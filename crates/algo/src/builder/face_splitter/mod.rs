//! Face splitting via 2D wire construction.
//!
//! For each face, collects boundary edges and section edges, converts
//! them to [`OrientedPCurveEdge`]s in the face's parameter space, calls
//! the wire builder, and produces [`SplitSubFace`]s.

mod containment;
mod conversion;
mod edge_splitting;
mod pole_round;
mod sampling;
mod special_cases;
mod tube_band;
pub(in crate::builder) use special_cases::cylinder_cone_remainder_interior;

use brepkit_math::madd::Madd;
pub use conversion::collect_wire_points;

use brepkit_math::region2d::point_in_region;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::classify_2d::{sample_interior_point, sample_interior_point_where, signed_area_2d};
use super::pcurve_compute::{
    compute_pcurve_on_surface, evaluate_edge_at_t, project_point_on_surface,
};
use super::plane_frame::PlaneFrame;
use super::split_types::{OrientedPCurveEdge, SectionEdge, SplitSubFace, SurfaceInfo};
use super::wire_builder::{
    build_wire_loops, build_wire_loops_dcel, build_wire_loops_dcel_measured,
    build_wire_loops_with_winding,
};

/// Quantized 3D endpoint pair key for loop-geometry equality tests.
type QKey3 = ((i64, i64, i64), (i64, i64, i64));
use crate::ds::Rank;

use containment::{find_point_outside_holes, is_inside_any_hole};
use conversion::{
    boundary_edges_to_pcurve, extract_plane_normal, is_point_on_boundary_uv,
    uv_endpoints_from_pcurve,
};
use edge_splitting::{
    find_splits_on_line, find_splits_on_nurbs_section, find_splits_on_section_arc,
    find_splits_on_section_ellipse, split_boundary_edges_at_3d_points,
};
use sampling::{sample_wire_loop_uv, sample_wire_loop_uv_periodic, sample_wire_loop_uv_via_frame};
use special_cases::{
    split_face_with_internal_loops, split_noseam_face_direct, split_periodic_face_into_bands,
    split_periodic_face_into_sectors, split_torus_by_coaxial_circles,
    try_split_crossing_plane_face, try_split_disk_by_chords,
};

/// Number of probe points (plus one for the closing sample) walked along a
/// section edge when testing whether it lies entirely inside an existing hole.
const HOLE_PROBE_SAMPLES: usize = 8;

/// Number of samples (plus one) walked along an arrangement arc input when
/// deciding whether a chord-crossing break point actually lies on the arc.
const ARR_ARC_SAMPLES: usize = 32;

/// Parameter `t` in `(0,1)` along segment `a0->a1` where it crosses segment
/// `b0->b1` in 2D, for a crossing strictly interior to `a` and within (or at
/// the ends of) `b`. `None` if parallel or out of range.
fn seg_cross_param(a0: Point2, a1: Point2, b0: Point2, b1: Point2) -> Option<f64> {
    let (rx, ry) = (a1.x() - a0.x(), a1.y() - a0.y());
    let (sx, sy) = (b1.x() - b0.x(), b1.y() - b0.y());
    let denom = rx.madd(sy, -(ry * sx));
    // `denom = |r x s| = |r||s| sin(theta)`; test it relative to the segment
    // lengths so near-parallel rejection is independent of model scale.
    let scale = (rx.hypot(ry) * sx.hypot(sy)).max(f64::MIN_POSITIVE);
    if denom.abs() <= 1e-9 * scale {
        return None;
    }
    let (qx, qy) = (b0.x() - a0.x(), b0.y() - a0.y());
    let t = qx.madd(sy, -(qy * sx)) / denom;
    let u = qx.madd(ry, -(qy * rx)) / denom;
    // `t`/`u` are normalized [0,1] parameters, so these epsilons are already
    // scale-invariant fractions of each segment.
    (t > 1e-6 && t < 1.0 - 1e-6 && u > -1e-6 && u < 1.0 + 1e-6).then_some(t)
}

/// Whether a boundary arc's chord crosses a section's chord in (u, v): a
/// chord-read arrangement then sees regions the arcs do not bound.
fn arc_chords_cross_sections(edges: &[OrientedPCurveEdge]) -> bool {
    let sections: Vec<(Point2, Point2)> = edges
        .iter()
        .filter(|e| e.source_edge_idx.is_some())
        .map(|e| (e.start_uv, e.end_uv))
        .collect();
    edges
        .iter()
        .filter(|e| e.source_edge_idx.is_none() && !edge_curve_is_straight(&e.curve_3d))
        .any(|e| {
            sections
                .iter()
                .any(|&(a, b)| seg_cross_param(e.start_uv, e.end_uv, a, b).is_some())
        })
}

/// Overlap span `[ta, tb]` (parameters along `a0->a1`, clamped to `[0,1]`)
/// where segment `a` rides collinearly on segment `b`: parallel within the
/// same near-parallel rejection band as [`seg_cross_param`], on the same
/// supporting line within a weld-scale offset, overlapping by more than a
/// parameter sliver. `None` otherwise.
fn collinear_overlap_span(a0: Point2, a1: Point2, b0: Point2, b1: Point2) -> Option<(f64, f64)> {
    let (rx, ry) = (a1.x() - a0.x(), a1.y() - a0.y());
    let (sx, sy) = (b1.x() - b0.x(), b1.y() - b0.y());
    let len2 = rx.madd(rx, ry * ry);
    if len2 < 1e-18 {
        return None;
    }
    let scale = (rx.hypot(ry) * sx.hypot(sy)).max(f64::MIN_POSITIVE);
    if rx.madd(sy, -(ry * sx)).abs() > 1e-9 * scale {
        return None;
    }
    let len = len2.sqrt();
    let off = |px: f64, py: f64| ((px - a0.x()) * ry - (py - a0.y()) * rx).abs() / len;
    if off(b0.x(), b0.y()) > 1e-6 || off(b1.x(), b1.y()) > 1e-6 {
        return None;
    }
    let t_of = |px: f64, py: f64| ((px - a0.x()) * rx + (py - a0.y()) * ry) / len2;
    let (mut ta, mut tb) = (t_of(b0.x(), b0.y()), t_of(b1.x(), b1.y()));
    if ta > tb {
        std::mem::swap(&mut ta, &mut tb);
    }
    let (ta, tb) = (ta.max(0.0), tb.min(1.0));
    (tb - ta > 1e-6).then_some((ta, tb))
}

/// A section piece's two quantized ends, in either order.
type EndsKey = (i64, i64, i64, i64, i64, i64);

/// A split piece's id, the source it was cut from, its ends in that
/// source's own direction, and the piece itself.
type PieceOf = (usize, Option<usize>, EndsKey, SectionEdge);

/// Split section edges at interior T-junctions with other sections.
///
/// `all_edges[section_start..]` holds the section edges as consecutive
/// forward/reverse pairs (both carrying the same `source_edge_idx`). When one
/// section's 3D endpoint lands strictly inside another section's span, the
/// crossed section is split there so both meet at a shared vertex — without
/// this the dangling end is pruned as a pendant and the face never splits at
/// that junction. Sections are split at `meetings` too, points where two of
/// them meet inside both. Boundary edges (`..section_start`) are left
/// untouched.
///
/// This covers the analytic (cylinder/cone) faces that
/// `try_split_crossing_plane_face` (plane-only) does not reach — e.g. a
/// rounded notch corner whose perpendicular-cut arc meets the axis-parallel
/// wall-top line on a corner cylinder. Each split piece gets a fresh unique
/// `source_edge_idx` (forward/reverse of a piece share it) so
/// `build_topology_face` still shares one topology edge per piece.
#[allow(clippy::too_many_arguments)]
fn split_sections_at_t_junctions(
    all_edges: &mut Vec<OrientedPCurveEdge>,
    section_start: usize,
    surface: &FaceSurface,
    frame: Option<&PlaneFrame>,
    wire_pts: &[Point3],
    tol: f64,
    mut split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
    meetings: &[Point3],
) {
    // Every distinct section endpoint (3D) is a candidate split point. Dedup
    // with a fine grid (cell = tol), then index the unique points in a COARSE
    // grid (cell = mean section length) so the per-section search below probes
    // only nearby candidates — near-linear instead of O(sections²) on a
    // perforated cap's many disjoint hole edges. The coarse query reproduces the
    // former all-endpoints scan: an endpoint on a section lies within `tol` of
    // it, hence in a coarse cell its bounding box (expanded by `tol`) overlaps.
    let fine = tol.max(f64::MIN_POSITIVE);
    let fine_inv = 1.0 / fine;
    let fine_cell = |p: Point3| -> (i64, i64, i64) {
        #[allow(clippy::cast_possible_truncation)]
        (
            (p.x() * fine_inv).floor() as i64,
            (p.y() * fine_inv).floor() as i64,
            (p.z() * fine_inv).floor() as i64,
        )
    };
    let mut endpoints: Vec<Point3> = Vec::new();
    let mut fine_grid: std::collections::HashMap<(i64, i64, i64), Vec<Point3>> =
        std::collections::HashMap::new();
    let mut len_sum = 0.0_f64;
    let mut len_cnt = 0.0_f64;
    let ends = all_edges[section_start..]
        .iter()
        .flat_map(|e| [e.start_3d, e.end_3d]);
    for e in &all_edges[section_start..] {
        len_sum += (e.end_3d - e.start_3d).length();
        len_cnt += 1.0;
    }
    for p in ends.chain(meetings.iter().copied()) {
        let (cx, cy, cz) = fine_cell(p);
        let dup = (-1..=1).any(|dx| {
            (-1..=1).any(|dy| {
                (-1..=1).any(|dz| {
                    fine_grid
                        .get(&(cx + dx, cy + dy, cz + dz))
                        .is_some_and(|pts| pts.iter().any(|q| (*q - p).length() < tol))
                })
            })
        });
        if !dup {
            endpoints.push(p);
            fine_grid.entry((cx, cy, cz)).or_default().push(p);
        }
    }
    // Coarse query grid: cell sized to the mean section length so a section
    // spans O(1) cells. Each endpoint is stored once.
    let coarse = if len_cnt > 0.0 {
        (len_sum / len_cnt).max(fine)
    } else {
        fine
    };
    let coarse_inv = 1.0 / coarse;
    let coarse_cell = |x: f64, y: f64, z: f64| -> (i64, i64, i64) {
        #[allow(clippy::cast_possible_truncation)]
        (
            (x * coarse_inv).floor() as i64,
            (y * coarse_inv).floor() as i64,
            (z * coarse_inv).floor() as i64,
        )
    };
    let mut coarse_grid: std::collections::HashMap<(i64, i64, i64), Vec<Point3>> =
        std::collections::HashMap::new();
    for &p in &endpoints {
        coarse_grid
            .entry(coarse_cell(p.x(), p.y(), p.z()))
            .or_default()
            .push(p);
    }
    // Candidate endpoints near a section edge's bounding box (expanded by tol).
    // A box spanning more cells than there are endpoints can't gain from the
    // grid, so fall back to the full set there (keeps the result identical).
    let candidates_near = |s3: Point3, e3: Point3| -> Vec<Point3> {
        let lo = coarse_cell(
            s3.x().min(e3.x()) - tol,
            s3.y().min(e3.y()) - tol,
            s3.z().min(e3.z()) - tol,
        );
        let hi = coarse_cell(
            s3.x().max(e3.x()) + tol,
            s3.y().max(e3.y()) + tol,
            s3.z().max(e3.z()) + tol,
        );
        let span = (hi.0 - lo.0 + 1)
            .saturating_mul(hi.1 - lo.1 + 1)
            .saturating_mul(hi.2 - lo.2 + 1);
        if span < 0 || span as usize > endpoints.len() {
            return endpoints.clone();
        }
        let mut out = Vec::new();
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                for cz in lo.2..=hi.2 {
                    if let Some(pts) = coarse_grid.get(&(cx, cy, cz)) {
                        out.extend_from_slice(pts);
                    }
                }
            }
        }
        out
    };

    let boundary: Vec<OrientedPCurveEdge> = all_edges[..section_start].to_vec();
    let sections: Vec<OrientedPCurveEdge> = all_edges[section_start..].to_vec();

    // A unique source id per geometric piece, kept stable across the
    // forward/reverse pair so the topology builder shares one edge per piece.
    // Start above every existing source id (sections already use ids ≥
    // section_start) so a fresh piece never collides with an unsplit edge.
    let mut next_src = all_edges
        .iter()
        .filter_map(|e| e.source_edge_idx)
        .max()
        .map_or(section_start, |m| m + 1);
    // Pieces by their ends, each with its id, the section it was cut from,
    // its ends in that section's own direction, and the piece as a section
    // read from its stored start. Two pieces cut from one section between
    // the same ends in its own direction are one curve, a piece and its
    // reverse twin; pieces of two sections with the same ends share an id
    // only where they are proved one curve (a circle's and an ellipse's
    // pieces between the two points where they meet are two).
    let mut piece_src: std::collections::HashMap<EndsKey, Vec<PieceOf>> =
        std::collections::HashMap::new();
    let cell = |a: Point3| {
        let q = |x: f64| (x / tol).round() as i64;
        (q(a.x()), q(a.y()), q(a.z()))
    };
    // The ends of a piece from `a` to `b`, in that order.
    let ordered = |a: Point3, b: Point3| -> EndsKey {
        let (ka, kb) = (cell(a), cell(b));
        (ka.0, ka.1, ka.2, kb.0, kb.1, kb.2)
    };
    // Order-independent so forward and reverse halves map to one id.
    let key = |a: Point3, b: Point3| -> EndsKey {
        if cell(a) <= cell(b) {
            ordered(a, b)
        } else {
            ordered(b, a)
        }
    };

    let mut new_sections: Vec<OrientedPCurveEdge> = Vec::with_capacity(sections.len());
    for edge in sections {
        let splits = match &edge.curve_3d {
            // Arc sections bulge beyond their chord, so an endpoint can lie on
            // the arc yet outside the chord's bounding box — the grid filter
            // (keyed on chord extent) would miss it. Arcs are rare (rounded
            // corners), so scan the full endpoint set for them; the
            // O(sections²) pressure comes from the many Line sections, which the
            // grid prunes. `find_splits_on_*` exclude the edge's own endpoints.
            // Circle AND ellipse sections use the shorter-arc
            // parameterization: each is pushed as a forward/reverse PAIR, and
            // the CCW-domain convention returns the long complement span for
            // the reverse twin (phantom interior splits from points outside
            // the arc — see `find_splits_on_section_arc` /
            // `find_splits_on_section_ellipse`). The shorter-arc convention
            // matches `evaluate_edge_at_t`, which both twins share.
            // A face boundary piece carried in as a section (a promoted hole's
            // pave-split rim) runs one way only and may span more than half
            // a turn. The shorter-arc reading then covers its complement: it
            // puts the complement's points on the piece and none of the
            // piece's own, so it can only mint false splits there. At exactly
            // half a turn (a rim split by a line through its centre) the
            // reading may take either half, which is no better.
            EdgeCurve::Circle(c)
                if edge.source_edge_idx.is_none() && {
                    let (s, e) = if edge.forward {
                        (edge.start_3d, edge.end_3d)
                    } else {
                        (edge.end_3d, edge.start_3d)
                    };
                    (c.project(e) - c.project(s)).rem_euclid(std::f64::consts::TAU)
                        > std::f64::consts::PI - 1e-9
                } =>
            {
                Vec::new()
            }
            EdgeCurve::Circle(_) => find_splits_on_section_arc(&edge, &endpoints, tol),
            EdgeCurve::Ellipse(_) => find_splits_on_section_ellipse(&edge, &endpoints, tol),
            // A marched-NURBS section (a plane×cone conic) bulges past its
            // chord like an arc — a junction endpoint mid-curve is invisible
            // to the chord-based search, so use sampled point-to-curve
            // projection over the full endpoint set (conics are rare).
            EdgeCurve::NurbsCurve(_) => find_splits_on_nurbs_section(&edge, &endpoints, tol, true),
            // Only endpoints near a line section's bounding box can land on it;
            // the grid query returns exactly that subset, preserving the former
            // full scan's result.
            EdgeCurve::Line => {
                find_splits_on_line(&edge, &candidates_near(edge.start_3d, edge.end_3d), tol)
            }
        };
        if splits.is_empty() {
            // Keep the original source id so an unsplit pair stays paired.
            new_sections.push(edge);
            continue;
        }

        // The face's section edges already carry UV unwrapped into the face's
        // continuous parameter window (the partial-band u-unwrap runs earlier),
        // but a fresh surface projection of a split point returns the raw
        // parameter (e.g. u in [0, 2pi)). Snap it to the period nearest the
        // running anchor so the split vertex stays in the same window.
        let (u_period, v_period) = super::pcurve_compute::surface_periods(surface);
        let project = |p: Point3, near: Point2| -> Point2 {
            let raw = if let Some(f) = frame {
                f.project(p)
            } else {
                project_point_on_surface(p, surface, wire_pts, None)
            };
            if frame.is_some() {
                return raw;
            }
            let snap = |val: f64, anchor: f64, period: Option<f64>| -> f64 {
                match period {
                    Some(p) if p > 1e-12 => val + ((anchor - val) / p).round() * p,
                    _ => val,
                }
            };
            Point2::new(
                snap(raw.x(), near.x(), u_period),
                snap(raw.y(), near.y(), v_period),
            )
        };

        let mut prev_3d = edge.start_3d;
        let mut prev_uv = edge.start_uv;
        let mut push_piece = |s3: Point3, e3: Point3, s_uv: Point2, e_uv: Point2| {
            let pcurve =
                compute_pcurve_on_surface(&edge.curve_3d, s3, e3, surface, wire_pts, frame);
            let (start, end) = if edge.forward { (s3, e3) } else { (e3, s3) };
            let piece = SectionEdge {
                curve_3d: edge.curve_3d.clone(),
                pcurve_a: pcurve.clone(),
                pcurve_b: pcurve.clone(),
                start,
                end,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
                pave_block_id: None,
            };
            let along = ordered(start, end);
            let same_ends = piece_src.entry(key(s3, e3)).or_default();
            let twin = same_ends.iter().find(|(_, from, ends, _)| {
                from.is_some() && *from == edge.source_edge_idx && *ends == along
            });
            let alike = || {
                same_ends.iter().find(|(_, from, _, other)| {
                    (from.is_none() || *from != edge.source_edge_idx)
                        && same_section_curve(other, &piece, 10.0 * tol)
                })
            };
            let src = if let Some(&(src, ..)) = twin.or_else(alike) {
                src
            } else {
                let v = next_src;
                next_src += 1;
                same_ends.push((v, edge.source_edge_idx, along, piece));
                v
            };
            new_sections.push(OrientedPCurveEdge {
                curve_3d: edge.curve_3d.clone(),
                pcurve,
                start_uv: s_uv,
                end_uv: e_uv,
                start_3d: s3,
                end_3d: e3,
                forward: edge.forward,
                source_edge_idx: Some(src),
                // A split piece is a sub-segment of the original section, so
                // it must not inherit the parent's pave_block_id — vertex
                // resolution would snap both halves to the PaveBlock's
                // (un-split) endpoints. Resolve by position instead;
                // cross-face sharing is recovered by merge_duplicate_edges.
                pave_block_id: None,
            });
        };
        // A circle or ellipse splits at the foot of the split point, which
        // both twins read alike; along the shorter arc, a half turn reads
        // either way to rounding.
        let at_split = |t: f64, sp: Point3| match &edge.curve_3d {
            EdgeCurve::Circle(c) => c.evaluate(c.project(sp)),
            EdgeCurve::Ellipse(e) => e.evaluate(e.project(sp)),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => {
                evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t)
            }
        };
        if let Some(reg) = split_registry.as_deref_mut()
            && let Some(pb_id) = edge.pave_block_id
        {
            for &(t, sp) in &splits {
                reg.entry(pb_id).or_default().push(at_split(t, sp));
            }
        }
        for &(t, sp) in &splits {
            let s3 = at_split(t, sp);
            let s_uv = project(s3, prev_uv);
            push_piece(prev_3d, s3, prev_uv, s_uv);
            prev_3d = s3;
            prev_uv = s_uv;
        }
        push_piece(prev_3d, edge.end_3d, prev_uv, edge.end_uv);
    }

    all_edges.truncate(0);
    all_edges.extend(boundary);
    all_edges.extend(new_sections);
}

/// Split a plane face's boundary arc/line edges at 3D points that land on their
/// interior. Used to attach a section whose endpoint lands mid-arc on a convex
/// rounded corner (the notch-straddle case).
///
/// Unlike [`split_boundary_edges_at_3d_points`], the arc split parameter is
/// computed with the SHORTER-arc convention that `evaluate_edge_at_t` uses, so
/// a corner arc traversed clockwise in its circle frame (as plane-face boundary
/// arcs are) is split at the geometrically-correct location rather than being
/// missed because the `domain_with_endpoints` CCW span excludes it.
fn split_plane_boundary_arcs_at_points(
    edges: Vec<OrientedPCurveEdge>,
    split_pts_3d: &[Point3],
    surface: &FaceSurface,
    frame: &PlaneFrame,
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    // Parameter t in (0,1) of `p` along the arc piece, in traversal order,
    // with its foot on the curve, or None if `p` is not on the piece's
    // interior. A piece runs its stored edge's own counter-clockwise span,
    // which past half a turn is the longer of the two arcs between its ends.
    let arc_param = |edge: &OrientedPCurveEdge, p: Point3| -> Option<(f64, Point3)> {
        let (angle, on_curve): (f64, Point3) = match &edge.curve_3d {
            EdgeCurve::Circle(c) => (c.project(p), c.evaluate(c.project(p))),
            EdgeCurve::Ellipse(e) => (e.project(p), e.evaluate(e.project(p))),
            // Only arc edges have a circle/ellipse parameter; a line or NURBS
            // edge is never split by this arc-only path.
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return None,
        };
        if (p - on_curve).length() > tol || (edge.start_3d - edge.end_3d).length() < tol {
            return None;
        }
        let (stored_start, stored_end) = if edge.forward {
            (edge.start_3d, edge.end_3d)
        } else {
            (edge.end_3d, edge.start_3d)
        };
        let (t0, t1) = edge
            .curve_3d
            .domain_with_endpoints(stored_start, stored_end);
        if (t1 - t0).abs() < 1e-12 {
            return None;
        }
        let along = sampling::normalize_angle_in_span(angle, t0, t1 - t0);
        let t = if edge.forward { along } else { 1.0 - along };
        (t > tol && t < 1.0 - tol).then_some((t, on_curve))
    };

    let mut result = Vec::with_capacity(edges.len());
    for edge in edges {
        let mut splits: Vec<(f64, Point3)> = match &edge.curve_3d {
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => split_pts_3d
                .iter()
                .filter_map(|&p| arc_param(&edge, p))
                .collect(),
            EdgeCurve::Line => {
                let dir = edge.end_3d - edge.start_3d;
                let len_sq = dir.dot(dir);
                if len_sq < tol * tol {
                    Vec::new()
                } else {
                    split_pts_3d
                        .iter()
                        .filter_map(|&p| {
                            let t = (p - edge.start_3d).dot(dir) / len_sq;
                            let closest = edge.start_3d + dir * t;
                            ((p - closest).length() < tol && t > tol && t < 1.0 - tol)
                                .then_some((t, closest))
                        })
                        .collect()
                }
            }
            EdgeCurve::NurbsCurve(_) => Vec::new(),
        };
        splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
        if splits.is_empty() {
            result.push(edge);
            continue;
        }

        let mut prev_3d = edge.start_3d;
        let mut prev_uv = edge.start_uv;
        let mut push_piece = |s3: Point3, e3: Point3, s_uv: Point2, e_uv: Point2| {
            let pcurve = if matches!(edge.curve_3d, EdgeCurve::Line) {
                compute_pcurve_on_surface(&edge.curve_3d, s3, e3, surface, &[], Some(frame))
            } else {
                edge_splitting::boundary_piece_pcurve(&edge, s3, e3, surface, Some(frame))
            };
            result.push(OrientedPCurveEdge {
                curve_3d: edge.curve_3d.clone(),
                pcurve,
                start_uv: s_uv,
                end_uv: e_uv,
                start_3d: s3,
                end_3d: e3,
                forward: edge.forward,
                source_edge_idx: None,
                pave_block_id: None,
            });
        };
        for &(_, s3) in &splits {
            let s_uv = frame.project(s3);
            push_piece(prev_3d, s3, prev_uv, s_uv);
            prev_3d = s3;
            prev_uv = s_uv;
        }
        push_piece(prev_3d, edge.end_3d, prev_uv, edge.end_uv);
    }
    result
}

/// The boundary with each arc piece whose ends are both ends of one straight
/// section split at its middle along its own span, or `None` when no piece is.
fn split_arcs_shadowing_chords(
    boundary: &[OrientedPCurveEdge],
    sections: &[OrientedPCurveEdge],
    surface: &FaceSurface,
    frame: &PlaneFrame,
    tol: f64,
) -> Option<Vec<OrientedPCurveEdge>> {
    let same = |a: Point3, b: Point3| (a - b).length() < tol;
    let chords: Vec<(Point3, Point3)> = sections
        .iter()
        .filter(|e| matches!(e.curve_3d, EdgeCurve::Line))
        .map(|e| (e.start_3d, e.end_3d))
        .collect();
    let shadows = |e: &OrientedPCurveEdge| {
        matches!(e.curve_3d, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_))
            && !same(e.start_3d, e.end_3d)
            && chords.iter().any(|&(p, q)| {
                (same(p, e.start_3d) && same(q, e.end_3d))
                    || (same(q, e.start_3d) && same(p, e.end_3d))
            })
    };
    if !boundary.iter().any(shadows) {
        return None;
    }
    let mut out = Vec::with_capacity(boundary.len() + 1);
    for e in boundary {
        if !shadows(e) {
            out.push(e.clone());
            continue;
        }
        let (stored_start, stored_end) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let (t0, t1) = e.curve_3d.domain_with_endpoints(stored_start, stored_end);
        let mid = e
            .curve_3d
            .evaluate_with_endpoints(0.5 * (t0 + t1), stored_start, stored_end);
        let mid_uv = frame.project(mid);
        for (from, to, from_uv, to_uv) in [
            (e.start_3d, mid, e.start_uv, mid_uv),
            (mid, e.end_3d, mid_uv, e.end_uv),
        ] {
            out.push(OrientedPCurveEdge {
                curve_3d: e.curve_3d.clone(),
                pcurve: edge_splitting::boundary_piece_pcurve(e, from, to, surface, Some(frame)),
                start_uv: from_uv,
                end_uv: to_uv,
                start_3d: from,
                end_3d: to,
                forward: e.forward,
                source_edge_idx: None,
                pave_block_id: None,
            });
        }
    }
    Some(out)
}

/// Whether an edge curve is geometrically a straight segment, independent of
/// its nominal type. A planar-NURBS extrusion wall's rim is a `NurbsCurve`
/// that is exactly straight (the tilted-divider cavity). Straightness is
/// decided from the control polygon — a NURBS whose control points are
/// collinear is straight everywhere, for any trim of the edge — so a trimmed
/// span of a genuinely curved carrier is never misjudged by sampling only part
/// of it. The collinearity band is the kernel's default linear tolerance
/// (1e-7); a control polygon that tight is straight for any splitting purpose.
pub fn edge_curve_is_straight(curve: &EdgeCurve) -> bool {
    match curve {
        EdgeCurve::Line => true,
        EdgeCurve::NurbsCurve(n) => {
            let pts = n.control_points();
            let (Some(first), Some(last)) = (pts.first(), pts.last()) else {
                return false;
            };
            let chord = *last - *first;
            let len = chord.length();
            if len < 1e-12 {
                return false;
            }
            pts.iter().all(|p| {
                let v = *p - *first;
                let along = v.dot(chord) / len;
                let dev_sq = along.madd(-along, v.dot(v));
                dev_sq < 1e-14
            })
        }
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => false,
    }
}

/// Weave hole boundaries into the section arrangement of a planar face.
///
/// When a holed planar face is cut by sections (e.g. a shelled box top with a
/// cavity opening, fused with a lip whose walls cross that opening), the
/// section runs partly through the cavity. Splitting only the outer boundary
/// leaves the hole un-split, so a sub-face ends up as a square carrying the
/// whole over-sized cavity hole instead of the true L-shaped rim. Trim each
/// section at the points where it crosses a hole edge — dropping the
/// sub-segment that lies inside the hole — and split the hole edges at those
/// crossings. The wire builder then traces the real material region.
///
/// Returns `(woven_edges, passthrough_hole_indices)` to append to the boundary,
/// or `None` to fall back to the attach-whole-hole path (curved holes/sections,
/// or no crossing — nothing to integrate). `passthrough_hole_indices` are the
/// holes that DON'T interact with any section: they are left out of the woven
/// arrangement (so their exact — possibly arc-bounded — geometry is not
/// chord-fragmented into spurious sliver regions) and must be attached whole by
/// the caller. Only holes a section actually crosses (or that cross a section)
/// need weaving; a baseplate top cut at one corner has 15 untouched cell
/// openings that must stay intact.
/// `source_edge_idx` base for weave/promotion section pieces: values at or
/// above this mark an edge as a synthetic hole-weave section rather than an
/// index into the caller's real section array (which is always far smaller).
const WEAVE_SECTION_SRC_BASE: usize = 1_000_000;

fn integrate_holes_plane(
    sections: &[SectionEdge],
    inner_wires: &[Vec<OrientedPCurveEdge>],
    frame: &PlaneFrame,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    base_src: usize,
) -> Option<(Vec<OrientedPCurveEdge>, Vec<usize>)> {
    // Line sections are split at hole crossings (the in-hole sub-segment is air,
    // dropped). Arc sections cannot be chord-trimmed here, so they are carried
    // through whole — valid only when an arc lies clear of every hole (a corner
    // arc on the OUTER boundary ring, well outside the inset cavity openings, in
    // the divider-lip fuse). If an arc would cross a hole's straight wall its
    // true geometry cannot be reproduced as one sub-edge, so the whole pass
    // bails to None and the caller's other paths handle the face.
    let (line_sections, arc_sections): (Vec<&SectionEdge>, Vec<&SectionEdge>) = sections
        .iter()
        .partition(|s| matches!(s.curve_3d, EdgeCurve::Line));

    // Identify holes that actually interact with a section: a section endpoint
    // lies inside the hole, OR a section segment crosses one of the hole's
    // edges. Non-interacting holes are left whole (returned as passthrough) so
    // their exact arc geometry is preserved instead of being chord-fragmented
    // by the arrangement subdivision.
    let sec_uv_all: Vec<(Point2, Point2)> = sections
        .iter()
        .map(|s| (frame.project(s.start), frame.project(s.end)))
        .collect();
    // Where an arc section ENDS on a hole's straight edge (a ring arc meeting
    // the opening's wall), as the parameter along that edge. The wall is split
    // at the foot of the arc's end, so the end must lie within a tenth of the
    // weld band for the two to meet as one vertex.
    let ends_on = |p: Point2, h0: Point2, h1: Point2| -> Option<f64> {
        let (dx, dy) = (h1.x() - h0.x(), h1.y() - h0.y());
        let len2 = dx.madd(dx, dy * dy);
        if len2 < 1e-24 {
            return None;
        }
        let t = (p.x() - h0.x()).madd(dx, (p.y() - h0.y()) * dy) / len2;
        if !(1e-6..=1.0 - 1e-6).contains(&t) {
            return None;
        }
        let q = Point2::new(dx.madd(t, h0.x()), dy.madd(t, h0.y()));
        ((q - p).length() < 1e-8).then_some(t)
    };
    // The weave reads regions by chords, so an arc whose chord lies along the
    // wall's own line (a rod's bottom arc between two points of a keyhole's
    // tail base) bounds nothing it can see; such an arc is left to the
    // un-woven path. The test is on the line, not the segment: the rod's own
    // bottom arc splits that base into two walls, one under each end.
    let arc_uv: Vec<(Point2, Point2)> = sections
        .iter()
        .filter(|s| !matches!(s.curve_3d, EdgeCurve::Line))
        .map(|s| (frame.project(s.start), frame.project(s.end)))
        .collect();
    let arc_ends_on = |h0: Point2, h1: Point2| -> Vec<f64> {
        arc_uv
            .iter()
            .filter(|&&(a, b)| !chord_on_line(a, b, h0, h1))
            .flat_map(|&(a, b)| [ends_on(a, h0, h1), ends_on(b, h0, h1)])
            .flatten()
            .collect()
    };
    let interacts = |hole: &[OrientedPCurveEdge]| -> bool {
        let poly: Vec<Point2> = hole.iter().map(|e| frame.project(e.start_3d)).collect();
        if poly.len() >= 3 {
            for (a, b) in &sec_uv_all {
                if super::classify_2d::point_in_polygon_2d(*a, &poly)
                    || super::classify_2d::point_in_polygon_2d(*b, &poly)
                {
                    return true;
                }
            }
        }
        for e in hole {
            let h0 = frame.project(e.start_3d);
            let h1 = frame.project(e.end_3d);
            for (a, b) in &sec_uv_all {
                if seg_cross_param(h0, h1, *a, *b).is_some() {
                    return true;
                }
            }
            if edge_curve_is_straight(&e.curve_3d) && !arc_ends_on(h0, h1).is_empty() {
                return true;
            }
        }
        false
    };
    let mut passthrough: Vec<usize> = Vec::new();
    let woven_inner_wires: Vec<Vec<OrientedPCurveEdge>> = inner_wires
        .iter()
        .enumerate()
        .filter_map(|(i, h)| {
            if interacts(h) {
                Some(h.clone())
            } else {
                passthrough.push(i);
                None
            }
        })
        .collect();
    let inner_wires: &[Vec<OrientedPCurveEdge>] = &woven_inner_wires;

    // Chord polygon per hole (arc edges contribute their start endpoint, which
    // is the right fidelity for the "is this section sub-segment inside the
    // cavity" point test — the test points are on the straight walls, far from
    // the corner arcs).
    let hole_polys: Vec<Vec<Point2>> = inner_wires
        .iter()
        .map(|w| w.iter().map(|e| frame.project(e.start_3d)).collect())
        .collect();
    // Straight hole edges only feed the section-split crossing set (a section
    // entering the cavity crosses a straight wall). Arc edges are carried
    // through separately below. "Straight" is geometric, not nominal: a
    // planar-NURBS extrusion wall's rim is a NurbsCurve edge that is exactly
    // straight (the tilted-divider cavity), and leaving it to the arc branch
    // makes the whole pass bail on the section that crosses it — the divider
    // cap is then never extracted and its top ring stays open.
    let hole_segs: Vec<(Point2, Point2, Point3, Point3)> = inner_wires
        .iter()
        .flatten()
        .filter(|e| edge_curve_is_straight(&e.curve_3d))
        .map(|e| {
            (
                frame.project(e.start_3d),
                frame.project(e.end_3d),
                e.start_3d,
                e.end_3d,
            )
        })
        .collect();

    let mk_line =
        |s_uv: Point2, e_uv: Point2, s3: Point3, e3: Point3, fwd: bool, src: Option<usize>| {
            use brepkit_math::curves2d::{Curve2D, Line2D};
            use brepkit_math::vec::Vec2;
            let d = Vec2::new(e_uv.x() - s_uv.x(), e_uv.y() - s_uv.y());
            let len = (d.x() * d.x() + d.y() * d.y()).sqrt();
            let dir = if len > 1e-12 {
                Vec2::new(d.x() / len, d.y() / len)
            } else {
                Vec2::new(1.0, 0.0)
            };
            let pcurve = Curve2D::Line(
                Line2D::new(s_uv, dir)
                    .or_else(|_| Line2D::new(s_uv, Vec2::new(1.0, 0.0)))
                    .ok()?,
            );
            Some(OrientedPCurveEdge {
                curve_3d: EdgeCurve::Line,
                pcurve,
                start_uv: s_uv,
                end_uv: e_uv,
                start_3d: s3,
                end_3d: e3,
                forward: fwd,
                source_edge_idx: src,
                pave_block_id: None,
            })
        };

    let mut out: Vec<OrientedPCurveEdge> = Vec::new();
    let mut any_crossing = false;
    let mut next_src = base_src;

    // Line sections: split at hole crossings, drop the in-hole sub-segments.
    for s in &line_sections {
        let s0 = frame.project(s.start);
        let s1 = frame.project(s.end);
        let mut ts: Vec<f64> = vec![0.0, 1.0];
        // Spans of this section that ride collinearly ON a hole edge. A
        // window inside such a span has a boundary point for a midpoint,
        // which the interior polygon test below cannot classify — the
        // verdict flips with ray direction, so of a mirrored pair of wall
        // cuts one reads inside and the other outside (the label-bracket
        // finger fuse). Split at the overlap ends and drop the riding
        // piece explicitly: the hole edge itself already bounds the
        // arrangement there, and emitting a duplicate would recreate the
        // parallel-twin hazard the dedup above guards against.
        let mut riding: Vec<(f64, f64)> = Vec::new();
        for (b0, b1, _, _) in &hole_segs {
            if let Some(t) = seg_cross_param(s0, s1, *b0, *b1) {
                ts.push(t);
                any_crossing = true;
            } else if let Some((ta, tb)) = collinear_overlap_span(s0, s1, *b0, *b1) {
                for t in [ta, tb] {
                    if t > 1e-6 && t < 1.0 - 1e-6 {
                        ts.push(t);
                        any_crossing = true;
                    }
                }
                riding.push((ta, tb));
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        for w in ts.windows(2) {
            let (ta, tb) = (w[0], w[1]);
            let tm = 0.5 * (ta + tb);
            if riding.iter().any(|&(a, b)| tm >= a && tm <= b) {
                continue; // rides the hole boundary — the hole edge covers it
            }
            let mid = Point2::new(
                s0.x() + (s1.x() - s0.x()) * tm,
                s0.y() + (s1.y() - s0.y()) * tm,
            );
            if hole_polys
                .iter()
                .any(|poly| super::classify_2d::point_in_polygon_2d(mid, poly))
            {
                continue; // sub-segment runs through the cavity — not material
            }
            let lerp2 = |t: f64| {
                Point2::new(
                    s0.x() + (s1.x() - s0.x()) * t,
                    s0.y() + (s1.y() - s0.y()) * t,
                )
            };
            let lerp3 = |t: f64| {
                Point3::new(
                    s.start.x() + (s.end.x() - s.start.x()) * t,
                    s.start.y() + (s.end.y() - s.start.y()) * t,
                    s.start.z() + (s.end.z() - s.start.z()) * t,
                )
            };
            let src = next_src;
            next_src += 1;
            let (ua, ub, pa, pb) = (lerp2(ta), lerp2(tb), lerp3(ta), lerp3(tb));
            out.push(mk_line(ua, ub, pa, pb, true, Some(src))?);
            out.push(mk_line(ub, ua, pb, pa, false, Some(src))?);
        }
    }

    // Hole edges: split at section crossings, keep their stored orientation.
    // Only Line sections split hole edges; arc sections are carried through
    // whole and (per the bail below) never cross a hole.
    let sec_uv: Vec<(Point2, Point2)> = line_sections
        .iter()
        .map(|s| (frame.project(s.start), frame.project(s.end)))
        .collect();
    for (h0, h1, p0, p1) in &hole_segs {
        let mut ts: Vec<f64> = vec![0.0, 1.0];
        for (a0, a1) in &sec_uv {
            if let Some(t) = seg_cross_param(*h0, *h1, *a0, *a1) {
                ts.push(t);
                any_crossing = true;
            }
        }
        // An arc section ending on the wall meets it at a T: the wall is
        // split there so the arc's end is a vertex of the arrangement.
        for t in arc_ends_on(*h0, *h1) {
            ts.push(t);
            any_crossing = true;
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        for w in ts.windows(2) {
            let (ta, tb) = (w[0], w[1]);
            let lerp2 = |t: f64| {
                Point2::new(
                    h0.x() + (h1.x() - h0.x()) * t,
                    h0.y() + (h1.y() - h0.y()) * t,
                )
            };
            let lerp3 = |t: f64| {
                Point3::new(
                    p0.x() + (p1.x() - p0.x()) * t,
                    p0.y() + (p1.y() - p0.y()) * t,
                    p0.z() + (p1.z() - p0.z()) * t,
                )
            };
            out.push(mk_line(
                lerp2(ta),
                lerp2(tb),
                lerp3(ta),
                lerp3(tb),
                true,
                None,
            )?);
        }
    }

    // Arc hole edges: preserve unchanged. Bail if any section crosses an arc's
    // chord (we don't split arcs here, and emitting the arc whole alongside a
    // section that cuts it would leave a dangling crossing).
    for arc in inner_wires
        .iter()
        .flatten()
        .filter(|e| !edge_curve_is_straight(&e.curve_3d))
    {
        let a0 = frame.project(arc.start_3d);
        let a1 = frame.project(arc.end_3d);
        for (s0, s1) in &sec_uv {
            if seg_cross_param(a0, a1, *s0, *s1).is_some() {
                return None;
            }
        }
        out.push(arc.clone());
    }

    // Arc sections: carried through whole as a forward/reverse pair (one shared
    // source id so `build_topology_face` welds the two sub-face uses to one
    // edge). An arc section that crosses a hole's straight wall (chord test)
    // cannot be reproduced as one sub-edge, so bail — the divider-lip case has
    // its corner arcs on the OUTER ring, clear of the inset cavity openings.
    for s in &arc_sections {
        let s0 = frame.project(s.start);
        let s1 = frame.project(s.end);
        for (b0, b1, _, _) in &hole_segs {
            if seg_cross_param(s0, s1, *b0, *b1).is_some() {
                return None;
            }
        }
        // An arc whose geometric midpoint (its bulge — not just the chord
        // midpoint) lies inside a hole is entirely over the cavity (air) — drop
        // it. Sampling the arc itself catches an arc that bows into a hole while
        // its endpoints and chord sit clear of it.
        let (ad0, ad1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
        let mid = frame.project(s.curve_3d.evaluate_with_endpoints(
            ad0 + 0.5 * (ad1 - ad0),
            s.start,
            s.end,
        ));
        if hole_polys
            .iter()
            .any(|poly| super::classify_2d::point_in_polygon_2d(mid, poly))
        {
            continue;
        }
        let src = next_src;
        next_src += 1;
        out.extend(curved_section_pair(s, frame, surface, wire_pts, src));
    }

    any_crossing.then_some((out, passthrough))
}

/// Where segments `a0`-`a1` and `b0`-`b1` cross at a point inside both.
fn segments_cross_at(a0: Point2, a1: Point2, b0: Point2, b1: Point2) -> Option<Point2> {
    let (rx, ry) = (a1.x() - a0.x(), a1.y() - a0.y());
    let (sx, sy) = (b1.x() - b0.x(), b1.y() - b0.y());
    let denom = rx.madd(sy, -(ry * sx));
    if denom.abs() <= 1e-12 * rx.hypot(ry) * sx.hypot(sy) {
        return None;
    }
    let (qx, qy) = (b0.x() - a0.x(), b0.y() - a0.y());
    let t = qx.madd(sy, -(qy * sx)) / denom;
    let u = qx.madd(ry, -(qy * rx)) / denom;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u))
        .then(|| Point2::new(rx.madd(t, a0.x()), ry.madd(t, a0.y())))
}

/// Whether the chord `a`-`b` lies along the line through `h0`-`h1`.
fn chord_on_line(a: Point2, b: Point2, h0: Point2, h1: Point2) -> bool {
    let (dx, dy) = (h1.x() - h0.x(), h1.y() - h0.y());
    let len = dx.hypot(dy);
    let off = |p: Point2| ((p.x() - h0.x()) * dy - (p.y() - h0.y()) * dx).abs() / len;
    len > 1e-12 && off(a) < 1e-8 && off(b) < 1e-8
}

/// A curved section on a plane face as a forward/reverse pair sharing one
/// source id, so `build_topology_face` welds the two sub-face uses to one
/// edge. The pcurve is refit in THIS face's frame: the stored one may come
/// from another plane frame or the opposing face, which would disconnect the
/// arc in this UV space.
fn curved_section_pair(
    s: &SectionEdge,
    frame: &PlaneFrame,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    src: usize,
) -> [OrientedPCurveEdge; 2] {
    let pcurve = super::pcurve_compute::compute_pcurve_on_surface(
        &s.curve_3d,
        s.start,
        s.end,
        surface,
        wire_pts,
        Some(frame),
    );
    let su = frame.project(s.start);
    let eu = frame.project(s.end);
    [
        OrientedPCurveEdge {
            curve_3d: s.curve_3d.clone(),
            pcurve: pcurve.clone(),
            start_uv: su,
            end_uv: eu,
            start_3d: s.start,
            end_3d: s.end,
            forward: true,
            source_edge_idx: Some(src),
            pave_block_id: s.pave_block_id,
        },
        OrientedPCurveEdge {
            curve_3d: s.curve_3d.clone(),
            pcurve,
            start_uv: eu,
            end_uv: su,
            start_3d: s.end,
            end_3d: s.start,
            forward: false,
            source_edge_idx: Some(src),
            pave_block_id: s.pave_block_id,
        },
    ]
}

/// How a loop sits relative to an outer loop's sampled polygon. Exact for
/// loops from a planar subdivision (loops never cross an outer, so boundary
/// containment of every sampled point decides region containment):
/// - `Nested`: every point inside or on the outer, at least one strictly
///   interior — a genuine hole candidate.
/// - `BoundaryCoincident`: every point ON the outer within its scale-aware
///   boundary tolerance — a re-trace of that outline (the shelled-cup lip
///   fuse weaves these from kept whole-edge duplicate sections).
/// - `Outside`: at least one point strictly outside — not contained.
#[derive(PartialEq, Eq, Clone, Copy)]
enum LoopContainment {
    Nested,
    BoundaryCoincident,
    Outside,
}

fn loop_containment(loop_pts: &[Point2], outer: &[Point2]) -> LoopContainment {
    if outer.len() < 3 {
        return LoopContainment::Outside;
    }
    let eps = super::classify_2d::boundary_eps(outer);
    let mut any_strict = false;
    for &p in loop_pts {
        let on_boundary = super::classify_2d::distance_to_polygon_boundary(p, outer) <= eps;
        if !on_boundary {
            if !super::classify_2d::point_in_polygon_2d(p, outer) {
                return LoopContainment::Outside;
            }
            any_strict = true;
        }
    }
    if any_strict {
        LoopContainment::Nested
    } else {
        LoopContainment::BoundaryCoincident
    }
}

/// Whether a `(u, v)` point lies inside a loop on a surface periodic in u
/// (and in v, a torus): a hole's samples and a split piece's outer wire can sit
/// whole turns apart when one was unwrapped into another period, so the point
/// is moved by the turns that bring it nearest the loop's extent first.
fn uv_in_loop_any_turn(p: Point2, loop_uv: &[Point2], surface: &FaceSurface) -> bool {
    use std::f64::consts::TAU;
    let u_periodic = matches!(
        surface,
        FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
    );
    let v_periodic = matches!(surface, FaceSurface::Torus(_));
    let (mut lo, mut hi) = (
        Point2::new(f64::MAX, f64::MAX),
        Point2::new(f64::MIN, f64::MIN),
    );
    for q in loop_uv {
        lo = Point2::new(lo.x().min(q.x()), lo.y().min(q.y()));
        hi = Point2::new(hi.x().max(q.x()), hi.y().max(q.y()));
    }
    // The shifts that bring the point nearest the loop's middle, one turn to
    // either side of it.
    let shifts = |periodic: bool, c: f64, a: f64, b: f64| -> [f64; 3] {
        if periodic && a <= b {
            let k = ((f64::midpoint(a, b) - c) / TAU).round() * TAU;
            [k, k - TAU, k + TAU]
        } else {
            [0.0; 3]
        }
    };
    let du = shifts(u_periodic, p.x(), lo.x(), hi.x());
    let dv = shifts(v_periodic, p.y(), lo.y(), hi.y());
    du.iter().any(|&a| {
        dv.iter().any(|&b| {
            super::classify_2d::point_in_polygon_2d(Point2::new(p.x() + a, p.y() + b), loop_uv)
        })
    })
}

/// Whether any straight section crosses a hole's boundary or ends inside it, so
/// the hole cannot stay whole in one piece.
fn sections_reach_a_hole(
    sections: &[SectionEdge],
    holes: &[Vec<OrientedPCurveEdge>],
    frame: &PlaneFrame,
) -> bool {
    holes.iter().any(|hole| {
        let polygon = sampling::sample_wire_loop_uv_via_frame(hole, frame);
        sections.iter().any(|section| {
            let (a, b) = (frame.project(section.start), frame.project(section.end));
            super::classify_2d::point_in_polygon_2d(a, &polygon)
                || super::classify_2d::point_in_polygon_2d(b, &polygon)
                || (0..polygon.len()).any(|i| {
                    let (p, q) = (polygon[i], polygon[(i + 1) % polygon.len()]);
                    seg_cross_param(a, b, p, q).is_some()
                })
        })
    })
}

/// Attach each whole hole to the sub-face that geometrically contains it.
///
/// A hole is assigned to the INNERMOST containing sub-face (the one whose own
/// interior point lies inside the most other containing sub-faces) so a hole
/// nested inside an annular region lands in the inner ring, not the outer disk.
/// Falls back to the largest sub-face when no sub-face contains the hole's
/// interior probe (degenerate sample or a hole straddling sub-face boundaries),
/// so the hole geometry is never silently dropped.
fn attach_whole_holes(sub_faces: &mut [SplitSubFace], holes: &[Vec<OrientedPCurveEdge>]) {
    if sub_faces.is_empty() || holes.is_empty() {
        return;
    }
    // On a periodic surface each loop is read unwrapped: a loop across the
    // parameter's cut read with a jump of a turn spans the whole period and
    // takes in every hole.
    let periods = |sf: &SplitSubFace| super::pcurve_compute::surface_periods(&sf.surface);
    let sub_outer_uv: Vec<Vec<Point2>> = sub_faces
        .iter()
        .map(|sf| {
            let (u, v) = periods(sf);
            sample_wire_loop_uv_periodic(&sf.outer_wire, u, v)
        })
        .collect();
    let sub_interior: Vec<Point2> = sub_outer_uv
        .iter()
        .map(|pts| super::classify_2d::sample_interior_point(pts))
        .collect();
    let largest_idx = || -> Option<usize> {
        sub_outer_uv
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                super::classify_2d::signed_area_2d(a)
                    .abs()
                    .partial_cmp(&super::classify_2d::signed_area_2d(b).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    };
    for hole in holes {
        let (u, v) = periods(&sub_faces[0]);
        let hole_pts = sample_wire_loop_uv_periodic(hole, u, v);
        if hole_pts.len() >= 3 {
            let probe = super::classify_2d::sample_interior_point(&hole_pts);
            let containing: Vec<usize> = (0..sub_faces.len())
                .filter(|&i| uv_in_loop_any_turn(probe, &sub_outer_uv[i], &sub_faces[i].surface))
                .collect();
            let best = containing.iter().copied().max_by_key(|&i| {
                containing
                    .iter()
                    .filter(|&&j| {
                        j != i
                            && super::classify_2d::point_in_polygon_2d(
                                sub_interior[i],
                                &sub_outer_uv[j],
                            )
                    })
                    .count()
            });
            if let Some(i) = best {
                sub_faces[i].inner_wires.push(hole.clone());
                continue;
            }
        }
        if let Some(idx) = largest_idx() {
            sub_faces[idx].inner_wires.push(hole.clone());
        }
    }
}

/// Whether the boundary of a u-periodic face runs counter-clockwise in
/// `(u, v)`, read as the sign of `-∮ v du` over its edges: `du` along each
/// edge does not depend on the period copy its ends sit at, so the edges can
/// come in any order. A band's two rims run a turn each way and its seam
/// copies cancel, so the integral is the face's signed area. Each edge is
/// integrated in pieces halved until a piece and its two halves agree, and
/// the sum of those differences bounds the error. `None` when the boundary
/// winds, or the integral does not clear its bound.
fn periodic_boundary_runs_ccw(
    boundary: &[OrientedPCurveEdge],
    surface: &FaceSurface,
) -> Option<bool> {
    let v0 = surface.project_point(boundary.first()?.start_3d)?.1;
    for tol in [1e-4, 1e-7, 1e-10] {
        let (mut area, mut bound, mut turn, mut size) = (0.0, 0.0, 0.0, 0.0);
        for e in boundary {
            let piece = edge_minus_v_du(e, surface, v0, tol)?;
            area += piece.area;
            bound += piece.bound;
            turn += piece.turn;
            size += piece.size;
        }
        if turn.abs() > 1e-6 {
            return None;
        }
        // Rounding in each piece's product, beyond its estimated error.
        let bound = bound + 1e-13 * size;
        if area.abs() > 2.0 * bound {
            return Some(area > 0.0);
        }
    }
    None
}

/// One boundary edge's share of `-∮ (v - v0) du`, in traversal order.
struct BoundaryIntegral {
    area: f64,
    bound: f64,
    /// The net change of `u` along the edge.
    turn: f64,
    /// The sum of `|v - v0| |du|` over the pieces, the scale of rounding.
    size: f64,
}

/// [`BoundaryIntegral`] of edge `e`, from 16 even pieces of its parameter,
/// each halved until the trapezoid over a piece and the two over its halves
/// differ by at most `tol` per unit of `|du|`.
fn edge_minus_v_du(
    e: &OrientedPCurveEdge,
    surface: &FaceSurface,
    v0: f64,
    tol: f64,
) -> Option<BoundaryIntegral> {
    use std::f64::consts::{PI, TAU};
    // A piece's ends, each its edge fraction and its `(u, v - v0)`.
    type End = (f64, (f64, f64));
    const PIECES: u32 = 16;
    const DEPTH: u32 = 14;
    let wrap = |d: f64| (d + PI).rem_euclid(TAU) - PI;
    let closed = (e.start_3d - e.end_3d).length() < 1e-10;
    let turn = if e.forward { TAU } else { -TAU };
    let (s3, e3) = if e.forward {
        (e.start_3d, e.end_3d)
    } else {
        (e.end_3d, e.start_3d)
    };
    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
    // The domain runs from the stored start (a marched curve's from either
    // end): read it backwards where it starts at the traversal's end.
    let flip = if closed {
        !e.forward
    } else {
        let p0 = e.curve_3d.evaluate_with_endpoints(t0, s3, e3);
        (p0 - e.start_3d).length() > (p0 - e.end_3d).length()
    };
    let point = |f: f64| -> Point3 {
        match &e.curve_3d {
            EdgeCurve::Line => e.start_3d + (e.end_3d - e.start_3d) * f,
            EdgeCurve::Circle(c) if closed => c.evaluate(turn.madd(f, c.project(e.start_3d))),
            EdgeCurve::Ellipse(el) if closed => el.evaluate(turn.madd(f, el.project(e.start_3d))),
            curve => {
                let g = if flip { 1.0 - f } else { f };
                curve.evaluate_with_endpoints((t1 - t0).madd(g, t0), s3, e3)
            }
        }
    };
    // An ellipse on a cylinder or a cone takes its `u` from the edge's start
    // over the run of its parameter: between two points a hair apart in it,
    // `u` can run most of a turn (a steep cut's near vertex).
    let at = |f: f64, u_near: f64| -> Option<(f64, f64)> {
        let (u, v) = surface.project_point(point(f))?;
        if let EdgeCurve::Ellipse(el) = &e.curve_3d {
            let run = if closed {
                turn * f
            } else if flip {
                -(t1 - t0) * f
            } else {
                (t1 - t0) * f
            };
            if let Some(lifted) =
                super::pcurve_compute::ellipse_u_after(el, surface, (e.start_uv.x(), u), run)
            {
                return Some((lifted, v - v0));
            }
        }
        Some((u_near + wrap(u - u_near), v - v0))
    };
    let trapezoid = |a: (f64, f64), b: (f64, f64)| -0.5 * (a.1 + b.1) * (b.0 - a.0);
    let mut out = BoundaryIntegral {
        area: 0.0,
        bound: 0.0,
        turn: 0.0,
        size: 0.0,
    };
    let mut stack: Vec<(End, End, u32)> = Vec::new();
    let mut prev = at(0.0, e.start_uv.x())?;
    let first = prev;
    for k in 1..=PIECES {
        let f = f64::from(k) / f64::from(PIECES);
        let next = at(f, prev.0)?;
        stack.push(((f - 1.0 / f64::from(PIECES), prev), (f, next), 0));
        prev = next;
    }
    out.turn = prev.0 - first.0;
    while let Some(((fa, a), (fb, b), depth)) = stack.pop() {
        let fm = 0.5 * (fa + fb);
        let m = at(fm, a.0)?;
        let one = trapezoid(a, b);
        let two = trapezoid(a, m) + trapezoid(m, b);
        let diff = (two - one).abs();
        if diff <= tol.madd((b.0 - a.0).abs(), 1e-15 * (a.1.abs() + b.1.abs())) || depth >= DEPTH {
            out.area += two;
            out.bound += diff;
            out.size += 0.5 * (a.1.abs() + b.1.abs()) * (b.0 - a.0).abs();
            continue;
        }
        stack.push(((fa, a), (fm, m), depth + 1));
        stack.push(((fm, m), (fb, b), depth + 1));
    }
    Some(out)
}

/// Whether a loop the DCEL traced on a u-periodic face is that face's
/// outside: the face stops short of a full turn, and the trace keeps every
/// region on its left, so the outside is a loop of boundary edges alone that
/// does not wind the period and runs clockwise read off the surface. A face
/// whose boundary winds clockwise in `(u, v)` (a cavity wall) traces its
/// outside along its own boundary edges, one winding counter-clockwise along
/// their reversed twins, so the test reads the loop's sense, not which
/// halves it holds. A region bounded by the boundary alone (one a section
/// loop holes) runs counter-clockwise and is kept.
fn is_periodic_face_outside(lp: &[OrientedPCurveEdge], surface: &FaceSurface) -> bool {
    use std::f64::consts::{PI, TAU};
    if lp.is_empty() || lp.iter().any(|e| e.source_edge_idx.is_some()) {
        return false;
    }
    // Each edge keeps its own run in u, and moves by whole turns to meet the
    // edge before at the vertex they share.
    let mut pts: Vec<Point2> = Vec::new();
    let mut end: Option<f64> = None;
    for (samples, edge_end) in sampling::sample_wire_edges_uv_on_surface(lp, surface) {
        let Some(start) = samples.first() else {
            continue;
        };
        let shift = end.map_or(0.0, |at| ((at - start.x()) / TAU).round() * TAU);
        pts.extend(samples.iter().map(|p| Point2::new(p.x() + shift, p.y())));
        end = Some(edge_end + shift);
    }
    let (Some(first), Some(end)) = (pts.first().copied(), end) else {
        return false;
    };
    (end - first.x()).abs() < PI && signed_area_2d(&pts) < 0.0
}

/// Whether every piece of `lp` is one of the face's own boundary pieces run
/// backwards: a pocket of the outside that the face's wire encloses on the
/// glued wall (a slot cut along a bore, across its seam), which the face
/// trace returns as an orbit of its boundary's twins. It is no region of the
/// face.
fn runs_boundary_backwards(
    lp: &[OrientedPCurveEdge],
    boundary: &[OrientedPCurveEdge],
    tol: f64,
) -> bool {
    let weld = tol * 100.0;
    !lp.is_empty()
        && lp.iter().all(|e| {
            e.source_edge_idx.is_none()
                && boundary.iter().any(|b| {
                    std::mem::discriminant(&b.curve_3d) == std::mem::discriminant(&e.curve_3d)
                        && (b.start_3d - e.end_3d).length() <= weld
                        && (b.end_3d - e.start_3d).length() <= weld
                        && (b.start_3d - b.end_3d).length() > weld
                })
                && !boundary.iter().any(|b| {
                    std::mem::discriminant(&b.curve_3d) == std::mem::discriminant(&e.curve_3d)
                        && (b.start_3d - e.start_3d).length() <= weld
                        && (b.end_3d - e.end_3d).length() <= weld
                })
        })
}

/// True when any traced loop's sampled UV polygon is area-degenerate — the
/// classifier's sliver guard would silently drop it, so the loops path
/// under-represents the face even though the loop COUNT looks fine. The
/// canonical case: a completed 4-way socket-junction circle traced as a
/// 2-arc closed loop whose pcurve-sampled polygon folds to ~zero area while
/// the true disc is πr² (the blind-recess cap the result must keep).
fn wire_loops_have_degenerate_area(loops: &[Vec<OrientedPCurveEdge>], tol: f64) -> bool {
    loops
        .iter()
        .any(|wl| loop_has_degenerate_area(wl, tol, None))
}

/// The signed area a loop on a plane encloses in `frame`: the polygon of its
/// pieces' ends plus, for each circle or ellipse piece, the segment between it
/// and its chord, swept on the piece's own span. A NURBS piece contributes
/// sampled points instead.
fn loop_area_in_frame(wire: &[OrientedPCurveEdge], frame: &PlaneFrame) -> f64 {
    let frame_normal = frame.u_axis().cross(frame.v_axis());
    let mut pts = Vec::with_capacity(wire.len());
    let mut segments = 0.0;
    for e in wire {
        let (axes, normal) = match &e.curve_3d {
            EdgeCurve::Circle(c) => (c.radius() * c.radius(), c.normal()),
            EdgeCurve::Ellipse(el) => (el.semi_major() * el.semi_minor(), el.normal()),
            EdgeCurve::Line => {
                pts.push(e.start_uv);
                continue;
            }
            EdgeCurve::NurbsCurve(_) => {
                pts.extend(sampling::sample_wire_loop_uv_via_frame(
                    std::slice::from_ref(e),
                    frame,
                ));
                continue;
            }
        };
        pts.push(e.start_uv);
        let (stored_start, stored_end) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let (t0, t1) = e.curve_3d.domain_with_endpoints(stored_start, stored_end);
        let sweep = t1 - t0;
        // The stored span turns counter-clockwise about the conic's normal.
        let turn = normal.dot(frame_normal).signum() * if e.forward { 1.0 } else { -1.0 };
        segments += turn * 0.5 * axes * (sweep - sweep.sin());
    }
    signed_area_2d(&pts) + segments
}

/// Whether planar sub-faces tile the area their boundary encloses: every outer
/// wire winding the same way (the arrangement traces each region
/// counter-clockwise, whichever way the face itself winds), no region's
/// interior inside another's, and their areas summing to the boundary's.
/// Only outlines of lines and conics are measured exactly, so with any NURBS
/// piece nothing more is checked: an arrangement over a NURBS outline can
/// leave out a region the boolean discards anyway (a lip's ring coincident
/// with the rim it is fused onto), and sampled areas cannot tell that from a
/// lost one. Where the walker's own loops are broken (`walker_broken`), the
/// arrangement over a NURBS outline must at least wind one way: two thin
/// regions traced opposite ways (a crescent between a rim arc and a section
/// arc whose chords cross) leave the face to the rescues after it. Over sound
/// walker loops a region read the other way round does not refuse it (a
/// face split further than the walker split it, its pieces whole).
fn subfaces_cover_boundary(
    subs: &[SplitSubFace],
    boundary: &[OrientedPCurveEdge],
    frame: &PlaneFrame,
    walker_broken: bool,
) -> bool {
    let nurbs = |w: &[OrientedPCurveEdge]| {
        w.iter()
            .any(|e| matches!(e.curve_3d, EdgeCurve::NurbsCurve(_)))
    };
    let outers: Vec<f64> = subs
        .iter()
        .map(|s| loop_area_in_frame(&s.outer_wire, frame))
        .collect();
    let mixed = outers.iter().any(|&a| a > 0.0) && outers.iter().any(|&a| a < 0.0);
    if nurbs(boundary)
        || subs
            .iter()
            .any(|s| nurbs(&s.outer_wire) || s.inner_wires.iter().any(|h| nurbs(h)))
    {
        return !(walker_broken && mixed);
    }
    if mixed {
        return false;
    }
    let enclosed = loop_area_in_frame(boundary, frame).abs();
    let regions: Vec<(Vec<Point2>, Vec<Vec<Point2>>)> = subs
        .iter()
        .map(|s| {
            (
                sampling::sample_wire_loop_uv_via_frame(&s.outer_wire, frame),
                s.inner_wires
                    .iter()
                    .map(|h| sampling::sample_wire_loop_uv_via_frame(h, frame))
                    .collect(),
            )
        })
        .collect();
    let holds = |(outer, holes): &(Vec<Point2>, Vec<Vec<Point2>>), p: Point2| {
        super::classify_2d::point_in_polygon_2d(p, outer)
            && !holes
                .iter()
                .any(|h| super::classify_2d::point_in_polygon_2d(p, h))
    };
    for (i, region) in regions.iter().enumerate() {
        let p = super::classify_2d::sample_interior_point(&region.0);
        if holds(region, p)
            && regions
                .iter()
                .enumerate()
                .any(|(j, other)| j != i && holds(other, p))
        {
            return false;
        }
    }
    let covered: f64 = subs
        .iter()
        .zip(&outers)
        .map(|(s, outer)| {
            outer.abs()
                - s.inner_wires
                    .iter()
                    .map(|h| loop_area_in_frame(h, frame).abs())
                    .sum::<f64>()
        })
        .sum();
    (covered - enclosed).abs() <= 1e-6 * enclosed.max(1.0)
}

/// Whether one loop encloses no area. On a plane face the loop is sampled
/// arc-true through its frame, as the loop classification does: the pcurve
/// sampler can fold a reversed boundary arc.
fn loop_has_degenerate_area(
    wl: &[OrientedPCurveEdge],
    tol: f64,
    plane: Option<&PlaneFrame>,
) -> bool {
    let pts = plane.map_or_else(
        || sample_wire_loop_uv(wl),
        |frame| sampling::sample_wire_loop_uv_via_frame(wl, frame),
    );
    if pts.len() < 3 {
        return true;
    }
    let area = signed_area_2d(&pts);
    let mut perimeter: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    if let (Some(first), Some(last)) = (pts.first(), pts.last()) {
        perimeter += (*last - *first).length();
    }
    area.abs() <= perimeter * tol
}

/// Greedy loops that enclose area. A zero-area loop (a section paired with
/// its own reverse, where a smooth boundary split at the section's end read as
/// a turn) is no region, so it must not count against a DCEL trace.
fn greedy_region_count(
    loops: &[Vec<OrientedPCurveEdge>],
    tol: f64,
    plane: Option<&PlaneFrame>,
) -> usize {
    loops
        .iter()
        .filter(|lp| !loop_has_degenerate_area(lp, tol, plane))
        .count()
}

/// Split a wire loop at UV vertices it visits more than once (a "pinch"): a
/// grand-tour trace that absorbed a sub-region as an excursion is separated
/// into the sub-cycle and the rest, recursively. Pure out-and-back excursions
/// separate into zero-area remnants, which are dropped (their edges' other
/// orientation still lives in the sibling region's loop). Comparison is on
/// the raw (unwrapped) UV, so a full-period band's two seam copies — one
/// period apart — never read as a pinch.
fn split_loop_at_pinch_vertices(
    wire: &[OrientedPCurveEdge],
    tol: f64,
) -> Vec<Vec<OrientedPCurveEdge>> {
    if wire.len() < 4 {
        return vec![wire.to_vec()];
    }
    let qscale = 1.0 / tol.max(1e-12);
    #[allow(clippy::cast_possible_truncation)]
    let qkey = |p: brepkit_math::vec::Point2| -> (i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
        )
    };
    let mut seen: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();
    let mut pinch: Option<(usize, usize)> = None;
    for (i, e) in wire.iter().enumerate() {
        if let Some(&j) = seen.get(&qkey(e.start_uv)) {
            pinch = Some((j, i));
            break;
        }
        seen.insert(qkey(e.start_uv), i);
    }
    let Some((j, i)) = pinch else {
        return vec![wire.to_vec()];
    };
    let sub: Vec<OrientedPCurveEdge> = wire[j..i].to_vec();
    let mut rest: Vec<OrientedPCurveEdge> = wire[..j].to_vec();
    rest.extend_from_slice(&wire[i..]);
    let keep = |lp: Vec<OrientedPCurveEdge>| -> Vec<Vec<OrientedPCurveEdge>> {
        if lp.is_empty() {
            return Vec::new();
        }
        let pts = sample_wire_loop_uv(&lp);
        let mut perimeter: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
        if let (Some(first), Some(last)) = (pts.first(), pts.last()) {
            perimeter += (*last - *first).length();
        }
        if signed_area_2d(&pts).abs() <= perimeter * tol {
            return Vec::new(); // out-and-back remnant
        }
        split_loop_at_pinch_vertices(&lp, tol)
    };
    let mut out = keep(sub);
    out.extend(keep(rest));
    out
}

/// True when any wire loop revisits a UV vertex — the signature of a
/// self-crossing trace from the angular wire builder, which the arrangement
/// decomposition can replace with simple (non-self-intersecting) regions even
/// when it produces FEWER loops. A simple closed loop visits each vertex once;
/// a figure-eight or out-and-back revisits one.
///
/// Detection is vertex-topological: it tests only the edges' endpoints
/// (`start_uv`). That is exactly the failure mode this gate targets — the
/// angular builder over-splits by walking out-and-back through a shared UV
/// vertex (see `remove_pendant_sections`), so the bad trace always reuses a
/// vertex. It deliberately does NOT detect a self-crossing that occurs only
/// along an edge's interior (e.g. an arc whose curved path crosses another
/// edge's chord in UV between their endpoints, with no shared vertex). No
/// wire-builder trace produces such a crossing here, so testing arc interiors
/// would add cost without changing any outcome.
fn wire_loops_self_cross(loops: &[Vec<OrientedPCurveEdge>], tol: f64) -> bool {
    let qscale = 1.0 / tol.max(1e-12);
    let qkey = |p: brepkit_math::vec::Point2| -> (i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
        )
    };
    for wire in loops {
        if wire.len() < 3 {
            continue;
        }
        let mut seen: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
        for e in wire {
            if !seen.insert(qkey(e.start_uv)) {
                return true;
            }
        }
    }
    false
}

/// Break co-endpoint circle-arc collisions inside adopted loops by splitting
/// every colliding arc at its angular midpoint — the sanctioned splitter-side
/// pattern for [`merge_duplicate_edges`]'s endpoint-pair key. Two
/// complementary half-rims share both endpoints; left whole, the merge
/// collapses them into one edge used out-and-back, the spur excision then
/// removes that pair, and the freed seam pair becomes wrap-adjacent and is
/// excised too — the cone∪box inscribed-rim fuse lost its entire base rim
/// and base disc this way. Splitting at midpoints gives every piece a unique
/// endpoint pair; `split_arc_edges_at_collinear_vertices` propagates the same
/// cuts to the neighbouring face's coincident rim so both partitions match.
///
/// Only groups whose members have DISTINCT arc midpoints are split: identical
/// twins (the same rim arc appearing once per adjacent band) must stay whole
/// so the merge can weld them, and complementary halves are exactly the
/// distinct-midpoint case. Line pairs are never touched — a cut-annulus band
/// legitimately traverses its seam edge twice, and that pair merging into one
/// edge used forward+reverse is correct topology.
fn split_coendpoint_loop_arcs(loops: &mut [Vec<OrientedPCurveEdge>], surface: &FaceSurface) {
    use std::collections::HashMap;
    use std::f64::consts::TAU;

    type QP = (i64, i64, i64);
    const QSCALE: f64 = 1e7;
    let q3 = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * QSCALE).round() as i64,
            (p.y() * QSCALE).round() as i64,
            (p.z() * QSCALE).round() as i64,
        )
    };
    let arc_mid = |e: &OrientedPCurveEdge| -> Option<Point3> {
        let EdgeCurve::Circle(ref c) = e.curve_3d else {
            return None;
        };
        let (ns, ne) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let s_ang = c.project(ns);
        let span = (c.project(ne) - s_ang).rem_euclid(TAU);
        if span < 1e-9 {
            return None; // closed or degenerate arc
        }
        Some(c.evaluate(s_ang + 0.5 * span))
    };

    let mut pair_mids: HashMap<(QP, QP), Vec<QP>> = HashMap::new();
    for lp in loops.iter() {
        for e in lp {
            let Some(mid) = arc_mid(e) else { continue };
            let (a, b) = (q3(e.start_3d), q3(e.end_3d));
            let key = if a <= b { (a, b) } else { (b, a) };
            pair_mids.entry(key).or_default().push(q3(mid));
        }
    }
    let colliding: std::collections::HashSet<(QP, QP)> = pair_mids
        .into_iter()
        .filter(|(_, mids)| mids.iter().any(|m| *m != mids[0]))
        .map(|(k, _)| k)
        .collect();
    if colliding.is_empty() {
        return;
    }

    for lp in loops.iter_mut() {
        let mut rebuilt: Vec<OrientedPCurveEdge> = Vec::with_capacity(lp.len() + 2);
        for e in lp.drain(..) {
            let (a, b) = (q3(e.start_3d), q3(e.end_3d));
            let key = if a <= b { (a, b) } else { (b, a) };
            let mid = if colliding.contains(&key) {
                arc_mid(&e)
            } else {
                None
            };
            let Some(mid_3d) = mid else {
                rebuilt.push(e);
                continue;
            };
            let Some((mu_raw, mv)) = surface.project_point(mid_3d) else {
                rebuilt.push(e);
                continue;
            };
            // Each piece takes the period copy of the projected mid-u nearest
            // its OWN retained endpoint. A seam-wrapping arc (stored e.g.
            // 4.71→1.57 through 2π) has its true midpoint at the wrap; the
            // endpoints' average sits half a period away and would fold the
            // piece across the middle of the domain. The two pieces may
            // legitimately land on different copies — that is exactly the
            // unwrapped representation of a seam-crossing split.
            let nearest_copy = |anchor: f64| -> f64 {
                anchor + (mu_raw - anchor + TAU / 2.0).rem_euclid(TAU) - TAU / 2.0
            };
            let mid_uv_first = Point2::new(nearest_copy(e.start_uv.x()), mv);
            let mid_uv_second = Point2::new(nearest_copy(e.end_uv.x()), mv);
            // A cross-section rim arc's UV image is a straight constant-v
            // segment, so the pieces get an exact Line pcurve. Keeping the
            // parent's curved pcurve would make period-aware samplers walk
            // the WHOLE parent curve per piece (they sample non-Line pcurves
            // over their full range), folding the loop polygon to zero area.
            let piece_pcurve = |a: Point2, b: Point2| -> Option<brepkit_math::curves2d::Curve2D> {
                if (a.y() - b.y()).abs() > 1e-9 {
                    return None;
                }
                let d = b - a;
                if d.length() < 1e-12 {
                    return None;
                }
                brepkit_math::curves2d::Line2D::new(a, d)
                    .ok()
                    .map(brepkit_math::curves2d::Curve2D::Line)
            };
            // Pieces drop the parent's pave block id, matching
            // `presplit_sections_at_registry`: the id keys a cross-face
            // edge-sharing cache, so two distinct sub-spans carrying the
            // parent's id could alias; the position-based merge welds the
            // shared pieces instead.
            let mut first = e.clone();
            first.end_3d = mid_3d;
            first.end_uv = mid_uv_first;
            first.source_edge_idx = None;
            first.pave_block_id = None;
            if let Some(pc) = piece_pcurve(first.start_uv, first.end_uv) {
                first.pcurve = pc;
            }
            let mut second = e;
            second.start_3d = mid_3d;
            second.start_uv = mid_uv_second;
            second.source_edge_idx = None;
            second.pave_block_id = None;
            if let Some(pc) = piece_pcurve(second.start_uv, second.end_uv) {
                second.pcurve = pc;
            }
            rebuilt.push(first);
            rebuilt.push(second);
        }
        *lp = rebuilt;
    }
}

/// The seam meridian's `u` from a face's boundary edges: the (unique)
/// non-degenerate boundary Line whose endpoints project to one `u` is the
/// generator the periodic face was cut open along.
fn boundary_seam_u(
    boundary_edges: &[OrientedPCurveEdge],
    surface: &FaceSurface,
    tol: f64,
) -> Option<f64> {
    use std::f64::consts::TAU;
    // A pointed cone's seam runs up to its apex, which has no u.
    let at_apex = |p: Point3| match surface {
        FaceSurface::Cone(cone) => (p - cone.apex()).length() <= tol,
        _ => false,
    };
    for e in boundary_edges {
        if !matches!(e.curve_3d, EdgeCurve::Line) || (e.start_3d - e.end_3d).length() <= tol {
            continue;
        }
        if at_apex(e.start_3d) || at_apex(e.end_3d) {
            let rim = if at_apex(e.start_3d) {
                e.end_3d
            } else {
                e.start_3d
            };
            if let Some((u, _)) = surface.project_point(rim) {
                return Some(u);
            }
            continue;
        }
        let (Some((u0, _)), Some((u1, _))) = (
            surface.project_point(e.start_3d),
            surface.project_point(e.end_3d),
        ) else {
            continue;
        };
        let d = (u1 - u0).rem_euclid(TAU);
        if d.min(TAU - d) < 1e-4 {
            return Some(u0);
        }
    }
    None
}

/// Split every section piece at each point where it crosses the seam
/// meridian ([`seam_crossings`]). Pieces already ending on the seam, or not
/// crossing it, pass through unchanged. Split pieces clear their UV hints
/// and pave block id, matching `presplit_sections_at_registry`.
fn split_sections_at_seam_meridian(
    sections: &[SectionEdge],
    surface: &FaceSurface,
    seam_u: f64,
    wire_pts: &[Point3],
    tol: f64,
) -> Vec<SectionEdge> {
    let mut out = Vec::with_capacity(sections.len() + 2);
    for s in sections {
        let crossings = seam_crossings(s, surface, seam_u, tol);
        if crossings.is_empty() {
            out.push(s.clone());
            continue;
        }
        let ends: Vec<Point3> = std::iter::once(s.start)
            .chain(crossings)
            .chain(std::iter::once(s.end))
            .collect();
        for w in ends.windows(2) {
            let mut piece = s.clone();
            piece.start = w[0];
            piece.end = w[1];
            piece.start_uv_a = None;
            piece.end_uv_a = None;
            piece.start_uv_b = None;
            piece.end_uv_b = None;
            piece.pave_block_id = None;
            // A fitted pcurve spans the whole section, and its ends would
            // stand for the piece's ends; a NURBS pcurve reads its UV ends off
            // its own ends too: each piece gets its own on this face.
            if !matches!(piece.curve_3d, EdgeCurve::Line)
                || matches!(piece.pcurve_a, brepkit_math::curves2d::Curve2D::Nurbs(_))
                || matches!(piece.pcurve_b, brepkit_math::curves2d::Curve2D::Nurbs(_))
            {
                let pcurve = compute_pcurve_on_surface(
                    &piece.curve_3d,
                    piece.start,
                    piece.end,
                    surface,
                    wire_pts,
                    None,
                );
                piece.pcurve_a = pcurve.clone();
                piece.pcurve_b = pcurve;
            }
            out.push(piece);
        }
    }
    out
}

/// The points, in order from its start, where section `s` crosses the seam
/// meridian at `seam_u`: where its `u`, lifted along it, passes a copy of
/// the seam's. A crossing within `10 tol` of an end is that end's own. A
/// closed section stays whole, as the band and hole splitters read it.
///
/// The `u` is lifted from samples along the span the section's pcurve
/// traces ([`super::pcurve_compute::edge_point_at`]: the shorter arc between
/// the ends of an open circle or ellipse, whichever way its own sense runs),
/// by the one-way lift along an ellipse on a cylinder or a cone, and from
/// sample to sample the short way round along any other curve. A piece's
/// ends alone, read the short way round, cannot tell a piece over half a
/// turn (a steep cone's ellipse across its near vertex) that crosses the
/// seam from one that does not. Each crossing is bisected between the two
/// samples it lies between, on the same span, with `u` lifted from the
/// first of them the same way.
fn seam_crossings(s: &SectionEdge, surface: &FaceSurface, seam_u: f64, tol: f64) -> Vec<Point3> {
    use std::f64::consts::{PI, TAU};
    const SAMPLES: u32 = 32;
    let wrap = |d: f64| -> f64 { (d + PI).rem_euclid(TAU) - PI };
    if (s.start - s.end).length() <= tol {
        return Vec::new();
    }
    let at = super::pcurve_compute::edge_point_at(&s.curve_3d, s.start, s.end);
    let param = |k: u32| f64::from(k) / f64::from(SAMPLES);
    let pts: Vec<Point3> = (0..=SAMPLES).map(|k| at(param(k))).collect();
    let Some(uvs) = super::pcurve_compute::edge_uv_samples(&s.curve_3d, surface, &pts) else {
        return Vec::new();
    };
    // Offsets from the seam, the start's in (-pi, pi], so the seam's copies
    // lie at whole turns.
    let shift = wrap(uvs[0].x() - seam_u) - uvs[0].x();
    let offsets: Vec<f64> = uvs.iter().map(|p| p.x() + shift).collect();
    // The offset at fraction `t` of the span, lifted from the offset `from`
    // at fraction `t_from`, at most a thirty-second of the span before it.
    let lifted = |t: f64, (t_from, from): (f64, f64)| -> Option<f64> {
        let p = at(t);
        let (u, _) = surface.project_point(p)?;
        let ellipse = match &s.curve_3d {
            EdgeCurve::Ellipse(e) => {
                let run =
                    super::pcurve_compute::shorter_arc_delta(e.project(p) - e.project(at(t_from)));
                super::pcurve_compute::ellipse_u_after(e, surface, (from - shift, u), run)
            }
            EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::NurbsCurve(_) => None,
        };
        Some(ellipse.map_or_else(|| from + wrap(u + shift - from), |u| u + shift))
    };
    let mut found: Vec<Point3> = Vec::new();
    for k in 0..SAMPLES {
        let (a, b) = (offsets[k as usize], offsets[k as usize + 1]);
        let (lo, hi) = (a.min(b), a.max(b));
        // Each copy of the seam in (lo, hi], so one met at a sample counts
        // in one bracket.
        for copy in (lo / TAU).floor() as i64 + 1..=(hi / TAU).floor() as i64 {
            let target = TAU * copy as f64;
            let (mut t_lo, mut t_hi) = (param(k), param(k + 1));
            let below = a < target;
            for _ in 0..60 {
                let mid = 0.5 * (t_lo + t_hi);
                let Some(off) = lifted(mid, (param(k), a)) else {
                    break;
                };
                if (off < target) == below {
                    t_lo = mid;
                } else {
                    t_hi = mid;
                }
            }
            let p = at(0.5 * (t_lo + t_hi));
            let on_seam = surface
                .project_point(p)
                .is_some_and(|(u, _)| wrap(u - seam_u).abs() < 1e-6);
            let clear = |q: Point3| (p - q).length() > tol * 10.0;
            if on_seam && clear(s.start) && clear(s.end) && found.last().is_none_or(|&q| clear(q)) {
                found.push(p);
            }
        }
    }
    found
}

/// Split a u-periodic cylinder/cone lateral into stacked bands along
/// seam-anchored section chains that each wind the periodic direction once:
/// the chain generalization of [`split_periodic_face_into_bands`], whose
/// separators must be closed circles. Mirrors its structure: same boundary
/// preconditions (a `band_stack`: two closed rims, or one and a cone's apex,
/// plus seam lines), same per-band wire
/// shape (lower separator, seam up, upper separator, seam down), and an
/// interior point per band. A bore through a tube's wall crosses the bore's
/// own wall along two such loops, giving three bands.
///
/// Sections that wind nothing close into loops clear of the seam: each is a
/// hole in the band it lies in, and a disc of its own (a rail through a post
/// whose end sits in another rail).
///
/// Returns `None` (caller falls through) unless: the boundary reads as a
/// band stack; at least one section chain winds the lateral; each chain has a
/// vertex on the seam meridian (the seam-anchoring pre-step guarantees this
/// for winding chains); every chain sample stays strictly between the
/// stack's ends; the chains do not cross (their order along the seam is
/// their order across the face); and every other section belongs to a loop
/// of winding zero clear of the seam inside one band.
#[allow(clippy::too_many_lines)]
fn split_periodic_face_by_winding_chain(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    let (_, period) = special_cases::band_period(surface)?;

    let mut chains: Vec<Vec<(usize, bool)>> = Vec::new();
    let mut free: Vec<usize> = (0..sections.len()).collect();
    while !free.is_empty() {
        let subset: Vec<SectionEdge> = free.iter().map(|&i| sections[i].clone()).collect();
        let Some((chain, _winding)) = winding_chain_in(&subset, surface, tol, true, period) else {
            break;
        };
        let used: Vec<usize> = chain.iter().map(|&(k, _)| free[k]).collect();
        chains.push(chain.iter().map(|&(k, fwd)| (free[k], fwd)).collect());
        free.retain(|i| !used.contains(i));
    }
    if chains.is_empty() {
        return None;
    }
    // The rest must close into holes; a section left out would be dropped.
    let hole_loops = closed_section_loops(sections, &free, tol)?;
    bands_from_winding_chains(
        surface,
        boundary_edges,
        sections,
        chains,
        &hole_loops,
        rank,
        reversed,
        face_id,
        tol,
    )
}

/// The bands of a u-periodic lateral cut by `chains`, each a closed chain of
/// `sections` (`(index, traversed start→end)`) winding the period once, and
/// the discs of `hole_loops`, closed loops of sections winding nothing, each
/// a hole in the band it lies in: the band builder of
/// [`split_periodic_face_by_winding_chain`]. `None` unless every chain
/// reaches the seam meridian, stays strictly between the rims, and the
/// chains stack without crossing, and every loop lies clear of the seam
/// inside one band.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn bands_from_winding_chains(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    chains: Vec<Vec<(usize, bool)>>,
    hole_loops: &[Vec<(usize, bool)>],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    struct Separator {
        v_seam: f64,
        v_opposite: f64,
        lower: Vec<OrientedPCurveEdge>,
        upper: Vec<OrientedPCurveEdge>,
        /// `(u` past the seam, unwrapped along the chain`, v)`.
        path: Vec<(f64, f64)>,
    }

    let (u_start, period) = special_cases::band_period(surface)?;
    let half = 0.5 * period;
    let close_tol = tol * 100.0;

    let special_cases::BandStack {
        seam_u,
        v_bot,
        bot,
        v_top,
        top,
        q_bot,
        q_top,
        lower_tan: ref_tan,
        ..
    } = special_cases::band_stack(surface, boundary_edges, close_tol)?;
    let wrap_pi = |d: f64| -> f64 { (d + half).rem_euclid(period) - half };
    let u_opposite = (seam_u + half - u_start).rem_euclid(period) + u_start;

    let traversal_start = |&(idx, fwd): &(usize, bool)| -> Point3 {
        let s = &sections[idx];
        if fwd { s.start } else { s.end }
    };
    let mut separators: Vec<Separator> = Vec::with_capacity(chains.len());
    for chain in chains {
        // Rotate the chain to start at its seam-anchored vertex.
        let seam_pos = chain.iter().position(|entry| {
            surface
                .project_point(traversal_start(entry))
                .is_some_and(|(u, _)| wrap_pi(u - seam_u).abs() < 1e-6)
        })?;
        let mut chain = chain;
        chain.rotate_left(seam_pos);
        let (_, v_seam) = surface.project_point(traversal_start(&chain[0]))?;

        // Every sample strictly between the rims; the one nearest the
        // antipodal meridian gives the chain's v there.
        let mut samples_uv: Vec<(f64, f64)> = Vec::new();
        for &(idx, _) in &chain {
            let s = &sections[idx];
            let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            for k in 0..=8 {
                let t = d0 + (d1 - d0) * (f64::from(k) / 8.0);
                let p = s.curve_3d.evaluate_with_endpoints(t, s.start, s.end);
                let (u, v) = surface.project_point(p)?;
                if v < v_bot + close_tol || v > v_top - close_tol {
                    return None;
                }
                samples_uv.push((u, v));
            }
        }
        let v_opposite = samples_uv
            .iter()
            .min_by(|a, b| {
                wrap_pi(a.0 - u_opposite)
                    .abs()
                    .total_cmp(&wrap_pi(b.0 - u_opposite).abs())
            })
            .map(|&(_, v)| v)?;

        let chain_tan = {
            let (idx, fwd) = chain[0];
            let s = &sections[idx];
            let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            let t_at = if fwd { d0 } else { d1 };
            let tan = s.curve_3d.tangent_with_endpoints(t_at, s.start, s.end);
            if fwd { tan } else { -tan }
        };
        let chain_is_lower_role = chain_tan.dot(ref_tan) > 0.0;

        // Materialize the chain as pcurve edges in a given traversal
        // direction, walking UV u with nearest-copy continuity from the seam.
        let build_chain = |as_ordered: bool| -> Option<Vec<OrientedPCurveEdge>> {
            let entries: Vec<(usize, bool)> = if as_ordered {
                chain.clone()
            } else {
                chain.iter().rev().map(|&(i, f)| (i, !f)).collect()
            };
            let mut out = Vec::with_capacity(entries.len());
            let mut u_prev = seam_u;
            for (idx, fwd) in entries {
                let s = &sections[idx];
                let (from, to) = if fwd {
                    (s.start, s.end)
                } else {
                    (s.end, s.start)
                };
                let (u_raw0, v0) = surface.project_point(from)?;
                let (u_raw1, v1) = surface.project_point(to)?;
                let u0 = u_prev + wrap_pi(u_raw0 - u_prev);
                let u1 = ellipse_section_u_run(s, fwd, surface)
                    .map_or_else(|| u0 + wrap_pi(u_raw1 - u_raw0), |run| u0 + run);
                u_prev = u1;
                let pcurve = match rank {
                    Rank::A => s.pcurve_a.clone(),
                    Rank::B => s.pcurve_b.clone(),
                };
                out.push(OrientedPCurveEdge {
                    curve_3d: s.curve_3d.clone(),
                    pcurve,
                    start_uv: Point2::new(u0, v0),
                    end_uv: Point2::new(u1, v1),
                    start_3d: from,
                    end_3d: to,
                    forward: fwd,
                    source_edge_idx: None,
                    pave_block_id: s.pave_block_id,
                });
            }
            Some(out)
        };
        // Read finer than the samples above: a hole may lie close beside it.
        let mut path: Vec<(f64, f64)> = Vec::new();
        for &(idx, fwd) in &chain {
            let s = &sections[idx];
            let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            let uv_at = |f: f64| {
                let t = if fwd { f } else { 1.0 - f };
                surface.project_point(s.curve_3d.evaluate_with_endpoints(
                    (d1 - d0).madd(t, d0),
                    s.start,
                    s.end,
                ))
            };
            push_uv_path(&mut path, seam_u, period, 0.5 * close_tol, &uv_at)?;
        }
        separators.push(Separator {
            v_seam,
            v_opposite,
            lower: build_chain(chain_is_lower_role)?,
            upper: build_chain(!chain_is_lower_role)?,
            path,
        });
    }

    // Stack the separators; crossing chains leave no consistent order.
    separators.sort_by(|a, b| a.v_seam.total_cmp(&b.v_seam));
    if separators
        .windows(2)
        .any(|w| w[1].v_opposite <= w[0].v_opposite || w[1].v_seam - w[0].v_seam < close_tol)
    {
        return None;
    }

    let mk_seam = |va: f64, vb: f64| -> Option<OrientedPCurveEdge> {
        let pa = surface.evaluate(seam_u, va)?;
        let pb = surface.evaluate(seam_u, vb)?;
        let dir = brepkit_math::vec::Vec2::new(0.0, if vb > va { 1.0 } else { -1.0 });
        let pcurve = brepkit_math::curves2d::Curve2D::Line(
            brepkit_math::curves2d::Line2D::new(Point2::new(seam_u, va), dir).ok()?,
        );
        Some(OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: Point2::new(seam_u, va),
            end_uv: Point2::new(seam_u, vb),
            start_3d: pa,
            end_3d: pb,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        })
    };

    // Each hole lies in one band: the one below every separator and rim
    // that passes above it, read at each of its samples.
    let n = separators.len();
    let end_path = |end: &special_cases::BandEnd<'_>, v: f64| -> Option<Vec<(f64, f64)>> {
        let special_cases::BandEnd::Rim(rim) = end else {
            return Some(vec![(-half, v), (half, v)]);
        };
        let (t0, t1) = rim.curve_3d.domain_with_endpoints(rim.start_3d, rim.end_3d);
        let uv_at = |f: f64| {
            surface.project_point(rim.curve_3d.evaluate_with_endpoints(
                (t1 - t0).madd(f, t0),
                rim.start_3d,
                rim.end_3d,
            ))
        };
        let mut path: Vec<(f64, f64)> = Vec::new();
        push_uv_path(&mut path, seam_u, period, 0.5 * close_tol, &uv_at)?;
        Some(path)
    };
    let mut levels: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n + 2);
    levels.push(end_path(&bot, v_bot)?);
    levels.extend(separators.iter().map(|s| s.path.clone()));
    levels.push(end_path(&top, v_top)?);
    let band_of = |d: f64, v: f64| -> Option<usize> {
        let mut below = 0;
        for level in &levels {
            let at = winding_path_v_at(level, d, period)?;
            if (at - v).abs() < close_tol {
                return None;
            }
            if at < v {
                below += 1;
            }
        }
        (1..=n + 1).contains(&below).then(|| below - 1)
    };
    // Whether the band's lower edge runs toward increasing `u`, which makes
    // its outer wire run counterclockwise in `(u, v)`.
    let along_u = {
        let mid = f64::midpoint(v_bot, v_top);
        let du = surface.evaluate(seam_u + 1e-4, mid)? - surface.evaluate(seam_u - 1e-4, mid)?;
        ref_tan.dot(du) > 0.0
    };
    // Samples along a section on a cylinder or a cone come lifted
    // ([`section_uv_run`]); on a periodic spline wall they come raw.
    let lifted = super::pcurve_compute::surface_periods(surface).0.is_some();
    // Each loop's band, `(u, v)` samples (`u` past the seam, kept on one side
    // of it), disc and hole wires.
    let mut holes: Vec<HoleLoop> = Vec::with_capacity(hole_loops.len());
    for entries in hole_loops {
        let mut uv: Vec<Point2> = Vec::new();
        let mut band: Option<usize> = None;
        // Each entry's samples lifted along it, and carried on from the
        // entry before at the vertex they share.
        let mut end: Option<f64> = None;
        for &(idx, fwd) in entries {
            let run = section_uv_run(&sections[idx], fwd, surface, HOLE_SAMPLES)?;
            let first = run.first()?.x();
            let mut d = end.map_or_else(
                || wrap_pi(first - seam_u),
                |e| e + wrap_pi(first - seam_u - e),
            );
            for (k, p) in run.iter().enumerate() {
                if k > 0 {
                    let step = p.x() - run[k - 1].x();
                    d += if lifted { step } else { wrap_pi(step) };
                }
                if k + 1 == run.len() {
                    end = Some(d);
                    break;
                }
                let in_band = band_of(wrap_pi(p.x() - seam_u), p.y())?;
                if band.is_some_and(|b| b != in_band) {
                    return None;
                }
                band = Some(in_band);
                uv.push(Point2::new(d, p.y()));
            }
        }
        let band = band?;
        // An inner wire cannot cross the seam the face is cut open along.
        let (lo, hi) = uv.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
            (lo.min(p.x()), hi.max(p.x()))
        });
        let clear = |a: f64, b: f64| a >= HOLE_SEAM_CLEARANCE && b <= period - HOLE_SEAM_CLEARANCE;
        if !clear(lo, hi) && !clear(-hi, -lo) {
            return None;
        }
        if hi < 0.0 {
            for p in &mut uv {
                *p = Point2::new(p.x() + period, p.y());
            }
        }
        let face_left = (signed_area_2d(&uv) > 0.0) == along_u;
        let reversed_entries: Vec<(usize, bool)> =
            entries.iter().rev().map(|&(i, f)| (i, !f)).collect();
        let (disc, hole) = if face_left {
            (entries.clone(), reversed_entries)
        } else {
            (reversed_entries, entries.clone())
        };
        let materialize = |loop_entries: &[(usize, bool)]| -> Option<Vec<OrientedPCurveEdge>> {
            let mut out = Vec::with_capacity(loop_entries.len());
            let mut u_prev: Option<f64> = None;
            for &(idx, fwd) in loop_entries {
                let s = &sections[idx];
                let (from, to) = if fwd {
                    (s.start, s.end)
                } else {
                    (s.end, s.start)
                };
                let (u_raw0, v0) = surface.project_point(from)?;
                let (u_raw1, v1) = surface.project_point(to)?;
                let u0 = u_prev.map_or_else(
                    || {
                        let past = (u_raw0 - seam_u).rem_euclid(period);
                        if along_u {
                            seam_u + past
                        } else {
                            seam_u + past - period
                        }
                    },
                    |prev| prev + wrap_pi(u_raw0 - prev),
                );
                let u1 = ellipse_section_u_run(s, fwd, surface)
                    .map_or_else(|| u0 + wrap_pi(u_raw1 - u_raw0), |run| u0 + run);
                u_prev = Some(u1);
                out.push(OrientedPCurveEdge {
                    curve_3d: s.curve_3d.clone(),
                    pcurve: match rank {
                        Rank::A => s.pcurve_a.clone(),
                        Rank::B => s.pcurve_b.clone(),
                    },
                    start_uv: Point2::new(u0, v0),
                    end_uv: Point2::new(u1, v1),
                    start_3d: from,
                    end_3d: to,
                    forward: fwd,
                    source_edge_idx: None,
                    pave_block_id: s.pave_block_id,
                });
            }
            Some(out)
        };
        holes.push(HoleLoop {
            band,
            disc: materialize(&disc)?,
            hole: materialize(&hole)?,
            uv,
            parent: None,
        });
    }
    // A loop inside another (a hollow rail's bore inside its wall) is a hole
    // of the disc around it, not of the band: each loop's parent is the
    // smallest loop holding it.
    for i in 0..holes.len() {
        let probe = *holes[i].uv.first()?;
        holes[i].parent = (0..holes.len())
            .filter(|&j| {
                j != i
                    && holes[j].band == holes[i].band
                    && super::classify_2d::point_in_polygon_2d(probe, &holes[j].uv)
            })
            .min_by(|&a, &b| {
                signed_area_2d(&holes[a].uv)
                    .abs()
                    .total_cmp(&signed_area_2d(&holes[b].uv).abs())
            });
    }
    let mut band_holes: Vec<Vec<Vec<OrientedPCurveEdge>>> = vec![Vec::new(); n + 1];
    let mut band_hole_uv: Vec<Vec<Vec<Point2>>> = vec![Vec::new(); n + 1];
    let mut discs: Vec<SplitSubFace> = Vec::new();
    for (i, h) in holes.iter().enumerate() {
        let children: Vec<&HoleLoop> = holes.iter().filter(|c| c.parent == Some(i)).collect();
        let child_uv: Vec<&[Point2]> = children.iter().map(|c| c.uv.as_slice()).collect();
        let inside = point_inside_region_uv(&h.uv, &child_uv)?;
        discs.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: h.disc.clone(),
            inner_wires: children.iter().map(|c| c.hole.clone()).collect(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(surface.evaluate(seam_u + inside.x(), inside.y())?),
        });
        if h.parent.is_none() {
            band_holes[h.band].push(h.hole.clone());
            band_hole_uv[h.band].push(h.uv.clone());
        }
    }

    // Each band: its lower separator in the lower role, the seam up, its
    // upper separator in the upper role, the seam back down. The rims close
    // the end bands.
    let mut bands = Vec::with_capacity(n + 1 + discs.len());
    for k in 0..=n {
        let (mut wire, v_low, q_low) = if k == 0 {
            (bot.edges(), v_bot, q_bot)
        } else {
            let below = &separators[k - 1];
            (below.lower.clone(), below.v_seam, below.v_opposite)
        };
        let (v_high, q_high) = if k == n {
            (v_top, q_top)
        } else {
            (separators[k].v_seam, separators[k].v_opposite)
        };
        wire.push(mk_seam(v_low, v_high)?);
        if k == n {
            wire.extend(top.edges());
        } else {
            wire.extend(separators[k].upper.iter().cloned());
        }
        wire.push(mk_seam(v_high, v_low)?);
        // A sample on the meridian opposite the seam, clear of the band's
        // holes.
        let d_opposite = wrap_pi(u_opposite - seam_u);
        let v_inside = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
            .into_iter()
            .map(|f| (q_high - q_low).madd(f, q_low))
            .find(|&v| {
                band_hole_uv[k].iter().all(|uv| {
                    [d_opposite - period, d_opposite, d_opposite + period]
                        .into_iter()
                        .all(|d| !super::classify_2d::point_in_polygon_2d(Point2::new(d, v), uv))
                })
            })?;
        bands.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: std::mem::take(&mut band_holes[k]),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(surface.evaluate(u_opposite, v_inside)?),
        });
    }
    bands.extend(discs);
    Some(bands)
}

/// Split a u-periodic cylinder/cone lateral along two saddles that touch:
/// the section of two crossing cylinders of one radius, four half ellipses
/// joining the two points where the cylinders touch. On each wall two of the
/// halves run each way round between those points; the lower of each side
/// form one saddle and the upper the other, each winding the wall once, and
/// between them lie two lenses (inside the other cylinder) that meet only at
/// the touch points. The wall splits into the band below the lower saddle,
/// the band above the upper one, and the two lenses. `None` (the caller
/// falls through) unless the sections are exactly such four paths.
#[allow(clippy::too_many_lines)]
fn split_periodic_face_by_touching_saddles(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    const SAMPLES: usize = 32;
    let (_, period) = special_cases::band_period(surface)?;
    let half = 0.5 * period;
    let weld = tol * 100.0;
    let wrap = |d: f64| -> f64 { (d + half).rem_euclid(period) - half };
    let seam_u = boundary_seam_u(boundary_edges, surface, tol)?;

    // Cut each section where it crosses the seam meridian, found along the
    // section itself: a half ellipse ends half a turn from where it starts,
    // so its ends alone do not tell which way round it runs.
    let mut pieces: Vec<SectionEdge> = Vec::with_capacity(sections.len() + 2);
    for s in sections {
        if (s.start - s.end).length() <= weld {
            return None;
        }
        let (t0, t1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
        let at = |t: f64| s.curve_3d.evaluate_with_endpoints(t, s.start, s.end);
        let off = |t: f64| surface.project_point(at(t)).map(|(u, _)| wrap(u - seam_u));
        let mut cut = None;
        for k in 0..SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let (ta, tb) = (
                t0 + (t1 - t0) * k as f64 / SAMPLES as f64,
                t0 + (t1 - t0) * (k + 1) as f64 / SAMPLES as f64,
            );
            let (fa, fb) = (off(ta)?, off(tb)?);
            if fa * fb < 0.0 && (fa - fb).abs() < half {
                let (mut lo, mut hi) = (ta, tb);
                for _ in 0..60 {
                    let tm = 0.5 * (lo + hi);
                    if (off(tm)? > 0.0) == (fa > 0.0) {
                        lo = tm;
                    } else {
                        hi = tm;
                    }
                }
                cut = Some(at(0.5 * (lo + hi)));
                break;
            }
        }
        match cut {
            Some(p) if (p - s.start).length() > weld && (p - s.end).length() > weld => {
                let mut first = s.clone();
                first.end = p;
                let mut second = s.clone();
                second.start = p;
                for piece in [&mut first, &mut second] {
                    piece.start_uv_a = None;
                    piece.end_uv_a = None;
                    piece.start_uv_b = None;
                    piece.end_uv_b = None;
                    piece.pave_block_id = None;
                }
                pieces.push(first);
                pieces.push(second);
            }
            _ => pieces.push(s.clone()),
        }
    }

    // The touch points: the two ends that four pieces meet at. Every other
    // end (a seam cut) joins exactly two.
    let mut ends: Vec<(Point3, usize)> = Vec::new();
    for s in &pieces {
        for p in [s.start, s.end] {
            match ends.iter_mut().find(|(q, _)| (*q - p).length() <= weld) {
                Some(entry) => entry.1 += 1,
                None => ends.push((p, 1)),
            }
        }
    }
    let touch: Vec<Point3> = ends.iter().filter(|e| e.1 == 4).map(|e| e.0).collect();
    if touch.len() != 2 || ends.iter().any(|e| e.1 != 4 && e.1 != 2) {
        return None;
    }
    let (p, q) = (touch[0], touch[1]);

    // The four paths from p to q, each piece as (index, traversed start→end).
    let mut used = vec![false; pieces.len()];
    let mut paths: Vec<Vec<(usize, bool)>> = Vec::new();
    for first in 0..pieces.len() {
        let leaves = if (pieces[first].start - p).length() <= weld {
            true
        } else if (pieces[first].end - p).length() <= weld {
            false
        } else {
            continue;
        };
        let mut path = vec![(first, leaves)];
        used[first] = true;
        loop {
            let &(i, fwd) = path.last()?;
            let to = if fwd { pieces[i].end } else { pieces[i].start };
            if (to - q).length() <= weld {
                break;
            }
            if (to - p).length() <= weld || path.len() > pieces.len() {
                return None;
            }
            let next = (0..pieces.len()).find_map(|j| {
                if used[j] {
                    None
                } else if (pieces[j].start - to).length() <= weld {
                    Some((j, true))
                } else if (pieces[j].end - to).length() <= weld {
                    Some((j, false))
                } else {
                    None
                }
            })?;
            used[next.0] = true;
            path.push(next);
        }
        paths.push(path);
    }
    if paths.len() != 4 || used.iter().any(|u| !u) {
        return None;
    }

    // Each path's (u, v) from p, u unwrapped along it.
    let path_uv = |path: &[(usize, bool)]| -> Option<Vec<(f64, f64)>> {
        let mut out: Vec<(f64, f64)> = Vec::new();
        for &(i, fwd) in path {
            let s = &pieces[i];
            let (t0, t1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            for k in 0..=SAMPLES {
                #[allow(clippy::cast_precision_loss)]
                let f = k as f64 / SAMPLES as f64;
                let f = if fwd { f } else { 1.0 - f };
                let pt = s
                    .curve_3d
                    .evaluate_with_endpoints(t0 + (t1 - t0) * f, s.start, s.end);
                let (u, v) = surface.project_point(pt)?;
                let u = out.last().map_or(u, |&(pu, _)| pu + wrap(u - pu));
                out.push((u, v));
            }
        }
        Some(out)
    };
    let uvs: Vec<Vec<(f64, f64)>> = paths
        .iter()
        .map(|path| path_uv(path))
        .collect::<Option<_>>()?;
    let turn = |uv: &[(f64, f64)]| uv.last().map_or(0.0, |l| l.0) - uv[0].0;
    // v at the middle of a side, read off a path at the u nearest it.
    let v_at = |uv: &[(f64, f64)], u: f64| {
        uv.iter()
            .min_by(|a, b| (a.0 - u).abs().total_cmp(&(b.0 - u).abs()))
            .map_or(0.0, |s| s.1)
    };

    // Two paths each way round, the lower and the upper of each side.
    let mut sides: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (k, uv) in uvs.iter().enumerate() {
        sides[usize::from(turn(uv) < 0.0)].push(k);
    }
    let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(2);
    for side in &sides {
        let [a, b] = side[..] else {
            return None;
        };
        let u_mid = uvs[a][0].0 + 0.5 * turn(&uvs[a]);
        let (va, vb) = (v_at(&uvs[a], u_mid), v_at(&uvs[b], u_mid));
        if (va - vb).abs() <= weld {
            return None;
        }
        let (lo, hi) = if va < vb { (a, b) } else { (b, a) };
        pairs.push((lo, hi, u_mid));
    }
    let reversed_path = |path: &[(usize, bool)]| -> Vec<(usize, bool)> {
        path.iter().rev().map(|&(i, f)| (i, !f)).collect()
    };
    let saddle = |a: usize, b: usize| -> Vec<(usize, bool)> {
        let mut chain = paths[a].clone();
        chain.extend(reversed_path(&paths[b]));
        chain
    };
    let chains = vec![
        saddle(pairs[0].0, pairs[1].0),
        saddle(pairs[0].1, pairs[1].1),
    ];
    let bands = bands_from_winding_chains(
        surface,
        boundary_edges,
        &pieces,
        chains,
        &[],
        rank,
        reversed,
        face_id,
        tol,
    )?;
    let [low, _pinched, high] = <[SplitSubFace; 3]>::try_from(bands).ok()?;

    // The lenses: on each side, out along the lower path and back along the
    // upper, wound counter-clockwise about the surface normal as every
    // stored outer wire is.
    let mut out = vec![low, high];
    for &(lo, hi, u_mid) in &pairs {
        let mut loop_entries = paths[lo].clone();
        loop_entries.extend(reversed_path(&paths[hi]));
        let mut wire: Vec<OrientedPCurveEdge> = Vec::with_capacity(loop_entries.len());
        let mut u_prev: Option<f64> = None;
        for (i, fwd) in loop_entries {
            let s = &pieces[i];
            let (from, to) = if fwd {
                (s.start, s.end)
            } else {
                (s.end, s.start)
            };
            let (u_raw0, v0) = surface.project_point(from)?;
            let (u_raw1, v1) = surface.project_point(to)?;
            let u0 = u_prev.map_or(u_raw0, |up| up + wrap(u_raw0 - up));
            let (t0, t1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            let mid = s
                .curve_3d
                .evaluate_with_endpoints(0.5 * (t0 + t1), s.start, s.end);
            let (u_mid_raw, _) = surface.project_point(mid)?;
            let u_half = u0 + wrap(u_mid_raw - u_raw0);
            let u1 = u_half + wrap(u_raw1 - u_mid_raw);
            u_prev = Some(u1);
            wire.push(OrientedPCurveEdge {
                curve_3d: s.curve_3d.clone(),
                pcurve: match rank {
                    Rank::A => s.pcurve_a.clone(),
                    Rank::B => s.pcurve_b.clone(),
                },
                start_uv: Point2::new(u0, v0),
                end_uv: Point2::new(u1, v1),
                start_3d: from,
                end_3d: to,
                forward: fwd,
                source_edge_idx: None,
                pave_block_id: s.pave_block_id,
            });
        }
        let mut poly: Vec<(f64, f64)> = uvs[lo].clone();
        poly.extend(uvs[hi].iter().rev().copied());
        let trapezoid: f64 = poly
            .iter()
            .zip(poly.iter().cycle().skip(1))
            .map(|(a, b)| (b.0 - a.0) * (b.1 + a.1))
            .sum();
        if trapezoid >= 0.0 {
            wire.reverse();
            for edge in &mut wire {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        }
        let v_mid = 0.5 * (v_at(&uvs[lo], u_mid) + v_at(&uvs[hi], u_mid));
        out.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(surface.evaluate(u_mid, v_mid)?),
        });
    }
    Some(out)
}

/// A loop of sections on a lateral that winds nothing: the band holding it,
/// its wires as a disc and as a hole, its `(u, v)` samples, and the loop
/// holding it, if any.
struct HoleLoop {
    band: usize,
    disc: Vec<OrientedPCurveEdge>,
    hole: Vec<OrientedPCurveEdge>,
    uv: Vec<Point2>,
    parent: Option<usize>,
}

/// Samples along each piece of a hole, or along a rim, read on a lateral.
const HOLE_SAMPLES: u32 = 32;
/// How near the seam meridian, in `u`, a hole may come: one crossing it
/// would need cutting there, which an inner wire cannot take.
const HOLE_SEAM_CLEARANCE: f64 = 1e-6;

/// The sections at `free` chained by their ends into closed loops, as
/// `(section, traversed start to end)` entries; `None` if one does not close.
fn closed_section_loops(
    sections: &[SectionEdge],
    free: &[usize],
    tol: f64,
) -> Option<Vec<Vec<(usize, bool)>>> {
    use std::collections::{HashMap, HashSet};

    let qscale = 1.0 / tol.max(1e-12);
    let q3 = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
            (p.z() * qscale).round() as i64,
        )
    };
    let mut adj: HashMap<(i64, i64, i64), Vec<(usize, bool)>> = HashMap::new();
    for &i in free {
        adj.entry(q3(sections[i].start))
            .or_default()
            .push((i, true));
        adj.entry(q3(sections[i].end)).or_default().push((i, false));
    }
    let mut used: HashSet<usize> = HashSet::new();
    let mut loops = Vec::new();
    for &first in free {
        if used.contains(&first) {
            continue;
        }
        let origin = q3(sections[first].start);
        let (mut cur, mut fwd) = (first, true);
        let mut entries = Vec::new();
        loop {
            used.insert(cur);
            entries.push((cur, fwd));
            let s = &sections[cur];
            let to = if fwd { s.end } else { s.start };
            if q3(to) == origin {
                break;
            }
            let &(next, next_fwd) = adj.get(&q3(to))?.iter().find(|(j, _)| !used.contains(j))?;
            cur = next;
            fwd = next_fwd;
        }
        loops.push(entries);
    }
    Some(loops)
}

/// Appends to `path` the `(u` past `seam_u`, unwrapped along the path`, v)`
/// samples of the curve that `uv_at` reads over `[0, 1]`: [`HOLE_SAMPLES`]
/// even steps, each halved while the curve's middle lies farther than
/// `within` in `v` from the line between its ends, so a `v` read between two
/// samples lies about that close to the curve.
fn push_uv_path(
    path: &mut Vec<(f64, f64)>,
    seam_u: f64,
    period: f64,
    within: f64,
    uv_at: &dyn Fn(f64) -> Option<(f64, f64)>,
) -> Option<()> {
    /// Halvings of one even step at most.
    const DEPTH: u32 = 10;
    let half = 0.5 * period;
    let past = |from: f64, u: f64| from + ((u - seam_u - from + half).rem_euclid(period) - half);
    let (u, v) = uv_at(0.0)?;
    path.push((past(path.last().map_or(0.0, |p| p.0), u), v));
    let mut left = 0.0;
    for k in 1..=HOLE_SAMPLES {
        let f = f64::from(k) / f64::from(HOLE_SAMPLES);
        let (u, v) = uv_at(f)?;
        let mut pending = vec![(f, u, v, 0)];
        while let Some(&(right, ur, vr, depth)) = pending.last() {
            let &(dl, vl) = path.last()?;
            let mid = f64::midpoint(left, right);
            let (um, vm) = uv_at(mid)?;
            let dm = past(dl, um);
            let dr = past(dm, ur);
            let chord = if (dr - dl).abs() > f64::EPSILON {
                (vr - vl).madd((dm - dl) / (dr - dl), vl)
            } else {
                vm
            };
            if depth < DEPTH && (vm - chord).abs() > within {
                pending.push((mid, um, vm, depth + 1));
            } else {
                pending.pop();
                path.push((past(dl, ur), vr));
                left = right;
            }
        }
    }
    Some(())
}

/// The `v` where a path winding a lateral once, as `(u` past the seam
/// unwrapped along it`, v)` samples, passes `u` past the seam `d`; `None`
/// where it passes there more than once or not at all.
fn winding_path_v_at(path: &[(f64, f64)], d: f64, period: f64) -> Option<f64> {
    let mut found: Option<f64> = None;
    for turn in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let target = period.madd(turn, d);
        for w in path.windows(2) {
            let ((d0, v0), (d1, v1)) = (w[0], w[1]);
            let (lo, hi) = if d0 < d1 { (d0, d1) } else { (d1, d0) };
            if (hi - lo).abs() < f64::EPSILON || target < lo || target >= hi {
                continue;
            }
            let v = (v1 - v0).madd((target - d0) / (d1 - d0), v0);
            match found {
                Some(prev) if (prev - v).abs() > 1e-9 => return None,
                _ => found = Some(v),
            }
        }
    }
    found
}

/// A point inside the closed `(u, v)` polygon `outer` and outside each of
/// `holes`: the middle of the widest span they leave, read by parity, on the
/// line through the mean `v` of `outer`.
fn point_inside_region_uv(outer: &[Point2], holes: &[&[Point2]]) -> Option<Point2> {
    #[allow(clippy::cast_precision_loss)]
    let v = outer.iter().map(|p| p.y()).sum::<f64>() / outer.len() as f64;
    let mut cuts: Vec<f64> = Vec::new();
    for poly in std::iter::once(outer).chain(holes.iter().copied()) {
        for k in 0..poly.len() {
            let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
            if (a.y() <= v) != (b.y() <= v) {
                cuts.push((b.x() - a.x()).madd((v - a.y()) / (b.y() - a.y()), a.x()));
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.chunks_exact(2)
        .max_by(|a, b| (a[1] - a[0]).total_cmp(&(b[1] - b[0])))
        .map(|span| Point2::new(f64::midpoint(span[0], span[1]), v))
}

/// The `(u, v)` of `samples + 1` points of section `s` at even steps of its
/// domain, read from its start when `forward` and from its end when not,
/// with `u` lifted along the section ([`super::pcurve_compute::edge_uv_samples`])
/// from the first point's own `u`.
fn section_uv_run(
    s: &SectionEdge,
    forward: bool,
    surface: &FaceSurface,
    samples: u32,
) -> Option<Vec<Point2>> {
    let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
    let pts: Vec<Point3> = (0..=samples)
        .map(|k| {
            let f = f64::from(k) / f64::from(samples);
            let f = if forward { f } else { 1.0 - f };
            s.curve_3d
                .evaluate_with_endpoints((d1 - d0).madd(f, d0), s.start, s.end)
        })
        .collect();
    super::pcurve_compute::edge_uv_samples(&s.curve_3d, surface, &pts)
}

/// The run in `u` of section `s` from its start to its end when `forward`,
/// and back when not, where it is an ellipse on a cylinder or a cone: by
/// [`super::pcurve_compute::ellipse_u_after`] over the run of its parameter, with no
/// samples to step over a turn. `None` for another curve or surface.
fn ellipse_section_u_run(s: &SectionEdge, forward: bool, surface: &FaceSurface) -> Option<f64> {
    let EdgeCurve::Ellipse(ellipse) = &s.curve_3d else {
        return None;
    };
    let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
    let (from, to, run) = if forward {
        (s.start, s.end, d1 - d0)
    } else {
        (s.end, s.start, d0 - d1)
    };
    let (u0, _) = surface.project_point(from)?;
    let (u1, _) = surface.project_point(to)?;
    Some(super::pcurve_compute::ellipse_u_after(ellipse, surface, (u0, u1), run)? - u0)
}

/// Split a u-periodic cylinder/cone lateral (two closed rims, or a pointed
/// cone's one rim and its apex, plus the seam) along closed section loops of
/// winding zero when some of them straddle the seam meridian (a bore through a
/// tube's wall at the wall's seam). The seam cannot run through a hole, so each
/// straddling hole notches the remainder's outer wire: going up one copy of the
/// seam the wire detours around the hole's half on that side, and coming down
/// the other copy around the other half. The seam's piece inside the hole
/// bounds the two half-discs the hole leaves. Loops clear of the seam stay
/// inner wires, each with its disc. The seam-anchor pre-pass cuts a straddling
/// loop at its seam crossings, so each such hole meets the seam at two section
/// vertices.
///
/// Returns `None` (caller falls through) unless the boundary is exactly two
/// closed rims plus seam edges (a pointed cone's apex standing in for one
/// rim), every section belongs to a closed loop of winding zero lying
/// strictly between the rims, every seam crossing is a section vertex, at
/// least one loop meets the seam and each that does meets it exactly twice,
/// and those holes' seam spans do not overlap.
#[allow(clippy::too_many_lines)]
fn split_periodic_face_around_seam_holes(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    use std::collections::HashMap;
    use std::f64::consts::{PI, TAU};
    const SAMPLES: u32 = 16;

    struct Loop {
        entries: Vec<(usize, bool)>,
        seam_at: Vec<usize>,
        area: f64,
        centroid_uv: (f64, f64),
    }
    struct Straddle {
        v_a: f64,
        v_b: f64,
        up: Vec<(usize, bool)>,
        down: Vec<(usize, bool)>,
    }

    if !matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) || sections.is_empty() {
        return None;
    }
    let close_tol = tol * 100.0;
    let wrap_pi = |d: f64| -> f64 { (d + PI).rem_euclid(TAU) - PI };

    let mut boundary_circles: Vec<&OrientedPCurveEdge> = Vec::new();
    let mut seam_edges: Vec<&OrientedPCurveEdge> = Vec::new();
    for e in boundary_edges {
        let is_closed = (e.start_3d - e.end_3d).length() < close_tol;
        match (&e.curve_3d, is_closed) {
            (EdgeCurve::Circle(_), true) => boundary_circles.push(e),
            (EdgeCurve::Line, false) => seam_edges.push(e),
            _ => return None,
        }
    }
    // A pointed cone's wall has one rim: its seam runs up to the apex and
    // back, and the apex stands in for the other rim.
    let apex = match surface {
        FaceSurface::Cone(cone) if boundary_circles.len() == 1 => Some(cone.apex()),
        _ => None,
    };
    if seam_edges.is_empty() || (boundary_circles.len() != 2 && apex.is_none()) {
        return None;
    }
    let off_apex = |p: Point3| apex.is_none_or(|a| (p - a).length() > close_tol);
    let seam_foot = [seam_edges[0].start_3d, seam_edges[0].end_3d]
        .into_iter()
        .find(|&p| off_apex(p))?;
    let (seam_u, _) = surface.project_point(seam_foot)?;
    let circle_v = |e: &OrientedPCurveEdge| -> Option<f64> {
        let (_, v) = surface.project_point(e.start_3d)?;
        let on_seam = surface.evaluate(seam_u, v)?;
        ((on_seam - e.start_3d).length() < close_tol).then_some(v)
    };
    let mut rims: Vec<(f64, Option<&OrientedPCurveEdge>)> = boundary_circles
        .iter()
        .map(|&e| circle_v(e).map(|v| (v, Some(e))))
        .collect::<Option<_>>()?;
    if let Some(a) = apex {
        rims.push((surface.project_point(a)?.1, None));
    }
    rims.sort_by(|a, b| a.0.total_cmp(&b.0));
    let [(v_bot, bot_edge), (v_top, top_edge)] = rims[..] else {
        return None;
    };
    if v_top - v_bot < close_tol {
        return None;
    }
    // Whether a rim leaves the seam toward +u; the two rims of a wall run
    // opposite ways.
    let leaves_up = |rim: &OrientedPCurveEdge| -> Option<bool> {
        let EdgeCurve::Circle(c) = &rim.curve_3d else {
            return None;
        };
        let t0 = c.project(rim.start_3d);
        let step = if rim.forward { 0.25 } else { -0.25 };
        let (u, _) = surface.project_point(c.evaluate(t0 + step))?;
        Some(wrap_pi(u - seam_u) > 0.0)
    };
    // Whether the bottom rim leaves the seam toward +u: the face lies on the
    // other side of the seam copy the wire climbs, and on this side of the
    // one it descends.
    let rim_ascends = match (bot_edge, top_edge) {
        (Some(bot), _) => leaves_up(bot)?,
        (None, Some(top)) => !leaves_up(top)?,
        (None, None) => return None,
    };

    let on_seam = |p: Point3| -> bool {
        surface
            .project_point(p)
            .and_then(|(_, v)| surface.evaluate(seam_u, v))
            .is_some_and(|q| (q - p).length() < close_tol)
    };
    let traversal = |&(idx, fwd): &(usize, bool)| -> (Point3, Point3) {
        let s = &sections[idx];
        if fwd {
            (s.start, s.end)
        } else {
            (s.end, s.start)
        }
    };
    // (offset from the seam, v) samples along an entry in traversal order,
    // the offset lifted along the entry ([`section_uv_run`]) from the first
    // sample's, read within half a turn of the seam.
    let entry_samples = |entry: &(usize, bool)| -> Option<Vec<(f64, f64)>> {
        let uv = section_uv_run(&sections[entry.0], entry.1, surface, SAMPLES)?;
        let first = uv.first()?.x() - seam_u;
        let shift = wrap_pi(first) - first;
        Some(uv.iter().map(|p| (p.x() - seam_u + shift, p.y())).collect())
    };
    // Each entry's samples carried on from the entry before at the vertex
    // they share.
    let carried = |entries: &[(usize, bool)]| -> Option<Vec<Vec<(f64, f64)>>> {
        let mut out: Vec<Vec<(f64, f64)>> = Vec::with_capacity(entries.len());
        for entry in entries {
            let mut run = entry_samples(entry)?;
            if let Some(&(prev, _)) = out.last().and_then(|r| r.last()) {
                let shift = ((prev - run.first()?.0) / TAU).round() * TAU;
                for p in &mut run {
                    p.0 += shift;
                }
            }
            out.push(run);
        }
        Some(out)
    };

    let qscale = 1.0 / tol.max(1e-12);
    let q3 = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
            (p.z() * qscale).round() as i64,
        )
    };
    let mut adj: HashMap<(i64, i64, i64), Vec<(usize, bool)>> = HashMap::new();
    for (i, s) in sections.iter().enumerate() {
        adj.entry(q3(s.start)).or_default().push((i, true));
        adj.entry(q3(s.end)).or_default().push((i, false));
    }
    let mut used = vec![false; sections.len()];
    let mut loops: Vec<Loop> = Vec::new();
    for first in 0..sections.len() {
        if used[first] {
            continue;
        }
        let origin = q3(sections[first].start);
        let mut entries = Vec::new();
        let (mut cur, mut fwd) = (first, true);
        loop {
            used[cur] = true;
            entries.push((cur, fwd));
            let (_, to) = traversal(&(cur, fwd));
            if q3(to) == origin {
                break;
            }
            let &(next, next_fwd) = adj.get(&q3(to))?.iter().find(|(j, _)| !used[*j])?;
            cur = next;
            fwd = next_fwd;
        }
        // Unwrapped UV walk: winding, area, centroid; crossings only at
        // vertices; strictly between the rims.
        let mut uv: Vec<(f64, f64)> = Vec::new();
        for samples in carried(&entries)? {
            for (k, &(d, v)) in samples.iter().enumerate() {
                if v < v_bot + close_tol || v > v_top - close_tol {
                    return None;
                }
                if k > 0 && k + 1 < samples.len() && wrap_pi(d).abs() < 1e-9 {
                    return None;
                }
                // Two inner samples on either side of a copy of the seam.
                if k > 1
                    && k + 1 < samples.len()
                    && ((samples[k - 1].0 / TAU).floor() - (d / TAU).floor()).abs() > 0.5
                {
                    return None;
                }
                if k + 1 < samples.len() {
                    uv.push((d, v));
                }
            }
        }
        let (u_first, _) = *uv.first()?;
        let (u_last, _) = *uv.last()?;
        if (u_last - u_first).abs() > PI {
            return None;
        }
        let mut area = 0.0;
        for k in 0..uv.len() {
            let (a, b) = (uv[k], uv[(k + 1) % uv.len()]);
            area += a.0 * b.1 - b.0 * a.1;
        }
        #[allow(clippy::cast_precision_loss)]
        let count = uv.len() as f64;
        let centroid_uv = (
            uv.iter().map(|p| p.0).sum::<f64>() / count,
            uv.iter().map(|p| p.1).sum::<f64>() / count,
        );
        let seam_at: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| on_seam(traversal(entry).0))
            .map(|(k, _)| k)
            .collect();
        if !seam_at.is_empty() && seam_at.len() != 2 {
            return None;
        }
        loops.push(Loop {
            entries,
            seam_at,
            area: 0.5 * area,
            centroid_uv,
        });
    }
    if loops.iter().all(|l| l.seam_at.is_empty()) {
        return None;
    }

    // An entry's run in u, read along it: a piece over half a turn ends on
    // the turn it runs to, not on the copy nearest its start.
    let entry_run = |entry: &(usize, bool)| -> Option<f64> {
        let samples = entry_samples(entry)?;
        Some(samples.last()?.0 - samples.first()?.0)
    };
    let materialize = |entries: &[(usize, bool)]| -> Option<Vec<OrientedPCurveEdge>> {
        let mut out = Vec::with_capacity(entries.len());
        let mut u_prev = seam_u;
        for entry in entries {
            let s = &sections[entry.0];
            let (from, to) = traversal(entry);
            let (u_raw0, v0) = surface.project_point(from)?;
            let (u_raw1, v1) = surface.project_point(to)?;
            let u0 = u_prev + wrap_pi(u_raw0 - u_prev);
            let walked = u0 + entry_run(entry)?;
            let u1 = walked + wrap_pi(u_raw1 - walked);
            u_prev = u1;
            out.push(OrientedPCurveEdge {
                curve_3d: s.curve_3d.clone(),
                pcurve: match rank {
                    Rank::A => s.pcurve_a.clone(),
                    Rank::B => s.pcurve_b.clone(),
                },
                start_uv: Point2::new(u0, v0),
                end_uv: Point2::new(u1, v1),
                start_3d: from,
                end_3d: to,
                forward: entry.1,
                source_edge_idx: None,
                pave_block_id: s.pave_block_id,
            });
        }
        Some(out)
    };
    let reversed_entries = |entries: &[(usize, bool)]| -> Vec<(usize, bool)> {
        entries.iter().rev().map(|&(i, f)| (i, !f)).collect()
    };
    let mk_seam = |va: f64, vb: f64| -> Option<OrientedPCurveEdge> {
        let pa = surface.evaluate(seam_u, va)?;
        let pb = surface.evaluate(seam_u, vb)?;
        let dir = brepkit_math::vec::Vec2::new(0.0, if vb > va { 1.0 } else { -1.0 });
        let pcurve = brepkit_math::curves2d::Curve2D::Line(
            brepkit_math::curves2d::Line2D::new(Point2::new(seam_u, va), dir).ok()?,
        );
        Some(OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: Point2::new(seam_u, va),
            end_uv: Point2::new(seam_u, vb),
            start_3d: pa,
            end_3d: pb,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        })
    };
    // A point inside a half-disc on the seam from `v_a` to `v_b`: at their
    // mid-height, halfway from the seam to where the half's chain first
    // crosses that height. A point level with one of the chain's samples
    // can leave the half where its edge slopes back toward the seam.
    let half_interior = |half: &[(usize, bool)], v_a: f64, v_b: f64| -> Option<Point3> {
        let v_mid = 0.5 * (v_a + v_b);
        // Offsets from the seam carried along the chain: a half wider than
        // half a turn runs past the opposite meridian.
        let chain: Vec<(f64, f64)> = carried(half)?.into_iter().flatten().collect();
        let mut nearest: Option<f64> = None;
        for w in chain.windows(2) {
            let ((d0, v0), (d1, v1)) = (w[0], w[1]);
            if (v0 - v_mid) * (v1 - v_mid) > 0.0 || (v1 - v0).abs() < f64::EPSILON {
                continue;
            }
            let d = d0 + (d1 - d0) * (v_mid - v0) / (v1 - v0);
            if d.abs() > 1e-12 && nearest.is_none_or(|n| d.abs() < n.abs()) {
                nearest = Some(d);
            }
        }
        surface.evaluate(seam_u + 0.5 * nearest?, v_mid)
    };

    let mut straddles: Vec<Straddle> = Vec::new();
    let mut sub_faces: Vec<SplitSubFace> = Vec::new();
    let mut holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
    let mut all_loops: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
    for lp in &loops {
        all_loops.push(materialize(&lp.entries)?);
        if lp.seam_at.is_empty() {
            let face_left = (lp.area > 0.0) == rim_ascends;
            let (disc, hole) = if face_left {
                (lp.entries.clone(), reversed_entries(&lp.entries))
            } else {
                (reversed_entries(&lp.entries), lp.entries.clone())
            };
            holes.push(materialize(&hole)?);
            sub_faces.push(SplitSubFace {
                surface: surface.clone(),
                outer_wire: materialize(&disc)?,
                inner_wires: Vec::new(),
                reversed,
                parent: face_id,
                rank,
                precomputed_interior: Some(
                    surface.evaluate(seam_u + lp.centroid_uv.0, lp.centroid_uv.1)?,
                ),
            });
            continue;
        }
        let mut entries = lp.entries.clone();
        entries.rotate_left(lp.seam_at[0]);
        let split = lp.seam_at[1] - lp.seam_at[0];
        let (first, second) = entries.split_at(split);
        let (_, v_first) = surface.project_point(traversal(&first[0]).0)?;
        let (_, v_second) = surface.project_point(traversal(&second[0]).0)?;
        // Offsets from the seam carried along the half from its seam vertex:
        // a half over half a turn runs past the opposite meridian, where a
        // wrapped offset changes sign.
        let side = |half: &[(usize, bool)]| -> Option<bool> {
            let mut sum = 0.0;
            for run in carried(half)? {
                let last = run.len() - 1;
                sum += run[1..last].iter().map(|&(d, _)| d).sum::<f64>();
            }
            Some(sum > 0.0)
        };
        let (s_first, s_second) = (side(first)?, side(second)?);
        if s_first == s_second {
            return None;
        }
        // `first` runs from its seam vertex to the other one; orient both
        // halves from the lower seam vertex (a) up to the upper (b).
        let (v_a, v_b, first_up, second_up) = if v_first < v_second {
            (v_first, v_second, first.to_vec(), reversed_entries(second))
        } else {
            (v_second, v_first, reversed_entries(first), second.to_vec())
        };
        let (up, down_a_to_b) = if s_first == rim_ascends {
            (second_up, first_up)
        } else {
            (first_up, second_up)
        };
        straddles.push(Straddle {
            v_a,
            v_b,
            up,
            down: reversed_entries(&down_a_to_b),
        });
    }
    straddles.sort_by(|a, b| a.v_a.total_cmp(&b.v_a));
    let mut v_floor = v_bot;
    for s in &straddles {
        if s.v_a < v_floor + close_tol || s.v_b < s.v_a + close_tol {
            return None;
        }
        v_floor = s.v_b;
    }
    if v_floor > v_top - close_tol {
        return None;
    }

    let mut outer: Vec<OrientedPCurveEdge> = bot_edge.into_iter().cloned().collect();
    let mut v_cur = v_bot;
    for s in &straddles {
        outer.push(mk_seam(v_cur, s.v_a)?);
        outer.extend(materialize(&s.up)?);
        v_cur = s.v_b;
    }
    outer.push(mk_seam(v_cur, v_top)?);
    outer.extend(top_edge.cloned());
    v_cur = v_top;
    for s in straddles.iter().rev() {
        outer.push(mk_seam(v_cur, s.v_b)?);
        outer.extend(materialize(&s.down)?);
        v_cur = s.v_a;
    }
    outer.push(mk_seam(v_cur, v_bot)?);

    for s in &straddles {
        let mut up_disc = vec![mk_seam(s.v_a, s.v_b)?];
        up_disc.extend(materialize(&reversed_entries(&s.up))?);
        let mut down_disc = vec![mk_seam(s.v_b, s.v_a)?];
        down_disc.extend(materialize(&reversed_entries(&s.down))?);
        for (wire, half) in [(up_disc, &s.up), (down_disc, &s.down)] {
            sub_faces.push(SplitSubFace {
                surface: surface.clone(),
                outer_wire: wire,
                inner_wires: Vec::new(),
                reversed,
                parent: face_id,
                rank,
                precomputed_interior: Some(half_interior(half, s.v_a, s.v_b)?),
            });
        }
    }

    let probe = SplitSubFace {
        surface: surface.clone(),
        outer_wire: outer.clone(),
        inner_wires: all_loops,
        reversed,
        parent: face_id,
        rank,
        precomputed_interior: None,
    };
    let remainder_interior = cylinder_cone_remainder_interior(&probe, tol)?;
    sub_faces.insert(
        0,
        SplitSubFace {
            surface: surface.clone(),
            outer_wire: outer,
            inner_wires: holes,
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(remainder_interior),
        },
    );
    Some(sub_faces)
}

/// Whether two or more sections chain into a loop that WINDS the surface's
/// periodic u direction, measured by endpoint progress. See
/// [`winding_section_chain`].
fn sections_form_winding_chain(sections: &[SectionEdge], surface: &FaceSurface, tol: f64) -> bool {
    winding_section_chain(sections, surface, tol, false).is_some()
}

/// The ordered section chain forming a loop that WINDS the surface's
/// periodic u direction (|net signed u-progress| > π after chaining by
/// endpoint position), as `(piece index, traversed start→end)` entries, with
/// the signed winding. Non-periodic surfaces never wind. Chains that fail to
/// close are conservatively reported as non-winding — the internal-loops
/// path has its own closure requirements.
///
/// `along_pieces` measures the progress along each piece, so a lone closed
/// section winds by its full turn and a piece may turn past half a period;
/// the band builder, which needs every such chain, asks for it. Without it
/// a chain needs two sections and each piece counts its endpoints' short
/// way round: the routing tests keep that reading, under which the two
/// crossing loops of equal perpendicular cylinders stay internal loops.
fn winding_section_chain(
    sections: &[SectionEdge],
    surface: &FaceSurface,
    tol: f64,
    along_pieces: bool,
) -> Option<(Vec<(usize, bool)>, f64)> {
    let (Some(_), _) = super::pcurve_compute::surface_periods(surface) else {
        return None;
    };
    winding_chain_in(sections, surface, tol, along_pieces, std::f64::consts::TAU)
}

/// [`winding_section_chain`] on a lateral whose `u` turns over `period`.
fn winding_chain_in(
    sections: &[SectionEdge],
    surface: &FaceSurface,
    tol: f64,
    along_pieces: bool,
    period: f64,
) -> Option<(Vec<(usize, bool)>, f64)> {
    use std::collections::HashMap;

    let half = 0.5 * period;
    if sections.len() < if along_pieces { 1 } else { 2 } {
        return None;
    }
    let proj_u = |p: Point3| -> Option<f64> { surface.project_point(p).map(|(u, _)| u) };
    let qscale = 1.0 / tol.max(1e-12);
    let q3 = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
            (p.z() * qscale).round() as i64,
        )
    };
    let wrap_pi = |d: f64| -> f64 { (d + half).rem_euclid(period) - half };

    // Endpoint-keyed adjacency: (piece index, leaves-from-start).
    let mut adj: HashMap<(i64, i64, i64), Vec<(usize, bool)>> = HashMap::new();
    for (i, s) in sections.iter().enumerate() {
        adj.entry(q3(s.start)).or_default().push((i, true));
        adj.entry(q3(s.end)).or_default().push((i, false));
    }

    let mut used = vec![false; sections.len()];
    for start in 0..sections.len() {
        if used[start] {
            continue;
        }
        let mut winding = 0.0_f64;
        let mut cur = start;
        let mut forward = true;
        let origin = q3(sections[start].start);
        let mut closed = false;
        let mut chain: Vec<(usize, bool)> = Vec::new();
        for _ in 0..sections.len() {
            used[cur] = true;
            chain.push((cur, forward));
            let s = &sections[cur];
            let to = if forward { s.end } else { s.start };
            let samples: u32 = if along_pieces { 16 } else { 1 };
            let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            let at = |k: u32| {
                let f = f64::from(k) / f64::from(samples);
                let t = if forward { f } else { 1.0 - f };
                proj_u(
                    s.curve_3d
                        .evaluate_with_endpoints(d0 + (d1 - d0) * t, s.start, s.end),
                )
            };
            if let Some(run) = ellipse_section_u_run(s, forward, surface) {
                winding += run;
            } else {
                let mut u_prev = at(0)?;
                for k in 1..=samples {
                    let u = at(k)?;
                    winding += wrap_pi(u - u_prev);
                    u_prev = u;
                }
            }
            let to_key = q3(to);
            if to_key == origin {
                closed = true;
                break;
            }
            let Some(next) = adj
                .get(&to_key)
                .and_then(|c| c.iter().find(|(j, _)| !used[*j]))
            else {
                break;
            };
            forward = next.1;
            cur = next.0;
        }
        if closed && winding.abs() > half {
            return Some((chain, winding));
        }
    }
    None
}

/// [`wire_loops_have_degenerate_area`] with period-aware sampling: a valid
/// full-period band loop's raw UV polygon folds to ~zero area at the seam
/// crossing, so the absolute check misjudges it. Downstream classification
/// measures periodic loops with the same period-aware sampler, so this is the
/// faithful degeneracy judgment for a periodic face.
fn wire_loops_have_degenerate_area_periodic(
    loops: &[Vec<OrientedPCurveEdge>],
    tol: f64,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> bool {
    loops.iter().any(|wl| {
        let pts = sample_wire_loop_uv_periodic(wl, u_period, v_period);
        if pts.len() < 3 {
            return true;
        }
        let area = signed_area_2d(&pts);
        let mut perimeter: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
        if let (Some(first), Some(last)) = (pts.first(), pts.last()) {
            perimeter += (*last - *first).length();
        }
        area.abs() <= perimeter * tol
    })
}

/// `loops` with the second of any two loops running the same edges dropped:
/// a loop inside a face comes back twice, as its region and as the hole it
/// leaves in the region around it.
fn without_twin_loops(loops: &[Vec<OrientedPCurveEdge>], tol: f64) -> Vec<Vec<OrientedPCurveEdge>> {
    let q = |p: Point2| ((p.x() / tol).round() as i64, (p.y() / tol).round() as i64);
    let mut seen: Vec<Vec<[i64; 4]>> = Vec::new();
    let mut out = Vec::new();
    for lp in loops {
        let mut keys: Vec<_> = lp
            .iter()
            .map(|e| {
                let (a, b) = (q(e.start_uv), q(e.end_uv));
                let (a, b) = if a <= b { (a, b) } else { (b, a) };
                [a.0, a.1, b.0, b.1]
            })
            .collect();
        keys.sort_unstable();
        if !seen.contains(&keys) {
            seen.push(keys);
            out.push(lp.clone());
        }
    }
    out
}

/// Whether one loop's edges are entirely duplicated by another loop — the
/// signature of a full ring section whose twin copies closed as a bare circle
/// while a band tour consumed the same arcs (a box tangent to a cone/cylinder
/// on all four walls: the section circle splits into four arcs, the greedy
/// spends them on an out-and-back seam tour, and the twins close as a
/// duplicate ring that later nests as an inner wire equal to the outer's own
/// rim while the true far rim is dropped from every loop).
///
/// Edge identity is the unordered pair of quantized UV endpoints plus the 3D
/// curve kind, so a chord and its co-endpoint arc never alias.
///
/// Also detects the WITHIN-loop form: the same geometric circle arc traversed
/// twice inside one loop (the 2-tangency grand tour, where the greedy hands
/// back a single loop that rides the section ring out and back instead of
/// splitting). Within-loop identity additionally includes the arc's 3D
/// midpoint: the two half-rims of one full circle share both endpoints inside
/// a perfectly valid band loop, and only a repeat of the SAME arc is broken.
fn wire_loops_duplicate_cover(loops: &[Vec<OrientedPCurveEdge>], tol: f64) -> bool {
    let qscale = 1.0 / tol.max(1e-12);
    let qkey = |p: Point2| -> (i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
        )
    };
    let ekey = |e: &OrientedPCurveEdge| -> ((i64, i64), (i64, i64), u8) {
        let (a, b) = (qkey(e.start_uv), qkey(e.end_uv));
        let kind = match e.curve_3d {
            EdgeCurve::Line => 0u8,
            EdgeCurve::Circle(_) => 1,
            EdgeCurve::Ellipse(_) => 2,
            EdgeCurve::NurbsCurve(_) => 3,
        };
        if a <= b { (a, b, kind) } else { (b, a, kind) }
    };
    let q3 = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
            (p.z() * qscale).round() as i64,
        )
    };
    let arc_mid_key = |e: &OrientedPCurveEdge| -> Option<(i64, i64, i64)> {
        let EdgeCurve::Circle(ref c) = e.curve_3d else {
            return None;
        };
        let (ns, ne) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let s_ang = c.project(ns);
        let span = (c.project(ne) - s_ang).rem_euclid(std::f64::consts::TAU);
        if span < 1e-9 {
            return None;
        }
        Some(q3(c.evaluate(s_ang + 0.5 * span)))
    };
    for lp in loops {
        let mut seen: std::collections::HashSet<_> = std::collections::HashSet::new();
        for e in lp {
            let Some(mid) = arc_mid_key(e) else { continue };
            if !seen.insert((ekey(e), mid)) {
                return true;
            }
        }
    }
    for (i, small) in loops.iter().enumerate() {
        if small.is_empty() {
            continue;
        }
        for (j, big) in loops.iter().enumerate() {
            if i == j || big.len() < small.len() {
                continue;
            }
            let keys: std::collections::HashSet<_> = big.iter().map(&ekey).collect();
            if small.iter().all(|e| keys.contains(&ekey(e))) {
                return true;
            }
        }
    }
    false
}

/// Whether the greedy wire loops form an INVALID (overlapping) partition: one
/// OUTER (positive-area) loop's material directly covers another outer loop,
/// with no hole region between them.
///
/// The angular wire builder partitions a plane face into loops, but when the
/// dividing sections form an incomplete interior boundary (e.g. an open inner
/// wall ring on a tool cap whose minuend cavity vents through wall cutouts) it
/// can hand back a partition that is not edge-disjoint: one outer loop traces
/// the whole face perimeter while sibling outer loops trace sub-regions sitting
/// in its MATERIAL. Those sub-faces double-cover that area and the assembled
/// shell goes non-manifold / inverted. The planar-arrangement decomposition is
/// edge-disjoint by construction, so it should be preferred in that case.
///
/// The check must NOT fire for a legitimate material ISLAND — an outer loop that
/// sits inside a HOLE of a larger outer ring (a washer with a central post: the
/// outer ring's polygon contains the post, but a hole separates them, so they do
/// not overlap). The distinguishing test is therefore: an outer loop B is
/// contained by a larger outer loop A AND there is NO hole (negative) loop H of
/// the same partition with B ⊆ H ⊆ A. With an intervening hole the configuration
/// is a valid island; without one, A's material overlaps B.
///
/// `cw_loops` flips the area sign for clockwise-wound boundaries so "outer"
/// (positive effective area) and "hole" (negative) are identified consistently
/// with the caller.
fn greedy_outer_loops_nested(loops: &[Vec<OrientedPCurveEdge>], cw_loops: bool) -> bool {
    // Sampled UV polygon + |effective area| for each loop, split by sign.
    let mut outers: Vec<(Vec<Point2>, f64)> = Vec::new();
    let mut holes: Vec<Vec<Point2>> = Vec::new();
    for wl in loops {
        let pts = sample_wire_loop_uv(wl);
        if pts.len() < 3 {
            continue;
        }
        let raw = signed_area_2d(&pts);
        let eff = if cw_loops { -raw } else { raw };
        if eff > 0.0 {
            outers.push((pts, eff));
        } else if eff < 0.0 {
            holes.push(pts);
        }
    }
    // `outer`'s polygon contains every sampled vertex of `inner` (within the
    // boundary tolerance).
    let poly_contains = |outer: &[Point2], inner: &[Point2]| -> bool {
        let eps = super::classify_2d::boundary_eps(outer);
        let inside = |p: Point2| {
            super::classify_2d::point_in_polygon_2d(p, outer)
                || super::classify_2d::distance_to_polygon_boundary(p, outer) <= eps
        };
        // Test each vertex AND each edge midpoint. For a concave `outer`, an
        // `inner` edge can exit and re-enter with both endpoints inside, so
        // vertex-only sampling would spuriously report containment.
        (0..inner.len()).all(|k| {
            let v = inner[k];
            let next = inner[(k + 1) % inner.len()];
            let mid = Point2::new(0.5 * (v.x() + next.x()), 0.5 * (v.y() + next.y()));
            inside(v) && inside(mid)
        })
    };
    for i in 0..outers.len() {
        for j in 0..outers.len() {
            // Strict containment: A (i) strictly larger AND geometrically holds
            // B (j). The area guard makes the relation asymmetric so two
            // coincident traces do not each "contain" the other.
            if i == j || outers[i].1 <= outers[j].1 {
                continue;
            }
            if !poly_contains(&outers[i].0, &outers[j].0) {
                continue;
            }
            // Valid island: some hole H sits between A and B (B ⊆ H ⊆ A). Then
            // A and B do not overlap and this is a legitimate post-in-washer.
            let separated_by_hole = holes
                .iter()
                .any(|h| poly_contains(h, &outers[j].0) && poly_contains(&outers[i].0, h));
            if !separated_by_hole {
                return true;
            }
        }
    }
    false
}

/// Whether both halves of one section lie in loops of the walk wound
/// against the outline, its holes. A section parts two regions, so at most
/// one of its halves bounds a hole: two holes side by side are regions the
/// walk took the wrong way round (sections clear of the outline, walked from
/// an edge that turns it round each of their regions clockwise). A loop of
/// no area is left to the degenerate test.
fn section_between_holes(
    loops: &[Vec<OrientedPCurveEdge>],
    frame: &PlaneFrame,
    cw_loops: bool,
    tol: f64,
) -> bool {
    let mut halves_in_holes: std::collections::HashMap<usize, u32> =
        std::collections::HashMap::new();
    for lp in loops {
        let pts = sampling::sample_wire_loop_uv_via_frame(lp, frame);
        if pts.len() < 3 {
            continue;
        }
        let area = signed_area_2d(&pts);
        let perimeter: f64 = pts
            .iter()
            .zip(pts.iter().cycle().skip(1))
            .map(|(a, b)| (*b - *a).length())
            .sum();
        if area.abs() <= perimeter * tol || (area < 0.0) == cw_loops {
            continue;
        }
        for e in lp {
            if let Some(id) = e.source_edge_idx {
                *halves_in_holes.entry(id).or_default() += 1;
            }
        }
    }
    halves_in_holes.values().any(|&n| n >= 2)
}

/// Quantized UV vertex key in the planar arrangement.
type UvKey = (i64, i64);

/// An undirected arrangement sub-segment: its two vertex keys, the source input
/// index, and whether it spans that whole input (so a whole arc can be emitted
/// with its true geometry).
type ArrSubEdge = (UvKey, UvKey, usize, bool);

/// A directed half-edge in the planar arrangement traced by
/// [`split_plane_face_by_arrangement`]. `seg_id` is the undirected sub-segment
/// index, shared by both directions so adjacent regions weld.
struct ArrHalfEdge {
    from: (i64, i64),
    to: (i64, i64),
    seg_id: usize,
    /// Direction angle at `from`.
    angle: f64,
    /// Direction angle at `from` along the edge itself: a whole arc's
    /// tangent, else `angle`. It orders edges whose chords leave a vertex
    /// along one line.
    tangent: f64,
}

/// One input edge to the arrangement (a boundary edge or a section), carrying
/// the true edge geometry (line or arc) plus its UV chord for the topological
/// subdivision. Arcs are represented by their chord while building the
/// arrangement (intersection, vertex merging, half-edge tracing all run on the
/// chord), then emitted as the true arc when the sub-edge spans the whole input.
struct ArrInput {
    /// UV chord start.
    a: Point2,
    /// UV chord end.
    b: Point2,
    /// True edge geometry to emit (3D curve, pcurve, endpoints, pave block).
    edge: OrientedPCurveEdge,
    /// Whether this input is an arc (non-Line). Arcs are emitted exactly only
    /// when un-split; if an arc would be split at an interior crossing the
    /// arrangement bails so the existing curved paths handle it.
    is_arc: bool,
    /// Whether this input came from the SECTION set (vs the face boundary).
    /// True line×arc crossings and exact-break sub-arc emission apply only to
    /// section arcs: boundary bay arcs arrive pre-split by the calibrated
    /// boundary-crossing machinery and must keep the historical chord path.
    is_section: bool,
}

/// Split co-endpoint SECTION arc pairs in an arrangement's input list at
/// their geometric midpoints (see the call site in
/// [`split_plane_face_by_arrangement`]). Only section arcs with a colliding
/// unordered chord-endpoint pair AND distinct midpoints are split; boundary
/// edges and identical duplicates are left untouched. A line between the
/// same two ends collides with an arc too: its chord is the arc's chord.
fn split_coendpoint_section_arc_inputs(inputs: &mut Vec<ArrInput>, frame: &PlaneFrame, tol: f64) {
    use std::collections::HashMap;
    type Q2 = (i64, i64);
    let qscale = 1.0 / tol.max(1e-12);
    let q2 = |p: Point2| -> Q2 {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
        )
    };
    let arc_mid = |e: &OrientedPCurveEdge| -> Option<Point3> {
        let EdgeCurve::Circle(ref c) = e.curve_3d else {
            return None;
        };
        let (ns, ne) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let s_ang = c.project(ns);
        let span = (c.project(ne) - s_ang).rem_euclid(std::f64::consts::TAU);
        if span < 1e-9 {
            return None;
        }
        Some(c.evaluate(s_ang + 0.5 * span))
    };
    let mut pair_mids: HashMap<(Q2, Q2), Vec<Q2>> = HashMap::new();
    for inp in inputs.iter() {
        let mid_uv = if matches!(inp.edge.curve_3d, EdgeCurve::Line) {
            Point2::new(
                f64::midpoint(inp.a.x(), inp.b.x()),
                f64::midpoint(inp.a.y(), inp.b.y()),
            )
        } else if inp.is_section
            && inp.is_arc
            && let Some(mid) = arc_mid(&inp.edge)
        {
            frame.project(mid)
        } else {
            continue;
        };
        let (a, b) = (q2(inp.a), q2(inp.b));
        let key = if a <= b { (a, b) } else { (b, a) };
        pair_mids.entry(key).or_default().push(q2(mid_uv));
    }
    let colliding: std::collections::HashSet<(Q2, Q2)> = pair_mids
        .into_iter()
        .filter(|(_, mids)| mids.iter().any(|m| *m != mids[0]))
        .map(|(k, _)| k)
        .collect();
    // A line through both ends of an arc is cut at them, and its piece
    // between them is the arc's chord too (a circle cut where a line crosses
    // it): the same band as the arrangement's breaks at ends on a chord.
    let band = tol * 100.0;
    let on_segment = |p: Point2, l: &ArrInput| {
        let d = l.b - l.a;
        let len2 = d.dot(d);
        len2 > 0.0 && {
            let t = (p - l.a).dot(d) / len2;
            (0.0..=1.0).contains(&t) && (p - (l.a + d * t)).length() <= band
        }
    };
    let along_line: Vec<bool> = inputs
        .iter()
        .map(|inp| {
            inp.is_section
                && inp.is_arc
                && inputs.iter().any(|l| {
                    matches!(l.edge.curve_3d, EdgeCurve::Line)
                        && on_segment(inp.a, l)
                        && on_segment(inp.b, l)
                })
        })
        .collect();
    if colliding.is_empty() && !along_line.contains(&true) {
        return;
    }
    let mut rebuilt: Vec<ArrInput> = Vec::with_capacity(inputs.len() + 2);
    for (i, inp) in inputs.drain(..).enumerate() {
        let (a, b) = (q2(inp.a), q2(inp.b));
        let key = if a <= b { (a, b) } else { (b, a) };
        let mid3 = if inp.is_section && inp.is_arc && (colliding.contains(&key) || along_line[i]) {
            arc_mid(&inp.edge)
        } else {
            None
        };
        let Some(mid_3d) = mid3 else {
            rebuilt.push(inp);
            continue;
        };
        let mid_uv = frame.project(mid_3d);
        let mut first_edge = inp.edge.clone();
        first_edge.end_3d = mid_3d;
        first_edge.end_uv = mid_uv;
        first_edge.source_edge_idx = None;
        first_edge.pave_block_id = None;
        let mut second_edge = inp.edge;
        second_edge.start_3d = mid_3d;
        second_edge.start_uv = mid_uv;
        second_edge.source_edge_idx = None;
        second_edge.pave_block_id = None;
        rebuilt.push(ArrInput {
            a: inp.a,
            b: mid_uv,
            edge: first_edge,
            is_arc: true,
            is_section: true,
        });
        rebuilt.push(ArrInput {
            a: mid_uv,
            b: inp.b,
            edge: second_edge,
            is_arc: true,
            is_section: true,
        });
    }
    *inputs = rebuilt;
}

/// How [`arrangement_regions_from_inputs`] turns its traced cycles into regions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArrNesting {
    /// Drop the largest cycle and emit the rest flat: faces with no integrated
    /// holes.
    DropLargest,
    /// Depth-even cycles are solid, depth-odd ones their holes: a holed cap's
    /// pre-woven edge list.
    EvenOdd,
    /// Every bounded face of the planar graph, holding the components nested in
    /// it as holes, with what bounds no region pruned first.
    Cells,
}

/// Whether two of a plane face's sections meet anywhere but end to end: a
/// crossing, or an end on the other's interior.
///
/// The sections of one closed solid on a face run apart or end to end. Tools
/// crossing each other, or a face's sections over a hole in the partner (whose
/// whole chord a hole weave wants), make a graph the region-tracing paths
/// read as one loop.
fn sections_meet_midway(sections: &[SectionEdge], frame: &PlaneFrame, tol: f64) -> bool {
    const CURVE_STEPS: u32 = 32;
    let band = tol * 100.0;
    let lines: Vec<Vec<Point2>> = sections
        .iter()
        .map(|s| {
            if matches!(s.curve_3d, EdgeCurve::Line) {
                return vec![frame.project(s.start), frame.project(s.end)];
            }
            let (t0, t1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            (0..=CURVE_STEPS)
                .map(|k| {
                    let t = (t1 - t0).madd(f64::from(k) / f64::from(CURVE_STEPS), t0);
                    frame.project(s.curve_3d.evaluate_with_endpoints(t, s.start, s.end))
                })
                .collect()
        })
        .collect();
    let ends: Vec<[Point2; 2]> = sections
        .iter()
        .map(|s| [frame.project(s.start), frame.project(s.end)])
        .collect();
    let box_of = |line: &[Point2]| {
        line.iter().fold(
            (f64::MAX, f64::MAX, f64::MIN, f64::MIN),
            |(lx, ly, hx, hy), p| (lx.min(p.x()), ly.min(p.y()), hx.max(p.x()), hy.max(p.y())),
        )
    };
    let boxes: Vec<_> = lines.iter().map(|line| box_of(line)).collect();
    let near_end = |p: Point2, e: &[Point2; 2]| e.iter().any(|&q| (p - q).length() <= band);
    let lands_on = |p: Point2, line: &[Point2]| {
        line.windows(2).any(|w| {
            let d = w[1] - w[0];
            let len2 = d.dot(d);
            if len2 <= 0.0 {
                return false;
            }
            let t = ((p - w[0]).dot(d) / len2).clamp(0.0, 1.0);
            (p - (w[0] + d * t)).length() <= band
        })
    };
    for i in 0..sections.len() {
        for j in i + 1..sections.len() {
            let (a, b) = (boxes[i], boxes[j]);
            if a.0 > b.2 + band || b.0 > a.2 + band || a.1 > b.3 + band || b.1 > a.3 + band {
                continue;
            }
            for p in ends[j] {
                if !near_end(p, &ends[i]) && lands_on(p, &lines[i]) {
                    return true;
                }
            }
            for p in ends[i] {
                if !near_end(p, &ends[j]) && lands_on(p, &lines[j]) {
                    return true;
                }
            }
            for u in lines[i].windows(2) {
                for v in lines[j].windows(2) {
                    let crosses = seg_cross_param(u[0], u[1], v[0], v[1]).is_some_and(|t| {
                        let at = u[0] + (u[1] - u[0]) * t;
                        !near_end(at, &ends[i]) && !near_end(at, &ends[j])
                    });
                    if crosses {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// A plane face's cells when its sections meet mid-way, or `None` when they do
/// not or the arrangement cannot hold its edges: the general splitter then
/// takes the face.
///
/// The arrangement reads each input by its chord, and a closed curve has none. A
/// closed circle or ellipse among the sections goes in as four arcs. A closed
/// spline's arcs would each read back as the whole loop, and a closed boundary
/// curve shares its edge with the face beside it, so it cannot take new vertices
/// here: either way the general splitter takes the face.
#[must_use]
pub fn split_plane_into_cells(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    rank: Rank,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    let face = topo.face(face_id).ok()?;
    let surface = face.surface().clone();
    if !matches!(surface, FaceSurface::Plane { .. }) {
        return None;
    }
    let wire_pts = collect_wire_points(topo, face.outer_wire());
    let frame = PlaneFrame::from_plane_face(extract_plane_normal(&surface), &wire_pts);
    if !sections_meet_midway(sections, &frame, tol) {
        return None;
    }
    let edges_of = |wire| boundary_edges_to_pcurve(topo, wire, &surface, &wire_pts, Some(&frame));
    let closed = |edge: &OrientedPCurveEdge| (edge.start_3d - edge.end_3d).length() <= tol;
    if edges_of(face.outer_wire()).iter().any(closed)
        || face
            .inner_wires()
            .iter()
            .any(|&wire| edges_of(wire).iter().any(closed))
    {
        return None;
    }
    let holes: Vec<Vec<OrientedPCurveEdge>> =
        face.inner_wires().iter().map(|&w| edges_of(w)).collect();
    let mut inputs: Vec<ArrInput> = edges_of(face.outer_wire())
        .into_iter()
        .chain(holes.iter().flatten().cloned())
        .map(|edge| ArrInput {
            a: edge.start_uv,
            b: edge.end_uv,
            is_arc: !matches!(edge.curve_3d, EdgeCurve::Line),
            is_section: false,
            edge,
        })
        .collect();
    for section in sections {
        let (t0, t1) = section
            .curve_3d
            .domain_with_endpoints(section.start, section.end);
        // A closed circle's domain starts at its parameter origin, wherever its
        // vertex sits.
        let from = match &section.curve_3d {
            EdgeCurve::Circle(circle) => circle.project(section.start),
            EdgeCurve::Ellipse(ellipse) => ellipse.project(section.start),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => t0,
        };
        let at = |f: f64| {
            section.curve_3d.evaluate_with_endpoints(
                (t1 - t0).madd(f, from),
                section.start,
                section.end,
            )
        };
        let closed = !matches!(section.curve_3d, EdgeCurve::Line)
            && (section.start - section.end).length() <= tol;
        if closed && matches!(section.curve_3d, EdgeCurve::NurbsCurve(_)) {
            return None;
        }
        let knots: Vec<Point3> = if closed {
            vec![section.start, at(0.25), at(0.5), at(0.75), section.end]
        } else {
            vec![section.start, section.end]
        };
        for span in knots.windows(2) {
            let (from, to) = (span[0], span[1]);
            let (a, b) = (frame.project(from), frame.project(to));
            inputs.push(ArrInput {
                a,
                b,
                edge: OrientedPCurveEdge {
                    curve_3d: section.curve_3d.clone(),
                    pcurve: compute_pcurve_on_surface(
                        &section.curve_3d,
                        from,
                        to,
                        &surface,
                        &wire_pts,
                        Some(&frame),
                    ),
                    start_uv: a,
                    end_uv: b,
                    start_3d: from,
                    end_3d: to,
                    forward: true,
                    source_edge_idx: None,
                    pave_block_id: None,
                },
                is_arc: !matches!(section.curve_3d, EdgeCurve::Line),
                is_section: true,
            });
        }
    }
    split_coendpoint_section_arc_inputs(&mut inputs, &frame, tol);
    arrangement_regions_from_inputs(
        &surface,
        inputs,
        rank,
        face.is_reversed(),
        face_id,
        &frame,
        tol,
        ArrNesting::Cells,
        &holes,
        None,
    )
}

/// The undirected edges of a graph lying on no cycle.
pub(super) fn bridge_edges(edges: &[(UvKey, UvKey)]) -> Vec<bool> {
    let mut around: std::collections::HashMap<UvKey, Vec<(UvKey, usize)>> =
        std::collections::HashMap::new();
    for (i, &(a, b)) in edges.iter().enumerate() {
        around.entry(a).or_default().push((b, i));
        around.entry(b).or_default().push((a, i));
    }
    let mut bridge = vec![false; edges.len()];
    let mut seen: std::collections::HashMap<UvKey, (usize, usize)> =
        std::collections::HashMap::new();
    let mut clock = 0;
    for &(root, _) in edges {
        if seen.contains_key(&root) {
            continue;
        }
        seen.insert(root, (clock, clock));
        clock += 1;
        // Vertex, the edge it was reached by, and its next neighbour to visit.
        let mut stack: Vec<(UvKey, usize, usize)> = vec![(root, usize::MAX, 0)];
        while let Some(&mut (at, via, ref mut next)) = stack.last_mut() {
            if let Some(&(to, edge)) = around[&at].get(*next) {
                *next += 1;
                if edge == via {
                    continue;
                }
                if let Some(&(order, _)) = seen.get(&to) {
                    if let Some(entry) = seen.get_mut(&at) {
                        entry.1 = entry.1.min(order);
                    }
                } else {
                    seen.insert(to, (clock, clock));
                    clock += 1;
                    stack.push((to, edge, 0));
                }
            } else {
                stack.pop();
                let low = seen[&at].1;
                if let Some(&(parent, _, _)) = stack.last() {
                    if let Some(entry) = seen.get_mut(&parent) {
                        entry.1 = entry.1.min(low);
                    }
                    if low > seen[&parent].0 {
                        bridge[via] = true;
                    }
                }
            }
        }
    }
    bridge
}

/// Decompose a planar face into its minimal interior regions via a 2D
/// arrangement of its boundary and section edges.
///
/// The angular wire builder ([`build_wire_loops`]) and the single-crossing
/// helper ([`try_split_crossing_plane_face`]) mis-partition a plane face cut by
/// three or more sections that form a partial grid (e.g. a notch side wall on a
/// SHELLED body, or an outer wall carved by a U-notch with rounded corners that
/// opens at the rim) — they hand back one self-crossing wire, which makes the
/// shared section edge non-manifold and forces a mesh fallback.
///
/// This builds the full planar subdivision instead: every boundary and section
/// edge (lines exactly, arcs via their chord) is split at all mutual
/// intersections, directed half-edges are traced into minimal faces by the
/// leftmost-turn rule, and the unbounded outer face is dropped. Each interior
/// region becomes a [`SplitSubFace`]. Straight sub-edges get 3D from the plane
/// frame (UV↔3D is an exact bijection on a plane); whole arc inputs are emitted
/// with their true `Circle`/`Ellipse` geometry so corner roundings are preserved
/// exactly.
///
/// Conservative on arcs: if an arc input would be split at an interior crossing
/// (so its true geometry cannot be reproduced as one edge), the function returns
/// `None` and the existing curved paths take over. Returns `None` when the
/// arrangement could not be traced or yields no interior region.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn split_plane_face_by_arrangement(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    frame: &PlaneFrame,
    tol: f64,
    split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
) -> Option<Vec<SplitSubFace>> {
    // Collect input edges (boundary + sections) with their true geometry. Each
    // arc keeps its source curve; the arrangement subdivision uses the chord.
    let mut inputs: Vec<ArrInput> = Vec::new();
    for e in boundary_edges {
        let is_arc = !matches!(e.curve_3d, EdgeCurve::Line);
        inputs.push(ArrInput {
            a: e.start_uv,
            b: e.end_uv,
            edge: e.clone(),
            is_arc,
            is_section: false,
        });
    }
    for s in sections {
        let is_arc = !matches!(s.curve_3d, EdgeCurve::Line);
        // UV endpoints for this face (rank A/B), falling back to projection.
        let (su, eu) = match rank {
            Rank::A => (s.start_uv_a, s.end_uv_a),
            Rank::B => (s.start_uv_b, s.end_uv_b),
        };
        let su = su.unwrap_or_else(|| frame.project(s.start));
        let eu = eu.unwrap_or_else(|| frame.project(s.end));
        // pcurve on this face: prefer the section's stored pcurve for the rank.
        let pcurve = match rank {
            Rank::A => s.pcurve_a.clone(),
            Rank::B => s.pcurve_b.clone(),
        };
        inputs.push(ArrInput {
            a: su,
            b: eu,
            edge: OrientedPCurveEdge {
                curve_3d: s.curve_3d.clone(),
                pcurve,
                start_uv: su,
                end_uv: eu,
                start_3d: s.start,
                end_3d: s.end,
                forward: true,
                source_edge_idx: None,
                pave_block_id: s.pave_block_id,
            },
            is_arc,
            is_section: true,
        });
    }
    // Two SECTION arcs sharing both endpoints (a 2-tangency section circle's
    // half-rims) have the SAME CHORD, so the chord-based subdivision would
    // collapse them into one segment and lose the region between them (the
    // box∪cylinder 2-tangency bottom face lost its far half this way). Split
    // each such arc at its geometric midpoint: the quarter chords are
    // distinct, the subdivision sees the ring, and each piece is still a
    // whole arc input emitted with true geometry. Identical duplicates (same
    // midpoint) stay whole — their chord collapse is the intended dedup.
    split_coendpoint_section_arc_inputs(&mut inputs, frame, tol);

    // Section-only entry point: keep the historical max-area drop + flat
    // emission (these faces have no integrated holes).
    arrangement_regions_from_inputs(
        surface,
        inputs,
        rank,
        reversed,
        face_id,
        frame,
        tol,
        ArrNesting::DropLargest,
        &[],
        split_registry,
    )
}

/// Holed-plane variant of [`split_plane_face_by_arrangement`]. The caller has
/// already woven the hole boundaries into a single combined edge list (via
/// [`integrate_holes_plane`]: sections trimmed at hole crossings, hole edges
/// split at section crossings), so the full planar subdivision is just that
/// list. Treat every edge as an arrangement input and decompose into minimal
/// regions. Unlike the section-based entry point, this path produces sub-faces
/// whose holes are already integral to the trace (no separate inner-wire
/// distribution), so it correctly partitions a holed cap whose cut also crosses
/// the holes — the divider-lip fuse onto a compartmented body, where the
/// stacking lip's footprint edge cuts each divider arm between the compartment
/// openings. The angular wire builder mis-traces that arrangement into one
/// region; this returns the true under-lip ring + exposed divider-cross regions.
///
/// Inputs may contain both orientations of a shared edge (the integrate output
/// emits forward/reverse pairs); the arrangement's undirected sub-edge dedup
/// collapses them. Returns `None` on the same conditions as the section path.
#[allow(clippy::too_many_arguments)]
fn arrangement_regions_from_combined(
    surface: &FaceSurface,
    combined_edges: &[OrientedPCurveEdge],
    inner_wires: &[Vec<OrientedPCurveEdge>],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    frame: &PlaneFrame,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    // A closed rim (a whole circle, one edge from a vertex back to itself)
    // has no chord, and the region it bounds would be lost with it.
    if combined_edges.iter().any(|e| {
        !matches!(e.curve_3d, EdgeCurve::Line)
            && (e.start_uv - e.end_uv).length() <= tol
            && (e.curve_3d.evaluate_with_endpoints(
                {
                    let (t0, t1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
                    0.5 * (t0 + t1)
                },
                e.start_3d,
                e.end_3d,
            ) - e.start_3d)
                .length()
                > tol
    }) {
        return None;
    }
    // The even-odd nesting pass resolves the nested overlapping faces (a holed
    // cap whose holes/sections form a component separate from the outer
    // boundary) and drops the air regions that fill the original holes.
    let inputs: Vec<ArrInput> = combined_edges
        .iter()
        .map(|e| ArrInput {
            a: e.start_uv,
            b: e.end_uv,
            edge: e.clone(),
            is_arc: !matches!(e.curve_3d, EdgeCurve::Line),
            is_section: false,
        })
        .collect();
    arrangement_regions_from_inputs(
        surface,
        inputs,
        rank,
        reversed,
        face_id,
        frame,
        tol,
        ArrNesting::EvenOdd,
        inner_wires,
        None,
    )
}

/// Decompose a planar arrangement (already-collected [`ArrInput`] edges) into
/// its minimal interior regions. Shared by [`split_plane_face_by_arrangement`]
/// (boundary + sections) and [`arrangement_regions_from_combined`] (a holed
/// face's pre-woven edge list).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn arrangement_regions_from_inputs(
    surface: &FaceSurface,
    mut inputs: Vec<ArrInput>,
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    frame: &PlaneFrame,
    tol: f64,
    // `EvenOdd` resolves overlapping nested minimal faces (a holed cap whose
    // holes/sections form a component separate from the outer-boundary loop)
    // via even-odd containment nesting: depth-even faces are solid regions
    // emitted with their direct-child faces as holes, depth-odd faces are holes,
    // and solid regions filling an `original_holes` opening are dropped (air).
    // `DropLargest` keeps the historical "drop only the max-area face, emit each
    // as a flat outer wire" behaviour for the section-only entry point.
    nesting: ArrNesting,
    // The face's ORIGINAL holes (compartment openings) in this face's frame.
    // Used by the nesting paths to drop solid regions that fill an opening.
    // Empty for the section-only entry point.
    original_holes: &[Vec<OrientedPCurveEdge>],
    // When present, section-input interior break points are recorded per pave
    // block (exact UV → 3D via the frame) so curved faces sharing the same
    // section curve pre-split at identical points.
    split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
) -> Option<Vec<SplitSubFace>> {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::Vec2;

    // Section break points for the registry, recorded only once the
    // arrangement succeeds: one that gives up (an arc broken at its chord)
    // has also broken lines at chords, which no other face shares (a section
    // line crossing the chord of a cap's major rim arc split the wall beside
    // it there).
    let mut pending_breaks: Vec<(usize, Point3)> = Vec::new();
    // Drop degenerate (zero-length) inputs.
    inputs.retain(|i| (i.a - i.b).length() > tol);
    if inputs.len() < 3 {
        return None;
    }

    // Every arc must actually LIE in this face's plane. A straddle arc (a corner
    // cylinder crossing the cap plane, whose endpoints/midpoint sit off the
    // plane) projects to a meaningless chord — its true geometry cannot be a
    // sub-edge of this planar arrangement. Bail so the existing curved paths
    // handle those faces. Test via the frame round-trip: an in-plane point maps
    // project→evaluate back to itself; an off-plane point does not. The band is
    // the weld scale (100·tol), not the vertex tolerance: a marched plane×cone
    // conic lies in the plane only to its curve-fit error (~1e-6), while a
    // genuine straddle arc leaves the plane by orders of magnitude more.
    let on_plane = |p: Point3| -> bool {
        let uv = frame.project(p);
        (frame.evaluate(uv.x(), uv.y()) - p).length() <= tol * 100.0
    };
    for inp in &inputs {
        if inp.is_arc {
            let mid =
                inp.edge
                    .curve_3d
                    .evaluate_with_endpoints(0.5, inp.edge.start_3d, inp.edge.end_3d);
            if !on_plane(inp.edge.start_3d) || !on_plane(inp.edge.end_3d) || !on_plane(mid) {
                return None;
            }
        }
    }

    // Quantize UV points so coincident vertices merge. The grid cell is the
    // linear tolerance: a point maps to `round(p / tol)`, so two points within
    // `tol` collapse to one key (UV on a plane is metric).
    let qscale = 1.0 / tol.max(1e-12);
    let qkey = |p: Point2| -> (i64, i64) {
        (
            (p.x() * qscale).round() as i64,
            (p.y() * qscale).round() as i64,
        )
    };

    // Split each chord at every intersection with every other chord, plus at
    // any other chord's endpoint that lands on its interior. Collect the
    // resulting break parameters, then emit sub-segments between consecutive
    // breaks.
    //
    // A computed break must UNIFY with the vertex set, not merely quantize: a
    // proper crossing is re-derived independently from each input's chord
    // fraction, and for nearly-parallel inputs (adjacent facet chains) the
    // two derivations scatter by crossing-noise/sin(angle) — far above the
    // quantization cell — leaving two degree-2 vertices where one degree-4
    // vertex belongs. Worse, a break landing near an input ENDPOINT splits
    // the chord at the scattered point while the neighbouring face keeps the
    // exact operand vertex, minting a near-copy boundary edge that never
    // welds. So: input endpoints register FIRST (ground truth), and every
    // later point within the snap band adopts the earliest registered vertex
    // via a coarse hash (first-wins).
    let snap_band = tol * 1000.0;
    let ckey = |p: Point2| -> (i64, i64) {
        (
            (p.x() / snap_band).round() as i64,
            (p.y() / snap_band).round() as i64,
        )
    };
    let mut vert_pos: std::collections::HashMap<(i64, i64), Point2> =
        std::collections::HashMap::new();
    let mut coarse: std::collections::HashMap<(i64, i64), Point2> =
        std::collections::HashMap::new();
    let mut register =
        |p: Point2, map: &mut std::collections::HashMap<(i64, i64), Point2>| -> (i64, i64) {
            let (cx, cy) = ckey(p);
            let mut adopted: Option<(f64, Point2)> = None;
            for dx in -1..=1_i64 {
                for dy in -1..=1_i64 {
                    if let Some(&q) = coarse.get(&(cx + dx, cy + dy)) {
                        let d = (q - p).length();
                        if d <= snap_band && adopted.is_none_or(|(bd, _)| d < bd) {
                            adopted = Some((d, q));
                        }
                    }
                }
            }
            let p = adopted.map_or(p, |(_, q)| q);
            let k = qkey(p);
            map.entry(k).or_insert(p);
            coarse.entry(ckey(p)).or_insert(p);
            k
        };
    // Exact 3D per registered input-endpoint vertex. The generic emission
    // reconstructs 3D as frame.evaluate(uv), which PROJECTS the point onto
    // the stored plane — but an operand's facet-chain vertex can sit ~1e-5
    // OFF that plane, and collapsing it moves the sub-face boundary off the
    // neighbouring faces' copy of the same vertex, minting a twin edge that
    // never welds. Boundary inputs register first, so their exact positions
    // win over section endpoints at the same vertex.
    let mut exact3d: std::collections::HashMap<(i64, i64), Point3> =
        std::collections::HashMap::new();
    for inp in &inputs {
        let ka = register(inp.a, &mut vert_pos);
        exact3d.entry(ka).or_insert(inp.edge.start_3d);
        let kb = register(inp.b, &mut vert_pos);
        exact3d.entry(kb).or_insert(inp.edge.end_3d);
    }

    // True when a UV point lies on input `idx`'s actual arc geometry (within
    // `tol`). A break parameter is computed against an arc's straight CHORD, so
    // a section that crosses the chord can still miss the real arc — most
    // commonly a section in the face interior crossing a convex BOUNDARY corner
    // arc's chord while passing well clear of the outward-bulging arc itself
    // (gridfinity scoop bases vs. the rounded-rect floor corners). Registering
    // that phantom break would subdivide an uncrossed arc and trip the
    // `is_arc && ts.len() > 2` bail, collapsing the whole arrangement back to
    // the self-crossing angular trace. Sampling the arc and rejecting breaks
    // farther than `tol` from it keeps the arc a single sub-edge.
    let chord_break_on_arc = |idx: usize, uv: Point2| -> bool {
        let e = &inputs[idx].edge;
        // `evaluate_with_endpoints` takes the curve's NATIVE parameter (radians
        // for Circle/Ellipse, knot value for NURBS), not a normalised [0,1].
        // Sample across the trimmed domain so the probe points lie on the real arc.
        let (d0, d1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
        (0..=ARR_ARC_SAMPLES)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)]
                let f = k as f64 / ARR_ARC_SAMPLES as f64;
                let p3 =
                    e.curve_3d
                        .evaluate_with_endpoints(d0 + (d1 - d0) * f, e.start_3d, e.end_3d);
                (frame.project(p3) - uv).length()
            })
            .fold(f64::MAX, f64::min)
            <= tol * 100.0
    };

    // Per-input chord bounding box (expanded by `tol`) for broad-phase pruning
    // of the pairwise subdivision below. Two chords whose boxes are disjoint can
    // neither cross nor host the other's endpoint on their interior, so the
    // O(inputs²) inner scan skips them — near-linear on an arrangement of many
    // disjoint loops (a perforated cap's hole edges). `ts` is sorted+deduped
    // after collection, so pruning never changes the emitted sub-segments.
    let chord_boxes: Vec<(f64, f64, f64, f64)> = inputs
        .iter()
        .map(|inp| {
            (
                inp.a.x().min(inp.b.x()) - tol,
                inp.a.y().min(inp.b.y()) - tol,
                inp.a.x().max(inp.b.x()) + tol,
                inp.a.y().max(inp.b.y()) + tol,
            )
        })
        .collect();

    // Undirected sub-segments keyed by endpoint vertex pair, each tagged with
    // its source input index and whether it spans that whole input.
    // UV polylines (with native curve parameters) for every arc input,
    // sampled once. Used for TRUE line×arc crossings: the chord×chord
    // crossing point can sit several 1e-4 from the real arc (the sagitta),
    // and registering it splits the LINE at a phantom vertex while the arc's
    // side rejects the break — a mismatched half-edge graph whose dangling
    // edges the face tracer walks out-and-back, emitting slit regions.
    let arc_polys: Vec<Option<Vec<(f64, Point2)>>> = inputs
        .iter()
        .map(|inp| {
            inp.is_arc.then(|| {
                let e = &inp.edge;
                let (d0, d1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
                (0..=ARR_ARC_SAMPLES)
                    .map(|k| {
                        #[allow(clippy::cast_precision_loss)]
                        let f = k as f64 / ARR_ARC_SAMPLES as f64;
                        let t = d0 + (d1 - d0) * f;
                        let p3 = e.curve_3d.evaluate_with_endpoints(t, e.start_3d, e.end_3d);
                        (t, frame.project(p3))
                    })
                    .collect()
            })
        })
        .collect();
    // True crossings of the segment (la, lb) with arc input `ai`, refined by
    // bisection on the arc's native parameter against the segment's line.
    // Returns the exact crossing UVs.
    // Max deviation of an arc's sampled polyline from its chord — the bound
    // within which a chord-derived crossing and the true crossing of the same
    // transversal can differ.
    let arc_sagitta = |ai: usize| -> f64 {
        let Some(poly) = arc_polys[ai].as_ref() else {
            return 0.0;
        };
        let a = inputs[ai].a;
        let b = inputs[ai].b;
        let d = b - a;
        let len = d.length().max(f64::MIN_POSITIVE);
        poly.iter()
            .map(|(_, p)| (d.x().madd(p.y() - a.y(), -(d.y() * (p.x() - a.x()))) / len).abs())
            .fold(0.0, f64::max)
    };
    let line_arc_crossings = |la: Point2, lb: Point2, ai: usize| -> Vec<Point2> {
        let Some(poly) = arc_polys[ai].as_ref() else {
            return Vec::new();
        };
        let e = &inputs[ai].edge;
        // An arc endpoint lying ON the line is an endpoint T-junction, owned
        // by the endpoint-break pass; fit-error sign flips near it fabricate
        // a phantom interior crossing that over-splits the arc.
        let guard = tol * 100.0;
        let near_end = |p: Point2| -> bool {
            (p - inputs[ai].a).length() <= guard
                || (p - inputs[ai].b).length() <= guard
                || (p - la).length() <= guard
                || (p - lb).length() <= guard
        };
        let ld = lb - la;
        let side = |p: Point2| -> f64 { ld.x().madd(p.y() - la.y(), -(ld.y() * (p.x() - la.x()))) };
        let mut out = Vec::new();
        for w in poly.windows(2) {
            let ((t0, p0), (t1, p1)) = (w[0], w[1]);
            if seg_cross_param(la, lb, p0, p1).is_none() {
                continue;
            }
            let (mut lo, mut hi) = (t0, t1);
            let mut s_lo = side(p0);
            for _ in 0..48 {
                let mid = 0.5 * (lo + hi);
                let pm = frame.project(
                    e.curve_3d
                        .evaluate_with_endpoints(mid, e.start_3d, e.end_3d),
                );
                let s_mid = side(pm);
                if (s_mid > 0.0) == (s_lo > 0.0) {
                    lo = mid;
                    s_lo = s_mid;
                } else {
                    hi = mid;
                }
            }
            let tm = 0.5 * (lo + hi);
            let uv = frame.project(e.curve_3d.evaluate_with_endpoints(tm, e.start_3d, e.end_3d));
            // A genuine crossing converges ONTO the line; a phantom from
            // fit-error sign noise (an arc endpoint sitting on the line makes
            // the initial side() pure noise) converges anywhere in the sample
            // window and lands well off it.
            let ld_len = ld.length().max(f64::MIN_POSITIVE);
            let dist_line = side(uv).abs() / ld_len;
            if dist_line <= guard && !near_end(uv) {
                out.push(uv);
            }
        }
        out
    };

    // Whether `uv` lies on the TRUE curve of arc input `ai` (not just near its
    // chord): the nearest sample is refined on the arc's native parameter.
    let on_true_arc = |ai: usize, uv: Point2| -> bool {
        let Some(poly) = arc_polys[ai].as_ref() else {
            return false;
        };
        let e = &inputs[ai].edge;
        let Some(k) = (0..poly.len()).min_by(|&x, &y| {
            (poly[x].1 - uv)
                .length()
                .total_cmp(&(poly[y].1 - uv).length())
        }) else {
            return false;
        };
        let (mut lo, mut hi) = (
            poly[k.saturating_sub(1)].0,
            poly[(k + 1).min(poly.len() - 1)].0,
        );
        let off = |t: f64| {
            (frame.project(e.curve_3d.evaluate_with_endpoints(t, e.start_3d, e.end_3d)) - uv)
                .length()
        };
        for _ in 0..60 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if off(m1) < off(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        off(f64::midpoint(lo, hi)) <= tol * 100.0
    };

    let mut sub_edges: Vec<ArrSubEdge> = Vec::new();
    for i in 0..inputs.len() {
        let (a0, a1) = (inputs[i].a, inputs[i].b);
        let d = a1 - a0;
        let len = d.length();
        if len < tol {
            continue;
        }
        let i_is_arc = inputs[i].is_arc;
        let (ilx, ily, ihx, ihy) = chord_boxes[i];
        // Break parameters along this chord (t in [0,1]), each optionally
        // carrying the EXACT break UV (true line×arc crossings; endpoint
        // T-junctions use the endpoint itself) so both sides of a junction
        // register the identical vertex.
        let mut ts: Vec<(f64, Option<Point2>)> = vec![(0.0, None), (1.0, None)];
        let mut inexact_arc_break = false;
        for (j, other) in inputs.iter().enumerate() {
            if i == j {
                continue;
            }
            // Box pruning is exact only when neither edge bulges past its
            // chord — i.e. both are lines. An arc can cross or be T-junctioned
            // outside its chord's box, so never prune a pair involving one
            // (arcs are rare: rounded corners). The honeycomb cap is all lines,
            // so this keeps the near-linear win where it matters.
            if !i_is_arc && !other.is_arc {
                let (jlx, jly, jhx, jhy) = chord_boxes[j];
                if ilx > jhx || jlx > ihx || ily > jhy || jly > ihy {
                    continue; // chord boxes disjoint → no crossing, no T-junction
                }
            }
            // A pair that survives the broad-phase does the real crossing /
            // T-junction work below — the cost the bbox prune keeps near-linear.
            crate::perf::bump_face_split_probe();
            let (b0, b1) = (other.a, other.b);
            // Proper interior crossings. Line×arc pairs use the TRUE crossing
            // (refined against the real arc) registered with the exact UV on
            // BOTH inputs; the chord-derived point can be a sagitta away and
            // splitting only one side desynchronizes the half-edge graph.
            match (i_is_arc, other.is_arc) {
                (false, false) => {
                    if let Some(t) = seg_cross_param(a0, a1, b0, b1) {
                        ts.push((t, None));
                    }
                }
                (false, true) if other.is_section => {
                    // True crossings replace the chord-derived point when one
                    // is found within the arc's sagitta of it; a chord
                    // crossing with NO nearby true
                    // crossing keeps the historical chord break (the sampled
                    // arc span can be unreliable for reversed arcs, and
                    // dropping calibrated breaks under-splits).
                    let cover = arc_sagitta(j) + tol * 100.0;
                    let truex = line_arc_crossings(a0, a1, j);
                    let mut covered_chord = false;
                    // A chord crossing within the weld band of the ARC's
                    // endpoints is the endpoint T-junction seen through the
                    // chord (a fit-error offset endpoint dips across the
                    // line); the endpoint-break pass owns it.
                    let chord_t = seg_cross_param(a0, a1, b0, b1).filter(|&t| {
                        let p = a0 + d * t;
                        (p - other.a).length() > tol * 100.0 && (p - other.b).length() > tol * 100.0
                    });
                    for uv in &truex {
                        let t = (*uv - a0).dot(d) / (len * len);
                        if t > 1e-6 && t < 1.0 - 1e-6 {
                            ts.push((t, Some(*uv)));
                            if let Some(ct) = chord_t
                                && (t - ct).abs() * len <= cover
                            {
                                covered_chord = true;
                            }
                        }
                    }
                    if let Some(ct) = chord_t
                        && !covered_chord
                    {
                        ts.push((ct, None));
                    }
                }
                (true, false) if inputs[i].is_section => {
                    let cover = arc_sagitta(i) + tol * 100.0;
                    let truex = line_arc_crossings(b0, b1, i);
                    let mut covered_chord = false;
                    let chord_t = seg_cross_param(a0, a1, b0, b1)
                        .filter(|&t| chord_break_on_arc(i, a0 + d * t))
                        .filter(|&t| t * len > tol * 100.0 && (1.0 - t) * len > tol * 100.0);
                    for uv in &truex {
                        let t = (*uv - a0).dot(d) / (len * len);
                        if t > 1e-6 && t < 1.0 - 1e-6 {
                            ts.push((t, Some(*uv)));
                            if let Some(ct) = chord_t
                                && (t - ct).abs() * len <= cover
                            {
                                covered_chord = true;
                            }
                        }
                    }
                    if let Some(ct) = chord_t
                        && !covered_chord
                    {
                        ts.push((ct, None));
                        inexact_arc_break = true;
                    }
                }
                _ => {
                    if let Some(t) = seg_cross_param(a0, a1, b0, b1)
                        && (!i_is_arc || chord_break_on_arc(i, a0 + d * t))
                    {
                        ts.push((t, None));
                        if i_is_arc {
                            inexact_arc_break = true;
                        }
                    }
                }
            }
            // Other chord's endpoints landing on this chord's interior
            // (T-junctions where a section merely touches another). The break
            // registers with the ENDPOINT itself as the exact vertex UV —
            // weld-scale band, not the vertex tolerance: a marched section's
            // endpoint (curve-fit error ~1e-6) landing on a boundary or
            // section chord is a REAL T-junction; at 1e-7 it is missed, the
            // chord dangles as a pendant, and the face tracer walks it twice.
            // A section ending on a SECTION arc's true curve (an earlier cut's
            // trace meeting a cylinder's section where it runs through) sits a
            // sagitta off the chord; it breaks the arc at that exact point. A
            // circle section (a rim that faceted sections of an exactly
            // coincident lip fan into) keeps the chord path.
            for bp in [b0, b1] {
                let w = (bp - a0).dot(d) / (len * len);
                if w > 1e-6 && w < 1.0 - 1e-6 {
                    let on = a0 + d * w;
                    let on_chord = (on - bp).length() < tol * 100.0;
                    if (on_chord && (!i_is_arc || chord_break_on_arc(i, bp)))
                        || (i_is_arc
                            && !matches!(inputs[i].edge.curve_3d, EdgeCurve::Circle(_))
                            && inputs[i].is_section
                            && !on_chord
                            && (bp - a0).length() > tol * 100.0
                            && (bp - a1).length() > tol * 100.0
                            && on_true_arc(i, bp))
                    {
                        ts.push((w, Some(bp)));
                    }
                }
            }
        }
        ts.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|x, y| (x.0 - y.0).abs() < 1e-6);
        // An arc split at a chord-derived (inexact) break cannot be emitted
        // faithfully; bail and let the existing curved paths handle it. Exact
        // true-crossing and endpoint-T breaks emit trimmed sub-arcs below.
        if inputs[i].is_arc && ts.len() > 2 && (inexact_arc_break || !inputs[i].is_section) {
            return None;
        }
        let n_breaks = ts.len();
        if n_breaks > 2
            && inputs[i].is_section
            && let Some(pb_id) = inputs[i].edge.pave_block_id
        {
            for brk in &ts[1..n_breaks - 1] {
                let uv = brk.1.unwrap_or(a0 + d * brk.0);
                pending_breaks.push((pb_id, frame.evaluate(uv.x(), uv.y())));
            }
        }
        for (wi, w) in ts.windows(2).enumerate() {
            let (ta, tb) = (w[0].0, w[1].0);
            if tb - ta < 1e-6 {
                continue;
            }
            let pa = w[0].1.unwrap_or(a0 + d * ta);
            let pb = w[1].1.unwrap_or(a0 + d * tb);
            let ka = register(pa, &mut vert_pos);
            let kb = register(pb, &mut vert_pos);
            // Whole = this is the only sub-segment of the input (no interior
            // breaks): exactly one window spanning [0,1].
            let whole = n_breaks == 2 && wi == 0;
            if ka != kb {
                sub_edges.push((ka, kb, i, whole));
            }
        }
    }

    // Deduplicate undirected sub-edges (the same physical edge can arise from
    // two overlapping input chords). Keep one record per vertex pair, preferring
    // a whole-arc source so the true arc geometry is emitted.
    sub_edges.sort_by(|l, r| {
        let lk = if l.0 <= l.1 { (l.0, l.1) } else { (l.1, l.0) };
        let rk = if r.0 <= r.1 { (r.0, r.1) } else { (r.1, r.0) };
        lk.cmp(&rk)
            // Prefer arc-whole inputs first within a vertex-pair group.
            .then_with(|| {
                let la = inputs[l.2].is_arc && l.3;
                let ra = inputs[r.2].is_arc && r.3;
                ra.cmp(&la)
            })
    });
    sub_edges.dedup_by(|a, b| {
        let ak = if a.0 <= a.1 { (a.0, a.1) } else { (a.1, a.0) };
        let bk = if b.0 <= b.1 { (b.0, b.1) } else { (b.1, b.0) };
        ak == bk
    });
    if nesting == ArrNesting::Cells {
        let pairs: Vec<(UvKey, UvKey)> = sub_edges.iter().map(|&(a, b, _, _)| (a, b)).collect();
        let bridge = bridge_edges(&pairs);
        let mut at = 0;
        sub_edges.retain(|_| {
            at += 1;
            !bridge[at - 1]
        });
    }
    if sub_edges.is_empty() {
        return None;
    }
    // Drop the per-edge source/whole tags into a parallel lookup keyed by seg_id
    // so the half-edge trace (which only needs vertex pairs) stays unchanged.
    let sub_edge_src: Vec<(usize, bool)> = sub_edges.iter().map(|&(_, _, i, w)| (i, w)).collect();
    let sub_edges: Vec<(UvKey, UvKey)> = sub_edges.iter().map(|&(a, b, _, _)| (a, b)).collect();

    // A whole arc's direction where it leaves vertex `at`, along the arc
    // rather than its chord. An arc's chord can lie on the line of another
    // edge at the same vertex (a disc's rim arc over a section and a chord
    // in one line), and the chords alone then leave the trace to rounding.
    // A whole sub-edge runs its input's way, so `forward` leaves its start.
    let arc_leaving_angle = |seg_id: usize, forward: bool, at: Point2| -> Option<f64> {
        let (i, whole) = sub_edge_src[seg_id];
        if !(inputs[i].is_arc && whole) {
            return None;
        }
        let e = &inputs[i].edge;
        let (d0, d1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
        let (t_end, t_far) = if forward { (d0, d1) } else { (d1, d0) };
        let uv =
            |t: f64| frame.project(e.curve_3d.evaluate_with_endpoints(t, e.start_3d, e.end_3d));
        let end = uv(t_end);
        if (end - at).length() > tol * 1000.0 {
            return None;
        }
        let dir = uv((t_far - t_end).madd(1e-4, t_end)) - end;
        (dir.length() > 0.0).then(|| dir.y().atan2(dir.x()))
    };

    // Build the directed half-edge adjacency. Each undirected sub-edge id maps
    // to two directed half-edges; both carry that id so adjacent regions share
    // one topology edge. Half-edge index 2*k = forward (va->vb), 2*k+1 = reverse.
    let mut halfs: Vec<ArrHalfEdge> = Vec::with_capacity(sub_edges.len() * 2);
    for (seg_id, &(ka, kb)) in sub_edges.iter().enumerate() {
        let pa = vert_pos[&ka];
        let pb = vert_pos[&kb];
        let fwd = pb - pa;
        let rev = pa - pb;
        let angle = fwd.y().atan2(fwd.x());
        let back = rev.y().atan2(rev.x());
        halfs.push(ArrHalfEdge {
            from: ka,
            to: kb,
            seg_id,
            angle,
            tangent: arc_leaving_angle(seg_id, true, pa).unwrap_or(angle),
        });
        halfs.push(ArrHalfEdge {
            from: kb,
            to: ka,
            seg_id,
            angle: back,
            tangent: arc_leaving_angle(seg_id, false, pb).unwrap_or(back),
        });
    }

    // Outgoing half-edges per vertex.
    let mut out_at: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (hi, h) in halfs.iter().enumerate() {
        out_at.entry(h.from).or_default().push(hi);
    }

    // Trace minimal faces. From each unused half-edge, at every arrival vertex
    // pick the next outgoing half-edge that turns most clockwise from the
    // arriving direction (the "next edge in face" rule for a CCW-bounded face),
    // i.e. minimize the CCW angle from the reverse-of-arrival to the candidate.
    let mut used = vec![false; halfs.len()];
    let mut faces: Vec<Vec<usize>> = Vec::new();
    for start in 0..halfs.len() {
        if used[start] {
            continue;
        }
        let mut face: Vec<usize> = Vec::new();
        let mut cur = start;
        let mut ok = true;
        loop {
            if used[cur] {
                // Returned to an already-used half-edge that is not the start —
                // this trace is degenerate; abandon it.
                ok = cur == start && !face.is_empty();
                break;
            }
            used[cur] = true;
            face.push(cur);
            let arrive_to = halfs[cur].to;
            if arrive_to == halfs[start].from && !face.is_empty() {
                // Closed the loop back to the start vertex.
                break;
            }
            // Incoming direction reversed = the direction we'd leave back along.
            let back_angle =
                (halfs[cur].angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU);
            let twin = cur ^ 1; // the reverse half-edge of `cur`
            let Some(cands) = out_at.get(&arrive_to) else {
                ok = false;
                break;
            };
            // Pick the candidate minimizing the CCW turn from `back_angle`,
            // excluding the immediate twin (which would U-turn). The smallest
            // positive CCW offset hugs the boundary on the left = minimal face.
            //
            // Chords leaving along one line (a whole arc's chord on another
            // edge's line) tie, and are ordered by their true directions: a
            // candidate along the way back sits just after it or just before
            // a full turn, and two tied candidates go clockwise first.
            let tie = 1e-9;
            let turn = |a: f64, b: f64| {
                (a - b + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI
            };
            let mut best: Option<usize> = None;
            let mut best_off = f64::MAX;
            for &c in cands {
                if used[c] || c == twin {
                    continue;
                }
                let mut off = (halfs[c].angle - back_angle).rem_euclid(std::f64::consts::TAU);
                if off < tie || off > std::f64::consts::TAU - tie {
                    let side = turn(halfs[c].tangent, halfs[twin].tangent);
                    if side > tie {
                        off = 0.0;
                    } else if side < -tie {
                        off = std::f64::consts::TAU;
                    }
                }
                let better = match best {
                    Some(b) if (off - best_off).abs() <= tie => {
                        let t = turn(halfs[c].tangent, halfs[b].tangent);
                        if t.abs() > tie {
                            t < 0.0
                        } else {
                            off < best_off
                        }
                    }
                    _ => off < best_off,
                };
                if better {
                    best_off = off;
                    best = Some(c);
                }
            }
            // If the only continuation is the twin (dangling edge), allow it so
            // the trace can retreat; otherwise abandon.
            let next = best.or_else(|| cands.iter().copied().find(|&c| !used[c] && c == twin));
            let Some(next) = next else {
                ok = false;
                break;
            };
            if next == start {
                break;
            }
            cur = next;
            if face.len() > halfs.len() {
                ok = false;
                break;
            }
        }
        if ok && face.len() >= 3 {
            faces.push(face);
        }
    }
    if faces.is_empty() {
        return None;
    }

    // Every simple arrangement that tiles a bounded region produces exactly one
    // unbounded "outer" face whose boundary trace re-walks the region perimeter;
    // its |area| equals the sum of all interior face |areas| and is therefore
    // strictly the largest single magnitude. Drop that one face and keep the
    // rest as interior regions — this is independent of the boundary winding
    // (which can be CW for a cavity wall, CCW for an outer wall).
    let face_area = |face: &[usize]| -> f64 {
        let pts: Vec<Point2> = face.iter().map(|&h| vert_pos[&halfs[h].from]).collect();
        signed_area_2d(&pts)
    };
    if faces.len() < 2 {
        return None;
    }
    // UV polygon of a traced face, for containment/probe tests. Whole-arc
    // sub-edges are densified by sampling their true 3D curve through the
    // frame (orientation-unambiguous): on a thin arc-bounded region (the
    // groove-mouth corner sliver, two lines + one r=4 arc) the chord polygon
    // misplaces the interior probe across the arc, misclassifying the region.
    let face_poly = |face: &[usize]| -> Vec<Point2> {
        let mut pts: Vec<Point2> = Vec::with_capacity(face.len() * 2);
        for &h in face {
            let he = &halfs[h];
            let su = vert_pos[&he.from];
            pts.push(su);
            let Some(&(input_idx, whole)) = sub_edge_src.get(he.seg_id) else {
                continue;
            };
            let inp = &inputs[input_idx];
            if !(whole && inp.is_arc) {
                continue;
            }
            let e = &inp.edge;
            let (a3, b3) = if e.forward {
                (e.start_3d, e.end_3d)
            } else {
                (e.end_3d, e.start_3d)
            };
            let (d0, d1) = e.curve_3d.domain_with_endpoints(a3, b3);
            let fwd_uv = frame.project(a3);
            let rev_uv = frame.project(b3);
            let from_matches_fwd = (fwd_uv - su).length() <= (rev_uv - su).length();
            for k in 1..8 {
                let f = f64::from(k) / 8.0;
                let f = if from_matches_fwd { f } else { 1.0 - f };
                let p3 = e
                    .curve_3d
                    .evaluate_with_endpoints(d0 + (d1 - d0) * f, a3, b3);
                pts.push(frame.project(p3));
            }
        }
        pts
    };
    // Measured with its arcs: a boundary of long arcs (a plate's rounded
    // outline) has a chord polygon smaller than a component inside it (a
    // disc's rim through the plate's holes), whose unbounded cycle the
    // chords would then keep as a region while dropping the face's own.
    let outer_idx = (0..faces.len()).max_by(|&a, &b| {
        signed_area_2d(&face_poly(&faces[a]))
            .abs()
            .partial_cmp(&signed_area_2d(&face_poly(&faces[b])).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    let interior: Vec<&Vec<usize>> = faces
        .iter()
        .enumerate()
        .filter(|(i, f)| *i != outer_idx && face_area(f).abs() > tol * tol)
        .map(|(_, f)| f)
        .collect();
    if interior.is_empty() {
        return None;
    }

    // Build sub-faces. Map each half-edge to an OrientedPCurveEdge line in UV
    // with 3D from the plane frame. A whole arc input is emitted with its true
    // curve geometry (oriented to match the requested direction).
    let mk_edge = |from: (i64, i64), to: (i64, i64), seg_id: usize| -> Option<OrientedPCurveEdge> {
        let su = vert_pos[&from];
        let eu = vert_pos[&to];
        // Reconstruct a whole arc input exactly.
        if let Some(&(input_idx, whole)) = sub_edge_src.get(seg_id) {
            let inp = &inputs[input_idx];
            if !whole && inp.is_arc {
                // Trimmed sub-arc: the same true curve with narrower endpoints
                // (the endpoint-trimmed convention every consumer follows via
                // `evaluate_with_endpoints`/`domain_with_endpoints`). Vertices
                // came from exact true-crossing/T registration, so the 3D from
                // the frame matches the neighbouring pieces.
                let s3 = exact3d
                    .get(&from)
                    .copied()
                    .unwrap_or_else(|| frame.evaluate(su.x(), su.y()));
                let e3 = exact3d
                    .get(&to)
                    .copied()
                    .unwrap_or_else(|| frame.evaluate(eu.x(), eu.y()));
                let pcurve = super::pcurve_compute::compute_pcurve_on_surface(
                    &inp.edge.curve_3d,
                    s3,
                    e3,
                    surface,
                    &[],
                    Some(frame),
                );
                let cd = inp.b - inp.a;
                let cl2 = cd.dot(cd).max(1e-24);
                let same_dir = (su - inp.a).dot(cd) / cl2 <= (eu - inp.a).dot(cd) / cl2;
                return Some(OrientedPCurveEdge {
                    curve_3d: inp.edge.curve_3d.clone(),
                    pcurve,
                    start_uv: su,
                    end_uv: eu,
                    start_3d: s3,
                    end_3d: e3,
                    forward: if same_dir {
                        inp.edge.forward
                    } else {
                        !inp.edge.forward
                    },
                    source_edge_idx: None,
                    // A sub-span of the section must not inherit the parent's
                    // pave_block_id — vertex resolution would snap both halves
                    // to the un-split PaveBlock endpoints.
                    pave_block_id: None,
                });
            }
            if whole && inp.is_arc {
                // Does the requested from->to match the input's a->b chord?
                let forward = (inp.a - su).length() < (inp.b - su).length();
                let base = &inp.edge;
                return Some(if forward {
                    base.clone()
                } else {
                    OrientedPCurveEdge {
                        curve_3d: base.curve_3d.clone(),
                        pcurve: base.pcurve.clone(),
                        start_uv: base.end_uv,
                        end_uv: base.start_uv,
                        start_3d: base.end_3d,
                        end_3d: base.start_3d,
                        forward: !base.forward,
                        // `None` (carried from `base`, where every input edge is
                        // built with `source_edge_idx: None`): these arrangement
                        // sub-faces are written straight to topology by
                        // `build_topology_face`, which does NOT weld via
                        // `source_edge_idx` (its `_shared_edge_cache` is unused).
                        // Each sub-face creates its own edges; the two directed
                        // uses of a shared interior edge carry identical 3D
                        // endpoints, so `merge_duplicate_edges` (position-keyed,
                        // post-build) unifies them. `source_edge_idx` is read only
                        // by the angular wire builder, which this path bypasses.
                        source_edge_idx: None,
                        pave_block_id: base.pave_block_id,
                    }
                });
            }
        }
        let dir = eu - su;
        let len = dir.length();
        let direction = if len > 1e-12 {
            Vec2::new(dir.x() / len, dir.y() / len)
        } else {
            Vec2::new(1.0, 0.0)
        };
        let pcurve = Curve2D::Line(
            Line2D::new(su, direction)
                .or_else(|_| Line2D::new(su, Vec2::new(1.0, 0.0)))
                .ok()?,
        );
        Some(OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: su,
            end_uv: eu,
            start_3d: exact3d
                .get(&from)
                .copied()
                .unwrap_or_else(|| frame.evaluate(su.x(), su.y())),
            end_3d: exact3d
                .get(&to)
                .copied()
                .unwrap_or_else(|| frame.evaluate(eu.x(), eu.y())),
            forward: true,
            source_edge_idx: Some(seg_id),
            pave_block_id: None,
        })
    };

    // Build a CCW wire (valid outer-wire winding) from a traced face.
    let build_ccw_wire = |face: &[usize]| -> Option<Vec<OrientedPCurveEdge>> {
        let reverse_each = face_area(face) < 0.0;
        let ccw: Vec<usize> = if reverse_each {
            face.iter().rev().copied().collect()
        } else {
            face.to_vec()
        };
        let mut wire = Vec::with_capacity(ccw.len());
        for &h in &ccw {
            let he = &halfs[h];
            let (from, to) = if reverse_each {
                (he.to, he.from)
            } else {
                (he.from, he.to)
            };
            wire.push(mk_edge(from, to, he.seg_id)?);
        }
        Some(wire)
    };

    if nesting == ArrNesting::Cells {
        // The trace turns as far right as it can, so a bounded face comes back
        // clockwise and each connected component's outline counter-clockwise.
        let mut component: std::collections::HashMap<UvKey, usize> =
            std::collections::HashMap::new();
        for (id, &(root, _)) in sub_edges.iter().enumerate() {
            if component.contains_key(&root) {
                continue;
            }
            let mut stack = vec![root];
            while let Some(v) = stack.pop() {
                if component.insert(v, id).is_none() {
                    for &(a, b) in &sub_edges {
                        if a == v {
                            stack.push(b);
                        } else if b == v {
                            stack.push(a);
                        }
                    }
                }
            }
        }
        let component_of = |f: &[usize]| component.get(&halfs[f[0]].from).copied();
        let cells: Vec<&Vec<usize>> = faces.iter().filter(|f| face_area(f) < -tol * tol).collect();
        let polys: Vec<Vec<Point2>> = cells.iter().map(|f| face_poly(f)).collect();
        let areas: Vec<f64> = cells.iter().map(|f| face_area(f).abs()).collect();
        let mut holes_of: Vec<Vec<Vec<OrientedPCurveEdge>>> = vec![Vec::new(); cells.len()];
        for outline in faces.iter().filter(|f| face_area(f) > tol * tol) {
            let area = face_area(outline);
            let ring = face_poly(outline);
            let holder = (0..cells.len())
                .filter(|&c| {
                    let eps = super::classify_2d::boundary_eps(&polys[c]);
                    component_of(cells[c]) != component_of(outline)
                        && areas[c] > area
                        && ring.iter().all(|&v| {
                            super::classify_2d::point_in_polygon_2d(v, &polys[c])
                                || super::classify_2d::distance_to_polygon_boundary(v, &polys[c])
                                    <= eps
                        })
                })
                .min_by(|&x, &y| areas[x].total_cmp(&areas[y]));
            if let Some(holder) = holder
                && let Some(mut wire) = build_ccw_wire(outline)
            {
                wire.reverse();
                for e in &mut wire {
                    std::mem::swap(&mut e.start_uv, &mut e.end_uv);
                    std::mem::swap(&mut e.start_3d, &mut e.end_3d);
                    e.forward = !e.forward;
                }
                holes_of[holder].push(wire);
            }
        }
        let air: Vec<Vec<Point2>> = original_holes
            .iter()
            .map(|w| sample_wire_loop_uv(w))
            .filter(|p| p.len() >= 3)
            .collect();
        let mut result = Vec::with_capacity(cells.len());
        for (c, inner) in holes_of.into_iter().enumerate() {
            let seed = if inner.is_empty() {
                sample_interior_point(&polys[c])
            } else {
                find_point_outside_holes(&polys[c], &inner, Some(frame))
            };
            if air
                .iter()
                .any(|hole| super::classify_2d::point_in_polygon_2d(seed, hole))
            {
                continue;
            }
            let outer_wire = build_ccw_wire(cells[c])?;
            let precomputed_interior =
                (!inner.is_empty()).then(|| frame.evaluate(seed.x(), seed.y()));
            result.push(SplitSubFace {
                surface: surface.clone(),
                outer_wire,
                inner_wires: inner,
                reversed,
                parent: face_id,
                rank,
                precomputed_interior,
            });
        }
        return (!result.is_empty()).then_some(result);
    }

    if nesting == ArrNesting::EvenOdd {
        // Even-odd nesting (holed-cap path). Minimal-face tracing of a
        // disconnected arrangement (a holed cap whose holes/sections form an
        // inner component separate from the outer boundary loop) yields nested
        // faces that OVERLAP: the outer-boundary disk covers the whole face, an
        // inner-component disk covers its sub-region, and the true regions sit
        // inside those. Resolve by containment depth: depth-even faces are solid
        // regions, depth-odd faces are holes in their container. Emit each solid
        // face with its DIRECT-child (depth+1) faces as inner wires, so a solid
        // disk that contains an inner disk becomes the correct annulus (e.g. the
        // ±41.75 cap perimeter ring around the ±40.55 inner-wall opening). Drop
        // any solid region whose interior lies in an original hole (air).
        let polys: Vec<Vec<Point2>> = interior.iter().map(|f| face_poly(f)).collect();
        let probes: Vec<Point2> = polys.iter().map(|p| sample_interior_point(p)).collect();
        let areas: Vec<f64> = polys.iter().map(|p| signed_area_2d(p).abs()).collect();
        // A CLEAN tiling — the interior faces' areas sum to the outer face's
        // area — is a proper subdivision: every region is simply connected and
        // nesting never applies. Without this cut-off, an arrangement whose
        // hole rings connect to the outer boundary (a groove mouth crossing
        // two pocket openings) has every region adjacent to the big material
        // region judged "contained" through the on-boundary tolerance below,
        // so ring cycles already present in the material outer wire get
        // re-attached as holes (double cover) and the mouth sliver is
        // swallowed instead of being emitted for classification. A trace with
        // OVERLAPPING faces (disconnected or bridge-connected components — a
        // holed cap's twin loops, the divider-lip weave) sums to MORE than the
        // outer area, and keeps the containment-depth nesting that resolves
        // it. The outer face is measured with its arcs, as the regions are: by
        // its chords a rim of arcs reads smaller than the regions it holds.
        let outer_area = signed_area_2d(&face_poly(&faces[outer_idx])).abs();
        let interior_area_sum: f64 = areas.iter().sum();
        let clean_tiling =
            (interior_area_sum - outer_area).abs() <= outer_area.madd(1e-6, tol * tol);
        // `outer` contains `inner` when EVERY one of inner's polygon vertices
        // lies inside (or on) outer's polygon AND outer is strictly larger.
        // A probe-only test is symmetric for the concentric disks that
        // disconnected components produce (both the ±41.75 and ±40.55 disks hold
        // the centre), so it can't order their nesting; the all-vertices +
        // larger-area test is asymmetric and orders them correctly.
        let contains = |outer: usize, inner: usize| -> bool {
            if clean_tiling || outer == inner || areas[outer] <= areas[inner] {
                return false;
            }
            let op = &polys[outer];
            let eps = super::classify_2d::boundary_eps(op);
            polys[inner].iter().all(|&v| {
                super::classify_2d::point_in_polygon_2d(v, op)
                    || super::classify_2d::distance_to_polygon_boundary(v, op) <= eps
            })
        };
        let depth: Vec<usize> = (0..interior.len())
            .map(|i| (0..interior.len()).filter(|&j| contains(j, i)).count())
            .collect();
        // Direct parent = the deepest container (max depth among containers).
        let direct_parent = |i: usize| -> Option<usize> {
            (0..interior.len())
                .filter(|&j| contains(j, i))
                .max_by_key(|&j| depth[j])
        };
        let hole_polys: Vec<Vec<Point2>> = original_holes
            .iter()
            .map(|w| sample_wire_loop_uv(w))
            .filter(|p| p.len() >= 3)
            .collect();

        let mut result = Vec::new();
        for i in 0..interior.len() {
            if !depth[i].is_multiple_of(2) {
                continue; // odd depth = hole in its container, not a solid region
            }
            // Direct children (depth+1, parented to i) become inner wires (the
            // holes of this solid region). A child that is itself an original
            // hole still bounds this region, so include it regardless.
            let mut inner = Vec::new();
            for j in 0..interior.len() {
                if depth[j] == depth[i] + 1 && direct_parent(j) == Some(i) {
                    // Inner wires must wind opposite the outer (CW); build_ccw
                    // gives CCW, so reverse.
                    if let Some(mut w) = build_ccw_wire(interior[j]) {
                        w.reverse();
                        for e in &mut w {
                            std::mem::swap(&mut e.start_uv, &mut e.end_uv);
                            std::mem::swap(&mut e.start_3d, &mut e.end_3d);
                            e.forward = !e.forward;
                        }
                        inner.push(w);
                    }
                }
            }
            // The region can be annular/non-convex; the classifier seed must lie
            // in the material, i.e. inside the outer polygon but OUTSIDE every
            // inner-wire (child) hole. `sample_interior_point` on the outer alone
            // can land in a child hole (e.g. a perimeter ring's centre sits in
            // its inner-wall opening), which would misclassify the region.
            let seed_uv = if inner.is_empty() {
                probes[i]
            } else {
                find_point_outside_holes(&polys[i], &inner, Some(frame))
            };
            // Drop a solid region that fills an original hole (air, not material).
            // Probe at the MATERIAL seed (outside this region's own inner-wire
            // holes), not the raw outer-polygon centroid: a large holed cap (a
            // 16-cell baseplate top) has a centroid that can land inside one of
            // its own cell openings, which would wrongly discard the entire cap.
            if hole_polys
                .iter()
                .any(|poly| super::classify_2d::point_in_polygon_2d(seed_uv, poly))
            {
                continue;
            }
            let Some(outer_wire) = build_ccw_wire(interior[i]) else {
                continue;
            };
            result.push(SplitSubFace {
                surface: surface.clone(),
                outer_wire,
                inner_wires: inner,
                reversed,
                parent: face_id,
                rank,
                precomputed_interior: Some(frame.evaluate(seed_uv.x(), seed_uv.y())),
            });
        }
        if result.is_empty() {
            return None;
        }
        if let Some(reg) = split_registry {
            for (pb_id, p) in pending_breaks {
                reg.entry(pb_id).or_default().push(p);
            }
        }
        return Some(result);
    }

    // A disconnected component of the arrangement — a closed section loop
    // touching neither the face boundary nor any other section — is traced
    // TWICE, once per orientation, because both directed cycles bound a
    // region of the trace graph. Flat emission would then duplicate the
    // loop's region AND leave the region that geometrically contains the
    // loop without a hole for it, so the container overlaps the duplicates
    // (the halfSockets socket fuse: four interior cell outlines emitted as
    // twin discs under a hole-less web face, collapsing the whole z=5
    // interface into one same-domain group that dropped every piece).
    // Resolve each twin pair: emit one cycle as the solid region and attach
    // the reversed twin as an inner wire of its direct container.
    let face_key = |f: &[usize], twin: bool| -> Vec<usize> {
        let mut k: Vec<usize> = f.iter().map(|&h| if twin { h ^ 1 } else { h }).collect();
        k.sort_unstable();
        k
    };
    let mut by_key: std::collections::HashMap<Vec<usize>, usize> = std::collections::HashMap::new();
    for (i, f) in interior.iter().enumerate() {
        by_key.insert(face_key(f, false), i);
    }
    // Pass 1: collect twin pairs — two traced faces whose half-edge sets are
    // exact twins (h ↔ h^1) with opposite winding. The lower-indexed member
    // stays a candidate solid region; the other becomes the hole cycle.
    // Whether the trace winds solid regions CCW or CW depends on the face's
    // boundary winding in the frame (a cavity wall winds CW), so no sign is
    // assumed — emission normalizes windings anyway.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut is_hole_cycle = vec![false; interior.len()];
    for i in 0..interior.len() {
        if is_hole_cycle[i] {
            continue;
        }
        let Some(&j) = by_key.get(&face_key(interior[i], true)) else {
            continue;
        };
        if j <= i || is_hole_cycle[j] {
            continue;
        }
        if (face_area(interior[i]) > 0.0) == (face_area(interior[j]) > 0.0) {
            continue;
        }
        is_hole_cycle[j] = true;
        pairs.push((i, j));
    }
    // Pass 2: attach each pair's hole cycle to its direct container — the
    // smallest-|area| non-hole face that contains the whole loop. As in the
    // even-odd nesting pass, containment is the asymmetric all-vertices +
    // strictly-larger-area test: a single interior probe would tie for
    // nested loops (the outer loop's probe can land inside the inner loop's
    // kept disc, parenting the hole to the wrong face) and rejects
    // boundary-touching containers without a tolerance band. A kept twin (a
    // nested loop's disc) is a legitimate container.
    let mut hole_twin: Vec<Option<usize>> = vec![None; interior.len()];
    for &(keep, hole) in &pairs {
        let loop_poly = face_poly(interior[keep]);
        let loop_area = face_area(interior[keep]).abs();
        let parent = (0..interior.len())
            .filter(|&k| {
                if k == keep || is_hole_cycle[k] || face_area(interior[k]).abs() <= loop_area {
                    return false;
                }
                let poly = face_poly(interior[k]);
                let eps = super::classify_2d::boundary_eps(&poly);
                loop_poly.iter().all(|&v| {
                    super::classify_2d::point_in_polygon_2d(v, &poly)
                        || super::classify_2d::distance_to_polygon_boundary(v, &poly) <= eps
                })
            })
            .min_by(|&a, &b| {
                face_area(interior[a])
                    .abs()
                    .partial_cmp(&face_area(interior[b]).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(parent) = parent {
            hole_twin[hole] = Some(parent);
        } else {
            // No container found (pathological: chord-sampled candidate
            // polygons under-covering a curved boundary). The loop's region
            // is still emitted once via `keep`; only the container's hole is
            // lost. Emitting the twin instead would recreate the duplicate
            // overlapping region this pass exists to prevent.
            log::debug!(
                "arrangement twin loop found no containing region for face {face_id:?}; \
                 hole not attached"
            );
        }
    }

    let mut inner_wires_of: Vec<Vec<Vec<OrientedPCurveEdge>>> = vec![Vec::new(); interior.len()];
    for (i, parent) in hole_twin.iter().enumerate() {
        let Some(parent) = *parent else { continue };
        // Inner wires must wind opposite the outer (CW); build_ccw gives
        // CCW, so reverse.
        if let Some(mut w) = build_ccw_wire(interior[i]) {
            w.reverse();
            for e in &mut w {
                std::mem::swap(&mut e.start_uv, &mut e.end_uv);
                std::mem::swap(&mut e.start_3d, &mut e.end_3d);
                e.forward = !e.forward;
            }
            inner_wires_of[parent].push(w);
        }
    }

    let mut result = Vec::new();
    for (i, face) in interior.iter().enumerate() {
        if is_hole_cycle[i] {
            continue;
        }
        let Some(wire) = build_ccw_wire(face) else {
            continue;
        };
        let inner = std::mem::take(&mut inner_wires_of[i]);
        // With holes attached, the generic interior sampler can land inside
        // one; pin a seed that avoids every hole. Hole-less regions keep
        // None: a region can be non-convex (an L), so the centroid is
        // unsafe, and `interior_point_3d` derives a robust interior sample.
        let precomputed_interior = if inner.is_empty() {
            None
        } else {
            let seed = find_point_outside_holes(&face_poly(face), &inner, Some(frame));
            Some(frame.evaluate(seed.x(), seed.y()))
        };
        result.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: inner,
            reversed,
            parent: face_id,
            rank,
            precomputed_interior,
        });
    }
    if result.is_empty() {
        return None;
    }
    if let Some(reg) = split_registry {
        for (pb_id, p) in pending_breaks {
            reg.entry(pb_id).or_default().push(p);
        }
    }
    Some(result)
}

/// Rectilinear-arrangement rescue for a u-periodic cylinder band whose greedy
/// wire trace self-crosses (a box cut notching the wall at partial overlap),
/// or traces cleanly but only part of the band, which the full face trace
/// refines (a mirrored cylinder, whose rims run against its u).
///
/// On a cylinder every edge is axis-aligned in UV: cross-section rings
/// (`EdgeCurve::Circle`) are horizontal (`v` const, `u` varies) and the seam and
/// side generators (`EdgeCurve::Line`) are vertical (`u` const, `v` varies). The
/// periodic `u` is cut open at the face seam (a boundary vertical generator) into
/// a planar strip `[u_s, u_s + 2π]`, with the seam duplicated to both strip edges
/// so a region wrapping the seam is bounded. The stored pcurve UV is unreliable
/// (a rim endpoint at the seam carries `u` wrapped by an extra 2π; the GFA's ring
/// FRAGMENTS disagree between 2D and 3D and over-cover kept arcs), so all
/// coordinates come from the exact 3D projection and the removed rectangles are
/// reconstructed from the reliable side generators: sorted by `u`, they pair
/// `(0,1),(2,3),...` into removed sectors by kept/removed alternation from the
/// (kept) seam. The planar subdivision of the full rims + seam + removed
/// rectangles is traced into minimal faces (leftmost-turn rule), the unbounded
/// outer face dropped, and each interior region emitted as a [`SplitSubFace`]
/// partition piece -- downstream classification decides material in/out.
///
/// Returns `None` (defer to the greedy path) on any geometry that is not a clean
/// rectilinear box notch: a non-axis-aligned line or circle, an ellipse/NURBS
/// section, a closed full-ring section, a missing seam, an odd number of side
/// generators, a generator pair that does not bound a rectangle, or a trace that
/// fails to yield a simple partition. The gates ensure this fires only when the
/// greedy trace is broken or under-split, so faces the greedy handles are
/// untouched. The face's holes do not enter it: callers gate on a hole-free
/// face.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn split_cylinder_band_by_arrangement(
    surface: &FaceSurface,
    all_edges: &[OrientedPCurveEdge],
    n_boundary_edges: usize,
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    use std::collections::HashMap;
    use std::f64::consts::TAU;

    use brepkit_math::curves::Circle3D;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::Vec2;

    // A vertical generator has |Δu| below this; a horizontal ring |Δv| below.
    // Generators have Δu exactly 0 and rings Δv exactly 0, so a straight line
    // that is NOT axis-parallel (helix chord) or a "circle" that is not a
    // cross-section is rejected -- the function only handles rectilinear cuts.
    const EPS_U: f64 = 1e-4;
    const EPS_V: f64 = 1e-4;
    // Snap band around the seam (u = 0 and u = 2π): the nearest real generator
    // sits ~0.7 rad away, so points within this band are the seam itself (both
    // copies must weld to a single 3D meridian).
    const SEAM_BAND: f64 = 1e-3;

    let FaceSurface::Cylinder(cyl) = surface else {
        return None;
    };

    let u_weld = tol.max(1e-9);
    let v_weld = tol.max(1e-9);

    // Cylinder (u, v): u angular in [0, 2π), v axial. All coordinates come from
    // the exact 3D projection, not the stored pcurve UV.
    let proj = |p: Point3| -> (f64, f64) { cyl.project_point(p) };

    // Seam anchor: the u of a boundary vertical generator (the meridian the
    // periodic face is cut along). All horizontal edges fit in one period
    // [u_s, u_s + 2π] without crossing it, so anchoring here avoids seam-crossing
    // rings; the seam generator becomes both strip edges.
    let mut seam_u: Option<f64> = None;
    for e in all_edges.iter().take(n_boundary_edges) {
        if !matches!(e.curve_3d, EdgeCurve::Line) || (e.start_3d - e.end_3d).length() <= tol {
            continue;
        }
        let (u0, _) = proj(e.start_3d);
        let (u1, _) = proj(e.end_3d);
        let d = (u1 - u0).rem_euclid(TAU);
        if d.min(TAU - d) >= EPS_U {
            continue; // boundary line is not axis-parallel -- not a generator
        }
        match seam_u {
            None => seam_u = Some(u0),
            Some(prev) => {
                let dd = (u0 - prev).rem_euclid(TAU);
                if dd.min(TAU - dd) > EPS_U {
                    return None; // conflicting boundary generators -- not one seam
                }
            }
        }
    }
    let u_s = seam_u?;

    let snap_u = |u: f64| -> f64 {
        if u.abs() < SEAM_BAND {
            0.0
        } else if (u - TAU).abs() < SEAM_BAND {
            TAU
        } else {
            u
        }
    };

    // Vertex registry: quantized (u_shift, v) key -> (u_shift, v, exact 3D).
    // Prefer the exact input endpoint 3D (a shared pave vertex) over a recomputed
    // point so reconstructed sub-edges match the adjacent faces' edges exactly.
    let mut verts: HashMap<(i64, i64), (f64, f64, Point3)> = HashMap::new();
    let register = |u: f64,
                    v: f64,
                    p3: Point3,
                    verts: &mut HashMap<(i64, i64), (f64, f64, Point3)>|
     -> (i64, i64) {
        let k = ((u / u_weld).round() as i64, (v / v_weld).round() as i64);
        verts.entry(k).or_insert((u, v, p3));
        k
    };

    // Phase A: register every input endpoint (exact 3D) and collect the reliable
    // structure -- section generator pieces (u_shift, v_lo, v_hi). The messy ring
    // FRAGMENTS (which over-cover kept arcs and disagree between 2D and 3D) are
    // used only to confirm the cut is rectilinear; the removed rectangles are
    // reconstructed from the generators, whose u/v are exact projections.
    let mut sec_pieces: Vec<(f64, f64, f64)> = Vec::new(); // (u_shift, v_lo, v_hi)
    let mut sec_rings: Vec<(f64, f64, f64)> = Vec::new(); // (v, u_lo, u_hi)
    for (i, e) in all_edges.iter().enumerate() {
        match &e.curve_3d {
            EdgeCurve::Line => {
                if (e.start_3d - e.end_3d).length() <= tol {
                    continue; // degenerate zero-length line
                }
                let (u0, v0p) = proj(e.start_3d);
                let (u1, v1p) = proj(e.end_3d);
                let du = (u1 - u0).rem_euclid(TAU);
                if du.min(TAU - du) >= EPS_U {
                    return None; // non-vertical line (helix chord) -- defer
                }
                let u = snap_u((u0 - u_s).rem_euclid(TAU));
                let (v0, v1, v0_3d, v1_3d) = if v0p <= v1p {
                    (v0p, v1p, e.start_3d, e.end_3d)
                } else {
                    (v1p, v0p, e.end_3d, e.start_3d)
                };
                register(u, v0, v0_3d, &mut verts);
                register(u, v1, v1_3d, &mut verts);
                if v1 - v0 <= tol {
                    continue;
                }
                // A non-seam generator (a tool side wall). Boundary verticals at
                // the seam are the meridian, not a cut.
                if i >= n_boundary_edges && u > SEAM_BAND && (TAU - u) > SEAM_BAND {
                    sec_pieces.push((u, v0, v1));
                }
            }
            EdgeCurve::Circle(_) => {
                if (e.start_3d - e.end_3d).length() <= tol * 100.0 {
                    return None; // closed full-ring section -- defer to greedy
                }
                let (u0, v0p) = proj(e.start_3d);
                let (u1, v1p) = proj(e.end_3d);
                if (v0p - v1p).abs() >= EPS_V {
                    return None; // non-horizontal circle (tilted section) -- defer
                }
                register(
                    snap_u((u0 - u_s).rem_euclid(TAU)),
                    v0p,
                    e.start_3d,
                    &mut verts,
                );
                register(
                    snap_u((u1 - u_s).rem_euclid(TAU)),
                    v1p,
                    e.end_3d,
                    &mut verts,
                );
                // A ring section over a sector the generators do NOT bound is a
                // real cut too: a partner plane that ends inside the band cuts
                // the arcs where it still exists, not only the sector the tool
                // removed. Reconstructing from the generator pairs alone leaves
                // those arcs uncut, and everything above them stays welded to
                // the material below (the lid magnet-post band, #1517).
                if i >= n_boundary_edges {
                    let (t0, t1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
                    let mid = e.curve_3d.evaluate_with_endpoints(
                        0.5_f64.madd(t1 - t0, t0),
                        e.start_3d,
                        e.end_3d,
                    );
                    let (um, _) = proj(mid);
                    let a = snap_u((u0 - u_s).rem_euclid(TAU));
                    let b = snap_u((u1 - u_s).rem_euclid(TAU));
                    let m = (um - u_s).rem_euclid(TAU);
                    // Endpoints alone leave the arc's side ambiguous; the
                    // midpoint picks it. An arc running the far way round the
                    // seam is emitted as its two strip pieces.
                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                    if m > lo && m < hi {
                        sec_rings.push((v0p, lo, hi));
                    } else {
                        sec_rings.push((v0p, 0.0, lo));
                        sec_rings.push((v0p, hi, TAU));
                    }
                }
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return None,
        }
    }
    if sec_pieces.is_empty() {
        return None;
    }

    // Band v-extent (bottom rim to top rim) from every registered endpoint.
    let (mut v_bottom, mut v_top) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, v, _) in verts.values() {
        v_bottom = v_bottom.min(v);
        v_top = v_top.max(v);
    }
    if v_top - v_bottom <= tol {
        return None;
    }

    // Merge generator pieces sharing a u (fwd/rev and pave splits) into one span.
    let mut gens: Vec<(f64, f64, f64)> = Vec::new();
    for &(u, v0, v1) in &sec_pieces {
        if let Some(g) = gens
            .iter_mut()
            .find(|(gu, _, _)| (u - *gu).abs() <= u_weld * 4.0)
        {
            g.1 = g.1.min(v0);
            g.2 = g.2.max(v1);
        } else {
            gens.push((u, v0, v1));
        }
    }
    gens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Each removed sector is bounded by a generator pair. Going around from the
    // seam (kept material -- it is an original boundary edge), the arcs alternate
    // kept / removed, so consecutive sorted generators pair (0,1),(2,3),... into
    // removed rectangles. An odd count or a pair with mismatched v-spans is not a
    // clean box notch -- defer.
    if !gens.len().is_multiple_of(2) {
        return None;
    }

    let cov = 1e-6;
    let mut verticals: Vec<(f64, f64, f64)> = Vec::new(); // (u_shift, v_lo, v_hi)
    let mut horizontals: Vec<(f64, f64, f64)> = Vec::new(); // (v, u_lo, u_hi)

    // Band frame: full rims and the seam on both strip edges.
    horizontals.push((v_bottom, 0.0, TAU));
    horizontals.push((v_top, 0.0, TAU));
    verticals.push((0.0, v_bottom, v_top));
    verticals.push((TAU, v_bottom, v_top));

    for pair in gens.chunks_exact(2) {
        let (ua, va0, va1) = pair[0];
        let (ub, vb0, vb1) = pair[1];
        if (va0 - vb0).abs() > tol * 100.0 || (va1 - vb1).abs() > tol * 100.0 {
            return None; // pair does not bound a single rectangle
        }
        let (v_lo, v_hi) = (va0.min(vb0), va1.max(vb1));
        verticals.push((ua, v_lo, v_hi));
        verticals.push((ub, v_lo, v_hi));
        if v_hi < v_top - tol {
            horizontals.push((v_hi, ua, ub));
        }
        if v_lo > v_bottom + tol {
            horizontals.push((v_lo, ua, ub));
        }
    }

    // The ring sections themselves, over whatever sectors they actually cover.
    // Rings at the rims are already the band frame.
    for &(v, u_lo, u_hi) in &sec_rings {
        if v > v_bottom + tol && v < v_top - tol && u_hi - u_lo > cov {
            horizontals.push((v, u_lo, u_hi));
        }
    }

    // Rectilinear split: cut each vertical at every horizontal ring's v that its
    // span brackets (and whose u-range covers the generator), and each horizontal
    // at every vertical generator's u interior to its span (whose v-range reaches
    // the ring). Emit undirected sub-segments; dedup collapses duplicates.
    let mut sub: Vec<((i64, i64), (i64, i64))> = Vec::new();
    // Every endpoint the edges brought cuts the frame lines through it too: a
    // ring section broken at a vertex of its own (where a partner face's rim
    // meets it) overlaps the whole ring a generator pair puts at its height,
    // and the two would stand side by side as different edges.
    let marks: Vec<(f64, f64)> = verts.values().map(|&(u, v, _)| (u, v)).collect();

    for &(u, v0, v1) in &verticals {
        let mut breaks: Vec<f64> = vec![v0, v1];
        for &(hv, hu0, hu1) in &horizontals {
            if hv > v0 + tol && hv < v1 - tol && u >= hu0 - cov && u <= hu1 + cov {
                breaks.push(hv);
            }
        }
        // Read round the turn: a vertex on the seam registers at u = 0 and
        // splits both of the seam's copies, at 0 and at 2 pi.
        for &(mu, mv) in &marks {
            let du = (mu - u).rem_euclid(TAU);
            if du.min(TAU - du) <= cov && mv > v0 + tol && mv < v1 - tol {
                breaks.push(mv);
            }
        }
        breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        breaks.dedup_by(|a, b| (*a - *b).abs() <= tol);
        for w in breaks.windows(2) {
            let (va, vb) = (w[0], w[1]);
            if vb - va <= tol {
                continue;
            }
            let ka = register(u, va, cyl.evaluate(u_s + u, va), &mut verts);
            let kb = register(u, vb, cyl.evaluate(u_s + u, vb), &mut verts);
            if ka != kb {
                sub.push((ka, kb));
            }
        }
    }

    for &(v, u0, u1) in &horizontals {
        let mut breaks: Vec<f64> = vec![u0, u1];
        for &(vu, vv0, vv1) in &verticals {
            if vu > u0 + cov && vu < u1 - cov && v >= vv0 - tol && v <= vv1 + tol {
                breaks.push(vu);
            }
        }
        for &(mu, mv) in &marks {
            if (mv - v).abs() <= tol && mu > u0 + cov && mu < u1 - cov {
                breaks.push(mu);
            }
        }
        breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        breaks.dedup_by(|a, b| (*a - *b).abs() <= cov);
        for w in breaks.windows(2) {
            let (ua, ub) = (w[0], w[1]);
            if ub - ua <= cov {
                continue;
            }
            let ka = register(ua, v, cyl.evaluate(u_s + ua, v), &mut verts);
            let kb = register(ub, v, cyl.evaluate(u_s + ub, v), &mut verts);
            if ka != kb {
                sub.push((ka, kb));
            }
        }
    }

    // Dedup undirected sub-edges by their vertex-key pair.
    sub.sort_by(|l, r| {
        let lk = if l.0 <= l.1 { (l.0, l.1) } else { (l.1, l.0) };
        let rk = if r.0 <= r.1 { (r.0, r.1) } else { (r.1, r.0) };
        lk.cmp(&rk)
    });
    sub.dedup_by(|a, b| {
        let ak = if a.0 <= a.1 { (a.0, a.1) } else { (a.1, a.0) };
        let bk = if b.0 <= b.1 { (b.0, b.1) } else { (b.1, b.0) };
        ak == bk
    });
    if sub.len() < 3 {
        return None;
    }

    // Shifted-uv position per vertex, for angles / areas / interior sampling.
    let pos: HashMap<(i64, i64), Point2> = verts
        .iter()
        .map(|(&k, &(u, v, _))| (k, Point2::new(u, v)))
        .collect();

    // Directed half-edges: index 2k = forward, 2k+1 = reverse.
    let mut halfs: Vec<ArrHalfEdge> = Vec::with_capacity(sub.len() * 2);
    for (seg_id, &(ka, kb)) in sub.iter().enumerate() {
        let pa = *pos.get(&ka)?;
        let pb = *pos.get(&kb)?;
        let fwd = pb - pa;
        let rev = pa - pb;
        halfs.push(ArrHalfEdge {
            from: ka,
            to: kb,
            seg_id,
            angle: fwd.y().atan2(fwd.x()),
            tangent: fwd.y().atan2(fwd.x()),
        });
        halfs.push(ArrHalfEdge {
            from: kb,
            to: ka,
            seg_id,
            angle: rev.y().atan2(rev.x()),
            tangent: rev.y().atan2(rev.x()),
        });
    }

    let mut out_at: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (hi, h) in halfs.iter().enumerate() {
        out_at.entry(h.from).or_default().push(hi);
    }

    // Trace minimal faces: at each arrival vertex pick the outgoing half-edge with
    // the smallest CCW turn from the reverse-of-arrival direction (leftmost turn),
    // excluding the immediate twin. Same rule as the plane arrangement tracer.
    let mut used = vec![false; halfs.len()];
    let mut faces: Vec<Vec<usize>> = Vec::new();
    for start in 0..halfs.len() {
        if used[start] {
            continue;
        }
        let mut face: Vec<usize> = Vec::new();
        let mut cur = start;
        let mut ok = true;
        loop {
            if used[cur] {
                ok = cur == start && !face.is_empty();
                break;
            }
            used[cur] = true;
            face.push(cur);
            let arrive_to = halfs[cur].to;
            if arrive_to == halfs[start].from && !face.is_empty() {
                break;
            }
            let back_angle = (halfs[cur].angle + std::f64::consts::PI).rem_euclid(TAU);
            let twin = cur ^ 1;
            let Some(cands) = out_at.get(&arrive_to) else {
                ok = false;
                break;
            };
            let mut best: Option<usize> = None;
            let mut best_off = f64::MAX;
            for &c in cands {
                if used[c] || c == twin {
                    continue;
                }
                let off = (halfs[c].angle - back_angle).rem_euclid(TAU);
                if off < best_off {
                    best_off = off;
                    best = Some(c);
                }
            }
            let next = best.or_else(|| cands.iter().copied().find(|&c| !used[c] && c == twin));
            let Some(next) = next else {
                ok = false;
                break;
            };
            if next == start {
                break;
            }
            cur = next;
            if face.len() > halfs.len() {
                ok = false;
                break;
            }
        }
        if ok && face.len() >= 3 {
            faces.push(face);
        }
    }
    if faces.len() < 2 {
        return None;
    }

    let face_area = |face: &[usize]| -> f64 {
        let pts: Vec<Point2> = face
            .iter()
            .filter_map(|&h| pos.get(&halfs[h].from).copied())
            .collect();
        signed_area_2d(&pts)
    };
    // The unbounded outer face re-walks the whole strip perimeter, so its |area|
    // equals the sum of every interior region and is the single largest. Drop it.
    let outer_idx = (0..faces.len()).max_by(|&a, &b| {
        face_area(&faces[a])
            .abs()
            .partial_cmp(&face_area(&faces[b]).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let interior: Vec<&Vec<usize>> = faces
        .iter()
        .enumerate()
        .filter(|(i, f)| *i != outer_idx && face_area(f).abs() > tol * tol)
        .map(|(_, f)| f)
        .collect();
    if interior.is_empty() {
        return None;
    }

    // Disconnected interior loop (traced twice as exact half-edge twins) would
    // need hole nesting this rectilinear path does not do -- defer to the greedy
    // path rather than emit a double-covered region.
    let mut face_keys: Vec<Vec<usize>> = interior
        .iter()
        .map(|f| {
            let mut segs: Vec<usize> = f.iter().map(|&h| halfs[h].seg_id).collect();
            segs.sort_unstable();
            segs
        })
        .collect();
    face_keys.sort();
    if face_keys.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }

    // Reconstruct a directed sub-edge (from -> to) as an exact cylinder edge:
    // a vertical generator is a Line, a horizontal ring a cross-section Circle.
    let mk_edge = |from: (i64, i64), to: (i64, i64)| -> Option<OrientedPCurveEdge> {
        let &(fu, fv, f3d) = verts.get(&from)?;
        let &(tu, tv, t3d) = verts.get(&to)?;
        let s_uv = Point2::new(u_s + fu, fv);
        let e_uv = Point2::new(u_s + tu, tv);
        let dir = e_uv - s_uv;
        let len = dir.length();
        let direction = if len > 1e-12 {
            Vec2::new(dir.x() / len, dir.y() / len)
        } else {
            Vec2::new(1.0, 0.0)
        };
        let pcurve = Curve2D::Line(
            Line2D::new(s_uv, direction)
                .or_else(|_| Line2D::new(s_uv, Vec2::new(1.0, 0.0)))
                .ok()?,
        );
        if (fu - tu).abs() < EPS_U {
            Some(OrientedPCurveEdge {
                curve_3d: EdgeCurve::Line,
                pcurve,
                start_uv: s_uv,
                end_uv: e_uv,
                start_3d: f3d,
                end_3d: t3d,
                forward: true,
                source_edge_idx: None,
                pave_block_id: None,
            })
        } else {
            // Cross-section circle at height v = fv, sharing the cylinder's frame
            // so its parameterization matches (an open arc spans the CCW range
            // from start to end; forward flips it when the wire runs decreasing-u).
            let center = cyl.origin() + cyl.axis() * fv;
            let circle =
                Circle3D::with_axes(center, cyl.axis(), cyl.radius(), cyl.x_axis(), cyl.y_axis())
                    .ok()?;
            Some(OrientedPCurveEdge {
                curve_3d: EdgeCurve::Circle(circle),
                pcurve,
                start_uv: s_uv,
                end_uv: e_uv,
                start_3d: f3d,
                end_3d: t3d,
                forward: tu > fu,
                source_edge_idx: None,
                pave_block_id: None,
            })
        }
    };

    // Build a CCW-wound outer wire (positive shifted-uv area) from a traced face.
    let build_ccw_wire = |face: &[usize]| -> Option<Vec<OrientedPCurveEdge>> {
        let reverse_each = face_area(face) < 0.0;
        let ordered: Vec<usize> = if reverse_each {
            face.iter().rev().copied().collect()
        } else {
            face.to_vec()
        };
        let mut wire = Vec::with_capacity(ordered.len());
        for &h in &ordered {
            let (from, to) = if reverse_each {
                (halfs[h].to, halfs[h].from)
            } else {
                (halfs[h].from, halfs[h].to)
            };
            wire.push(mk_edge(from, to)?);
        }
        Some(wire)
    };

    let mut result = Vec::new();
    for face in interior {
        let poly: Vec<Point2> = face
            .iter()
            .filter_map(|&h| pos.get(&halfs[h].from).copied())
            .collect();
        if poly.len() < 3 {
            continue;
        }
        let Some(outer_wire) = build_ccw_wire(face) else {
            continue;
        };
        let seed = sample_interior_point(&poly);
        result.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(cyl.evaluate(u_s + seed.x(), seed.y())),
        });
    }
    (!result.is_empty()).then_some(result)
}

/// Drop sections that run OFF-FACE through a concavity of the OUTER wire
/// (a cylinder/cone wall whose boundary carries an earlier cut's bite).
///
/// The inner-wire air filter above `split_face_2d`'s arrangement only tests
/// holes; a deepening cut's sections can overhang the face through the outer
/// boundary instead (the snapClip deepened notch: the new box cutter's rim
/// arc lies entirely inside the old notch bite, and the wall-section tails
/// hug the old boundary edge 1e-4 off it — marched fits of the same exact
/// intersection disagree at fit-error scale, far above weld). Keeping them
/// makes the weave emit the WHOLE face plus disconnected bite fragments.
///
/// Verdict per section, sampled against the outer polygon in unwrapped UV:
/// any sample clearly inside → keep; else any sample clearly outside → drop
/// (fully off-face); else (every sample within the fit band of the boundary)
/// it re-traces a boundary SUB-SPAN → drop, unless it duplicates a WHOLE
/// boundary edge (kept, mirroring the plane-path re-trace discipline).
fn clip_sections_to_outer_region(
    sections: Vec<SectionEdge>,
    boundary_edges: &[OrientedPCurveEdge],
    surface: &FaceSurface,
    wire_pts: &[Point3],
) -> (Vec<SectionEdge>, Vec<Point3>) {
    use std::f64::consts::{PI, TAU};
    // Marched-fit mutual-disagreement tier: two independent fits of the same
    // exact intersection land ~1e-4 apart (vertex 1e-7 < weld 1e-5 < fit).
    const FIT_BAND: f64 = 1e-3;
    // v-disagreement gate for stale parent pcurves (weld scale in UV).
    const WELD_UV: f64 = 1e-4;
    const CURVE_SAMPLES: usize = 12;
    const SEC_SAMPLES: usize = 32;

    // Outer polygon in a continuous (unwrapped-u) UV window, sampled from the
    // 3D curves in native orientation (pcurve conventions are ambiguous).
    let mut poly: Vec<Point2> = Vec::new();
    let mut prev: Option<Point2> = None;
    for e in boundary_edges {
        // Endpoint order follows the traversal flag — for circle/ellipse
        // edges `domain_with_endpoints` takes the CCW span from its first
        // argument, so swapped endpoints would select the COMPLEMENTARY arc.
        // The resulting samples are then oriented to wire order EMPIRICALLY
        // (first sample nearest start_3d): a whole-edge NURBS returns the
        // full forward domain for either traversal orientation, so its trace
        // direction is the curve's own and the flag alone is not reliable.
        let (s3, e3) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
        #[allow(clippy::cast_precision_loss)]
        let mut pts3: Vec<Point3> = (0..=CURVE_SAMPLES)
            .map(|k| {
                let t = (t1 - t0).madd(k as f64 / CURVE_SAMPLES as f64, t0);
                e.curve_3d.evaluate_with_endpoints(t, s3, e3)
            })
            .collect();
        if let (Some(first), Some(last)) = (pts3.first(), pts3.last())
            && (*first - e.start_3d).length() > (*last - e.start_3d).length()
        {
            pts3.reverse();
        }
        // Lifted along the edge, and carried on from the edge before at the
        // vertex they share.
        pts3.retain(|&p| surface.project_point(p).is_some());
        let Some(samples) = super::pcurve_compute::edge_uv_samples(&e.curve_3d, surface, &pts3)
        else {
            continue;
        };
        let shift = match (prev, samples.first()) {
            (Some(pv), Some(first)) => {
                let du = ((first.x() - pv.x() + PI).rem_euclid(TAU)) - PI;
                pv.x() + du - first.x()
            }
            _ => 0.0,
        };
        for s in samples {
            let unwrapped = Point2::new(s.x() + shift, s.y());
            prev = Some(unwrapped);
            poly.push(unwrapped);
        }
    }
    if poly.len() < 3 {
        return (sections, Vec::new());
    }
    // The outer-overhang class requires an EARLIER cut's bite in the outer
    // wire — a marched NURBS boundary edge — and a partial-band face. A
    // primitive full-revolution lateral (plain bore cuts) has neither, and
    // its seam-doubled polygon would produce garbage verdicts.
    if !boundary_edges
        .iter()
        .any(|e| matches!(e.curve_3d, EdgeCurve::NurbsCurve(_)))
    {
        return (sections, Vec::new());
    }
    let (u_lo, u_hi) = poly.iter().fold((f64::MAX, f64::MIN), |(a, b), p| {
        (a.min(p.x()), b.max(p.x()))
    });
    if u_hi - u_lo > TAU - 0.1 {
        return (sections, Vec::new());
    }
    let band = FIT_BAND.max(super::classify_2d::boundary_eps(&poly));
    let to_win = |p: Point3| -> Option<Point2> {
        let (u, v) = surface.project_point(p)?;
        // Test the 2-pi translates and keep the one nearest the polygon.
        let best = [-1.0f64, 0.0, 1.0]
            .iter()
            .map(|k| Point2::new(k.madd(TAU, u), v))
            .min_by(|a, b| {
                super::classify_2d::distance_to_polygon_boundary(*a, &poly)
                    .partial_cmp(&super::classify_2d::distance_to_polygon_boundary(*b, &poly))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
        Some(best)
    };

    // Project a 3D point onto the nearest boundary edge's curve (sampled +
    // ternary-refined). Returns the on-curve foot when within the fit band.
    let snap_to_boundary = |p: Point3| -> Option<Point3> {
        let mut best: Option<(f64, Point3)> = None;
        for e in boundary_edges {
            let (t0, t1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
            let eval = |t: f64| e.curve_3d.evaluate_with_endpoints(t, e.start_3d, e.end_3d);
            let n = 32usize;
            let (mut bi, mut bd) = (0usize, f64::MAX);
            let mut spacing = 0.0_f64;
            let mut last: Option<Point3> = None;
            for k in 0..=n {
                #[allow(clippy::cast_precision_loss)]
                let t = (t1 - t0).madd(k as f64 / n as f64, t0);
                let q = eval(t);
                let d = (q - p).length();
                if d < bd {
                    bd = d;
                    bi = k;
                }
                if let Some(l) = last {
                    spacing = spacing.max((q - l).length());
                }
                last = Some(q);
            }
            // The nearest sample can sit half a sample step from a point on
            // the curve itself.
            if bd > FIT_BAND.madd(4.0, spacing) {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let (mut lo, mut hi) = (
                (t1 - t0).madd(bi.saturating_sub(1) as f64 / n as f64, t0),
                (t1 - t0).madd(((bi + 1).min(n)) as f64 / n as f64, t0),
            );
            for _ in 0..48 {
                let m1 = lo + (hi - lo) / 3.0;
                let m2 = hi - (hi - lo) / 3.0;
                if (eval(m1) - p).length() < (eval(m2) - p).length() {
                    hi = m2;
                } else {
                    lo = m1;
                }
            }
            let t = f64::midpoint(lo, hi);
            let foot = eval(t);
            let d = (foot - p).length();
            if best.is_none_or(|(bd2, _)| d < bd2) {
                best = Some((d, foot));
            }
        }
        best.and_then(|(d, foot)| (d <= FIT_BAND).then_some(foot))
    };

    let inside_at = |p: Point3| -> Option<(bool, f64)> {
        let uv = to_win(p)?;
        Some((
            super::classify_2d::point_in_polygon_2d(uv, &poly),
            super::classify_2d::distance_to_polygon_boundary(uv, &poly),
        ))
    };

    let mut kept: Vec<SectionEdge> = Vec::new();
    let mut anchors: Vec<Point3> = Vec::new();
    for s in sections {
        // Sample the in/out profile.
        let mut states: Vec<Option<(bool, f64)>> = Vec::with_capacity(SEC_SAMPLES + 1);
        for i in 0..=SEC_SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f64 / SEC_SAMPLES as f64;
            states.push(inside_at(evaluate_edge_at_t(
                &s.curve_3d,
                s.start,
                s.end,
                t,
            )));
        }
        if states.iter().any(Option::is_none) {
            kept.push(s);
            continue;
        }
        let clear_in = |st: &Option<(bool, f64)>| st.is_some_and(|(i, d)| i && d > band);
        let clear_out = |st: &Option<(bool, f64)>| st.is_some_and(|(i, d)| !i && d > band);
        let any_in = states.iter().any(clear_in);
        let any_out = states.iter().any(clear_out);
        if any_in && !any_out {
            kept.push(s);
            continue;
        }
        if !any_in {
            // No clearly-interior portion: fully off-face, or a band-hugging
            // boundary re-trace. Keep only a WHOLE-edge duplicate.
            let whole_dup = !any_out
                && boundary_edges.iter().any(|e| {
                    ((s.start - e.start_3d).length() < FIT_BAND
                        && (s.end - e.end_3d).length() < FIT_BAND)
                        || ((s.start - e.end_3d).length() < FIT_BAND
                            && (s.end - e.start_3d).length() < FIT_BAND)
                });
            if whole_dup {
                kept.push(s);
            }
            continue;
        }
        // Mixed: split at each clear in↔out transition, bisection-refined,
        // junction snapped onto the boundary curve so both the boundary
        // splitter (exact on-curve gate) and the piece share one vertex.
        let raw_inside = |st: &Option<(bool, f64)>| st.is_some_and(|(i, _)| i);
        let mut cuts: Vec<(f64, Point3)> = Vec::new();
        for w in 0..SEC_SAMPLES {
            let (a, b) = (&states[w], &states[w + 1]);
            // A crossing whose inside sample lands within the band still
            // counts when the outside one is clearly out; samples hugging the
            // boundary on both sides are no crossing.
            let flip = raw_inside(a) != raw_inside(b)
                && (clear_in(a) || clear_in(b) || clear_out(a) || clear_out(b));
            if !flip {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let (mut lo, mut hi) = (
                w as f64 / SEC_SAMPLES as f64,
                (w + 1) as f64 / SEC_SAMPLES as f64,
            );
            let lo_in = raw_inside(a);
            for _ in 0..48 {
                let m = f64::midpoint(lo, hi);
                let p = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, m);
                let m_in = inside_at(p).is_some_and(|(i, _)| i);
                if m_in == lo_in {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let mut tc = f64::midpoint(lo, hi);
            let pc = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, tc);
            let mut j = snap_to_boundary(pc).unwrap_or(pc);
            // The sampled polygon puts the crossing off the boundary arc by
            // its chord sag; alternate projections between the section and
            // the boundary curve to land on the crossing itself, where the
            // boundary edge was split for the other operand.
            #[allow(clippy::cast_precision_loss)]
            let reach = 1.0 / SEC_SAMPLES as f64;
            for _ in 0..40 {
                let (mut a, mut b) = ((tc - reach).max(0.0), (tc + reach).min(1.0));
                for _ in 0..48 {
                    let m1 = a + (b - a) / 3.0;
                    let m2 = b - (b - a) / 3.0;
                    let d1 = (evaluate_edge_at_t(&s.curve_3d, s.start, s.end, m1) - j).length();
                    let d2 = (evaluate_edge_at_t(&s.curve_3d, s.start, s.end, m2) - j).length();
                    if d1 < d2 {
                        b = m2;
                    } else {
                        a = m1;
                    }
                }
                tc = f64::midpoint(a, b);
                let Some(next) =
                    snap_to_boundary(evaluate_edge_at_t(&s.curve_3d, s.start, s.end, tc))
                else {
                    break;
                };
                let moved = (next - j).length();
                j = next;
                if moved < 1e-12 {
                    break;
                }
            }
            cuts.push((tc, j));
        }
        if cuts.is_empty() {
            kept.push(s);
            continue;
        }
        cuts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut piece_bounds: Vec<(f64, Point3)> = Vec::with_capacity(cuts.len() + 2);
        piece_bounds.push((0.0, s.start));
        piece_bounds.extend(cuts.iter().copied());
        piece_bounds.push((1.0, s.end));
        for w in piece_bounds.windows(2) {
            let (t_a, p_a) = &w[0];
            let (t_b, p_b) = &w[1];
            if (*p_b - *p_a).length() < FIT_BAND {
                continue;
            }
            let tm = f64::midpoint(*t_a, *t_b);
            let pm = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, tm);
            let keep_piece = inside_at(pm).is_some_and(|(i, d)| i && d > band);
            if !keep_piece {
                continue;
            }
            let mut piece = s.clone();
            piece.start = *p_a;
            piece.end = *p_b;
            // The stored pcurves span the WHOLE original section; a piece
            // keeping them would evaluate (and UV-anchor) the full span.
            // Refit on this face over the piece's own endpoints. Both slots
            // are overwritten — this `sections` vec is face-local.
            let pc = super::pcurve_compute::compute_pcurve_on_surface(
                &piece.curve_3d,
                piece.start,
                piece.end,
                surface,
                wire_pts,
                None,
            );
            piece.pcurve_a = pc.clone();
            piece.pcurve_b = pc;
            piece.start_uv_a = None;
            piece.end_uv_a = None;
            piece.start_uv_b = None;
            piece.end_uv_b = None;
            kept.push(piece);
        }
        for (_, j) in &cuts {
            anchors.push(*j);
        }
    }
    // A registry-presplit piece keeps its PARENT's pcurve, whose endpoint UVs
    // are the parent's (the deepened-notch sec pieces evaluated their B/A ends
    // at the parent's B'/A' rim UVs, disconnecting them from the boundary in
    // UV). Detect by v-disagreement — v is non-periodic, so a real mismatch is
    // unambiguous where a u shift could be a legitimate 2π translate — and
    // refit on this face.
    for s in &mut kept {
        let v_bad = |p: Point3, uv_stored: Option<Point2>| -> bool {
            match (surface.project_point(p), uv_stored) {
                (Some((_, v_true)), Some(uv)) => (uv.y() - v_true).abs() > WELD_UV,
                _ => false,
            }
        };
        let (su, eu) = uv_endpoints_from_pcurve(&s.pcurve_a, s.start, s.end, surface, wire_pts);
        if v_bad(s.start, Some(su)) || v_bad(s.end, Some(eu)) {
            let pc = super::pcurve_compute::compute_pcurve_on_surface(
                &s.curve_3d,
                s.start,
                s.end,
                surface,
                wire_pts,
                None,
            );
            s.pcurve_a = pc.clone();
            s.pcurve_b = pc;
            s.start_uv_a = None;
            s.end_uv_a = None;
            s.start_uv_b = None;
            s.end_uv_b = None;
        }
    }

    (kept, anchors)
}

/// Whether any greedy loop contains an out-and-back spur: an edge immediately
/// followed (cyclically) by its exact UV reverse. The angular walker emits
/// this signature only when it has woven twin section edges into one loop
/// instead of closing a region between them; a clean partition never does.
///
/// Loops of two edges are exempt by default: consecutive edges always share a
/// vertex, so for `n == 2` closure alone forces the reverse pattern and every
/// lens region (an arc plus its co-endpoint chord) matches. Only at three or
/// more edges does the pattern actually mean the walker doubled back — at two
/// it fired on the honeycomb cap arrangement's legitimate lens and cost
/// `pcut3` its zero free-edge pin. With `two_edge_spurs` (a promoted hole in
/// the arrangement) a two-edge loop of two STRAIGHT edges counts: that is a
/// bridge section walked to the opening and straight back, never a region,
/// while a lens still keeps its arc and stays exempt.
fn loops_have_out_and_back(
    loops: &[Vec<OrientedPCurveEdge>],
    tol: f64,
    two_edge_spurs: bool,
) -> bool {
    for lp in loops {
        let n = lp.len();
        if n < 2 {
            continue;
        }
        if n == 2 && !(two_edge_spurs && lp.iter().all(|e| matches!(e.curve_3d, EdgeCurve::Line))) {
            continue;
        }
        for i in 0..n {
            let a = &lp[i];
            let b = &lp[(i + 1) % n];
            if (a.start_uv - a.end_uv).length() > tol
                && (a.start_uv - b.end_uv).length() < tol
                && (a.end_uv - b.start_uv).length() < tol
            {
                return true;
            }
        }
    }
    false
}

/// Join a face's holes that meet along a section, run one way by one hole
/// and back by the other, into one hole without it. Both sides of a
/// section lying in an opening are opening: a pin's flat across its bore
/// leaves the bore's rim in two loops, the flat's chord between them, and a
/// face carrying both holes uses the chord twice.
fn merge_holes_across_twin_sections(holes: &mut Vec<Vec<OrientedPCurveEdge>>, tol: f64) {
    let twins = |a: &OrientedPCurveEdge, b: &OrientedPCurveEdge| {
        a.source_edge_idx.is_some()
            && a.source_edge_idx == b.source_edge_idx
            && (a.start_3d - b.end_3d).length() < tol
            && (a.end_3d - b.start_3d).length() < tol
            && (a.start_3d - a.end_3d).length() > tol
    };
    'again: loop {
        for i in 0..holes.len() {
            for j in i + 1..holes.len() {
                let found = holes[i].iter().enumerate().find_map(|(ki, a)| {
                    holes[j].iter().position(|b| twins(a, b)).map(|kj| (ki, kj))
                });
                if let Some((ki, kj)) = found {
                    let b = holes.remove(j);
                    let a = &holes[i];
                    let mut joined = Vec::with_capacity(a.len() + b.len() - 2);
                    joined.extend_from_slice(&a[..ki]);
                    joined.extend_from_slice(&b[kj + 1..]);
                    joined.extend_from_slice(&b[..kj]);
                    joined.extend_from_slice(&a[ki + 1..]);
                    holes[i] = joined;
                    continue 'again;
                }
            }
        }
        break;
    }
}

/// Whether a loop runs a section out and, later, back: the weave of
/// [`loops_have_out_and_back`] with other edges between the two twins. Where
/// a hole's rim reaches the outline through sections at both its ends (a
/// bore whose diameter a slot's floor runs along), the walk can take the
/// pieces on either side of the hole round it as one loop, crossing each
/// section twice.
fn loops_run_a_section_both_ways(loops: &[Vec<OrientedPCurveEdge>], tol: f64) -> bool {
    loops.iter().any(|lp| {
        lp.iter().enumerate().any(|(i, a)| {
            a.source_edge_idx.is_some()
                && (a.start_uv - a.end_uv).length() > tol
                && lp[i + 1..].iter().any(|b| {
                    b.source_edge_idx == a.source_edge_idx
                        && (a.start_uv - b.end_uv).length() < tol
                        && (a.end_uv - b.start_uv).length() < tol
                })
        })
    })
}

/// Puts the section breaks of the try whose pieces a face keeps,
/// `breaks`, in place of the registry `registry` they were copied from.
fn keep_breaks(
    registry: &mut Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
    breaks: Option<std::collections::HashMap<usize, Vec<Point3>>>,
) {
    if let (Some(registry), Some(breaks)) = (registry.as_deref_mut(), breaks) {
        *registry = breaks;
    }
}

/// Split a face by its section edges, producing sub-faces.
///
/// If there are no section edges, returns a single sub-face covering
/// the entire face (pass-through).
///
/// Wraps [`split_face_2d_impl`] to salvage closed circle, ellipse and curved
/// loop *cap* sections on plane faces (the drilled-socket → bin-body fuse: the
/// socket's screw-hole rims land as closed circle sections on the body's
/// coincident bottom plane; a tilted plane across a tube carries two nested
/// ellipses; a plane across a torus's tube carries its loops around the tube).
/// The impl drops closed circles in both its arrangement path
/// ([`arrangement_regions_from_inputs`] discards any input whose UV chord is
/// zero-length, and a closed circle has `start == end`) and its wire-builder
/// fallback, leaving the drilled cylinder rims as free edges; the only impl
/// path that carves a closed circle is the single-closed fast path (exactly
/// one section, closed), which never fires when the circle is mixed with open
/// sections or when there are two or more closed curves. Peel the caps off,
/// split the plane by the remaining sections, then carve each cap into the
/// sub-face that contains it ([`distribute_cap_circles`]).
///
/// # Arguments
/// - `topo` -- the topology arena (immutable read)
/// - `face_id` -- the face to split
/// - `sections` -- intersection curves that cut this face (already trimmed)
/// - `rank` -- which solid this face belongs to (A or B)
/// - `tol` -- tolerance (`.linear` for 3D matching, UV tol derived internally)
/// - `frame` -- cached `PlaneFrame` for this face (avoids origin mismatch)
/// - `info` -- cached `SurfaceInfo` for periodicity flags
#[allow(clippy::too_many_arguments)]
pub fn split_face_2d(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    rank: Rank,
    tol: &brepkit_math::tolerance::Tolerance,
    frame: Option<&PlaneFrame>,
    info: Option<&SurfaceInfo>,
    edge_images: &std::collections::HashMap<
        brepkit_topology::edge::EdgeId,
        Vec<brepkit_topology::edge::EdgeId>,
        impl std::hash::BuildHasher,
    >,
    kept_splits: &std::collections::BTreeSet<brepkit_topology::vertex::VertexId>,
    mut split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
) -> Vec<SplitSubFace> {
    // A ball face with a pole of every chart on it is split in a chart with
    // one pole off it where its own paths find no chart with its poles
    // clear, and so is one its own paths leave whole or split into nothing.
    // That split stands only where its loops were read in that chart: about
    // the pole it holds, or as loops that wind no pole. Elsewhere the face's
    // own paths stand.
    if let Ok(face) = topo.face(face_id)
        && matches!(face.surface(), FaceSurface::Sphere(_))
        && face.inner_wires().is_empty()
        && !sections.is_empty()
    {
        let up_front = matches!(
            pole_clear_chart(topo, face_id, face, sections, true),
            Some(BallChart::PoleOff)
        );
        // Each try splits against its own copy of the registry of section
        // breaks, and only the try whose pieces the face keeps adds its
        // breaks to it: a try refused or set aside leaves no break that the
        // kept pieces do not have for a later face to split at.
        let attempt = |pole_off: Option<&mut bool>| {
            let mut breaks = split_registry.as_deref().cloned();
            let pieces = split_face_2d_impl(
                topo,
                face_id,
                sections,
                rank,
                tol,
                frame,
                info,
                edge_images,
                kept_splits,
                breaks.as_mut(),
                pole_off,
            );
            (pieces, breaks)
        };
        if up_front {
            let mut read_about_a_pole = false;
            let (pieces, breaks) = attempt(Some(&mut read_about_a_pole));
            if read_about_a_pole {
                keep_breaks(&mut split_registry, breaks);
                return pieces;
            }
        }
        let (first, first_breaks) = attempt(None);
        let whole = match first.as_slice() {
            [] => true,
            [only] => only
                .outer_wire
                .iter()
                .chain(only.inner_wires.iter().flatten())
                .all(|e| e.source_edge_idx.is_none()),
            _ => false,
        };
        if !whole || up_front {
            keep_breaks(&mut split_registry, first_breaks);
            return first;
        }
        let mut read_about_a_pole = false;
        let (retry, retry_breaks) = attempt(Some(&mut read_about_a_pole));
        return if read_about_a_pole && retry.len() >= 2 {
            keep_breaks(&mut split_registry, retry_breaks);
            retry
        } else {
            keep_breaks(&mut split_registry, first_breaks);
            first
        };
    }
    let run_impl =
        |secs: &[SectionEdge],
         registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>| {
            split_face_2d_impl(
                topo,
                face_id,
                secs,
                rank,
                tol,
                frame,
                info,
                edge_images,
                kept_splits,
                registry,
                None,
            )
        };

    // Cheap gate for the common case: no closed curved section, nothing to
    // salvage — skip all frame and polygon work below.
    let any_closed_curve = sections.iter().any(|s| {
        (s.start - s.end).length() < tol.linear
            && matches!(
                s.curve_3d,
                EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
            )
    });
    if !any_closed_curve {
        return run_impl(sections, split_registry);
    }

    // Salvage applies only to plane faces (curved faces route closed circles
    // through their own band/internal-loop paths).
    let Ok(face) = topo.face(face_id) else {
        return run_impl(sections, split_registry);
    };
    let surface = face.surface().clone();
    if !matches!(surface, FaceSurface::Plane { .. }) {
        return run_impl(sections, split_registry);
    }

    // Build the face's outer polygon in UV so a *genuine* cap (a full closed
    // circle or ellipse strictly interior to the face, a drilled hole rim) can be
    // told apart from the zero-span arc remnants that the FF/pave machinery
    // leaves at faceted-corner junctions (those sit ON the boundary, share a
    // point with an open arc section, and their "circle" is the corner cylinder
    // tangent to the face outline). Only genuine caps are peeled off; everything
    // else stays in `sections` and reaches the impl unchanged.
    // The frame must be built from the same points the impl uses, so the
    // sub-face UVs and the projected circle centres agree; the containment
    // polygon follows wire traversal order along each edge's own curve, so
    // a face bounded by one circle (one vertex) still has an outline.
    let frame_pts = collect_wire_points(topo, face.outer_wire());
    let owned_frame;
    let cap_frame = if let Some(f) = frame {
        f
    } else {
        let normal = extract_plane_normal(&surface);
        owned_frame = PlaneFrame::from_plane_face(normal, &frame_pts);
        &owned_frame
    };
    let outer_poly: Vec<Point2> = sample_wire_outline(topo, face.outer_wire())
        .iter()
        .map(|&p| cap_frame.project(p))
        .collect();

    let mut cap_sections: Vec<SectionEdge> = Vec::new();
    let mut cap_points: Vec<Point3> = Vec::new();
    let mut rest_sections: Vec<SectionEdge> = Vec::new();
    let mut is_cap: Vec<bool> = Vec::with_capacity(sections.len());
    for s in sections {
        // An oblique plane's ellipse lies within its semi-major axis of its
        // centre, so it clears the outline when that circle would.
        let conic = match &s.curve_3d {
            EdgeCurve::Circle(c) => Some((c.center(), c.radius())),
            EdgeCurve::Ellipse(e) => Some((e.center(), e.semi_major())),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
        };
        let closed = (s.start - s.end).length() < tol.linear && outer_poly.len() >= 3;
        // A ring near the face's rim (one standing on a tube's top) has
        // its centre within the margin of the outline but lies on the face
        // all round, crossing none of its boundary, as a tangent corner
        // circle does not.
        let cap_center = if closed && let Some((center, reach)) = conic {
            let center_uv = cap_frame.project(center);
            (super::classify_2d::point_in_polygon_2d(center_uv, &outer_poly)
                && (super::classify_2d::distance_to_polygon_boundary(center_uv, &outer_poly)
                    > reach * CAP_INTERIORITY_MARGIN
                    || super::fill_images_faces::conic_side_of_plane_face(
                        topo,
                        face_id,
                        &s.curve_3d,
                        tol.linear,
                    ) == Some(true)))
            .then_some(center)
        } else if closed && matches!(s.curve_3d, EdgeCurve::NurbsCurve(_)) {
            interior_loop_center(s, cap_frame, &outer_poly)
        } else {
            None
        };
        is_cap.push(cap_center.is_some());
    }

    // A circle that another section meets is no lone cap: it is cut where
    // they meet and goes to the split with the rest.
    let Some(meets) = circle_cap_meets(sections, &is_cap, tol.linear) else {
        log::debug!(
            "split_face_2d: face {face_id:?} has a whole circle that a section may meet, \
             at no point found; refused"
        );
        return Vec::new();
    };
    let mut cut_any = false;
    for (i, s) in sections.iter().enumerate() {
        if !is_cap[i] {
            rest_sections.push(s.clone());
        } else if let (EdgeCurve::Circle(circle), false) = (&s.curve_3d, meets[i].is_empty()) {
            let (arcs, breaks) = circle_cut_at(
                s, circle, &meets[i], &surface, &frame_pts, cap_frame, tol.linear,
            );
            if let (Some(pb), Some(reg)) = (s.pave_block_id, split_registry.as_deref_mut()) {
                reg.entry(pb).or_default().extend(breaks);
            }
            rest_sections.extend(arcs);
            cut_any = true;
        } else {
            cap_sections.push(s.clone());
            cap_points.push(s.start);
        }
    }

    // Engage only where the impl would silently drop caps: at least one
    // genuine cap AND not the single-closed fast path (exactly one section,
    // that one closed curve) which the impl already handles correctly.
    if cap_sections.is_empty() && cut_any {
        return run_impl(&rest_sections, split_registry);
    }
    if cap_sections.is_empty() || (cap_sections.len() == 1 && sections.len() == 1) {
        return run_impl(sections, split_registry);
    }

    // The impl ignores closed curves on plane faces outside its single-closed
    // fast path, so withholding the caps does not change how the remaining
    // sections split.
    let base = run_impl(&rest_sections, split_registry);

    distribute_cap_circles(topo, face_id, base, &cap_sections, &cap_points, rank, frame)
}

/// How a face's ball is charted for its split.
enum BallChart {
    /// Another surface, or a ball face no chart is read for: one with no
    /// sections, one whose outline is all chords or all arcs of its chart's
    /// equator (a hemisphere, whose own paths read its poles), or one that
    /// holds a pole of every axis tried.
    Unclear,
    /// The face's own chart, whose poles keep clear of its wires and
    /// sections. A face that holds one of them winds the chart.
    Own,
    /// A chart whose poles lie off the face, as far from its wires and
    /// sections as a spread of axes allows; `clear` when that is as far as
    /// the face's own chart must keep them to be read as it lies.
    Turned { chart: FaceSurface, clear: bool },
    /// Where no chart has both poles off the face and clear of it, and the
    /// face's own chart has both poles on it or one near its wires: a chart
    /// with one pole off the face and the other on it, both clear
    /// ([`pole_off_ball_axis`]). The face's outline winds it once.
    PoleOff,
}

/// The face's ball in a chart whose poles lie off the face and clear of its
/// wires and sections. At a pole every `u` names one point: a wire or a
/// section that reaches one jumps in `u` there, and the traces cannot follow
/// it. A pole inside the face would make its outline wind the chart.
fn pole_clear_sphere(
    topo: &Topology,
    face_id: FaceId,
    face: &brepkit_topology::face::Face,
    sections: &[SectionEdge],
) -> BallChart {
    pole_clear_chart(topo, face_id, face, sections, false).unwrap_or(BallChart::Unclear)
}

/// A ball face's wires and sections seen from its ball's centre: unit
/// directions sampled along each edge, and whether its outline is all
/// chords or all arcs of its chart's equator. `None` for another surface,
/// a face with no sections, or a lookup that fails.
fn ball_face_directions(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
    sections: &[SectionEdge],
) -> Option<(Vec<Vec3>, bool)> {
    // Samples per edge: a half turn of a great circle is read every 2.8
    // degrees, inside the 5.7 degrees that counts as near a pole.
    const SAMPLES: u32 = 64;
    let FaceSurface::Sphere(sphere) = face.surface() else {
        return None;
    };
    if sections.is_empty() {
        return None;
    }
    let center = sphere.center();
    let mut dirs: Vec<Vec3> = Vec::new();
    let sample = |dirs: &mut Vec<Vec3>, curve: &EdgeCurve, start: Point3, end: Point3| {
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        for k in 0..=SAMPLES {
            let t = (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0);
            let d = curve.evaluate_with_endpoints(t, start, end) - center;
            let len = d.length();
            if len > 0.0 {
                dirs.push(d * (1.0 / len));
            }
        }
    };
    let mut chords_only = true;
    for (i, wid) in std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .enumerate()
    {
        let wire = topo.wire(wid).ok()?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).ok()?;
            let start = topo.vertex(edge.start()).ok()?.point();
            let end = topo.vertex(edge.end()).ok()?.point();
            let chord = matches!(edge.curve(), EdgeCurve::Line) && (start - end).length() > 0.0;
            if i == 0 && !chord && !special_cases::on_chart_equator(sphere, edge.curve(), 1e-7) {
                chords_only = false;
            }
            sample(&mut dirs, edge.curve(), start, end);
        }
    }
    for s in sections {
        sample(&mut dirs, &s.curve_3d, s.start, s.end);
    }
    Some((dirs, chords_only))
}

/// A chart of a ball face's ball with one pole off the face and the other
/// on it, both clear of the face's wires and sections
/// ([`pole_off_ball_axis`]), whatever its outline.
fn pole_off_chart(
    topo: &Topology,
    face_id: FaceId,
    face: &brepkit_topology::face::Face,
    sections: &[SectionEdge],
) -> Option<FaceSurface> {
    let FaceSurface::Sphere(sphere) = face.surface() else {
        return None;
    };
    let (dirs, _) = ball_face_directions(topo, face, sections)?;
    let trim = crate::classifier::SphereFaceTrim::new(topo, face_id).ok()??;
    let (center, radius) = (sphere.center(), sphere.radius());
    let axis = pole_off_ball_axis(center, radius, &dirs, &trim)?;
    brepkit_math::surfaces::SphericalSurface::with_axis(center, radius, axis)
        .ok()
        .map(FaceSurface::Sphere)
}

/// [`pole_clear_sphere`]'s chart; with `pole_off`, [`BallChart::PoleOff`]
/// where it serves.
fn pole_clear_chart(
    topo: &Topology,
    face_id: FaceId,
    face: &brepkit_topology::face::Face,
    sections: &[SectionEdge],
    pole_off: bool,
) -> Option<BallChart> {
    let FaceSurface::Sphere(sphere) = face.surface() else {
        return None;
    };
    let (dirs, chords_only) = ball_face_directions(topo, face, sections)?;
    if chords_only {
        return None;
    }
    let center = sphere.center();
    // The cosine of the angle from the nearer pole about `axis` to the
    // nearest sample.
    let nearest = |axis: Vec3| dirs.iter().fold(0.0_f64, |m, d| m.max(d.dot(axis).abs()));
    let radius = sphere.radius();
    let trim = crate::classifier::SphereFaceTrim::new(topo, face_id)
        .ok()
        .flatten();
    let one_pole_off = |t: &crate::classifier::SphereFaceTrim| {
        pole_off && pole_off_ball_axis(center, radius, &dirs, t).is_some()
    };
    // A face that holds a pole of its own chart winds the chart, however
    // far its wires keep from the pole: it takes a chart whose poles lie off
    // it and clear of its wires where one is found.
    if nearest(sphere.z_axis()) < 0.1_f64.cos() {
        let z = sphere.z_axis();
        let mut both_held = false;
        let turned = trim.as_ref().and_then(|t| {
            let held = [z, -z].map(|a| t.holds(center + a * radius, 1e-6 * radius));
            both_held = held[0] && held[1];
            held.contains(&true)
                .then(|| clear_ball_axis(center, radius, &dirs, t))
                .flatten()
                .filter(|&axis| nearest(axis) < 0.1_f64.cos())
        });
        return Some(
            match turned.and_then(|axis| {
                brepkit_math::surfaces::SphericalSurface::with_axis(center, radius, axis).ok()
            }) {
                Some(chart) => BallChart::Turned {
                    chart: FaceSurface::Sphere(chart),
                    clear: true,
                },
                None if both_held && trim.as_ref().is_some_and(one_pole_off) => BallChart::PoleOff,
                None => BallChart::Own,
            },
        );
    }
    let trim = trim?;
    let Some(best) = clear_ball_axis(center, radius, &dirs, &trim) else {
        return one_pole_off(&trim).then_some(BallChart::PoleOff);
    };
    brepkit_math::surfaces::SphericalSurface::with_axis(center, radius, best)
        .ok()
        .map(|chart| BallChart::Turned {
            chart: FaceSurface::Sphere(chart),
            clear: nearest(best) < 0.1_f64.cos(),
        })
}

/// Of a spread of axes through `center`, the one whose poles keep farthest
/// from the unit directions `dirs` (a face's wires and sections seen from
/// its ball's centre), each pole off the face that `trim` reads. `None`
/// when every axis has a pole on the face.
fn clear_ball_axis(
    center: Point3,
    radius: f64,
    dirs: &[Vec3],
    trim: &crate::classifier::SphereFaceTrim,
) -> Option<Vec3> {
    const CANDIDATES: u32 = 256;
    // The cosine of the angle from the nearer pole about `axis` to the
    // nearest direction.
    let nearest = |axis: Vec3| dirs.iter().fold(0.0_f64, |m, d| m.max(d.dot(axis).abs()));
    let off_face = |axis: Vec3| {
        [axis, -axis]
            .iter()
            .all(|&a| !trim.holds(center + a * radius, 1e-6 * radius))
    };
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..CANDIDATES)
        .map(|i| {
            let z = 1.0 - (2.0 * f64::from(i) + 1.0) / f64::from(CANDIDATES);
            let r = (1.0 - z * z).max(0.0).sqrt();
            let (s, c) = (golden * f64::from(i)).sin_cos();
            Vec3::new(r * c, r * s, z)
        })
        .filter(|&axis| off_face(axis))
        .map(|axis| (nearest(axis), axis))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, axis)| axis)
}

/// Of the axes through `center`, one with a pole off the face that `trim`
/// reads and the other on it, whose poles keep clear of the unit directions
/// `dirs` (the face's wires and sections seen from its ball's centre): the
/// pole on the face by 0.1 rad, as a chart of the face's own must, and the
/// pole off it by 0.035 rad. The pole off the face lies in a hole or past
/// the outline, and a narrow hole holds it nearer its wires than that. Both
/// clear the half step between the samples of a half turn of a great circle
/// (0.025 rad), so no wire runs through either pole between its samples.
///
/// A spread of axes is searched, then the best of them is moved in steps
/// while its poles keep their sides and the nearer pole, measured against
/// its own clearance, gains. `None` when no axis keeps both clearances.
fn pole_off_ball_axis(
    center: Point3,
    radius: f64,
    dirs: &[Vec3],
    trim: &crate::classifier::SphereFaceTrim,
) -> Option<Vec3> {
    const CANDIDATES: u32 = 256;
    const HELD_CLEAR: f64 = 0.1;
    const OFF_CLEAR: f64 = 0.035;
    let near = 1e-6 * radius;
    // The angle from `pole` to the nearest direction.
    let gap = |pole: Vec3| {
        dirs.iter()
            .fold(-1.0_f64, |m, d| m.max(d.dot(pole)))
            .clamp(-1.0, 1.0)
            .acos()
    };
    // The axis's poles, the one the face holds first, when it holds one.
    let sides = |axis: Vec3| match (
        trim.holds(center + axis * radius, near),
        trim.holds(center - axis * radius, near),
    ) {
        (true, false) => Some((axis, -axis)),
        (false, true) => Some((-axis, axis)),
        _ => None,
    };
    // The nearer pole's gap as a share of its clearance: 1 or more clears.
    let score = |axis: Vec3| {
        sides(axis).map(|(held, off)| (gap(held) / HELD_CLEAR).min(gap(off) / OFF_CLEAR))
    };
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let (mut best, mut axis) = (0..CANDIDATES)
        .map(|i| {
            let z = 1.0 - (2.0 * f64::from(i) + 1.0) / f64::from(CANDIDATES);
            let r = (1.0 - z * z).max(0.0).sqrt();
            let (s, c) = (golden * f64::from(i)).sin_cos();
            Vec3::new(r * c, r * s, z)
        })
        .filter_map(|axis| score(axis).map(|s| (s, axis)))
        .max_by(|a, b| a.0.total_cmp(&b.0))?;
    // The spread's axes lie about 0.16 rad apart: step about the best one,
    // halving the step where no step gains, down to a ten-thousandth.
    let mut step = 0.08_f64;
    for _ in 0..200 {
        if step < 1e-4 {
            break;
        }
        let side = if axis.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let Ok(t1) = axis.cross(side).normalize() else {
            break;
        };
        let t2 = axis.cross(t1);
        let gain = [t1, -t1, t2, -t2]
            .into_iter()
            .filter_map(|d| (axis + d * step).normalize().ok())
            .filter_map(|a| score(a).map(|s| (s, a)))
            .filter(|&(s, _)| s > best)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        match gain {
            Some((s, a)) => (best, axis) = (s, a),
            None => step *= 0.5,
        }
    }
    (best >= 1.0).then_some(axis)
}

/// Whether each end of each section meets exactly one other end (a closed
/// section's two ends meet each other). Sections cross nowhere but at their
/// ends, so loops chained from such sections share no point: any two lie
/// apart or one inside the other. Where three ends or more meet, two loops
/// can cross there.
fn sections_meet_in_pairs(sections: &[SectionEdge], weld: f64) -> bool {
    let ends: Vec<Point3> = sections.iter().flat_map(|s| [s.start, s.end]).collect();
    ends.iter()
        .all(|&p| ends.iter().filter(|&&q| (q - p).length() < weld).count() == 2)
}

/// The centroid of a closed curved section's samples when every sample
/// clears the face outline by a twentieth of the loop's extent (a loop that
/// touches the outline is no cap).
fn interior_loop_center(
    section: &SectionEdge,
    frame: &PlaneFrame,
    outer_poly: &[Point2],
) -> Option<Point3> {
    let (t0, t1) = section
        .curve_3d
        .domain_with_endpoints(section.start, section.end);
    let n = 32_i32;
    let pts: Vec<Point3> = (0..n)
        .map(|k| {
            section.curve_3d.evaluate_with_endpoints(
                (t1 - t0).madd(f64::from(k) / f64::from(n), t0),
                section.start,
                section.end,
            )
        })
        .collect();
    let uv: Vec<Point2> = pts.iter().map(|&p| frame.project(p)).collect();
    let (mut lo, mut hi) = (uv[0], uv[0]);
    for p in &uv {
        lo = Point2::new(lo.x().min(p.x()), lo.y().min(p.y()));
        hi = Point2::new(hi.x().max(p.x()), hi.y().max(p.y()));
    }
    let margin = 0.05 * (hi - lo).length();
    if margin <= 0.0
        || !uv.iter().all(|&p| {
            super::classify_2d::point_in_polygon_2d(p, outer_poly)
                && super::classify_2d::distance_to_polygon_boundary(p, outer_poly) > margin
        })
    {
        return None;
    }
    let sum = pts.iter().fold(Vec3::new(0.0, 0.0, 0.0), |acc, p| {
        acc + Vec3::new(p.x(), p.y(), p.z())
    });
    let mean = sum * (1.0 / f64::from(n));
    Some(Point3::new(mean.x(), mean.y(), mean.z()))
}

/// Samples per curved edge of an outline: a chord of a circle in 64 sags
/// 0.12% of its radius.
const OUTLINE_CURVE_SAMPLES: usize = 64;

/// Whether a ball face holds one pole of a chart, with its one wire running
/// once round the chart's axis. The face's trim must hold one pole
/// and not the other, and the wire's turn about the axis, read along its
/// edges' own curves, must be one whole turn either way. A zone between two
/// latitudes, bounded by a seam and two rims, holds neither pole and runs
/// its rims round the axis both ways, for no net turn. A wire that comes
/// within 0.02 rad of the axis, or whose samples step more than a quarter
/// turn round it, is not read.
///
/// `Some(true)` when the face holds the north pole of `chart`, a chart of
/// its ball, and `Some(false)` when it holds the south pole.
fn pole_held_round(
    topo: &Topology,
    face_id: FaceId,
    chart: &brepkit_math::surfaces::SphericalSurface,
) -> Option<bool> {
    use std::f64::consts::{FRAC_PI_2, TAU};
    let face = topo.face(face_id).ok()?;
    if !face.inner_wires().is_empty() {
        return None;
    }
    let (center, radius) = (chart.center(), chart.radius());
    let (x, y, z) = (chart.x_axis(), chart.y_axis(), chart.z_axis());
    let trim = crate::classifier::SphereFaceTrim::new(topo, face_id).ok()??;
    let held = [z, -z].map(|a| trim.holds(center + a * radius, 1e-6 * radius));
    if held[0] == held[1] {
        return None;
    }
    let pts = sample_wire_outline(topo, face.outer_wire());
    if pts.len() < 3 {
        return None;
    }
    let mut angles = Vec::with_capacity(pts.len());
    for p in &pts {
        let d = *p - center;
        let (a, b) = (d.dot(x), d.dot(y));
        if a.hypot(b) < 0.02 * radius {
            return None;
        }
        angles.push(b.atan2(a));
    }
    let mut turn = 0.0;
    for (k, &a) in angles.iter().enumerate() {
        let next = angles[(k + 1) % angles.len()];
        let step = (next - a + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
        if step.abs() > FRAC_PI_2 {
            return None;
        }
        turn += step;
    }
    ((turn.abs() / TAU - 1.0).abs() < 1e-6).then_some(held[0])
}

/// A wire's outline in traversal order: each line's start, and each curved
/// edge sampled along its own span.
fn sample_wire_outline(topo: &Topology, wire_id: brepkit_topology::wire::WireId) -> Vec<Point3> {
    let Ok(wire) = topo.wire(wire_id) else {
        return Vec::new();
    };
    let mut pts = Vec::new();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            continue;
        };
        let n = if matches!(edge.curve(), EdgeCurve::Line) {
            1
        } else {
            OUTLINE_CURVE_SAMPLES
        };
        super::pcurve_compute::sample_edge_uniform_native(
            edge.curve(),
            sv.point(),
            ev.point(),
            n,
            oe.is_forward(),
            &mut pts,
        );
    }
    pts
}

/// [`sample_wire_outline`] in `surface`'s parameters: each edge's samples
/// lifted along it ([`super::pcurve_compute::edge_uv_samples`]), and moved
/// by whole periods to meet the edge before at the vertex they share.
fn sample_wire_outline_uv(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    surface: &FaceSurface,
) -> Vec<Point2> {
    let Ok(wire) = topo.wire(wire_id) else {
        return Vec::new();
    };
    let (u_period, v_period) = super::pcurve_compute::surface_periods(surface);
    let turns =
        |x: f64, to: f64, period: Option<f64>| period.map_or(0.0, |p| ((to - x) / p).round() * p);
    let mut pts: Vec<Point2> = Vec::new();
    let mut end: Option<Point2> = None;
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            continue;
        };
        let n = if matches!(edge.curve(), EdgeCurve::Line) {
            1
        } else {
            OUTLINE_CURVE_SAMPLES
        };
        let mut chain = Vec::with_capacity(n + 1);
        super::pcurve_compute::sample_edge_uniform_native(
            edge.curve(),
            sv.point(),
            ev.point(),
            n,
            oe.is_forward(),
            &mut chain,
        );
        chain.push(if oe.is_forward() {
            ev.point()
        } else {
            sv.point()
        });
        chain.retain(|&p| surface.project_point(p).is_some());
        let Some(uv) = super::pcurve_compute::edge_uv_samples(edge.curve(), surface, &chain) else {
            continue;
        };
        let (Some(&first), Some(&last)) = (uv.first(), uv.last()) else {
            continue;
        };
        let shift = end.map_or(brepkit_math::vec::Vec2::new(0.0, 0.0), |at| {
            brepkit_math::vec::Vec2::new(
                turns(first.x(), at.x(), u_period),
                turns(first.y(), at.y(), v_period),
            )
        });
        pts.extend(uv[..uv.len() - 1].iter().map(|&p| p + shift));
        end = Some(last + shift);
    }
    pts
}

/// Slack on the cap interiority test. A genuine drilled hole's centre clears
/// the face outline by more than its radius, while a corner cylinder's section
/// circle is exactly tangent (centre exactly one radius out); the 5% margin
/// keeps float noise on such a tangent circle from reading as interior. An
/// ellipse is held to its semi-major axis, so one that touches the outline
/// anywhere is refused.
const CAP_INTERIORITY_MARGIN: f64 = 1.05;

/// A loop with the two halves of one of `rims` back to back (the rim split
/// only at its seam's antipode, touched by no section) with that rim whole
/// in their place, starting where the first half starts. Two pieces of a
/// rim meeting at one of `trace_splits`, a point the rim was split at only
/// for the trace, are joined into one arc the same way.
fn rejoin_untouched_rims(
    wire: Vec<OrientedPCurveEdge>,
    rims: &[OrientedPCurveEdge],
    trace_splits: &[Point3],
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    let near = |a: Point3, b: Point3| (a - b).length() < 100.0 * tol;
    let same_circle = |a: &EdgeCurve, b: &EdgeCurve| match (a, b) {
        (EdgeCurve::Circle(p), EdgeCurve::Circle(q)) => {
            near(p.center(), q.center()) && (p.radius() - q.radius()).abs() < 100.0 * tol
        }
        _ => false,
    };
    let n = wire.len();
    if rims.is_empty() || n < 3 {
        return wire;
    }
    // A rim cut into several pieces that run round it from its start back to
    // it, touched by no section, is the rim whole. Joined a pair at a time,
    // the pieces would pass through half a turn, which the rule below keeps
    // split.
    for i in 0..n {
        let first = &wire[i];
        if first.source_edge_idx.is_some() {
            continue;
        }
        let Some(rim) = rims.iter().find(|r| {
            same_circle(&r.curve_3d, &first.curve_3d) && near(r.start_3d, first.start_3d)
        }) else {
            continue;
        };
        let mut last = i;
        let mut count = 1;
        while count < n - 1 {
            let next = &wire[(last + 1) % n];
            if next.source_edge_idx.is_some()
                || !same_circle(&rim.curve_3d, &next.curve_3d)
                || !near(wire[last].end_3d, next.start_3d)
            {
                break;
            }
            last = (last + 1) % n;
            count += 1;
            if near(next.end_3d, first.start_3d) {
                break;
            }
        }
        if count < 3 || !near(wire[last].end_3d, first.start_3d) {
            continue;
        }
        // A joint another edge of the loop meets (a section ending on the
        // rim) is a real vertex of the rim.
        let run: Vec<usize> = (0..count).map(|k| (i + k) % n).collect();
        let met = run[..count - 1].iter().any(|&k| {
            let joint = wire[k].end_3d;
            (0..n)
                .filter(|j| !run.contains(j))
                .any(|j| near(wire[j].start_3d, joint) || near(wire[j].end_3d, joint))
        });
        if met {
            continue;
        }
        let Ok(line) = Line2D::new(first.start_uv, wire[last].end_uv - first.start_uv) else {
            continue;
        };
        let joined = OrientedPCurveEdge {
            curve_3d: rim.curve_3d.clone(),
            pcurve: Curve2D::Line(line),
            start_uv: first.start_uv,
            end_uv: wire[last].end_uv,
            start_3d: first.start_3d,
            end_3d: first.start_3d,
            forward: first.forward,
            source_edge_idx: None,
            pave_block_id: rim.pave_block_id,
        };
        let mut out: Vec<OrientedPCurveEdge> = Vec::with_capacity(n - count + 1);
        for k in 0..n {
            if k == i {
                out.push(joined.clone());
            } else if !run.contains(&k) {
                out.push(wire[k].clone());
            }
        }
        return rejoin_untouched_rims(out, rims, trace_splits, tol);
    }
    for i in 0..n {
        let (first, second) = (&wire[i], &wire[(i + 1) % n]);
        if first.source_edge_idx.is_some() || second.source_edge_idx.is_some() {
            continue;
        }
        let Some(rim) = rims.iter().find(|r| {
            same_circle(&r.curve_3d, &first.curve_3d) && same_circle(&r.curve_3d, &second.curve_3d)
        }) else {
            continue;
        };
        if !near(first.end_3d, second.start_3d) {
            continue;
        }
        let whole = near(rim.start_3d, first.start_3d) && near(second.end_3d, first.start_3d);
        if !whole && !trace_splits.iter().any(|&p| near(p, first.end_3d)) {
            continue;
        }
        // Joined into half a turn or more, the arc would run between the
        // same two points as the rest of the rim, and the edge merge would
        // weld the two: keep the split.
        if !whole && (second.end_uv.x() - first.start_uv.x()).abs() >= std::f64::consts::PI - 1e-6 {
            continue;
        }
        let Ok(line) = Line2D::new(first.start_uv, second.end_uv - first.start_uv) else {
            continue;
        };
        let joined = OrientedPCurveEdge {
            curve_3d: rim.curve_3d.clone(),
            pcurve: Curve2D::Line(line),
            start_uv: first.start_uv,
            end_uv: second.end_uv,
            start_3d: first.start_3d,
            end_3d: if whole { first.start_3d } else { second.end_3d },
            forward: first.forward,
            source_edge_idx: None,
            pave_block_id: if whole { rim.pave_block_id } else { None },
        };
        let mut out: Vec<OrientedPCurveEdge> = Vec::with_capacity(n - 1);
        for k in 0..n {
            if k == i {
                out.push(joined.clone());
            } else if k != (i + 1) % n {
                out.push(wire[k].clone());
            }
        }
        return rejoin_untouched_rims(out, rims, trace_splits, tol);
    }
    wire
}

/// Lay a periodic face's outer boundary out as one walk in `u` from its
/// first closed rim: every edge moves by whole periods to start where the
/// edge before it ends, so the seam's two copies land a period apart with
/// the rims running between them. A closed rim's UVs run its curve's own
/// sense whatever its traversal, so one traversed against its curve leaves
/// the walk a turn back from its start. False, with nothing moved, when
/// there is no closed rim to walk from or a closed rim is neither a circle
/// nor an ellipse (the split of a closed spline rim takes its pieces' u from
/// projection, not from the rim's own span).
fn walk_boundary_from_closed_rim(edges: &mut [OrientedPCurveEdge]) -> bool {
    use std::f64::consts::TAU;
    let closed = |e: &OrientedPCurveEdge| (e.start_3d - e.end_3d).length() < 1e-10;
    if edges
        .iter()
        .any(|e| closed(e) && !matches!(e.curve_3d, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)))
    {
        return false;
    }
    let Some(first) = edges.iter().position(closed) else {
        return false;
    };
    let n = edges.len();
    let mut cur = edges[first].start_uv.x();
    for k in 0..n {
        let e = &mut edges[(first + k) % n];
        let shift = ((cur - e.start_uv.x()) / TAU).round() * TAU;
        if shift != 0.0 {
            e.start_uv = Point2::new(e.start_uv.x() + shift, e.start_uv.y());
            e.end_uv = Point2::new(e.end_uv.x() + shift, e.end_uv.y());
        }
        cur = if closed(e) && !e.forward {
            2.0f64.madd(e.start_uv.x(), -e.end_uv.x())
        } else {
            e.end_uv.x()
        };
    }
    true
}

/// Move an edge's UVs and its pcurve together by `(du, dv)`: a pcurve left on
/// the old copy of the period reads as stale, and the walkers then take the
/// edge's chord for its direction at each end.
fn move_edge_in_uv(e: &mut OrientedPCurveEdge, du: f64, dv: f64) {
    use brepkit_math::vec::Vec2;
    if du == 0.0 && dv == 0.0 {
        return;
    }
    let by = Vec2::new(du, dv);
    e.start_uv = e.start_uv + by;
    e.end_uv = e.end_uv + by;
    if let Some(pcurve) = edge_splitting::shifted_pcurve(&e.pcurve, du, dv) {
        e.pcurve = pcurve;
    }
}

/// The pieces of a curved face's `hole` split where section `ends` lie on
/// it, on the copy of the period those ends' UVs take, or `None` when no end
/// lies on it.
fn weave_curved_hole(
    hole: &[OrientedPCurveEdge],
    ends: &[(Point3, Point2)],
    surface: &FaceSurface,
    tol: f64,
) -> Option<Vec<OrientedPCurveEdge>> {
    let weld = tol * 100.0;
    let (u_period, v_period) = super::pcurve_compute::surface_periods(surface);
    let nearest = |x: f64, target: f64, period: Option<f64>| match period {
        Some(p) => x - ((x - target) / p).round() * p,
        None => x,
    };
    let on_hole: Vec<(Point3, Point2)> = ends
        .iter()
        .copied()
        .filter(|&(p, _)| hole.iter().any(|e| point_on_oriented_edge(e, p, weld)))
        .collect();
    if on_hole.is_empty() {
        return None;
    }
    let pts: Vec<Point3> = on_hole.iter().map(|&(p, _)| p).collect();
    let mut pieces =
        edge_splitting::split_hole_edges_at_3d_points(hole.to_vec(), &pts, None, surface, tol);
    // The hole's own UVs come from its own walk, which can sit a period away
    // from the face's window: move it to the copy whose point under a
    // section's end lies at that end's UV. Only an end with a piece's end
    // under it can say which copy: the split can refuse an end that the
    // contact test takes.
    let piece_ends: Vec<(Point3, Point2)> = pieces
        .iter()
        .flat_map(|e| [(e.start_3d, e.start_uv), (e.end_3d, e.end_uv)])
        .collect();
    let (anchor_uv, at) = on_hole.iter().find_map(|&(p, uv)| {
        piece_ends
            .iter()
            .find(|&&(q, _)| (q - p).length() <= weld)
            .map(|&(_, at)| (uv, at))
    })?;
    let du = nearest(at.x(), anchor_uv.x(), u_period) - at.x();
    let dv = nearest(at.y(), anchor_uv.y(), v_period) - at.y();
    for e in &mut pieces {
        move_edge_in_uv(e, du, dv);
        for &(p, uv) in &on_hole {
            if (e.start_3d - p).length() <= weld {
                e.start_uv = uv;
            }
            if (e.end_3d - p).length() <= weld {
                e.end_uv = uv;
            }
        }
        e.source_edge_idx = None;
    }
    Some(pieces)
}

/// Whether `p` lies on `e` within `band`, read along its curve between its
/// ends.
fn point_on_oriented_edge(e: &OrientedPCurveEdge, p: Point3, band: f64) -> bool {
    const SAMPLES: u32 = 64;
    if (e.start_3d - p).length() <= band || (e.end_3d - p).length() <= band {
        return true;
    }
    let (s3, e3) = if e.forward {
        (e.start_3d, e.end_3d)
    } else {
        (e.end_3d, e.start_3d)
    };
    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
    // A spline is read by projection onto the part the edge runs on: fixed
    // samples along its parameter can step over a short knot span (an
    // imported polyline's spike) and miss a point on it.
    if let EdgeCurve::NurbsCurve(whole) = &e.curve_3d {
        use brepkit_math::nurbs::knot_ops::curve_split;
        use brepkit_math::nurbs::projection::project_point_to_curve;
        use brepkit_math::traits::ParametricCurve;
        let (lo, hi) = (t0.min(t1), t0.max(t1));
        let (d0, d1) = ParametricCurve::domain(whole);
        let slack = 1e-9 * (d1 - d0);
        let mut span = whole.clone();
        if lo > d0 + slack {
            let Ok((_, rest)) = curve_split(&span, lo) else {
                return false;
            };
            span = rest;
        }
        if hi < d1 - slack {
            let Ok((kept, _)) = curve_split(&span, hi) else {
                return false;
            };
            span = kept;
        }
        return project_point_to_curve(&span, p, band * 1e-3).is_ok_and(|hit| hit.distance <= band);
    }
    // A circle or an ellipse is read at its own projection, which its
    // splitter takes too: a chord of a long arc lies up to its sag inside
    // the arc, and a point in that lens is not on it.
    let conic = match &e.curve_3d {
        EdgeCurve::Circle(c) => {
            let angle = c.project(p);
            Some((angle, c.evaluate(angle)))
        }
        EdgeCurve::Ellipse(el) => {
            let angle = el.project(p);
            Some((angle, el.evaluate(angle)))
        }
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
    };
    if let Some((angle, foot)) = conic {
        use std::f64::consts::TAU;
        let along = if t1 >= t0 {
            (angle - t0).rem_euclid(TAU)
        } else {
            (t0 - angle).rem_euclid(TAU)
        };
        return along <= (t1 - t0).abs() && (foot - p).length() <= band;
    }
    let at = |k: u32| {
        e.curve_3d.evaluate_with_endpoints(
            (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0),
            s3,
            e3,
        )
    };
    (0..SAMPLES).any(|k| {
        let (a, b) = (at(k), at(k + 1));
        let d = b - a;
        let l2 = d.dot(d);
        let f = if l2 > 0.0 {
            ((p - a).dot(d) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let sag = (at(k) + (at(k + 1) - at(k)) * 0.5 - {
            let tm = (t1 - t0).madd((f64::from(k) + 0.5) / f64::from(SAMPLES), t0);
            e.curve_3d.evaluate_with_endpoints(tm, s3, e3)
        })
        .length();
        (a + d * f - p).length() <= band + sag
    })
}

/// Move a section edge's ends by whole periods into the one-turn window
/// `[lo, lo + period]` its face's boundary spans. An open section's interior
/// lies inside the window, so its curve's samples, unwrapped from the
/// middle, say which copy each end takes, an end on the seam included; its
/// own end UVs can come from separate projections and need not be one span.
/// A closed section winds once from the seam and moves as a whole.
fn place_section_in_turn(e: &mut OrientedPCurveEdge, surface: &FaceSurface, lo: f64, period: f64) {
    const SAMPLES: u32 = 8;
    let nearest = |u: f64, target: f64| u - ((u - target) / period).round() * period;
    let shift_to = |e: &mut OrientedPCurveEdge, start_u: f64, end_u: f64| {
        e.start_uv = Point2::new(start_u, e.start_uv.y());
        e.end_uv = Point2::new(end_u, e.end_uv.y());
    };
    if (e.start_3d - e.end_3d).length() < 1e-10 {
        let mid = f64::midpoint(e.start_uv.x(), e.end_uv.x());
        let shift = -((mid - lo) / period).floor() * period;
        shift_to(e, e.start_uv.x() + shift, e.end_uv.x() + shift);
        move_pcurve_to_ends(e, period);
        return;
    }
    let (s3, e3) = if e.forward {
        (e.start_3d, e.end_3d)
    } else {
        (e.end_3d, e.start_3d)
    };
    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
    let pts: Vec<Point3> = (0..=SAMPLES)
        .map(|k| {
            let t = (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0);
            e.curve_3d.evaluate_with_endpoints(t, s3, e3)
        })
        .collect();
    // Lifted along the section, then moved by whole periods so its middle
    // lies in the window.
    let Some(uv) = super::pcurve_compute::edge_uv_samples(&e.curve_3d, surface, &pts) else {
        return;
    };
    let mid = uv.len() / 2;
    let shift = lo + (uv[mid].x() - lo).rem_euclid(period) - uv[mid].x();
    let path: Vec<f64> = uv.iter().map(|p| p.x() + shift).collect();
    // A marched curve's domain can run from either end.
    let first = e.curve_3d.evaluate_with_endpoints(t0, s3, e3);
    let (at_start, at_end) = if (first - e.start_3d).length() <= (first - e.end_3d).length() {
        (path[0], path[path.len() - 1])
    } else {
        (path[path.len() - 1], path[0])
    };
    shift_to(
        e,
        nearest(e.start_uv.x(), at_start),
        nearest(e.end_uv.x(), at_end),
    );
    move_pcurve_to_ends(e, period);
}

/// Move a section's NURBS pcurve by the whole turns of `u` that its ends
/// moved. A pcurve left a turn from its ends reads as stale, and the trace
/// then reads the section's way out of an end from its chord. A section
/// with both ends on the seam has its chord along the seam line, and the
/// trace cannot tell it from the seam there.
fn move_pcurve_to_ends(e: &mut OrientedPCurveEdge, period: f64) {
    const NEAR: f64 = 1e-6;
    let brepkit_math::curves2d::Curve2D::Nurbs(n) = &e.pcurve else {
        return;
    };
    let knots = n.knots();
    let (Some(&k0), Some(&k1)) = (knots.first(), knots.last()) else {
        return;
    };
    let (a, b) = (n.evaluate(k0), n.evaluate(k1));
    let turns = |from: Point2, to: Point2| -> Option<f64> {
        let k = ((to.x() - from.x()) / period).round();
        ((period.madd(k, from.x()) - to.x()).abs() < NEAR && (from.y() - to.y()).abs() < NEAR)
            .then_some(k)
    };
    let k = match (
        turns(a, e.start_uv).zip(turns(b, e.end_uv)),
        turns(b, e.start_uv).zip(turns(a, e.end_uv)),
    ) {
        (Some((k, l)), _) | (_, Some((k, l))) if (k - l).abs() < 0.5 => k,
        _ => return,
    };
    if k == 0.0 {
        return;
    }
    if let Some(moved) = edge_splitting::shifted_pcurve(&e.pcurve, k * period, 0.0) {
        e.pcurve = moved;
    }
}

/// Give each end of a sliver section lying on the seam the seam copy of the
/// section it continues. Where a section loop crosses the seam at its fold,
/// the marched pieces can end a hair apart on either side, joined by a
/// sliver whose samples all sit on the seam: [`place_section_in_turn`] lays
/// it wholly on one copy, and the piece on the other copy then dangles. The
/// sliver has to step from one copy to the other.
fn place_seam_slivers(sections: &mut [OrientedPCurveEdge], lo: f64, period: f64, tol: f64) {
    let near = 100.0 * tol;
    let on_seam = |u: f64| {
        let d = (u - lo).rem_euclid(period);
        d.min(period - d) < 1e-6
    };
    let seam_sliver = |e: &OrientedPCurveEdge| {
        is_sliver(e, near) && on_seam(e.start_uv.x()) && on_seam(e.end_uv.x())
    };
    let ends: Vec<(Point3, f64)> = sections
        .iter()
        .filter(|e| !seam_sliver(e))
        .flat_map(|e| [(e.start_3d, e.start_uv.x()), (e.end_3d, e.end_uv.x())])
        .filter(|&(_, u)| on_seam(u))
        .collect();
    let copy_at = |p: Point3| {
        ends.iter()
            .map(|&(q, u)| ((q - p).length(), u))
            .filter(|&(d, _)| d < near)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, u)| u)
    };
    for e in sections.iter_mut().filter(|e| seam_sliver(e)) {
        if let Some(u) = copy_at(e.start_3d) {
            e.start_uv = Point2::new(u, e.start_uv.y());
        }
        if let Some(u) = copy_at(e.end_3d) {
            e.end_uv = Point2::new(u, e.end_uv.y());
        }
    }
}

/// Whether an edge's ends and its middle all lie within `size` of one
/// another: a sliver, not a closed curve whose ends merely meet. An edge
/// whose ends meet on an open curve is a point of it: its span reads as the
/// whole curve, whose own ends lie apart.
fn is_sliver(e: &OrientedPCurveEdge, size: f64) -> bool {
    let (s, t) = (e.start_3d, e.end_3d);
    if (s - t).length() >= size {
        return false;
    }
    let (t0, t1) = e.curve_3d.domain_with_endpoints(s, t);
    let at = |u: f64| e.curve_3d.evaluate_with_endpoints(u, s, t);
    (at(t0) - at(t1)).length() >= size || (at(f64::midpoint(t0, t1)) - s).length() < size
}

/// For each whole circle cap among `sections` (`is_cap`), the points where
/// another section meets it; empty for a cap no section meets, and for every
/// other section. A pair of caps that cross get the same points. `None` where
/// no bound proves a section apart from a cap and no point is found where it
/// meets it ([`circle_meets`]).
fn circle_cap_meets(
    sections: &[SectionEdge],
    is_cap: &[bool],
    tol: f64,
) -> Option<Vec<Vec<Point3>>> {
    let mut meets: Vec<Vec<Point3>> = vec![Vec::new(); sections.len()];
    for (i, cap) in sections.iter().enumerate() {
        let EdgeCurve::Circle(circle) = &cap.curve_3d else {
            continue;
        };
        if !is_cap[i] {
            continue;
        }
        for (j, other) in sections.iter().enumerate() {
            let other_cap = is_cap[j] && matches!(other.curve_3d, EdgeCurve::Circle(_));
            if j == i || (other_cap && j < i) {
                continue;
            }
            let points = circle_meets(circle, other, tol)?;
            if other_cap {
                meets[j].extend(points.iter().copied());
            }
            meets[i].extend(points);
        }
    }
    Some(meets)
}

/// The points where `other`, a section on the plane of `circle`, meets the
/// whole circle: an end of `other` on it, or a crossing, within the weld
/// (`100 tol`). Empty where a bound proves `other` keeps farther than the
/// weld from the circle; `None` where no bound does and no point is found.
///
/// A line's points come from the closed form of
/// [`Circle3D::intersect_segment`], and a circle's from
/// [`Circle3D::intersect_circle`], kept where they lie on `other`'s arc (read
/// along its curve or the short way round). An ellipse or a spline is only
/// proved apart: it lies within its semi-major axis of its centre, and a
/// spline's pieces within the hull of their control points.
fn circle_meets(
    circle: &brepkit_math::curves::Circle3D,
    other: &SectionEdge,
    tol: f64,
) -> Option<Vec<Point3>> {
    let weld = 100.0 * tol;
    let (center, r) = (circle.center(), circle.radius());
    let closed = (other.start - other.end).length() < tol;
    let mut at: Vec<Point3> = Vec::new();
    // A computed point that lies on an end of `other` is that end, so the
    // pieces of both meet at one point.
    let mut add = |p: Point3| {
        let p = [other.start, other.end]
            .into_iter()
            .find(|e| (*e - p).length() <= weld)
            .unwrap_or(p);
        if !at.iter().any(|q| (*q - p).length() <= weld) {
            at.push(p);
        }
    };
    if !closed {
        for p in [other.start, other.end] {
            if ((p - center).length() - r).abs() <= weld {
                add(p);
            }
        }
    }
    // The least and greatest distance from the centre to `other`.
    let (near, far) = match &other.curve_3d {
        EdgeCurve::Line => {
            let d = other.end - other.start;
            let len2 = d.dot(d);
            let t = if len2 > 0.0 {
                ((center - other.start).dot(d) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (
                (other.start + d * t - center).length(),
                (other.start - center)
                    .length()
                    .max((other.end - center).length()),
            )
        }
        EdgeCurve::Circle(q) => {
            let apart = (q.center() - center).length();
            ((apart - q.radius()).abs(), apart + q.radius())
        }
        EdgeCurve::Ellipse(e) => {
            let apart = (e.center() - center).length();
            ((apart - e.semi_major()).max(0.0), apart + e.semi_major())
        }
        EdgeCurve::NurbsCurve(n) => {
            let (t0, t1) = other.curve_3d.domain_with_endpoints(other.start, other.end);
            return nurbs_apart_from_circle(n, (t0, t1), center, r, weld).then(Vec::new);
        }
    };
    if near > r + weld || far < r - weld {
        return Some(Vec::new());
    }
    match &other.curve_3d {
        EdgeCurve::Line => {
            for (p, _) in circle.intersect_segment(other.start, other.end, tol) {
                add(p);
            }
            (!at.is_empty()).then_some(at)
        }
        EdgeCurve::Circle(q) => {
            if (q.center() - center).length() <= weld && (q.radius() - r).abs() <= weld {
                // `other` runs on the circle: its ends cut it, and a whole
                // copy of it is left as it was.
                return Some(at);
            }
            let crossings = circle.intersect_circle(q, tol);
            if crossings.is_empty() {
                return None;
            }
            for (p, _) in crossings {
                if closed || on_section_arc(q, other, p, weld) {
                    add(p);
                }
            }
            Some(at)
        }
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => None,
    }
}

/// Whether `p`, a point of `circle`, lies on the open arc `section` of it,
/// read along its curve from its start or the short way round between its
/// ends, `slack` along it.
fn on_section_arc(
    circle: &brepkit_math::curves::Circle3D,
    section: &SectionEdge,
    p: Point3,
    slack: f64,
) -> bool {
    use std::f64::consts::TAU;
    let slack = slack / circle.radius();
    let at = circle.project(p);
    let (t0, t1) = section
        .curve_3d
        .domain_with_endpoints(section.start, section.end);
    let along =
        (at - t0).rem_euclid(TAU) <= (t1 - t0) + slack || (at - t0).rem_euclid(TAU) >= TAU - slack;
    let start = circle.project(section.start);
    let delta = super::pcurve_compute::shorter_arc_delta(circle.project(section.end) - start);
    let (from, run) = if delta >= 0.0 {
        (start, delta)
    } else {
        (start + delta, -delta)
    };
    let short =
        (at - from).rem_euclid(TAU) <= run + slack || (at - from).rem_euclid(TAU) >= TAU - slack;
    along || short
}

/// Whether the span `(t0, t1)` of spline `n` keeps farther than `weld` from
/// the circle about `center` of radius `r`: each piece's control points all
/// lie within `r - weld` of the centre, or beyond `r + weld` along the line
/// from the centre to their mean. The pieces start as the knot spans the
/// range covers, each a Bezier piece of its own; a piece neither test proves
/// is halved, ten times at most.
fn nurbs_apart_from_circle(
    n: &brepkit_math::nurbs::curve::NurbsCurve,
    (t0, t1): (f64, f64),
    center: Point3,
    r: f64,
    weld: f64,
) -> bool {
    let (lo, hi) = (t0.min(t1), t0.max(t1));
    let mut pending: Vec<(f64, f64, u32)> = n
        .knots()
        .windows(2)
        .map(|w| (w[0].max(lo), w[1].min(hi)))
        .filter(|(a, b)| a < b)
        .map(|(a, b)| (a, b, 0))
        .collect();
    if pending.is_empty() {
        return false;
    }
    while let Some((a, b, depth)) = pending.pop() {
        let Ok(homogeneous) = brepkit_math::nurbs::decompose::curve_bezier_between(n, a, b) else {
            return false;
        };
        let mut points = Vec::with_capacity(homogeneous.len());
        for [x, y, z, w] in homogeneous {
            if w.is_nan() || w <= 0.0 {
                return false;
            }
            points.push(Point3::new(x / w, y / w, z / w));
        }
        if points.is_empty() {
            return false;
        }
        let inside = points.iter().all(|&p| (p - center).length() < r - weld);
        #[allow(clippy::cast_precision_loss)]
        let mean = points
            .iter()
            .fold(Vec3::new(0.0, 0.0, 0.0), |acc, &p| acc + (p - center))
            * (1.0 / points.len() as f64);
        let outside = mean
            .normalize()
            .is_ok_and(|toward| points.iter().all(|&p| (p - center).dot(toward) > r + weld));
        if inside || outside {
            continue;
        }
        if depth >= 10 {
            return false;
        }
        let mid = f64::midpoint(a, b);
        pending.push((a, mid, depth + 1));
        pending.push((mid, b, depth + 1));
    }
    true
}

/// The whole circle section `cap` cut at `points` into arcs of at most a
/// third of a turn, each counter-clockwise about the circle's axis, and the
/// points it was cut at. A run between two points longer than a third of a
/// turn is cut in equal parts: sections are read the short way round
/// between their ends.
fn circle_cut_at(
    cap: &SectionEdge,
    circle: &brepkit_math::curves::Circle3D,
    points: &[Point3],
    surface: &FaceSurface,
    wire_pts: &[Point3],
    frame: &PlaneFrame,
    tol: f64,
) -> (Vec<SectionEdge>, Vec<Point3>) {
    use std::f64::consts::TAU;
    let mut cuts: Vec<(f64, Point3)> = points.iter().map(|&p| (circle.project(p), p)).collect();
    cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
    cuts.dedup_by(|a, b| (a.1 - b.1).length() <= 100.0 * tol);
    let mut breaks: Vec<Point3> = Vec::new();
    for (k, &(t, p)) in cuts.iter().enumerate() {
        breaks.push(p);
        let next = cuts[(k + 1) % cuts.len()].0;
        let run = if cuts.len() == 1 {
            TAU
        } else {
            (next - t).rem_euclid(TAU)
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let parts = (run / (TAU / 3.0)).ceil().max(1.0) as u32;
        for j in 1..parts {
            breaks.push(circle.evaluate(run.madd(f64::from(j) / f64::from(parts), t)));
        }
    }
    let arcs = (0..breaks.len())
        .map(|k| {
            let (start, end) = (breaks[k], breaks[(k + 1) % breaks.len()]);
            let pcurve = compute_pcurve_on_surface(
                &cap.curve_3d,
                start,
                end,
                surface,
                wire_pts,
                Some(frame),
            );
            SectionEdge {
                curve_3d: cap.curve_3d.clone(),
                pcurve_a: pcurve.clone(),
                pcurve_b: pcurve,
                start,
                end,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: cap.target_face,
                pave_block_id: None,
            }
        })
        .collect();
    (arcs, breaks)
}

/// Carve closed cap circles and ellipses (see [`split_face_2d`]) into the
/// base sub-faces.
///
/// Each cap circle is assigned to the first base sub-face whose outer boundary
/// contains a point on the circle in UV, then that sub-face is re-split via
/// [`split_face_with_internal_loops`] (disc cap + holed remainder, the
/// sub-face's holes inside the circle moving to the disc). A circle on a
/// sub-face hole's inside is air — the rim of a drill emerging inside an
/// existing opening — and a circle contained by no sub-face has no home; both
/// are dropped, exactly as the impl's air filter and arrangement paths drop
/// them. The circle's centre decides neither: a counterbore's rim circles a
/// hole, so its centre lies in the hole while the circle does not.
fn distribute_cap_circles(
    topo: &Topology,
    face_id: FaceId,
    base: Vec<SplitSubFace>,
    cap_sections: &[SectionEdge],
    cap_points: &[Point3],
    rank: Rank,
    frame: Option<&PlaneFrame>,
) -> Vec<SplitSubFace> {
    let Ok(face) = topo.face(face_id) else {
        return base;
    };
    let surface = face.surface().clone();
    if !matches!(surface, FaceSurface::Plane { .. }) {
        return base;
    }
    // Same frame convention as the impl (see [`split_face_2d`]).
    let wire_pts = collect_wire_points(topo, face.outer_wire());
    let owned_frame;
    let frame = if let Some(f) = frame {
        f
    } else {
        let normal = extract_plane_normal(&surface);
        owned_frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        &owned_frame
    };

    let points_uv: Vec<Point2> = cap_points.iter().map(|&c| frame.project(c)).collect();
    // A sub-face's loops are read in this frame from their 3D points: a
    // piece from the internal-loops path carries its UVs in a frame of its
    // own (a coplanar footprint's square read degenerate here, the circles
    // inside it went to the ring around it, and the ring kept them as holes).
    let in_frame = |lp: &[OrientedPCurveEdge]| -> Vec<OrientedPCurveEdge> {
        lp.iter()
            .map(|e| {
                let mut e = e.clone();
                e.start_uv = frame.project(e.start_3d);
                e.end_uv = frame.project(e.end_3d);
                e
            })
            .collect()
    };

    let mut assigned = vec![false; cap_sections.len()];
    let mut result: Vec<SplitSubFace> = Vec::with_capacity(base.len() + cap_sections.len());
    for sf in base {
        // The frame-based sampler is orientation-unambiguous; the stored-pcurve
        // sampler can fold a loop carrying reversed arc sections into a
        // self-crossing polygon (see [`sample_wire_loop_uv_via_frame`]).
        let outer = in_frame(&sf.outer_wire);
        let holes: Vec<Vec<OrientedPCurveEdge>> =
            sf.inner_wires.iter().map(|h| in_frame(h)).collect();
        let poly = sample_wire_loop_uv_via_frame(&outer, frame);
        // Read on the sub-face's own lines and arcs where it has no NURBS
        // edge: eight chords of a circular outline leave a cap's point off it.
        let exact = |wire: &[OrientedPCurveEdge]| containment::exact_hole(wire, frame);
        let outer_exact = exact(&outer);
        let holes_exact: Vec<_> = holes.iter().map(|h| exact(h)).collect();
        let in_outer = |p: Point2| match &outer_exact {
            Some(pieces) => point_in_region(pieces, p, 0.0) == Some(true),
            None => super::classify_2d::point_in_polygon_2d(p, &poly),
        };
        let in_hole = |p: Point2| {
            holes
                .iter()
                .zip(&holes_exact)
                .any(|(hole, pieces)| match pieces {
                    Some(pieces) => point_in_region(pieces, p, 0.0) == Some(true),
                    None => is_inside_any_hole(&p, std::slice::from_ref(hole)),
                })
        };
        let contained: Vec<SectionEdge> = if poly.len() < 3 {
            Vec::new()
        } else {
            cap_sections
                .iter()
                .enumerate()
                .filter_map(|(i, cs)| {
                    if assigned[i] || !in_outer(points_uv[i]) || in_hole(points_uv[i]) {
                        return None;
                    }
                    assigned[i] = true;
                    Some(cs.clone())
                })
                .collect()
        };
        if contained.is_empty() {
            result.push(sf);
        } else {
            let sub_wire_pts: Vec<Point3> = sf
                .outer_wire
                .iter()
                .flat_map(|e| [e.start_3d, e.end_3d])
                .collect();
            // Caps are closed curves, which the internal-loops path never
            // declines.
            if let Some(carved) = split_face_with_internal_loops(
                &sf.surface,
                &sf.outer_wire,
                &sf.inner_wires,
                &contained,
                rank,
                sf.reversed,
                face_id,
                &sub_wire_pts,
                // A plane face: the wall search this clears is never run.
                brepkit_math::tolerance::Tolerance::new().linear,
            ) {
                result.extend(carved);
            } else {
                result.push(sf);
            }
        }
    }
    result
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn split_face_2d_impl(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    rank: Rank,
    tol: &brepkit_math::tolerance::Tolerance,
    frame: Option<&PlaneFrame>,
    info: Option<&SurfaceInfo>,
    edge_images: &std::collections::HashMap<
        brepkit_topology::edge::EdgeId,
        Vec<brepkit_topology::edge::EdgeId>,
        impl std::hash::BuildHasher,
    >,
    kept_splits: &std::collections::BTreeSet<brepkit_topology::vertex::VertexId>,
    mut split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
    pole_off: Option<&mut bool>,
) -> Vec<SplitSubFace> {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    // Two FF pairs emit the SAME section when the partner faces are coplanar
    // neighbours and the curve rides their shared boundary (a slot-corner
    // cylinder against a wall face and the coplanar lip band above it).
    // Duplicate sections become parallel twin edges in the trace input, and
    // the greedy walker's angular successor is ill-defined on parallel
    // twins — it grand-tours adjacent regions into one slitted loop instead
    // of partitioning them. Drop duplicates: sections with the same ends
    // that a bound over their whole length holds together, as two curves
    // through the same ends and middle (a circle and an ellipse) can part
    // between them.
    let deduped_sections: Option<Vec<SectionEdge>> = {
        let band = tol.linear * 10.0;
        let same_pt = |p: Point3, q: Point3| (p - q).length() <= band;
        let mut keep = vec![true; sections.len()];
        let mut any_dup = false;
        for i in 0..sections.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..sections.len() {
                if !keep[j] {
                    continue;
                }
                let (a, b) = (&sections[i], &sections[j]);
                let ends_match = (same_pt(a.start, b.start) && same_pt(a.end, b.end))
                    || (same_pt(a.start, b.end) && same_pt(a.end, b.start));
                if ends_match && same_section_curve(a, b, band) {
                    keep[j] = false;
                    any_dup = true;
                }
            }
        }
        any_dup.then(|| {
            sections
                .iter()
                .zip(&keep)
                .filter(|&(_, &k)| k)
                .map(|(s, _)| s.clone())
                .collect()
        })
    };
    let sections: &[SectionEdge] = deduped_sections.as_deref().unwrap_or(sections);
    let trace_split =
        std::env::var("BK_SPLIT_TRACE").is_ok_and(|v| v == format!("{}", face_id.index()));
    // With `pole_off`, the face takes a chart with one pole off it, and the
    // flag reads whether the split in that chart stands.
    let ball_chart = if pole_off.is_some() {
        match pole_off_chart(topo, face_id, face, sections) {
            Some(chart) => BallChart::Turned { chart, clear: true },
            None => return Vec::new(),
        }
    } else {
        pole_clear_sphere(topo, face_id, face, sections)
    };
    let (rechart, ball_chart_clear) = match ball_chart {
        BallChart::Unclear | BallChart::PoleOff => (None, false),
        BallChart::Own => (None, true),
        BallChart::Turned { chart, clear } => (Some(chart), clear),
    };
    // The sections' pcurves and end hints read the face's own chart.
    let recharted_sections: Vec<SectionEdge>;
    let sections: &[SectionEdge] = match &rechart {
        Some(chart) => {
            recharted_sections = sections
                .iter()
                .map(|s| {
                    let pcurve = super::pcurve_compute::compute_pcurve_on_surface(
                        &s.curve_3d,
                        s.start,
                        s.end,
                        chart,
                        &[],
                        None,
                    );
                    SectionEdge {
                        pcurve_a: pcurve.clone(),
                        pcurve_b: pcurve,
                        start_uv_a: None,
                        end_uv_a: None,
                        start_uv_b: None,
                        end_uv_b: None,
                        ..s.clone()
                    }
                })
                .collect();
            &recharted_sections
        }
        None => sections,
    };
    let surface = rechart.unwrap_or_else(|| face.surface().clone());
    let reversed = face.is_reversed();
    let is_plane = matches!(surface, FaceSurface::Plane { .. });
    if trace_split {
        log::debug!(
            "STRACE face={face_id:?} plane={is_plane} sections={} info={}",
            sections.len(),
            info.map_or_else(|| "none".to_string(), |i| format!("{:?}", i.periodicity()))
        );
        let mut rows: Vec<String> = sections
            .iter()
            .map(|sec| {
                format!(
                    "STRACE-SEC s=({:.7},{:.7},{:.7}) e=({:.7},{:.7},{:.7}) pb={:?}",
                    sec.start.x(),
                    sec.start.y(),
                    sec.start.z(),
                    sec.end.x(),
                    sec.end.y(),
                    sec.end.z(),
                    sec.pave_block_id
                )
            })
            .collect();
        rows.sort();
        for r in rows {
            log::debug!("{r}");
        }
    }

    // Use provided frame or build one from wire points (plane faces only).
    let wire_pts = collect_wire_points(topo, face.outer_wire());
    let owned_frame;
    let frame = if let Some(f) = frame {
        f
    } else if is_plane {
        let normal = extract_plane_normal(&surface);
        owned_frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        &owned_frame
    } else {
        // For non-plane faces, PlaneFrame is not used -- set a dummy.
        // All UV projection goes through surface.project_point().
        owned_frame = PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &[]);
        &owned_frame
    };

    // Extract periodicity from SurfaceInfo.
    // Periodic quantization is needed for boundary wire connectivity (circle
    // end at u=2pi connects to seam start at u=0). Keep it enabled.
    let (u_periodic, v_periodic) = info.map_or((false, false), SurfaceInfo::periodicity);

    // Outer-wire image expansion is gated to HOLE-FREE PLANAR faces: a
    // periodic lateral's seam is a Line edge too, and expanding it hands
    // the calibrated seam machinery multiple seam pieces where it expects
    // one (the perpendicular-cylinder fuse loses both mutually-trimmed
    // walls); a holed cap's integrate-holes weave is likewise calibrated
    // against the unexpanded boundary (the divider-lip fuse de-analytics
    // with expansion applied).
    let section_anchor_pts: Vec<Point3> = sections
        .iter()
        .flat_map(|sct| [sct.start, sct.end])
        .collect();
    // Line expansion stays planar-only (the periodic seam machinery expects
    // one seam piece); NURBS expansion also serves non-planar hole-free
    // faces, whose boundary arcs never split in their own machinery either
    // (the coaxial wedge's cylinder patches refused their axial sections
    // for the same reason its z-planes did). A holed plane keeps its outer
    // wire whole for the integrate-holes weave; a holed cylinder (a band
    // pierced by earlier slot boxes) has no such weave and expands its NURBS
    // boundary like a hole-free one.
    let mut boundary_edges =
        if face.inner_wires().is_empty() || matches!(surface, FaceSurface::Cylinder(_)) {
            super::face_splitter::conversion::boundary_edges_to_pcurve_with_images(
                topo,
                face.outer_wire(),
                &surface,
                &wire_pts,
                if is_plane { Some(frame) } else { None },
                edge_images,
                kept_splits,
                &section_anchor_pts,
                is_plane,
            )
        } else {
            boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, None)
        };

    // Convert original inner wires (holes) to OrientedPCurveEdge.
    let original_inner_wires: Vec<Vec<OrientedPCurveEdge>> = face
        .inner_wires()
        .iter()
        .filter_map(|&iw_id| {
            let iw_pts = collect_wire_points(topo, iw_id);
            let edges = if is_plane {
                boundary_edges_to_pcurve(topo, iw_id, &surface, &iw_pts, Some(frame))
            } else {
                boundary_edges_to_pcurve(topo, iw_id, &surface, &iw_pts, None)
            };
            // A hole bounded by closed curved edges (e.g. a single full
            // circle) has fewer than 3 distinct wire points but is a valid
            // inner wire; only polyline-style wires need 3+ points.
            let has_closed_curve = edges.iter().any(|e| {
                !matches!(e.curve_3d, EdgeCurve::Line)
                    && (e.start_3d - e.end_3d).length() < tol.linear * 100.0
            });
            // Two vertices bound an area too when an edge between them is
            // curved (a D-shaped hole: its flat and its arc).
            let curved_pair = iw_pts.len() == 2
                && edges.len() >= 2
                && edges.iter().any(|e| !edge_curve_is_straight(&e.curve_3d));
            if edges.is_empty() || (iw_pts.len() < 3 && !has_closed_curve && !curved_pair) {
                None
            } else {
                Some(edges)
            }
        })
        .collect();

    // Expand plane-face hole wires through the pave-level edge splits so the
    // PROMOTION pass below can match hole vertices against the exact pave
    // vertices the boundary machinery minted. Kept SEPARATE from
    // `original_inner_wires`: the hole weave (`integrate_holes_plane`) is
    // calibrated on unsplit hole edges — its whole-edge-duplicate re-trace
    // discriminant misfires on pave-split pieces (the honeycomb pcut3 cap).
    let mut expanded_inner_wires: Vec<Option<Vec<OrientedPCurveEdge>>> = if is_plane {
        let iw_ids = face.inner_wires().to_vec();
        original_inner_wires
            .iter()
            .enumerate()
            .map(|(hi, _)| {
                let &iw_id = iw_ids.get(hi)?;
                let wire = topo.wire(iw_id).ok()?;
                let needs = wire.edges().iter().any(|oe| {
                    edge_images
                        .get(&oe.edge())
                        .is_some_and(|imgs| imgs.len() > 1)
                });
                if !needs {
                    return None;
                }
                let mut out: Vec<OrientedPCurveEdge> = Vec::new();
                for oe in wire.edges() {
                    let pieces: Vec<brepkit_topology::edge::EdgeId> =
                        match edge_images.get(&oe.edge()) {
                            Some(imgs) if imgs.len() > 1 => {
                                let mut v = imgs.clone();
                                if !oe.is_forward() {
                                    v.reverse();
                                }
                                v
                            }
                            _ => vec![oe.edge()],
                        };
                    for pid in pieces {
                        let Ok(e) = topo.edge(pid) else { continue };
                        let (Ok(vs), Ok(ve)) = (topo.vertex(e.start()), topo.vertex(e.end()))
                        else {
                            continue;
                        };
                        let (s3, e3) = if oe.is_forward() {
                            (vs.point(), ve.point())
                        } else {
                            (ve.point(), vs.point())
                        };
                        if (s3 - e3).length() < tol.linear {
                            continue;
                        }
                        // A piece of a split rim can run more than a half
                        // turn: it traces its own span, not the shorter arc
                        // between its ends.
                        let pcurve = super::pcurve_compute::compute_boundary_pcurve_on_surface(
                            e.curve(),
                            vs.point(),
                            ve.point(),
                            oe.is_forward(),
                            &surface,
                            &wire_pts,
                            Some(frame),
                        );
                        out.push(OrientedPCurveEdge {
                            curve_3d: e.curve().clone(),
                            pcurve,
                            start_uv: frame.project(s3),
                            end_uv: frame.project(e3),
                            start_3d: s3,
                            end_3d: e3,
                            forward: oe.is_forward(),
                            source_edge_idx: None,
                            pave_block_id: None,
                        });
                    }
                }
                (out.len() >= 3).then_some(out)
            })
            .collect()
    } else {
        vec![None; original_inner_wires.len()]
    };

    // Normalize hole winding: an inner wire must wind OPPOSITE the outer wire
    // in the projected UV frame for every consumer that trusts stored
    // orientation — `integrate_holes_plane` weaves hole pieces as-is, and a
    // same-winding hole makes the angular wire builder trace the material
    // region wound CW plus a spurious loop spanning the opening wound CCW
    // (the halfSockets lip fuse shipped that loop as a membrane across the
    // bin throat while the real ledge region vanished with the face it was
    // hole-matched onto). Upstream operations can emit same-winding holes
    // (the halfSockets body's cavity cut), so fix the winding here where the
    // wires enter the splitter.
    let original_inner_wires: Vec<Vec<OrientedPCurveEdge>> = if is_plane {
        // Arc-true via the frame: the pcurve sampler walks a reversed
        // boundary arc backwards (boundary pcurves are fitted in traversal
        // order), and a rim split into reversed pieces then reads with the
        // wrong sign (a barrel's end disc, see `sample_wire_loop_uv_via_frame`).
        let outer_sign = signed_area_2d(&sampling::sample_wire_loop_uv_via_frame(
            &boundary_edges,
            frame,
        )) >= 0.0;
        let flip_wire = |hole: &mut Vec<OrientedPCurveEdge>| {
            hole.reverse();
            for edge in hole.iter_mut() {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        };
        original_inner_wires
            .into_iter()
            .enumerate()
            .map(|(hi, mut hole)| {
                // Read the same way as the outer wire: the pcurve sampler
                // walks a hole's reversed arcs backwards too (a keyhole
                // whose rod arcs all run against their edges reads the
                // wrong sign and was flipped to wind with the rim).
                let pts = sampling::sample_wire_loop_uv_via_frame(&hole, frame);
                if pts.len() < 3 {
                    return hole;
                }
                let area = signed_area_2d(&pts);
                // A sliver hole encloses less area than a tol-wide band along
                // its own perimeter; its winding sign is numeric noise, so
                // leave it untouched rather than flip on noise.
                let mut perimeter: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
                if let (Some(first), Some(last)) = (pts.first(), pts.last()) {
                    perimeter += (*last - *first).length();
                }
                if area.abs() > perimeter * tol.linear && (area >= 0.0) == outer_sign {
                    flip_wire(&mut hole);
                    // The expanded twin must stay orientation-consistent with
                    // the wire it stands in for.
                    if let Some(Some(exp)) = expanded_inner_wires.get_mut(hi) {
                        flip_wire(exp);
                    }
                }
                hole
            })
            .collect()
    } else {
        original_inner_wires
    };

    // A section edge lying entirely inside an existing hole runs through
    // air, not face material (a tool passing through a cavity opening still
    // intersects the face's surface plane inside the hole). Keeping it would
    // stamp a spurious nested loop onto the face, leaving free edges.
    let filtered_sections: Vec<SectionEdge>;
    let sections = if original_inner_wires.is_empty() {
        sections
    } else {
        let to_uv = |p: Point3| -> Option<Point2> {
            if is_plane {
                Some(frame.project(p))
            } else {
                surface.project_point(p).map(|(u, v)| Point2::new(u, v))
            }
        };
        // Sample along the actual curve, not the start/mid/end chord: a
        // strongly curved section edge can bow outside the hole while its
        // endpoints and chord midpoint all sit inside it. Walking the curve
        // via `evaluate_edge_at_t` also covers closed-circle sections
        // (start == end), where chord sampling collapses to a single point.
        //
        // Uniform samples alone are NOT sufficient: a Line section bridging
        // between two cavities can cross a sliver of material (the 1.2 mm
        // divider cap between the openings, on a 200+ mm span) that every
        // uniform probe misses, so the section reads as pure air and the
        // divider cap is never split out (tilted-divider lip fuse). Probe the
        // midpoints between consecutive hole-boundary crossings as well —
        // the same sub-segment structure the hole weave itself uses — so any
        // material sub-segment, however thin, keeps the section alive.
        let crossing_midpoint_probes = |s: &SectionEdge| -> Vec<Point2> {
            if !matches!(s.curve_3d, EdgeCurve::Line) {
                return Vec::new();
            }
            let (Some(s0), Some(s1)) = (to_uv(s.start), to_uv(s.end)) else {
                return Vec::new();
            };
            let mut ts: Vec<f64> = vec![0.0, 1.0];
            for hole in &original_inner_wires {
                // Only straight hole edges: for those the endpoint chord IS the
                // edge, so the crossing parameters are exact. A curved rim's
                // chord would place crossings (and thus the probe midpoints)
                // off the true geometry; those edges contribute no crossings
                // here, which errs on the side of keeping the uniform-probe
                // verdict rather than fabricating a rescue.
                for e in hole.iter().filter(|e| edge_curve_is_straight(&e.curve_3d)) {
                    let h0 = frame.project(e.start_3d);
                    let h1 = frame.project(e.end_3d);
                    if let Some(t) = seg_cross_param(s0, s1, h0, h1) {
                        ts.push(t);
                    }
                }
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ts.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            ts.windows(2)
                .map(|w| {
                    let tm = f64::midpoint(w[0], w[1]);
                    Point2::new(
                        s0.x() + (s1.x() - s0.x()) * tm,
                        s0.y() + (s1.y() - s0.y()) * tm,
                    )
                })
                .collect()
        };
        filtered_sections = sections
            .iter()
            .filter(|s| {
                let uniform_in_hole = (0..=HOLE_PROBE_SAMPLES).all(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = i as f64 / HOLE_PROBE_SAMPLES as f64;
                    let p = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, t);
                    to_uv(p).is_some_and(|uv| is_inside_any_hole(&uv, &original_inner_wires))
                });
                if !uniform_in_hole {
                    return true;
                }
                if is_plane {
                    let probes = crossing_midpoint_probes(s);
                    if !probes
                        .iter()
                        .all(|uv| is_inside_any_hole(uv, &original_inner_wires))
                    {
                        return true;
                    }
                }
                false
            })
            .cloned()
            .collect();
        &filtered_sections
    };

    // Deduplicate sections sharing endpoints: a face-face interference can be
    // recorded more than once (e.g. the same wall reached via two adjacent
    // tool faces). A duplicated dividing section makes the wire builder weave
    // a zero-area slit instead of splitting the face, which reads as a spurious
    // genus-1 handle in the assembled solid.
    let deduped_sections: Vec<SectionEdge>;
    let sections = {
        // Quantize at the kernel's linear tolerance so dedup only collapses
        // genuinely-coincident sections (a doubly-recorded interference) and
        // never distinct splitters that happen to be close on a small model.
        let cell = tol.linear.max(1e-12);
        let scale = 1.0 / cell;
        let q = |p: Point3| -> (i64, i64, i64) {
            (
                (p.x() * scale).round() as i64,
                (p.y() * scale).round() as i64,
                (p.z() * scale).round() as i64,
            )
        };
        // Two sections with the same quantized ends are one only where a
        // bound over their whole length holds them together: an arc and the
        // rest of its circle share their ends, and so do the two halves of a
        // circle. Ends in one cell lie within `sqrt(3)` cells of each other.
        let mut seen: std::collections::HashMap<_, Vec<usize>> = std::collections::HashMap::new();
        let mut kept: Vec<SectionEdge> = Vec::with_capacity(sections.len());
        for s in sections {
            let (a, b) = (q(s.start), q(s.end));
            let twins = seen
                .entry(if a <= b { (a, b) } else { (b, a) })
                .or_default();
            if twins
                .iter()
                .any(|&i| same_section_curve(&kept[i], s, 2.0 * cell))
            {
                continue;
            }
            twins.push(kept.len());
            kept.push(s.clone());
        }
        deduped_sections = kept;
        &deduped_sections[..]
    };

    let outer_clipped_sections: Vec<SectionEdge>;
    let mut outer_clip_anchors: Vec<Point3> = Vec::new();
    let sections = if !is_plane
        && matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
    {
        let (clipped, anchors) =
            clip_sections_to_outer_region(sections.to_vec(), &boundary_edges, &surface, &wire_pts);
        outer_clipped_sections = clipped;
        outer_clip_anchors = anchors;
        &outer_clipped_sections[..]
    } else {
        sections
    };

    // If no section edges, the face is unsplit -- return as-is with original holes.
    if sections.is_empty() {
        return vec![SplitSubFace {
            surface,
            outer_wire: boundary_edges,
            inner_wires: original_inner_wires,
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        }];
    }

    // No-seam face shortcut: faces whose boundary is entirely Line edges
    // (no seam edges) can't be split by the wire builder (it needs vertical
    // seam connections to form rectangular bands). Construct cap + band
    // sub-faces directly instead. Applies to sphere hemispheres and any
    // other face topology without seam edges.
    let all_boundary_line = boundary_edges.iter().all(|e| {
        matches!(e.curve_3d, EdgeCurve::Line)
            // Exclude degenerate seam edges (start approx end) -- those are periodic
            // seam connections (e.g., torus), not true line boundaries.
            && (e.start_3d - e.end_3d).length() > tol.linear
    });
    // A hemisphere bounded by its chart's equator has no seam edge either.
    let equator_outline = match &surface {
        FaceSurface::Sphere(sphere) => {
            !boundary_edges.is_empty()
                && boundary_edges
                    .iter()
                    .all(|e| special_cases::on_chart_equator(sphere, &e.curve_3d, tol.linear))
        }
        FaceSurface::Plane { .. }
        | FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Nurbs(_) => false,
    };
    // `split_noseam_face_direct` carves cap+band from OPEN section arcs (it
    // skips full-circle sections, relying on the FF boundary-crossing split to
    // pre-open them). A hemisphere cut by a cylinder yields a CLOSED circle
    // interior to the face (no equator crossing) — there are no open arcs, so
    // that path would unsplit it. Defer those to the internal-loops path
    // below, which carves the cap disc as a hole.
    let has_open_section = sections
        .iter()
        .any(|s| (s.start - s.end).length() > tol.linear);
    // Gated to periodic surfaces (and spheres, whose disjoint-arc
    // arrangement fallback is sphere-specific): only there does the wire
    // builder genuinely need seam connections it cannot get from an
    // all-Line boundary. A NON-periodic no-seam face (a ruled strut's
    // bilinear quad crossed by two wedge planes) is topologically a plane
    // face in UV, and the generic arrangement below splits it — the
    // shortcut's disjoint-section fallback would return it unsplit.
    // Open sections that chain into loops inside a sphere face, winding no
    // pole (a column's bottom corners dipping into a hemisphere), are holes
    // of the face and discs of their own, which the internal-loops path
    // below carves; the shortcut only places chains that reach the seam.
    let inside_loops = matches!(surface, FaceSurface::Sphere(_))
        && sections.iter().all(|s| {
            !is_point_on_boundary_uv(s.start, &surface, &boundary_edges, 0.01)
                && !is_point_on_boundary_uv(s.end, &surface, &boundary_edges, 0.01)
        })
        && !sections_form_winding_chain(sections, &surface, tol.linear);
    if (all_boundary_line || equator_outline)
        && !is_plane
        && has_open_section
        && !inside_loops
        && pole_off.is_none()
        && (u_periodic || v_periodic || matches!(surface, FaceSurface::Sphere(_)))
    {
        let mut pieces = split_noseam_face_direct(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            &wire_pts,
            &original_inner_wires,
            tol.linear,
        );
        if matches!(surface, FaceSurface::Sphere(_)) {
            attach_sphere_holes(&mut pieces, &original_inner_wires);
        } else {
            attach_whole_holes(&mut pieces, &original_inner_wires);
        }
        return pieces;
    }

    if matches!(surface, FaceSurface::Torus(_))
        && original_inner_wires.is_empty()
        && let Some(bands) = split_torus_by_coaxial_circles(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            tol.linear,
        )
        .or_else(|| {
            special_cases::split_torus_by_ring_loops(
                &surface,
                &boundary_edges,
                sections,
                rank,
                reversed,
                face_id,
                tol.linear,
            )
        })
        .or_else(|| {
            tube_band::split_tube_band(
                &surface,
                &boundary_edges,
                sections,
                rank,
                reversed,
                face_id,
                tol.linear,
            )
        })
    {
        return bands;
    }

    // Seam-anchor a winding section chain: a chain that winds the periodic
    // direction must connect to the seam in the trace graph, or it floats as
    // an island and every wire builder mis-handles it (the closed-circle
    // band shortcut gets this via the seam-anchor pre-pass; a chain of arcs
    // and conic pieces gets it here). Splitting the crossing piece at the
    // seam meridian makes the crossing point a section ENDPOINT, so the
    // boundary-splitting below anchors the seam edge at the same 3D point
    // and the graphs share a vertex.
    // A section crossing a cone's or a cylinder's seam ruling needs the same
    // anchor to pass from one seam copy to the other: a loop that closes
    // across a cylinder's seam (a pin through a bore's mouth, its seam inside
    // the bite) has its pieces laid a turn apart in UV, and only a vertex on
    // the seam joins them.
    let seam_anchored_sections: Vec<SectionEdge>;
    let sections: &[SectionEdge] = if !is_plane
        && super::pcurve_compute::surface_periods(&surface).0.is_some()
        && (matches!(surface, FaceSurface::Cone(_) | FaceSurface::Cylinder(_))
            || sections_form_winding_chain(sections, &surface, tol.linear))
        && let Some(seam_u) = boundary_seam_u(&boundary_edges, &surface, tol.linear)
    {
        seam_anchored_sections =
            split_sections_at_seam_meridian(sections, &surface, seam_u, &wire_pts, tol.linear);
        &seam_anchored_sections
    } else {
        sections
    };

    // Two equal cylinders crossing: four half ellipses between the points
    // where they touch, two saddles winding the wall and two lenses.
    if u_periodic
        && !is_plane
        && original_inner_wires.is_empty()
        && let Some(pieces) = split_periodic_face_by_touching_saddles(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            tol.linear,
        )
    {
        return pieces;
    }

    // Band shortcut: closed section circles on a u-periodic face split it
    // into stacked bands, not discs. Requires seam-anchored circles (see
    // the seam-anchor pre-pass in fill_images_faces); falls through to the
    // generic paths when preconditions don't hold.
    // A NURBS tube closed in `u` (swept, lofted, or converted from a
    // cylinder) reads its `u` as non-periodic everywhere else, but its rims
    // and winding sections stack into bands just the same.
    let closed_nurbs = matches!(&surface, FaceSurface::Nurbs(n) if n.is_periodic_u());
    if (u_periodic || closed_nurbs)
        && !is_plane
        && original_inner_wires.is_empty()
        && let Some(bands) = split_periodic_face_into_bands(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            tol.linear,
        )
    {
        return bands;
    }

    // Chain-band shortcut: a seam-anchored section chain that WINDS the
    // periodic direction separates the lateral into two bands, exactly like
    // a closed section circle but with a wavy separator (the circle-outside
    // cone∪box fuse: 4 corner ring-arcs + 4 wall arches). The greedy and
    // DCEL both mistrace the chain's identical-tangent parallel twins, so
    // the bands are emitted directly.
    if (u_periodic || closed_nurbs)
        && !is_plane
        && original_inner_wires.is_empty()
        && let Some(bands) = split_periodic_face_by_winding_chain(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            tol.linear,
        )
    {
        return bands;
    }

    // The face's own holes go whole to the pieces that hold them when no
    // section ends on one: a wall with a tunnel's two holes, crossed above
    // them by a rod whose footprint straddles the seam on one side and lies
    // clear of it on the other. The generic paths dropped the clear loop.
    if u_periodic
        && !is_plane
        && sections.iter().all(|s| {
            original_inner_wires.iter().flatten().all(|e| {
                !point_on_oriented_edge(e, s.start, tol.linear * 100.0)
                    && !point_on_oriented_edge(e, s.end, tol.linear * 100.0)
            })
        })
        && let Some(mut pieces) = split_periodic_face_around_seam_holes(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            tol.linear,
        )
    {
        let held: Vec<usize> = pieces.iter().map(|p| p.inner_wires.len()).collect();
        attach_whole_holes(&mut pieces, &original_inner_wires);
        // Each piece's cached interior point was found clear of the section
        // loops alone, and can lie in a hole it now holds.
        for (piece, held) in pieces.iter_mut().zip(held) {
            if piece.inner_wires.len() > held {
                piece.precomputed_interior = None;
            }
        }
        return pieces;
    }

    // Internal section edge shortcut: when section edges form closed loops
    // entirely within the face (not connecting to boundary edges), the wire
    // builder struggles with periodic UV and 4-way junctions. Instead, group
    // the section edges into closed loops and construct sub-faces directly.
    //
    // Detection: check if ALL section endpoints are far from the face
    // boundary in UV space. Project each section endpoint to UV and test
    // if it lies on any boundary edge's UV segment (within tolerance).
    // This is surface-type agnostic and handles curved boundary edges.
    let mut deduped_line_loops: Option<Vec<SectionEdge>> = None;
    let all_sections_internal = if sections.is_empty() {
        false
    } else if is_plane {
        // Plane faces: exactly 1 closed section curve, or all-Line
        // sections forming closed loops strictly inside the boundary
        // (nested coplanar footprints). Multiple circles on the same
        // plane face still need the wire builder for loop formation.
        let single_closed = sections.len() == 1
            && sections.iter().all(|s| {
                (s.start - s.end).length() < tol.linear // closed curve
            });
        if single_closed {
            true
        } else {
            // The holes' vertices where the pave splits put them.
            let hole_points: Vec<Point3> = expanded_inner_wires
                .iter()
                .zip(&original_inner_wires)
                .flat_map(|(split, whole)| split.as_ref().unwrap_or(whole).iter())
                .map(|e| e.start_3d)
                .collect();
            deduped_line_loops = plane_internal_line_loops(
                sections,
                frame,
                &boundary_edges,
                &original_inner_wires,
                &hole_points,
                tol.linear,
            );
            deduped_line_loops.is_some()
        }
    } else {
        // Non-plane faces: check if all section endpoints are off the
        // boundary in UV space.
        let uv_tol = 0.01; // ~0.6 deg in angular coordinates
        // A whole ring has no boundary: its seam placeholders stand for the
        // seam lines, which a section may cross like any other point. (A ring
        // with holes already keeps the boundary test, and its interiors come
        // from the generic paths.)
        let whole_ring = matches!(surface, FaceSurface::Torus(_))
            && original_inner_wires.is_empty()
            && boundary_edges.iter().all(|e| {
                matches!(e.curve_3d, EdgeCurve::Line)
                    && (e.start_3d - e.end_3d).length() < tol.linear * 100.0
            });
        // A closed section that starts on the boundary but runs off it
        // everywhere else only touches it there: a cylinder whose wall is
        // tangent to the face's rim, its section loop started at the
        // tangent point. It bounds a hole all the same, unless it winds
        // round the wall (a whole rim section from the seam's vertex).
        let touches_only = |s: &SectionEdge| {
            if (s.start - s.end).length() >= tol.linear
                || winding_section_chain(std::slice::from_ref(s), &surface, tol.linear, true)
                    .is_some()
            {
                return false;
            }
            let (t0, t1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            // Samples between sixteenths, clear of the start and of the
            // loop's halfway point, where a second touch often lies.
            (0..16).all(|k| {
                let t = t0 + (t1 - t0) * (f64::from(k) + 0.5) / 16.0;
                let p = s.curve_3d.evaluate_with_endpoints(t, s.start, s.end);
                !is_point_on_boundary_uv(p, &surface, &boundary_edges, uv_tol)
            })
        };
        // A section that ends on a hole runs from it, as from the outer
        // boundary: a rod's footprint on a bore, in a U from a window's
        // bottom edge and back, closes on the window, not on itself.
        let on_hole = |p: Point3| {
            original_inner_wires
                .iter()
                .flatten()
                .any(|e| point_on_oriented_edge(e, p, tol.linear * 100.0))
        };
        let endpoints_internal = whole_ring
            || sections.iter().all(|s| {
                let start_on_boundary =
                    is_point_on_boundary_uv(s.start, &surface, &boundary_edges, uv_tol)
                        || on_hole(s.start);
                let end_on_boundary =
                    is_point_on_boundary_uv(s.end, &surface, &boundary_edges, uv_tol)
                        || on_hole(s.end);
                (!start_on_boundary && !end_on_boundary) || touches_only(s)
            });
        // Winding veto: on a u-periodic lateral, a section chain that winds
        // the full period is a band separator, not a contractible hole — an
        // annulus loop with winding number 1 bounds no disc. Treating it as
        // internal attaches the chain as an inner wire on the UNSPLIT face
        // (the "circle outside" cone∪box fuse: 4 corner ring-arcs + 4 wall
        // arches chain into one winding loop, and the whole lateral was kept
        // with the loop as a hole, orphaning the top rim). Winding chains
        // fall through to the general splitter, whose band machinery handles
        // them. A genuine lens hole (a tube poking the wall) has winding 0
        // and keeps the internal path.
        endpoints_internal && !sections_form_winding_chain(sections, &surface, tol.linear)
    };

    // The shortcut takes each chained loop as a disc, and as a hole of the
    // rest of the face: that tiles the face only where no two loops cross.
    // On a ball face, loops that meet (two small circles that cross) go to
    // the general split. The weld is the one the shortcut chains by.
    let loops_apart = || {
        if is_plane {
            // On a plane face the loops must be proved apart: two that cross
            // or touch where neither has an end, or a pair the finder cannot
            // settle, go to the general split.
            let spans: Vec<_> = deduped_line_loops
                .as_deref()
                .unwrap_or(sections)
                .iter()
                .map(|s| edge_splitting::SectionSpan {
                    curve: &s.curve_3d,
                    start: s.start,
                    end: s.end,
                })
                .collect();
            edge_splitting::section_meeting_points(&spans, tol.linear).is_ok_and(|m| m.is_empty())
        } else {
            !matches!(surface, FaceSurface::Sphere(_))
                || sections_meet_in_pairs(
                    deduped_line_loops.as_deref().unwrap_or(sections),
                    tol.linear * 100.0,
                )
        }
    };
    if all_sections_internal && loops_apart() {
        let secs = deduped_line_loops.as_deref().unwrap_or(sections);
        log::debug!(
            "split_face_2d: face {face_id:?} routed to internal-loops path ({} sections)",
            secs.len()
        );
        if let Some(pieces) = split_face_with_internal_loops(
            &surface,
            &boundary_edges,
            &original_inner_wires,
            secs,
            rank,
            reversed,
            face_id,
            &wire_pts,
            tol.linear,
        ) {
            // Loops that wind no pole of the chart are holes of the face and
            // pieces of their own in any chart.
            if let Some(read) = pole_off {
                *read = pieces.len() >= 2;
            }
            return pieces;
        }
    }

    let mut split_pts_3d: Vec<Point3> = sections.iter().flat_map(|s| [s.start, s.end]).collect();
    split_pts_3d.append(&mut outer_clip_anchors);
    if !is_plane && let Some(reg) = split_registry.as_deref_mut() {
        split_pts_3d.extend(reg.values().flatten().copied());
    }
    // A closed rim is re-split here from the raw edge, so it must break at
    // every vertex its pave images carry (the caps sharing it take those
    // images), not only at the sections' ends.
    if matches!(surface, FaceSurface::Cone(_))
        && let Ok(outer) = topo.wire(face.outer_wire())
    {
        for oe in outer.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            if edge.start() != edge.end() {
                continue;
            }
            let Some(imgs) = edge_images.get(&oe.edge()).filter(|imgs| imgs.len() > 1) else {
                continue;
            };
            for &img in imgs {
                if let Ok(piece) = topo.edge(img)
                    && let (Ok(a), Ok(b)) = (topo.vertex(piece.start()), topo.vertex(piece.end()))
                {
                    split_pts_3d.push(a.point());
                    split_pts_3d.push(b.point());
                }
            }
        }
    }

    // A torus band round the axis has a meridian arc for its seam, at one u
    // from end to end, where a cylinder or cone has a line.
    let torus_band = matches!(surface, FaceSurface::Torus(_)) && u_periodic && !v_periodic;
    let meridian = |e: &OrientedPCurveEdge| match (&surface, &e.curve_3d) {
        (FaceSurface::Torus(t), EdgeCurve::Circle(c)) => {
            torus_band
                && (e.start_3d - e.end_3d).length() >= 1e-10
                && (c.radius() - t.minor_radius()).abs() <= tol.linear
                && c.normal().dot(t.z_axis()).abs() <= 1e-9
                && c.normal().dot(c.center() - t.center()).abs() <= tol.linear
        }
        _ => false,
    };
    // A meridian's two ends project to `u` on either side of the angle's
    // origin when the seam lies on it.
    for e in &mut boundary_edges {
        if meridian(e) {
            e.end_uv = Point2::new(e.start_uv.x(), e.end_uv.y());
        }
    }
    let seam_u_opt = boundary_edges.iter().find_map(|e| {
        if matches!(e.curve_3d, EdgeCurve::Line) || meridian(e) {
            surface.project_point(e.start_3d).map(|(u, _)| u)
        } else {
            None
        }
    });
    // For periodic faces, align closed boundary edge UV with seam edge UV.
    // The same 3D vertex projects to u=0 (from circle unwrapping) and u=seam
    // (from Line edge projection). Shift the circle UV so it starts at seam_u.
    let mut walked_from_closed_rim = false;
    if u_periodic {
        // On a cylinder laid out in one window (seam lines a period apart),
        // a closed rim's turn, run from its start_uv the way it is
        // traversed, belongs in that window: it moves there by whole
        // periods, or stays where the boundary walk put it.
        let window = if matches!(surface, FaceSurface::Cylinder(_)) {
            let us: Vec<f64> = boundary_edges
                .iter()
                .filter(|e| {
                    matches!(e.curve_3d, EdgeCurve::Line)
                        && (e.start_uv.x() - e.end_uv.x()).abs() <= 1e-9
                })
                .map(|e| e.start_uv.x())
                .collect();
            let lo = us.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = us.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (us.len() >= 2 && ((hi - lo) - std::f64::consts::TAU).abs() <= 1e-6).then_some(lo)
        } else {
            None
        };
        if let Some(seam_u) = seam_u_opt {
            for edge in &mut boundary_edges {
                if (edge.start_3d - edge.end_3d).length() < 1e-10
                    && let Some(lo) = window
                {
                    let native = if edge.end_uv.x() < edge.start_uv.x() {
                        -1.0_f64
                    } else {
                        1.0
                    };
                    let dir = if edge.forward { native } else { -native };
                    let run_lo = edge
                        .start_uv
                        .x()
                        .min(std::f64::consts::TAU.madd(dir, edge.start_uv.x()));
                    let turns = (lo - run_lo) / std::f64::consts::TAU;
                    if (turns - turns.round()).abs() <= 1e-6 {
                        let by = turns.round() * std::f64::consts::TAU;
                        edge.start_uv = Point2::new(edge.start_uv.x() + by, edge.start_uv.y());
                        edge.end_uv = Point2::new(edge.end_uv.x() + by, edge.end_uv.y());
                        continue;
                    }
                }
                if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                    // Closed edge: shift UV so start_uv.x() == seam_u. One
                    // already starting on a copy of the seam's u was placed
                    // there by the boundary walk, in step with its seam
                    // lines, and stays.
                    let shift = seam_u - edge.start_uv.x();
                    let on_seam_copy = matches!(surface, FaceSurface::Cone(_))
                        && (shift
                            - (shift / std::f64::consts::TAU).round() * std::f64::consts::TAU)
                            .abs()
                            <= 1e-6;
                    if shift.abs() > 0.01 && !on_seam_copy {
                        edge.start_uv = Point2::new(edge.start_uv.x() + shift, edge.start_uv.y());
                        edge.end_uv = Point2::new(edge.end_uv.x() + shift, edge.end_uv.y());
                    }
                }
            }
            // A pointed cone's two seam copies meet at its apex, which a trace
            // cut open along the seam cannot close across.
            let frustum = match &surface {
                FaceSurface::Cone(cone) => boundary_edges.iter().all(|e| {
                    (e.start_3d - cone.apex()).length() > tol.linear
                        && (e.end_3d - cone.apex()).length() > tol.linear
                }),
                _ => false,
            };
            if matches!(surface, FaceSurface::Cylinder(_)) || torus_band || frustum {
                walked_from_closed_rim = walk_boundary_from_closed_rim(&mut boundary_edges);
            }
        }
    }

    // For periodic faces with section edges, split closed boundary edges
    // (full circles) at the point diametrically opposite the seam vertex
    // in the surface's UV parameterization (u = seam_u + pi).
    //
    // The seam vertex (where the boundary circle starts/ends) is shared
    // with the seam Line edge. Splitting the circle at the UV-antipodal
    // point creates half-arcs whose endpoints match the seam edge vertices,
    // enabling the wire builder to form proper rectangular bands.
    // A cylinder's rim points split here for the trace alone, which no
    // neighbouring face shares: the pieces meeting at one rejoin afterwards.
    let mut trace_splits: Vec<Point3> = Vec::new();
    if u_periodic && !sections.is_empty() {
        let seam_u = seam_u_opt.unwrap_or(0.0);
        let anti_u = (seam_u + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU);

        for edge in &boundary_edges {
            if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                // Closed edge: find the 3D point at u = seam_u + pi on the surface.
                // Project the boundary vertex to get v, then evaluate surface at (anti_u, v).
                if let Some((_, v)) = surface.project_point(edge.start_3d)
                    && let Some(anti_pt) = surface.evaluate(anti_u, v)
                {
                    // A section already ending at the antipode (a cone's top
                    // rim touched there) would leave two halves sharing both
                    // ends, which the edge merge welds: quarter them too.
                    let taken = split_pts_3d
                        .iter()
                        .any(|&p| (p - anti_pt).length() < 100.0 * tol.linear);
                    split_pts_3d.push(anti_pt);
                    if !taken && matches!(surface, FaceSurface::Cylinder(_)) {
                        trace_splits.push(anti_pt);
                    }
                    if taken && matches!(surface, FaceSurface::Cone(_)) {
                        for quarter in [0.5, 1.5] {
                            if let Some(p) =
                                surface.evaluate(std::f64::consts::PI.madd(quarter, seam_u), v)
                            {
                                split_pts_3d.push(p);
                            }
                        }
                    }
                }
            }
        }
    }

    // A LINE section can cross a boundary ARC mid-span — the groove-mouth
    // corner, where a pocket ring absorbed into the outer boundary (a bay)
    // bulges across the notch wall line. Section ENDPOINTS alone leave that
    // crossing unsplit, and the arrangement's chord-based crossing detection
    // misses the real arc by the sagitta (the arc's chord stays clear of the
    // section), so the notch region gets traced through the pocket bite to a
    // phantom corner inside air — an unpaired mouth triangle on the kept top
    // face. Collect the true circle×line crossings here; they are applied to
    // the boundary ONLY on the combined-arrangement path below (the angular
    // wire-builder paths are calibrated to the unsplit boundary — splitting
    // globally regressed the d-series lip fuses).
    let mut boundary_cross_pts: Vec<Point3> = Vec::new();
    if is_plane {
        for edge in &boundary_edges {
            let EdgeCurve::Circle(c) = &edge.curve_3d else {
                continue;
            };
            for s in sections {
                if !matches!(s.curve_3d, EdgeCurve::Line) {
                    continue;
                }
                for (p, _) in c.intersect_segment(s.start, s.end, tol.linear) {
                    boundary_cross_pts.push(p);
                }
            }
        }
    }
    let boundary_arc_crossed = !boundary_cross_pts.is_empty();

    // The closed rims as they arrive, before the split at the seam's
    // antipode above: a rim no section touches is put back whole in the loop
    // that runs both its halves, so it stays the edge its cap shares (a cone's
    // base, or the top of a hole whose plate a box presses on from below).
    let closed_rims: Vec<OrientedPCurveEdge> =
        if matches!(surface, FaceSurface::Cone(_) | FaceSurface::Cylinder(_)) {
            boundary_edges
                .iter()
                .filter(|e| {
                    matches!(e.curve_3d, EdgeCurve::Circle(_))
                        && (e.start_3d - e.end_3d).length() < 1e-10
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

    let boundary_edges = if ball_chart_clear {
        edge_splitting::split_clear_ball_edges_at_3d_points(
            boundary_edges,
            &split_pts_3d,
            &surface,
            tol.linear,
        )
    } else {
        split_boundary_edges_at_3d_points(
            boundary_edges,
            &split_pts_3d,
            if is_plane { Some(frame) } else { None },
            &surface,
            tol.linear,
        )
    };

    // Reorder boundary edges: Line (seam) edges first, then curved (circle)
    // edges. This ensures the wire builder starts loops from seam edges,
    // forming rectangular bands before circle arcs can self-close.
    let boundary_edges = if u_periodic && !sections.is_empty() {
        let (mut lines, curves): (Vec<_>, Vec<_>) = boundary_edges
            .into_iter()
            .partition(|e| matches!(e.curve_3d, EdgeCurve::Line));
        lines.extend(curves);
        lines
    } else {
        boundary_edges
    };

    // Weld plane-face section endpoints to coincident boundary (and earlier
    // section) endpoints within the weld-scale band (100·tol). A marched-NURBS
    // section endpoint carries the curve-fit error (~1e-6) while its chain
    // partner's endpoint is an exact clip value; the difference exceeds the
    // 1e-7 vertex quantization used by both the wire builder and the planar
    // arrangement, so the chain junction never forms — the trace walks
    // out-and-back along the section and the face is left unsplit (the
    // snap-slot wall's socket-profile silhouette). Exact junctions are
    // untouched (zero distance ⇒ no-op). Precomputed UVs are cleared for
    // moved endpoints so consumers re-derive them from the welded 3D.
    let welded_sections: Vec<SectionEdge>;
    let sections: &[SectionEdge] = if is_plane && !sections.is_empty() {
        let weld = tol.linear * 100.0;
        let mut anchors: Vec<Point3> = boundary_edges
            .iter()
            .flat_map(|e| [e.start_3d, e.end_3d])
            .collect();
        let mut out = sections.to_vec();
        for s in &mut out {
            let mut moved = false;
            for pick_start in [true, false] {
                let p = if pick_start { s.start } else { s.end };
                let snapped = anchors
                    .iter()
                    .find(|a| {
                        let d = (**a - p).length();
                        d > 1e-12 && d <= weld
                    })
                    .copied();
                if let Some(a) = snapped {
                    if pick_start {
                        s.start = a;
                    } else {
                        s.end = a;
                    }
                    moved = true;
                } else {
                    anchors.push(p);
                }
            }
            if moved {
                s.start_uv_a = None;
                s.end_uv_a = None;
                s.start_uv_b = None;
                s.end_uv_b = None;
            }
        }
        // Second pass: an endpoint can also land just OFF another section's
        // INTERIOR (a T-junction, not a shared corner — e.g. a plane×cone
        // conic ending on the top-plane section where the cone rim meets it;
        // the marched endpoint carries ~1e-6 of fit error). No anchor exists
        // mid-span, so project the endpoint onto the Line sections and snap it
        // to the nearest strictly-interior foot within the weld band. With the
        // endpoint exactly ON the line, the downstream T-junction split (1e-7
        // on-curve test) fires and the crossed section divides. Only CURVED
        // (marched/fitted) sections' endpoints are candidates — Line section
        // endpoints come from exact clips, so the Line geometry referenced
        // here never moves during this pass.
        let mut lines: Vec<(usize, Point3, Point3)> = out
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s.curve_3d, EdgeCurve::Line))
            .map(|(i, s)| (i, s.start, s.end))
            .collect();
        // BOUNDARY Line edges are projection targets too: a curved section
        // ending mid-span of the face's own boundary (a cone conic landing on
        // the box wall's bottom edge) has no anchor there either, and the
        // arrangement's T-break test needs the endpoint exactly ON the edge.
        // usize::MAX-based ids keep them distinct from every section index.
        lines.extend(
            boundary_edges
                .iter()
                .enumerate()
                .filter(|(_, e)| matches!(e.curve_3d, EdgeCurve::Line))
                .map(|(i, e)| (usize::MAX - i, e.start_3d, e.end_3d)),
        );
        for si in 0..out.len() {
            if matches!(out[si].curve_3d, EdgeCurve::Line) {
                continue;
            }
            let mut moved = false;
            for pick_start in [true, false] {
                let p = if pick_start {
                    out[si].start
                } else {
                    out[si].end
                };
                let mut best: Option<(f64, Point3)> = None;
                for &(li, a, b) in &lines {
                    if li == si {
                        continue;
                    }
                    let ab = b - a;
                    let len2 = ab.dot(ab);
                    if len2 <= 0.0 {
                        continue;
                    }
                    let len = len2.sqrt();
                    let t = (p - a).dot(ab) / len2;
                    // Strictly interior along the SEGMENT: a foot near or past
                    // either end is the anchor pass's job (and a point near
                    // the line's extension must not snap onto the span).
                    if t * len <= weld || (1.0 - t) * len <= weld {
                        continue;
                    }
                    let foot = a + ab * t;
                    let d = (p - foot).length();
                    if d > 1e-12 && d <= weld && best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, foot));
                    }
                }
                if let Some((_, foot)) = best {
                    if pick_start {
                        out[si].start = foot;
                    } else {
                        out[si].end = foot;
                    }
                    moved = true;
                }
            }
            if moved {
                out[si].start_uv_a = None;
                out[si].end_uv_a = None;
                out[si].start_uv_b = None;
                out[si].end_uv_b = None;
            }
        }
        welded_sections = out;
        &welded_sections
    } else {
        sections
    };

    let boundary_edges_backup = if is_plane && sections.len() >= 2 {
        Some(boundary_edges.clone())
    } else {
        None
    };

    // Convert section edges to OrientedPCurveEdge (both orientations).
    let mut all_edges = boundary_edges;
    let n_boundary_edges = all_edges.len();

    // Holed planar face cut by sections: weave the hole boundaries into the
    // arrangement (trim sections at hole crossings, split hole edges) so the
    // wire builder traces the true material region. Only holes a section
    // actually interacts with are woven; non-interacting holes are returned as
    // `passthrough` indices and attached whole below (so a cap with many
    // untouched openings — a baseplate top cut at one corner — keeps the other
    // openings' exact arc geometry instead of chord-fragmenting them).
    let mut woven_hole_indices: Vec<usize> = Vec::new();
    let mut holes_promoted = false;
    let mut holes_integrated = if is_plane && !original_inner_wires.is_empty() {
        if let Some((extra, passthrough)) = integrate_holes_plane(
            sections,
            &original_inner_wires,
            frame,
            &surface,
            &wire_pts,
            WEAVE_SECTION_SRC_BASE,
        ) {
            all_edges.extend(extra);
            let pass: std::collections::HashSet<usize> = passthrough.into_iter().collect();
            woven_hole_indices = (0..original_inner_wires.len())
                .filter(|i| !pass.contains(i))
                .collect();
            true
        } else {
            false
        }
    } else {
        false
    };
    // Promote pave-split passthrough holes into the arrangement: when a hole
    // wire's (image-expanded) vertex lies exactly ON a section segment, the
    // section runs THROUGH the opening — attaching the hole whole after the
    // split leaves the notch piece overlapping the opening covered by the
    // kept face with nothing below it (the fit-offset groove-mouth sliver).
    // A section that ENDS at a hole vertex needs the hole just the same: it
    // is a bridge from the outer wire to the opening (a bracket plane's
    // trace across a barrel's end annulus, clipped to the ring), and with
    // the hole attached whole its inner end dangles and the bridge is
    // dropped as a pendant.
    if is_plane && !original_inner_wires.is_empty() {
        let sec_segs: Vec<(Point3, Point3)> = sections
            .iter()
            .filter(|sct| matches!(sct.curve_3d, EdgeCurve::Line))
            .map(|sct| (sct.start, sct.end))
            .collect();
        // A curved section ending at a hole's vertex bridges to the opening
        // just as a line does (a cylinder cut through a ring of round holes:
        // its circle's arcs run from hole to hole, each ending on a rim).
        let arc_ends: Vec<Point3> = sections
            .iter()
            .filter(|sct| {
                !matches!(sct.curve_3d, EdgeCurve::Line)
                    && (sct.start - sct.end).length() > tol.linear
            })
            .flat_map(|sct| [sct.start, sct.end])
            .collect();
        let on_seg = |p: Point3, a: Point3, b: Point3| -> bool {
            let d = b - a;
            let len2 = d.dot(d);
            if len2 < 1e-18 {
                return false;
            }
            if (p - a).length() < tol.linear || (p - b).length() < tol.linear {
                return true;
            }
            let t = (p - a).dot(d) / len2;
            if !(1e-6..=1.0 - 1e-6).contains(&t) {
                return false;
            }
            ((a + d * t) - p).length() < tol.linear * 10.0
        };
        let woven_set: std::collections::HashSet<usize> =
            woven_hole_indices.iter().copied().collect();
        for (i, orig_hole) in original_inner_wires.iter().enumerate() {
            // The pave-split expansion carries the exact minted vertices the
            // coincidence test needs; fall back to the stored wire otherwise.
            let hole = expanded_inner_wires
                .get(i)
                .and_then(Option::as_ref)
                .unwrap_or(orig_hole);
            if woven_set.contains(&i) {
                continue;
            }
            let mut coincident: Vec<Point3> = Vec::new();
            for e in hole {
                for p in [e.start_3d, e.end_3d] {
                    if (sec_segs.iter().any(|&(a, b)| on_seg(p, a, b))
                        || arc_ends
                            .iter()
                            .any(|q| (*q - p).length() < tol.linear * 10.0))
                        && !coincident.iter().any(|q| (*q - p).length() < tol.linear)
                    {
                        coincident.push(p);
                    }
                }
            }
            if coincident.is_empty() {
                continue;
            }
            if !holes_integrated {
                holes_integrated = true;
                holes_promoted = true;
                // Holes of lines and conic arcs are read exactly; a dense
                // polygon of each hole's rim finds the section's crossings.
                let hole_exact: Vec<Option<Vec<brepkit_math::region2d::Boundary2>>> =
                    original_inner_wires
                        .iter()
                        .map(|h| containment::exact_hole(h, frame))
                        .collect();
                let hole_polys: Vec<Vec<Point2>> = original_inner_wires
                    .iter()
                    .map(|h| {
                        h.iter()
                            .flat_map(|e| {
                                let n: u32 = if matches!(e.curve_3d, EdgeCurve::Line) {
                                    1
                                } else {
                                    64
                                };
                                (0..n).map(move |k| {
                                    frame.project(evaluate_edge_at_t(
                                        &e.curve_3d,
                                        e.start_3d,
                                        e.end_3d,
                                        f64::from(k) / f64::from(n),
                                    ))
                                })
                            })
                            .collect()
                    })
                    .collect();
                // A section clear of every hole at its samples can still dip
                // into a narrow opening between two of them: within one step
                // of its polyline, the stretch between two crossings of a
                // hole's boundary lies inside the hole. A section touching a
                // hole's rim (tangent, or ending on it) has no such stretch.
                let enters_hole = |sct: &SectionEdge| {
                    const STEPS: u32 = 64;
                    let (d0, d1) = sct.curve_3d.domain_with_endpoints(sct.start, sct.end);
                    let pts: Vec<Point2> = (0..=STEPS)
                        .map(|k| {
                            let t = (d1 - d0).madd(f64::from(k) / f64::from(STEPS), d0);
                            frame.project(
                                sct.curve_3d.evaluate_with_endpoints(t, sct.start, sct.end),
                            )
                        })
                        .collect();
                    hole_polys
                        .iter()
                        .zip(&hole_exact)
                        .filter(|(poly, _)| poly.len() >= 3)
                        .any(|(poly, exact)| {
                            let in_hole = |p: Point2| match exact {
                                Some(pieces) => {
                                    brepkit_math::region2d::point_in_region(
                                        pieces,
                                        p,
                                        tol.linear * 10.0,
                                    ) == Some(true)
                                }
                                None => {
                                    super::classify_2d::point_in_polygon_2d(p, poly)
                                        && super::classify_2d::distance_to_polygon_boundary(p, poly)
                                            > tol.linear * 10.0
                                }
                            };
                            let dips_in = |w: &[Point2]| {
                                let from = w[0];
                                let reach = |q: &Point2| (q.x() - from.x()).hypot(q.y() - from.y());
                                let mut at: Vec<Point2> = (0..poly.len())
                                    .filter_map(|i| {
                                        segments_cross_at(
                                            w[0],
                                            w[1],
                                            poly[i],
                                            poly[(i + 1) % poly.len()],
                                        )
                                    })
                                    .collect();
                                at.sort_by(|a, b| reach(a).total_cmp(&reach(b)));
                                at.windows(2).any(|c| {
                                    in_hole(Point2::new(
                                        0.5 * (c[0].x() + c[1].x()),
                                        0.5 * (c[0].y() + c[1].y()),
                                    ))
                                })
                            };
                            pts[1..pts.len() - 1].iter().any(|&p| in_hole(p))
                                || pts.windows(2).any(dips_in)
                        })
                };
                let hole_lines: Vec<(Point2, Point2)> = original_inner_wires
                    .iter()
                    .flatten()
                    .filter(|e| edge_curve_is_straight(&e.curve_3d))
                    .map(|e| (frame.project(e.start_3d), frame.project(e.end_3d)))
                    .collect();
                let chord_on_hole_line = |sct: &SectionEdge| {
                    let (a, b) = (frame.project(sct.start), frame.project(sct.end));
                    hole_lines
                        .iter()
                        .any(|&(h0, h1)| chord_on_line(a, b, h0, h1))
                };
                for (si, sct) in sections.iter().enumerate() {
                    if !matches!(sct.curve_3d, EdgeCurve::Line) {
                        // A curved section clear of every opening (a corner
                        // arc closing a cell outline on the outer ring) is
                        // carried whole: without it the line sections it
                        // joins dangle and the whole arrangement is pruned.
                        // One reaching into an opening has no split here,
                        // and one whose chord lies along a straight hole
                        // edge's line is left out as the weave leaves it:
                        // the arrangement reads regions by chords.
                        if (sct.start - sct.end).length() >= 1e-10
                            && !enters_hole(sct)
                            && !chord_on_hole_line(sct)
                        {
                            all_edges.extend(curved_section_pair(
                                sct,
                                frame,
                                &surface,
                                &wire_pts,
                                WEAVE_SECTION_SRC_BASE + si,
                            ));
                        }
                        continue;
                    }
                    let s0 = frame.project(sct.start);
                    let s1 = frame.project(sct.end);
                    let mk = |su: Point2, eu: Point2, s3: Point3, e3: Point3, fwd: bool| {
                        use brepkit_math::curves2d::{Curve2D, Line2D};
                        use brepkit_math::vec::Vec2;
                        let d = Vec2::new(eu.x() - su.x(), eu.y() - su.y());
                        let len = (d.x() * d.x() + d.y() * d.y()).sqrt();
                        let dir = if len > 1e-12 {
                            Vec2::new(d.x() / len, d.y() / len)
                        } else {
                            Vec2::new(1.0, 0.0)
                        };
                        Line2D::new(su, dir).ok().map(|l| OrientedPCurveEdge {
                            curve_3d: EdgeCurve::Line,
                            pcurve: Curve2D::Line(l),
                            start_uv: su,
                            end_uv: eu,
                            start_3d: s3,
                            end_3d: e3,
                            forward: fwd,
                            source_edge_idx: Some(WEAVE_SECTION_SRC_BASE + si),
                            pave_block_id: None,
                        })
                    };
                    if let Some(e1) = mk(s0, s1, sct.start, sct.end, true) {
                        all_edges.push(e1);
                    }
                    if let Some(e2) = mk(s1, s0, sct.end, sct.start, false) {
                        all_edges.push(e2);
                    }
                }
            }
            woven_hole_indices.push(i);
            all_edges.extend(hole.iter().cloned());
            // Arc-true: a rim split in two has a two-point chord polygon, and
            // a line through its centre leaves a diameter, on which the
            // segment test below reads neither side.
            let poly: Vec<Point2> = sample_wire_loop_uv_via_frame(hole, frame);
            let arcs: Vec<(Point2, f64, Point2, Point2)> = hole
                .iter()
                .filter_map(|e| {
                    let EdgeCurve::Circle(c3) = &e.curve_3d else {
                        return None;
                    };
                    Some((
                        frame.project(c3.center()),
                        c3.radius(),
                        frame.project(e.start_3d),
                        frame.project(e.end_3d),
                    ))
                })
                .collect();
            let side = |a: Point2, b: Point2, p: Point2| -> f64 {
                (b.x() - a.x()).madd(p.y() - a.y(), -((b.y() - a.y()) * (p.x() - a.x())))
            };
            // A section's run across the hole between the ends of one rim arc
            // lies on an edge of the vertex polygon and on the arc's chord,
            // where the first two tests read it as outside the hole.
            let exact = containment::exact_hole(orig_hole, frame);
            let in_region = |p: Point2| -> bool {
                (super::classify_2d::point_in_polygon_2d(p, &poly)
                    && super::classify_2d::distance_to_polygon_boundary(p, &poly)
                        > tol.linear * 10.0)
                    || arcs.iter().any(|&(c, r, u0, u1)| {
                        let d = ((p.x() - c.x()).powi(2) + (p.y() - c.y()).powi(2)).sqrt();
                        d < r && side(u0, u1, p) * side(u0, u1, c) < 0.0
                    })
                    || exact.as_ref().is_some_and(|pieces| {
                        brepkit_math::region2d::point_in_region(pieces, p, tol.linear * 10.0)
                            == Some(true)
                    })
            };
            let mut rebuilt: Vec<OrientedPCurveEdge> = Vec::new();
            for e in std::mem::take(&mut all_edges) {
                let is_weave_section = e
                    .source_edge_idx
                    .is_some_and(|si| si >= WEAVE_SECTION_SRC_BASE)
                    && matches!(e.curve_3d, EdgeCurve::Line);
                if !is_weave_section {
                    rebuilt.push(e);
                    continue;
                }
                let dir = e.end_3d - e.start_3d;
                let len2 = dir.dot(dir);
                let mut ts: Vec<f64> = vec![0.0, 1.0];
                for p in &coincident {
                    if len2 > 1e-18 {
                        let t = (*p - e.start_3d).dot(dir) / len2;
                        if (1e-6..=1.0 - 1e-6).contains(&t)
                            && ((e.start_3d + dir * t) - *p).length() < tol.linear * 10.0
                        {
                            ts.push(t);
                        }
                    }
                }
                if ts.len() == 2 {
                    rebuilt.push(e);
                    continue;
                }
                ts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                ts.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
                for w in ts.windows(2) {
                    let (ta, tb) = (w[0], w[1]);
                    let pm3 = e.start_3d + dir * (0.5 * (ta + tb));
                    if in_region(frame.project(pm3)) {
                        continue;
                    }
                    let sa = e.start_3d + dir * ta;
                    let sb = e.start_3d + dir * tb;
                    let mut piece = e.clone();
                    piece.start_3d = sa;
                    piece.end_3d = sb;
                    piece.start_uv = frame.project(sa);
                    piece.end_uv = frame.project(sb);
                    rebuilt.push(piece);
                }
            }
            all_edges = rebuilt;
        }
    }

    // The holes the arrangement actually consumed (woven). The passthrough holes
    // are attached whole after the split (see the `!holes_integrated` /
    // passthrough attach pass below).
    let woven_inner_wires: Vec<Vec<OrientedPCurveEdge>> = woven_hole_indices
        .iter()
        .map(|&i| original_inner_wires[i].clone())
        .collect();
    let passthrough_inner_wires: Vec<Vec<OrientedPCurveEdge>> = if holes_integrated {
        let woven: std::collections::HashSet<usize> = woven_hole_indices.iter().copied().collect();
        (0..original_inner_wires.len())
            .filter(|i| !woven.contains(i))
            .map(|i| original_inner_wires[i].clone())
            .collect()
    } else {
        Vec::new()
    };

    // A whole circle across a cylinder's wall starts and ends at one point,
    // on the seam line, where the wall's two seam copies are glued into one
    // vertex of the trace graph. Two such circles (a pin through a plate) and
    // the seam between them bound a band the face traces cannot tell from
    // the bands beside it at that pinch, and it was dropped. Each circle is
    // split in thirds from its start, so its pieces run between plain
    // vertices, each under the half turn the traces' unwrapping allows.
    let split_circles: Vec<SectionEdge>;
    let mut circles_split = false;
    let sections: &[SectionEdge] = if u_periodic
        && matches!(surface, FaceSurface::Cylinder(_))
        && sections.iter().any(|s| {
            matches!(s.curve_3d, EdgeCurve::Circle(_)) && (s.start - s.end).length() < 1e-10
        }) {
        split_circles = sections
            .iter()
            .flat_map(|s| {
                if !matches!(s.curve_3d, EdgeCurve::Circle(_))
                    || (s.start - s.end).length() >= 1e-10
                {
                    return vec![s.clone()];
                }
                // The circle's own UV run, laid as the whole section would
                // be (a turn along one v, in the face's window), is cut in
                // thirds with it: a fresh projection would put the pieces at
                // the surface's own u, off the window.
                let pcurve = match rank {
                    Rank::A => &s.pcurve_a,
                    Rank::B => &s.pcurve_b,
                };
                let hints = match rank {
                    Rank::A => s.start_uv_a.zip(s.end_uv_a),
                    Rank::B => s.start_uv_b.zip(s.end_uv_b),
                };
                let (su, eu) = hints.unwrap_or_else(|| {
                    uv_endpoints_from_pcurve(pcurve, s.start, s.end, &surface, &wire_pts)
                });
                if ((eu.x() - su.x()).abs() - std::f64::consts::TAU).abs() > 1e-6 {
                    return vec![s.clone()];
                }
                // The circle lies at one v, so each third's point is the
                // surface's at its UV.
                let at = |k: f64| {
                    let uv = Point2::new((eu.x() - su.x()).madd(k / 3.0, su.x()), su.y());
                    (surface.evaluate(uv.x(), uv.y()).unwrap_or(s.start), uv)
                };
                let pts = [at(0.0), at(1.0), at(2.0), at(3.0)];
                pts.windows(2)
                    .map(|w| {
                        let ((a, ua), (b, ub)) = (w[0], w[1]);
                        let (a, b) = (
                            if (a - s.start).length() < 1e-9 {
                                s.start
                            } else {
                                a
                            },
                            if (b - s.start).length() < 1e-9 {
                                s.start
                            } else {
                                b
                            },
                        );
                        let mut piece = s.clone();
                        piece.start = a;
                        piece.end = b;
                        piece.start_uv_a = Some(ua);
                        piece.end_uv_a = Some(ub);
                        piece.start_uv_b = Some(ua);
                        piece.end_uv_b = Some(ub);
                        piece.pave_block_id = None;
                        let dir = brepkit_math::vec::Vec2::new(ub.x() - ua.x(), ub.y() - ua.y());
                        let line = dir
                            .normalize()
                            .ok()
                            .and_then(|d| brepkit_math::curves2d::Line2D::new(ua, d).ok());
                        if let Some(l) = line {
                            piece.pcurve_a = brepkit_math::curves2d::Curve2D::Line(l.clone());
                            piece.pcurve_b = brepkit_math::curves2d::Curve2D::Line(l);
                        }
                        piece
                    })
                    .collect()
            })
            .collect();
        circles_split = true;
        &split_circles
    } else {
        sections
    };

    for section in sections {
        if holes_integrated {
            break;
        }
        // Skip full-circle section edges on plane faces -- they have
        // start approx end in 3D and would produce degenerate UV edges.
        // The half-arc section edges handle the plane face correctly.
        let is_closed_edge = (section.start - section.end).length() < 1e-10;
        if is_closed_edge && is_plane {
            continue;
        }

        // A section lying entirely inside one of the face's holes is over
        // air, not material. The hole weave drops these itself, but when it
        // bails (a section crossing a hole's corner ARC cannot be woven) the
        // general path used to carry them into the arrangement, where they
        // close into phantom in-hole loops — garbage faces plus punched
        // holes (the spacer foot plate's corner lenses, #1570). Arc-true
        // sampling on both the section and the hole rims; strict interior
        // with a margin so a rim-riding section is never eaten.
        if is_plane && !original_inner_wires.is_empty() {
            let (d0, d1) = section
                .curve_3d
                .domain_with_endpoints(section.start, section.end);
            let samples: Vec<Point2> = (0..=8)
                .map(|k| {
                    let t = d0 + (d1 - d0) * f64::from(k) / 8.0;
                    frame.project(section.curve_3d.evaluate_with_endpoints(
                        t,
                        section.start,
                        section.end,
                    ))
                })
                .collect();
            let margin = tol.linear * 10.0;
            let wholly_in_hole = original_inner_wires.iter().any(|hole| {
                let poly = sample_wire_loop_uv_via_frame(hole, frame);
                poly.len() >= 3
                    && samples.iter().all(|&p| {
                        super::classify_2d::point_in_polygon_2d(p, &poly)
                            && super::classify_2d::distance_to_polygon_boundary(p, &poly) > margin
                    })
            });
            if trace_split {
                log::debug!(
                    "STRACE-INHOLE sec ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}) {} wholly_in_hole={wholly_in_hole} holes={}",
                    section.start.x(),
                    section.start.y(),
                    section.start.z(),
                    section.end.x(),
                    section.end.y(),
                    section.end.z(),
                    section.curve_3d.type_tag(),
                    original_inner_wires.len()
                );
            }
            if wholly_in_hole {
                continue;
            }
        }

        // Curved sections on plane faces must live in the same PlaneFrame
        // as the boundary edges. The pcurve from build_section_edges was
        // fitted in a frame anchored at the original (pre-split) wire, so
        // its UV space — and its NURBS parameter domain — need not match
        // `frame`; using it would disconnect the section from the boundary
        // in UV. Refit it in this face's frame.
        let owned_pcurve;
        let pcurve_on_this_face = if is_plane && !matches!(section.curve_3d, EdgeCurve::Line) {
            owned_pcurve = super::pcurve_compute::compute_pcurve_on_surface(
                &section.curve_3d,
                section.start,
                section.end,
                &surface,
                &wire_pts,
                Some(frame),
            );
            &owned_pcurve
        } else {
            match rank {
                Rank::A => &section.pcurve_a,
                Rank::B => &section.pcurve_b,
            }
        };

        // Project section endpoints to UV.
        // Use pre-computed UV endpoints when available (e.g. seam-split half-arcs
        // where the unwrapped UV was computed from the arc samples). Otherwise,
        // for non-plane faces, use the pcurve's endpoint evaluations instead
        // of independent surface projection -- this ensures UV endpoints are
        // consistent with the pcurve's unwrapped parameterization (e.g. arc
        // ending at u=2pi rather than u=0 after periodic unwrapping).
        let (start_uv, end_uv) = if is_plane {
            // Plane faces: project in the boundary's frame. Precomputed UVs
            // (when present) come from build_section_edges' own frame and
            // would not connect to the boundary edges in UV.
            (frame.project(section.start), frame.project(section.end))
        } else {
            match rank {
                Rank::A => {
                    if let (Some(su), Some(eu)) = (section.start_uv_a, section.end_uv_a) {
                        (su, eu)
                    } else {
                        uv_endpoints_from_pcurve(
                            pcurve_on_this_face,
                            section.start,
                            section.end,
                            &surface,
                            &wire_pts,
                        )
                    }
                }
                Rank::B => {
                    if let (Some(su), Some(eu)) = (section.start_uv_b, section.end_uv_b) {
                        (su, eu)
                    } else {
                        uv_endpoints_from_pcurve(
                            pcurve_on_this_face,
                            section.start,
                            section.end,
                            &surface,
                            &wire_pts,
                        )
                    }
                }
            }
        };

        // Forward direction. Both forward and reverse share the same
        // source_edge_idx so build_topology_face creates one shared edge.
        let section_idx = all_edges.len();
        let pb_id = section.pave_block_id;
        all_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
            source_edge_idx: Some(section_idx),
            pave_block_id: pb_id,
        });
        // Reverse direction (for the adjacent sub-face).
        all_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv: end_uv,
            end_uv: start_uv,
            start_3d: section.end,
            end_3d: section.start,
            forward: false,
            source_edge_idx: Some(section_idx),
            pave_block_id: pb_id,
        });
    }

    // One u window for the whole wall: the boundary walk can lay a rim a
    // whole turn off the window its seam copies bound (a rim walked against
    // its curve runs a turn below its start), and sections sit at their
    // points' own u, which need not be in that window either; parts a turn
    // apart never join. When the seam copies are one period apart, every
    // closed rim's run of pieces that sections end on, and every section,
    // moves by whole periods into the window between them.
    if u_periodic
        && matches!(surface, FaceSurface::Cylinder(_))
        && all_edges.len() > n_boundary_edges
        && let (Some(u_period), _) = super::pcurve_compute::surface_periods(&surface)
    {
        let weld = tol.linear * 100.0;
        let seam_us: Vec<f64> = all_edges[..n_boundary_edges]
            .iter()
            .filter(|e| {
                matches!(e.curve_3d, EdgeCurve::Line)
                    && (e.start_uv.x() - e.end_uv.x()).abs() <= 1e-9
            })
            .map(|e| e.start_uv.x())
            .collect();
        let lo = seam_us.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = seam_us.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let banded = seam_us.len() >= 2
            && ((hi - lo) - u_period).abs() <= 1e-6
            && seam_us
                .iter()
                .all(|&u| (u - lo).abs() <= 1e-6 || (u - hi).abs() <= 1e-6);
        if banded {
            let eps = 1e-6;
            let shift = |e: &mut OrientedPCurveEdge, by: f64| {
                e.start_uv = Point2::new(e.start_uv.x() + by, e.start_uv.y());
                e.end_uv = Point2::new(e.end_uv.x() + by, e.end_uv.y());
            };
            let same_rim =
                |a: &OrientedPCurveEdge, b: &OrientedPCurveEdge| match (&a.curve_3d, &b.curve_3d) {
                    (EdgeCurve::Circle(p), EdgeCurve::Circle(q)) => {
                        (p.center() - q.center()).length() <= weld
                            && (p.radius() - q.radius()).abs() <= weld
                            && (a.end_3d - b.start_3d).length() <= weld
                    }
                    _ => false,
                };
            let mut i = 0;
            while i < n_boundary_edges {
                if !matches!(all_edges[i].curve_3d, EdgeCurve::Circle(_)) {
                    i += 1;
                    continue;
                }
                let mut j = i;
                while j + 1 < n_boundary_edges && same_rim(&all_edges[j], &all_edges[j + 1]) {
                    j += 1;
                }
                let (run_lo, run_hi) = all_edges[i..=j]
                    .iter()
                    .flat_map(|e| [e.start_uv.x(), e.end_uv.x()])
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), u| {
                        (a.min(u), b.max(u))
                    });
                // Only a rim that sections end on: elsewhere the walk's own
                // layout of a whole rim is what the tracer reads.
                let joined = (i..j).any(|k| {
                    let joint = all_edges[k].end_3d;
                    all_edges[n_boundary_edges..].iter().any(|s| {
                        (s.start_3d - joint).length() <= weld || (s.end_3d - joint).length() <= weld
                    })
                });
                let turns = ((lo - run_lo) / u_period).round();
                let by = turns * u_period;
                if joined && turns != 0.0 && run_lo + by >= lo - eps && run_hi + by <= hi + eps {
                    for e in &mut all_edges[i..=j] {
                        shift(e, by);
                    }
                }
                i = j + 1;
            }
            for e in &mut all_edges[n_boundary_edges..] {
                let mid = f64::midpoint(e.start_uv.x(), e.end_uv.x());
                let by = -((mid - lo) / u_period).floor() * u_period;
                if by != 0.0 {
                    shift(e, by);
                }
            }
        }
    }

    // A section on a cylinder sits at its points' own u, which can be whole
    // turns off the boundary it ends on (an arc leaving u = 0 on a wall laid
    // out around 2pi). So can a section on a ball face charted clear of its
    // poles, whose wires run across the chart's seam. When an end meets a
    // boundary vertex in 3D but no copy of it in UV, move the section by the
    // one whole-turn shift that joins every such end.
    if u_periodic
        && (matches!(surface, FaceSurface::Cylinder(_))
            || ball_chart_clear && matches!(surface, FaceSurface::Sphere(_)))
        && all_edges.len() > n_boundary_edges
        && let (Some(u_period), _) = super::pcurve_compute::surface_periods(&surface)
    {
        let weld = tol.linear * 100.0;
        let corners: Vec<(Point3, Point2)> = all_edges[..n_boundary_edges]
            .iter()
            .flat_map(|e| [(e.start_3d, e.start_uv), (e.end_3d, e.end_uv)])
            .collect();
        // A ruling keeps the u its split gave it (a seam copy's own).
        for k in n_boundary_edges..all_edges.len() {
            if matches!(all_edges[k].curve_3d, EdgeCurve::Line) {
                continue;
            }
            let mut turns: Option<Vec<i64>> = None;
            for (p, uv) in [
                (all_edges[k].start_3d, all_edges[k].start_uv),
                (all_edges[k].end_3d, all_edges[k].end_uv),
            ] {
                let at: Vec<Point2> = corners
                    .iter()
                    .filter(|(q, _)| (*q - p).length() <= weld)
                    .map(|&(_, q)| q)
                    .collect();
                if at.is_empty() {
                    continue;
                }
                if at.iter().any(|q| (q.x() - uv.x()).abs() <= 1e-6) {
                    turns = Some(vec![0]);
                    break;
                }
                #[allow(clippy::cast_possible_truncation)]
                let mut here: Vec<i64> = at
                    .iter()
                    .filter_map(|q| {
                        let t = ((q.x() - uv.x()) / u_period).round();
                        ((q.x() - uv.x() - t * u_period).abs() <= 1e-6
                            && (q.y() - uv.y()).abs() <= 1e-6)
                            .then_some(t as i64)
                    })
                    .collect();
                here.sort_unstable();
                here.dedup();
                turns = Some(match turns {
                    None => here,
                    Some(prev) => prev.into_iter().filter(|t| here.contains(t)).collect(),
                });
            }
            if let Some(t) = turns
                && let [t] = t[..]
                && t != 0
            {
                #[allow(clippy::cast_precision_loss)]
                let by = t as f64 * u_period;
                move_edge_in_uv(&mut all_edges[k], by, 0.0);
            }
        }
    }

    // Full-band u window: the boundary walk lays a periodic face's rims and
    // seam lines out in one continuous u window, which need not be the
    // principal one; a section projected to its principal u would then sit
    // a period away from the rims it ends on. Move each section by whole
    // periods so its midpoint falls inside the window. A cylinder's boundary
    // walked from a closed rim spans exactly one turn, and so does a torus
    // band's outline run unbroken round the axis; their sections go in end
    // by end (`place_section_in_turn`): their own end UVs need not be one
    // continuous span. Traced glued at its seam, a band's rim piece over half
    // a turn long reads as running back the short way.
    let mut one_turn: Option<(f64, f64)> = None;
    if u_periodic
        && matches!(
            surface,
            FaceSurface::Cone(_) | FaceSurface::Cylinder(_) | FaceSurface::Torus(_)
        )
        && all_edges.len() > n_boundary_edges
        && let (Some(u_period), _) = super::pcurve_compute::surface_periods(&surface)
    {
        let (lo, hi) = all_edges[..n_boundary_edges]
            .iter()
            .flat_map(|e| [e.start_uv.x(), e.end_uv.x()])
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), u| {
                (lo.min(u), hi.max(u))
            });
        let boundary = &all_edges[..n_boundary_edges];
        let unbroken = boundary
            .iter()
            .zip(boundary.iter().cycle().skip(1))
            .all(|(a, b)| (b.start_uv.x() - a.end_uv.x()).abs() < 1e-9);
        // A frustum walked from its closed rims lies in one turn as a
        // cylinder does.
        let cylinder = matches!(surface, FaceSurface::Cylinder(_))
            || matches!(surface, FaceSurface::Cone(_)) && walked_from_closed_rim;
        if (cylinder && walked_from_closed_rim || torus_band && unbroken)
            && (hi - lo - u_period).abs() < 1e-6
        {
            for e in &mut all_edges[n_boundary_edges..] {
                place_section_in_turn(e, &surface, lo, u_period);
            }
            if matches!(surface, FaceSurface::Cylinder(_)) {
                place_seam_slivers(&mut all_edges[n_boundary_edges..], lo, u_period, tol.linear);
            }
            one_turn = Some((lo, u_period));
        } else if !cylinder
            // A torus face over half a turn about the axis or less that holds
            // the whole tube (a half ring) is read periodic in `u` too.
            && (hi - lo > 0.5 * u_period
                || matches!(surface, FaceSurface::Torus(_)) && hi - lo > 1e-6)
        {
            for e in &mut all_edges[n_boundary_edges..] {
                let mid = f64::midpoint(e.start_uv.x(), e.end_uv.x());
                let shift = -((mid - lo) / u_period).floor() * u_period;
                if shift != 0.0 {
                    e.start_uv = Point2::new(e.start_uv.x() + shift, e.start_uv.y());
                    e.end_uv = Point2::new(e.end_uv.x() + shift, e.end_uv.y());
                }
            }
        }
    }

    // A ball face that holds one pole of its chart winds the chart: its
    // outline runs one whole turn of u. A section that meets no vertex of
    // the outline keeps its points' own u, which can lie a turn off that
    // window, and its region then never closes. The region that holds the
    // pole runs that turn too. A zone between two latitudes spans a turn of
    // u as well, but holds no pole: its seam closes its regions.
    let mut round_a_pole = false;
    let mut pole_view: Option<pole_round::PoleView> = None;

    if u_periodic
        && ball_chart_clear
        && let FaceSurface::Sphere(chart) = &surface
        && original_inner_wires.is_empty()
        && all_edges.len() > n_boundary_edges
        && let (Some(u_period), _) = super::pcurve_compute::surface_periods(&surface)
        && let Some(north) = pole_held_round(topo, face_id, chart)
    {
        let (lo, hi) = all_edges[..n_boundary_edges]
            .iter()
            .flat_map(|e| [e.start_uv.x(), e.end_uv.x()])
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), u| {
                (lo.min(u), hi.max(u))
            });
        if (hi - lo - u_period).abs() < 1e-6 {
            for e in &mut all_edges[n_boundary_edges..] {
                place_section_in_turn(e, &surface, lo, u_period);
            }
            round_a_pole = true;
            pole_view = Some(pole_round::PoleView::new(chart, north));
        }
    }

    // Partial-band u unwrap: a face whose u-window touches the period seam
    // (e.g. a rounded-rect corner cylinder spanning [3pi/2, 2pi]) carries
    // mixed u anchors — surface projection returns u in [0, 2pi), so
    // endpoints exactly on the seam come back as 0 while their neighbours
    // sit near 2pi. Partial bands are treated as non-periodic (see
    // build_surface_info), so quantized junction keys would never match.
    // Remap every endpoint u into the continuous window that starts after
    // the largest angular gap.
    if !u_periodic
        && !is_plane
        && let (Some(u_period), _) = super::pcurve_compute::surface_periods(&surface)
    {
        // The boundary walk lays the face's outline out in one unbroken run
        // of u, which spans the face: the window starts at its low end. The
        // endpoints' widest gap cannot always say which side is the face: a
        // half band's vertices all sit on its two rulings half a turn apart,
        // its two gaps tie, and rounding can pick the side it does not cover.
        // A whole ring's seam placeholders all sit at one u and span nothing.
        let boundary = &all_edges[..n_boundary_edges];
        let unbroken = !boundary.is_empty()
            && boundary
                .iter()
                .zip(boundary.iter().cycle().skip(1))
                .all(|(a, b)| (b.start_uv.x() - a.end_uv.x()).abs() < 1e-9);
        let (walk_lo, walk_hi) = boundary
            .iter()
            .flat_map(|e| [e.start_uv.x(), e.end_uv.x()])
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), u| {
                (lo.min(u), hi.max(u))
            });
        let span = walk_hi - walk_lo;
        let walked = (unbroken && span > 1e-6 && span < u_period - 1e-6).then_some(walk_lo);
        let mut us: Vec<f64> = all_edges
            .iter()
            .flat_map(|e| [e.start_uv.x(), e.end_uv.x()])
            .map(|u| u.rem_euclid(u_period))
            .collect();
        us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if us.len() >= 2 {
            let mut gap_start = us[us.len() - 1];
            let mut max_gap = u_period - (us[us.len() - 1] - us[0]);
            for w in us.windows(2) {
                if w[1] - w[0] > max_gap {
                    max_gap = w[1] - w[0];
                    gap_start = w[0];
                }
            }
            if max_gap > 0.05 {
                let lo = walked.unwrap_or(gap_start + max_gap);
                for e in &mut all_edges {
                    let remap = |uv: Point2| -> Point2 {
                        let mut d = (uv.x() - lo).rem_euclid(u_period);
                        if d > u_period - 1e-6 {
                            d = 0.0;
                        }
                        Point2::new(lo + d, uv.y())
                    };
                    e.start_uv = remap(e.start_uv);
                    e.end_uv = remap(e.end_uv);
                }
            }
        }
    }

    // The same for a partial torus band's tube angle `v`.
    if !v_periodic
        && !is_plane
        && let (_, Some(v_period)) = super::pcurve_compute::surface_periods(&surface)
    {
        let mut vs: Vec<f64> = all_edges
            .iter()
            .flat_map(|e| [e.start_uv.y(), e.end_uv.y()])
            .map(|v| v.rem_euclid(v_period))
            .collect();
        vs.sort_by(f64::total_cmp);
        if vs.len() >= 2 {
            let mut gap_start = vs[vs.len() - 1];
            let mut max_gap = v_period - (vs[vs.len() - 1] - vs[0]);
            for w in vs.windows(2) {
                if w[1] - w[0] > max_gap {
                    max_gap = w[1] - w[0];
                    gap_start = w[0];
                }
            }
            if max_gap > 0.05 {
                // A band round the axis over more than half the tube has only
                // its two rims' tube angles for ends, and the wider gap
                // between them runs across the face: its outline says where
                // the face starts.
                let lo =
                    super::super::pave_filler::phase_ff::face_tube_window(topo, face_id, &surface)
                        .map_or(gap_start + max_gap, |(start, _)| start);
                for e in &mut all_edges {
                    let remap = |uv: Point2| -> Point2 {
                        let mut d = (uv.y() - lo).rem_euclid(v_period);
                        if d > v_period - 1e-6 {
                            d = 0.0;
                        }
                        Point2::new(uv.x(), lo + d)
                    };
                    e.start_uv = remap(e.start_uv);
                    e.end_uv = remap(e.end_uv);
                }
            }
        }
    }

    // Co-register section endpoints with the boundary graph on curved
    // faces. A marched chain end and the boundary split vertex at the same
    // junction share ONE 3D vertex (the pave machinery welded them), but
    // their UVs come from different paths — surface projection vs boundary
    // pcurve — and disagree at marched precision (~1e-6), far above the
    // wire graph's exact-tol weld. Left apart, the pendant pruner reads
    // every chain end as dangling and eats whole section chains (the wedge
    // cylinder kept its in-strut corner span for exactly this reason).
    // Junction identity is the 3D point, so adopt the boundary's UV
    // wherever the 3D positions coincide within the weld-scale band.
    // Non-periodic faces only: a periodic face's seam carries TWO boundary
    // vertices at the same 3D position whose UVs differ by a full period,
    // and nearest-3D adoption could pick the wrong copy.
    if !is_plane && !u_periodic && !v_periodic && all_edges.len() > n_boundary_edges {
        let weld3_sq = (tol.linear * 100.0) * (tol.linear * 100.0);
        let boundary_uvs: Vec<(Point3, Point2)> = all_edges[..n_boundary_edges]
            .iter()
            .flat_map(|e| [(e.start_3d, e.start_uv), (e.end_3d, e.end_uv)])
            .collect();
        let adopt = |p3: Point3| -> Option<Point2> {
            let mut best: Option<(f64, Point2)> = None;
            for &(q3, quv) in &boundary_uvs {
                let d = (q3 - p3).length_squared();
                if d <= weld3_sq && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, quv));
                }
            }
            best.map(|(_, uv)| uv)
        };
        for e in &mut all_edges[n_boundary_edges..] {
            if let Some(uv) = adopt(e.start_3d) {
                e.start_uv = uv;
            }
            if let Some(uv) = adopt(e.end_3d) {
                e.end_uv = uv;
            }
        }
    }

    // Split section edges where another section's endpoint lands on their
    // interior (L/T junctions). Needs ≥ 2 distinct sections (one pair to cross
    // another); runs after the partial-band u-unwrap so split UVs match the
    // face's continuous window.
    if all_edges.len() > n_boundary_edges + 2 {
        // Two sections may also meet inside both, where no end of either
        // lies (a circle and an ellipse touching at one point).
        // A pair whose meetings cannot be settled leaves the face unsplit,
        // which the builder refuses: a guess could split it wrongly.
        let meetings: Vec<Point3> = if is_plane {
            let spans = edge_splitting::section_spans(&all_edges[n_boundary_edges..]);
            let Ok(found) = edge_splitting::section_meeting_points(&spans, tol.linear) else {
                log::debug!("split_face_2d: face {face_id:?} has sections that may meet unproved");
                return Vec::new();
            };
            found
                .into_iter()
                .map(|m| {
                    log::debug!(
                        "split_face_2d: face {face_id:?}: sections meet at {:?}, witnesses {:e} apart",
                        m.at,
                        m.apart
                    );
                    m.at
                })
                .collect()
        } else {
            Vec::new()
        };
        split_sections_at_t_junctions(
            &mut all_edges,
            n_boundary_edges,
            &surface,
            if is_plane { Some(frame) } else { None },
            &wire_pts,
            tol.linear,
            split_registry.as_deref_mut(),
            &meetings,
        );
    }

    // Interior-junction UV co-registration, the T-split counterpart of the
    // boundary pass above: a T-split mints its vertex UV by projecting the
    // junction endpoint, while the CROSSING section's own endpoint UV came
    // from an independent projection of a 3D point that differs by marched
    // rounding (~1e-6) — above the wire graph's weld, so the pendant pruner
    // reads the crossing section as dangling and eats it (the wedge-corner
    // junction: the cylinder section pruned, two loops instead of three).
    // Canonicalize section endpoint UVs by 3D identity, seeding with the
    // boundary vertices so the boundary pass's welds are preserved.
    if !is_plane && !u_periodic && !v_periodic && all_edges.len() > n_boundary_edges {
        let weld3_sq = (tol.linear * 100.0) * (tol.linear * 100.0);
        let mut canon: Vec<(Point3, Point2)> = all_edges[..n_boundary_edges]
            .iter()
            .flat_map(|e| [(e.start_3d, e.start_uv), (e.end_3d, e.end_uv)])
            .collect();
        for e in &mut all_edges[n_boundary_edges..] {
            for pick_end in [false, true] {
                let (p3, uv) = if pick_end {
                    (e.end_3d, e.end_uv)
                } else {
                    (e.start_3d, e.start_uv)
                };
                let adopted = canon
                    .iter()
                    .filter(|(q3, _)| (*q3 - p3).length_squared() <= weld3_sq)
                    .min_by(|(a3, _), (b3, _)| {
                        (*a3 - p3)
                            .length_squared()
                            .total_cmp(&(*b3 - p3).length_squared())
                    })
                    .map(|&(_, quv)| quv);
                let new_uv = adopted.unwrap_or_else(|| {
                    canon.push((p3, uv));
                    uv
                });
                if pick_end {
                    e.end_uv = new_uv;
                } else {
                    e.start_uv = new_uv;
                }
            }
        }
    }

    // Split BOUNDARY arc edges where a section endpoint lands strictly on their
    // interior. A section clipped out to a convex rounded corner's TRUE arc (a
    // notch corner straddling a wall's top edge) ends MID-ARC on the boundary;
    // without splitting the arc there, the wire builder can't route through the
    // junction and the section is pruned as a pendant. Plane faces only: their
    // frame projection is periodicity-free, so the split UV is unambiguous (the
    // periodic-cylinder boundary is handled by the section-side T-junction split
    // above).
    let mut n_boundary_edges = n_boundary_edges;
    if is_plane && all_edges.len() > n_boundary_edges {
        // Dedup section endpoints with a grid (cell = tol) so a cap with many
        // hole edges costs O(sections) here, not O(sections²). Probing the 3×3×3
        // neighbourhood reproduces the former within-`tol` linear scan exactly.
        let cell = tol.linear.max(f64::MIN_POSITIVE);
        let inv = 1.0 / cell;
        let cell_of = |p: Point3| -> (i64, i64, i64) {
            #[allow(clippy::cast_possible_truncation)]
            (
                (p.x() * inv).floor() as i64,
                (p.y() * inv).floor() as i64,
                (p.z() * inv).floor() as i64,
            )
        };
        let mut ep_grid: std::collections::HashMap<(i64, i64, i64), Vec<Point3>> =
            std::collections::HashMap::new();
        let mut section_endpoints: Vec<Point3> = Vec::new();
        for e in &all_edges[n_boundary_edges..] {
            for p in [e.start_3d, e.end_3d] {
                let (cx, cy, cz) = cell_of(p);
                let dup = (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| {
                        (-1..=1).any(|dz| {
                            ep_grid
                                .get(&(cx + dx, cy + dy, cz + dz))
                                .is_some_and(|pts| {
                                    pts.iter().any(|q| (*q - p).length() < tol.linear)
                                })
                        })
                    })
                });
                if !dup {
                    section_endpoints.push(p);
                    ep_grid.entry((cx, cy, cz)).or_default().push(p);
                }
            }
        }
        let boundary: Vec<OrientedPCurveEdge> = all_edges[..n_boundary_edges].to_vec();
        let split_boundary = split_plane_boundary_arcs_at_points(
            boundary,
            &section_endpoints,
            &surface,
            frame,
            tol.linear,
        );
        if split_boundary.len() != n_boundary_edges {
            let sections_tail: Vec<OrientedPCurveEdge> = all_edges[n_boundary_edges..].to_vec();
            n_boundary_edges = split_boundary.len();
            all_edges.clear();
            all_edges.extend(split_boundary);
            all_edges.extend(sections_tail);
        }
        // A boundary arc piece running between the two ends of a straight
        // section bounds a region with that chord alone, and the assembler's
        // endpoint-keyed edge merge would weld the two (a keyhole's rim below
        // its tail's base line): give the arc a vertex at its middle.
        if let Some(boundary) = split_arcs_shadowing_chords(
            &all_edges[..n_boundary_edges],
            &all_edges[n_boundary_edges..],
            &surface,
            frame,
            tol.linear,
        ) {
            let sections_tail: Vec<OrientedPCurveEdge> = all_edges[n_boundary_edges..].to_vec();
            n_boundary_edges = boundary.len();
            all_edges.clear();
            all_edges.extend(boundary);
            all_edges.extend(sections_tail);
        }
    }

    // Snap section UV endpoints onto 3D-coincident boundary UV copies. The
    // wire graph keys junctions on UV quantized at the exact tolerance, but a
    // section endpoint's UV comes from its own pcurve evaluation while the
    // boundary split's UV comes from span interpolation — the same exact 3D
    // junction can carry UV copies ~1e-6 apart (the fit-error class), which
    // lands them in different graph cells and the section reads as pendant.
    // Adopting the boundary's UV copy is a sub-weld no-op in 3D.
    // On a ball face round a pole, a section can end a whole turn from the
    // boundary's copy of its end: the one-turn window holds the outline and
    // the section's middle. The section keeps its own copy of the end,
    // moved onto the boundary's.
    {
        let turn = if round_a_pole {
            super::pcurve_compute::surface_periods(&surface).0
        } else {
            None
        };
        let near_copy = |auv: Point2, uv: Point2| {
            let du = auv.x() - uv.x();
            let du = turn.map_or(du, |p| du - (du / p).round() * p);
            Point2::new(uv.x() + du, auv.y())
        };
        let (bnd, secs) = all_edges.split_at_mut(n_boundary_edges);
        let mut anchors: Vec<(Point3, Point2)> = bnd
            .iter()
            .flat_map(|e| [(e.start_3d, e.start_uv), (e.end_3d, e.end_uv)])
            .collect();
        for e in secs {
            for pick_start in [true, false] {
                let (p3, uv) = if pick_start {
                    (e.start_3d, e.start_uv)
                } else {
                    (e.end_3d, e.end_uv)
                };
                let snapped = anchors
                    .iter()
                    .map(|&(a3, auv)| (a3, near_copy(auv, uv)))
                    .filter(|(a3, auv)| {
                        (*a3 - p3).length() < tol.linear && (*auv - uv).length() < 1e-2
                    })
                    .min_by(|(_, a), (_, b)| (*a - uv).length().total_cmp(&(*b - uv).length()))
                    .map(|(_, auv)| auv);
                let final_uv = snapped.unwrap_or(uv);
                if pick_start {
                    e.start_uv = final_uv;
                } else {
                    e.end_uv = final_uv;
                }
                anchors.push((p3, final_uv));
            }
        }
    }

    // Bridge a pendant section end to a nearby boundary vertex BEFORE the
    // pendant drop. Independently built operands disagree at their shared
    // outlines by up to ~2.4e-3 (measured, kumiko z~9.9 rim): the pair-level
    // mutual-overlap clip then legitimately ends a section where the line
    // leaves the PARTNER face, a fraction short of where this face's own
    // boundary carries the exit pave (now visible in the boundary via the
    // image expansion). Bridging the pendant end to its unambiguous nearest
    // boundary vertex completes the partition with the micro-facet edge the
    // true result needs. Genuine features sit >= 1.1e-2 apart, so the band
    // cannot conflate distinct corners.
    let mut bridge_sections: Vec<super::split_types::SectionEdge> = Vec::new();
    // Bridging is gated to HOLE-FREE planar faces: on a holed cap the
    // integrate-holes weave owns chain-end reconciliation, and a bridge
    // minted next to a woven hole boundary de-analytics the divider-lip
    // fuse (its production fixture drops from >=32 curved faces to 0).
    if is_plane && original_inner_wires.is_empty() {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::vec::Vec2;
        const BRIDGE_BAND: f64 = 3e-3;
        let weld = tol.linear * 100.0;
        let n_all = all_edges.len();
        let mut bridges: Vec<OrientedPCurveEdge> = Vec::new();
        let mut pendants: Vec<(Point3, Point2, Point3)> = Vec::new();
        for si in n_boundary_edges..n_all {
            for pick_start in [true, false] {
                let (p3, puv) = if pick_start {
                    (all_edges[si].start_3d, all_edges[si].start_uv)
                } else {
                    (all_edges[si].end_3d, all_edges[si].end_uv)
                };
                // Pendant: coincides with no other edge endpoint (boundary or
                // section) within the weld band. Sections arrive as
                // forward/reverse twin pairs, so an edge with the same
                // undirected endpoint set is the twin, not a continuation.
                let (s_si, e_si) = (all_edges[si].start_3d, all_edges[si].end_3d);
                let shared = all_edges.iter().enumerate().any(|(oi, oe)| {
                    if oi == si {
                        return false;
                    }
                    let twin = ((oe.start_3d - s_si).length() <= weld
                        && (oe.end_3d - e_si).length() <= weld)
                        || ((oe.start_3d - e_si).length() <= weld
                            && (oe.end_3d - s_si).length() <= weld);
                    !twin
                        && ((oe.start_3d - p3).length() <= weld
                            || (oe.end_3d - p3).length() <= weld)
                });
                if shared {
                    continue;
                }
                // The pendant edge's own other endpoint is not a bridge
                // target: bridging back onto it would duplicate the pendant
                // edge itself.
                let own_other = if pick_start {
                    all_edges[si].end_3d
                } else {
                    all_edges[si].start_3d
                };
                let mut best: Option<(f64, Point3, Point2)> = None;
                let mut second = f64::INFINITY;
                for be in &all_edges[..n_boundary_edges] {
                    for (b3, buv) in [(be.start_3d, be.start_uv), (be.end_3d, be.end_uv)] {
                        if (b3 - own_other).length() <= weld {
                            continue;
                        }
                        let d = (b3 - p3).length();
                        match &best {
                            Some((bd, bp, _)) => {
                                if d < *bd {
                                    if (*bp - b3).length() > weld {
                                        second = *bd;
                                    }
                                    best = Some((d, b3, buv));
                                } else if (*bp - b3).length() > weld {
                                    second = second.min(d);
                                }
                            }
                            None => best = Some((d, b3, buv)),
                        }
                    }
                }
                // A target vertex some SECTION already reaches is part of
                // the section network; the pendant connects to it through
                // that network and a direct bridge would over-connect the
                // corner (edge use-3 at the z=4.5 corner). Bridge only onto
                // section-free boundary vertices — the exit paves nothing
                // else reaches.
                let target_in_sections = |b3: Point3| -> bool {
                    all_edges[n_boundary_edges..].iter().any(|se| {
                        (se.start_3d - b3).length() <= weld || (se.end_3d - b3).length() <= weld
                    })
                };
                let boundary_bridge = best.is_some_and(|(d, b3, _)| {
                    d > weld && d <= BRIDGE_BAND && second >= 1e-2 && !target_in_sections(b3)
                });
                if !boundary_bridge {
                    pendants.push((p3, puv, own_other));
                }
                if let Some((_d, b3, buv)) = best
                    && boundary_bridge
                    && !bridges.iter().any(|br: &OrientedPCurveEdge| {
                        ((br.start_3d - p3).length() <= weld && (br.end_3d - b3).length() <= weld)
                            || ((br.start_3d - b3).length() <= weld
                                && (br.end_3d - p3).length() <= weld)
                    })
                {
                    use brepkit_math::curves2d::{Curve2D, Line2D};
                    use brepkit_math::vec::Vec2;
                    let duv = Vec2::new(buv.x() - puv.x(), buv.y() - puv.y());
                    let len = duv.length();
                    let dir = if len > 1e-12 {
                        Vec2::new(duv.x() / len, duv.y() / len)
                    } else {
                        Vec2::new(1.0, 0.0)
                    };
                    let Ok(l2d) = Line2D::new(puv, dir) else {
                        continue;
                    };
                    bridges.push(OrientedPCurveEdge {
                        curve_3d: EdgeCurve::Line,
                        pcurve: Curve2D::Line(l2d),
                        start_uv: puv,
                        end_uv: buv,
                        start_3d: p3,
                        end_3d: b3,
                        forward: true,
                        source_edge_idx: None,
                        pave_block_id: None,
                    });
                }
            }
        }
        // Pendant-to-pendant bridging: a chain gap between TWO section ends
        // (the triple-point wedge on the kumiko slope, where the pair
        // sections stop 2.3e-3 apart at the operands' disagreement) has no
        // boundary vertex to target — the missing connector runs between
        // the pendants themselves. Bridge mutually-nearest pendant pairs
        // within the band, with the same isolation guard against every
        // other pendant.
        // Each section arrives as a forward/reverse twin pair, so a pendant
        // endpoint is recorded once per twin — dedupe by position or the
        // zero-distance duplicate defeats the mutual-nearest pairing.
        pendants.dedup_by(|a, b| (a.0 - b.0).length() <= weld);
        for i in 0..pendants.len() {
            let (pi, uvi, own_i) = pendants[i];
            let mut best_j: Option<(f64, usize)> = None;
            let mut second = f64::INFINITY;
            for (j, &(pj, _, _)) in pendants.iter().enumerate() {
                if j == i {
                    continue;
                }
                let d = (pj - pi).length();
                match best_j {
                    Some((bd, _)) if d < bd => {
                        second = bd;
                        best_j = Some((d, j));
                    }
                    Some(_) => second = second.min(d),
                    None => best_j = Some((d, j)),
                }
            }
            let Some((d, j)) = best_j else { continue };
            let (pj, uvj, own_j) = pendants[j];
            if d <= weld || d > BRIDGE_BAND || second < 1e-2 {
                continue;
            }
            let mutual = pendants
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|(_, &(pk, _, _))| (pk - pj).length())
                .fold(f64::INFINITY, f64::min)
                >= d - 1e-12;
            if !mutual
                || (pj - own_i).length() <= weld
                || (pi - own_j).length() <= weld
                || bridges.iter().any(|br: &OrientedPCurveEdge| {
                    ((br.start_3d - pi).length() <= weld && (br.end_3d - pj).length() <= weld)
                        || ((br.start_3d - pj).length() <= weld
                            && (br.end_3d - pi).length() <= weld)
                })
            {
                continue;
            }
            let duv = Vec2::new(uvj.x() - uvi.x(), uvj.y() - uvi.y());
            let len = duv.length();
            let dir = if len > 1e-12 {
                Vec2::new(duv.x() / len, duv.y() / len)
            } else {
                Vec2::new(1.0, 0.0)
            };
            let Ok(l2d) = Line2D::new(uvi, dir) else {
                continue;
            };
            bridges.push(OrientedPCurveEdge {
                curve_3d: EdgeCurve::Line,
                pcurve: Curve2D::Line(l2d),
                start_uv: uvi,
                end_uv: uvj,
                start_3d: pi,
                end_3d: pj,
                forward: true,
                source_edge_idx: None,
                pave_block_id: None,
            });
        }
        if !bridges.is_empty() {
            bridge_sections = bridges
                .iter()
                .map(|br| super::split_types::SectionEdge {
                    curve_3d: EdgeCurve::Line,
                    pcurve_a: br.pcurve.clone(),
                    pcurve_b: br.pcurve.clone(),
                    start: br.start_3d,
                    end: br.end_3d,
                    start_uv_a: Some(br.start_uv),
                    end_uv_a: Some(br.end_uv),
                    start_uv_b: Some(br.start_uv),
                    end_uv_b: Some(br.end_uv),
                    target_face: None,
                    pave_block_id: None,
                })
                .collect();
        }
        all_edges.extend(bridges);
    }
    // The arrangement path reads `sections`, not `all_edges` — augment it so
    // a bridged partition survives arrangement adoption.
    let sections_aug: Vec<super::split_types::SectionEdge>;
    let sections: &[super::split_types::SectionEdge] = if bridge_sections.is_empty() {
        sections
    } else {
        sections_aug = sections.iter().cloned().chain(bridge_sections).collect();
        &sections_aug
    };

    // Drop pendant section edges that dangle into the face interior — left
    // in, the traversal walks out and back along them, spuriously
    // over-splitting the face (boundary edges are never removed, so the
    // boundary prefix and `n_boundary_edges` stay valid).

    if trace_split {
        let mut rows: Vec<String> = all_edges
            .iter()
            .map(|e| {
                format!(
                    "STRACE-PRE ({:.7},{:.7})->({:.7},{:.7}) src={:?} fwd={} {} 3d=({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3})",
                    e.start_uv.x(),
                    e.start_uv.y(),
                    e.end_uv.x(),
                    e.end_uv.y(),
                    e.source_edge_idx,
                    e.forward,
                    e.curve_3d.type_tag(),
                    e.start_3d.x(),
                    e.start_3d.y(),
                    e.start_3d.z(),
                    e.end_3d.x(),
                    e.end_3d.y(),
                    e.end_3d.z()
                )
            })
            .collect();
        rows.sort();
        for r in rows {
            log::debug!("{r}");
        }
        for e in &all_edges {
            if let brepkit_math::curves2d::Curve2D::Nurbs(n) = &e.pcurve {
                let k = n.knots();
                log::debug!(
                    "PCDBG {:?}->{:?} nurbs a={:?} b={:?} mid={:?}",
                    e.start_uv,
                    e.end_uv,
                    n.evaluate(k[0]),
                    n.evaluate(k[k.len() - 1]),
                    n.evaluate(0.5 * (k[0] + k[k.len() - 1]))
                );
            } else if !matches!(e.curve_3d, EdgeCurve::Line) {
                log::debug!("PCDBG {:?}->{:?} other pcurve", e.start_uv, e.end_uv);
            }
        }
    }
    // A curved face's hole that a section ends on joins the arrangement: the
    // section bridges the face to the opening (a strut's footprint on a bore,
    // closed by the bottom of a window cut through the bore). Attached whole
    // after the split, the hole would leave the section's end dangling, and
    // the pendant pass would drop it.
    let mut woven_curved_holes: Vec<usize> = Vec::new();
    if !is_plane && !original_inner_wires.is_empty() && all_edges.len() > n_boundary_edges {
        let ends: Vec<(Point3, Point2)> = all_edges[n_boundary_edges..]
            .iter()
            .flat_map(|e| [(e.start_3d, e.start_uv), (e.end_3d, e.end_uv)])
            .collect();
        for (hi, hole) in original_inner_wires.iter().enumerate() {
            if let Some(pieces) = weave_curved_hole(hole, &ends, &surface, tol.linear) {
                all_edges.extend(pieces);
                woven_curved_holes.push(hi);
            }
        }
    }
    // Whether a section piece off a flat face bounds something: its ends lie
    // apart; its chart image spans 0.01 (the scale of the boundary-proximity
    // `uv_tol` above, ~0.6 deg in angular coordinates), as a closed circle
    // round the chart's axis does; or, read about a pole, its curve is not
    // proved to stay within a weld of its start (a closed circle round no
    // pole spans neither).
    let has_extent = |e: &OrientedPCurveEdge| {
        const ZERO_EXTENT_UV: f64 = 0.01;
        (e.start_3d - e.end_3d).length() >= tol.linear
            || (e.start_uv - e.end_uv).length() >= ZERO_EXTENT_UV
            || (pole_view.is_some() && super::wire_builder::runs_off(e, 100.0 * tol.linear))
    };
    // What the face received, read about a pole: each half must run once in
    // its pieces, whatever the pruning below drops. A piece with no extent
    // (a self-split remnant) bounds nothing.
    let received: Option<Vec<OrientedPCurveEdge>> = pole_view.as_ref().map(|_| {
        all_edges
            .iter()
            .enumerate()
            .filter(|(i, e)| *i < n_boundary_edges || has_extent(e))
            .map(|(_, e)| e.clone())
            .collect()
    });
    let all_edges = if pole_view.is_some() {
        super::wire_builder::remove_pendant_sections_but_loops(
            &all_edges, tol.linear, u_periodic, v_periodic,
        )
    } else {
        super::wire_builder::remove_pendant_sections(&all_edges, tol.linear, u_periodic, v_periodic)
    };

    // Drop zero-extent section edges (a T-junction split at a section's own
    // endpoint mints them); a self-loop edge derails the angular walker into
    // degenerate single-edge sub-faces.
    let all_edges: Vec<OrientedPCurveEdge> = if is_plane {
        // Plane frames are isometric, so a sub-weld 3D chord IS a degenerate
        // piece (a self-split remnant ~fit-error long); real sliver edges
        // (corner lenses) are orders of magnitude longer.
        let n_b = n_boundary_edges;
        let weld = tol.linear * 100.0;
        all_edges
            .into_iter()
            .enumerate()
            .filter(|(i, e)| *i < n_b || (e.start_3d - e.end_3d).length() >= weld)
            .map(|(_, e)| e)
            .collect()
    } else {
        let n_b = n_boundary_edges;
        all_edges
            .into_iter()
            .enumerate()
            .filter(|(i, e)| *i < n_b || has_extent(e))
            .map(|(_, e)| e)
            .collect()
    };

    // Drop boundary-collinear section edges on plane faces: a section whose
    // whole UV image lies ON the boundary polyline cannot partition the face
    // interior (its endpoints already split the boundary edges), but its twin
    // pair double-covers the boundary segment and derails the angular walker
    // (the pad bottom-cap chord riding the inset wall's bottom edge left the
    // wall unsplit at both true crossings).
    let all_edges: Vec<OrientedPCurveEdge> = if is_plane && !u_periodic && !v_periodic {
        // Via-frame sampling: stored pcurves fold on reversed boundary arcs
        // (two orientation conventions coexist), which would corrupt the
        // polygon and mis-measure every distance below.
        let boundary_poly = sample_wire_loop_uv_via_frame(&all_edges[..n_boundary_edges], frame);
        let eps = tol.linear * 10.0;
        let n_b = n_boundary_edges;
        all_edges
            .into_iter()
            .enumerate()
            .filter(|(i, e)| {
                if *i < n_b {
                    return true;
                }
                // Only straight (Line-pcurve) sections can be fully
                // boundary-collinear without interior evidence; arcs bulge
                // and are kept. Probe nine points along the chord — a
                // splitter across a concave region keeps interior evidence
                // at some interior fraction.
                if !matches!(e.pcurve, brepkit_math::curves2d::Curve2D::Line(_)) {
                    return true;
                }
                let (s, t) = (e.start_uv, e.end_uv);
                !(0..=8).all(|k| {
                    let f = f64::from(k) / 8.0;
                    let p = Point2::new(f.madd(t.x() - s.x(), s.x()), f.madd(t.y() - s.y(), s.y()));
                    super::classify_2d::distance_to_polygon_boundary(p, &boundary_poly) <= eps
                })
            })
            .map(|(_, e)| e)
            .collect()
    } else {
        all_edges
    };

    // Laid out in one turn, the face is a strip cut open along its seam and
    // is traced as one: glued at the seam, a vertex there offers the edges
    // leaving both of its copies, and the walk can cross to the far copy (an
    // ellipse closing at the seam's end, where it touches the rim). A sliver
    // run from one seam copy to the other (`place_seam_slivers`) is a step
    // across the seam, not a chord across the strip: such a face is traced
    // periodic.
    let strip = one_turn.is_some_and(|(lo, period)| {
        all_edges.iter().all(|e| {
            let (a, b) = (e.start_uv.x(), e.end_uv.x());
            a.min(b) >= lo - 1e-6
                && a.max(b) <= lo + period + 1e-6
                && ((b - a).abs() < 0.5 * period || !is_sliver(e, 100.0 * tol.linear))
        })
    });
    // Build wire loops via angular-sorting traversal.
    let mut loops = if round_a_pole {
        super::wire_builder::build_wire_loops_round_a_pole(&all_edges, tol.linear)
    } else {
        build_wire_loops(&all_edges, tol.linear, u_periodic && !strip, v_periodic)
    };
    // A woven hole's edges run one way only, and at a corner of the hole
    // where a section leaves on the material's side (a strut's footprint
    // under a window), the greedy walk turns into the hole's next side. The
    // face trace of the strip reads each region whole, and an orbit that
    // runs a rim or hole edge backward lies outside the face.
    let mut woven_trace = false;
    if !woven_curved_holes.is_empty() && (strip || !u_periodic) && !v_periodic {
        let rim_ccw = signed_area_2d(&sample_wire_loop_uv(&all_edges[..n_boundary_edges])) > 0.0;
        // Read along the curves: a region of one or two curved edges has
        // no area between its vertices alone.
        let curve_area = |lp: &[OrientedPCurveEdge]| signed_area_2d(&sample_wire_loop_uv(lp));
        let dcel = super::wire_builder::build_wire_loops_dcel_measured(
            &all_edges,
            tol.linear,
            false,
            false,
            Some(&curve_area),
            Some(rim_ccw),
        );
        if !dcel.is_empty()
            && !wire_loops_have_degenerate_area(&dcel, tol.linear)
            && !wire_loops_self_cross(&dcel, tol.linear)
        {
            loops = dcel;
            woven_trace = true;
        }
    }
    if trace_split {
        let mut rows: Vec<String> = all_edges
            .iter()
            .map(|e| {
                format!(
                    "STRACE-IN ({:.7},{:.7})->({:.7},{:.7}) src={:?}",
                    e.start_uv.x(),
                    e.start_uv.y(),
                    e.end_uv.x(),
                    e.end_uv.y(),
                    e.source_edge_idx
                )
            })
            .collect();
        rows.sort();
        for r in rows {
            log::debug!("{r}");
        }
        let mut lrows: Vec<String> = loops
            .iter()
            .map(|lp| {
                let mut vs: Vec<String> = lp
                    .iter()
                    .map(|e| format!("({:.7},{:.7})", e.start_uv.x(), e.start_uv.y()))
                    .collect();
                vs.sort();
                let walk: Vec<String> = lp
                    .iter()
                    .map(|e| {
                        format!(
                            "({:.3},{:.3}){}{}",
                            e.start_uv.x(),
                            e.start_uv.y(),
                            if e.forward { "+" } else { "-" },
                            match e.curve_3d {
                                EdgeCurve::Line => "l",
                                _ => "c",
                            }
                        )
                    })
                    .collect();
                format!(
                    "STRACE-LOOP n={} {} walk={}",
                    lp.len(),
                    vs.join(""),
                    walk.join(">")
                )
            })
            .collect();
        lrows.sort();
        for r in lrows {
            log::debug!("{r}");
        }
    }
    // Clockwise-boundary handling: this face's UV frame derives from the raw
    // surface normal, not the effective face orientation, so an inner-shell
    // (cavity) wall winds CW in UV while the outer wall winds CCW. Two effects
    // follow when the boundary is CW, and both must be corrected:
    //   1. Every sub-loop comes out with negated signed area, so the
    //      area-based outer/hole split below would call every band a hole.
    //      `cw_loops` flips the sign back during classification.
    //   2. The min-clockwise turn rule can also merge everything into a single
    //      loop; when that under-split happens, retry with the mirrored rule.
    // Detect the CW boundary once and set `cw_loops` regardless of whether the
    // default traversal already split correctly — otherwise a correctly-split
    // CW face (e.g. a rounded-rect cavity corner cut by a constant-z section)
    // has all its bands misclassified as holes and collapses to one sub-face.
    let mut cw_loops = false;
    if all_edges.len() > n_boundary_edges && !u_periodic && !v_periodic {
        // Same arc-true sampling as the loop classification below: the
        // pcurve sampler folds reversed boundary arcs and called a barrel's
        // end disc clockwise, so both of its regions were emitted inside out.
        let boundary_pts = if is_plane {
            sampling::sample_wire_loop_uv_via_frame(&all_edges[..n_boundary_edges], frame)
        } else {
            sample_wire_loop_uv(&all_edges[..n_boundary_edges])
        };
        if signed_area_2d(&boundary_pts) < 0.0 {
            cw_loops = true;
            if loops.len() <= 1 {
                let retry = build_wire_loops_with_winding(
                    &all_edges, tol.linear, u_periodic, v_periodic, true,
                );
                if retry.len() > loops.len() {
                    loops = retry;
                }
            }
        }
    }

    // Holes-integrated planar arrangement. When `integrate_holes_plane` wove the
    // hole boundaries into `all_edges`, that list is a complete planar
    // subdivision (boundary + trimmed sections + split hole edges). With TWO OR
    // MORE original holes the cut can bridge across the material between them —
    // the divider-lip fuse onto a compartmented body, where the lip footprint
    // cuts each divider arm between the compartment openings — and the angular
    // wire builder fragments that into wrong loops (the under-lip ring and the
    // exposed divider cross get wound together). The even-odd arrangement
    // decomposition resolves the nesting correctly there. A SINGLE-hole holed
    // cap (the shelled wall-cutout rim, one cavity opening) is already
    // partitioned cleanly by the wire builder and the arrangement's nesting can
    // pick a worse decomposition, so restrict this path to the multi-hole case.
    // Second entry condition: sections cross the OUTER boundary's bay arcs
    // (pocket rings absorbed into the outer wire by earlier cuts) but touch no
    // remaining inner ring — the last groove of a fit-offset export, whose two
    // mouth cells are both bays. Nothing integrates, so the multi-hole gate
    // above never fires and the angular wire builder mis-traces the bay mouths
    // (phantom-corner slivers). The pre-split boundary + sections are already a
    // complete arrangement; the untouched rings all attach whole afterward.
    // Mirrors the >=2-hole restriction of the integrated branch: a SINGLE-hole
    // cap (the d-series shelled-cup lip fuse) is partitioned correctly by the
    // calibrated wire-builder path, and the arrangement picks a worse
    // decomposition there.
    // The boundary-arc×section crossings are applied HERE, not globally: the
    // arrangement needs the bay arcs pre-split at the crossings (each arc
    // piece is then uncrossed and its endpoints register as T-junctions on the
    // section lines), while the angular wire-builder paths below are
    // calibrated to the unsplit boundary.
    let arr_edges: Vec<OrientedPCurveEdge>;
    let arr_input: &[OrientedPCurveEdge] = if is_plane && boundary_arc_crossed {
        let (bnd, rest) = all_edges.split_at(n_boundary_edges.min(all_edges.len()));
        let split_bnd = split_boundary_edges_at_3d_points(
            bnd.to_vec(),
            &boundary_cross_pts,
            Some(frame),
            &surface,
            tol.linear,
        );
        arr_edges = split_bnd.into_iter().chain(rest.iter().cloned()).collect();
        &arr_edges
    } else {
        &all_edges
    };
    // The arrangement reads each edge by its chord, so a boundary arc whose
    // chord crosses a section (a disc's long rim arc beyond a box's corner,
    // its chord passing under the box's top) bounds regions it cannot see;
    // such a face is left to the wire builder.
    let bay_mouth_arrangement = !holes_integrated
        && boundary_arc_crossed
        && original_inner_wires.len() >= 2
        && !arc_chords_cross_sections(arr_input);
    // Third entry condition: the angular walker demonstrably WOVE the greedy
    // loops — some loop contains an edge immediately followed by its exact
    // reverse (an out-and-back spur, the label-tab corner-crescent weave where
    // both twins of a salvaged corner chord ride one loop instead of closing
    // the crescent). A clean partition never produces that signature, so this
    // trigger cannot demote a correctly-split single-hole cap; it only engages
    // the arrangement where the calibrated wire-builder path has already
    // failed.
    let woven_spur = is_plane
        && holes_integrated
        && !woven_inner_wires.is_empty()
        && (loops_have_out_and_back(&loops, tol.linear, holes_promoted)
            || loops_run_a_section_both_ways(&loops, tol.linear));
    if is_plane
        && ((holes_integrated && original_inner_wires.len() >= 2 && !woven_inner_wires.is_empty())
            || bay_mouth_arrangement
            || woven_spur)
        && let Some(mut result) = arrangement_regions_from_combined(
            &surface,
            arr_input,
            &woven_inner_wires,
            rank,
            reversed,
            face_id,
            frame,
            tol.linear,
        )
    {
        // Attach the passthrough holes (openings no section touched) whole to
        // the sub-face that geometrically contains each — they were deliberately
        // kept out of the woven arrangement so their exact arc geometry survives.
        if bay_mouth_arrangement {
            attach_whole_holes(&mut result, &original_inner_wires);
        } else if !passthrough_inner_wires.is_empty() {
            attach_whole_holes(&mut result, &passthrough_inner_wires);
        }
        return result;
    }

    // Geometric crossing/T-junction split. The wire builder under-partitions
    // a plane face whose two sections cross (X, 4 regions) or meet in a T (one
    // section's endpoint mid-way on the other, 3 regions): it merges everything
    // into one loop, or splits on only one section. Prefer the direct geometric
    // construction whenever it yields more regions than the wire builder did.
    if sections.len() >= 2
        && is_plane
        && !holes_integrated
        && !sections_reach_a_hole(sections, &original_inner_wires, frame)
        && let Some(ref boundary) = boundary_edges_backup
        && let Some(mut result) = try_split_crossing_plane_face(
            &surface, boundary, sections, rank, reversed, face_id, frame, tol,
        )
        && result.len() > loops.len()
    {
        attach_whole_holes(&mut result, &original_inner_wires);
        // Each piece's cached interior point was read off its outer wire alone,
        // and can lie in the hole it now holds.
        for piece in result.iter_mut().filter(|p| !p.inner_wires.is_empty()) {
            piece.precomputed_interior = None;
        }
        return result;
    }

    // General planar arrangement fallback: a plane face cut by three or more
    // sections forming a partial grid (e.g. a notch side wall on a SHELLED body
    // crossed by the outer wall, the inner cavity wall and the rim, or an outer
    // wall carved by a U-notch with rounded corners opening at the rim) is not
    // covered by `try_split_crossing_plane_face` (2/4-section X/T/star only), and
    // the angular wire builder hands back a self-crossing loop. Decompose the
    // full arrangement into minimal regions when it yields more regions than the
    // wire builder, when the wire builder's loops self-cross (the arrangement
    // can replace a broken trace with simple regions even at an equal/lower
    // count), OR when the wire builder's loops OVERLAP — one outer loop's
    // material directly covers another (`greedy_outer_loops_nested`), the
    // signature of a tool cap whose minuend cavity vents through wall cutouts so
    // its inner-wall ring is incomplete and the angular builder hands back the
    // whole perimeter plus sub-regions sitting inside it. Lines are exact;
    // in-plane arcs (corner roundings) are preserved via their true geometry —
    // `split_plane_face_by_arrangement` bails on off-plane straddle arcs so those
    // faces keep the existing curved paths.
    //
    // Skip when the face has un-integrated original holes: the arrangement
    // builds purely from the OUTER boundary + sections and never sees an inner
    // wire, so on a holed face (e.g. the lip-bottom annulus of the 3×3
    // stacking-lip fuse, whose corner arcs make the LOOPS trace self-cross) it
    // would decompose the annulus into hole-less disk regions and triple-share
    // the perimeter. Those faces fall through to the LOOPS path below, which
    // carries the holes correctly. (`holes_integrated` covers the case where
    // the holes WERE folded into the section set; this covers the case where
    // they were not.)
    if sections.len() >= 2
        && is_plane
        && !holes_integrated
        && original_inner_wires.is_empty()
        && let Some(ref boundary) = boundary_edges_backup
    {
        // Disc cap (circle boundary) cut by chords: the chord-based arrangement
        // cannot represent the disc's major arc (its chord cuts across the
        // face), so the greedy trace drops the remnant. Split it natively from
        // the analytic circle + chords first; the gate inside returns None for
        // any non-disc boundary.
        if let Some(result) = try_split_disk_by_chords(
            &surface, boundary, sections, rank, reversed, face_id, frame, tol.linear,
        ) && (result.len() > loops.len()
            || wire_loops_self_cross(&loops, tol.linear)
            || greedy_outer_loops_nested(&loops, cw_loops)
            || wire_loops_have_degenerate_area(&loops, tol.linear))
        {
            return result;
        }

        // The trial's section breaks reach the registry only if its pieces
        // are kept: the walker's loops, kept instead, never split the
        // sections there (a section line crossing the chord of a cap's major
        // rim arc split the wall beside it at the crossing).
        let mut trial_breaks: std::collections::HashMap<usize, Vec<Point3>> =
            std::collections::HashMap::new();
        let arr = split_plane_face_by_arrangement(
            &surface,
            boundary,
            sections,
            rank,
            reversed,
            face_id,
            frame,
            tol.linear,
            Some(&mut trial_breaks),
        );
        // The arrangement traces chords: a section cluster that never meets
        // the rim (a keyhole cap inside a disc) leaves the ring around it no
        // cell, and a rim arc whose chord crosses sections (a knuckle's rim
        // past a pin's cap) has its region traced the wrong way round.
        let walker_broken = wire_loops_self_cross(&loops, tol.linear)
            || greedy_outer_loops_nested(&loops, cw_loops)
            || wire_loops_have_degenerate_area(&loops, tol.linear)
            || section_between_holes(&loops, frame, cw_loops, tol.linear);
        if let Some(result) = arr
            && (result.len() > loops.len() || walker_broken)
            && subfaces_cover_boundary(&result, boundary, frame, walker_broken)
        {
            if let Some(reg) = split_registry {
                for (pb_id, pts) in trial_breaks {
                    reg.entry(pb_id).or_default().extend(pts);
                }
            }
            return result;
        }
    }

    // Rectilinear-arrangement rescue for a u-periodic cylinder band whose greedy
    // wire trace broke (self-crossing, overlapping, or degenerate loops) -- a box
    // cut notching the wall at partial overlap figure-eights the angular builder.
    // Gated exactly like the plane arrangement rescue: fires only when the greedy
    // loops are already broken (or, below, under-split), so it never changes a
    // face the greedy handles.
    // (A face is either a plane disc or a cylinder band, so this never overlaps
    // the disc-chord / plane-arrangement paths above.)
    let greedy_broken = wire_loops_self_cross(&loops, tol.linear)
        || greedy_outer_loops_nested(&loops, cw_loops)
        || wire_loops_have_degenerate_area(&loops, tol.linear);
    // A ring section's twin copies closed as a bare duplicate of arcs another
    // loop already consumed (box tangent on all four walls). The partition
    // double-covers the ring and orphans the far rim, so it is broken even
    // though no loop self-crosses. Detected separately from `greedy_broken`
    // so the cone arm below can gate on exactly this signature.
    // An island and the hole it leaves are one pair, not a duplicated ring.
    let ring_duplicated =
        wire_loops_duplicate_cover(&without_twin_loops(&loops, tol.linear), tol.linear);

    // Sector rescue for an UNDER-split u-periodic lateral: one full-height
    // ruling plus the glued seam should sector the strip, but to the glued
    // walker an annulus cut once is a single wrapped region (the mid-wall
    // pad whose second wall crossing rides the seam). Fires only when the
    // greedy produced no split at all, so configs the greedy handles (all
    // rulings in-face) never take this path. A pointed cone's rulings meet
    // at its apex, where its boundary has no edge to close the greedy's
    // loops on, so it always takes this path.
    let pointed_cone = matches!(&surface, FaceSurface::Cone(cone) if all_edges[..n_boundary_edges]
        .iter()
        .any(|e| (e.start_3d - cone.apex()).length() < 100.0 * tol.linear));
    if u_periodic
        && !v_periodic
        && (loops.len() <= 1 || pointed_cone)
        && !sections.is_empty()
        && matches!(&surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
        && original_inner_wires.is_empty()
        && let Some(sectors) = split_periodic_face_into_sectors(
            &surface,
            &all_edges[..n_boundary_edges],
            sections,
            rank,
            reversed,
            face_id,
            tol.linear,
        )
    {
        if trace_split {
            log::debug!("STRACE-SECTORS {} pieces", sectors.len());
        }
        return sectors;
    }
    // Whole circles split in thirds (above) give the walker's glued seam
    // vertex several equally good ways on, and its loops dead-end or tour
    // the bands between the circles as one; the arrangement does not take
    // circles either. The face trace's successor is a function of the graph:
    // when the walker's loops are broken (or there are none), adopt the
    // trace's partition if it encloses area in every loop (read
    // period-aware: a band between two circles winds the whole period) and
    // covers nothing twice.
    // Only on a wall laid out in one window, its seam copies a period apart:
    // elsewhere the pieces' u are mixed copies, and no loop's winding reads
    // right (a region and its hole twin trade places).
    let one_window = {
        let seam_us: Vec<f64> = all_edges[..n_boundary_edges]
            .iter()
            .filter(|e| {
                matches!(e.curve_3d, EdgeCurve::Line)
                    && (e.start_uv.x() - e.end_uv.x()).abs() <= 1e-9
            })
            .map(|e| e.start_uv.x())
            .collect();
        let lo = seam_us.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = seam_us.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        seam_us.len() >= 2 && ((hi - lo) - std::f64::consts::TAU).abs() <= 1e-6
    };
    let circle_bands = if circles_split
        && u_periodic
        && !v_periodic
        && one_window
        && (greedy_broken || loops.is_empty())
    {
        let mut dcel = build_wire_loops_dcel(&all_edges, tol.linear, u_periodic, v_periodic);
        dcel.retain(|lp| !is_periodic_face_outside(lp, &surface));
        // A loop inside the face comes back twice, as its region and as the
        // hole it leaves in the region around it: one pair, not a
        // duplicated ring.
        let duplicated =
            wire_loops_duplicate_cover(&without_twin_loops(&dcel, tol.linear), tol.linear);
        let degenerate = wire_loops_have_degenerate_area_periodic(
            &dcel,
            tol.linear,
            Some(std::f64::consts::TAU),
            None,
        );
        if trace_split {
            log::debug!(
                "STRACE-CIRCLES dcel={} duplicated={duplicated} degenerate={degenerate}",
                dcel.len()
            );
            for lp in &loops {
                log::debug!(
                    "STRACE-CIRCLES greedy loop {}",
                    lp.iter()
                        .map(|e| format!("({:.3},{:.3})", e.start_uv.x(), e.start_uv.y()))
                        .collect::<Vec<_>>()
                        .join(">")
                );
            }
        }
        (!dcel.is_empty() && !duplicated && !degenerate).then_some(dcel)
    } else {
        None
    };
    if trace_split {
        log::debug!(
            "STRACE-CIRCLES split={circles_split} greedy={} broken={greedy_broken} adopted={}",
            loops.len(),
            circle_bands.as_ref().map_or(0, Vec::len)
        );
    }
    if let Some(bands) = circle_bands {
        // The trace walks every region one way round, whichever way the
        // face's boundary runs in its UV. A region runs the face's rims the
        // way the face's boundary does, so where the trace's loops run the
        // rim pieces against it, it went the other way: turn every loop
        // round (a patch's region and its hole twin would otherwise trade
        // places, and the patch face come out inside out).
        let near = |a: Point2, b: Point2| (a - b).length() <= 1e-6;
        let (mut with, mut against) = (0_usize, 0_usize);
        for e in bands.iter().flatten() {
            for b in all_edges[..n_boundary_edges]
                .iter()
                .filter(|b| !matches!(b.curve_3d, EdgeCurve::Line))
            {
                if near(b.start_uv, e.start_uv) && near(b.end_uv, e.end_uv) {
                    with += 1;
                } else if near(b.start_uv, e.end_uv) && near(b.end_uv, e.start_uv) {
                    against += 1;
                }
            }
        }
        if trace_split {
            log::debug!("STRACE-CIRCLES rims with={with} against={against}");
            for lp in &bands {
                log::debug!(
                    "STRACE-CIRCLES band area={:.4} {}",
                    signed_area_2d(&sample_wire_loop_uv(lp)),
                    lp.iter()
                        .map(|e| format!("({:.3},{:.3})", e.start_uv.x(), e.start_uv.y()))
                        .collect::<Vec<_>>()
                        .join(">")
                );
            }
        }
        loops = if against > 0 && with == 0 {
            bands
                .iter()
                .map(|lp| special_cases::reverse_loop(lp))
                .collect()
        } else {
            bands
        };
    } else if u_periodic
        && !v_periodic
        && !sections.is_empty()
        && (matches!(&surface, FaceSurface::Cylinder(_)) && (greedy_broken || ring_duplicated)
            || matches!(&surface, FaceSurface::Cone(_))
                && (ring_duplicated || (pointed_cone && greedy_broken)))
    {
        if let Some(result) = split_cylinder_band_by_arrangement(
            &surface,
            &all_edges,
            n_boundary_edges,
            rank,
            reversed,
            face_id,
            tol.linear,
        ) {
            return result;
        }
        // Oblique cuts (ellipse/conic sections — e.g. a socket profile biting
        // a pad wall) are outside the rectilinear arrangement's domain. The
        // greedy walker's `used` filter makes its successor ORDER-DEPENDENT
        // (an early loop steals edges from later regions — the grand tour);
        // the DCEL trace's successor is a pure function of the graph, so try
        // it first and adopt when strictly healthier.
        //
        // On the duplicated-ring signature, split co-endpoint arc pairs in
        // the trace INPUT first: two half-rims sharing both endpoints (a
        // 2-tangency section circle, or a primitive's half-circle rims) are
        // PARALLEL edges, which make the angular successor at those nodes
        // ill-defined — the raw trace grand-tours the whole face and
        // re-emits the duplicate ring. The same split is required downstream
        // anyway (the endpoint-pair merge key), so this only moves it before
        // the trace.
        let dcel_input: Vec<OrientedPCurveEdge> = if ring_duplicated {
            let mut wrapped = vec![all_edges.clone()];
            split_coendpoint_loop_arcs(&mut wrapped, &surface);
            wrapped.swap_remove(0)
        } else {
            all_edges.clone()
        };
        let mut dcel = build_wire_loops_dcel(&dcel_input, tol.linear, u_periodic, v_periodic);
        dcel.retain(|lp| !is_periodic_face_outside(lp, &surface));
        // No pinch-split on the DCEL result: this branch is u-periodic by its
        // gate, and on a full-period face the band orbit legitimately
        // revisits the glued seam node with both visits stored at the same
        // period copy — the pinch splitter would shear the band there into
        // wrong fragments (three over-shared wall sub-faces on the lite
        // diagonal-pad fuse).
        // The duplicated-ring rescue judges the DCEL's area health with
        // period-aware sampling (a full-period band folds under the raw
        // sampler); every other path keeps the original absolute veto.
        let dcel_ring_ok = ring_duplicated
            && !wire_loops_duplicate_cover(&dcel, tol.linear)
            && !wire_loops_have_degenerate_area_periodic(
                &dcel,
                tol.linear,
                Some(std::f64::consts::TAU),
                None,
            );
        let dcel_adopted = ((!wire_loops_have_degenerate_area(&dcel, tol.linear)
            && (dcel.len() > loops.len() || wire_loops_have_degenerate_area(&loops, tol.linear)))
            || dcel_ring_ok)
            && (!wire_loops_self_cross(&dcel, tol.linear)
                || wire_loops_self_cross(&loops, tol.linear))
            && (!greedy_outer_loops_nested(&dcel, cw_loops)
                || greedy_outer_loops_nested(&loops, cw_loops));
        if dcel_adopted {
            loops = dcel;
            split_coendpoint_loop_arcs(&mut loops, &surface);
        } else {
            // The mirrored turn rule as a legacy fallback: adopt the retry only
            // when it is strictly healthier — more loops and none of the broken
            // signatures.
            let retry =
                build_wire_loops_with_winding(&all_edges, tol.linear, u_periodic, v_periodic, true);
            // Adoption bar: zero degenerate areas, NO NEW broken flags relative to
            // the greedy result, and a strict improvement — either the greedy had
            // degenerate loops (which the retry cleared) or the retry partitions
            // into more loops. `wire_loops_self_cross` is not periodic-aware — a
            // full-period band loop legitimately visits the seam vertex at both
            // unwrapped copies — so it may stay set on both sides; what matters is
            // the retry does not introduce it.
            if !wire_loops_have_degenerate_area(&retry, tol.linear)
                && (!wire_loops_self_cross(&retry, tol.linear)
                    || wire_loops_self_cross(&loops, tol.linear))
                && (!greedy_outer_loops_nested(&retry, cw_loops)
                    || greedy_outer_loops_nested(&loops, cw_loops))
                && (retry.len() > loops.len()
                    || wire_loops_have_degenerate_area(&loops, tol.linear))
            {
                loops = retry;
            }
        }
        // Pinch resolution: a grand-tour loop that revisits a UV vertex is
        // two (or more) regions traced as one — split it there, whichever
        // GREEDY-family builder produced the winning loops. Working in the
        // UNWRAPPED uv keeps the legitimate seam double-visit out (its two
        // copies differ by a period). Zero-area remnants (pure out-and-back
        // excursions) are dropped; the sliver guard below would misclassify
        // them as holes. An ADOPTED DCEL result is exempt: it is already the
        // true subdivision, and its band orbit can revisit the glued seam
        // node at the SAME stored period copy — pinch-shearing there
        // recreates the over-shared wall fragments the trace fixed.
        if !dcel_adopted && wire_loops_self_cross(&loops, tol.linear) {
            let resolved: Vec<Vec<OrientedPCurveEdge>> = loops
                .iter()
                .flat_map(|lp| split_loop_at_pinch_vertices(lp, tol.linear))
                .collect();
            // Adopt only when the split UNCOVERED a region (strictly more
            // loops) and the self-cross cleared. Equal-count adoption (shed
            // remnant only) was tried and REFUTED: an out-and-back remnant
            // can be load-bearing — dropping it on the straight-graze pad
            // wall regressed the lite_pad_graze_fuse fixture to mesh
            // fallback. Downstream reconciliation consumes those remnants.
            if resolved.len() > loops.len() && !wire_loops_self_cross(&resolved, tol.linear) {
                loops = resolved;
            }
        }
    } else if u_periodic
        && !v_periodic
        && !woven_trace
        && !sections.is_empty()
        && matches!(&surface, FaceSurface::Cylinder(_))
    {
        // UNDER-split rescue: a greedy trace can merge regions that the
        // section chains fully separate WITHOUT tripping any broken-loop
        // signature (the lite pad's B-side wall folded its socket-notch strip
        // into the buried band — 2 clean-looking loops where the graph
        // carries 4 regions — and the merged piece then classified as one,
        // deleting the notch strip). The DCEL face trace enumerates the true
        // subdivision; adopt it only when it strictly refines the greedy
        // partition and is clean by every loop-health signature.
        // A hole across the seam that an earlier cut left notches the
        // boundary, and the trace closes the notch's reversed twins round the
        // hole: a loop on the face's outside that winds like a region.
        let mut dcel = build_wire_loops_dcel_measured(
            &all_edges,
            tol.linear,
            u_periodic,
            v_periodic,
            None,
            periodic_boundary_runs_ccw(&all_edges[..n_boundary_edges], &surface),
        );
        dcel.retain(|lp| {
            !is_periodic_face_outside(lp, &surface)
                && !runs_boundary_backwards(lp, &all_edges[..n_boundary_edges], tol.linear)
        });
        // This branch is u-periodic by its gate: the pinch splitter is
        // unsafe on the glued seam graph (see the DCEL consult above), so
        // the trace is used as-is. The broken-loop flags are RELATIVE to the
        // greedy result, not absolute — `wire_loops_self_cross` is not
        // periodic-aware, so a valid winding band orbit revisiting the
        // quantized seam vertex would fail an absolute gate and the refined
        // partition (the separated strip) would be rejected.
        let refines = dcel.len() > loops.len()
            && !wire_loops_have_degenerate_area(&dcel, tol.linear)
            && (!greedy_outer_loops_nested(&dcel, cw_loops)
                || greedy_outer_loops_nested(&loops, cw_loops));
        if trace_split {
            log::debug!(
                "STRACE-DCEL greedy={} dcel={} refines={refines} self_cross={}/{}",
                loops.len(),
                dcel.len(),
                wire_loops_self_cross(&dcel, tol.linear),
                wire_loops_self_cross(&loops, tol.linear)
            );
        }
        if refines
            && (!wire_loops_self_cross(&dcel, tol.linear)
                || wire_loops_self_cross(&loops, tol.linear))
        {
            loops = dcel;
        } else if refines
            && original_inner_wires.is_empty()
            && let Some(result) = split_cylinder_band_by_arrangement(
                &surface,
                &all_edges,
                n_boundary_edges,
                rank,
                reversed,
                face_id,
                tol.linear,
            )
        {
            // The trace's band orbit revisits the glued seam (which the
            // self-cross test reads as a crossing) where the greedy walk,
            // clean but short, traced only the regions away from it: a
            // cylinder whose rims run against its u (a mirrored one) lays
            // its sections on the rims' own period copy. The arrangement
            // partitions the band as it does the greedy's broken cases.
            return result;
        }
    } else if !u_periodic && !v_periodic && !is_plane && !sections.is_empty() && greedy_broken {
        // Non-periodic band grand tour: sections meeting at T-junctions
        // (a slot-corner cylinder crossing a wall-face pair and the lip
        // taper) give the greedy walker degree-3 nodes where its
        // order-dependent successor tours adjacent regions into one
        // slitted loop. The DCEL successor is a pure function of the
        // graph; adopt it only when it strictly refines the partition and
        // is clean by EVERY absolute loop-health signature (no seam on a
        // non-periodic face, so no periodic-aware relaxation applies).
        let dcel = build_wire_loops_dcel(&all_edges, tol.linear, u_periodic, v_periodic);
        if dcel.len() > greedy_region_count(&loops, tol.linear, None)
            && !wire_loops_have_degenerate_area(&dcel, tol.linear)
            && !wire_loops_self_cross(&dcel, tol.linear)
            && (!greedy_outer_loops_nested(&dcel, cw_loops)
                || greedy_outer_loops_nested(&loops, cw_loops))
        {
            loops = dcel;
        }
    }

    // A greedy plane trace that revisits a vertex toured several regions as
    // one loop (a section run out and back, or a smooth section passing a
    // junction where another branches off), and one that closed a section
    // on its own reverse left the region it bounds unsplit. The DCEL face
    // trace partitions the arrangement itself; adopt it when it has at least
    // the greedy partition's regions and is clean by every loop-health
    // signature. As many regions is not too few: a greedy walk that turns
    // right at a vertex where a left turn bounds its region traces a pinched
    // loop in place of that region and still counts as many (a disc under
    // three coplanar faces of a yoke's back, where a cap's arc and a small
    // arc leave one end of a line). Holes no section reaches stay out of the
    // arrangement and attach whole afterwards, so they do not bar the trace.
    // Holes woven in bar it only while the greedy walk holds no spur: their
    // edges then run one way only, and a region outside the face runs them
    // the other way (an annulus round a hole that two lenses of a spigot's
    // wall bite into).
    // A face with holes in the arrangement also takes the trace for a section
    // closed on its own reverse (an arc ending where the face's boundary turns
    // away from it, a wall's notch over a bore: the walker follows the notch).
    // So does a greedy walk whose regions cover each other with no hole
    // between them. Where two section circles cross inside the face, the walk
    // turns right at a crossing and traces the outline of both discs as one
    // region, each whole circle as a hole, and the lens as a region inside.
    if is_plane
        && !sections.is_empty()
        && ((((woven_inner_wires.is_empty() && !holes_promoted) || woven_spur)
            && (loops
                .iter()
                .any(|lp| split_loop_at_pinch_vertices(lp, tol.linear).len() != 1)
                || greedy_outer_loops_nested(&loops, cw_loops)))
            || loops
                .iter()
                .any(|lp| loop_has_degenerate_area(lp, tol.linear, Some(frame))))
    {
        let frame_area = |lp: &[OrientedPCurveEdge]| loop_area_in_frame(lp, frame);
        let dcel = build_wire_loops_dcel_measured(
            &all_edges,
            tol.linear,
            u_periodic,
            v_periodic,
            Some(&frame_area),
            Some(!cw_loops),
        );
        if dcel.len() >= greedy_region_count(&loops, tol.linear, Some(frame))
            && !dcel
                .iter()
                .any(|lp| loop_has_degenerate_area(lp, tol.linear, Some(frame)))
            && !wire_loops_self_cross(&dcel, tol.linear)
            && (!greedy_outer_loops_nested(&dcel, cw_loops)
                || greedy_outer_loops_nested(&loops, cw_loops))
        {
            loops = dcel;
        }
    }

    // Classify each loop as outer (positive area) or hole (negative).
    // For loops with curved edges, sample intermediate UV points to get
    // an accurate area -- using only start_uv gives degenerate polygons
    // for 2-edge circles.
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, f64)> = Vec::new();
    let mut holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();

    let u_per_opt = if u_periodic {
        Some(std::f64::consts::TAU)
    } else {
        None
    };
    let v_per_opt = if v_periodic {
        Some(std::f64::consts::TAU)
    } else {
        None
    };

    // For periodic faces with section edges, use structural classification
    // instead of signed area. Band loops (containing seam + section edges)
    // are outer wires. Circle-only self-loops are holes. Signed area on
    // periodic surfaces is unreliable because UV wraps around the period.
    //
    // A PARTIAL analytic band (a non-periodic cylinder/cone quarter, e.g. a
    // rounded-rect corner) split boundary-to-boundary by a section CHAIN
    // produces two complementary bands that wind OPPOSITELY in UV (they share
    // the chain with flipped orientation), so the signed-area rule calls the
    // reversed one a hole and nests it inside the other. Both are genuine
    // sub-faces. The opposite-winding signature distinguishes this from a
    // genuinely nested band/hole (which winds the SAME way as its container,
    // e.g. a single plane×corner-cylinder lip section whose two seam-bounded
    // loops are both positive): only when two seam-carrying loops have
    // OPPOSITE-sign effective areas is the negative one a flipped band rather
    // than a hole. (An arc-only interior loop — no seam Line — is always a
    // hole.) For periodic bands the existing seam-based structural rule still
    // applies unconditionally.
    let loop_eff_area = |wl: &[OrientedPCurveEdge]| -> f64 {
        let pts = sample_wire_loop_uv_periodic(wl, u_per_opt, v_per_opt);
        let raw = signed_area_2d(&pts);
        if cw_loops { -raw } else { raw }
    };
    let has_seam_and_arc = |wl: &[OrientedPCurveEdge]| -> bool {
        wl.iter().any(|e| matches!(e.curve_3d, EdgeCurve::Line))
            && wl.iter().any(|e| !matches!(e.curve_3d, EdgeCurve::Line))
    };
    let partial_band_chain_split = !is_plane
        && !u_periodic
        && matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
        && !sections.is_empty()
        && {
            // Require a positive AND a negative seam+arc loop (the flipped pair).
            let mut has_pos = false;
            let mut has_neg = false;
            for wl in &loops {
                if has_seam_and_arc(wl) {
                    let a = loop_eff_area(wl);
                    if a > 0.0 {
                        has_pos = true;
                    } else if a < 0.0 {
                        has_neg = true;
                    }
                }
            }
            has_pos && has_neg
        };
    let use_structural_classification =
        (u_periodic || partial_band_chain_split) && !sections.is_empty();
    // The way the face's outline winds in (u, v), where it bounds an area
    // there: a face round a pole of its chart (a whole hemisphere) does not.
    let outline_sign = {
        let uv = sample_wire_outline_uv(topo, face.outer_wire(), &surface);
        let area = signed_area_2d(&uv);
        let perimeter: f64 = uv
            .iter()
            .zip(uv.iter().cycle().skip(1))
            .map(|(a, b)| (*b - *a).length())
            .sum();
        (uv.len() >= 3 && area.abs() > 1e-3 * perimeter).then(|| area.signum())
    };
    // A loop along the outline and a section bounds a piece only where it
    // winds as the outline does: one that runs against it closes round the
    // rest of the surface.
    let winds_with_outline = |wl: &[OrientedPCurveEdge]| -> bool {
        let pts = sample_wire_loop_uv_periodic(wl, u_per_opt, v_per_opt);
        let area = signed_area_2d(&pts);
        let perimeter: f64 = pts
            .iter()
            .zip(pts.iter().cycle().skip(1))
            .map(|(a, b)| (*b - *a).length())
            .sum();
        outline_sign.is_some_and(|sign| area.abs() > perimeter * tol.linear && area * sign > 0.0)
    };

    // A ball face round a pole of its chart, its other pole off the face, is
    // read in a plane about the pole it holds, where every loop bounds an
    // area ([`pole_round`]).
    let mut pole_holes: Option<Vec<Vec<Vec<OrientedPCurveEdge>>>> = None;
    let read_about_pole = pole_view.as_ref().and_then(|view| {
        pole_round::read(
            view,
            &all_edges,
            received.as_deref()?,
            n_boundary_edges,
            loops.clone(),
            tol.linear,
        )
    });
    let loops = match read_about_pole {
        Some(pieces) => {
            let mut per_piece = Vec::with_capacity(pieces.len());
            for (outline, piece_holes) in pieces {
                outers.push((outline, 1.0));
                per_piece.push(piece_holes);
            }
            pole_holes = Some(per_piece);
            Vec::new()
        }
        None => loops,
    };
    if let Some(read) = pole_off {
        *read = pole_holes.is_some();
    }

    // A woven hole's pieces carry no section index, like the outline's
    // edges. A loop runs on the outline only where it holds an outline edge.
    let on_outline = |e: &OrientedPCurveEdge| {
        let weld = 100.0 * tol.linear;
        all_edges[..n_boundary_edges].iter().any(|b| {
            ((b.start_3d - e.start_3d).length() <= weld && (b.end_3d - e.end_3d).length() <= weld)
                || ((b.start_3d - e.end_3d).length() <= weld
                    && (b.end_3d - e.start_3d).length() <= weld)
        })
    };
    for wire_loop in loops {
        // On a wall traced with a woven hole, a loop of the hole's pieces and
        // sections only (a window's sides and a rod's U from its bottom edge)
        // is read by its winding: the piece the U cuts off the window's edge
        // is a region, the window widened by it a hole of the region round
        // it.
        if use_structural_classification && woven_trace && !wire_loop.iter().any(&on_outline) {
            if loop_eff_area(&wire_loop) > 0.0 {
                outers.push((wire_loop, 1.0));
            } else {
                holes.push(wire_loop);
            }
            continue;
        }
        if use_structural_classification {
            // Structural: a loop containing both Line edges (seam) and
            // non-Line edges (section arcs / circles) is a band = outer.
            let has_line = wire_loop
                .iter()
                .any(|e| matches!(e.curve_3d, EdgeCurve::Line));
            let has_nonline = wire_loop
                .iter()
                .any(|e| !matches!(e.curve_3d, EdgeCurve::Line));
            // A loop running along part of the face's boundary and back
            // along a section (a conic tongue off a cone's rim, a cylinder
            // band's piece between two sections running rim to rim, the lens
            // between a cylinder's rim and an oblique plane's ellipse traced
            // on its strip) bounds a region of the face too; only a loop of
            // sections alone is a hole. A torus band's seam is a meridian
            // arc, not a line, so its pieces read this way as well; but a
            // section that retraces a stretch of the band's rim (two half
            // rings meeting along it) closes a sliver with the rim, or runs
            // out and back in the loop round the whole band, and neither
            // bounds a piece of the face.
            let torus_piece = matches!(surface, FaceSurface::Torus(_)) && {
                let runs_both_ways = wire_loop.iter().any(|a| {
                    a.source_edge_idx.is_some()
                        && wire_loop.iter().any(|b| {
                            b.source_edge_idx == a.source_edge_idx && b.forward != a.forward
                        })
                });
                let pts = sample_wire_loop_uv_periodic(&wire_loop, u_per_opt, v_per_opt);
                let perimeter: f64 = pts
                    .iter()
                    .zip(pts.iter().cycle().skip(1))
                    .map(|(a, b)| (*b - *a).length())
                    .sum();
                !runs_both_ways && signed_area_2d(&pts).abs() > perimeter * tol.linear
            };
            let on_boundary = (matches!(surface, FaceSurface::Cone(_) | FaceSurface::Cylinder(_))
                || torus_piece
                || strip
                || (!matches!(surface, FaceSurface::Torus(_)) && winds_with_outline(&wire_loop)))
                && wire_loop.iter().any(|e| e.source_edge_idx.is_none())
                && wire_loop.iter().any(|e| e.source_edge_idx.is_some());
            // On a partial band, a loop of sections alone (a strut poking
            // through the wall, its rulings and arcs closing a triangle) is
            // traced both ways: the island is one, the hole around it the
            // other, told apart by their winding.
            // So is a patch of a periodic wall bounded by sections alone, lines
            // among them (a block's side face and its top and bottom edges
            // cutting a pin's wall clear of the seam): read as a band by its
            // lines, its hole twin became a second patch and the wall round
            // it lost its hole.
            // A loop of curved sections alone whose corners lie on the
            // face's own boundary (two equal walls crossing, one truncated
            // through the other's axis: the patch between their arcs runs
            // rim to rim and touches the seam) has no region round it to be
            // a hole of, and no hole twin: it is told by its winding too.
            let touches_boundary = || {
                let boundary_vertex = |p: Point3| {
                    all_edges[..n_boundary_edges].iter().any(|b| {
                        (b.start_3d - p).length() <= tol.linear
                            || (b.end_3d - p).length() <= tol.linear
                    })
                };
                let net_u: f64 = wire_loop
                    .iter()
                    .map(|e| e.end_uv.x() - e.start_uv.x())
                    .sum();
                net_u.abs() < std::f64::consts::PI
                    && wire_loop.iter().any(|e| boundary_vertex(e.start_3d))
            };
            let island = wire_loop.iter().all(|e| e.source_edge_idx.is_some())
                && (!u_periodic
                    || (has_line && has_nonline)
                    || (!has_line
                        && matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
                        && touches_boundary()));
            if island {
                if loop_eff_area(&wire_loop) > 0.0 {
                    outers.push((wire_loop, 1.0));
                } else {
                    holes.push(wire_loop);
                }
            } else if (has_line && has_nonline) || on_boundary {
                outers.push((wire_loop, 1.0)); // area placeholder
            } else {
                holes.push(wire_loop);
            }
        } else {
            // Plane faces: classify on the arc-true via-frame polygon. The
            // pcurve sampler chords Circle2D arcs and can fold
            // reversed-boundary arcs, flipping the SIGN for a thin band
            // between two near-parallel arcs (the crescent between a bin
            // bottom's corner arc and a base socket outline): the correctly
            // wound band then classifies as a hole and the adjacent-region
            // promotion below inverts it — a same-sense shell defect. Same
            // rationale as the via-frame switch in the nesting test.
            let pts = if is_plane {
                sampling::sample_wire_loop_uv_via_frame(&wire_loop, frame)
            } else {
                sample_wire_loop_uv_periodic(&wire_loop, u_per_opt, v_per_opt)
            };
            let area = if cw_loops {
                -signed_area_2d(&pts)
            } else {
                signed_area_2d(&pts)
            };
            // Sliver guard: a loop enclosing less area than a tol-wide band
            // along its own perimeter is degenerate — e.g. an arc traversed
            // forward then backward when a coplanar partner's boundary
            // coincides with the face's own corner arc. Classifying it as
            // outer creates a zero-area face; as hole, a spurious inner wire.
            let mut perimeter: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
            if let (Some(first), Some(last)) = (pts.first(), pts.last()) {
                perimeter += (*last - *first).length();
            }
            if area.abs() <= perimeter * tol.linear {
                continue;
            }
            if trace_split {
                log::debug!(
                    "STRACE-CLASS n={} area={area:.5} cw_loops={cw_loops} -> {}",
                    wire_loop.len(),
                    if area > 0.0 { "outer" } else { "hole" }
                );
            }
            if area > 0.0 {
                outers.push((wire_loop, area));
            } else {
                holes.push(wire_loop);
            }
        }
    }

    // If all loops are CW (negative area), the winding is reversed. Promote
    // the LARGEST-area loop as the outer — promoting whichever loop the wire
    // builder happened to trace first is order-dependent: when a down-facing
    // annulus (a lip's base ring) traces throat-first, the throat became the
    // "outer", the outline was then separately promoted by the nesting pass,
    // and the throat was never re-attached as a hole — two hole-less regions
    // whose spurious full disc capped the bin interior (the mid-cell-dividers
    // lip fuse: 3-way shared ring edge → open hole shell → mesh fallback).
    if !use_structural_classification && outers.is_empty() && !holes.is_empty() {
        for hole in &mut holes {
            hole.reverse();
            for edge in hole.iter_mut() {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        }
        let areas: Vec<f64> = holes
            .iter()
            .map(|h| signed_area_2d(&sample_wire_loop_uv_periodic(h, u_per_opt, v_per_opt)))
            .collect();
        let largest = areas
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);
        let area = areas[largest];
        outers.push((holes.remove(largest), area));
    }

    // A negative-area loop is only a true hole if it is geometrically NESTED
    // inside an outer loop. When a plane face is split by a single section line
    // into two side-by-side regions, the wire builder can hand back the second
    // region wound CW (negative area) even though it is ADJACENT, not nested
    // (e.g. the above-vs-below halves of a notch-straddle tool face split at the
    // wall-top line). Loops come from a planar subdivision, so containment is
    // decidable from the whole sampled boundary ([`loop_containment`]); a loop
    // is promoted to a region only when it has points STRICTLY outside every
    // outer. A single interior probe was tried and fails on thin regions — the
    // crescent between a bin bottom's corner arc and a base socket outline's
    // chamfer is ~0.1 mm wide, and the probe slips across the shared boundary
    // into the adjacent socket region, wrongly keeping the crescent as a hole
    // there (the socket-assembly fuse's free edges at every bin corner).
    // `BoundaryCoincident` loops (re-traces of a sibling outline, woven from
    // kept whole-edge duplicate sections) must NOT be promoted: they stay
    // holes so the matching below threads their edges through the split — the
    // shelled-cup lip fuse regresses if they become regions or are dropped.
    if !use_structural_classification && !outers.is_empty() && !holes.is_empty() {
        // Plane faces: arc-true via-frame polygons. The pcurve sampler chords
        // Circle2D arcs (and can fold reversed-boundary arcs), so a hole whose
        // corner arcs poke past the outer's chord-approximated corners read as
        // "outside" and got promoted — a lip base annulus lost its throat hole
        // and the spurious full disc capped the bin interior (the
        // mid-cell-dividers lip fuse).
        let sample_loop = |wl: &[OrientedPCurveEdge]| -> Vec<Point2> {
            if is_plane {
                sampling::sample_wire_loop_uv_via_frame(wl, frame)
            } else {
                sample_wire_loop_uv(wl)
            }
        };
        let outer_uv: Vec<Vec<Point2>> = outers.iter().map(|(w, _)| sample_loop(w)).collect();
        // An outer of lines and conic arcs read as it is: its polygon cuts
        // each arc's corners off, by half a unit on a rim that is one circle
        // of radius seven, and a loop that runs nearer the rim than that
        // reads as outside it.
        let outer_exact: Vec<Option<Vec<brepkit_math::region2d::Boundary2>>> = outers
            .iter()
            .map(|(w, _)| {
                is_plane
                    .then(|| containment::exact_hole(w, frame))
                    .flatten()
            })
            .collect();
        let within_exact = |pts: &[Point2]| {
            outer_exact.iter().flatten().any(|pieces| {
                pts.iter().all(|&p| {
                    brepkit_math::region2d::point_in_region(pieces, p, tol.linear * 10.0)
                        != Some(false)
                })
            })
        };
        let mut promoted: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
        holes.retain(|hole| {
            let hole_pts = sample_loop(hole);
            if hole_pts.len() < 3 {
                return true;
            }
            let nested = outer_uv
                .iter()
                .any(|o| loop_containment(&hole_pts, o) != LoopContainment::Outside)
                || within_exact(&hole_pts);
            if nested {
                true
            } else {
                promoted.push(hole.clone());
                false
            }
        });
        for mut region in promoted {
            if trace_split {
                log::debug!(
                    "STRACE-PROMOTE hole loop n={} promoted to a region (reversed)",
                    region.len()
                );
            }
            region.reverse();
            for edge in &mut region {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
            let pts: Vec<Point2> = sample_wire_loop_uv(&region);
            let area = signed_area_2d(&pts).abs();
            outers.push((region, area));
        }
    }

    if std::env::var("BK_BAND_TRACE").is_ok() && !is_plane {
        for (wi, (w, _)) in outers.iter().enumerate() {
            let desc: Vec<String> = w
                .iter()
                .map(|e| {
                    let closed = (e.start_3d - e.end_3d).length() < 1e-9;
                    format!(
                        "{:?}{} fwd={} z=({:.1},{:.1})",
                        std::mem::discriminant(&e.curve_3d),
                        if closed { "*" } else { "" },
                        e.forward,
                        e.start_3d.z(),
                        e.end_3d.z()
                    )
                })
                .collect();
            log::debug!("BAND outer[{wi}] face {face_id:?}: {desc:?}");
        }
    }
    // A traced region whose interior lies strictly inside one of the face's
    // ORIGINAL holes is not material — a face's sub-faces tile its outer
    // region minus its holes, always. When a hole's rim arrives in the
    // arrangement as split boundary pieces (a pave-image-expanded inner wire)
    // and sections cross the opening (the 16-foot wall grid running under a
    // bin's insert-pocket mouth, #1517 4x4 row), the angular walker traces
    // the in-hole cells as loops and they used to be emitted as phantom
    // material — a disc over the pocket mouth that SD then paired against
    // the tool's genuine caps, dropping both and leaving the foot open.
    // Only for the vertex-coincidence PROMOTION path: it splices hole-split
    // line sections into the arrangement without the full weave, so the
    // in-hole cells emerge as disconnected-island traces (garbage). The real
    // weave (integrate_holes_plane) creates in-hole regions INTENTIONALLY
    // (the tilted-divider cap), and the un-woven path attaches the original
    // hole AFTER the split so regions legitimately span the opening (the
    // dovetail holecut chain) — no discard in either.
    let hole_polys_for_discard: Vec<Vec<Point2>> =
        if is_plane && holes_promoted && !original_inner_wires.is_empty() {
            original_inner_wires
                .iter()
                .map(|h| sample_wire_loop_uv_via_frame(h, frame))
                .filter(|p| p.len() >= 3)
                .collect()
        } else {
            Vec::new()
        };
    let hole_margin = tol.linear * 10.0;
    let mut sub_faces = Vec::new();
    let mut halves: Vec<(usize, Vec<OrientedPCurveEdge>)> = Vec::new();
    for (outer_wire, _area) in outers {
        if !hole_polys_for_discard.is_empty() {
            let poly = sample_wire_loop_uv_via_frame(&outer_wire, frame);
            if poly.len() >= 3 {
                let ip = super::classify_2d::sample_interior_point(&poly);
                // A region round the hole (the outer rim of a ring the
                // sections cut only near its hole) is no island, though its
                // own interior point, read without its holes, lies in it.
                let surrounds = |hp: &[Point2]| {
                    let (lo, hi) = hp.iter().fold(
                        (
                            Point2::new(f64::MAX, f64::MAX),
                            Point2::new(f64::MIN, f64::MIN),
                        ),
                        |(lo, hi), q| {
                            (
                                Point2::new(lo.x().min(q.x()), lo.y().min(q.y())),
                                Point2::new(hi.x().max(q.x()), hi.y().max(q.y())),
                            )
                        },
                    );
                    let clear = 0.01 * (hi - lo).length();
                    hp.iter().any(|&q| {
                        super::classify_2d::point_in_polygon_2d(q, &poly)
                            && super::classify_2d::distance_to_polygon_boundary(q, &poly) > clear
                    })
                };
                let matched_hole = hole_polys_for_discard.iter().find(|hp| {
                    super::classify_2d::point_in_polygon_2d(ip, hp)
                        && super::classify_2d::distance_to_polygon_boundary(ip, hp) > hole_margin
                        && !surrounds(hp)
                });
                // A region that holds a hole is not inside one: the point
                // read off its outline alone can lie in the hole it holds (a
                // ring's outer rim, whose middle is the middle of its hole).
                let holds_a_hole = matched_hole.is_some()
                    && containment::exact_hole(&outer_wire, frame).is_some_and(|outline| {
                        original_inner_wires.iter().flatten().any(|e| {
                            brepkit_math::region2d::point_in_region(
                                &outline,
                                frame.project(e.start_3d),
                                hole_margin,
                            ) == Some(true)
                        })
                    });
                if let Some(hole_poly) = matched_hole.filter(|_| !holds_a_hole) {
                    if trace_split {
                        log::debug!(
                            "STRACE-DROP face={face_id:?} in-hole island (ip=({:.4},{:.4}))",
                            ip.x(),
                            ip.y()
                        );
                    }
                    // The island trace IS the opening's rim: a disconnected
                    // island component inside a cell is unreachable from the
                    // cell's own angular walk, so the containing region would
                    // otherwise emerge with no inner wire and over-cover the
                    // opening. Hand the outline (reversed to hole orientation)
                    // to the hole-matching pass below — but only when the
                    // original-hole attach pass will NOT also run, or the
                    // opening would be represented twice (the dovetail
                    // a1corner holecut aborted to mesh on the doubled rim).
                    // Orient the outline to the ORIGINAL inner wire's
                    // winding — the ground truth for this face's hole sense
                    // (a raw-frame-CW face mirrors every convention; a blind
                    // reversal left 34 same-direction shared edges around
                    // the pocket rim).
                    let mut hole_loop = outer_wire;
                    if signed_area_2d(&poly) * signed_area_2d(hole_poly) < 0.0 {
                        hole_loop.reverse();
                        for edge in &mut hole_loop {
                            std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                            std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                            edge.forward = !edge.forward;
                        }
                    }
                    holes.push(hole_loop);
                    continue;
                }
            }
        }
        let mut sub_face = SplitSubFace {
            surface: surface.clone(),
            outer_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        };
        // A whole rim spans a period in one edge, which the periodic
        // sampler folds away: the interior point is read from the halves,
        // once the holes are attached.
        let rejoined = rejoin_untouched_rims(
            sub_face.outer_wire.clone(),
            &closed_rims,
            &trace_splits,
            tol.linear,
        );
        if rejoined.len() != sub_face.outer_wire.len() {
            halves.push((
                sub_faces.len(),
                std::mem::replace(&mut sub_face.outer_wire, rejoined),
            ));
        }
        sub_faces.push(sub_face);
    }

    if let Some(per_piece) = pole_holes {
        for (sub_face, piece_holes) in sub_faces.iter_mut().zip(per_piece) {
            sub_face.inner_wires.extend(piece_holes);
        }
    }

    // Simple hole matching: each hole goes to the outer that contains its
    // first vertex (via 2D point-in-polygon), with a first-sub-face fallback.
    // Deliberately NOT the whole-boundary [`loop_containment`] criterion the
    // promotion pass uses: a `BoundaryCoincident` hole (the woven image of a
    // kept whole-edge re-trace section, every point ON a sibling outline)
    // must not be attached to the outline it duplicates — that pairs the
    // outline with itself as a zero-area annulus and the shelled-cup lip fuse
    // loses its lip (whole-boundary matching, strict-interior matching, and
    // dropping such holes were each tried; all three regress
    // gridfinity_d4_full_1x1_bin). The first-vertex probe lands those loops in
    // the surrounding region, threading their edges through the rebuild.
    let qscale = 1.0 / tol.linear;
    let loop_key = move |wl: &[OrientedPCurveEdge]| -> Vec<QKey3> {
        let q = |p: Point3| -> (i64, i64, i64) {
            (
                (p.x() * qscale).round() as i64,
                (p.y() * qscale).round() as i64,
                (p.z() * qscale).round() as i64,
            )
        };
        let mut keys: Vec<_> = wl
            .iter()
            .map(|e| {
                let (a, b) = (q(e.start_3d), q(e.end_3d));
                if a <= b { (a, b) } else { (b, a) }
            })
            .collect();
        keys.sort_unstable();
        keys
    };
    let loop_area_abs = |wl: &[OrientedPCurveEdge]| -> f64 {
        let pts = if is_plane {
            sampling::sample_wire_loop_uv_via_frame(wl, frame)
        } else {
            sample_wire_loop_uv(wl)
        };
        signed_area_2d(&pts).abs()
    };
    for hole in holes {
        if let Some(first_pt) = hole.first().map(|e| e.start_uv) {
            let hole_key = loop_key(&hole);
            let hole_area = loop_area_abs(&hole);
            let mut assigned = false;
            let mut fallback_idx: Option<usize> = None;
            for (i, sf) in sub_faces.iter_mut().enumerate() {
                // Never attach a hole to the sub-face whose outer wire IS the
                // hole's own geometry (the reversed twin of a section loop):
                // the first-vertex probe sits exactly ON that boundary and
                // float jitter in the strict ray-cast can land it "inside",
                // pairing the outline with itself as a zero-area bubble (the
                // custom-shape lip ring lost its cap this way). Skipping the
                // twin sends the hole to the true enclosing region, which is
                // the documented intent of the first-vertex probe.
                if loop_key(&sf.outer_wire) == hole_key {
                    continue;
                }
                // A hole cannot be carried by a region smaller than itself.
                // The first-vertex probe reads "inside" when the hole's traced
                // loop happens to START at a vertex it shares with a small
                // sibling region (a woven wall-notch rect shares its inner
                // corners with the cavity-mouth loop on a shelled bin rim), and
                // first-match order then hands the whole mouth to the notch —
                // the containing disc loses its hole and the mouth re-emerges
                // as a same-sense doubled cover (#1536). Area dominance is the
                // one containment invariant the probe cannot fake.
                if loop_area_abs(&sf.outer_wire) < hole_area * (1.0 - 1e-6) {
                    continue;
                }
                if fallback_idx.is_none() {
                    fallback_idx = Some(i);
                }
                let outer_pts = sample_wire_loop_uv(&sf.outer_wire);
                if super::classify_2d::point_in_polygon_2d(first_pt, &outer_pts) {
                    sf.inner_wires.push(hole.clone());
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                // Last resort mirrors the pre-gate behaviour (never silently
                // drop a hole): the first area-passing candidate, else the
                // largest non-twin sub-face.
                let idx = fallback_idx.or_else(|| {
                    sub_faces
                        .iter()
                        .enumerate()
                        .filter(|(_, sf)| loop_key(&sf.outer_wire) != hole_key)
                        .map(|(i, sf)| (i, loop_area_abs(&sf.outer_wire)))
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(i, _)| i)
                });
                if let Some(i) = idx {
                    sub_faces[i].inner_wires.push(hole);
                }
            }
        }
    }
    for sf in &mut sub_faces {
        merge_holes_across_twin_sections(&mut sf.inner_wires, tol.linear);
    }

    // Distribute original inner wires (holes from the source face) to sub-faces.
    // Each hole is assigned to the sub-face whose outer wire contains its
    // interior sample point (a point inside the hole's enclosed area, not
    // its first vertex — that vertex often sits exactly on a sub-face
    // boundary when the split passes through it, and `point_in_polygon_2d`'s
    // strict ray-cast returns false for every sub-face). If no sub-face
    // claims the hole — degenerate UV sample, hole straddling multiple
    // sub-faces, etc. — fall back to the largest-area sub-face. A warning
    // fires for the fallback so the case stays visible; what we never do is
    // silently drop the hole as the earlier code did.
    if !original_inner_wires.is_empty() && !holes_integrated {
        let largest_sub_face_idx = |sub_faces: &[SplitSubFace]| -> Option<usize> {
            sub_faces
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let area_a =
                        super::classify_2d::signed_area_2d(&sample_wire_loop_uv(&a.outer_wire))
                            .abs();
                    let area_b =
                        super::classify_2d::signed_area_2d(&sample_wire_loop_uv(&b.outer_wire))
                            .abs();
                    area_a
                        .partial_cmp(&area_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        };

        // Pre-sample each sub-face's outer wire in UV once, plus a guaranteed
        // interior point. Reused below to resolve nesting between sub-faces.
        let sub_outer_uv: Vec<Vec<Point2>> = sub_faces
            .iter()
            .map(|sf| sample_wire_loop_uv(&sf.outer_wire))
            .collect();
        let sub_interior: Vec<Point2> = sub_outer_uv
            .iter()
            .map(|pts| super::classify_2d::sample_interior_point(pts))
            .collect();

        for (hi, hole) in original_inner_wires.iter().enumerate() {
            if woven_curved_holes.contains(&hi) {
                continue;
            }
            let hole_pts = sample_wire_loop_uv(hole);
            let assigned = if hole_pts.len() >= 3 {
                let probe = super::classify_2d::sample_interior_point(&hole_pts);
                // Assign the hole to the INNERMOST sub-face that contains it.
                // A section can split a holed face into nested annular regions
                // (e.g. a lip-bottom ring ext 15->21 cut at ext 19 yields rings
                // 15->19 and 19->21); the original ext-15 hole lies inside both
                // the ext-21 outer wire and the ext-19 ring, but belongs to the
                // inner (ext-19) region. Pick by mutual containment rather than
                // UV area: `sample_wire_loop_uv` can under-measure a rounded
                // arc wire's area, so an outer ring's polygon area can read
                // smaller than the ring it encloses. Point-in-polygon nesting
                // is robust to that sampling error. The innermost containing
                // sub-face is the one whose own interior point lies inside the
                // most other containing sub-faces.
                let containing: Vec<usize> = (0..sub_faces.len())
                    .filter(|&i| uv_in_loop_any_turn(probe, &sub_outer_uv[i], &surface))
                    .collect();
                let best = containing.iter().copied().max_by_key(|&i| {
                    containing
                        .iter()
                        .filter(|&&j| {
                            j != i
                                && super::classify_2d::point_in_polygon_2d(
                                    sub_interior[i],
                                    &sub_outer_uv[j],
                                )
                        })
                        .count()
                });
                best.map(|i| sub_faces[i].inner_wires.push(hole.clone()))
            } else {
                None
            };
            if assigned.is_some() {
                continue;
            }
            // Fallback path: degenerate sample OR no sub-face contained the
            // probe point. Attach to the largest sub-face so the geometry is
            // preserved.
            let reason = if hole_pts.len() < 3 {
                "produced a degenerate UV sample (<3 pts)"
            } else {
                "did not contain-test in any sub-face"
            };
            log::warn!(
                "face_splitter: hole with {} edges {reason}; attaching to largest sub-face \
                 as fallback",
                hole.len()
            );
            if let Some(idx) = largest_sub_face_idx(&sub_faces) {
                sub_faces[idx].inner_wires.push(hole.clone());
            }
        }
    }

    // Holes-integrated loops fallback. The arrangement happy path
    // (`arrangement_regions_from_combined`) attaches the passthrough holes
    // (openings no section touched, deliberately kept out of the woven
    // arrangement to preserve their exact arc geometry) before returning. When
    // that path is SKIPPED — the arrangement returned `None` on a degenerate
    // weave (e.g. a woven-hole arc endpoint coinciding with another woven edge
    // chord trips the arc-split bail) — execution falls through to this loops
    // path, where the `!holes_integrated`-gated distribution above does not
    // fire. Without this, the passthrough holes are silently dropped. The woven
    // holes are already carried here via `all_edges` (their edges were extended
    // in and the wire builder traces them), so only the passthrough set needs
    // re-attaching. `attach_whole_holes` never drops — it falls back to the
    // largest sub-face when no sub-face contains a hole's interior probe.
    //
    // No dedicated regression test: reaching this branch needs the woven
    // arrangement to bail to `None` (an exact arc-endpoint/chord coincidence
    // deep in `arrangement_regions_from_inputs`) WHILE passthrough holes exist,
    // a degenerate float-coincident internal state not feasibly constructible
    // from the public solid API. The reused `attach_whole_holes` never-drop
    // contract is exercised by the happy path (`dovetail_tongue_groove_cut_inmem`).
    if holes_integrated && !passthrough_inner_wires.is_empty() {
        attach_whole_holes(&mut sub_faces, &passthrough_inner_wires);
    }

    // The interior point is read once the holes are in: one read before can
    // sit in a hole. A wall with a lens hole is searched clear of its holes in
    // 3D, as `fill_images_faces` searches one: the lens's one closed edge
    // gives no hole polygon in UV.
    for (i, wire) in halves {
        let sub_face = &mut sub_faces[i];
        let whole = std::mem::replace(&mut sub_face.outer_wire, wire);
        let lens = sub_face.inner_wires.iter().any(|hole| {
            matches!(
                hole.as_slice(),
                [e] if matches!(
                    e.curve_3d,
                    EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
                )
            )
        });
        sub_face.precomputed_interior = if lens {
            cylinder_cone_remainder_interior(sub_face, tol.linear)
        } else {
            interior_point_3d(sub_face, None, tol.linear)
        };
        sub_face.outer_wire = whole;
    }

    sub_faces
}

/// Whether a face is a cylinder/cone lateral wall carrying CURVED inner-wire
/// loops — the lens-hole signature (a closed Circle/Ellipse/NURBS edge where
/// another quadric crosses the wall). For these the generic `interior_point_3d`
/// /`sample_face_interior` fallbacks are unsafe: each lens loop is a single
/// closed edge with a degenerate start/end UV, so the assembled hole polygon is
/// unusable and a generic sample can land inside the removed lens. The dedicated
/// `cylinder_cone_remainder_interior` handles them; when even its dense grid
/// finds no contained point, the analytic split must abort to mesh rather than
/// classify the wall from inside the removed region.
pub fn face_has_curved_lens_holes(topo: &Topology, face_id: FaceId) -> bool {
    use brepkit_topology::edge::EdgeCurve;
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    if !matches!(
        face.surface(),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
    ) {
        return false;
    }
    face.inner_wires().iter().any(|&wid| {
        topo.wire(wid).is_ok_and(|wire| {
            // The degenerate lens hole is a SINGLE closed curved edge (the seam
            // ellipse). A regular curved hole (e.g. a drilled bore) has multiple
            // edges and a working generic interior — don't force it to mesh.
            let [oe] = wire.edges() else {
                return false;
            };
            topo.edge(oe.edge()).is_ok_and(|e| {
                matches!(
                    e.curve(),
                    EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
                )
            })
        })
    })
}

/// A point on a whole cylinder/cone wall that lies clear of its lens holes:
/// the search of [`cylinder_cone_remainder_interior`], for a wall that no
/// section cut. The generic sample of an uncut face does not look at holes.
pub fn whole_lens_wall_interior(
    topo: &Topology,
    face_id: FaceId,
    clearance: f64,
) -> Option<Point3> {
    let face = topo.face(face_id).ok()?;
    let surface = face.surface().clone();
    let edges_of = |wire| {
        let pts = collect_wire_points(topo, wire);
        boundary_edges_to_pcurve(topo, wire, &surface, &pts, None)
    };
    let outer_wire = edges_of(face.outer_wire());
    let inner_wires: Vec<_> = face.inner_wires().iter().map(|&w| edges_of(w)).collect();
    if outer_wire.is_empty() || inner_wires.iter().any(Vec::is_empty) {
        return None;
    }
    let whole = SplitSubFace {
        surface,
        outer_wire,
        inner_wires,
        reversed: face.is_reversed(),
        parent: face_id,
        rank: Rank::A,
        precomputed_interior: None,
    };
    cylinder_cone_remainder_interior(&whole, clearance)
}

/// Get a point guaranteed inside a sub-face's outer wire (in UV space),
/// not inside any inner wire (hole), then evaluate it to 3D via the surface.
/// `None` for a plane piece bounded by lines and conic arcs when no point
/// inside them stands clear of them: a point off its own curves would
/// classify a neighbour.
#[allow(clippy::too_many_lines)]
pub fn interior_point_3d(
    sub_face: &SplitSubFace,
    frame: Option<&PlaneFrame>,
    clearance: f64,
) -> Option<Point3> {
    // For a lateral analytic band (cylinder/cone), the section edges' pcurves
    // can evaluate to a different 2pi window than the boundary edges' stored
    // (already-unwrapped) UV — e.g. a rounded-rect corner band split by a
    // faceted ramp, whose staircase arc pcurves land near u=pi while the seam
    // Lines sit near u=3pi. The two windows differ by 2pi, so the assembled UV
    // polygon self-crosses and `point_in_polygon_2d` mislabels the interior
    // sample (it ends up on the wrong side of the section). Unwrapping the
    // sampled points to one continuous u-window first makes the polygon simple
    // again so the centroid/edge-walk interior point is geometrically valid.
    // A cone's pieces already carry a consistent layout (a pointed cone's
    // piece around its apex closes across it, a whole period in u at v = 0),
    // which unwrapping folds wherever a rim piece spans over a half turn or
    // crosses the apex, and whose seam copies' pcurves need not follow the
    // traversal: sample each edge's 3D curve from its laid-out start.
    let pts_2d = if matches!(&sub_face.surface, FaceSurface::Cone(_)) {
        sampling::sample_wire_loop_uv_on_surface(&sub_face.outer_wire, &sub_face.surface)
    } else if matches!(&sub_face.surface, FaceSurface::Cylinder(_)) {
        let (u_period, v_period) = super::pcurve_compute::surface_periods(&sub_face.surface);
        // A wire laid out in one unbroken run of u, each edge starting where
        // the one before it ends, is sampled as it lies: unwrapping would
        // fold a piece spanning half a turn or more onto its complement.
        let wire = &sub_face.outer_wire;
        let unbroken = wire
            .iter()
            .zip(wire.iter().cycle().skip(1))
            .all(|(a, b)| (b.start_uv.x() - a.end_uv.x()).abs() < 1e-9);
        sample_wire_loop_uv_periodic(wire, if unbroken { None } else { u_period }, v_period)
    } else if let Some(pts) = sphere_loop_uv(sub_face).or_else(|| torus_loop_uv(sub_face)) {
        pts
    } else if let (FaceSurface::Plane { .. }, Some(f)) = (&sub_face.surface, frame) {
        // Plane faces with a frame: sample the 3D curves, never the pcurves.
        // A wire can mix pcurve orientation conventions (reversed boundary
        // arcs vs section arcs), and a convention-blind pcurve sampler folds
        // thin regions (a socket-outline corner crescent) into self-crossing
        // polygons whose "interior" point lands in the neighboring region —
        // misclassifying the sub-face and dropping it from the result.
        sampling::sample_wire_loop_uv_via_frame(&sub_face.outer_wire, f)
    } else {
        sample_wire_loop_uv(&sub_face.outer_wire)
    };
    // A plane loop of lines and conic arcs is read on its curves: its sampled
    // polygon runs on each arc's chords, and the centroid of a crescent whose
    // inner arc passes near it can sit between that arc and its chords, in the
    // neighbouring region. A crescent thinner than its chords' sag has no
    // candidate across its polygon inside it, so one is stepped off its curves.
    // The holes are read too where each reads, to check the point moved clear
    // of them below. Each hole is its own region, not one parity with the
    // outline: a face can carry a hole twice, and the two copies would cancel.
    let (exact, holes) = match (&sub_face.surface, frame) {
        (FaceSurface::Plane { .. }, Some(f)) => {
            let outline = containment::region_with_fitted_splines(&sub_face.outer_wire, f);
            let holes = outline.as_ref().and_then(|_| {
                sub_face
                    .inner_wires
                    .iter()
                    .map(|w| containment::region_with_fitted_splines(w, f))
                    .collect::<Option<Vec<_>>>()
            });
            (outline, holes)
        }
        _ => (None, None),
    };
    // A point clear of the outline's curves and of every hole's.
    let clear_of_holes = |p: Point2| {
        exact
            .as_ref()
            .is_some_and(|pieces| point_in_region(pieces, p, clearance) == Some(true))
            && holes
                .iter()
                .flatten()
                .all(|hole| point_in_region(hole, p, clearance) == Some(false))
    };
    let mut interior_uv = match exact.as_ref() {
        Some(pieces) => {
            let clear = |p: Point2| point_in_region(pieces, p, clearance) == Some(true);
            sample_interior_point_where(&pts_2d, clear)
                .or_else(|| containment::point_stepped_into_region(pieces, &pts_2d, clearance))
                // The polygon's own sample asks for more room than the
                // curves do (a thin rectangle's centre clears its sides).
                .or_else(|| Some(sample_interior_point(&pts_2d)).filter(|&p| clear(p)))?
        }
        None => sample_interior_point(&pts_2d),
    };

    // Periodic lateral walls (cone/cylinder): the closed boundary circles
    // share a seam, and `sample_wire_loop_uv` can emit a lopsided uv polygon
    // (most samples clustered on one bounding circle, plus seam-wrapped u
    // values outside [0, 2pi)). `sample_interior_point` is then pulled onto a
    // v-extreme — i.e. onto a bounding circle. For a flush/coincident cap that
    // circle is the shared rim with the opposing solid, so the classifier
    // samples exactly on the boundary and misclassifies the wall (dropping the
    // cavity face on a Cut). Snap v to the axial midpoint, which is interior
    // between the two bounding circles at the sampled u. Mirrors the
    // sphere-cap fix above.
    if matches!(
        &sub_face.surface,
        FaceSurface::Cone(_) | FaceSurface::Cylinder(_)
    ) && !pts_2d.is_empty()
    {
        let v_min = pts_2d.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min);
        let v_max = pts_2d
            .iter()
            .map(|p| p.y())
            .fold(f64::NEG_INFINITY, f64::max);
        let range = v_max - v_min;
        if range > 1e-9 {
            let margin = 0.05 * range;
            let snapped = Point2::new(interior_uv.x(), 0.5 * (v_min + v_max));
            if (interior_uv.y() < v_min + margin || interior_uv.y() > v_max - margin)
                && (!matches!(&sub_face.surface, FaceSurface::Cone(_))
                    || super::classify_2d::point_in_polygon_2d(snapped, &pts_2d))
            {
                interior_uv = snapped;
            }
        }
    }

    // Sphere cap fix: sphere sub-faces with degenerate UV boundaries (thin
    // strip at constant v) need the interior UV offset toward the pole.
    // The outer wire of a sphere cap maps to a horizontal line in UV,
    // producing a near-zero-area polygon whose centroid lies on the boundary.
    if let FaceSurface::Sphere(_) = &sub_face.surface
        && !pts_2d.is_empty()
    {
        let v_min = pts_2d.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min);
        let v_max = pts_2d
            .iter()
            .map(|p| p.y())
            .fold(f64::NEG_INFINITY, f64::max);
        if (v_max - v_min) < 0.1 {
            let v_boundary = (v_min + v_max) * 0.5;
            let u_center = pts_2d.iter().map(|p| p.x()).sum::<f64>() / pts_2d.len() as f64;

            // A band sub-face's outer wire sits at the equator (v ≈ 0) for
            // BOTH hemispheres, so its v-sign cannot say which hemisphere the
            // band covers. The hole (the cut tunnel rim) carries that sign:
            // aim the interior into the annular ring between the equator and
            // the hole. Without a hole the strip is a polar cap whose own
            // v-sign points at the enclosed pole.
            let hole_v: Option<f64> = {
                let vs: Vec<f64> = sub_face
                    .inner_wires
                    .iter()
                    .flatten()
                    .map(|e| 0.5 * (e.start_uv.y() + e.end_uv.y()))
                    .collect();
                vs.iter()
                    .copied()
                    .max_by(|a, b| a.abs().total_cmp(&b.abs()))
            };
            let target_v = match hole_v {
                Some(hv) if (hv - v_boundary).abs() > 1e-9 => 0.5 * (v_boundary + hv),
                _ => {
                    let v_pole = if v_boundary >= 0.0 {
                        std::f64::consts::FRAC_PI_2
                    } else {
                        -std::f64::consts::FRAC_PI_2
                    };
                    0.5 * (v_boundary + v_pole)
                }
            };
            interior_uv = Point2::new(u_center, target_v);
        } else {
            // A loop that winds the axis holds a pole on its left (progress
            // toward +u puts the north pole there), and its (u, v) polygon,
            // open across the seam, encloses no part of it: its centroid can
            // land beyond the loop. Aim between the loop's nearest approach
            // to that pole and the pole, short of any hole that winds the
            // axis too. The loops are sampled on their edges' own curves: a
            // pcurve can run past a pole, and holds a closed latitude circle
            // as a single point.
            let outer =
                sampling::sample_wire_loop_uv_on_surface(&sub_face.outer_wire, &sub_face.surface);
            if let Some(north) = winds_the_axis(&outer) {
                let toward = |v: f64| if north { v } else { -v };
                let reach = outer
                    .iter()
                    .map(|p| toward(p.y()))
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut far = std::f64::consts::FRAC_PI_2;
                let mut islands: Vec<Vec<Point2>> = Vec::new();
                for hole in &sub_face.inner_wires {
                    let hole_2d = sampling::sample_wire_loop_uv_on_surface(hole, &sub_face.surface);
                    if winds_the_axis(&hole_2d).is_some() {
                        let near = hole_2d
                            .iter()
                            .map(|p| toward(p.y()))
                            .fold(f64::INFINITY, f64::min);
                        far = far.min(near);
                    } else if hole_2d.len() >= 3 {
                        // A hole runs with the piece on its left: turned
                        // round, it has its own region there.
                        let own: Vec<Point2> = hole_2d.iter().rev().copied().collect();
                        islands.push(unwrapped_u(&own));
                    }
                }
                // A hole that does not wind the axis covers only part of any
                // latitude, and can cover the loop's first longitude there.
                if reach < far {
                    let v = toward(0.5 * (reach + far));
                    let free =
                        (0..16).map(|k| outer[0].x() + std::f64::consts::TAU * f64::from(k) / 16.0);
                    for u in free {
                        let covered = islands.iter().any(|hole| {
                            let u_min = hole.iter().map(|q| q.x()).fold(f64::INFINITY, f64::min);
                            let at = u_min + (u - u_min).rem_euclid(std::f64::consts::TAU);
                            super::classify_2d::point_in_polygon_2d(Point2::new(at, v), hole)
                        });
                        if !covered && let Some(p) = sub_face.surface.evaluate(u, v) {
                            return Some(p);
                        }
                    }
                    interior_uv = Point2::new(outer[0].x(), v);
                }
            }
        }
    }

    // If the point falls inside a hole, find a point between the outer wire
    // and the nearest hole boundary. (`find_point_outside_holes` steps inward
    // in small increments so it lands in a thin ring rather than overshooting
    // back into the hole.) For a planar face with holes, a centroid sampled
    // from an under-resolved outer-wire polygon can sit on the wrong side of a
    // thin annular ring even when it is not strictly inside a hole, so always
    // re-derive the interior point from the ring between outer and holes.
    //
    // A bounding-shape proxy must NOT be used for the hole test: a circle
    // around the hole centroid wildly over-covers an elongated/rectangular
    // hole (a cavity opening), flagging a legitimate thin-rim point as
    // "inside" and then displacing it to the farthest corner — which on a
    // multi-hole frame (two adjacent cavities sharing a divider) lands inside
    // the OTHER hole and silently drops the whole frame face. The accurate
    // `is_inside_any_hole` UV point-in-polygon test avoids that.
    // The ring's chord polygons can put the point back past an outer arc (a
    // notch hugging a round boss: 3e-3 inside the arc). Where the outline and
    // holes read exactly, such a point is replaced by one read on them.
    if (matches!(&sub_face.surface, FaceSurface::Plane { .. }) && !sub_face.inner_wires.is_empty())
        || is_inside_any_hole(&interior_uv, &sub_face.inner_wires)
    {
        let ring = find_point_outside_holes(&pts_2d, &sub_face.inner_wires, frame);
        interior_uv = match (exact.as_ref(), holes.as_ref()) {
            (Some(outline), Some(hole_regions)) if !clear_of_holes(ring) => {
                // A ring of crescents (a pocket whose lobes reach the rim)
                // holds no candidate across the polygon: step in off the
                // curves, the outline and holes read as one region.
                let region: Vec<brepkit_math::region2d::Boundary2> = outline
                    .iter()
                    .chain(hole_regions.iter().flatten())
                    .copied()
                    .collect();
                sample_interior_point_where(&pts_2d, clear_of_holes)
                    .or_else(|| {
                        containment::point_stepped_into_region(&region, &pts_2d, clearance)
                            .filter(|&p| clear_of_holes(p))
                    })
                    .unwrap_or(ring)
            }
            _ => ring,
        };
    }

    // Evaluate back to 3D.
    if let Some(p) = sub_face.surface.evaluate(interior_uv.x(), interior_uv.y()) {
        return Some(p);
    }

    // For plane faces, evaluate via PlaneFrame.
    if let FaceSurface::Plane { normal, .. } = &sub_face.surface {
        if let Some(f) = frame {
            return Some(f.evaluate(interior_uv.x(), interior_uv.y()));
        }
        let wire_pts: Vec<Point3> = sub_face.outer_wire.iter().map(|e| e.start_3d).collect();
        let f = PlaneFrame::from_plane_face(*normal, &wire_pts);
        return Some(f.evaluate(interior_uv.x(), interior_uv.y()));
    }

    // Last resort: average of 3D endpoints.
    let sum: Point3 = sub_face
        .outer_wire
        .iter()
        .fold(Point3::new(0.0, 0.0, 0.0), |acc, e| {
            acc + (e.start_3d - Point3::new(0.0, 0.0, 0.0))
        });
    let n = sub_face.outer_wire.len() as f64;
    Some(Point3::new(sum.x() / n, sum.y() / n, sum.z() / n))
}

/// Give each of a sphere face's holes, none of which a section touches, to the
/// innermost piece whose region holds it. A piece's region is on its outer
/// loop's left, and a loop that winds the axis bounds no `(u, v)` polygon of
/// its own: its region is closed through the pole on its left.
fn attach_sphere_holes(pieces: &mut [SplitSubFace], holes: &[Vec<OrientedPCurveEdge>]) {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let wrap = |d: f64| (d + PI).rem_euclid(TAU) - PI;
    let region = |pts: &[Point2]| -> Vec<Point2> {
        let mut poly = unwrapped_u(pts);
        if let Some(north) = winds_the_axis(pts) {
            let pole = if north { FRAC_PI_2 } else { -FRAC_PI_2 };
            let (first, last) = (pts[0], poly[poly.len() - 1]);
            let turned = last.x() + wrap(first.x() - last.x());
            poly.push(Point2::new(turned, first.y()));
            poly.push(Point2::new(turned, pole));
            poly.push(Point2::new(first.x(), pole));
        }
        poly
    };
    let holds = |poly: &[Point2], p: Point2| {
        let u_min = poly.iter().map(|q| q.x()).fold(f64::INFINITY, f64::min);
        let u = u_min + (p.x() - u_min).rem_euclid(TAU);
        [u, u + TAU, u - TAU]
            .into_iter()
            .any(|u| super::classify_2d::point_in_polygon_2d(Point2::new(u, p.y()), poly))
    };
    let regions: Vec<Option<Vec<Point2>>> = pieces
        .iter()
        .map(|piece| {
            let pts = sampling::sample_wire_loop_uv_on_surface(&piece.outer_wire, &piece.surface);
            (pts.len() >= 3).then(|| region(&pts))
        })
        .collect();
    let mut unplaced = Vec::new();
    for hole in holes {
        let probe = pieces.first().and_then(|piece| {
            sampling::sample_wire_loop_uv_on_surface(hole, &piece.surface)
                .first()
                .copied()
        });
        let innermost = probe.and_then(|p| {
            regions
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.as_ref().filter(|r| holds(r, p)).map(|r| (i, r)))
                .min_by(|(_, a), (_, b)| {
                    super::classify_2d::signed_area_2d(a)
                        .abs()
                        .total_cmp(&super::classify_2d::signed_area_2d(b).abs())
                })
                .map(|(i, _)| i)
        });
        match innermost {
            Some(i) => pieces[i].inner_wires.push(hole.clone()),
            None => unplaced.push(hole.clone()),
        }
    }
    attach_whole_holes(pieces, &unplaced);
}

/// A torus piece's outer loop in `(u, v)`, sampled on its edges' own curves
/// with both angles carried on continuously, when it winds neither the axis
/// nor the tube. The edges' own `(u, v)` can put a stretch of the loop a turn
/// of the tube away from the rest: a half ring's meridian arc from the tube
/// angle 3pi/2 up to its outer equator ends at 0, not 2pi, and the polygon
/// then folds across the piece beside it.
///
/// `None` for a piece with holes: they are read on their own stored
/// `(u, v)`, which can lie a turn from the window this reading takes. `None`
/// on a spindle torus too: a point of its sheet across the axis projects
/// onto the near tube's sheet.
fn torus_loop_uv(sub_face: &SplitSubFace) -> Option<Vec<Point2>> {
    use std::f64::consts::{PI, TAU};
    const CURVE_SAMPLES: u32 = 8;
    let FaceSurface::Torus(torus) = &sub_face.surface else {
        return None;
    };
    if torus.minor_radius() >= torus.major_radius() || !sub_face.inner_wires.is_empty() {
        return None;
    }
    let mut points = Vec::with_capacity(sub_face.outer_wire.len() * CURVE_SAMPLES as usize);
    for e in &sub_face.outer_wire {
        points.push(e.start_3d);
        if matches!(e.curve_3d, EdgeCurve::Line) {
            continue;
        }
        let closed = (e.start_3d - e.end_3d).length() < 1e-10;
        let fraction = |k: u32| f64::from(k) / f64::from(CURVE_SAMPLES);
        match &e.curve_3d {
            EdgeCurve::Circle(c) if closed => {
                let (a0, turn) = (c.project(e.start_3d), if e.forward { TAU } else { -TAU });
                points.extend((1..CURVE_SAMPLES).map(|k| c.evaluate(turn.madd(fraction(k), a0))));
            }
            _ if closed => return None,
            curve => {
                let (s3, e3) = if e.forward {
                    (e.start_3d, e.end_3d)
                } else {
                    (e.end_3d, e.start_3d)
                };
                let (t0, t1) = curve.domain_with_endpoints(s3, e3);
                let mut samples: Vec<Point3> = (1..CURVE_SAMPLES)
                    .map(|k| curve.evaluate_with_endpoints((t1 - t0).madd(fraction(k), t0), s3, e3))
                    .collect();
                // A marched curve's domain can run from either end.
                let at_t0 = curve.evaluate_with_endpoints(t0, s3, e3);
                if (at_t0 - e.start_3d).length() > (at_t0 - e.end_3d).length() {
                    samples.reverse();
                }
                points.extend(samples);
            }
        }
    }
    let nearest = |from: f64, to: f64| from + (to - from + PI).rem_euclid(TAU) - PI;
    let mut pts: Vec<Point2> = Vec::with_capacity(points.len());
    for p in points {
        let (u, v) = sub_face.surface.project_point(p)?;
        let uv = match pts.last() {
            Some(last) => Point2::new(nearest(last.x(), u), nearest(last.y(), v)),
            None => Point2::new(u, v),
        };
        pts.push(uv);
    }
    let (first, last) = (*pts.first()?, *pts.last()?);
    let closes = (nearest(last.x(), first.x()) - first.x()).abs() < 1e-9
        && (nearest(last.y(), first.y()) - first.y()).abs() < 1e-9;
    (pts.len() >= 3 && closes).then_some(pts)
}

/// A point of ball piece `sub_face` that its face's `trim` holds farther than
/// `near` from its wires, found among the candidates [`interior_point_3d`]
/// reads across the piece's outline in `(u, v)`. The outline is read in a
/// chart whose poles lie off the piece and as far from its wires as a spread
/// of axes allows: through a pole, its `u` jumps. `None` for a piece of
/// another surface, when no such chart is found or its loop is not read
/// ([`sphere_loop_uv`]), and when no candidate passes.
pub fn ball_interior_point_clear(
    sub_face: &SplitSubFace,
    trim: &crate::classifier::SphereFaceTrim,
    near: f64,
) -> Option<Point3> {
    const SAMPLES: u32 = 64;
    let FaceSurface::Sphere(sphere) = &sub_face.surface else {
        return None;
    };
    let (center, radius) = (sphere.center(), sphere.radius());
    let mut dirs: Vec<Vec3> = Vec::new();
    for e in sub_face
        .outer_wire
        .iter()
        .chain(sub_face.inner_wires.iter().flatten())
    {
        let (start, end) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let (t0, t1) = e.curve_3d.domain_with_endpoints(start, end);
        for k in 0..=SAMPLES {
            let t = (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0);
            if let Ok(d) = (e.curve_3d.evaluate_with_endpoints(t, start, end) - center).normalize()
            {
                dirs.push(d);
            }
        }
    }
    let axis = clear_ball_axis(center, radius, &dirs, trim)?;
    let chart = FaceSurface::Sphere(
        brepkit_math::surfaces::SphericalSurface::with_axis(center, radius, axis).ok()?,
    );
    let relaid = |wire: &[OrientedPCurveEdge]| -> Option<Vec<OrientedPCurveEdge>> {
        wire.iter()
            .map(|e| {
                let (su, sv) = chart.project_point(e.start_3d)?;
                let (eu, ev) = chart.project_point(e.end_3d)?;
                Some(OrientedPCurveEdge {
                    start_uv: Point2::new(su, sv),
                    end_uv: Point2::new(eu, ev),
                    ..e.clone()
                })
            })
            .collect()
    };
    let piece = SplitSubFace {
        outer_wire: relaid(&sub_face.outer_wire)?,
        inner_wires: sub_face
            .inner_wires
            .iter()
            .map(|w| relaid(w))
            .collect::<Option<_>>()?,
        surface: chart,
        ..sub_face.clone()
    };
    let pts = sphere_loop_uv(&piece)?;
    let at = |uv: Point2| piece.surface.evaluate(uv.x(), uv.y());
    let uv =
        sample_interior_point_where(&pts, |uv| at(uv).is_some_and(|p| trim.holds_clear(p, near)))?;
    at(uv)
}

/// A sphere piece's outer loop in `(u, v)`, sampled on its edges' own curves
/// and read in one continuous `u` window, when its steps in `u` sum to none:
/// it does not wind the axis, and a pole it runs through is crossed the way
/// that bounds the piece. A loop through both poles is left out (its two
/// crossings can cancel the wrong way). Pcurves computed on demand (a
/// mirrored ball's) can hand consecutive edges windows a turn apart, which
/// folds the polygon and puts its interior on the far side of the sphere.
fn sphere_loop_uv(sub_face: &SplitSubFace) -> Option<Vec<Point2>> {
    if !matches!(sub_face.surface, FaceSurface::Sphere(_)) {
        return None;
    }
    let pts = sampling::sample_wire_loop_uv_on_surface(&sub_face.outer_wire, &sub_face.surface);
    if pts.len() < 3 {
        return None;
    }
    let at_pole = |north: bool| {
        pts.iter().any(|p| {
            let v = if north { p.y() } else { -p.y() };
            v >= std::f64::consts::FRAC_PI_2 - 1e-6
        })
    };
    if at_pole(true) && at_pole(false) {
        return None;
    }
    let pts = unwrapped_u(&pts);
    let closes = (pts[pts.len() - 1].x() - pts[0].x()).abs() < std::f64::consts::PI;
    closes.then_some(pts)
}

/// A sphere loop's `(u, v)` samples with `u` carried continuously along it.
/// The loop's region is on its left.
///
/// A point at a pole has no `u` of its own. The loop arrives there on one
/// meridian and leaves on another, and the nearest way round between the two
/// is not the loop's way: for meridians a half turn apart, rounding alone
/// picks it. The region lies below the north pole and above the south one,
/// so with the region on its left the loop passes the north pole toward `-u`
/// and the south pole toward `+u`. Each run of pole points becomes the two
/// corners of that passage, and a loop that starts at a pole comes back
/// starting at its first point off the poles.
fn unwrapped_u(pts: &[Point2]) -> Vec<Point2> {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    // A point projected from a pole reads a latitude short of the pole by
    // the square root of its rounding, about 1.5e-8. A point farther off
    // has a longitude of its own: a small loop beside the pole keeps the
    // nearest way.
    let at_pole = |p: &Point2| p.y().abs() >= FRAC_PI_2 - 1e-7;
    let nearest = |from: f64, to: f64| (to - from + PI).rem_euclid(TAU) - PI;
    // A loop with no point off the poles has no meridian to pass a pole
    // from: it keeps the nearest way, as a loop with no point at a pole does.
    let first = pts.iter().position(|p| !at_pole(p));
    let (Some(first), true) = (first, pts.iter().any(at_pole)) else {
        let mut u = pts[0].x();
        let mut out = vec![pts[0]];
        for w in pts.windows(2) {
            u += nearest(w[0].x(), w[1].x());
            out.push(Point2::new(u, w[1].y()));
        }
        return out;
    };
    // The way past a pole from the meridian `from` to the meridian `to`.
    let past_pole = |from: f64, to: f64, north: bool| {
        let turn = if north { from - to } else { to - from }.rem_euclid(TAU);
        let turn = if turn > TAU - 1e-9 { 0.0 } else { turn };
        if north { -turn } else { turn }
    };
    let count = pts.len();
    let mut u = pts[first].x();
    let mut last_raw = u;
    let mut out = vec![pts[first]];
    let mut pole: Option<f64> = None;
    for k in 1..=count {
        let p = pts[(first + k) % count];
        if at_pole(&p) {
            pole = Some(p.y());
            continue;
        }
        if let Some(v) = pole.take() {
            out.push(Point2::new(u, v));
            u += past_pole(last_raw, p.x(), v > 0.0);
            out.push(Point2::new(u, v));
        } else {
            u += nearest(last_raw, p.x());
        }
        last_raw = p.x();
        // The last step comes back to the first point, which is there.
        if k < count {
            out.push(Point2::new(u, p.y()));
        }
    }
    out
}

/// Whether a sphere loop sampled in `(u, v)` winds the axis, and which way:
/// `Some(true)` when its progress in `u` runs a turn toward `+u`, which puts
/// the north pole on its left. A loop through a pole winds nothing: its
/// crossing there has no longitude, and its `(u, v)` polygon closes anyway.
fn winds_the_axis(pts: &[Point2]) -> Option<bool> {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    if pts.iter().any(|p| p.y().abs() >= FRAC_PI_2 - 1e-6) {
        return None;
    }
    let us: Vec<f64> = pts.iter().map(|p| p.x()).collect();
    let progress: f64 = us
        .iter()
        .zip(us.iter().cycle().skip(1))
        .map(|(a, b)| (b - a + PI).rem_euclid(TAU) - PI)
        .sum();
    (progress.abs() > PI).then_some(progress > 0.0)
}

/// Detect section edges (lines, open arcs, and open NURBS conics) forming
/// closed loops strictly inside a plane face's boundary (nested coplanar
/// footprints, or a box wall's socket-profile silhouette), and dedup
/// repeated segments. Both the coplanar-contact pass and adjacent-face plane-plane
/// intersections can emit the same footprint segment, so sections with the
/// same quantized ends (either way round) that are one curve
/// ([`same_section_curve`]) collapse to one.
///
/// Returns the deduped sections when every quantized endpoint has degree
/// exactly 2 (disjoint closed loops) and every endpoint lies strictly
/// interior to the boundary polygon; `None` routes back to the generic
/// wire-builder path.
fn plane_internal_line_loops(
    sections: &[SectionEdge],
    frame: &PlaneFrame,
    boundary_edges: &[OrientedPCurveEdge],
    inner_wires: &[Vec<OrientedPCurveEdge>],
    hole_points: &[Point3],
    tol_linear: f64,
) -> Option<Vec<SectionEdge>> {
    use std::collections::HashMap;

    type QPt = (i64, i64, i64);

    // Accept Line and open arc (Circle/Ellipse) sections: a rounded-rect
    // tool footprint stamps a mixed line+arc loop onto a coplanar cap.
    // Closed curves (start == end) are handled by the single-closed path.
    if sections.len() < 3
        || !sections.iter().all(|s| match s.curve_3d {
            EdgeCurve::Line => true,
            // Open curved sections chain fine: the loop builder connects by
            // endpoints and `split_face_with_internal_loops` preserves the
            // stored curve geometry. Open marched-NURBS conics matter here —
            // a box wall crossing a socket-profile stack receives its
            // silhouette as hyperbola pieces chained with lines (the
            // snap-slot wall), which must carve an internal loop.
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                (s.start - s.end).length() > tol_linear
            }
        })
    {
        return None;
    }
    let polygon: Vec<Point2> = boundary_edges.iter().map(|e| e.start_uv).collect();
    if polygon.len() < 3 {
        return None;
    }

    let quant = |p: Point3| -> QPt {
        let s = 1.0 / (tol_linear * 100.0);
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };

    let margin = tol_linear * 100.0;
    let on_plane = |p: Point3| {
        let uv = frame.project(p);
        (frame.evaluate(uv.x(), uv.y()) - p).length() <= margin
    };

    // Sections kept so far, by their quantized ends. Two sections with the
    // same ends are one only when they are one curve: two arcs between one
    // pair of points (a lens), or an arc and its chord, bound a region.
    let mut seen: HashMap<(QPt, QPt), Vec<usize>> = HashMap::new();
    let mut deduped: Vec<SectionEdge> = Vec::new();
    for s in sections {
        // A section can only bound a sub-face of this plane if it lies on
        // the plane; off-plane segments (e.g. lateral edges grazing the
        // face at one endpoint) are noise for this face.
        if !on_plane(s.start) || !on_plane(s.end) {
            continue;
        }
        let a = quant(s.start);
        let b = quant(s.end);
        if a == b {
            return None;
        }
        let key = if a <= b { (a, b) } else { (b, a) };
        let twins = seen.entry(key).or_default();
        if twins
            .iter()
            .any(|&i| same_section_curve(&deduped[i], s, margin))
        {
            continue;
        }
        twins.push(deduped.len());
        deduped.push(s.clone());
    }
    if deduped.len() < 3 {
        return None;
    }
    // A loop that runs through a hole's vertex crosses the hole there (a box
    // footprint over round openings: the rims are split where its sides cross
    // them). The hole is then part of the arrangement, not a loop inside it.
    let through_hole = hole_points.iter().any(|&p| {
        deduped
            .iter()
            .filter(|s| matches!(s.curve_3d, EdgeCurve::Line))
            .any(|s| {
                let d = s.end - s.start;
                let len2 = d.dot(d);
                let t = (p - s.start).dot(d) / len2;
                len2 > margin * margin
                    && t > 1e-6
                    && t < 1.0 - 1e-6
                    && (s.start + d * t - p).length() <= margin
            })
    });
    // So does a loop that meets a hole at its vertices (a disc whose rim
    // runs along a hole's arc: the rim's other arcs end where the hole's
    // edges do).
    let meets_hole = hole_points.iter().any(|&p| {
        deduped
            .iter()
            .any(|s| (s.start - p).length() <= margin || (s.end - p).length() <= margin)
    });
    if through_hole || meets_hole {
        return None;
    }

    // The same footprint side can arrive both whole and as sub-segments
    // split at paves. Drop any segment that another section's endpoint
    // subdivides (collinear, strictly interior) — the sub-segments carry
    // the same geometry. If the sub-segments turn out incomplete, the
    // degree check below rejects and the generic path takes over. Only a
    // line along the segment is a sub-segment: an arc ending on it (a circle
    // cut where a line crosses it) leaves it whole, and its loop is no
    // lone loop.
    //
    // A subdividing endpoint must lie ON the segment, hence inside its
    // bounding box. Index the endpoints in a coarse grid and probe only the
    // cells the segment spans, so a face with many sections (a perforated
    // panel's cap) costs O(sections) here instead of O(sections²).
    let endpoints: Vec<(Point3, Point3)> = deduped
        .iter()
        .filter(|s| matches!(s.curve_3d, EdgeCurve::Line))
        .flat_map(|s| [(s.start, s.end), (s.end, s.start)])
        .collect();
    let grid_cell = {
        let mut sum = 0.0;
        let mut cnt = 0.0;
        for s in &deduped {
            if matches!(s.curve_3d, EdgeCurve::Line) {
                sum += (s.end - s.start).length();
                cnt += 1.0;
            }
        }
        if cnt > 0.0 { sum / cnt } else { margin }
    }
    .max(margin);
    let grid_inv = 1.0 / grid_cell;
    let cell_of = |p: Point3| -> QPt {
        #[allow(clippy::cast_possible_truncation)]
        (
            (p.x() * grid_inv).floor() as i64,
            (p.y() * grid_inv).floor() as i64,
            (p.z() * grid_inv).floor() as i64,
        )
    };
    let mut ep_grid: HashMap<QPt, Vec<(Point3, Point3)>> = HashMap::new();
    for &(p, other_end) in &endpoints {
        ep_grid.entry(cell_of(p)).or_default().push((p, other_end));
    }
    deduped.retain(|s| {
        if !matches!(s.curve_3d, EdgeCurve::Line) {
            return true;
        }
        let dir = s.end - s.start;
        let len2 = dir.dot(dir);
        if len2 < margin * margin {
            return true;
        }
        let lo = cell_of(Point3::new(
            s.start.x().min(s.end.x()) - margin,
            s.start.y().min(s.end.y()) - margin,
            s.start.z().min(s.end.z()) - margin,
        ));
        let hi = cell_of(Point3::new(
            s.start.x().max(s.end.x()) + margin,
            s.start.y().max(s.end.y()) + margin,
            s.start.z().max(s.end.z()) + margin,
        ));
        let off_line = |p: Point3| {
            let t = (p - s.start).dot(dir) / len2;
            (p - (s.start + dir * t)).length()
        };
        let subdivided = |p: Point3, other_end: Point3| -> bool {
            if (p - s.start).length() < margin || (p - s.end).length() < margin {
                return false;
            }
            let t = (p - s.start).dot(dir) / len2;
            if !(0.0..=1.0).contains(&t) {
                return false;
            }
            off_line(p) < margin && off_line(other_end) < margin
        };
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                for cz in lo.2..=hi.2 {
                    if let Some(pts) = ep_grid.get(&(cx, cy, cz)) {
                        for &(p, other_end) in pts {
                            crate::perf::bump_face_split_probe();
                            if subdivided(p, other_end) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    });

    let mut degree: HashMap<QPt, u32> = HashMap::new();
    for s in &deduped {
        *degree.entry(quant(s.start)).or_insert(0) += 1;
        *degree.entry(quant(s.end)).or_insert(0) += 1;
        for p in [s.start, s.end] {
            let uv = frame.project(p);
            if !super::classify_2d::point_in_polygon_2d(uv, &polygon)
                || super::classify_2d::distance_to_polygon_boundary(uv, &polygon) <= margin
            {
                log::debug!(
                    "plane_internal_line_loops: endpoint {p:?} not strictly interior (dist {})",
                    super::classify_2d::distance_to_polygon_boundary(uv, &polygon)
                );
                return None;
            }
        }
    }
    if degree.values().any(|&d| d != 2) {
        // Chains whose loose ends dangle strictly inside an INNER wire are
        // fence loops overshooting into a hole: their in-hole tails lie
        // outside the face material and the connecting geometry (a tool
        // column swallowed by the hole) is not on this face at all, so the
        // loop legitimately closes along the hole rim (the circleinsert
        // slab's foot rings across the pocket). Trim each such chain at its
        // exact rim crossing and close it with the rim arc between its two
        // anchors. Any other degree defect keeps the historical bail.
        if let Some(rescued) =
            rescue_hole_dangling_loops(&deduped, &degree, &quant, inner_wires, frame, tol_linear)
        {
            log::debug!(
                "plane_internal_line_loops: rescued {} hole-dangling sections into {} loop sections",
                deduped.len(),
                rescued.len()
            );
            return Some(rescued);
        }
        let bad: Vec<_> = degree.iter().filter(|&(_, &d)| d != 2).collect();
        log::debug!(
            "plane_internal_line_loops: {} endpoints with degree != 2 (deduped {} of {}): {bad:?}",
            bad.len(),
            deduped.len(),
            degree.len()
        );
        return None;
    }
    Some(deduped)
}

/// Whether `a` and `b`, edges of one face that share their two ends, are one
/// curve within `tol`: `Some` where both are lines, circles or ellipses, read
/// by [`same_section_curve`] each from its stored start, and `None` where
/// either is a B-spline.
pub fn edges_one_curve(a: &OrientedPCurveEdge, b: &OrientedPCurveEdge, tol: f64) -> Option<bool> {
    let analytic = |e: &OrientedPCurveEdge| !matches!(e.curve_3d, EdgeCurve::NurbsCurve(_));
    if !(analytic(a) && analytic(b)) {
        return None;
    }
    let section = |e: &OrientedPCurveEdge| {
        let (start, end) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        SectionEdge {
            curve_3d: e.curve_3d.clone(),
            pcurve_a: e.pcurve.clone(),
            pcurve_b: e.pcurve.clone(),
            start,
            end,
            start_uv_a: None,
            end_uv_a: None,
            start_uv_b: None,
            end_uv_b: None,
            target_face: None,
            pave_block_id: None,
        }
    };
    Some(same_section_curve(&section(a), &section(b), tol))
}

/// Whether sections `a` and `b`, whose ends lie within `tol` of each other
/// (either way round), are one curve within `tol`, by a bound over their
/// whole length. Two lines are. Two circular or elliptic arcs are where
/// [`circle_arcs_apart`] or [`ellipse_arcs_apart`] holds them within `tol`,
/// read the shorter way round between their ends and along the curve from
/// each start alike. A B-spline is a line's segment when its control points
/// all lie within `tol` of the segment, as the curve lies in their hull and
/// runs from end to end; and is another B-spline's trim where
/// [`nurbs_trims_apart`] holds them within `tol`. Their middles must meet
/// too, a quick refusal and never a proof. Any other pair is taken for two
/// curves.
fn same_section_curve(a: &SectionEdge, b: &SectionEdge, tol: f64) -> bool {
    use std::borrow::Cow;
    // Each bound reads a conic's sizes from its radius or semi-axes, true
    // only on unit, square axes. A line or a B-spline is read as it is.
    fn unit(s: &SectionEdge) -> Option<(Cow<'_, SectionEdge>, f64)> {
        let (curve_3d, gap) = crate::builder::pcurve_compute::unit_conic(&s.curve_3d)?;
        let section = match curve_3d {
            Cow::Borrowed(_) => Cow::Borrowed(s),
            Cow::Owned(curve_3d) => Cow::Owned(SectionEdge {
                curve_3d,
                ..s.clone()
            }),
        };
        Some((section, gap))
    }
    let (Some((a, gap_a)), Some((b, gap_b))) = (unit(a), unit(b)) else {
        return false;
    };
    let (a, b): (&SectionEdge, &SectionEdge) = (&a, &b);
    // Each stored arc lies within its arc gap of its unit form's, and the
    // two gaps may take at most a quarter of the tolerance.
    let off_forms = crate::builder::pcurve_compute::unit_conic_arc_gap(gap_a)
        + crate::builder::pcurve_compute::unit_conic_arc_gap(gap_b);
    if off_forms.is_nan() || off_forms > 0.25 * tol {
        return false;
    }
    let along = |s: &SectionEdge, t: f64| {
        let (t0, t1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
        s.curve_3d
            .evaluate_with_endpoints((t1 - t0).madd(t, t0), s.start, s.end)
    };
    let middles_meet = (evaluate_edge_at_t(&a.curve_3d, a.start, a.end, 0.5)
        - evaluate_edge_at_t(&b.curve_3d, b.start, b.end, 0.5))
    .length()
        <= tol
        && (along(a, 0.5) - along(b, 0.5)).length() <= tol;
    let straight = |nurbs: &brepkit_math::nurbs::curve::NurbsCurve, line: &SectionEdge| {
        let d = line.end - line.start;
        let len2 = d.dot(d);
        len2 > 0.0
            && nurbs.weights().iter().all(|&w| w > 0.0)
            && nurbs.control_points().iter().all(|&p| {
                let t = ((p - line.start).dot(d) / len2).clamp(0.0, 1.0);
                (line.start + d * t - p).length() <= tol
            })
    };
    // An open arc reads two ways: the shorter way round between its ends,
    // or along its curve from its start. An arc and the rest of its circle
    // stored from the other end are one arc read the first way and two the
    // second, so the trims must meet both ways.
    let readings = |s: &SectionEdge, project: &dyn Fn(Point3) -> f64| {
        let t0 = project(s.start);
        let shorter = (
            t0,
            crate::builder::pcurve_compute::shorter_arc_delta(project(s.end) - t0),
        );
        let (n0, n1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
        [shorter, (n0, n1 - n0)]
    };
    let within = |bound: f64| !bound.is_nan() && bound + off_forms <= tol;
    match (&a.curve_3d, &b.curve_3d) {
        (EdgeCurve::Line, EdgeCurve::Line) => true,
        (EdgeCurve::Circle(p), EdgeCurve::Circle(q)) => {
            middles_meet && {
                let (ra, rb) = (
                    readings(a, &|x| p.project(x)),
                    readings(b, &|x| q.project(x)),
                );
                ra.iter()
                    .zip(&rb)
                    .all(|(&ia, &ib)| within(circle_arcs_apart(p, ia, q, ib)))
            }
        }
        (EdgeCurve::Ellipse(p), EdgeCurve::Ellipse(q)) => {
            middles_meet && {
                let (ra, rb) = (
                    readings(a, &|x| p.project(x)),
                    readings(b, &|x| q.project(x)),
                );
                ra.iter()
                    .zip(&rb)
                    .all(|(&ia, &ib)| within(ellipse_arcs_apart(p, ia, q, ib)))
            }
        }
        // A circle is the ellipse of two equal semi-axes on its axes, and is
        // read so against an ellipse.
        (EdgeCurve::Circle(c), EdgeCurve::Ellipse(q)) => {
            middles_meet
                && round_ellipse(c).is_some_and(|p| {
                    let (ra, rb) = (
                        readings(a, &|x| p.project(x)),
                        readings(b, &|x| q.project(x)),
                    );
                    ra.iter()
                        .zip(&rb)
                        .all(|(&ia, &ib)| within(ellipse_arcs_apart(&p, ia, q, ib)))
                })
        }
        (EdgeCurve::Ellipse(p), EdgeCurve::Circle(c)) => {
            middles_meet
                && round_ellipse(c).is_some_and(|q| {
                    let (ra, rb) = (
                        readings(a, &|x| p.project(x)),
                        readings(b, &|x| q.project(x)),
                    );
                    ra.iter()
                        .zip(&rb)
                        .all(|(&ia, &ib)| within(ellipse_arcs_apart(p, ia, &q, ib)))
                })
        }
        (EdgeCurve::Line, EdgeCurve::NurbsCurve(n)) => straight(n, a),
        (EdgeCurve::NurbsCurve(n), EdgeCurve::Line) => straight(n, b),
        (EdgeCurve::NurbsCurve(m), EdgeCurve::NurbsCurve(n)) => {
            middles_meet
                && within(nurbs_trims_apart(
                    m,
                    a.curve_3d.domain_with_endpoints(a.start, a.end),
                    n,
                    b.curve_3d.domain_with_endpoints(b.start, b.end),
                ))
        }
        _ => false,
    }
}

/// Circle `c` as the ellipse of two semi-axes its radius long on its axes,
/// which it is point for point at each parameter.
fn round_ellipse(c: &brepkit_math::curves::Circle3D) -> Option<brepkit_math::curves::Ellipse3D> {
    brepkit_math::curves::Ellipse3D::with_axes(
        c.center(),
        c.normal(),
        c.radius(),
        c.radius(),
        c.u_axis(),
        c.v_axis(),
    )
    .ok()
}

/// How far apart, in its own parameter, the span from `a.0` on by `a.1`
/// lies from the span from `b.0` on by `b.1`, read as sets: the larger gap
/// between their ends, matched the same way or the other way round,
/// whichever is less. A span a whole turn longer than another is a turn
/// apart.
fn spans_apart((a0, ad): (f64, f64), (b0, bd): (f64, f64)) -> f64 {
    use crate::builder::pcurve_compute::{max_or_nan, shorter_arc_delta};
    let same = shorter_arc_delta(b0 - a0);
    let back = shorter_arc_delta(b0 + bd - a0);
    let gap = |start: f64, end: f64| max_or_nan(start.abs(), end.abs());
    let (same, back) = (gap(same, same + bd - ad), gap(back, back - bd - ad));
    if same.is_nan() || back.is_nan() {
        f64::NAN
    } else {
        same.min(back)
    }
}

/// A bound on the Hausdorff distance between the arc of circle `p` from its
/// parameter `pa.0` on by `pa.1` and the arc of circle `q` over `qa`.
///
/// The rotation that takes `q`'s normal (or its reverse, whichever lies
/// nearer) onto `p`'s carries `q` onto `p`, each point moving at most
/// `|c_p - c_q| + |r_p - r_q| + r_p |n_p - n_q|` (`n_q` the sign used), and turns its arc
/// into an arc of `p`. That arc's ends lie within the span gap
/// ([`spans_apart`]) of `p`'s arc's ends along `p`, so the two arcs of `p`
/// lie within `r_p` times that gap of each other.
pub fn circle_arcs_apart(
    p: &brepkit_math::curves::Circle3D,
    pa: (f64, f64),
    q: &brepkit_math::curves::Circle3D,
    (q0, qd): (f64, f64),
) -> f64 {
    let n1 = p.normal();
    let (n2, sweep) = if n1.dot(q.normal()) < 0.0 {
        (-q.normal(), -qd)
    } else {
        (q.normal(), qd)
    };
    let cos = n2.dot(n1);
    let w = n2.cross(n1);
    let turn = |e: Vec3| e * cos + w.cross(e) + w * (w.dot(e) / (1.0 + cos));
    // `|n1 - n2|` is `2 sin(angle / 2)`, read from the components with no
    // cancellation, where `sqrt(2 - 2 cos)` loses a small turn to rounding;
    // a few units of rounding cover normals unit only within it. Normals
    // alike to the bit make `w` nothing, and the turn moves no point.
    let alike = [(n1.x(), n2.x()), (n1.y(), n2.y()), (n1.z(), n2.z())]
        .iter()
        .all(|&(a, b)| (a + 0.0).to_bits() == (b + 0.0).to_bits());
    let tilt = if alike {
        0.0
    } else {
        8.0f64.madd(f64::EPSILON, (n1 - n2).length())
    };
    let moved =
        (p.center() - q.center()).length() + (p.radius() - q.radius()).abs() + p.radius() * tilt;
    let (s, c) = q0.sin_cos();
    let start = turn(q.u_axis() * c + q.v_axis() * s);
    let start = start.dot(p.v_axis()).atan2(start.dot(p.u_axis()));
    p.radius().madd(spans_apart(pa, (start, sweep)), moved)
}

/// A bound on the Hausdorff distance between the arc of ellipse `p` from
/// its parameter `pa.0` on by `pa.1` and the arc of ellipse `q` over `qa`.
///
/// With `q`'s axes taken with signs `su`, `sv`, its point at `t2` is
/// `c_q + a_q cos t (su U_q) + b_q sin t (sv V_q)`, `t` its parameter on `p`
/// (`cos t = su cos t2`, `sin t = sv sin t2`), so it stands within
/// `|c_p - c_q| + sqrt(|a_q su U_q - a_p U_p|^2 + |b_q sv V_q - b_p V_p|^2)` of
/// `p`'s point at `t`. The arc of `p` over the matching span then lies within
/// `a_p` times the span gap ([`spans_apart`]) of `p`'s own arc.
///
/// A second bound reads each ellipse as the circle on its axes whose radius
/// is its mean semi-axis. At each parameter the ellipse lies within half the
/// gap between its semi-axes of that circle, so [`circle_arcs_apart`] between
/// the circles plus both half gaps bounds the ellipses too. It reads two round
/// ellipses on turned axes as one. The least bound stands.
pub fn ellipse_arcs_apart(
    p: &brepkit_math::curves::Ellipse3D,
    pa: (f64, f64),
    q: &brepkit_math::curves::Ellipse3D,
    (q0, qd): (f64, f64),
) -> f64 {
    let mut best = f64::INFINITY;
    for (su, sv) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        let major = q.u_axis() * (su * q.semi_major()) - p.u_axis() * p.semi_major();
        let minor = q.v_axis() * (sv * q.semi_minor()) - p.v_axis() * p.semi_minor();
        let moved = (p.center() - q.center()).length() + major.length().hypot(minor.length());
        let start = (sv * q0.sin()).atan2(su * q0.cos());
        let bound = p
            .semi_major()
            .madd(spans_apart(pa, (start, su * sv * qd)), moved);
        if bound.is_nan() {
            return f64::NAN;
        }
        best = best.min(bound);
    }
    let round = |e: &brepkit_math::curves::Ellipse3D| {
        let radius = 0.5 * (e.semi_major() + e.semi_minor());
        let off = (0.5 * (e.semi_major() - e.semi_minor())).abs() + f64::EPSILON * radius;
        brepkit_math::curves::Circle3D::with_axes(
            e.center(),
            e.normal(),
            radius,
            e.u_axis(),
            e.v_axis(),
        )
        .ok()
        .map(|circle| (circle, off))
    };
    if let (Some((cp, off_p)), Some((cq, off_q))) = (round(p), round(q)) {
        let bound = circle_arcs_apart(&cp, pa, &cq, (q0, qd)) + off_p + off_q;
        if !bound.is_nan() {
            best = best.min(bound);
        }
    }
    best
}

/// A bound on the Hausdorff distance between the trim of B-spline `m` over
/// its parameters from `ma.0` to `ma.1` and the trim of `n` over `na`.
///
/// Where [`nurbs_data_apart`] holds `m` within `d` of `n` at each parameter
/// (or of `n` read backward), `m`'s trim lies within `d` of `n` over the
/// same span (or its mirror), and that lies within `n`'s speed bound
/// ([`nurbs_speed_bound`]) times the larger gap between the spans' ends of
/// `n`'s own trim. NaN where no bound holds.
fn nurbs_trims_apart(
    m: &brepkit_math::nurbs::curve::NurbsCurve,
    ma: (f64, f64),
    n: &brepkit_math::nurbs::curve::NurbsCurve,
    na: (f64, f64),
) -> f64 {
    use crate::builder::pcurve_compute::max_or_nan;
    let Some(speed) = nurbs_speed_bound(n) else {
        return f64::NAN;
    };
    let span = |(x, y): (f64, f64)| (x.min(y), x.max(y));
    let gap = |a: (f64, f64), b: (f64, f64)| {
        let (a, b) = (span(a), span(b));
        max_or_nan((a.0 - b.0).abs(), (a.1 - b.1).abs())
    };
    let knots = n.knots();
    let (lo, hi) = (knots[0], knots[knots.len() - 1]);
    let (forward, backward) = nurbs_data_apart(m, n);
    let mut best = f64::NAN;
    for (data, theirs) in [(forward, na), (backward, (lo + hi - na.0, lo + hi - na.1))] {
        let Some(d) = data else { continue };
        let bound = speed.madd(gap(ma, theirs), d);
        if !bound.is_nan() && (best.is_nan() || bound < best) {
            best = bound;
        }
    }
    best
}

/// A bound on the speed of B-spline `n` in its own parameter: with `A` the
/// homogeneous curve about its first control point `P_0` and `w` its weight,
/// `C' = (A' - w' (C - P_0)) / w`, each of `A'` and `w'` at most its largest
/// derivative control point, `|C - P_0|` at most the control polygon's reach
/// and `w` at least the least weight. `None` for a weight not above nothing
/// or a bound that is not finite.
fn nurbs_speed_bound(n: &brepkit_math::nurbs::curve::NurbsCurve) -> Option<f64> {
    use crate::builder::pcurve_compute::max_or_nan;
    let (p, pts, ws, knots) = (n.degree(), n.control_points(), n.weights(), n.knots());
    if pts.is_empty() || ws.iter().any(|&w| w.is_nan() || w <= 0.0) {
        return None;
    }
    // The curve keeps its shape with its weights scaled alike, and read
    // against the largest they neither underflow nor overflow.
    let top = ws.iter().fold(0.0_f64, |m, &w| max_or_nan(m, w));
    let ws: Vec<f64> = ws.iter().map(|&w| w / top).collect();
    let origin = pts[0];
    let least = ws.iter().fold(f64::INFINITY, |m, &w| m.min(w));
    let reach = pts
        .iter()
        .fold(0.0_f64, |m, &q| max_or_nan(m, (q - origin).length()));
    #[allow(clippy::cast_precision_loss)]
    let degree = p as f64;
    let (mut lead, mut weight_rate) = (0.0_f64, 0.0_f64);
    for i in 0..pts.len().saturating_sub(1) {
        let du = knots[i + p + 1] - knots[i + 1];
        if du <= 0.0 {
            continue;
        }
        let step = (pts[i + 1] - origin) * ws[i + 1] - (pts[i] - origin) * ws[i];
        lead = max_or_nan(lead, step.length() * degree / du);
        weight_rate = max_or_nan(weight_rate, (ws[i + 1] - ws[i]).abs() * degree / du);
    }
    let speed = weight_rate.madd(reach, lead) / least;
    speed.is_finite().then_some(speed)
}

/// Bounds on how far B-spline `n` stands from `m` at each parameter, read
/// forward and read backward (`n` at `lo + hi - t` against `m` at `t`): each
/// `Some` only where the degree and knot vector match to the bit that way
/// round, and the weights are positive.
///
/// On one knot vector the curves share their basis functions `N_i`, and each
/// point is the mean of the control points under the shares `N_i w_i`. With
/// the second's weights a common scale times `w_i (1 + e_i)`, `|e_i| <= d`,
/// each share moves by at most `2d / (1 - d)` of itself. So at each
/// parameter the second curve lies within
/// `max |Q_i - P_i| + 2d / (1 - d) max |P_i - P_0|` of the first.
fn nurbs_data_apart(
    m: &brepkit_math::nurbs::curve::NurbsCurve,
    n: &brepkit_math::nurbs::curve::NurbsCurve,
) -> (Option<f64>, Option<f64>) {
    let (mk, nk) = (m.knots(), n.knots());
    let (mp, np) = (m.control_points(), n.control_points());
    let (mw, nw) = (m.weights(), n.weights());
    if m.degree() != n.degree() || mk.len() != nk.len() || mp.len() != np.len() || mp.is_empty() {
        return (None, None);
    }
    let positive = |w: &[f64]| w.iter().all(|&x| x.is_finite() && x > 0.0);
    if !positive(mw) || !positive(nw) {
        return (None, None);
    }
    let max_or_nan = crate::builder::pcurve_compute::max_or_nan;
    let most = |w: &[f64]| w.iter().copied().fold(0.0_f64, max_or_nan);
    let (top_m, top_n) = (most(mw), most(nw));
    let reach = mp
        .iter()
        .map(|&p| (p - mp[0]).length())
        .fold(0.0_f64, max_or_nan);
    // Each set of weights is read against its own largest, so the ratios
    // stay near one whatever the scale: raw products of weights can
    // underflow to 0 / 0 or overflow to inf / inf.
    let apart = |pairs: &mut dyn Iterator<Item = ((Point3, f64), (Point3, f64))>| {
        let (mut moved, mut ratio) = (0.0_f64, 0.0_f64);
        for ((p, wp), (q, wq)) in pairs {
            moved = max_or_nan(moved, (q - p).length());
            ratio = max_or_nan(ratio, ((wq / top_n) / (wp / top_m) - 1.0).abs());
        }
        let bound = (2.0 * ratio / (1.0 - ratio)).madd(reach, moved);
        (ratio < 0.5 && bound.is_finite()).then_some(bound)
    };
    let ours = || mp.iter().copied().zip(mw.iter().copied());
    let forward = if mk.iter().zip(nk).all(|(x, y)| x.to_bits() == y.to_bits()) {
        apart(&mut ours().zip(np.iter().copied().zip(nw.iter().copied())))
    } else {
        None
    };
    let (lo, hi) = (nk[0], nk[nk.len() - 1]);
    let backward = if mk
        .iter()
        .zip(nk.iter().rev())
        .all(|(x, &y)| x.to_bits() == (lo + hi - y).to_bits())
    {
        apart(&mut ours().zip(np.iter().copied().zip(nw.iter().copied()).rev()))
    } else {
        None
    };
    (forward, backward)
}

/// Rescue for [`plane_internal_line_loops`]: every degree-1 endpoint lies
/// strictly inside one of the face's inner wires (holes). Trims each loose
/// chain at its exact hole-rim crossing, drops the fully-in-hole tail
/// sections, and closes each chain with the rim arc between its two anchors
/// (the candidate arc containing no other chains' anchors). Returns `None`
/// unless every step resolves unambiguously — the historical bail then
/// stands.
#[allow(clippy::too_many_lines)]
fn rescue_hole_dangling_loops(
    deduped: &[SectionEdge],
    degree: &std::collections::HashMap<(i64, i64, i64), u32>,
    quant: &dyn Fn(Point3) -> (i64, i64, i64),
    inner_wires: &[Vec<OrientedPCurveEdge>],
    frame: &PlaneFrame,
    tol_linear: f64,
) -> Option<Vec<SectionEdge>> {
    if inner_wires.is_empty() || degree.values().any(|&d| d > 2) {
        return None;
    }
    let margin = tol_linear * 100.0;

    // Arc-true hole polygons in the local frame, plus per-hole edge geometry.
    let mut hole_polys: Vec<Vec<Point2>> = Vec::new();
    for wire in inner_wires {
        let mut poly = Vec::new();
        for e in wire {
            poly.push(frame.project(e.start_3d));
            if !matches!(e.curve_3d, EdgeCurve::Line) {
                let (t0, t1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
                for k in 1..32 {
                    let t = (t1 - t0).madd(f64::from(k) / 32.0, t0);
                    poly.push(
                        frame.project(e.curve_3d.evaluate_with_endpoints(t, e.start_3d, e.end_3d)),
                    );
                }
            }
        }
        if poly.len() < 3 {
            return None;
        }
        hole_polys.push(poly);
    }
    let hole_of = |p: Point3| -> Option<usize> {
        let uv = frame.project(p);
        hole_polys.iter().position(|poly| {
            super::classify_2d::point_in_polygon_2d(uv, poly)
                && super::classify_2d::distance_to_polygon_boundary(uv, poly) > margin
        })
    };

    // Gate: every loose end strictly inside some hole.
    let loose: Vec<(i64, i64, i64)> = degree
        .iter()
        .filter(|&(_, &d)| d == 1)
        .map(|(&q, _)| q)
        .collect();
    if loose.is_empty() {
        return None;
    }
    let mut loose_hole: std::collections::HashMap<(i64, i64, i64), usize> =
        std::collections::HashMap::new();
    for s in deduped {
        for p in [s.start, s.end] {
            if loose.contains(&quant(p)) {
                let hi = hole_of(p)?;
                loose_hole.insert(quant(p), hi);
            }
        }
    }
    if loose_hole.len() != loose.len() {
        return None;
    }

    // Exact crossing of a Line section with a hole edge, in the local frame.
    let line_hole_crossing = |sec_start: Point3, sec_end: Point3, hi: usize| -> Option<Point3> {
        let wire = &inner_wires[hi];
        let a2 = frame.project(sec_start);
        let b2 = frame.project(sec_end);
        let d2 = b2 - a2;
        let mut best: Option<(f64, Point3)> = None;
        for e in wire {
            match &e.curve_3d {
                EdgeCurve::Circle(c) => {
                    let c2 = frame.project(c.center());
                    let r = c.radius();
                    // |a2 + t*d2 - c2|^2 = r^2
                    let f = a2 - c2;
                    let qa = d2.dot(d2);
                    let qb = 2.0 * f.dot(d2);
                    let qc = f.dot(f) - r * r;
                    let disc = qb.madd(qb, -4.0 * qa * qc);
                    if disc < 0.0 || qa.abs() < 1e-30 {
                        continue;
                    }
                    for t in [
                        (-qb - disc.sqrt()) / (2.0 * qa),
                        (-qb + disc.sqrt()) / (2.0 * qa),
                    ] {
                        if !(0.0..=1.0).contains(&t) {
                            continue;
                        }
                        let p3 = sec_start + (sec_end - sec_start) * t;
                        // Within the arc's own span?
                        let (t0, t1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
                        let ang = c.project(p3);
                        let span = t1 - t0;
                        let rel = (ang - t0).rem_euclid(std::f64::consts::TAU);
                        if rel <= span + 1e-9 {
                            match best {
                                Some((bt, _)) if bt <= t => {}
                                _ => best = Some((t, p3)),
                            }
                        }
                    }
                }
                EdgeCurve::Line => {
                    let s2 = frame.project(e.start_3d);
                    let e2 = frame.project(e.end_3d);
                    let sd = e2 - s2;
                    let det = d2.x().madd(sd.y(), -(d2.y() * sd.x()));
                    if det.abs() < 1e-14 {
                        continue;
                    }
                    let dx = s2 - a2;
                    let t = dx.x().madd(sd.y(), -(dx.y() * sd.x())) / det;
                    let u = dx.x().madd(d2.y(), -(dx.y() * d2.x())) / det;
                    if (0.0..=1.0).contains(&t) && (-1e-9..=1.0 + 1e-9).contains(&u) {
                        let p3 = sec_start + (sec_end - sec_start) * t;
                        match best {
                            Some((bt, _)) if bt <= t => {}
                            _ => best = Some((t, p3)),
                        }
                    }
                }
                _ => return None,
            }
        }
        best.map(|(_, p)| p)
    };

    // Walk each loose chain, trimming at the rim.
    let mut sections: Vec<Option<SectionEdge>> = deduped.iter().cloned().map(Some).collect();
    let mut adjacency: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, s) in deduped.iter().enumerate() {
        adjacency.entry(quant(s.start)).or_default().push(i);
        adjacency.entry(quant(s.end)).or_default().push(i);
    }
    // (hole idx, anchor point, chain id) per trimmed end.
    let mut anchors: Vec<(usize, Point3, usize)> = Vec::new();
    let mut visited_loose: std::collections::HashSet<(i64, i64, i64)> =
        std::collections::HashSet::new();
    let mut chain_id = 0usize;
    for &start_q in &loose {
        if visited_loose.contains(&start_q) {
            continue;
        }
        let hi = *loose_hole.get(&start_q)?;
        // Walk from this loose end through degree-2 nodes to the other end.
        let mut prev_q = start_q;
        let mut cur_idx = *adjacency.get(&start_q)?.first()?;
        let mut chain_members: Vec<usize> = Vec::new();
        loop {
            chain_members.push(cur_idx);
            let s = deduped.get(cur_idx)?;
            let (aq, bq) = (quant(s.start), quant(s.end));
            let next_q = if aq == prev_q { bq } else { aq };
            if loose.contains(&next_q) {
                visited_loose.insert(start_q);
                visited_loose.insert(next_q);
                break;
            }
            let nbrs = adjacency.get(&next_q)?;
            let next_idx = *nbrs.iter().find(|&&i| i != cur_idx)?;
            prev_q = next_q;
            cur_idx = next_idx;
        }
        // Trim both ends of this chain.
        for end_of_chain in [false, true] {
            let iter: Box<dyn Iterator<Item = &usize>> = if end_of_chain {
                Box::new(chain_members.iter().rev())
            } else {
                Box::new(chain_members.iter())
            };
            let mut trimmed = false;
            for &mi in iter {
                let Some(sec) = sections[mi].clone() else {
                    continue;
                };
                let in_s = hole_of(sec.start).is_some();
                let in_e = hole_of(sec.end).is_some();
                if in_s && in_e {
                    sections[mi] = None;
                    continue;
                }
                if !in_s && !in_e {
                    return None;
                }
                if !matches!(sec.curve_3d, EdgeCurve::Line) {
                    return None;
                }
                let cross = line_hole_crossing(sec.start, sec.end, hi)?;
                let mut new_sec = sec;
                if in_s {
                    new_sec.start = cross;
                } else {
                    new_sec.end = cross;
                }
                new_sec.start_uv_a = None;
                new_sec.end_uv_a = None;
                new_sec.start_uv_b = None;
                new_sec.end_uv_b = None;
                sections[mi] = Some(new_sec);
                anchors.push((hi, cross, chain_id));
                trimmed = true;
                break;
            }
            if !trimmed {
                return None;
            }
        }
        chain_id += 1;
    }
    if anchors.len() != 2 * chain_id {
        return None;
    }

    // Rim connector per chain: the arc between its two anchors that contains
    // no other chains' anchors.
    let mut out: Vec<SectionEdge> = sections.into_iter().flatten().collect();
    for cid in 0..chain_id {
        let pair: Vec<&(usize, Point3, usize)> =
            anchors.iter().filter(|(_, _, c)| *c == cid).collect();
        let [(hi_a, pa, _), (hi_b, pb, _)] = pair[..] else {
            return None;
        };
        if hi_a != hi_b {
            return None;
        }
        // One shared circle for the whole hole rim.
        let circle = inner_wires[*hi_a].iter().find_map(|e| {
            if let EdgeCurve::Circle(c) = &e.curve_3d {
                Some(c.clone())
            } else {
                None
            }
        })?;
        let on_rim = |p: Point3| ((p - circle.center()).length() - circle.radius()).abs() <= margin;
        if !on_rim(*pa) || !on_rim(*pb) {
            return None;
        }
        let ang = |p: Point3| circle.project(p);
        let (ta, tb) = (ang(*pa), ang(*pb));
        let span_ab = (tb - ta).rem_euclid(std::f64::consts::TAU);
        let others: Vec<f64> = anchors
            .iter()
            .filter(|(h, _, c)| *c != cid && h == hi_a)
            .map(|(_, p, _)| ang(*p))
            .collect();
        let contains_other = |from: f64, span: f64| {
            others.iter().any(|&o| {
                let rel = (o - from).rem_euclid(std::f64::consts::TAU);
                rel > 1e-9 && rel < span - 1e-9
            })
        };
        let ab_clear = !contains_other(ta, span_ab);
        let ba_clear = !contains_other(tb, std::f64::consts::TAU - span_ab);
        log::debug!(
            "rescue connector: ta={ta:.4} tb={tb:.4} span_ab={span_ab:.4} ab_clear={ab_clear} ba_clear={ba_clear}"
        );
        let (from_p, to_p) = match (ab_clear, ba_clear) {
            (true, false) => (*pa, *pb),
            (false, true) => (*pb, *pa),
            (true, true) => {
                if span_ab <= std::f64::consts::PI {
                    (*pa, *pb)
                } else {
                    (*pb, *pa)
                }
            }
            (false, false) => return None,
        };
        let uv_c = frame.project(circle.center());
        let pc2 = brepkit_math::curves2d::Circle2D::new(uv_c, circle.radius()).ok()?;
        out.push(SectionEdge {
            curve_3d: EdgeCurve::Circle(circle),
            pcurve_a: brepkit_math::curves2d::Curve2D::Circle(pc2.clone()),
            pcurve_b: brepkit_math::curves2d::Curve2D::Circle(pc2),
            start: from_p,
            end: to_p,
            start_uv_a: None,
            end_uv_a: None,
            start_uv_b: None,
            end_uv_b: None,
            target_face: None,
            pave_block_id: None,
        });
    }

    // The rescued web must be fully degree-2.
    let mut deg2: std::collections::HashMap<(i64, i64, i64), u32> =
        std::collections::HashMap::new();
    for s in &out {
        *deg2.entry(quant(s.start)).or_insert(0) += 1;
        *deg2.entry(quant(s.end)).or_insert(0) += 1;
    }
    if deg2.values().any(|&d| d != 2) {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::curves2d::Line2D;
    use brepkit_math::vec::Vec2;
    use brepkit_topology::test_utils::make_unit_square_face;

    fn dummy_pcurve() -> brepkit_math::curves2d::Curve2D {
        brepkit_math::curves2d::Curve2D::Line(
            Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap(),
        )
    }

    /// Half of the ellipse `v = 2 + cos(u)` on a unit cylinder: read between
    /// the samples, the path stays within its bound of the curve. Even steps
    /// alone miss it by 1.2e-3 a sixty-fourth of a turn along.
    #[test]
    fn a_separator_path_reads_its_curve_between_samples() {
        use std::f64::consts::{PI, TAU};
        let uv_at = |f: f64| Some((PI * f, 2.0 + (PI * f).cos()));
        let within = 5e-6;
        let mut path = Vec::new();
        push_uv_path(&mut path, 0.0, TAU, within, &uv_at).unwrap();
        for k in 1..64 {
            let u = PI * f64::from(k) / 64.0;
            let v = winding_path_v_at(&path, u, TAU).unwrap();
            assert!((v - (2.0 + u.cos())).abs() < 2.0 * within, "at {u}: {v}");
        }
        let mut coarse = Vec::new();
        push_uv_path(&mut coarse, 0.0, TAU, f64::MAX, &uv_at).unwrap();
        let u = PI / 64.0;
        let v = winding_path_v_at(&coarse, u, TAU).unwrap();
        assert!((v - (2.0 + u.cos())).abs() > 1e-3);
    }

    /// A closed spline among the sections is the general splitter's: its quarters
    /// would each read back as the whole loop.
    #[test]
    fn a_closed_spline_section_leaves_the_face_to_the_general_splitter() {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::vec::{Point2, Vec2};
        use brepkit_topology::builder::make_planar_face;

        let section = |curve_3d: EdgeCurve, start: Point3, end: Point3| {
            let flat =
                Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap());
            SectionEdge {
                curve_3d,
                pcurve_a: flat.clone(),
                pcurve_b: flat,
                start,
                end,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
                pave_block_id: None,
            }
        };
        let mut topo = Topology::new();
        let corners = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
            .map(|(x, y)| Point3::new(x, y, 0.0));
        let face = make_planar_face(&mut topo, &corners, 1e-7).unwrap();
        let at = Point3::new(4.0, 0.0, 0.0);
        let points = vec![
            at,
            Point3::new(4.0, 6.0, 0.0),
            Point3::new(-4.0, 6.0, 0.0),
            Point3::new(-4.0, -6.0, 0.0),
            Point3::new(4.0, -6.0, 0.0),
            at,
        ];
        let weights = vec![1.0; points.len()];
        let knots = vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0];
        let loop_curve = NurbsCurve::new(2, knots, points, weights).unwrap();
        let sections = [
            section(EdgeCurve::NurbsCurve(loop_curve), at, at),
            section(
                EdgeCurve::Line,
                Point3::new(-10.0, 1.0, 0.0),
                Point3::new(10.0, 1.0, 0.0),
            ),
        ];
        assert!(split_plane_into_cells(&topo, face, &sections, Rank::B, 1e-7).is_none());
    }

    /// A closed section whose vertex sits off its circle's parameter origin,
    /// crossed by a line: two half discs and the square's two parts around them.
    #[test]
    fn a_circle_read_from_its_own_vertex_and_a_line_make_four_cells() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::vec::{Point2, Vec2};
        use brepkit_topology::builder::make_planar_face;

        let section = |curve_3d: EdgeCurve, start: Point3, end: Point3| {
            let flat =
                Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap());
            SectionEdge {
                curve_3d,
                pcurve_a: flat.clone(),
                pcurve_b: flat,
                start,
                end,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
                pave_block_id: None,
            }
        };
        for start_at in [0.0, 2.4, 5.0] {
            let mut topo = Topology::new();
            let corners = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
                .map(|(x, y)| Point3::new(x, y, 0.0));
            let face = make_planar_face(&mut topo, &corners, 1e-7).unwrap();
            let round =
                Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
            let on_round = round.evaluate(start_at);
            let (left, right) = (Point3::new(-10.0, 1.0, 0.0), Point3::new(10.0, 1.0, 0.0));
            let sections = [
                section(EdgeCurve::Circle(round), on_round, on_round),
                section(EdgeCurve::Line, left, right),
            ];
            let cells = split_plane_into_cells(&topo, face, &sections, Rank::B, 1e-7).unwrap();
            assert_eq!(cells.len(), 4, "from {start_at}");
            let swept: f64 = cells
                .iter()
                .flat_map(|cell| &cell.outer_wire)
                .filter(|edge| matches!(edge.curve_3d, EdgeCurve::Circle(_)))
                .map(|edge| {
                    let (from, to) = if edge.forward {
                        (edge.start_3d, edge.end_3d)
                    } else {
                        (edge.end_3d, edge.start_3d)
                    };
                    let (t0, t1) = edge.curve_3d.domain_with_endpoints(from, to);
                    t1 - t0
                })
                .sum();
            assert!(
                (swept - 4.0 * std::f64::consts::PI).abs() < 1e-6,
                "from {start_at}: its cells run {swept} round the circle, not once each way"
            );
        }
    }

    /// A disc's rim, its upper half one arc, a line across it along that
    /// arc's chord, and a small half disc on the line: at each end of the line
    /// the arc and the line leave along one direction, and the arc's own
    /// direction orders them. A half annulus, the small half disc and the
    /// lower half disc, in turns that leave the chords tied exactly and only
    /// to rounding.
    #[test]
    fn rim_arcs_whose_chords_run_along_a_line_bound_their_own_regions() {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::PI;

        for turn in [0.0, 0.7, 1.3, PI / 2.0, 2.9, 4.4] {
            let (s, c) = turn.sin_cos();
            let at = |x: f64, y: f64| Point3::new(c * x - s * y, s * x + c * y, 0.0);
            let normal = Vec3::new(0.0, 0.0, 1.0);
            let surface = FaceSurface::Plane { normal, d: 0.0 };
            let rim = [at(5.0, 0.0), at(0.0, 5.0), at(-5.0, 0.0), at(0.0, -5.0)];
            let frame = PlaneFrame::from_plane_face(normal, &rim);
            let edge = |curve_3d: EdgeCurve, from: Point3, to: Point3| OrientedPCurveEdge {
                pcurve: compute_pcurve_on_surface(
                    &curve_3d,
                    from,
                    to,
                    &surface,
                    &rim,
                    Some(&frame),
                ),
                curve_3d,
                start_uv: frame.project(from),
                end_uv: frame.project(to),
                start_3d: from,
                end_3d: to,
                forward: true,
                source_edge_idx: None,
                pave_block_id: None,
            };
            let round = |r: f64| {
                EdgeCurve::Circle(Circle3D::new(Point3::new(0.0, 0.0, 0.0), normal, r).unwrap())
            };
            let edges = [
                edge(round(5.0), at(5.0, 0.0), at(-5.0, 0.0)),
                edge(round(5.0), at(-5.0, 0.0), at(0.0, -5.0)),
                edge(round(5.0), at(0.0, -5.0), at(5.0, 0.0)),
                edge(EdgeCurve::Line, at(-5.0, 0.0), at(-2.0, 0.0)),
                edge(EdgeCurve::Line, at(-2.0, 0.0), at(2.0, 0.0)),
                edge(EdgeCurve::Line, at(2.0, 0.0), at(5.0, 0.0)),
                edge(round(2.0), at(2.0, 0.0), at(0.0, 2.0)),
                edge(round(2.0), at(0.0, 2.0), at(-2.0, 0.0)),
            ];
            let pieces = arrangement_regions_from_combined(
                &surface,
                &edges,
                &[],
                Rank::B,
                false,
                make_unit_square_face(&mut Topology::new()),
                &frame,
                1e-7,
            );
            assert!(pieces.is_some(), "turn {turn}: no regions");
            let pieces = pieces.unwrap();
            let mut areas: Vec<f64> = pieces
                .iter()
                .map(|p| loop_area_in_frame(&p.outer_wire, &frame).abs())
                .collect();
            areas.sort_by(f64::total_cmp);
            let want = [2.0 * PI, 10.5 * PI, 12.5 * PI];
            assert_eq!(areas.len(), want.len(), "turn {turn}: regions {areas:?}");
            for (got, want) in areas.iter().zip(want) {
                assert!((got - want).abs() < 1e-6, "turn {turn}: regions {areas:?}");
            }
        }
    }

    /// A band of a unit cylinder between two tilted rims 1e-4 apart,
    /// `cos u <= v <= cos u + 1e-4`, its rims cut into pieces at different
    /// angles: its boundary runs counter-clockwise (area about 2π·1e-4), and
    /// the same boundary run the other way runs clockwise.
    #[test]
    fn a_thin_band_between_tilted_rims_reads_its_sense() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, TAU};

        let z = Vec3::new(0.0, 0.0, 1.0);
        let cyl = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z, 1.0).unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let rim = |lift: f64| {
            Ellipse3D::with_axes(
                Point3::new(0.0, 0.0, lift),
                Vec3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2),
                2.0_f64.sqrt(),
                1.0,
                Vec3::new(FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2),
                Vec3::new(0.0, 1.0, 0.0),
            )
            .unwrap()
        };
        let uv = |p: Point3| {
            let (u, v) = surface.project_point(p).unwrap();
            Point2::new(u, v)
        };
        let edge = |curve: EdgeCurve, from: Point3, to: Point3, forward: bool| OrientedPCurveEdge {
            curve_3d: curve,
            pcurve: dummy_pcurve(),
            start_uv: uv(from),
            end_uv: uv(to),
            start_3d: from,
            end_3d: to,
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let (low, high) = (rim(0.0), rim(1e-4));
        let mut band = Vec::new();
        for w in [0.0, FRAC_PI_2, 1.25 * PI, TAU].windows(2) {
            let (a, b) = (low.evaluate(w[0]), low.evaluate(w[1]));
            band.push(edge(EdgeCurve::Ellipse(low.clone()), a, b, true));
        }
        band.push(edge(
            EdgeCurve::Line,
            low.evaluate(TAU),
            high.evaluate(TAU),
            true,
        ));
        for w in [TAU, 1.5 * PI, PI, FRAC_PI_2, 0.0].windows(2) {
            let (a, b) = (high.evaluate(w[0]), high.evaluate(w[1]));
            band.push(edge(EdgeCurve::Ellipse(high.clone()), a, b, false));
        }
        band.push(edge(
            EdgeCurve::Line,
            high.evaluate(0.0),
            low.evaluate(0.0),
            true,
        ));
        assert_eq!(periodic_boundary_runs_ccw(&band, &surface), Some(true));
        let reversed: Vec<OrientedPCurveEdge> = band
            .iter()
            .rev()
            .map(|e| OrientedPCurveEdge {
                start_uv: e.end_uv,
                end_uv: e.start_uv,
                start_3d: e.end_3d,
                end_3d: e.start_3d,
                forward: !e.forward,
                ..e.clone()
            })
            .collect();
        assert_eq!(periodic_boundary_runs_ccw(&reversed, &surface), Some(false));
    }

    /// The boundary of a cylinder wall 300 degrees round, one unit tall: run
    /// counter-clockwise in `(u, v)` it bounds the wall and is kept, run
    /// clockwise it is the wall's outside, whichever way the face itself
    /// winds, and a loop holding a section is never the outside.
    #[test]
    fn a_periodic_walls_outside_is_its_boundary_run_clockwise() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        let z = Vec3::new(0.0, 0.0, 1.0);
        let cyl = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z, 1.0).unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let at = |u: f64, v: f64| surface.evaluate(u, v).unwrap();
        let (u0, u1) = (0.5, 5.8);
        let rim =
            |v: f64| EdgeCurve::Circle(Circle3D::new(Point3::new(0.0, 0.0, v), z, 1.0).unwrap());
        let edge =
            |curve: EdgeCurve, a: (f64, f64), b: (f64, f64), forward: bool| OrientedPCurveEdge {
                curve_3d: curve,
                pcurve: dummy_pcurve(),
                start_uv: Point2::new(a.0, a.1),
                end_uv: Point2::new(b.0, b.1),
                start_3d: at(a.0, a.1),
                end_3d: at(b.0, b.1),
                forward,
                source_edge_idx: None,
                pave_block_id: None,
            };
        let wall = vec![
            edge(rim(0.0), (u0, 0.0), (u1, 0.0), true),
            edge(EdgeCurve::Line, (u1, 0.0), (u1, 1.0), true),
            edge(rim(1.0), (u1, 1.0), (u0, 1.0), false),
            edge(EdgeCurve::Line, (u0, 1.0), (u0, 0.0), true),
        ];
        let outside: Vec<OrientedPCurveEdge> = wall
            .iter()
            .rev()
            .map(|e| OrientedPCurveEdge {
                start_uv: e.end_uv,
                end_uv: e.start_uv,
                start_3d: e.end_3d,
                end_3d: e.start_3d,
                forward: !e.forward,
                ..e.clone()
            })
            .collect();
        assert!(!is_periodic_face_outside(&wall, &surface));
        assert!(is_periodic_face_outside(&outside, &surface));
        let mut sectioned = outside;
        sectioned[1].source_edge_idx = Some(0);
        assert!(!is_periodic_face_outside(&sectioned, &surface));
    }

    /// The hole-dangling rescue: a fence chain whose two loose ends dangle
    /// inside an inner wire must be trimmed at its exact rim crossings and
    /// closed with the rim arc between the anchors (through the side away
    /// from the chain), yielding a fully degree-2 loop web. The hole edge is
    /// a closed circle (seam vertex), the codebase's common hole shape.
    #[test]
    fn hole_dangling_chain_is_rescued_with_a_rim_connector() {
        use brepkit_math::curves::Circle3D;
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let face = topo.face(face_id).unwrap();
        let surface = face.surface().clone();
        let wire_pts = collect_wire_points(&topo, face.outer_wire());
        let normal = extract_plane_normal(&surface);
        let frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        let boundary =
            boundary_edges_to_pcurve(&topo, face.outer_wire(), &surface, &wire_pts, Some(&frame));

        let center = Point3::new(0.5, 0.5, 0.0);
        let r = 0.2;
        let circle = Circle3D::new(center, brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        let seam = Point3::new(0.7, 0.5, 0.0);
        let hole = vec![OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(circle),
            pcurve: brepkit_math::curves2d::Curve2D::Circle(
                brepkit_math::curves2d::Circle2D::new(frame.project(center), r).unwrap(),
            ),
            start_uv: frame.project(seam),
            end_uv: frame.project(seam),
            start_3d: seam,
            end_3d: seam,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        }];

        // U-shaped chain; both loose ends dangle inside the hole.
        let sections = [
            line_section(Point3::new(0.1, 0.4, 0.0), Point3::new(0.45, 0.4, 0.0)),
            line_section(Point3::new(0.1, 0.4, 0.0), Point3::new(0.1, 0.6, 0.0)),
            line_section(Point3::new(0.1, 0.6, 0.0), Point3::new(0.45, 0.6, 0.0)),
        ];
        let rescued = plane_internal_line_loops(&sections, &frame, &boundary, &[hole], &[], 1e-7)
            .expect("chain must be rescued");
        assert_eq!(rescued.len(), 4, "3 trimmed sections + 1 rim connector");
        let connector = rescued
            .iter()
            .find(|s| matches!(s.curve_3d, EdgeCurve::Circle(_)))
            .expect("rim connector arc");
        for p in [connector.start, connector.end] {
            assert!(
                ((p - center).length() - r).abs() < 1e-6,
                "connector anchors on the rim"
            );
        }
        // Trimmed line ends sit exactly on the rim; x = 0.5 - sqrt(r^2 - 0.1^2).
        let x_rim = 0.5 - (r * r - 0.01_f64).sqrt();
        for s in &rescued {
            if matches!(s.curve_3d, EdgeCurve::Line) {
                for p in [s.start, s.end] {
                    let on_rim = ((p - center).length() - r).abs() < 1e-6;
                    let inside = (p - center).length() < r - 1e-6;
                    assert!(!inside, "no endpoint may remain inside the hole");
                    if on_rim {
                        assert!((p.x() - x_rim).abs() < 1e-6);
                    }
                }
            }
        }
    }

    /// A NON-periodic all-Line-boundary NURBS quad (a ruled strut's bilinear
    /// side patch) crossed by two disjoint transversal sections must split
    /// into three pieces via the generic arrangement. The no-seam shortcut
    /// used to intercept every all-Line-boundary curved face, and its
    /// disjoint-section fallback is sphere-only — the quad came back
    /// unsplit, the in-wedge piece classified whole, and the fuse shelled
    /// apart (the kumiko diagonal strut).
    #[test]
    fn nonperiodic_line_bounded_nurbs_quad_splits_by_disjoint_sections() {
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let corners = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let vids: Vec<_> = corners
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        let eids: Vec<_> = (0..4)
            .map(|i| topo.add_edge(Edge::new(vids[i], vids[(i + 1) % 4], EdgeCurve::Line)))
            .collect();
        let wire = Wire::new(
            eids.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let surface = brepkit_math::nurbs::surface::NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![vec![corners[0], corners[3]], vec![corners[1], corners[2]]],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();
        let face_id = topo.add_face(Face::new(
            wid,
            vec![],
            brepkit_topology::face::FaceSurface::Nurbs(surface),
        ));

        let uv_line_section = |x: f64| -> SectionEdge {
            SectionEdge {
                curve_3d: EdgeCurve::Line,
                pcurve_a: brepkit_math::curves2d::Curve2D::Line(
                    Line2D::new(Point2::new(x, 0.0), Vec2::new(0.0, 1.0)).unwrap(),
                ),
                pcurve_b: dummy_pcurve(),
                start: Point3::new(x, 0.0, 0.0),
                end: Point3::new(x, 1.0, 0.0),
                start_uv_a: Some(Point2::new(x, 0.0)),
                end_uv_a: Some(Point2::new(x, 1.0)),
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
                pave_block_id: None,
            }
        };
        let sections = [uv_line_section(0.3), uv_line_section(0.6)];
        let tol = brepkit_math::tolerance::Tolerance::new();
        let edge_images: std::collections::HashMap<
            brepkit_topology::edge::EdgeId,
            Vec<brepkit_topology::edge::EdgeId>,
        > = std::collections::HashMap::new();
        let subs = split_face_2d(
            &topo,
            face_id,
            &sections,
            Rank::A,
            &tol,
            None,
            None,
            &edge_images,
            &std::collections::BTreeSet::new(),
            None,
        );
        assert_eq!(
            subs.len(),
            3,
            "two disjoint transversal sections must split the quad into 3 pieces"
        );
    }

    /// A square plate split by a T of two sections, a square hole clear of both
    /// in one of its three parts. That part keeps the hole, the others stay
    /// whole: however many pieces a face splits into, an untouched hole survives.
    #[test]
    fn a_t_of_sections_leaves_a_hole_clear_of_it_in_its_own_piece() {
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let mut loop_of = |corners: [(f64, f64); 4]| {
            let vertices =
                corners.map(|(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)));
            let edges: Vec<_> = (0..4)
                .map(|i| {
                    let edge = Edge::new(vertices[i], vertices[(i + 1) % 4], EdgeCurve::Line);
                    OrientedEdge::new(topo.add_edge(edge), true)
                })
                .collect();
            topo.add_wire(Wire::new(edges, true).unwrap())
        };
        let outer = loop_of([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        // Over the point the piece's outer corners average to, (2.5, 4).
        let hole = loop_of([(2.0, 3.5), (2.0, 4.5), (3.0, 4.5), (3.0, 3.5)]);
        let face = topo.add_face(Face::new(
            outer,
            vec![hole],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let sections = [
            line_section(Point3::new(0.0, 8.0, 0.0), Point3::new(10.0, 8.0, 0.0)),
            line_section(Point3::new(5.0, 8.0, 0.0), Point3::new(5.0, 0.0, 0.0)),
        ];
        let subs = split_face_2d(
            &topo,
            face,
            &sections,
            Rank::A,
            &brepkit_math::tolerance::Tolerance::new(),
            None,
            None,
            &std::collections::HashMap::new(),
            &std::collections::BTreeSet::new(),
            None,
        );
        assert_eq!(subs.len(), 3);
        let holed: Vec<_> = subs.iter().filter(|s| !s.inner_wires.is_empty()).collect();
        assert_eq!(holed.len(), 1, "the hole must land in exactly one piece");
        let lower_left = holed[0]
            .outer_wire
            .iter()
            .all(|e| e.start_3d.x() <= 5.0 + 1e-9 && e.start_3d.y() <= 8.0 + 1e-9);
        assert!(lower_left, "the hole belongs to the piece around it");
        assert!(
            holed[0].precomputed_interior.is_none(),
            "an interior point read without the hole can lie in it"
        );
    }

    /// A section across a hole, or ending in it, leaves no piece that can hold
    /// it whole, so the T path declines and the hole goes on to the general
    /// splitter. One clear of it leaves it whole.
    #[test]
    fn a_section_across_a_hole_reaches_it() {
        let frame =
            PlaneFrame::from_normal_and_point(Vec3::new(0.0, 0.0, 1.0), Point3::new(0.0, 0.0, 0.0));
        let corners = [(4.0, 7.0), (4.0, 9.0), (6.0, 9.0), (6.0, 7.0)];
        let hole: Vec<OrientedPCurveEdge> = (0..4)
            .map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                let (start_3d, end_3d) = (Point3::new(a.0, a.1, 0.0), Point3::new(b.0, b.1, 0.0));
                OrientedPCurveEdge {
                    curve_3d: EdgeCurve::Line,
                    pcurve: dummy_pcurve(),
                    start_uv: frame.project(start_3d),
                    end_uv: frame.project(end_3d),
                    start_3d,
                    end_3d,
                    forward: true,
                    source_edge_idx: None,
                    pave_block_id: None,
                }
            })
            .collect();
        let holes = [hole];
        let reaches = |a: (f64, f64), b: (f64, f64)| {
            sections_reach_a_hole(
                &[line_section(
                    Point3::new(a.0, a.1, 0.0),
                    Point3::new(b.0, b.1, 0.0),
                )],
                &holes,
                &frame,
            )
        };
        assert!(reaches((0.0, 8.0), (10.0, 8.0)), "across it");
        assert!(reaches((5.0, 8.0), (5.0, 0.0)), "ending in it");
        assert!(!reaches((0.0, 2.0), (10.0, 2.0)), "clear below it");
        assert!(!reaches((3.0, 0.0), (3.0, 10.0)), "clear beside it");
    }

    fn line_section(start: Point3, end: Point3) -> SectionEdge {
        SectionEdge {
            curve_3d: EdgeCurve::Line,
            pcurve_a: dummy_pcurve(),
            pcurve_b: dummy_pcurve(),
            start,
            end,
            start_uv_a: None,
            end_uv_a: None,
            start_uv_b: None,
            end_uv_b: None,
            target_face: None,
            pave_block_id: None,
        }
    }

    /// The face-64 slit-web regression: a plane face cut by a straight
    /// section plus a marched-NURBS section whose endpoint forms a
    /// T-junction MID-SPAN of the straight one, carrying ~1e-6 of curve-fit
    /// error. The chord-based subdivision split the line at a phantom point
    /// the arc side rejected, the half-edge graph desynchronized, and the
    /// tracer's dangling-edge retreat emitted regions with SLIT (doubled)
    /// edges. With true line×arc crossings and exact-UV co-registration the
    /// arrangement yields the three real regions with every edge used once
    /// per region.
    #[test]
    fn arrangement_splits_fit_error_t_junction_web_without_slits() {
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let face = topo.face(face_id).unwrap();
        let surface = face.surface().clone();
        let wire_pts = collect_wire_points(&topo, face.outer_wire());
        let normal = extract_plane_normal(&surface);
        let frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        let boundary =
            boundary_edges_to_pcurve(&topo, face.outer_wire(), &surface, &wire_pts, Some(&frame));

        // Straight section spanning the square at y = 0.5.
        let s_line = line_section(Point3::new(0.0, 0.5, 0.0), Point3::new(1.0, 0.5, 0.0));

        // Marched section from a T mid-span of the line (with 1e-6 fit error
        // off it) down to the right boundary edge, bulging like a conic.
        let pts = [
            Point3::new(0.5, 0.5 + 1.0e-6, 0.0),
            Point3::new(0.63, 0.44, 0.0),
            Point3::new(0.75, 0.385, 0.0),
            Point3::new(0.87, 0.315, 0.0),
            Point3::new(1.0, 0.25, 0.0),
        ];
        let nurbs = brepkit_math::nurbs::fitting::interpolate(&pts, 3).unwrap();
        let s_arc = SectionEdge {
            curve_3d: EdgeCurve::NurbsCurve(nurbs),
            pcurve_a: dummy_pcurve(),
            pcurve_b: dummy_pcurve(),
            start: pts[0],
            end: pts[4],
            start_uv_a: None,
            end_uv_a: None,
            start_uv_b: None,
            end_uv_b: None,
            target_face: None,
            pave_block_id: None,
        };

        let sections = vec![s_line, s_arc];
        let result = split_plane_face_by_arrangement(
            &surface,
            &boundary,
            &sections,
            Rank::A,
            false,
            face_id,
            &frame,
            1.0e-7,
            None,
        )
        .expect("arrangement must trace the T-junction web");

        // Upper half, lower-left region, lower-right region.
        assert_eq!(result.len(), 3, "expected the three real regions");

        // No region may traverse the same undirected edge twice (a slit).
        for sub in &result {
            let mut seen = std::collections::HashMap::new();
            for e in &sub.outer_wire {
                let q = |p: Point3| {
                    (
                        (p.x() * 1.0e6).round() as i64,
                        (p.y() * 1.0e6).round() as i64,
                        (p.z() * 1.0e6).round() as i64,
                    )
                };
                let (a, b) = (q(e.start_3d), q(e.end_3d));
                let key = if a <= b { (a, b) } else { (b, a) };
                *seen.entry(key).or_insert(0usize) += 1;
            }
            assert!(
                seen.values().all(|&c| c <= 1),
                "region wire traverses an edge twice (slit): {:?}",
                seen.iter().filter(|&(_, &c)| c > 1).collect::<Vec<_>>()
            );
        }
    }

    /// A stadium of long arcs (its chord polygon a thin rectangle) holding a
    /// circle of four quarter arcs, a component apart from it (a disc's rim
    /// woven through two holes): the stadium's region has the circle as its
    /// hole, and the disc inside is a region of its own. Read by chords, the
    /// circle's unbounded cycle had the larger area and was dropped in place
    /// of the stadium's, which then came out twice, facing both ways, and
    /// neither with the hole.
    #[test]
    fn a_component_inside_an_outline_of_long_arcs_is_its_hole() {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let x = Vec3::new(1.0, 0.0, 0.0);
        let (half, r) = (1.0, 20.0);
        let mut vertex = |p: Point3| topo.add_vertex(Vertex::new(p, 1e-7));
        let corners = [
            Point3::new(half, -r, 0.0),
            Point3::new(half, r, 0.0),
            Point3::new(-half, r, 0.0),
            Point3::new(-half, -r, 0.0),
        ]
        .map(&mut vertex);
        let rim = [0.0, 1.0, 2.0, 3.0].map(|k| {
            let a = k * std::f64::consts::FRAC_PI_2;
            vertex(Point3::new(8.0 * a.cos(), 8.0 * a.sin(), 0.0))
        });
        let circle = |topo: &mut Topology, c: Point3, radius: f64, a, b| {
            let curve = Circle3D::new_with_ref(c, z, radius, x).unwrap();
            topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(curve)))
        };
        let right = circle(
            &mut topo,
            Point3::new(half, 0.0, 0.0),
            r,
            corners[0],
            corners[1],
        );
        let top = topo.add_edge(Edge::new(corners[1], corners[2], EdgeCurve::Line));
        let left = circle(
            &mut topo,
            Point3::new(-half, 0.0, 0.0),
            r,
            corners[2],
            corners[3],
        );
        let bottom = topo.add_edge(Edge::new(corners[3], corners[0], EdgeCurve::Line));
        let outline = topo.add_wire(
            Wire::new(
                [right, top, left, bottom]
                    .map(|e| OrientedEdge::new(e, true))
                    .to_vec(),
                true,
            )
            .unwrap(),
        );
        let quarters: Vec<_> = (0..4)
            .map(|k| {
                circle(
                    &mut topo,
                    Point3::new(0.0, 0.0, 0.0),
                    8.0,
                    rim[k],
                    rim[(k + 1) % 4],
                )
            })
            .collect();
        let ring = topo.add_wire(
            Wire::new(
                quarters
                    .iter()
                    .map(|&e| OrientedEdge::new(e, true))
                    .collect(),
                true,
            )
            .unwrap(),
        );
        let surface = FaceSurface::Plane { normal: z, d: 0.0 };
        let face_id = topo.add_face(Face::new(outline, vec![], surface.clone()));
        let wire_pts = collect_wire_points(&topo, outline);
        let frame = PlaneFrame::from_plane_face(z, &wire_pts);
        let mut edges = boundary_edges_to_pcurve(&topo, outline, &surface, &wire_pts, Some(&frame));
        edges.extend(boundary_edges_to_pcurve(
            &topo,
            ring,
            &surface,
            &wire_pts,
            Some(&frame),
        ));
        let regions = arrangement_regions_from_combined(
            &surface,
            &edges,
            &[],
            Rank::A,
            false,
            face_id,
            &frame,
            1e-7,
        )
        .expect("the arrangement traces its regions");
        let holes: Vec<usize> = regions.iter().map(|r| r.inner_wires.len()).collect();
        assert_eq!(regions.len(), 2, "regions with holes {holes:?}");
        assert!(
            holes.contains(&1) && holes.contains(&0),
            "regions with holes {holes:?}"
        );
    }

    fn cyl_edge(curve_3d: EdgeCurve, start_3d: Point3, end_3d: Point3) -> OrientedPCurveEdge {
        // The cylinder arrangement derives every coordinate from the 3D endpoints
        // via `project_point`, so the pcurve / UV / flags are unread placeholders.
        OrientedPCurveEdge {
            curve_3d,
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(0.0, 0.0),
            start_3d,
            end_3d,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    /// A box notch removing the `u ∈ [π/2, π]`, `v ∈ [0, 1]` corner of a unit
    /// cylinder band (`z ∈ [0, 2]`). The angular wire builder figure-eights this
    /// partial-overlap cut; the rectilinear arrangement must instead hand back the
    /// kept comb region plus the removed rectangle, each a simple (non-revisiting)
    /// wire in UV.
    #[test]
    fn cylinder_band_partial_notch_splits_into_comb_and_rectangle() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let bottom = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
            .map(EdgeCurve::Circle)
            .unwrap();
        let top = Circle3D::new(Point3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
            .map(EdgeCurve::Circle)
            .unwrap();
        let ring = Circle3D::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
            .map(EdgeCurve::Circle)
            .unwrap();

        // Boundary (seam + rims split at the seam so no edge is a closed circle).
        let boundary = [
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 2.0),
            ),
            cyl_edge(
                bottom.clone(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
            ),
            cyl_edge(
                bottom,
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ),
            cyl_edge(
                top.clone(),
                Point3::new(1.0, 0.0, 2.0),
                Point3::new(-1.0, 0.0, 2.0),
            ),
            cyl_edge(top, Point3::new(-1.0, 0.0, 2.0), Point3::new(1.0, 0.0, 2.0)),
        ];
        let n_boundary = boundary.len();
        // Section: two side generators + the notch's top ring.
        let sections = [
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 1.0),
            ),
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(-1.0, 0.0, 1.0),
            ),
            cyl_edge(
                ring,
                Point3::new(0.0, 1.0, 1.0),
                Point3::new(-1.0, 0.0, 1.0),
            ),
        ];
        let all_edges: Vec<OrientedPCurveEdge> = boundary.into_iter().chain(sections).collect();

        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let result = split_cylinder_band_by_arrangement(
            &surface,
            &all_edges,
            n_boundary,
            Rank::A,
            false,
            face_id,
            1.0e-7,
        )
        .expect("cylinder band arrangement must trace the notch");

        assert_eq!(
            result.len(),
            2,
            "expected the kept comb + the removed rectangle"
        );

        // Neither region may revisit a UV vertex (a figure-eight): the seam sits
        // at both u = u_s and u = u_s + 2π, which are distinct UV points, so a
        // correct partition never lands two edges on the same UV vertex.
        for sub in &result {
            let mut seen = std::collections::HashSet::new();
            for e in &sub.outer_wire {
                let key = (
                    (e.start_uv.x() * 1.0e6).round() as i64,
                    (e.start_uv.y() * 1.0e6).round() as i64,
                );
                assert!(
                    seen.insert(key),
                    "region wire revisits a UV vertex (figure-eight)"
                );
            }
            assert!(sub.outer_wire.len() >= 3, "degenerate region wire");
        }

        // The removed rectangle spans v ∈ [0, 1]; the kept comb reaches v = 2.
        let v_extent = |sub: &SplitSubFace| -> f64 {
            sub.outer_wire
                .iter()
                .map(|e| e.start_uv.y())
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let mut extents: Vec<f64> = result.iter().map(v_extent).collect();
        extents.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            extents[0] <= 1.0 + 1.0e-6,
            "removed rectangle should stay at v <= 1"
        );
        assert!(
            extents[1] >= 2.0 - 1.0e-6,
            "kept comb should reach the top rim v = 2"
        );
    }

    /// A quadratic hole edge over (0, 0), (1, 1), (2, 0) with weights 1, -0.5
    /// and 1 runs through (1, -1), outside its control polygon's box: the
    /// weave's split still finds it there.
    #[test]
    fn a_spline_hole_edge_with_a_negative_weight_splits_outside_its_control_box() {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::vec::Vec2;
        let spline = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
            ],
            vec![1.0, -0.5, 1.0],
        )
        .unwrap();
        let low = Point3::new(1.0, -1.0, 0.0);
        assert!((spline.evaluate(0.5) - low).length() < 1e-12);
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::NurbsCurve(spline),
            pcurve: Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(2.0, 0.0)).unwrap()),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(2.0, 0.0),
            start_3d: Point3::new(0.0, 0.0, 0.0),
            end_3d: Point3::new(2.0, 0.0, 0.0),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let pieces =
            edge_splitting::split_hole_edges_at_3d_points(vec![edge], &[low], None, &plane, 1e-7);
        assert_eq!(pieces.len(), 2, "the spline did not split at (1, -1)");
        assert!((pieces[0].end_3d - low).length() < 1e-7);
        assert!((pieces[1].start_3d - low).length() < 1e-7);
    }

    /// A hole edge that is a degree-1 spline with a spike in one short knot
    /// span holds the spike's tip and its sides, where fixed samples along
    /// the parameter step over the span.
    #[test]
    fn a_point_on_a_short_span_of_a_spline_hole_edge_is_on_it() {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::vec::Vec2;
        let spline = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.503, 0.504, 0.505, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.503, 0.0, 0.0),
                Point3::new(0.504, 0.1, 0.0),
                Point3::new(0.505, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ],
            vec![1.0; 5],
        )
        .unwrap();
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::NurbsCurve(spline),
            pcurve: Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(1.0, 0.0),
            start_3d: Point3::new(0.0, 0.0, 0.0),
            end_3d: Point3::new(1.0, 0.0, 0.0),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let band = 1e-5;
        assert!(point_on_oriented_edge(
            &edge,
            Point3::new(0.504, 0.1, 0.0),
            band
        ));
        assert!(point_on_oriented_edge(
            &edge,
            Point3::new(0.5035, 0.05, 0.0),
            band
        ));
        assert!(point_on_oriented_edge(
            &edge,
            Point3::new(0.75, 0.0, 0.0),
            band
        ));
        assert!(!point_on_oriented_edge(
            &edge,
            Point3::new(0.504, 0.2, 0.0),
            band
        ));
        assert!(!point_on_oriented_edge(
            &edge,
            Point3::new(0.75, 0.01, 0.0),
            band
        ));
        // The weave's split finds the tip where the contact test does.
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let tip = Point3::new(0.504, 0.1, 0.0);
        let pieces = edge_splitting::split_hole_edges_at_3d_points(
            vec![edge.clone()],
            &[tip],
            None,
            &plane,
            1e-7,
        );
        assert_eq!(pieces.len(), 2, "the spline did not split at its tip");
        assert!((pieces[0].end_3d - tip).length() < 1e-7);
        assert!((pieces[1].start_3d - tip).length() < 1e-7);
        // An edge on the part before the spike holds none of it.
        let mut first_part = edge;
        first_part.end_3d = Point3::new(0.503, 0.0, 0.0);
        let on_first = |p: Point3| point_on_oriented_edge(&first_part, p, band);
        assert!(on_first(Point3::new(0.25, 0.0, 0.0)));
        assert!(!on_first(Point3::new(0.504, 0.1, 0.0)));
        assert!(!on_first(Point3::new(0.75, 0.0, 0.0)));
    }

    /// A window on a cylinder of radius 38 over one radian (u = 0.2 to 1.2)
    /// from z = 3 to 4.5, the fan bore's scale: its bottom rim, its right
    /// ruling, its top rim run back, and its left ruling.
    fn bore_window() -> (FaceSurface, Vec<OrientedPCurveEdge>) {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::surfaces::CylindricalSurface;
        let surface = FaceSurface::Cylinder(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 38.0)
                .unwrap(),
        );
        let at = |u: f64, v: f64| surface.evaluate(u, v).unwrap();
        let rim = |z: f64| {
            EdgeCurve::Circle(
                Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), 38.0).unwrap(),
            )
        };
        let edge = |curve: EdgeCurve, a: Point2, b: Point2, forward: bool| OrientedPCurveEdge {
            curve_3d: curve,
            pcurve: Curve2D::Line(Line2D::new(a, b - a).unwrap()),
            start_uv: a,
            end_uv: b,
            start_3d: at(a.x(), a.y()),
            end_3d: at(b.x(), b.y()),
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let (u0, u1, z0, z1) = (0.2, 1.2, 3.0, 4.5);
        let hole = vec![
            edge(rim(z0), Point2::new(u0, z0), Point2::new(u1, z0), true),
            edge(
                EdgeCurve::Line,
                Point2::new(u1, z0),
                Point2::new(u1, z1),
                true,
            ),
            edge(rim(z1), Point2::new(u1, z1), Point2::new(u0, z1), false),
            edge(
                EdgeCurve::Line,
                Point2::new(u0, z1),
                Point2::new(u0, z0),
                true,
            ),
        ];
        (surface, hole)
    }

    /// A point between a chord of the first sampling interval and the arc,
    /// 1e-4 off the chord's middle, lies about 1.06e-3 off the bore window's
    /// rim: it is off the rim, as a point 2e-5 outside is, and a point 5e-6
    /// outside is on it.
    #[test]
    fn a_point_inside_a_long_arcs_sampling_chord_is_off_the_arc() {
        let (surface, hole) = bore_window();
        let at = |u: f64, v: f64| surface.evaluate(u, v).unwrap();
        let out = |u: f64, by: f64| {
            let p = at(u, 3.0);
            p + Vec3::new(p.x(), p.y(), 0.0).normalize().unwrap() * by
        };
        let step = 1.0 / 64.0;
        let chord_middle = at(0.2, 3.0) + (at(0.2 + step, 3.0) - at(0.2, 3.0)) * 0.5;
        let lens = chord_middle
            + (at(0.2 + 0.5 * step, 3.0) - chord_middle)
                .normalize()
                .unwrap()
                * 1e-4;
        let band = 1e-5;
        for rim in [&hole[0], &hole[2]] {
            let z = rim.start_3d.z();
            let lift = |p: Point3| Point3::new(p.x(), p.y(), z);
            assert!(!point_on_oriented_edge(rim, lift(lens), band));
            assert!(point_on_oriented_edge(
                rim,
                lift(at(0.2 + 0.5 * step, 3.0)),
                band
            ));
            assert!(point_on_oriented_edge(rim, lift(out(0.7, 5e-6)), band));
            assert!(!point_on_oriented_edge(rim, lift(out(0.7, 2e-5)), band));
            assert!(!point_on_oriented_edge(rim, lift(at(1.4, 3.0)), band));
        }
    }

    /// The bore window's bottom rim with three section ends: one inside a
    /// sampling chord (off the rim), one 5e-7 outside the rim (in the weld
    /// band, so the rim splits at its foot), then one on it at u = 0.9. The
    /// two on the rim have UVs a period on, and the hole moves a period to
    /// them.
    #[test]
    fn a_curved_hole_weaves_at_its_contact_after_near_misses() {
        let (surface, hole) = bore_window();
        let at = |u: f64, v: f64| surface.evaluate(u, v).unwrap();
        let step = 1.0 / 64.0;
        let chord_middle = at(0.2, 3.0) + (at(0.2 + step, 3.0) - at(0.2, 3.0)) * 0.5;
        let lens = chord_middle
            + (at(0.2 + 0.5 * step, 3.0) - chord_middle)
                .normalize()
                .unwrap()
                * 1e-4;
        let near = {
            let p = at(0.6, 3.0);
            p + Vec3::new(p.x(), p.y(), 0.0).normalize().unwrap() * 5e-7
        };
        let tau = std::f64::consts::TAU;
        let contact = at(0.9, 3.0);
        let uv = |p: Point3| {
            let (u, v) = surface.project_point(p).unwrap();
            Point2::new(u, v)
        };
        let ends = [
            (lens, uv(lens)),
            (near, uv(near) + Vec2::new(tau, 0.0)),
            (contact, Point2::new(0.9 + tau, 3.0)),
        ];
        let pieces =
            weave_curved_hole(&hole, &ends, &surface, 1e-7).expect("the hole stayed whole");
        assert_eq!(pieces.len(), 6);
        let foot = at(0.6, 3.0);
        assert!(
            pieces.iter().any(|e| (e.end_3d - foot).length() < 1e-9),
            "no piece ends at the foot of the end in the weld band"
        );
        let into = pieces
            .iter()
            .find(|e| (e.end_3d - contact).length() < 1e-9)
            .expect("no piece ends at the contact");
        assert!((into.end_uv - Point2::new(0.9 + tau, 3.0)).length() < 1e-9);
        assert!(
            pieces
                .iter()
                .all(|e| e.start_uv.x() > tau && e.start_uv.x() < tau + 1.3),
            "the hole did not move to the contact's copy"
        );
    }

    /// A woven hole's piece whose pcurve is a NURBS arc, moved a period along
    /// u to the face's window: its pcurve moves with its ends, so it still
    /// starts and ends where the edge does.
    #[test]
    fn a_hole_moved_a_period_keeps_its_curved_pcurve_on_its_ends() {
        use brepkit_math::curves2d::{Curve2D, NurbsCurve2D};
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let arc = NurbsCurve2D::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point2::new(3.7, 3.0),
                Point2::new(3.75, 3.6),
                Point2::new(3.8, 3.0),
            ],
            vec![1.0, w, 1.0],
        )
        .unwrap();
        let (start, end) = (arc.evaluate(0.0), arc.evaluate(1.0));
        let mut edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: Curve2D::Nurbs(arc),
            start_uv: start,
            end_uv: end,
            start_3d: Point3::new(0.0, 0.0, 0.0),
            end_3d: Point3::new(1.0, 0.0, 0.0),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let du = std::f64::consts::TAU;
        move_edge_in_uv(&mut edge, du, 0.0);
        assert!((edge.start_uv.x() - (start.x() + du)).abs() < 1e-12);
        assert!(matches!(edge.pcurve, Curve2D::Nurbs(_)), "not a NURBS arc");
        if let Curve2D::Nurbs(moved) = &edge.pcurve {
            assert!((moved.evaluate(0.0) - edge.start_uv).length() < 1e-12);
            assert!((moved.evaluate(1.0) - edge.end_uv).length() < 1e-12);
            let middle = moved.evaluate(0.5);
            assert!(
                (middle.x() - (3.75 + du)).abs() < 1e-9 && middle.y() > 3.1,
                "{middle:?}"
            );
        }
    }

    /// [`cylinder_band_partial_notch_splits_into_comb_and_rectangle`] with the
    /// seam split at v = 1 ahead of time: the band's two copies of the seam,
    /// at u = 0 and at u = 2 pi, split at the same heights.
    #[test]
    fn cylinder_band_splits_both_copies_of_a_split_seam_alike() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let circle = |z: f64| {
            Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .map(EdgeCurve::Circle)
                .unwrap()
        };
        let boundary = [
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 1.0),
            ),
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(1.0, 0.0, 1.0),
                Point3::new(1.0, 0.0, 2.0),
            ),
            cyl_edge(
                circle(0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
            ),
            cyl_edge(
                circle(0.0),
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ),
            cyl_edge(
                circle(2.0),
                Point3::new(1.0, 0.0, 2.0),
                Point3::new(-1.0, 0.0, 2.0),
            ),
            cyl_edge(
                circle(2.0),
                Point3::new(-1.0, 0.0, 2.0),
                Point3::new(1.0, 0.0, 2.0),
            ),
        ];
        let n_boundary = boundary.len();
        let sections = [
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 1.0),
            ),
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(-1.0, 0.0, 1.0),
            ),
            cyl_edge(
                circle(1.0),
                Point3::new(0.0, 1.0, 1.0),
                Point3::new(-1.0, 0.0, 1.0),
            ),
        ];
        let all_edges: Vec<OrientedPCurveEdge> = boundary.into_iter().chain(sections).collect();

        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let result = split_cylinder_band_by_arrangement(
            &surface,
            &all_edges,
            n_boundary,
            Rank::A,
            false,
            face_id,
            1.0e-7,
        )
        .expect("cylinder band arrangement must trace the notch");

        let seam_u: Vec<f64> = result
            .iter()
            .flat_map(|sub| &sub.outer_wire)
            .filter(|e| matches!(e.curve_3d, EdgeCurve::Line))
            .filter(|e| (e.start_3d.x() - 1.0).abs() < 1e-9 && e.start_3d.y().abs() < 1e-9)
            .map(|e| e.start_uv.x())
            .collect();
        let (lo, hi) = seam_u
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &u| {
                (lo.min(u), hi.max(u))
            });
        let heights = |u: f64| {
            let mut vs: Vec<i64> = result
                .iter()
                .flat_map(|sub| &sub.outer_wire)
                .filter(|e| matches!(e.curve_3d, EdgeCurve::Line))
                .filter(|e| (e.start_uv.x() - u).abs() < 1e-6 && (e.end_uv.x() - u).abs() < 1e-6)
                .flat_map(|e| [e.start_uv.y(), e.end_uv.y()])
                .map(|v| (v * 1.0e6).round() as i64)
                .collect();
            vs.sort_unstable();
            vs.dedup();
            vs
        };
        assert!(hi - lo > 6.0, "both copies of the seam bound a region");
        assert_eq!(heights(lo), heights(hi), "the seam's copies split apart");
        assert_eq!(heights(lo), vec![0, 1_000_000, 2_000_000]);
    }

    /// A tilted (non-axis-aligned) section makes the cut non-rectilinear, so the
    /// function must defer (`None`) and let the greedy path keep the face.
    #[test]
    fn cylinder_band_arrangement_defers_on_non_rectilinear_section() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let bottom = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
            .map(EdgeCurve::Circle)
            .unwrap();
        let top = Circle3D::new(Point3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
            .map(EdgeCurve::Circle)
            .unwrap();

        let boundary = [
            cyl_edge(
                EdgeCurve::Line,
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 2.0),
            ),
            cyl_edge(
                bottom.clone(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
            ),
            cyl_edge(
                bottom,
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ),
            cyl_edge(
                top.clone(),
                Point3::new(1.0, 0.0, 2.0),
                Point3::new(-1.0, 0.0, 2.0),
            ),
            cyl_edge(top, Point3::new(-1.0, 0.0, 2.0), Point3::new(1.0, 0.0, 2.0)),
        ];
        let n_boundary = boundary.len();
        // A "line" whose endpoints project to different u (a helix chord).
        let sections = [cyl_edge(
            EdgeCurve::Line,
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 1.0),
        )];
        let all_edges: Vec<OrientedPCurveEdge> = boundary.into_iter().chain(sections).collect();

        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        assert!(
            split_cylinder_band_by_arrangement(
                &surface,
                &all_edges,
                n_boundary,
                Rank::A,
                false,
                face_id,
                1.0e-7,
            )
            .is_none(),
            "a non-axis-aligned section must defer to the greedy path"
        );
    }

    /// A cap circle whose centre lies inside a sub-face hole is air (the rim
    /// of a drill emerging inside an existing opening) and must be dropped,
    /// while a genuine cap over material is carved into disc + remainder.
    #[test]
    fn distribute_cap_circles_drops_caps_inside_holes() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::curves2d::{Circle2D, Curve2D};

        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let face = topo.face(face_id).unwrap();
        let surface = face.surface().clone();
        let wire_pts = collect_wire_points(&topo, face.outer_wire());
        let normal = extract_plane_normal(&surface);
        let frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        let boundary =
            boundary_edges_to_pcurve(&topo, face.outer_wire(), &surface, &wire_pts, Some(&frame));

        // Square hole around (0.3, 0.3). All UVs must come from the same
        // frame projection the boundary and cap centres use.
        let hole_corners = [
            Point3::new(0.15, 0.15, 0.0),
            Point3::new(0.45, 0.15, 0.0),
            Point3::new(0.45, 0.45, 0.0),
            Point3::new(0.15, 0.45, 0.0),
        ];
        let hole: Vec<OrientedPCurveEdge> = (0..4)
            .map(|i| {
                let (a, b) = (hole_corners[i], hole_corners[(i + 1) % 4]);
                OrientedPCurveEdge {
                    curve_3d: EdgeCurve::Line,
                    pcurve: dummy_pcurve(),
                    start_uv: frame.project(a),
                    end_uv: frame.project(b),
                    start_3d: a,
                    end_3d: b,
                    forward: true,
                    source_edge_idx: None,
                    pave_block_id: None,
                }
            })
            .collect();

        let base = SplitSubFace {
            surface,
            outer_wire: boundary,
            inner_wires: vec![hole],
            reversed: false,
            parent: face_id,
            rank: Rank::A,
            precomputed_interior: None,
        };

        let cap = |cx: f64, cy: f64| -> SectionEdge {
            let r = 0.05;
            let circle =
                Circle3D::new(Point3::new(cx, cy, 0.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
            let rim3 = Point3::new(cx + r, cy, 0.0);
            let rim_uv = frame.project(rim3);
            let pcurve =
                Curve2D::Circle(Circle2D::new(frame.project(Point3::new(cx, cy, 0.0)), r).unwrap());
            SectionEdge {
                curve_3d: EdgeCurve::Circle(circle),
                pcurve_a: pcurve.clone(),
                pcurve_b: pcurve,
                start: rim3,
                end: rim3,
                start_uv_a: Some(rim_uv),
                end_uv_a: Some(rim_uv),
                start_uv_b: Some(rim_uv),
                end_uv_b: Some(rim_uv),
                target_face: None,
                pave_block_id: None,
            }
        };
        let in_hole = cap(0.3, 0.3);
        let genuine = cap(0.7, 0.7);
        let centers = [Point3::new(0.3, 0.3, 0.0), Point3::new(0.7, 0.7, 0.0)];

        let result = distribute_cap_circles(
            &topo,
            face_id,
            vec![base],
            &[in_hole, genuine],
            &centers,
            Rank::A,
            Some(&frame),
        );

        assert_eq!(
            result.len(),
            2,
            "genuine cap carves disc + remainder; in-hole cap must be dropped"
        );
        let discs: Vec<&SplitSubFace> = result
            .iter()
            .filter(|sf| {
                sf.outer_wire.len() == 1
                    && matches!(sf.outer_wire[0].curve_3d, EdgeCurve::Circle(_))
            })
            .collect();
        assert_eq!(discs.len(), 1, "exactly one cap disc must be carved");
        let EdgeCurve::Circle(c) = &discs[0].outer_wire[0].curve_3d else {
            unreachable!();
        };
        assert!(
            (c.center() - Point3::new(0.7, 0.7, 0.0)).length() < 1e-9,
            "the carved disc must be the genuine cap, not the in-hole one"
        );
    }

    fn uv_line_edge(a: Point2, b: Point2) -> OrientedPCurveEdge {
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: dummy_pcurve(),
            start_uv: a,
            end_uv: b,
            start_3d: Point3::new(a.x(), a.y(), 0.0),
            end_3d: Point3::new(b.x(), b.y(), 0.0),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    #[test]
    fn pinch_split_separates_figure_eight_into_two_squares() {
        let p = |x: f64, y: f64| Point2::new(x, y);
        let wire = vec![
            uv_line_edge(p(0.0, 0.0), p(1.0, 0.0)),
            uv_line_edge(p(1.0, 0.0), p(1.0, 1.0)),
            uv_line_edge(p(1.0, 1.0), p(2.0, 1.0)),
            uv_line_edge(p(2.0, 1.0), p(2.0, 2.0)),
            uv_line_edge(p(2.0, 2.0), p(1.0, 2.0)),
            uv_line_edge(p(1.0, 2.0), p(1.0, 1.0)),
            uv_line_edge(p(1.0, 1.0), p(0.0, 1.0)),
            uv_line_edge(p(0.0, 1.0), p(0.0, 0.0)),
        ];
        let out = split_loop_at_pinch_vertices(&wire, 1e-7);
        assert_eq!(out.len(), 2, "figure-eight must split into its two cycles");
        for lp in &out {
            assert_eq!(lp.len(), 4);
            let pts = sample_wire_loop_uv(lp);
            assert!((signed_area_2d(&pts).abs() - 1.0).abs() < 1e-9);
        }
        assert!(!wire_loops_self_cross(&out, 1e-7));
    }

    #[test]
    fn pinch_split_drops_pure_out_and_back_excursion() {
        let p = |x: f64, y: f64| Point2::new(x, y);
        let wire = vec![
            uv_line_edge(p(0.0, 0.0), p(1.0, 0.0)),
            uv_line_edge(p(1.0, 0.0), p(1.0, 1.0)),
            uv_line_edge(p(1.0, 1.0), p(2.0, 1.0)),
            uv_line_edge(p(2.0, 1.0), p(1.0, 1.0)),
            uv_line_edge(p(1.0, 1.0), p(0.0, 1.0)),
            uv_line_edge(p(0.0, 1.0), p(0.0, 0.0)),
        ];
        let out = split_loop_at_pinch_vertices(&wire, 1e-7);
        assert_eq!(out.len(), 1, "the zero-area excursion must be dropped");
        assert_eq!(out[0].len(), 4);
        let pts = sample_wire_loop_uv(&out[0]);
        assert!((signed_area_2d(&pts).abs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pinch_split_keeps_simple_loop_whole() {
        let p = |x: f64, y: f64| Point2::new(x, y);
        let wire = vec![
            uv_line_edge(p(0.0, 0.0), p(1.0, 0.0)),
            uv_line_edge(p(1.0, 0.0), p(1.0, 1.0)),
            uv_line_edge(p(1.0, 1.0), p(0.0, 1.0)),
            uv_line_edge(p(0.0, 1.0), p(0.0, 0.0)),
        ];
        let out = split_loop_at_pinch_vertices(&wire, 1e-7);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 4);
    }

    /// The crescent a disc keeps outside a second disc of its radius whose
    /// rim runs through its centre, turned through every degree in its
    /// plane. The longest chord across its sampled polygon starts beside its
    /// seam and runs past the disc's centre along the other rim, between
    /// that arc and its chords; the interior point must lie in the crescent.
    #[test]
    fn a_crescents_interior_point_lies_off_its_inner_arc() {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_3};
        let z = Vec3::new(0.0, 0.0, 1.0);
        let frame = PlaneFrame::from_normal_and_point(z, Point3::new(0.0, 0.0, 0.0));
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let r = 5.0_f64;
        let b = Point3::new(0.0, 0.0, 0.0);
        for degree in 0..360 {
            let turn = f64::from(degree).to_radians();
            let on = |c: Point3, at: f64| {
                Point3::new(
                    r.madd((turn + at).cos(), c.x()),
                    r.madd((turn + at).sin(), c.y()),
                    0.0,
                )
            };
            let a = on(b, 0.0);
            let (seam, p, q) = (on(b, FRAC_PI_2), on(b, FRAC_PI_3), on(b, -FRAC_PI_3));
            let arc =
                |centre: Point3, from: Point3, to: Point3, forward: bool| OrientedPCurveEdge {
                    curve_3d: EdgeCurve::Circle(Circle3D::new(centre, z, r).unwrap()),
                    pcurve: dummy_pcurve(),
                    start_uv: frame.project(from),
                    end_uv: frame.project(to),
                    start_3d: from,
                    end_3d: to,
                    forward,
                    source_edge_idx: None,
                    pave_block_id: None,
                };
            let crescent = SplitSubFace {
                surface: FaceSurface::Plane { normal: z, d: 0.0 },
                outer_wire: vec![
                    arc(b, seam, p, false),
                    arc(a, p, b, true),
                    arc(a, b, q, true),
                    arc(b, q, seam, false),
                ],
                inner_wires: vec![],
                reversed: false,
                parent: face_id,
                rank: Rank::A,
                precomputed_interior: None,
            };
            let pt =
                interior_point_3d(&crescent, Some(&frame), containment::SEED_CLEARANCE).unwrap();
            assert!(
                (pt - b).length() < r && (pt - a).length() > r,
                "turned {degree} degrees: {pt:?}"
            );
        }
    }

    /// The crescent a disc keeps outside a second of its radius whose centre
    /// lies `gap` from its own, turned every 15 degrees in its plane: of
    /// radius 5 with a gap of 0.001, of radius 1e6 with a gap of 1e-4, and of
    /// radius 5 with gaps of 3e-7 and 2.2e-7, whose widest points clear both
    /// arcs by only 1.5e-7 and 1.1e-7, between two halving steps.
    /// Each is far thinner than its arcs' chords' sag, so no candidate taken
    /// across its sampled polygon lies inside it; the interior point must
    /// still lie in the crescent. One of radius 5 with a gap of 1e-8 holds no
    /// point clear of its arcs, and gets none.
    #[test]
    fn a_thin_crescents_interior_point_lies_inside_it() {
        use brepkit_math::curves::Circle3D;
        let z = Vec3::new(0.0, 0.0, 1.0);
        let frame = PlaneFrame::from_normal_and_point(z, Point3::new(0.0, 0.0, 0.0));
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let point_in = |r: f64, gap: f64, degrees: i32| {
            let turn = f64::from(degrees).to_radians();
            let (c, s) = (turn.cos(), turn.sin());
            let at = |x: f64, y: f64| Point3::new(c * x - s * y, s * x + c * y, 0.0);
            let reach = gap.madd(-gap * 0.25, r * r).sqrt();
            let (kept, cut) = (at(0.0, 0.0), at(gap, 0.0));
            let (top, bottom) = (at(gap * 0.5, reach), at(gap * 0.5, -reach));
            let arc =
                |centre: Point3, from: Point3, to: Point3, forward: bool| OrientedPCurveEdge {
                    curve_3d: EdgeCurve::Circle(Circle3D::new(centre, z, r).unwrap()),
                    pcurve: dummy_pcurve(),
                    start_uv: frame.project(from),
                    end_uv: frame.project(to),
                    start_3d: from,
                    end_3d: to,
                    forward,
                    source_edge_idx: None,
                    pave_block_id: None,
                };
            let crescent = SplitSubFace {
                surface: FaceSurface::Plane { normal: z, d: 0.0 },
                outer_wire: vec![arc(kept, top, bottom, true), arc(cut, bottom, top, false)],
                inner_wires: vec![],
                reversed: false,
                parent: face_id,
                rank: Rank::A,
                precomputed_interior: None,
            };
            (
                interior_point_3d(&crescent, Some(&frame), containment::SEED_CLEARANCE),
                kept,
                cut,
            )
        };
        for (r, gap) in [
            (5.0_f64, 0.001_f64),
            (1e6, 1e-4),
            (5.0, 3e-7),
            (5.0, 2.2e-7),
        ] {
            for step in 0..24 {
                let (pt, kept, cut) = point_in(r, gap, step * 15);
                let pt = pt.unwrap();
                assert!(
                    (pt - kept).length() < r && (pt - cut).length() > r,
                    "radius {r}, turned {} degrees: {pt:?}",
                    step * 15
                );
            }
        }
        assert!(point_in(5.0, 1e-8, 0).0.is_none());
    }

    /// Two regions of a square split down its middle, one traced backwards,
    /// do not tile it, though a straight B-spline runs along the split: the
    /// winding is read on NURBS outlines too where the walker's own loops are
    /// broken (a crescent between a rim arc and a section arc whose chords
    /// cross came back from the arrangement wound against its neighbour, and
    /// the result's faces disagreed along it).
    #[test]
    fn regions_wound_both_ways_do_not_cover_a_nurbs_outline() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let frame = PlaneFrame::from_normal_and_point(z, Point3::new(0.0, 0.0, 0.0));
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let edge = |from: Point3, to: Point3, nurbs: bool| OrientedPCurveEdge {
            curve_3d: if nurbs {
                EdgeCurve::NurbsCurve(
                    brepkit_math::nurbs::curve::NurbsCurve::new(
                        1,
                        vec![0.0, 0.0, 1.0, 1.0],
                        vec![from, to],
                        vec![1.0; 2],
                    )
                    .unwrap(),
                )
            } else {
                EdgeCurve::Line
            },
            pcurve: dummy_pcurve(),
            start_uv: frame.project(from),
            end_uv: frame.project(to),
            start_3d: from,
            end_3d: to,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let piece = |corners: &[Point3]| SplitSubFace {
            surface: FaceSurface::Plane { normal: z, d: 0.0 },
            outer_wire: (0..corners.len())
                .map(|k| {
                    let (from, to) = (corners[k], corners[(k + 1) % corners.len()]);
                    edge(
                        from,
                        to,
                        (from.x() - 1.0).abs() < 1e-12 && (to.x() - 1.0).abs() < 1e-12,
                    )
                })
                .collect(),
            inner_wires: vec![],
            reversed: false,
            parent: face_id,
            rank: Rank::A,
            precomputed_interior: None,
        };
        let boundary = piece(&[p(0.0, 0.0), p(2.0, 0.0), p(2.0, 1.0), p(0.0, 1.0)]).outer_wire;
        let left = piece(&[p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]);
        let right = piece(&[p(1.0, 0.0), p(2.0, 0.0), p(2.0, 1.0), p(1.0, 1.0)]);
        let right_backwards = piece(&[p(1.0, 0.0), p(1.0, 1.0), p(2.0, 1.0), p(2.0, 0.0)]);
        assert!(subfaces_cover_boundary(
            &[left.clone(), right],
            &boundary,
            &frame,
            true
        ));
        // Taken over broken walker loops, regions wound both ways refuse;
        // over sound ones (the arrangement only splitting further) they do
        // not.
        assert!(!subfaces_cover_boundary(
            &[left.clone(), right_backwards.clone()],
            &boundary,
            &frame,
            true
        ));
        assert!(subfaces_cover_boundary(
            &[left, right_backwards],
            &boundary,
            &frame,
            false
        ));
    }

    /// Rectangles 1 by 1e-6 and 100 by 1e-5 get a point inside them at the
    /// default clearance, though their sampled polygons ask for more room,
    /// and corner steps stay too near their ends; one 1e-3 by
    /// 1.5e-7 gets one at a clearance of 1e-11, but none at 1e-7, which no
    /// point of it clears.
    #[test]
    fn a_thin_rectangles_interior_point_lies_inside_it() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let frame = PlaneFrame::from_normal_and_point(z, Point3::new(0.0, 0.0, 0.0));
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let rectangle = |w: f64, h: f64| {
            let corners = [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(w, 0.0, 0.0),
                Point3::new(w, h, 0.0),
                Point3::new(0.0, h, 0.0),
            ];
            SplitSubFace {
                surface: FaceSurface::Plane { normal: z, d: 0.0 },
                outer_wire: (0..4)
                    .map(|k| {
                        let (from, to) = (corners[k], corners[(k + 1) % 4]);
                        OrientedPCurveEdge {
                            curve_3d: EdgeCurve::Line,
                            pcurve: dummy_pcurve(),
                            start_uv: frame.project(from),
                            end_uv: frame.project(to),
                            start_3d: from,
                            end_3d: to,
                            forward: true,
                            source_edge_idx: None,
                            pave_block_id: None,
                        }
                    })
                    .collect(),
                inner_wires: vec![],
                reversed: false,
                parent: face_id,
                rank: Rank::A,
                precomputed_interior: None,
            }
        };
        let inside =
            |p: Point3, w: f64, h: f64| p.x() > 0.0 && p.x() < w && p.y() > 0.0 && p.y() < h;
        let p = interior_point_3d(
            &rectangle(1.0, 1e-6),
            Some(&frame),
            containment::SEED_CLEARANCE,
        )
        .unwrap();
        assert!(inside(p, 1.0, 1e-6), "{p:?}");
        let long = rectangle(100.0, 1e-5);
        let p = interior_point_3d(&long, Some(&frame), containment::SEED_CLEARANCE).unwrap();
        assert!(inside(p, 100.0, 1e-5), "{p:?}");
        let thin = rectangle(1e-3, 1.5e-7);
        let p = interior_point_3d(&thin, Some(&frame), 1e-11).unwrap();
        assert!(inside(p, 1e-3, 1.5e-7), "{p:?}");
        assert!(interior_point_3d(&thin, Some(&frame), containment::SEED_CLEARANCE).is_none());
    }

    /// A rectangle in a torus's `(u, v)`, its edges straight between its
    /// corners and its `(u, v)` stored as given.
    fn torus_rectangle(
        torus: &brepkit_math::surfaces::ToroidalSurface,
        (u0, u1): (f64, f64),
        (v0, v1): (f64, f64),
        counterclockwise: bool,
    ) -> Vec<OrientedPCurveEdge> {
        let mut corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)];
        if !counterclockwise {
            corners.reverse();
        }
        (0..4)
            .map(|k| {
                let (from, to) = (corners[k], corners[(k + 1) % 4]);
                let (s, e) = (Point2::new(from.0, from.1), Point2::new(to.0, to.1));
                OrientedPCurveEdge {
                    curve_3d: EdgeCurve::Line,
                    pcurve: brepkit_math::curves2d::Curve2D::Line(Line2D::new(s, e - s).unwrap()),
                    start_uv: s,
                    end_uv: e,
                    start_3d: torus.evaluate(from.0, from.1),
                    end_3d: torus.evaluate(to.0, to.1),
                    forward: true,
                    source_edge_idx: None,
                    pave_block_id: None,
                }
            })
            .collect()
    }

    /// A torus patch with a hole in its middle, both laid out a turn of `u`
    /// on from the principal window. The patch's interior point lies on it,
    /// off the hole.
    #[test]
    fn a_holed_torus_patch_a_turn_on_keeps_its_point_off_the_hole() {
        use std::f64::consts::TAU;
        let torus =
            brepkit_math::surfaces::ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5)
                .unwrap();
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let patch = SplitSubFace {
            surface: FaceSurface::Torus(torus.clone()),
            outer_wire: torus_rectangle(&torus, (TAU + 0.2, TAU + 1.2), (0.5, 2.5), true),
            inner_wires: vec![torus_rectangle(
                &torus,
                (TAU + 0.5, TAU + 0.9),
                (1.2, 1.8),
                false,
            )],
            reversed: false,
            parent: face_id,
            rank: Rank::A,
            precomputed_interior: None,
        };
        let p = interior_point_3d(&patch, None, 1e-7).unwrap();
        let (u, v) = torus.project_point(p);
        let u = u.rem_euclid(TAU);
        assert!(
            (0.2..1.2).contains(&u) && (0.5..2.5).contains(&v),
            "({u}, {v})"
        );
        assert!(
            !((0.5..0.9).contains(&u) && (1.2..1.8).contains(&v)),
            "({u}, {v}) lies in the hole"
        );
    }

    /// A patch of a spindle torus (radii 0.1 and 2.45) on its sheet across
    /// the axis, round the point at `u = 0`, `v = pi`. Projected onto the
    /// torus, its corners read the near tube's sheet; the patch's interior
    /// point lies on its own sheet.
    #[test]
    fn a_spindle_torus_patch_across_the_axis_keeps_its_point_on_its_sheet() {
        use std::f64::consts::PI;
        let torus =
            brepkit_math::surfaces::ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 0.1, 2.45)
                .unwrap();
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let patch = SplitSubFace {
            surface: FaceSurface::Torus(torus.clone()),
            outer_wire: torus_rectangle(&torus, (-0.2, 0.2), (PI - 0.3, PI + 0.3), true),
            inner_wires: vec![],
            reversed: false,
            parent: face_id,
            rank: Rank::A,
            precomputed_interior: None,
        };
        let p = interior_point_3d(&patch, None, 1e-7).unwrap();
        let want = torus.evaluate(0.0, PI);
        assert!((p - want).length() < 0.05, "{p:?} against {want:?}");
    }

    /// A loop through the north pole between two meridians a half turn
    /// apart, with its region on its left: up the meridian `u = pi`, down
    /// the meridian `u = 0`, and back along a latitude through `u = pi/2`.
    /// The pole's own `u`, and the last bit of the seam meridian's `u`, are
    /// noise, and neither moves the region.
    #[test]
    fn a_loop_through_a_pole_is_read_round_its_own_half() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};
        let low = 0.34;
        for pole_u in [0.0, 1.2, 5.0341, -2.0] {
            for seam in [0.0, 1e-15, TAU - 1e-15, TAU] {
                let mut pts = vec![
                    Point2::new(PI, low),
                    Point2::new(PI, 1.0),
                    Point2::new(pole_u, FRAC_PI_2),
                    Point2::new(seam, 1.0),
                    Point2::new(seam, low),
                    Point2::new(0.0, low),
                    Point2::new(FRAC_PI_2, low),
                ];
                for start in 0..pts.len() {
                    let poly = unwrapped_u(&pts);
                    let holds = |u: f64, v: f64| {
                        (-2..=2).any(|k| {
                            let at = Point2::new(TAU * f64::from(k) + u, v);
                            super::super::classify_2d::point_in_polygon_2d(at, &poly)
                        })
                    };
                    let label = format!("pole_u={pole_u} seam={seam} start={start}");
                    assert!(holds(FRAC_PI_2, 1.0), "{label}: its own half");
                    assert!(holds(FRAC_PI_4, 1.5), "{label}: near the pole");
                    assert!(!holds(3.0 * FRAC_PI_2, 1.0), "{label}: the other half");
                    assert!(!holds(FRAC_PI_2, 0.2), "{label}: below the latitude");
                    pts.rotate_left(1);
                }
            }
        }
    }

    /// The same loop about the south pole runs the other way round, and a
    /// loop off the poles keeps the nearest way.
    #[test]
    fn a_loop_through_the_south_pole_and_one_off_the_poles() {
        use std::f64::consts::{FRAC_PI_2, PI, TAU};
        let south = [
            Point2::new(0.0, -0.34),
            Point2::new(0.0, -1.0),
            Point2::new(4.0, -FRAC_PI_2),
            Point2::new(PI, -1.0),
            Point2::new(PI, -0.34),
            Point2::new(FRAC_PI_2, -0.34),
        ];
        let poly = unwrapped_u(&south);
        let holds = |u: f64, v: f64| {
            (-2..=2).any(|k| {
                let at = Point2::new(TAU * f64::from(k) + u, v);
                super::super::classify_2d::point_in_polygon_2d(at, &poly)
            })
        };
        assert!(holds(FRAC_PI_2, -1.0));
        assert!(!holds(3.0 * FRAC_PI_2, -1.0));

        let across_the_seam = [
            Point2::new(6.0, 0.1),
            Point2::new(0.2, 0.1),
            Point2::new(0.2, 0.5),
            Point2::new(6.0, 0.5),
        ];
        // The same loop within the pole's margin has no point off the pole.
        let lift = |by: f64| -> Vec<Point2> {
            across_the_seam
                .iter()
                .map(|p| Point2::new(p.x(), if p.y() < 0.3 { by } else { by + 6e-7 }))
                .collect()
        };
        for loop_uv in [
            across_the_seam.to_vec(),
            lift(FRAC_PI_2 - 8e-7),
            lift(FRAC_PI_2 - 6.5e-7),
            lift(FRAC_PI_2 - 6.2e-7),
        ] {
            let us: Vec<f64> = unwrapped_u(&loop_uv).iter().map(|p| p.x()).collect();
            assert_eq!(us.len(), 4);
            for (got, want) in us.iter().zip([6.0, 0.2 + TAU, 0.2 + TAU, 6.0]) {
                assert!((got - want).abs() < 1e-12, "{us:?}");
            }
        }
    }

    #[test]
    fn a_closed_section_moved_a_turn_takes_its_pcurve_along() {
        use brepkit_math::curves2d::{Curve2D, NurbsCurve2D};
        use brepkit_math::surfaces::CylindricalSurface;
        use std::f64::consts::TAU;
        let z = Vec3::new(0.0, 0.0, 1.0);
        let cyl = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z, 1.0).unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        // A whole turn of the cylinder at v = 0.5, laid out a turn past
        // [0, 2 pi): ends and pcurve both one turn up.
        let u0 = TAU + 0.5;
        let (a, b) = (Point2::new(u0, 0.5), Point2::new(u0 + TAU, 0.5));
        let at = Point3::new(0.5_f64.cos(), 0.5_f64.sin(), 0.5);
        let mut e = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: Curve2D::Nurbs(NurbsCurve2D::from_line(a, b).unwrap()),
            start_uv: a,
            end_uv: b,
            start_3d: at,
            end_3d: at,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        super::place_section_in_turn(&mut e, &surface, 0.0, TAU);
        assert!(
            (e.start_uv.x() - 0.5).abs() < 1e-12,
            "start {:?}",
            e.start_uv
        );
        let Curve2D::Nurbs(n) = &e.pcurve else {
            unreachable!("the pcurve stays a B-spline");
        };
        let (k0, k1) = (n.knots()[0], n.knots()[n.knots().len() - 1]);
        let (p0, p1) = (n.evaluate(k0), n.evaluate(k1));
        assert!(
            (p0 - e.start_uv).length() < 1e-9 && (p1 - e.end_uv).length() < 1e-9,
            "the pcurve runs {p0:?} to {p1:?}, its ends {:?} to {:?}",
            e.start_uv,
            e.end_uv
        );
    }

    /// A unit cylinder's wall from `v = 0` to `v = 10`, seamed at `u = 0`,
    /// with a hole round the patch from `u = -1` to `u = 4` between `v = 4`
    /// and `v = 6`: arcs along the two heights, split where they cross the
    /// seam, and rulings at the patch's sides; and the patch mirrored in the
    /// seam. The arcs from the seam to `u = 4` run more than half a turn.
    /// Each piece of the hole in the split faces runs its own way round in
    /// `u`, and so ends on its own turn.
    #[test]
    fn a_seam_hole_piece_over_half_a_turn_keeps_its_far_end_on_its_turn() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use std::f64::consts::TAU;

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let on = |u: f64, v: f64| surface.evaluate(u, v).unwrap();
        let ring = |v: f64, up: bool| {
            let normal = Vec3::new(0.0, 0.0, if up { 1.0 } else { -1.0 });
            let circle = Circle3D::new(Point3::new(0.0, 0.0, v), normal, 1.0).unwrap();
            let t = circle.project(on(0.0, v));
            let ahead = circle.evaluate(t + 0.1);
            let step = if up { 0.1 } else { -0.1 };
            assert!(
                (ahead - on(step, v)).length() < 1e-12,
                "the ring turns against u"
            );
            EdgeCurve::Circle(circle)
        };
        let rim = |v: f64, forward: bool| OrientedPCurveEdge {
            curve_3d: ring(v, true),
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(if forward { 0.0 } else { TAU }, v),
            end_uv: Point2::new(if forward { TAU } else { 0.0 }, v),
            start_3d: on(0.0, v),
            end_3d: on(0.0, v),
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let seam = |from: f64, to: f64, u: f64| OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(u, from),
            end_uv: Point2::new(u, to),
            start_3d: on(u, from),
            end_3d: on(u, to),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let boundary = vec![
            rim(0.0, true),
            seam(0.0, 10.0, TAU),
            rim(10.0, false),
            seam(10.0, 0.0, 0.0),
        ];
        // Each piece's ends in (u, v).
        let patch: [((f64, f64), (f64, f64)); 6] = [
            ((-1.0, 4.0), (0.0, 4.0)),
            ((0.0, 4.0), (4.0, 4.0)),
            ((4.0, 4.0), (4.0, 6.0)),
            ((4.0, 6.0), (0.0, 6.0)),
            ((0.0, 6.0), (-1.0, 6.0)),
            ((-1.0, 6.0), (-1.0, 4.0)),
        ];
        // The patch as given, and mirrored in the seam: its long arm on the
        // other side.
        for side in [1.0, -1.0] {
            let pieces = patch.map(|((u0, v0), (u1, v1))| ((side * u0, v0), (side * u1, v1)));
            let sections: Vec<SectionEdge> = pieces
                .iter()
                .map(|&((u0, v0), (u1, v1))| {
                    let curve = if (v1 - v0).abs() > 1e-12 {
                        EdgeCurve::Line
                    } else {
                        ring(v0, u1 > u0)
                    };
                    SectionEdge {
                        curve_3d: curve,
                        ..line_section(on(u0, v0), on(u1, v1))
                    }
                })
                .collect();
            let mut topo = Topology::new();
            let face_id = make_unit_square_face(&mut topo);
            let split = split_periodic_face_around_seam_holes(
                &surface,
                &boundary,
                &sections,
                Rank::A,
                false,
                face_id,
                1e-7,
            )
            .expect("the wall splits round the hole");
            assert_eq!(split.len(), 3, "side {side}: the remainder and two halves");
            let mut checked = 0;
            for face in &split {
                for e in face
                    .outer_wire
                    .iter()
                    .chain(face.inner_wires.iter().flatten())
                {
                    let run = pieces.iter().find_map(|&((u0, v0), (u1, v1))| {
                        let (a, b) = (on(u0, v0), on(u1, v1));
                        if (e.start_3d - a).length() < 1e-9 && (e.end_3d - b).length() < 1e-9 {
                            Some(u1 - u0)
                        } else if (e.start_3d - b).length() < 1e-9 && (e.end_3d - a).length() < 1e-9
                        {
                            Some(u0 - u1)
                        } else {
                            None
                        }
                    });
                    let Some(run) = run else { continue };
                    let got = e.end_uv.x() - e.start_uv.x();
                    assert!(
                        (got - run).abs() < 1e-9,
                        "side {side}: a piece from {:?} to {:?} runs {got} in u, not {run}",
                        e.start_uv,
                        e.end_uv
                    );
                    checked += 1;
                }
            }
            assert_eq!(
                checked, 12,
                "side {side}: each piece in the outline and in its half"
            );
        }
    }

    /// The wall of the cone of [`cone_ellipse_fixture::oblique_cone`] at
    /// slope 0.99999, between rims at `v = 0.1` and `v = 1e4`, with a hole
    /// whose loop runs along the ellipse from its parameter 3.12 to 3.4, up a
    /// ruling to `v = 5000`, back along that height, and down a ruling. The
    /// ellipse crosses the seam at its parameter 3.1343, and the arc above
    /// it at the same `u`; both are split there. From the seam the ellipse's
    /// `u` runs 5.149 to 3.4, nearly all of it round the near vertex. Run
    /// either way round, every piece of the hole in the split faces runs its
    /// own way in `u`, by its exact run.
    ///
    /// [`cone_ellipse_fixture::oblique_cone`]: crate::builder::pcurve_compute::cone_ellipse_fixture::oblique_cone
    #[test]
    fn a_seam_hole_on_a_steep_cone_keeps_each_pieces_run() {
        use crate::builder::pcurve_compute::cone_ellipse_fixture::{lifted_u, oblique_cone};
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::{FRAC_PI_4, TAU};

        let (surface, ellipse) = oblique_cone(0.99999);
        let on = |u: f64, v: f64| surface.evaluate(u, v).unwrap();
        let ring = |v: f64, up: bool| {
            let z = v * FRAC_PI_4.sin();
            let normal = Vec3::new(0.0, 0.0, if up { 1.0 } else { -1.0 });
            Circle3D::new(Point3::new(0.0, 0.0, z), normal, v * FRAC_PI_4.cos()).unwrap()
        };
        let (a, b) = (3.12, 3.4);
        let (u_a, u_b) = lifted_u(&surface, &ellipse, a, b);
        // Where the arc crosses the seam, `u` a whole number of turns.
        let seam = (u_a / TAU).ceil() * TAU;
        assert!(seam < u_b);
        let (mut lo, mut hi) = (a, b);
        for _ in 0..100 {
            let mid = f64::midpoint(lo, hi);
            if lifted_u(&surface, &ellipse, a, mid).1 < seam {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t_seam = f64::midpoint(lo, hi);
        assert!(
            t_seam - a > 1e-3 && u_b - seam > 5.0,
            "the arc crosses the seam at {t_seam}, and runs {} on from it",
            u_b - seam
        );
        let (_, v_a) = surface.project_point(ellipse.evaluate(a)).unwrap();
        let (_, v_b) = surface.project_point(ellipse.evaluate(b)).unwrap();
        let (v_bot, v_high, v_top) = (0.1, 5000.0, 1e4);
        let rim = |v: f64, forward: bool| OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(ring(v, true)),
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(if forward { 0.0 } else { TAU }, v),
            end_uv: Point2::new(if forward { TAU } else { 0.0 }, v),
            start_3d: on(0.0, v),
            end_3d: on(0.0, v),
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let seam_line = |from: f64, to: f64, u: f64| OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(u, from),
            end_uv: Point2::new(u, to),
            start_3d: on(u, from),
            end_3d: on(u, to),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let boundary = vec![
            rim(v_bot, true),
            seam_line(v_bot, v_top, TAU),
            rim(v_top, false),
            seam_line(v_top, v_bot, 0.0),
        ];
        let section = |curve: EdgeCurve, start: Point3, end: Point3| SectionEdge {
            curve_3d: curve,
            ..line_section(start, end)
        };
        let at = |t: f64| ellipse.evaluate(t);
        let high = |u: f64| on(u, v_high);
        // Each piece as stored, and its run in u from its stored start.
        let pieces: Vec<(SectionEdge, f64)> = vec![
            (
                section(EdgeCurve::Ellipse(ellipse.clone()), at(a), at(t_seam)),
                seam - u_a,
            ),
            (
                section(EdgeCurve::Ellipse(ellipse.clone()), at(t_seam), at(b)),
                u_b - seam,
            ),
            (section(EdgeCurve::Line, at(b), high(u_b)), 0.0),
            (
                section(
                    EdgeCurve::Circle(ring(v_high, false)),
                    high(u_b),
                    high(seam),
                ),
                seam - u_b,
            ),
            (
                section(
                    EdgeCurve::Circle(ring(v_high, false)),
                    high(seam),
                    high(u_a),
                ),
                u_a - seam,
            ),
            (section(EdgeCurve::Line, high(u_a), at(a)), 0.0),
        ];
        assert!(v_a < v_high && v_b < v_high);
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        // Chained from the first piece as stored, the loop runs along the
        // ellipse first; from the last ruling stored the other way, it runs
        // the loop backwards.
        let mut backwards = pieces.clone();
        let (last, run) = backwards.pop().unwrap();
        backwards.insert(
            0,
            (
                SectionEdge {
                    start: last.end,
                    end: last.start,
                    ..last
                },
                run,
            ),
        );
        for (way, pieces) in [("along", pieces), ("back", backwards)] {
            let sections: Vec<SectionEdge> = pieces.iter().map(|(s, _)| s.clone()).collect();
            let split = split_periodic_face_around_seam_holes(
                &surface,
                &boundary,
                &sections,
                Rank::A,
                false,
                face_id,
                1e-7,
            )
            .expect("the wall splits round the hole");
            assert_eq!(split.len(), 3, "{way}: the remainder and two halves");
            let mut checked = 0;
            for face in &split {
                for e in face
                    .outer_wire
                    .iter()
                    .chain(face.inner_wires.iter().flatten())
                {
                    let run = pieces.iter().find_map(|(s, run)| {
                        if (e.start_3d - s.start).length() < 1e-6
                            && (e.end_3d - s.end).length() < 1e-6
                        {
                            Some(*run)
                        } else if (e.start_3d - s.end).length() < 1e-6
                            && (e.end_3d - s.start).length() < 1e-6
                        {
                            Some(-*run)
                        } else {
                            None
                        }
                    });
                    let Some(run) = run else { continue };
                    let got = e.end_uv.x() - e.start_uv.x();
                    assert!(
                        (got - run).abs() < 1e-9,
                        "{way}: a piece from {:?} to {:?} runs {got} in u, not {run}",
                        e.start_uv,
                        e.end_uv
                    );
                    checked += 1;
                }
            }
            assert_eq!(checked, 12, "{way}: each piece in the outline and its half");
        }
    }

    /// Pieces of the ellipse of `cone_ellipse_fixture::oblique_cone` at
    /// slope 0.99999, each way along it. The piece across the near vertex
    /// runs most of a turn of `u`, and crosses the seam meridian once, where
    /// its `u` lifted along it passes the seam's: with the seam at the
    /// vertex, and half a radian on from the piece's start. Read from its
    /// ends the short way round, it crossed nowhere. A short piece crosses
    /// the seam at its middle, and not the meridian opposite.
    #[test]
    fn a_section_over_half_a_turn_splits_where_it_crosses_the_seam() {
        use crate::builder::pcurve_compute::cone_ellipse_fixture::{lifted_u, oblique_cone};
        use std::f64::consts::{PI, TAU};
        let (surface, ellipse) = oblique_cone(0.99999);
        let curve = EdgeCurve::Ellipse(ellipse.clone());
        let reversed = curve.reversed().unwrap();
        let check = |(a, b): (f64, f64), seam: f64, crossings: usize| {
            for (c, start, end) in [
                (&curve, ellipse.evaluate(a), ellipse.evaluate(b)),
                (&reversed, ellipse.evaluate(b), ellipse.evaluate(a)),
            ] {
                let s = SectionEdge {
                    curve_3d: c.clone(),
                    ..line_section(start, end)
                };
                let split = split_sections_at_seam_meridian(
                    &[s],
                    &surface,
                    seam.rem_euclid(TAU),
                    &[],
                    1e-7,
                );
                assert_eq!(split.len(), crossings + 1, "({a}, {b}) at seam {seam}");
                assert!((split[0].start - start).length() < 1e-12);
                assert!((split[crossings].end - end).length() < 1e-12);
                for k in 0..crossings {
                    let p = split[k].end;
                    assert!((split[k + 1].start - p).length() < 1e-12);
                    let t = (ellipse.project(p) - a).rem_euclid(TAU) + a;
                    assert!((ellipse.evaluate(t) - p).length() < 1e-9, "off the ellipse");
                    assert!(t > a && t < b, "({a}, {b}) at seam {seam}: crossed at {t}");
                    let (_, u) = lifted_u(&surface, &ellipse, a, t);
                    assert!(
                        (u - seam).abs() < 1e-6,
                        "({a}, {b}): crossed at u {u}, the seam at {seam}"
                    );
                }
            }
        };
        let near = (PI - 0.02, PI + 0.02);
        let (u_a, u_b) = lifted_u(&surface, &ellipse, near.0, near.1);
        assert!((u_b - u_a).abs() > PI, "the piece runs {}", u_b - u_a);
        let (_, u_vertex) = lifted_u(&surface, &ellipse, near.0, PI);
        check(near, u_vertex, 1);
        check(near, u_a + 0.5f64.copysign(u_b - u_a), 1);
        let short = (1.5 * PI, 1.75 * PI);
        let (u_c, u_d) = lifted_u(&surface, &ellipse, short.0, short.1);
        assert!((u_d - u_c).abs() < 0.5, "the piece runs {}", u_d - u_c);
        check(short, f64::midpoint(u_c, u_d), 1);
        check(short, f64::midpoint(u_c, u_d) + PI, 0);
    }

    /// An arc of a ring round a unit cylinder from u = 0.2 to u = -0.2, the
    /// short way round as its pcurve runs, written with the ring's normal
    /// up and down and each way along it, so in some cases its own sense
    /// runs the long way between its ends. It crosses the seam at u = 0 at
    /// (1, 0, z), and not the seam at u = pi on the long way round: walked
    /// along that long way, it took a point at (-1, 0, z), off the arc.
    #[test]
    fn a_ring_arc_crosses_the_seam_on_the_short_way_its_pcurve_runs() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use std::f64::consts::PI;
        let z = 2.0;
        let surface = FaceSurface::Cylinder(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap(),
        );
        let at = |u: f64| surface.evaluate(u, z).unwrap();
        for normal in [1.0, -1.0] {
            let ring = EdgeCurve::Circle(
                Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, normal), 1.0).unwrap(),
            );
            for (from, to) in [(0.2, -0.2), (-0.2, 0.2)] {
                let arc = SectionEdge {
                    curve_3d: ring.clone(),
                    ..line_section(at(from), at(to))
                };
                let what = format!("normal {normal}, {from} to {to}");
                let split = split_sections_at_seam_meridian(
                    std::slice::from_ref(&arc),
                    &surface,
                    PI,
                    &[],
                    1e-7,
                );
                assert_eq!(split.len(), 1, "{what}: split at the far seam");
                let split = split_sections_at_seam_meridian(&[arc], &surface, 0.0, &[], 1e-7);
                assert_eq!(split.len(), 2, "{what}: no split at the near seam");
                assert!(
                    (split[0].end - at(0.0)).length() < 1e-9,
                    "{what}: split at {:?}",
                    split[0].end
                );
            }
        }
    }

    /// A rational spline on a unit cylinder of quarter arcs from u = 0 to
    /// -pi/2, back to 0 and on to pi/2, rising from z = 0 to 3 a unit each,
    /// each way along it. It starts on the seam at u = 0 and crosses it
    /// again at z = 2, where it splits. A crossing at the seam copy its
    /// start lies on was dropped wherever it fell.
    #[test]
    fn a_section_leaving_the_seam_splits_where_it_crosses_it_again() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::surfaces::CylindricalSurface;
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};
        let surface = FaceSurface::Cylinder(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap(),
        );
        // The point at `u` on the ring of radius `r` about the axis at `z`.
        let at = |u: f64, z: f64, r: f64| {
            let axis = Point3::new(0.0, 0.0, z);
            axis + (surface.evaluate(u, z).unwrap() - axis) * r
        };
        let turns = [0.0, -FRAC_PI_2, 0.0, FRAC_PI_2];
        let mut poles = vec![at(0.0, 0.0, 1.0)];
        let mut weights = vec![1.0];
        for (k, z0) in [0.0, 1.0, 2.0].into_iter().enumerate() {
            let z1 = z0 + 1.0;
            let middle = 0.5 * (turns[k] + turns[k + 1]);
            poles.push(at(middle, 0.5 * (z0 + z1), 1.0 / FRAC_PI_4.cos()));
            poles.push(at(turns[k + 1], z1, 1.0));
            weights.extend([FRAC_1_SQRT_2, 1.0]);
        }
        let third = 1.0 / 3.0;
        let knots = vec![
            0.0,
            0.0,
            0.0,
            third,
            third,
            2.0 * third,
            2.0 * third,
            1.0,
            1.0,
            1.0,
        ];
        let spline =
            EdgeCurve::NurbsCurve(NurbsCurve::new(2, knots, poles.clone(), weights).unwrap());
        let (first, last) = (poles[0], poles[poles.len() - 1]);
        for (curve, start, end) in [
            (spline.clone(), first, last),
            (spline.reversed().unwrap(), last, first),
        ] {
            let section = SectionEdge {
                curve_3d: curve,
                ..line_section(start, end)
            };
            let split = split_sections_at_seam_meridian(&[section], &surface, 0.0, &[], 1e-7);
            assert_eq!(split.len(), 2, "from {start:?}: {split:?}");
            assert!(
                (split[0].end - at(0.0, 2.0, 1.0)).length() < 1e-9,
                "from {start:?}: split at {:?}",
                split[0].end
            );
        }
    }

    /// A ball face round its north pole, below which its one rim of three
    /// arcs runs once round the axis, holds that pole. A zone between two
    /// latitudes, bounded by a meridian seam and two rims of three arcs each,
    /// spans a turn of `u` too, but holds no pole and turns its rims both
    /// ways: the walk that closes a loop a turn round does not run on it.
    #[test]
    fn only_a_ball_face_round_a_pole_closes_its_loops_a_turn_round() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeId};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let axis = Vec3::new(0.0, 0.0, 1.0);
        let rim = |topo: &mut Topology, h: f64| -> Vec<EdgeId> {
            let w = (1.0 - h * h).sqrt();
            let circle = Circle3D::new(Point3::new(0.0, 0.0, h), axis, w).unwrap();
            let vs: Vec<_> = (0..3)
                .map(|k| {
                    let a = std::f64::consts::TAU * f64::from(k) / 3.0;
                    topo.add_vertex(Vertex::new(Point3::new(w * a.cos(), w * a.sin(), h), 1e-7))
                })
                .collect();
            (0..3)
                .map(|k| {
                    topo.add_edge(Edge::new(
                        vs[k],
                        vs[(k + 1) % 3],
                        EdgeCurve::Circle(circle.clone()),
                    ))
                })
                .collect()
        };
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let ball = || FaceSurface::Sphere(sphere.clone());

        let mut topo = Topology::new();
        let low = rim(&mut topo, -0.4);
        let cap = topo.add_wire(
            Wire::new(
                low.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let cap = topo.add_face(Face::new(cap, vec![], ball()));
        assert!(pole_held_round(&topo, cap, &sphere).is_some());

        let high = rim(&mut topo, 0.6);
        let (from, to) = (
            topo.edge(low[0]).unwrap().start(),
            topo.edge(high[0]).unwrap().start(),
        );
        let at = topo.vertex(from).unwrap().point();
        let meridian = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            1.0,
            at - Point3::new(0.0, 0.0, 0.0),
        )
        .unwrap();
        let seam = topo.add_edge(Edge::new(from, to, EdgeCurve::Circle(meridian)));
        let mut zone: Vec<OrientedEdge> = low.iter().map(|&e| OrientedEdge::new(e, true)).collect();
        zone.push(OrientedEdge::new(seam, true));
        zone.extend(high.iter().rev().map(|&e| OrientedEdge::new(e, false)));
        zone.push(OrientedEdge::new(seam, false));
        let zone = topo.add_wire(Wire::new(zone, true).unwrap());
        let zone = topo.add_face(Face::new(zone, vec![], ball()));
        assert!(pole_held_round(&topo, zone, &sphere).is_none());
    }

    /// The unit ball less the cap round the unit direction `cap` beyond
    /// `0.95`, one face bounded by the cap's rim; two small circles on it,
    /// 0.2 rad round, their middles 0.3 rad apart either side of the unit
    /// direction `m` along the unit direction `e` square to it, so that they
    /// cross twice, and a whole circle 0.1 rad round the unit direction
    /// `lone` where one is given, as a circle or as a squeezed spline
    /// ([`squeezed_spline_circle`]); split by `split_face_2d` along their
    /// arcs, laid in the order of their circles.
    fn split_with_crossing_circles(
        cap: Vec3,
        m: Vec3,
        e: Vec3,
        lone: Option<(Vec3, bool)>,
    ) -> Vec<SplitSubFace> {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let origin = Point3::new(0.0, 0.0, 0.0);
        let circle =
            |d: Vec3, rho: f64| Circle3D::new(origin + d * rho.cos(), d, rho.sin()).unwrap();
        let sphere = SphericalSurface::new(origin, 1.0).unwrap();
        let surface = FaceSurface::Sphere(sphere);
        let mut topo = Topology::new();
        // The rim runs round `-cap`, so that the face lies on its left.
        let rim = circle(-cap, std::f64::consts::PI - 0.95_f64.acos());
        let at = topo.add_vertex(Vertex::new(rim.evaluate(0.0), 1e-7));
        let rim = topo.add_edge(Edge::new(at, at, EdgeCurve::Circle(rim)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let face_id = topo.add_face(Face::new(wire, vec![], surface.clone()));

        let (rho, apart) = (0.2_f64, 0.3_f64);
        let (s, c) = (0.5 * apart).sin_cos();
        let n = m.cross(e);
        let along = rho.cos() / c;
        let across = (1.0 - along * along).sqrt();
        let p = origin + m * along + n * across;
        let q = origin + m * along - n * across;
        let mut sections = Vec::new();
        for d in [m * c + e * s, m * c - e * s] {
            let round = circle(d, rho);
            // Each arc in pieces under half a turn, as the face-face phase
            // emits them.
            let mut arcs = Vec::new();
            for (from, to) in [(p, q), (q, p)] {
                let (a, b) = (round.project(from), round.project(to));
                let span = (b - a).rem_euclid(std::f64::consts::TAU);
                if span < 0.999 * std::f64::consts::PI {
                    arcs.push((from, to));
                } else {
                    let middle = round.evaluate(a + 0.5 * span);
                    arcs.extend([(from, middle), (middle, to)]);
                }
            }
            let curve = EdgeCurve::Circle(round);
            for (from, to) in arcs {
                let pcurve = super::super::pcurve_compute::compute_pcurve_on_surface(
                    &curve,
                    from,
                    to,
                    &surface,
                    &[],
                    None,
                );
                sections.push(SectionEdge {
                    curve_3d: curve.clone(),
                    pcurve_a: pcurve,
                    pcurve_b: dummy_pcurve(),
                    start: from,
                    end: to,
                    start_uv_a: None,
                    end_uv_a: None,
                    start_uv_b: None,
                    end_uv_b: None,
                    target_face: None,
                    pave_block_id: None,
                });
            }
        }
        if let Some((d, squeezed)) = lone {
            let curve = if squeezed {
                EdgeCurve::NurbsCurve(squeezed_spline_circle(&circle(d, 0.1)))
            } else {
                EdgeCurve::Circle(circle(d, 0.1))
            };
            let at = curve.evaluate_with_endpoints(0.0, origin, origin);
            let pcurve = super::super::pcurve_compute::compute_pcurve_on_surface(
                &curve,
                at,
                at,
                &surface,
                &[],
                None,
            );
            sections.push(SectionEdge {
                curve_3d: curve,
                pcurve_a: pcurve,
                pcurve_b: dummy_pcurve(),
                start: at,
                end: at,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
                pave_block_id: None,
            });
        }
        let tol = brepkit_math::tolerance::Tolerance::new();
        let edge_images: std::collections::HashMap<
            brepkit_topology::edge::EdgeId,
            Vec<brepkit_topology::edge::EdgeId>,
        > = std::collections::HashMap::new();
        let info = SurfaceInfo::Parametric {
            u_periodic: true,
            v_periodic: false,
        };
        split_face_2d(
            &topo,
            face_id,
            &sections,
            Rank::A,
            &tol,
            None,
            Some(&info),
            &edge_images,
            &std::collections::BTreeSet::new(),
            None,
        )
    }

    /// `circle` as a closed rational quadratic spline in four arcs: three
    /// over the first 1e-6 of its domain run all but 1e-6 rad of the turn,
    /// and the fourth runs the rest over the remainder of the domain.
    fn squeezed_spline_circle(
        circle: &brepkit_math::curves::Circle3D,
    ) -> brepkit_math::nurbs::curve::NurbsCurve {
        use std::f64::consts::TAU;
        let (c, r) = (circle.center(), circle.radius());
        let (u, v) = (circle.u_axis(), circle.v_axis());
        let on = |a: f64| c + u * (r * a.cos()) + v * (r * a.sin());
        let last = 1e-6;
        let big = (TAU - last) / 3.0;
        let ends = [0.0, big, 2.0 * big, TAU - last, TAU];
        let mut points = vec![on(0.0)];
        let mut weights = vec![1.0];
        for w in ends.windows(2) {
            let half = 0.5 * (w[1] - w[0]);
            points.push(c + (on(w[0] + half) - c) * (1.0 / half.cos()));
            weights.push(half.cos());
            points.push(on(w[1]));
            weights.push(1.0);
        }
        let third = 1e-6 / 3.0;
        let knots = vec![
            0.0,
            0.0,
            0.0,
            third,
            third,
            2.0 * third,
            2.0 * third,
            1e-6,
            1e-6,
            1.0,
            1.0,
            1.0,
        ];
        brepkit_math::nurbs::curve::NurbsCurve::new(2, knots, points, weights).unwrap()
    }

    /// The arcs a piece's wires run, as the middles of their circles.
    fn circles_run(piece: &SplitSubFace) -> Vec<Vec<Point3>> {
        std::iter::once(&piece.outer_wire)
            .chain(&piece.inner_wires)
            .map(|w| {
                w.iter()
                    .filter_map(|e| match &e.curve_3d {
                        EdgeCurve::Circle(c) => Some(c.center()),
                        _ => None,
                    })
                    .collect()
            })
            .collect()
    }

    /// Two small circles that cross on a ball face split it in four: the
    /// face less both discs, each disc less the other, and their lens. Read
    /// as two whole discs, the discs overlap on the lens and the face's
    /// two holes overlap there too.
    #[test]
    fn two_small_circles_that_cross_on_a_ball_face_split_it_in_four() {
        let unit = |x: f64, y: f64, z: f64| Vec3::new(x, y, z).normalize().unwrap();
        // The face holds the south pole of its own chart; and, with the cap
        // round +x, both poles, so that it takes a chart with one pole off it.
        for (cap, m, e) in [
            (
                Vec3::new(0.0, 0.0, 1.0),
                unit(1.0, 0.0, -1.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            (
                Vec3::new(1.0, 0.0, 0.0),
                unit(-0.5, 0.5, -0.7),
                unit(1.0, 1.0, 0.0),
            ),
        ] {
            let split = split_with_crossing_circles(cap, m, e, None);
            assert_eq!(split.len(), 4, "cap {cap:?}");
            // No wire runs one small circle alone: a whole disc.
            let rim = Point3::new(0.0, 0.0, 0.0) + cap * 0.95;
            for piece in &split {
                for wire in circles_run(piece) {
                    let one_circle = wire.iter().all(|c| (*c - wire[0]).length() < 1e-9);
                    assert!(
                        !one_circle || (wire[0] - rim).length() < 1e-9,
                        "cap {cap:?}: a wire runs a whole small circle"
                    );
                }
            }
            assert_eq!(split.iter().map(|p| p.inner_wires.len()).sum::<usize>(), 1);
        }
    }

    /// Two small circles that cross and a whole circle apart from them,
    /// a circle or a squeezed spline, split a ball face in five: the whole
    /// circle's disc stands as a piece, and the face keeps it as a hole.
    #[test]
    fn crossing_circles_and_a_lone_circle_on_a_ball_face_split_it_in_five() {
        let unit = |x: f64, y: f64, z: f64| Vec3::new(x, y, z).normalize().unwrap();
        let cases = [
            (
                Vec3::new(0.0, 0.0, 1.0),
                unit(1.0, 0.0, -1.0),
                Vec3::new(0.0, 1.0, 0.0),
                unit(-1.0, 0.0, -1.0),
            ),
            (
                Vec3::new(1.0, 0.0, 0.0),
                unit(-0.5, 0.5, -0.7),
                unit(1.0, 1.0, 0.0),
                unit(-0.5, -0.5, -0.7),
            ),
        ];
        for (case, squeezed) in cases.iter().flat_map(|c| [(c, false), (c, true)]) {
            let &(cap, m, e, lone) = case;
            let split = split_with_crossing_circles(cap, m, e, Some((lone, squeezed)));
            assert_eq!(split.len(), 5, "cap {cap:?}, squeezed {squeezed}");
            assert_eq!(
                split.iter().map(|p| p.inner_wires.len()).sum::<usize>(),
                2,
                "cap {cap:?}"
            );
            // The lone loop's disc: one wire of one edge, the lone loop.
            let middle = Point3::new(0.0, 0.0, 0.0) + lone * 0.1_f64.cos();
            let disc = split.iter().filter(|p| {
                p.inner_wires.is_empty()
                    && p.outer_wire.len() == 1
                    && (p.outer_wire[0].start_3d - middle).length() < 0.11
            });
            assert_eq!(disc.count(), 1, "cap {cap:?}, squeezed {squeezed}");
        }
    }

    /// The unit ball less the cap round `+z` beyond `0.95`, one face bounded
    /// by the cap's rim; a circle 0.15 rad round a point of the rim, whose
    /// arc below the rim cuts a piece off the face; and a whole circle 0.1
    /// rad round the unit direction `lone`; split by `split_face_2d`.
    fn split_with_an_arc_and_a_lone_circle(lone: Vec3) -> Vec<SplitSubFace> {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let origin = Point3::new(0.0, 0.0, 0.0);
        let circle =
            |d: Vec3, rho: f64| Circle3D::new(origin + d * rho.cos(), d, rho.sin()).unwrap();
        let sphere = SphericalSurface::new(origin, 1.0).unwrap();
        let surface = FaceSurface::Sphere(sphere);
        let mut topo = Topology::new();
        let polar = 0.95_f64.acos();
        let rim = circle(Vec3::new(0.0, 0.0, -1.0), std::f64::consts::PI - polar);
        let at = topo.add_vertex(Vertex::new(rim.evaluate(0.0), 1e-7));
        let rim = topo.add_edge(Edge::new(at, at, EdgeCurve::Circle(rim)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let face_id = topo.add_face(Face::new(wire, vec![], surface.clone()));
        // The arc's ends: where the circle crosses z = 0.95, by bisection.
        let round = circle(Vec3::new(polar.sin(), 0.0, polar.cos()), 0.15);
        let above = |t: f64| round.evaluate(t).z() > 0.95;
        let steps: Vec<f64> = (0..=360).map(|k| f64::from(k).to_radians()).collect();
        let crossings: Vec<f64> = steps
            .windows(2)
            .filter(|w| above(w[0]) != above(w[1]))
            .map(|w| {
                let (mut lo, mut hi) = (w[0], w[1]);
                for _ in 0..80 {
                    let mid = f64::midpoint(lo, hi);
                    if above(mid) == above(lo) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                f64::midpoint(lo, hi)
            })
            .collect();
        assert_eq!(crossings.len(), 2);
        let (from, to) = if above(f64::midpoint(crossings[0], crossings[1])) {
            (crossings[1], crossings[0])
        } else {
            (crossings[0], crossings[1])
        };
        let lone = circle(lone, 0.1);
        let (ring, p, q) = (lone.evaluate(0.0), round.evaluate(from), round.evaluate(to));
        let mut sections = Vec::new();
        for (curve, a, b) in [
            (EdgeCurve::Circle(round), p, q),
            (EdgeCurve::Circle(lone), ring, ring),
        ] {
            let pcurve = super::super::pcurve_compute::compute_pcurve_on_surface(
                &curve,
                a,
                b,
                &surface,
                &[],
                None,
            );
            sections.push(SectionEdge {
                curve_3d: curve,
                pcurve_a: pcurve,
                pcurve_b: dummy_pcurve(),
                start: a,
                end: b,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
                pave_block_id: None,
            });
        }
        let tol = brepkit_math::tolerance::Tolerance::new();
        let edge_images: std::collections::HashMap<
            brepkit_topology::edge::EdgeId,
            Vec<brepkit_topology::edge::EdgeId>,
        > = std::collections::HashMap::new();
        let info = SurfaceInfo::Parametric {
            u_periodic: true,
            v_periodic: false,
        };
        split_face_2d(
            &topo,
            face_id,
            &sections,
            Rank::A,
            &tol,
            None,
            Some(&info),
            &edge_images,
            &std::collections::BTreeSet::new(),
            None,
        )
    }

    /// An arc across a ball face's rim and a whole circle apart from it split
    /// the face in three: the piece the arc cuts off, the whole circle's
    /// disc, and the rest of the face with the disc as its hole.
    #[test]
    fn an_arc_across_the_rim_and_a_lone_circle_split_a_ball_face_in_three() {
        let lone = Vec3::new(-1.0, 0.0, -1.0).normalize().unwrap();
        let split = split_with_an_arc_and_a_lone_circle(lone);
        assert_eq!(split.len(), 3);
        assert_eq!(split.iter().map(|p| p.inner_wires.len()).sum::<usize>(), 1);
    }

    /// A square plate 10 wide, and on it two arcs from `(4, 5)` to `(6, 5)`,
    /// each a quarter of a circle of radius `sqrt(2)`, one bulging up about
    /// `(5, 4)` and one down about `(5, 6)`, with two lines from those
    /// points up to `(5, 8)`. The sections and the plate's face.
    fn two_arcs_between_one_pair_of_points() -> (Topology, FaceId, Vec<SectionEdge>) {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
            .map(|(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)));
        let edges: Vec<_> = (0..4)
            .map(|i| {
                let edge = Edge::new(corners[i], corners[(i + 1) % 4], EdgeCurve::Line);
                OrientedEdge::new(topo.add_edge(edge), true)
            })
            .collect();
        let outer = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = topo.add_face(Face::new(
            outer,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let (p, q, r) = (
            Point3::new(4.0, 5.0, 0.0),
            Point3::new(6.0, 5.0, 0.0),
            Point3::new(5.0, 8.0, 0.0),
        );
        // Each arc runs counter-clockwise about +z from its start, under a
        // half turn.
        let arc = |centre_y: f64, from: Point3, to: Point3| {
            let circle = Circle3D::new(
                Point3::new(5.0, centre_y, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                std::f64::consts::SQRT_2,
            )
            .unwrap();
            SectionEdge {
                curve_3d: EdgeCurve::Circle(circle),
                ..line_section(from, to)
            }
        };
        let sections = vec![
            arc(4.0, q, p),
            arc(6.0, p, q),
            line_section(q, r),
            line_section(r, p),
        ];
        (topo, face, sections)
    }

    /// The middle of a section along its shorter arc.
    fn section_middle(s: &SectionEdge) -> Point3 {
        evaluate_edge_at_t(&s.curve_3d, s.start, s.end, 0.5)
    }

    /// Two different curves between one pair of points (an arc bulging up
    /// and one bulging down, or the arc up and the straight line) are two
    /// sections: the loop finder keeps both. The arc up repeated is one
    /// section, and goes once.
    #[test]
    fn plane_internal_line_loops_keeps_two_curves_that_share_their_ends() {
        let (topo, face_id, sections) = two_arcs_between_one_pair_of_points();
        let face = topo.face(face_id).unwrap();
        let surface = face.surface().clone();
        let wire_pts = collect_wire_points(&topo, face.outer_wire());
        let frame = PlaneFrame::from_plane_face(extract_plane_normal(&surface), &wire_pts);
        let boundary =
            boundary_edges_to_pcurve(&topo, face.outer_wire(), &surface, &wire_pts, Some(&frame));
        let chord = line_section(sections[0].start, sections[0].end);
        for (what, pair) in [
            ("two arcs", [sections[0].clone(), sections[1].clone()]),
            ("an arc and a line", [sections[0].clone(), chord]),
        ] {
            let mut given = pair.to_vec();
            given.extend_from_slice(&sections[2..]);
            if let Some(kept) = plane_internal_line_loops(&given, &frame, &boundary, &[], &[], 1e-7)
            {
                for curve in &pair {
                    let middle = section_middle(curve);
                    assert!(
                        kept.iter()
                            .any(|s| (section_middle(s) - middle).length() < 1e-9),
                        "{what}: the curve through {middle:?} was dropped, {} of {} kept",
                        kept.len(),
                        given.len()
                    );
                }
            }
        }
        let repeated = vec![
            sections[0].clone(),
            sections[2].clone(),
            sections[0].clone(),
            sections[3].clone(),
        ];
        let kept = plane_internal_line_loops(&repeated, &frame, &boundary, &[], &[], 1e-7)
            .expect("the arc up and the two lines close one loop");
        assert_eq!(kept.len(), 3);
    }

    /// The same plate split by those sections: the lens between the two
    /// arcs, the triangle above the arc up, and the rest of the plate, each
    /// the region its sections bound.
    #[test]
    fn two_arcs_between_one_pair_of_points_split_a_plate_into_their_regions() {
        let (topo, face_id, sections) = two_arcs_between_one_pair_of_points();
        let subs = split_face_2d(
            &topo,
            face_id,
            &sections,
            Rank::A,
            &brepkit_math::tolerance::Tolerance::new(),
            None,
            None,
            &std::collections::HashMap::new(),
            &std::collections::BTreeSet::new(),
            None,
        );
        // Each region's area, by the shoelace formula over its wires, each
        // edge read at 2000 points.
        let wire_area = |wire: &[OrientedPCurveEdge]| {
            let mut pts = Vec::new();
            for e in wire {
                for k in 0..2000 {
                    let t = f64::from(k) / 2000.0;
                    pts.push(evaluate_edge_at_t(&e.curve_3d, e.start_3d, e.end_3d, t));
                }
            }
            let mut twice = 0.0;
            for (i, a) in pts.iter().enumerate() {
                let b = pts[(i + 1) % pts.len()];
                twice += a.x() * b.y() - b.x() * a.y();
            }
            (0.5 * twice).abs()
        };
        let mut areas: Vec<f64> = subs
            .iter()
            .map(|s| {
                wire_area(&s.outer_wire) - s.inner_wires.iter().map(|w| wire_area(w)).sum::<f64>()
            })
            .collect();
        areas.sort_by(f64::total_cmp);
        // Each arc cuts a segment of area pi/2 - 1 off its quarter disc; the
        // triangle's area is 3.
        let pi = std::f64::consts::PI;
        let want = [pi - 2.0, 4.0 - 0.5 * pi, 98.0 - 0.5 * pi];
        assert_eq!(areas.len(), 3, "regions of areas {areas:?}");
        for (got, want) in areas.iter().zip(want) {
            assert!(
                (got - want).abs() < 1e-5,
                "regions of areas {areas:?}, want {want}"
            );
        }
    }

    /// A section along the B-spline of degree `degree` on `knots` through
    /// `points` (in z = 0) with `weights`, from its first point to its last.
    fn spline_section(
        degree: usize,
        knots: &[f64],
        points: &[(f64, f64)],
        weights: &[f64],
    ) -> SectionEdge {
        let points: Vec<Point3> = points
            .iter()
            .map(|&(x, y)| Point3::new(x, y, 0.0))
            .collect();
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            degree,
            knots.to_vec(),
            points.clone(),
            weights.to_vec(),
        )
        .unwrap();
        SectionEdge {
            curve_3d: EdgeCurve::NurbsCurve(curve),
            ..line_section(points[0], points[points.len() - 1])
        }
    }

    /// The most `y` a section reaches, read at 2001 points along it.
    fn highest_y(s: &SectionEdge) -> f64 {
        (0..=2000)
            .map(|k| evaluate_edge_at_t(&s.curve_3d, s.start, s.end, f64::from(k) / 2000.0).y())
            .fold(
                f64::NEG_INFINITY,
                crate::builder::pcurve_compute::max_or_nan,
            )
    }

    /// Two cubics on one knot vector through one set of control points, with
    /// weights `s` at all four, and `s, 100 s, 100 s, s`, for `s` of 1e-15,
    /// 1e-200 and 1e200: their ends and middles meet, but the first reaches
    /// y = 0.289 and the second 0.855. Raw products of the weights underflow
    /// or overflow at the two far scales. Weights that differ only by a
    /// common scale are one curve.
    #[test]
    fn b_splines_whose_weights_differ_in_ratio_are_two_curves() {
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let points = [(0.0, 0.0), (1.0, 1.0), (2.0, -1.0), (3.0, 0.0)];
        for s in [1e-15, 1e-200, 1e200] {
            let even = spline_section(3, &knots, &points, &[s; 4]);
            let uneven = spline_section(3, &knots, &points, &[s, 100.0 * s, 100.0 * s, s]);
            assert!((section_middle(&even) - section_middle(&uneven)).length() < 1e-9);
            let (low, high) = (highest_y(&even), highest_y(&uneven));
            assert!(
                (low - 0.289).abs() < 1e-3 && (high - 0.855).abs() < 1e-3,
                "scale {s}: {low} {high}"
            );
            assert!(!same_section_curve(&even, &uneven, 1e-5), "scale {s}");
            assert!(!same_section_curve(&uneven, &even, 1e-5), "scale {s}");
            let scaled = spline_section(3, &knots, &points, &[2.5 * s; 4]);
            assert!(same_section_curve(&even, &scaled, 1e-5), "scale {s}");
            let scaled = spline_section(
                3,
                &knots,
                &points,
                &[2.5 * s, 250.0 * s, 250.0 * s, 2.5 * s],
            );
            assert!(same_section_curve(&uneven, &scaled, 1e-5), "scale {s}");
        }
    }

    /// Two cubics on the parameter span [0, 1e-12] through six control
    /// points symmetric about (1.5, 0), with inner knots at a quarter and
    /// three quarters of the span, and at 0.4 and 0.6 of it. Their ends and
    /// middles meet, but they part by more than the tolerance between.
    #[test]
    fn b_splines_with_other_knots_on_a_short_span_are_two_curves() {
        let span = 1e-12;
        let knots = |a: f64, b: f64| {
            [
                0.0,
                0.0,
                0.0,
                0.0,
                a * span,
                b * span,
                span,
                span,
                span,
                span,
            ]
        };
        let points = [
            (0.0, 0.0),
            (0.6, 1.0),
            (1.2, 1.0),
            (1.8, -1.0),
            (2.4, -1.0),
            (3.0, 0.0),
        ];
        let quarters = spline_section(3, &knots(0.25, 0.75), &points, &[1.0; 6]);
        let tenths = spline_section(3, &knots(0.4, 0.6), &points, &[1.0; 6]);
        assert!((section_middle(&quarters) - section_middle(&tenths)).length() < 1e-9);
        let apart = (highest_y(&quarters) - highest_y(&tenths)).abs();
        assert!(apart > 1e-3, "the two cubics part by {apart}");
        assert!(!same_section_curve(&quarters, &tenths, 1e-5));
        assert!(same_section_curve(&quarters, &quarters.clone(), 1e-5));
    }

    /// The plate of [`two_arcs_between_one_pair_of_points`] split by
    /// `sections`, its regions' areas sorted, each read by the shoelace
    /// formula over its wires with each edge read at 2000 points. An edge
    /// runs its curve from its stored start, its samples then turned to the
    /// wire's way, as [`sampling::sample_wire_loop_uv_via_frame`] reads it.
    /// Each outline must wind counter-clockwise about the plate's normal
    /// `+z` and each hole clockwise; a region's area is the sum of its
    /// wires' signed areas.
    fn plate_region_areas(sections: &[SectionEdge]) -> Vec<f64> {
        let (topo, face_id, _) = two_arcs_between_one_pair_of_points();
        let subs = split_face_2d(
            &topo,
            face_id,
            sections,
            Rank::A,
            &brepkit_math::tolerance::Tolerance::new(),
            None,
            None,
            &std::collections::HashMap::new(),
            &std::collections::BTreeSet::new(),
            None,
        );
        let wire_area = |wire: &[OrientedPCurveEdge]| {
            let mut pts = Vec::new();
            for e in wire {
                let (s3, e3) = if e.forward {
                    (e.start_3d, e.end_3d)
                } else {
                    (e.end_3d, e.start_3d)
                };
                let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
                for k in 0..2000 {
                    let f = f64::from(k) / 2000.0;
                    let f = if e.forward { f } else { 1.0 - f };
                    pts.push(
                        e.curve_3d
                            .evaluate_with_endpoints((t1 - t0).madd(f, t0), s3, e3),
                    );
                }
            }
            let mut twice = 0.0;
            for (i, a) in pts.iter().enumerate() {
                let b = pts[(i + 1) % pts.len()];
                twice += a.x() * b.y() - b.x() * a.y();
            }
            0.5 * twice
        };
        let mut areas: Vec<f64> = subs
            .iter()
            .map(|s| {
                let outline = wire_area(&s.outer_wire);
                assert!(outline > 0.0, "an outline winds clockwise: {outline}");
                let holes: Vec<f64> = s.inner_wires.iter().map(|w| wire_area(w)).collect();
                assert!(
                    holes.iter().all(|&h| h < 0.0),
                    "a hole winds counter-clockwise: {holes:?}"
                );
                outline + holes.iter().sum::<f64>()
            })
            .collect();
        areas.sort_by(f64::total_cmp);
        areas
    }

    /// One circle of radius 2 about (5, 5), as two halves between (7, 5)
    /// and (3, 5), each half a turn counter-clockwise from its start: two
    /// trims of one circle with one pair of ends, and the plate splits into
    /// the disc and the rest. An arc of a sixth of a turn, and the rest of
    /// its circle stored from the other end, are two curves too: read by
    /// the shorter way round they are one arc, but read along the circle
    /// from each start they are the two arcs of the circle.
    #[test]
    fn trims_of_one_circle_between_one_pair_of_points_stay_two() {
        use brepkit_math::curves::Circle3D;
        let circle =
            Circle3D::new(Point3::new(5.0, 5.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let trim = |a: f64, b: f64| SectionEdge {
            curve_3d: EdgeCurve::Circle(circle.clone()),
            ..line_section(circle.evaluate(a), circle.evaluate(b))
        };
        let pi = std::f64::consts::PI;
        let (upper, lower) = (trim(0.0, pi), trim(pi, 0.0));
        assert!(!same_section_curve(&upper, &lower, 1e-5));
        let areas = plate_region_areas(&[upper, lower]);
        let want = [4.0 * pi, 100.0 - 4.0 * pi];
        assert_eq!(areas.len(), 2, "regions of areas {areas:?}");
        for (got, want) in areas.iter().zip(want) {
            assert!(
                (got - want).abs() < 1e-4,
                "regions of areas {areas:?}, want {want}"
            );
        }
        let (short, rest) = (trim(0.2, 0.2 + pi / 3.0), trim(0.2 + pi / 3.0, 0.2));
        assert!(!same_section_curve(&short, &rest, 1e-5));
        assert!(!same_section_curve(&rest, &short, 1e-5));
        assert!(same_section_curve(&short, &short.clone(), 1e-5));
    }

    /// The plate of [`two_arcs_between_one_pair_of_points`] laid in the frame
    /// at `o` with axes `u` and `v`, its normal `u x v`: the plate's (x, y)
    /// lies at `o + x u + y v`.
    fn plate_in_frame(o: Point3, u: Vec3, v: Vec3) -> (Topology, FaceId) {
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
            .map(|(x, y)| topo.add_vertex(Vertex::new(o + u * x + v * y, 1e-7)));
        let edges: Vec<_> = (0..4)
            .map(|i| {
                let edge = Edge::new(corners[i], corners[(i + 1) % 4], EdgeCurve::Line);
                OrientedEdge::new(topo.add_edge(edge), true)
            })
            .collect();
        let outer = topo.add_wire(Wire::new(edges, true).unwrap());
        let normal = u.cross(v);
        let face = topo.add_face(Face::new(
            outer,
            vec![],
            FaceSurface::Plane {
                normal,
                d: normal.dot(o - Point3::new(0.0, 0.0, 0.0)),
            },
        ));
        (topo, face)
    }

    /// The area a wire of lines and circular arcs bounds in the frame at `o`
    /// with axes `u` and `v`, signed by its way round about `u x v`. Each
    /// edge adds its chord's term of the shoelace formula, and an arc adds
    /// the segment between it and its chord, `r^2 (phi - sin phi) / 2` for
    /// a sweep `phi`, so the sum is exact.
    fn exact_wire_area(wire: &[OrientedPCurveEdge], o: Point3, u: Vec3, v: Vec3) -> f64 {
        let local = |p: Point3| ((p - o).dot(u), (p - o).dot(v));
        let normal = u.cross(v);
        wire.iter()
            .map(|e| {
                let (s3, e3) = if e.forward {
                    (e.start_3d, e.end_3d)
                } else {
                    (e.end_3d, e.start_3d)
                };
                let (a, b) = (local(s3), local(e3));
                let chord = 0.5 * (a.0 * b.1 - b.0 * a.1);
                let segment = match &e.curve_3d {
                    EdgeCurve::Line => 0.0,
                    EdgeCurve::Circle(c) => {
                        let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
                        let phi = t1 - t0;
                        let r = c.radius();
                        c.normal().dot(normal).signum() * 0.5 * r * r * (phi - phi.sin())
                    }
                    // An ellipse is a circle scaled by its semi-axes, which
                    // scales its segments' areas by their product.
                    EdgeCurve::Ellipse(el) => {
                        let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
                        let phi = t1 - t0;
                        let ab = el.semi_major() * el.semi_minor();
                        el.normal().dot(normal).signum() * 0.5 * ab * (phi - phi.sin())
                    }
                    other @ EdgeCurve::NurbsCurve(_) => unreachable!("a wire edge on {other:?}"),
                };
                if e.forward {
                    chord + segment
                } else {
                    -(chord + segment)
                }
            })
            .sum()
    }

    /// The regions' areas, sorted, of the plate in the frame at `o` with
    /// axes `u` and `v` split by `sections`. Each outline must wind
    /// counter-clockwise about `u x v` and each hole clockwise.
    fn framed_plate_region_areas(
        o: Point3,
        u: Vec3,
        v: Vec3,
        sections: &[SectionEdge],
    ) -> Vec<f64> {
        let (topo, face_id) = plate_in_frame(o, u, v);
        let subs = split_face_2d(
            &topo,
            face_id,
            sections,
            Rank::A,
            &brepkit_math::tolerance::Tolerance::new(),
            None,
            None,
            &std::collections::HashMap::new(),
            &std::collections::BTreeSet::new(),
            None,
        );
        let mut areas: Vec<f64> = subs
            .iter()
            .map(|s| {
                let outline = exact_wire_area(&s.outer_wire, o, u, v);
                assert!(outline > 0.0, "an outline winds clockwise: {outline}");
                let holes: Vec<f64> = s
                    .inner_wires
                    .iter()
                    .map(|w| exact_wire_area(w, o, u, v))
                    .collect();
                assert!(
                    holes.iter().all(|&h| h < 0.0),
                    "a hole winds counter-clockwise: {holes:?}"
                );
                outline + holes.iter().sum::<f64>()
            })
            .collect();
        areas.sort_by(f64::total_cmp);
        areas
    }

    /// The plate's frames: upright, turned about a slant axis and moved,
    /// and that turn mirrored in the plane `x = 0`.
    fn plate_frames() -> [(&'static str, Point3, Vec3, Vec3); 3] {
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn = |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let mirror = |e: Vec3| Vec3::new(-e.x(), e.y(), e.z());
        let (u, v) = (
            turn(Vec3::new(1.0, 0.0, 0.0)),
            turn(Vec3::new(0.0, 1.0, 0.0)),
        );
        let o = Point3::new(3.0, -2.0, 5.0);
        [
            (
                "upright",
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            ("turned", o, u, v),
            (
                "mirrored",
                Point3::new(-o.x(), o.y(), o.z()),
                mirror(u),
                mirror(v),
            ),
        ]
    }

    /// A circle of radius 2 about the plate's (5, 5), cut at two points a
    /// sixth of a turn apart into an arc and the rest of the circle: the
    /// plate splits into the disc and the rest, in each frame. The rest is
    /// stored from the arc's end round to its start, or from the arc's start
    /// about the reversed normal; either way its shorter-arc middle is the
    /// arc's, and only a reading along each curve keeps both.
    #[test]
    fn an_arc_and_the_rest_of_its_circle_split_a_plate_into_the_disc_and_the_rest() {
        use brepkit_math::curves::Circle3D;
        let pi = std::f64::consts::PI;
        for (pose, o, u, v) in plate_frames() {
            let normal = u.cross(v);
            let centre = o + u * 5.0 + v * 5.0;
            let circle = Circle3D::new(centre, normal, 2.0).unwrap();
            let back = Circle3D::new(centre, -normal, 2.0).unwrap();
            let (p, q) = (circle.evaluate(0.2), circle.evaluate(0.2 + pi / 3.0));
            let on = |c: &Circle3D, from: Point3, to: Point3| SectionEdge {
                curve_3d: EdgeCurve::Circle(c.clone()),
                ..line_section(from, to)
            };
            for (how, arc, rest) in [
                (
                    "rest from the arc's end",
                    on(&circle, p, q),
                    on(&circle, q, p),
                ),
                (
                    "rest about the reversed normal",
                    on(&circle, p, q),
                    on(&back, p, q),
                ),
                (
                    "arc about the reversed normal",
                    on(&back, q, p),
                    on(&circle, q, p),
                ),
            ] {
                assert!(
                    (section_middle(&arc) - section_middle(&rest)).length() < 1e-9,
                    "{pose}, {how}: the shorter-arc middles differ"
                );
                let want = [4.0 * pi, 100.0 - 4.0 * pi];
                for sections in [[arc.clone(), rest.clone()], [rest, arc]] {
                    let areas = framed_plate_region_areas(o, u, v, &sections);
                    assert_eq!(areas.len(), 2, "{pose}, {how}: regions of areas {areas:?}");
                    for (got, want) in areas.iter().zip(want) {
                        assert!(
                            (got - want).abs() < 1e-9,
                            "{pose}, {how}: regions of areas {areas:?}, want {want}"
                        );
                    }
                }
            }
        }
    }

    /// A section edge along `curve` from `from` to `to`, its source `src`.
    fn meeting_edge(curve: EdgeCurve, from: Point3, to: Point3, src: usize) -> OrientedPCurveEdge {
        OrientedPCurveEdge {
            curve_3d: curve,
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(0.0, 0.0),
            start_3d: from,
            end_3d: to,
            forward: true,
            source_edge_idx: Some(src),
            pave_block_id: None,
        }
    }

    /// The points where the sections meet inside both, at 1e-7.
    fn meetings(edges: &[OrientedPCurveEdge]) -> Vec<Point3> {
        let found =
            edge_splitting::section_meeting_points(&edge_splitting::section_spans(edges), 1e-7)
                .unwrap();
        assert!(found.iter().all(|m| m.apart <= 1e-7), "{found:?}");
        found.into_iter().map(|m| m.at).collect()
    }

    /// The half of a circle of radius 2 about the plate's (5, 5) from angle 0
    /// round to pi, and the half of a circle of radius 0.5 outside it that
    /// touches it at angle 0.02, `gap` off. At angle 0.02 the touch lies
    /// within the first of 64 even steps along the first, nearer its start.
    fn touch_near_an_arcs_start(
        o: Point3,
        u: Vec3,
        v: Vec3,
        gap: f64,
    ) -> (Vec<OrientedPCurveEdge>, Point3) {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::PI;
        let n = u.cross(v);
        let centre = o + u * 5.0 + v * 5.0;
        let big = Circle3D::with_axes(centre, n, 2.0, u, v).unwrap();
        let (sin, cos) = 0.02f64.sin_cos();
        let out = u * cos + v * sin;
        let small = Circle3D::with_axes(centre + out * (2.5 + gap), n, 0.5, u, v).unwrap();
        let touch = centre + out * 2.0;
        let edges = vec![
            meeting_edge(
                EdgeCurve::Circle(big.clone()),
                big.evaluate(0.0),
                big.evaluate(PI),
                1,
            ),
            meeting_edge(
                EdgeCurve::Circle(small.clone()),
                small.evaluate(0.02 + 0.5 * PI),
                small.evaluate(0.02 + 1.5 * PI),
                2,
            ),
        ];
        (edges, touch)
    }

    /// Two circles touching a fiftieth of a radian from one's start, nearer
    /// it than its first sample at 64 even steps: the touch is found in each
    /// frame.
    #[test]
    fn a_touch_between_samples_near_an_arcs_start_is_found() {
        for (pose, o, u, v) in plate_frames() {
            let (edges, touch) = touch_near_an_arcs_start(o, u, v, 0.0);
            let found = meetings(&edges);
            assert!(
                found.len() == 1 && (found[0] - touch).length() < 1e-6,
                "{pose}: meetings {found:?}, want {touch:?}"
            );
        }
    }

    /// The same two circles two tolerances apart do not meet, in each frame.
    #[test]
    fn circles_two_tolerances_apart_do_not_meet() {
        for (pose, o, u, v) in plate_frames() {
            let (edges, _) = touch_near_an_arcs_start(o, u, v, 2e-7);
            let found = meetings(&edges);
            assert!(found.is_empty(), "{pose}: meetings {found:?}");
        }
    }

    /// A circle of radius 1 on axes of length 2, so of radius 2 read along
    /// them, over (-0.5, 0.5), and the line `x = 2` from `y = -1` to `1`:
    /// they touch at (2, 0), in each frame.
    #[test]
    fn a_circle_on_long_axes_touching_a_line_is_found() {
        use brepkit_math::curves::Circle3D;
        for (pose, o, u, v) in plate_frames() {
            let at = |x: f64, y: f64| o + u * x + v * y;
            let circle =
                Circle3D::with_axes(at(0.0, 0.0), u.cross(v), 1.0, u * 2.0, v * 2.0).unwrap();
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Circle(circle.clone()),
                    circle.evaluate(-0.5),
                    circle.evaluate(0.5),
                    1,
                ),
                meeting_edge(EdgeCurve::Line, at(2.0, -1.0), at(2.0, 1.0), 2),
            ];
            let found = meetings(&edges);
            assert!(
                found.len() == 1 && (found[0] - at(2.0, 0.0)).length() < 1e-6,
                "{pose}: meetings {found:?}"
            );
        }
    }

    /// Ellipses of semi-axes 3 and 2 about one centre, the second turned a
    /// quarter: their arcs over the first quadrant cross at
    /// `(6 / sqrt(13), 6 / sqrt(13))`, in each frame.
    #[test]
    fn concentric_ellipses_turned_a_quarter_cross() {
        use brepkit_math::curves::Ellipse3D;
        use std::f64::consts::FRAC_PI_2;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let first = Ellipse3D::with_axes(o, n, 3.0, 2.0, u, v).unwrap();
            let turned = Ellipse3D::with_axes(o, n, 3.0, 2.0, v, -u).unwrap();
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Ellipse(first.clone()),
                    first.evaluate(0.0),
                    first.evaluate(FRAC_PI_2),
                    1,
                ),
                meeting_edge(
                    EdgeCurve::Ellipse(turned.clone()),
                    turned.evaluate(-FRAC_PI_2),
                    turned.evaluate(0.0),
                    2,
                ),
            ];
            let s = 6.0 / 13.0f64.sqrt();
            let want = o + (u + v) * s;
            let found = meetings(&edges);
            assert!(
                found.len() == 1 && (found[0] - want).length() < 1e-6,
                "{pose}: meetings {found:?}, want {want:?}"
            );
        }
    }

    /// The circle and the ellipse of
    /// [`a_circle_and_an_ellipse_through_three_points_split_a_plate_in_four`],
    /// and a line from (7.000005, 5), fifty tolerances off both at their
    /// touch, to (9, 5): the touch at (7, 5) is still found, in each frame.
    #[test]
    fn a_point_off_both_curves_does_not_hide_their_touch() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        use std::f64::consts::PI;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let circle = Circle3D::with_axes(at(5.0, 5.0), n, 2.0, u, v).unwrap();
            let ellipse = Ellipse3D::with_axes(at(4.0, 5.0), n, 3.0, 5.4f64.sqrt(), u, v).unwrap();
            let (p, q) = (circle.evaluate(-PI / 3.0), circle.evaluate(PI / 3.0));
            let edges = vec![
                meeting_edge(EdgeCurve::Circle(circle), p, q, 1),
                meeting_edge(EdgeCurve::Ellipse(ellipse), p, q, 2),
                meeting_edge(EdgeCurve::Line, at(7.000_005, 5.0), at(9.0, 5.0), 3),
            ];
            let found = meetings(&edges);
            assert!(
                found.len() == 1 && (found[0] - at(7.0, 5.0)).length() < 1e-6,
                "{pose}: meetings {found:?}"
            );
        }
    }

    /// The regions of the plate split by `sections` match `inner`, the
    /// regions inside the plate's edge, and the rest of the plate, in each
    /// frame `sections(o, u, v)` is laid in.
    fn plate_regions_match(
        sections: &dyn Fn(Point3, Vec3, Vec3) -> Vec<SectionEdge>,
        inner: &[f64],
        within: f64,
    ) {
        let mut want: Vec<f64> = inner.to_vec();
        want.push(100.0 - inner.iter().sum::<f64>());
        want.sort_by(f64::total_cmp);
        for (pose, o, u, v) in plate_frames() {
            let areas = framed_plate_region_areas(o, u, v, &sections(o, u, v));
            assert_eq!(
                areas.len(),
                want.len(),
                "{pose}: regions of areas {areas:?}, want {want:?}"
            );
            for (got, want) in areas.iter().zip(&want) {
                assert!(
                    (got - want).abs() < within,
                    "{pose}: regions of areas {areas:?}, want {want}"
                );
            }
        }
    }

    /// A section along `curve` from `start` to `end`.
    fn curve_section(curve: EdgeCurve, start: Point3, end: Point3) -> SectionEdge {
        SectionEdge {
            curve_3d: curve,
            ..line_section(start, end)
        }
    }

    /// Ellipses about one centre on one frame, of semi-axes (1000.00005,
    /// 0.9999) and (1000, 1), over (-0.03, 0.03), cross twice about 0.0447
    /// apart near the end of their long axes, and part by 5e-5 between: two
    /// meetings, in each frame.
    #[test]
    fn ellipses_that_cross_twice_near_a_long_axis_end_meet_twice() {
        use brepkit_math::curves::Ellipse3D;
        let (aa, ab) = (1000.0f64, 1.0f64);
        let (ba, bb) = (1_000.000_05_f64, 0.9999f64);
        let (alpha, beta) = ((aa / ba) * (aa / ba), (ab / bb) * (ab / bb));
        let t = ((1.0 - alpha) / (beta - alpha)).sqrt().asin();
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let first = Ellipse3D::with_axes(at(-995.0, 5.0), n, ba, bb, u, v).unwrap();
            let second = Ellipse3D::with_axes(at(-995.0, 5.0), n, aa, ab, u, v).unwrap();
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Ellipse(first.clone()),
                    first.evaluate(-0.03),
                    first.evaluate(0.03),
                    1,
                ),
                meeting_edge(
                    EdgeCurve::Ellipse(second.clone()),
                    second.evaluate(-0.03),
                    second.evaluate(0.03),
                    2,
                ),
            ];
            let found = meetings(&edges);
            let crossings = [second.evaluate(-t), second.evaluate(t)];
            assert!(
                found.len() == 2
                    && crossings
                        .iter()
                        .all(|&x| found.iter().any(|&f| (f - x).length() < 1e-6)),
                "{pose}: meetings {found:?}, want {crossings:?}"
            );
        }
    }

    /// Unit circles about the plate's (5, 5) and (5.000005, 5), over a
    /// quarter turn on each side of the top, cross at x = 5.0000025: one
    /// meeting, in each frame, though they lie within 50 tolerances.
    #[test]
    fn circles_fifty_tolerances_apart_cross_where_they_meet() {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::FRAC_PI_4;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let first = Circle3D::with_axes(at(5.0, 5.0), n, 1.0, u, v).unwrap();
            let second = Circle3D::with_axes(at(5.000_005, 5.0), n, 1.0, u, v).unwrap();
            let arc = |c: &Circle3D, src| {
                meeting_edge(
                    EdgeCurve::Circle(c.clone()),
                    c.evaluate(FRAC_PI_4),
                    c.evaluate(3.0 * FRAC_PI_4),
                    src,
                )
            };
            let found = meetings(&[arc(&first, 1), arc(&second, 2)]);
            let crossing = at(5.000_002_5, 5.0 + (1.0 - 6.25e-12f64).sqrt());
            assert!(
                found.len() == 1 && (found[0] - crossing).length() < 1e-9,
                "{pose}: meetings {found:?}, want {crossing:?}"
            );
        }
    }

    /// Ellipses of semi-axes 300000 and 200000 about one centre, one turned
    /// 5e-10 from the other, over 0.01 inside (-pi/4, 3pi/4): alike within 100
    /// tolerances along their axes but more than 400 apart between, they
    /// cross twice, near the ends of the axes, in each frame.
    #[test]
    fn large_ellipses_turned_a_little_cross_near_their_axes() {
        use brepkit_math::curves::Ellipse3D;
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let (a, b, turn) = (300_000.0f64, 200_000.0f64, 5e-10f64);
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let centre = o + u * (5.0 - a) + v * 5.0;
            let (ts, tc) = turn.sin_cos();
            let (tu, tv) = (u * tc + v * ts, v * tc - u * ts);
            let first = Ellipse3D::with_axes(centre, n, a, b, u, v).unwrap();
            let second = Ellipse3D::with_axes(centre, n, a, b, tu, tv).unwrap();
            let arc = |e: &Ellipse3D, src| {
                meeting_edge(
                    EdgeCurve::Ellipse(e.clone()),
                    e.evaluate(0.01 - FRAC_PI_4),
                    e.evaluate(3.0 * FRAC_PI_4 - 0.01),
                    src,
                )
            };
            let found = meetings(&[arc(&first, 1), arc(&second, 2)]);
            // Congruent ellipses turned about their centre cross where the
            // polar angle is half the turn, and a quarter turn on.
            let crossings = [0.5 * turn, FRAC_PI_2 + 0.5 * turn]
                .map(|f: f64| first.evaluate((a * f.sin()).atan2(b * f.cos())));
            assert!(
                found.len() == 2
                    && crossings
                        .iter()
                        .all(|&x| found.iter().any(|&f| (f - x).length() < 10.0)),
                "{pose}: meetings {found:?}, want {crossings:?}"
            );
        }
    }

    /// Ellipses on one frame about the plate's (5, 5), of semi-axes (3, 2)
    /// and (3.00001, 2), 1e-5 apart at the ends of their long axes, over
    /// 0.01 inside half a turn, touch at (5, 7): one meeting, in each frame.
    #[test]
    fn ellipses_apart_along_one_axis_touch_at_the_other() {
        use brepkit_math::curves::Ellipse3D;
        use std::f64::consts::PI;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let first = Ellipse3D::with_axes(at(5.0, 5.0), n, 3.0, 2.0, u, v).unwrap();
            let second = Ellipse3D::with_axes(at(5.0, 5.0), n, 3.000_01, 2.0, u, v).unwrap();
            let arc = |e: &Ellipse3D, src| {
                meeting_edge(
                    EdgeCurve::Ellipse(e.clone()),
                    e.evaluate(0.01),
                    e.evaluate(PI - 0.01),
                    src,
                )
            };
            let found = meetings(&[arc(&first, 1), arc(&second, 2)]);
            assert!(
                found.len() == 1 && (found[0] - at(5.0, 7.0)).length() < 1e-6,
                "{pose}: meetings {found:?}"
            );
        }
    }

    /// A circle of radius 2 about the plate's (5, 5) and the ellipse of
    /// semi-axes (2, 2) on its frame are one curve: over one span, with a
    /// chord, they split the plate into the segment and the rest; over spans
    /// apart, each with its chord, into two segments and the rest; over
    /// spans that overlap they do not meet. In each frame.
    #[test]
    fn a_circle_and_a_round_ellipse_on_its_frame_are_one_curve() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        use std::f64::consts::PI;
        let segment = 2.0 * (2.0 * PI / 3.0 - (2.0 * PI / 3.0).sin());
        let curves = |o: Point3, u: Vec3, v: Vec3| {
            let n = u.cross(v);
            let centre = o + u * 5.0 + v * 5.0;
            (
                Circle3D::with_axes(centre, n, 2.0, u, v).unwrap(),
                Ellipse3D::with_axes(centre, n, 2.0, 2.0, u, v).unwrap(),
            )
        };
        let one_span = |o: Point3, u: Vec3, v: Vec3| {
            let (circle, ellipse) = curves(o, u, v);
            let (p, q) = (circle.evaluate(-PI / 3.0), circle.evaluate(PI / 3.0));
            vec![
                curve_section(EdgeCurve::Circle(circle), p, q),
                curve_section(EdgeCurve::Ellipse(ellipse), p, q),
                line_section(q, p),
            ]
        };
        plate_regions_match(&one_span, &[segment], 1e-9);
        let apart = |o: Point3, u: Vec3, v: Vec3| {
            let (circle, ellipse) = curves(o, u, v);
            let (p, q) = (circle.evaluate(-PI / 3.0), circle.evaluate(PI / 3.0));
            let (r, s) = (
                ellipse.evaluate(2.0 * PI / 3.0),
                ellipse.evaluate(4.0 * PI / 3.0),
            );
            vec![
                curve_section(EdgeCurve::Circle(circle), p, q),
                line_section(q, p),
                curve_section(EdgeCurve::Ellipse(ellipse), r, s),
                line_section(s, r),
            ]
        };
        plate_regions_match(&apart, &[segment, segment], 1e-9);
        for (pose, o, u, v) in plate_frames() {
            let (circle, ellipse) = curves(o, u, v);
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Circle(circle.clone()),
                    circle.evaluate(-PI / 3.0),
                    circle.evaluate(PI / 3.0),
                    1,
                ),
                meeting_edge(
                    EdgeCurve::Ellipse(ellipse.clone()),
                    ellipse.evaluate(0.0),
                    ellipse.evaluate(2.0 * PI / 3.0),
                    2,
                ),
            ];
            let found = meetings(&edges);
            assert!(found.is_empty(), "{pose}: meetings {found:?}");
        }
    }

    /// Unit circles about the plate's (4, 5) and (6.0000001, 5), the halves
    /// that face each other, lie exactly one tolerance apart at their
    /// nearest: whether they meet there is not settled, and the finder
    /// says so, in each frame.
    #[test]
    fn circles_one_tolerance_apart_are_not_settled() {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::FRAC_PI_2;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let first = Circle3D::with_axes(at(4.0, 5.0), n, 1.0, u, v).unwrap();
            let second = Circle3D::with_axes(at(6.000_000_1, 5.0), n, 1.0, u, v).unwrap();
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Circle(first.clone()),
                    first.evaluate(-FRAC_PI_2),
                    first.evaluate(FRAC_PI_2),
                    1,
                ),
                meeting_edge(
                    EdgeCurve::Circle(second.clone()),
                    second.evaluate(FRAC_PI_2),
                    second.evaluate(3.0 * FRAC_PI_2),
                    2,
                ),
            ];
            let spans = edge_splitting::section_spans(&edges);
            let found = edge_splitting::section_meeting_points(&spans, 1e-7);
            assert!(found.is_err(), "{pose}: meetings {found:?}");
        }
    }

    /// A circle of radius 2 about the plate's (5, 5) over (-0.5, 0.5), and
    /// the segment from (7.0000001, 5.5) to (7.0000001, 5.8): the segment's
    /// line lies one tolerance off the circle at (7, 5), but the segment
    /// itself lies more than 0.06 from the arc. They are apart, in each
    /// frame.
    #[test]
    fn an_arc_and_a_segment_whose_line_touches_it_elsewhere_are_apart() {
        use brepkit_math::curves::Circle3D;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let circle = Circle3D::with_axes(at(5.0, 5.0), n, 2.0, u, v).unwrap();
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Circle(circle.clone()),
                    circle.evaluate(-0.5),
                    circle.evaluate(0.5),
                    1,
                ),
                meeting_edge(
                    EdgeCurve::Line,
                    at(7.000_000_1, 5.5),
                    at(7.000_000_1, 5.8),
                    2,
                ),
            ];
            let found = meetings(&edges);
            assert!(found.is_empty(), "{pose}: meetings {found:?}");
        }
    }

    /// Unit circles about the plate's (4, 5) and (6.0000001, 5), one
    /// tolerance apart at their nearest: one arc over a half turn about its
    /// nearest point, the other over the quarter turn about its top, far
    /// from its own. They are apart whichever is read along, in each frame.
    #[test]
    fn arcs_whose_circles_touch_beyond_one_of_them_are_apart() {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let left = Circle3D::with_axes(at(4.0, 5.0), n, 1.0, u, v).unwrap();
            let right = Circle3D::with_axes(at(6.000_000_1, 5.0), n, 1.0, u, v).unwrap();
            let arc = |c: &Circle3D, from: f64, to: f64, src| {
                meeting_edge(
                    EdgeCurve::Circle(c.clone()),
                    c.evaluate(from),
                    c.evaluate(to),
                    src,
                )
            };
            let near = arc(&left, -FRAC_PI_2, FRAC_PI_2, 1);
            let top = arc(&right, FRAC_PI_4, 3.0 * FRAC_PI_4, 2);
            for edges in [vec![near.clone(), top.clone()], vec![top, near]] {
                let found = meetings(&edges);
                assert!(found.is_empty(), "{pose}: meetings {found:?}");
            }
        }
    }

    /// Three unit circles about the plate's (3, 5), (4, 5) and (5.0000001,
    /// 5), each cut into three arcs: the first two cross, and the first and
    /// the last lie one tolerance apart at their nearest, which the finder
    /// cannot settle. The loops are not proved apart, so the face is not
    /// split into discs, in each frame.
    #[test]
    fn loops_that_cross_beside_a_pair_not_settled_are_refused() {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::PI;
        let cuts = [0.5 * PI, 7.0 * PI / 6.0, 11.0 * PI / 6.0, 2.5 * PI];
        let sections = |o: Point3, u: Vec3, v: Vec3| {
            let n = u.cross(v);
            let mut sections = Vec::new();
            for x in [3.0, 4.0, 5.000_000_1] {
                let c = Circle3D::with_axes(o + u * x + v * 5.0, n, 1.0, u, v).unwrap();
                let ends: Vec<Point3> = cuts.iter().map(|&t| c.evaluate(t)).collect();
                for w in 0..3 {
                    let end = if w == 2 { ends[0] } else { ends[w + 1] };
                    sections.push(curve_section(EdgeCurve::Circle(c.clone()), ends[w], end));
                }
            }
            sections
        };
        for (pose, o, u, v) in plate_frames() {
            let areas = framed_plate_region_areas(o, u, v, &sections(o, u, v));
            assert!(areas.is_empty(), "{pose}: regions of areas {areas:?}");
        }
    }

    /// An arc of the ellipse of semi-axes 2e8 along the plate's x and 2
    /// about (100000005, 5), its centre 1e8 off, over 4 units about the
    /// parameter `2 pi / 3`, where it passes (5, 5 + sqrt 3): flat to well
    /// within a tolerance, its ends on the ellipse to twice the precision of
    /// one. In each frame, with the arc and `other(o, u, v, ends)`.
    fn flat_arc_of_an_ellipse_centred_far_off(
        other: &dyn Fn(Point3, Vec3, Vec3, (Point3, Point3)) -> OrientedPCurveEdge,
    ) -> Vec<(&'static str, Vec<OrientedPCurveEdge>, Point3)> {
        use brepkit_math::curves::Ellipse3D;
        let third = 2.0 * std::f64::consts::PI / 3.0;
        let half = 2.0 / (2e8 * third.sin());
        plate_frames()
            .into_iter()
            .map(|(pose, o, u, v)| {
                let at = |x: f64, y: f64| o + u * x + v * y;
                let ellipse = EdgeCurve::Ellipse(
                    Ellipse3D::with_axes(at(5.0 + 1e8, 5.0), u.cross(v), 2e8, 2.0, u, v).unwrap(),
                );
                let ends = (
                    edge_splitting::conic_point(&ellipse, third - half).unwrap(),
                    edge_splitting::conic_point(&ellipse, third + half).unwrap(),
                );
                let edges = vec![
                    meeting_edge(ellipse, ends.0, ends.1, 1),
                    other(o, u, v, ends),
                ];
                (pose, edges, at(5.0, 5.0 + 3.0f64.sqrt()))
            })
            .collect()
    }

    /// The flat arc of [`flat_arc_of_an_ellipse_centred_far_off`] and the
    /// line `x = 5` from `y = 6` to `7.5` across it meet where they cross,
    /// in each frame.
    #[test]
    fn a_flat_arc_of_an_ellipse_centred_far_off_meets_a_line_across_it() {
        let across = |o: Point3, u: Vec3, v: Vec3, _: (Point3, Point3)| {
            meeting_edge(
                EdgeCurve::Line,
                o + u * 5.0 + v * 6.0,
                o + u * 5.0 + v * 7.5,
                2,
            )
        };
        for (pose, edges, crossing) in flat_arc_of_an_ellipse_centred_far_off(&across) {
            let found = meetings(&edges);
            assert!(
                found.len() == 1 && (found[0] - crossing).length() < 1e-6,
                "{pose}: meetings {found:?}, want {crossing:?}"
            );
        }
    }

    /// The flat arc of [`flat_arc_of_an_ellipse_centred_far_off`] and the
    /// segment along its chord two tolerances off it are apart, in each
    /// frame.
    #[test]
    fn a_flat_arc_two_tolerances_from_a_line_is_apart() {
        let beside = |_: Point3, u: Vec3, v: Vec3, (p, q): (Point3, Point3)| {
            let off = u.cross(v).cross((q - p).normalize().unwrap()) * 2e-7;
            meeting_edge(EdgeCurve::Line, p + off, q + off, 2)
        };
        for (pose, edges, _) in flat_arc_of_an_ellipse_centred_far_off(&beside) {
            let found = meetings(&edges);
            assert!(found.is_empty(), "{pose}: meetings {found:?}");
        }
    }

    /// A rational quadratic from the plate's (0, 0) to (10, 0) about (5, 5)
    /// with weights 1, -0.5 and 1, which passes (5, -5), and the line from
    /// (5, -8) to it: the curve is split there, and each piece and its
    /// reverse twin keep one id, though no bound reads the curve as one with
    /// itself. In each frame.
    #[test]
    fn a_piece_and_its_twin_keep_one_id_whatever_their_curve() {
        use brepkit_math::nurbs::NurbsCurve;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let frame = PlaneFrame::from_normal_and_point(n, o);
            let surface = FaceSurface::Plane {
                normal: n,
                d: n.dot(o - Point3::new(0.0, 0.0, 0.0)),
            };
            let curve = EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    vec![at(0.0, 0.0), at(5.0, 5.0), at(10.0, 0.0)],
                    vec![1.0, -0.5, 1.0],
                )
                .unwrap(),
            );
            let both = |curve: &EdgeCurve, from: Point3, to: Point3, src: usize| {
                let forward = OrientedPCurveEdge {
                    start_uv: frame.project(from),
                    end_uv: frame.project(to),
                    ..meeting_edge(curve.clone(), from, to, src)
                };
                let reverse = OrientedPCurveEdge {
                    start_uv: forward.end_uv,
                    end_uv: forward.start_uv,
                    start_3d: to,
                    end_3d: from,
                    forward: false,
                    ..forward.clone()
                };
                [forward, reverse]
            };
            let mut edges: Vec<OrientedPCurveEdge> = both(&curve, at(0.0, 0.0), at(10.0, 0.0), 0)
                .into_iter()
                .chain(both(&EdgeCurve::Line, at(5.0, -8.0), at(5.0, -5.0), 1))
                .collect();
            split_sections_at_t_junctions(
                &mut edges,
                0,
                &surface,
                Some(&frame),
                &[],
                1e-7,
                None,
                &[],
            );
            let pieces: Vec<&OrientedPCurveEdge> = edges
                .iter()
                .filter(|e| matches!(e.curve_3d, EdgeCurve::NurbsCurve(_)))
                .collect();
            assert_eq!(pieces.len(), 4, "{pose}: pieces {pieces:?}");
            for piece in pieces.iter().filter(|e| e.forward) {
                let twin = pieces.iter().find(|e| {
                    !e.forward
                        && (e.start_3d - piece.end_3d).length() < 1e-9
                        && (e.end_3d - piece.start_3d).length() < 1e-9
                });
                assert!(
                    twin.is_some_and(|t| t.source_edge_idx == piece.source_edge_idx),
                    "{pose}: piece {piece:?} and twin {twin:?}"
                );
            }
        }
    }

    /// Two segments from about (1e10, 1e10) one unit along each axis, the
    /// second one unit of rounding off the first in its largest coordinate,
    /// about 1e-6 apart: they never meet, in each frame.
    #[test]
    fn segments_far_off_a_unit_of_rounding_apart_never_meet() {
        for (pose, o, u, v) in plate_frames() {
            let at = |x: f64, y: f64| o + u * x + v * y;
            let (p0, p1) = (at(1e10, 1e10), at(1e10 + 1.0, 1e10 + 1.0));
            let axis = [p0.x(), p0.y(), p0.z()]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .map_or(0, |(k, _)| k);
            let bump = |p: Point3| {
                let mut c = [p.x(), p.y(), p.z()];
                c[axis] = c[axis].next_up();
                Point3::new(c[0], c[1], c[2])
            };
            let line = EdgeCurve::Line;
            let (a, b) = (
                edge_splitting::SectionSpan {
                    curve: &line,
                    start: p0,
                    end: p1,
                },
                edge_splitting::SectionSpan {
                    curve: &line,
                    start: bump(p0),
                    end: bump(p1),
                },
            );
            let found = edge_splitting::segments_meet((&a, 0.0), (&b, 0.0), 1e-7);
            assert!(!matches!(found, Ok(Some(_))), "{pose}: {found:?}");
        }
    }

    /// The flat arc of a circle of radius 1e9 over the plate's x from 4 to
    /// 6 near y = 5, and a B-spline whose chord lies one tolerance above the
    /// arc's but which bows a half up between: the finder leaves the
    /// B-spline out and does not refuse the pair, in each frame.
    #[test]
    fn a_flat_arc_beside_a_b_spline_whose_chord_touches_it_is_left_alone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::nurbs::NurbsCurve;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let circle = EdgeCurve::Circle(
                Circle3D::with_axes(at(5.0, 5.0 - 1e9), n, 1e9, v, n.cross(v)).unwrap(),
            );
            let (right, left) = (
                edge_splitting::conic_point(&circle, -1e-9).unwrap(),
                edge_splitting::conic_point(&circle, 1e-9).unwrap(),
            );
            let lift = v * 1e-7;
            let spline = EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    vec![right + lift - u * 0.5, at(5.0, 6.0), left + lift + u * 0.5],
                    vec![1.0, 1.0, 1.0],
                )
                .unwrap(),
            );
            let edges = vec![
                meeting_edge(circle, right, left, 1),
                meeting_edge(spline, right + lift - u * 0.5, left + lift + u * 0.5, 2),
            ];
            let found = meetings(&edges);
            assert!(found.is_empty(), "{pose}: meetings {found:?}");
        }
    }

    /// Circles of radius 1e-3, 1, 1e6 and 1e9, about the plate's (5, 5) and
    /// 1e8 along its x from there, each over half a radian up to an end at
    /// angle 0 (the seam) or 1, and a circle of the same radius across each
    /// two and a half tolerances past that end: they meet nowhere on both
    /// sections, in each frame.
    #[test]
    fn an_arc_does_not_meet_another_past_its_end() {
        use brepkit_math::curves::Circle3D;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            for r in [1e-3, 1.0, 1e6, 1e9] {
                for offset in [0.0, 1e8] {
                    for end in [0.0f64, 1.0] {
                        let b = Circle3D::with_axes(o + u * (5.0 + offset) + v * 5.0, n, r, u, v)
                            .unwrap();
                        let past = end + 2.5e-7 / r;
                        let across = u * (-past.sin()) + v * past.cos();
                        let a = Circle3D::with_axes(
                            b.evaluate(past) + across * r,
                            n,
                            r,
                            -across,
                            n.cross(-across),
                        )
                        .unwrap();
                        let edges = vec![
                            meeting_edge(
                                EdgeCurve::Circle(b.clone()),
                                b.evaluate(end - 0.5),
                                b.evaluate(end),
                                1,
                            ),
                            meeting_edge(
                                EdgeCurve::Circle(a.clone()),
                                a.evaluate(-0.3),
                                a.evaluate(0.3),
                                2,
                            ),
                        ];
                        let spans = edge_splitting::section_spans(&edges);
                        let found = edge_splitting::section_meeting_points(&spans, 1e-7);
                        assert!(
                            !matches!(&found, Ok(m) if !m.is_empty()),
                            "{pose}, radius {r:e}, offset {offset:e}, end {end}: {found:?}"
                        );
                    }
                }
            }
        }
    }

    /// The ellipse of semi-axes 3 along the plate's x and 1 about (5, 5)
    /// over `pi / 4` to `7 pi / 6`, and the segment from (4, 6) to (6, 6),
    /// which touches it at (5, 6) far from both its ends: one meeting there,
    /// in each frame.
    #[test]
    fn a_long_ellipse_arc_touched_far_from_its_ends_meets() {
        use brepkit_math::curves::Ellipse3D;
        use std::f64::consts::PI;
        for (pose, o, u, v) in plate_frames() {
            let at = |x: f64, y: f64| o + u * x + v * y;
            let ellipse = Ellipse3D::with_axes(at(5.0, 5.0), u.cross(v), 3.0, 1.0, u, v).unwrap();
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Ellipse(ellipse.clone()),
                    ellipse.evaluate(0.25 * PI),
                    ellipse.evaluate(7.0 * PI / 6.0),
                    1,
                ),
                meeting_edge(EdgeCurve::Line, at(4.0, 6.0), at(6.0, 6.0), 2),
            ];
            let found = meetings(&edges);
            assert!(
                found.len() == 1 && (found[0] - at(5.0, 6.0)).length() < 1e-6,
                "{pose}: meetings {found:?}"
            );
        }
    }

    /// The ellipse of [`a_long_ellipse_arc_touched_far_from_its_ends_meets`]
    /// over all but a hundredth of a half turn from `pi / 4`: a segment
    /// across it at (5, 6) meets it there, and one across its curve two and
    /// a half tolerances past either end meets it nowhere, in each frame.
    #[test]
    fn an_ellipse_arc_meets_inside_its_ends_and_not_past_them() {
        use brepkit_math::curves::Ellipse3D;
        use std::f64::consts::PI;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let ellipse = Ellipse3D::with_axes(at(5.0, 5.0), n, 3.0, 1.0, u, v).unwrap();
            let (t0, t1) = (0.25 * PI, 0.25 * PI + 0.99 * PI);
            let arc = meeting_edge(
                EdgeCurve::Ellipse(ellipse.clone()),
                ellipse.evaluate(t0),
                ellipse.evaluate(t1),
                1,
            );
            let across = meeting_edge(EdgeCurve::Line, at(5.0, 5.5), at(5.0, 6.5), 2);
            let found = meetings(&[arc.clone(), across]);
            assert!(
                found.len() == 1 && (found[0] - at(5.0, 6.0)).length() < 1e-6,
                "{pose}: meetings {found:?}"
            );
            for (end, out) in [(t0, -1.0), (t1, 1.0)] {
                let ahead = u * (-3.0 * end.sin()) + v * end.cos();
                let past = end + out * 2.5e-7 / ahead.length();
                let x = ellipse.evaluate(past);
                let normal = n.cross(ahead).normalize().unwrap();
                let edges = [
                    arc.clone(),
                    meeting_edge(EdgeCurve::Line, x - normal * 0.5, x + normal * 0.5, 2),
                ];
                let found = edge_splitting::section_meeting_points(
                    &edge_splitting::section_spans(&edges),
                    1e-7,
                );
                assert!(
                    !matches!(&found, Ok(m) if !m.is_empty()),
                    "{pose}, past the end at {end:.3}: {found:?}"
                );
            }
        }
    }

    /// Segments of length 1e-3, 1 and 1e6 from the plate's (5, 5) and from
    /// 1e8 along its x, each crossed on its line two tolerances past its end
    /// by a segment square to it: they never meet, in each frame.
    #[test]
    fn segments_crossing_past_an_end_never_meet() {
        for (pose, o, u, v) in plate_frames() {
            let at = |x: f64, y: f64| o + u * x + v * y;
            let line = EdgeCurve::Line;
            for length in [1e-3, 1.0, 1e6] {
                for offset in [0.0, 1e8] {
                    let past = 5.0 + offset + length + 2e-7;
                    let (a, b) = (
                        edge_splitting::SectionSpan {
                            curve: &line,
                            start: at(5.0 + offset, 5.0),
                            end: at(5.0 + offset + length, 5.0),
                        },
                        edge_splitting::SectionSpan {
                            curve: &line,
                            start: at(past, 4.0),
                            end: at(past, 6.0),
                        },
                    );
                    for (x, y) in [(&a, &b), (&b, &a)] {
                        let found = edge_splitting::segments_meet((x, 0.0), (y, 0.0), 1e-7);
                        assert!(
                            !matches!(found, Ok(Some(_))),
                            "{pose}, length {length:e}, offset {offset:e}: {found:?}"
                        );
                    }
                }
            }
        }
    }

    /// The flat arc of a circle of radius 1e9 over the plate's x from 4 to 6
    /// near y = 5, and a segment square to its chord two tolerances past
    /// its end at x = 6, and another along its chord over x from 5 to 7:
    /// neither meets it inside both, in each frame.
    #[test]
    fn a_flat_arc_meets_no_segment_past_its_end_or_along_its_chord() {
        use brepkit_math::curves::Circle3D;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let circle = EdgeCurve::Circle(
                Circle3D::with_axes(at(5.0, 5.0 - 1e9), n, 1e9, v, n.cross(v)).unwrap(),
            );
            let (right, left) = (
                edge_splitting::conic_point(&circle, -1e-9).unwrap(),
                edge_splitting::conic_point(&circle, 1e-9).unwrap(),
            );
            let along = (right - left) * (1.0 / (right - left).length());
            let across = n.cross(along);
            let past = right + along * 2e-7;
            for other in [
                meeting_edge(EdgeCurve::Line, past - across, past + across, 2),
                meeting_edge(EdgeCurve::Line, right - along, right + along, 2),
            ] {
                let edges = vec![meeting_edge(circle.clone(), right, left, 1), other];
                let found = edge_splitting::section_meeting_points(
                    &edge_splitting::section_spans(&edges),
                    1e-7,
                );
                assert!(
                    !matches!(&found, Ok(m) if !m.is_empty()),
                    "{pose}: {found:?}"
                );
            }
        }
    }

    /// A half turn of a circle of radius 2 or an ellipse of semi-axes 3 and
    /// 1.5 about the plate's (5, 5) from each of several angles, with its
    /// chord: given twice, as is or the second stored the other way round,
    /// it bounds the half and the rest with one arc; with the other half
    /// from the same ends, the two halves and the rest. In each frame.
    #[test]
    fn half_turns_with_the_same_ends_are_one_on_one_half_and_two_on_two() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        use std::f64::consts::PI;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let centre = o + u * 5.0 + v * 5.0;
            for (curve, ab) in [
                (
                    EdgeCurve::Circle(Circle3D::with_axes(centre, n, 2.0, u, v).unwrap()),
                    4.0,
                ),
                (
                    EdgeCurve::Ellipse(Ellipse3D::with_axes(centre, n, 3.0, 1.5, u, v).unwrap()),
                    4.5,
                ),
            ] {
                let half = 0.5 * PI * ab;
                let reversed = curve.reversed().unwrap();
                for start in [0.0, 0.3, 0.5 * PI, PI, 1.7] {
                    let point = |t: f64| curve.evaluate_with_endpoints(t, centre, centre);
                    let (p, q) = (point(start), point(start + PI));
                    for case in ["twice", "reversed", "opposite"] {
                        let second = match case {
                            "twice" => curve_section(curve.clone(), p, q),
                            "reversed" => curve_section(reversed.clone(), q, p),
                            _ => curve_section(curve.clone(), q, p),
                        };
                        let sections = [
                            curve_section(curve.clone(), p, q),
                            second,
                            line_section(q, p),
                        ];
                        let (topo, face_id) = plate_in_frame(o, u, v);
                        let subs = split_face_2d(
                            &topo,
                            face_id,
                            &sections,
                            Rank::A,
                            &brepkit_math::tolerance::Tolerance::new(),
                            None,
                            None,
                            &std::collections::HashMap::new(),
                            &std::collections::BTreeSet::new(),
                            None,
                        );
                        let mut areas: Vec<f64> = subs
                            .iter()
                            .map(|s| {
                                exact_wire_area(&s.outer_wire, o, u, v)
                                    + s.inner_wires
                                        .iter()
                                        .map(|w| exact_wire_area(w, o, u, v))
                                        .sum::<f64>()
                            })
                            .collect();
                        areas.sort_by(f64::total_cmp);
                        let mut want = if case == "opposite" {
                            vec![half, half, 100.0 - 2.0 * half]
                        } else {
                            vec![half, 100.0 - half]
                        };
                        want.sort_by(f64::total_cmp);
                        let what = format!("{pose}, {}, from {start:.3}, {case}", curve.type_tag());
                        assert!(
                            areas.len() == want.len()
                                && areas.iter().zip(&want).all(|(a, w)| (a - w).abs() < 1e-9),
                            "{what}: regions of areas {areas:?}, want {want:?}"
                        );
                        // Each half is bounded by its chord and one arc.
                        for sub in &subs {
                            let arcs = sub
                                .outer_wire
                                .iter()
                                .filter(|e| !matches!(e.curve_3d, EdgeCurve::Line))
                                .count();
                            let lines = sub.outer_wire.len() - arcs;
                            assert!(
                                lines == 4 || (arcs == 1 && lines == 1),
                                "{what}: a region of {arcs} arcs and {lines} lines"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A circle of radius 2 and an ellipse of semi-axes 2 and 1.5 about the
    /// plate's (5, 5): a section over half a turn from each of several
    /// angles, its twin with it, and a line ending on its middle split both
    /// twins there alike, each piece and its twin one id and the two halves
    /// two. A closed section is left whole, its twins one id. In each frame.
    #[test]
    fn a_half_turn_section_and_its_twin_split_alike() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        use std::f64::consts::PI;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let frame = PlaneFrame::from_normal_and_point(n, o);
            let surface = FaceSurface::Plane {
                normal: n,
                d: n.dot(o - Point3::new(0.0, 0.0, 0.0)),
            };
            let both = |curve: &EdgeCurve, from: Point3, to: Point3, src: usize| {
                let forward = OrientedPCurveEdge {
                    start_uv: frame.project(from),
                    end_uv: frame.project(to),
                    ..meeting_edge(curve.clone(), from, to, src)
                };
                let reverse = OrientedPCurveEdge {
                    start_uv: forward.end_uv,
                    end_uv: forward.start_uv,
                    start_3d: to,
                    end_3d: from,
                    forward: false,
                    ..forward.clone()
                };
                [forward, reverse]
            };
            let centre = o + u * 5.0 + v * 5.0;
            let curves = [
                EdgeCurve::Circle(Circle3D::with_axes(centre, n, 2.0, u, v).unwrap()),
                EdgeCurve::Ellipse(Ellipse3D::with_axes(centre, n, 2.0, 1.5, u, v).unwrap()),
            ];
            for curve in &curves {
                let point = |t: f64| curve.evaluate_with_endpoints(t, centre, centre);
                for start in [0.0, 0.3, 0.5 * PI, PI, 1.7] {
                    for closed in [false, true] {
                        let to = if closed { start } else { start + PI };
                        let cuts = if closed {
                            vec![start + 0.4, start + 0.4 + PI]
                        } else {
                            vec![start + 0.5 * PI]
                        };
                        let mut edges = both(curve, point(start), point(to), 0).to_vec();
                        for (k, &t) in cuts.iter().enumerate() {
                            let p = point(t);
                            let out = (p - centre).normalize().unwrap();
                            edges.extend(both(&EdgeCurve::Line, p + out * 3.0, p, 1 + k));
                        }
                        split_sections_at_t_junctions(
                            &mut edges,
                            0,
                            &surface,
                            Some(&frame),
                            &[],
                            1e-7,
                            None,
                            &[],
                        );
                        let pieces: Vec<&OrientedPCurveEdge> = edges
                            .iter()
                            .filter(|e| !matches!(e.curve_3d, EdgeCurve::Line))
                            .collect();
                        let what = format!(
                            "{pose}, {}, from {start:.3}, closed {closed}",
                            curve.type_tag()
                        );
                        assert_eq!(
                            pieces.len(),
                            if closed { 2 } else { 4 },
                            "{what}: {pieces:?}"
                        );
                        let mut ids = std::collections::BTreeSet::new();
                        for piece in pieces.iter().filter(|e| e.forward) {
                            ids.insert(piece.source_edge_idx);
                            assert!(
                                pieces.iter().any(|e| {
                                    !e.forward
                                        && e.source_edge_idx == piece.source_edge_idx
                                        && (e.start_3d - piece.end_3d).length() < 1e-9
                                        && (e.end_3d - piece.start_3d).length() < 1e-9
                                }),
                                "{what}: no twin for {piece:?} in {pieces:?}"
                            );
                        }
                        assert_eq!(ids.len(), if closed { 1 } else { 2 }, "{what}: ids {ids:?}");
                    }
                }
            }
        }
    }

    /// An arc in the plane square to the plate along the chord from (3, 5)
    /// to (7, 5), which is its shadow on the plate, and a line square to the
    /// arc's plane from (6, 7) to the chord at (6, 5), a hair off square to
    /// the plate's x: the two lie in no one plane, and the arc meets the
    /// line nowhere, in each frame.
    #[test]
    fn an_arc_square_to_the_plate_meets_no_line_across_its_plane() {
        use brepkit_math::curves::Circle3D;
        for (pose, o, u, v) in plate_frames() {
            let n = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let circle = Circle3D::with_axes(at(5.0, 5.0) + n * 1.5, v, 2.5, u, -n).unwrap();
            let t0 = 0.6f64.asin();
            let edges = vec![
                meeting_edge(
                    EdgeCurve::Circle(circle.clone()),
                    circle.evaluate(t0),
                    circle.evaluate(std::f64::consts::PI - t0),
                    1,
                ),
                meeting_edge(EdgeCurve::Line, at(6.0 + 1e-14, 7.0), at(6.0, 5.0), 2),
            ];
            let found = meetings(&edges);
            assert!(found.is_empty(), "{pose}: meetings {found:?}");
        }
    }

    /// A circle of radius 2 about the plate's (5, 5) from -60 to 60 degrees,
    /// an ellipse about (4, 5) of semi-axes 3 along the plate's x and
    /// `sqrt(5.4)` through the same three points `P`, `M = (7, 5)` and `Q`,
    /// and the chord from `Q` to `P`, in each frame. The two curves touch at
    /// `M`, the ellipse inside, so the plate splits in four: the ellipse's
    /// segment, two lenses between the curves meeting at `M`, and the rest.
    #[test]
    fn a_circle_and_an_ellipse_through_three_points_split_a_plate_in_four() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        let pi = std::f64::consts::PI;
        let b = 5.4f64.sqrt();
        let s0 = (2.0f64 / 3.0).acos();
        let circle_segment = 2.0 * (2.0 * pi / 3.0 - (2.0 * pi / 3.0).sin());
        let ellipse_segment = 0.5 * 3.0 * b * (2.0 * s0 - (2.0 * s0).sin());
        let lens = 0.5 * (circle_segment - ellipse_segment);
        let mut want = [ellipse_segment, lens, lens, 100.0 - circle_segment];
        want.sort_by(f64::total_cmp);
        for (pose, o, u, v) in plate_frames() {
            let normal = u.cross(v);
            let at = |x: f64, y: f64| o + u * x + v * y;
            let circle = Circle3D::with_axes(at(5.0, 5.0), normal, 2.0, u, v).unwrap();
            let ellipse = Ellipse3D::with_axes(at(4.0, 5.0), normal, 3.0, b, u, v).unwrap();
            let (p, q) = (circle.evaluate(-pi / 3.0), circle.evaluate(pi / 3.0));
            for (x, t) in [(p, -s0), (q, s0)] {
                assert!((ellipse.evaluate(t) - x).length() < 1e-12, "{pose}");
            }
            let sections = [
                SectionEdge {
                    curve_3d: EdgeCurve::Circle(circle),
                    ..line_section(p, q)
                },
                SectionEdge {
                    curve_3d: EdgeCurve::Ellipse(ellipse),
                    ..line_section(p, q)
                },
                line_section(q, p),
            ];
            let areas = framed_plate_region_areas(o, u, v, &sections);
            assert_eq!(
                areas.len(),
                4,
                "{pose}: regions of areas {areas:?}, want {want:?}"
            );
            for (got, want) in areas.iter().zip(want) {
                assert!(
                    (got - want).abs() < 1e-9,
                    "{pose}: regions of areas {areas:?}, want {want}"
                );
            }
        }
    }

    /// A closed B-spline (an exact circle of radius 2 about (5, 5), nine
    /// rational quadratic points) cut at its closure point and at 0.6 of its
    /// span into two trims with one pair of ends: they are two curves, and
    /// the plate splits into the disc and the rest.
    #[test]
    fn trims_of_one_closed_b_spline_between_one_pair_of_points_stay_two() {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let points: Vec<Point3> = [
            (7.0, 5.0),
            (7.0, 7.0),
            (5.0, 7.0),
            (3.0, 7.0),
            (3.0, 5.0),
            (3.0, 3.0),
            (5.0, 3.0),
            (7.0, 3.0),
            (7.0, 5.0),
        ]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect();
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            points,
            vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0],
        )
        .unwrap();
        let (a, b) = (curve.evaluate(0.0), curve.evaluate(0.6));
        let trim = |from: Point3, to: Point3| SectionEdge {
            curve_3d: EdgeCurve::NurbsCurve(curve.clone()),
            ..line_section(from, to)
        };
        let (first, second) = (trim(a, b), trim(b, a));
        assert!(
            (section_middle(&first) - section_middle(&second)).length() > 1.0,
            "the two trims read as one piece"
        );
        assert!(!same_section_curve(&first, &second, 1e-5));
        let pi = std::f64::consts::PI;
        let areas = plate_region_areas(&[first, second]);
        let want = [4.0 * pi, 100.0 - 4.0 * pi];
        assert_eq!(areas.len(), 2, "regions of areas {areas:?}");
        for (got, want) in areas.iter().zip(want) {
            assert!(
                (got - want).abs() < 1e-4,
                "regions of areas {areas:?}, want {want}"
            );
        }
    }

    /// The greatest distance from points of section `a`, read at 2001
    /// points, to section `b`, read at 20001.
    fn farthest_from(a: &SectionEdge, b: &SectionEdge) -> f64 {
        let on_b: Vec<Point3> = (0..=20000)
            .map(|k| evaluate_edge_at_t(&b.curve_3d, b.start, b.end, f64::from(k) / 20000.0))
            .collect();
        (0..=2000)
            .map(|k| {
                let p = evaluate_edge_at_t(&a.curve_3d, a.start, a.end, f64::from(k) / 2000.0);
                on_b.iter()
                    .map(|&q| (q - p).length())
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, crate::builder::pcurve_compute::max_or_nan)
    }

    /// Two arcs of a sweep of pi - 0.1 from one pair of angles: one on the
    /// unit circle, one on a circle 0.53 of the tolerance larger whose centre
    /// moves as far toward the arc's quarter point. Their ends and middles
    /// lie within the tolerance of each other, but at the quarter point
    /// they stand 1.06 of it apart: they are two curves.
    #[test]
    fn arcs_whose_ends_and_middles_meet_but_part_between_are_two_curves() {
        use brepkit_math::curves::Circle3D;
        let tol = 1e-5;
        let (from, sweep) = (0.3_f64, std::f64::consts::PI - 0.1);
        let quarter = sweep.madd(0.25, from);
        let z = Vec3::new(0.0, 0.0, 1.0);
        let unit = Circle3D::new(Point3::new(5.0, 5.0, 0.0), z, 1.0).unwrap();
        let shift = (unit.evaluate(quarter) - unit.center()) * (0.53 * tol);
        let wider = Circle3D::new(
            Point3::new(5.0, 5.0, 0.0) + shift,
            z,
            0.53f64.madd(tol, 1.0),
        )
        .unwrap();
        let arc = |c: &Circle3D| SectionEdge {
            curve_3d: EdgeCurve::Circle(c.clone()),
            ..line_section(c.evaluate(from), c.evaluate(from + sweep))
        };
        let (a, b) = (arc(&unit), arc(&wider));
        assert!((section_middle(&a) - section_middle(&b)).length() < tol);
        assert!((a.start - b.start).length() < tol && (a.end - b.end).length() < tol);
        let apart = farthest_from(&b, &a);
        assert!(apart > tol, "the arcs part by {apart:e}");
        assert!(!same_section_curve(&a, &b, tol));
        assert!(!same_section_curve(&b, &a, tol));
    }

    /// Complementary trims of a closed spline of lines through (0, 0),
    /// (2, 0), (2, 2), (-2, 3) and a point 4e-6 off (2, 1), over [0, 0.6] and
    /// [0.6, 1]: their ends meet and their middles lie 4e-6 apart, but they
    /// are two curves.
    #[test]
    fn complementary_trims_of_a_closed_spline_with_meeting_middles_are_two_curves() {
        let points: Vec<Point3> = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (-2.0, 3.0),
            (2.0 + 4e-6, 1.0),
            (0.0, 0.0),
        ]
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect();
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            points,
            vec![1.0; 6],
        )
        .unwrap();
        let trim = |from: Point3, to: Point3| SectionEdge {
            curve_3d: EdgeCurve::NurbsCurve(curve.clone()),
            ..line_section(from, to)
        };
        let (a, b) = (curve.evaluate(0.0), curve.evaluate(0.6));
        let (first, second) = (trim(a, b), trim(b, a));
        assert!((section_middle(&first) - section_middle(&second)).length() < 1e-5);
        assert!(farthest_from(&first, &second) > 0.5);
        assert!(!same_section_curve(&first, &second, 1e-5));
        assert!(!same_section_curve(&second, &first, 1e-5));
    }

    /// Sections that are one curve still merge: an arc and its copy moved
    /// 1e-7, the same arc stored from its other end on its circle turned
    /// over, an elliptic arc and the same arc on its ellipse with both axes
    /// reversed, and a trim of an open cubic stored either way round.
    #[test]
    fn sections_that_are_one_curve_still_merge() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        let tol = 1e-5;
        let z = Vec3::new(0.0, 0.0, 1.0);
        let centre = Point3::new(5.0, 5.0, 0.0);
        let circle = Circle3D::new(centre, z, 2.0).unwrap();
        let (a, b) = (circle.evaluate(0.4), circle.evaluate(1.9));
        let arc = SectionEdge {
            curve_3d: EdgeCurve::Circle(circle),
            ..line_section(a, b)
        };
        let moved = Circle3D::new(centre + Vec3::new(1e-7, 0.0, 0.0), z, 2.0).unwrap();
        let near = SectionEdge {
            curve_3d: EdgeCurve::Circle(moved.clone()),
            ..line_section(moved.evaluate(0.4), moved.evaluate(1.9))
        };
        let over = SectionEdge {
            curve_3d: EdgeCurve::Circle(Circle3D::new(centre, -z, 2.0).unwrap()),
            ..line_section(b, a)
        };
        assert!(same_section_curve(&arc, &near, tol));
        assert!(same_section_curve(&arc, &over, tol));
        assert!(same_section_curve(&over, &arc, tol));

        let x = Vec3::new(1.0, 0.0, 0.0);
        let ellipse = Ellipse3D::new_with_ref(centre, z, 3.0, 1.5, x).unwrap();
        let flipped = Ellipse3D::new_with_ref(centre, z, 3.0, 1.5, -x).unwrap();
        let (a, b) = (ellipse.evaluate(0.4), ellipse.evaluate(1.9));
        let elliptic = |e: &Ellipse3D| SectionEdge {
            curve_3d: EdgeCurve::Ellipse(e.clone()),
            ..line_section(a, b)
        };
        assert!(same_section_curve(
            &elliptic(&ellipse),
            &elliptic(&flipped),
            tol
        ));

        let cubic = brepkit_math::nurbs::curve::NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(2.0, -1.0, 0.0),
                Point3::new(3.0, 1.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0; 5],
        )
        .unwrap();
        let (a, b) = (cubic.evaluate(0.2), cubic.evaluate(0.7));
        let trim = |from: Point3, to: Point3| SectionEdge {
            curve_3d: EdgeCurve::NurbsCurve(cubic.clone()),
            ..line_section(from, to)
        };
        assert!(same_section_curve(&trim(a, b), &trim(a, b), tol));
        assert!(same_section_curve(&trim(a, b), &trim(b, a), tol));
    }

    /// An arc of a circle of radius 1e4 over [3 pi / 8, 7 pi / 8], and the
    /// same arc with its circle turned by `angle` about the line through
    /// its centre at angle 1/8 + 0.35 of a turn in its plane.
    fn large_arc_and_its_turned_copy(angle: f64) -> (SectionEdge, SectionEdge) {
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::PI;
        let centre = Point3::new(3.0, -2.0, 1.0);
        let normal = Vec3::new(0.2, -0.3, 0.9).normalize().unwrap();
        let circle = Circle3D::new(centre, normal, 1e4).unwrap();
        let toward = 0.35 + PI / 8.0;
        let axis = circle.u_axis() * toward.cos() + circle.v_axis() * toward.sin();
        let (s, c) = angle.sin_cos();
        let turn = |v: Vec3| v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c));
        let turned =
            Circle3D::new_with_ref(centre, turn(normal), 1e4, turn(circle.u_axis())).unwrap();
        let arc = |k: &Circle3D| SectionEdge {
            curve_3d: EdgeCurve::Circle(k.clone()),
            ..line_section(k.evaluate(3.0 * PI / 8.0), k.evaluate(7.0 * PI / 8.0))
        };
        (arc(&circle), arc(&turned))
    }

    /// The most a section's points, read at 2001, stand off the whole
    /// circle of a circular section `of`.
    fn off_circle(s: &SectionEdge, of: &SectionEdge) -> f64 {
        let EdgeCurve::Circle(c) = &of.curve_3d else {
            return f64::NAN;
        };
        (0..=2000)
            .map(|k| {
                let p = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, f64::from(k) / 2000.0);
                let w = p - c.center();
                let h = w.dot(c.normal());
                h.hypot((w - c.normal() * h).length() - c.radius())
            })
            .fold(0.0, crate::builder::pcurve_compute::max_or_nan)
    }

    /// An arc of radius 1e4 and its copy turned by 1.06e-9: their ends and
    /// middles lie within the tolerance of each other, but the copy stands
    /// 1.06e-5 off between, so they are two curves. Turned by nothing, or by
    /// 1e-14, the copy is the same curve.
    #[test]
    fn a_large_arc_and_its_slightly_turned_copy_merge_only_when_within_the_tolerance() {
        let tol = 1e-5;
        let (arc, copy) = large_arc_and_its_turned_copy(1.06e-9);
        assert!((section_middle(&arc) - section_middle(&copy)).length() < tol);
        assert!((arc.start - copy.start).length() < tol && (arc.end - copy.end).length() < tol);
        let off = off_circle(&copy, &arc);
        assert!(off > tol, "the turned copy stands {off:e} off");
        assert!(!same_section_curve(&arc, &copy, tol));
        assert!(!same_section_curve(&copy, &arc, tol));
        for angle in [0.0, 1e-14] {
            let (arc, copy) = large_arc_and_its_turned_copy(angle);
            assert!(same_section_curve(&arc, &copy, tol), "turned by {angle:e}");
            assert!(same_section_curve(&copy, &arc, tol), "turned by {angle:e}");
        }
    }

    /// A section's circle stored on axes `scale` long, its radius divided by
    /// `scale`: the same points at the same parameters.
    fn on_long_axes(s: &SectionEdge, scale: f64) -> SectionEdge {
        let EdgeCurve::Circle(c) = &s.curve_3d else {
            return s.clone();
        };
        let long = brepkit_math::curves::Circle3D::with_axes(
            c.center(),
            c.normal(),
            c.radius() / scale,
            c.u_axis() * scale,
            c.v_axis() * scale,
        )
        .unwrap();
        SectionEdge {
            curve_3d: EdgeCurve::Circle(long),
            ..s.clone()
        }
    }

    /// The pair of [`large_arc_and_its_turned_copy`] with both circles
    /// stored on axes 1e12 long: the copy turned by 1.06e-9 still stays a
    /// curve of its own, and the copy turned by nothing still merges. An
    /// ellipse on axes of other lengths, whose projection misreads its
    /// points, or on axes slightly askew, is refused.
    #[test]
    fn arcs_on_long_axes_merge_as_their_unit_forms_do() {
        use brepkit_math::curves::Ellipse3D;
        let tol = 1e-5;
        for (angle, merges) in [(1.06e-9, false), (0.0, true)] {
            let (arc, copy) = large_arc_and_its_turned_copy(angle);
            assert_eq!(
                same_section_curve(&arc, &copy, tol),
                merges,
                "unit, {angle:e}"
            );
            let (long_arc, long_copy) = (on_long_axes(&arc, 1e12), on_long_axes(&copy, 1e12));
            for (a, b) in [
                (&long_arc, &long_copy),
                (&arc, &long_copy),
                (&long_arc, &copy),
            ] {
                assert_eq!(
                    same_section_curve(a, b, tol),
                    merges,
                    "long axes, {angle:e}"
                );
            }
        }
        let z = Vec3::new(0.0, 0.0, 1.0);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let centre = Point3::new(5.0, 5.0, 0.0);
        let ellipse = Ellipse3D::with_axes(centre, z, 3.0, 1.5, x, y).unwrap();
        let (a, b) = (ellipse.evaluate(0.4), ellipse.evaluate(1.9));
        let section = |e: Ellipse3D| SectionEdge {
            curve_3d: EdgeCurve::Ellipse(e),
            ..line_section(a, b)
        };
        let scaled = Ellipse3D::with_axes(centre, z, 1.5, 0.5, x * 2.0, y * 3.0).unwrap();
        assert!((scaled.evaluate(0.4) - a).length() < 1e-12);
        assert!(!same_section_curve(
            &section(ellipse.clone()),
            &section(scaled),
            tol
        ));
        assert!(same_section_curve(
            &section(ellipse.clone()),
            &section(ellipse.clone()),
            tol
        ));
        let askew =
            Ellipse3D::with_axes(centre, z, 3.0, 1.5, x, (y + x * 1e-6).normalize().unwrap())
                .unwrap();
        assert!(!same_section_curve(&section(ellipse), &section(askew), tol));
    }

    /// The upper half of a circle of radius 2e9, stored on the unit axes
    /// and on the axes (1, 0, 0) and (0, 1 + 1.4e-14, 0): the two stand
    /// 2.8e-5 apart at the top, past the 1e-5 margin, so they are two
    /// curves. A copy on the unit axes is the same curve.
    #[test]
    fn a_huge_circle_on_axes_of_slightly_unequal_length_is_its_own_curve() {
        use brepkit_math::curves::Circle3D;
        let tol = 1e-5;
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let centre = Point3::new(0.0, 0.0, 0.0);
        let unit = Circle3D::with_axes(centre, z, 2e9, x, y).unwrap();
        let longer = Circle3D::with_axes(centre, z, 2e9, x, y * (1.0 + 1.4e-14)).unwrap();
        let half = |c: &Circle3D| SectionEdge {
            curve_3d: EdgeCurve::Circle(c.clone()),
            ..line_section(c.evaluate(0.0), c.evaluate(std::f64::consts::PI))
        };
        let top = (longer.evaluate(std::f64::consts::FRAC_PI_2)
            - unit.evaluate(std::f64::consts::FRAC_PI_2))
        .length();
        assert!(top > tol, "the halves stand {top:e} apart at the top");
        assert!(!same_section_curve(&half(&unit), &half(&longer), tol));
        assert!(!same_section_curve(&half(&longer), &half(&unit), tol));
        assert!(same_section_curve(&half(&unit), &half(&unit.clone()), tol));
    }

    /// A section on the 10 by 10 plate, in the plate's own coordinates.
    #[derive(Clone, Copy, PartialEq)]
    enum PlateCut {
        Line((f64, f64), (f64, f64)),
        /// The shorter arc about a centre, counter-clockwise from one point
        /// to the next.
        Arc((f64, f64), f64, (f64, f64), (f64, f64)),
    }

    /// The plate's frame in a pose: its origin and the images of its x and
    /// y axes. A mirrored pose also lists the plate's corners the other way
    /// round, so its face normal runs against its arcs' axes.
    struct PlatePose {
        origin: Point3,
        x: Vec3,
        y: Vec3,
        mirrored: bool,
    }

    impl PlatePose {
        fn at(&self, (x, y): (f64, f64)) -> Point3 {
            self.origin + self.x * x + self.y * y
        }

        fn normal(&self) -> Vec3 {
            self.x.cross(self.y)
        }
    }

    fn turned(v: Vec3, axis: Vec3, angle: f64) -> Vec3 {
        let a = axis.normalize().unwrap();
        v * angle.cos() + a.cross(v) * angle.sin() + a * (a.dot(v) * (1.0 - angle.cos()))
    }

    /// Upright; turned and moved; and mirrored in its plane, turned and moved.
    fn plate_poses() -> [PlatePose; 3] {
        let (ex, ey) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let a = Vec3::new(1.0, 2.0, 2.0);
        let b = Vec3::new(-2.0, 1.0, 2.0);
        [
            PlatePose {
                origin: Point3::new(0.0, 0.0, 0.0),
                x: ex,
                y: ey,
                mirrored: false,
            },
            PlatePose {
                origin: Point3::new(3.0, -7.0, 11.0),
                x: turned(ex, a, 0.9),
                y: turned(ey, a, 0.9),
                mirrored: false,
            },
            PlatePose {
                origin: Point3::new(-5.0, 2.0, 4.0),
                x: turned(ex, b, 2.1),
                y: -turned(ey, b, 2.1),
                mirrored: true,
            },
        ]
    }

    /// Twice the signed area a wire of lines and circle arcs bounds about
    /// `normal`, in closed form, and NaN for a wire with any other curve. An
    /// arc runs counter-clockwise about its circle's normal from its stored
    /// start to its stored end, as the face builder lays it: from `start_3d`
    /// when forward, from `end_3d` when not.
    fn twice_wire_area(wire: &[OrientedPCurveEdge], normal: Vec3) -> f64 {
        let origin = Point3::new(0.0, 0.0, 0.0);
        wire.iter()
            .map(|e| {
                let (p, q) = (e.start_3d - origin, e.end_3d - origin);
                match &e.curve_3d {
                    EdgeCurve::Line => p.cross(q).dot(normal),
                    EdgeCurve::Circle(c) => {
                        let (a, b) = if e.forward {
                            (e.start_3d, e.end_3d)
                        } else {
                            (e.end_3d, e.start_3d)
                        };
                        let sweep = if (a - b).length() < 1e-12 {
                            std::f64::consts::TAU
                        } else {
                            (c.project(b) - c.project(a)).rem_euclid(std::f64::consts::TAU)
                        };
                        let turn = if e.forward { sweep } else { -sweep };
                        let r = c.radius();
                        (c.center() - origin).cross(q - p).dot(normal)
                            + r * r * turn * c.normal().dot(normal)
                    }
                    EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => f64::NAN,
                }
            })
            .sum()
    }

    /// The 10 by 10 plate in `pose`, with square holes (each its lower left
    /// corner and side), and `cuts` as its sections.
    fn plate_with_cuts(
        pose: &PlatePose,
        cuts: &[PlateCut],
        holes: &[((f64, f64), f64)],
    ) -> (Topology, FaceId, Vec<SectionEdge>) {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::builder::{make_planar_face, make_polygon_wire};

        let mut topo = Topology::new();
        let mut corners: Vec<Point3> = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
            .iter()
            .map(|&p| pose.at(p))
            .collect();
        if pose.mirrored {
            corners.reverse();
        }
        let face = make_planar_face(&mut topo, &corners, 1e-7).unwrap();
        for &((x, y), side) in holes {
            let mut square: Vec<Point3> =
                [(x, y), (x, y + side), (x + side, y + side), (x + side, y)]
                    .iter()
                    .map(|&p| pose.at(p))
                    .collect();
            if pose.mirrored {
                square.reverse();
            }
            let hole = make_polygon_wire(&mut topo, &square, 1e-7).unwrap();
            topo.face_mut(face).unwrap().inner_wires_mut().push(hole);
        }
        let sections = cuts
            .iter()
            .map(|cut| {
                let (curve_3d, start, end) = match *cut {
                    PlateCut::Line(a, b) => (EdgeCurve::Line, pose.at(a), pose.at(b)),
                    PlateCut::Arc(c, r, a, b) => {
                        let circle =
                            Circle3D::with_axes(pose.at(c), pose.normal(), r, pose.x, pose.y)
                                .unwrap();
                        (EdgeCurve::Circle(circle), pose.at(a), pose.at(b))
                    }
                };
                SectionEdge {
                    curve_3d,
                    start,
                    end,
                    ..line_section(start, end)
                }
            })
            .collect();
        (topo, face, sections)
    }

    /// Each piece's area, its outer wire's less its holes', sorted. Each
    /// wire must close edge to edge.
    fn piece_areas(pieces: &[SplitSubFace], normal: Vec3) -> Vec<f64> {
        let mut areas: Vec<f64> = pieces
            .iter()
            .map(|piece| {
                for wire in std::iter::once(&piece.outer_wire).chain(&piece.inner_wires) {
                    for (k, e) in wire.iter().enumerate() {
                        let next = &wire[(k + 1) % wire.len()];
                        assert!(
                            (e.end_3d - next.start_3d).length() < 1e-9,
                            "a wire breaks at {:?}",
                            e.end_3d
                        );
                    }
                }
                let holes: f64 = piece
                    .inner_wires
                    .iter()
                    .map(|w| twice_wire_area(w, normal).abs())
                    .sum();
                0.5 * (twice_wire_area(&piece.outer_wire, normal).abs() - holes)
            })
            .collect();
        areas.sort_by(f64::total_cmp);
        areas
    }

    fn assert_areas(areas: &[f64], expected: &[f64], label: &str) {
        let mut expected = expected.to_vec();
        expected.sort_by(f64::total_cmp);
        assert!(
            areas.len() == expected.len()
                && areas
                    .iter()
                    .zip(&expected)
                    .all(|(a, e)| (a - e).abs() < 1e-9),
            "{label}: pieces of area {areas:?}, expected {expected:?}"
        );
    }

    /// The arc about (5, 4) of radius root 2 from (6, 5) to (4, 5), the arc
    /// about (5, 6) bulging the other way, and the line between their ends.
    fn arcs_and_chord() -> (PlateCut, PlateCut, PlateCut) {
        use std::f64::consts::SQRT_2;
        (
            PlateCut::Arc((5.0, 4.0), SQRT_2, (6.0, 5.0), (4.0, 5.0)),
            PlateCut::Arc((5.0, 6.0), SQRT_2, (4.0, 5.0), (6.0, 5.0)),
            PlateCut::Line((4.0, 5.0), (6.0, 5.0)),
        )
    }

    /// The plate's arrangement reads an arc and the line between its ends as
    /// two edges, in either order, so the segment between them is a region:
    /// on a line across the plate with the arc on either side, with an arc on
    /// each side, and with the rest of the arc's circle in two arcs; and with
    /// two lines from the chord's ends to the rim.
    #[test]
    fn the_arrangement_keeps_an_arc_apart_from_its_chord() {
        use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};
        let segment = FRAC_PI_2 - 1.0;
        let disc_below = 2.0 * PI - segment;
        let (up, down, chord) = arcs_and_chord();
        let left = PlateCut::Line((0.0, 5.0), (4.0, 5.0));
        let right = PlateCut::Line((6.0, 5.0), (10.0, 5.0));
        let to_apex = PlateCut::Line((6.0, 5.0), (5.0, 10.0));
        let from_apex = PlateCut::Line((5.0, 10.0), (4.0, 5.0));
        let bottom = (5.0, 4.0 - SQRT_2);
        let rest_a = PlateCut::Arc((5.0, 4.0), SQRT_2, (4.0, 5.0), bottom);
        let rest_b = PlateCut::Arc((5.0, 4.0), SQRT_2, bottom, (6.0, 5.0));
        let cases = [
            (
                vec![up, chord, right, left],
                vec![segment, 50.0 - segment, 50.0],
            ),
            (
                vec![down, chord, right, left],
                vec![segment, 50.0 - segment, 50.0],
            ),
            (
                vec![up, chord, to_apex, from_apex],
                vec![segment, 5.0 - segment, 95.0],
            ),
            (
                vec![up, down, chord, right, left],
                vec![segment, segment, 50.0 - segment, 50.0 - segment],
            ),
            (
                vec![up, rest_a, rest_b, chord, right, left],
                vec![segment, disc_below, 50.0 - segment, 50.0 - disc_below],
            ),
        ];
        for (k, (cuts, expected)) in cases.iter().enumerate() {
            for (i, pose) in plate_poses().iter().enumerate() {
                for chord_first in [false, true] {
                    let mut cuts = cuts.clone();
                    if chord_first {
                        let at = cuts.iter().position(|c| *c == chord).unwrap();
                        let line = cuts.remove(at);
                        cuts.insert(0, line);
                    }
                    let (topo, face_id, sections) = plate_with_cuts(pose, &cuts, &[]);
                    let face = topo.face(face_id).unwrap();
                    let surface = face.surface().clone();
                    let wire_pts = collect_wire_points(&topo, face.outer_wire());
                    let frame =
                        PlaneFrame::from_plane_face(extract_plane_normal(&surface), &wire_pts);
                    let boundary = boundary_edges_to_pcurve(
                        &topo,
                        face.outer_wire(),
                        &surface,
                        &wire_pts,
                        Some(&frame),
                    );
                    let pieces = split_plane_face_by_arrangement(
                        &surface,
                        &boundary,
                        &sections,
                        Rank::A,
                        false,
                        face_id,
                        &frame,
                        1e-7,
                        None,
                    )
                    .unwrap();
                    assert_areas(
                        &piece_areas(&pieces, pose.normal()),
                        expected,
                        &format!("case {k}, pose {i}, chord first {chord_first}"),
                    );
                }
            }
        }
    }

    /// An arc and its chord inside the plate, clear of its rim, split it with
    /// the segment between them as a piece of its own: with two lines from the
    /// chord's ends to (5, 8), with the arc on either side of the chord, with
    /// an arc on each side, and with the rest of the arc's circle in two arcs;
    /// in either order, and on a plate with a hole.
    #[test]
    fn an_arc_and_its_chord_inside_the_plate_split_off_their_segment() {
        use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};
        let segment = FRAC_PI_2 - 1.0;
        let disc = 2.0 * PI;
        let (up, down, chord) = arcs_and_chord();
        let l1 = PlateCut::Line((6.0, 5.0), (5.0, 8.0));
        let l2 = PlateCut::Line((5.0, 8.0), (4.0, 5.0));
        let bottom = (5.0, 4.0 - SQRT_2);
        let rest_a = PlateCut::Arc((5.0, 4.0), SQRT_2, (4.0, 5.0), bottom);
        let rest_b = PlateCut::Arc((5.0, 4.0), SQRT_2, bottom, (6.0, 5.0));
        // Each case's pieces; the last holds the plate's rim, and with a unit
        // square hole at (1, 1) it loses 1.
        let cases = [
            (vec![up, chord, l1, l2], vec![segment, 3.0 - segment, 97.0]),
            (
                vec![down, chord, l1, l2],
                vec![segment, 3.0, 97.0 - segment],
            ),
            (vec![up, chord], vec![segment, 100.0 - segment]),
            (
                vec![up, down, chord],
                vec![segment, segment, 100.0 - 2.0 * segment],
            ),
            (
                vec![up, rest_a, rest_b, chord],
                vec![segment, disc - segment, 100.0 - disc],
            ),
        ];
        for (k, (cuts, whole)) in cases.iter().enumerate() {
            let mut holed = whole.clone();
            if let Some(rim) = holed.last_mut() {
                *rim -= 1.0;
            }
            for (holes, expected) in [(vec![], whole), (vec![((1.0, 1.0), 1.0)], &holed)] {
                for (i, pose) in plate_poses().iter().enumerate() {
                    for chord_first in [false, true] {
                        let mut cuts = cuts.clone();
                        if chord_first {
                            let at = cuts.iter().position(|c| *c == chord).unwrap();
                            let line = cuts.remove(at);
                            cuts.insert(0, line);
                        }
                        let (topo, face, sections) = plate_with_cuts(pose, &cuts, &holes);
                        let pieces = split_face_2d(
                            &topo,
                            face,
                            &sections,
                            Rank::A,
                            &brepkit_math::tolerance::Tolerance::new(),
                            None,
                            None,
                            &std::collections::HashMap::new(),
                            &std::collections::BTreeSet::new(),
                            None,
                        );
                        let label = format!(
                            "case {k}, holes {}, pose {i}, chord first {chord_first}",
                            holes.len()
                        );
                        assert_areas(&piece_areas(&pieces, pose.normal()), expected, &label);
                    }
                }
            }
        }
    }

    /// An arc and its chord whose ends meet the rest of the cut, which runs to
    /// the plate's rim, split the plate with the segment between them as a
    /// piece of its own: a line across the plate through the chord, with the
    /// arc on either side, and two lines from the chord's ends to the rim.
    #[test]
    fn an_arc_and_its_chord_meeting_the_rim_split_off_their_segment() {
        use std::f64::consts::FRAC_PI_2;
        let segment = FRAC_PI_2 - 1.0;
        let (up, down, chord) = arcs_and_chord();
        let left = PlateCut::Line((0.0, 5.0), (4.0, 5.0));
        let right = PlateCut::Line((6.0, 5.0), (10.0, 5.0));
        let to_apex = PlateCut::Line((6.0, 5.0), (5.0, 10.0));
        let from_apex = PlateCut::Line((5.0, 10.0), (4.0, 5.0));
        // Each case's pieces, without and with a unit square hole at (1, 1).
        let cases = [
            (
                vec![up, chord, right, left],
                [segment, 50.0 - segment, 50.0],
                [segment, 50.0 - segment, 49.0],
            ),
            (
                vec![down, chord, right, left],
                [segment, 50.0 - segment, 50.0],
                [segment, 49.0 - segment, 50.0],
            ),
            (
                vec![up, chord, to_apex, from_apex],
                [segment, 5.0 - segment, 95.0],
                [segment, 5.0 - segment, 94.0],
            ),
        ];
        for (k, (cuts, whole, holed)) in cases.iter().enumerate() {
            for (holes, expected) in [(vec![], whole), (vec![((1.0, 1.0), 1.0)], holed)] {
                for (i, pose) in plate_poses().iter().enumerate() {
                    for chord_first in [false, true] {
                        let mut cuts = cuts.clone();
                        if chord_first {
                            let at = cuts.iter().position(|c| *c == chord).unwrap();
                            let line = cuts.remove(at);
                            cuts.insert(0, line);
                        }
                        let (topo, face, sections) = plate_with_cuts(pose, &cuts, &holes);
                        let pieces = split_face_2d(
                            &topo,
                            face,
                            &sections,
                            Rank::A,
                            &brepkit_math::tolerance::Tolerance::new(),
                            None,
                            None,
                            &std::collections::HashMap::new(),
                            &std::collections::BTreeSet::new(),
                            None,
                        );
                        let label = format!(
                            "case {k}, holes {}, pose {i}, chord first {chord_first}",
                            holes.len()
                        );
                        assert_areas(&piece_areas(&pieces, pose.normal()), expected, &label);
                    }
                }
            }
        }
    }

    /// The whole circle about `center` of radius `r` in the plate's plane,
    /// closed at its lowest point, as a section in `pose`.
    fn plate_circle(pose: &PlatePose, center: (f64, f64), r: f64) -> SectionEdge {
        let circle = brepkit_math::curves::Circle3D::with_axes(
            pose.at(center),
            pose.normal(),
            r,
            pose.x,
            pose.y,
        )
        .unwrap();
        let at = pose.at((center.0, center.1 - r));
        SectionEdge {
            curve_3d: EdgeCurve::Circle(circle),
            start: at,
            end: at,
            ..line_section(at, at)
        }
    }

    /// Two discs' common area, radii `a` and `b`, centres `d` apart.
    fn lens_area(a: f64, b: f64, d: f64) -> f64 {
        let p = a * a * ((d * d + a * a - b * b) / (2.0 * d * a)).acos();
        let q = b * b * ((d * d + b * b - a * a) / (2.0 * d * b)).acos();
        let r = 0.5 * ((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b)).sqrt();
        p + q - r
    }

    /// The plate split by `cuts` and whole circles `(centre, radius)` in
    /// `pose`.
    fn plate_split_with_circles(
        pose: &PlatePose,
        cuts: &[PlateCut],
        circles: &[((f64, f64), f64)],
        more: &[SectionEdge],
    ) -> Vec<SplitSubFace> {
        let (topo, face, mut sections) = plate_with_cuts(pose, cuts, &[]);
        for &(c, r) in circles {
            sections.push(plate_circle(pose, c, r));
        }
        sections.extend(more.iter().cloned());
        split_face_2d(
            &topo,
            face,
            &sections,
            Rank::A,
            &brepkit_math::tolerance::Tolerance::new(),
            None,
            None,
            &std::collections::HashMap::new(),
            &std::collections::BTreeSet::new(),
            None,
        )
    }

    /// A whole circle about (5, 4) of radius root 2 on the plate, with a
    /// chord from (4, 5) to (6, 5), with a line across the plate through the
    /// same points, or with a circle of radius 1 about (5, 5.5) that crosses
    /// it: it is cut where they meet, and each region comes out once. A line
    /// from its centre out across it, and a line clear of it, leave its disc
    /// whole.
    #[test]
    fn a_whole_circle_that_other_sections_meet_is_cut_where_they_meet() {
        use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};
        let segment = FRAC_PI_2 - 1.0;
        let disc = 2.0 * PI;
        let lens = lens_area(SQRT_2, 1.0, 1.5);
        let cases = [
            (
                "chord",
                vec![PlateCut::Line((4.0, 5.0), (6.0, 5.0))],
                vec![((5.0, 4.0), SQRT_2)],
                vec![segment, disc - segment, 100.0 - disc],
            ),
            (
                "across",
                vec![PlateCut::Line((0.0, 5.0), (10.0, 5.0))],
                vec![((5.0, 4.0), SQRT_2)],
                vec![
                    segment,
                    disc - segment,
                    50.0 - segment,
                    50.0 - disc + segment,
                ],
            ),
            (
                "two circles",
                vec![],
                vec![((5.0, 4.0), SQRT_2), ((5.0, 5.5), 1.0)],
                vec![lens, disc - lens, PI - lens, 100.0 - disc - PI + lens],
            ),
            (
                "into",
                vec![PlateCut::Line((5.0, 4.0), (8.0, 4.0))],
                vec![((5.0, 4.0), SQRT_2)],
                vec![disc, 100.0 - disc],
            ),
            (
                "clear",
                vec![PlateCut::Line((0.0, 8.0), (10.0, 8.0))],
                vec![((5.0, 4.0), SQRT_2)],
                vec![disc, 80.0 - disc, 20.0],
            ),
        ];
        for (name, cuts, circles, expected) in &cases {
            for (i, pose) in plate_poses().iter().enumerate() {
                let pieces = plate_split_with_circles(pose, cuts, circles, &[]);
                let label = format!("{name}, pose {i}");
                assert_areas(&piece_areas(&pieces, pose.normal()), expected, &label);
            }
        }
    }

    /// Arcs of the circle about (5, 4) whose ends lie on a line across the
    /// plate, the line whole or in pieces, and one arc running more than a
    /// half turn: the segment between the short arc and the line is a piece
    /// of its own.
    #[test]
    fn arcs_ending_on_a_line_across_the_plate_split_off_their_segment() {
        use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};
        let segment = FRAC_PI_2 - 1.0;
        let disc = 2.0 * PI;
        let expected = [
            segment,
            disc - segment,
            50.0 - segment,
            50.0 - disc + segment,
        ];
        let arc = |from: (f64, f64), to: (f64, f64)| PlateCut::Arc((5.0, 4.0), SQRT_2, from, to);
        let (left, right, bottom) = ((4.0, 5.0), (6.0, 5.0), (5.0, 4.0 - SQRT_2));
        let east = (5.0 + SQRT_2, 4.0);
        let across = PlateCut::Line((0.0, 5.0), (10.0, 5.0));
        let pieces = [
            PlateCut::Line((0.0, 5.0), left),
            PlateCut::Line(left, right),
            PlateCut::Line(right, (10.0, 5.0)),
        ];
        let cases = [
            vec![
                arc(right, left),
                arc(left, bottom),
                arc(bottom, right),
                across,
            ],
            vec![arc(right, left), arc(left, east), arc(east, right), across],
            [
                vec![arc(right, left), arc(left, east), arc(east, right)],
                pieces.to_vec(),
            ]
            .concat(),
        ];
        for (k, cuts) in cases.iter().enumerate() {
            for (i, pose) in plate_poses().iter().enumerate() {
                let pieces = plate_split_with_circles(pose, cuts, &[], &[]);
                let label = format!("case {k}, pose {i}");
                assert_areas(&piece_areas(&pieces, pose.normal()), &expected, &label);
            }
        }
    }

    /// A whole circle about (5, 4) of radius root 2 and a quadratic spline
    /// from (2, 4) to (8, 4) about a control point above it. Over (5, 8) the
    /// spline keeps more than 0.5 off the circle, which its control points prove, and
    /// the circle stays a cap; over (5, 6) it crosses the circle at points no
    /// closed form finds, and the split is refused.
    #[test]
    fn a_whole_circle_and_a_spline_that_may_cross_it_refuse_the_split() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use std::f64::consts::{PI, SQRT_2};
        for (i, pose) in plate_poses().iter().enumerate() {
            for (top, expected) in [(8.0, Some([2.0 * PI, 100.0 - 2.0 * PI])), (6.0, None)] {
                let points = vec![
                    pose.at((2.0, 4.0)),
                    pose.at((5.0, top)),
                    pose.at((8.0, 4.0)),
                ];
                let spline = NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    points.clone(),
                    vec![1.0; 3],
                )
                .unwrap();
                let section = SectionEdge {
                    curve_3d: EdgeCurve::NurbsCurve(spline),
                    ..line_section(points[0], points[2])
                };
                let pieces =
                    plate_split_with_circles(pose, &[], &[((5.0, 4.0), SQRT_2)], &[section]);
                let label = format!("control at height {top}, pose {i}");
                match expected {
                    Some(areas) => {
                        assert_areas(&piece_areas(&pieces, pose.normal()), &areas, &label);
                    }
                    None => assert!(pieces.is_empty(), "{label}: split into {}", pieces.len()),
                }
            }
        }
    }

    /// A whole circle about (5, 4) of radius root 2 and a quadratic spline in
    /// two knot spans. Below the circle, over (3, 1) and (7, 1.5), the spline
    /// keeps well off it in both spans and the circle stays a cap; through
    /// (3, 1) and (5, 4), its second span runs into the circle and the split
    /// is refused.
    #[test]
    fn a_whole_circle_and_a_spline_of_two_spans_are_read_span_by_span() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use std::f64::consts::{PI, SQRT_2};
        let below = [(1.0, 1.0), (3.0, 1.0), (7.0, 1.5), (9.0, 1.0)];
        let through = [(1.0, 1.0), (3.0, 1.0), (5.0, 4.0), (9.0, 4.0)];
        for (i, pose) in plate_poses().iter().enumerate() {
            for (poles, expected) in [(below, Some([2.0 * PI, 100.0 - 2.0 * PI])), (through, None)]
            {
                let points: Vec<Point3> = poles.iter().map(|&p| pose.at(p)).collect();
                let spline = NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
                    points.clone(),
                    vec![1.0; 4],
                )
                .unwrap();
                let section = SectionEdge {
                    curve_3d: EdgeCurve::NurbsCurve(spline),
                    ..line_section(points[0], points[3])
                };
                let pieces =
                    plate_split_with_circles(pose, &[], &[((5.0, 4.0), SQRT_2)], &[section]);
                let label = format!("poles {poles:?}, pose {i}");
                match expected {
                    Some(areas) => {
                        assert_areas(&piece_areas(&pieces, pose.normal()), &areas, &label);
                    }
                    None => assert!(pieces.is_empty(), "{label}: split into {}", pieces.len()),
                }
            }
        }
    }
}
