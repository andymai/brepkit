//! Face splitting via 2D wire construction.
//!
//! For each face, collects boundary edges and section edges, converts
//! them to [`OrientedPCurveEdge`]s in the face's parameter space, calls
//! the wire builder, and produces [`SplitSubFace`]s.

mod containment;
mod conversion;
mod edge_splitting;
mod sampling;
mod special_cases;
pub(in crate::builder) use special_cases::cylinder_cone_remainder_interior;

pub use conversion::collect_wire_points;

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::classify_2d::{sample_interior_point, signed_area_2d};
use super::pcurve_compute::{
    compute_pcurve_on_surface, evaluate_edge_at_t, project_point_on_surface,
};
use super::plane_frame::PlaneFrame;
use super::split_types::{OrientedPCurveEdge, SectionEdge, SplitSubFace, SurfaceInfo};
use super::wire_builder::{build_wire_loops, build_wire_loops_dcel, build_wire_loops_with_winding};

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
    split_periodic_face_into_sectors, split_torus_band_by_arrangement,
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
    let denom = rx.mul_add(sy, -(ry * sx));
    // `denom = |r x s| = |r||s| sin(theta)`; test it relative to the segment
    // lengths so near-parallel rejection is independent of model scale.
    let scale = (rx.hypot(ry) * sx.hypot(sy)).max(f64::MIN_POSITIVE);
    if denom.abs() <= 1e-9 * scale {
        return None;
    }
    let (qx, qy) = (b0.x() - a0.x(), b0.y() - a0.y());
    let t = qx.mul_add(sy, -(qy * sx)) / denom;
    let u = qx.mul_add(ry, -(qy * rx)) / denom;
    // `t`/`u` are normalized [0,1] parameters, so these epsilons are already
    // scale-invariant fractions of each segment.
    (t > 1e-6 && t < 1.0 - 1e-6 && u > -1e-6 && u < 1.0 + 1e-6).then_some(t)
}

/// Split section edges at interior T-junctions with other sections.
///
/// `all_edges[section_start..]` holds the section edges as consecutive
/// forward/reverse pairs (both carrying the same `source_edge_idx`). When one
/// section's 3D endpoint lands strictly inside another section's span, the
/// crossed section is split there so both meet at a shared vertex — without
/// this the dangling end is pruned as a pendant and the face never splits at
/// that junction. Boundary edges (`..section_start`) are left untouched.
///
/// This covers the analytic (cylinder/cone) faces that
/// `try_split_crossing_plane_face` (plane-only) does not reach — e.g. a
/// rounded notch corner whose perpendicular-cut arc meets the axis-parallel
/// wall-top line on a corner cylinder. Each split piece gets a fresh unique
/// `source_edge_idx` (forward/reverse of a piece share it) so
/// `build_topology_face` still shares one topology edge per piece.
fn split_sections_at_t_junctions(
    all_edges: &mut Vec<OrientedPCurveEdge>,
    section_start: usize,
    surface: &FaceSurface,
    frame: Option<&PlaneFrame>,
    wire_pts: &[Point3],
    tol: f64,
    mut split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
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
    for e in &all_edges[section_start..] {
        len_sum += (e.end_3d - e.start_3d).length();
        len_cnt += 1.0;
        for p in [e.start_3d, e.end_3d] {
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
    let mut piece_src: std::collections::HashMap<(i64, i64, i64, i64, i64, i64), usize> =
        std::collections::HashMap::new();
    let key = |a: Point3, b: Point3| -> (i64, i64, i64, i64, i64, i64) {
        let q = |x: f64| (x / tol).round() as i64;
        let ka = (q(a.x()), q(a.y()), q(a.z()));
        let kb = (q(b.x()), q(b.y()), q(b.z()));
        // Order-independent so forward and reverse halves map to one id.
        if ka <= kb {
            (ka.0, ka.1, ka.2, kb.0, kb.1, kb.2)
        } else {
            (kb.0, kb.1, kb.2, ka.0, ka.1, ka.2)
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
            EdgeCurve::Circle(_) => find_splits_on_section_arc(&edge, &endpoints, tol),
            EdgeCurve::Ellipse(_) => find_splits_on_section_ellipse(&edge, &endpoints, tol),
            // A marched-NURBS section (a plane×cone conic) bulges past its
            // chord like an arc — a junction endpoint mid-curve is invisible
            // to the chord-based search, so use sampled point-to-curve
            // projection over the full endpoint set (conics are rare).
            EdgeCurve::NurbsCurve(_) => find_splits_on_nurbs_section(&edge, &endpoints, tol),
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
            let src = *piece_src.entry(key(s3, e3)).or_insert_with(|| {
                let v = next_src;
                next_src += 1;
                v
            });
            let pcurve =
                compute_pcurve_on_surface(&edge.curve_3d, s3, e3, surface, wire_pts, frame);
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
        if let Some(reg) = split_registry.as_deref_mut()
            && let Some(pb_id) = edge.pave_block_id
        {
            for &(t, _) in &splits {
                let s3 = evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
                reg.entry(pb_id).or_default().push(s3);
            }
        }
        for &(t, _) in &splits {
            let s3 = evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
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
    // Shorter-arc parameter t in (0,1) of `p` on the arc edge from `start` to
    // `end`, or None if `p` is not on the arc interior.
    let arc_param = |curve: &EdgeCurve, start: Point3, end: Point3, p: Point3| -> Option<f64> {
        let (circle_proj, on_curve): (f64, Point3) = match curve {
            EdgeCurve::Circle(c) => (c.project(p), c.evaluate(c.project(p))),
            EdgeCurve::Ellipse(e) => (e.project(p), e.evaluate(e.project(p))),
            // Only arc edges have a circle/ellipse parameter; a line or NURBS
            // edge is never split by this arc-only path.
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return None,
        };
        if (p - on_curve).length() > tol {
            return None;
        }
        let (a0, a_end) = match curve {
            EdgeCurve::Circle(c) => (c.project(start), c.project(end)),
            EdgeCurve::Ellipse(e) => (e.project(start), e.project(end)),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return None,
        };
        let span = super::pcurve_compute::shorter_arc_delta(a_end - a0);
        if span.abs() < 1e-12 {
            return None;
        }
        let d = super::pcurve_compute::shorter_arc_delta(circle_proj - a0);
        let t = d / span;
        (t > tol && t < 1.0 - tol).then_some(t)
    };

    let mut result = Vec::with_capacity(edges.len());
    for edge in edges {
        let mut splits: Vec<f64> = match &edge.curve_3d {
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => split_pts_3d
                .iter()
                .filter_map(|&p| arc_param(&edge.curve_3d, edge.start_3d, edge.end_3d, p))
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
                            ((p - closest).length() < tol && t > tol && t < 1.0 - tol).then_some(t)
                        })
                        .collect()
                }
            }
            EdgeCurve::NurbsCurve(_) => Vec::new(),
        };
        splits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        splits.dedup_by(|a, b| (*a - *b).abs() < tol);
        if splits.is_empty() {
            result.push(edge);
            continue;
        }

        let mut prev_3d = edge.start_3d;
        let mut prev_uv = edge.start_uv;
        let mut push_piece = |s3: Point3, e3: Point3, s_uv: Point2, e_uv: Point2| {
            let pcurve =
                compute_pcurve_on_surface(&edge.curve_3d, s3, e3, surface, &[], Some(frame));
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
        for &t in &splits {
            let s3 = evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
            let s_uv = frame.project(s3);
            push_piece(prev_3d, s3, prev_uv, s_uv);
            prev_3d = s3;
            prev_uv = s_uv;
        }
        push_piece(prev_3d, edge.end_3d, prev_uv, edge.end_uv);
    }
    result
}

/// Whether an edge curve is geometrically a straight segment, independent of
/// its nominal type. A planar-NURBS extrusion wall's rim is a `NurbsCurve`
/// that is exactly straight (the tilted-divider cavity). Straightness is
/// decided from the control polygon — a NURBS whose control points are
/// collinear is straight everywhere, for any trim of the edge — so a trimmed
/// span of a genuinely curved carrier is never misjudged by sampling only part
/// of it. The collinearity band is the kernel's default linear tolerance
/// (1e-7); a control polygon that tight is straight for any splitting purpose.
fn edge_curve_is_straight(curve: &EdgeCurve) -> bool {
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
                let dev_sq = along.mul_add(-along, v.dot(v));
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
        for (b0, b1, _, _) in &hole_segs {
            if let Some(t) = seg_cross_param(s0, s1, *b0, *b1) {
                ts.push(t);
                any_crossing = true;
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        for w in ts.windows(2) {
            let (ta, tb) = (w[0], w[1]);
            let tm = 0.5 * (ta + tb);
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
        // Recompute the arc's pcurve in THIS face's frame (a plane face: project
        // the 3D arc into `frame`). The stored `pcurve_a` may have been fitted in
        // a different plane frame or on the opposing face, which would disconnect
        // the arc in this UV space — mirror the main section path's plane refit.
        let arc_pcurve = super::pcurve_compute::compute_pcurve_on_surface(
            &s.curve_3d,
            s.start,
            s.end,
            surface,
            wire_pts,
            Some(frame),
        );
        let pcurve_uv = |p: Point3| frame.project(p);
        let src = next_src;
        next_src += 1;
        let su = pcurve_uv(s.start);
        let eu = pcurve_uv(s.end);
        out.push(OrientedPCurveEdge {
            curve_3d: s.curve_3d.clone(),
            pcurve: arc_pcurve.clone(),
            start_uv: su,
            end_uv: eu,
            start_3d: s.start,
            end_3d: s.end,
            forward: true,
            source_edge_idx: Some(src),
            pave_block_id: s.pave_block_id,
        });
        out.push(OrientedPCurveEdge {
            curve_3d: s.curve_3d.clone(),
            pcurve: arc_pcurve,
            start_uv: eu,
            end_uv: su,
            start_3d: s.end,
            end_3d: s.start,
            forward: false,
            source_edge_idx: Some(src),
            pave_block_id: s.pave_block_id,
        });
    }

    any_crossing.then_some((out, passthrough))
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
    let sub_outer_uv: Vec<Vec<Point2>> = sub_faces
        .iter()
        .map(|sf| sample_wire_loop_uv(&sf.outer_wire))
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
        let hole_pts = sample_wire_loop_uv(hole);
        if hole_pts.len() >= 3 {
            let probe = super::classify_2d::sample_interior_point(&hole_pts);
            let containing: Vec<usize> = (0..sub_faces.len())
                .filter(|&i| super::classify_2d::point_in_polygon_2d(probe, &sub_outer_uv[i]))
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

/// True when any traced loop's sampled UV polygon is area-degenerate — the
/// classifier's sliver guard would silently drop it, so the loops path
/// under-represents the face even though the loop COUNT looks fine. The
/// canonical case: a completed 4-way socket-junction circle traced as a
/// 2-arc closed loop whose pcurve-sampled polygon folds to ~zero area while
/// the true disc is πr² (the blind-recess cap the result must keep).
fn wire_loops_have_degenerate_area(loops: &[Vec<OrientedPCurveEdge>], tol: f64) -> bool {
    loops.iter().any(|wl| {
        let pts = sample_wire_loop_uv(wl);
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
    for e in boundary_edges {
        if !matches!(e.curve_3d, EdgeCurve::Line) || (e.start_3d - e.end_3d).length() <= tol {
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

/// Split every section piece that crosses the seam meridian at the crossing
/// point (bisected on the piece's own curve). Pieces are assumed u-monotone
/// between their endpoints (chain pieces subtend well under a half-turn);
/// pieces already ending on the seam, or not straddling it, pass through
/// unchanged. Split halves clear their UV hints and pave block id, matching
/// `presplit_sections_at_registry`.
fn split_sections_at_seam_meridian(
    sections: &[SectionEdge],
    surface: &FaceSurface,
    seam_u: f64,
    tol: f64,
) -> Vec<SectionEdge> {
    use std::f64::consts::{PI, TAU};
    let wrap = |d: f64| -> f64 { (d + PI).rem_euclid(TAU) - PI };
    let delta_u =
        |p: Point3| -> Option<f64> { surface.project_point(p).map(|(u, _)| wrap(u - seam_u)) };
    let mut out = Vec::with_capacity(sections.len() + 2);
    for s in sections {
        let (Some(d0), Some(d1)) = (delta_u(s.start), delta_u(s.end)) else {
            out.push(s.clone());
            continue;
        };
        let straddles = d0 * d1 < 0.0 && (d0 - d1).abs() < PI && d0.abs() > 1e-9 && d1.abs() > 1e-9;
        if !straddles {
            out.push(s.clone());
            continue;
        }
        let (t0, t1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
        let (mut lo, mut hi, f_lo) = (t0, t1, d0);
        for _ in 0..60 {
            let tm = 0.5 * (lo + hi);
            let Some(fm) = delta_u(s.curve_3d.evaluate_with_endpoints(tm, s.start, s.end)) else {
                break;
            };
            if (fm > 0.0) == (f_lo > 0.0) {
                lo = tm;
            } else {
                hi = tm;
            }
        }
        let p = s
            .curve_3d
            .evaluate_with_endpoints(0.5 * (lo + hi), s.start, s.end);
        if (p - s.start).length() <= tol * 10.0 || (p - s.end).length() <= tol * 10.0 {
            out.push(s.clone());
            continue;
        }
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
        out.push(first);
        out.push(second);
    }
    out
}

/// Split a u-periodic cylinder/cone lateral into TWO bands along a
/// seam-anchored section chain that winds the periodic direction — the chain
/// generalization of [`split_periodic_face_into_bands`], whose separator must
/// be a closed circle. Mirrors its structure: same boundary preconditions
/// (two closed rim circles + seam lines), same per-band wire shape
/// (rim, seam, separator, seam), same precomputed interior points.
///
/// Returns `None` (caller falls through) unless: the boundary is exactly two
/// closed rims + seam edges; ALL sections belong to one winding chain; the
/// chain has a vertex on the seam meridian (the seam-anchoring pre-step
/// guarantees this for winding chains); and every chain sample stays
/// strictly between the rims.
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
    use std::f64::consts::{PI, TAU};

    if !matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) {
        return None;
    }
    let close_tol = tol * 100.0;

    let (chain, _winding) = winding_section_chain(sections, surface, tol)?;
    if chain.len() != sections.len() {
        // Sections outside the chain would be silently dropped — defer.
        return None;
    }

    // Boundary: exactly two closed rim circles plus seam Line edges.
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
    if boundary_circles.len() != 2 || seam_edges.is_empty() {
        return None;
    }
    let (seam_u, _) = surface.project_point(seam_edges[0].start_3d)?;
    let wrap_pi = |d: f64| -> f64 { (d + PI).rem_euclid(TAU) - PI };

    let circle_v = |e: &OrientedPCurveEdge| -> Option<f64> {
        let (_, v) = surface.project_point(e.start_3d)?;
        let on_seam = surface.evaluate(seam_u, v)?;
        ((on_seam - e.start_3d).length() < close_tol).then_some(v)
    };
    let v0 = circle_v(boundary_circles[0])?;
    let v1 = circle_v(boundary_circles[1])?;
    let (v_bot, bot_edge, v_top, top_edge) = if v0 < v1 {
        (v0, boundary_circles[0], v1, boundary_circles[1])
    } else {
        (v1, boundary_circles[1], v0, boundary_circles[0])
    };
    if v_top - v_bot < close_tol {
        return None;
    }

    // Rotate the chain to start at its seam-anchored vertex.
    let traversal_start = |&(idx, fwd): &(usize, bool)| -> Point3 {
        let s = &sections[idx];
        if fwd { s.start } else { s.end }
    };
    let seam_pos = chain.iter().position(|entry| {
        surface
            .project_point(traversal_start(entry))
            .is_some_and(|(u, _)| wrap_pi(u - seam_u).abs() < 1e-6)
    })?;
    let mut chain: Vec<(usize, bool)> = chain;
    chain.rotate_left(seam_pos);
    let (_, v_x) = surface.project_point(traversal_start(&chain[0]))?;

    // Every chain sample must sit strictly between the rims, and the chain's
    // v profile is recorded per sample for the interior-point lookup below.
    let mut samples_uv: Vec<(f64, f64)> = Vec::new();
    for &(idx, _) in &chain {
        let s = &sections[idx];
        for k in 0..=8 {
            let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            let t = d0 + (d1 - d0) * (f64::from(k) / 8.0);
            let p = s.curve_3d.evaluate_with_endpoints(t, s.start, s.end);
            let (u, v) = surface.project_point(p)?;
            if v < v_bot + close_tol || v > v_top - close_tol {
                return None;
            }
            samples_uv.push((u, v));
        }
    }

    // Traversal tangent of the chain at its seam start, compared with the
    // bottom rim's traversal tangent there (the circle-band rule): the chain
    // list as ordered plays the LOWER role when aligned, else it is flipped.
    let ref_tan = {
        let EdgeCurve::Circle(c) = &bot_edge.curve_3d else {
            return None;
        };
        let t = c.tangent(c.project(bot_edge.start_3d));
        if bot_edge.forward { t } else { -t }
    };
    let chain_tan = {
        let (idx, fwd) = chain[0];
        let s = &sections[idx];
        let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
        let t_at = if fwd { d0 } else { d1 };
        let tan = s.curve_3d.tangent_with_endpoints(t_at, s.start, s.end);
        if fwd { tan } else { -tan }
    };
    let chain_is_lower_role = chain_tan.dot(ref_tan) > 0.0;

    // Materialize the chain as pcurve edges in a given traversal direction,
    // walking UV u with nearest-copy continuity from the seam.
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
            let u1 = u0 + wrap_pi(u_raw1 - u_raw0);
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
    let chain_lower = build_chain(chain_is_lower_role)?;
    let chain_upper = build_chain(!chain_is_lower_role)?;

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

    // Interior points: the chain's v at the antipodal meridian, from the
    // sample nearest u = seam_u + π.
    let u_q = (seam_u + PI).rem_euclid(TAU);
    let v_at_q = samples_uv
        .iter()
        .min_by(|a, b| {
            wrap_pi(a.0 - u_q)
                .abs()
                .partial_cmp(&wrap_pi(b.0 - u_q).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|&(_, v)| v)?;
    let lower_interior = surface.evaluate(u_q, f64::midpoint(v_bot, v_at_q))?;
    let upper_interior = surface.evaluate(u_q, f64::midpoint(v_at_q, v_top))?;

    // Lower band: bottom rim, seam up to the chain's seam vertex, the chain
    // in the upper role, seam back down. Upper band symmetric.
    let mut lower_wire = vec![bot_edge.clone(), mk_seam(v_bot, v_x)?];
    lower_wire.extend(chain_upper);
    lower_wire.push(mk_seam(v_x, v_bot)?);
    let mut upper_wire = chain_lower;
    upper_wire.push(mk_seam(v_x, v_top)?);
    upper_wire.push(top_edge.clone());
    upper_wire.push(mk_seam(v_top, v_x)?);

    Some(vec![
        SplitSubFace {
            surface: surface.clone(),
            outer_wire: lower_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(lower_interior),
        },
        SplitSubFace {
            surface: surface.clone(),
            outer_wire: upper_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(upper_interior),
        },
    ])
}

/// Whether the sections chain into a loop that WINDS the surface's periodic
/// u direction. See [`winding_section_chain`].
fn sections_form_winding_chain(sections: &[SectionEdge], surface: &FaceSurface, tol: f64) -> bool {
    winding_section_chain(sections, surface, tol).is_some()
}

/// The ordered section chain forming a loop that WINDS the surface's
/// periodic u direction (|net signed u-progress| > π after chaining by
/// endpoint position), as `(piece index, traversed start→end)` entries, with
/// the signed winding. Non-periodic surfaces never wind. Chains that fail to
/// close are conservatively reported as non-winding — the internal-loops
/// path has its own closure requirements.
fn winding_section_chain(
    sections: &[SectionEdge],
    surface: &FaceSurface,
    tol: f64,
) -> Option<(Vec<(usize, bool)>, f64)> {
    use std::collections::HashMap;
    use std::f64::consts::{PI, TAU};

    let (Some(_), _) = super::pcurve_compute::surface_periods(surface) else {
        return None;
    };
    if sections.len() < 2 {
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
    let wrap_pi = |d: f64| -> f64 { (d + PI).rem_euclid(TAU) - PI };

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
            let (from, to) = if forward {
                (s.start, s.end)
            } else {
                (s.end, s.start)
            };
            let (Some(u0), Some(u1)) = (proj_u(from), proj_u(to)) else {
                return None;
            };
            winding += wrap_pi(u1 - u0);
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
        if closed && winding.abs() > PI {
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
/// edges and identical duplicates are left untouched.
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
        if !(inp.is_section && inp.is_arc) {
            continue;
        }
        let Some(mid) = arc_mid(&inp.edge) else {
            continue;
        };
        let (a, b) = (q2(inp.a), q2(inp.b));
        let key = if a <= b { (a, b) } else { (b, a) };
        pair_mids
            .entry(key)
            .or_default()
            .push(q2(frame.project(mid)));
    }
    let colliding: std::collections::HashSet<(Q2, Q2)> = pair_mids
        .into_iter()
        .filter(|(_, mids)| mids.iter().any(|m| *m != mids[0]))
        .map(|(k, _)| k)
        .collect();
    if colliding.is_empty() {
        return;
    }
    let mut rebuilt: Vec<ArrInput> = Vec::with_capacity(inputs.len() + 2);
    for inp in inputs.drain(..) {
        let (a, b) = (q2(inp.a), q2(inp.b));
        let key = if a <= b { (a, b) } else { (b, a) };
        let mid3 = if inp.is_section && inp.is_arc && colliding.contains(&key) {
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
        false,
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
        true,
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
    // When true, resolve overlapping nested minimal faces (a holed cap whose
    // holes/sections form a component separate from the outer-boundary loop)
    // via even-odd containment nesting: depth-even faces are solid regions
    // emitted with their direct-child faces as holes, depth-odd faces are holes,
    // and solid regions filling an `original_holes` opening are dropped (air).
    // False keeps the historical "drop only the max-area face, emit each as a
    // flat outer wire" behaviour for the section-only entry point.
    even_odd_nesting: bool,
    // The face's ORIGINAL holes (compartment openings) in this face's frame.
    // Used by the even-odd path to drop solid regions that fill an opening.
    // Empty for the section-only entry point.
    original_holes: &[Vec<OrientedPCurveEdge>],
    // When present, section-input interior break points are recorded per pave
    // block (exact UV → 3D via the frame) so curved faces sharing the same
    // section curve pre-split at identical points.
    mut split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
) -> Option<Vec<SplitSubFace>> {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::Vec2;

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
            .map(|(_, p)| (d.x().mul_add(p.y() - a.y(), -(d.y() * (p.x() - a.x()))) / len).abs())
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
        let side =
            |p: Point2| -> f64 { ld.x().mul_add(p.y() - la.y(), -(ld.y() * (p.x() - la.x()))) };
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
            for bp in [b0, b1] {
                let w = (bp - a0).dot(d) / (len * len);
                if w > 1e-6 && w < 1.0 - 1e-6 {
                    let on = a0 + d * w;
                    if (on - bp).length() < tol * 100.0 && (!i_is_arc || chord_break_on_arc(i, bp))
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
            && let Some(reg) = split_registry.as_deref_mut()
            && let Some(pb_id) = inputs[i].edge.pave_block_id
        {
            for brk in &ts[1..n_breaks - 1] {
                let uv = brk.1.unwrap_or(a0 + d * brk.0);
                reg.entry(pb_id)
                    .or_default()
                    .push(frame.evaluate(uv.x(), uv.y()));
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
    if sub_edges.is_empty() {
        return None;
    }
    // Drop the per-edge source/whole tags into a parallel lookup keyed by seg_id
    // so the half-edge trace (which only needs vertex pairs) stays unchanged.
    let sub_edge_src: Vec<(usize, bool)> = sub_edges.iter().map(|&(_, _, i, w)| (i, w)).collect();
    let sub_edges: Vec<(UvKey, UvKey)> = sub_edges.iter().map(|&(a, b, _, _)| (a, b)).collect();

    // Build the directed half-edge adjacency. Each undirected sub-edge id maps
    // to two directed half-edges; both carry that id so adjacent regions share
    // one topology edge. Half-edge index 2*k = forward (va->vb), 2*k+1 = reverse.
    let mut halfs: Vec<ArrHalfEdge> = Vec::with_capacity(sub_edges.len() * 2);
    for (seg_id, &(ka, kb)) in sub_edges.iter().enumerate() {
        let pa = vert_pos[&ka];
        let pb = vert_pos[&kb];
        let fwd = pb - pa;
        let rev = pa - pb;
        halfs.push(ArrHalfEdge {
            from: ka,
            to: kb,
            seg_id,
            angle: fwd.y().atan2(fwd.x()),
        });
        halfs.push(ArrHalfEdge {
            from: kb,
            to: ka,
            seg_id,
            angle: rev.y().atan2(rev.x()),
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
            let mut best: Option<usize> = None;
            let mut best_off = f64::MAX;
            for &c in cands {
                if used[c] || c == twin {
                    continue;
                }
                let off = (halfs[c].angle - back_angle).rem_euclid(std::f64::consts::TAU);
                if off < best_off {
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

    if even_odd_nesting {
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
        // it.
        let outer_area = face_area(&faces[outer_idx]).abs();
        let interior_area_sum: f64 = areas.iter().sum();
        let clean_tiling =
            (interior_area_sum - outer_area).abs() <= outer_area.mul_add(1e-6, tol * tol);
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
        return (!result.is_empty()).then_some(result);
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
    Some(result)
}

/// Rectilinear-arrangement rescue for a u-periodic cylinder band whose greedy
/// wire trace self-crosses (a box cut notching the wall at partial overlap).
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
/// fails to yield a simple partition. The gate above ensures this fires only when
/// the greedy trace is already broken, so faces the greedy handles are untouched.
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
                        0.5_f64.mul_add(t1 - t0, t0),
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

    for &(u, v0, v1) in &verticals {
        let mut breaks: Vec<f64> = vec![v0, v1];
        for &(hv, hu0, hu1) in &horizontals {
            if hv > v0 + tol && hv < v1 - tol && u >= hu0 - cov && u <= hu1 + cov {
                breaks.push(hv);
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
        });
        halfs.push(ArrHalfEdge {
            from: kb,
            to: ka,
            seg_id,
            angle: rev.y().atan2(rev.x()),
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
                let t = (t1 - t0).mul_add(k as f64 / CURVE_SAMPLES as f64, t0);
                e.curve_3d.evaluate_with_endpoints(t, s3, e3)
            })
            .collect();
        if let (Some(first), Some(last)) = (pts3.first(), pts3.last())
            && (*first - e.start_3d).length() > (*last - e.start_3d).length()
        {
            pts3.reverse();
        }
        let samples: Vec<Point2> = pts3
            .into_iter()
            .filter_map(|p| surface.project_point(p).map(|(u, v)| Point2::new(u, v)))
            .collect();
        for s in samples {
            let unwrapped = match prev {
                Some(pv) => {
                    let du = ((s.x() - pv.x() + PI).rem_euclid(TAU)) - PI;
                    Point2::new(pv.x() + du, s.y())
                }
                None => s,
            };
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
            .map(|k| Point2::new(k.mul_add(TAU, u), v))
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
            for k in 0..=n {
                #[allow(clippy::cast_precision_loss)]
                let t = (t1 - t0).mul_add(k as f64 / n as f64, t0);
                let d = (eval(t) - p).length();
                if d < bd {
                    bd = d;
                    bi = k;
                }
            }
            if bd > FIT_BAND * 4.0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let (mut lo, mut hi) = (
                (t1 - t0).mul_add(bi.saturating_sub(1) as f64 / n as f64, t0),
                (t1 - t0).mul_add(((bi + 1).min(n)) as f64 / n as f64, t0),
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
            let flip = (clear_in(a) && !raw_inside(b)) || (!raw_inside(a) && clear_in(b));
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
            let tc = f64::midpoint(lo, hi);
            let pc = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, tc);
            let j = snap_to_boundary(pc).unwrap_or(pc);
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
/// Loops of two edges are exempt: consecutive edges always share a vertex, so
/// for `n == 2` closure alone forces the reverse pattern and every lens region
/// (an arc plus its co-endpoint chord) matches. Only at three or more edges
/// does the pattern actually mean the walker doubled back — at two it fired on
/// the honeycomb cap arrangement's legitimate lens and cost `pcut3` its zero
/// free-edge pin.
fn loops_have_out_and_back(loops: &[Vec<OrientedPCurveEdge>], tol: f64) -> bool {
    for lp in loops {
        let n = lp.len();
        if n < 3 {
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

/// Split a face by its section edges, producing sub-faces.
///
/// If there are no section edges, returns a single sub-face covering
/// the entire face (pass-through).
///
/// Wraps [`split_face_2d_impl`] to salvage closed-circle *cap* sections on
/// plane faces (the drilled-socket → bin-body fuse: the socket's screw-hole
/// rims land as closed circle sections on the body's coincident bottom plane).
/// The impl drops closed circles in both its arrangement path
/// ([`arrangement_regions_from_inputs`] discards any input whose UV chord is
/// zero-length, and a closed circle has `start == end`) and its wire-builder
/// fallback, leaving the drilled cylinder rims as free edges; the only impl
/// path that carves a closed circle is the single-closed fast path (exactly
/// one section, closed), which never fires when the circle is mixed with open
/// sections or when there are two or more circles. Peel the cap circles off,
/// split the plane by the remaining sections, then carve each cap circle into
/// the sub-face that contains it ([`distribute_cap_circles`]).
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
    split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
) -> Vec<SplitSubFace> {
    let run_impl = move |secs: &[SectionEdge]| {
        split_face_2d_impl(
            topo,
            face_id,
            secs,
            rank,
            tol,
            frame,
            info,
            edge_images,
            split_registry,
        )
    };

    // Cheap gate for the common case: no closed-circle section, nothing to
    // salvage — skip all frame and polygon work below.
    let any_closed_circle = sections.iter().any(|s| {
        (s.start - s.end).length() < tol.linear && matches!(s.curve_3d, EdgeCurve::Circle(_))
    });
    if !any_closed_circle {
        return run_impl(sections);
    }

    // Salvage applies only to plane faces (curved faces route closed circles
    // through their own band/internal-loop paths).
    let Ok(face) = topo.face(face_id) else {
        return run_impl(sections);
    };
    let surface = face.surface().clone();
    if !matches!(surface, FaceSurface::Plane { .. }) {
        return run_impl(sections);
    }

    // Build the face's outer polygon in UV so a *genuine* cap circle (a full
    // closed circle strictly interior to the face — a drilled hole rim) can be
    // told apart from the zero-span arc remnants that the FF/pave machinery
    // leaves at faceted-corner junctions (those sit ON the boundary, share a
    // point with an open arc section, and their "circle" is the corner cylinder
    // tangent to the face outline). Only genuine caps are peeled off; everything
    // else stays in `sections` and reaches the impl unchanged.
    // The frame must be built from the same points the impl uses, so the
    // sub-face UVs and the projected circle centres agree; the containment
    // polygon must follow wire traversal order (see
    // [`collect_wire_points_oriented`]).
    let frame_pts = collect_wire_points(topo, face.outer_wire());
    let owned_frame;
    let cap_frame = if let Some(f) = frame {
        f
    } else {
        let normal = extract_plane_normal(&surface);
        owned_frame = PlaneFrame::from_plane_face(normal, &frame_pts);
        &owned_frame
    };
    let outer_poly: Vec<Point2> = collect_wire_points_oriented(topo, face.outer_wire())
        .iter()
        .map(|&p| cap_frame.project(p))
        .collect();

    let mut cap_sections: Vec<SectionEdge> = Vec::new();
    let mut cap_centers: Vec<Point3> = Vec::new();
    let mut rest_sections: Vec<SectionEdge> = Vec::new();
    for s in sections {
        let cap_center = if (s.start - s.end).length() < tol.linear
            && outer_poly.len() >= 3
            && let EdgeCurve::Circle(circle) = &s.curve_3d
        {
            let center_uv = cap_frame.project(circle.center());
            (super::classify_2d::point_in_polygon_2d(center_uv, &outer_poly)
                && super::classify_2d::distance_to_polygon_boundary(center_uv, &outer_poly)
                    > circle.radius() * CAP_INTERIORITY_MARGIN)
                .then(|| circle.center())
        } else {
            None
        };
        match cap_center {
            Some(c) => {
                cap_sections.push(s.clone());
                cap_centers.push(c);
            }
            None => rest_sections.push(s.clone()),
        }
    }

    // Engage only where the impl would silently drop cap circles: at least one
    // genuine cap circle AND not the single-closed fast path (exactly one
    // section, that one circle) which the impl already handles correctly.
    if cap_sections.is_empty() || (cap_sections.len() == 1 && sections.len() == 1) {
        return run_impl(sections);
    }

    // The impl ignores closed circles on plane faces outside its single-closed
    // fast path, so withholding the caps does not change how the remaining
    // sections split.
    let base = run_impl(&rest_sections);

    distribute_cap_circles(
        topo,
        face_id,
        base,
        &cap_sections,
        &cap_centers,
        rank,
        frame,
    )
}

/// Slack on the cap-circle interiority test. A genuine drilled hole's centre
/// clears the face outline by more than its radius, while a corner cylinder's
/// section circle is exactly tangent (centre exactly one radius out); the 5%
/// margin keeps float noise on such a tangent circle from reading as interior.
const CAP_INTERIORITY_MARGIN: f64 = 1.05;

/// Wire points in TRAVERSAL order, honouring each oriented edge's direction.
///
/// [`collect_wire_points`] pushes every edge's stored `start()` and ignores the
/// orientation flag, so a wire carrying reversed edges comes back scrambled.
/// Containment tests need the traversal-ordered outline, so they use this.
fn collect_wire_points_oriented(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Vec<Point3> {
    let Ok(wire) = topo.wire(wire_id) else {
        return Vec::new();
    };
    let mut pts = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            if let Ok(v) = topo.vertex(vid) {
                pts.push(v.point());
            }
        }
    }
    pts
}

/// Carve closed cap circles (see [`split_face_2d`]) into the base sub-faces.
///
/// Each cap circle is assigned to the first base sub-face whose outer boundary
/// contains the circle's centre in UV, then that sub-face is re-split via
/// [`split_face_with_internal_loops`] (disc cap + holed remainder). A circle
/// whose centre lies inside a sub-face hole is air — the rim of a drill
/// emerging inside an existing opening — and a circle contained by no sub-face
/// has no home; both are dropped, exactly as the impl's air filter and
/// arrangement paths drop them.
fn distribute_cap_circles(
    topo: &Topology,
    face_id: FaceId,
    base: Vec<SplitSubFace>,
    cap_sections: &[SectionEdge],
    cap_centers: &[Point3],
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

    let centers_uv: Vec<Point2> = cap_centers.iter().map(|&c| frame.project(c)).collect();

    let mut assigned = vec![false; cap_sections.len()];
    let mut result: Vec<SplitSubFace> = Vec::with_capacity(base.len() + cap_sections.len());
    for sf in base {
        // The frame-based sampler is orientation-unambiguous; the stored-pcurve
        // sampler can fold a loop carrying reversed arc sections into a
        // self-crossing polygon (see [`sample_wire_loop_uv_via_frame`]).
        let poly = sample_wire_loop_uv_via_frame(&sf.outer_wire, frame);
        let contained: Vec<SectionEdge> = if poly.len() < 3 {
            Vec::new()
        } else {
            cap_sections
                .iter()
                .enumerate()
                .filter_map(|(i, cs)| {
                    if assigned[i]
                        || !super::classify_2d::point_in_polygon_2d(centers_uv[i], &poly)
                        || is_inside_any_hole(&centers_uv[i], &sf.inner_wires)
                    {
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
            let carved = split_face_with_internal_loops(
                &sf.surface,
                &sf.outer_wire,
                &sf.inner_wires,
                &contained,
                rank,
                sf.reversed,
                face_id,
                &sub_wire_pts,
            );
            result.extend(carved);
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
    mut split_registry: Option<&mut std::collections::HashMap<usize, Vec<Point3>>>,
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
    // of partitioning them. Drop near-exact geometric duplicates.
    let deduped_sections: Option<Vec<SectionEdge>> = {
        let band = tol.linear * 10.0;
        let same_pt = |p: Point3, q: Point3| (p - q).length() <= band;
        let mids: Vec<Point3> = sections
            .iter()
            .map(|s| s.curve_3d.evaluate_with_endpoints(0.5, s.start, s.end))
            .collect();
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
                if ends_match && same_pt(mids[i], mids[j]) {
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
    let surface = face.surface().clone();
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
    // for the same reason its z-planes did).
    let mut boundary_edges = if face.inner_wires().is_empty() {
        super::face_splitter::conversion::boundary_edges_to_pcurve_with_images(
            topo,
            face.outer_wire(),
            &surface,
            &wire_pts,
            if is_plane { Some(frame) } else { None },
            edge_images,
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
            if edges.is_empty() || (iw_pts.len() < 3 && !has_closed_curve) {
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
                        let pcurve = super::pcurve_compute::compute_pcurve_on_surface(
                            e.curve(),
                            s3,
                            e3,
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
        let outer_sign = signed_area_2d(&sample_wire_loop_uv(&boundary_edges)) >= 0.0;
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
                let pts = sample_wire_loop_uv(&hole);
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
        let scale = 1.0 / tol.linear.max(1e-12);
        let q = |p: Point3| -> (i64, i64, i64) {
            (
                (p.x() * scale).round() as i64,
                (p.y() * scale).round() as i64,
                (p.z() * scale).round() as i64,
            )
        };
        let mut seen = std::collections::HashSet::new();
        deduped_sections = sections
            .iter()
            .filter(|s| {
                // Key on the endpoints plus a midpoint sample so two distinct
                // arcs sharing endpoints (e.g. the two halves of a split
                // circle) are not collapsed into one.
                let (a, b) = (q(s.start), q(s.end));
                let mid = q(evaluate_edge_at_t(&s.curve_3d, s.start, s.end, 0.5));
                seen.insert((if a <= b { (a, b) } else { (b, a) }, mid))
            })
            .cloned()
            .collect();
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
    // `split_noseam_face_direct` carves cap+band from OPEN section arcs (it
    // skips full-circle sections, relying on the FF boundary-crossing split to
    // pre-open them). A hemisphere cut by a cylinder yields a CLOSED circle
    // interior to the face (no equator crossing) — there are no open arcs, so
    // that path would unsplit it. Defer those to the internal-loops path
    // below, which carves the cap disc as a hole.
    let has_open_section = sections
        .iter()
        .any(|s| (s.start - s.end).length() > tol.linear);
    if all_boundary_line && !is_plane && has_open_section {
        return split_noseam_face_direct(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            &wire_pts,
            tol.linear,
        );
    }

    // Torus notch band: a box (or analogous) cut removes a sector of the ring
    // whose surface boundary is two φ-wrapping loops; the kept torus is the
    // u-band between them. Contained tracer — defers (None) when the in-box arcs
    // don't stitch into exactly two φ-wrapping loops. Restricted to ALL-OPEN
    // sections: a CLOSED-loop section (a section circle/loop entirely interior to
    // the torus, e.g. a small tool poking a closed hole) bounds its own region
    // that the band tracer would silently drop, so defer those to the
    // internal-loops path.
    let all_sections_open = !sections.is_empty()
        && sections
            .iter()
            .all(|s| (s.start - s.end).length() > tol.linear);
    if matches!(surface, FaceSurface::Torus(_))
        && original_inner_wires.is_empty()
        && all_sections_open
        && let Some(band) =
            split_torus_band_by_arrangement(&surface, sections, rank, reversed, face_id, tol.linear)
    {
        return band;
    }

    // Seam-anchor a winding section chain: a chain that winds the periodic
    // direction must connect to the seam in the trace graph, or it floats as
    // an island and every wire builder mis-handles it (the closed-circle
    // band shortcut gets this via the seam-anchor pre-pass; a chain of arcs
    // and conic pieces gets it here). Splitting the crossing piece at the
    // seam meridian makes the crossing point a section ENDPOINT, so the
    // boundary-splitting below anchors the seam edge at the same 3D point
    // and the graphs share a vertex.
    let seam_anchored_sections: Vec<SectionEdge>;
    let sections: &[SectionEdge] = if !is_plane
        && super::pcurve_compute::surface_periods(&surface).0.is_some()
        && sections_form_winding_chain(sections, &surface, tol.linear)
        && let Some(seam_u) = boundary_seam_u(&boundary_edges, &surface, tol.linear)
    {
        seam_anchored_sections =
            split_sections_at_seam_meridian(sections, &surface, seam_u, tol.linear);
        &seam_anchored_sections
    } else {
        sections
    };

    // Band shortcut: closed section circles on a u-periodic face split it
    // into stacked bands, not discs. Requires seam-anchored circles (see
    // the seam-anchor pre-pass in fill_images_faces); falls through to the
    // generic paths when preconditions don't hold.
    if u_periodic
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
    if u_periodic
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
            deduped_line_loops = plane_internal_line_loops(
                sections,
                frame,
                &boundary_edges,
                &original_inner_wires,
                tol.linear,
            );
            deduped_line_loops.is_some()
        }
    } else {
        // Non-plane faces: check if all section endpoints are off the
        // boundary in UV space.
        let uv_tol = 0.01; // ~0.6 deg in angular coordinates
        let endpoints_internal = sections.iter().all(|s| {
            let start_on_boundary =
                is_point_on_boundary_uv(s.start, &surface, &boundary_edges, uv_tol);
            let end_on_boundary = is_point_on_boundary_uv(s.end, &surface, &boundary_edges, uv_tol);
            !start_on_boundary && !end_on_boundary
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

    if all_sections_internal {
        let secs = deduped_line_loops.as_deref().unwrap_or(sections);
        log::debug!(
            "split_face_2d: face {face_id:?} routed to internal-loops path ({} sections)",
            secs.len()
        );
        return split_face_with_internal_loops(
            &surface,
            &boundary_edges,
            &original_inner_wires,
            secs,
            rank,
            reversed,
            face_id,
            &wire_pts,
        );
    }

    let mut split_pts_3d: Vec<Point3> = sections.iter().flat_map(|s| [s.start, s.end]).collect();
    split_pts_3d.append(&mut outer_clip_anchors);
    if !is_plane && let Some(reg) = split_registry.as_deref_mut() {
        split_pts_3d.extend(reg.values().flatten().copied());
    }

    // For periodic faces, align closed boundary edge UV with seam edge UV.
    // The same 3D vertex projects to u=0 (from circle unwrapping) and u=seam
    // (from Line edge projection). Shift the circle UV so it starts at seam_u.
    if u_periodic {
        let seam_u_opt = boundary_edges.iter().find_map(|e| {
            if matches!(e.curve_3d, EdgeCurve::Line) {
                surface.project_point(e.start_3d).map(|(u, _)| u)
            } else {
                None
            }
        });
        if let Some(seam_u) = seam_u_opt {
            for edge in &mut boundary_edges {
                if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                    // Closed edge: shift UV so start_uv.x() == seam_u.
                    let shift = seam_u - edge.start_uv.x();
                    if shift.abs() > 0.01 {
                        edge.start_uv = Point2::new(edge.start_uv.x() + shift, edge.start_uv.y());
                        edge.end_uv = Point2::new(edge.end_uv.x() + shift, edge.end_uv.y());
                    }
                }
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
    if u_periodic && !sections.is_empty() {
        // Find the seam Line edge's vertex UV to determine seam_u.
        let seam_u = {
            let mut su = 0.0_f64;
            for edge in &boundary_edges {
                if matches!(edge.curve_3d, EdgeCurve::Line)
                    && let Some((u, _)) = surface.project_point(edge.start_3d)
                {
                    su = u;
                    break;
                }
            }
            su
        };
        let anti_u = (seam_u + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU);

        for edge in &boundary_edges {
            if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                // Closed edge: find the 3D point at u = seam_u + pi on the surface.
                // Project the boundary vertex to get v, then evaluate surface at (anti_u, v).
                if let Some((_, v)) = surface.project_point(edge.start_3d)
                    && let Some(anti_pt) = surface.evaluate(anti_u, v)
                {
                    split_pts_3d.push(anti_pt);
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

    let boundary_edges = split_boundary_edges_at_3d_points(
        boundary_edges,
        &split_pts_3d,
        if is_plane { Some(frame) } else { None },
        &surface,
        tol.linear,
    );

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
    if is_plane && !original_inner_wires.is_empty() {
        let sec_segs: Vec<(Point3, Point3)> = sections
            .iter()
            .filter(|sct| matches!(sct.curve_3d, EdgeCurve::Line))
            .map(|sct| (sct.start, sct.end))
            .collect();
        let on_seg = |p: Point3, a: Point3, b: Point3| -> bool {
            let d = b - a;
            let len2 = d.dot(d);
            if len2 < 1e-18 {
                return false;
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
                    if sec_segs.iter().any(|&(a, b)| on_seg(p, a, b))
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
                for (si, sct) in sections.iter().enumerate() {
                    if !matches!(sct.curve_3d, EdgeCurve::Line) {
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
            let poly: Vec<Point2> = hole.iter().map(|e| frame.project(e.start_3d)).collect();
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
                (b.x() - a.x()).mul_add(p.y() - a.y(), -((b.y() - a.y()) * (p.x() - a.x())))
            };
            let in_region = |p: Point2| -> bool {
                super::classify_2d::point_in_polygon_2d(p, &poly)
                    || arcs.iter().any(|&(c, r, u0, u1)| {
                        let d = ((p.x() - c.x()).powi(2) + (p.y() - c.y()).powi(2)).sqrt();
                        d < r && side(u0, u1, p) * side(u0, u1, c) < 0.0
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
                let lo = gap_start + max_gap;
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

    // Split section edges where another section's endpoint lands on their
    // interior (L/T junctions). Needs ≥ 2 distinct sections (one pair to cross
    // another); runs after the partial-band u-unwrap so split UVs match the
    // face's continuous window.
    if all_edges.len() > n_boundary_edges + 2 {
        split_sections_at_t_junctions(
            &mut all_edges,
            n_boundary_edges,
            &surface,
            if is_plane { Some(frame) } else { None },
            &wire_pts,
            tol.linear,
            split_registry.as_deref_mut(),
        );
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
    }

    // Snap section UV endpoints onto 3D-coincident boundary UV copies. The
    // wire graph keys junctions on UV quantized at the exact tolerance, but a
    // section endpoint's UV comes from its own pcurve evaluation while the
    // boundary split's UV comes from span interpolation — the same exact 3D
    // junction can carry UV copies ~1e-6 apart (the fit-error class), which
    // lands them in different graph cells and the section reads as pendant.
    // Adopting the boundary's UV copy is a sub-weld no-op in 3D.
    {
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
                    .filter(|(a3, auv)| {
                        (*a3 - p3).length() < tol.linear && (*auv - uv).length() < 1e-2
                    })
                    .min_by(|(_, a), (_, b)| (*a - uv).length().total_cmp(&(*b - uv).length()))
                    .map(|(_, auv)| *auv);
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
    let all_edges = super::wire_builder::remove_pendant_sections(
        &all_edges, tol.linear, u_periodic, v_periodic,
    );

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
            .filter(|(i, e)| {
                // Same scale as the boundary-proximity `uv_tol` above
                // (~0.6 deg in angular coordinates): a closed circle section
                // has a zero 3D chord but a full-period UV extent.
                const ZERO_EXTENT_UV: f64 = 0.01;
                *i < n_b
                    || (e.start_3d - e.end_3d).length() >= tol.linear
                    || (e.start_uv - e.end_uv).length() >= ZERO_EXTENT_UV
            })
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
                    let p = Point2::new(
                        f.mul_add(t.x() - s.x(), s.x()),
                        f.mul_add(t.y() - s.y(), s.y()),
                    );
                    super::classify_2d::distance_to_polygon_boundary(p, &boundary_poly) <= eps
                })
            })
            .map(|(_, e)| e)
            .collect()
    } else {
        all_edges
    };

    // Build wire loops via angular-sorting traversal.
    let mut loops = build_wire_loops(&all_edges, tol.linear, u_periodic, v_periodic);
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
                format!("STRACE-LOOP n={} {}", lp.len(), vs.join(""))
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
        let boundary_pts = sample_wire_loop_uv(&all_edges[..n_boundary_edges]);
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
    let bay_mouth_arrangement =
        !holes_integrated && boundary_arc_crossed && original_inner_wires.len() >= 2;
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
        && loops_have_out_and_back(&loops, tol.linear);
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
        && let Some(ref boundary) = boundary_edges_backup
        && let Some(result) = try_split_crossing_plane_face(
            &surface, boundary, sections, rank, reversed, face_id, frame, tol,
        )
        && result.len() > loops.len()
    {
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

        let arr = split_plane_face_by_arrangement(
            &surface,
            boundary,
            sections,
            rank,
            reversed,
            face_id,
            frame,
            tol.linear,
            split_registry,
        );
        if let Some(result) = arr
            && (result.len() > loops.len()
                || wire_loops_self_cross(&loops, tol.linear)
                || greedy_outer_loops_nested(&loops, cw_loops)
                || wire_loops_have_degenerate_area(&loops, tol.linear))
        {
            return result;
        }
    }

    // Rectilinear-arrangement rescue for a u-periodic cylinder band whose greedy
    // wire trace broke (self-crossing, overlapping, or degenerate loops) -- a box
    // cut notching the wall at partial overlap figure-eights the angular builder.
    // Gated exactly like the plane arrangement rescue: fires only when the greedy
    // loops are already broken, so it never changes a face the greedy handles.
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
    let ring_duplicated = wire_loops_duplicate_cover(&loops, tol.linear);
    // Sector rescue for an UNDER-split u-periodic lateral: one full-height
    // ruling plus the glued seam should sector the strip, but to the glued
    // walker an annulus cut once is a single wrapped region (the mid-wall
    // pad whose second wall crossing rides the seam). Fires only when the
    // greedy produced no split at all, so configs the greedy handles (all
    // rulings in-face) never take this path.
    if u_periodic
        && !v_periodic
        && loops.len() <= 1
        && !sections.is_empty()
        && matches!(&surface, FaceSurface::Cylinder(_))
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
        return sectors;
    }
    if u_periodic
        && !v_periodic
        && !sections.is_empty()
        && (matches!(&surface, FaceSurface::Cylinder(_)) && (greedy_broken || ring_duplicated)
            || matches!(&surface, FaceSurface::Cone(_)) && ring_duplicated)
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
        let dcel = build_wire_loops_dcel(&dcel_input, tol.linear, u_periodic, v_periodic);
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
        let dcel = build_wire_loops_dcel(&all_edges, tol.linear, u_periodic, v_periodic);
        // This branch is u-periodic by its gate: the pinch splitter is
        // unsafe on the glued seam graph (see the DCEL consult above), so
        // the trace is used as-is. The broken-loop flags are RELATIVE to the
        // greedy result, not absolute — `wire_loops_self_cross` is not
        // periodic-aware, so a valid winding band orbit revisiting the
        // quantized seam vertex would fail an absolute gate and the refined
        // partition (the separated strip) would be rejected.
        if dcel.len() > loops.len()
            && !wire_loops_have_degenerate_area(&dcel, tol.linear)
            && (!wire_loops_self_cross(&dcel, tol.linear)
                || wire_loops_self_cross(&loops, tol.linear))
            && (!greedy_outer_loops_nested(&dcel, cw_loops)
                || greedy_outer_loops_nested(&loops, cw_loops))
        {
            loops = dcel;
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
        if dcel.len() > loops.len()
            && !wire_loops_have_degenerate_area(&dcel, tol.linear)
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

    for wire_loop in loops {
        if use_structural_classification {
            // Structural: a loop containing both Line edges (seam) and
            // non-Line edges (section arcs / circles) is a band = outer.
            let has_line = wire_loop
                .iter()
                .any(|e| matches!(e.curve_3d, EdgeCurve::Line));
            let has_nonline = wire_loop
                .iter()
                .any(|e| !matches!(e.curve_3d, EdgeCurve::Line));
            if has_line && has_nonline {
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
        let mut promoted: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
        holes.retain(|hole| {
            let hole_pts = sample_loop(hole);
            if hole_pts.len() < 3 {
                return true;
            }
            let nested = outer_uv
                .iter()
                .any(|o| loop_containment(&hole_pts, o) != LoopContainment::Outside);
            if nested {
                true
            } else {
                promoted.push(hole.clone());
                false
            }
        });
        for mut region in promoted {
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
    for (outer_wire, _area) in outers {
        if !hole_polys_for_discard.is_empty() {
            let poly = sample_wire_loop_uv_via_frame(&outer_wire, frame);
            if poly.len() >= 3 {
                let ip = super::classify_2d::sample_interior_point(&poly);
                let matched_hole = hole_polys_for_discard.iter().find(|hp| {
                    super::classify_2d::point_in_polygon_2d(ip, hp)
                        && super::classify_2d::distance_to_polygon_boundary(ip, hp) > hole_margin
                });
                if let Some(hole_poly) = matched_hole {
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
        sub_faces.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        });
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

        for hole in &original_inner_wires {
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
                    .filter(|&i| super::classify_2d::point_in_polygon_2d(probe, &sub_outer_uv[i]))
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

/// Get a point guaranteed inside a sub-face's outer wire (in UV space),
/// not inside any inner wire (hole), then evaluate it to 3D via the surface.
#[allow(clippy::too_many_lines)]
pub fn interior_point_3d(sub_face: &SplitSubFace, frame: Option<&PlaneFrame>) -> Point3 {
    // For a lateral analytic band (cylinder/cone), the section edges' pcurves
    // can evaluate to a different 2pi window than the boundary edges' stored
    // (already-unwrapped) UV — e.g. a rounded-rect corner band split by a
    // faceted ramp, whose staircase arc pcurves land near u=pi while the seam
    // Lines sit near u=3pi. The two windows differ by 2pi, so the assembled UV
    // polygon self-crosses and `point_in_polygon_2d` mislabels the interior
    // sample (it ends up on the wrong side of the section). Unwrapping the
    // sampled points to one continuous u-window first makes the polygon simple
    // again so the centroid/edge-walk interior point is geometrically valid.
    let pts_2d = if matches!(
        &sub_face.surface,
        FaceSurface::Cone(_) | FaceSurface::Cylinder(_)
    ) {
        let (u_period, v_period) = super::pcurve_compute::surface_periods(&sub_face.surface);
        sample_wire_loop_uv_periodic(&sub_face.outer_wire, u_period, v_period)
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
    let mut interior_uv = sample_interior_point(&pts_2d);

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
            if interior_uv.y() < v_min + margin || interior_uv.y() > v_max - margin {
                interior_uv = Point2::new(interior_uv.x(), 0.5 * (v_min + v_max));
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
    if matches!(&sub_face.surface, FaceSurface::Plane { .. }) && !sub_face.inner_wires.is_empty() {
        interior_uv = find_point_outside_holes(&pts_2d, &sub_face.inner_wires, frame);
    } else if is_inside_any_hole(&interior_uv, &sub_face.inner_wires) {
        interior_uv = find_point_outside_holes(&pts_2d, &sub_face.inner_wires, frame);
    }

    // Evaluate back to 3D.
    if let Some(p) = sub_face.surface.evaluate(interior_uv.x(), interior_uv.y()) {
        return p;
    }

    // For plane faces, evaluate via PlaneFrame.
    if let FaceSurface::Plane { normal, .. } = &sub_face.surface {
        if let Some(f) = frame {
            return f.evaluate(interior_uv.x(), interior_uv.y());
        }
        let wire_pts: Vec<Point3> = sub_face.outer_wire.iter().map(|e| e.start_3d).collect();
        let f = PlaneFrame::from_plane_face(*normal, &wire_pts);
        return f.evaluate(interior_uv.x(), interior_uv.y());
    }

    // Last resort: average of 3D endpoints.
    let sum: Point3 = sub_face
        .outer_wire
        .iter()
        .fold(Point3::new(0.0, 0.0, 0.0), |acc, e| {
            acc + (e.start_3d - Point3::new(0.0, 0.0, 0.0))
        });
    let n = sub_face.outer_wire.len() as f64;
    Point3::new(sum.x() / n, sum.y() / n, sum.z() / n)
}

/// Detect section edges (lines, open arcs, and open NURBS conics) forming
/// closed loops strictly inside a plane face's boundary (nested coplanar
/// footprints, or a box wall's socket-profile silhouette), and dedup
/// repeated segments. Both the coplanar-contact pass and adjacent-face plane-plane
/// intersections can emit the same footprint segment, so identical
/// segments (by unordered quantized endpoints) collapse to one.
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
    tol_linear: f64,
) -> Option<Vec<SectionEdge>> {
    use std::collections::{HashMap, HashSet};

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

    let mut seen: HashSet<(QPt, QPt)> = HashSet::new();
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
        if seen.insert(key) {
            deduped.push(s.clone());
        }
    }
    if deduped.len() < 3 {
        return None;
    }

    // The same footprint side can arrive both whole and as sub-segments
    // split at paves. Drop any segment that another section's endpoint
    // subdivides (collinear, strictly interior) — the sub-segments carry
    // the same geometry. If the sub-segments turn out incomplete, the
    // degree check below rejects and the generic path takes over.
    //
    // A subdividing endpoint must lie ON the segment, hence inside its
    // bounding box. Index the endpoints in a coarse grid and probe only the
    // cells the segment spans, so a face with many sections (a perforated
    // panel's cap) costs O(sections) here instead of O(sections²).
    let endpoints: Vec<Point3> = deduped.iter().flat_map(|s| [s.start, s.end]).collect();
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
    let mut ep_grid: HashMap<QPt, Vec<Point3>> = HashMap::new();
    for &p in &endpoints {
        ep_grid.entry(cell_of(p)).or_default().push(p);
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
        let subdivided = |p: Point3| -> bool {
            if (p - s.start).length() < margin || (p - s.end).length() < margin {
                return false;
            }
            let t = (p - s.start).dot(dir) / len2;
            if !(0.0..=1.0).contains(&t) {
                return false;
            }
            let foot = s.start + dir * t;
            (p - foot).length() < margin
        };
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                for cz in lo.2..=hi.2 {
                    if let Some(pts) = ep_grid.get(&(cx, cy, cz)) {
                        for &p in pts {
                            crate::perf::bump_face_split_probe();
                            if subdivided(p) {
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
                    let t = (t1 - t0).mul_add(f64::from(k) / 32.0, t0);
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
                    let disc = qb.mul_add(qb, -4.0 * qa * qc);
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
                    let det = d2.x().mul_add(sd.y(), -(d2.y() * sd.x()));
                    if det.abs() < 1e-14 {
                        continue;
                    }
                    let dx = s2 - a2;
                    let t = dx.x().mul_add(sd.y(), -(dx.y() * sd.x())) / det;
                    let u = dx.x().mul_add(d2.y(), -(dx.y() * d2.x())) / det;
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
        let rescued = plane_internal_line_loops(&sections, &frame, &boundary, &[hole], 1e-7)
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
}
