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
use super::wire_builder::{build_wire_loops, build_wire_loops_with_winding};
use crate::ds::Rank;

use containment::{find_point_outside_holes, is_inside_any_hole};
use conversion::{
    boundary_edges_to_pcurve, extract_plane_normal, is_point_on_boundary_uv,
    uv_endpoints_from_pcurve,
};
use edge_splitting::{
    find_splits_on_circle, find_splits_on_ellipse, find_splits_on_line,
    split_boundary_edges_at_3d_points,
};
use sampling::{sample_wire_loop_uv, sample_wire_loop_uv_periodic};
use special_cases::{
    split_face_with_internal_loops, split_noseam_face_direct, split_periodic_face_into_bands,
    try_split_crossing_plane_face,
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
            EdgeCurve::Circle(circle) => find_splits_on_circle(circle, &edge, &endpoints, tol),
            EdgeCurve::Ellipse(ellipse) => find_splits_on_ellipse(ellipse, &edge, &endpoints, tol),
            // Only endpoints near a line section's bounding box can land on it;
            // the grid query returns exactly that subset, preserving the former
            // full scan's result. A NURBS section (rare here) has no specialized
            // splitter, so the chord-based line search is the closest match.
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => {
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
    // through separately below.
    let hole_segs: Vec<(Point2, Point2, Point3, Point3)> = inner_wires
        .iter()
        .flatten()
        .filter(|e| matches!(e.curve_3d, EdgeCurve::Line))
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
        .filter(|e| !matches!(e.curve_3d, EdgeCurve::Line))
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
        });
    }
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
    // project→evaluate back to itself; an off-plane point does not.
    let on_plane = |p: Point3| -> bool {
        let uv = frame.project(p);
        (frame.evaluate(uv.x(), uv.y()) - p).length() <= tol
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
    let mut vert_pos: std::collections::HashMap<(i64, i64), Point2> =
        std::collections::HashMap::new();
    let register =
        |p: Point2, map: &mut std::collections::HashMap<(i64, i64), Point2>| -> (i64, i64) {
            let k = qkey(p);
            map.entry(k).or_insert(p);
            k
        };

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
            <= tol
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
        // Break parameters along this chord (t in [0,1]).
        let mut ts: Vec<f64> = vec![0.0, 1.0];
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
            // Proper interior crossing. For an arc input, only honour the break
            // when the crossing point is on the real arc (not just its chord).
            if let Some(t) = seg_cross_param(a0, a1, b0, b1)
                && (!i_is_arc || chord_break_on_arc(i, a0 + d * t))
            {
                ts.push(t);
            }
            // Other chord's endpoints landing on this chord's interior
            // (T-junctions where a section merely touches another).
            for bp in [b0, b1] {
                let w = (bp - a0).dot(d) / (len * len);
                if w > 1e-6 && w < 1.0 - 1e-6 {
                    let on = a0 + d * w;
                    if (on - bp).length() < tol && (!i_is_arc || chord_break_on_arc(i, on)) {
                        ts.push(w);
                    }
                }
            }
        }
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|x, y| (*x - *y).abs() < 1e-6);
        // An arc whose chord is split at an interior crossing cannot be emitted
        // as one true arc; bail and let the existing curved paths handle it.
        if inputs[i].is_arc && ts.len() > 2 {
            return None;
        }
        let n_breaks = ts.len();
        for (wi, w) in ts.windows(2).enumerate() {
            let (ta, tb) = (w[0], w[1]);
            if tb - ta < 1e-6 {
                continue;
            }
            let pa = a0 + d * ta;
            let pb = a0 + d * tb;
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
            start_3d: frame.evaluate(su.x(), su.y()),
            end_3d: frame.evaluate(eu.x(), eu.y()),
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
    // UV polygon (chord points) of a traced face, for containment tests.
    let face_poly = |face: &[usize]| -> Vec<Point2> {
        face.iter().map(|&h| vert_pos[&halfs[h].from]).collect()
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
        // `outer` contains `inner` when EVERY one of inner's polygon vertices
        // lies inside (or on) outer's polygon AND outer is strictly larger.
        // A probe-only test is symmetric for the concentric disks that
        // disconnected components produce (both the ±41.75 and ±40.55 disks hold
        // the centre), so it can't order their nesting; the all-vertices +
        // larger-area test is asymmetric and orders them correctly.
        let contains = |outer: usize, inner: usize| -> bool {
            if outer == inner || areas[outer] <= areas[inner] {
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
                find_point_outside_holes(&polys[i], &inner)
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

    let mut result = Vec::new();
    for face in interior {
        let Some(wire) = build_ccw_wire(face) else {
            continue;
        };
        result.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            // Leave None: a region can be non-convex (an L), so the centroid
            // is unsafe. `interior_point_3d` derives a robust interior sample.
            precomputed_interior: None,
        });
    }
    if result.is_empty() {
        return None;
    }
    Some(result)
}

/// Split a face by its section edges, producing sub-faces.
///
/// If there are no section edges, returns a single sub-face covering
/// the entire face (pass-through).
///
/// # Arguments
/// - `topo` -- the topology arena (immutable read)
/// - `face_id` -- the face to split
/// - `sections` -- intersection curves that cut this face (already trimmed)
/// - `rank` -- which solid this face belongs to (A or B)
/// - `tol` -- tolerance (`.linear` for 3D matching, UV tol derived internally)
/// - `frame` -- cached `PlaneFrame` for this face (avoids origin mismatch)
/// - `info` -- cached `SurfaceInfo` for periodicity flags
#[allow(clippy::too_many_lines)]
pub fn split_face_2d(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    rank: Rank,
    tol: &brepkit_math::tolerance::Tolerance,
    frame: Option<&PlaneFrame>,
    info: Option<&SurfaceInfo>,
) -> Vec<SplitSubFace> {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let is_plane = matches!(surface, FaceSurface::Plane { .. });

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

    let mut boundary_edges = if is_plane {
        boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, Some(frame))
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
        filtered_sections = sections
            .iter()
            .filter(|s| {
                let all_in_hole = (0..=HOLE_PROBE_SAMPLES).all(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = i as f64 / HOLE_PROBE_SAMPLES as f64;
                    let p = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, t);
                    to_uv(p).is_some_and(|uv| is_inside_any_hole(&uv, &original_inner_wires))
                });
                !all_in_hole
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
            deduped_line_loops =
                plane_internal_line_loops(sections, frame, &boundary_edges, tol.linear);
            deduped_line_loops.is_some()
        }
    } else {
        // Non-plane faces: check if all section endpoints are off the
        // boundary in UV space.
        let uv_tol = 0.01; // ~0.6 deg in angular coordinates
        sections.iter().all(|s| {
            let start_on_boundary =
                is_point_on_boundary_uv(s.start, &surface, &boundary_edges, uv_tol);
            let end_on_boundary = is_point_on_boundary_uv(s.end, &surface, &boundary_edges, uv_tol);
            !start_on_boundary && !end_on_boundary
        })
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
    let holes_integrated = if is_plane && !original_inner_wires.is_empty() {
        if let Some((extra, passthrough)) = integrate_holes_plane(
            sections,
            &original_inner_wires,
            frame,
            &surface,
            &wire_pts,
            1_000_000,
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

    // Drop pendant section edges that dangle into the face interior — left
    // in, the traversal walks out and back along them, spuriously
    // over-splitting the face (boundary edges are never removed, so the
    // boundary prefix and `n_boundary_edges` stay valid).
    let all_edges = super::wire_builder::remove_pendant_sections(
        &all_edges, tol.linear, u_periodic, v_periodic,
    );

    // Build wire loops via angular-sorting traversal.
    let mut loops = build_wire_loops(&all_edges, tol.linear, u_periodic, v_periodic);

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
    if holes_integrated
        && is_plane
        && original_inner_wires.len() >= 2
        && !woven_inner_wires.is_empty()
        && let Some(mut result) = arrangement_regions_from_combined(
            &surface,
            &all_edges,
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
        if !passthrough_inner_wires.is_empty() {
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
        && let Some(result) = split_plane_face_by_arrangement(
            &surface, boundary, sections, rank, reversed, face_id, frame, tol.linear,
        )
        && (result.len() > loops.len()
            || wire_loops_self_cross(&loops, tol.linear)
            || greedy_outer_loops_nested(&loops, cw_loops))
    {
        return result;
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
            let pts = sample_wire_loop_uv_periodic(&wire_loop, u_per_opt, v_per_opt);
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

    // If all loops are CW (negative area), the winding is reversed.
    if !use_structural_classification && outers.is_empty() && !holes.is_empty() {
        for hole in &mut holes {
            hole.reverse();
            for edge in hole.iter_mut() {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        }
        let pts: Vec<Point2> = holes[0].iter().map(|e| e.start_uv).collect();
        let area = signed_area_2d(&pts);
        outers.push((holes.remove(0), area));
    }

    // A negative-area loop is only a true hole if it is geometrically NESTED
    // inside an outer loop. When a plane face is split by a single section line
    // into two side-by-side regions, the wire builder can hand back the second
    // region wound CW (negative area) even though it is ADJACENT, not nested
    // (e.g. the above-vs-below halves of a notch-straddle tool face split at the
    // wall-top line). Detect that by probing a point strictly inside the
    // candidate hole: if it lies in no outer's interior, the loop is a separate
    // region — reverse it to CCW and promote it to an outer. A genuinely nested
    // hole's probe lies inside its containing outer, so it is left alone.
    if !use_structural_classification && !outers.is_empty() && !holes.is_empty() {
        let outer_uv: Vec<Vec<Point2>> =
            outers.iter().map(|(w, _)| sample_wire_loop_uv(w)).collect();
        let mut promoted: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
        holes.retain(|hole| {
            let hole_pts = sample_wire_loop_uv(hole);
            if hole_pts.len() < 3 {
                return true;
            }
            let probe = super::classify_2d::sample_interior_point(&hole_pts);
            let nested = outer_uv
                .iter()
                .any(|o| super::classify_2d::point_in_polygon_2d(probe, o));
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

    let mut sub_faces = Vec::new();
    for (outer_wire, _area) in outers {
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
    // first vertex (via 2D point-in-polygon). Uses sampled UV points for
    // accurate containment with curved outer wires.
    for hole in holes {
        if let Some(first_pt) = hole.first().map(|e| e.start_uv) {
            let mut assigned = false;
            for sf in &mut sub_faces {
                let outer_pts = sample_wire_loop_uv(&sf.outer_wire);
                if super::classify_2d::point_in_polygon_2d(first_pt, &outer_pts) {
                    sf.inner_wires.push(hole.clone());
                    assigned = true;
                    break;
                }
            }
            if !assigned && let Some(sf) = sub_faces.first_mut() {
                sf.inner_wires.push(hole);
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
        interior_uv = find_point_outside_holes(&pts_2d, &sub_face.inner_wires);
    } else if is_inside_any_hole(&interior_uv, &sub_face.inner_wires) {
        interior_uv = find_point_outside_holes(&pts_2d, &sub_face.inner_wires);
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

/// Detect all-Line section edges forming closed loops strictly inside a
/// plane face's boundary (nested coplanar footprints), and dedup repeated
/// segments. Both the coplanar-contact pass and adjacent-face plane-plane
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
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => (s.start - s.end).length() > tol_linear,
            EdgeCurve::NurbsCurve(_) => false,
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
