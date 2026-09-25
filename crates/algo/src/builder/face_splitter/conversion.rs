//! Coordinate/type conversions between 3D, UV, and topology types.

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;

use super::super::pcurve_compute::{
    compute_pcurve_on_surface, project_point_on_surface, sample_edge_to_uv,
};
use super::super::plane_frame::PlaneFrame;
use super::super::split_types::OrientedPCurveEdge;

/// Collect 3D vertex positions from a wire's edges.
pub fn collect_wire_points(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Vec<Point3> {
    let wire = match topo.wire(wire_id) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };
    let mut pts = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge())
            && let Ok(v) = topo.vertex(edge.start())
        {
            pts.push(v.point());
        }
    }
    pts
}

/// Extract the plane normal from a `FaceSurface`, defaulting to +Z.
pub(super) fn extract_plane_normal(surface: &FaceSurface) -> Vec3 {
    if let FaceSurface::Plane { normal, .. } = surface {
        *normal
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

/// Convert a wire's edges to `OrientedPCurveEdge`s on a surface.
pub(super) fn boundary_edges_to_pcurve(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    frame: Option<&PlaneFrame>,
) -> Vec<OrientedPCurveEdge> {
    boundary_edges_to_pcurve_with_images(
        topo,
        wire_id,
        surface,
        wire_pts,
        frame,
        &std::collections::HashMap::new(),
        &[],
        false,
    )
}

/// [`boundary_edges_to_pcurve`] with pave-split edge images applied: a wire
/// edge the pave machinery split (an EF/EE crossing pave on the operand
/// boundary) is expanded into its image pieces, so the face splitter sees
/// the boundary VERTEX at each crossing and can end a partition there.
/// Without the expansion the outer boundary keeps the unsplit original, the
/// face cannot partition at its own exit pave, and its sub-faces mismatch
/// the neighbouring faces' image-split edges along the whole span (the
/// kumiko z~9.9 missing-face root). Line edges only, mirroring
/// `rebuild_face_with_edge_images`.
#[allow(clippy::too_many_arguments)]
pub(super) fn boundary_edges_to_pcurve_with_images<S: std::hash::BuildHasher>(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    frame: Option<&PlaneFrame>,
    edge_images: &std::collections::HashMap<
        brepkit_topology::edge::EdgeId,
        Vec<brepkit_topology::edge::EdgeId>,
        S,
    >,
    anchors: &[Point3],
    expand_lines: bool,
) -> Vec<OrientedPCurveEdge> {
    // Demand-driven: expand an edge only when one of its interior image
    // junctions sits near a section endpoint (an exit pave a chain must
    // anchor to). Every other face keeps its unexpanded boundary and stays
    // byte-identical to the historical behaviour.
    const ANCHOR_BAND: f64 = 3e-3;
    // A junction COINCIDENT with a section endpoint (within the weld band)
    // is already served by the calibrated boundary-splitting machinery —
    // expanding there perturbs partitions that were correct (the
    // divider-lip fuse de-analytics). Only a junction the sections point AT
    // but do not REACH (the operands' own disagreement scale) demands the
    // expansion.
    const WELD_BAND: f64 = 1e-5;
    let wire = match topo.wire(wire_id) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };

    let junction_near_anchor = |imgs: &[brepkit_topology::edge::EdgeId]| -> bool {
        for w in imgs.windows(2) {
            let (Ok(e0), Ok(e1)) = (topo.edge(w[0]), topo.edge(w[1])) else {
                continue;
            };
            for vid in [e0.start(), e0.end()] {
                if (vid == e1.start() || vid == e1.end())
                    && let Ok(v) = topo.vertex(vid)
                {
                    let p = v.point();
                    if anchors.iter().any(|a| {
                        let d = (*a - p).length();
                        d <= ANCHOR_BAND && d > WELD_BAND
                    }) && !anchors.iter().any(|a| (*a - p).length() <= WELD_BAND)
                    {
                        return true;
                    }
                }
            }
        }
        false
    };

    // A NURBS boundary arc does NOT split inside the planar arrangement (its
    // arrangement representation is chord-based), so a section endpoint that
    // COINCIDES with one of its pave junctions still cannot anchor — the
    // opposite of the Line case, where coincident junctions are served by the
    // calibrated boundary-splitting machinery and expansion would perturb it.
    // Expand a NURBS edge only when a junction is weld-COINCIDENT with a
    // section endpoint (the coaxial wedge: EF paves the arcs at the exact
    // crossings and the clip lands the section endpoints on those same
    // points, agreeing to ~1e-9). Marched sections whose endpoints sit in
    // the wider anchor band but off the junction (~1e-3, the snapClip
    // deepened-notch cut) go through their calibrated un-expanded machinery;
    // expanding there broke that fixture's edge pairing.
    // Circle-likeness gate: the expansion serves analytic revolve arcs
    // serialized as NURBS. A marched free-form NURBS boundary (the snapClip
    // deepened-notch walls) has its own calibrated machinery that expansion
    // breaks, and it is never a circle: sample five points, fit the circle
    // through three, and require the rest to sit on it within 1e-6.
    let nurbs_is_circular = |eid: brepkit_topology::edge::EdgeId| -> bool {
        let Ok(edge) = topo.edge(eid) else {
            return false;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            return false;
        };
        let (sp, ep) = (sv.point(), ev.point());
        let pts: Vec<Point3> = [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|&f| super::super::pcurve_compute::evaluate_edge_at_t(edge.curve(), sp, ep, f))
            .collect();
        let (a, b, c) = (pts[0], pts[2], pts[4]);
        let (u, v) = (b - a, c - a);
        let w = u.cross(v);
        let w2 = w.length_squared();
        if w2 < 1e-18 {
            return false;
        }
        let center = a
            + (v.cross(w) * u.length_squared() + w.cross(u) * v.length_squared())
                * (1.0 / (2.0 * w2));
        let r = (a - center).length();
        if r < 1e-9 {
            return false;
        }
        pts.iter()
            .all(|p| ((*p - center).length() - r).abs() <= 1e-6 * r.max(1.0))
    };

    let junction_in_band_nurbs = |imgs: &[brepkit_topology::edge::EdgeId]| -> bool {
        for w in imgs.windows(2) {
            let (Ok(e0), Ok(e1)) = (topo.edge(w[0]), topo.edge(w[1])) else {
                continue;
            };
            for vid in [e0.start(), e0.end()] {
                if (vid == e1.start() || vid == e1.end())
                    && let Ok(v) = topo.vertex(vid)
                {
                    let p = v.point();
                    if anchors.iter().any(|a| (*a - p).length() <= WELD_BAND) {
                        return true;
                    }
                }
            }
        }
        false
    };

    let mut pieces: Vec<(brepkit_topology::edge::EdgeId, bool)> = Vec::new();
    for oe in wire.edges() {
        let curve_kind = topo.edge(oe.edge()).map(|e| e.curve().clone()).ok();
        let is_line = matches!(curve_kind, Some(brepkit_topology::edge::EdgeCurve::Line));
        let is_nurbs = matches!(
            curve_kind,
            Some(brepkit_topology::edge::EdgeCurve::NurbsCurve(_))
        );
        match edge_images.get(&oe.edge()) {
            Some(imgs)
                if imgs.len() > 1
                    && ((is_line && expand_lines && junction_near_anchor(imgs))
                        || (is_nurbs
                            && nurbs_is_circular(oe.edge())
                            && junction_in_band_nurbs(imgs))) =>
            {
                if oe.is_forward() {
                    pieces.extend(imgs.iter().map(|&i| (i, true)));
                } else {
                    pieces.extend(imgs.iter().rev().map(|&i| (i, false)));
                }
            }
            _ => pieces.push((oe.edge(), oe.is_forward())),
        }
    }

    let mut result = Vec::new();
    for (eid, forward) in pieces {
        let edge = match topo.edge(eid) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let start_v = match topo.vertex(if forward { edge.start() } else { edge.end() }) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let end_v = match topo.vertex(if forward { edge.end() } else { edge.start() }) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let start_3d = start_v.point();
        let end_3d = end_v.point();

        let pcurve =
            compute_pcurve_on_surface(edge.curve(), start_3d, end_3d, surface, wire_pts, frame);

        // For closed edges (start_3d approx end_3d, e.g. full circle), projecting
        // start and end to UV gives the same point. Use pcurve sampling to
        // get distinct UV endpoints spanning the full curve.
        let is_closed = (start_3d - end_3d).length() < 1e-10;
        let (start_uv, end_uv) = if is_closed && !matches!(surface, FaceSurface::Plane { .. }) {
            let uv_samples = sample_edge_to_uv(edge.curve(), start_3d, end_3d, surface);
            let su = uv_samples
                .first()
                .copied()
                .unwrap_or_else(|| project_point_on_surface(start_3d, surface, wire_pts, frame));
            let eu = uv_samples
                .last()
                .copied()
                .unwrap_or_else(|| project_point_on_surface(end_3d, surface, wire_pts, frame));
            (su, eu)
        } else {
            (
                project_point_on_surface(start_3d, surface, wire_pts, frame),
                project_point_on_surface(end_3d, surface, wire_pts, frame),
            )
        };

        result.push(OrientedPCurveEdge {
            curve_3d: edge.curve().clone(),
            pcurve,
            start_uv,
            end_uv,
            start_3d,
            end_3d,
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        });
    }
    if frame.is_none() {
        resolve_seam_endpoint_uv(&mut result, surface);
    }
    result
}

/// Unwrap the boundary's u along the wire on a periodic surface, so no
/// consecutive endpoints differ by more than half a period; this also
/// resolves the 0-vs-2π ambiguity of endpoints that sit exactly ON the
/// u-seam.
///
/// `project_point_on_surface` normalizes u into [0, TAU), so a sector face
/// whose window is [3π/2, 2π] (the fourth-quadrant corner cone of a socket
/// pocket) gets its seam-side endpoints projected to u=0 — the wrapped rim
/// arcs' UV chords then cover the COMPLEMENT span [0, 3π/2], the whole UV
/// window is inconsistent, and sections projected inside the true window
/// (u≈5.5) dangle unconnected and are dropped as pendants (the face returns
/// unsplit, leaving the cut's intersection curves unpaired). Each at-seam
/// endpoint takes the seam image (0 or TAU) closest to its reference — wire
/// continuity for a start, the edge's own other endpoint for an end (boundary
/// sector arcs are minor in u, so the closer image is the consistent one;
/// deriving the span from the circle's own parameterization is unreliable
/// because a stored normal opposite the surface axis flips the sign).
/// Every open piece then takes the period copy of each endpoint nearest the
/// running u (start, native-arc midpoint, end), so a rim that passes through
/// the seam at a plain vertex stays continuous; a face whose u is already
/// consistent is a no-op because every endpoint is already the nearest copy.
fn resolve_seam_endpoint_uv(edges: &mut [OrientedPCurveEdge], surface: &FaceSurface) {
    use std::f64::consts::TAU;

    if !matches!(
        surface,
        FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
    ) {
        return;
    }
    let at_seam = |u: f64| -> bool { u.abs() < 1e-9 || (u - TAU).abs() < 1e-9 };
    // A pointed cone's seam runs up to the apex and back, where u is
    // undefined: each apex end takes its ruling's u, and the walk starts on
    // the ruling leaving the apex, so the seam's two copies land a period
    // apart with the rim running between them.
    let apex = match surface {
        FaceSurface::Cone(cn) => Some(cn.apex()),
        _ => None,
    };
    let at_apex = |p: Point3| apex.is_some_and(|a| (p - a).length() < 1e-9);
    let leaves_apex = edges
        .iter()
        .position(|e| at_apex(e.start_3d) && !at_apex(e.end_3d));
    // Walk from an edge anchored off the seam so continuity has a reference.
    let Some(first) = leaves_apex.or_else(|| edges.iter().position(|e| !at_seam(e.start_uv.x())))
    else {
        return;
    };
    let n = edges.len();
    // Every endpoint takes the period copy nearest the running u, not just
    // the ones exactly on the seam: a rim that passes THROUGH the seam at a
    // vertex (a bore wall whose face seam sits a quarter turn from the
    // surface's, after a slot pave splits its rims) has principal-value
    // pieces on the far side that jump back a period, the loop folds, and
    // the arrangement traces one loop with rim arcs repeated. An open piece
    // is walked start, midpoint, end so a span near a half period still
    // lands on the right copy. A closed rim is left alone: its pcurve spans
    // the conventional full period [0, TAU] whatever u its seam vertex
    // projects to, so shifting it "nearest" would move it a period away
    // from the seam lines it shares vertices with.
    let nearest = |u: f64, target: f64| -> f64 { u - ((u - target) / TAU).round() * TAU };
    let mut cur = if leaves_apex.is_some() {
        edges[first].end_uv.x()
    } else {
        edges[first].start_uv.x()
    };
    for k in 0..n {
        let e = &mut edges[(first + k) % n];
        let is_closed = (e.start_3d - e.end_3d).length() < 1e-10;
        if is_closed && !matches!(surface, FaceSurface::Cone(_)) {
            cur = e.end_uv.x();
            continue;
        }
        if is_closed {
            // Its samples start at its vertex and run the curve's own sense:
            // move them by whole periods onto the running u, and leave the
            // walk a turn on in the traversal's sense.
            let shift = nearest(e.start_uv.x(), cur) - e.start_uv.x();
            let native = e.end_uv.x() - e.start_uv.x();
            e.start_uv = Point2::new(e.start_uv.x() + shift, e.start_uv.y());
            e.end_uv = Point2::new(e.end_uv.x() + shift, e.end_uv.y());
            cur = if e.forward {
                e.end_uv.x()
            } else {
                e.start_uv.x() - native
            };
            continue;
        }
        let su = if at_apex(e.start_3d) {
            nearest(e.end_uv.x(), cur)
        } else {
            nearest(e.start_uv.x(), cur)
        };
        // Native orientation: `domain_with_endpoints` takes the positive
        // parametric span from its first point, so a major arc sampled from
        // swapped endpoints (or by the shorter-arc helper) would put the
        // midpoint on the complementary arc.
        let (s3, e3) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
        let mid_3d = e.curve_3d.evaluate_with_endpoints(0.5 * (t0 + t1), s3, e3);
        // At a sphere pole or a cone apex u is undefined (atan2 of zero), so
        // a meridian piece through it hops straight to its end.
        let u_defined = match surface {
            FaceSurface::Sphere(sp) => {
                let d = mid_3d - sp.center();
                let axial = d.dot(sp.z_axis());
                (d - sp.z_axis() * axial).length() > 1e-9 * sp.radius().max(1.0)
            }
            FaceSurface::Cone(cn) => (mid_3d - cn.apex()).length() > 1e-9,
            _ => true,
        };
        let mu = if u_defined {
            surface
                .project_point(mid_3d)
                .map_or(su, |(u, _)| nearest(u, su))
        } else {
            su
        };
        let eu = if at_apex(e.end_3d) {
            su
        } else {
            nearest(e.end_uv.x(), mu)
        };
        if (su - e.start_uv.x()).abs() > 1e-12 {
            e.start_uv = Point2::new(su, e.start_uv.y());
        }
        if (eu - e.end_uv.x()).abs() > 1e-12 {
            e.end_uv = Point2::new(eu, e.end_uv.y());
        }
        cur = eu;
    }
}

/// Check if a 3D point lies on any boundary edge in UV space.
///
/// Projects the point to UV (trying periodic shifts for seam-adjacent
/// points), then checks if the projected UV is within tolerance of any
/// boundary edge's UV segment.
pub(super) fn is_point_on_boundary_uv(
    point: Point3,
    surface: &FaceSurface,
    boundary: &[OrientedPCurveEdge],
    tol: f64,
) -> bool {
    let Some((pu, pv)) = surface.project_point(point) else {
        return false;
    };

    // Circle boundary edges are tested against their true 3D arc first. A
    // boundary arc whose u-span wraps the seam has UV endpoints normalized
    // into [0, TAU), so its UV chord below covers the COMPLEMENT of the actual
    // arc — a point on the wrapped span misses the chord by up to the whole
    // period and the ±TAU candidates cannot recover it. 3D is unambiguous.
    for edge in boundary {
        let brepkit_topology::edge::EdgeCurve::Circle(c) = &edge.curve_3d else {
            continue;
        };
        let foot_t = c.project(point);
        if (c.evaluate(foot_t) - point).length() > c.radius() * tol {
            continue;
        }
        // `domain_with_endpoints` returns the CCW span between its arguments;
        // a reversed-traversal edge covers the CCW span END→START, so orient
        // by the flag or the complement arc is tested instead.
        let (a3, b3) = if edge.forward {
            (edge.start_3d, edge.end_3d)
        } else {
            (edge.end_3d, edge.start_3d)
        };
        let (d0, d1) = edge.curve_3d.domain_with_endpoints(a3, b3);
        let span = (d1 - d0).rem_euclid(std::f64::consts::TAU);
        let span = if span < 1e-12 {
            std::f64::consts::TAU
        } else {
            span
        };
        let off = (foot_t - d0).rem_euclid(std::f64::consts::TAU);
        if off <= span + tol || off >= std::f64::consts::TAU - tol {
            return true;
        }
    }

    // For periodic surfaces, try the original u and u +/- 2pi.
    let u_period = match surface {
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => Some(std::f64::consts::TAU),
        _ => None,
    };
    let u_candidates: Vec<f64> = if let Some(period) = u_period {
        vec![pu, pu - period, pu + period]
    } else {
        vec![pu]
    };

    for &u in &u_candidates {
        let pt_uv = Point2::new(u, pv);
        for edge in boundary {
            // A sphere's arcs were settled in 3D above: the UV chord of one
            // ending at a pole runs to an arbitrary pole `u` and sweeps across
            // the face (a meridian's chord to a pole at `u = 0`).
            if matches!(surface, FaceSurface::Sphere(_))
                && matches!(edge.curve_3d, brepkit_topology::edge::EdgeCurve::Circle(_))
            {
                continue;
            }
            let su = edge.start_uv;
            let eu = edge.end_uv;
            let dx = eu.x() - su.x();
            let dy = eu.y() - su.y();
            let seg_len_sq = dx * dx + dy * dy;

            if seg_len_sq < 1e-20 {
                // Closed edge (circle) -- check v-distance only.
                if (pv - su.y()).abs() < tol {
                    return true;
                }
            } else {
                let t = ((pt_uv.x() - su.x()) * dx + (pt_uv.y() - su.y()) * dy) / seg_len_sq;
                let t = t.clamp(0.0, 1.0);
                let cx = su.x() + t * dx;
                let cy = su.y() + t * dy;
                let dist = ((pt_uv.x() - cx).powi(2) + (pt_uv.y() - cy).powi(2)).sqrt();
                if dist < tol {
                    return true;
                }
            }
        }
    }
    false
}

/// Extract UV endpoints from a pcurve's evaluation rather than independent
/// surface projection. This ensures consistency -- e.g. a pcurve that goes
/// from (pi, v) to (2pi, v) won't have its end snapped to (0, v) by the
/// surface's `project_point` which normalizes u into `[0, 2pi)`.
pub(super) fn uv_endpoints_from_pcurve(
    pcurve: &brepkit_math::curves2d::Curve2D,
    start_3d: Point3,
    end_3d: Point3,
    surface: &FaceSurface,
    wire_pts: &[Point3],
) -> (Point2, Point2) {
    use brepkit_math::curves2d::Curve2D;

    match pcurve {
        Curve2D::Line(line) => {
            // Line2D: start is at t=0. End is estimated by projecting the
            // 3D endpoint and computing the 2D distance along the line.
            let su = line.evaluate(0.0);
            let eu_proj = project_point_on_surface(end_3d, surface, wire_pts, None);
            let du = eu_proj.x() - su.x();
            let dv = eu_proj.y() - su.y();
            let len_2d = (du * du + dv * dv).sqrt();
            let eu = line.evaluate(len_2d);
            // Sanity: if the Line2D evaluation diverges from the projected
            // endpoint by more than pi (half a period), the line direction
            // is wrong -- fall back to direct projection.
            if (eu.x() - eu_proj.x()).abs() > std::f64::consts::PI
                || (eu.y() - eu_proj.y()).abs() > std::f64::consts::PI
            {
                (su, eu_proj)
            } else {
                (su, eu)
            }
        }
        Curve2D::Nurbs(nurbs) => {
            let knots = nurbs.knots();
            if knots.len() >= 2 {
                let t0 = knots[0];
                let tn = knots[knots.len() - 1];
                (nurbs.evaluate(t0), nurbs.evaluate(tn))
            } else {
                (
                    project_point_on_surface(start_3d, surface, wire_pts, None),
                    project_point_on_surface(end_3d, surface, wire_pts, None),
                )
            }
        }
        _ => (
            project_point_on_surface(start_3d, surface, wire_pts, None),
            project_point_on_surface(end_3d, surface, wire_pts, None),
        ),
    }
}
