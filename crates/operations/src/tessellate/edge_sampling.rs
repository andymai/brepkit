//! Edge curve sampling and parametrization.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;

/// Combined linear+angular segment count for a circular arc.
///
/// Delegates to [`brepkit_math::chord::segments_for_chord_deviation_with_angle`]
/// with no minimum-edge-length clamp. `apply_curvature_floor` is forwarded:
/// constant-curvature circles pass `false` (the chord formula is exact),
/// variable/doubly-curved geometry passes `true`.
pub(super) fn segments_for_chord_deviation_a(
    radius: f64,
    arc_range: f64,
    deflection: f64,
    angular_tol: f64,
    apply_curvature_floor: bool,
) -> usize {
    brepkit_math::chord::segments_for_chord_deviation_with_angle(
        radius,
        arc_range,
        deflection,
        angular_tol,
        0.0,
        apply_curvature_floor,
    )
}

/// The segments every mesher samples an arc of a circle of `radius` over
/// `arc_range` of its parameter in. `circle_floor` keeps the curvature
/// floor (see [`edge_sample_count`]).
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the arc would take
/// more samples than the tessellator allows.
pub(super) fn circle_segments(
    radius: f64,
    arc_range: f64,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
) -> Result<usize, crate::OperationsError> {
    let n =
        segments_for_chord_deviation_a(radius, arc_range, deflection, angular_tol, circle_floor);
    super::check_count("a circle edge", n.saturating_add(1), deflection)?;
    Ok(n)
}

/// The segments every mesher samples an arc of `ellipse` over `arc_range`
/// of its parameter in: as many as the deflection needs.
///
/// Density is driven by the largest radius of curvature (a^2/b, at the
/// minor-axis ends). Under uniform-parameter sampling the per-segment chord
/// deviation is set by how far the parameter sweeps in arc length, which
/// peaks where curvature is lowest; the small-radius criterion (b^2/a)
/// satisfies pointwise sag but lets the integrated (area/volume) error grow
/// about 15 times. Using a^2/b keeps both bounded.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the arc would take
/// more samples than the tessellator allows.
pub(super) fn ellipse_segments(
    ellipse: &brepkit_math::curves::Ellipse3D,
    arc_range: f64,
    deflection: f64,
    angular_tol: f64,
) -> Result<usize, crate::OperationsError> {
    let (a, b) = (ellipse.semi_major(), ellipse.semi_minor());
    let n = segments_for_chord_deviation_a(a * a / b, arc_range, deflection, angular_tol, true);
    super::check_count("an ellipse edge", n.saturating_add(1), deflection)?;
    Ok(n)
}

/// Compute orthogonal axes for a plane given its normal.
///
/// Falls back to identity axes if the normal is degenerate (should not
/// happen for valid face data).
pub(super) fn plane_axes(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = normal
        .cross(up)
        .normalize()
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let v_axis = normal
        .cross(u_axis)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
    (u_axis, v_axis)
}

/// Compute the number of sample points for an edge based on deflection.
///
/// Uses edge length and curvature to determine sampling density.
///
/// `circle_floor` selects whether a circular edge keeps the curvature floor.
/// Display callers pass `false` (the chord count is exact for a constant-
/// curvature circle); the boolean mesh-fallback passes `true` because its
/// co-refinement robustness depends on the denser floored sampling.
pub(super) fn edge_sample_count(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
) -> Result<usize, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    Ok(match edge.curve() {
        EdgeCurve::Line => 2,
        EdgeCurve::Circle(c) => {
            // Use the same segments_for_chord_deviation formula that
            // tessellate_analytic uses for the grid density. This ensures
            // edge sample points align with the analytic grid boundary,
            // allowing the snap path to achieve watertight stitching.
            let arc_range = circle_param_range(topo, edge, c)
                .map_or(std::f64::consts::TAU, |(t_start, t_end)| {
                    (t_end - t_start).abs()
                });
            circle_segments(c.radius(), arc_range, deflection, angular_tol, circle_floor)? + 1
        }
        EdgeCurve::Ellipse(ellipse) => {
            let arc_range = if edge.is_closed() {
                std::f64::consts::TAU
            } else if let (Ok(sp), Ok(ep)) = (
                topo.vertex(edge.start())
                    .map(brepkit_topology::vertex::Vertex::point),
                topo.vertex(edge.end())
                    .map(brepkit_topology::vertex::Vertex::point),
            ) {
                let ts = ellipse.project(sp);
                let mut te = ellipse.project(ep);
                if te <= ts {
                    te += std::f64::consts::TAU;
                }
                te - ts
            } else {
                std::f64::consts::TAU
            };
            ellipse_segments(ellipse, arc_range, deflection, angular_tol)?
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            // Endpoint-trimmed convention: a section edge can be a validated
            // sub-span of its stored curve; measuring the FULL knot domain
            // would size (and later sample) the whole parent curve.
            let (u0, u1) = match (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                (Ok(sv), Ok(ev)) => edge.curve().domain_with_endpoints(sv.point(), ev.point()),
                _ => nurbs.domain(),
            };
            nurbs_edge_breaks(nurbs, u0, u1, deflection, angular_tol)?.len() - 1
        }
    })
}

/// The parameters at which a NURBS edge from `t0` to `t1` is sampled, both
/// ends included: a span is cut until its curve stays within `deflection`
/// of its chord and turns by no more than half of `angular_tol` (`0.0` for no
/// limit). A straight run takes one span and a tight turn takes many, so the
/// count follows the curve and not its tightest place.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the edge would take
/// more samples than the tessellator allows.
pub(super) fn nurbs_edge_breaks(
    stored: &brepkit_math::nurbs::curve::NurbsCurve,
    t0: f64,
    t1: f64,
    deflection: f64,
    angular_tol: f64,
) -> Result<Vec<f64>, crate::OperationsError> {
    // A span holding more knots than this is halved unread: its probes could
    // step over a polynomial piece.
    const MAX_KNOTS: usize = 8;
    const MIN_TURN_LENGTH: f64 = 0.1;
    // The rounding of one coordinate of a point read off a chord, in units
    // of that coordinate's relative precision at its largest size there:
    // the error of the chord's evaluated ends and of the offset itself.
    const ROUNDING: f64 = 4.0;

    // The curve is read from its first control point: the rounding of what
    // is read then follows the curve's own size, not its distance from the
    // origin. The parameters, which are what this gives, are the same.
    let first_point = stored.control_points()[0];
    let shifted = brepkit_math::nurbs::curve::NurbsCurve::new(
        stored.degree(),
        stored.knots().to_vec(),
        stored
            .control_points()
            .iter()
            .map(|p| {
                Point3::new(
                    p.x() - first_point.x(),
                    p.y() - first_point.y(),
                    p.z() - first_point.z(),
                )
            })
            .collect(),
        stored.weights().to_vec(),
    )
    .ok();
    let nurbs = shifted.as_ref().unwrap_or(stored);

    let (lo, hi) = (t0.min(t1), t0.max(t1));
    let mut knots: Vec<f64> = nurbs
        .knots()
        .iter()
        .copied()
        .filter(|&k| k > lo && k < hi)
        .collect();
    knots.dedup();
    let turn_limit = 0.5 * angular_tol;
    // The offset of `p` from the chord's segment `a b`.
    let off_chord_by = |p: Point3, a: Point3, b: Point3| {
        let chord = b - a;
        let len2 = chord.dot(chord);
        let f = if len2 > 0.0 {
            ((p - a).dot(chord) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        p - (a + chord * f)
    };
    let off_chord = |p: Point3, a: Point3, b: Point3| off_chord_by(p, a, b).length();
    // While its weights are positive, the curve over a span lies in the hull
    // of the control points whose basis functions reach into the span: no
    // point of it is farther from the chord than the farthest of them. The
    // span is straight as stored when each coordinate of each of their
    // offsets is within the rounding of that coordinate, at its largest
    // size among the stored control points there: a coordinate far from
    // zero cannot hide another's offset.
    let hull_off_chord = |span_lo: f64, span_hi: f64, pa: Point3, pb: Point3| {
        let (all_knots, controls) = (nurbs.knots(), nurbs.control_points());
        let (degree, last) = (nurbs.degree(), controls.len() - 1);
        let span_of = |at: usize| at.saturating_sub(1).clamp(degree, last);
        let first = span_of(all_knots.partition_point(|&k| k <= span_lo)) - degree;
        let end = span_of(all_knots.partition_point(|&k| k < span_hi));
        let mut scale = [0.0_f64; 3];
        for p in &stored.control_points()[first..=end] {
            for (s, c) in scale.iter_mut().zip([p.x(), p.y(), p.z()]) {
                *s = s.max(c.abs());
            }
        }
        let mut far = 0.0_f64;
        let mut straight = true;
        for i in first..=end {
            let w = nurbs.weights()[i];
            if !(w > 0.0 && w.is_finite()) {
                return (f64::INFINITY, false);
            }
            let off = off_chord_by(controls[i], pa, pb);
            far = far.max(off.length());
            straight &= [off.x(), off.y(), off.z()]
                .iter()
                .zip(scale)
                .all(|(d, s)| d.abs() <= ROUNDING * f64::EPSILON * s);
        }
        (far, straight)
    };
    let parts = |a: f64, b: f64| {
        let (span_lo, span_hi) = (a.min(b), a.max(b));
        let from = knots.partition_point(|&k| k <= span_lo);
        let to = knots.partition_point(|&k| k < span_hi).max(from);
        let (pa, pb) = (nurbs.evaluate(a), nurbs.evaluate(b));
        // A closed curve has no chord: however small, it takes three spans
        // to enclose anything. A part of the edge whose ends meet is a piece
        // where the curve stands still, not a loop: cut as a closed curve,
        // it would be cut in three at every level without end.
        let whole = a.to_bits() == t0.to_bits() && b.to_bits() == t1.to_bits();
        let closed = whole && (pb - pa).length() <= 1e-9;
        // A span whose control points lie within rounding of its chord is
        // straight as stored: what its probes read off the chord, and the
        // turn of its tangents, is rounding. It takes one span, as no cut
        // can bring its samples closer to the curve than their rounding. A
        // closed edge still takes its three.
        let (hull, straight) = hull_off_chord(span_lo, span_hi, pa, pb);
        if straight && !closed {
            return 1;
        }
        if to - from > MAX_KNOTS {
            return if closed { 3 } else { 2 };
        }
        let mut probes = super::nurbs::span_probes(a, b, &knots[from..to]);
        probes.sort_by(f64::total_cmp);
        let points: Vec<Point3> = probes.iter().map(|&t| nurbs.evaluate(t)).collect();
        // The probes read the sag from below and the hull bounds it from
        // above: more than the bound is rounding in the probes.
        let sag = points
            .iter()
            .map(|&p| off_chord(p, pa, pb))
            .fold(0.0, f64::max)
            .min(hull);
        // The probes run up the parameter, so from the span's lower end.
        let (first, last) = if a <= b { (pa, pb) } else { (pb, pa) };
        let length = std::iter::once(first)
            .chain(points.iter().copied())
            .chain(std::iter::once(last))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum::<f64>();
        if closed && sag > 0.0 {
            return super::nurbs::parts_needed(sag / deflection, 0.0).max(3);
        }
        let mut turned = 0.0;
        // A span shorter than a tenth of the deflection is not cut for its
        // turn: a sharp bend cut that fine gives samples the face's (u, v)
        // cannot tell apart.
        if turn_limit > 0.0 && length > MIN_TURN_LENGTH * deflection {
            // Up the parameter; at the span's upper end the tangent is read
            // just below it, on the span's own piece: a knot there can turn
            // the curve's tangent, as a degree-one spline's does.
            let width = span_hi - span_lo;
            let mut last = nurbs.tangent(span_lo).ok();
            for t in [
                span_lo + 0.25 * width,
                span_lo + 0.5 * width,
                span_lo + 0.75 * width,
                span_hi.next_down(),
            ] {
                let next = nurbs.tangent(t.max(span_lo)).ok();
                if let (Some(p), Some(q)) = (last, next) {
                    turned += p.dot(q).clamp(-1.0, 1.0).acos();
                }
                last = next.or(last);
            }
            turned /= turn_limit;
        }
        super::nurbs::parts_needed(sag / deflection, turned)
    };

    // The edge is broken first at each knot where its tangent turns (a knot
    // as deep as the degree can turn it), so no span holds a break in its
    // tangent: a degree-one spline is lines between its corners. Then level
    // by level, so every part of the span is read alike at each level.
    let turns_at = |k: f64| match (nurbs.tangent(k.next_down()), nurbs.tangent(k)) {
        (Ok(before), Ok(after)) => before.dot(after) < 1.0 - 1e-12,
        _ => true,
    };
    let mut corners: Vec<f64> = Vec::new();
    for run in nurbs.knots().chunk_by(|x, y| x.to_bits() == y.to_bits()) {
        if run.len() >= nurbs.degree() && run[0] > lo && run[0] < hi && turns_at(run[0]) {
            corners.push(run[0]);
            // The edge takes a sample at each corner and at its two ends.
            super::check_count("a NURBS edge", corners.len() + 2, deflection)?;
        }
    }
    if t0 > t1 {
        corners.reverse();
    }
    let mut spans: Vec<(f64, f64, bool)> = std::iter::once(t0)
        .chain(corners.iter().copied())
        .zip(corners.iter().copied().chain(std::iter::once(t1)))
        .map(|(a, b)| (a, b, false))
        .collect();
    // A closed edge broken at one corner has two spans between the same two
    // points, and two samples enclose nothing: the longer span is halved, so
    // the edge keeps the three spans a closed curve needs. With no corner,
    // the whole edge is cut in three or more as a closed curve.
    let closed_edge = (nurbs.evaluate(t1) - nurbs.evaluate(t0)).length() <= 1e-9;
    if closed_edge && spans.len() == 2 {
        let longer = usize::from((spans[1].1 - spans[1].0).abs() > (spans[0].1 - spans[0].0).abs());
        let (a, b, _) = spans[longer];
        let mid = a + (b - a) * 0.5;
        if mid.to_bits() == a.to_bits() || mid.to_bits() == b.to_bits() {
            return Err(super::endless_samples("a NURBS edge", deflection));
        }
        spans.splice(longer..=longer, [(a, mid, false), (mid, b, false)]);
    }
    loop {
        let mut next = Vec::with_capacity(spans.len() * 2);
        let mut cut = false;
        for &(a, b, held) in &spans {
            let n = if held { 1 } else { parts(a, b) };
            if n == 1 {
                next.push((a, b, true));
            } else {
                cut = true;
                let mid = a + (b - a) * 0.5;
                if mid.to_bits() == a.to_bits() || mid.to_bits() == b.to_bits() {
                    return Err(super::endless_samples("a NURBS edge", deflection));
                }
                for part in 0..n {
                    #[allow(clippy::cast_precision_loss)]
                    let (from, to) = (
                        if part == 0 {
                            a
                        } else {
                            a + (b - a) * part as f64 / n as f64
                        },
                        if part + 1 == n {
                            b
                        } else {
                            a + (b - a) * (part + 1) as f64 / n as f64
                        },
                    );
                    next.push((from, to, false));
                }
            }
            // Every span is counted as it is kept, cut or not: a straight
            // span adds a sample as a cut part does.
            super::check_count("a NURBS edge", next.len() + 1, deflection)?;
        }
        spans = next;
        if !cut {
            break;
        }
    }
    // A curved edge keeps its middle however shallow it is: a face between
    // it and a straight edge, or between two such arcs, has three boundary
    // points to mesh between.
    if let [(a, b, _)] = spans[..] {
        let mid = 0.5 * (a + b);
        let (pa, pb) = (nurbs.evaluate(a), nurbs.evaluate(b));
        if off_chord(nurbs.evaluate(mid), pa, pb) > 1e-9 * (1.0 + (pb - pa).length()) {
            return Ok(vec![a, mid, b]);
        }
    }
    // A span whose curve barely moves over part of it (its control points
    // bunched) is cut by parameter into pieces a hair long, which a face's
    // (u, v) cannot tell apart; a break that close to the last one kept is
    // dropped, moving the polyline no farther than that.
    let prune = |gap: f64| {
        let mut kept: Vec<(f64, Point3)> = vec![(t0, nurbs.evaluate(t0))];
        for &(_, b, _) in &spans {
            let p = nurbs.evaluate(b);
            if (p - kept[kept.len() - 1].1).length() > gap {
                kept.push((b, p));
            } else if b.to_bits() == t1.to_bits() && kept.len() > 1 {
                // The edge's own end stays; the break before it goes.
                let last = kept.len() - 1;
                kept[last] = (b, p);
            } else if b.to_bits() == t1.to_bits() {
                kept.push((b, p));
            }
        }
        kept.into_iter().map(|(t, _)| t).collect::<Vec<f64>>()
    };
    let kept = prune(1e-3 * deflection);
    // A closed curve smaller than that gap would lose the three spans it
    // needs to enclose anything: it keeps every break that moves at all.
    if kept.len() < 4 && closed_edge {
        return Ok(prune(0.0));
    }
    Ok(kept)
}

/// Get the parameter range for a circle edge.
///
/// # Errors
///
/// Returns an error if vertex lookup fails.
pub(super) fn circle_param_range(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    circle: &brepkit_math::curves::Circle3D,
) -> Result<(f64, f64), crate::OperationsError> {
    if edge.is_closed() {
        Ok((0.0, std::f64::consts::TAU))
    } else {
        let sp = topo.vertex(edge.start())?.point();
        let ep = topo.vertex(edge.end())?.point();
        let ts = circle.project(sp);
        let mut te = circle.project(ep);
        if te <= ts {
            te += std::f64::consts::TAU;
        }
        Ok((ts, te))
    }
}

/// The samples of each straight edge of a twisted bilinear NURBS face, as
/// fine as the grid its twist needs across the face's own parameter box: an
/// edge is cut into as many pieces as the grid's cells its parameters cross.
/// Empty for a face that needs no such grid.
///
/// # Errors
///
/// Returns an error if a wire, edge or vertex lookup fails.
pub(super) fn twisted_straight_edge_samples(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
) -> Result<Vec<(usize, Vec<Point3>)>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;

    let surface = face.surface();
    let FaceSurface::Nurbs(nurbs) = surface else {
        return Ok(Vec::new());
    };
    if nurbs.degree_u() != 1 || nurbs.degree_v() != 1 {
        return Ok(Vec::new());
    }
    let (mut u_lo, mut u_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut lines = Vec::new();
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let uvs: Vec<(f64, f64)> =
                sample_edge(topo, edge, deflection, angular_tol, circle_floor)?
                    .into_iter()
                    .filter_map(|p| surface.project_point(p))
                    .collect();
            for &(u, v) in &uvs {
                (u_lo, u_hi) = (u_lo.min(u), u_hi.max(u));
                (v_lo, v_hi) = (v_lo.min(v), v_hi.max(v));
            }
            if matches!(edge.curve(), EdgeCurve::Line)
                && let (Some(&start_uv), Some(&end_uv)) = (uvs.first(), uvs.last())
            {
                let ends = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                lines.push((oe.edge().index(), ends, start_uv, end_uv));
            }
        }
    }
    let (du, dv) = (u_hi - u_lo, v_hi - v_lo);
    if !(du > 1e-15 && dv > 1e-15) {
        return Ok(Vec::new());
    }
    let at = |u: f64, v: f64| nurbs.evaluate(u, v);
    // The grid only counts samples along straight edges: a line has no sag
    // or turn along it for a chord test to read.
    let (n_u, n_v) = super::nurbs::untwisted_grid(
        &at,
        ((u_lo, u_hi), (v_lo, v_hi)),
        (1, 1),
        deflection,
        super::nonplanar::NURBS_MAX_DIVISIONS,
        None,
    );
    if n_u.max(n_v) <= 1 {
        return Ok(Vec::new());
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let samples = lines
        .into_iter()
        .map(|(idx, (a, b), (ua, va), (ub, vb))| {
            let cells = ((ub - ua).abs() / du * n_u as f64).max((vb - va).abs() / dv * n_v as f64);
            // A whole number of cells, read past its rounding.
            let pieces = ((cells * (1.0 - 1e-9)).ceil() as usize).max(1);
            let points = (0..=pieces)
                .map(|k| a + (b - a) * (k as f64 / pieces as f64))
                .collect();
            (idx, points)
        })
        .collect();
    Ok(samples)
}

/// Sample an edge curve to produce a list of 3D points (start to end).
///
/// The sampling density is driven by `deflection`. For a `Line`, only the
/// two endpoints are returned. For curves, the point count is proportional
/// to curvature. `circle_floor` is forwarded to [`edge_sample_count`].
///
/// # Errors
///
/// Returns an error if vertex lookup fails for edge endpoints.
pub(super) fn sample_edge(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
) -> Result<Vec<Point3>, crate::OperationsError> {
    sample_edge_with_params(topo, edge, deflection, angular_tol, circle_floor)
        .map(|(points, _)| points)
}

/// [`sample_edge`], with a NURBS edge's sample parameters (`None` for the
/// other curves).
///
/// # Errors
///
/// Returns an error if vertex lookup fails for edge endpoints.
pub(super) fn sample_edge_with_params(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
) -> Result<(Vec<Point3>, Option<Vec<f64>>), crate::OperationsError> {
    use brepkit_geometry::sampling::sample_uniform;
    use brepkit_topology::edge::EdgeCurve;

    let n = if matches!(edge.curve(), EdgeCurve::NurbsCurve(_)) {
        0
    } else {
        edge_sample_count(topo, edge, deflection, angular_tol, circle_floor)?
    };

    let mut nurbs_params = None;
    let points = match edge.curve() {
        EdgeCurve::Line => {
            vec![
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            ]
        }
        EdgeCurve::Circle(circle) => {
            let (t_start, t_end) = circle_param_range(topo, edge, circle)?;
            sample_uniform(circle, t_start, t_end, n)
        }
        EdgeCurve::Ellipse(ellipse) => {
            let (t_start, t_end) = if edge.is_closed() {
                (0.0, std::f64::consts::TAU)
            } else {
                let sp = topo.vertex(edge.start())?.point();
                let ep = topo.vertex(edge.end())?.point();
                let ts = ellipse.project(sp);
                let mut te = ellipse.project(ep);
                if te <= ts {
                    te += std::f64::consts::TAU;
                }
                (ts, te)
            };
            sample_uniform(ellipse, t_start, t_end, n)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            // Endpoint-trimmed convention: a validated sub-span samples only
            // the edge's own piece of the stored curve (already start→end);
            // sampling the full knot domain traces the whole parent section
            // curve and rips a crack along the un-shared part.
            let params = nurbs_edge_sample_params(topo, edge, nurbs, deflection, angular_tol)?;
            let points = params.iter().map(|&t| nurbs.evaluate(t)).collect();
            nurbs_params = Some(params);
            points
        }
    };

    // A fitted curve's evaluated end can miss its vertex by the fit error (a
    // marched section's end sits ~1e-8 off the pave, a fillet's contact curve
    // 1.4e-6 off its corner), enough to land in another merge cell than the
    // neighbouring edge's end and leave a sliver an exporter collapses: its
    // ends are the vertices every edge meeting there shares. A file's vertex
    // can sit off any of its curves, circles too, by the file's tolerance
    // (6e-4 on a part read from STEP), and an end left unsnapped there broke
    // the face's outline at one deflection and not the next: ends within a
    // file's tolerance snap too, never by more than the deflection. The
    // samples already run start to end; a closed circle's start is its
    // frame's, not its vertex.
    if matches!(edge.curve(), EdgeCurve::Line)
        || (edge.is_closed() && !matches!(edge.curve(), EdgeCurve::NurbsCurve(_)))
    {
        return Ok((points, nurbs_params));
    }
    let snap = (1e-2 * deflection).max(deflection.min(1e-3)).max(1e-6);
    let mut points = points;
    let (start, end) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    if let Some(first) = points.first_mut()
        && (*first - start).length() < snap
    {
        *first = start;
    }
    if let Some(last) = points.last_mut()
        && (*last - end).length() < snap
    {
        *last = end;
    }

    Ok((points, nurbs_params))
}

/// The curve parameters of a NURBS edge's samples, in the order
/// [`sample_edge`] returns its points: from its start vertex to its end.
pub(super) fn nurbs_edge_sample_params(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    nurbs: &brepkit_math::nurbs::curve::NurbsCurve,
    deflection: f64,
    angular_tol: f64,
) -> Result<Vec<f64>, crate::OperationsError> {
    let sp = topo.vertex(edge.start())?.point();
    let ep = topo.vertex(edge.end())?.point();
    let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
    let (u0, u1) = nurbs.domain();
    let is_subspan = (t0 - u0).abs() > 1e-12 || (t1 - u1).abs() > 1e-12;
    let mut params = nurbs_edge_breaks(nurbs, t0, t1, deflection, angular_tol)?;
    // Normalize to edge (start→end vertex) order so every consumer's
    // `is_forward` walk holds even for section edges whose stored
    // curve runs end→start. Sub-spans are already endpoint-ordered.
    if !is_subspan && nurbs_runs_end_to_start(topo, edge, nurbs)? {
        params.reverse();
    }
    Ok(params)
}

/// Whether an open NURBS edge's stored curve runs from the edge's END vertex
/// back to its START vertex. GFA section edges can store traversal-order
/// vertices over an unreversed curve, so a sampler that walks the knot domain
/// trusting `oe.is_forward()` alone folds the boundary polyline back on
/// itself (a double-covered strip along the shared section curve).
pub(super) fn nurbs_runs_end_to_start(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    nurbs: &brepkit_math::nurbs::curve::NurbsCurve,
) -> Result<bool, crate::OperationsError> {
    if edge.start() == edge.end() {
        return Ok(false);
    }
    let s = topo.vertex(edge.start())?.point();
    let e = topo.vertex(edge.end())?.point();
    let (u0, u1) = nurbs.domain();
    let p0 = nurbs.evaluate(u0);
    let p1 = nurbs.evaluate(u1);
    let aligned = (p0 - s).length() + (p1 - e).length();
    let flipped = (p0 - e).length() + (p1 - s).length();
    Ok(flipped < aligned)
}

/// Sample a wire into a list of 3D positions, skipping consecutive duplicates.
pub(super) fn sample_wire_positions(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
    tol: f64,
    deflection: f64,
    angular_tol: f64,
) -> Result<Vec<Point3>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    let mut positions = Vec::new();

    let sample_curve_into = |evaluate: &dyn Fn(f64) -> Point3,
                             t_for_index: &dyn Fn(usize) -> f64,
                             n_samples: usize,
                             forward: bool,
                             positions: &mut Vec<Point3>| {
        let indices: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..n_samples)
        } else {
            // Reversed traversal walks t_end -> t_start; the [traversal
            // start, traversal end) convention therefore needs indices
            // n..=1, not (0..n).rev(): excluding t_end here dropped the
            // junction vertex with the PREVIOUS edge (nobody else supplies
            // it), and the CDT outline then shortcut the polygon corner
            // with a chord whose area bite scales with the neighbour
            // edge's length. t_start is excluded instead - the next edge
            // supplies it, same as the forward case.
            Box::new((1..=n_samples).rev())
        };
        for i in indices {
            #[allow(clippy::cast_precision_loss)]
            let t = t_for_index(i);
            let pt = evaluate(t);
            if positions
                .last()
                .is_none_or(|p: &Point3| (*p - pt).length() > tol)
            {
                positions.push(pt);
            }
        }
    };

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        match edge.curve() {
            EdgeCurve::Circle(circle) => {
                let (t_start, t_end) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    edge.curve().domain_with_endpoints(sp, ep)
                };
                let n_samples = circle_segments(
                    circle.radius(),
                    (t_end - t_start).abs(),
                    deflection,
                    angular_tol,
                    false,
                )?;
                #[allow(clippy::cast_precision_loss)]
                sample_curve_into(
                    &|t| circle.evaluate(t),
                    &|i| t_start + (t_end - t_start) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
                    &mut positions,
                );
            }
            EdgeCurve::Ellipse(ellipse) => {
                let (t_start, t_end) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    let ts = ellipse.project(sp);
                    let mut te = ellipse.project(ep);
                    if te <= ts {
                        te += std::f64::consts::TAU;
                    }
                    (ts, te)
                };
                let n_samples =
                    ellipse_segments(ellipse, t_end - t_start, deflection, angular_tol)?;
                #[allow(clippy::cast_precision_loss)]
                sample_curve_into(
                    &|t| ellipse.evaluate(t),
                    &|i| t_start + (t_end - t_start) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
                    &mut positions,
                );
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                // Endpoint-trimmed convention: sample only the edge's own
                // sub-span of the stored curve (see `sample_edge`).
                let sp = topo.vertex(edge.start())?.point();
                let ep = topo.vertex(edge.end())?.point();
                let (u0, u1) = edge.curve().domain_with_endpoints(sp, ep);
                let full = nurbs.domain();
                let is_subspan = (u0 - full.0).abs() > 1e-12 || (u1 - full.1).abs() > 1e-12;
                let breaks = nurbs_edge_breaks(nurbs, u0, u1, deflection, angular_tol)?;
                let n_samples = breaks.len() - 1;
                let forward = if is_subspan {
                    // Sub-spans are already endpoint-ordered start→end.
                    oe.is_forward()
                } else {
                    oe.is_forward() != nurbs_runs_end_to_start(topo, edge, nurbs)?
                };
                #[allow(clippy::cast_precision_loss)]
                sample_curve_into(
                    &|t| nurbs.evaluate(t),
                    &|i| breaks[i],
                    n_samples,
                    forward,
                    &mut positions,
                );
            }
            EdgeCurve::Line => {
                let vid = if oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                let pt = topo.vertex(vid)?.point();
                if positions
                    .last()
                    .is_none_or(|p: &Point3| (*p - pt).length() > tol)
                {
                    positions.push(pt);
                }
            }
        }
    }

    if positions.len() > 2
        && let (Some(first), Some(last)) = (positions.first(), positions.last())
        && (*last - *first).length() < tol
    {
        positions.pop();
    }

    Ok(positions)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;

    /// A degree-one spline is lines between its knots, each with a tangent
    /// that breaks at the knot. Asked for a deflection of 1e-18, it is broken
    /// at its knot into two lines that need no cutting, not cut toward the
    /// corner until the parameters stop moving.
    #[test]
    fn a_degree_one_spline_breaks_at_its_knot_and_nowhere_else() {
        let curve = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            vec![1.0; 3],
        )
        .unwrap();
        let started = std::time::Instant::now();
        let breaks = nurbs_edge_breaks(&curve, 0.0, 1.0, 1e-18, 0.35);
        assert_eq!(
            breaks.as_ref().ok(),
            Some(&vec![0.0, 0.5, 1.0]),
            "{breaks:?}"
        );
        let backward = nurbs_edge_breaks(&curve, 1.0, 0.0, 1e-18, 0.35);
        assert_eq!(
            backward.as_ref().ok(),
            Some(&vec![1.0, 0.5, 0.0]),
            "{backward:?}"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// A degree-one zigzag is straight between its knots, so no span of it
    /// is cut and its count is the count of its corners. One of 1,000,001
    /// pieces would take 1,000,002 samples, past the limit for one edge, and
    /// is refused at the first corner past the limit, before its spans are
    /// built; one of 1000 pieces keeps a break at each.
    #[test]
    fn a_zigzag_past_the_limit_is_refused_for_its_corners() {
        let zigzag = |pieces: u32| {
            let points: Vec<Point3> = (0..=pieces)
                .map(|i| Point3::new(f64::from(i), f64::from(i % 2), 0.0))
                .collect();
            let mut knots = vec![0.0];
            knots.extend((0..=pieces).map(f64::from));
            knots.push(f64::from(pieces));
            let weights = vec![1.0; points.len()];
            NurbsCurve::new(1, knots, points, weights).unwrap()
        };
        let small = nurbs_edge_breaks(&zigzag(1000), 0.0, 1000.0, 0.01, 0.35);
        assert_eq!(
            small.as_ref().ok(),
            Some(&(0..=1000).map(f64::from).collect::<Vec<f64>>()),
            "{small:?}"
        );
        let big = nurbs_edge_breaks(&zigzag(1_000_001), 0.0, 1_000_001.0, 0.01, 0.35);
        // The refusal carries the count it stopped at: no more corners were
        // read than the limit and the edge's two ends.
        assert!(
            matches!(big, Err(crate::OperationsError::TooManySamples { samples, limit, .. })
                if samples > limit && samples <= limit + 2.0),
            "{:?}",
            big.map(|b| b.len())
        );
    }

    /// A straight edge in B-spline form, its control points on one line
    /// (evenly spaced or not), takes one span at any length and far from the
    /// origin, at a deflection below its coordinates' rounding and an angle
    /// below its tangents' rounding: its probes and tangents read only
    /// rounding off the line. Read so, it was cut until its parameters
    /// stopped moving, or past the limit. A spline bowed 1e-6 off its chord
    /// is still cut for it.
    #[test]
    fn a_straight_spline_takes_one_span() {
        let spline = |degree: usize, points: Vec<Point3>| {
            let n = points.len();
            let mut knots = vec![0.0; degree + 1];
            let inner = n - degree - 1;
            #[allow(clippy::cast_precision_loss)]
            knots.extend((1..=inner).map(|k| k as f64 / (inner + 1) as f64));
            knots.extend(vec![1.0; degree + 1]);
            NurbsCurve::new(degree, knots, points, vec![1.0; n]).unwrap()
        };
        let along = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        for degree in [2, 3] {
            for (length, offset, uneven) in [1e-3, 1.0, 1e6]
                .into_iter()
                .flat_map(|l| [0.0, 1e8].map(|o| (l, o)))
                .flat_map(|(l, o)| [false, true].map(|u| (l, o, u)))
            {
                let base = Point3::new(offset, 0.5 * offset, -0.25 * offset);
                #[allow(clippy::cast_precision_loss)]
                let points = (0..degree + 2)
                    .map(|i| {
                        let f = i as f64 / (degree + 1) as f64;
                        base + along * (length * if uneven { f * f } else { f })
                    })
                    .collect();
                let curve = spline(degree, points);
                for deflection in [1e-6, 1e-9, 1e-12] {
                    for angle in [0.0, 1e-9, 0.35] {
                        let breaks = nurbs_edge_breaks(&curve, 0.0, 1.0, deflection, angle);
                        assert!(
                            matches!(&breaks, Ok(b) if b.len() <= 3),
                            "degree {degree}, {length} long at {offset} (uneven {uneven}), \
                             {deflection} and {angle}: {:?}",
                            breaks.map(|b| b.len())
                        );
                    }
                }
            }
        }
        let bowed = spline(
            2,
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.5, 1e-6, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ],
        );
        let breaks = nurbs_edge_breaks(&bowed, 0.0, 1.0, 1e-9, 0.0);
        assert!(matches!(&breaks, Ok(b) if b.len() > 3), "{breaks:?}");
    }

    /// A spline whose control points lie on one line but double back (x at
    /// 0, 10, 10 and 5, or 0, 10 and 5) runs out past its end and back, with
    /// no knot where it turns: its control points lie off the chord's
    /// segment, so it is not taken as straight. Upright, turned and
    /// mirrored, at the origin and 1e8 from it, either way along, its samples
    /// reach the curve's true extent to within the deflection.
    #[test]
    fn a_straight_spline_that_doubles_back_keeps_its_extent() {
        let poses = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0).normalize().unwrap(),
            Vec3::new(-3.0, 1.0, -2.0).normalize().unwrap(),
        ];
        for (degree, xs) in [(3, vec![0.0, 10.0, 10.0, 5.0]), (2, vec![0.0, 10.0, 5.0])] {
            for along in poses {
                for offset in [0.0, 1e8] {
                    let base = Point3::new(offset, -0.5 * offset, 0.25 * offset);
                    let n = xs.len();
                    let mut knots = vec![0.0; degree + 1];
                    knots.extend(vec![1.0; degree + 1]);
                    let points = xs.iter().map(|&x| base + along * x).collect();
                    let curve = NurbsCurve::new(degree, knots, points, vec![1.0; n]).unwrap();
                    let reach_of = |t: f64| (curve.evaluate(t) - base).dot(along);
                    #[allow(clippy::cast_precision_loss)]
                    let extent = (0..=100_000)
                        .map(|k| reach_of(k as f64 / 100_000.0))
                        .fold(f64::MIN, f64::max);
                    for deflection in [1e-3, 1e-6] {
                        for angle in [0.0, 0.35] {
                            for (t0, t1) in [(0.0, 1.0), (1.0, 0.0)] {
                                let breaks =
                                    nurbs_edge_breaks(&curve, t0, t1, deflection, angle).unwrap();
                                let reach =
                                    breaks.iter().map(|&t| reach_of(t)).fold(f64::MIN, f64::max);
                                assert!(
                                    reach >= extent - deflection - 1e-7,
                                    "degree {degree} along {along:?} at {offset}, {deflection} and \
                                     {angle}, {t0} to {t1}: reach {reach} of {extent} in {} breaks",
                                    breaks.len()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// A loop 4e-6 across, 1e10 from the origin along x (a clamped cubic,
    /// its ends at one point): its offsets are far below the rounding of x
    /// but well above that of y and z, so it is not straight. It takes at
    /// least three spans and keeps within the deflection, 1e-7, in y and z,
    /// where its coordinates resolve it. Turned so its offsets mix into
    /// every coordinate, it still takes three spans. An open spline bowed
    /// 4e-6 off its chord there is cut.
    #[test]
    fn a_small_curve_far_out_keeps_its_shape() {
        let cubic = |points: [Point3; 4]| {
            let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
            NurbsCurve::new(3, knots, points.to_vec(), vec![1.0; 4]).unwrap()
        };
        let distinct = |curve: &NurbsCurve, breaks: &[f64]| {
            let mut points: Vec<Point3> = breaks.iter().map(|&t| curve.evaluate(t)).collect();
            points.dedup_by(|p, q| (*p - *q).length() == 0.0);
            if points.len() > 1 && (points[0] - points[points.len() - 1]).length() == 0.0 {
                points.pop();
            }
            points.len()
        };
        let base = Point3::new(1e10, 0.0, 0.0);
        let offsets = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 4e-6, 4e-6),
            Vec3::new(0.0, -4e-6, 4e-6),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let deflection = 1e-7;
        for angle in [0.0, 0.35] {
            let curve = cubic(offsets.map(|o| base + o));
            let breaks = nurbs_edge_breaks(&curve, 0.0, 1.0, deflection, angle).unwrap();
            assert!(distinct(&curve, &breaks) >= 3, "{breaks:?}");
            // In y and z, the curve against its polyline.
            let flat = |p: Point3| (p.y(), p.z());
            let polyline: Vec<(f64, f64)> =
                breaks.iter().map(|&t| flat(curve.evaluate(t))).collect();
            let off = |q: (f64, f64)| {
                polyline
                    .windows(2)
                    .map(|w| {
                        let (a, b) = (w[0], w[1]);
                        let (cx, cy) = (b.0 - a.0, b.1 - a.1);
                        let len2 = cx * cx + cy * cy;
                        let f = if len2 > 0.0 {
                            (((q.0 - a.0) * cx + (q.1 - a.1) * cy) / len2).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        (q.0 - a.0 - cx * f).hypot(q.1 - a.1 - cy * f)
                    })
                    .fold(f64::INFINITY, f64::min)
            };
            #[allow(clippy::cast_precision_loss)]
            let worst = (0..=10_000)
                .map(|k| off(flat(curve.evaluate(f64::from(k) / 10_000.0))))
                .fold(0.0, f64::max);
            assert!(
                worst <= deflection * 1.001,
                "off by {worst} in {} breaks",
                breaks.len()
            );

            let turn = brepkit_math::mat::Mat4::rotation_x(0.7)
                * brepkit_math::mat::Mat4::rotation_z(0.4)
                * brepkit_math::mat::Mat4::rotation_y(1.1);
            let turned = cubic(offsets.map(|o| turn.mul_point(base + o)));
            let breaks = nurbs_edge_breaks(&turned, 0.0, 1.0, deflection, angle).unwrap();
            assert!(distinct(&turned, &breaks) >= 3, "turned: {breaks:?}");
        }
        let bowed = cubic([
            base,
            base + Vec3::new(0.0, 1.0 / 3.0, 4e-6),
            base + Vec3::new(0.0, 2.0 / 3.0, 4e-6),
            base + Vec3::new(0.0, 1.0, 0.0),
        ]);
        let breaks = nurbs_edge_breaks(&bowed, 0.0, 1.0, deflection, 0.0);
        assert!(matches!(&breaks, Ok(b) if b.len() > 3), "bowed: {breaks:?}");
    }

    /// A spline that turns a corner on a middle piece that barely moves (its
    /// three control points there a hair apart, not in line) has parts whose
    /// ends meet. Each is a still piece
    /// of an open curve, cut like any other span, not a closed loop cut in
    /// three at every level: the edge takes a handful of breaks, where it
    /// was cut without end.
    #[test]
    fn a_still_piece_of_an_open_spline_is_not_cut_without_end() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0 + 1e-12, 1e-12, 0.0),
            Point3::new(1.0, 2e-12, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let third = 1.0 / 3.0;
        let curve = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, third, 2.0 * third, 1.0, 1.0, 1.0],
            points,
            vec![1.0; 5],
        )
        .unwrap();
        let breaks = nurbs_edge_breaks(&curve, 0.0, 1.0, 0.05, 0.35).unwrap();
        assert!(breaks.len() < 100, "{}", breaks.len());
    }

    #[test]
    fn a_fitted_edge_meshes_to_its_vertices() {
        let mut topo = Topology::new();
        let (a, b) = (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0));
        // The curve's end misses its vertex by a fillet contact curve's fit
        // error.
        let curve = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![a, Point3::new(1.0, 0.0, 1.4e-6)],
            vec![1.0, 1.0],
        )
        .unwrap();
        let (va, vb) = (
            topo.add_vertex(Vertex::new(a, 1e-7)),
            topo.add_vertex(Vertex::new(b, 1e-7)),
        );
        let edge = topo.add_edge(Edge::new(va, vb, EdgeCurve::NurbsCurve(curve)));
        let points = sample_edge(&topo, topo.edge(edge).unwrap(), 0.01, 0.3, false).unwrap();
        assert_eq!((points.first(), points.last()), (Some(&a), Some(&b)));
    }

    /// A long straight run that ends in a tight hook: a degree 2 curve whose
    /// last span turns back within a hundredth of its length.
    fn hooked_curve() -> NurbsCurve {
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(50.0, 0.0, 0.0),
                Point3::new(100.0, 0.0, 0.0),
                Point3::new(101.0, 0.5, 0.0),
                Point3::new(100.0, 1.0, 0.0),
            ],
            vec![1.0; 5],
        )
        .unwrap()
    }

    #[test]
    fn a_straight_spline_edge_takes_one_span() {
        let line = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(8.0, 0.0, 0.0)],
            vec![1.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            nurbs_edge_breaks(&line, 0.0, 1.0, 0.01, 0.3).unwrap(),
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn a_tight_hook_does_not_size_the_straight_run_before_it() {
        let curve = hooked_curve();
        let (deflection, angular) = (0.01, 0.3);
        let breaks = nurbs_edge_breaks(&curve, 0.0, 3.0, deflection, angular).unwrap();
        let on_run = breaks.iter().filter(|&&t| t < 1.0).count();
        let on_hook = breaks.iter().filter(|&&t| t > 2.0).count();
        assert!(on_run <= 4, "{on_run} breaks on the straight run");
        assert!(on_hook > 2 * on_run, "{on_hook} breaks on the hook");
        // Every span keeps its curve within the deflection of its chord.
        for pair in breaks.windows(2) {
            let (a, b) = (curve.evaluate(pair[0]), curve.evaluate(pair[1]));
            let chord = b - a;
            for k in 1..16 {
                let p = curve.evaluate(pair[0] + (pair[1] - pair[0]) * f64::from(k) / 16.0);
                let along = ((p - a).dot(chord) / chord.dot(chord)).clamp(0.0, 1.0);
                let off = (p - (a + chord * along)).length();
                assert!(
                    off <= deflection,
                    "span {pair:?} stands {off} off its chord"
                );
            }
        }
    }

    #[test]
    fn breaks_run_the_way_the_edge_does() {
        let curve = hooked_curve();
        let breaks = nurbs_edge_breaks(&curve, 3.0, 0.0, 0.01, 0.3).unwrap();
        assert_eq!((breaks.first(), breaks.last()), (Some(&3.0), Some(&0.0)));
        assert!(breaks.windows(2).all(|pair| pair[1] < pair[0]));
    }

    /// A cubic that is straight but for one polynomial piece, between the
    /// knots 0.51 and 0.54, which bumps 0.75 off the line: the knots lie on
    /// the line and no even fraction of the whole falls inside the piece.
    #[test]
    fn a_bump_between_two_knots_is_read() {
        let knots = vec![
            0.0, 0.0, 0.0, 0.0, 0.51, 0.51, 0.51, 0.54, 0.54, 0.54, 1.0, 1.0, 1.0, 1.0,
        ];
        let x = |t: f64| 100.0 * t;
        let straight =
            |from: f64, to: f64, k: f64| Point3::new(x(from + (to - from) * k / 3.0), 0.0, 0.0);
        let lift = |p: Point3| Point3::new(p.x(), 1.0, 0.0);
        let points = vec![
            straight(0.0, 0.51, 0.0),
            straight(0.0, 0.51, 1.0),
            straight(0.0, 0.51, 2.0),
            straight(0.51, 0.54, 0.0),
            lift(straight(0.51, 0.54, 1.0)),
            lift(straight(0.51, 0.54, 2.0)),
            straight(0.54, 1.0, 0.0),
            straight(0.54, 1.0, 1.0),
            straight(0.54, 1.0, 2.0),
            straight(0.54, 1.0, 3.0),
        ];
        let curve = NurbsCurve::new(3, knots, points, vec![1.0; 10]).unwrap();
        assert!((curve.evaluate(0.525).y() - 0.75).abs() < 1e-9);
        let breaks = nurbs_edge_breaks(&curve, 0.0, 1.0, 0.01, 0.0).unwrap();
        let on_bump = breaks.iter().filter(|&&t| t > 0.51 && t < 0.54).count();
        assert!(on_bump >= 4, "{on_bump} breaks on the bump: {breaks:?}");
    }

    /// A closed spline smaller than the deflection still encloses its disc,
    /// down to one smaller than the gap at which close breaks are dropped.
    #[test]
    fn a_closed_spline_smaller_than_the_deflection_keeps_three_spans() {
        let deflection = 0.1;
        for radius in [0.01, 1e-4, 1e-5, 1e-6] {
            let circle = brepkit_math::curves::Circle3D::new(
                Point3::new(0.0, 0.0, 0.0),
                brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
                radius,
            )
            .unwrap();
            let spline =
                brepkit_geometry::convert::circle_to_nurbs(&circle, 0.0, std::f64::consts::TAU)
                    .unwrap();
            let (t0, t1) = spline.domain();
            let breaks = nurbs_edge_breaks(&spline, t0, t1, deflection, 0.0).unwrap();
            assert!(breaks.len() >= 4, "radius {radius}: {breaks:?}");
            let points: Vec<Point3> = breaks.iter().map(|&t| spline.evaluate(t)).collect();
            for pair in points.windows(2) {
                assert!(
                    (pair[1] - pair[0]).length() > 0.1 * radius,
                    "radius {radius}: a span {} long in {breaks:?}",
                    (pair[1] - pair[0]).length()
                );
            }
        }
    }

    /// A closed spline with more knots than a span reads at once is cut
    /// unread, and in three, not two: two spans enclose nothing.
    #[test]
    fn a_closed_spline_with_many_knots_keeps_three_spans() {
        let circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
            1e-3,
        )
        .unwrap();
        let spline =
            brepkit_geometry::convert::circle_to_nurbs(&circle, 0.0, std::f64::consts::TAU)
                .unwrap();
        let (t0, t1) = spline.domain();
        let extra: Vec<f64> = (1..11)
            .map(|k| t0 + (t1 - t0) * (f64::from(k) + 0.37) / 12.0)
            .collect();
        let spline = brepkit_math::nurbs::knot_ops::curve_knot_refine(&spline, &extra).unwrap();
        let breaks = nurbs_edge_breaks(&spline, t0, t1, 0.1, 0.0).unwrap();
        assert!(breaks.len() >= 4, "{breaks:?}");
    }

    /// A hook far smaller than the deflection turns half a turn, but cutting
    /// it for its turn down to the depth limit gives samples a hair apart
    /// (7e-7 here), which a face's (u, v) cannot tell apart: once a span is
    /// shorter than a tenth of the deflection, it is cut for its sag alone.
    #[test]
    fn a_hook_shorter_than_a_tenth_of_the_deflection_is_not_cut_for_its_turn() {
        let curve = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(50.0, 0.0, 0.0),
                Point3::new(100.0, 0.0, 0.0),
                Point3::new(100.004, 0.002, 0.0),
                Point3::new(100.0, 0.004, 0.0),
            ],
            vec![1.0; 5],
        )
        .unwrap();
        let (deflection, angular) = (0.5, 0.5);
        let breaks = nurbs_edge_breaks(&curve, 0.0, 3.0, deflection, angular).unwrap();
        let points: Vec<Point3> = breaks.iter().map(|&t| curve.evaluate(t)).collect();
        let closest = points
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).length())
            .fold(f64::INFINITY, f64::min);
        assert!(
            closest > 1e-4 * deflection,
            "samples {closest} apart: {breaks:?}"
        );
    }

    /// An arc shallower than the deflection keeps its middle, so the face
    /// between it and its chord has three points.
    #[test]
    fn a_shallow_arc_keeps_its_middle() {
        let arc = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.5, 0.01, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ],
            vec![1.0; 3],
        )
        .unwrap();
        assert_eq!(
            nurbs_edge_breaks(&arc, 0.0, 1.0, 0.01, 0.3).unwrap(),
            vec![0.0, 0.5, 1.0]
        );
    }
}
