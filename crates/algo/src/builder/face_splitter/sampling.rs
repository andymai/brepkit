//! Wire and surface sampling functions for UV space.

use brepkit_math::madd::Madd;
use brepkit_math::vec::Point2;

use super::super::split_types::OrientedPCurveEdge;

/// Sample UV points along a wire loop, interpolating along curved edges.
///
/// For line edges, uses only the start point. For curved edges (Circle,
/// Ellipse, NurbsCurve), samples N intermediate points to approximate the
/// true curve shape in UV. This is critical for signed area computation
/// and point-in-polygon tests on loops with curved edges.
pub(super) fn sample_wire_loop_uv(wire: &[OrientedPCurveEdge]) -> Vec<Point2> {
    sample_wire_loop_uv_periodic(wire, None, None)
}

/// Sample a plane-face wire loop by evaluating each edge's 3D curve and
/// projecting through the face's `PlaneFrame` — never the stored pcurve.
///
/// Pcurves reach a wire under two orientation conventions (sections carry the
/// curve's natural parameterization plus a traversal flag; boundary edges are
/// fit in traversal order but keep the topology orientation flag), so a
/// pcurve-driven sampler can trace a reversed-boundary arc backwards and fold
/// the loop polygon into a self-crossing zig-zag. The 3D curve with the
/// traversal endpoints is orientation-unambiguous. Mirrors the arc-true hole
/// polygons in `find_point_outside_holes`.
pub(super) fn sample_wire_loop_uv_via_frame(
    wire: &[OrientedPCurveEdge],
    frame: &super::super::plane_frame::PlaneFrame,
) -> Vec<Point2> {
    use brepkit_topology::edge::EdgeCurve;
    const CURVE_SAMPLES: usize = 8;

    let mut pts = Vec::with_capacity(wire.len() * CURVE_SAMPLES);
    for e in wire {
        pts.push(e.start_uv);
        if matches!(e.curve_3d, EdgeCurve::Line) {
            continue;
        }
        // Sample in the edge's NATIVE orientation: `domain_with_endpoints`
        // always takes the positive parametric span, so swapped endpoints
        // would select the COMPLEMENTARY arc. Restore wire order afterwards.
        let (s3, e3) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let (mut t0, mut t1) = e.curve_3d.domain_with_endpoints(s3, e3);
        // A closed circle's domain starts at its frame's parameter 0, not at
        // its vertex: run the full turn from the vertex.
        if let EdgeCurve::Circle(c) = &e.curve_3d
            && (s3 - e3).length() < 1e-9
        {
            t0 = c.project(s3);
            t1 = t0 + std::f64::consts::TAU;
        }
        // A conic's samples follow its turn, one at least every 1/64 of it:
        // eight over a whole circle make an octagon whose sides cut inside
        // it by 7.6% of its radius, past which a hole near the rim pokes.
        let n = match &e.curve_3d {
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let by_turn = ((t1 - t0).abs() / (std::f64::consts::PI / 32.0)).ceil() as usize;
                by_turn.clamp(CURVE_SAMPLES, 64)
            }
            _ => CURVE_SAMPLES,
        };
        #[allow(clippy::cast_precision_loss)]
        let mut samples: Vec<Point2> = (1..n)
            .map(|k| {
                let t = (t1 - t0).madd(k as f64 / n as f64, t0);
                frame.project(e.curve_3d.evaluate_with_endpoints(t, s3, e3))
            })
            .collect();
        if !e.forward {
            samples.reverse();
        }
        pts.extend(samples);
    }
    pts
}

/// Sample a periodic-face wire loop by evaluating each edge's 3D curve in
/// traversal order and projecting it onto the surface, each edge from its
/// laid-out start ([`sample_wire_edges_uv_on_surface`]). Used where the
/// stored pcurves cannot be trusted to follow the traversal (a pointed
/// cone's seam copies).
pub(super) fn sample_wire_loop_uv_on_surface(
    wire: &[OrientedPCurveEdge],
    surface: &brepkit_topology::face::FaceSurface,
) -> Vec<Point2> {
    sample_wire_edges_uv_on_surface(wire, surface)
        .into_iter()
        .flat_map(|(samples, _)| samples)
        .collect()
}

/// Each edge of a periodic-face wire loop sampled on its 3D curve in
/// traversal order and projected onto the surface: its laid-out start, then
/// samples that run on from it as [`edge_uv_samples`] lifts them; with the
/// `u` its end reaches that way.
///
/// [`edge_uv_samples`]: super::super::pcurve_compute::edge_uv_samples
pub(super) fn sample_wire_edges_uv_on_surface(
    wire: &[OrientedPCurveEdge],
    surface: &brepkit_topology::face::FaceSurface,
) -> Vec<(Vec<Point2>, f64)> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::TAU;
    const CURVE_SAMPLES: usize = 8;

    let mut out = Vec::with_capacity(wire.len());
    for e in wire {
        let start = e.start_uv;
        let mut pts = vec![start];
        if matches!(e.curve_3d, EdgeCurve::Line) {
            out.push((pts, e.end_uv.x()));
            continue;
        }
        let (s3, e3) = if e.forward {
            (e.start_3d, e.end_3d)
        } else {
            (e.end_3d, e.start_3d)
        };
        let closed = (e.start_3d - e.end_3d).length() < 1e-10;
        #[allow(clippy::cast_precision_loss)]
        let fraction = |k: usize| k as f64 / CURVE_SAMPLES as f64;
        let samples: Vec<_> = match &e.curve_3d {
            // A closed conic's own domain starts wherever its frame puts
            // parameter 0: run a full turn from its vertex, in its own sense
            // when forward.
            EdgeCurve::Circle(c) if closed => {
                let (a0, turn) = (c.project(e.start_3d), if e.forward { TAU } else { -TAU });
                (1..CURVE_SAMPLES)
                    .map(|k| c.evaluate(turn.madd(fraction(k), a0)))
                    .collect()
            }
            EdgeCurve::Ellipse(el) if closed => {
                let (a0, turn) = (el.project(e.start_3d), if e.forward { TAU } else { -TAU });
                (1..CURVE_SAMPLES)
                    .map(|k| el.evaluate(turn.madd(fraction(k), a0)))
                    .collect()
            }
            _ => {
                let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
                let mut samples: Vec<_> = (1..CURVE_SAMPLES)
                    .map(|k| {
                        e.curve_3d
                            .evaluate_with_endpoints((t1 - t0).madd(fraction(k), t0), s3, e3)
                    })
                    .collect();
                // A marched curve's domain can run from either end: orient by
                // where its start actually evaluates.
                let at_t0 = e.curve_3d.evaluate_with_endpoints(t0, s3, e3);
                if (at_t0 - e.start_3d).length() > (at_t0 - e.end_3d).length() {
                    samples.reverse();
                }
                samples
            }
        };
        let mut chain = vec![e.start_3d];
        chain.extend(samples);
        chain.push(e.end_3d);
        chain.retain(|&p| surface.project_point(p).is_some());
        let lifted = super::super::pcurve_compute::edge_uv_samples(&e.curve_3d, surface, &chain)
            .filter(|l| l.len() >= 2);
        let Some(lifted) = lifted else {
            out.push((pts, e.end_uv.x()));
            continue;
        };
        let onto = ((start.x() - lifted[0].x()) / TAU).round() * TAU;
        for p in &lifted[1..lifted.len() - 1] {
            pts.push(Point2::new(p.x() + onto, p.y()));
        }
        out.push((pts, lifted[lifted.len() - 1].x() + onto));
    }
    out
}

/// Sample UV points along a wire loop with optional periodic unwrapping.
///
/// When `u_period`/`v_period` is set, each edge's samples move by whole
/// periods to meet the edge before at their shared vertex, so the UV path
/// is continuous (no jumps of ~2pi between edges connected via periodic
/// quantization); within an edge the samples keep its pcurve's own run.
pub(super) fn sample_wire_loop_uv_periodic(
    wire: &[OrientedPCurveEdge],
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> Vec<Point2> {
    use brepkit_math::curves2d::Curve2D;
    const CURVE_SAMPLES: usize = 8;

    let mut pts = Vec::new();
    let has_period = u_period.is_some() || v_period.is_some();
    let mut end: Option<Point2> = None;
    for edge in wire {
        let mut block: Vec<Point2> = Vec::new();
        let mut block_end = edge.end_uv;
        match &edge.pcurve {
            Curve2D::Line(_) => {
                // For periodic surfaces, push both start and end, so the
                // edge's own run in a periodic direction stays in the
                // polygon.
                block.push(edge.start_uv);
                if has_period {
                    // Steps well under half a period along that run. Only
                    // the periodic directions are stepped: a wall's height
                    // in model units needs none.
                    let d = edge.end_uv - edge.start_uv;
                    let turn = |delta: f64, period: Option<f64>| {
                        period.map_or(0.0, |p| delta.abs() * std::f64::consts::TAU / p)
                    };
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let steps = turn(d.x(), u_period)
                        .max(turn(d.y(), v_period))
                        .ceil()
                        .max(1.0) as usize;
                    #[allow(clippy::cast_precision_loss)]
                    for i in 1..steps {
                        let t = i as f64 / steps as f64;
                        block.push(Point2::new(
                            d.x().madd(t, edge.start_uv.x()),
                            d.y().madd(t, edge.start_uv.y()),
                        ));
                    }
                    block.push(edge.end_uv);
                }
            }
            Curve2D::Nurbs(nurbs) => {
                let knots = nurbs.knots();
                if knots.len() >= 2 {
                    let t0 = knots[0];
                    let tn = knots[knots.len() - 1];
                    // A pcurve fit on one period copy can reach a wire whose
                    // end UVs were moved to another (a section starting on a
                    // seam): sample it on the copy its start sits at.
                    // Both ends must land on the edge's end UVs under the one
                    // shift.
                    // A marched section's pcurve is a fit, its ends a few
                    // 1e-4 off the edge's own UVs; whole turns are what tell
                    // the copy apart, not the fit's residual.
                    let whole_periods = |at_start: f64, at_end: f64| {
                        let turns = (at_start / std::f64::consts::TAU).round();
                        let shift = turns * std::f64::consts::TAU;
                        let residual = (at_start - shift).abs().max((at_end - shift).abs());
                        (shift, (residual < 1e-2).then_some(residual))
                    };
                    let shifts = |t_first: f64, t_last: f64| {
                        let (first, last) = (nurbs.evaluate(t_first), nurbs.evaluate(t_last));
                        let (du, u_fit) = whole_periods(
                            edge.start_uv.x() - first.x(),
                            edge.end_uv.x() - last.x(),
                        );
                        let (dv, v_fit) = whole_periods(
                            edge.start_uv.y() - first.y(),
                            edge.end_uv.y() - last.y(),
                        );
                        let fit = u_fit.zip(v_fit).map(|(a, b)| a.max(b));
                        (
                            if u_fit.is_some() { du } else { 0.0 },
                            if v_fit.is_some() { dv } else { 0.0 },
                            fit,
                        )
                    };
                    // A section's pcurve runs the curve's own way, so a
                    // reverse edge is read from tn to t0; a boundary edge's
                    // is fit in traversal order. Read it the way whose ends
                    // land nearer the edge's own (on an edge shorter than the
                    // fit's band, both ways land within it).
                    let (from, to) = if edge.forward { (t0, tn) } else { (tn, t0) };
                    let (stored, turned) = (shifts(from, to), shifts(to, from));
                    // Where neither way lands within whole turns of the
                    // edge's ends, the way whose ends land clearly nearer,
                    // read modulo the surface's periods, is the one the wire
                    // runs, whatever the flag says (a plane facing against a
                    // circle's axis has its rim arcs both ways).
                    let nearer_turned = || {
                        let span_off = |dx: f64, dy: f64| {
                            let reduce = |x: f64, period: Option<f64>| {
                                period.map_or(x, |p| x - (x / p).round() * p)
                            };
                            reduce(dx, u_period).hypot(reduce(dy, v_period))
                        };
                        let misfit = |t_first: f64, t_last: f64| {
                            let (first, last) = (nurbs.evaluate(t_first), nurbs.evaluate(t_last));
                            span_off(edge.start_uv.x() - first.x(), edge.start_uv.y() - first.y())
                                + span_off(edge.end_uv.x() - last.x(), edge.end_uv.y() - last.y())
                        };
                        let (flagged, other) = (misfit(from, to), misfit(to, from));
                        (edge.end_uv.x() - edge.start_uv.x())
                            .hypot(edge.end_uv.y() - edge.start_uv.y())
                            > 1e-9
                            && flagged > 1e-6
                            && other < 0.25 * flagged
                    };
                    let reads_turned = match (stored.2, turned.2) {
                        (Some(a), Some(b)) => b < a,
                        (None, Some(_)) => true,
                        (Some(_), None) => false,
                        (None, None) => nearer_turned(),
                    };
                    let (from, to, (du, dv, _)) = if reads_turned {
                        (to, from, turned)
                    } else {
                        (from, to, stored)
                    };
                    #[allow(clippy::cast_precision_loss)]
                    for i in 0..CURVE_SAMPLES {
                        let frac = i as f64 / CURVE_SAMPLES as f64;
                        let t = from + (to - from) * frac;
                        let p = nurbs.evaluate(t);
                        block.push(Point2::new(p.x() + du, p.y() + dv));
                    }
                    let p = nurbs.evaluate(to);
                    block_end = Point2::new(p.x() + du, p.y() + dv);
                } else {
                    block.push(edge.start_uv);
                }
            }
            Curve2D::Circle(_) | Curve2D::Ellipse(_) => {
                // Circle2D/Ellipse2D pcurves: interpolate between start_uv
                // and end_uv. This is approximate (chord, not arc) but these
                // pcurve types are rare in the pipeline -- section edges use
                // NURBS and boundary edges use Line2D.
                #[allow(clippy::cast_precision_loss)]
                for i in 0..CURVE_SAMPLES {
                    let t = i as f64 / CURVE_SAMPLES as f64;
                    block.push(Point2::new(
                        edge.start_uv.x() + (edge.end_uv.x() - edge.start_uv.x()) * t,
                        edge.start_uv.y() + (edge.end_uv.y() - edge.start_uv.y()) * t,
                    ));
                }
            }
        }
        // The edge's own run stays as it is: a pcurve moves continuously,
        // and between two of its samples it can run most of a turn (a steep
        // cut's near vertex), which a step to the nearest copy reads as a
        // step back. The edge moves by whole periods to the copy that meets
        // the edge before at the vertex they share.
        let (Some(first), Some(prev)) = (block.first().copied(), end) else {
            pts.extend(block);
            end = Some(block_end);
            continue;
        };
        let turns = |x: f64, to: f64, period: Option<f64>| {
            period.map_or(0.0, |p| ((to - x) / p).round() * p)
        };
        let shift = brepkit_math::vec::Vec2::new(
            turns(first.x(), prev.x(), u_period),
            turns(first.y(), prev.y(), v_period),
        );
        pts.extend(block.into_iter().map(|q| q + shift));
        end = Some(block_end + shift);
    }

    pts
}

/// Normalize an angle into the `[0, 1]` parameter range of an edge span.
///
/// `t0` is the start angle, `span = t1 - t0` is the signed angular range.
/// Returns `(angle - t0) / span`, wrapping by 2pi to stay within the arc.
pub(super) fn normalize_angle_in_span(angle: f64, t0: f64, span: f64) -> f64 {
    use std::f64::consts::TAU;
    let mut delta = angle - t0;
    if span > 0.0 {
        // CCW arc: delta should be in [0, span].
        // At most 2 wraps needed (angle is in (-pi, pi]).
        for _ in 0..3 {
            if delta >= -1e-10 {
                break;
            }
            delta += TAU;
        }
        for _ in 0..3 {
            if delta <= span + 1e-10 {
                break;
            }
            delta -= TAU;
        }
    } else {
        // CW arc: delta should be in [span, 0].
        for _ in 0..3 {
            if delta <= 1e-10 {
                break;
            }
            delta -= TAU;
        }
        for _ in 0..3 {
            if delta >= span - 1e-10 {
                break;
            }
            delta += TAU;
        }
    }
    delta / span
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves2d::{Curve2D, NurbsCurve2D};
    use brepkit_math::vec::{Point2, Point3};
    use brepkit_topology::edge::EdgeCurve;

    use super::*;

    fn edge(pcurve: (Point2, Point2), uv: (Point2, Point2), forward: bool) -> OrientedPCurveEdge {
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: Curve2D::Nurbs(NurbsCurve2D::from_line(pcurve.0, pcurve.1).unwrap()),
            start_uv: uv.0,
            end_uv: uv.1,
            start_3d: Point3::new(0.0, 0.0, 0.0),
            end_3d: Point3::new(1.0, 0.0, 0.0),
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    /// A reverse edge's pcurve may run the curve's own way (a section's) or
    /// the way the edge is walked (a boundary edge's), on the edge's own
    /// period copy or another: either way its samples run from the edge's
    /// start to its end.
    #[test]
    fn a_reverse_edge_samples_from_its_start_whichever_way_its_pcurve_runs() {
        let tau = std::f64::consts::TAU;
        let (a, b) = (Point2::new(1.0, 2.0), Point2::new(1.0, 4.0));
        let short = Point2::new(1.0, 2.005);
        for (shift, pcurve) in [
            (0.0, (b, a)),
            (0.0, (a, b)),
            (tau, (b, a)),
            (tau, (a, b)),
            (0.0, (short, a)),
            (0.0, (a, short)),
        ] {
            let end = if pcurve.0 == short || pcurve.1 == short {
                short
            } else {
                b
            };
            let uv = (
                Point2::new(a.x(), a.y() + shift),
                Point2::new(end.x(), end.y() + shift),
            );
            let pts = sample_wire_loop_uv_periodic(&[edge(pcurve, uv, false)], None, None);
            assert!((pts[0] - uv.0).length() < 1e-12, "{pcurve:?}: {pts:?}");
            assert!(
                pts.windows(2).all(|w| w[1].y() > w[0].y()),
                "{pcurve:?}: {pts:?}"
            );
        }
    }

    /// A pcurve whose ends land on neither of the edge's ends within whole
    /// turns, either way it is read (a fit 0.05 off them), is read the way
    /// whose ends land clearly nearer, whatever the flag says.
    #[test]
    fn a_pcurve_off_its_edge_is_read_the_way_that_lands_nearer() {
        let (a, b) = (Point2::new(1.0, 2.0), Point2::new(3.0, 2.5));
        let off = |p: Point2| Point2::new(p.x() + 0.05, p.y() - 0.05);
        // The pcurve runs from b to a, the edge from a to b, the flag forward.
        let pts = sample_wire_loop_uv_periodic(&[edge((off(b), off(a)), (a, b), true)], None, None);
        assert!((pts[0] - off(a)).length() < 1e-12, "{pts:?}");
        assert!(pts.windows(2).all(|w| w[1].x() > w[0].x()), "{pts:?}");
    }

    fn line(a: Point2, b: Point2) -> OrientedPCurveEdge {
        OrientedPCurveEdge {
            pcurve: Curve2D::Line(brepkit_math::curves2d::Line2D::new(a, b - a).unwrap()),
            ..edge((a, b), (a, b), true)
        }
    }

    /// On a cylinder only `u` turns: a ruling 100 000 high is its two ends,
    /// and a rim half (half a turn) is stepped under a radian at a time
    /// whatever the wall's height.
    #[test]
    fn a_line_is_stepped_along_its_periodic_direction_only() {
        let tau = std::f64::consts::TAU;
        let ruling = line(Point2::new(1.0, 0.0), Point2::new(1.0, 1e5));
        let pts = sample_wire_loop_uv_periodic(&[ruling], Some(tau), None);
        assert_eq!(pts.len(), 2, "{} samples", pts.len());
        let rim = line(Point2::new(0.0, 1e5), Point2::new(tau / 2.0, 1e5));
        let pts = sample_wire_loop_uv_periodic(&[rim], Some(tau), None);
        assert_eq!(pts.len(), 5, "{pts:?}");
        assert!(pts.windows(2).all(|w| w[1].x() - w[0].x() < 1.0), "{pts:?}");
    }

    /// A loop on the cone of `cone_ellipse_fixture::oblique_cone` at slope
    /// 0.99999: along the ellipse from its parameter 3 to 3.4, where `u`
    /// runs 5.091 between two of its eight samples round the near vertex, up
    /// a ruling to `v = 5000`, back along that height, and down a ruling;
    /// and the loop run the other way. The pcurve sampler keeps the
    /// ellipse's pcurve as it runs, and the next edge starts where it ends
    /// (the loop winds nothing, so after the last edge it is back at its
    /// first point); the sampler on the 3D curves lifts the ellipse by its
    /// exact run.
    #[test]
    fn wire_samplers_keep_a_steep_cone_ellipse_on_its_turn() {
        use crate::builder::pcurve_compute::compute_boundary_pcurve_on_surface;
        use crate::builder::pcurve_compute::cone_ellipse_fixture::{lifted_u, oblique_cone};
        use brepkit_math::curves::Circle3D;
        use brepkit_math::curves2d::Line2D;
        use brepkit_math::vec::Vec3;
        use std::f64::consts::{FRAC_PI_4, PI, TAU};

        let (surface, ellipse) = oblique_cone(0.99999);
        let curve = EdgeCurve::Ellipse(ellipse.clone());
        let (a, b) = (3.0, 3.4);
        let (u_a, u_b) = lifted_u(&surface, &ellipse, a, b);
        let (_, v_a) = surface.project_point(ellipse.evaluate(a)).unwrap();
        let (_, v_b) = surface.project_point(ellipse.evaluate(b)).unwrap();
        let v_high = 5000.0;
        let on = |u: f64, v: f64| surface.evaluate(u, v).unwrap();
        let line = |from: Point2, to: Point2, s3: Point3, e3: Point3| OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: Curve2D::Line(Line2D::new(from, to - from).unwrap()),
            start_uv: from,
            end_uv: to,
            start_3d: s3,
            end_3d: e3,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let height = Circle3D::new(
            Point3::new(0.0, 0.0, v_high * FRAC_PI_4.sin()),
            Vec3::new(0.0, 0.0, -1.0),
            v_high * FRAC_PI_4.cos(),
        )
        .unwrap();
        let (pa, pb) = (Point2::new(u_a, v_a), Point2::new(u_b, v_b));
        let (ha, hb) = (Point2::new(u_a, v_high), Point2::new(u_b, v_high));
        let arc = |forward: bool| OrientedPCurveEdge {
            curve_3d: curve.clone(),
            pcurve: compute_boundary_pcurve_on_surface(
                &curve,
                ellipse.evaluate(a),
                ellipse.evaluate(b),
                forward,
                &surface,
                &[],
                None,
            ),
            start_uv: if forward { pa } else { pb },
            end_uv: if forward { pb } else { pa },
            start_3d: ellipse.evaluate(if forward { a } else { b }),
            end_3d: ellipse.evaluate(if forward { b } else { a }),
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let around = |forward: bool| OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(height.clone()),
            pcurve: if forward {
                Curve2D::Line(Line2D::new(hb, ha - hb).unwrap())
            } else {
                Curve2D::Line(Line2D::new(ha, hb - ha).unwrap())
            },
            start_uv: if forward { hb } else { ha },
            end_uv: if forward { ha } else { hb },
            start_3d: on(if forward { u_b } else { u_a }, v_high),
            end_3d: on(if forward { u_a } else { u_b }, v_high),
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let along = vec![
            arc(true),
            line(pb, hb, ellipse.evaluate(b), on(u_b, v_high)),
            around(true),
            line(ha, pa, on(u_a, v_high), ellipse.evaluate(a)),
        ];
        let back = vec![
            line(pa, ha, ellipse.evaluate(a), on(u_a, v_high)),
            around(false),
            line(hb, pb, on(u_b, v_high), ellipse.evaluate(b)),
            arc(false),
        ];
        for (way, wire, arc_at, run) in
            [("along", along, 0, u_b - u_a), ("back", back, 3, u_a - u_b)]
        {
            let pts = sample_wire_loop_uv_periodic(&wire, Some(TAU), None);
            // The arc's eight samples run as its pcurve does, one step most
            // of a turn, and the edge after it starts at its end.
            let start = if arc_at == 0 { 0 } else { pts.len() - 8 };
            let block = &pts[start..start + 8];
            let pcurve = &wire[arc_at].pcurve;
            for (k, q) in block.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let want = pcurve.evaluate(k as f64 / 8.0);
                assert!(
                    (q.x() - block[0].x() - (want.x() - pcurve.evaluate(0.0).x())).abs() < 1e-9,
                    "{way}: sample {k} of the arc at {q:?}, its pcurve at {want:?}"
                );
            }
            let steps: Vec<f64> = block.windows(2).map(|w| w[1].x() - w[0].x()).collect();
            assert!(
                steps.iter().all(|&d| d * run > 0.0) && steps.iter().any(|d| d.abs() > PI),
                "{way}: the arc's steps {steps:?}"
            );
            // The loop winds nothing, so it comes back to its first point.
            let after = if arc_at == 0 { pts[8] } else { pts[0] };
            let end = block[0].x() + run;
            assert!(
                (after.x() - end).abs() < 1e-6,
                "{way}: the edge after the arc starts at {after:?}, the arc ends at u {end}"
            );
            // The sampler on the 3D curves: the arc's run, exact, its samples
            // running one way.
            let edges = sample_wire_edges_uv_on_surface(&wire, &surface);
            let (samples, edge_end) = &edges[arc_at];
            assert!(
                (edge_end - samples[0].x() - run).abs() < 1e-9,
                "{way}: the arc's samples run {} in u, the arc {run}",
                edge_end - samples[0].x()
            );
            assert!(
                samples
                    .windows(2)
                    .all(|w| (w[1].x() - w[0].x()) * run > 0.0),
                "{way}: the arc's samples step back: {samples:?}"
            );
        }
    }
}
