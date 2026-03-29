//! Wire and surface sampling functions for UV space.

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

/// Sample UV points along a wire loop with optional periodic unwrapping.
///
/// When `u_period`/`v_period` is set, unwraps consecutive points so the
/// UV path is continuous (no jumps of ~2pi between edges connected via
/// periodic quantization).
pub(super) fn sample_wire_loop_uv_periodic(
    wire: &[OrientedPCurveEdge],
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> Vec<Point2> {
    use brepkit_math::curves2d::Curve2D;
    const CURVE_SAMPLES: usize = 8;

    let mut pts = Vec::new();
    let has_period = u_period.is_some() || v_period.is_some();
    for edge in wire {
        match &edge.pcurve {
            Curve2D::Line(_) => {
                // For periodic surfaces, push both start and end to enable
                // proper unwrapping across periodic jumps at seam vertices.
                pts.push(edge.start_uv);
                if has_period {
                    pts.push(edge.end_uv);
                }
            }
            Curve2D::Nurbs(nurbs) => {
                let knots = nurbs.knots();
                if knots.len() >= 2 {
                    let t0 = knots[0];
                    let tn = knots[knots.len() - 1];
                    // For reverse edges, the pcurve was computed for the forward
                    // direction. Evaluate from tn->t0 to trace the reverse path.
                    #[allow(clippy::cast_precision_loss)]
                    for i in 0..CURVE_SAMPLES {
                        let frac = i as f64 / CURVE_SAMPLES as f64;
                        let t = if edge.forward {
                            t0 + (tn - t0) * frac
                        } else {
                            tn - (tn - t0) * frac
                        };
                        pts.push(nurbs.evaluate(t));
                    }
                } else {
                    pts.push(edge.start_uv);
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
                    pts.push(Point2::new(
                        edge.start_uv.x() + (edge.end_uv.x() - edge.start_uv.x()) * t,
                        edge.start_uv.y() + (edge.end_uv.y() - edge.start_uv.y()) * t,
                    ));
                }
            }
        }
    }

    // Unwrap periodic UV jumps between consecutive points.
    if pts.len() >= 2 {
        super::super::pcurve_compute::unwrap_periodic_params_pub(&mut pts, u_period, v_period);
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
