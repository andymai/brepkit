//! Edge curve sampling and parametrization.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;

use super::shorter_arc_range;

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
) -> usize {
    use brepkit_topology::edge::EdgeCurve;

    match edge.curve() {
        EdgeCurve::Line => 2,
        EdgeCurve::Circle(c) => {
            let radius = c.radius();
            // Use the same segments_for_chord_deviation formula that
            // tessellate_analytic uses for the grid density. This ensures
            // edge sample points align with the analytic grid boundary,
            // allowing the snap path to achieve watertight stitching.
            if let Ok((t_start, t_end)) = circle_param_range(topo, edge, c) {
                let arc_range = (t_end - t_start).abs();
                segments_for_chord_deviation_a(
                    radius,
                    arc_range,
                    deflection,
                    angular_tol,
                    circle_floor,
                ) + 1
            } else {
                segments_for_chord_deviation_a(
                    radius,
                    std::f64::consts::TAU,
                    deflection,
                    angular_tol,
                    circle_floor,
                ) + 1
            }
        }
        EdgeCurve::Ellipse(ellipse) => {
            // Density is driven by the LARGEST radius of curvature (a^2/b, at the
            // minor-axis ends). Under uniform-parameter sampling the per-segment
            // chord deviation is set by how far the parameter sweeps in arc length,
            // which peaks where curvature is lowest; the small-radius criterion
            // (b^2/a) satisfies pointwise sag but lets the integrated (area/volume)
            // error grow ~15x. Using a^2/b keeps both bounded.
            let a = ellipse.semi_major();
            let b = ellipse.semi_minor();
            let max_curv_radius = a * a / b;
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
            segments_for_chord_deviation_a(
                max_curv_radius,
                arc_range,
                deflection,
                angular_tol,
                true,
            )
            .min(4096)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            // Adaptive: coarse-pass deviation measurement, then refine if the
            // chord sag OR the per-segment turn exceeds tolerance.
            let (u0, u1) = nurbs.domain();
            let n_spans = nurbs
                .control_points()
                .len()
                .saturating_sub(nurbs.degree())
                .max(1);
            let coarse_n = (n_spans * 4).clamp(8, 128);
            let max_dev = measure_max_chord_deviation(nurbs, u0, u1, coarse_n);
            let max_turn = measure_max_segment_turn(nurbs, u0, u1, coarse_n);
            let sag_ok = max_dev <= deflection;
            let turn_ok = angular_tol <= 0.0 || max_turn <= angular_tol * 0.5;
            if sag_ok && turn_ok {
                coarse_n
            } else {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let sag_n = if sag_ok {
                    coarse_n
                } else {
                    ((coarse_n as f64) * (max_dev / deflection).sqrt()).ceil() as usize
                };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let turn_n = if turn_ok {
                    coarse_n
                } else {
                    ((coarse_n as f64) * (max_turn / (angular_tol * 0.5))).ceil() as usize
                };
                sag_n.max(turn_n).clamp(8, 4096)
            }
        }
    }
}

/// Measure the maximum midpoint chord deviation across `n` segments of a NURBS curve.
///
/// For each segment `[u_i, u_{i+1}]`, evaluates the curve at the midpoint and
/// measures its distance from the chord midpoint. Returns the maximum deviation.
pub(super) fn measure_max_chord_deviation(
    nurbs: &brepkit_math::nurbs::curve::NurbsCurve,
    u0: f64,
    u1: f64,
    n: usize,
) -> f64 {
    let mut max_dev: f64 = 0.0;
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let t0 = u0 + (u1 - u0) * (i as f64) / (n as f64);
        let t1 = u0 + (u1 - u0) * ((i + 1) as f64) / (n as f64);
        let p0 = nurbs.evaluate(t0);
        let p1 = nurbs.evaluate(t1);
        let mid_chord = Point3::new(
            (p0.x() + p1.x()) * 0.5,
            (p0.y() + p1.y()) * 0.5,
            (p0.z() + p1.z()) * 0.5,
        );
        let mid_curve = nurbs.evaluate((t0 + t1) * 0.5);
        let dev = (mid_curve - mid_chord).length();
        max_dev = max_dev.max(dev);
    }
    max_dev
}

/// Measure the maximum tangent turn angle (radians) at segment midpoints of a
/// NURBS curve sampled over `n` uniform segments.
///
/// For each segment the curve tangent is compared at the segment endpoints; the
/// angle between them is the swing across that segment.
pub(super) fn measure_max_segment_turn(
    nurbs: &brepkit_math::nurbs::curve::NurbsCurve,
    u0: f64,
    u1: f64,
    n: usize,
) -> f64 {
    let mut max_turn: f64 = 0.0;
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let t0 = u0 + (u1 - u0) * (i as f64) / (n as f64);
        let t1 = u0 + (u1 - u0) * ((i + 1) as f64) / (n as f64);
        if let (Ok(a), Ok(b)) = (nurbs.tangent(t0), nurbs.tangent(t1)) {
            let dot = a.dot(b).clamp(-1.0, 1.0);
            max_turn = max_turn.max(dot.acos());
        }
    }
    max_turn
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
    use brepkit_geometry::sampling::sample_uniform;
    use brepkit_topology::edge::EdgeCurve;

    let n = edge_sample_count(topo, edge, deflection, angular_tol, circle_floor);

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
            let (u0, u1) = nurbs.domain();
            sample_uniform(nurbs, u0, u1, n)
        }
    };

    Ok(points)
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
            Box::new((0..n_samples).rev())
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
                    shorter_arc_range(circle, topo, edge)?
                };
                let arc_range = (t_end - t_start).abs();
                let n_samples = segments_for_chord_deviation_a(
                    circle.radius(),
                    arc_range,
                    deflection,
                    angular_tol,
                    false,
                );
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
                let arc_range = t_end - t_start;
                // Largest radius of curvature (a^2/b) governs uniform-parameter
                // sampling density; see edge_sample_count for the rationale.
                let max_curv_radius =
                    ellipse.semi_major() * ellipse.semi_major() / ellipse.semi_minor();
                let n_samples = segments_for_chord_deviation_a(
                    max_curv_radius,
                    arc_range,
                    deflection,
                    angular_tol,
                    true,
                );
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
                let (u0, u1) = nurbs.domain();
                let n_spans = nurbs
                    .control_points()
                    .len()
                    .saturating_sub(nurbs.degree())
                    .max(1);
                let coarse_n = (n_spans * 4).clamp(8, 128);
                let max_dev = measure_max_chord_deviation(nurbs, u0, u1, coarse_n);
                let max_turn = measure_max_segment_turn(nurbs, u0, u1, coarse_n);
                let sag_ok = max_dev <= deflection;
                let turn_ok = angular_tol <= 0.0 || max_turn <= angular_tol * 0.5;
                #[allow(clippy::cast_sign_loss)]
                let n_samples = if sag_ok && turn_ok {
                    coarse_n
                } else {
                    let sag_n = if sag_ok {
                        coarse_n
                    } else {
                        ((coarse_n as f64) * (max_dev / deflection).sqrt()).ceil() as usize
                    };
                    let turn_n = if turn_ok {
                        coarse_n
                    } else {
                        ((coarse_n as f64) * (max_turn / (angular_tol * 0.5))).ceil() as usize
                    };
                    sag_n.max(turn_n)
                }
                .clamp(8, 4096);
                #[allow(clippy::cast_precision_loss)]
                sample_curve_into(
                    &|t| nurbs.evaluate(t),
                    &|i| u0 + (u1 - u0) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
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
