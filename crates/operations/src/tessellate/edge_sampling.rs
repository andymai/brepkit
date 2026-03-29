//! Edge curve sampling and parametrization.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;

use super::shorter_arc_range;

/// Compute the angular resolution needed for a circular arc to achieve
/// a given chord deviation (sag).
///
/// Delegates to [`brepkit_math::chord::segments_for_chord_deviation`].
pub(super) fn segments_for_chord_deviation(radius: f64, arc_range: f64, deflection: f64) -> usize {
    brepkit_math::chord::segments_for_chord_deviation(radius, arc_range, deflection)
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
pub(super) fn edge_sample_count(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    deflection: f64,
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
                segments_for_chord_deviation(radius, arc_range, deflection) + 1
            } else {
                segments_for_chord_deviation(radius, std::f64::consts::TAU, deflection) + 1
            }
        }
        EdgeCurve::Ellipse(ellipse) => {
            // Use chord-deviation formula with max curvature radius (a^2/b).
            // An ellipse's curvature is highest at the ends of the semi-major axis
            // where the radius of curvature equals a^2/b.
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
            segments_for_chord_deviation(max_curv_radius, arc_range, deflection).min(4096)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            // Adaptive: coarse-pass deviation measurement, then refine if needed.
            let (u0, u1) = nurbs.domain();
            let n_spans = nurbs
                .control_points()
                .len()
                .saturating_sub(nurbs.degree())
                .max(1);
            let coarse_n = (n_spans * 4).clamp(8, 128);
            let max_dev = measure_max_chord_deviation(nurbs, u0, u1, coarse_n);
            if max_dev <= deflection {
                coarse_n
            } else {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let refined = ((coarse_n as f64) * (max_dev / deflection).sqrt()).ceil() as usize;
                refined.clamp(8, 4096)
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
/// to curvature.
///
/// # Errors
///
/// Returns an error if vertex lookup fails for edge endpoints.
pub(super) fn sample_edge(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    deflection: f64,
) -> Result<Vec<Point3>, crate::OperationsError> {
    use brepkit_geometry::sampling::sample_uniform;
    use brepkit_topology::edge::EdgeCurve;

    let n = edge_sample_count(topo, edge, deflection);

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
                let n_samples =
                    segments_for_chord_deviation(circle.radius(), arc_range, deflection);
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
                let n_samples =
                    segments_for_chord_deviation(ellipse.semi_major(), arc_range, deflection);
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
                #[allow(clippy::cast_sign_loss)]
                let n_samples = if max_dev <= deflection {
                    coarse_n
                } else {
                    ((coarse_n as f64) * (max_dev / deflection).sqrt()).ceil() as usize
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

    // Remove closing duplicate.
    if positions.len() > 2 {
        if let (Some(first), Some(last)) = (positions.first(), positions.last()) {
            if (*last - *first).length() < tol {
                positions.pop();
            }
        }
    }

    Ok(positions)
}
