//! Shared helpers for measurement operations.

use std::collections::HashSet;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// Collect deduplicated vertex positions from a solid.
pub(super) fn collect_solid_vertex_points(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let solid_data = topo.solid(solid)?;

    for shell_id in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id)?;
        for &fid in shell.faces() {
            let face = topo.face(fid)?;
            for wire_id in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge())?;
                    vertex_ids.insert(edge.start());
                    vertex_ids.insert(edge.end());
                }
            }
        }
    }

    let mut points = Vec::with_capacity(vertex_ids.len());
    for vid in vertex_ids {
        points.push(topo.vertex(vid)?.point());
    }
    Ok(points)
}

/// Collect all face IDs from a solid's shells.
pub(super) fn collect_solid_face_ids(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<FaceId>, crate::OperationsError> {
    let mut face_ids = Vec::new();
    let solid_data = topo.solid(solid)?;

    for shell_id in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id)?;
        face_ids.extend_from_slice(shell.faces());
    }
    Ok(face_ids)
}

/// Collect ordered vertex positions from a wire.
pub(super) fn collect_wire_positions(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
) -> Result<Vec<Point3>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    let mut positions = Vec::new();
    let n_samples = 256_usize;
    let tol = 1e-10;

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        match edge.curve() {
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
            EdgeCurve::Circle(c) => {
                let (t0, t1) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    let ts = c.project(sp);
                    let mut te = c.project(ep);
                    if te <= ts {
                        te += std::f64::consts::TAU;
                    }
                    (ts, te)
                };
                sample_edge_curve(
                    &|t| c.evaluate(t),
                    t0,
                    t1,
                    n_samples,
                    oe.is_forward(),
                    tol,
                    &mut positions,
                );
            }
            EdgeCurve::Ellipse(e) => {
                let (t0, t1) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    let ts = e.project(sp);
                    let mut te = e.project(ep);
                    if te <= ts {
                        te += std::f64::consts::TAU;
                    }
                    (ts, te)
                };
                sample_edge_curve(
                    &|t| e.evaluate(t),
                    t0,
                    t1,
                    n_samples,
                    oe.is_forward(),
                    tol,
                    &mut positions,
                );
            }
            EdgeCurve::NurbsCurve(nc) => {
                let (u0, u1) = nc.domain();
                sample_edge_curve(
                    &|t| nc.evaluate(t),
                    u0,
                    u1,
                    n_samples,
                    oe.is_forward(),
                    tol,
                    &mut positions,
                );
            }
        }
    }
    Ok(positions)
}

/// Sample points along a parametric curve for area/distance calculations.
///
/// Uses open-endpoint sampling (`i / n_samples`, NOT `i / (n-1)`) so that
/// closed curves (full circles) do not duplicate the start/end point.
#[allow(clippy::cast_precision_loss)]
fn sample_edge_curve(
    evaluate: &dyn Fn(f64) -> Point3,
    t0: f64,
    t1: f64,
    n_samples: usize,
    forward: bool,
    tol: f64,
    positions: &mut Vec<Point3>,
) {
    let indices: Box<dyn Iterator<Item = usize>> = if forward {
        Box::new(0..n_samples)
    } else {
        Box::new((0..n_samples).rev())
    };
    for i in indices {
        let t = t0 + (t1 - t0) * (i as f64) / (n_samples as f64);
        let pt = evaluate(t);
        if positions
            .last()
            .is_none_or(|p: &Point3| (*p - pt).length() > tol)
        {
            positions.push(pt);
        }
    }
}

/// Compute the angular range `(u_start, u_end)` from a set of projected u values.
///
/// Detects the largest angular gap and treats it as the boundary between the
/// face's angular extent. For full revolutions (no significant gap), returns
/// `(0, 2*pi)`.
pub(super) fn compute_angular_range(u_vals: &mut Vec<f64>) -> (f64, f64) {
    use std::f64::consts::TAU;
    let tol_lin = brepkit_math::tolerance::Tolerance::default().linear;

    u_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    u_vals.dedup_by(|a, b| (*a - *b).abs() < tol_lin);

    if u_vals.len() < 3 {
        return (0.0, TAU);
    }

    let mut max_gap = 0.0_f64;
    let mut gap_end_idx = 0_usize;
    for i in 0..u_vals.len() {
        let j = (i + 1) % u_vals.len();
        let gap = if j > i {
            u_vals[j] - u_vals[i]
        } else {
            u_vals[j] + TAU - u_vals[i]
        };
        if gap > max_gap {
            max_gap = gap;
            gap_end_idx = j;
        }
    }
    let n_angles = u_vals.len() as f64;
    let even_gap = TAU / n_angles;
    let gap_threshold = (2.5 * even_gap).min(TAU / 3.0);
    if max_gap < gap_threshold {
        (0.0, TAU)
    } else {
        let u_start = u_vals[gap_end_idx];
        let gap_start_idx = if gap_end_idx == 0 {
            u_vals.len() - 1
        } else {
            gap_end_idx - 1
        };
        let u_end = u_vals[gap_start_idx];
        if u_end > u_start {
            (u_start, u_end)
        } else {
            (u_start, u_end + TAU)
        }
    }
}
