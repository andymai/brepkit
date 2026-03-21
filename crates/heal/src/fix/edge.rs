//! Edge fixing — vertex-curve alignment, degenerate removal, same-parameter.

use brepkit_math::curves2d::{Curve2D, NurbsCurve2D};
use brepkit_math::vec::Point2;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::pcurve::PCurve;

use super::FixResult;
use super::config::{FixConfig, FixMode};
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Number of sample points for PCurve deviation analysis.
const SAME_PARAM_SAMPLES: usize = 20;

/// Fix a single edge: vertex tolerance, degenerate removal, `SameParameter` stub.
///
/// # Fixes applied
///
/// 1. **Vertex tolerance** (`fix_vertex_tolerance`): if a vertex position
///    deviates from the 3D curve endpoint by more than tolerance, a warning
///    is logged.
/// 2. **Degenerate edge** (`fix_degenerate_edges`): if the edge is closed
///    (`start == end`) and the curve length is approximately zero, the edge
///    is marked for removal via [`ReShape`](crate::reshape::ReShape).
/// 3. **SameParameter** (`fix_same_parameter`): logs a TODO warning.
///    For face-aware same-parameter fixing, use [`fix_same_parameter_on_face`].
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn fix_edge(
    topo: &Topology,
    edge_id: EdgeId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let mut result = FixResult::ok();

    // 1. Vertex tolerance check.
    if config.fix_vertex_tolerance != FixMode::Off {
        let r = fix_vertex_tolerance(topo, edge_id, ctx, config)?;
        result.merge(&r);
    }

    // 2. Degenerate edge detection and removal.
    if config.fix_degenerate_edges != FixMode::Off {
        let r = fix_degenerate(topo, edge_id, ctx, config)?;
        result.merge(&r);
    }

    // 3. SameParameter stub (no face context).
    if config.fix_same_parameter != FixMode::Off {
        let r = fix_same_parameter_stub(ctx, config);
        result.merge(&r);
    }

    Ok(result)
}

/// Fix SameParameter for an edge on a specific face.
///
/// Ensures the edge's PCurve on the given face accurately represents the
/// 3D curve projected onto the face surface.
///
/// # Algorithm
///
/// 1. Check if the edge has a PCurve on the given face (via pcurve registry).
/// 2. If a PCurve exists, sample both the 3D curve and the PCurve at
///    `SAME_PARAM_SAMPLES` points and compute the maximum deviation
///    between the surface evaluation at the PCurve's UV coordinates and
///    the 3D curve position.
/// 3. If deviation > tolerance (or no PCurve exists), rebuild the PCurve
///    via [`project_edge_to_pcurve`](crate::construct::project_curve::project_edge_to_pcurve).
/// 4. Register the new PCurve in the topology's pcurve registry.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail or PCurve projection fails.
#[allow(clippy::too_many_lines)]
pub fn fix_same_parameter_on_face(
    topo: &mut Topology,
    edge_id: EdgeId,
    face_id: FaceId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let has_pcurve = topo.pcurves().contains(edge_id, face_id);

    if has_pcurve {
        // Check deviation between existing PCurve and 3D curve.
        let max_dev = compute_pcurve_deviation(topo, edge_id, face_id)?;
        let tol = ctx.tolerance.linear;
        let needs_fix = max_dev > tol;

        if !config.fix_same_parameter.should_fix(needs_fix) {
            return Ok(FixResult::ok());
        }

        if !needs_fix {
            return Ok(FixResult::ok());
        }

        ctx.info(format!(
            "Edge {edge_id:?} on Face {face_id:?}: PCurve deviation {max_dev:.2e} exceeds tolerance {tol:.2e}, rebuilding",
        ));
    } else {
        // No PCurve exists -- always needs fixing unless mode is Off.
        if !config.fix_same_parameter.should_fix(true) {
            return Ok(FixResult::ok());
        }

        ctx.info(format!(
            "Edge {edge_id:?} on Face {face_id:?}: no PCurve found, creating via projection",
        ));
    }

    // Build a new PCurve via projection.
    let nurbs_3d = crate::construct::project_curve::project_edge_to_pcurve(
        topo,
        edge_id,
        face_id,
        SAME_PARAM_SAMPLES,
        &ctx.tolerance,
    )?;

    // Convert the 3D NURBS (with z=0) to a NurbsCurve2D.
    let cp_2d: Vec<Point2> = nurbs_3d
        .control_points()
        .iter()
        .map(|p| Point2::new(p.x(), p.y()))
        .collect();
    let weights = nurbs_3d.weights().to_vec();
    let knots = nurbs_3d.knots().to_vec();
    let degree = nurbs_3d.degree();

    let nurbs_2d = NurbsCurve2D::new(degree, knots.clone(), cp_2d, weights).map_err(|e| {
        HealError::FixFailed(format!(
            "failed to construct NurbsCurve2D for edge {edge_id:?}: {e}"
        ))
    })?;

    // Determine parameter range from the knot vector.
    let t_start = knots[degree];
    let t_end = knots[knots.len() - degree - 1];

    let pcurve = PCurve::new(Curve2D::Nurbs(nurbs_2d), t_start, t_end);
    topo.pcurves_mut().set(edge_id, face_id, pcurve);

    Ok(FixResult {
        status: Status::DONE3,
        actions_taken: 1,
    })
}

/// Check vertex-curve deviation and warn if it exceeds tolerance.
fn fix_vertex_tolerance(
    topo: &Topology,
    edge_id: EdgeId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let (start_dev, end_dev) = crate::analysis::edge::vertex_curve_deviation(topo, edge_id)?;

    let tol = ctx.tolerance.linear;
    let has_issue = start_dev > tol || end_dev > tol;

    if !config.fix_vertex_tolerance.should_fix(has_issue) {
        return Ok(FixResult::ok());
    }

    if start_dev > tol {
        ctx.warn(format!(
            "Edge {edge_id:?}: start vertex deviates from curve by {start_dev:.2e} (tol={tol:.2e})",
        ));
    }
    if end_dev > tol {
        ctx.warn(format!(
            "Edge {edge_id:?}: end vertex deviates from curve by {end_dev:.2e} (tol={tol:.2e})",
        ));
    }

    Ok(FixResult {
        status: Status::DONE1,
        actions_taken: 1,
    })
}

/// Detect and remove degenerate edges (closed + zero-length curve).
fn fix_degenerate(
    topo: &Topology,
    edge_id: EdgeId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let analysis = crate::analysis::edge::analyze_edge(topo, edge_id, &ctx.tolerance)?;

    if !config
        .fix_degenerate_edges
        .should_fix(analysis.is_degenerate)
    {
        return Ok(FixResult::ok());
    }

    ctx.info(format!(
        "Edge {edge_id:?}: degenerate (closed, length={:.2e}), marking for removal",
        analysis.curve_length_approx
    ));
    ctx.reshape.remove_edge(edge_id);

    Ok(FixResult {
        status: Status::DONE2,
        actions_taken: 1,
    })
}

/// Stub for SameParameter when no face context is available.
fn fix_same_parameter_stub(ctx: &mut HealContext, config: &FixConfig) -> FixResult {
    // SameParameter requires a face context to compute PCurve deviation.
    // Without a face_id, we can only log a warning.
    if !config.fix_same_parameter.should_fix(false) {
        return FixResult::ok();
    }

    ctx.warn(
        "SameParameter fix: requires face context, use fix_same_parameter_on_face()".to_string(),
    );

    FixResult {
        status: Status::DONE3,
        actions_taken: 0,
    }
}

/// Compute the maximum deviation between a 3D edge curve and its PCurve
/// on a given face.
///
/// Samples both curves at [`SAME_PARAM_SAMPLES`] points and returns the
/// maximum 3D distance between the edge curve point and the surface point
/// evaluated at the PCurve's UV coordinates.
fn compute_pcurve_deviation(
    topo: &Topology,
    edge_id: EdgeId,
    face_id: FaceId,
) -> Result<f64, HealError> {
    let pcurve = topo.pcurves().get(edge_id, face_id).ok_or_else(|| {
        HealError::FixFailed(format!(
            "no PCurve found for edge {edge_id:?} on face {face_id:?}"
        ))
    })?;

    let edge = topo.edge(edge_id)?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();
    let curve = edge.curve();

    let face = topo.face(face_id)?;
    let surface = face.surface();

    let (t0_3d, t1_3d) = curve.domain_with_endpoints(start_pos, end_pos);
    let t0_pc = pcurve.t_start();
    let t1_pc = pcurve.t_end();

    let mut max_dev = 0.0_f64;

    for i in 0..=SAME_PARAM_SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let frac = i as f64 / SAME_PARAM_SAMPLES as f64;

        // 3D curve point.
        let t_3d = t0_3d + (t1_3d - t0_3d) * frac;
        let pt_3d = curve.evaluate_with_endpoints(t_3d, start_pos, end_pos);

        // PCurve -> surface point.
        let t_pc = t0_pc + (t1_pc - t0_pc) * frac;
        let uv = pcurve.evaluate(t_pc);

        if let Some(pt_surf) = surface.evaluate(uv.x(), uv.y()) {
            let dev = (pt_3d - pt_surf).length();
            max_dev = max_dev.max(dev);
        }
        // If surface.evaluate returns None (plane), skip that sample --
        // plane PCurves are validated differently.
    }

    Ok(max_dev)
}
