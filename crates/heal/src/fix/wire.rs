//! Wire fixing — reorder, close gaps, remove degenerate/small edges.
//!
//! The fix sequence follows a standard wire-repair pipeline:
//!
//! 1. Reorder edges to form a connected chain
//! 2. Close gaps between consecutive edges (merge near vertices)
//! 3. Ensure wire closure
//! 4. Remove small edges (shorter than tolerance)
//! 5. Remove degenerate edges (closed + zero length)
//! 6. Close 3D gaps (delegates to connectivity fix)
//! 7. Remove trailing short edges
//!
//! Stubs exist for self-intersection, lacking, notched, intersecting, seam.

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use super::FixResult;
use super::config::{FixConfig, FixMode};
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Fix a single wire: reorder, close gaps, remove degenerate/small edges.
///
/// Applies fixes in order of increasing invasiveness.  Each sub-fix
/// respects the corresponding [`FixMode`] in `config`.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
#[allow(clippy::too_many_lines)]
pub fn fix_wire(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    fix_wire_impl(topo, wire_id, None, ctx, config)
}

/// Overload that also accepts a face ID (used by `fix_face`).
///
/// The face ID enables PCurve-aware fixes (`fix_lacking`) and
/// periodic-surface fixes (`fix_missing_seam`).
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn fix_wire_on_face(
    topo: &mut Topology,
    wire_id: WireId,
    face_id: FaceId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    fix_wire_impl(topo, wire_id, Some(face_id), ctx, config)
}

/// Shared implementation for wire fixing with an optional face context.
#[allow(clippy::too_many_lines)]
fn fix_wire_impl(
    topo: &mut Topology,
    wire_id: WireId,
    face_id: Option<FaceId>,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let mut result = FixResult::ok();

    // 1. Reorder edges to form a connected chain.
    if config.fix_reorder != FixMode::Off {
        let r = fix_reorder(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 2. Close gaps between consecutive edges.
    if config.fix_connectivity != FixMode::Off {
        let r = fix_connected(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 3. Ensure wire closure.
    if config.fix_closure != FixMode::Off {
        let r = fix_closed(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 4. Remove small edges.
    if config.fix_small_edges != FixMode::Off {
        let r = fix_small(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 5. Remove degenerate edges (closed + zero length).
    if config.fix_degenerate_edges != FixMode::Off {
        let r = fix_degenerate(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 6. Close 3D gaps (delegates to connectivity fix).
    if config.fix_gaps_3d != FixMode::Off {
        let r = fix_gaps_3d(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 7. Remove trailing short edges.
    if config.fix_tail != FixMode::Off {
        let r = fix_tail(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 8. Self-intersection detection.
    if config.fix_self_intersection != FixMode::Off {
        let r = fix_self_intersection(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 9. Lacking (PCurve/3D divergence) fix.
    if config.fix_lacking != FixMode::Off {
        if let Some(fid) = face_id {
            let r = fix_lacking(topo, wire_id, fid, ctx, config)?;
            result.merge(&r);
        }
    }

    // 10. Notched edges fix.
    if config.fix_notched != FixMode::Off {
        let r = fix_notched(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 11. Intersecting edges fix.
    if config.fix_intersecting_edges != FixMode::Off {
        let r = fix_intersecting_edges(topo, wire_id, ctx, config)?;
        result.merge(&r);
    }

    // 12. Missing seam fix.
    if config.fix_missing_seam != FixMode::Off {
        if let Some(fid) = face_id {
            let r = fix_missing_seam(topo, wire_id, fid, ctx, config)?;
            result.merge(&r);
        }
    }

    Ok(result)
}

// ── Fix implementations ─────────────────────────────────────────────────

/// Reorder edges to form a connected chain.
///
/// Uses the greedy nearest-neighbour algorithm from
/// [`analysis::wire_order`](crate::analysis::wire_order).
fn fix_reorder(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let analysis = crate::analysis::wire::analyze_wire(topo, wire_id, &ctx.tolerance)?;
    let has_issue = !analysis.is_ordered;

    if !config.fix_reorder.should_fix(has_issue) {
        return Ok(FixResult::ok());
    }

    let order_result = crate::analysis::wire_order::compute_wire_order(topo, wire_id)?;

    // If already in order, nothing to do.
    let is_identity = order_result.order.iter().enumerate().all(|(i, &o)| o == i)
        && order_result.flips.iter().all(|&f| !f);
    if is_identity {
        return Ok(FixResult::ok());
    }

    // Snapshot: read original edges.
    let wire = topo.wire(wire_id)?;
    let old_edges: Vec<OrientedEdge> = wire.edges().to_vec();
    let is_closed = wire.is_closed();

    if old_edges.is_empty() {
        return Ok(FixResult::ok());
    }

    // Build reordered edge list.
    let mut new_edges = Vec::with_capacity(old_edges.len());
    for (idx, &orig_idx) in order_result.order.iter().enumerate() {
        let oe = old_edges[orig_idx];
        let flip = order_result.flips[idx];
        let forward = if flip {
            !oe.is_forward()
        } else {
            oe.is_forward()
        };
        new_edges.push(OrientedEdge::new(oe.edge(), forward));
    }

    // Allocate new wire.
    let new_wire = Wire::new(new_edges, is_closed)?;
    let new_wire_id = topo.add_wire(new_wire);
    ctx.reshape.replace_wire(wire_id, new_wire_id);

    ctx.info(format!(
        "Wire {wire_id:?}: reordered {count} edges (max_gap={gap:.2e})",
        count = old_edges.len(),
        gap = order_result.max_gap,
    ));

    Ok(FixResult {
        status: Status::DONE1,
        actions_taken: 1,
    })
}

/// Close gaps between consecutive edges by merging nearby vertices.
///
/// Uses `ctx.tolerance.linear` as the merge threshold. For widened-
/// tolerance gap closing, see [`fix_gaps_3d`] which retries with a
/// wider threshold when nominal-tolerance closing leaves gaps behind.
///
/// Ported from `operations::heal::close_wire_gaps`.
fn fix_connected(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let merge_tol = ctx.tolerance.linear;
    fix_connected_with_tol(topo, wire_id, ctx, config, merge_tol)
}

/// Parameterized gap-closing helper — used by both [`fix_connected`]
/// (nominal tolerance) and [`fix_gaps_3d`] (widened tolerance).
#[allow(clippy::needless_pass_by_ref_mut)]
fn fix_connected_with_tol(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
    merge_tol: f64,
) -> Result<FixResult, HealError> {
    let analysis = crate::analysis::wire::analyze_wire(topo, wire_id, &ctx.tolerance)?;
    let has_issue = !analysis.gaps.is_empty();

    if !config.fix_connectivity.should_fix(has_issue) {
        return Ok(FixResult::ok());
    }

    // Snapshot: read wire edges and gather vertex pairs to merge.
    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let n_edges = edges_list.len();

    if n_edges < 2 {
        return Ok(FixResult::ok());
    }

    let tol_sq = merge_tol * merge_tol;
    let mut merge_pairs: Vec<(VertexId, VertexId)> = Vec::new();

    // Snapshot all oriented endpoint vertex IDs.
    let mut end_vids = Vec::with_capacity(n_edges);
    let mut start_vids = Vec::with_capacity(n_edges);
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        end_vids.push(oe.oriented_end(edge));
        start_vids.push(oe.oriented_start(edge));
    }

    // Compare consecutive edge pairs (including wrap-around for closed wires).
    let pairs = if n_edges > 1 { n_edges } else { 0 };
    for i in 0..pairs {
        let next_i = (i + 1) % n_edges;
        if next_i == 0 && !topo.wire(wire_id)?.is_closed() {
            continue; // Open wire: don't close last→first.
        }

        let end_vid = end_vids[i];
        let start_vid = start_vids[next_i];

        if end_vid == start_vid {
            continue; // Already connected.
        }

        let end_pos = topo.vertex(end_vid)?.point();
        let start_pos = topo.vertex(start_vid)?.point();
        let dist_sq = (end_pos - start_pos).length_squared();

        if dist_sq < tol_sq {
            merge_pairs.push((start_vid, end_vid)); // merge start_vid into end_vid
        }
    }

    if merge_pairs.is_empty() {
        return Ok(FixResult::ok());
    }

    let gaps_closed = merge_pairs.len();

    // Record vertex replacements in reshape.
    for (from_vid, to_vid) in &merge_pairs {
        ctx.reshape.replace_vertex(*from_vid, *to_vid);
    }

    ctx.info(format!(
        "Wire {wire_id:?}: closed {gaps_closed} gap(s) by merging vertices",
    ));

    Ok(FixResult {
        status: Status::DONE2,
        actions_taken: gaps_closed,
    })
}

/// Ensure wire closure: if the wire is flagged as closed but the last
/// edge's end doesn't connect to the first edge's start, merge them.
#[allow(clippy::needless_pass_by_ref_mut)]
fn fix_closed(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let wire = topo.wire(wire_id)?;
    if !wire.is_closed() {
        return Ok(FixResult::ok());
    }

    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let n = edges_list.len();
    if n == 0 {
        return Ok(FixResult::ok());
    }

    // Get last edge's oriented end and first edge's oriented start.
    let last_edge = topo.edge(edges_list[n - 1].edge())?;
    let first_edge = topo.edge(edges_list[0].edge())?;
    let last_end = edges_list[n - 1].oriented_end(last_edge);
    let first_start = edges_list[0].oriented_start(first_edge);

    if last_end == first_start {
        return Ok(FixResult::ok()); // Already closed.
    }

    let end_pos = topo.vertex(last_end)?.point();
    let start_pos = topo.vertex(first_start)?.point();
    let dist = (end_pos - start_pos).length();

    let has_issue = dist <= ctx.tolerance.linear;

    if !config.fix_closure.should_fix(has_issue) {
        return Ok(FixResult::ok());
    }

    if dist > ctx.tolerance.linear {
        ctx.warn(format!(
            "Wire {wire_id:?}: closure gap {dist:.2e} exceeds tolerance, cannot close",
        ));
        return Ok(FixResult {
            status: Status::FAIL1,
            actions_taken: 0,
        });
    }

    // Merge first_start into last_end to close the wire.
    ctx.reshape.replace_vertex(first_start, last_end);

    ctx.info(format!(
        "Wire {wire_id:?}: closed closure gap of {dist:.2e}",
    ));

    Ok(FixResult {
        status: Status::DONE3,
        actions_taken: 1,
    })
}

/// Remove edges shorter than tolerance and merge their vertices.
#[allow(clippy::needless_pass_by_ref_mut)]
fn fix_small(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let analysis = crate::analysis::wire::analyze_wire(topo, wire_id, &ctx.tolerance)?;
    let has_issue = !analysis.small_edges.is_empty();

    if !config.fix_small_edges.should_fix(has_issue) {
        return Ok(FixResult::ok());
    }

    let mut removed = 0;
    for se in &analysis.small_edges {
        // Snapshot edge data before modifying.
        let edge = topo.edge(se.edge_id)?;
        let start_vid = edge.start();
        let end_vid = edge.end();

        // Don't remove closed edges here — that's handled by fix_degenerate.
        if start_vid == end_vid {
            continue;
        }

        // Merge end vertex into start vertex and remove the edge.
        ctx.reshape.replace_vertex(end_vid, start_vid);
        ctx.reshape.remove_edge(se.edge_id);
        removed += 1;

        ctx.info(format!(
            "Wire {wire_id:?}: removed small edge {:?} (length={:.2e})",
            se.edge_id, se.length,
        ));
    }

    if removed == 0 {
        return Ok(FixResult::ok());
    }

    Ok(FixResult {
        status: Status::DONE4,
        actions_taken: removed,
    })
}

/// Remove degenerate edges: closed (start == end) with zero-length curve.
///
/// Ported from `operations::heal::remove_degenerate_edges`.
fn fix_degenerate(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let analysis = crate::analysis::wire::analyze_wire(topo, wire_id, &ctx.tolerance)?;
    let has_issue = !analysis.degenerate_edges.is_empty();

    if !config.fix_degenerate_edges.should_fix(has_issue) {
        return Ok(FixResult::ok());
    }

    // Snapshot: read the wire's edge list.
    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let is_closed = wire.is_closed();

    let mut new_edges = Vec::with_capacity(edges_list.len());
    let mut removed = 0;

    for (i, oe) in edges_list.iter().enumerate() {
        if analysis.degenerate_edges.contains(&i) {
            ctx.reshape.remove_edge(oe.edge());
            removed += 1;
            ctx.info(format!(
                "Wire {wire_id:?}: removed degenerate edge {:?} at index {i}",
                oe.edge(),
            ));
        } else {
            new_edges.push(*oe);
        }
    }

    if removed == 0 {
        return Ok(FixResult::ok());
    }

    // Rebuild wire without degenerate edges, if any remain.
    if !new_edges.is_empty() {
        let new_wire = Wire::new(new_edges, is_closed)?;
        let new_wire_id = topo.add_wire(new_wire);
        ctx.reshape.replace_wire(wire_id, new_wire_id);
    }

    Ok(FixResult {
        status: Status::DONE5,
        actions_taken: removed,
    })
}

/// Close 3D gaps between consecutive edges, including gaps wider than
/// the nominal `ctx.tolerance.linear` that [`fix_connected`] would not
/// close.
///
/// # Algorithm
///
/// 1. First attempt at `ctx.tolerance.linear`.
/// 2. If `analyze_wire` still reports gaps after step 1, recompute the
///    largest residual gap and retry once at a widened tolerance:
///    `widened = min(2 * largest_gap, 100 * linear, MAX_GAP_3D_BOUND)`.
///    The 2× safety factor on `largest_gap` ensures we close the
///    measured gap; the `100 * linear` cap prevents runaway widening
///    on degenerate inputs; the absolute upper bound
///    [`MAX_GAP_3D_BOUND`] caps catastrophic merges in mm-scale
///    geometry.
/// 3. If gaps remain after the widened pass, return the partial
///    result (caller / pipeline can decide whether to escalate).
fn fix_gaps_3d(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let mut result = fix_connected(topo, wire_id, ctx, config)?;

    // Re-analyze: did nominal-tolerance pass close every gap?
    let post = crate::analysis::wire::analyze_wire(topo, wire_id, &ctx.tolerance)?;
    if post.gaps.is_empty() {
        return Ok(result);
    }

    let largest_gap = post.gaps.iter().map(|g| g.distance).fold(0.0_f64, f64::max);

    let nominal = ctx.tolerance.linear;
    let widened = (2.0 * largest_gap)
        .min(100.0 * nominal)
        .min(MAX_GAP_3D_BOUND);
    if widened <= nominal {
        // Nothing more we can do — the residual gaps are below our
        // floor or the bounded widened tolerance doesn't actually
        // exceed the nominal one.
        return Ok(result);
    }

    ctx.info(format!(
        "Wire {wire_id:?}: retrying gap close at widened tolerance \
         {widened:.2e} (nominal={nominal:.2e}, largest_gap={largest_gap:.2e})",
    ));

    let widened_result = fix_connected_with_tol(topo, wire_id, ctx, config, widened)?;
    result.merge(&widened_result);
    Ok(result)
}

/// Absolute upper bound on the widened tolerance used for 3D gap
/// closing. 1mm is a sensible cap for STEP/IGES imports of mm-scale
/// CAD parts — gaps larger than this typically indicate a real
/// modeling error, not numerical drift.
const MAX_GAP_3D_BOUND: f64 = 1e-3;

/// Remove short trailing edges at wire ends (open wires only).
fn fix_tail(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let wire = topo.wire(wire_id)?;
    if wire.is_closed() {
        return Ok(FixResult::ok()); // Tail fix only applies to open wires.
    }

    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let n = edges_list.len();
    if n < 2 {
        return Ok(FixResult::ok());
    }

    let tol = ctx.tolerance.linear;

    // Check trailing edge (last edge).
    let last_oe = edges_list[n - 1];
    let last_edge = topo.edge(last_oe.edge())?;
    let last_start = topo.vertex(last_edge.start())?.point();
    let last_end = topo.vertex(last_edge.end())?.point();
    let last_len = (last_end - last_start).length();

    let has_tail = last_len < tol;

    if !config.fix_tail.should_fix(has_tail) {
        return Ok(FixResult::ok());
    }

    if !has_tail {
        return Ok(FixResult::ok());
    }

    // Remove trailing edge.
    let new_edges: Vec<OrientedEdge> = edges_list[..n - 1].to_vec();
    if new_edges.is_empty() {
        return Ok(FixResult::ok());
    }

    let new_wire = Wire::new(new_edges, false)?;
    let new_wire_id = topo.add_wire(new_wire);
    ctx.reshape.replace_wire(wire_id, new_wire_id);
    ctx.reshape.remove_edge(last_oe.edge());

    ctx.info(format!(
        "Wire {wire_id:?}: removed trailing edge {:?} (length={last_len:.2e})",
        last_oe.edge(),
    ));

    Ok(FixResult {
        status: Status::DONE6,
        actions_taken: 1,
    })
}

// ── Wire fix implementations ─────────────────────────────────────────

/// Detect self-intersections in the wire by checking non-adjacent edge
/// pairs for polyline crossings.
///
/// Currently detection-only: crossings are reported via `ctx.warn()`
/// but no topology modifications are made.
#[allow(clippy::too_many_lines, clippy::needless_pass_by_ref_mut)]
fn fix_self_intersection(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    // Snapshot wire edges.
    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let n = edges_list.len();

    if n < 3 {
        // Need at least 3 edges for non-adjacent pairs.
        return Ok(FixResult::ok());
    }

    // Sample each edge to a polyline of 10 points.
    let num_samples = 10;
    let mut polylines: Vec<Vec<brepkit_math::vec::Point3>> = Vec::with_capacity(n);
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let start_pos = topo.vertex(oe.oriented_start(edge))?.point();
        let end_pos = topo.vertex(oe.oriented_end(edge))?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
        let mut pts = Vec::with_capacity(num_samples);
        for s in 0..num_samples {
            #[allow(clippy::cast_precision_loss)]
            let t = t0 + (t1 - t0) * (s as f64) / ((num_samples - 1) as f64);
            pts.push(edge.curve().evaluate_with_endpoints(t, start_pos, end_pos));
        }
        polylines.push(pts);
    }

    // Find the dominant projection plane from all points.
    let dominant_axis = find_dominant_axis(&polylines);

    let mut crossings_found = 0usize;

    // Check all non-adjacent edge pairs.
    for i in 0..n {
        for j in (i + 2)..n {
            // Skip adjacent pair (last, first) in closed wires.
            if j == n - 1 && i == 0 {
                continue;
            }
            if segments_cross_2d(&polylines[i], &polylines[j], dominant_axis) {
                crossings_found += 1;
                ctx.warn(format!(
                    "Wire {wire_id:?}: self-intersection detected between edges \
                     {i} and {j}",
                ));
            }
        }
    }

    if !config.fix_self_intersection.should_fix(crossings_found > 0) {
        return Ok(FixResult::ok());
    }

    if crossings_found > 0 {
        Ok(FixResult {
            status: Status::DONE1,
            actions_taken: crossings_found,
        })
    } else {
        Ok(FixResult::ok())
    }
}

/// Find the axis with the least variance across all sampled points,
/// returning `0` (project onto YZ), `1` (XZ), or `2` (XY).
fn find_dominant_axis(polylines: &[Vec<brepkit_math::vec::Point3>]) -> usize {
    use brepkit_math::vec::Vec3;

    // Compute centroid.
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    let mut count = 0usize;
    for pts in polylines {
        for p in pts {
            sum += Vec3::new(p.x(), p.y(), p.z());
            count += 1;
        }
    }
    if count == 0 {
        return 2;
    }
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f64;
    let cx = sum.x() * inv;
    let cy = sum.y() * inv;
    let cz = sum.z() * inv;

    // Compute variance per axis.
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    let mut vz = 0.0_f64;
    for pts in polylines {
        for p in pts {
            let dx = p.x() - cx;
            let dy = p.y() - cy;
            let dz = p.z() - cz;
            vx += dx * dx;
            vy += dy * dy;
            vz += dz * dz;
        }
    }

    // The axis with least variance is the "normal" direction; project
    // onto the plane perpendicular to it.
    if vx <= vy && vx <= vz {
        0 // Project onto YZ plane
    } else if vy <= vz {
        1 // Project onto XZ plane
    } else {
        2 // Project onto XY plane
    }
}

/// Check if any segment in polyline `a` crosses any segment in polyline `b`
/// when projected onto a 2D plane (dropping the `drop_axis` coordinate).
fn segments_cross_2d(
    a: &[brepkit_math::vec::Point3],
    b: &[brepkit_math::vec::Point3],
    drop_axis: usize,
) -> bool {
    let project = |p: &brepkit_math::vec::Point3| -> (f64, f64) {
        match drop_axis {
            0 => (p.y(), p.z()),
            1 => (p.x(), p.z()),
            _ => (p.x(), p.y()),
        }
    };

    for ai in 0..a.len().saturating_sub(1) {
        let (ax1, ay1) = project(&a[ai]);
        let (ax2, ay2) = project(&a[ai + 1]);
        for bi in 0..b.len().saturating_sub(1) {
            let (bx1, by1) = project(&b[bi]);
            let (bx2, by2) = project(&b[bi + 1]);
            if segments_intersect_2d(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) {
                return true;
            }
        }
    }
    false
}

/// Test if two 2D line segments (p1-p2) and (p3-p4) properly cross.
#[allow(clippy::too_many_arguments)]
fn segments_intersect_2d(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
    x4: f64,
    y4: f64,
) -> bool {
    let d1 = cross_2d(x3, y3, x4, y4, x1, y1);
    let d2 = cross_2d(x3, y3, x4, y4, x2, y2);
    let d3 = cross_2d(x1, y1, x2, y2, x3, y3);
    let d4 = cross_2d(x1, y1, x2, y2, x4, y4);

    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    false
}

/// 2D cross product: sign of `(b - a) x (c - a)`.
fn cross_2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Fix edges where the 3D curve endpoints diverge from vertex positions.
///
/// For each edge, checks that the curve evaluated at its domain bounds
/// matches the bounding vertex positions.  If the deviation exceeds
/// tolerance, adjusts the vertex position to match the curve endpoint.
#[allow(clippy::needless_pass_by_ref_mut)]
fn fix_lacking(
    topo: &mut Topology,
    wire_id: WireId,
    _face_id: FaceId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    // Snapshot edge data: for each edge, store (edge_id, curve_type,
    // start_vid, end_vid, start_pos, end_pos, domain).
    struct EdgeSnapshot {
        start_vid: VertexId,
        end_vid: VertexId,
        start_pos: brepkit_math::vec::Point3,
        end_pos: brepkit_math::vec::Point3,
        curve_start: brepkit_math::vec::Point3,
        curve_end: brepkit_math::vec::Point3,
    }

    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let mut snapshots = Vec::with_capacity(edges_list.len());
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let start_vid = edge.start();
        let end_vid = edge.end();
        let start_pos = topo.vertex(start_vid)?.point();
        let end_pos = topo.vertex(end_vid)?.point();

        // For Line edges, endpoints are determined by vertices, so skip.
        if matches!(edge.curve(), EdgeCurve::Line) {
            continue;
        }

        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
        let curve_start = edge.curve().evaluate_with_endpoints(t0, start_pos, end_pos);
        let curve_end = edge.curve().evaluate_with_endpoints(t1, start_pos, end_pos);

        snapshots.push(EdgeSnapshot {
            start_vid,
            end_vid,
            start_pos,
            end_pos,
            curve_start,
            curve_end,
        });
    }

    let tol = ctx.tolerance.linear;
    let mut adjusted = 0usize;

    for snap in &snapshots {
        let start_dev = (snap.start_pos - snap.curve_start).length();
        if start_dev > tol {
            // Create a new vertex at the corrected position and replace via reshape.
            let new_vid =
                topo.add_vertex(brepkit_topology::vertex::Vertex::new(snap.curve_start, tol));
            ctx.reshape.replace_vertex(snap.start_vid, new_vid);
            adjusted += 1;
            ctx.info(format!(
                "Wire {wire_id:?}: adjusted start vertex {:?} by {start_dev:.2e}",
                snap.start_vid,
            ));
        }

        let end_dev = (snap.end_pos - snap.curve_end).length();
        if end_dev > tol {
            let new_vid =
                topo.add_vertex(brepkit_topology::vertex::Vertex::new(snap.curve_end, tol));
            ctx.reshape.replace_vertex(snap.end_vid, new_vid);
            adjusted += 1;
            ctx.info(format!(
                "Wire {wire_id:?}: adjusted end vertex {:?} by {end_dev:.2e}",
                snap.end_vid,
            ));
        }
    }

    if !config.fix_lacking.should_fix(adjusted > 0) {
        return Ok(FixResult::ok());
    }

    if adjusted > 0 {
        Ok(FixResult {
            status: Status::DONE1,
            actions_taken: adjusted,
        })
    } else {
        Ok(FixResult::ok())
    }
}

/// Remove cusp/notched edges: consecutive edges whose tangents reverse.
///
/// When two consecutive edges form a cusp (tangent dot product < -0.9)
/// and the shorter edge is below tolerance length, the shorter edge is
/// removed and its vertices are merged.
#[allow(clippy::too_many_lines, clippy::needless_pass_by_ref_mut)]
fn fix_notched(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    // Snapshot edge data for cusp detection.
    struct EdgeInfo {
        oe: OrientedEdge,
        start_vid: VertexId,
        end_vid: VertexId,
        length: f64,
        end_tangent: brepkit_math::vec::Vec3,
        start_tangent: brepkit_math::vec::Vec3,
    }

    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let n = edges_list.len();

    if n < 2 {
        return Ok(FixResult::ok());
    }

    let mut infos = Vec::with_capacity(n);
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let raw_start = topo.vertex(edge.start())?.point();
        let raw_end = topo.vertex(edge.end())?.point();
        let start_vid = oe.oriented_start(edge);
        let end_vid = oe.oriented_end(edge);
        let o_start_pos = topo.vertex(start_vid)?.point();
        let o_end_pos = topo.vertex(end_vid)?.point();
        let length = (o_end_pos - o_start_pos).length();

        let (t0, t1) = edge.curve().domain_with_endpoints(raw_start, raw_end);
        // Tangent at the oriented start and end.
        let (ts, te) = if oe.is_forward() {
            (
                edge.curve().tangent_with_endpoints(t0, raw_start, raw_end),
                edge.curve().tangent_with_endpoints(t1, raw_start, raw_end),
            )
        } else {
            // Reversed edge: flip tangent directions.
            let t_end = edge.curve().tangent_with_endpoints(t0, raw_start, raw_end);
            let t_start = edge.curve().tangent_with_endpoints(t1, raw_start, raw_end);
            (t_start * -1.0, t_end * -1.0)
        };

        infos.push(EdgeInfo {
            oe: *oe,
            start_vid,
            end_vid,
            length,
            end_tangent: te,
            start_tangent: ts,
        });
    }

    let tol = ctx.tolerance.linear;
    let mut cusps_found = 0usize;
    let mut edges_to_remove = std::collections::HashSet::new();

    // Check consecutive pairs for cusps.
    let pairs = if topo.wire(wire_id)?.is_closed() {
        n
    } else {
        n - 1
    };
    for i in 0..pairs {
        let j = (i + 1) % n;
        if edges_to_remove.contains(&i) || edges_to_remove.contains(&j) {
            continue;
        }

        let end_tan = infos[i].end_tangent;
        let start_tan = infos[j].start_tangent;

        // Normalize tangents for dot product.
        let end_len = end_tan.length();
        let start_len = start_tan.length();
        if end_len < 1e-15 || start_len < 1e-15 {
            continue;
        }
        let dot = end_tan.dot(start_tan) / (end_len * start_len);

        if dot < -0.9 {
            // Cusp detected. Remove the shorter edge if below tolerance.
            let (remove_idx, keep_idx) = if infos[i].length <= infos[j].length {
                (i, j)
            } else {
                (j, i)
            };

            if infos[remove_idx].length < tol {
                edges_to_remove.insert(remove_idx);
                // Merge vertices of the removed edge.
                let from = infos[remove_idx].end_vid;
                let to = infos[keep_idx].start_vid;
                if from != to {
                    ctx.reshape.replace_vertex(from, to);
                }
                ctx.reshape.remove_edge(infos[remove_idx].oe.edge());
                cusps_found += 1;
                ctx.info(format!(
                    "Wire {wire_id:?}: removed notched edge at index {remove_idx} \
                     (cusp dot={dot:.3}, length={:.2e})",
                    infos[remove_idx].length,
                ));
            } else {
                ctx.warn(format!(
                    "Wire {wire_id:?}: cusp detected at edge {i}/{j} (dot={dot:.3}) \
                     but shorter edge length {:.2e} exceeds tolerance",
                    infos[remove_idx].length,
                ));
            }
        }
    }

    if !config.fix_notched.should_fix(cusps_found > 0) {
        return Ok(FixResult::ok());
    }

    if cusps_found > 0 {
        Ok(FixResult {
            status: Status::DONE1,
            actions_taken: cusps_found,
        })
    } else {
        Ok(FixResult::ok())
    }
}

/// Detect crossings between adjacent (consecutive) edges.
///
/// Samples each pair of consecutive edges densely and checks for
/// polyline crossings. Currently detection-only: crossings are
/// reported via `ctx.warn()`.
#[allow(clippy::needless_pass_by_ref_mut)]
fn fix_intersecting_edges(
    topo: &mut Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let n = edges_list.len();

    if n < 2 {
        return Ok(FixResult::ok());
    }

    // Sample each edge to a polyline of 20 points.
    let num_samples = 20;
    let mut polylines: Vec<Vec<brepkit_math::vec::Point3>> = Vec::with_capacity(n);
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let start_pos = topo.vertex(oe.oriented_start(edge))?.point();
        let end_pos = topo.vertex(oe.oriented_end(edge))?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
        let mut pts = Vec::with_capacity(num_samples);
        for s in 0..num_samples {
            #[allow(clippy::cast_precision_loss)]
            let t = t0 + (t1 - t0) * (s as f64) / ((num_samples - 1) as f64);
            pts.push(edge.curve().evaluate_with_endpoints(t, start_pos, end_pos));
        }
        polylines.push(pts);
    }

    let dominant_axis = find_dominant_axis(&polylines);
    let mut crossings_found = 0usize;

    // Check consecutive edge pairs (excluding the shared vertex endpoints).
    let pairs = if topo.wire(wire_id)?.is_closed() {
        n
    } else {
        n - 1
    };
    for i in 0..pairs {
        let j = (i + 1) % n;
        // Skip the first segment of edge j and last segment of edge i
        // (they share a vertex).
        let a = &polylines[i];
        let b = &polylines[j];

        if a.len() < 3 || b.len() < 3 {
            continue;
        }

        // Check interior segments only (skip last seg of a, first seg of b).
        let a_interior = &a[..a.len() - 1];
        let b_interior = &b[1..];

        if segments_cross_2d(a_interior, b_interior, dominant_axis) {
            crossings_found += 1;
            ctx.warn(format!(
                "Wire {wire_id:?}: crossing detected between adjacent edges {i} and {j}",
            ));
        }
    }

    if !config
        .fix_intersecting_edges
        .should_fix(crossings_found > 0)
    {
        return Ok(FixResult::ok());
    }

    if crossings_found > 0 {
        Ok(FixResult {
            status: Status::DONE1,
            actions_taken: crossings_found,
        })
    } else {
        Ok(FixResult::ok())
    }
}

/// Detect missing seam edges on periodic surfaces.
///
/// Checks if the face's surface is periodic (Cylinder, Cone, Sphere,
/// Torus) and whether the wire spans most of the period without a
/// seam edge. Currently detection-only: missing seams are reported
/// via `ctx.warn()`.
#[allow(clippy::needless_pass_by_ref_mut)]
fn fix_missing_seam(
    topo: &mut Topology,
    wire_id: WireId,
    face_id: FaceId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    use std::f64::consts::TAU;

    // Check if the face surface is periodic.
    let face = topo.face(face_id)?;
    let surface = face.surface().clone();

    let period = match &surface {
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => TAU,
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => {
            // Not periodic.
            return Ok(FixResult::ok());
        }
    };

    // Sample wire edge endpoints and project onto the surface to get
    // the u-parameter range.
    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();

    if edges_list.is_empty() {
        return Ok(FixResult::ok());
    }

    let mut u_min = f64::MAX;
    let mut u_max = f64::MIN;

    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let start_pos = topo.vertex(oe.oriented_start(edge))?.point();
        let end_pos = topo.vertex(oe.oriented_end(edge))?.point();

        for pos in [start_pos, end_pos] {
            if let Some((u, _v)) = surface.project_point(pos) {
                if u < u_min {
                    u_min = u;
                }
                if u > u_max {
                    u_max = u;
                }
            }
        }
    }

    if u_min >= u_max {
        return Ok(FixResult::ok());
    }

    let u_span = u_max - u_min;
    let threshold = 0.9 * period;

    // Check if wire has a seam edge (an edge whose start and end project
    // to very different u values, spanning most of the period).
    let mut has_seam = false;
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let start_pos = topo.vertex(oe.oriented_start(edge))?.point();
        let end_pos = topo.vertex(oe.oriented_end(edge))?.point();

        if let (Some((u_s, _)), Some((u_e, _))) = (
            surface.project_point(start_pos),
            surface.project_point(end_pos),
        ) {
            // A seam edge has endpoints at roughly u=0 and u=2*pi.
            let du = (u_s - u_e).abs();
            if du > threshold {
                has_seam = true;
                break;
            }
        }
    }

    let has_issue = u_span > threshold && !has_seam;

    if !config.fix_missing_seam.should_fix(has_issue) {
        return Ok(FixResult::ok());
    }

    if has_issue {
        ctx.warn(format!(
            "Wire {wire_id:?} on face {face_id:?}: wire spans {u_span:.3} \
             of period {period:.3} but has no seam edge \
             (seam insertion not yet implemented)",
        ));
        Ok(FixResult {
            status: Status::DONE1,
            actions_taken: 0,
        })
    } else {
        Ok(FixResult::ok())
    }
}

/// Merge coincident vertices in a wire (helper used by connectivity fixes).
///
/// Finds vertex pairs within tolerance and records replacements in the
/// reshape context.  Returns the number of merges.
#[allow(dead_code)]
fn fix_coincident_vertices(
    topo: &Topology,
    wire_id: WireId,
    ctx: &mut HealContext,
) -> Result<usize, HealError> {
    let wire = topo.wire(wire_id)?;
    let edges_list = wire.edges();
    let tol_sq = ctx.tolerance.linear * ctx.tolerance.linear;

    // Collect unique vertex IDs and positions from this wire.
    let mut vertex_ids: Vec<VertexId> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for oe in edges_list {
        let edge = topo.edge(oe.edge())?;
        for vid in [edge.start(), edge.end()] {
            if seen.insert(vid.index()) {
                vertex_ids.push(vid);
            }
        }
    }

    let mut merged = 0usize;
    let n = vertex_ids.len();

    // Snapshot all positions.
    let positions: Vec<_> = vertex_ids
        .iter()
        .map(|&vid| -> Result<_, HealError> { Ok(topo.vertex(vid)?.point()) })
        .collect::<Result<Vec<_>, _>>()?;

    // O(n^2) pairwise comparison — wires are small.
    let mut already_merged = std::collections::HashSet::new();
    for i in 0..n {
        if already_merged.contains(&i) {
            continue;
        }
        for j in (i + 1)..n {
            if already_merged.contains(&j) {
                continue;
            }
            let dist_sq = (positions[i] - positions[j]).length_squared();
            if dist_sq < tol_sq {
                ctx.reshape.replace_vertex(vertex_ids[j], vertex_ids[i]);
                already_merged.insert(j);
                merged += 1;
            }
        }
    }

    Ok(merged)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::context::HealContext;
    use crate::fix::config::FixConfig;
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::Edge;
    use brepkit_topology::vertex::Vertex;

    /// Build an open wire of two collinear edges with a gap of `gap`
    /// between the end of edge1 and the start of edge2 (along +x).
    /// Returns the wire id and the inner gap-vertex IDs (start_v2 then
    /// end_v1) so tests can assert on reshape state.
    fn make_two_edges_with_gap(topo: &mut Topology, gap: f64) -> (WireId, VertexId, VertexId) {
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        // Both edges have line geometry (EdgeCurve::Line), endpoints
        // come from vertex positions.
        let v1_end = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2_start = topo.add_vertex(Vertex::new(Point3::new(1.0 + gap, 0.0, 0.0), 1e-7));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 0.0), 1e-7));

        let e1 = topo.add_edge(Edge::new(v0, v1_end, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2_start, v3, EdgeCurve::Line));

        let wire = Wire::new(
            vec![OrientedEdge::new(e1, true), OrientedEdge::new(e2, true)],
            false,
        )
        .expect("wire");
        let wid = topo.add_wire(wire);
        (wid, v2_start, v1_end)
    }

    #[test]
    fn fix_gaps_3d_widens_tolerance_to_close_gap_above_nominal() {
        // Gap ~5e-6 — above nominal linear tol (1e-7) but well below
        // the 1mm widening cap. fix_gaps_3d should close it via the
        // widened-tolerance second pass.
        let mut topo = Topology::new();
        let (wid, v_from, v_to) = make_two_edges_with_gap(&mut topo, 5e-6);

        let mut ctx = HealContext::new();
        let cfg = FixConfig::default();
        let result = fix_gaps_3d(&mut topo, wid, &mut ctx, &cfg).unwrap();

        // The widened pass should have merged the two endpoint vertices
        // (recorded in reshape).
        assert_eq!(
            ctx.reshape.resolve_vertex(v_from),
            v_to,
            "v_from should resolve to v_to after gap close"
        );
        assert!(result.actions_taken >= 1);
    }

    #[test]
    fn fix_gaps_3d_does_not_close_gap_above_max_bound() {
        // Gap of 5mm is above the 1mm MAX_GAP_3D_BOUND, so neither the
        // nominal pass nor the widened pass should close it. This
        // protects against runaway widening on real modeling errors.
        let mut topo = Topology::new();
        let (wid, v_from, _) = make_two_edges_with_gap(&mut topo, 5e-3);

        let mut ctx = HealContext::new();
        let cfg = FixConfig::default();
        let _ = fix_gaps_3d(&mut topo, wid, &mut ctx, &cfg).unwrap();

        // v_from should NOT have been merged (no replacement recorded).
        assert_eq!(
            ctx.reshape.resolve_vertex(v_from),
            v_from,
            "5mm gap is above MAX_GAP_3D_BOUND (1mm); should not be closed"
        );
    }
}
