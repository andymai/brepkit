//! 2D wire construction from edge soup via angular-sorting traversal.
//!
//! Given an unordered set of [`OrientedPCurveEdge`]s on a face, builds
//! minimal closed wire loops by traversing edges using the minimum
//! clockwise angle rule at each vertex.
//!
//! Uses the minimum clockwise angle traversal algorithm for face splitting.

use std::collections::HashMap;
use std::f64::consts::TAU;

use brepkit_math::vec::Point2;

use super::split_types::OrientedPCurveEdge;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Quantized UV endpoint pair, keying edges by their (start, end) node cells.
type QuantizedPair = ((i64, i64), (i64, i64));

/// An entry in the vertex adjacency map.
struct VertexEntry {
    /// Index into the input edge slice.
    edge_idx: usize,
    /// `true` if this vertex is the START of the edge (outgoing).
    outgoing: bool,
    /// Tangent angle at this vertex in \[0, 2pi).
    angle: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build closed wire loops from an unordered set of 2D edges.
///
/// Each edge has start/end UV coordinates. The algorithm:
/// 1. Builds a vertex adjacency map (quantized UV endpoints).
/// 2. Computes outgoing angles for each edge at each vertex.
/// 3. Traverses edges using the minimum clockwise angle rule.
/// 4. Returns a list of closed loops, each a `Vec<OrientedPCurveEdge>`.
///
/// `tol` is the UV-space tolerance for vertex deduplication.
/// `u_periodic` and `v_periodic` indicate whether the surface parameter wraps
/// (e.g. cylinder u in \[0, 2pi)). When `true`, UV coordinates are normalized
/// into \[0, period) before quantizing, and edge angles correct for seam-crossing.
pub fn build_wire_loops(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    u_periodic: bool,
    v_periodic: bool,
) -> Vec<Vec<OrientedPCurveEdge>> {
    build_wire_loops_with_winding(edges, tol, u_periodic, v_periodic, false)
}

/// [`build_wire_loops`] with explicit boundary winding.
///
/// The minimum-clockwise-angle rule produces minimal loops only when the
/// face boundary winds counter-clockwise in UV. For faces whose boundary
/// winds clockwise (legal — UV frames are built from the surface normal,
/// not the effective face orientation), pass `cw_boundary = true` to use
/// the mirrored minimum-counter-clockwise rule instead.
pub fn build_wire_loops_with_winding(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    u_periodic: bool,
    v_periodic: bool,
    cw_boundary: bool,
) -> Vec<Vec<OrientedPCurveEdge>> {
    if edges.is_empty() {
        return Vec::new();
    }

    let turn_score = |incoming: f64, candidate: f64| -> f64 {
        let cw = clockwise_angle(incoming, candidate);
        // `clockwise_angle` returns TAU for exact continuation, which must rank
        // as the worst (largest) score in both windings. The mirrored rule maps
        // every other angle to `TAU - cw` but would map a continuation to 0
        // (best), so keep it pinned at TAU.
        if cw_boundary {
            if cw >= TAU { TAU } else { TAU - cw }
        } else {
            cw
        }
    };

    let u_period = if u_periodic { Some(TAU) } else { None };
    let v_period = if v_periodic { Some(TAU) } else { None };

    // 1. Build vertex adjacency map.
    let mut adj: HashMap<(i64, i64), Vec<VertexEntry>> = HashMap::new();
    for (idx, edge) in edges.iter().enumerate() {
        let start_key = quantize_uv_periodic(edge.start_uv, tol, u_period, v_period);
        let end_key = quantize_uv_periodic(edge.end_uv, tol, u_period, v_period);
        let angle_out = edge_angle_at_vertex_periodic(edge, true, u_period, v_period);
        let angle_in = edge_angle_at_vertex_periodic(edge, false, u_period, v_period);
        adj.entry(start_key).or_default().push(VertexEntry {
            edge_idx: idx,
            outgoing: true,
            angle: angle_out,
        });
        adj.entry(end_key).or_default().push(VertexEntry {
            edge_idx: idx,
            outgoing: false,
            angle: angle_in,
        });
    }

    // 2. Greedy traversal.
    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();

    while let Some(start_idx) = used.iter().position(|u| !u) {
        used[start_idx] = true;

        let start_vertex = quantize_uv_periodic(edges[start_idx].start_uv, tol, u_period, v_period);
        let mut current_loop = vec![edges[start_idx].clone()];
        let mut current_idx = start_idx;

        // Walk edges until we close the loop.
        loop {
            let current_edge = &edges[current_idx];
            let end_vertex = quantize_uv_periodic(current_edge.end_uv, tol, u_period, v_period);

            // Check for loop closure: quantized keys must match AND the raw
            // 2D distance must be small. On periodic surfaces, seam-opposite
            // vertices quantize to the same key but have large UV distance
            // (~2pi). Check raw UV distance to reject seam-boundary false closures.
            let is_closed = if end_vertex == start_vertex {
                let raw_du = (current_edge.end_uv.x() - edges[start_idx].start_uv.x()).abs();
                let raw_dv = (current_edge.end_uv.y() - edges[start_idx].start_uv.y()).abs();
                let seam_threshold = std::f64::consts::PI;
                // Reject seam-boundary false closures: vertices on opposite sides
                // of a periodic seam quantize to the same key but have large UV distance.
                !(u_periodic && raw_du > seam_threshold || v_periodic && raw_dv > seam_threshold)
            } else {
                false
            };
            if is_closed {
                loops.push(current_loop);
                break;
            }

            // Incoming angle at the end vertex.
            let incoming_angle =
                edge_angle_at_vertex_periodic(current_edge, false, u_period, v_period);
            let arriving_start =
                quantize_uv_periodic(current_edge.start_uv, tol, u_period, v_period);

            // Find best outgoing edge at end_vertex.
            let Some(entries) = adj.get(&end_vertex) else {
                // Dead end -- discard this incomplete loop.
                break;
            };

            let mut best_cw = f64::MAX;
            let mut best_idx = None;
            for entry in entries {
                if !entry.outgoing || used[entry.edge_idx] {
                    continue;
                }
                // Skip the reverse of the arriving edge (prevents U-turns
                // along section edges that appear as forward+backward pairs).
                let candidate = &edges[entry.edge_idx];
                if quantize_uv_periodic(candidate.end_uv, tol, u_period, v_period) == arriving_start
                {
                    // Check if this is truly the reverse edge (same line, opposite direction).
                    let angle_diff = (entry.angle - incoming_angle).abs();
                    let is_reverse = angle_diff < 0.1 || (angle_diff - TAU).abs() < 0.1;
                    if is_reverse {
                        continue;
                    }
                }

                let cw = turn_score(incoming_angle, entry.angle);
                if cw < best_cw {
                    best_cw = cw;
                    best_idx = Some(entry.edge_idx);
                }
            }

            let Some(next_idx) = best_idx else {
                // No way out (all edges used or only reverse available).
                // Try again without the reverse-edge skip.
                let fallback = entries
                    .iter()
                    .filter(|e| e.outgoing && !used[e.edge_idx])
                    .min_by(|a, b| {
                        turn_score(incoming_angle, a.angle)
                            .partial_cmp(&turn_score(incoming_angle, b.angle))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some(fb) = fallback {
                    used[fb.edge_idx] = true;
                    current_loop.push(edges[fb.edge_idx].clone());
                    current_idx = fb.edge_idx;
                    continue;
                }
                // Truly no way out -- discard loop.
                break;
            };

            used[next_idx] = true;
            current_loop.push(edges[next_idx].clone());
            current_idx = next_idx;
        }
    }

    loops
}

/// Deterministic DCEL-style face trace over an already-junction-split edge
/// set: the canonical planar-subdivision face enumeration.
///
/// The greedy walker above consumes edges globally (`used`), so an early loop
/// can steal edges that belong to a later region and grand-tour several
/// regions as one (the lite diagonal-pad wall). Here the successor of a
/// half-edge is `rotational-next-of-twin` — a BIJECTION, so every half-edge
/// lies on exactly one clean face orbit regardless of enumeration order, and
/// near-parallel arrivals at tangential junctions cannot funnel into one
/// chain (the defect of a min-angle successor).
///
/// Sections already arrive as forward+reverse pairs; boundary edges get
/// SYNTHETIC reversed twins for the trace only. Faces that use a synthetic
/// half are the outer face (or touch an unpaired defect) and are dropped;
/// the remaining orbits are the interior partition. Requires the input
/// discipline the face splitter provides: junctions pre-split, endpoint UVs
/// reconciled.
pub fn build_wire_loops_dcel(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    u_periodic: bool,
    v_periodic: bool,
) -> Vec<Vec<OrientedPCurveEdge>> {
    if edges.is_empty() {
        return Vec::new();
    }
    let u_period = if u_periodic { Some(TAU) } else { None };
    let v_period = if v_periodic { Some(TAU) } else { None };

    // Half-edge table: the input edges, plus synthetic reversed twins for
    // every input half that has no reverse partner. `real` marks inputs.
    let mut halves: Vec<OrientedPCurveEdge> = edges.to_vec();
    let n_real = halves.len();
    let key_of = |p: Point2| quantize_uv_periodic(p, tol, u_period, v_period);

    // Twin pairing among the real halves: same unordered endpoint pair,
    // opposite direction. Prefer the same `source_edge_idx` (a section's own
    // fwd/rev pair); co-endpoint DISTINCT edges (the lens class) only pair
    // with each other if no same-source partner exists and both are free.
    let mut twin: Vec<Option<usize>> = vec![None; n_real];
    let mut by_pair: HashMap<QuantizedPair, Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        by_pair
            .entry((key_of(e.start_uv), key_of(e.end_uv)))
            .or_default()
            .push(i);
    }
    for (i, e) in edges.iter().enumerate() {
        if twin[i].is_some() {
            continue;
        }
        let rev_key = (key_of(e.end_uv), key_of(e.start_uv));
        if let Some(cands) = by_pair.get(&rev_key) {
            let pick = cands
                .iter()
                .copied()
                .filter(|&j| j != i && twin[j].is_none())
                .max_by_key(|&j| {
                    usize::from(
                        edges[j].source_edge_idx.is_some()
                            && edges[j].source_edge_idx == e.source_edge_idx,
                    )
                });
            if let Some(j) = pick {
                twin[i] = Some(j);
                twin[j] = Some(i);
            }
        }
    }
    // Synthetic reversed twins for the unpaired (boundary) halves.
    let mut twin_all: Vec<usize> = (0..n_real).map(|i| twin[i].unwrap_or(usize::MAX)).collect();
    for i in 0..n_real {
        if twin_all[i] != usize::MAX {
            continue;
        }
        let e = &edges[i];
        let mut r = e.clone();
        std::mem::swap(&mut r.start_uv, &mut r.end_uv);
        std::mem::swap(&mut r.start_3d, &mut r.end_3d);
        r.forward = !r.forward;
        let j = halves.len();
        halves.push(r);
        twin_all[i] = j;
        twin_all.push(i);
    }
    let n_all = halves.len();

    // Per-node circular order of OUTGOING halves by departure angle (ties
    // broken by index for a strict, deterministic order).
    let mut node_out: HashMap<(i64, i64), Vec<(f64, usize)>> = HashMap::new();
    for (i, h) in halves.iter().enumerate() {
        let ang = edge_angle_at_vertex_periodic(h, true, u_period, v_period);
        node_out
            .entry(key_of(h.start_uv))
            .or_default()
            .push((ang, i));
    }
    for list in node_out.values_mut() {
        list.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
    }

    // Successor: the outgoing half CLOCKWISE-next from twin(h) in the
    // circular order at h's end node (twin(h) starts there). Clockwise-next
    // in an angle-ascending list is the PREVIOUS entry, wrapping.
    let succ: Vec<usize> = (0..n_all)
        .map(|i| {
            let t = twin_all[i];
            let node = key_of(halves[t].start_uv);
            let list = &node_out[&node];
            let pos = list.iter().position(|&(_, j)| j == t).unwrap_or(0);
            let prev = (pos + list.len() - 1) % list.len();
            list[prev].1
        })
        .collect();

    // Orbits of the successor bijection = faces of the subdivision. The
    // single unbounded (outer) face is the orbit with the most NEGATIVE
    // signed area — drop it; every other orbit is an interior region. A
    // region may legitimately use synthetic boundary twins (a face whose
    // boundary winds clockwise in UV traverses its rim reversed), so
    // synthetic halves are emitted as reversed traversals, not filtered.
    let mut visited = vec![false; n_all];
    let mut orbits: Vec<(f64, Vec<usize>)> = Vec::new();
    for start in 0..n_all {
        if visited[start] {
            continue;
        }
        let mut orbit = Vec::new();
        let mut cur = start;
        let mut broken = false;
        loop {
            visited[cur] = true;
            orbit.push(cur);
            cur = succ[cur];
            if cur == start {
                break;
            }
            if visited[cur] {
                // Successor is a bijection, so this cannot happen unless the
                // node data was degenerate; discard defensively.
                broken = true;
                break;
            }
        }
        if broken {
            continue;
        }
        // Shoelace over LIFTED vertices: on a periodic surface the node keys
        // glue the seam (correct cylinder connectivity), but edge UVs are
        // stored at whichever period copy their pcurve produced — a raw
        // shoelace over mixed copies is garbage (the full-period remainder
        // band read as strongly negative and was dropped as "outer"). Lift
        // each successive vertex by the minimal image against its
        // predecessor; the net lift after closing the orbit is the WINDING —
        // nonzero only for the unbounded rim rings of a periodic face.
        let lift = |val: f64, anchor: f64, period: Option<f64>| -> f64 {
            match period {
                Some(p) if p > 1e-12 => val + ((anchor - val) / p).round() * p,
                _ => val,
            }
        };
        let mut area = 0.0;
        let mut perimeter = 0.0;
        let first = halves[orbit[0]].start_uv;
        let mut prev = first;
        let mut lifted_first = first;
        for k in 0..orbit.len() {
            let raw = halves[orbit[(k + 1) % orbit.len()]].start_uv;
            let b = Point2::new(
                lift(raw.x(), prev.x(), u_period),
                lift(raw.y(), prev.y(), v_period),
            );
            if k + 1 == orbit.len() {
                lifted_first = b;
            }
            area += prev.x().mul_add(b.y(), -(b.x() * prev.y()));
            perimeter += (b - prev).length();
            prev = b;
        }
        let area = area * 0.5;
        let winding_u = u_period.is_some_and(|p| (lifted_first.x() - first.x()).abs() > p * 0.5);
        let winding_v = v_period.is_some_and(|p| (lifted_first.y() - first.y()).abs() > p * 0.5);
        if (winding_u || winding_v) && orbit.iter().all(|&i| i >= n_real) {
            // A period-winding orbit of pure synthetic halves is an unbounded
            // rim ring (the periodic face's counterpart of the planar outer
            // face). A winding orbit WITH real halves is a genuine
            // full-period band region — a cylinder band's boundary rims are
            // separate rings in the glued graph, so the band itself winds.
            continue;
        }
        // A zero-area orbit walks out-and-back around a dangling chain (the
        // graph can carry genuinely unpaired sections); it is not a region.
        if area.abs() <= perimeter * tol {
            continue;
        }
        orbits.push((area, orbit));
    }
    // The planar unbounded face is the most-negative-area orbit. On a
    // u-periodic face the unbounded sides are the winding rim rings already
    // dropped above, so every surviving closed orbit is a genuine region.
    let outer = if u_periodic || v_periodic {
        None
    } else {
        orbits
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.0
                    .partial_cmp(&b.1.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    };
    orbits
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| Some(i) != outer)
        .map(|(_, (_, orbit))| orbit.iter().map(|&i| halves[i].clone()).collect())
        .collect()
}

/// Drop pendant section edges before loop building.
///
/// Mirrors the reference kernel's "shapes to avoid" pass: a section edge that
/// fails to connect two parts of the face boundary dangles into the interior.
/// Left in, the angular traversal is forced to walk out and back along it,
/// producing a zero-area spur and over-splitting the face into spurious
/// coplanar pieces.
///
/// A vertex is a free end if every alive edge touching it belongs to the same
/// section (same `source_edge_idx`). Each section appears as a forward+reverse
/// pair, so a genuine free end carries exactly that pair and nothing else.
/// Boundary edges (`source_edge_idx == None`) never dangle — the outer wire is
/// closed — so they anchor their vertices and are never removed. Peeling is
/// iterative: removing one pendant can expose the next along a chain.
pub fn remove_pendant_sections(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    u_periodic: bool,
    v_periodic: bool,
) -> Vec<OrientedPCurveEdge> {
    let u_period = if u_periodic { Some(TAU) } else { None };
    let v_period = if v_periodic { Some(TAU) } else { None };
    let mut alive = vec![true; edges.len()];

    loop {
        let mut vmap: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, e) in edges.iter().enumerate() {
            if !alive[i] {
                continue;
            }
            let sk = quantize_uv_periodic(e.start_uv, tol, u_period, v_period);
            let ek = quantize_uv_periodic(e.end_uv, tol, u_period, v_period);
            vmap.entry(sk).or_default().push(i);
            if ek != sk {
                vmap.entry(ek).or_default().push(i);
            }
        }

        let mut pendant_src: Option<usize> = None;
        for incident in vmap.values() {
            let mut srcs = incident.iter().map(|&i| edges[i].source_edge_idx);
            let first = srcs.next().flatten();
            if let Some(src) = first
                && incident
                    .iter()
                    .all(|&i| edges[i].source_edge_idx == Some(src))
            {
                pendant_src = Some(src);
                break;
            }
        }

        match pendant_src {
            Some(src) => {
                for (i, e) in edges.iter().enumerate() {
                    if e.source_edge_idx == Some(src) {
                        alive[i] = false;
                    }
                }
            }
            None => break,
        }
    }

    let mut kept = Vec::with_capacity(edges.len());
    for (i, e) in edges.iter().enumerate() {
        if alive[i] {
            kept.push(e.clone());
        }
    }
    kept
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Quantize a 2D point to an integer key for vertex deduplication.
#[allow(dead_code)]
fn quantize_uv(p: Point2, tol: f64) -> (i64, i64) {
    quantize_uv_periodic(p, tol, None, None)
}

/// Quantize a 2D point with periodic normalization.
///
/// When `u_period`/`v_period` is `Some`, the coordinate is normalized into
/// `[0, period)` via `rem_euclid` before quantizing. This ensures vertices
/// near the seam (e.g. u=6.28 and u=0.001 on a cylinder) hash to the same key.
fn quantize_uv_periodic(
    p: Point2,
    tol: f64,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> (i64, i64) {
    // Guard against non-positive or non-finite tolerance.
    let safe_tol = if tol <= 0.0 || !tol.is_finite() {
        1e-7
    } else {
        tol
    };
    let resolution = 1.0 / safe_tol;
    (
        quantize_coord(p.x(), resolution, u_period),
        quantize_coord(p.y(), resolution, v_period),
    )
}

/// Quantize a single coordinate, wrapping at the period boundary so that
/// values near `0` and near `period` hash to the same key.
fn quantize_coord(val: f64, resolution: f64, period: Option<f64>) -> i64 {
    if let Some(p) = period {
        let normalized = val.rem_euclid(p);
        let q = (normalized * resolution).round() as i64;
        let period_q = (p * resolution).round() as i64;
        if q >= period_q { 0 } else { q }
    } else {
        (val * resolution).round() as i64
    }
}

/// Compute the outgoing angle of an edge at a vertex in UV space.
///
/// Returns the angle in \[0, 2pi) of the tangent direction at the vertex.
///
/// - `at_start = true`: outgoing direction (start -> end).
/// - `at_start = false`: incoming direction (end -> start, pointing back along the edge).
#[allow(dead_code)]
fn edge_angle_at_vertex(edge: &OrientedPCurveEdge, at_start: bool) -> f64 {
    edge_angle_at_vertex_periodic(edge, at_start, None, None)
}

/// Compute the outgoing angle of an edge at a vertex, with periodic wrapping.
///
/// For curved edges (NURBS pcurve), evaluates the pcurve near the endpoint
/// to get the true tangent direction. For straight edges, uses the chord.
///
/// When a period is set and the raw `dx` exceeds half the period, the
/// difference is wrapped so seam-crossing edges get correct tangent angles.
fn edge_angle_at_vertex_periodic(
    edge: &OrientedPCurveEdge,
    at_start: bool,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> f64 {
    let (mut dx, mut dy) = pcurve_tangent_at_endpoint(edge, at_start);
    if let Some(period) = u_period {
        let half = period * 0.5;
        if dx.abs() > half {
            dx -= dx.signum() * period;
        }
    }
    if let Some(period) = v_period {
        let half = period * 0.5;
        if dy.abs() > half {
            dy -= dy.signum() * period;
        }
    }
    let angle = dy.atan2(dx);
    if angle < 0.0 { angle + TAU } else { angle }
}

/// Compute the tangent direction at an edge endpoint in UV space.
///
/// For `Line2D` pcurves, returns the chord direction (exact).
/// For `NurbsCurve2D` pcurves, evaluates the pcurve near the endpoint
/// to approximate the true tangent -- important for half-circle arcs where
/// the chord direction can be perpendicular to the actual tangent.
fn pcurve_tangent_at_endpoint(edge: &OrientedPCurveEdge, at_start: bool) -> (f64, f64) {
    use brepkit_math::curves2d::Curve2D;

    // For NURBS pcurves, sample near the endpoint for tangent direction.
    // Reverse edges reuse the same pcurve -- swap t0/tn to match the
    // oriented edge direction.
    if let Curve2D::Nurbs(ref nurbs) = edge.pcurve {
        let knots = nurbs.knots();
        if knots.len() >= 2 {
            let t0_raw = knots[0];
            let tn_raw = knots[knots.len() - 1];
            // For reverse edges, the pcurve's t0 corresponds to the edge's
            // end and tn corresponds to the edge's start.
            let (t_start, t_end) = if edge.forward {
                (t0_raw, tn_raw)
            } else {
                (tn_raw, t0_raw)
            };
            let span = (t_end - t_start).abs();
            let delta = span * 0.01;

            if at_start {
                let p0 = nurbs.evaluate(t_start);
                let p1 = nurbs.evaluate(t_start + (t_end - t_start).signum() * delta);
                return (p1.x() - p0.x(), p1.y() - p0.y());
            }
            // at_end: incoming direction (from end back toward start).
            let p0 = nurbs.evaluate(t_end);
            let p1 = nurbs.evaluate(t_end - (t_end - t_start).signum() * delta);
            return (p1.x() - p0.x(), p1.y() - p0.y());
        }
    }

    // For Line2D and fallback: use chord direction.
    if at_start {
        (
            edge.end_uv.x() - edge.start_uv.x(),
            edge.end_uv.y() - edge.start_uv.y(),
        )
    } else {
        (
            edge.start_uv.x() - edge.end_uv.x(),
            edge.start_uv.y() - edge.end_uv.y(),
        )
    }
}

/// Compute the clockwise sweep angle from `angle_in` to `angle_out`.
///
/// `angle_in` is the incoming edge's angle at the vertex (points back along
/// the arriving edge). `angle_out` is the candidate outgoing edge's angle.
///
/// The formula computes the CCW angle from `angle_out` to the travel
/// direction (`angle_in + pi`). Minimum value = rightmost turn = traces
/// minimal enclosed regions.
///
/// Returns a value in (0, 2pi].
fn clockwise_angle(angle_in: f64, angle_out: f64) -> f64 {
    // A candidate that continues straight ahead (angle_out ~= travel) must rank
    // worst so the traversal hugs the rightmost real turn instead of running on
    // through a collinear junction. At a T-junction the bar passes straight
    // through (two collinear boundary edges meet the section stem), and `da`
    // for that continuation collapses to rounding noise: the angles flow through
    // `atan2`, subtraction, and `rem_euclid`, so the residual lands anywhere in
    // ~[0, 1e-12] depending on the vertex coordinates. A 1e-14 cutoff caught it
    // only by luck (the same tongue split correctly at y=9 with da=1.8e-15 but
    // wove a zero-area slit at y=49 with da=1.7e-13). Treat anything within a
    // small angular band of a straight continuation as a continuation. This is
    // safe: at a 2-edge junction the lone continuation is still the min and gets
    // taken regardless, and at a 3+-edge junction a numerically-collinear
    // continuation should always lose to a genuine branch (that is exactly the
    // face split the traversal exists to find).
    const CONTINUATION_EPS: f64 = 1e-9;
    // Travel direction = opposite of incoming (which points backward).
    let travel = (angle_in + std::f64::consts::PI).rem_euclid(TAU);
    let da = (travel - angle_out).rem_euclid(TAU);
    if (CONTINUATION_EPS..=TAU - CONTINUATION_EPS).contains(&da) {
        da
    } else {
        TAU // Straight continuation -> maximum angle (worst).
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::Vec2;
    use brepkit_topology::edge::EdgeCurve;

    fn make_line_edge(start: Point2, end: Point2) -> OrientedPCurveEdge {
        let dir = Vec2::new(end.x() - start.x(), end.y() - start.y());
        let pcurve =
            Curve2D::Line(Line2D::new(start, dir).expect("non-zero direction for test edge"));
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: start,
            end_uv: end,
            start_3d: brepkit_math::vec::Point3::new(start.x(), start.y(), 0.0),
            end_3d: brepkit_math::vec::Point3::new(end.x(), end.y(), 0.0),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    fn make_section_edge(start: Point2, end: Point2, src: usize) -> OrientedPCurveEdge {
        let mut e = make_line_edge(start, end);
        e.source_edge_idx = Some(src);
        e
    }

    #[test]
    fn remove_pendant_sections_drops_dangling_keeps_dividing() {
        // Square boundary, bottom + top split at x=5 so a divider lands on
        // existing vertices.
        let mut edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(5.0, 0.0)),
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
        ];
        let n_boundary = edges.len();
        // Dividing section (both ends on the boundary) — kept.
        edges.push(make_section_edge(
            Point2::new(5.0, 0.0),
            Point2::new(5.0, 10.0),
            100,
        ));
        edges.push(make_section_edge(
            Point2::new(5.0, 10.0),
            Point2::new(5.0, 0.0),
            100,
        ));
        // Pendant section (free end at (7,5) inside the face) — removed.
        edges.push(make_section_edge(
            Point2::new(5.0, 0.0),
            Point2::new(7.0, 5.0),
            200,
        ));
        edges.push(make_section_edge(
            Point2::new(7.0, 5.0),
            Point2::new(5.0, 0.0),
            200,
        ));

        let kept = remove_pendant_sections(&edges, 1e-7, false, false);

        assert_eq!(
            kept.len(),
            n_boundary + 2,
            "pendant pair removed, dividing pair kept"
        );
        assert!(
            kept.iter().all(|e| e.source_edge_idx != Some(200)),
            "pendant section must be dropped"
        );
        assert_eq!(
            kept.iter()
                .filter(|e| e.source_edge_idx == Some(100))
                .count(),
            2,
            "dividing section must survive"
        );
    }

    #[test]
    fn square_cut_by_vertical_line_produces_two_loops() {
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(5.0, 0.0)),
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(5.0, 0.0)),
        ];

        let loops = build_wire_loops(&edges, 1e-7, false, false);

        assert_eq!(loops.len(), 2, "expected 2 loops, got {}", loops.len());
        for (i, lp) in loops.iter().enumerate() {
            assert_eq!(
                lp.len(),
                4,
                "loop {i} should have 4 edges, got {}",
                lp.len()
            );
        }
    }

    #[test]
    fn single_square_returns_one_loop() {
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
        ];

        let loops = build_wire_loops(&edges, 1e-7, false, false);

        assert_eq!(loops.len(), 1, "expected 1 loop, got {}", loops.len());
        assert_eq!(loops[0].len(), 4);
    }

    #[test]
    fn clockwise_angle_basics() {
        use std::f64::consts::PI;
        let cw = clockwise_angle(PI, PI / 2.0);
        assert!((cw - 3.0 * PI / 2.0).abs() < 1e-10, "cw = {cw}");

        let cw2 = clockwise_angle(PI, 3.0 * PI / 2.0);
        assert!((cw2 - PI / 2.0).abs() < 1e-10, "cw2 = {cw2}");

        let cw3 = clockwise_angle(PI, 0.0);
        assert!((cw3 - TAU).abs() < 1e-10, "cw3 = {cw3}");
    }

    /// A near-collinear continuation (incoming and candidate almost exactly
    /// opposite, so `da` is rounding noise just above 0) must rank as a straight
    /// continuation (TAU = worst), not a razor-thin rightmost turn. The
    /// multi-tongue baseplate fuse wove a zero-area slit because this `da`
    /// landed at 1.7e-13 at one tongue but 1.8e-15 at another, and only the
    /// latter cleared the old 1e-14 cutoff.
    #[test]
    fn clockwise_angle_treats_near_collinear_as_continuation() {
        // angle_out within rounding noise of the travel direction (angle_in+pi).
        let angle_in = 1.374_674_651_606_426_8;
        let angle_out = 4.516_267_305_196_049; // travel ends up ~1.7e-13 larger
        let cw = clockwise_angle(angle_in, angle_out);
        assert!(
            (cw - TAU).abs() < 1e-10,
            "near-collinear continuation must score TAU, got {cw}"
        );
    }

    /// A convex quad whose bottom edge is split at an interior vertex by a
    /// section stem rising to the top edge (a T-junction with a collinear bar)
    /// must split into two loops, never one loop plus a degenerate slit. This is
    /// the minimal 2D form of the tongue-cap split that the collinear-junction
    /// rounding bug broke.
    #[test]
    fn t_junction_with_collinear_bar_splits_into_two() {
        // Bottom edge runs straight from (0,0) to (10,0), split at (4,0). The
        // section stem rises from (4,0) to (4,10) on the top edge.
        let mut edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            make_line_edge(Point2::new(4.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(4.0, 10.0)),
            make_line_edge(Point2::new(4.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
        ];
        // Section stem, both orientations (one per adjacent sub-face).
        edges.push(make_section_edge(
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 10.0),
            100,
        ));
        edges.push(make_section_edge(
            Point2::new(4.0, 10.0),
            Point2::new(4.0, 0.0),
            100,
        ));

        let loops = build_wire_loops(&edges, 1e-7, false, false);
        assert_eq!(loops.len(), 2, "T-junction must split into two loops");
        for l in &loops {
            assert!(
                l.len() >= 3,
                "no degenerate slit loop, got {} edges",
                l.len()
            );
        }
    }

    #[test]
    fn triangle_returns_one_loop() {
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 0.0)),
        ];

        let loops = build_wire_loops(&edges, 1e-7, false, false);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 3);
    }

    #[test]
    fn quantize_uv_periodic_wraps_near_seam() {
        let p1 = Point2::new(TAU + 0.001, 1.0);
        let p2 = Point2::new(0.001, 1.0);
        let tol = 1e-7;
        let k1 = quantize_uv_periodic(p1, tol, Some(TAU), None);
        let k2 = quantize_uv_periodic(p2, tol, Some(TAU), None);
        assert_eq!(
            k1, k2,
            "points near seam should hash equal: {k1:?} vs {k2:?}"
        );

        let p3 = Point2::new(TAU, 1.0);
        let p4 = Point2::new(0.0, 1.0);
        let k3 = quantize_uv_periodic(p3, tol, Some(TAU), None);
        let k4 = quantize_uv_periodic(p4, tol, Some(TAU), None);
        assert_eq!(k3, k4, "u=TAU should wrap to u=0: {k3:?} vs {k4:?}");

        let k5 = quantize_uv_periodic(p1, tol, None, None);
        let k6 = quantize_uv_periodic(p2, tol, None, None);
        assert_ne!(k5, k6, "non-periodic should keep them distinct");
    }

    #[test]
    fn edge_angle_periodic_wraps_large_du() {
        let edge = make_line_edge(Point2::new(6.0, 0.0), Point2::new(0.3, 0.0));
        let angle = edge_angle_at_vertex_periodic(&edge, true, Some(TAU), None);
        assert!(
            !(0.5..=TAU - 0.5).contains(&angle),
            "angle should be near 0 (rightward), got {angle}"
        );

        let angle_raw = edge_angle_at_vertex(&edge, true);
        assert!(
            (angle_raw - std::f64::consts::PI).abs() < 0.5,
            "non-periodic angle should be near pi, got {angle_raw}"
        );
    }

    #[test]
    fn wire_loop_crossing_seam() {
        let tau_05 = TAU + 0.5;
        let edges = vec![
            make_line_edge(Point2::new(5.5, 0.0), Point2::new(tau_05, 0.0)),
            make_line_edge(Point2::new(tau_05, 0.0), Point2::new(tau_05, 5.0)),
            make_line_edge(Point2::new(tau_05, 5.0), Point2::new(5.5, 5.0)),
            make_line_edge(Point2::new(5.5, 5.0), Point2::new(5.5, 0.0)),
        ];
        let loops = build_wire_loops(&edges, 1e-7, true, false);
        assert_eq!(loops.len(), 1, "expected 1 loop, got {}", loops.len());
        assert_eq!(loops[0].len(), 4);
    }

    #[test]
    fn dcel_plain_square_is_one_loop() {
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
        ];
        let loops = build_wire_loops_dcel(&edges, 1e-7, false, false);
        assert_eq!(loops.len(), 1, "square must close as one loop");
        assert_eq!(loops[0].len(), 4);
    }

    #[test]
    fn dcel_divided_square_is_two_loops() {
        // Boundary split at x=5 on both rails + a divider section (fwd+rev).
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(5.0, 0.0)),
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
            make_section_edge(Point2::new(5.0, 0.0), Point2::new(5.0, 10.0), 6),
            make_section_edge(Point2::new(5.0, 10.0), Point2::new(5.0, 0.0), 6),
        ];
        let loops = build_wire_loops_dcel(&edges, 1e-7, false, false);
        assert_eq!(
            loops.len(),
            2,
            "divided square must be two cells, got {}",
            loops.len()
        );
        for lp in &loops {
            assert_eq!(lp.len(), 4);
        }
    }

    #[test]
    fn dcel_glued_cylinder_strip_partition() {
        // A full-period cylinder band (u in [0, 2pi], v in [0, 1]) with seam
        // edges at u=0 and two vertical sections at u=1 and u=2. The glued
        // node keys identify u=0 with u=2pi; the true partition is three
        // cells. The top/bottom rim rings traced via synthetic twins are
        // period-winding pure-synthetic orbits and must be dropped, while
        // every cell (none winding here — the seam edges cut the period)
        // must survive. Rim pieces are stored with u=2pi endpoints, so the
        // lifted shoelace must bridge the copies.
        let u2 = TAU;
        let edges = vec![
            // bottom rim pieces, all spans < pi (the lift precondition;
            // real rim chains arrive pre-split well below pi)
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)),
            make_line_edge(Point2::new(1.0, 0.0), Point2::new(2.0, 0.0)),
            make_line_edge(Point2::new(2.0, 0.0), Point2::new(4.0, 0.0)),
            make_line_edge(Point2::new(4.0, 0.0), Point2::new(u2, 0.0)),
            // top rim pieces
            make_line_edge(Point2::new(1.0, 1.0), Point2::new(0.0, 1.0)),
            make_line_edge(Point2::new(2.0, 1.0), Point2::new(1.0, 1.0)),
            make_line_edge(Point2::new(4.0, 1.0), Point2::new(2.0, 1.0)),
            make_line_edge(Point2::new(u2, 1.0), Point2::new(4.0, 1.0)),
            // seam pair (the full-period face's boundary contains the seam twice)
            make_line_edge(Point2::new(0.0, 1.0), Point2::new(0.0, 0.0)),
            make_line_edge(Point2::new(u2, 0.0), Point2::new(u2, 1.0)),
            // vertical sections at u=1 and u=2 (fwd+rev pairs)
            make_section_edge(Point2::new(1.0, 0.0), Point2::new(1.0, 1.0), 100),
            make_section_edge(Point2::new(1.0, 1.0), Point2::new(1.0, 0.0), 100),
            make_section_edge(Point2::new(2.0, 0.0), Point2::new(2.0, 1.0), 200),
            make_section_edge(Point2::new(2.0, 1.0), Point2::new(2.0, 0.0), 200),
        ];
        let loops = build_wire_loops_dcel(&edges, 1e-7, true, false);
        assert_eq!(
            loops.len(),
            3,
            "glued strip must partition into three cells, got {}",
            loops.len()
        );
        let mut sizes: Vec<usize> = loops.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(
            sizes,
            vec![4, 4, 6],
            "two square cells + the two-piece-rimmed third"
        );
    }

    #[test]
    fn dcel_winding_band_with_real_halves_is_kept() {
        // A full-period band with NO seam edges, cut once at u=1: the single
        // band region winds the period in the glued graph (its rims are
        // separate rings). A winding orbit carrying real halves is a genuine
        // region and must be kept; the pure-synthetic winding rim rings must
        // be dropped (the lite diagonal-pad remainder band was dropped as
        // "outer" by the planar most-negative rule before the lifted
        // shoelace + winding rules).
        let u2 = TAU;
        let edges = vec![
            // bottom rim split at u=1 into sub-pi pieces (boundary, single copies)
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)),
            make_line_edge(Point2::new(1.0, 0.0), Point2::new(3.0, 0.0)),
            make_line_edge(Point2::new(3.0, 0.0), Point2::new(5.0, 0.0)),
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(u2, 0.0)),
            // top rim, same pieces westward
            make_line_edge(Point2::new(1.0, 1.0), Point2::new(0.0, 1.0)),
            make_line_edge(Point2::new(3.0, 1.0), Point2::new(1.0, 1.0)),
            make_line_edge(Point2::new(5.0, 1.0), Point2::new(3.0, 1.0)),
            make_line_edge(Point2::new(u2, 1.0), Point2::new(5.0, 1.0)),
            // the single vertical section at u=1
            make_section_edge(Point2::new(1.0, 0.0), Point2::new(1.0, 1.0), 100),
            make_section_edge(Point2::new(1.0, 1.0), Point2::new(1.0, 0.0), 100),
        ];
        let loops = build_wire_loops_dcel(&edges, 1e-7, true, false);
        assert_eq!(
            loops.len(),
            1,
            "the winding band must survive as one region, got {}",
            loops.len()
        );
        assert_eq!(
            loops[0].len(),
            10,
            "band uses all eight rim pieces + the section twice"
        );
    }
}
