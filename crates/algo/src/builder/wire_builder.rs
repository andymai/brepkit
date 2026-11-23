//! 2D wire construction from edge soup via angular-sorting traversal.
//!
//! Given an unordered set of [`OrientedPCurveEdge`]s on a face, builds
//! minimal closed wire loops by traversing edges using the minimum
//! clockwise angle rule at each vertex.
//!
//! Uses the minimum clockwise angle traversal algorithm for face splitting.

use brepkit_math::madd::Madd;
use std::collections::HashMap;
use std::f64::consts::TAU;

use brepkit_math::vec::{Point2, Vec2};

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
    /// A NURBS pcurve's derivative angle there, pointing into the edge.
    exact: Option<f64>,
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
            exact: exact_nurbs_angle(edge, true),
        });
        adj.entry(end_key).or_default().push(VertexEntry {
            edge_idx: idx,
            outgoing: false,
            angle: angle_in,
            exact: exact_nurbs_angle(edge, false),
        });
    }

    // 2. Greedy traversal.
    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();
    let wire_trace = log::log_enabled!(log::Level::Debug) && std::env::var("BK_WIRE_TRACE").is_ok();

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
            let incoming_exact = exact_nurbs_angle(current_edge, false);
            // A straight continuation ranks after every turn; between two
            // of them (a section leaving tangentially), the sampled tangents
            // still order by curvature.
            let score = |entry: &VertexEntry| {
                let turn = turn_score(incoming_angle, entry.angle);
                if continues_straight(incoming_exact, entry.exact)
                    || continues_one_conic(
                        current_edge,
                        &edges[entry.edge_idx],
                        incoming_angle,
                        entry.angle,
                    )
                {
                    TAU + turn
                } else {
                    turn
                }
            };
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
                if wire_trace {
                    let c = &edges[entry.edge_idx];
                    log::debug!(
                        "WTRACE at ({:.4},{:.4}) in={:.4} cand ({:.4},{:.4})->({:.4},{:.4}) ang={:.4} score={:.4}",
                        current_edge.end_uv.x(),
                        current_edge.end_uv.y(),
                        incoming_angle,
                        c.start_uv.x(),
                        c.start_uv.y(),
                        c.end_uv.x(),
                        c.end_uv.y(),
                        entry.angle,
                        score(entry)
                    );
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

                let cw = score(entry);
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
                        score(a)
                            .partial_cmp(&score(b))
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
/// SYNTHETIC reversed twins for the trace only. An interior region may
/// legitimately use synthetic halves (a CW-wound face traverses its rim
/// reversed), so unbounded orbits are identified by area and winding, not by
/// synthetic content: on planar faces the most-negative-area orbit is the
/// outer face; on periodic faces the outer faces are the period-winding
/// orbits made purely of synthetic rim twins, while a winding orbit carrying
/// real halves is a genuine full-period band region. Orbit areas use a
/// lifted minimal-image shoelace, so edge UVs may sit at either period copy
/// as long as single-edge spans stay below half a period. Requires the input
/// discipline the face splitter provides: junctions pre-split, endpoint UVs
/// reconciled.
pub fn build_wire_loops_dcel(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    u_periodic: bool,
    v_periodic: bool,
) -> Vec<Vec<OrientedPCurveEdge>> {
    build_wire_loops_dcel_measured(edges, tol, u_periodic, v_periodic, None, None)
}

/// Reads a loop's signed area in its face's parameter plane.
pub type LoopArea<'a> = &'a dyn Fn(&[OrientedPCurveEdge]) -> f64;

/// [`build_wire_loops_dcel`] with each orbit's signed area read by `area_of`
/// instead of its vertex polygon. A rim that is one closed curve has a single
/// vertex, so its polygon encloses nothing and both the face's own region and
/// its unbounded face would be dropped as out-and-back walks.
///
/// `rim_ccw` gives the rim's winding in UV when the caller knows it: every
/// region then runs each rim edge forward (counter-clockwise) or reversed
/// (clockwise), and an orbit running one the other way lies outside the face,
/// the unbounded face among them, however its area reads (a section across
/// the mouth of a notch in the rim closes the notch into a positive orbit).
pub fn build_wire_loops_dcel_measured(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    u_periodic: bool,
    v_periodic: bool,
    area_of: Option<LoopArea<'_>>,
    rim_ccw: Option<bool>,
) -> Vec<Vec<OrientedPCurveEdge>> {
    if edges.is_empty() {
        return Vec::new();
    }
    let u_period = if u_periodic { Some(TAU) } else { None };
    let v_period = if v_periodic { Some(TAU) } else { None };

    // A closed boundary edge on a periodic face keeps its UVs in its curve's
    // own sense, whatever its traversal (`walk_boundary_from_closed_rim`).
    // The trace reads each half from its start UV, so one run against its
    // curve is turned to run its traversal. Left the curve's way, the region
    // beside it took its reversed twin, and the face's piece ran the edge
    // the same way as the face it shares the edge with.
    let traversed: Vec<OrientedPCurveEdge> = edges
        .iter()
        .map(|e| {
            let mut h = e.clone();
            if h.source_edge_idx.is_none()
                && !h.forward
                && (h.start_3d - h.end_3d).length() < 1e-10
                && (h.start_uv - h.end_uv).length() > tol
            {
                std::mem::swap(&mut h.start_uv, &mut h.end_uv);
            }
            h
        })
        .collect();
    let edges = traversed.as_slice();

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
    // Same-source pairs first: a boundary arc sharing both ends with a
    // section (a wall's notch over a bore, the bore's arc below it) must not
    // take the section's forward half before its reverse half pairs with it.
    for same_source_only in [true, false] {
        for (i, e) in edges.iter().enumerate() {
            if twin[i].is_some() {
                continue;
            }
            let rev_key = (key_of(e.end_uv), key_of(e.start_uv));
            if let Some(cands) = by_pair.get(&rev_key) {
                let same = |j: usize| {
                    edges[j].source_edge_idx.is_some()
                        && edges[j].source_edge_idx == e.source_edge_idx
                };
                let pick = cands
                    .iter()
                    .copied()
                    .filter(|&j| j != i && twin[j].is_none() && (!same_source_only || same(j)))
                    .max_by_key(|&j| usize::from(same(j)));
                if let Some(j) = pick {
                    twin[i] = Some(j);
                    twin[j] = Some(i);
                }
            }
        }
    }
    let rim: Vec<bool> = twin.iter().map(Option::is_none).collect();
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
        let against_rim = |&i: &usize| match rim_ccw {
            Some(true) => i >= n_real,
            Some(false) => i < n_real && rim[i],
            None => false,
        };
        if orbit.iter().any(against_rim) {
            continue;
        }
        let lift = lift_orbit(orbit.iter().map(|&i| &halves[i]), u_period, v_period);
        let perimeter = lift.perimeter;
        let area = area_of.map_or(lift.area, |f| {
            f(&orbit.iter().map(|&i| halves[i].clone()).collect::<Vec<_>>())
        });
        let winding_u = u_period.is_some_and(|p| lift.turn.x().abs() > p * 0.5);
        let winding_v = v_period.is_some_and(|p| lift.turn.y().abs() > p * 0.5);
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
    let outer = if u_periodic || v_periodic || rim_ccw.is_some() {
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

        // Peel the lowest-indexed pendant source each round: `vmap` is a
        // HashMap, so taking the first hit in iteration order made the peel
        // order (and with it sub-face edge sets downstream) vary run to run.
        let mut pendant_src: Option<usize> = None;
        for incident in vmap.values() {
            let mut srcs = incident.iter().map(|&i| edges[i].source_edge_idx);
            let first = srcs.next().flatten();
            if let Some(src) = first
                && incident
                    .iter()
                    .all(|&i| edges[i].source_edge_idx == Some(src))
            {
                pendant_src = Some(pendant_src.map_or(src, |cur| cur.min(src)));
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

/// An orbit of halves laid out in UV with each half's vertices lifted off
/// the period copies they are stored on.
struct OrbitLift {
    /// The signed area the lifted vertices enclose.
    area: f64,
    /// The length of the lifted vertices' polygon.
    perimeter: f64,
    /// How far the last vertex lies from the first. For a closed orbit
    /// these are whole periods, the turns it makes round each period.
    turn: Vec2,
}

/// Lift `halves`, an orbit in order, vertex by vertex: each half moves the
/// last vertex on by its run ([`half_run`]).
///
/// On a periodic surface the node keys glue the seam (correct cylinder
/// connectivity), but edge UVs are stored at whichever period copy their
/// pcurve produced: a raw shoelace over mixed copies is garbage (the
/// full-period remainder band read as strongly negative and was dropped as
/// "outer"). The net lift after the orbit closes is its winding, nonzero
/// only for the unbounded rim rings of a periodic face.
fn lift_orbit<'a>(
    halves: impl IntoIterator<Item = &'a OrientedPCurveEdge>,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> OrbitLift {
    let mut halves = halves.into_iter().peekable();
    let Some(first) = halves.peek().map(|h| h.start_uv) else {
        return OrbitLift {
            area: 0.0,
            perimeter: 0.0,
            turn: Vec2::new(0.0, 0.0),
        };
    };
    let mut area = 0.0;
    let mut perimeter = 0.0;
    let mut prev = first;
    for h in halves {
        let b = prev + half_run(h, u_period, v_period);
        // Taken from the first vertex: far from the origin the products of
        // raw parameters round away a small orbit's area.
        let (p, q) = (prev - first, b - first);
        area += p.x().madd(q.y(), -(q.x() * p.y()));
        perimeter += (b - prev).length();
        prev = b;
    }
    // An orbit that winds a period ends a period from its start, and its
    // sum about the origin holds this term for the gap between.
    let turn = prev - first;
    area += first.x().madd(turn.y(), -(turn.x() * first.y()));
    OrbitLift {
        area: area * 0.5,
        perimeter,
        turn,
    }
}

/// How far a half runs in UV from its start to its end.
///
/// The ends of a closed half are one point, so its run is whole periods: a
/// whole rim in one edge goes once round. Taken the short way, its twin's
/// orbit lost its turn to the last bit of its span, and it stood as a
/// region of no area in place of an unbounded ring.
///
/// An open half runs as its pcurve runs, when the pcurve lies on its ends
/// ([`pcurve_along_half`]): its end lies on the copy nearest the pcurve's
/// end. A pcurve follows its curve's samples lifted along the curve, and a
/// piece of a steep cone's ellipse across its near vertex runs most of a
/// turn, which its ends read the short way round take for a short step
/// back.
///
/// With no such pcurve, an open half runs the short way round. A step of
/// exactly half a period (a rim cut in halves, as files write circles)
/// keeps its stored direction: lifting each vertex against the last left
/// that to the last bit, and a band below such a rim read as winding with
/// no area and was dropped.
fn half_run(h: &OrientedPCurveEdge, u_period: Option<f64>, v_period: Option<f64>) -> Vec2 {
    let d = h.end_uv - h.start_uv;
    let closed = (h.start_3d - h.end_3d).length() < 1e-10;
    let pcurve = if closed {
        None
    } else {
        pcurve_along_half(h, u_period, v_period)
            .map(|(nurbs, from, to)| nurbs.evaluate(to) - nurbs.evaluate(from))
    };
    let along = |d: f64, period: Option<f64>, pcurve: Option<f64>| -> f64 {
        match (period, pcurve) {
            (Some(p), _) if p > 1e-12 && closed => (d / p).round() * p,
            (Some(p), Some(run)) if p > 1e-12 => d - ((d - run) / p).round() * p,
            (Some(p), None) if p > 1e-12 && d.abs() > p * 0.5 * (1.0 + 1e-9) => {
                d - (d / p).round() * p
            }
            _ => d,
        }
    };
    Vec2::new(
        along(d.x(), u_period, pcurve.map(Vec2::x)),
        along(d.y(), v_period, pcurve.map(Vec2::y)),
    )
}

/// A half's NURBS pcurve and its parameters at the half's start and end,
/// when its two ends lie on the half's two ends, each on some copy of the
/// periods, one way round only. `None` for another pcurve, for a stale one,
/// and for one whose ends lie on the same point of the periodic plane.
fn pcurve_along_half(
    h: &OrientedPCurveEdge,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> Option<(&brepkit_math::curves2d::NurbsCurve2D, f64, f64)> {
    use brepkit_math::curves2d::Curve2D;
    // Within the gap a weld leaves between a section's ends and its pcurve.
    const NEAR: f64 = 1e-6;
    let Curve2D::Nurbs(ref nurbs) = h.pcurve else {
        return None;
    };
    let (t0, t1) = nurbs.domain();
    let (a, b) = (nurbs.evaluate(t0), nurbs.evaluate(t1));
    let off = |x: f64, y: f64, period: Option<f64>| {
        let d = x - y;
        let d = period
            .filter(|&p| p > 1e-12)
            .map_or(d, |p| d - (d / p).round() * p);
        d.abs() <= NEAR * y.abs().max(1.0)
    };
    let on = |p: Point2, q: Point2| off(p.x(), q.x(), u_period) && off(p.y(), q.y(), v_period);
    match (
        on(a, h.start_uv) && on(b, h.end_uv),
        on(b, h.start_uv) && on(a, h.end_uv),
    ) {
        (true, false) => Some((nurbs, t0, t1)),
        (false, true) => Some((nurbs, t1, t0)),
        _ => None,
    }
}

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
    let (mut dx, mut dy) = long_pcurve_tangent(edge, at_start, u_period, v_period)
        .unwrap_or_else(|| pcurve_tangent_at_endpoint(edge, at_start, (u_period, v_period)));
    // A step of exactly half a period reads the same either way round (a
    // rim cut in halves, as files write circles), so only a step past the
    // half by more than rounding wraps; at the half the edge keeps its
    // stored direction rather than whichever way the last bit tips it.
    if let Some(period) = u_period {
        let half = period * 0.5;
        if dx.abs() > half * (1.0 + 1e-9) {
            dx -= dx.signum() * period;
        }
    }
    if let Some(period) = v_period {
        let half = period * 0.5;
        if dy.abs() > half * (1.0 + 1e-9) {
            dy -= dy.signum() * period;
        }
    }
    let angle = dy.atan2(dx);
    if angle < 0.0 { angle + TAU } else { angle }
}

/// A NURBS pcurve read at an edge endpoint: the curve, the parameter there,
/// the sign of a parameter step into the edge, and the edge's parameter
/// span. `None` for other pcurves and for a stale one.
///
/// A pcurve may run in the edge's traversal order or in its stored order (a
/// reverse edge reusing its forward pcurve): it is read from whichever end
/// sits on the edge's start, falling back to the traversal flag when both
/// ends do (a closed loop in the plane).
fn nurbs_endpoint(
    edge: &OrientedPCurveEdge,
    at_start: bool,
) -> Option<(&brepkit_math::curves2d::NurbsCurve2D, f64, f64, f64)> {
    use brepkit_math::curves2d::Curve2D;

    let Curve2D::Nurbs(ref nurbs) = edge.pcurve else {
        return None;
    };
    let knots = nurbs.knots();
    if knots.len() < 2 {
        return None;
    }
    let t0_raw = knots[0];
    let tn_raw = knots[knots.len() - 1];
    // A closed half that goes round its surface's period has its ends a
    // period apart, and its pcurve can lie a period from them, which no
    // tangent depends on. It is read in the order whose span runs the way
    // its stored span runs: read from the end nearer its start, a whole rim
    // run against its curve read as stale, and its chord, folded by a
    // period, gave it its twin's heading.
    let stored = edge.end_uv - edge.start_uv;
    if (edge.start_3d - edge.end_3d).length() < 1e-10 && stored.length() > 1e-9 {
        let spans = nurbs.evaluate(tn_raw) - nurbs.evaluate(t0_raw);
        let along = spans.dot(stored) > 0.0;
        let off = if along {
            spans - stored
        } else {
            spans + stored
        };
        if off.length() > 1e-9 {
            return None;
        }
        let (t_start, t_end) = if along {
            (t0_raw, tn_raw)
        } else {
            (tn_raw, t0_raw)
        };
        let inward = (t_end - t_start).signum() * if at_start { 1.0 } else { -1.0 };
        let t_anchor = if at_start { t_start } else { t_end };
        return Some((nurbs, t_anchor, inward, (t_end - t_start).abs()));
    }
    let off_start = |t: f64| {
        let p = nurbs.evaluate(t);
        (p.x() - edge.start_uv.x()).powi(2) + (p.y() - edge.start_uv.y()).powi(2)
    };
    let (d0, dn) = (off_start(t0_raw), off_start(tn_raw));
    let traversal_order = if (d0 - dn).abs() <= 1e-18 {
        edge.forward
    } else {
        d0 < dn
    };
    let (t_start, t_end) = if traversal_order {
        (t0_raw, tn_raw)
    } else {
        (tn_raw, t0_raw)
    };

    // The stored pcurve can be STALE relative to the edge's UV
    // endpoints: junction co-registration welds start_uv/end_uv
    // across faces without refitting the pcurve, so a short
    // T-split piece's sampled departure direction disagrees with
    // where the edge actually starts, and one bad angle in the
    // rotation mis-orders every walker at that junction. When the
    // pcurve's own endpoint sits measurably off the stored UV,
    // fall through to the chord, which follows the welded points.
    let anchor = if at_start { edge.start_uv } else { edge.end_uv };
    let t_anchor = if at_start { t_start } else { t_end };
    let pe = nurbs.evaluate(t_anchor);
    let stale_eps_sq: f64 = 1e-9 * 1e-9;
    if (pe.x() - anchor.x()).powi(2) + (pe.y() - anchor.y()).powi(2) > stale_eps_sq {
        return None;
    }
    let inward = (t_end - t_start).signum() * if at_start { 1.0 } else { -1.0 };
    Some((nurbs, t_anchor, inward, (t_end - t_start).abs()))
}

/// Compute the tangent direction at an edge endpoint in UV space.
///
/// For `Line2D` pcurves, returns the chord direction (exact).
/// For `NurbsCurve2D` pcurves, evaluates the pcurve near the endpoint
/// to approximate the true tangent ([`pcurve_heading`]) -- important for
/// half-circle arcs where the chord direction can be perpendicular to the
/// actual tangent. `periods` are the surface's periods in `u` and `v`.
fn pcurve_tangent_at_endpoint(
    edge: &OrientedPCurveEdge,
    at_start: bool,
    periods: (Option<f64>, Option<f64>),
) -> (f64, f64) {
    // For NURBS pcurves, sample near the endpoint for tangent direction.
    if let Some((nurbs, t, inward, span)) = nurbs_endpoint(edge, at_start) {
        return pcurve_heading(nurbs, (t, inward.madd(span, t)), periods);
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

/// The tangent at an edge endpoint, pointing into the edge, of a NURBS
/// pcurve that runs over half a period round and lies on the edge's ends
/// on other copies ([`pcurve_along_half`]). `None` for any other edge.
///
/// [`nurbs_endpoint`] reads a pcurve off its edge's copy as stale, and the
/// edge's chord, folded the short way round, points against a piece that
/// runs most of a turn.
fn long_pcurve_tangent(
    edge: &OrientedPCurveEdge,
    at_start: bool,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> Option<(f64, f64)> {
    if nurbs_endpoint(edge, at_start).is_some() {
        return None;
    }
    let (nurbs, from, to) = pcurve_along_half(edge, u_period, v_period)?;
    let run = nurbs.evaluate(to) - nurbs.evaluate(from);
    let long = |r: f64, period: Option<f64>| period.is_some_and(|p| r.abs() > 0.5 * p);
    if !long(run.x(), u_period) && !long(run.y(), v_period) {
        return None;
    }
    let ends = if at_start { (from, to) } else { (to, from) };
    Some(pcurve_heading(nurbs, ends, (u_period, v_period)))
}

/// The heading of a NURBS pcurve at its parameter `t`, into its span toward
/// its parameter `toward`: the chord to a hundredth of the way along, or,
/// where that chord runs a quarter period or more, the derivative at `t`,
/// signed into the span and of unit length.
///
/// A pcurve that packs most of a turn into its first knots runs that far
/// within the hundredth: folded the short way round, as the heading is
/// read, the chord pointed against the pcurve. The pcurve itself has no
/// copies to fold between, and its derivative heads the way it leaves `t`.
fn pcurve_heading(
    nurbs: &brepkit_math::curves2d::NurbsCurve2D,
    (t, toward): (f64, f64),
    (u_period, v_period): (Option<f64>, Option<f64>),
) -> (f64, f64) {
    let chord = nurbs.evaluate((toward - t).madd(0.01, t)) - nurbs.evaluate(t);
    let packed = |c: f64, period: Option<f64>| period.is_some_and(|p| c.abs() >= 0.25 * p);
    if packed(chord.x(), u_period) || packed(chord.y(), v_period) {
        let d = nurbs.tangent(t) * (toward - t).signum();
        let len = d.length();
        if len.is_finite() && len > 0.0 {
            return (d.x() / len, d.y() / len);
        }
    }
    (chord.x(), chord.y())
}

/// The angle in \[0, 2pi) of a NURBS pcurve's derivative at an edge
/// endpoint, pointing into the edge; `None` where [`nurbs_endpoint`] reads
/// nothing or the derivative vanishes.
fn exact_nurbs_angle(edge: &OrientedPCurveEdge, at_start: bool) -> Option<f64> {
    let (nurbs, t, inward, span) = nurbs_endpoint(edge, at_start)?;
    let d = nurbs.tangent(t);
    (d.x().is_finite() && d.y().is_finite() && d.length() * span > 1e-9)
        .then(|| (inward * d.y()).atan2(inward * d.x()).rem_euclid(TAU))
}

/// Whether an edge leaving a vertex continues the arriving edge straight
/// on, read from both pcurves' derivatives.
///
/// The sampled tangents are chords a hundredth of a span long, so a curved
/// boundary split at a vertex kinks there by its curvature: into a slight
/// right turn where it curves away from the face, which the walk prefers
/// to a section leaving that vertex to the left, and the region that
/// section bounds never closes. The derivatives of two pieces of one curve
/// agree to 2.4e-7 (a scoop's side edge). A hole's arc pieces, fitted one by
/// one, kink by 3e-6 to 1e-5, and the hole weave relies on the walk reading
/// those as turns (the keyholed knuckle cuts), so the band stays below them.
fn continues_straight(incoming: Option<f64>, outgoing: Option<f64>) -> bool {
    const STRAIGHT: f64 = 1e-6;
    let (Some(back), Some(out)) = (incoming, outgoing) else {
        return false;
    };
    let da = (back + std::f64::consts::PI - out).rem_euclid(TAU);
    !(STRAIGHT..=TAU - STRAIGHT).contains(&da)
}

/// Whether a boundary piece leaving a vertex runs on along the circle or
/// ellipse the arriving boundary piece lies on.
///
/// Two pieces of one conic meet at a shared point with one tangent line,
/// so a piece that leaves within a quarter turn of the travel direction
/// continues the arriving piece exactly. Their pcurves are fitted piece by
/// piece and their end derivatives differ by up to 6e-5 (a 60 degree piece
/// after a 30 degree piece of a hole's rim), which [`continues_straight`]
/// reads as a turn: toward the hole where the rim curves that way, and the
/// walk then passes the section that leaves the rim at that vertex.
fn continues_one_conic(
    arriving: &OrientedPCurveEdge,
    leaving: &OrientedPCurveEdge,
    incoming_angle: f64,
    outgoing_angle: f64,
) -> bool {
    use brepkit_topology::edge::EdgeCurve;
    const SAME: f64 = 1e-9;
    // Sections run both ways, and one that continues another is a turn like
    // any other where a third section leaves between them.
    if arriving.source_edge_idx.is_some() || leaving.source_edge_idx.is_some() {
        return false;
    }
    let one_conic = match (&arriving.curve_3d, &leaving.curve_3d) {
        (EdgeCurve::Circle(a), EdgeCurve::Circle(b)) => {
            (a.center() - b.center()).length() <= SAME
                && (a.radius() - b.radius()).abs() <= SAME
                && a.normal().cross(b.normal()).length() <= SAME
        }
        (EdgeCurve::Ellipse(a), EdgeCurve::Ellipse(b)) => {
            (a.center() - b.center()).length() <= SAME
                && (a.semi_major() - b.semi_major()).abs() <= SAME
                && (a.semi_minor() - b.semi_minor()).abs() <= SAME
                && a.normal().cross(b.normal()).length() <= SAME
                && a.u_axis().cross(b.u_axis()).length() <= SAME
        }
        _ => false,
    };
    if !one_conic {
        return false;
    }
    let off_travel = (incoming_angle + std::f64::consts::PI - outgoing_angle).rem_euclid(TAU);
    !(std::f64::consts::FRAC_PI_2..=TAU - std::f64::consts::FRAC_PI_2).contains(&off_travel)
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

    fn make_bezier_edge(points: [Point2; 3]) -> OrientedPCurveEdge {
        let pcurve = Curve2D::Nurbs(
            brepkit_math::curves2d::NurbsCurve2D::new(
                2,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                points.to_vec(),
                vec![1.0; 3],
            )
            .unwrap(),
        );
        let mut e = make_line_edge(points[0], points[2]);
        e.pcurve = pcurve;
        e
    }

    /// A boundary curving away from a section that leaves its split point
    /// a few degrees to the left runs straight on through that point: its
    /// two pieces share their tangent there. Read as chords a hundredth of
    /// a span long, the pieces kink by the curvature into a slight right
    /// turn, which the walk prefers to the section, so the region the
    /// section bounds was never closed.
    #[test]
    fn a_curved_boundary_runs_straight_through_its_split_point() {
        // The right side bulges in to x 8 at y 5, as two halves of one
        // parabola; the section rises from there to the top edge.
        let mut edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)),
            make_bezier_edge([
                Point2::new(10.0, 0.0),
                Point2::new(8.0, 2.5),
                Point2::new(8.0, 5.0),
            ]),
            make_bezier_edge([
                Point2::new(8.0, 5.0),
                Point2::new(8.0, 7.5),
                Point2::new(10.0, 10.0),
            ]),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(7.5, 10.0)),
            make_line_edge(Point2::new(7.5, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
        ];
        edges.push(make_section_edge(
            Point2::new(8.0, 5.0),
            Point2::new(7.5, 10.0),
            0,
        ));
        edges.push(make_section_edge(
            Point2::new(7.5, 10.0),
            Point2::new(8.0, 5.0),
            0,
        ));
        let loops = build_wire_loops(&edges, 1e-7, false, false);
        let mut sizes: Vec<usize> = loops.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![3, 5], "{loops:?}");
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

    /// A piece of the circle of radius `r` about the origin from angle `a`
    /// to angle `b` (degrees), with a pcurve fitted through 17 samples as
    /// the face splitter fits a boundary piece.
    fn make_arc_edge(r: f64, a: f64, b: f64) -> OrientedPCurveEdge {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::{Point3, Vec3};
        let at = |deg: f64| {
            let t = deg.to_radians();
            Point2::new(r * t.cos(), r * t.sin())
        };
        let samples: Vec<Point2> = (0..=16)
            .map(|k| at(a + (b - a) * f64::from(k) / 16.0))
            .collect();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        let (start, end) = (at(a), at(b));
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(circle),
            pcurve: crate::builder::pcurve_compute::fit_nurbs2d_through_points(&samples),
            start_uv: start,
            end_uv: end,
            start_3d: Point3::new(start.x(), start.y(), 0.0),
            end_3d: Point3::new(end.x(), end.y(), 0.0),
            forward: b > a,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    /// A ring cut by two radial sections 60 degrees apart. The hole's rim
    /// runs clockwise, so a piece of it that follows another curves toward
    /// the hole; the fitted pcurves of a 120 degree and a 60 degree piece
    /// differ at their shared end, and the walk read that as a turn toward
    /// the hole and passed the section that leaves the rim there.
    #[test]
    fn a_holes_rim_piece_runs_on_and_the_section_at_its_end_is_taken() {
        let stem = |deg: f64, src: usize| {
            let t = deg.to_radians();
            let (inner, outer) = (
                Point2::new(2.0 * t.cos(), 2.0 * t.sin()),
                Point2::new(3.0 * t.cos(), 3.0 * t.sin()),
            );
            [
                make_section_edge(inner, outer, src),
                make_section_edge(outer, inner, src),
            ]
        };
        let mut edges = vec![
            make_arc_edge(3.0, 0.0, 60.0),
            make_arc_edge(3.0, 60.0, 180.0),
            make_arc_edge(3.0, 180.0, 360.0),
            make_arc_edge(2.0, 360.0, 180.0),
            make_arc_edge(2.0, 180.0, 60.0),
            make_arc_edge(2.0, 60.0, 0.0),
        ];
        edges.extend(stem(0.0, 0));
        edges.extend(stem(60.0, 1));
        let loops = build_wire_loops(&edges, 1e-7, false, false);
        let mut sizes: Vec<usize> = loops.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, [4, 6], "a 60 degree piece and the rest of the ring");
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
    fn dcel_section_across_a_notch_mouth_is_no_region() {
        let rim = [
            (0.0, 0.0),
            (4.0, 0.0),
            (5.0, 2.0),
            (6.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ];
        let p = |(x, y): (f64, f64)| Point2::new(x, y);
        let mut edges: Vec<OrientedPCurveEdge> = (0..rim.len())
            .map(|i| make_line_edge(p(rim[i]), p(rim[(i + 1) % rim.len()])))
            .collect();
        edges.push(make_section_edge(p((4.0, 0.0)), p((6.0, 0.0)), 7));
        edges.push(make_section_edge(p((6.0, 0.0)), p((4.0, 0.0)), 7));
        let by_area = build_wire_loops_dcel(&edges, 1e-7, false, false);
        assert_eq!(by_area.len(), 2, "the notch reads as a region by area");
        let loops = build_wire_loops_dcel_measured(&edges, 1e-7, false, false, None, Some(true));
        assert_eq!(loops.len(), 1, "only the face itself, got {}", loops.len());
        assert_eq!(loops[0].len(), 7);

        let mut cw: Vec<OrientedPCurveEdge> = edges[..rim.len()]
            .iter()
            .rev()
            .map(|e| make_line_edge(e.end_uv, e.start_uv))
            .collect();
        cw.extend_from_slice(&edges[rim.len()..]);
        let loops = build_wire_loops_dcel_measured(&cw, 1e-7, false, false, None, Some(false));
        assert_eq!(loops.len(), 1, "a clockwise rim keeps only the face too");
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
            // bottom rim pieces, each lifted by its own step the short way
            // round (a step of exactly half a turn keeps its direction)
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

    #[test]
    fn dcel_band_under_a_rim_in_halves_is_kept() {
        // A pin's wall below the mouth of a hole of its own radius: the seam
        // at u = a, the pin's end rim in two halves of exactly half a turn
        // (as files write circles), and the hole's rim above in pieces. Each
        // half reads the same either way round; read the wrong way, the
        // band wound the period with no area and was dropped.
        let a = 1.283_690_632_958_161_4;
        let p = Point2::new;
        let edges = vec![
            make_line_edge(p(a, 30.0), p(a, 0.0)),
            make_line_edge(p(a, 0.0), p(4.425_283_286_547_954_6, 0.0)),
            make_line_edge(p(4.425_283_286_547_954_6, 0.0), p(a + TAU, 0.0)),
            make_line_edge(p(a + TAU, 0.0), p(a + TAU, 30.0)),
            make_line_edge(p(a + TAU, 30.0), p(4.778_341_232_904_937, 30.0)),
            make_line_edge(
                p(4.778_341_232_904_937, 30.0),
                p(1.989_806_525_672_153_8, 30.0),
            ),
            make_line_edge(
                p(1.989_806_525_672_153_8, 30.0),
                p(1.636_748_579_315_157_5, 30.0),
            ),
            make_line_edge(p(1.636_748_579_315_157_5, 30.0), p(a, 30.0)),
        ];
        let loops = build_wire_loops_dcel(&edges, 1e-7, true, false);
        assert_eq!(loops.len(), 1, "the band below the rim is one region");
        assert_eq!(loops[0].len(), 8);
    }

    #[test]
    fn dcel_twin_of_a_whole_rim_in_one_edge_is_no_region() {
        // A band between the seam's two copies, its top rim one closed edge
        // whose stored span is a period and a last bit. The rim's twin walks
        // once round above the band, an unbounded ring. Taken the short way,
        // the rim's step was the last bit: the ring stood as a region, and
        // the band, with no turn left in it, read as no area.
        let a = 4.712_388_980_384_691;
        let turn = 6.283_185_307_179_587;
        let p = Point2::new;
        let mut rim = make_line_edge(p(a + turn, 1.0), p(a, 1.0));
        rim.end_3d = rim.start_3d;
        let edges = vec![
            make_line_edge(p(a, 0.0), p(a + 2.0, 0.0)),
            make_line_edge(p(a + 2.0, 0.0), p(a + 4.0, 0.0)),
            make_line_edge(p(a + 4.0, 0.0), p(a + TAU, 0.0)),
            make_line_edge(p(a + TAU, 0.0), p(a + TAU, 1.0)),
            rim,
            make_line_edge(p(a, 1.0), p(a, 0.0)),
        ];
        let loops = build_wire_loops_dcel(&edges, 1e-7, true, false);
        assert_eq!(loops.len(), 1, "the band is the one region");
        assert_eq!(loops[0].len(), 6);
    }

    #[test]
    fn a_whole_rim_a_period_from_its_pcurve_heads_the_way_it_runs() {
        // A top rim in one closed edge, run against its curve: its ends sit
        // a period up from its pcurve, which bows up to v = 3.1 between
        // them. Each half leaves its start along its pcurve, whichever copy
        // the pcurve lies on. Read as stale, the rim took its chord, which a
        // period folds to nothing, and headed +u like its twin.
        use std::f64::consts::PI;
        let a = 4.712_388_980_384_691;
        let v = 2.963;
        let mut rim = make_bezier_edge([
            Point2::new(a, v),
            Point2::new(a + PI, 3.1),
            Point2::new(a + TAU, v),
        ]);
        rim.start_uv = Point2::new(a + 2.0 * TAU, v);
        rim.end_uv = Point2::new(a + TAU, v);
        rim.end_3d = rim.start_3d;
        rim.forward = false;
        // The heading is read on a chord a hundredth of the curve long,
        // which turns it by about 5e-4 from the curve's tangent.
        let bow = (2.0 * (3.1 - v) / TAU).atan();
        let angle = edge_angle_at_vertex_periodic(&rim, true, Some(TAU), None);
        assert!((angle - (PI - bow)).abs() < 1e-3, "rim leaves at {angle}");

        let mut twin = rim;
        std::mem::swap(&mut twin.start_uv, &mut twin.end_uv);
        twin.forward = true;
        let angle = edge_angle_at_vertex_periodic(&twin, true, Some(TAU), None);
        assert!((angle - bow).abs() < 1e-3, "twin leaves at {angle}");
    }

    #[test]
    fn a_half_turn_step_keeps_its_stored_direction() {
        // A rim half of a cylinder, a hair past half a turn by rounding,
        // leaves its start heading +u as stored; wrapped, it read as heading
        // -u, and the face trace walked the rim backwards out of the face.
        let start = 1.989_806_525_672_153_8;
        let half = make_line_edge(
            Point2::new(start, 0.0),
            Point2::new(start + std::f64::consts::PI * (1.0 + 1e-15), 0.0),
        );
        let angle = edge_angle_at_vertex_periodic(&half, true, Some(TAU), None);
        assert!(
            angle.abs() < 1e-9 || (angle - TAU).abs() < 1e-9,
            "angle {angle}"
        );
    }

    /// A piece of a conic on `surface` from its parameter `from` to `to`,
    /// with its pcurve as the face splitter fits a boundary piece and its
    /// end UVs projected, so on the copies the projection puts them on.
    fn conic_piece(
        curve: &EdgeCurve,
        surface: &brepkit_topology::face::FaceSurface,
        from: f64,
        to: f64,
    ) -> OrientedPCurveEdge {
        let at = |t: f64| match curve {
            EdgeCurve::Circle(c) => brepkit_math::traits::ParametricCurve::evaluate(c, t),
            EdgeCurve::Ellipse(e) => brepkit_math::traits::ParametricCurve::evaluate(e, t),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => unreachable!(),
        };
        let (start, end) = (at(from), at(to));
        let uv = |p| {
            let (u, v) = surface.project_point(p).unwrap();
            Point2::new(u, v)
        };
        OrientedPCurveEdge {
            curve_3d: curve.clone(),
            pcurve: crate::builder::pcurve_compute::compute_pcurve_on_surface(
                curve,
                start,
                end,
                surface,
                &[],
                None,
            ),
            start_uv: uv(start),
            end_uv: uv(end),
            start_3d: start,
            end_3d: end,
            forward: to > from,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    /// The band of the cone `x^2 + y^2 = z^2` between its ellipse at slope
    /// 0.99999 (`cone_ellipse_fixture::oblique_cone`) and the ring round the
    /// axis at height `ring_z`, laid out counter-clockwise in UV, with a
    /// ruling across it from the ring to the ellipse. The ellipse is cut at
    /// its parameters pi -+ 0.02, 1.5 pi and 0: the first piece, across its
    /// near vertex, runs most of a turn of `u`, and the others share the
    /// rest. Below the ellipse its pieces run `-u`; above it, `+u`.
    fn steep_cone_band(ring_z: f64) -> Vec<OrientedPCurveEdge> {
        use crate::builder::pcurve_compute::cone_ellipse_fixture::oblique_cone;
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::{Point3, Vec3};
        use std::f64::consts::PI;
        let (surface, ellipse) = oblique_cone(0.99999);
        let sense = crate::builder::pcurve_compute::ellipse_u_sense(&ellipse, &surface).unwrap();
        let below = ring_z < 0.5;
        // Up the band's left side the ellipse runs -u below, +u above.
        let up = if below { -sense } else { sense };
        let ellipse_curve = EdgeCurve::Ellipse(ellipse.clone());
        let cuts = [PI - 0.02, PI + 0.02, 1.5 * PI, 2.0 * PI];
        let mut edges: Vec<OrientedPCurveEdge> = (0..4)
            .map(|k| {
                let (a, b) = (cuts[k], cuts[(k + 1) % 4] + if k == 3 { TAU } else { 0.0 });
                if up > 0.0 {
                    conic_piece(&ellipse_curve, &surface, a, b)
                } else {
                    conic_piece(&ellipse_curve, &surface, b, a)
                }
            })
            .collect();
        // The ruling through the ellipse's point at 1.5 pi, from the apex.
        let apex = Point3::new(0.0, 0.0, 0.0);
        let foot = ellipse.evaluate(1.5 * PI);
        let on_ring = apex + (foot - apex) * (ring_z / foot.z());
        let ring = EdgeCurve::Circle(
            Circle3D::new(
                Point3::new(0.0, 0.0, ring_z),
                Vec3::new(0.0, 0.0, 1.0),
                ring_z,
            )
            .unwrap(),
        );
        let start = match &ring {
            EdgeCurve::Circle(c) => c.project(on_ring),
            _ => unreachable!(),
        };
        for k in 0..3 {
            let (a, b) = (
                start + TAU * f64::from(k) / 3.0,
                start + TAU * f64::from(k + 1) / 3.0,
            );
            edges.push(if below {
                conic_piece(&ring, &surface, a, b)
            } else {
                conic_piece(&ring, &surface, b, a)
            });
        }
        let uv = |p| {
            let (u, v) = surface.project_point(p).unwrap();
            Point2::new(u, v)
        };
        for (from, to) in [(on_ring, foot), (foot, on_ring)] {
            let mut section = make_section_edge(uv(from), uv(to), 0);
            section.start_3d = from;
            section.end_3d = to;
            edges.push(section);
        }
        edges
    }

    /// The band of [`steep_cone_band`] below its ellipse and above it, so
    /// with the piece across the ellipse's near vertex run each way. The
    /// piece runs over half a turn of `u`, and leaves and reaches its ends
    /// the way it runs. The rims' twins are rings round the face, one turn
    /// each, and are no region: read from its ends the short way round, the
    /// piece ran a turn short, its end read as arriving against its run, and
    /// the ellipse's ring wound nothing and stood as a region beside the
    /// band. The band's lifted area is positive, and run the other way round,
    /// negative.
    #[test]
    fn a_band_under_a_piece_over_half_a_turn_is_the_one_region() {
        for ring_z in [0.25, 2e5] {
            let edges = steep_cone_band(ring_z);
            let lifted = lift_orbit(std::iter::once(&edges[0]), Some(TAU), None);
            let run = lifted.turn.x();
            assert!(
                run.abs() > std::f64::consts::PI,
                "ring {ring_z}: the piece across the near vertex runs {run}"
            );
            let out = edge_angle_at_vertex_periodic(&edges[0], true, Some(TAU), None);
            let back = edge_angle_at_vertex_periodic(&edges[0], false, Some(TAU), None);
            assert!(
                out.cos() * run > 0.0 && back.cos() * run < 0.0,
                "ring {ring_z}: the piece running {run} leaves at {out} and arrives from {back}"
            );
            let loops = build_wire_loops_dcel(&edges, 1e-7, true, false);
            assert_eq!(loops.len(), 1, "ring {ring_z}: {} regions", loops.len());
            assert_eq!(loops[0].len(), edges.len(), "ring {ring_z}");
            let lift = lift_orbit(loops[0].iter(), Some(TAU), None);
            assert!(lift.area > 0.0, "ring {ring_z}: area {}", lift.area);
            let back: Vec<OrientedPCurveEdge> = loops[0]
                .iter()
                .rev()
                .map(|e| {
                    let mut r = e.clone();
                    std::mem::swap(&mut r.start_uv, &mut r.end_uv);
                    std::mem::swap(&mut r.start_3d, &mut r.end_3d);
                    r.forward = !r.forward;
                    r
                })
                .collect();
            let lift = lift_orbit(back.iter(), Some(TAU), None);
            assert!(
                lift.area < 0.0,
                "ring {ring_z}: reversed area {}",
                lift.area
            );
        }
    }

    /// A half whose pcurve packs 4.5 of its 5 radians of `u` into the first
    /// five hundredths of a percent of its knots, with its ends stored on
    /// the pcurve's own copy and a period off it, run each way. It leaves
    /// its start, and its twin reaches its end, heading `+u` as the pcurve
    /// does there. A chord a hundredth of the pcurve long ran 4.5 in `u`,
    /// and folded the short way round it pointed `-u`.
    #[test]
    fn a_packed_pcurve_leaves_its_end_along_its_derivative() {
        use brepkit_math::curves2d::NurbsCurve2D;
        let pcurve = Curve2D::Nurbs(
            NurbsCurve2D::new(
                1,
                vec![0.0, 0.0, 0.0005, 1.0, 1.0],
                vec![
                    Point2::new(0.0, 0.0),
                    Point2::new(4.5, 0.1),
                    Point2::new(5.0, 1.0),
                ],
                vec![1.0; 3],
            )
            .unwrap(),
        );
        let heading = 0.1_f64.atan2(4.5);
        for shift in [0.0, TAU] {
            let mut half = make_line_edge(Point2::new(shift, 0.0), Point2::new(5.0 + shift, 1.0));
            half.pcurve = pcurve.clone();
            let mut twin = half.clone();
            std::mem::swap(&mut twin.start_uv, &mut twin.end_uv);
            std::mem::swap(&mut twin.start_3d, &mut twin.end_3d);
            twin.forward = false;
            for (what, edge, at_start) in [("half", &half, true), ("twin", &twin, false)] {
                let angle = edge_angle_at_vertex_periodic(edge, at_start, Some(TAU), None);
                let off =
                    (angle - heading + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
                assert!(
                    off.abs() < 1e-9,
                    "{what} stored {shift} off: heads {angle}, the pcurve {heading}"
                );
            }
        }
    }

    #[test]
    fn multi_pendant_peel_is_deterministic() {
        // A closed square boundary with THREE independent dangling sections
        // (sources deliberately out of order). Every repetition must peel the
        // same set the same way: the peel picks the lowest source index per
        // round, so the kept set is invariant regardless of the hash order the
        // candidate scan encounters.
        let p = Point2::new;
        let build = || {
            vec![
                make_line_edge(p(0.0, 0.0), p(10.0, 0.0)),
                make_line_edge(p(10.0, 0.0), p(10.0, 10.0)),
                make_line_edge(p(10.0, 10.0), p(0.0, 10.0)),
                make_line_edge(p(0.0, 10.0), p(0.0, 0.0)),
                make_section_edge(p(5.0, 0.0), p(5.0, 4.0), 7),
                make_section_edge(p(5.0, 4.0), p(5.0, 0.0), 7),
                make_section_edge(p(0.0, 5.0), p(3.0, 5.0), 2),
                make_section_edge(p(3.0, 5.0), p(0.0, 5.0), 2),
                make_section_edge(p(10.0, 5.0), p(7.0, 5.0), 5),
                make_section_edge(p(7.0, 5.0), p(10.0, 5.0), 5),
            ]
        };
        let reference = remove_pendant_sections(&build(), 1e-7, false, false);
        assert_eq!(
            reference.len(),
            4,
            "all three dangling sections must be peeled, keeping the boundary"
        );
        for _ in 0..50 {
            let again = remove_pendant_sections(&build(), 1e-7, false, false);
            let key = |es: &[OrientedPCurveEdge]| -> Vec<(String, String)> {
                es.iter()
                    .map(|e| (format!("{:?}", e.start_uv), format!("{:?}", e.end_uv)))
                    .collect()
            };
            assert_eq!(key(&again), key(&reference), "peel result must not vary");
        }
    }
}
