//! Shared helpers for measurement operations.

use std::collections::HashSet;

use brepkit_math::vec::{Point3, Vec3};
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
    // Endpoint-INCLUSIVE: the final sample is a polygon corner where the
    // next boundary edge attaches, and skipping it shortcuts the corner
    // with a chord whose area bite scales with the NEIGHBOUR edge's
    // length, not the sample pitch (a 256-sample quarter arc lost 0.064
    // of a notched cap's area this way). The dedup guard below absorbs
    // the coincidence with the next edge's start vertex.
    let indices: Box<dyn Iterator<Item = usize>> = if forward {
        Box::new(0..=n_samples)
    } else {
        Box::new((0..=n_samples).rev())
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

/// Angular range `(u_start, u_end)` of a face on a surface periodic in `u`,
/// from the arcs its outer wire actually covers rather than from a sparse
/// sample of vertex angles.
///
/// Each circle or ellipse boundary edge is walked along the shorter arc
/// between its endpoints (the convention the midpoint sampling this replaces
/// followed; a single rim edge past 180 degrees reads short here as it did
/// there), unwrapped so the walk never jumps a period, and
/// contributes the interval of `u` it covers (`project` returns the surface
/// `u` of a point); the union of those intervals is the face's extent and the
/// largest uncovered gap is its opening. A wall keeping 270 degrees around a
/// bracket sampled at its vertices and arc midpoints reads as five angles
/// with a 90 degree gap, which the density heuristic of
/// [`compute_angular_range`] calls a full turn. Returns `None` when the wire
/// has no curved edge, or when a curved edge is a spline: a blend band's
/// rational arc keeps its full circle as its domain, so walking it would
/// cover the whole turn (the second-pass fillet fixture).
pub(super) fn angular_range_from_wire_arcs(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
    project: impl Fn(Point3) -> f64,
) -> Option<(f64, f64)> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::{PI, TAU};
    const SAMPLES: usize = 16;
    let mut intervals: Vec<(f64, f64)> = Vec::new();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        match edge.curve() {
            EdgeCurve::Line => continue,
            EdgeCurve::NurbsCurve(_) => return None,
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {}
        }
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            continue;
        };
        let (sp, ep) = (sv.point(), ev.point());
        if edge.is_closed() {
            return Some((0.0, TAU));
        }
        let point_at = |f: f64| -> Point3 {
            match edge.curve() {
                EdgeCurve::Circle(c) => {
                    let (ts, te) = (c.project(sp), c.project(ep));
                    let fwd = (te - ts).rem_euclid(TAU);
                    let a = if fwd <= PI {
                        fwd.mul_add(f, ts)
                    } else {
                        (TAU - fwd).mul_add(-f, ts)
                    };
                    c.evaluate(a)
                }
                EdgeCurve::Ellipse(e) => {
                    let (ts, te) = (e.project(sp), e.project(ep));
                    let fwd = (te - ts).rem_euclid(TAU);
                    let a = if fwd <= PI {
                        fwd.mul_add(f, ts)
                    } else {
                        (TAU - fwd).mul_add(-f, ts)
                    };
                    e.evaluate(a)
                }
                EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => sp,
            }
        };
        let mut prev: Option<f64> = None;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..=SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let mut u = project(point_at(k as f64 / SAMPLES as f64));
            if let Some(p) = prev {
                u -= ((u - p) / TAU).round() * TAU;
            }
            prev = Some(u);
            lo = lo.min(u);
            hi = hi.max(u);
        }
        if hi - lo >= TAU - 1e-9 {
            return Some((0.0, TAU));
        }
        let shift = lo.rem_euclid(TAU) - lo;
        intervals.push((lo + shift, hi + shift));
    }
    if intervals.is_empty() {
        return None;
    }
    // Union on the circle: split intervals crossing the period end, sort,
    // merge, then find the widest uncovered gap.
    let mut pieces: Vec<(f64, f64)> = Vec::new();
    for (a, b) in intervals {
        if b > TAU {
            pieces.push((a, TAU));
            pieces.push((0.0, b - TAU));
        } else {
            pieces.push((a, b));
        }
    }
    pieces.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in pieces {
        match merged.last_mut() {
            Some(last) if a <= last.1 + 1e-9 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let mut best_gap = 0.0_f64;
    let mut range = (0.0, TAU);
    for i in 0..merged.len() {
        let (_, end) = merged[i];
        let next_start = if i + 1 < merged.len() {
            merged[i + 1].0
        } else {
            merged[0].0 + TAU
        };
        let gap = next_start - end;
        if gap > best_gap {
            best_gap = gap;
            let start = next_start.rem_euclid(TAU);
            range = (start, start + (TAU - gap));
        }
    }
    if best_gap <= 1e-9 {
        return Some((0.0, TAU));
    }
    Some(range)
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

/// Green's-theorem signed doubled area (`∮(x dy − y dx)`) of one planar wire in
/// the `(ex, ey)` frame, plus its circular- and elliptic-arc edge count.
/// `Ok(None)` when an edge is neither a line nor a conic arc, so the caller
/// falls back to tessellation.
pub(super) fn planar_wire_signed_area2(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    ex: Vec3,
    ey: Vec3,
) -> Result<Option<(f64, usize)>, crate::OperationsError> {
    let to_2d = |p: Point3| {
        let v = Vec3::new(p.x(), p.y(), p.z());
        (v.dot(ex), v.dot(ey))
    };
    let tol_lin = brepkit_math::tolerance::Tolerance::default().linear;
    let mut area2: f64 = 0.0; // accumulates 2·A (Green's ∮(x dy − y dx))
    let mut arc_edges = 0_usize;
    let mut anchor: Option<(f64, f64)> = None;
    {
        let wire = topo.wire(wire_id)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let (sv, ev) = if oe.is_forward() {
                (edge.start(), edge.end())
            } else {
                (edge.end(), edge.start())
            };
            let pa = topo.vertex(sv)?.point();
            let pb = topo.vertex(ev)?.point();
            let (ax, ay) = to_2d(pa);
            let (bx, by) = to_2d(pb);
            // Chord term: triangle (anchor, a, b) doubled, about the wire's
            // first vertex so far-off coordinates do not cancel the area.
            let (qx, qy) = *anchor.get_or_insert((ax, ay));
            area2 += (ax - qx) * (by - qy) - (bx - qx) * (ay - qy);

            // A degenerate edge collapsed to a point that is NOT a closed circle
            // (e.g. the inner "arc" at the axis where a disc cap reaches r = 0, or
            // a zero-length line) contributes no chord and no bulge — skip it (and
            // do NOT let curve recognition on a zero-length arc decline the whole
            // cap). A CLOSED `Circle` rim also has coincident endpoints but bounds
            // a full disc, so it falls through to the arc handler below.
            let is_closed_circle =
                matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
                    && edge.start() == edge.end();
            if let brepkit_topology::edge::EdgeCurve::Ellipse(e) = edge.curve() {
                // An ellipse is a circle stretched along its minor axis, so the
                // segment between an arc and its chord is `a b / 2 (Δ − sin Δ)`
                // over its parametric sweep Δ, counterclockwise about its normal.
                let nat_start = topo.vertex(edge.start())?.point();
                let nat_end = topo.vertex(edge.end())?.point();
                let (t0, t1) = edge.curve().domain_with_endpoints(nat_start, nat_end);
                let sweep = t1 - t0;
                let facing = e.normal().dot(ex.cross(ey)).signum();
                let turn = if oe.is_forward() { facing } else { -facing };
                area2 += turn * e.semi_major() * e.semi_minor() * (sweep - sweep.sin());
                arc_edges += 1;
                continue;
            }
            if (pa - pb).length() < tol_lin && !is_closed_circle {
                continue;
            }

            // Circular-arc bulge correction (segment between the arc and its
            // chord). A `Line` has no bulge. A `Circle`/arc-`NurbsCurve` adds
            // sign·ρ²·(|α| − sin|α|), α the signed sweep about the arc centre.
            let arc = match edge.curve() {
                brepkit_topology::edge::EdgeCurve::Line => None,
                // An ellipse took its bulge above.
                brepkit_topology::edge::EdgeCurve::Ellipse(_) => None,
                brepkit_topology::edge::EdgeCurve::Circle(c) => Some((c.center(), c.radius())),
                brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => {
                    let tol = brepkit_math::tolerance::Tolerance::default().linear * 100.0;
                    match brepkit_geometry::convert::recognize_curve(nc, tol) {
                        brepkit_geometry::convert::RecognizedCurve::Circle {
                            center,
                            radius,
                            ..
                        } => Some((center, radius)),
                        brepkit_geometry::convert::RecognizedCurve::Line { .. } => None,
                        _ => return Ok(None),
                    }
                }
            };

            if let Some((center, radius)) = arc {
                arc_edges += 1;
                // The bulge correction (circular segment between the arc and its
                // chord) is `sign·ρ²·(|α| − sin|α|)`. Compute the sweep in the
                // curve's NATURAL direction (start→mid→end), then flip its sign for
                // a reversed `OrientedEdge`, so the bulge is consistent with the
                // chord term above (which uses the oriented endpoints). Without the
                // flip, a reversed inner rim of an annulus ADDS its segment instead
                // of subtracting it (inflated area).
                let nat_alpha = if is_closed_circle {
                    // A full circle sweeps 2π in its natural (CCW) direction → the
                    // bulge gives the disc area πρ². (The seam endpoint's antipode is
                    // NOT the domain midpoint, so the open-arc disambiguation below
                    // does not apply.)
                    std::f64::consts::TAU
                } else {
                    // Sample the arc at its DOMAIN midpoint (the domain need not be
                    // [0,1]) to disambiguate the signed sweep > π for a major arc.
                    let nat_start = topo.vertex(edge.start())?.point();
                    let nat_end = topo.vertex(edge.end())?.point();
                    let (t0, t1) = edge.curve().domain_with_endpoints(nat_start, nat_end);
                    let mid_pt = edge.curve().evaluate_with_endpoints(
                        f64::midpoint(t0, t1),
                        nat_start,
                        nat_end,
                    );
                    let (cx, cy) = to_2d(center);
                    let (sx, sy) = to_2d(nat_start);
                    let (ex, ey) = to_2d(nat_end);
                    let (mx, my) = to_2d(mid_pt);
                    let va = (sx - cx, sy - cy);
                    let vm = (mx - cx, my - cy);
                    let vb = (ex - cx, ey - cy);
                    // Signed sweep start→mid→end (each leg in (−π, π]).
                    let ang = |u: (f64, f64), w: (f64, f64)| -> f64 {
                        (u.0 * w.1 - u.1 * w.0).atan2(u.0 * w.0 + u.1 * w.1)
                    };
                    ang(va, vm) + ang(vm, vb)
                };
                let alpha = if oe.is_forward() {
                    nat_alpha
                } else {
                    -nat_alpha
                };
                area2 += alpha.signum() * radius * radius * (alpha.abs() - alpha.abs().sin());
            }
        }
    }
    Ok(Some((area2, arc_edges)))
}

/// Parameter spans that walk `edge` from its traversal-start vertex to its
/// traversal-end vertex. `domain_with_endpoints` gives a whole NURBS edge its
/// curve's own domain even where the curve runs from the edge's end vertex,
/// and a closed edge the span from its curve's origin; either would walk a
/// wire out of order in the unwrapped `(u, v)` plane.
pub(super) fn traversal_spans(
    edge: &brepkit_topology::edge::Edge,
    forward: bool,
    sp: Point3,
    ep: Point3,
) -> Vec<(f64, f64)> {
    use brepkit_topology::edge::EdgeCurve;

    let curve = edge.curve();
    let (t0, t1) = curve.domain_with_endpoints(sp, ep);
    let spans = if edge.start() == edge.end() {
        match curve {
            EdgeCurve::Circle(c) => {
                let tv = c.project(sp);
                vec![(tv, tv + (t1 - t0))]
            }
            EdgeCurve::Ellipse(e) => {
                let tv = e.project(sp);
                vec![(tv, tv + (t1 - t0))]
            }
            EdgeCurve::NurbsCurve(n) => {
                match brepkit_math::nurbs::projection::project_point_to_curve(n, sp, 1e-9) {
                    Ok(hit) => vec![(hit.parameter, t1), (t0, hit.parameter)],
                    Err(_) => vec![(t0, t1)],
                }
            }
            EdgeCurve::Line => vec![(t0, t1)],
        }
    } else {
        let at = |t: f64| curve.evaluate_with_endpoints(t, sp, ep);
        if (at(t0) - sp).length() <= (at(t0) - ep).length() {
            vec![(t0, t1)]
        } else {
            vec![(t1, t0)]
        }
    };
    if forward {
        spans
    } else {
        spans.into_iter().rev().map(|(a, b)| (b, a)).collect()
    }
}

/// The tube-angle range `(v0, v1)`, `v0 < v1 < v0 + 2 pi`, of a torus band
/// bounded by two closed rim circles at distinct `v` and seamed by open arcs
/// along a meridian: of the two arcs of `v` between the rims, the one holding
/// a seam arc's middle (open arcs run counterclockwise from their start
/// vertex). `None` for any other torus face.
pub(super) fn torus_band_v_range(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
    torus: &brepkit_math::surfaces::ToroidalSurface,
) -> Result<Option<(f64, f64)>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::TAU;

    if !face.inner_wires().is_empty() {
        return Ok(None);
    }
    let mut rims: Vec<f64> = Vec::new();
    let mut middle: Option<f64> = None;
    for oe in topo.wire(face.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        let EdgeCurve::Circle(circle) = edge.curve() else {
            return Ok(None);
        };
        let start = topo.vertex(edge.start())?.point();
        if edge.start() == edge.end() {
            if circle.normal().cross(torus.z_axis()).length() > 1e-9 {
                return Ok(None);
            }
            let (_, v) = torus.project_point(start);
            rims.push(v.rem_euclid(TAU));
        } else {
            let end = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let mid = edge
                .curve()
                .evaluate_with_endpoints(f64::midpoint(t0, t1), start, end);
            middle = Some(torus.project_point(mid).1.rem_euclid(TAU));
        }
    }
    let ([a, b], Some(m)) = (rims.as_slice(), middle) else {
        return Ok(None);
    };
    let (lo, hi) = if a < b { (*a, *b) } else { (*b, *a) };
    if hi - lo < 1e-9 {
        return Ok(None);
    }
    Ok(Some(if lo < m && m < hi {
        (lo, hi)
    } else {
        (hi, lo + TAU)
    }))
}

/// The ring-angle range `(u0, u1)`, `u0 < u1 < u0 + 2 pi`, of a torus sector
/// bounded by two tube cross-sections (meridian circles) and seamed by open
/// arcs along a latitude: of the two arcs of `u` between the cross-sections,
/// the one holding a seam arc's middle. `None` for any other torus face.
pub(super) fn torus_sector_u_range(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
    torus: &brepkit_math::surfaces::ToroidalSurface,
) -> Result<Option<(f64, f64)>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::TAU;

    if !face.inner_wires().is_empty() {
        return Ok(None);
    }
    let mut rims: Vec<f64> = Vec::new();
    let mut middle: Option<f64> = None;
    for oe in topo.wire(face.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        let EdgeCurve::Circle(circle) = edge.curve() else {
            return Ok(None);
        };
        let start = topo.vertex(edge.start())?.point();
        if edge.start() == edge.end() {
            if circle.normal().dot(torus.z_axis()).abs() > 1e-9 {
                return Ok(None);
            }
            let (u, _) = torus.project_point(circle.center());
            rims.push(u.rem_euclid(TAU));
        } else {
            let end = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let mid = edge
                .curve()
                .evaluate_with_endpoints(f64::midpoint(t0, t1), start, end);
            middle = Some(torus.project_point(mid).0.rem_euclid(TAU));
        }
    }
    let ([a, b], Some(m)) = (rims.as_slice(), middle) else {
        return Ok(None);
    };
    let (lo, hi) = if a < b { (*a, *b) } else { (*b, *a) };
    if hi - lo < 1e-9 {
        return Ok(None);
    }
    Ok(Some(if lo < m && m < hi {
        (lo, hi)
    } else {
        (hi, lo + TAU)
    }))
}
