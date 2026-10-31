//! Exact integrals over a face bounded on a surface of revolution, taken
//! along its wires' own curves by Green's theorem in `(u, v)`.
//!
//! With `F(u, v) = ∫₀ᵘ g(s, v) ds`, the region `R` on the wires' left gives
//! `∫∫_R g du dv = ∮ F dv`, `u` carried continuously along each wire. A wire
//! that winds the axis `k` times ends a `k`-fold turn from where it began,
//! which the seam its region is cut along makes good: `-2πk H(v₀)`, with
//! `H(v) = ∫ ḡ` from the reference latitude and `ḡ` the mean of `g` over a
//! turn, `v₀` the latitude the wire starts at. A region holding a pole adds
//! `2π H` there, taken toward growing `v`. Nothing is sampled into a polygon, so the result is
//! exact to quadrature precision however an edge bows.

use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{Face, FaceSurface};

use super::face_integrator::FaceContribution;
use crate::CheckError;

/// Integrands a face's properties sum at `(u, v)`: area, the volume flux
/// about `about`, the three volume moments and the three area moments.
pub(super) fn integrand<S: ParametricSurface>(
    surface: &S,
    u: f64,
    v: f64,
    about: Vec3,
) -> [f64; 8] {
    let p = surface.evaluate(u, v);
    let n = surface.partial_u(u, v).cross(surface.partial_v(u, v));
    let n_len = n.length();
    let pv = Vec3::new(p.x(), p.y(), p.z()) - about;
    [
        n_len,
        pv.dot(n) / 3.0,
        0.5 * p.x() * p.x() * n.x(),
        0.5 * p.y() * p.y() * n.y(),
        0.5 * p.z() * p.z() * n.z(),
        p.x() * n_len,
        p.y() * n_len,
        p.z() * n_len,
    ]
}

/// One edge of a wire in its traversal: its curve, end points, and the
/// curve parameter it runs over from where the wire enters it.
pub(super) struct BoundaryEdge<'a> {
    pub curve: &'a EdgeCurve,
    pub start: Point3,
    pub end: Point3,
    pub from: f64,
    pub to: f64,
}

/// The surface, how points map to its `(u, v)`, and where `H` starts.
pub(super) struct Chart<'a, S: ParametricSurface> {
    pub surface: &'a S,
    pub project: &'a dyn Fn(Point3) -> (f64, f64),
    /// The latitude `H` is taken from: a pole (or apex) the wires may run
    /// through, where `H` vanishes and so does a wire's jump across it.
    pub v_ref: f64,
    /// The poles at the low and high ends of `v`, where the surface has
    /// them (a cone's apex at one end, a sphere's at both).
    pub low: Option<f64>,
    pub high: Option<f64>,
    /// `v` is an angle too (round a torus's tube): it is carried along each
    /// wire as `u` is, and a wire winding round the tube needs nothing
    /// more, every integrand being periodic in it.
    pub v_periodic: bool,
}

const TAU: f64 = std::f64::consts::TAU;

fn wrap_pi(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI
}

fn add(acc: &mut [f64; 8], g: &[f64; 8], w: f64) {
    for (a, x) in acc.iter_mut().zip(g) {
        *a += w * x;
    }
}

/// `∫ f` over `[a, b]` by `rule`, each interval halved until its two halves
/// agree with it to 1e-12 of their size or to `floor`, or `None` past
/// `depth` halvings or where `f` is.
fn adaptive(
    f: &mut dyn FnMut(f64) -> Option<[f64; 8]>,
    a: f64,
    b: f64,
    rule: &[brepkit_math::quadrature::GaussPoint],
    depth: u32,
    floor: f64,
) -> Option<[f64; 8]> {
    let mut over = |a: f64, b: f64| -> Option<[f64; 8]> {
        let (half, mid) = (0.5 * (b - a), 0.5 * (a + b));
        let mut acc = [0.0; 8];
        for gp in rule {
            add(&mut acc, &f(half.mul_add(gp.x, mid))?, half * gp.w);
        }
        Some(acc)
    };
    let m = f64::midpoint(a, b);
    let whole = over(a, b)?;
    let (left, right) = (over(a, m)?, over(m, b)?);
    let mut halves = left;
    add(&mut halves, &right, 1.0);
    let size = halves.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let gap = whole
        .iter()
        .zip(&halves)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    if gap <= (1e-12 * size).max(floor) {
        return Some(halves);
    }
    if depth == 0 {
        return None;
    }
    let mut sum = adaptive(f, a, m, rule, depth - 1, floor)?;
    add(&mut sum, &adaptive(f, m, b, rule, depth - 1, floor)?, 1.0);
    Some(sum)
}

/// `∫₀ᵗᵘʳⁿ g(s, v) ds` by the trapezoid rule, exact for the low-order
/// trigonometric integrands of a surface of revolution.
fn turn_integral<S: ParametricSurface>(surface: &S, v: f64, about: Vec3) -> [f64; 8] {
    const N: u32 = 32;
    let mut acc = [0.0; 8];
    for k in 0..N {
        let s = TAU * f64::from(k) / f64::from(N);
        add(
            &mut acc,
            &integrand(surface, s, v, about),
            TAU / f64::from(N),
        );
    }
    acc
}

/// `∫₀ᵘ g(s, v) ds` for any `u`: whole turns by [`turn_integral`], the rest
/// by Gauss-Legendre in panels of at most a quarter turn.
fn antiderivative<S: ParametricSurface>(surface: &S, u: f64, v: f64, about: Vec3) -> [f64; 8] {
    let turns = (u / TAU).floor();
    let rest = u - turns * TAU;
    let mut acc = [0.0; 8];
    if turns != 0.0 {
        add(&mut acc, &turn_integral(surface, v, about), turns);
    }
    let gauss = gauss_legendre_points(10);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let panels = (rest / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as u32;
    let width = rest / f64::from(panels);
    for i in 0..panels {
        let mid = width.mul_add(f64::from(i) + 0.5, 0.0);
        for gp in gauss {
            let s = (0.5 * width).mul_add(gp.x, mid);
            add(
                &mut acc,
                &integrand(surface, s, v, about),
                0.5 * width * gp.w,
            );
        }
    }
    acc
}

/// `H(v) = ∫ ḡ` from `v_ref` to `v`, `ḡ` the mean over a turn.
fn mean_integral<S: ParametricSurface>(chart: &Chart<'_, S>, v: f64, about: Vec3) -> [f64; 8] {
    let gauss = gauss_legendre_points(10);
    let span = v - chart.v_ref;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let panels = (span.abs() / std::f64::consts::FRAC_PI_4)
        .ceil()
        .clamp(1.0, 64.0) as u32;
    let width = span / f64::from(panels);
    let mut acc = [0.0; 8];
    for i in 0..panels {
        let mid = width.mul_add(f64::from(i) + 0.5, chart.v_ref);
        for gp in gauss {
            let w = (0.5 * width).mul_add(gp.x, mid);
            add(
                &mut acc,
                &turn_integral(chart.surface, w, about),
                0.5 * width * gp.w / TAU,
            );
        }
    }
    acc
}

/// An edge's parameter span in traversal order, cut where the quadrature
/// wants it: a conic every eighth of a turn (its parameter is its angle), a
/// NURBS curve at its knots, a line in two.
fn edge_pieces(edge: &BoundaryEdge<'_>) -> Vec<(f64, f64)> {
    let span = edge.to - edge.from;
    let uniform = |n: u32| {
        (0..n)
            .map(|i| {
                (
                    span.mul_add(f64::from(i) / f64::from(n), edge.from),
                    span.mul_add(f64::from(i + 1) / f64::from(n), edge.from),
                )
            })
            .collect()
    };
    match edge.curve {
        EdgeCurve::Line => uniform(2),
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => uniform(
            (span.abs() / std::f64::consts::FRAC_PI_8)
                .ceil()
                .clamp(1.0, 16.0) as u32,
        ),
        EdgeCurve::NurbsCurve(nc) => {
            let (lo, hi) = (edge.from.min(edge.to), edge.from.max(edge.to));
            let mut cuts: Vec<f64> = nc
                .knots()
                .iter()
                .copied()
                .filter(|&k| k > lo + 1e-12 * (hi - lo) && k < hi - 1e-12 * (hi - lo))
                .collect();
            cuts.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (hi - lo));
            let mut ends = vec![lo];
            ends.extend(cuts);
            ends.push(hi);
            let mut pieces: Vec<(f64, f64)> = ends.windows(2).map(|w| (w[0], w[1])).collect();
            if edge.to < edge.from {
                pieces = pieces.into_iter().rev().map(|(a, b)| (b, a)).collect();
            }
            pieces
        }
    }
}

/// A face's integrals over the region on its wires' left, each wire's edges
/// in traversal order: area and area moments unsigned, the volume terms
/// times `sign` (the face's orientation against its surface's normal).
/// `None` when the wires do not read: consecutive edges that do not meet, a
/// piece that turns a quarter turn about the axis however finely it is cut
/// (a wire through a pole), a turn that is not whole, a wire of no area whose side the
/// sum of turns cannot tell, a pole the surface does not have, or a region
/// of no area.
#[allow(clippy::too_many_lines)]
pub(super) fn integrate_by_boundary<S: ParametricSurface>(
    chart: &Chart<'_, S>,
    wires: &[Vec<BoundaryEdge<'_>>],
    sign: f64,
    about: Vec3,
) -> Option<FaceContribution> {
    const AREA_SLACK: f64 = 1e-12;
    let gauss = gauss_legendre_points(8);
    // A point with no `u` of its own (a cone's apex) may take any.
    let singular = |u: f64, v: f64| {
        chart.surface.partial_u(u, v).length() <= 1e-6 * chart.surface.partial_v(u, v).length()
    };
    let quarter = std::f64::consts::FRAC_PI_2;
    let mut total = [0.0; 8];
    let (mut turns_sum, mut winds, mut patch, mut holes) = (0_i64, false, false, true);
    for wire in wires {
        let first = wire.first()?;
        let entry = first
            .curve
            .evaluate_with_endpoints(first.from, first.start, first.end);
        let (mut u, v_start) = (chart.project)(entry);
        let u_start = u;
        let mut v = v_start;
        let v_wrap = |x: f64| if chart.v_periodic { wrap_pi(x) } else { x };
        let mut acc = [0.0; 8];
        let mut twice_area = 0.0;
        let (mut exit, mut chord) = (entry, 0.0);
        for edge in wire {
            let point = |t: f64| edge.curve.evaluate_with_endpoints(t, edge.start, edge.end);
            let at = |t: f64| (chart.project)(point(t));
            let (enter, leave) = (point(edge.from), point(edge.to));
            chord = (leave - enter).length();
            if (enter - exit).length() > 1e-6 + 1e-3 * chord {
                return None;
            }
            exit = leave;
            // A piece that turns a quarter turn about the axis is halved (a
            // NURBS rim's knot span may be a third of a circle); one that
            // still does at a thousandth of its edge runs past a pole.
            let floor = 1e-3 * (edge.to - edge.from).abs();
            let mut pending = edge_pieces(edge);
            pending.reverse();
            while let Some((ta, tb)) = pending.pop() {
                let (ua, va) = at(ta);
                let u_a = u + wrap_pi(ua - u);
                let v_a = v + v_wrap(va - v);
                let (ub, vb) = at(tb);
                let u_b = u_a + wrap_pi(ub - u_a);
                let v_b = v_a + v_wrap(vb - v_a);
                let mid = f64::midpoint(ta, tb);
                // The rule's nodes over `[a, b]`, `u` and `v` carried on from
                // the piece's start.
                let nodes_over = |a: f64, b: f64| {
                    let (half, centre) = (0.5 * (b - a), f64::midpoint(a, b));
                    let mut nodes = [(0.0, 0.0, 0.0, 0.0); 8];
                    for (node, gp) in nodes.iter_mut().zip(gauss) {
                        let t = half.mul_add(gp.x, centre);
                        let (un, vn) = at(t);
                        *node = (
                            t,
                            u_a + wrap_pi(un - u_a),
                            v_a + v_wrap(vn - v_a),
                            half * gp.w,
                        );
                    }
                    nodes
                };
                let nodes = nodes_over(ta, tb);
                let swings = !singular(ua, va)
                    && (nodes.iter().any(|n| (n.1 - u_a).abs() > quarter)
                        || (!singular(ub, vb) && (u_b - u_a).abs() > quarter));
                if swings {
                    if (tb - ta).abs() <= floor {
                        return None;
                    }
                    pending.push((mid, tb));
                    pending.push((ta, mid));
                    continue;
                }
                // The integrals over `[a, b]` by the rule, the area's last,
                // and the most the stencil's rounding can move any of them.
                let sums = |a: f64, b: f64, nodes: [(f64, f64, f64, f64); 8]| {
                    // A five-point stencil wide enough that rounding in
                    // far-off coordinates stays small against it.
                    let h = 1e-3 * (b - a);
                    let mut sums = [0.0; 9];
                    let mut rounding = 0.0_f64;
                    for (t, un, vn, w) in nodes {
                        let [(u1, v1), (u2, v2), (u3, v3), (u4, v4)] =
                            [t + h, t - h, t + 2.0 * h, t - 2.0 * h].map(at);
                        let du = 8.0f64.mul_add(wrap_pi(u1 - u2), -wrap_pi(u3 - u4)) / (12.0 * h);
                        let dv = 8.0f64.mul_add(v_wrap(v1 - v2), -v_wrap(v3 - v4)) / (12.0 * h);
                        // Each difference rounds by an ulp of its coordinate.
                        let (nu, nv) = (
                            1.5 * f64::EPSILON * (1.0 + un.abs()) / h.abs(),
                            1.5 * f64::EPSILON * (1.0 + vn.abs()) / h.abs(),
                        );
                        // A latitude (a rim circle about the axis) adds nothing.
                        if dv.abs() > 1e-12 * (1.0 + vn.abs()) {
                            let g = antiderivative(chart.surface, un, vn, about);
                            for (s, x) in sums.iter_mut().zip(g) {
                                *s += w * dv * x;
                                rounding = rounding.max((w * nv * x).abs());
                            }
                        }
                        sums[8] += w * (un * dv - vn * du);
                        rounding = rounding.max((w * (un.abs() * nv + vn.abs() * nu)).abs());
                    }
                    (sums, 8.0 * rounding)
                };
                let (mut piece, _) = sums(ta, tb, nodes);
                // A NURBS edge's parameter can run unevenly along the surface
                // (a rational arc whose weights grow along it, or a polynomial
                // whose speed swings), so its piece is kept only where its
                // halves agree with it, and halved otherwise.
                if matches!(edge.curve, EdgeCurve::NurbsCurve(_)) {
                    let ((left, left_rounding), (right, right_rounding)) = (
                        sums(ta, mid, nodes_over(ta, mid)),
                        sums(mid, tb, nodes_over(mid, tb)),
                    );
                    let halves: [f64; 9] = std::array::from_fn(|k| left[k] + right[k]);
                    let rounding = left_rounding + right_rounding;
                    let size = halves.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                    let gap = piece
                        .iter()
                        .zip(&halves)
                        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
                    // The gap measures the whole piece's error; the halves
                    // kept are far closer. A gap within the stencil's
                    // rounding is no error of the rule (a knot span 6e-11
                    // wide, or a stretch where the edge barely moves), and
                    // halving only multiplies that rounding.
                    if gap > 1e-9 * size && gap > 100.0 * rounding {
                        if (tb - ta).abs() <= floor {
                            return None;
                        }
                        pending.push((mid, tb));
                        pending.push((ta, mid));
                        continue;
                    }
                    piece = halves;
                }
                for (a, x) in acc.iter_mut().zip(&piece) {
                    *a += x;
                }
                twice_area += piece[8];
                u = u_b;
                v = v_b;
            }
        }
        if (entry - exit).length() > 1e-6 + 1e-3 * chord {
            return None;
        }
        let turn = u - u_start;
        let turns = (turn / TAU).round();
        if (turn - turns * TAU).abs() > 1e-6 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        let turns = turns as i64;
        let tube_turns = ((v - v_start) / TAU).round();
        if chart.v_periodic && (v - v_start - tube_turns * TAU).abs() > 1e-6 {
            return None;
        }
        if chart.v_periodic && tube_turns != 0.0 {
            if turns != 0 {
                return None;
            }
            add(&mut total, &acc, 1.0);
            winds = true;
            continue;
        }
        if turns != 0 {
            let h = mean_integral(chart, v_start, about);
            #[allow(clippy::cast_precision_loss)]
            add(&mut acc, &h, -TAU * turns as f64);
        } else if twice_area.abs() <= AREA_SLACK {
            return None;
        }
        add(&mut total, &acc, 1.0);
        turns_sum += turns;
        winds |= turns != 0;
        patch |= turns == 0 && twice_area > AREA_SLACK;
        holes &= turns == 0 && twice_area < -AREA_SLACK;
    }
    // A total of one turn holds the high pole (the face on the wire's left,
    // toward growing `v`), minus one the low; with none, a band or a patch
    // holds neither and a face of clockwise holes both. A pole the surface
    // does not have (a cone's open end) holds no face.
    let (holds_low, holds_high) = match turns_sum {
        1 => (false, true),
        -1 => (true, false),
        0 if winds || patch => (false, false),
        0 if holes => (true, true),
        _ => return None,
    };
    if holds_high {
        add(&mut total, &mean_integral(chart, chart.high?, about), TAU);
    }
    if holds_low {
        add(&mut total, &mean_integral(chart, chart.low?, about), -TAU);
    }
    // On a torus, a wire round the tube carries `u` on from wherever it
    // starts, and one round the axis reads `H` at whichever turn of `v` it
    // starts in: a turn more or less moves either by the whole surface's
    // integral. A face holds more than none of the surface and less than all
    // of it.
    if chart.v_periodic && winds {
        let mut whole = mean_integral(chart, chart.v_ref + TAU, about);
        for x in &mut whole {
            *x *= TAU;
        }
        if whole[0] <= 0.0 {
            return None;
        }
        let turns = (total[0] / whole[0]).floor();
        add(&mut total, &whole, -turns);
    }
    if total[0] <= 0.0 {
        return None;
    }
    Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    })
}

/// A face's wires as [`BoundaryEdge`]s in traversal order. Each open edge
/// runs from where the wire stands (the stored flags need not chain), each
/// closed one from its vertex. `None` for a closed line.
fn boundary_wires<'t>(
    topo: &'t Topology,
    face: &Face,
) -> Result<Option<Vec<Vec<BoundaryEdge<'t>>>>, CheckError> {
    let ids: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    wires_of(topo, &ids)
}

/// [`boundary_wires`] for the wires `ids`.
fn wires_of<'t>(
    topo: &'t Topology,
    ids: &[brepkit_topology::wire::WireId],
) -> Result<Option<Vec<Vec<BoundaryEdge<'t>>>>, CheckError> {
    let mut wires = Vec::new();
    for &wid in ids {
        let mut edges = Vec::new();
        let mut cursor: Option<Point3> = None;
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            let curve = edge.curve();
            if edge.start() == edge.end() {
                let spans = match curve {
                    EdgeCurve::Circle(c) => {
                        let at = c.project(start);
                        vec![(at, at + TAU)]
                    }
                    EdgeCurve::Ellipse(e) => {
                        let at = e.project(start);
                        vec![(at, at + TAU)]
                    }
                    // A closed NURBS rim runs from its vertex to its domain's
                    // end and on from its start back to the vertex.
                    EdgeCurve::NurbsCurve(nc) => {
                        let (u0, u1) = nc.domain();
                        let at = crate::util::nurbs_seam_parameter(nc, start, u0, u1);
                        vec![(at, u1), (u0, at)]
                    }
                    EdgeCurve::Line => return Ok(None),
                };
                let spans: Vec<(f64, f64)> = if oe.is_forward() {
                    spans
                } else {
                    spans.into_iter().rev().map(|(a, b)| (b, a)).collect()
                };
                for (from, to) in spans {
                    if (to - from).abs() > 0.0 {
                        edges.push(BoundaryEdge {
                            curve,
                            start,
                            end,
                            from,
                            to,
                        });
                    }
                }
                cursor = Some(start);
                continue;
            }
            if matches!(curve, EdgeCurve::Line) && (end - start).length() < 1e-12 {
                continue;
            }
            let (t0, t1) = curve.domain_with_endpoints(start, end);
            let first = curve.evaluate_with_endpoints(t0, start, end);
            let runs_start_to_end = (first - start).length() <= (first - end).length();
            let forward = cursor.map_or_else(
                || oe.is_forward(),
                |c| (start - c).length() <= (end - c).length(),
            );
            let (from, to) = if forward == runs_start_to_end {
                (t0, t1)
            } else {
                (t1, t0)
            };
            edges.push(BoundaryEdge {
                curve,
                start,
                end,
                from,
                to,
            });
            cursor = Some(if forward { end } else { start });
        }
        if !edges.is_empty() {
            wires.push(edges);
        }
    }
    Ok(Some(wires))
}

/// Points along every wire, sixty-five to an edge.
fn wire_samples(wires: &[Vec<BoundaryEdge<'_>>]) -> Vec<Point3> {
    wires
        .iter()
        .flatten()
        .flat_map(|e| {
            (0..=64).map(move |k| {
                let t = (e.to - e.from).mul_add(f64::from(k) / 64.0, e.from);
                e.curve.evaluate_with_endpoints(t, e.start, e.end)
            })
        })
        .collect()
}

/// A cylinder, cone or sphere face's integrals by [`integrate_by_boundary`],
/// `sign` its orientation against its surface. `None` for another surface,
/// or a face the boundary does not read (a cone face across its apex, a
/// sphere face whose wires come near every candidate axis's poles).
pub(super) fn curved_face_by_boundary(
    topo: &Topology,
    face: &Face,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    if !matches!(
        face.surface(),
        FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
    ) {
        return Ok(None);
    }
    if let FaceSurface::Torus(t) = face.surface()
        && t.minor_radius() < t.major_radius()
        && whole_surface(topo, face)?
    {
        return whole_torus_less_holes(topo, face, t, sign, about);
    }
    let Some(wires) = boundary_wires(topo, face)? else {
        return Ok(None);
    };
    Ok(match face.surface() {
        FaceSurface::Cylinder(s) => {
            let project = |p: Point3| s.project_point(p);
            // No pole: `H` from where the first wire starts, where one panel
            // of its integral stays short.
            let v_ref = wires.first().and_then(|w| w.first()).map_or(0.0, |e| {
                project(e.curve.evaluate_with_endpoints(e.from, e.start, e.end)).1
            });
            let chart = Chart {
                surface: s,
                project: &project,
                v_ref,
                low: None,
                high: None,
                v_periodic: false,
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        FaceSurface::Cone(s) => {
            let vs: Vec<f64> = wire_samples(&wires)
                .into_iter()
                .map(|p| s.project_point(p).1)
                .collect();
            let slack = 1e-9 * vs.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            let (low, high) = if vs.iter().all(|&v| v >= -slack) {
                (Some(0.0), None)
            } else if vs.iter().all(|&v| v <= slack) {
                (None, Some(0.0))
            } else {
                return Ok(None);
            };
            let project = |p: Point3| s.project_point(p);
            let chart = Chart {
                surface: s,
                project: &project,
                v_ref: 0.0,
                low,
                high,
                v_periodic: false,
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        FaceSurface::Sphere(s) => {
            let samples = wire_samples(&wires);
            let reach = |axis: Vec3| {
                samples.iter().fold(0.0_f64, |m, p| {
                    let w = *p - s.center();
                    m.max((w.dot(axis) / w.length().max(1e-300)).abs())
                })
            };
            let axis = [
                s.z_axis(),
                s.x_axis(),
                s.y_axis(),
                Vec3::new(
                    0.447_213_595_499_957_9,
                    0.547_722_557_505_166_1,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
                Vec3::new(-0.5, 0.763_762_615_825_973_4, 0.408_248_290_463_863),
                Vec3::new(
                    0.597_614_304_667_196_8,
                    -0.377_964_473_009_227_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
            ]
            .into_iter()
            .map(|a| (reach(a), a))
            .filter(|(m, _)| *m <= 3.0_f64.to_radians().cos())
            .min_by(|a, b| a.0.total_cmp(&b.0));
            let Some((_, axis)) = axis else {
                return Ok(None);
            };
            let Ok(turned) = SphericalSurface::with_axis(s.center(), s.radius(), axis) else {
                return Ok(None);
            };
            let project = |p: Point3| turned.project_point(p);
            let chart = Chart {
                surface: &turned,
                project: &project,
                v_ref: -std::f64::consts::FRAC_PI_2,
                low: Some(-std::f64::consts::FRAC_PI_2),
                high: Some(std::f64::consts::FRAC_PI_2),
                v_periodic: false,
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        // A ring torus: no pole, and its tube angle `v` wraps as `u` does.
        FaceSurface::Torus(t) if t.minor_radius() < t.major_radius() => {
            let project = |p: Point3| t.project_point(p);
            let chart = Chart {
                surface: t,
                project: &project,
                v_ref: 0.0,
                low: None,
                high: None,
                v_periodic: true,
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        FaceSurface::Plane { .. } | FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => None,
    })
}

/// Whether a face's outer wire is its surface's seams alone, closed edges of
/// no length: the face is the whole surface, less its holes.
fn whole_surface(topo: &Topology, face: &Face) -> Result<bool, CheckError> {
    let wire = topo.wire(face.outer_wire())?;
    let mut all = true;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        all &= edge.start() == edge.end() && matches!(edge.curve(), EdgeCurve::Line);
    }
    Ok(all && !wire.edges().is_empty())
}

/// A ring torus face that is the whole torus less its holes: the torus's
/// integrals over both turns, by the trapezoid rule (exact for its
/// trigonometric integrands), less each hole's, read on the hole's wire run
/// the other way round, which holds the hole on its left.
fn whole_torus_less_holes(
    topo: &Topology,
    face: &Face,
    t: &brepkit_math::surfaces::ToroidalSurface,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    const N: u32 = 32;
    let step = TAU / f64::from(N);
    let mut total = [0.0; 8];
    for i in 0..N {
        for j in 0..N {
            let g = integrand(t, step * f64::from(i), step * f64::from(j), about);
            add(&mut total, &g, step * step);
        }
    }
    let project = |p: Point3| t.project_point(p);
    let chart = Chart {
        surface: t,
        project: &project,
        v_ref: 0.0,
        low: None,
        high: None,
        v_periodic: true,
    };
    for &hole in face.inner_wires() {
        let Some(wires) = wires_of(topo, &[hole])? else {
            return Ok(None);
        };
        let around: Vec<Vec<BoundaryEdge<'_>>> = wires
            .into_iter()
            .map(|wire| {
                wire.into_iter()
                    .rev()
                    .map(|e| BoundaryEdge {
                        from: e.to,
                        to: e.from,
                        ..e
                    })
                    .collect()
            })
            .collect();
        let Some(inside) = integrate_by_boundary(&chart, &around, 1.0, about) else {
            return Ok(None);
        };
        let inside = [
            inside.area,
            inside.volume,
            inside.volume_moment_x,
            inside.volume_moment_y,
            inside.volume_moment_z,
            inside.centroid_x,
            inside.centroid_y,
            inside.centroid_z,
        ];
        add(&mut total, &inside, -1.0);
    }
    if total[0] <= 0.0 {
        return Ok(None);
    }
    Ok(Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    }))
}

/// A face that is its whole open NURBS surface, its integrals over the
/// surface's `(u, v)` domain directly, in panels between its knots. Green's
/// theorem along its boundary comes to the same through a projection of
/// every boundary point, which a face cut from nothing need not pay for: a
/// sweep tiles its tube with hundreds of such patches. Taken only where the
/// face is shown to be the whole surface: no holes, and its wire straight
/// edges from one corner of the domain to the next, each side of the
/// surface shown straight by its control points lying on its edge. The
/// surface is polynomial with clamped knots, so its sides are the outer rows
/// of its control points, which bound them, and its integrands are smooth
/// in each panel. `None` for any other face.
pub(super) fn nurbs_face_over_domain(
    topo: &Topology,
    face: &Face,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    const DEPTH: u32 = 24;
    let FaceSurface::Nurbs(s) = face.surface() else {
        return Ok(None);
    };
    let clamped = |knots: &[f64], degree: usize| {
        knots.len() > 2 * degree + 1
            && knots[..=degree]
                .iter()
                .all(|&k| k.to_bits() == knots[0].to_bits())
            && knots[knots.len() - degree - 1..]
                .iter()
                .all(|&k| k.to_bits() == knots[knots.len() - 1].to_bits())
    };
    // A knot repeated more often than the degree breaks the surface apart:
    // its sides' control points no longer show it continuous.
    let continuous = |knots: &[f64], degree: usize| {
        knots[degree + 1..knots.len() - degree - 1]
            .chunk_by(|a, b| a.to_bits() == b.to_bits())
            .all(|run| run.len() <= degree)
    };
    if !face.inner_wires().is_empty()
        || s.is_rational()
        || !clamped(s.knots_u(), s.degree_u())
        || !clamped(s.knots_v(), s.degree_v())
        || !continuous(s.knots_u(), s.degree_u())
        || !continuous(s.knots_v(), s.degree_v())
        || !open_and_poleless(s)
    {
        return Ok(None);
    }
    let cps = s.control_points();
    let (nu, nv) = (cps.len(), cps.first().map_or(0, Vec::len));
    if nu < 2 || nv < 2 {
        return Ok(None);
    }
    // Side `k` runs from corner `k` to the next, along `v = v0`, `u = u1`,
    // `v = v1` and `u = u0` in turn, its control points the outer row there.
    let corners = [
        cps[0][0],
        cps[nu - 1][0],
        cps[nu - 1][nv - 1],
        cps[0][nv - 1],
    ];
    let side = |k: usize| -> Vec<Point3> {
        match k {
            0 => (0..nu).map(|i| cps[i][0]).collect(),
            1 => (0..nv).map(|j| cps[nu - 1][j]).collect(),
            2 => (0..nu).map(|i| cps[i][nv - 1]).collect(),
            _ => (0..nv).map(|j| cps[0][j]).collect(),
        }
    };
    // Matched to the shortest side: a tolerance taken from the whole patch
    // would let a long thin one's trim pass for its side.
    let shortest = (0..4)
        .map(|k| (corners[(k + 1) % 4] - corners[k]).length())
        .fold(f64::INFINITY, f64::min);
    if !shortest.is_finite() || shortest <= 0.0 {
        return Ok(None);
    }
    let reach = 1e-9 * shortest;
    let on_segment = |p: Point3, a: Point3, b: Point3| {
        let (ab, ap) = (b - a, p - a);
        let w = (ap.dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
        (ap - ab * w).length() <= reach
    };
    let corner_of = |p: Point3| corners.iter().position(|&c| (c - p).length() <= reach);
    let mut covered = [false; 4];
    for oe in topo.wire(face.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        if !matches!(edge.curve(), EdgeCurve::Line) {
            return Ok(None);
        }
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (Some(i), Some(j)) = (corner_of(a), corner_of(b)) else {
            return Ok(None);
        };
        let k = if (i + 1) % 4 == j {
            i
        } else if (j + 1) % 4 == i {
            j
        } else {
            return Ok(None);
        };
        if !side(k).iter().all(|&p| on_segment(p, a, b)) {
            return Ok(None);
        }
        covered[k] = true;
    }
    if !covered.iter().all(|&c| c) {
        return Ok(None);
    }
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();
    // Every distinct knot cuts: a span however short can hold a share of
    // the surface.
    let cuts = |knots: &[f64], lo: f64, hi: f64| {
        let mut c: Vec<f64> = std::iter::once(lo)
            .chain(knots.iter().copied().filter(|&k| k > lo && k < hi))
            .chain(std::iter::once(hi))
            .collect();
        c.dedup_by(|a, b| a.to_bits() == b.to_bits());
        c
    };
    let (u_cuts, v_cuts) = (cuts(s.knots_u(), u0, u1), cuts(s.knots_v(), v0, v1));
    // How often a panel may be halved before its halves come within a
    // thousand units in the last place of its parameters, where the rule's
    // points round onto each other and a halving reads the same panel twice.
    // A panel too narrow to halve four times is not read here.
    let halvings = |a: f64, b: f64| {
        let ulp = f64::EPSILON * a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let depth = ((b - a) / (1024.0 * ulp)).log2().floor().max(0.0) as u32;
        depth.min(DEPTH)
    };
    if u_cuts
        .windows(2)
        .chain(v_cuts.windows(2))
        .any(|w| halvings(w[0], w[1]) < 4)
    {
        return Ok(None);
    }
    let size = (corners[2] - corners[0])
        .length()
        .max((corners[1] - corners[3]).length());
    let floor = 1e-14 * size.powi(3);
    // The volume's integrand is a polynomial of a few times the surface's
    // degree in each parameter; each panel is halved until it settles, so
    // a rule a little above the degree serves.
    let order = (2 + s.degree_u().max(s.degree_v())).min(8);
    let gauss = gauss_legendre_points(order);
    let mut across = |v: f64| -> Option<[f64; 8]> {
        let mut acc = [0.0; 8];
        for w in u_cuts.windows(2) {
            let mut g = |u: f64| Some(integrand(s, u, v, about));
            add(
                &mut acc,
                &adaptive(&mut g, w[0], w[1], gauss, halvings(w[0], w[1]), floor)?,
                1.0,
            );
        }
        Some(acc)
    };
    let mut total = [0.0; 8];
    for w in v_cuts.windows(2) {
        let Some(panel) = adaptive(&mut across, w[0], w[1], gauss, halvings(w[0], w[1]), floor)
        else {
            return Ok(None);
        };
        add(&mut total, &panel, 1.0);
    }
    if total[0] <= 1e-12 {
        return Ok(None);
    }
    Ok(Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    }))
}

/// A face on an open NURBS surface, its integrals by Green's theorem as
/// [`integrate_by_boundary`] takes them, with `F(u, v) = ∫ g(s, v) ds` from
/// the start of `u`'s domain in panels between the surface's knots, and
/// each edge cut where it crosses a knot line, so every panel's integrand
/// is smooth. `None` for a surface that closes on itself or shrinks to a
/// point along a side (its `(u, v)` would not be one-to-one), a wire a
/// point of which does not project, or a region of no area.
pub(super) fn nurbs_face_by_boundary(
    topo: &Topology,
    face: &Face,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    use brepkit_math::nurbs::projection::{
        project_point_to_surface, project_point_to_surface_from,
    };
    const DEPTH: u32 = 24;
    let FaceSurface::Nurbs(s) = face.surface() else {
        return Ok(None);
    };
    if !open_and_poleless(s) {
        return Ok(None);
    }
    let Some(wires) = boundary_wires(topo, face)? else {
        return Ok(None);
    };
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();
    let interior = |knots: &[f64], lo: f64, hi: f64| {
        let mut k: Vec<f64> = knots
            .iter()
            .copied()
            .filter(|&k| k > lo && k < hi)
            .collect();
        k.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (hi - lo));
        k
    };
    let (u_knots, v_knots) = (interior(s.knots_u(), u0, u1), interior(s.knots_v(), v0, v1));
    let size = (s.evaluate(u1, v1) - s.evaluate(u0, v0))
        .length()
        .max((s.evaluate(u1, v0) - s.evaluate(u0, v1)).length());
    let reach = 1e-6 * size.max(1.0);
    // Where an integrand vanishes (along an edge at the start of `u`) its
    // halves agree only to rounding.
    let floor = 1e-14 * size.powi(3);
    // Each integral is halved until it settles: a rational surface can
    // crowd its area into a sliver of its parameter.
    let gauss_u = gauss_legendre_points(10);
    let antiderivative = |u: f64, v: f64| -> Option<[f64; 8]> {
        let mut acc = [0.0; 8];
        let mut from = u0;
        for to in u_knots.iter().copied().filter(|&k| k < u).chain([u]) {
            if to > from {
                let mut g = |s_u: f64| Some(integrand(s, s_u, v, about));
                add(
                    &mut acc,
                    &adaptive(&mut g, from, to, gauss_u, DEPTH, floor)?,
                    1.0,
                );
            }
            from = to;
        }
        Some(acc)
    };
    let gauss = gauss_legendre_points(8);
    let mut total = [0.0; 8];
    for (k, wire) in wires.iter().enumerate() {
        let mut region = [0.0; 8];
        for edge in wire {
            let point = |t: f64| edge.curve.evaluate_with_endpoints(t, edge.start, edge.end);
            // A point's `(u, v)` by Newton from a neighbour's where that lands
            // on the surface, else by a search of the whole surface.
            let on = |t: f64, seed: Option<(f64, f64)>| -> Option<(f64, f64)> {
                let p = point(t);
                if let Some(seed) = seed
                    && let Ok(q) = project_point_to_surface_from(s, p, seed, 1e-12)
                    && q.distance <= reach
                {
                    return Some((q.u, q.v));
                }
                project_point_to_surface(s, p, 1e-12)
                    .ok()
                    .filter(|q| q.distance <= reach)
                    .map(|q| (q.u, q.v))
            };
            let mut pieces = Vec::new();
            for (ta, tb) in edge_pieces(edge) {
                // Where the edge crosses a knot line the integrand bends:
                // cut there, found by bisection between samples astride it.
                let mut samples: Vec<(f64, (f64, f64))> = Vec::with_capacity(9);
                let mut seed = None;
                for k in 0..=8 {
                    let t = (tb - ta).mul_add(f64::from(k) / 8.0, ta);
                    let Some(q) = on(t, seed) else {
                        return Ok(None);
                    };
                    samples.push((t, q));
                    seed = Some(q);
                }
                let mut cuts = vec![ta, tb];
                // A sample on a knot line is a cut itself; the bisection
                // below only takes a knot strictly between two samples.
                let (u_span, v_span) = (u1 - u0, v1 - v0);
                for &(t, q) in &samples {
                    let on_knot = u_knots.iter().any(|&k| (q.0 - k).abs() <= 1e-12 * u_span)
                        || v_knots.iter().any(|&k| (q.1 - k).abs() <= 1e-12 * v_span);
                    if on_knot {
                        cuts.push(t);
                    }
                }
                for w in samples.windows(2) {
                    let ((t_a, a), (t_b, b)) = (w[0], w[1]);
                    for (knots, pick) in [(&u_knots, 0_usize), (&v_knots, 1_usize)] {
                        let coord = |q: (f64, f64)| if pick == 0 { q.0 } else { q.1 };
                        for &k in knots {
                            if (coord(a) - k) * (coord(b) - k) >= 0.0 {
                                continue;
                            }
                            let (mut lo, mut hi) = (t_a, t_b);
                            for _ in 0..50 {
                                let mid = 0.5 * (lo + hi);
                                let Some(q) = on(mid, Some(a)) else {
                                    return Ok(None);
                                };
                                if (coord(q) - k) * (coord(a) - k) > 0.0 {
                                    lo = mid;
                                } else {
                                    hi = mid;
                                }
                            }
                            cuts.push(0.5 * (lo + hi));
                        }
                    }
                }
                cuts.sort_by(f64::total_cmp);
                if tb < ta {
                    cuts.reverse();
                }
                cuts.dedup_by(|x, y| (*x - *y).abs() <= 1e-12 * (tb - ta).abs());
                let start = samples[0].1;
                pieces.extend(cuts.windows(2).map(|w| (w[0], w[1], start)));
            }
            for (ta, tb, start) in pieces {
                let mut seed = start;
                let mut along = |t: f64| -> Option<[f64; 8]> {
                    let c = on(t, Some(seed))?;
                    seed = c;
                    // `dv/dt` from the curve's velocity in the surface's
                    // tangent plane, by the first fundamental form.
                    let velocity = edge_velocity(edge.curve, t, edge.start, edge.end);
                    let d = s.derivatives(c.0, c.1, 1);
                    let (su, sv) = (d[1][0], d[0][1]);
                    let (e, f, g) = (su.dot(su), su.dot(sv), sv.dot(sv));
                    let det = e.mul_add(g, -f * f);
                    if det <= 1e-24 * (e * g).max(1e-300) {
                        return None;
                    }
                    let dv = e.mul_add(sv.dot(velocity), -f * su.dot(velocity)) / det;
                    let mut term = antiderivative(c.0, c.1)?;
                    for x in &mut term {
                        *x *= dv;
                    }
                    Some(term)
                };
                let Some(piece) = adaptive(&mut along, ta, tb, gauss, DEPTH, floor) else {
                    return Ok(None);
                };
                add(&mut region, &piece, 1.0);
            }
        }
        // Each wire counts the region it bounds, whichever way it runs in
        // `(u, v)`; holes are taken away.
        let side = region[0].signum() * if k == 0 { 1.0 } else { -1.0 };
        add(&mut total, &region, side);
    }
    if total[0] <= 1e-12 {
        return Ok(None);
    }
    Ok(Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    }))
}

/// `d/dt` of `curve` at `t` as `evaluate_with_endpoints` runs it from
/// `start` to `end`.
fn edge_velocity(curve: &EdgeCurve, t: f64, start: Point3, end: Point3) -> Vec3 {
    match curve {
        EdgeCurve::Line => end - start,
        EdgeCurve::Circle(c) => c.tangent(t) * c.radius(),
        EdgeCurve::Ellipse(e) => e.tangent(t),
        EdgeCurve::NurbsCurve(n) => n.derivatives(t, 1)[1],
    }
}

/// Whether a NURBS surface neither closes on itself along `u` or `v` nor
/// shrinks to a point along a side.
fn open_and_poleless(s: &brepkit_math::nurbs::surface::NurbsSurface) -> bool {
    const N: u32 = 8;
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();
    let lerp = |a: f64, b: f64, k: u32| (b - a).mul_add(f64::from(k) / f64::from(N), a);
    let side = |along_u: bool, at: f64| -> Vec<Point3> {
        (0..=N)
            .map(|k| {
                if along_u {
                    s.evaluate(lerp(u0, u1, k), at)
                } else {
                    s.evaluate(at, lerp(v0, v1, k))
                }
            })
            .collect()
    };
    let length = |pts: &[Point3]| pts.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>();
    let sides = [
        side(false, u0),
        side(false, u1),
        side(true, v0),
        side(true, v1),
    ];
    let size = sides.iter().map(|p| length(p)).fold(0.0_f64, f64::max);
    let near = 1e-9 * size.max(1.0);
    let same =
        |a: &[Point3], b: &[Point3]| a.iter().zip(b).all(|(p, q)| (*p - *q).length() <= near);
    size > near
        && !same(&sides[0], &sides[1])
        && !same(&sides[2], &sides[3])
        && sides.iter().all(|p| length(p) > near)
}
