//! A round run off its ledge: the walls under the ledge meet the band.
//!
//! A round between a wall and a ledge (a face along the wall's foot, or its
//! top) has its contact on the ledge. Where the ledge narrows to a tip
//! tighter than the round, that contact runs off the ledge's outline
//! before the round's end, and the band hangs past the ledge. The walls
//! under the ledge's outline there then meet the band: the band ends on
//! the chain of its intersections with them, from its contact on the
//! wall, across each wall the ledge's outline passes, to where its contact
//! comes back on to the ledge across the outline.
//!
//! The round is first built as it ends in the plane square to its spine,
//! closed by a runout patch of three edges: the band's end section `b`
//! from its contact on the wall `P` to its contact on the ledge `E`, the
//! wall's edge `w` from the ledge's corner `C` to `P`, and a line `l` on
//! the ledge from `C` to `E`. The ledge's outline from `C` runs on through
//! the edges `e1 .. ek` (each between the ledge and a wall `Wi` under it)
//! until the contact `c` from `E` crosses it at `X`, on `ek`. That patch
//! goes; each wall `Wi` takes the band's intersection with it in place of
//! its piece of the outline, and the ledge and the band each take the
//! contact from `X` on.
//!
//! A concave round (adding material over the ledge) has the walls run up to
//! the band: each wall's surface is carried on past the ledge, and between
//! two walls the edge they meet in is carried on up to the band. A convex
//! one (taking material off the ledge's tip) has them cut down to it: the
//! edge two walls meet in is cut where the band crosses it, and the round's
//! wall, which ran up its corner edge from `P` and back down, keeps the
//! edge's piece below `P`.
//!
//! The wall at the round's end may itself run on under the ledge's outline
//! from the corner (a boss's wall standing above the ledge beside the
//! round, below it beyond): it is then the first wall, and `w` goes, the
//! wall's outline reaching `P` down `w`'s neighbour. Where the outline
//! passes a little beyond the band's end section before the contact
//! crosses it, the band is run on past its end (its last span carried on).

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use super::{
    band_meets_cap, cylinder_meets_cap, edge_nearest, rewrite_wire, straight_edge, torus_meets_cap,
    torus_parallel_meets_plane, torus_parallel_meets_surface,
};
use crate::BlendError;

/// The most edges of the ledge's outline a contact may run off past.
const MOST_WALLS: usize = 4;

/// Has the walls under a ledge meet each round run off it
/// ([`run_walls_up`]), each runout patch in turn. Returns the faces left.
pub(super) fn run_walls_up_over_runouts(
    topo: &mut Topology,
    faces: Vec<FaceId>,
    input_edges: &std::collections::HashSet<EdgeId>,
) -> Result<Vec<FaceId>, BlendError> {
    let mut faces = faces;
    'runouts: loop {
        for (index, &face_id) in faces.iter().enumerate() {
            if run_walls_up(topo, &faces, face_id, input_edges)? {
                faces.remove(index);
                continue 'runouts;
            }
            if let Some(corner) = split_runout_notch(topo, &faces, face_id, input_edges)? {
                faces.push(corner);
                continue 'runouts;
            }
        }
        break;
    }
    Ok(faces)
}

/// Splits off the corner of a ledge's outline that a runout patch's line
/// cuts across, where `face_id` is the patch (see the module's docs).
///
/// The patch's line `l` on the ledge runs from the contact's end `E` to the
/// ledge's corner `C`. Where the outline from `C` first steps a little the
/// round's way (a short edge, a file's sliver) and then crosses `l` at `X`,
/// the ledge's loop ran out past `X` and back, crossing itself. The ledge
/// keeps `l` from `E` to `X` and its outline on from `X`; the corner the
/// line cut off (`C`, the short edges, `X`), on the ledge's surface,
/// becomes a face of its own, and the patch takes `l` in its two pieces.
/// Returns the new face.
#[allow(clippy::too_many_lines)]
fn split_runout_notch(
    topo: &mut Topology,
    faces: &[FaceId],
    face_id: FaceId,
    input_edges: &std::collections::HashSet<EdgeId>,
) -> Result<Option<FaceId>, BlendError> {
    // How much shorter than `l` the outline's step before the crossing is.
    const SHORT: f64 = 0.1;
    for patch in patches(topo, faces, face_id, input_edges)? {
        let (l_id, ledge, ledge_wire) = patch.l;
        let l_edge = topo.edge(l_id)?.clone();
        if !matches!(l_edge.curve(), EdgeCurve::Line) {
            continue;
        }
        let [cp, ep] = [patch.corner, patch.e].map(|v| topo.vertex(v).map(Vertex::point));
        let (cp, ep) = (cp?, ep?);
        let length = (cp - ep).length();
        let tol = 1e-6 * length.max(1.0);
        let ledge_edges = topo.wire(ledge_wire)?.edges().to_vec();
        let Some(at) = ledge_edges.iter().position(|oe| oe.edge() == l_id) else {
            continue;
        };
        let n = ledge_edges.len();
        if n < 4 {
            continue;
        }
        let oriented_start = |oe: &OrientedEdge| -> Result<VertexId, BlendError> {
            let edge = topo.edge(oe.edge())?;
            Ok(if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            })
        };
        // Walked through the wire backwards from `l` where `l` taken this
        // way starts at the corner.
        let backward = oriented_start(&ledge_edges[at])? == patch.corner;
        // The outline from the corner: a short edge, then the one `l`
        // crosses.
        let index = |step: usize| {
            if backward {
                (at + n - step) % n
            } else {
                (at + step) % n
            }
        };
        let (first, second) = (ledge_edges[index(1)].edge(), ledge_edges[index(2)].edge());
        if first == l_id || second == l_id || first == second {
            continue;
        }
        let first_edge = topo.edge(first)?.clone();
        if first_edge.start() != patch.corner && first_edge.end() != patch.corner {
            continue;
        }
        let near = if first_edge.start() == patch.corner {
            first_edge.end()
        } else {
            first_edge.start()
        };
        let np = topo.vertex(near)?.point();
        if (np - cp).length() > SHORT * length {
            continue;
        }
        let second_edge = topo.edge(second)?.clone();
        if second_edge.start() != near && second_edge.end() != near {
            continue;
        }
        let far = if second_edge.start() == near {
            second_edge.end()
        } else {
            second_edge.start()
        };
        let (lt0, lt1) = l_edge.curve().domain_with_endpoints(
            topo.vertex(l_edge.start())?.point(),
            topo.vertex(l_edge.end())?.point(),
        );
        let Some((_, x)) = crossing(
            topo,
            second,
            (
                l_edge.curve(),
                topo.vertex(l_edge.start())?.point(),
                topo.vertex(l_edge.end())?.point(),
            ),
            (lt0, lt1),
            tol,
        )?
        else {
            continue;
        };
        let fp = topo.vertex(far)?.point();
        if [cp, ep, np, fp]
            .iter()
            .any(|&end| (end - x).length() <= 10.0 * tol)
        {
            continue;
        }
        let Some((_, other_wire)) = one_use(topo, faces, ledge, second)? else {
            continue;
        };
        let (s_curve, s_start, s_end) = edge_curve(topo, &second_edge)?;
        let (s0, s1) = s_curve.domain_with_endpoints(s_start, s_end);
        let (te, gap) = nearest_param((&s_curve, s_start, s_end), (s0, s1), x);
        if gap > 10.0 * tol {
            continue;
        }
        // The pieces: `l` from `E` to `X` and from `X` to the corner, the
        // crossed edge from `X` to each end.
        let xv = topo.add_vertex(Vertex::new(x, 1e-7));
        let l1 = topo.add_edge(Edge::new(patch.e, xv, EdgeCurve::Line));
        let l2 = topo.add_edge(Edge::new(xv, patch.corner, EdgeCurve::Line));
        let to_far = edge_piece(topo, second, te, xv, second_edge.end() == far)?;
        let to_near = edge_piece(topo, second, te, xv, second_edge.end() == near)?;
        let starts = |oe: OrientedEdge, v: VertexId| {
            let edge = &if oe.edge() == l_id {
                l_edge.clone()
            } else {
                second_edge.clone()
            };
            (if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            }) == v
        };
        let paths = |topo: &Topology, steps: &[(EdgeId, VertexId)]| path(topo, steps);
        let ledge_l = paths(topo, &[(l1, patch.e)])?;
        let ledge_l_back = paths(topo, &[(l1, xv)])?;
        let ledge_far = paths(topo, &[(to_far, xv)])?;
        let ledge_far_back = paths(topo, &[(to_far, far)])?;
        rewrite_wire(topo, ledge_wire, |oe| {
            if oe.edge() == l_id {
                if starts(oe, patch.e) {
                    ledge_l.clone()
                } else {
                    ledge_l_back.clone()
                }
            } else if oe.edge() == first {
                vec![]
            } else if oe.edge() == second {
                if starts(oe, near) {
                    ledge_far.clone()
                } else {
                    ledge_far_back.clone()
                }
            } else {
                vec![oe]
            }
        })?;
        let other_on = paths(topo, &[(to_near, near), (to_far, xv)])?;
        let other_back = paths(topo, &[(to_far, far), (to_near, xv)])?;
        rewrite_wire(topo, other_wire, |oe| {
            if oe.edge() == second {
                if starts(oe, near) {
                    other_on.clone()
                } else {
                    other_back.clone()
                }
            } else {
                vec![oe]
            }
        })?;
        let patch_wire = topo.face(face_id)?.outer_wire();
        let patch_on = paths(topo, &[(l1, patch.e), (l2, xv)])?;
        let patch_back = paths(topo, &[(l2, patch.corner), (l1, xv)])?;
        rewrite_wire(topo, patch_wire, |oe| {
            if oe.edge() == l_id {
                if starts(oe, patch.e) {
                    patch_on.clone()
                } else {
                    patch_back.clone()
                }
            } else {
                vec![oe]
            }
        })?;
        // The corner cut off, its loop the ledge's way round: the short edge
        // and the crossed edge's piece to `X` as the ledge ran them, then
        // `l` back to the corner.
        let loop_edges = if backward {
            paths(topo, &[(to_near, xv), (first, near), (l2, patch.corner)])?
        } else {
            paths(topo, &[(first, patch.corner), (to_near, near), (l2, xv)])?
        };
        let wire = topo.add_wire(Wire::new(loop_edges, true)?);
        let ledge_face = topo.face(ledge)?;
        let mut corner = Face::new(wire, Vec::new(), ledge_face.surface().clone());
        corner.set_reversed(ledge_face.is_reversed());
        log::debug!(
            "runout {face_id:?}: its line cut off the ledge's corner {:?}, now a face",
            patch.corner
        );
        return Ok(Some(topo.add_face(corner)));
    }
    Ok(None)
}

/// Every use of `edge` among `faces` but `face_id`'s: the face and wire.
fn uses_of(
    topo: &Topology,
    faces: &[FaceId],
    skip: FaceId,
    edge: EdgeId,
) -> Result<Vec<(FaceId, WireId)>, BlendError> {
    let mut uses = Vec::new();
    for &other in faces {
        if other == skip {
            continue;
        }
        let f = topo.face(other)?;
        for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oe in topo.wire(w)?.edges() {
                if oe.edge() == edge {
                    uses.push((other, w));
                }
            }
        }
    }
    Ok(uses)
}

/// The one use of `edge` among `faces` but `skip`'s.
fn one_use(
    topo: &Topology,
    faces: &[FaceId],
    skip: FaceId,
    edge: EdgeId,
) -> Result<Option<(FaceId, WireId)>, BlendError> {
    Ok(match uses_of(topo, faces, skip, edge)?.as_slice() {
        [one] => Some(*one),
        _ => None,
    })
}

/// A point's offset from `surface` along its normal there (either way, the
/// same way at every point near the surface), `None` where it does not
/// project on to it.
fn surface_offset(surface: &FaceSurface, point: Point3) -> Option<f64> {
    if let FaceSurface::Plane { normal, d } = surface {
        return Some(normal.dot(Vec3::new(point.x(), point.y(), point.z())) - d);
    }
    let (u, v) = surface.project_point(point)?;
    let on = surface.evaluate(u, v)?;
    let normal = surface.normal(u, v).normalize().ok()?;
    Some((point - on).dot(normal))
}

/// A wall's surface carried on past its edges, so points of it above the
/// ledge lie on it and project on to it: planes and the analytic surfaces
/// run on already; a B-spline wall straight in one direction (an extrusion:
/// one span that way, its control points evenly spaced along a line and
/// weighted alike, a straight line at any degree) is rebuilt of degree one
/// that way and run on `reach` times its span each way, its parameters
/// unchanged where it was. `None` for any other wall.
fn run_on_surface(surface: &FaceSurface, reach: f64) -> Option<FaceSurface> {
    let FaceSurface::Nurbs(nurbs) = surface else {
        return Some(surface.clone());
    };
    let points = nurbs.control_points();
    let weights = nurbs.weights();
    // `line` is evenly spaced along a line from its first point to its last,
    // weighted alike.
    let straight = |line: &[Point3], line_weights: &[f64]| -> bool {
        let n = line.len();
        if n < 2
            || line_weights
                .iter()
                .any(|&w| (w - line_weights[0]).abs() > 1e-12 * w.abs())
        {
            return false;
        }
        let (a, b) = (line[0], line[n - 1]);
        let tol = 1e-9 * (b - a).length().max(1.0);
        line.iter().enumerate().all(|(k, &p)| {
            #[allow(clippy::cast_precision_loss)]
            let f = k as f64 / (n - 1) as f64;
            (p - (a + (b - a) * f)).length() <= tol
        })
    };
    let one_span = |knots: &[f64], degree: usize| knots.len() == 2 * (degree + 1);
    let run = |a: Point3, b: Point3| (a + (a - b) * reach, b + (b - a) * reach);
    let stretch = |knots: &[f64]| {
        let (lo, hi) = (knots[0], knots[knots.len() - 1]);
        let span = (hi - lo) * reach;
        vec![lo - span, lo - span, hi + span, hi + span]
    };
    let rows = points.len();
    let cols = points.first().map_or(0, Vec::len);
    // Straight along v: every row.
    if one_span(nurbs.knots_v(), nurbs.degree_v())
        && points.iter().zip(weights).all(|(row, w)| straight(row, w))
    {
        let grid: Vec<Vec<Point3>> = points
            .iter()
            .map(|row| {
                let (a, b) = run(row[0], row[cols - 1]);
                vec![a, b]
            })
            .collect();
        let grid_weights: Vec<Vec<f64>> = weights.iter().map(|w| vec![w[0], w[0]]).collect();
        let surface = NurbsSurface::new(
            nurbs.degree_u(),
            1,
            nurbs.knots_u().to_vec(),
            stretch(nurbs.knots_v()),
            grid,
            grid_weights,
        )
        .ok()?;
        return Some(FaceSurface::Nurbs(surface));
    }
    // Straight along u: every column.
    let column = |j: usize| -> (Vec<Point3>, Vec<f64>) {
        (
            points.iter().map(|row| row[j]).collect(),
            weights.iter().map(|w| w[j]).collect(),
        )
    };
    if rows >= 2
        && one_span(nurbs.knots_u(), nurbs.degree_u())
        && (0..cols).all(|j| {
            let (line, w) = column(j);
            straight(&line, &w)
        })
    {
        let (first, last): (Vec<Point3>, Vec<Point3>) = points[0]
            .iter()
            .zip(&points[rows - 1])
            .map(|(&a, &b)| run(a, b))
            .unzip();
        let surface = NurbsSurface::new(
            1,
            nurbs.degree_v(),
            stretch(nurbs.knots_u()),
            nurbs.knots_v().to_vec(),
            vec![first, last],
            vec![weights[0].clone(), weights[rows - 1].clone()],
        )
        .ok()?;
        return Some(FaceSurface::Nurbs(surface));
    }
    None
}

/// The band's surface run on past its end section `b` by about the runout
/// patch's size, its parameters unchanged where it was (the last span's
/// polynomial carried on). `None` for a band that is not a B-spline, or
/// whose end section is not on an end of its domain.
fn band_run_on(
    topo: &Topology,
    patch: &Patch,
    band: &FaceSurface,
) -> Result<Option<FaceSurface>, BlendError> {
    let FaceSurface::Nurbs(nurbs) = band else {
        return Ok(None);
    };
    let [pp, cp, ep] = [patch.p, patch.corner, patch.e].map(|v| topo.vertex(v).map(Vertex::point));
    let (pp, cp, ep) = (pp?, cp?, ep?);
    let reach = (pp - cp).length().max((ep - cp).length());
    let b_edge = topo.edge(patch.b.0)?.clone();
    let (curve, start, end) = edge_curve(topo, &b_edge)?;
    let (t0, t1) = curve.domain_with_endpoints(start, end);
    let middle = curve.evaluate_with_endpoints(f64::midpoint(t0, t1), start, end);
    let Some((u, v)) = band.project_point(middle) else {
        return Ok(None);
    };
    let ((u0, u1), (v0, v1)) = (nurbs.domain_u(), nurbs.domain_v());
    let at = |x: f64, end: f64, lo: f64, hi: f64| (x - end).abs() <= 1e-6 * (hi - lo);
    let (along_v, at_end, lo, hi) = if at(v, v1, v0, v1) {
        (true, true, v0, v1)
    } else if at(v, v0, v0, v1) {
        (true, false, v0, v1)
    } else if at(u, u1, u0, u1) {
        (false, true, u0, u1)
    } else if at(u, u0, u0, u1) {
        (false, false, u0, u1)
    } else {
        return Ok(None);
    };
    let edge_of = if at_end { hi } else { lo };
    let step = 1e-3 * (hi - lo) * if at_end { -1.0 } else { 1.0 };
    let point = |t: f64| {
        if along_v {
            nurbs.evaluate(u, t)
        } else {
            nurbs.evaluate(t, v)
        }
    };
    let speed = (point(edge_of) - point(edge_of + step)).length() / step.abs();
    if !(speed > 0.0 && reach > 0.0) {
        return Ok(None);
    }
    Ok(nurbs_run_on(nurbs, along_v, at_end, reach / speed).map(FaceSurface::Nurbs))
}

/// `surface` run on by `by` in its parameter past one end of its domain
/// (along v or u, at the end or the start).
fn nurbs_run_on(
    surface: &NurbsSurface,
    along_v: bool,
    at_end: bool,
    by: f64,
) -> Option<NurbsSurface> {
    let mut work = surface.clone();
    if !along_v {
        work = nurbs_transposed(&work)?;
    }
    if !at_end {
        work = nurbs_mirrored_v(&work)?;
    }
    let mut out = nurbs_run_on_v_end(&work, by)?;
    if !at_end {
        out = nurbs_mirrored_v(&out)?;
    }
    if !along_v {
        out = nurbs_transposed(&out)?;
    }
    Some(out)
}

/// `surface` with u and v swapped.
fn nurbs_transposed(surface: &NurbsSurface) -> Option<NurbsSurface> {
    let (points, weights) = (surface.control_points(), surface.weights());
    let cols = points.first().map_or(0, Vec::len);
    let swap = |grid: &[Vec<f64>]| -> Vec<Vec<f64>> {
        (0..cols)
            .map(|j| grid.iter().map(|row| row[j]).collect())
            .collect()
    };
    let swapped: Vec<Vec<Point3>> = (0..cols)
        .map(|j| points.iter().map(|row| row[j]).collect())
        .collect();
    NurbsSurface::new(
        surface.degree_v(),
        surface.degree_u(),
        surface.knots_v().to_vec(),
        surface.knots_u().to_vec(),
        swapped,
        swap(weights),
    )
    .ok()
}

/// `surface` with v run the other way over the same domain.
fn nurbs_mirrored_v(surface: &NurbsSurface) -> Option<NurbsSurface> {
    let knots = surface.knots_v();
    let (lo, hi) = (*knots.first()?, *knots.last()?);
    let mirrored: Vec<f64> = knots.iter().rev().map(|&t| lo + hi - t).collect();
    let flip = |row: &Vec<Point3>| row.iter().rev().copied().collect::<Vec<_>>();
    let flip_w = |row: &Vec<f64>| row.iter().rev().copied().collect::<Vec<_>>();
    NurbsSurface::new(
        surface.degree_u(),
        surface.degree_v(),
        surface.knots_u().to_vec(),
        mirrored,
        surface.control_points().iter().map(flip).collect(),
        surface.weights().iter().map(flip_w).collect(),
    )
    .ok()
}

/// `surface` run on by `by` past the end of v: the last span's knot
/// raised to the degree, so its control points are that span's Bezier
/// points, and those extrapolated by de Casteljau's steps.
fn nurbs_run_on_v_end(surface: &NurbsSurface, by: f64) -> Option<NurbsSurface> {
    let degree = surface.degree_v();
    let knots = surface.knots_v();
    let end = *knots.last()?;
    let span_start = knots.iter().rev().copied().find(|&t| t < end)?;
    let multiplicity = knots
        .iter()
        .filter(|&&t| (t - span_start).abs() <= 1e-12 * (end - span_start))
        .count();
    let surface = if multiplicity < degree {
        brepkit_math::nurbs::knot_ops::surface_knot_insert_v(
            surface,
            span_start,
            degree - multiplicity,
        )
        .ok()?
    } else {
        surface.clone()
    };
    let t = (end + by - span_start) / (end - span_start);
    let mut points = surface.control_points().to_vec();
    let mut weights = surface.weights().to_vec();
    let cols = points.first().map_or(0, Vec::len);
    if cols < degree + 1 {
        return None;
    }
    let first = cols - degree - 1;
    for (row, row_weights) in points.iter_mut().zip(weights.iter_mut()) {
        let mut bezier: Vec<[f64; 4]> = (first..cols)
            .map(|j| {
                let (p, w) = (row[j], row_weights[j]);
                [p.x() * w, p.y() * w, p.z() * w, w]
            })
            .collect();
        let mut left = vec![bezier[0]];
        for level in 1..=degree {
            for k in 0..=(degree - level) {
                let (a, b) = (bezier[k], bezier[k + 1]);
                bezier[k] = [0, 1, 2, 3].map(|i| a[i] + (b[i] - a[i]) * t);
            }
            left.push(bezier[0]);
        }
        for (k, h) in left.iter().enumerate() {
            if h[3] <= 0.0 {
                return None;
            }
            row[first + k] = Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
            row_weights[first + k] = h[3];
        }
    }
    let mut run_knots = surface.knots_v().to_vec();
    let len = run_knots.len();
    for knot in &mut run_knots[len - degree - 1..] {
        *knot = end + by;
    }
    NurbsSurface::new(
        surface.degree_u(),
        degree,
        surface.knots_u().to_vec(),
        run_knots,
        points,
        weights,
    )
    .ok()
}

/// An edge's curve between its stored ends, and those ends' points.
fn edge_curve(topo: &Topology, edge: &Edge) -> Result<(EdgeCurve, Point3, Point3), BlendError> {
    Ok((
        edge.curve().clone(),
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    ))
}

/// The edge's piece from the point at parameter `t` to its `keep_end`
/// (its stored end when true, else its start), from `split` (a vertex at
/// that point), in the edge's stored direction.
fn edge_piece(
    topo: &mut Topology,
    edge_id: EdgeId,
    t: f64,
    split: VertexId,
    keep_end: bool,
) -> Result<EdgeId, BlendError> {
    let edge = topo.edge(edge_id)?.clone();
    let curve = match edge.curve() {
        EdgeCurve::NurbsCurve(nurbs) => {
            let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(nurbs, t)?;
            EdgeCurve::NurbsCurve(if keep_end { right } else { left })
        }
        other => other.clone(),
    };
    Ok(topo.add_edge(if keep_end {
        Edge::new(split, edge.end(), curve)
    } else {
        Edge::new(edge.start(), split, curve)
    }))
}

/// Edges run from vertex to vertex, each taken the way that starts it at
/// its `from`.
fn path(topo: &Topology, steps: &[(EdgeId, VertexId)]) -> Result<Vec<OrientedEdge>, BlendError> {
    steps
        .iter()
        .map(|&(edge, from)| Ok(OrientedEdge::new(edge, topo.edge(edge)?.start() == from)))
        .collect()
}

/// The path reversed: its edges in the other order, each the other way.
fn reversed(path: &[OrientedEdge]) -> Vec<OrientedEdge> {
    path.iter()
        .rev()
        .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
        .collect()
}

/// Where along `curve` (between `t0` and `t1`, from `a` to `b`) it crosses
/// the edge: its parameter and point nearest the edge, where they come
/// within `tol`.
fn crossing(
    topo: &Topology,
    edge: EdgeId,
    (curve, a, b): (&EdgeCurve, Point3, Point3),
    (t0, t1): (f64, f64),
    tol: f64,
) -> Result<Option<(f64, Point3)>, BlendError> {
    const SAMPLES: usize = 96;
    #[allow(clippy::cast_precision_loss)]
    let param = |k: usize| t0 + (t1 - t0) * (k as f64) / (SAMPLES as f64);
    let gap = |t: f64| -> Result<f64, BlendError> {
        Ok(edge_nearest(topo, edge, curve.evaluate_with_endpoints(t, a, b), None)?.0)
    };
    let mut best = (0, f64::INFINITY);
    for k in 0..=SAMPLES {
        let g = gap(param(k))?;
        if g < best.1 {
            best = (k, g);
        }
    }
    let (mut lo, mut hi) = (
        param(best.0.saturating_sub(1)),
        param((best.0 + 1).min(SAMPLES)),
    );
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..80 {
        let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if gap(x1)? <= gap(x2)? {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    let t = f64::midpoint(lo, hi);
    if gap(t)? > tol {
        return Ok(None);
    }
    Ok(Some((t, curve.evaluate_with_endpoints(t, a, b))))
}

/// The parameter of `curve` (between `t0` and `t1`, from `a` to `b`)
/// nearest `point`, and the gap there.
fn nearest_param(
    (curve, a, b): (&EdgeCurve, Point3, Point3),
    (t0, t1): (f64, f64),
    point: Point3,
) -> (f64, f64) {
    const SAMPLES: usize = 256;
    #[allow(clippy::cast_precision_loss)]
    let param = |k: usize| t0 + (t1 - t0) * (k as f64) / (SAMPLES as f64);
    let gap = |t: f64| (curve.evaluate_with_endpoints(t, a, b) - point).length();
    let k = (0..=SAMPLES)
        .min_by(|&i, &j| gap(param(i)).total_cmp(&gap(param(j))))
        .unwrap_or(0);
    let (mut lo, mut hi) = (param(k.saturating_sub(1)), param((k + 1).min(SAMPLES)));
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..80 {
        let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if gap(x1) <= gap(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    let t = f64::midpoint(lo, hi);
    (t, gap(t))
}

/// Where the line from `from` along `up` meets the band: by a scan out to
/// `reach` and bisection on the point's offset from the band. `None` where
/// it does not within `reach`, or meets it farther than `tol` off.
fn line_meets_band(
    band: &FaceSurface,
    from: Point3,
    up: Vec3,
    reach: f64,
    tol: f64,
) -> Option<Point3> {
    const SAMPLES: usize = 200;
    let offset = |s: f64| surface_offset(band, from + up * s);
    #[allow(clippy::cast_precision_loss)]
    let at = |k: usize| reach * (k as f64) / (SAMPLES as f64);
    let mut last = offset(0.0)?;
    let mut bracket = None;
    for k in 1..=SAMPLES {
        let next = offset(at(k))?;
        if (last <= 0.0) != (next <= 0.0) {
            bracket = Some((at(k - 1), at(k), last <= 0.0));
            break;
        }
        last = next;
    }
    let (mut lo, mut hi, low_inside) = bracket?;
    for _ in 0..100 {
        let middle = f64::midpoint(lo, hi);
        if (offset(middle)? <= 0.0) == low_inside {
            lo = middle;
        } else {
            hi = middle;
        }
    }
    let point = from + up * f64::midpoint(lo, hi);
    let (u, v) = band.project_point(point)?;
    ((band.evaluate(u, v)? - point).length() <= tol).then_some(point)
}

/// A runout patch's roles: the band's end section `b`, the wall's run up
/// `w` and the line on the ledge `l` (each with the face and wire beyond
/// it), the corner `C`, the wall contact's end `P` and the ledge contact's
/// end `E`.
struct Patch {
    b: (EdgeId, FaceId, WireId),
    w: EdgeId,
    l: (EdgeId, FaceId, WireId),
    wall: (FaceId, WireId),
    corner: VertexId,
    p: VertexId,
    e: VertexId,
}

/// The ways the three-edge patch `face_id` could be a runout patch.
fn patches(
    topo: &Topology,
    faces: &[FaceId],
    face_id: FaceId,
    input_edges: &std::collections::HashSet<EdgeId>,
) -> Result<Vec<Patch>, BlendError> {
    let face = topo.face(face_id)?;
    if !face.inner_wires().is_empty() {
        return Ok(Vec::new());
    }
    let ring = topo.wire(face.outer_wire())?.edges().to_vec();
    if ring.len() != 3 || ring.iter().any(|oe| input_edges.contains(&oe.edge())) {
        return Ok(Vec::new());
    }
    let mut users = Vec::with_capacity(3);
    for oe in &ring {
        let Some(user) = one_use(topo, faces, face_id, oe.edge())? else {
            return Ok(Vec::new());
        };
        users.push(user);
    }
    let mut found = Vec::new();
    for b in 0..3 {
        if !matches!(
            topo.face(users[b].0)?.surface(),
            FaceSurface::Nurbs(_) | FaceSurface::Torus(_) | FaceSurface::Cylinder(_)
        ) {
            continue;
        }
        for (kw, kl) in [((b + 1) % 3, (b + 2) % 3), ((b + 2) % 3, (b + 1) % 3)] {
            let (be, we, le) = (
                topo.edge(ring[b].edge())?,
                topo.edge(ring[kw].edge())?,
                topo.edge(ring[kl].edge())?,
            );
            let ends = |e: &Edge| [e.start(), e.end()];
            let Some(corner) = ends(we).into_iter().find(|v| ends(le).contains(v)) else {
                continue;
            };
            let other = |e: &Edge| {
                if e.start() == corner {
                    e.end()
                } else {
                    e.start()
                }
            };
            let (p, e) = (other(we), other(le));
            if p == e || !ends(be).contains(&p) || !ends(be).contains(&e) {
                continue;
            }
            let (wall, ledge, band) = (users[kw].0, users[kl].0, users[b].0);
            if wall == ledge || wall == band || ledge == band {
                continue;
            }
            found.push(Patch {
                b: (ring[b].edge(), users[b].0, users[b].1),
                w: ring[kw].edge(),
                l: (ring[kl].edge(), users[kl].0, users[kl].1),
                wall: users[kw],
                corner,
                p,
                e,
            });
        }
    }
    Ok(found)
}

/// Has the walls under a ledge meet a round run off it, where `face_id` is
/// the round's runout patch (see the module's docs). Returns whether it
/// did; the patch is then to go.
#[allow(clippy::too_many_lines)]
fn run_walls_up(
    topo: &mut Topology,
    faces: &[FaceId],
    face_id: FaceId,
    input_edges: &std::collections::HashSet<EdgeId>,
) -> Result<bool, BlendError> {
    for patch in patches(topo, faces, face_id, input_edges)? {
        let band = topo.face(patch.b.1)?.surface().clone();
        let mut plan = plan_run_up(topo, faces, face_id, &patch, &band)?;
        // The ledge's outline may pass a little beyond the band's end
        // section before the contact crosses it: the band run on past it.
        if plan.is_none()
            && let Some(longer) = band_run_on(topo, &patch, &band)?
        {
            plan = plan_run_up(topo, faces, face_id, &patch, &longer)?.map(|plan| RunUp {
                band: Some(longer),
                ..plan
            });
        }
        if let Some(plan) = plan {
            build_run_up(topo, &patch, plan)?;
            log::debug!(
                "runout {face_id:?}: the round runs off its ledge at corner {:?}, the walls under it meet the band",
                patch.corner
            );
            return Ok(true);
        }
    }
    Ok(false)
}

/// Where a convex round's `P` lies on the edge its wall and the first wall
/// meet in at the corner: that edge and `P`'s parameter on it, `None`
/// where there is no such edge or `P` is off it.
fn corner_cut(
    topo: &Topology,
    patch: &Patch,
    first: &(FaceId, WireId, EdgeId, VertexId),
    pp: Point3,
    tol: f64,
) -> Result<Option<(EdgeId, f64)>, BlendError> {
    let wall_edges = topo.wire(patch.wall.1)?.edges().to_vec();
    let first_edges = topo.wire(first.1)?.edges().to_vec();
    let Some(join) = wall_edges.iter().map(OrientedEdge::edge).find(|&e| {
        e != patch.w
            && e != first.2
            && first_edges.iter().any(|oe| oe.edge() == e)
            && topo
                .edge(e)
                .is_ok_and(|edge| edge.start() == patch.corner || edge.end() == patch.corner)
    }) else {
        return Ok(None);
    };
    let edge = topo.edge(join)?.clone();
    let (curve, a, b) = edge_curve(topo, &edge)?;
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    let (t, gap) = nearest_param((&curve, a, b), (t0, t1), pp);
    let inside = (t - t0.min(t1)).min(t0.max(t1) - t) > 1e-6 * (t1 - t0).abs();
    Ok((gap <= tol && inside).then_some((join, t)))
}

/// Where the band crosses `edge`, first from its end `from`: the edge's
/// parameter and the point, `None` where it does not cross it.
fn band_crossing_on_edge(
    topo: &Topology,
    band: &FaceSurface,
    edge_id: EdgeId,
    from: VertexId,
    tol: f64,
) -> Result<Option<(f64, Point3)>, BlendError> {
    const SAMPLES: usize = 200;
    let edge = topo.edge(edge_id)?.clone();
    let (curve, a, b) = edge_curve(topo, &edge)?;
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    let (ta, tb) = if edge.start() == from {
        (t0, t1)
    } else {
        (t1, t0)
    };
    let offset = |t: f64| surface_offset(band, curve.evaluate_with_endpoints(t, a, b));
    #[allow(clippy::cast_precision_loss)]
    let at = |k: usize| ta + (tb - ta) * (k as f64) / (SAMPLES as f64);
    let Some(mut last) = offset(at(1)) else {
        return Ok(None);
    };
    let mut bracket = None;
    for k in 2..=SAMPLES {
        let Some(next) = offset(at(k)) else {
            return Ok(None);
        };
        if (last <= 0.0) != (next <= 0.0) {
            bracket = Some((at(k - 1), at(k), last <= 0.0));
            break;
        }
        last = next;
    }
    let Some((mut lo, mut hi, low_inside)) = bracket else {
        return Ok(None);
    };
    for _ in 0..100 {
        let middle = f64::midpoint(lo, hi);
        match offset(middle) {
            Some(off) if (off <= 0.0) == low_inside => lo = middle,
            Some(_) => hi = middle,
            None => return Ok(None),
        }
    }
    let t = f64::midpoint(lo, hi);
    let point = curve.evaluate_with_endpoints(t, a, b);
    let on = band
        .project_point(point)
        .and_then(|(u, v)| band.evaluate(u, v))
        .is_some_and(|q| (q - point).length() <= tol);
    Ok(on.then_some((t, point)))
}

/// The band's intersection with a wall from `a` to `b`: for a fitted band
/// its rows followed to the wall, for a torus its parallels, for a
/// cylinder its rulings.
fn band_meets_wall(
    band: &FaceSurface,
    wall: &FaceSurface,
    a: Point3,
    b: Point3,
    tol: f64,
) -> Option<EdgeCurve> {
    match band {
        FaceSurface::Torus(torus) => match wall {
            FaceSurface::Plane { normal, d } => torus_meets_cap(torus, a, b, |v, near| {
                torus_parallel_meets_plane(torus, *normal, *d, v, near)
            }),
            _ => torus_meets_cap(torus, a, b, |v, near| {
                torus_parallel_meets_surface(torus, wall, v, near, tol)
            }),
        },
        FaceSurface::Cylinder(cylinder) => match wall {
            FaceSurface::Plane { normal, d } => {
                crate::builder_utils::cylinder_plane_arc(cylinder, *normal, *d, a, b)
            }
            _ => cylinder_meets_cap(cylinder, wall, a, b),
        },
        _ => band_meets_cap(band, a, b, |point| surface_offset(wall, point)),
    }
}

/// Where the band meets the edge two walls meet in: that edge's end on the
/// ledge, the point, and (where the band crosses the edge itself) the edge
/// and the point's parameter on it.
type Top = (VertexId, Point3, Option<(EdgeId, f64)>);

/// What the run up builds: the walls' faces and wires and the outline's
/// edges they meet the ledge in (from the corner on), the walls' surfaces
/// run on, the points where the band meets the runs up between walls, the
/// crossing `X` (on the contact `c` at `tc`, on the last edge at `te`), and
/// the band's intersection with each wall.
struct RunUp {
    walls: Vec<(FaceId, WireId, EdgeId, VertexId)>,
    surfaces: Vec<FaceSurface>,
    tops: Vec<Top>,
    corner: Option<(EdgeId, f64)>,
    contact: EdgeId,
    tc: f64,
    te: f64,
    x: Point3,
    meets: Vec<EdgeCurve>,
    /// The band's surface run on past its end, where the plan needed it.
    band: Option<FaceSurface>,
}

/// Plans a runout patch's run up, `None` where the round does not run off
/// its ledge so (its contact crossing no edge of the ledge's outline from
/// the corner, a wall not straight up past the ledge, the band missing a
/// wall).
#[allow(clippy::too_many_lines)]
fn plan_run_up(
    topo: &Topology,
    faces: &[FaceId],
    face_id: FaceId,
    patch: &Patch,
    band: &FaceSurface,
) -> Result<Option<RunUp>, BlendError> {
    let (b_id, band_face, band_wire) = patch.b;
    let (l_id, ledge, ledge_wire) = patch.l;
    let [pp, cp, ep] = [patch.p, patch.corner, patch.e].map(|v| topo.vertex(v).map(Vertex::point));
    let (pp, cp, ep) = (pp?, cp?, ep?);
    let scale = (pp - cp).length().max((ep - cp).length()).max(1e-9);
    let tol = 1e-6 * scale.max(1.0);
    // The band's contact on the ledge, from `E`.
    let band_edges = topo.wire(band_wire)?.edges().to_vec();
    let ledge_edges = topo.wire(ledge_wire)?.edges().to_vec();
    let Some(contact) = band_edges.iter().map(OrientedEdge::edge).find(|&e| {
        e != b_id
            && ledge_edges.iter().any(|oe| oe.edge() == e)
            && topo
                .edge(e)
                .is_ok_and(|edge| edge.start() == patch.e || edge.end() == patch.e)
    }) else {
        return Ok(None);
    };
    let contact_edge = topo.edge(contact)?.clone();
    let (c_curve, c_start, c_end) = edge_curve(topo, &contact_edge)?;
    let (t0, t1) = c_curve.domain_with_endpoints(c_start, c_end);
    let from_e = if contact_edge.start() == patch.e {
        (t0, t1)
    } else {
        (t1, t0)
    };
    // The ledge's outline from the corner, away from `l`.
    let Some(at) = ledge_edges.iter().position(|oe| oe.edge() == l_id) else {
        return Ok(None);
    };
    let n = ledge_edges.len();
    let l_starts_at_corner = topo.edge(l_id)?.start() == patch.corner;
    // `l` taken forward runs from its start: the outline before it ends at
    // its oriented start.
    let l_oriented_start = if ledge_edges[at].is_forward() == l_starts_at_corner {
        patch.corner
    } else {
        patch.e
    };
    let backward = l_oriented_start == patch.corner;
    let mut walls = Vec::new();
    let mut from = patch.corner;
    let mut found = None;
    for step in 1..=MOST_WALLS.min(n.saturating_sub(2)) {
        let index = if backward {
            (at + n - step) % n
        } else {
            (at + step) % n
        };
        let edge_id = ledge_edges[index].edge();
        if edge_id == contact || band_edges.iter().any(|oe| oe.edge() == edge_id) {
            return Ok(None);
        }
        let edge = topo.edge(edge_id)?;
        if edge.start() != from && edge.end() != from {
            return Ok(None);
        }
        let to = if edge.start() == from {
            edge.end()
        } else {
            edge.start()
        };
        let Some((wall, wall_wire)) = one_use(topo, faces, ledge, edge_id)? else {
            return Ok(None);
        };
        // The round's own wall may run on under the ledge from the corner
        // (a wall the ledge crosses, the round on its upper part): it is
        // then the first wall, its edge up from the corner dropped.
        if wall == face_id || wall == band_face || (wall == patch.wall.0 && step > 1) {
            return Ok(None);
        }
        walls.push((wall, wall_wire, edge_id, from));
        if let Some((tc, x)) = crossing(topo, edge_id, (&c_curve, c_start, c_end), from_e, tol)? {
            // Not at the contact's far end, and not at the edge's ends.
            let (_, xe) = edge_nearest(topo, edge_id, x, None)?;
            let ends = [topo.vertex(from)?.point(), topo.vertex(to)?.point()];
            if ends.iter().any(|&end| (end - xe).length() <= 10.0 * tol)
                || (tc - from_e.1).abs() <= 1e-9 * (from_e.1 - from_e.0).abs()
            {
                return Ok(None);
            }
            found = Some((tc, x));
            break;
        }
        from = to;
    }
    let Some((tc, x)) = found else {
        return Ok(None);
    };
    // Each wall straight up past the ledge, `P` on the first.
    let mut surfaces = Vec::with_capacity(walls.len());
    for &(wall, ..) in &walls {
        let Some(surface) = run_on_surface(topo.face(wall)?.surface(), 2.0) else {
            return Ok(None);
        };
        surfaces.push(surface);
    }
    // The corner: a concave round's wall contact ends on the first wall's
    // surface run on above the ledge; a convex round's on the edge the
    // round's wall and the first wall meet in at the corner (the round's
    // wall running up it from `P` to the corner and back down).
    let corner = corner_cut(topo, patch, &walls[0], pp, tol)?;
    if corner.is_none() && surface_offset(&surfaces[0], pp).is_none_or(|off| off.abs() > tol) {
        return Ok(None);
    }
    // Between walls `i` and `i + 1`, the straight edge they meet in at the
    // ledge, run up to the band.
    let mut tops = Vec::new();
    for pair in walls.windows(2) {
        let (first, second) = (&pair[0], &pair[1]);
        let vertex = second.3;
        let (first_edges, second_edges) = (
            topo.wire(first.1)?.edges().to_vec(),
            topo.wire(second.1)?.edges().to_vec(),
        );
        let Some(join) = first_edges.iter().map(OrientedEdge::edge).find(|&e| {
            e != first.2
                && second_edges.iter().any(|oe| oe.edge() == e)
                && topo
                    .edge(e)
                    .is_ok_and(|edge| edge.start() == vertex || edge.end() == vertex)
        }) else {
            return Ok(None);
        };
        let join_edge = topo.edge(join)?;
        if !straight_edge(topo, join_edge, 1e-6 * scale)? {
            return Ok(None);
        }
        let below = if join_edge.start() == vertex {
            join_edge.end()
        } else {
            join_edge.start()
        };
        // A convex round's band crosses the edge below the ledge; a concave
        // round's meets the edge carried on above it.
        if let Some((t, point)) = band_crossing_on_edge(topo, band, join, vertex, tol)? {
            tops.push((vertex, point, Some((join, t))));
            continue;
        }
        let vp = topo.vertex(vertex)?.point();
        let up = (vp - topo.vertex(below)?.point()).normalize()?;
        let Some(top) = line_meets_band(band, vp, up, 4.0 * scale, tol) else {
            return Ok(None);
        };
        tops.push((vertex, top, None));
    }
    // The band's intersection with each wall, `P` to `X` across them.
    let mut ends = vec![pp];
    ends.extend(tops.iter().map(|&(_, top, _)| top));
    ends.push(x);
    let mut meets = Vec::with_capacity(walls.len());
    for (k, surface) in surfaces.iter().enumerate() {
        let Some(meet) = band_meets_wall(band, surface, ends[k], ends[k + 1], tol) else {
            return Ok(None);
        };
        meets.push(meet);
    }
    let last = walls[walls.len() - 1].2;
    let last_edge = topo.edge(last)?.clone();
    let (e_curve, e_start, e_end) = edge_curve(topo, &last_edge)?;
    let (s0, s1) = e_curve.domain_with_endpoints(e_start, e_end);
    let (te, gap) = nearest_param((&e_curve, e_start, e_end), (s0, s1), x);
    if gap > 10.0 * tol {
        return Ok(None);
    }
    Ok(Some(RunUp {
        walls,
        surfaces,
        tops,
        corner,
        contact,
        tc,
        te,
        x,
        meets,
        band: None,
    }))
}

/// Builds a planned run up: the walls run on (or cut down) and rewired,
/// the band's end and the ledge cut at `X`.
#[allow(clippy::too_many_lines)]
fn build_run_up(topo: &mut Topology, patch: &Patch, plan: RunUp) -> Result<(), BlendError> {
    let RunUp {
        walls,
        surfaces,
        tops,
        corner,
        contact,
        tc,
        te,
        x,
        meets,
        band,
    } = plan;
    let (b_id, band_face, band_wire) = patch.b;
    if let Some(band) = band {
        topo.face_mut(band_face)?.set_surface(band);
    }
    let (l_id, _, ledge_wire) = patch.l;
    let x_vertex = topo.add_vertex(Vertex::new(x, 1e-7));
    let top_vertices: Vec<VertexId> = tops
        .iter()
        .map(|&(_, top, _)| topo.add_vertex(Vertex::new(top, 1e-7)))
        .collect();
    // The chain of the band's intersections, `P` to `X`.
    let mut chain_vertices = vec![patch.p];
    chain_vertices.extend(top_vertices.iter().copied());
    chain_vertices.push(x_vertex);
    let chain: Vec<EdgeId> = meets
        .into_iter()
        .enumerate()
        .map(|(k, curve)| topo.add_edge(Edge::new(chain_vertices[k], chain_vertices[k + 1], curve)))
        .collect();
    // Between walls: a run up from the ledge to the band (concave), or the
    // edge they meet in cut where the band crosses it, its piece below kept
    // (convex). The same at the corner, between the round's wall and the
    // first wall.
    let mut runs: Vec<Option<EdgeId>> = Vec::with_capacity(tops.len());
    let mut pieces: Vec<(EdgeId, EdgeId)> = Vec::new();
    for (&(vertex, _, cut), &top) in tops.iter().zip(&top_vertices) {
        if let Some((join, t)) = cut {
            let keep_end = topo.edge(join)?.start() == vertex;
            pieces.push((join, edge_piece(topo, join, t, top, keep_end)?));
            runs.push(None);
        } else {
            runs.push(Some(topo.add_edge(Edge::new(vertex, top, EdgeCurve::Line))));
        }
    }
    if let Some((join, t)) = corner {
        let keep_end = topo.edge(join)?.start() == patch.corner;
        pieces.push((join, edge_piece(topo, join, t, patch.p, keep_end)?));
    }
    let piece_of = |edge: EdgeId| pieces.iter().find(|(whole, _)| *whole == edge).map(|p| p.1);
    // The contact from `X` on, and the last edge from `X` on.
    let contact_edge = topo.edge(contact)?.clone();
    let contact_kept = edge_piece(topo, contact, tc, x_vertex, contact_edge.start() == patch.e)?;
    let (_, _, last, last_from) = walls[walls.len() - 1];
    let last_edge = topo.edge(last)?.clone();
    let last_kept = edge_piece(topo, last, te, x_vertex, last_edge.start() == last_from)?;
    // The ledge: `l`, the outline up to `X` and the contact up to `X` go.
    let dropped: Vec<EdgeId> = walls[..walls.len() - 1]
        .iter()
        .map(|wall| wall.2)
        .chain([l_id])
        .collect();
    rewrite_wire(topo, ledge_wire, |oe| {
        if dropped.contains(&oe.edge()) {
            vec![]
        } else if oe.edge() == contact {
            vec![OrientedEdge::new(contact_kept, oe.is_forward())]
        } else if oe.edge() == last {
            vec![OrientedEdge::new(last_kept, oe.is_forward())]
        } else {
            vec![oe]
        }
    })?;
    // The band: the chain in place of its end section.
    let mut chain_steps = Vec::with_capacity(chain.len());
    for (k, &edge) in chain.iter().enumerate() {
        chain_steps.push((edge, chain_vertices[k]));
    }
    let forward_chain = path(topo, &chain_steps)?;
    let b_edge = topo.edge(b_id)?.clone();
    rewrite_wire(topo, band_wire, |oe| {
        if oe.edge() == b_id {
            // `b` taken this way runs from `P` when its oriented start is.
            let from_p = (b_edge.start() == patch.p) == oe.is_forward();
            if from_p {
                forward_chain.clone()
            } else {
                reversed(&forward_chain)
            }
        } else if oe.edge() == contact {
            vec![OrientedEdge::new(contact_kept, oe.is_forward())]
        } else {
            vec![oe]
        }
    })?;
    // The round's wall: a convex round's spur up the corner and back goes,
    // the corner edge's piece below `P` in its place.
    if corner.is_some() {
        rewrite_wire(topo, patch.wall.1, |oe| {
            if oe.edge() == patch.w {
                vec![]
            } else if let Some(piece) = piece_of(oe.edge()) {
                vec![OrientedEdge::new(piece, oe.is_forward())]
            } else {
                vec![oe]
            }
        })?;
    }
    // Each wall: its piece of the outline replaced by the run up (where
    // concave), the band's intersection and the run down (where concave),
    // the last wall's by those and its edge from `X` on; the edges it
    // meets its neighbours in replaced by their pieces below the band
    // (where convex).
    let own_wall = walls[0].0 == patch.wall.0;
    for (k, &(wall, wire, edge, from)) in walls.iter().enumerate() {
        let mut steps = Vec::with_capacity(3);
        let up = if k == 0 {
            (corner.is_none() && !own_wall).then_some(patch.w)
        } else {
            runs[k - 1]
        };
        if let Some(up) = up {
            steps.push((up, from));
        }
        // On the round's own wall `w`, from the corner up to `P`, goes: the
        // wall's outline already reaches `P` down `w`'s neighbour.
        let drop_w = k == 0 && own_wall;
        steps.push((chain[k], chain_vertices[k]));
        if k + 1 < walls.len() {
            if let Some(down) = runs[k] {
                steps.push((down, chain_vertices[k + 1]));
            }
        } else {
            steps.push((last_kept, x_vertex));
        }
        let forward = path(topo, &steps)?;
        let stored = topo.edge(edge)?.clone();
        topo.face_mut(wall)?.set_surface(surfaces[k].clone());
        rewrite_wire(topo, wire, |oe| {
            if drop_w && oe.edge() == patch.w {
                vec![]
            } else if oe.edge() == edge {
                // The piece taken this way runs from `from` when its oriented
                // start is (from `P` on the round's own wall).
                let from_start = (stored.start() == from) == oe.is_forward();
                if from_start {
                    forward.clone()
                } else {
                    reversed(&forward)
                }
            } else if let Some(piece) = piece_of(oe.edge()) {
                vec![OrientedEdge::new(piece, oe.is_forward())]
            } else {
                vec![oe]
            }
        })?;
    }
    Ok(())
}
