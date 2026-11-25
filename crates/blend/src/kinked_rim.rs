//! A closed rim whose curve meets itself at an angle.
//!
//! A closed B-spline rim whose ends meet at an angle (an outline drawn as one
//! curve, its start and end tangents apart) has no smooth closed round: the
//! sections at its start and its end, square to two different tangents, are
//! not the same arc, so a band walked once round closes on neither. The wall
//! beside such a rim is one face whose seam runs down from the rim's vertex,
//! on both sides of the corner there.
//!
//! [`split_kinked_rims`] cuts the rim in two halves and the wall along its
//! ruling from the halfway point, so the rim is two edges between different
//! faces: two rounds, which meet at the rim's vertex as rounds of any two
//! edges at a corner beside a sharp edge do (mitred at a tip, pivoted through
//! a horn torus at a dimple), and meet halfway round at a tangent joint
//! across the ruling, where they share a section.

use brepkit_math::madd::Madd;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::knot_ops::curve_split;
use brepkit_math::nurbs::projection::project_point_to_curve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::radius_law::RadiusLaw;
use crate::trimmer::{self, SplitScope};

/// Angle between two tangents, in radians, past which the curves meet in a
/// corner rather than running on: a closed walk across a kink of `a` misses
/// its start by about `r a`, past any closing tolerance long before this.
const KINK: f64 = 1e-3;

/// How far, relative to the part's size, a point may stand off a curve, or a
/// ruling off its chord, and still be on it.
const ON: f64 = 1e-7;

/// The angle between the end and start tangents of a closed B-spline edge.
fn closure_kink(topo: &Topology, edge: EdgeId) -> Option<f64> {
    // An edge the topology does not hold fails on its own later.
    let Ok(data) = topo.edge(edge) else {
        return None;
    };
    if data.start() != data.end() {
        return None;
    }
    let EdgeCurve::NurbsCurve(nurbs) = data.curve() else {
        return None;
    };
    let (u0, u1) = nurbs.domain();
    let (Ok(start), Ok(end)) = (
        nurbs.derivatives(u0, 1)[1].normalize(),
        nurbs.derivatives(u1, 1)[1].normalize(),
    ) else {
        return None;
    };
    Some(start.dot(end).clamp(-1.0, 1.0).acos())
}

/// The faces of `solid`'s outer shell whose outer wire uses `edge`.
fn faces_using(topo: &Topology, solid: SolidId, edge: EdgeId) -> Result<Vec<FaceId>, BlendError> {
    let shell = topo.shell(topo.solid(solid)?.outer_shell())?;
    let mut found = Vec::new();
    for &face_id in shell.faces() {
        let wire = topo.wire(topo.face(face_id)?.outer_wire())?;
        if wire.edges().iter().any(|oe| oe.edge() == edge) {
            found.push(face_id);
        }
    }
    Ok(found)
}

/// The straight ruling of a wall across its rim at a point.
struct Ruling {
    /// Where it ends across the face.
    far: Point3,
    /// Whether the rim runs along the surface's u (the ruling along v).
    rim_along_u: bool,
    /// The rim's parameter of the ruling.
    at: f64,
}

/// The ruling of `surface` through `point`, on the rim that also runs
/// through `along`: the parameter line across the rim's, which must be
/// straight.
fn ruling(surface: &FaceSurface, point: Point3, along: Point3) -> Option<Ruling> {
    let FaceSurface::Nurbs(nurbs) = surface else {
        return None;
    };
    let (u, v) = surface.project_point(point)?;
    let (u2, v2) = surface.project_point(along)?;
    let ((u0, u1), (v0, v1)) = (nurbs.domain_u(), nurbs.domain_v());
    // The rim runs along whichever parameter moves more between its two
    // points, as a share of its domain; the ruling runs along the other.
    let rim_along_u = (u2 - u).abs() / (u1 - u0) > (v2 - v).abs() / (v1 - v0);
    let at = |t: f64| {
        if rim_along_u {
            surface.evaluate(u, (v1 - v0).madd(t, v0))
        } else {
            surface.evaluate((u1 - u0).madd(t, u0), v)
        }
    };
    let (first, last) = (at(0.0)?, at(1.0)?);
    let far = if (first - point).length() > (last - point).length() {
        first
    } else {
        last
    };
    let chord = far - point;
    let length = chord.length();
    let direction = chord.normalize().ok()?;
    for k in 1..8 {
        let q = at(f64::from(k) / 8.0)?;
        let off = (q - point) - direction * (q - point).dot(direction);
        if off.length() > ON * length.max(1.0) {
            return None;
        }
    }
    Some(Ruling {
        far,
        rim_along_u,
        at: if rim_along_u { u } else { v },
    })
}

/// `surface` cut in two at `at` along the rim's parameter: the part before
/// and the part after.
fn split_surface(
    surface: &NurbsSurface,
    rim_along_u: bool,
    at: f64,
) -> Result<(NurbsSurface, NurbsSurface), BlendError> {
    let points = surface.control_points();
    let weights = surface.weights();
    let (rows, columns) = (points.len(), points.first().map_or(0, Vec::len));
    // The surface's lines along the rim's parameter, each cut at `at`.
    let lines = if rim_along_u { columns } else { rows };
    let mut halves = Vec::with_capacity(lines);
    for line in 0..lines {
        let (line_points, line_weights): (Vec<Point3>, Vec<f64>) = if rim_along_u {
            (0..rows)
                .map(|i| (points[i][line], weights[i][line]))
                .unzip()
        } else {
            (points[line].clone(), weights[line].clone())
        };
        let (degree, knots) = if rim_along_u {
            (surface.degree_u(), surface.knots_u().to_vec())
        } else {
            (surface.degree_v(), surface.knots_v().to_vec())
        };
        let curve = NurbsCurve::new(degree, knots, line_points, line_weights)?;
        halves.push(curve_split(&curve, at)?);
    }
    let part =
        |pick: fn(&(NurbsCurve, NurbsCurve)) -> &NurbsCurve| -> Result<NurbsSurface, BlendError> {
            let first = pick(&halves[0]);
            let count = first.control_points().len();
            let (grid, grid_weights): (Vec<Vec<Point3>>, Vec<Vec<f64>>) = if rim_along_u {
                (0..count)
                    .map(|i| {
                        halves
                            .iter()
                            .map(|h| (pick(h).control_points()[i], pick(h).weights()[i]))
                            .unzip()
                    })
                    .unzip()
            } else {
                halves
                    .iter()
                    .map(|h| {
                        (
                            pick(h).control_points().to_vec(),
                            pick(h).weights().to_vec(),
                        )
                    })
                    .unzip()
            };
            let surface = if rim_along_u {
                NurbsSurface::new(
                    surface.degree_u(),
                    surface.degree_v(),
                    first.knots().to_vec(),
                    surface.knots_v().to_vec(),
                    grid,
                    grid_weights,
                )
            } else {
                NurbsSurface::new(
                    surface.degree_u(),
                    surface.degree_v(),
                    surface.knots_u().to_vec(),
                    first.knots().to_vec(),
                    grid,
                    grid_weights,
                )
            };
            Ok(surface?)
        };
    Ok((part(|h| &h.0)?, part(|h| &h.1)?))
}

/// An edge cut in two, and its two pieces.
type Cut = (EdgeId, [EdgeId; 2]);

/// The vertex of `face`'s outer wire at `point`, splitting the line or
/// B-spline edge of the wire it lies on there when it is no vertex yet (the
/// edge cut is returned too).
fn vertex_on_wire(
    topo: &mut Topology,
    scope: &SplitScope,
    face: FaceId,
    point: Point3,
) -> Result<Option<(VertexId, Option<Cut>)>, BlendError> {
    let edges = topo.wire(topo.face(face)?.outer_wire())?.edges().to_vec();
    let tol = 10.0 * ON * (point - Point3::new(0.0, 0.0, 0.0)).length().max(1.0);
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        for vertex in [edge.start(), edge.end()] {
            if (topo.vertex(vertex)?.point() - point).length() <= tol {
                return Ok(Some((vertex, None)));
            }
        }
    }
    for oe in &edges {
        let edge = topo.edge(oe.edge())?.clone();
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        match edge.curve() {
            EdgeCurve::Line => {
                let chord = b - a;
                let t = (point - a).dot(chord) / chord.dot(chord);
                if t <= 0.0 || t >= 1.0 || (a + chord * t - point).length() > tol {
                    continue;
                }
                let vertex = topo.add_vertex(Vertex::new(point, ON));
                let (one, two) = trimmer::split_edge_at(
                    topo,
                    scope,
                    &OrientedEdge::new(oe.edge(), true),
                    vertex,
                )?;
                return Ok(Some((vertex, Some((oe.edge(), [one.edge(), two.edge()])))));
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                let hit = project_point_to_curve(nurbs, point, ON)?;
                let (d0, d1) = nurbs.domain();
                let margin = 1e-9 * (d1 - d0);
                if hit.distance > tol
                    || hit.parameter <= d0 + margin
                    || hit.parameter >= d1 - margin
                {
                    continue;
                }
                let (left, right) = curve_split(nurbs, hit.parameter)?;
                let vertex = topo.add_vertex(Vertex::new(hit.point, ON));
                let (one, two) = trimmer::split_edge_at_with_curves(
                    topo,
                    scope,
                    &OrientedEdge::new(oe.edge(), true),
                    vertex,
                    left,
                    right,
                )?;
                return Ok(Some((vertex, Some((oe.edge(), [one.edge(), two.edge()])))));
            }
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {}
        }
    }
    Ok(None)
}

/// Cut `face` in two along a straight edge between `from` and `to`, two
/// vertices its outer wire passes once each. The face keeps the run from
/// `from` to `to`, a new face on the same surface takes the run back, and the
/// solid's outer shell takes both.
fn cut_face(
    topo: &mut Topology,
    solid: SolidId,
    face: FaceId,
    from: VertexId,
    to: VertexId,
    (whole, rim_along_u, at): (&NurbsSurface, bool, f64),
    (before, after): (NurbsSurface, NurbsSurface),
) -> Result<bool, BlendError> {
    let data = topo.face(face)?.clone();
    let edges = topo.wire(data.outer_wire())?.edges().to_vec();
    let mut ends = Vec::with_capacity(edges.len());
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        ends.push(if oe.is_forward() {
            edge.end()
        } else {
            edge.start()
        });
    }
    if ends.iter().filter(|&&v| v == from).count() != 1
        || ends.iter().filter(|&&v| v == to).count() != 1
    {
        return Ok(false);
    }
    let (Some(p), Some(q)) = (
        ends.iter().position(|&v| v == from),
        ends.iter().position(|&v| v == to),
    ) else {
        return Ok(false);
    };
    let count = edges.len();
    // The wire's edges after the one ending at `after`, up to the one ending
    // at `until`.
    let run = |after: usize, until: usize| {
        let mut run = Vec::new();
        let mut i = (after + 1) % count;
        loop {
            run.push(edges[i]);
            if i == until {
                break;
            }
            i = (i + 1) % count;
        }
        run
    };
    let cut = topo.add_edge(Edge::new(from, to, EdgeCurve::Line));
    let mut kept = run(p, q);
    kept.push(OrientedEdge::new(cut, false));
    let mut other = run(q, p);
    other.push(OrientedEdge::new(cut, true));
    // The kept run starts along the rim from `from`: the middle of its first
    // edge tells which side of the ruling it lies on.
    let first = topo.edge(edges[(p + 1) % count].edge())?;
    let (a, b) = (
        topo.vertex(first.start())?.point(),
        topo.vertex(first.end())?.point(),
    );
    let (t0, t1) = first.curve().domain_with_endpoints(a, b);
    let middle = first
        .curve()
        .evaluate_with_endpoints((t1 - t0).madd(0.5, t0), a, b);
    let Some((u, v)) = FaceSurface::Nurbs(whole.clone()).project_point(middle) else {
        return Ok(false);
    };
    let kept_before = (if rim_along_u { u } else { v }) < at;
    let (kept_surface, other_surface) = if kept_before {
        (before, after)
    } else {
        (after, before)
    };
    let kept_wire = topo.add_wire(Wire::new(kept, true)?);
    let other_wire = topo.add_wire(Wire::new(other, true)?);
    topo.face_mut(face)?.set_outer_wire(kept_wire);
    topo.face_mut(face)?
        .set_surface(FaceSurface::Nurbs(kept_surface));
    let other_surface = FaceSurface::Nurbs(other_surface);
    let new_face = if data.is_reversed() {
        Face::new_reversed(other_wire, Vec::new(), other_surface)
    } else {
        Face::new(other_wire, Vec::new(), other_surface)
    };
    let new_face = topo.add_face(new_face);
    let shell_id = topo.solid(solid)?.outer_shell();
    let mut faces = topo.shell(shell_id)?.faces().to_vec();
    faces.push(new_face);
    *topo.shell_mut(shell_id)? = Shell::new(faces)?;
    Ok(true)
}

/// Cut each picked closed B-spline rim whose ends meet at an angle past
/// [`KINK`] in two halves at its middle, and its B-spline wall (a face
/// without holes, ruled straight across the rim) along the ruling from
/// there; the halves are picked in the rim's place. A rim with no such wall
/// is left whole.
///
/// # Errors
///
/// Returns [`BlendError`] if a topology lookup or a curve's split fails.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn split_kinked_rims(
    topo: &mut Topology,
    scope: &SplitScope,
    solid: SolidId,
    edge_sets: &mut [(Vec<EdgeId>, RadiusLaw)],
) -> Result<(), BlendError> {
    let picks: Vec<EdgeId> = edge_sets
        .iter()
        .flat_map(|(e, _)| e.iter().copied())
        .collect();
    // Each edge cut, by its pieces; a ruling cut across a wall from one rim
    // may cut another picked rim.
    let mut cuts = std::collections::HashMap::<EdgeId, [EdgeId; 2]>::new();
    for edge in picks {
        if cuts.contains_key(&edge) {
            continue;
        }
        let made = split_kinked_rim(topo, scope, solid, edge)?;
        if !made.is_empty() {
            log::debug!("kinked closed rim {edge:?} rounded as two halves");
        }
        cuts.extend(made);
    }
    for (edges, _) in edge_sets.iter_mut() {
        let mut picked = Vec::with_capacity(edges.len() + 2);
        let mut stack: Vec<EdgeId> = edges.iter().rev().copied().collect();
        while let Some(edge) = stack.pop() {
            match cuts.get(&edge) {
                Some(&[first, second]) => {
                    stack.push(second);
                    stack.push(first);
                }
                None => picked.push(edge),
            }
        }
        *edges = picked;
    }
    Ok(())
}

/// [`split_kinked_rims`] for one edge: the edges it cut (the rim, and the
/// edge across the wall the ruling ends on, if that was no vertex yet).
fn split_kinked_rim(
    topo: &mut Topology,
    scope: &SplitScope,
    solid: SolidId,
    edge: EdgeId,
) -> Result<Vec<Cut>, BlendError> {
    if !closure_kink(topo, edge).is_some_and(|kink| kink > KINK) {
        return Ok(Vec::new());
    }
    let EdgeCurve::NurbsCurve(nurbs) = topo.edge(edge)?.curve().clone() else {
        return Ok(Vec::new());
    };
    let walls: Vec<FaceId> = faces_using(topo, solid, edge)?
        .into_iter()
        .filter(|&f| {
            topo.face(f).is_ok_and(|f| {
                matches!(f.surface(), FaceSurface::Nurbs(_)) && f.inner_wires().is_empty()
            })
        })
        .collect();
    let [wall] = walls[..] else {
        return Ok(Vec::new());
    };
    let (u0, u1) = nurbs.domain();
    let middle = (u1 - u0).madd(0.5, u0);
    let point = nurbs.evaluate(middle);
    let along = nurbs.evaluate((u1 - u0).madd(0.25, u0));
    let Some(Ruling {
        far,
        rim_along_u,
        at,
    }) = ruling(topo.face(wall)?.surface(), point, along)
    else {
        return Ok(Vec::new());
    };
    if (far - point).length() <= ON {
        return Ok(Vec::new());
    }
    let FaceSurface::Nurbs(surface) = topo.face(wall)?.surface().clone() else {
        return Ok(Vec::new());
    };
    let Ok((before, after)) = split_surface(&surface, rim_along_u, at) else {
        return Ok(Vec::new());
    };
    let (left, right) = curve_split(&nurbs, middle)?;
    let vertex = topo.add_vertex(Vertex::new(point, ON));
    let (first, second) = trimmer::split_edge_at_with_curves(
        topo,
        scope,
        &OrientedEdge::new(edge, true),
        vertex,
        left,
        right,
    )?;
    let Some((far_vertex, far_cut)) = vertex_on_wire(topo, scope, wall, far)? else {
        return Err(BlendError::PlanningFailure {
            reason: format!("the ruling across the wall of kinked rim {edge:?} ends off its wire"),
        });
    };
    if !cut_face(
        topo,
        solid,
        wall,
        vertex,
        far_vertex,
        (&surface, rim_along_u, at),
        (before, after),
    )? {
        return Err(BlendError::PlanningFailure {
            reason: format!("the wall of kinked rim {edge:?} cannot be cut along its ruling"),
        });
    }
    Ok(std::iter::once((edge, [first.edge(), second.edge()]))
        .chain(far_cut)
        .collect())
}
