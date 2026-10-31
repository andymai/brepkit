//! Sweep sides that keep the profile's own curves.
//!
//! Each section along the path is the profile carried by a similarity: the
//! frame's rotation, the scale law's factor and a translation. An affine map
//! takes a rational curve to the rational curve on its control points'
//! images with the same weights, so a side surface whose u rows are a profile
//! edge's control points, each running along its own interpolated
//! trajectory, has the moved profile edge as its section wherever those
//! trajectories hold their points. Between, each surface point is a convex
//! combination of trajectory points under the edge's nonnegative rational
//! basis, so the surface stays within the trajectories' fit error of the
//! true sweep.

use std::collections::HashMap;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use super::{
    Frame, ProfilePlacement, SweepContactMode, SweepOptions, frames_at, orthogonalize,
    pick_reference_axis, profile_basis, resolve_placement,
};
use crate::dot_normal_point;

/// A trajectory's allowed fit error, as a fraction of the sweep's extent.
const FIT_REL_TOL: f64 = 1e-7;

/// The most path segments a trajectory is fitted over.
const MAX_SEGMENTS: usize = 1024;

/// The narrowest knot span of the path or guide, on its `[0, 1]`
/// parameter, the fit samples on its own.
const MIN_SPAN: f64 = 1e-9;

/// Samples along each edge for the profile's plane and winding.
const EDGE_SAMPLES: usize = 16;

/// One edge of a profile loop, in the order the loop runs it.
struct LoopEdge {
    edge: OrientedEdge,
    start: VertexId,
    /// The edge's own curve from its start to its end along the loop.
    span: NurbsCurve,
    is_line: bool,
}

struct ProfileLoop {
    edges: Vec<LoopEdge>,
    /// Newell area vector of the loop's samples, dotted with the profile
    /// normal: positive when the loop runs anticlockwise about it.
    winding: f64,
}

/// Sweep `profile` along the open, kink-free `path` with `options`, every
/// profile edge sweeping one side face on its own curve. `None` when the
/// profile's boundary does not lie in a plane, which the plane caps need.
///
/// # Errors
///
/// Returns an error if a curve or surface cannot be built, the scale law
/// leaves a section without size, or the profile is degenerate.
#[allow(clippy::too_many_lines)]
pub(super) fn sweep_exact(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
    options: &SweepOptions,
    placement: ProfilePlacement,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let tol = Tolerance::new();
    let face = topo.face(profile)?;
    let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();

    let mut loops = Vec::with_capacity(wire_ids.len());
    for wid in wire_ids {
        let oriented = topo.wire(wid)?.edges().to_vec();
        let pieces = crate::extrude::split_closed_wire_exact(topo, &oriented, tol.linear)?;
        let mut edges = Vec::with_capacity(pieces.len());
        for oe in pieces {
            let e = topo.edge(oe.edge())?;
            let (start, end) = if oe.is_forward() {
                (e.start(), e.end())
            } else {
                (e.end(), e.start())
            };
            let (from, to) = (topo.vertex(start)?.point(), topo.vertex(end)?.point());
            let curve = e.curve().clone();
            let is_line = matches!(curve, EdgeCurve::Line);
            let span = match crate::revolve::edge_span_nurbs(&curve, oe.is_forward(), (from, to))? {
                Some(span) => {
                    // An unclamped curve's end control points are not its
                    // ends, and moving them would change the curve.
                    let cps = span.control_points();
                    let (t0, t1) = span.domain();
                    let reach = 10.0 * tol.linear;
                    if (span.evaluate(t0) - cps[0]).length() > reach
                        || (span.evaluate(t1) - cps[cps.len() - 1]).length() > reach
                    {
                        return Ok(None);
                    }
                    with_ends(&span, from, to)?
                }
                None => NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![from, to], vec![1.0; 2])?,
            };
            edges.push(LoopEdge {
                edge: oe,
                start,
                span,
                is_line,
            });
        }
        if edges.is_empty() {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep profile has no edges".into(),
            });
        }
        loops.push(ProfileLoop {
            edges,
            winding: 0.0,
        });
    }

    let samples: Vec<Vec<Point3>> = loops.iter().map(loop_samples).collect();
    let outer_area = crate::winding::newell_normal(&samples[0]);
    let mut input_normal = outer_area.normalize()?;
    let (d0, d1) = path.domain();
    let path_tangent_0 = path.tangent(d0)?;
    if input_normal.dot(path_tangent_0) < 0.0 {
        input_normal = -input_normal;
    }
    let all_samples: Vec<Point3> = samples.iter().flatten().copied().collect();
    let anchor = samples[0][0];
    let extent = diagonal(&all_samples);
    // Every control point, which bound each edge, not samples between them:
    // a span can leave the plane between any samples.
    let off_plane = loops
        .iter()
        .flat_map(|lp| lp.edges.iter())
        .flat_map(|le| le.span.control_points().iter())
        .map(|&p| (p - anchor).dot(input_normal).abs())
        .fold(0.0, f64::max);
    if off_plane > 10.0f64.mul_add(tol.linear, 1e-9 * extent) {
        return Ok(None);
    }
    for (lp, pts) in loops.iter_mut().zip(&samples) {
        lp.winding = crate::winding::newell_normal(pts).dot(input_normal);
    }

    let up_hint = orthogonalize(input_normal, path_tangent_0);
    let at = |t: f64| (d1 - d0).mul_add(t, d0);
    let frames_for = |ts: &[f64]| -> Result<Vec<Frame>, crate::OperationsError> {
        if let Some(aux) = options.aux_spine.as_ref() {
            let (a0, a1) = aux.domain();
            let mut out = Vec::with_capacity(ts.len());
            let mut prev_up: Option<Vec3> = None;
            for &t in ts {
                let origin = path.evaluate(at(t));
                let tangent = path.tangent(at(t)).unwrap_or(path_tangent_0);
                let guide_dir = aux.evaluate((a1 - a0).mul_add(t, a0)) - origin;
                // Where the guide meets the spine its direction is undefined;
                // the previous frame's up carries on.
                let up = if guide_dir.length() < 1e-9 {
                    let seed = prev_up.unwrap_or_else(|| pick_reference_axis(tangent));
                    orthogonalize(seed, tangent)
                } else {
                    orthogonalize(guide_dir, tangent)
                };
                prev_up = Some(up);
                out.push(Frame {
                    origin,
                    tangent,
                    up,
                    right: tangent.cross(up),
                });
            }
            return Ok(out);
        }
        Ok(match options.contact_mode {
            SweepContactMode::RotationMinimizing => {
                let params: Vec<f64> = ts.iter().map(|&t| at(t)).collect();
                frames_at(path, &params, up_hint)?
            }
            SweepContactMode::Fixed => {
                // The profile keeps its orientation: the whole frame stays as
                // it starts, and the sections are translations of it.
                let up = orthogonalize(input_normal, path_tangent_0);
                let right = path_tangent_0.cross(up);
                ts.iter()
                    .map(|&t| Frame {
                        origin: path.evaluate(at(t)),
                        tangent: path_tangent_0,
                        up,
                        right,
                    })
                    .collect()
            }
            SweepContactMode::ConstantNormal(normal_dir) => ts
                .iter()
                .map(|&t| {
                    let tangent = path.tangent(at(t)).unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                    let up = orthogonalize(normal_dir, tangent);
                    Frame {
                        origin: path.evaluate(at(t)),
                        tangent,
                        up,
                        right: tangent.cross(up),
                    }
                })
                .collect(),
        })
    };

    let first = frames_for(&[0.0])?;
    let (reference, basis) = match resolve_placement(placement, input_normal, path_tangent_0) {
        ProfilePlacement::AsPositioned => (
            first[0].origin,
            [first[0].right, first[0].up, first[0].tangent],
        ),
        ProfilePlacement::CentroidOnPath => {
            let (r, u, t) = profile_basis(input_normal);
            (crate::winding::polygon_centroid(&samples[0]), [r, u, t])
        }
    };
    let maps_for = |ts: &[f64]| -> Result<Vec<Mat4>, crate::OperationsError> {
        let frames = frames_for(ts)?;
        ts.iter()
            .zip(&frames)
            .map(|(&t, frame)| {
                let scale = options.scale_law.as_ref().map_or(1.0, |law| law(t));
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(crate::OperationsError::InvalidInput {
                        reason: format!("sweep scale law gives {scale} at {t}, not a size"),
                    });
                }
                Ok(section_map(frame, reference, basis, scale))
            })
            .collect()
    };

    // Every control point of every edge, each fitted along the path.
    let points: Vec<Point3> = loops
        .iter()
        .flat_map(|lp| lp.edges.iter())
        .flat_map(|le| le.span.control_points().iter().copied())
        .collect();
    // The path's and guide's interior knots, on the shared [0, 1] parameter:
    // either curve may bend in its parameter there.
    let interior = |curve: &NurbsCurve| -> Vec<f64> {
        let (k0, k1) = curve.domain();
        curve
            .knots()
            .iter()
            .map(|&k| (k - k0) / (k1 - k0))
            .filter(|&t| t > 0.0 && t < 1.0)
            .collect()
    };
    let mut breaks = interior(path);
    if let Some(aux) = options.aux_spine.as_ref() {
        breaks.extend(interior(aux));
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|a, b| a.to_bits() == b.to_bits());
    // A knot span too narrow for the fit's samples to resolve can hold a
    // bend between them; the ring-by-ring sweep reads such a span inside.
    let narrow = std::iter::once(0.0)
        .chain(breaks.iter().copied())
        .chain(std::iter::once(1.0))
        .collect::<Vec<f64>>()
        .windows(2)
        .any(|w| w[1] - w[0] < MIN_SPAN);
    if narrow {
        return Ok(None);
    }
    let min_segments = if options.segments > 0 {
        options.segments.min(MAX_SEGMENTS)
    } else {
        (path.control_points().len() * 2).max(4)
    };
    let (rails, maps_0, maps_1) =
        fit_trajectories(&points, &breaks, min_segments, extent, tol, &maps_for)?;

    // Topology: one ring of vertices and edges at each end, one rail edge per
    // profile vertex, one side face per profile edge.
    let mut ring_vertex: [HashMap<VertexId, VertexId>; 2] = [HashMap::new(), HashMap::new()];
    let mut rail_edge: HashMap<VertexId, brepkit_topology::edge::EdgeId> = HashMap::new();
    let mut ring_edges: [Vec<Vec<OrientedEdge>>; 2] = [Vec::new(), Vec::new()];
    let straight = rails.first().is_some_and(|r| r.degree() == 1);
    let mut cursor = 0;
    let mut faces = Vec::new();
    for lp in &loops {
        for le in &lp.edges {
            let pos = topo.vertex(le.start)?.point();
            for (end, map) in [maps_0, maps_1].iter().enumerate() {
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    ring_vertex[end].entry(le.start)
                {
                    slot.insert(topo.add_vertex(Vertex::new(map.mul_point(pos), tol.linear)));
                }
            }
            if let std::collections::hash_map::Entry::Vacant(slot) = rail_edge.entry(le.start) {
                let curve = if straight {
                    EdgeCurve::Line
                } else {
                    EdgeCurve::NurbsCurve(rails[cursor].clone())
                };
                let edge = Edge::new(ring_vertex[0][&le.start], ring_vertex[1][&le.start], curve);
                slot.insert(topo.add_edge(edge));
            }
            cursor += le.span.control_points().len();
        }
    }
    for (end, map) in [maps_0, maps_1].iter().enumerate() {
        for lp in &loops {
            let mut ring = Vec::with_capacity(lp.edges.len());
            for le in &lp.edges {
                let e = topo.edge(le.edge.edge())?;
                let (s, t) = (e.start(), e.end());
                let curve = crate::transform::curve_image(e.curve(), map)?.0;
                let edge = Edge::new(ring_vertex[end][&s], ring_vertex[end][&t], curve);
                ring.push(OrientedEdge::new(topo.add_edge(edge), le.edge.is_forward()));
            }
            ring_edges[end].push(ring);
        }
    }

    let n0 = linear_image(maps_0, input_normal).normalize()?;
    let n1 = linear_image(maps_1, input_normal).normalize()?;
    let mut cursor = 0;
    for (li, lp) in loops.iter().enumerate() {
        // The material lies inside the outer loop and outside each hole.
        let sense = if (li == 0) == (lp.winding > 0.0) {
            1.0
        } else {
            -1.0
        };
        let count = lp.edges.len();
        for (i, le) in lp.edges.iter().enumerate() {
            let n_cp = le.span.control_points().len();
            let rows: Vec<Vec<Point3>> = rails[cursor..cursor + n_cp]
                .iter()
                .map(|r| r.control_points().to_vec())
                .collect();
            let cols = rows[0].len();
            let weights: Vec<Vec<f64>> = le.span.weights().iter().map(|&w| vec![w; cols]).collect();
            cursor += n_cp;
            let surface = NurbsSurface::new(
                le.span.degree(),
                rails[0].degree(),
                le.span.knots().to_vec(),
                rails[0].knots().to_vec(),
                rows,
                weights,
            )?;
            let (u_lo, u_hi) = surface.domain_u();
            let derivs = surface.derivatives(f64::midpoint(u_lo, u_hi), 0.0, 1);
            let natural = derivs[1][0].cross(derivs[0][1]);
            let outward = derivs[1][0].cross(n0) * sense;
            let reversed = natural.dot(outward) < 0.0;

            let next = &lp.edges[(i + 1) % count];
            let (ring_0, ring_1) = (ring_edges[0][li][i], ring_edges[1][li][i]);
            let (rail_a, rail_b) = (rail_edge[&le.start], rail_edge[&next.start]);
            let edges = if reversed {
                vec![
                    OrientedEdge::new(rail_a, true),
                    ring_1,
                    OrientedEdge::new(rail_b, false),
                    flip(ring_0),
                ]
            } else {
                vec![
                    ring_0,
                    OrientedEdge::new(rail_b, true),
                    flip(ring_1),
                    OrientedEdge::new(rail_a, false),
                ]
            };
            let wire = topo.add_wire(Wire::new(edges, true)?);

            let corners = [
                maps_0.mul_point(le.span.control_points()[0]),
                maps_0.mul_point(le.span.control_points()[n_cp - 1]),
                maps_1.mul_point(le.span.control_points()[0]),
                maps_1.mul_point(le.span.control_points()[n_cp - 1]),
            ];
            let flat = (le.is_line && straight)
                .then(|| plane_through(&corners, natural, tol.linear))
                .flatten();
            let face = if let Some(normal) = flat {
                let normal = if reversed { -normal } else { normal };
                Face::new(
                    wire,
                    vec![],
                    FaceSurface::Plane {
                        normal,
                        d: dot_normal_point(normal, corners[0]),
                    },
                )
            } else {
                let mut face = Face::new(wire, vec![], FaceSurface::Nurbs(surface));
                face.set_reversed(reversed);
                face
            };
            faces.push(topo.add_face(face));
        }
    }

    for (end, (normal, map)) in [(-n0, maps_0), (n1, maps_1)].into_iter().enumerate() {
        // About the cap's outward normal the outer loop runs anticlockwise and
        // every hole clockwise.
        let toward = if end == 0 { -1.0 } else { 1.0 };
        let mut wires = Vec::with_capacity(loops.len());
        for (li, lp) in loops.iter().enumerate() {
            let ring = &ring_edges[end][li];
            let anticlockwise = lp.winding * toward > 0.0;
            let edges = if anticlockwise == (li == 0) {
                ring.clone()
            } else {
                ring.iter().rev().map(|&oe| flip(oe)).collect()
            };
            wires.push(topo.add_wire(Wire::new(edges, true)?));
        }
        let on = map.mul_point(anchor);
        let surface = FaceSurface::Plane {
            normal,
            d: dot_normal_point(normal, on),
        };
        let outer = wires.remove(0);
        faces.push(topo.add_face(Face::new(outer, wires, surface)));
    }

    let shell = topo.add_shell(Shell::new(faces)?);
    Ok(Some(topo.add_solid(Solid::new(shell, vec![]))))
}

/// The edge's own curve with its first and last control points on its
/// vertices: a clamped curve's ends are those points, off them only by
/// rounding.
fn with_ends(
    span: &NurbsCurve,
    from: Point3,
    to: Point3,
) -> Result<NurbsCurve, crate::OperationsError> {
    let mut points = span.control_points().to_vec();
    let last = points.len() - 1;
    points[0] = from;
    points[last] = to;
    Ok(NurbsCurve::new(
        span.degree(),
        span.knots().to_vec(),
        points,
        span.weights().to_vec(),
    )?)
}

/// Points along a loop in its order, each edge read along its own curve.
fn loop_samples(lp: &ProfileLoop) -> Vec<Point3> {
    let mut out = Vec::new();
    for le in &lp.edges {
        let (t0, t1) = le.span.domain();
        let n = if le.is_line { 1 } else { EDGE_SAMPLES };
        for k in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let f = k as f64 / n as f64;
            out.push(le.span.evaluate((t1 - t0).mul_add(f, t0)));
        }
    }
    out
}

fn diagonal(points: &[Point3]) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for (i, c) in [p.x(), p.y(), p.z()].into_iter().enumerate() {
            lo[i] = lo[i].min(c);
            hi[i] = hi[i].max(c);
        }
    }
    Vec3::new(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).length()
}

/// The similarity taking the profile, read in `basis` about `reference`, to
/// its section in `frame`, scaled by `scale` about the frame's origin.
fn section_map(frame: &Frame, reference: Point3, basis: [Vec3; 3], scale: f64) -> Mat4 {
    let image = [frame.right, frame.up, frame.tangent];
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().take(3).enumerate() {
        for (j, cell) in row.iter_mut().take(3).enumerate() {
            let sum: f64 = (0..3)
                .map(|k| coord(image[k], i) * coord(basis[k], j))
                .sum();
            *cell = scale * sum;
        }
    }
    let r = [reference.x(), reference.y(), reference.z()];
    let o = [frame.origin.x(), frame.origin.y(), frame.origin.z()];
    for i in 0..3 {
        m[i][3] = o[i] - (0..3).map(|j| m[i][j] * r[j]).sum::<f64>();
    }
    m[3][3] = 1.0;
    Mat4(m)
}

fn coord(v: Vec3, i: usize) -> f64 {
    match i {
        0 => v.x(),
        1 => v.y(),
        _ => v.z(),
    }
}

fn linear_image(map: Mat4, v: Vec3) -> Vec3 {
    map.mul_point(Point3::new(v.x(), v.y(), v.z())) - map.mul_point(Point3::new(0.0, 0.0, 0.0))
}

const fn flip(oe: OrientedEdge) -> OrientedEdge {
    OrientedEdge::new(oe.edge(), !oe.is_forward())
}

/// The unit normal of the plane through four corners, on the side of
/// `toward`, when they lie in one.
fn plane_through(corners: &[Point3; 4], toward: Vec3, tol: f64) -> Option<Vec3> {
    let normal = (corners[1] - corners[0])
        .cross(corners[2] - corners[0])
        .normalize()
        .ok()?;
    let normal = if normal.dot(toward) < 0.0 {
        -normal
    } else {
        normal
    };
    corners
        .iter()
        .all(|&p| (p - corners[0]).dot(normal).abs() <= 10.0 * tol)
        .then_some(normal)
}

type Fitted = (Vec<NurbsCurve>, Mat4, Mat4);

/// The section maps at a list of path parameters.
type SectionMaps<'a> = dyn Fn(&[f64]) -> Result<Vec<Mat4>, crate::OperationsError> + 'a;

/// Each point's trajectory along the path, fitted through its images at a
/// set of path parameters and checked halfway between them. The path is cut
/// at `breaks`, where the path or guide may bend in its parameter, and each
/// piece is fitted on its own and joined to the next at their shared point.
/// The pieces' samples grow until every trajectory holds within the fit
/// tolerance or the path reaches [`MAX_SEGMENTS`]. Trajectories that all run
/// straight at an even pace come back as lines. Also the section maps at the
/// path's two ends.
fn fit_trajectories(
    points: &[Point3],
    breaks: &[f64],
    min_segments: usize,
    profile_extent: f64,
    tol: Tolerance,
    maps_for: &SectionMaps<'_>,
) -> Result<Fitted, crate::OperationsError> {
    let mut cuts: Vec<f64> = std::iter::once(0.0)
        .chain(breaks.iter().copied())
        .chain(std::iter::once(1.0))
        .collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let spans: Vec<(f64, f64)> = cuts.windows(2).map(|w| (w[0], w[1])).collect();
    // A cubic per piece needs four samples.
    #[allow(clippy::cast_precision_loss)]
    let per_span = |total: usize, (a, b): (f64, f64)| -> usize {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = ((b - a) * total as f64).ceil() as usize;
        k.max(3)
    };
    let mut total = min_segments.max(2);
    let mut fit_tol: Option<f64> = None;
    let mut last_count = 0;
    loop {
        // Pieces at their floor of three segments add none while `total` is
        // small against their number.
        let count: usize = spans.iter().map(|&span| per_span(total, span)).sum();
        if count == last_count {
            total *= 2;
            continue;
        }
        last_count = count;
        // Each piece's parameters, its first shared with the previous piece's
        // last, and every midpoint between neighbours. `slots[i][j]` is where
        // piece i's parameter j sits in `all`; its midpoint to j + 1 follows.
        let mut all: Vec<f64> = vec![0.0];
        let mut pieces: Vec<Vec<f64>> = Vec::with_capacity(spans.len());
        let mut slots: Vec<Vec<usize>> = Vec::with_capacity(spans.len());
        for &(a, b) in &spans {
            let k = per_span(total, (a, b));
            #[allow(clippy::cast_precision_loss)]
            let mut piece: Vec<f64> = (0..=k)
                .map(|j| (b - a).mul_add(j as f64 / k as f64, a))
                .collect();
            piece[0] = a;
            piece[k] = b;
            let mut slot = vec![all.len() - 1];
            for w in piece.windows(2) {
                all.push(f64::midpoint(w[0], w[1]));
                all.push(w[1]);
                slot.push(all.len() - 1);
            }
            pieces.push(piece);
            slots.push(slot);
        }
        let maps = maps_for(&all)?;

        let fit = *fit_tol.get_or_insert_with(|| {
            let moved: Vec<Point3> = maps
                .iter()
                .flat_map(|m| points.iter().map(|&p| m.mul_point(p)))
                .collect();
            (FIT_REL_TOL * diagonal(&moved).max(profile_extent)).max(tol.linear)
        });

        let (first, last) = (maps[0], maps[maps.len() - 1]);
        let straight = points.iter().all(|&p| {
            let (a, b) = (first.mul_point(p), last.mul_point(p));
            all.iter()
                .zip(&maps)
                .all(|(&t, m)| (a + (b - a) * t - m.mul_point(p)).length() <= fit)
        });
        if straight {
            let lines = points
                .iter()
                .map(|&p| {
                    NurbsCurve::new(
                        1,
                        vec![0.0, 0.0, 1.0, 1.0],
                        vec![first.mul_point(p), last.mul_point(p)],
                        vec![1.0; 2],
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok((lines, first, last));
        }

        let mut rails = Vec::with_capacity(points.len());
        let mut worst = 0.0_f64;
        for &p in points {
            let mut fitted = Vec::with_capacity(pieces.len());
            for (piece, slot) in pieces.iter().zip(&slots) {
                let track: Vec<Point3> = slot.iter().map(|&i| maps[i].mul_point(p)).collect();
                let rail = crate::loft::interpolate_at(&track, 3, piece)?;
                for &i in &slot[..slot.len() - 1] {
                    let q = maps[i + 1].mul_point(p);
                    worst = worst.max((rail.evaluate(all[i + 1]) - q).length());
                }
                fitted.push(rail);
            }
            rails.push(join(&fitted)?);
        }
        let segments: usize = pieces.iter().map(|piece| piece.len() - 1).sum();
        if worst <= fit || segments >= MAX_SEGMENTS {
            log::debug!(
                "sweep_exact: {} trajectories over {segments} segments in {} pieces, worst {worst:.3e} (fit {fit:.3e})",
                rails.len(),
                pieces.len()
            );
            return Ok((rails, first, last));
        }
        // A cubic's error falls with the fourth power of its step: grow the
        // samples by what that asks, with a tenth to spare, rather than
        // doubling past a fit that was only just missed.
        let grow = (1.1 * (worst / fit).powf(0.25)).clamp(1.2, 2.0);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        {
            total = ((total as f64 * grow).ceil() as usize).max(total + 1);
        }
    }
}

/// Clamped cubics on consecutive parameter ranges, each starting where the
/// one before ends, as one curve meeting itself at each join.
fn join(pieces: &[NurbsCurve]) -> Result<NurbsCurve, crate::OperationsError> {
    let p = pieces[0].degree();
    let mut knots: Vec<f64> = Vec::new();
    let mut points: Vec<Point3> = Vec::new();
    for (i, piece) in pieces.iter().enumerate() {
        let k = piece.knots();
        let interior = &k[p + 1..k.len() - p - 1];
        if i == 0 {
            knots.extend(std::iter::repeat_n(k[0], p + 1));
            points.extend_from_slice(piece.control_points());
        } else {
            knots.extend(std::iter::repeat_n(k[0], p));
            points.extend_from_slice(&piece.control_points()[1..]);
        }
        knots.extend_from_slice(interior);
    }
    let end = pieces[pieces.len() - 1].knots();
    knots.extend(std::iter::repeat_n(end[end.len() - 1], p + 1));
    let weights = vec![1.0; points.len()];
    Ok(NurbsCurve::new(p, knots, points, weights)?)
}
