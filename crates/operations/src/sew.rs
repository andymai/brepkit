//! Topology sewing: merge loose faces into connected shells.
//!
//! Takes a set of independent faces and merges coincident edges/vertices
//! to create a topologically connected shell.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

/// Sew a set of loose faces into a solid.
///
/// Finds geometrically coincident edges between faces (within
/// `tolerance`) and merges them, creating shared topology. Then
/// assembles the sewn faces into a shell and solid.
///
/// # Algorithm
///
/// 1. Merge coincident vertices using spatial hashing
/// 2. Merge coincident edges: the same two vertices, and each edge's points
///    on the other, so the two halves of a circle, or two closed edges
///    through one vertex, stay apart
/// 3. Rebuild each face, holes included, from the merged edges, each keeping
///    its own curve
/// 4. Assemble into a shell and solid
///
/// # Errors
///
/// Returns an error if fewer than 2 faces are provided or a topology lookup
/// fails.
pub fn sew_faces(
    topo: &mut Topology,
    faces: &[FaceId],
    tolerance: f64,
) -> Result<SolidId, crate::OperationsError> {
    if faces.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sewing requires at least 2 faces".into(),
        });
    }

    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };

    let mut face_snapshots: Vec<FaceSnapshot> = Vec::with_capacity(faces.len());
    for &fid in faces {
        let face = topo.face(fid)?;
        let mut wires = Vec::with_capacity(1 + face.inner_wires().len());
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let mut edges = Vec::new();
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                let start = topo.vertex(edge.start())?.point();
                let end = topo.vertex(edge.end())?.point();
                edges.push(EdgeSnapshot {
                    span: Span::new(edge.curve().clone(), start, end),
                    forward: oe.is_forward(),
                    source: oe.edge().index(),
                    closed: edge.start() == edge.end(),
                });
            }
            wires.push(edges);
        }
        face_snapshots.push(FaceSnapshot {
            surface: face.surface().clone(),
            reversed: face.is_reversed(),
            wires,
        });
    }

    let resolution = 1.0 / tol;
    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> = HashMap::new();
    let mut vertex_at = |topo: &mut Topology, p: Point3| {
        #[allow(clippy::cast_possible_truncation)]
        let key = (
            (p.x() * resolution).round() as i64,
            (p.y() * resolution).round() as i64,
            (p.z() * resolution).round() as i64,
        );
        *vertex_map
            .entry(key)
            .or_insert_with(|| topo.add_vertex(Vertex::new(p, tol)))
    };

    // The edges made so far between each pair of vertices, and the edge
    // each input edge became, with whether it runs the same way.
    let mut edge_map: HashMap<(usize, usize), Vec<MadeEdge>> = HashMap::new();
    let mut by_source: HashMap<usize, (EdgeId, bool)> = HashMap::new();
    let mut new_face_ids: Vec<FaceId> = Vec::with_capacity(face_snapshots.len());

    for snap in &face_snapshots {
        let mut wire_ids = Vec::with_capacity(snap.wires.len());
        for edges in &snap.wires {
            let mut oriented_edges: Vec<OrientedEdge> = Vec::with_capacity(edges.len());
            for e in edges {
                let v_start = vertex_at(topo, e.span.start);
                let v_end = vertex_at(topo, e.span.end);
                // An edge whose two ends weld into one vertex and whose curve
                // stays within the tolerance of them closes nothing. One that
                // reaches farther, an arc just short of a whole turn, becomes
                // a closed edge.
                if v_start == v_end
                    && !e.closed
                    && e.span
                        .samples()
                        .all(|f| (e.span.at(f) - e.span.start).length() <= tol)
                {
                    continue;
                }
                let key = if v_start.index() <= v_end.index() {
                    (v_start.index(), v_end.index())
                } else {
                    (v_end.index(), v_start.index())
                };
                let made = edge_map.entry(key).or_default();
                let (edge_id, along) = if let Some(&shared) = by_source.get(&e.source) {
                    shared
                } else if let Some(m) = made
                    .iter()
                    .find(|m| e.span.lies_on(&m.span, tol) && m.span.lies_on(&e.span, tol))
                {
                    let along = if v_start == v_end {
                        e.span.runs_with(&m.span)
                    } else {
                        m.start == v_start
                    };
                    (m.edge, along)
                } else {
                    // A closed edge runs its curve's whole domain, so a piece
                    // of a longer curve whose ends welded is cut to its span.
                    let curve = match &e.span.curve {
                        EdgeCurve::NurbsCurve(c) if v_start == v_end && !e.closed => {
                            nurbs_between(c, e.span.t0, e.span.t1)
                                .map_or_else(|| e.span.curve.clone(), EdgeCurve::NurbsCurve)
                        }
                        c => c.clone(),
                    };
                    let edge = topo.add_edge(Edge::new(v_start, v_end, curve));
                    made.push(MadeEdge {
                        span: e.span.clone(),
                        edge,
                        start: v_start,
                    });
                    (edge, true)
                };
                by_source.insert(e.source, (edge_id, along));
                // An edge met again the other way along runs against its
                // first face's traversal in this one.
                let forward = e.forward == along;
                oriented_edges.push(OrientedEdge::new(edge_id, forward));
            }
            let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
            wire_ids.push(topo.add_wire(wire));
        }
        let (outer, holes) = wire_ids
            .split_first()
            .map(|(o, h)| (*o, h.to_vec()))
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: "a face to sew has no boundary".into(),
            })?;
        let mut face = Face::new(outer, holes, snap.surface.clone());
        if snap.reversed {
            face.set_reversed(true);
        }
        new_face_ids.push(topo.add_face(face));
    }

    // Faces sewn from separate operations need not face one way: a loft's
    // wall sewn in as a hole's faces away from the hole. They take the first
    // face's sense across their shared edges, and all turn outward if they
    // close the other way round.
    if let Some(&seed) = new_face_ids.first() {
        brepkit_topology::orientation::propagate_orientation(topo, &new_face_ids, &[seed])?;
    }
    let closes = closes(topo, &new_face_ids)?;
    let shell = Shell::new(new_face_ids.clone()).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = topo.add_solid(Solid::new(shell_id, vec![]));
    if closes {
        let b = crate::measure::solid_bounding_box(topo, solid)?;
        let deflection = 0.01 * (b.max - b.min).length().max(tol);
        if crate::measure::oriented_solid_volume(topo, solid, deflection)? < 0.0 {
            for &fid in &new_face_ids {
                let reversed = topo.face(fid)?.is_reversed();
                topo.face_mut(fid)?.set_reversed(!reversed);
            }
        }
    }
    Ok(solid)
}

/// Whether every edge of `faces` bounds exactly two of them.
fn closes(topo: &Topology, faces: &[FaceId]) -> Result<bool, crate::OperationsError> {
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    Ok(!uses.is_empty() && uses.values().all(|&n| n == 2))
}

/// An edge sewing has made: its curve, and the vertex it starts at.
struct MadeEdge {
    span: Span,
    edge: EdgeId,
    start: VertexId,
}

/// One edge of a face to sew, as its wire runs it.
struct EdgeSnapshot {
    span: Span,
    forward: bool,
    /// The input edge's index.
    source: usize,
    /// The input edge starts and ends at one vertex.
    closed: bool,
}

/// An edge's curve from its start to its end, read by the fraction of the
/// way along its parameter range, so a range of any scale reads alike.
#[derive(Clone)]
struct Span {
    curve: EdgeCurve,
    start: Point3,
    end: Point3,
    t0: f64,
    t1: f64,
    /// The span cut into pieces each held in a ball, together covering it.
    pieces: Vec<Piece>,
}

/// A stretch of a [`Span`] from fraction `from` to `to`, held in the ball
/// about `centre` of radius `radius`, with `inside` points read inside it.
#[derive(Clone, Copy)]
struct Piece {
    from: f64,
    to: f64,
    centre: Point3,
    radius: f64,
    inside: u32,
}

impl Span {
    fn new(curve: EdgeCurve, start: Point3, end: Point3) -> Self {
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        let mut span = Self {
            curve,
            start,
            end,
            t0,
            t1,
            pieces: Vec::new(),
        };
        span.pieces = match &span.curve {
            EdgeCurve::Line => vec![span.chord_piece(0.0, 1.0, 0.0, 3)],
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                let reach = match &span.curve {
                    EdgeCurve::Circle(c) => c.radius(),
                    EdgeCurve::Ellipse(e) => e.semi_major(),
                    _ => 0.0,
                };
                let eighth = std::f64::consts::FRAC_PI_8;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let n = ((t1 - t0).abs() / eighth).ceil().max(1.0) as u32;
                let bow = reach * (1.0 - (0.5 * eighth).cos());
                (0..n)
                    .map(|k| {
                        let (a, b) = (f64::from(k) / f64::from(n), f64::from(k + 1) / f64::from(n));
                        span.chord_piece(a, b, 2.0 * bow, 3)
                    })
                    .collect()
            }
            EdgeCurve::NurbsCurve(nurbs) => nurbs_pieces(nurbs, t0, t1)
                .unwrap_or_else(|| vec![span.chord_piece(0.0, 1.0, f64::INFINITY, 16)]),
        };
        span
    }

    /// The piece from `a` to `b` held in the ball on its chord, widened by
    /// `bow`, the most the curve strays from the chord there.
    fn chord_piece(&self, a: f64, b: f64, bow: f64, inside: u32) -> Piece {
        let (pa, pb) = (self.at(a), self.at(b));
        Piece {
            from: a,
            to: b,
            centre: pa + (pb - pa) * 0.5,
            radius: 0.5 * (pb - pa).length() + bow,
            inside,
        }
    }

    /// The point a fraction `f` of the way along the parameter range.
    fn at(&self, f: f64) -> Point3 {
        self.curve.evaluate_with_endpoints(
            f.mul_add(self.t1 - self.t0, self.t0),
            self.start,
            self.end,
        )
    }

    /// The fractions that decide whether this span lies on another curve:
    /// every piece's ends and points inside it.
    fn samples(&self) -> impl Iterator<Item = f64> + '_ {
        self.pieces.iter().flat_map(|piece| {
            let n = f64::from(piece.inside + 1);
            (0..=piece.inside + 1)
                .map(move |k| (piece.to - piece.from).mul_add(f64::from(k) / n, piece.from))
        })
    }

    /// The fraction along this span nearest `p`, and its distance. Every
    /// piece whose ball could hold a nearer point than the nearest piece end
    /// is closed in on by golden section, nearest ball first.
    fn nearest(&self, p: Point3) -> (f64, f64) {
        let dist = |f: f64| (self.at(f) - p).length();
        let mut best = (0.0, f64::INFINITY);
        for piece in &self.pieces {
            for f in [piece.from, piece.to] {
                let d = dist(f);
                if d < best.1 {
                    best = (f, d);
                }
            }
        }
        let mut order: Vec<(f64, &Piece)> = self
            .pieces
            .iter()
            .map(|piece| ((piece.centre - p).length() - piece.radius, piece))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
        for (bound, piece) in order {
            if bound >= best.1 {
                break;
            }
            // A piece is near enough flat that its distance has one least
            // between a few samples: start from the least of them.
            let n = piece.inside + 1;
            let fs: Vec<f64> = (0..=n)
                .map(|k| (piece.to - piece.from).mul_add(f64::from(k) / f64::from(n), piece.from))
                .collect();
            let ds: Vec<f64> = fs.iter().map(|&f| dist(f)).collect();
            let i = (0..ds.len())
                .min_by(|&a, &b| ds[a].total_cmp(&ds[b]))
                .unwrap_or(0);
            let (mut a, mut b) = (fs[i.saturating_sub(1)], fs[(i + 1).min(fs.len() - 1)]);
            let (mut c, mut e) = (b - golden * (b - a), golden.mul_add(b - a, a));
            let (mut dc, mut de) = (dist(c), dist(e));
            for _ in 0..80 {
                if dc <= de {
                    (b, e, de) = (e, c, dc);
                    c = b - golden * (b - a);
                    dc = dist(c);
                } else {
                    (a, c, dc) = (c, e, de);
                    e = golden.mul_add(b - a, a);
                    de = dist(e);
                }
            }
            for found in [(fs[i], ds[i]), (c, dc), (e, de)] {
                if found.1 < best.1 {
                    best = found;
                }
            }
        }
        best
    }

    /// Whether points through every piece of this span lie on `other`,
    /// however either is parameterized.
    fn lies_on(&self, other: &Self, tol: f64) -> bool {
        self.samples().all(|f| other.nearest(self.at(f)).1 <= tol)
    }

    /// The way the span runs at fraction `f`.
    fn heading(&self, f: f64) -> Vec3 {
        let h = 1e-6;
        self.at((f + h).min(1.0)) - self.at((f - h).max(0.0))
    }

    /// Whether this span runs the same way along as `other`, which it lies
    /// on. A closed edge's ends cannot tell.
    fn runs_with(&self, other: &Self) -> bool {
        let there = other.nearest(self.at(0.5)).0;
        self.heading(0.5).dot(other.heading(there)) > 0.0
    }
}

/// `curve` from parameter `t0` to `t1`, run backward when `t1 < t0`, or
/// `None` if it cannot be cut there. The curve is cut over its domain moved
/// to `[0, 1]`, so a domain of any scale is cut alike.
fn nurbs_between(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    t0: f64,
    t1: f64,
) -> Option<brepkit_math::nurbs::curve::NurbsCurve> {
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::nurbs::knot_ops::curve_split;
    // The knot routines' resolution: within it of an end there is no room to
    // cut, and nothing past it to cut off.
    const AT_END: f64 = 1e-15;
    let (d0, d1) = curve.domain();
    let length = d1 - d0;
    if length.is_nan() || length <= 0.0 {
        return None;
    }
    let unit = |t: f64| (t - d0) / length;
    let mut span = NurbsCurve::new(
        curve.degree(),
        curve.knots().iter().map(|&k| unit(k)).collect(),
        curve.control_points().to_vec(),
        curve.weights().to_vec(),
    )
    .ok()?;
    // Cut wherever the span leaves the domain, however near its end: a knot
    // span as narrow as a parameter's noise can hold a long piece.
    let (lo, hi) = (unit(t0.min(t1)), unit(t0.max(t1)));
    if hi < 1.0 - AT_END {
        span = curve_split(&span, hi).ok()?.0;
    }
    if lo > AT_END {
        span = curve_split(&span, lo).ok()?.1;
    }
    if t1 >= t0 {
        return Some(span);
    }
    let (a, b) = span.domain();
    NurbsCurve::new(
        span.degree(),
        span.knots().iter().rev().map(|&k| a + b - k).collect(),
        span.control_points().iter().rev().copied().collect(),
        span.weights().iter().rev().copied().collect(),
    )
    .ok()
}

/// A NURBS edge from `t0` to `t1` cut into its Bezier spans, each halved
/// until its control points lie within half a percent of the chord between
/// its ends, so a bump or a fold between samples is cut out, and its
/// weights lie within a factor of two of each other, so its points spread
/// along it with its parameter. Each piece is held in the ball about its
/// control points, which hold the curve where its weights are positive.
/// `None` where a weight is not.
fn nurbs_pieces(
    nurbs: &brepkit_math::nurbs::curve::NurbsCurve,
    t0: f64,
    t1: f64,
) -> Option<Vec<Piece>> {
    const MAX_DEPTH: u32 = 48;
    if nurbs.weights().iter().any(|&w| !w.is_finite() || w <= 0.0) || (t1 - t0).abs() <= 0.0 {
        return None;
    }
    let (lo, hi) = (t0.min(t1), t0.max(t1));
    let fraction = |t: f64| (t - t0) / (t1 - t0);
    let knots = nurbs.knots();
    let mut pieces = Vec::new();
    for i in nurbs.degree()..nurbs.control_points().len() {
        let (a, b) = (knots[i].max(lo), knots[i + 1].min(hi));
        if b <= a {
            continue;
        }
        let pts = brepkit_math::nurbs::decompose::curve_bezier_between(nurbs, a, b).ok()?;
        let mut stack = vec![(a, b, pts, 0_u32)];
        while let Some((a, b, pts, depth)) = stack.pop() {
            let cart: Vec<Point3> = pts
                .iter()
                .map(|h| Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
                .collect();
            let (first, last) = (cart[0], cart[cart.len() - 1]);
            let chord = last - first;
            let length = chord.length();
            let bow = cart.iter().fold(0.0_f64, |m, &q| {
                let d = q - first;
                let along = if length > 0.0 {
                    d.dot(chord) / (length * length)
                } else {
                    0.0
                };
                m.max((d - chord * along.clamp(0.0, 1.0)).length())
            });
            let (least, most) = pts.iter().fold((f64::INFINITY, 0.0_f64), |(l, m), h| {
                (l.min(h[3]), m.max(h[3]))
            });
            if (bow > 0.005 * length || most > 2.0 * least) && depth < MAX_DEPTH {
                let (left, right) = split_bezier(&pts, 0.5);
                let mid = f64::midpoint(a, b);
                stack.push((mid, b, right, depth + 1));
                stack.push((a, mid, left, depth + 1));
                continue;
            }
            let lo_corner = cart.iter().fold(cart[0], |m, q| {
                Point3::new(m.x().min(q.x()), m.y().min(q.y()), m.z().min(q.z()))
            });
            let hi_corner = cart.iter().fold(cart[0], |m, q| {
                Point3::new(m.x().max(q.x()), m.y().max(q.y()), m.z().max(q.z()))
            });
            let centre = lo_corner + (hi_corner - lo_corner) * 0.5;
            let radius = cart
                .iter()
                .fold(0.0_f64, |m, &q| m.max((q - centre).length()));
            #[allow(clippy::cast_possible_truncation)]
            let inside = (pts.len() as u32).max(3);
            let (fa, fb) = (fraction(a), fraction(b));
            pieces.push(Piece {
                from: fa.min(fb),
                to: fa.max(fb),
                centre,
                radius,
                inside,
            });
        }
    }
    pieces.sort_by(|x, y| x.from.total_cmp(&y.from));
    (!pieces.is_empty()).then_some(pieces)
}

/// A Bezier curve's homogeneous control points split at `s` in `[0, 1]` of
/// its parameter, by de Casteljau.
fn split_bezier(pts: &[[f64; 4]], s: f64) -> (Vec<[f64; 4]>, Vec<[f64; 4]>) {
    let n = pts.len();
    let mut work = pts.to_vec();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    left.push(work[0]);
    right.push(work[n - 1]);
    for r in 1..n {
        for k in 0..n - r {
            for c in 0..4 {
                work[k][c] = (work[k + 1][c] - work[k][c]).mul_add(s, work[k][c]);
            }
        }
        left.push(work[0]);
        right.push(work[n - 1 - r]);
    }
    right.reverse();
    (left, right)
}

/// Snapshot of a face's geometry for the sewing algorithm.
struct FaceSnapshot {
    surface: FaceSurface,
    reversed: bool,
    /// The outer wire first, then the holes.
    wires: Vec<Vec<EdgeSnapshot>>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// A box's six faces, one sewn turned inside out and then all of them,
    /// come back facing out of the box: its volume read with its sign is
    /// the box's 24.
    #[test]
    fn sewn_faces_face_out_of_what_they_close() {
        for flipped in [vec![0], (0..6).collect()] {
            let mut topo = Topology::new();
            let b = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
            let faces = brepkit_topology::explorer::solid_faces(&topo, b).unwrap();
            for &i in &flipped {
                let r = topo.face(faces[i]).unwrap().is_reversed();
                topo.face_mut(faces[i]).unwrap().set_reversed(!r);
            }
            let solid = sew_faces(&mut topo, &faces, 1e-6).unwrap();
            let v = crate::measure::oriented_solid_volume(&topo, solid, 0.01).unwrap();
            assert!((v - 24.0).abs() < 1e-9, "{flipped:?} flipped: volume {v}");
        }
    }

    /// Helper: create a standalone quad face with independent vertices/edges.
    fn make_loose_quad(
        topo: &mut Topology,
        p0: Point3,
        p1: Point3,
        p2: Point3,
        p3: Point3,
        normal: Vec3,
        d: f64,
    ) -> FaceId {
        let tol_val = 1e-7;
        let v0 = topo.add_vertex(Vertex::new(p0, tol_val));
        let v1 = topo.add_vertex(Vertex::new(p1, tol_val));
        let v2 = topo.add_vertex(Vertex::new(p2, tol_val));
        let v3 = topo.add_vertex(Vertex::new(p3, tol_val));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }))
    }

    #[test]
    fn sew_two_adjacent_quads() {
        let mut topo = Topology::new();

        // Two quads sharing an edge along x=1.
        let f0 = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let f1 = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );

        let solid = sew_faces(&mut topo, &[f0, f1], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 2, "sewn result should have 2 faces");
    }

    #[test]
    fn sew_shares_coincident_edges() {
        let mut topo = Topology::new();

        let f0 = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let f1 = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );

        let solid = sew_faces(&mut topo, &[f0, f1], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        let mut edge_set = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_set.insert(oe.edge().index());
            }
        }

        // 2 quads with 4 edges each, sharing 1 edge = 7 unique edges.
        assert_eq!(
            edge_set.len(),
            7,
            "two adjacent quads should share 1 edge (7 unique), got {}",
            edge_set.len()
        );
    }

    #[test]
    fn sew_six_cube_faces() {
        let mut topo = Topology::new();

        let bottom = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            0.0,
        );
        let top = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        );
        let front = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.0,
        );
        let back = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
        );
        let left = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.0,
        );
        let right = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
        );

        let solid = sew_faces(&mut topo, &[bottom, top, front, back, left, right], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 6, "sewn cube should have 6 faces");

        // Count unique edges — a cube has 12.
        let mut edge_set = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_set.insert(oe.edge().index());
            }
        }
        assert_eq!(
            edge_set.len(),
            12,
            "cube should have 12 edges, got {}",
            edge_set.len()
        );
    }

    #[test]
    fn sew_single_face_error() {
        let mut topo = Topology::new();
        let f = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        assert!(sew_faces(&mut topo, &[f], 1e-6).is_err());
    }

    /// The kinds of curve a solid's edges carry, in face order.
    fn curve_kinds(topo: &Topology, solid: SolidId) -> Vec<Vec<&'static str>> {
        brepkit_topology::explorer::solid_faces(topo, solid)
            .unwrap()
            .iter()
            .map(|&f| {
                let face = topo.face(f).unwrap();
                std::iter::once(face.outer_wire())
                    .chain(face.inner_wires().iter().copied())
                    .flat_map(|w| topo.wire(w).unwrap().edges().to_vec())
                    .map(|oe| topo.edge(oe.edge()).unwrap().curve().type_tag())
                    .collect()
            })
            .collect()
    }

    /// A cylinder's three faces sew back into the cylinder: its rims stay
    /// circles and it holds π r² h.
    #[test]
    fn sew_a_cylinders_faces_keeps_its_circles() {
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 10.0, 20.0).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, cyl).unwrap();
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        assert_eq!(curve_kinds(&topo, sewn), curve_kinds(&topo, cyl));
        assert!(
            crate::validate::validate_solid(&topo, sewn)
                .unwrap()
                .is_valid()
        );
        let volume = crate::measure::solid_volume(&topo, sewn, 0.01).unwrap();
        let exact = std::f64::consts::PI * 100.0 * 20.0;
        assert!((volume - exact).abs() < 1e-3 * exact, "{volume}");
    }

    /// A quarter-circle sector sewn beside a far triangle keeps its arc and
    /// its area of 25π/4.
    #[test]
    fn sew_keeps_a_sectors_arc() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let v = |topo: &mut Topology, x: f64, y: f64| {
            topo.add_vertex(Vertex::new(Point3::new(x, y, 20.0), 1e-7))
        };
        let (c, a, b) = (
            v(&mut topo, 20.0, 15.0),
            v(&mut topo, 20.0, 20.0),
            v(&mut topo, 25.0, 15.0),
        );
        let circle = Circle3D::new(
            Point3::new(20.0, 15.0, 20.0),
            Vec3::new(0.0, 0.0, -1.0),
            5.0,
        )
        .unwrap();
        let edges = [
            topo.add_edge(Edge::new(c, a, EdgeCurve::Line)),
            topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
            topo.add_edge(Edge::new(b, c, EdgeCurve::Line)),
        ];
        let wire = topo.add_wire(
            Wire::new(
                edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let sector = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: -20.0,
            },
        ));
        let far = make_loose_quad(
            &mut topo,
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(110.0, 0.0, 0.0),
            Point3::new(110.0, 10.0, 0.0),
            Point3::new(100.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let sewn = sew_faces(&mut topo, &[sector, far], 1e-7).unwrap();
        let first = brepkit_topology::explorer::solid_faces(&topo, sewn).unwrap()[0];
        let area = crate::measure::face_area(&topo, first, 0.001).unwrap();
        let exact = 25.0 * std::f64::consts::FRAC_PI_4;
        assert!((area - exact).abs() < 1e-3 * exact, "{area}");
        assert!(curve_kinds(&topo, sewn)[0].contains(&"circle"));
    }

    /// A disc bounded by two half circles between the same two points keeps
    /// both: its area stays 25π.
    #[test]
    fn sew_keeps_two_arcs_between_one_pair_of_points() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(5.0, 0.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(-5.0, 0.0, 0.0), 1e-7));
        let up = Vec3::new(0.0, 0.0, 1.0);
        let circle =
            || EdgeCurve::Circle(Circle3D::new(Point3::new(0.0, 0.0, 0.0), up, 5.0).unwrap());
        let edges = [
            topo.add_edge(Edge::new(a, b, circle())),
            topo.add_edge(Edge::new(b, a, circle())),
        ];
        let wire = topo.add_wire(
            Wire::new(
                edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let disc = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: up, d: 0.0 },
        ));
        let far = make_loose_quad(
            &mut topo,
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(110.0, 0.0, 0.0),
            Point3::new(110.0, 10.0, 0.0),
            Point3::new(100.0, 10.0, 0.0),
            up,
            0.0,
        );
        let sewn = sew_faces(&mut topo, &[disc, far], 1e-7).unwrap();
        let first = brepkit_topology::explorer::solid_faces(&topo, sewn).unwrap()[0];
        let area = crate::measure::face_area(&topo, first, 0.001).unwrap();
        let exact = 25.0 * std::f64::consts::PI;
        assert!((area - exact).abs() < 1e-3 * exact, "{area}");
    }

    /// A span of a degree-1 curve through `points` in the plane z = 0 over
    /// `knots`, from its first point to its last.
    fn polyline_span(knots: Vec<f64>, points: &[(f64, f64)]) -> Span {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let pts: Vec<Point3> = points
            .iter()
            .map(|&(x, y)| Point3::new(x, y, 0.0))
            .collect();
        let (start, end) = (pts[0], pts[pts.len() - 1]);
        let curve = NurbsCurve::new(1, knots, pts.clone(), vec![1.0; pts.len()]).unwrap();
        Span::new(EdgeCurve::NurbsCurve(curve), start, end)
    }

    /// A unit segment whose knots run to 1e9 lies on its copy.
    #[test]
    fn a_span_lies_on_its_copy_however_far_its_knots_run() {
        let span = polyline_span(vec![0.0, 0.0, 1e9, 1e9], &[(0.0, 0.0), (1.0, 0.0)]);
        assert!(span.lies_on(&span.clone(), 1e-7));
    }

    /// A curve folding back 0.001 above itself lies on its copy: a point on
    /// its second leg is read on that leg, not on the first.
    #[test]
    fn a_folded_span_lies_on_its_copy() {
        let span = polyline_span(
            vec![0.0, 0.0, 0.4, 0.5, 1.0, 1.0],
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 0.001), (0.0, 0.001)],
        );
        assert!(span.lies_on(&span.clone(), 1e-7));
    }

    /// A curve with a bump 0.1 high inside its first knot span, from 0 to
    /// 0.1, does not lie on the unit segment. The segment's points all lie
    /// on the curve's long straight span, so the bump is what keeps the two
    /// apart, merging needing each on the other.
    #[test]
    fn a_bump_in_a_short_knot_span_keeps_curves_apart() {
        let bumped = polyline_span(
            vec![0.0, 0.0, 0.05, 0.1, 1.0, 1.0],
            &[(0.0, 0.0), (0.05, 0.1), (0.1, 0.0), (1.0, 0.0)],
        );
        let straight = Span::new(
            EdgeCurve::Line,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        );
        assert!(!bumped.lies_on(&straight, 1e-7));
        assert!(straight.lies_on(&bumped, 1e-7));
    }

    /// A bump 0.1 high inside a knot span 1e-13 long keeps the curve off
    /// the unit segment: a short parameter span is no short stretch.
    #[test]
    fn a_bump_in_a_vanishing_knot_span_keeps_curves_apart() {
        let bumped = polyline_span(
            vec![0.0, 0.0, 1e-13, 2e-13, 1.0, 1.0],
            &[(0.0, 0.0), (0.05, 0.1), (0.1, 0.0), (1.0, 0.0)],
        );
        let straight = Span::new(
            EdgeCurve::Line,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        );
        assert!(!bumped.lies_on(&straight, 1e-7));
    }

    /// Two copies of the cubic `(t (t - 0.2) (t - 0.25), 0.001 t, 0)`, which
    /// folds back on itself twice within a thousandth, lie on each other.
    #[test]
    fn copies_of_a_folded_cubic_lie_on_each_other() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.05 / 3.0, 0.001 / 3.0, 0.0),
            Point3::new(0.1 / 3.0 - 0.15, 0.002 / 3.0, 0.0),
            Point3::new(0.6, 0.001, 0.0),
        ];
        let curve = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            pts.clone(),
            vec![1.0; 4],
        )
        .unwrap();
        let fold = |c: &NurbsCurve| Span::new(EdgeCurve::NurbsCurve(c.clone()), pts[0], pts[3]);
        let (a, b) = (fold(&curve), fold(&curve));
        assert!(a.lies_on(&b, 1e-7) && b.lies_on(&a, 1e-7));
    }

    /// A rational cubic through (0, 0), (0, 0.01), (0, 0) and (1, 0) with
    /// weights 1e-12, 1e-6, 1 and 1 strays about 0.0043 off the unit segment
    /// near its start, and does not lie on it; nor with its second point at
    /// (0, 0.004), straying about 0.0017.
    #[test]
    fn a_rational_cubics_excursion_keeps_it_off_a_segment() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let straight = Span::new(EdgeCurve::Line, p(0.0, 0.0), p(1.0, 0.0));
        for height in [0.01, 0.004] {
            let curve = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![p(0.0, 0.0), p(0.0, height), p(0.0, 0.0), p(1.0, 0.0)],
                vec![1e-12, 1e-6, 1.0, 1.0],
            )
            .unwrap();
            let rational = Span::new(EdgeCurve::NurbsCurve(curve), p(0.0, 0.0), p(1.0, 0.0));
            assert!(!rational.lies_on(&straight, 1e-7), "height {height}");
        }
    }

    /// The quadratic through (0, 0), (1, 1) and (2, 0) on the unclamped
    /// knots 0 to 5 runs from (0.5, 0.5) to (1.5, 0.5) through (1, 0.75):
    /// each piece's ball holds it, and a copy lies on it.
    #[test]
    fn an_unclamped_quadratics_pieces_hold_it() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let curve = NurbsCurve::new(
            2,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![p(0.0, 0.0), p(1.0, 1.0), p(2.0, 0.0)],
            vec![1.0; 3],
        )
        .unwrap();
        assert!((curve.evaluate(2.5) - p(1.0, 0.75)).length() < 1e-12);
        let span =
            |c: &NurbsCurve| Span::new(EdgeCurve::NurbsCurve(c.clone()), p(0.5, 0.5), p(1.5, 0.5));
        let (a, b) = (span(&curve), span(&curve));
        for piece in &a.pieces {
            for k in 0..=8 {
                let f = (piece.to - piece.from).mul_add(f64::from(k) / 8.0, piece.from);
                assert!((a.at(f) - piece.centre).length() <= piece.radius + 1e-12);
            }
        }
        assert!(a.lies_on(&b, 1e-7) && b.lies_on(&a, 1e-7));
    }

    /// A unit circle's arc 4e-8 long, shorter than the tolerance, welds to
    /// one vertex and is dropped, rather than kept as a whole circle.
    #[test]
    fn sew_drops_an_arc_shorter_than_the_tolerance() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            up,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let [a, b, c] = [
            Point3::new(1.0, 0.0, 0.0),
            circle.evaluate(4e-8),
            Point3::new(0.0, 1.0, 0.0),
        ]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let edges = [
            topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
            topo.add_edge(Edge::new(b, c, EdgeCurve::Line)),
            topo.add_edge(Edge::new(c, a, EdgeCurve::Line)),
        ];
        let wire = topo.add_wire(
            Wire::new(
                edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let sliver = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: up, d: 0.0 },
        ));
        let far = make_loose_quad(
            &mut topo,
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(110.0, 0.0, 0.0),
            Point3::new(110.0, 10.0, 0.0),
            Point3::new(100.0, 10.0, 0.0),
            up,
            0.0,
        );
        let sewn = sew_faces(&mut topo, &[sliver, far], 1e-7).unwrap();
        assert!(
            curve_kinds(&topo, sewn)[0]
                .iter()
                .all(|&kind| kind == "line")
        );
    }

    /// A unit circle's arc from angle 0 to 2π - 4e-8 welds its ends into one
    /// vertex but runs almost the whole way round: it stays, as a closed
    /// edge, while the chord closing it, shorter than the tolerance, goes.
    #[test]
    fn sew_keeps_an_arc_just_short_of_a_whole_turn() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            up,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let [a, b] = [
            Point3::new(1.0, 0.0, 0.0),
            circle.evaluate(std::f64::consts::TAU - 4e-8),
        ]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let edges = [
            topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
            topo.add_edge(Edge::new(b, a, EdgeCurve::Line)),
        ];
        let wire = topo.add_wire(
            Wire::new(
                edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let disc = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: up, d: 0.0 },
        ));
        let far = make_loose_quad(
            &mut topo,
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(110.0, 0.0, 0.0),
            Point3::new(110.0, 10.0, 0.0),
            Point3::new(100.0, 10.0, 0.0),
            up,
            0.0,
        );
        let sewn = sew_faces(&mut topo, &[disc, far], 1e-7).unwrap();
        assert_eq!(curve_kinds(&topo, sewn)[0], ["circle"]);
        let face = brepkit_topology::explorer::solid_faces(&topo, sewn).unwrap()[0];
        let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
        let edge = topo.edge(wire.edges()[0].edge()).unwrap();
        assert_eq!(edge.start(), edge.end());
    }

    /// A spiral `r = 1 + 6e-9 θ` from θ = -π/2 to 2π, fit through 81 of its
    /// points, with an edge on it from θ = 0 to its end, whose ends lie
    /// 3.8e-8 apart and weld: the edge becomes a closed edge one turn long,
    /// cut from the curve, not the curve's whole turn and a quarter, whether
    /// its knots run over 1 or over 1e-13.
    /// A polyline whose first or last knot span is a millionth of a
    /// millionth of its domain yet holds a unit segment: a cut halfway along
    /// that span drops half the segment, at either end and either way along.
    #[test]
    fn nurbs_between_cuts_inside_a_narrow_knot_span() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let polyline = |knot: f64, corner: Point3| {
            NurbsCurve::new(
                1,
                vec![0.0, 0.0, knot, 1.0, 1.0],
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    corner,
                    Point3::new(1.0, 1.0, 0.0),
                ],
                vec![1.0; 3],
            )
            .unwrap()
        };
        let cuts_to = |c: &NurbsCurve, t0: f64, t1: f64| {
            let piece = nurbs_between(c, t0, t1).unwrap();
            let (a, b) = piece.domain();
            (piece.evaluate(a) - c.evaluate(t0)).length() < 1e-12
                && (piece.evaluate(b) - c.evaluate(t1)).length() < 1e-12
        };

        let first = polyline(1e-12, Point3::new(1.0, 0.0, 0.0));
        let half = 5e-13;
        assert!((first.evaluate(half) - Point3::new(0.5, 0.0, 0.0)).length() < 1e-3);
        assert!(cuts_to(&first, half, 1.0));
        assert!(cuts_to(&first, 1.0, half));

        let last = polyline(1.0 - 1e-12, Point3::new(0.0, 1.0, 0.0));
        let half = 1.0 - 5e-13;
        assert!((last.evaluate(half) - Point3::new(0.5, 1.0, 0.0)).length() < 1e-3);
        assert!(cuts_to(&last, 0.0, half));
        assert!(cuts_to(&last, half, 0.0));
        assert!(cuts_to(&last, 0.0, 1.0));
    }

    #[test]
    fn sew_cuts_a_welded_piece_of_a_longer_curve_to_its_span() {
        for scale in [1.0, 1e-13] {
            sew_a_welded_spiral_piece(scale);
        }
    }

    fn sew_a_welded_spiral_piece(scale: f64) {
        let mut topo = Topology::new();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let spiral = |theta: f64| {
            let r = 6e-9_f64.mul_add(theta, 1.0);
            Point3::new(r * theta.cos(), r * theta.sin(), 0.0)
        };
        let quarter = std::f64::consts::FRAC_PI_2;
        let points: Vec<Point3> = (0..=80)
            .map(|k| spiral((5.0 * quarter).mul_add(f64::from(k) / 80.0, -quarter)))
            .collect();
        let fit = brepkit_math::nurbs::fitting::interpolate(&points, 3).unwrap();
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            fit.degree(),
            fit.knots().iter().map(|k| k * scale).collect(),
            fit.control_points().to_vec(),
            fit.weights().to_vec(),
        )
        .unwrap();
        let [a, b] = [points[16], points[80]].map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let edges = [
            topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve))),
            topo.add_edge(Edge::new(b, a, EdgeCurve::Line)),
        ];
        let wire = topo.add_wire(
            Wire::new(
                edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let disc = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: up, d: 0.0 },
        ));
        let far = make_loose_quad(
            &mut topo,
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(110.0, 0.0, 0.0),
            Point3::new(110.0, 10.0, 0.0),
            Point3::new(100.0, 10.0, 0.0),
            up,
            0.0,
        );
        let sewn = sew_faces(&mut topo, &[disc, far], 1e-7).unwrap();
        let face = brepkit_topology::explorer::solid_faces(&topo, sewn).unwrap()[0];
        let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
        assert_eq!(wire.edges().len(), 1);
        let edge = topo.edge(wire.edges()[0].edge()).unwrap();
        let p = topo.vertex(edge.start()).unwrap().point();
        assert_eq!(edge.start(), edge.end());
        let (t0, t1) = edge.curve().domain_with_endpoints(p, p);
        let length: f64 = (0..2000)
            .map(|k| {
                let at = |k: i32| {
                    let t = (t1 - t0).mul_add(f64::from(k) / 2000.0, t0);
                    edge.curve().evaluate_with_endpoints(t, p, p)
                };
                (at(k + 1) - at(k)).length()
            })
            .sum();
        assert!(
            (length - std::f64::consts::TAU).abs() < 1e-3,
            "knots over {scale}: {length}"
        );
    }

    /// The cylinder's faces with one cap's rim swapped for `curve`, used
    /// forward or not.
    fn cylinder_with_rim(
        topo: &mut Topology,
        top: bool,
        curve: brepkit_math::curves::Circle3D,
        forward: bool,
    ) -> Vec<FaceId> {
        let cyl = crate::primitives::make_cylinder(topo, 10.0, 20.0).unwrap();
        let mut faces = brepkit_topology::explorer::solid_faces(topo, cyl).unwrap();
        let z = if top { 20.0 } else { 0.0 };
        let at = faces
            .iter()
            .position(|&f| {
                let face = topo.face(f).unwrap();
                face.surface().is_planar() && {
                    let wire = topo.wire(face.outer_wire()).unwrap();
                    let edge = topo.edge(wire.edges()[0].edge()).unwrap();
                    (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
                }
            })
            .unwrap();
        let (surface, reversed, vertex) = {
            let face = topo.face(faces[at]).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            let edge = topo.edge(wire.edges()[0].edge()).unwrap();
            (face.surface().clone(), face.is_reversed(), edge.start())
        };
        let edge = topo.add_edge(Edge::new(vertex, vertex, EdgeCurve::Circle(curve)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, forward)], true).unwrap());
        let mut cap = Face::new(wire, vec![], surface);
        cap.set_reversed(reversed);
        faces[at] = topo.add_face(cap);
        faces
    }

    fn assert_whole_cylinder(topo: &Topology, sewn: SolidId) {
        let edges: std::collections::HashSet<usize> =
            brepkit_topology::explorer::solid_faces(topo, sewn)
                .unwrap()
                .iter()
                .flat_map(|&f| {
                    let face = topo.face(f).unwrap();
                    topo.wire(face.outer_wire()).unwrap().edges().to_vec()
                })
                .map(|oe| oe.edge().index())
                .collect();
        assert_eq!(edges.len(), 3);
        assert!(
            crate::validate::validate_solid(topo, sewn)
                .unwrap()
                .is_valid()
        );
        let volume = crate::measure::solid_volume(topo, sewn, 0.01).unwrap();
        let exact = std::f64::consts::PI * 100.0 * 20.0;
        assert!((volume - exact).abs() < 1e-3 * exact, "{volume}");
    }

    /// A bottom cap whose rim is the circle about -z, run forward, meets
    /// the wall's circle about +z, from the same start, the other way along.
    #[test]
    fn sew_turns_a_closed_rim_running_the_other_way() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let rim = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            10.0,
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let faces = cylinder_with_rim(&mut topo, false, rim, true);
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        assert_whole_cylinder(&topo, sewn);
    }

    /// A top cap whose rim circle starts a quarter turn round from the
    /// wall's still shares the wall's edge.
    #[test]
    fn sew_shares_a_closed_rim_whose_circle_starts_elsewhere() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let rim = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 20.0),
            Vec3::new(0.0, 0.0, 1.0),
            10.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let faces = cylinder_with_rim(&mut topo, true, rim, true);
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        assert_whole_cylinder(&topo, sewn);
    }

    /// Two discs on perpendicular planes, both closed at (1, 0, 0) with the
    /// same middle point, keep their own circles.
    #[test]
    fn sew_keeps_perpendicular_circles_through_one_point_apart() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let x = Vec3::new(1.0, 0.0, 0.0);
        let disc = |topo: &mut Topology, normal: Vec3| {
            let v = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
            let circle =
                Circle3D::new_with_ref(Point3::new(0.0, 0.0, 0.0), normal, 1.0, x).unwrap();
            let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
            topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane { normal, d: 0.0 },
            ))
        };
        let flat = disc(&mut topo, Vec3::new(0.0, 0.0, 1.0));
        let upright = disc(&mut topo, Vec3::new(0.0, 1.0, 0.0));
        let sewn = sew_faces(&mut topo, &[flat, upright], 1e-7).unwrap();
        let rims: Vec<_> = brepkit_topology::explorer::solid_faces(&topo, sewn)
            .unwrap()
            .iter()
            .map(|&f| {
                let wire = topo.wire(topo.face(f).unwrap().outer_wire()).unwrap();
                wire.edges()[0].edge()
            })
            .collect();
        assert_ne!(rims[0], rims[1]);
        let EdgeCurve::Circle(upright_rim) = topo.edge(rims[1]).unwrap().curve() else {
            unreachable!("the rim is a circle")
        };
        assert!(upright_rim.evaluate(1.0).y().abs() < 1e-12);
    }

    /// A cube whose bottom and front meet along one segment carried by two
    /// differently parameterized curves sews into the closed cube.
    #[test]
    fn sew_shares_a_segment_however_its_curves_run() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let mut topo = Topology::new();
        let p = Point3::new;
        let segment = |xs: [f64; 3]| {
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    xs.iter().map(|&x| p(x, 0.0, 0.0)).collect(),
                    vec![1.0; 3],
                )
                .unwrap(),
            )
        };
        let quad = |topo: &mut Topology,
                    corners: [Point3; 4],
                    normal: Vec3,
                    d: f64,
                    first: Option<(EdgeCurve, bool)>| {
            let v: Vec<_> = corners
                .iter()
                .map(|&c| topo.add_vertex(Vertex::new(c, 1e-7)))
                .collect();
            let mut edges = Vec::new();
            for i in 0..4 {
                let (a, b) = (v[i], v[(i + 1) % 4]);
                let oriented = match (&first, i) {
                    (Some((curve, true)), 0) => {
                        OrientedEdge::new(topo.add_edge(Edge::new(a, b, curve.clone())), true)
                    }
                    (Some((curve, false)), 0) => {
                        OrientedEdge::new(topo.add_edge(Edge::new(b, a, curve.clone())), false)
                    }
                    _ => OrientedEdge::new(topo.add_edge(Edge::new(a, b, EdgeCurve::Line)), true),
                };
                edges.push(oriented);
            }
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }))
        };
        let faces = [
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 0.0),
                    p(1.0, 0.0, 0.0),
                    p(1.0, 1.0, 0.0),
                    p(0.0, 1.0, 0.0),
                ],
                Vec3::new(0.0, 0.0, -1.0),
                0.0,
                Some((segment([0.0, 0.2, 1.0]), true)),
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 1.0),
                    p(1.0, 0.0, 1.0),
                    p(1.0, 1.0, 1.0),
                    p(0.0, 1.0, 1.0),
                ],
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                None,
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 0.0),
                    p(1.0, 0.0, 0.0),
                    p(1.0, 0.0, 1.0),
                    p(0.0, 0.0, 1.0),
                ],
                Vec3::new(0.0, -1.0, 0.0),
                0.0,
                Some((segment([1.0, 0.8, 0.0]), false)),
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 1.0, 0.0),
                    p(1.0, 1.0, 0.0),
                    p(1.0, 1.0, 1.0),
                    p(0.0, 1.0, 1.0),
                ],
                Vec3::new(0.0, 1.0, 0.0),
                1.0,
                None,
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 0.0),
                    p(0.0, 1.0, 0.0),
                    p(0.0, 1.0, 1.0),
                    p(0.0, 0.0, 1.0),
                ],
                Vec3::new(-1.0, 0.0, 0.0),
                0.0,
                None,
            ),
            quad(
                &mut topo,
                [
                    p(1.0, 0.0, 0.0),
                    p(1.0, 1.0, 0.0),
                    p(1.0, 1.0, 1.0),
                    p(1.0, 0.0, 1.0),
                ],
                Vec3::new(1.0, 0.0, 0.0),
                1.0,
                None,
            ),
        ];
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        let edges: std::collections::HashSet<usize> =
            brepkit_topology::explorer::solid_faces(&topo, sewn)
                .unwrap()
                .iter()
                .flat_map(|&f| {
                    let face = topo.face(f).unwrap();
                    topo.wire(face.outer_wire()).unwrap().edges().to_vec()
                })
                .map(|oe| oe.edge().index())
                .collect();
        assert_eq!(edges.len(), 12);
    }

    /// A block with a round hole through it, its faces sewn again, keeps
    /// the hole: the caps keep their inner circles and the solid its volume.
    #[test]
    fn sew_keeps_a_faces_holes() {
        use crate::boolean::{BooleanOp, boolean};

        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
        let rod = crate::primitives::make_cylinder(&mut topo, 4.0, 30.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            rod,
            &brepkit_math::mat::Mat4::translation(10.0, 10.0, -10.0),
        )
        .unwrap();
        let holed = boolean(&mut topo, BooleanOp::Cut, block, rod).unwrap();
        let before = crate::measure::solid_volume(&topo, holed, 0.01).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, holed).unwrap();
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        let holes: usize = brepkit_topology::explorer::solid_faces(&topo, sewn)
            .unwrap()
            .iter()
            .map(|&f| topo.face(f).unwrap().inner_wires().len())
            .sum();
        assert_eq!(holes, 2);
        assert!(
            crate::validate::validate_solid(&topo, sewn)
                .unwrap()
                .is_valid()
        );
        let after = crate::measure::solid_volume(&topo, sewn, 0.01).unwrap();
        assert!(
            (after - before).abs() < 1e-6 * before,
            "{after} vs {before}"
        );
    }
}
