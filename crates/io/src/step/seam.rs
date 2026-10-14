//! Seams that a STEP file leaves out of a face of revolution.
//!
//! A STEP face may bound a full cylinder or cone band by its two rims alone,
//! and a pointed cone by its base rim alone (with or without a vertex loop at
//! the apex), with no edge where the surface's angle wraps. brepkit bounds
//! every such face with one loop that runs up a seam line and back, as its
//! own primitives do, and its meshers and booleans rely on that seam. Once a
//! shell's faces are built, [`close_seams`] adds the seam each one lacks.

use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, TAU};

use brepkit_math::frame::Frame3;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::IoError;

/// Samples per edge when measuring how far a loop turns about the axis.
const WINDING_SAMPLES: u32 = 16;
/// Samples per edge when looking for where a rim crosses a generator.
const CROSSING_SAMPLES: u32 = 64;

/// Add the missing seam to every face of `faces` that needs one. `apexes`
/// holds the vertex a face's vertex loop puts at its cone's apex.
pub(super) fn close_seams(
    topo: &mut Topology,
    faces: &[FaceId],
    apexes: &HashMap<FaceId, VertexId>,
) -> Result<(), IoError> {
    for &fid in faces {
        close_seam(topo, faces, fid, apexes.get(&fid).copied())?;
    }
    Ok(())
}

/// The axis a cylinder or cone turns about, and its frame for angles.
struct Revolution {
    frame: Frame3,
    /// A cone's apex, through which all its generators run.
    apex: Option<Point3>,
}

impl Revolution {
    fn of(surface: &FaceSurface) -> Option<Self> {
        let (origin, axis, apex) = match surface {
            FaceSurface::Cylinder(c) => (c.origin(), c.axis(), None),
            FaceSurface::Cone(c) => (c.apex(), c.axis(), Some(c.apex())),
            _ => return None,
        };
        let frame = Frame3::from_normal(origin, axis).ok()?;
        Some(Self { frame, apex })
    }

    fn angle(&self, p: Point3) -> f64 {
        let d = p - self.frame.origin;
        d.dot(self.frame.y).atan2(d.dot(self.frame.x))
    }

    /// How far `b` lies off the generator through `a`.
    fn generator_gap(&self, a: Point3, b: Point3) -> f64 {
        let along = match self.apex {
            None => self.frame.z,
            Some(apex) => match (a - apex).normalize() {
                Ok(d) => d,
                Err(_) => return f64::INFINITY,
            },
        };
        let d = b - a;
        (d - along * d.dot(along)).length()
    }

    /// The gap below which two points share a generator: the file's
    /// coordinates carry about fifteen digits of the part's size.
    fn tolerance(&self, p: Point3) -> f64 {
        1e-7 * (1.0 + (p - self.frame.origin).length())
    }
}

fn oriented_ends(topo: &Topology, oe: OrientedEdge) -> Result<(VertexId, VertexId), IoError> {
    let e = topo.edge(oe.edge())?;
    Ok(if oe.is_forward() {
        (e.start(), e.end())
    } else {
        (e.end(), e.start())
    })
}

/// Points along an oriented edge from its start, its end left out.
fn edge_samples(topo: &Topology, oe: OrientedEdge, n: u32) -> Result<Vec<Point3>, IoError> {
    let e = topo.edge(oe.edge())?;
    let (a, b) = (
        topo.vertex(e.start())?.point(),
        topo.vertex(e.end())?.point(),
    );
    let (t0, t1) = e.curve().domain_with_endpoints(a, b);
    let mut points: Vec<Point3> = (0..=n)
        .map(|k| {
            let t = (t1 - t0).mul_add(f64::from(k) / f64::from(n), t0);
            e.curve().evaluate_with_endpoints(t, a, b)
        })
        .collect();
    if !oe.is_forward() {
        points.reverse();
    }
    points.pop();
    Ok(points)
}

/// Whole turns a loop makes about the axis.
fn winding(topo: &Topology, rev: &Revolution, wire: WireId) -> Result<i64, IoError> {
    let mut points = Vec::new();
    for &oe in topo.wire(wire)?.edges() {
        points.extend(edge_samples(topo, oe, WINDING_SAMPLES)?);
    }
    let mut turn = 0.0;
    for (i, &p) in points.iter().enumerate() {
        let q = points[(i + 1) % points.len()];
        let step = rev.angle(q) - rev.angle(p);
        turn += (step + PI).rem_euclid(TAU) - PI;
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok((turn / TAU).round() as i64)
}

/// A loop that already runs down a seam uses some edge both ways.
fn has_seam(topo: &Topology, wire: WireId) -> Result<bool, IoError> {
    let mut seen = HashSet::new();
    for oe in topo.wire(wire)?.edges() {
        if !seen.insert(oe.edge()) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[allow(clippy::too_many_lines)]
fn close_seam(
    topo: &mut Topology,
    faces: &[FaceId],
    fid: FaceId,
    apex_vertex: Option<VertexId>,
) -> Result<(), IoError> {
    let face = topo.face(fid)?;
    let Some(rev) = Revolution::of(face.surface()) else {
        return Ok(());
    };
    let wires: Vec<WireId> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    let mut rims = Vec::new();
    let mut holes = Vec::new();
    for &w in &wires {
        if has_seam(topo, w)? {
            return Ok(());
        }
        if winding(topo, &rev, w)? == 0 {
            holes.push(w);
        } else {
            rims.push(w);
        }
    }

    let outer = match (rims.as_slice(), rev.apex) {
        ([], _) => return Ok(()),
        (&[rim], Some(apex)) => {
            let rim_edges = topo.wire(rim)?.edges().to_vec();
            let (a, _) = oriented_ends(topo, rim_edges[0])?;
            let apex_vertex = match apex_vertex {
                Some(v) => v,
                None => topo.add_vertex(Vertex::new(apex, 1e-7)),
            };
            let seam = topo.add_edge(Edge::new(a, apex_vertex, EdgeCurve::Line));
            let mut edges = rim_edges;
            edges.push(OrientedEdge::new(seam, true));
            edges.push(OrientedEdge::new(seam, false));
            edges
        }
        (&[first, second], _) => {
            let (a, b) = seam_ends(topo, faces, &rev, first, second)?;
            // Splitting a rim edge gives its loop a new id; the holes keep
            // theirs.
            let face = topo.face(fid)?;
            let rims: Vec<WireId> = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .filter(|w| !holes.contains(w))
                .collect();
            let rim_from = |v: VertexId| -> Result<Option<Vec<OrientedEdge>>, IoError> {
                for &w in &rims {
                    let edges = topo.wire(w)?.edges().to_vec();
                    for k in 0..edges.len() {
                        if oriented_ends(topo, edges[k])?.0 == v {
                            let mut rotated = edges[k..].to_vec();
                            rotated.extend_from_slice(&edges[..k]);
                            return Ok(Some(rotated));
                        }
                    }
                }
                Ok(None)
            };
            let (Some(rim_a), Some(rim_b)) = (rim_from(a)?, rim_from(b)?) else {
                return Err(IoError::ParseError {
                    reason: format!("face {fid:?}: a seam end is on no rim"),
                });
            };
            let seam = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
            let mut edges = rim_a;
            edges.push(OrientedEdge::new(seam, true));
            edges.extend(rim_b);
            edges.push(OrientedEdge::new(seam, false));
            edges
        }
        _ => {
            log::warn!(
                "STEP face {fid:?}: {} loops wind about its axis; left without a seam",
                rims.len()
            );
            return Ok(());
        }
    };

    let outer = topo.add_wire(Wire::new(outer, true)?);
    let face = topo.face_mut(fid)?;
    face.set_outer_wire(outer);
    *face.inner_wires_mut() = holes;
    Ok(())
}

/// The ends of a seam between two rims: a vertex of each on one generator,
/// made by moving or splitting a rim edge where the rims share none.
fn seam_ends(
    topo: &mut Topology,
    faces: &[FaceId],
    rev: &Revolution,
    first: WireId,
    second: WireId,
) -> Result<(VertexId, VertexId), IoError> {
    let starts = |topo: &Topology, w: WireId| -> Result<Vec<VertexId>, IoError> {
        topo.wire(w)?
            .edges()
            .iter()
            .map(|&oe| oriented_ends(topo, oe).map(|(s, _)| s))
            .collect()
    };
    let (va, vb) = (starts(topo, first)?, starts(topo, second)?);
    let mut best: Option<(f64, VertexId, VertexId)> = None;
    for &a in &va {
        let pa = topo.vertex(a)?.point();
        for &b in &vb {
            let gap = rev.generator_gap(pa, topo.vertex(b)?.point());
            if gap < rev.tolerance(pa) && best.is_none_or(|(g, _, _)| gap < g) {
                best = Some((gap, a, b));
            }
        }
    }
    if let Some((_, a, b)) = best {
        return Ok((a, b));
    }

    // Put the new vertex on whichever rim takes it without a split.
    let movable = |topo: &Topology, w: WireId| -> Result<bool, IoError> {
        let edges = topo.wire(w)?.edges();
        Ok(edges.len() == 1 && closed_conic_alone(topo, faces, edges[0].edge())?)
    };
    let (fixed, onto) = if movable(topo, second)? || !movable(topo, first)? {
        (va[0], second)
    } else {
        (vb[0], first)
    };
    let at = topo.vertex(fixed)?.point();
    let new = vertex_on_rim(topo, faces, rev, onto, at)?;
    Ok(if onto == second {
        (fixed, new)
    } else {
        (new, fixed)
    })
}

/// Whether `edge` is a closed circle or ellipse whose vertex no other edge
/// of the shell uses, so the vertex can slide round it.
fn closed_conic_alone(topo: &Topology, faces: &[FaceId], edge: EdgeId) -> Result<bool, IoError> {
    let e = topo.edge(edge)?;
    if e.start() != e.end() || !matches!(e.curve(), EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
        return Ok(false);
    }
    let v = e.start();
    for &fid in faces {
        let face = topo.face(fid)?;
        for w in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(w)?.edges() {
                let other = topo.edge(oe.edge())?;
                if oe.edge() != edge && (other.start() == v || other.end() == v) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// A vertex of rim `rim` on the generator through `at`: its existing vertex
/// slid there when the rim is a lone closed conic, else a new vertex
/// splitting the rim edge that crosses the generator.
fn vertex_on_rim(
    topo: &mut Topology,
    faces: &[FaceId],
    rev: &Revolution,
    rim: WireId,
    at: Point3,
) -> Result<VertexId, IoError> {
    let target = rev.angle(at);
    let off = |p: Point3| (rev.angle(p) - target + PI).rem_euclid(TAU) - PI;
    let rim_edges = topo.wire(rim)?.edges().to_vec();
    let mut crossing = None;
    'edges: for &oe in &rim_edges {
        let e = topo.edge(oe.edge())?;
        let (a, b) = (
            topo.vertex(e.start())?.point(),
            topo.vertex(e.end())?.point(),
        );
        let (t0, t1) = e.curve().domain_with_endpoints(a, b);
        let point_at = |t: f64| e.curve().evaluate_with_endpoints(t, a, b);
        let mut lo = t0;
        for k in 1..=CROSSING_SAMPLES {
            let hi = (t1 - t0).mul_add(f64::from(k) / f64::from(CROSSING_SAMPLES), t0);
            let (f_lo, f_hi) = (off(point_at(lo)), off(point_at(hi)));
            // A sign change across the far side of the circle is the wrap of
            // the angle, not a crossing.
            if f_lo.signum() != f_hi.signum() && (f_lo - f_hi).abs() < PI {
                let (mut l, mut h) = (lo, hi);
                for _ in 0..100 {
                    let m = 0.5 * (l + h);
                    if off(point_at(m)).signum() == f_lo.signum() {
                        l = m;
                    } else {
                        h = m;
                    }
                }
                let t = 0.5 * (l + h);
                crossing = Some((oe.edge(), t, point_at(t)));
                break 'edges;
            }
            lo = hi;
        }
    }
    let Some((edge, t, point)) = crossing else {
        return Err(IoError::ParseError {
            reason: "a rim of a seamless band never crosses the other rim's generator".into(),
        });
    };

    if closed_conic_alone(topo, faces, edge)? {
        let v = topo.edge(edge)?.start();
        topo.vertex_mut(v)?.set_point(point);
        anchor_circle(topo, edge)?;
        return Ok(v);
    }

    let (start, end, curve) = {
        let e = topo.edge(edge)?;
        (e.start(), e.end(), e.curve().clone())
    };
    // A B-spline edge's ends are found by projecting its vertices onto its
    // curve, which cannot tell a closed curve's start from its end: each part
    // gets its own curve.
    let (head_curve, tail_curve) = match &curve {
        EdgeCurve::NurbsCurve(n) if start == end => {
            let (h, t) = brepkit_math::nurbs::knot_ops::curve_split(n, t).map_err(|e| {
                IoError::ParseError {
                    reason: format!("splitting a rim at its seam: {e}"),
                }
            })?;
            (EdgeCurve::NurbsCurve(h), EdgeCurve::NurbsCurve(t))
        }
        _ => (curve.clone(), curve),
    };
    let v = topo.add_vertex(Vertex::new(point, 1e-7));
    let head = topo.add_edge(Edge::new(start, v, head_curve));
    let tail = topo.add_edge(Edge::new(v, end, tail_curve));
    split_in_wires(topo, faces, edge, head, tail)?;
    Ok(v)
}

/// Start a closed circle at its vertex, where its loop starts and ends.
pub(super) fn anchor_circle(topo: &mut Topology, edge: EdgeId) -> Result<(), IoError> {
    let e = topo.edge(edge)?;
    let EdgeCurve::Circle(c) = e.curve() else {
        return Ok(());
    };
    if e.start() != e.end() {
        return Ok(());
    }
    let from_center = topo.vertex(e.start())?.point() - c.center();
    let Ok(anchored) = brepkit_math::curves::Circle3D::new_with_ref(
        c.center(),
        c.normal(),
        c.radius(),
        from_center,
    ) else {
        return Ok(());
    };
    topo.edge_mut(edge)?.set_curve(EdgeCurve::Circle(anchored));
    Ok(())
}

/// Replace `edge` by `head` then `tail` in every loop of `faces` that uses it.
fn split_in_wires(
    topo: &mut Topology,
    faces: &[FaceId],
    edge: EdgeId,
    head: EdgeId,
    tail: EdgeId,
) -> Result<(), IoError> {
    for &fid in faces {
        let face = topo.face(fid)?;
        let wires: Vec<WireId> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for w in wires {
            let wire = topo.wire(w)?;
            if !wire.edges().iter().any(|oe| oe.edge() == edge) {
                continue;
            }
            let closed = wire.is_closed();
            let mut edges = Vec::with_capacity(wire.edges().len() + 1);
            for &oe in wire.edges() {
                if oe.edge() != edge {
                    edges.push(oe);
                } else if oe.is_forward() {
                    edges.push(OrientedEdge::new(head, true));
                    edges.push(OrientedEdge::new(tail, true));
                } else {
                    edges.push(OrientedEdge::new(tail, false));
                    edges.push(OrientedEdge::new(head, false));
                }
            }
            let new = topo.add_wire(Wire::new(edges, closed)?);
            let face = topo.face_mut(fid)?;
            if face.outer_wire() == w {
                face.set_outer_wire(new);
            }
            for inner in face.inner_wires_mut().iter_mut() {
                if *inner == w {
                    *inner = new;
                }
            }
        }
    }
    Ok(())
}
