//! Chamfers along tangent chains of lines and arcs on one flat cap.
//!
//! A rounded prism's rim is a chain of straight edges between planes and arcs
//! between the cap and a coaxial cylinder, each edge tangent to the next. The
//! same triangle runs along the whole chain, so its chamfer can be written
//! down: each edge moves into the cap and along the wall, a plane (over a
//! line) or a cone (over an arc) joins the two, and a chain stopping at a
//! tangent neighbour ends in a flat triangle across the edge. Nothing here
//! depends on how the solid is posed, and the result's volume has a closed
//! form, which is checked before it is returned.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::TAU;

use brepkit_geometry::extrema::segment_segment_distance;
use brepkit_math::curves::Circle3D;
use brepkit_math::frame::Frame3;
use brepkit_math::region2d::point_in_region;
use brepkit_math::surfaces::{ConicalSurface, CylindricalSurface};
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::planar::face_boundary_2d;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::OperationsError;

/// How far two readings of one point or direction may differ at a join.
const JOIN_TOL: f64 = 1e-7;

/// Samples along an edge when testing it against a chamfer's strip.
const STRIP_SAMPLES: u32 = 32;

/// How far apart the change in volume and its closed form may lie, as a
/// fraction of all the volume the chains sweep, so a convex chain and a concave
/// one cancelling leave the margin intact. Far looser than either reads: a face
/// without an exact reader elsewhere on the solid makes both volumes meshes.
const VOLUME_MARGIN: f64 = 0.01;

/// One selected edge: a line between two planes, or an open arc between the
/// cap and a coaxial cylinder.
struct Piece {
    edge: EdgeId,
    wall: FaceId,
    start: VertexId,
    end: VertexId,
    curve: EdgeCurve,
    start_point: Point3,
    end_point: Point3,
    /// `cap normal x tangent`, signed to point into the cap's material.
    cap_sign: f64,
    /// Into the wall's material: `wall normal x tangent`, signed, on a plane;
    /// along the axis on a cylinder.
    wall_side: WallSide,
}

#[derive(Clone, Copy)]
enum WallSide {
    Plane { normal: Vec3, sign: f64 },
    Axis(Vec3),
}

impl Piece {
    fn at(&self, vertex: VertexId) -> Point3 {
        if vertex == self.start {
            self.start_point
        } else {
            self.end_point
        }
    }

    /// The direction of the edge's stored sense at `p`.
    fn tangent(&self, p: Point3) -> Result<Vec3, OperationsError> {
        Ok(match &self.curve {
            EdgeCurve::Line => (self.end_point - self.start_point).normalize()?,
            EdgeCurve::Circle(c) => c.tangent(c.project(p)).normalize()?,
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                return Err(OperationsError::InvalidInput {
                    reason: "a chamfer chain runs only along lines and arcs".into(),
                });
            }
        })
    }

    /// The direction leaving the edge's other end through `vertex`.
    fn leaving(&self, vertex: VertexId) -> Result<Vec3, OperationsError> {
        let tangent = self.tangent(self.at(vertex))?;
        Ok(if vertex == self.end {
            tangent
        } else {
            -tangent
        })
    }

    fn wall_dir(&self, p: Point3) -> Result<Vec3, OperationsError> {
        Ok(match self.wall_side {
            WallSide::Plane { normal, sign } => normal.cross(self.tangent(p)?) * sign,
            WallSide::Axis(axis) => axis,
        })
    }

    fn midpoint(&self) -> Point3 {
        let (t0, t1) = self
            .curve
            .domain_with_endpoints(self.start_point, self.end_point);
        self.curve
            .evaluate_with_endpoints(0.5 * (t0 + t1), self.start_point, self.end_point)
    }
}

/// Where the chamfer's triangle stands at one vertex of a chain.
#[derive(Clone, Copy)]
struct Section {
    cap: Point3,
    wall: Point3,
    cap_dir: Vec3,
    wall_dir: Vec3,
}

/// A chain vertex where two walls meet along a ruling, which shortens.
struct Joint {
    vertex: VertexId,
    ruling: EdgeId,
    walls: [FaceId; 2],
}

/// A chain's free end, where its wall meets the tangent neighbour along a
/// ruling, which the chamfer takes the lower part of.
struct End {
    vertex: VertexId,
    ruling: EdgeId,
    wall: FaceId,
    beyond: FaceId,
}

/// One tangent chain, read off the solid and ready to build.
struct Plan {
    cap: FaceId,
    pieces: Vec<Piece>,
    sections: Vec<(VertexId, Section)>,
    joints: Vec<Joint>,
    ends: Vec<End>,
    /// The chain's change in the solid's volume.
    volume_change: f64,
    /// Whether the chain needs this path: it holds an arc or stops at a
    /// curved neighbour, where the other engines' ends are not exact.
    curved: bool,
}

impl Plan {
    fn section(&self, vertex: VertexId) -> Option<Section> {
        self.sections
            .iter()
            .find(|(v, _)| *v == vertex)
            .map(|&(_, s)| s)
    }

    /// Whether `p`, on the cap, lies in the strip the chamfer takes from it.
    fn strip_holds(&self, p: Point3, d: f64, tol: f64) -> Result<bool, OperationsError> {
        for piece in &self.pieces {
            let held = match &piece.curve {
                EdgeCurve::Line => {
                    let along = piece.end_point - piece.start_point;
                    let length = along.length();
                    let tangent = along.normalize()?;
                    let offset = p - piece.start_point;
                    let s = offset.dot(tangent);
                    let Some(section) = self.section(piece.start) else {
                        continue;
                    };
                    let h = offset.dot(section.cap_dir);
                    s > tol && s < length - tol && h > tol && h <= d + tol
                }
                EdgeCurve::Circle(c) => {
                    let Some(section) = self.section(piece.start) else {
                        continue;
                    };
                    let radial = (piece.start_point - c.center()).normalize()?;
                    let toward = section.cap_dir.dot(radial);
                    let h = ((p - c.center()).length() - c.radius()) * toward;
                    let (t0, t1) = piece
                        .curve
                        .domain_with_endpoints(piece.start_point, piece.end_point);
                    let swept = (c.project(p) - t0).rem_euclid(TAU);
                    swept < t1 - t0 && h > tol && h <= d + tol
                }
                EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
            };
            if held {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn edge_samples(curve: &EdgeCurve, start: Point3, end: Point3) -> Vec<Point3> {
    let (t0, t1) = curve.domain_with_endpoints(start, end);
    (0..=STRIP_SAMPLES)
        .map(|k| {
            let t = t0 + (t1 - t0) * f64::from(k) / f64::from(STRIP_SAMPLES);
            curve.evaluate_with_endpoints(t, start, end)
        })
        .collect()
}

/// A stretch of a chain's new boundary, or of another edge on its cap.
enum Span {
    Segment(Point3, Point3),
    Arc {
        circle: Circle3D,
        t0: f64,
        sweep: f64,
    },
}

impl Span {
    /// `None` for an ellipse or NURBS edge, which is not read exactly here.
    fn of(curve: &EdgeCurve, start: Point3, end: Point3) -> Option<Self> {
        match curve {
            EdgeCurve::Line => Some(Self::Segment(start, end)),
            EdgeCurve::Circle(circle) => {
                let (t0, t1) = curve.domain_with_endpoints(start, end);
                Some(Self::Arc {
                    circle: circle.clone(),
                    t0,
                    sweep: t1 - t0,
                })
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => None,
        }
    }

    /// Where the span starts and ends.
    fn ends(&self) -> [Point3; 2] {
        match self {
            Self::Segment(a, b) => [*a, *b],
            Self::Arc { circle, t0, sweep } => {
                [circle.evaluate(*t0), circle.evaluate(*t0 + *sweep)]
            }
        }
    }

    fn holds(circle: &Circle3D, t0: f64, sweep: f64, p: Point3, tol: f64) -> bool {
        let slack = tol / circle.radius();
        (circle.project(p) - t0 + slack).rem_euclid(TAU) <= sweep + 2.0 * slack
    }

    /// Where two spans on one plane touch, exactly: a crossing between any
    /// two samples of them still counts.
    fn touches(&self, other: &Self, tol: f64) -> Vec<Point3> {
        match (self, other) {
            (Self::Segment(a, b), Self::Segment(c, d)) => {
                let (distance, p, _) = segment_segment_distance(*a, *b, *c, *d);
                if distance <= tol { vec![p] } else { Vec::new() }
            }
            (Self::Segment(a, b), Self::Arc { circle, t0, sweep })
            | (Self::Arc { circle, t0, sweep }, Self::Segment(a, b)) => circle
                .intersect_segment(*a, *b, tol)
                .into_iter()
                .map(|(p, _)| p)
                .filter(|&p| Self::holds(circle, *t0, *sweep, p, tol))
                .collect(),
            (
                Self::Arc { circle, t0, sweep },
                Self::Arc {
                    circle: other,
                    t0: other_t0,
                    sweep: other_sweep,
                },
            ) => {
                // One circle: they touch only where an end of one lies on the other.
                if (circle.center() - other.center()).length() <= tol
                    && (circle.radius() - other.radius()).abs() <= tol
                {
                    let ends = [*t0, *t0 + *sweep];
                    let other_ends = [*other_t0, *other_t0 + *other_sweep];
                    return ends
                        .map(|t| circle.evaluate(t))
                        .into_iter()
                        .filter(|&p| Self::holds(other, *other_t0, *other_sweep, p, tol))
                        .chain(
                            other_ends
                                .map(|t| other.evaluate(t))
                                .into_iter()
                                .filter(|&p| Self::holds(circle, *t0, *sweep, p, tol)),
                        )
                        .collect();
                }
                circle
                    .intersect_circle(other, tol)
                    .into_iter()
                    .map(|(p, _)| p)
                    .filter(|&p| {
                        Self::holds(circle, *t0, *sweep, p, tol)
                            && Self::holds(other, *other_t0, *other_sweep, p, tol)
                    })
                    .collect()
            }
        }
    }
}

/// The plane normal of a face that is a plane.
fn plane_normal(topo: &Topology, face: FaceId) -> Result<Option<Vec3>, OperationsError> {
    Ok(match topo.face(face)?.surface() {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Nurbs(_) => None,
    })
}

/// The cylinder of a face that is one.
fn cylinder_of(
    topo: &Topology,
    face: FaceId,
) -> Result<Option<CylindricalSurface>, OperationsError> {
    Ok(match topo.face(face)?.surface() {
        FaceSurface::Cylinder(cylinder) => Some(cylinder.clone()),
        FaceSurface::Plane { .. }
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Nurbs(_) => None,
    })
}

/// Which way from `m`, along `dir`, the planar `face` lies: `1` along `dir`,
/// `-1` against it, `None` when neither or both.
fn side_of_plane_face(
    topo: &Topology,
    face: FaceId,
    normal: Vec3,
    m: Point3,
    dir: Vec3,
    step: f64,
) -> Result<Option<f64>, OperationsError> {
    let frame = Frame3::from_normal(m, normal)?;
    let Some(boundary) = face_boundary_2d(topo, face, m, frame.x, frame.y)? else {
        return Ok(None);
    };
    let flat = Point2::new(dir.dot(frame.x) * step, dir.dot(frame.y) * step);
    let opposite = Point2::new(-flat.x(), -flat.y());
    let tol = step * 1e-3;
    Ok(
        match (
            point_in_region(&boundary, flat, tol),
            point_in_region(&boundary, opposite, tol),
        ) {
            (Some(true), Some(false)) => Some(1.0),
            (Some(false), Some(true)) => Some(-1.0),
            _ => None,
        },
    )
}

/// The ruling of `wall` leaving `vertex` along `dir`, and how long it runs.
fn ruling_along(
    topo: &Topology,
    wall: FaceId,
    vertex: VertexId,
    dir: Vec3,
) -> Result<Option<(EdgeId, f64)>, OperationsError> {
    let at = topo.vertex(vertex)?.point();
    for oe in topo.wire(topo.face(wall)?.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        if !matches!(edge.curve(), EdgeCurve::Line) {
            continue;
        }
        let other = if edge.start() == vertex {
            edge.end()
        } else if edge.end() == vertex {
            edge.start()
        } else {
            continue;
        };
        let run = topo.vertex(other)?.point() - at;
        if run.normalize()?.cross(dir).length() < JOIN_TOL && run.dot(dir) > 0.0 {
            return Ok(Some((oe.edge(), run.length())));
        }
    }
    Ok(None)
}

/// The one way the wire uses `edge`.
fn wire_flag(topo: &Topology, face: FaceId, edge: EdgeId) -> Result<Option<bool>, OperationsError> {
    let data = topo.face(face)?;
    for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            if oe.edge() == edge {
                return Ok(Some(oe.is_forward()));
            }
        }
    }
    Ok(None)
}

/// Read one connected group of selected edges off the solid as a chain, or
/// `None` when it is not one this path can build exactly.
fn plan_chain(
    topo: &Topology,
    faces: &[FaceId],
    group: &[EdgeId],
    d: f64,
) -> Result<Option<Plan>, OperationsError> {
    let adjacency = AdjacencyIndex::build_from_faces(topo, faces)?;

    // The cap is a plane every edge of the chain lies on.
    let mut caps: Option<Vec<FaceId>> = None;
    for &edge in group {
        let mut around: Vec<FaceId> = adjacency.faces_for_edge(edge).to_vec();
        around.dedup();
        if around.len() != 2 {
            return Ok(None);
        }
        let mut planes = Vec::with_capacity(2);
        for &f in &around {
            if plane_normal(topo, f)?.is_some() {
                planes.push(f);
            }
        }
        caps = Some(match caps {
            None => planes,
            Some(kept) => kept.into_iter().filter(|f| planes.contains(f)).collect(),
        });
    }
    // A line lies on two planes and either may be its cap: a shell's face order
    // has no say in which one the chain fits.
    for cap in caps.unwrap_or_default() {
        if let Some(plan) = plan_on_cap(topo, &adjacency, group, cap, d)? {
            return Ok(Some(plan));
        }
    }
    Ok(None)
}

/// [`plan_chain`] with `cap` as the plane the chain lies on.
#[allow(clippy::too_many_lines)]
fn plan_on_cap(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    group: &[EdgeId],
    cap: FaceId,
    d: f64,
) -> Result<Option<Plan>, OperationsError> {
    let tol = Tolerance::new().linear;
    let Some(cap_normal) = plane_normal(topo, cap)? else {
        return Ok(None);
    };
    let step = d * 1e-3;

    let mut pieces = Vec::with_capacity(group.len());
    for &edge in group {
        let data = topo.edge(edge)?;
        let (start, end) = (data.start(), data.end());
        let start_point = topo.vertex(start)?.point();
        let end_point = topo.vertex(end)?.point();
        let curve = data.curve().clone();
        let Some(&wall) = adjacency.faces_for_edge(edge).iter().find(|&&f| f != cap) else {
            return Ok(None);
        };
        let wall_side = match &curve {
            EdgeCurve::Line => {
                let Some(normal) = plane_normal(topo, wall)? else {
                    return Ok(None);
                };
                let tangent = (end_point - start_point).normalize()?;
                let m = start_point + (end_point - start_point) * 0.5;
                let across = normal.cross(tangent);
                let Some(sign) = side_of_plane_face(topo, wall, normal, m, across, step)? else {
                    return Ok(None);
                };
                WallSide::Plane { normal, sign }
            }
            EdgeCurve::Circle(circle) => {
                if start == end {
                    return Ok(None);
                }
                let Some(cylinder) = cylinder_of(topo, wall)? else {
                    return Ok(None);
                };
                let axis = cylinder.axis();
                let off = circle.center() - cylinder.origin();
                let coaxial = circle.normal().cross(axis).length() < JOIN_TOL
                    && cap_normal.cross(axis).length() < JOIN_TOL
                    && off.cross(axis).length() < JOIN_TOL
                    && (circle.radius() - cylinder.radius()).abs() < JOIN_TOL;
                if !coaxial {
                    return Ok(None);
                }
                // Toward the wall's far side along its axis.
                let along = |p: Point3| (p - cylinder.origin()).dot(axis);
                let rim = along(start_point);
                let mut far = rim;
                for oe in topo.wire(topo.face(wall)?.outer_wire())?.edges() {
                    let e = topo.edge(oe.edge())?;
                    for v in [e.start(), e.end()] {
                        let height = along(topo.vertex(v)?.point());
                        if (height - rim).abs() > (far - rim).abs() {
                            far = height;
                        }
                    }
                }
                if (far - rim).abs() <= tol {
                    return Ok(None);
                }
                WallSide::Axis(axis * (far - rim).signum())
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
        };
        // Which way the cap's material lies, read at the edge's middle.
        let mut piece = Piece {
            edge,
            wall,
            start,
            end,
            curve,
            start_point,
            end_point,
            cap_sign: 1.0,
            wall_side,
        };
        let m = piece.midpoint();
        let across = cap_normal.cross(piece.tangent(m)?);
        let Some(sign) = side_of_plane_face(topo, cap, cap_normal, m, across, step)? else {
            return Ok(None);
        };
        piece.cap_sign = sign;
        pieces.push(piece);
    }

    // The section at each vertex, on which every edge there must agree.
    let mut sections: Vec<(VertexId, Section)> = Vec::new();
    let mut uses: Vec<(VertexId, Vec<usize>)> = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        for vertex in [piece.start, piece.end] {
            let p = piece.at(vertex);
            let cap_dir = cap_normal.cross(piece.tangent(p)?) * piece.cap_sign;
            let wall_dir = piece.wall_dir(p)?;
            let section = Section {
                cap: p + cap_dir * d,
                wall: p + wall_dir * d,
                cap_dir,
                wall_dir,
            };
            if let Some(&(_, seen)) = sections.iter().find(|(v, _)| *v == vertex) {
                if (seen.cap - section.cap).length() > JOIN_TOL
                    || (seen.wall - section.wall).length() > JOIN_TOL
                {
                    return Ok(None);
                }
            } else {
                sections.push((vertex, section));
            }
            match uses.iter_mut().find(|(v, _)| *v == vertex) {
                Some((_, at)) => at.push(index),
                None => uses.push((vertex, vec![index])),
            }
        }
    }

    // What the chain does at each vertex: join two edges or end at a tangent
    // neighbour.
    let mut joints = Vec::new();
    let mut ends = Vec::new();
    let mut curved = pieces
        .iter()
        .any(|p| matches!(p.curve, EdgeCurve::Circle(_)));
    for (vertex, at) in &uses {
        let vertex = *vertex;
        let Some(section) = sections.iter().find(|(v, _)| *v == vertex).map(|s| s.1) else {
            return Ok(None);
        };
        match at[..] {
            [a, b] => {
                let (first, second) = (&pieces[a], &pieces[b]);
                let through = first.leaving(vertex)? + second.leaving(vertex)?;
                if through.length() > JOIN_TOL {
                    return Ok(None);
                }
                let ruling = ruling_along(topo, first.wall, vertex, section.wall_dir)?;
                if let Some((_, length)) = ruling
                    && length <= d + tol
                {
                    return Ok(None);
                }
                if first.wall == second.wall {
                    // A seam between two pieces of one wall's rim shortens in
                    // both its uses; pieces of a flat wall's edge have none.
                    if let Some((ruling, _)) = ruling {
                        joints.push(Joint {
                            vertex,
                            ruling,
                            walls: [first.wall, first.wall],
                        });
                    }
                } else {
                    let Some((ruling, _)) = ruling else {
                        return Ok(None);
                    };
                    if !adjacency.faces_for_edge(ruling).contains(&second.wall) {
                        return Ok(None);
                    }
                    joints.push(Joint {
                        vertex,
                        ruling,
                        walls: [first.wall, second.wall],
                    });
                }
            }
            [a] => {
                let piece = &pieces[a];
                // The cap's next edge carries straight on from this one.
                let leaving = piece.leaving(vertex)?;
                let cap_data = topo.face(cap)?;
                let mut next = None;
                for wid in std::iter::once(cap_data.outer_wire())
                    .chain(cap_data.inner_wires().iter().copied())
                {
                    for oe in topo.wire(wid)?.edges() {
                        let e = topo.edge(oe.edge())?;
                        if oe.edge() != piece.edge && (e.start() == vertex || e.end() == vertex) {
                            next = Some(oe.edge());
                        }
                    }
                }
                let Some(next) = next else {
                    return Ok(None);
                };
                let next_data = topo.edge(next)?;
                let (na, nb) = (
                    topo.vertex(next_data.start())?.point(),
                    topo.vertex(next_data.end())?.point(),
                );
                let next_tangent = match next_data.curve() {
                    EdgeCurve::Line => (nb - na).normalize()?,
                    EdgeCurve::Circle(c) => c.tangent(c.project(piece.at(vertex))).normalize()?,
                    EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
                };
                let away = if next_data.start() == vertex {
                    next_tangent
                } else {
                    -next_tangent
                };
                if (away - leaving).length() > JOIN_TOL {
                    return Ok(None);
                }
                let Some(&beyond) = adjacency.faces_for_edge(next).iter().find(|&&f| f != cap)
                else {
                    return Ok(None);
                };
                let Some((ruling, length)) =
                    ruling_along(topo, piece.wall, vertex, section.wall_dir)?
                else {
                    return Ok(None);
                };
                if length <= d + tol {
                    return Ok(None);
                }
                if beyond == piece.wall || !adjacency.faces_for_edge(ruling).contains(&beyond) {
                    return Ok(None);
                }
                if plane_normal(topo, beyond)?.is_none() {
                    curved = true;
                }
                ends.push(End {
                    vertex,
                    ruling,
                    wall: piece.wall,
                    beyond,
                });
            }
            _ => return Ok(None),
        }
    }

    // The solid's change: the triangle's area along its centroid's path.
    let cap_out = if topo.face(cap)?.is_reversed() {
        -cap_normal
    } else {
        cap_normal
    };
    let mut volume_change = 0.0;
    for piece in &pieces {
        let Some(section) = sections
            .iter()
            .find(|(v, _)| *v == piece.start)
            .map(|s| s.1)
        else {
            return Ok(None);
        };
        let area = 0.5 * section.cap_dir.cross(section.wall_dir).length() * d * d;
        let path = match &piece.curve {
            EdgeCurve::Line => (piece.end_point - piece.start_point).length(),
            EdgeCurve::Circle(c) => {
                let radial = (piece.start_point - c.center()).normalize()?;
                let toward = section.cap_dir.dot(radial);
                // A cap point past the arc's axis has no circle to run on.
                if c.radius() + toward * d <= tol {
                    return Ok(None);
                }
                let (t0, t1) = piece
                    .curve
                    .domain_with_endpoints(piece.start_point, piece.end_point);
                (t1 - t0) * (c.radius() + toward * d / 3.0)
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
        };
        // A wall running under the cap's outward side is a convex edge, whose
        // triangle is material; above it, the triangle fills a corner.
        let sign = if section.wall_dir.dot(cap_out) < 0.0 {
            -1.0
        } else {
            1.0
        };
        volume_change += sign * area * path;
    }

    let plan = Plan {
        cap,
        pieces,
        sections,
        joints,
        ends,
        volume_change,
        curved,
    };

    // The chain's new boundary on the cap must meet nothing else there: not
    // an edge it crosses, which an exact test finds wherever it happens, nor
    // one lying wholly in the strip, which crosses nothing. A cross edge starts
    // on the neighbouring edge it is allowed to touch.
    let mut new_spans: Vec<(Span, Option<Point3>)> = Vec::new();
    for piece in &plan.pieces {
        let (Some(a), Some(b)) = (plan.section(piece.start), plan.section(piece.end)) else {
            return Ok(None);
        };
        let span = match &piece.curve {
            EdgeCurve::Line => Span::Segment(a.cap, b.cap),
            EdgeCurve::Circle(c) => {
                let (t0, t1) = piece
                    .curve
                    .domain_with_endpoints(piece.start_point, piece.end_point);
                let radial = (piece.start_point - c.center()).normalize()?;
                let radius = c.radius() + a.cap_dir.dot(radial) * d;
                Span::Arc {
                    circle: Circle3D::with_axes(
                        c.center(),
                        c.normal(),
                        radius,
                        c.u_axis(),
                        c.v_axis(),
                    )?,
                    t0,
                    sweep: t1 - t0,
                }
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
        };
        new_spans.push((span, None));
    }
    for end in &plan.ends {
        let Some(section) = plan.section(end.vertex) else {
            return Ok(None);
        };
        let at = topo.vertex(end.vertex)?.point();
        new_spans.push((Span::Segment(at, section.cap), Some(at)));
    }
    // Nor may it meet itself away from where its pieces join: across a narrow
    // neck, the two sides' new edges cross.
    for (i, (first, _)) in new_spans.iter().enumerate() {
        for (second, _) in &new_spans[i + 1..] {
            for p in first.touches(second, tol) {
                let joins = |span: &Span| {
                    span.ends()
                        .iter()
                        .any(|&end| (p - end).length() <= 10.0 * tol)
                };
                if !(joins(first) && joins(second)) {
                    return Ok(None);
                }
            }
        }
    }
    let cap_data = topo.face(cap)?;
    for wid in std::iter::once(cap_data.outer_wire()).chain(cap_data.inner_wires().iter().copied())
    {
        for oe in topo.wire(wid)?.edges() {
            if plan.pieces.iter().any(|p| p.edge == oe.edge()) {
                continue;
            }
            let edge = topo.edge(oe.edge())?;
            let a = topo.vertex(edge.start())?.point();
            let b = topo.vertex(edge.end())?.point();
            let Some(span) = Span::of(edge.curve(), a, b) else {
                return Ok(None);
            };
            for (new_span, allowed) in &new_spans {
                for p in new_span.touches(&span, tol) {
                    if allowed.is_none_or(|at| (p - at).length() > 10.0 * tol) {
                        return Ok(None);
                    }
                }
            }
            for p in edge_samples(edge.curve(), a, b) {
                if plan.strip_holds(p, d, tol)? {
                    return Ok(None);
                }
            }
        }
    }

    // Nor on a wall, whose rim rises by `d` and which keeps every other edge
    // and hole where it was. A line stays clear if its ends do, and a circle
    // on the wall's cylinder runs at one height.
    for piece in &plan.pieces {
        let Some(section) = plan.section(piece.start) else {
            return Ok(None);
        };
        let wall = topo.face(piece.wall)?;
        if !wall.inner_wires().is_empty() {
            return Ok(None);
        }
        for oe in topo.wire(wall.outer_wire())?.edges() {
            let rim_or_ruling = plan.pieces.iter().any(|p| p.edge == oe.edge())
                || plan.joints.iter().any(|j| j.ruling == oe.edge())
                || plan.ends.iter().any(|e| e.ruling == oe.edge());
            if rim_or_ruling {
                continue;
            }
            let edge = topo.edge(oe.edge())?;
            match edge.curve() {
                EdgeCurve::Line => {}
                EdgeCurve::Circle(c) => {
                    if c.normal().cross(section.wall_dir).length() > JOIN_TOL {
                        return Ok(None);
                    }
                }
                EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
            }
            for v in [edge.start(), edge.end()] {
                let height = (topo.vertex(v)?.point() - piece.start_point).dot(section.wall_dir);
                if height <= d + tol {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(plan))
}

/// The selected edges, grouped by the vertices they share, each group and the
/// groups' order following the edges' ids, so a result does not depend on the
/// order they were listed in.
fn connected_groups(
    topo: &Topology,
    edges: &[EdgeId],
) -> Result<Vec<Vec<EdgeId>>, OperationsError> {
    let mut sorted = edges.to_vec();
    sorted.sort_by_key(|e| e.index());
    sorted.dedup();
    let mut groups: Vec<Vec<EdgeId>> = Vec::new();
    let mut ends_of: Vec<Vec<VertexId>> = Vec::new();
    for edge in sorted {
        let data = topo.edge(edge)?;
        let ends = [data.start(), data.end()];
        let mut merged = vec![edge];
        let mut merged_ends = ends.to_vec();
        let mut k = 0;
        while k < groups.len() {
            if ends_of[k].iter().any(|v| merged_ends.contains(v)) {
                merged.extend(groups.remove(k));
                merged_ends.extend(ends_of.remove(k));
            } else {
                k += 1;
            }
        }
        merged.sort_by_key(|e| e.index());
        groups.push(merged);
        ends_of.push(merged_ends);
    }
    groups.sort_by_key(|g| g.first().map_or(0, |e| e.index()));
    Ok(groups)
}

/// Each edge of `original` paired with its image in `copy`, which lists its
/// faces, wires and edges in the same order.
fn edge_images(
    topo: &Topology,
    original: SolidId,
    copy: SolidId,
) -> Result<Option<HashMap<EdgeId, EdgeId>>, OperationsError> {
    let from = brepkit_topology::explorer::solid_faces(topo, original)?;
    let to = brepkit_topology::explorer::solid_faces(topo, copy)?;
    if from.len() != to.len() {
        return Ok(None);
    }
    let mut images = HashMap::new();
    for (&a, &b) in from.iter().zip(&to) {
        let (fa, fb) = (topo.face(a)?, topo.face(b)?);
        let wires_a: Vec<WireId> = std::iter::once(fa.outer_wire())
            .chain(fa.inner_wires().iter().copied())
            .collect();
        let wires_b: Vec<WireId> = std::iter::once(fb.outer_wire())
            .chain(fb.inner_wires().iter().copied())
            .collect();
        if wires_a.len() != wires_b.len() {
            return Ok(None);
        }
        for (&wa, &wb) in wires_a.iter().zip(&wires_b) {
            let (ea, eb) = (topo.wire(wa)?.edges(), topo.wire(wb)?.edges());
            if ea.len() != eb.len() {
                return Ok(None);
            }
            for (x, y) in ea.iter().zip(eb) {
                images.insert(x.edge(), y.edge());
            }
        }
    }
    Ok(Some(images))
}

/// An edge's replacement in one face: the pieces running from its stored start
/// to its stored end, each with the flag it takes on that run.
type Run = Vec<(EdgeId, bool)>;

/// `wire_id` with each listed edge replaced by its run, read backward where
/// the wire traverses the edge backward.
fn rewrite_wire(
    topo: &mut Topology,
    wire_id: WireId,
    swaps: &HashMap<EdgeId, Run>,
) -> Result<WireId, OperationsError> {
    let wire = topo.wire(wire_id)?;
    let closed = wire.is_closed();
    let mut edges = Vec::with_capacity(wire.edges().len());
    for oe in wire.edges() {
        match swaps.get(&oe.edge()) {
            None => edges.push(OrientedEdge::new(oe.edge(), oe.is_forward())),
            Some(run) if oe.is_forward() => {
                edges.extend(run.iter().map(|&(e, f)| OrientedEdge::new(e, f)));
            }
            Some(run) => {
                edges.extend(run.iter().rev().map(|&(e, f)| OrientedEdge::new(e, !f)));
            }
        }
    }
    Ok(topo.add_wire(Wire::new(edges, closed)?))
}

/// The normal of a planar loop through `corners`, wound counter-clockwise
/// about it.
fn loop_normal(corners: &[Point3]) -> Result<Vec3, OperationsError> {
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    for (i, &p) in corners.iter().enumerate() {
        let q = corners[(i + 1) % corners.len()];
        sum += (p - corners[0]).cross(q - corners[0]);
    }
    Ok(sum.normalize()?)
}

fn plane_through(normal: Vec3, p: Point3) -> FaceSurface {
    FaceSurface::Plane {
        normal,
        d: normal.dot(Vec3::new(p.x(), p.y(), p.z())),
    }
}

/// Build one planned chain into the solid it was read from, whose cap, walls
/// and neighbours are rewritten in place. Returns the new faces, or `None`
/// when the shell's faces do not agree on the edges they share.
#[allow(clippy::too_many_lines)]
fn build_chain(
    topo: &mut Topology,
    plan: &Plan,
    d: f64,
) -> Result<Option<Vec<FaceId>>, OperationsError> {
    let tol = Tolerance::new().linear;
    let mut cap_vertex: HashMap<VertexId, VertexId> = HashMap::new();
    let mut wall_vertex: HashMap<VertexId, VertexId> = HashMap::new();
    let mut across: HashMap<VertexId, EdgeId> = HashMap::new();
    for &(vertex, section) in &plan.sections {
        let c = topo.add_vertex(Vertex::new(section.cap, tol));
        let w = topo.add_vertex(Vertex::new(section.wall, tol));
        cap_vertex.insert(vertex, c);
        wall_vertex.insert(vertex, w);
        across.insert(vertex, topo.add_edge(Edge::new(c, w, EdgeCurve::Line)));
    }

    // Each rewritten face's replacements, its id first so faces rewrite in a
    // fixed order.
    let mut swaps: BTreeMap<usize, (FaceId, HashMap<EdgeId, Run>)> = BTreeMap::new();
    let mut swap = |face: FaceId, edge: EdgeId, run: Run| {
        swaps
            .entry(face.index())
            .or_insert_with(|| (face, HashMap::new()))
            .1
            .insert(edge, run);
    };

    // The chain's free ends: the cap steps in along a cross edge, the wall's
    // ruling keeps its far part, and the neighbour takes both parts.
    let mut crossing: HashMap<VertexId, EdgeId> = HashMap::new();
    let mut lower: HashMap<VertexId, (EdgeId, bool)> = HashMap::new();
    for end in &plan.ends {
        let c = cap_vertex[&end.vertex];
        let w = wall_vertex[&end.vertex];
        crossing.insert(
            end.vertex,
            topo.add_edge(Edge::new(end.vertex, c, EdgeCurve::Line)),
        );
        let ruling = topo.edge(end.ruling)?;
        let (start, stop) = (ruling.start(), ruling.end());
        let from_vertex = start == end.vertex;
        let first = topo.add_edge(Edge::new(start, w, EdgeCurve::Line));
        let second = topo.add_edge(Edge::new(w, stop, EdgeCurve::Line));
        let (near, far) = if from_vertex {
            (first, second)
        } else {
            (second, first)
        };
        lower.insert(end.vertex, (near, from_vertex));
        swap(end.wall, end.ruling, vec![(far, true)]);
        swap(end.beyond, end.ruling, vec![(first, true), (second, true)]);
    }

    // Between two walls, the ruling shortens for both.
    for joint in &plan.joints {
        let w = wall_vertex[&joint.vertex];
        let ruling = topo.edge(joint.ruling)?;
        let (start, stop) = (ruling.start(), ruling.end());
        let shorter = if start == joint.vertex {
            Edge::new(w, stop, EdgeCurve::Line)
        } else {
            Edge::new(start, w, EdgeCurve::Line)
        };
        let shorter = topo.add_edge(shorter);
        for wall in joint.walls {
            swap(wall, joint.ruling, vec![(shorter, true)]);
        }
    }

    // Each edge moves into the cap and along its wall, and a plane or a cone
    // takes the band between.
    let mut new_faces = Vec::with_capacity(plan.pieces.len() + plan.ends.len());
    let mut across_uses: HashMap<VertexId, bool> = HashMap::new();
    let cap_reversed = topo.face(plan.cap)?.is_reversed();
    let Some(cap_normal) = plane_normal(topo, plan.cap)? else {
        return Ok(None);
    };
    let cap_out = if cap_reversed {
        -cap_normal
    } else {
        cap_normal
    };
    for piece in &plan.pieces {
        let (Some(a), Some(b)) = (plan.section(piece.start), plan.section(piece.end)) else {
            return Ok(None);
        };
        let (Some(cap_flag), Some(wall_flag)) = (
            wire_flag(topo, plan.cap, piece.edge)?,
            wire_flag(topo, piece.wall, piece.edge)?,
        ) else {
            return Ok(None);
        };
        let wall_reversed = topo.face(piece.wall)?.is_reversed();
        let cap_use = cap_flag != cap_reversed;
        let wall_use = wall_flag != wall_reversed;
        if cap_use == wall_use {
            return Ok(None);
        }

        let (cap_curve, wall_curve, band) = match &piece.curve {
            EdgeCurve::Line => (EdgeCurve::Line, EdgeCurve::Line, None),
            EdgeCurve::Circle(c) => {
                let radial = (piece.start_point - c.center()).normalize()?;
                let radius = c.radius() + a.cap_dir.dot(radial) * d;
                let wall_center = c.center() + a.wall_dir * d;
                let cap_circle =
                    Circle3D::with_axes(c.center(), c.normal(), radius, c.u_axis(), c.v_axis())?;
                let wall_circle = Circle3D::with_axes(
                    wall_center,
                    c.normal(),
                    c.radius(),
                    c.u_axis(),
                    c.v_axis(),
                )?;
                // The cone through both circles.
                let (r1, r2) = (radius, c.radius());
                let t0 = -r1 / (r2 - r1);
                let apex = c.center() + (wall_center - c.center()) * t0;
                let toward =
                    ((c.center() + (wall_center - c.center()) * 0.5) - apex).normalize()?;
                let height = (c.center() - apex).dot(toward).abs();
                let cone = ConicalSurface::new(apex, toward, height.atan2(r1))?;
                let wall_normal = {
                    let Some(cylinder) = cylinder_of(topo, piece.wall)? else {
                        return Ok(None);
                    };
                    let (u, v) = cylinder.project_point(piece.start_point);
                    cylinder.normal(u, v)
                };
                let wall_out = if wall_reversed {
                    -wall_normal
                } else {
                    wall_normal
                };
                let mid = a.cap + (a.wall - a.cap) * 0.5;
                let (u, v) = cone.project_point(mid);
                let reversed = cone.normal(u, v).dot(cap_out + wall_out) < 0.0;
                (
                    EdgeCurve::Circle(cap_circle),
                    EdgeCurve::Circle(wall_circle),
                    Some((cone, reversed)),
                )
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
        };
        let new_cap = topo.add_edge(Edge::new(
            cap_vertex[&piece.start],
            cap_vertex[&piece.end],
            cap_curve,
        ));
        let new_wall = topo.add_edge(Edge::new(
            wall_vertex[&piece.start],
            wall_vertex[&piece.end],
            wall_curve,
        ));

        let mut cap_run: Run = Vec::with_capacity(3);
        if let Some(&cross) = crossing.get(&piece.start) {
            cap_run.push((cross, true));
        }
        cap_run.push((new_cap, true));
        if let Some(&cross) = crossing.get(&piece.end) {
            cap_run.push((cross, false));
        }
        swap(plan.cap, piece.edge, cap_run);
        swap(piece.wall, piece.edge, vec![(new_wall, true)]);

        // The band runs its new cap edge against the cap's use of it.
        let reversed = band.as_ref().is_some_and(|&(_, r)| r);
        let forward = cap_use == reversed;
        let (start_across, end_across) = (across[&piece.start], across[&piece.end]);
        let (uses, corners) = if forward {
            (
                vec![
                    (new_cap, true),
                    (end_across, true),
                    (new_wall, false),
                    (start_across, false),
                ],
                [a.cap, b.cap, b.wall, a.wall],
            )
        } else {
            (
                vec![
                    (start_across, true),
                    (new_wall, true),
                    (end_across, false),
                    (new_cap, false),
                ],
                [a.cap, a.wall, b.wall, b.cap],
            )
        };
        across_uses.insert(piece.start, forward == reversed);
        across_uses.insert(piece.end, forward != reversed);
        let wire = topo.add_wire(Wire::new(
            uses.iter().map(|&(e, f)| OrientedEdge::new(e, f)).collect(),
            true,
        )?);
        let face = if let Some((cone, _)) = band {
            let mut face = Face::new(wire, Vec::new(), FaceSurface::Cone(cone));
            face.set_reversed(reversed);
            face
        } else {
            Face::new(
                wire,
                Vec::new(),
                plane_through(loop_normal(&corners)?, a.cap),
            )
        };
        new_faces.push(topo.add_face(face));
    }

    // Each free end closes in a flat triangle, running every edge against the
    // face beside it.
    for end in &plan.ends {
        let Some(section) = plan.section(end.vertex) else {
            return Ok(None);
        };
        let at = topo.vertex(end.vertex)?.point();
        let Some(&(near, from_vertex)) = lower.get(&end.vertex) else {
            return Ok(None);
        };
        let across_edge = across[&end.vertex];
        let cross = crossing[&end.vertex];
        let Some(&band_use) = across_uses.get(&end.vertex) else {
            return Ok(None);
        };
        let (uses, corners) = if band_use {
            // The band runs cap to wall, so this runs wall to cap.
            (
                vec![(across_edge, false), (cross, false), (near, from_vertex)],
                [section.wall, section.cap, at],
            )
        } else {
            (
                vec![(across_edge, true), (near, !from_vertex), (cross, true)],
                [section.cap, section.wall, at],
            )
        };
        let wire = topo.add_wire(Wire::new(
            uses.iter().map(|&(e, f)| OrientedEdge::new(e, f)).collect(),
            true,
        )?);
        let surface = plane_through(loop_normal(&corners)?, at);
        new_faces.push(topo.add_face(Face::new(wire, Vec::new(), surface)));
    }

    for (face, face_swaps) in swaps.into_values() {
        // An edge this face no longer uses keeps no trim on it.
        for &edge in face_swaps.keys() {
            topo.pcurves_mut().remove(edge, face);
        }
        let data = topo.face(face)?;
        let (outer, inners) = (data.outer_wire(), data.inner_wires().to_vec());
        let new_outer = rewrite_wire(topo, outer, &face_swaps)?;
        let mut new_inners = Vec::with_capacity(inners.len());
        for wire in inners {
            new_inners.push(rewrite_wire(topo, wire, &face_swaps)?);
        }
        let data = topo.face_mut(face)?;
        data.set_outer_wire(new_outer);
        *data.inner_wires_mut() = new_inners;
    }
    Ok(Some(new_faces))
}

/// Chamfer `edges` of `solid` at `d` on both faces, when they form tangent
/// chains of lines and arcs this path can build exactly.
///
/// Each chain lies on one flat cap, its lines between planes and its arcs
/// between the cap and a coaxial cylinder, every edge tangent to the next, and
/// any free end stops where the next edge carries straight on. A selection of
/// plain lines that meet no curved neighbour is left to the other engines.
///
/// Returns `None`, leaving `solid` as it was, when the selection is not one
/// of these or the chamfer would not fit: past an arc's axis, a wall's far end
/// or another edge of the cap.
///
/// # Errors
///
/// Returns an error if a topology lookup fails, or the chamfer would break
/// into one of the solid's cavities.
#[allow(clippy::too_many_lines)]
pub fn chamfer_tangent_chains(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    d: f64,
) -> Result<Option<SolidId>, OperationsError> {
    if d <= 0.0 || edges.is_empty() {
        return Ok(None);
    }
    let outer_faces = topo
        .shell(topo.solid(solid)?.outer_shell())?
        .faces()
        .to_vec();
    let groups = connected_groups(topo, edges)?;
    let mut plans = Vec::with_capacity(groups.len());
    for group in &groups {
        let Some(plan) = plan_chain(topo, &outer_faces, group, d)? else {
            return Ok(None);
        };
        plans.push(plan);
    }
    if !plans.iter().any(|p| p.curved) {
        return Ok(None);
    }
    let change: f64 = plans.iter().map(|p| p.volume_change).sum();
    let swept: f64 = plans.iter().map(|p| p.volume_change.abs()).sum();

    let copy = crate::copy::copy_solid(topo, solid)?;
    let Some(images) = edge_images(topo, solid, copy)? else {
        return Ok(None);
    };
    let data = topo.solid(copy)?;
    let (outer, inners) = (data.outer_shell(), data.inner_shells().to_vec());
    // A copy leaves its pcurves behind, and a face this chamfer never touches
    // keeps its trims.
    let source = topo.solid(solid)?;
    let source_shells: Vec<_> = std::iter::once(source.outer_shell())
        .chain(source.inner_shells().iter().copied())
        .collect();
    let copied_shells: Vec<_> = std::iter::once(outer)
        .chain(inners.iter().copied())
        .collect();
    for (&from, &to) in source_shells.iter().zip(&copied_shells) {
        crate::cavities::copy_pcurves(topo, from, to)?;
    }
    let mut faces = topo.shell(outer)?.faces().to_vec();
    let mut made = Vec::new();
    for group in &groups {
        let mut image = Vec::with_capacity(group.len());
        for edge in group {
            let Some(&e) = images.get(edge) else {
                return Ok(None);
            };
            image.push(e);
        }
        // Read after the chains before it, which shorten rulings this one
        // shares and leave new edges on a cap it shares, which it must clear.
        let Some(plan) = plan_chain(topo, &faces, &image, d)? else {
            return Ok(None);
        };
        let Some(new_faces) = build_chain(topo, &plan, d)? else {
            return Ok(None);
        };
        faces.extend_from_slice(&new_faces);
        made.extend(new_faces);
    }
    let outer = topo.add_shell(Shell::new(faces)?);
    let result = topo.add_solid(Solid::new(outer, inners));
    crate::cavities::clear_of_own_cavities(topo, result, &made, "chamfer")?;

    // The closed form decides whether what was built is what was meant.
    let deflection = (d * 0.01).max(1e-4);
    let before = crate::measure::solid_volume(topo, solid, deflection)?;
    let after = crate::measure::solid_volume(topo, result, deflection)?;
    let oriented = crate::measure::oriented_solid_volume(topo, result, deflection)?;
    if oriented <= 0.0 || (after - before - change).abs() > VOLUME_MARGIN * swept {
        log::debug!(
            "chamfer chain: volume went {before} to {after}, a change of {}, expected {change}",
            after - before
        );
        return Ok(None);
    }
    // And a shell whose faces disagree can still hold the right volume.
    if !crate::validate::validate_solid(topo, result)?.is_valid() {
        return Ok(None);
    }
    Ok(Some(result))
}
