//! Chamfers along chains of lines and arcs.
//!
//! A rounded prism's rim is a chain of straight edges between planes and arcs
//! between a flat cap and a coaxial cylinder. The same triangle runs along a
//! chain, so its chamfer can be written down: each edge moves into its cap
//! and along its wall, and a plane (over a line) or a cone (over an arc) joins
//! the two. A straight edge along a ruling of a cylinder, where a cap
//! parallel to the axis meets the cylinder, moves along the wall by turning
//! about the axis, and a plane joins the two.
//!
//! Where two edges meet tangent, their sections agree. Where two edges meet
//! at an angle, with one face in common and the third edge at the vertex left
//! sharp, the two chamfers meet along the curve where their faces cross (a
//! mitre). A chain's free end stops where the next edge carries straight on,
//! and ends in a flat triangle across the edge, or at a plane through the
//! end, square or slanted to the edge, which takes the chamfer's end. Each
//! edge can take one distance into its cap and another along its wall.
//! Nothing here depends on how the solid is posed. The result's volume is
//! checked against the change that each edge's triangle makes along it
//! before it is returned.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::{PI, TAU};

use brepkit_geometry::extrema::segment_segment_distance;
use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::curves::Circle3D;
use brepkit_math::frame::Frame3;
use brepkit_math::nurbs::curve::NurbsCurve;
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

/// How far a mitre's curve may stray from the two surfaces it lies on, read
/// between its ends.
const CURVE_TOL: f64 = 1e-9;

/// One selected edge: a line between two planes, an open arc between a flat
/// cap and a coaxial cylinder, or a line along a ruling of a cylinder where a
/// cap parallel to its axis meets the cylinder.
struct Piece {
    edge: EdgeId,
    cap: FaceId,
    cap_normal: Vec3,
    wall: FaceId,
    start: VertexId,
    end: VertexId,
    curve: EdgeCurve,
    start_point: Point3,
    end_point: Point3,
    /// `cap normal x tangent`, signed to point into the cap's material.
    cap_sign: f64,
    /// Into the wall's material.
    wall_side: WallSide,
    /// How far the chamfer reaches into the cap and along the wall (a chord
    /// round a cylinder beside a ruling).
    cap_d: f64,
    wall_d: f64,
}

#[derive(Clone)]
enum WallSide {
    /// A plane wall: `wall normal x tangent`, signed.
    Plane { normal: Vec3, sign: f64 },
    /// A coaxial cylinder under an arc: along the axis.
    Axis(Vec3),
    /// A cylinder along whose ruling the edge runs: the contact lies the
    /// signed angle `turn` about the axis from the edge.
    Ruling {
        cylinder: CylindricalSurface,
        turn: f64,
    },
}

impl Piece {
    fn at(&self, vertex: VertexId) -> Point3 {
        if vertex == self.start {
            self.start_point
        } else {
            self.end_point
        }
    }

    /// 0 for the edge's start, 1 for its end.
    fn slot(&self, vertex: VertexId) -> usize {
        usize::from(vertex != self.start)
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

    fn midpoint(&self) -> Point3 {
        let (t0, t1) = self
            .curve
            .domain_with_endpoints(self.start_point, self.end_point);
        self.curve
            .evaluate_with_endpoints(0.5 * (t0 + t1), self.start_point, self.end_point)
    }

    /// The farther of the chamfer's two reaches.
    fn reach(&self) -> f64 {
        self.cap_d.max(self.wall_d)
    }

    /// The chamfer's triangle at `p` on the edge.
    fn section(&self, p: Point3) -> Result<Section, OperationsError> {
        let tangent = self.tangent(p)?;
        let cap_dir = self.cap_normal.cross(tangent) * self.cap_sign;
        let wall = match &self.wall_side {
            WallSide::Plane { normal, sign } => p + normal.cross(tangent) * (sign * self.wall_d),
            WallSide::Axis(axis) => p + *axis * self.wall_d,
            WallSide::Ruling { cylinder, turn } => turn_about(cylinder, p, *turn),
        };
        Ok(Section {
            cap: p + cap_dir * self.cap_d,
            wall,
            cap_dir,
            wall_dir: (wall - p).normalize()?,
        })
    }

    /// The curves the contacts run on, on the cap and on the wall, from the
    /// section `at` the edge's point `p`.
    fn tracks(&self, p: Point3, at: &Section) -> Result<[Track; 2], OperationsError> {
        Ok(match &self.curve {
            EdgeCurve::Circle(c) => {
                let radial = (p - c.center()).normalize()?;
                let radius = c.radius() + at.cap_dir.dot(radial) * self.cap_d;
                [
                    Track::Circle(Circle3D::with_axes(
                        c.center(),
                        c.normal(),
                        radius,
                        c.u_axis(),
                        c.v_axis(),
                    )?),
                    Track::Circle(Circle3D::with_axes(
                        c.center() + at.wall_dir * self.wall_d,
                        c.normal(),
                        c.radius(),
                        c.u_axis(),
                        c.v_axis(),
                    )?),
                ]
            }
            EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                let dir = self.tangent(p)?;
                [
                    Track::Line { at: at.cap, dir },
                    Track::Line { at: at.wall, dir },
                ]
            }
        })
    }

    /// The surface of the chamfer face, from its sections at the edge's start
    /// `a` and end `b`.
    fn band(&self, a: &Section, b: &Section) -> Result<Band, OperationsError> {
        Ok(match &self.curve {
            EdgeCurve::Circle(c) => {
                let radial = (self.start_point - c.center()).normalize()?;
                let r1 = c.radius() + a.cap_dir.dot(radial) * self.cap_d;
                let r2 = c.radius();
                let wall_center = c.center() + a.wall_dir * self.wall_d;
                // The cone through the cap's and the wall's circles.
                let t0 = -r1 / (r2 - r1);
                let apex = c.center() + (wall_center - c.center()) * t0;
                let toward =
                    ((c.center() + (wall_center - c.center()) * 0.5) - apex).normalize()?;
                let height = (c.center() - apex).dot(toward).abs();
                Band::Cone(ConicalSurface::new(apex, toward, height.atan2(r1))?)
            }
            EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                let normal = loop_normal(&[a.cap, b.cap, b.wall, a.wall])?;
                Band::Plane {
                    normal,
                    d: normal.dot(to_vec(a.cap)),
                }
            }
        })
    }

    /// The contact's curve for the cap and the wall, between two of its
    /// points.
    fn contact_curves(&self, a: &Section) -> Result<[EdgeCurve; 2], OperationsError> {
        Ok(match &self.curve {
            EdgeCurve::Circle(c) => {
                let radial = (self.start_point - c.center()).normalize()?;
                let radius = c.radius() + a.cap_dir.dot(radial) * self.cap_d;
                [
                    EdgeCurve::Circle(Circle3D::with_axes(
                        c.center(),
                        c.normal(),
                        radius,
                        c.u_axis(),
                        c.v_axis(),
                    )?),
                    EdgeCurve::Circle(Circle3D::with_axes(
                        c.center() + a.wall_dir * self.wall_d,
                        c.normal(),
                        c.radius(),
                        c.u_axis(),
                        c.v_axis(),
                    )?),
                ]
            }
            EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                [EdgeCurve::Line, EdgeCurve::Line]
            }
        })
    }
}

fn to_vec(p: Point3) -> Vec3 {
    Vec3::new(p.x(), p.y(), p.z())
}

/// `p` on `cylinder` turned `angle` about its axis.
fn turn_about(cylinder: &CylindricalSurface, p: Point3, angle: f64) -> Point3 {
    let axis = cylinder.axis();
    let foot = cylinder.origin() + axis * axis.dot(p - cylinder.origin());
    let radial = p - foot;
    let (sin, cos) = angle.sin_cos();
    foot + radial * cos + axis.cross(radial) * sin
}

/// Where the chamfer's triangle stands at one vertex of a chain.
#[derive(Clone, Copy)]
struct Section {
    cap: Point3,
    wall: Point3,
    cap_dir: Vec3,
    wall_dir: Vec3,
}

/// The line or circle a contact runs on.
enum Track {
    Line { at: Point3, dir: Vec3 },
    Circle(Circle3D),
}

impl Track {
    /// Where the track meets the line through `at` along `dir`, nearest
    /// `near`.
    fn meets_line(&self, at: Point3, dir: Vec3, near: Point3, tol: f64) -> Option<Point3> {
        match self {
            Self::Line { at: a, dir: u } => line_line(*a, *u, at, dir, tol),
            Self::Circle(c) => circle_line(c, at, dir, near, tol),
        }
    }

    /// Where two tracks cross, nearest `near`.
    fn meets(&self, other: &Self, near: Point3, tol: f64) -> Option<Point3> {
        match (self, other) {
            (_, Self::Line { at, dir }) => self.meets_line(*at, *dir, near, tol),
            (Self::Line { at, dir }, Self::Circle(_)) => other.meets_line(*at, *dir, near, tol),
            (Self::Circle(a), Self::Circle(b)) => a
                .intersect_circle(b, tol)
                .into_iter()
                .map(|(p, _)| p)
                .min_by(|p, q| (*p - near).length().total_cmp(&(*q - near).length())),
        }
    }
}

/// Where two lines cross, `None` when they are parallel or pass more than
/// `tol` apart.
fn line_line(a: Point3, u: Vec3, b: Point3, v: Vec3, tol: f64) -> Option<Point3> {
    let w = a - b;
    let (uu, uv, vv) = (u.dot(u), u.dot(v), v.dot(v));
    let (uw, vw) = (u.dot(w), v.dot(w));
    let den = uu * vv - uv * uv;
    if den <= 1e-12 * uu * vv {
        return None;
    }
    let s = (uv * vw - vv * uw) / den;
    let t = (uu * vw - uv * uw) / den;
    let (p, q) = (a + u * s, b + v * t);
    ((p - q).length() <= tol).then(|| p + (q - p) * 0.5)
}

/// Where the line through `at` along `dir` meets the circle `c`, nearest
/// `near`.
fn circle_line(c: &Circle3D, at: Point3, dir: Vec3, near: Point3, tol: f64) -> Option<Point3> {
    let dir = dir.normalize().ok()?;
    let n = c.normal();
    let w = at - c.center();
    let across = dir.dot(n);
    if across.abs() > 1e-9 {
        let p = at + dir * (-w.dot(n) / across);
        return (((p - c.center()).length() - c.radius()).abs() <= tol).then_some(p);
    }
    if w.dot(n).abs() > tol {
        return None;
    }
    let b = w.dot(dir);
    let disc = b * b - (w.dot(w) - c.radius() * c.radius());
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    [-b - root, -b + root]
        .into_iter()
        .map(|t| at + dir * t)
        .min_by(|p, q| (*p - near).length().total_cmp(&(*q - near).length()))
}

/// A chamfer face's surface.
#[derive(Clone)]
enum Band {
    Plane { normal: Vec3, d: f64 },
    Cone(ConicalSurface),
}

impl Band {
    /// Zero on the surface, of opposite signs on its two sides.
    fn level(&self, p: Point3) -> f64 {
        match self {
            Self::Plane { normal, d } => normal.dot(to_vec(p)) - d,
            Self::Cone(cone) => {
                let off = p - cone.apex();
                let h = cone.axis().dot(off);
                let rho = (off - cone.axis() * h).length();
                let (sin, cos) = cone.half_angle().sin_cos();
                rho * sin - h * cos
            }
        }
    }

    fn normal_at(&self, p: Point3) -> Vec3 {
        match self {
            Self::Plane { normal, .. } => *normal,
            Self::Cone(cone) => {
                let (u, v) = cone.project_point(p);
                cone.normal(u, v)
            }
        }
    }
}

/// Where `f` changes sign between `0` and `1`, halving the bracket until it
/// stops shrinking. `None` when it does not change sign.
fn root_between(f: impl Fn(f64) -> f64) -> Option<f64> {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let (f_lo, f_hi) = (f(lo), f(hi));
    if f_lo == 0.0 {
        return Some(lo);
    }
    if f_hi == 0.0 {
        return Some(hi);
    }
    if f_lo.signum() == f_hi.signum() {
        return None;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        let f_mid = f(mid);
        if f_mid == 0.0 {
            return Some(mid);
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// The arc from `a` to `b` of the conic where the plane of normal `normal`
/// through them meets `cone`, as a rational quadratic: its middle control
/// point where the arc's end tangents cross, and its weight from where the
/// arc crosses the line from the chord's middle to that point. `None` when
/// the tangents do not cross ahead of both ends, or the curve strays from
/// either surface.
fn conic_arc(normal: Vec3, cone: &ConicalSurface, a: Point3, b: Point3) -> Option<EdgeCurve> {
    let band = Band::Cone(cone.clone());
    let chord = b - a;
    let length = chord.length();
    if length <= JOIN_TOL {
        return None;
    }
    let aim = |t: Vec3| -> Option<Vec3> {
        let t = t.normalize().ok()?;
        Some(if t.dot(chord) < 0.0 { -t } else { t })
    };
    let ta = aim(normal.cross(band.normal_at(a)))?;
    let tb = aim(normal.cross(band.normal_at(b)))?;
    let unit = chord * (1.0 / length);
    if ta.cross(unit).length() <= 1e-12 && tb.cross(unit).length() <= 1e-12 {
        let mid = a + chord * 0.5;
        return (band.level(mid).abs() <= CURVE_TOL).then_some(EdgeCurve::Line);
    }
    // a + s ta = b - r tb, both ahead.
    let (aa, ab, bb) = (ta.dot(ta), ta.dot(tb), tb.dot(tb));
    let (ac, bc) = (ta.dot(chord), tb.dot(chord));
    let den = aa * bb - ab * ab;
    if den <= 1e-18 {
        return None;
    }
    let s = (ac * bb - ab * bc) / den;
    let r = (aa * bc - ab * ac) / den;
    if s <= 0.0 || r <= 0.0 {
        return None;
    }
    let control = a + ta * s;
    let mid = a + chord * 0.5;
    let lambda = root_between(|t| band.level(mid + (control - mid) * t))?;
    if lambda <= 0.0 || lambda >= 1.0 {
        return None;
    }
    let weight = lambda / (1.0 - lambda);
    let curve = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![a, control, b],
        vec![1.0, weight, 1.0],
    )
    .ok()?;
    for k in 1..8 {
        let p = curve.evaluate(f64::from(k) / 8.0);
        if band.level(p).abs() > CURVE_TOL || normal.dot(p - a).abs() > CURVE_TOL {
            return None;
        }
    }
    Some(EdgeCurve::NurbsCurve(curve))
}

/// The curve where two chamfer faces, or a chamfer face and a plane, cross
/// between `a` and `b`: a line where both are planes, a conic where one is a
/// cone. `None` for two cones.
fn crossing_curve(first: &Band, second: &Band, a: Point3, b: Point3) -> Option<EdgeCurve> {
    match (first, second) {
        (Band::Plane { .. }, Band::Plane { .. }) => Some(EdgeCurve::Line),
        (Band::Plane { normal, .. }, Band::Cone(cone))
        | (Band::Cone(cone), Band::Plane { normal, .. }) => conic_arc(*normal, cone, a, b),
        (Band::Cone(_), Band::Cone(_)) => None,
    }
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

/// A chain's free end at a plane through its vertex that holds the plane's
/// edges along the cap and along the wall. The contacts end where they reach
/// those edges, and the plane takes the curve where the chamfer face crosses
/// it between the two points: the end's section, where the plane stands
/// square to the edge, and a line or a conic where it slants. Each edge runs
/// on from the vertex through its point, which splits it, or, at a square
/// end, the other way, so that the plane takes the stretch from the vertex
/// to the point as well (an end at an inside corner).
struct PlaneEnd {
    vertex: VertexId,
    piece: usize,
    face: FaceId,
    /// The plane's edges along the cap and along the wall.
    edges: [EdgeId; 2],
    /// Whether each point splits its edge.
    split: [bool; 2],
    /// Where the contacts end, on the cap and on the wall.
    points: [Point3; 2],
    /// From the cap's point to the wall's, between the chamfer face and the
    /// plane.
    cross: EdgeCurve,
    /// Whether the plane slants to the edge.
    slanted: bool,
}

/// Two edges meeting at an angle, with `shared` the face along both and
/// `third` the sharp edge between their other two faces. Each edge's contact
/// on `shared` runs to where the two cross (`meet`); the other contacts run
/// to `third`, which shortens to `reach`, the farther of the two. The nearer
/// one ends where the farther one's chamfer face crosses it (`cut`), which
/// leaves a sliver of its face to that chamfer.
struct Corner {
    vertex: VertexId,
    shared: FaceId,
    third: EdgeId,
    /// The piece whose contact reaches `third` farther, then the other.
    pieces: [usize; 2],
    /// Each piece's face other than `shared`, in the order of `pieces`.
    faces: [FaceId; 2],
    meet: Point3,
    reach: Point3,
    cut: Option<Point3>,
    /// From `cut` (or `reach`) to `meet`, between the two chamfer faces.
    mitre: EdgeCurve,
    /// From `reach` to `cut`, between the first piece's chamfer face and the
    /// second piece's face other than `shared`.
    sliver: Option<EdgeCurve>,
}

impl Corner {
    /// Where piece `index`'s contacts end here: on the shared face, then on
    /// its other face.
    fn ends_of(&self, index: usize) -> (Point3, Point3) {
        let other = if index == self.pieces[0] {
            self.reach
        } else {
            self.cut.unwrap_or(self.reach)
        };
        (self.meet, other)
    }
}

/// One chain, read off the solid and ready to build.
struct Plan {
    pieces: Vec<Piece>,
    /// Each piece's sections at its start and end.
    piece_sections: Vec<[Section; 2]>,
    /// The one section at each vertex other than a corner.
    sections: Vec<(VertexId, Section)>,
    joints: Vec<Joint>,
    ends: Vec<End>,
    plane_ends: Vec<PlaneEnd>,
    corners: Vec<Corner>,
    /// The chain's change in the solid's volume, each edge's triangle swept
    /// along it.
    volume_change: f64,
    /// How far the change may stray from `volume_change` at the corners and
    /// the slanted ends, where the chamfers end past or short of their edges'
    /// ends.
    corner_allowance: f64,
    /// Whether the chain needs this path: it holds an arc or a ruling of a
    /// cylinder, or stops at a curved neighbour, where the other engines'
    /// ends are not exact.
    curved: bool,
}

impl Plan {
    fn section(&self, vertex: VertexId) -> Option<Section> {
        self.sections
            .iter()
            .find(|(v, _)| *v == vertex)
            .map(|&(_, s)| s)
    }

    fn corner(&self, vertex: VertexId) -> Option<&Corner> {
        self.corners.iter().find(|c| c.vertex == vertex)
    }

    fn plane_end(&self, vertex: VertexId) -> Option<&PlaneEnd> {
        self.plane_ends.iter().find(|e| e.vertex == vertex)
    }

    /// Where piece `index`'s contacts end at `vertex`: on its cap, then on
    /// its wall.
    fn contact_ends(&self, index: usize, vertex: VertexId) -> (Point3, Point3) {
        let piece = &self.pieces[index];
        if let Some(corner) = self.corner(vertex) {
            let (on_shared, on_other) = corner.ends_of(index);
            return if corner.shared == piece.cap {
                (on_shared, on_other)
            } else {
                (on_other, on_shared)
            };
        }
        if let Some(end) = self.plane_end(vertex) {
            return (end.points[0], end.points[1]);
        }
        let s = self.piece_sections[index][piece.slot(vertex)];
        (s.cap, s.wall)
    }

    /// Whether `p`, on `cap`, lies in the strip a chamfer takes from it.
    fn strip_holds(&self, cap: FaceId, p: Point3, tol: f64) -> Result<bool, OperationsError> {
        for (index, piece) in self.pieces.iter().enumerate() {
            if piece.cap == cap && self.cap_holds(index, p, tol)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether `p`, on piece `index`'s cap, lies in the strip its chamfer
    /// takes from the cap.
    fn cap_holds(&self, index: usize, p: Point3, tol: f64) -> Result<bool, OperationsError> {
        let piece = &self.pieces[index];
        let section = self.piece_sections[index][0];
        Ok(match &piece.curve {
            EdgeCurve::Line => {
                let along = piece.end_point - piece.start_point;
                let length = along.length();
                let tangent = along.normalize()?;
                let offset = p - piece.start_point;
                let s = offset.dot(tangent);
                let h = offset.dot(section.cap_dir);
                s > tol && s < length - tol && h > tol && h <= piece.cap_d + tol
            }
            EdgeCurve::Circle(c) => {
                let radial = (piece.start_point - c.center()).normalize()?;
                let toward = section.cap_dir.dot(radial);
                let h = ((p - c.center()).length() - c.radius()) * toward;
                let (t0, t1) = piece
                    .curve
                    .domain_with_endpoints(piece.start_point, piece.end_point);
                let swept = (c.project(p) - t0).rem_euclid(TAU);
                swept < t1 - t0 && h > tol && h <= piece.cap_d + tol
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
        })
    }

    /// Whether `p`, on `face`, lies in the strip piece `index`'s chamfer
    /// takes from it as its cap ([`Plan::cap_holds`]) or as its wall: beside
    /// a line on a plane wall, under an arc on its cylinder, or beside a
    /// ruling.
    fn takes(
        &self,
        index: usize,
        face: FaceId,
        p: Point3,
        tol: f64,
    ) -> Result<bool, OperationsError> {
        let piece = &self.pieces[index];
        if piece.cap == face && self.cap_holds(index, p, tol)? {
            return Ok(true);
        }
        if piece.wall != face {
            return Ok(false);
        }
        let rise = (p - piece.start_point).dot(self.piece_sections[index][0].wall_dir);
        Ok(match (&piece.curve, &piece.wall_side) {
            (EdgeCurve::Line, WallSide::Plane { .. }) => {
                let along = piece.end_point - piece.start_point;
                let s = (p - piece.start_point).dot(along.normalize()?);
                s > tol && s < along.length() - tol && rise > tol && rise <= piece.wall_d + tol
            }
            (EdgeCurve::Circle(c), WallSide::Axis(_)) => {
                let (t0, t1) = piece
                    .curve
                    .domain_with_endpoints(piece.start_point, piece.end_point);
                let swept = (c.project(p) - t0).rem_euclid(TAU);
                swept < t1 - t0 && rise > tol && rise <= piece.wall_d + tol
            }
            (EdgeCurve::Line, WallSide::Ruling { cylinder, turn }) => {
                let Some((from, width, low, high)) = cylinder_strip(cylinder, piece) else {
                    return Ok(false);
                };
                let (u, _) = cylinder.project_point(p);
                let height = cylinder.axis().dot(p - cylinder.origin());
                let slack = tol / cylinder.radius();
                let off = (u - from).rem_euclid(TAU);
                // Past the edge, up to and on the contact.
                let in_angles = if *turn < 0.0 {
                    off < width - slack || off > TAU - slack
                } else {
                    off > slack && off <= width + slack
                };
                in_angles && height > low + tol && height < high - tol
            }
            _ => false,
        })
    }

    /// A box that holds the strip piece `index`'s chamfer takes from its cap
    /// (`side` 0) or its wall (`side` 1), where [`Plan::takes`] can hold a
    /// point: the edge and the contact at its sections, the arcs between
    /// their ends and, beside a ruling, the bulge of the wall's arc between
    /// the edge and the contact.
    fn strip_box(&self, index: usize, side: usize) -> Result<Aabb3, OperationsError> {
        let tol = Tolerance::new().linear;
        let piece = &self.pieces[index];
        let [s0, s1] = self.piece_sections[index];
        let (a, b) = if side == 0 {
            (s0.cap, s1.cap)
        } else {
            (s0.wall, s1.wall)
        };
        let mut corners = vec![piece.start_point, piece.end_point, a, b];
        let mut margin = tol;
        if let EdgeCurve::Circle(c) = &piece.curve {
            let (t0, t1) = piece
                .curve
                .domain_with_endpoints(piece.start_point, piece.end_point);
            let contact = &piece.contact_curves(&s0)?[side];
            for arc in std::iter::once(c).chain(match contact {
                EdgeCurve::Circle(on) => Some(on),
                EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => None,
            }) {
                let bounds = arc.arc_aabb(t0, t1);
                corners.extend([bounds.min, bounds.max]);
            }
        } else if let (1, WallSide::Ruling { cylinder, turn }) = (side, &piece.wall_side) {
            margin += cylinder.radius() * (1.0 - (0.5 * turn).cos());
        }
        Ok(Aabb3::from_points(corners).expanded(margin))
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

    /// The span's box.
    fn bounds(&self) -> Aabb3 {
        match self {
            Self::Segment(a, b) => Aabb3::from_points([*a, *b]),
            Self::Arc { circle, t0, sweep } => circle.arc_aabb(*t0, *t0 + *sweep),
        }
    }

    /// Points along the span, its ends among them.
    fn samples(&self) -> Vec<Point3> {
        let at = |k: u32| f64::from(k) / f64::from(STRIP_SAMPLES);
        match self {
            Self::Segment(a, b) => (0..=STRIP_SAMPLES)
                .map(|k| *a + (*b - *a) * at(k))
                .collect(),
            Self::Arc { circle, t0, sweep } => (0..=STRIP_SAMPLES)
                .map(|k| circle.evaluate(*t0 + *sweep * at(k)))
                .collect(),
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
        if !matches!(read_curve(topo, oe.edge())?, EdgeCurve::Line) {
            continue;
        }
        let edge = topo.edge(oe.edge())?;
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

/// The edges of `face` at `vertex`.
fn edges_at(
    topo: &Topology,
    face: FaceId,
    vertex: VertexId,
) -> Result<Vec<EdgeId>, OperationsError> {
    let data = topo.face(face)?;
    let mut found = Vec::new();
    for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let e = topo.edge(oe.edge())?;
            if (e.start() == vertex || e.end() == vertex) && !found.contains(&oe.edge()) {
                found.push(oe.edge());
            }
        }
    }
    Ok(found)
}

/// An edge's curve, a NURBS curve lying on the segment between the edge's
/// ends read as the line it is.
fn read_curve(topo: &Topology, edge: EdgeId) -> Result<EdgeCurve, OperationsError> {
    let data = topo.edge(edge)?;
    let curve = data.curve();
    if matches!(curve, EdgeCurve::NurbsCurve(_))
        && curve.is_straight_between(
            topo.vertex(data.start())?.point(),
            topo.vertex(data.end())?.point(),
            JOIN_TOL,
        )
    {
        return Ok(EdgeCurve::Line);
    }
    Ok(curve.clone())
}

/// The other end of `edge` from `vertex`.
fn far_end(topo: &Topology, edge: EdgeId, vertex: VertexId) -> Result<VertexId, OperationsError> {
    let e = topo.edge(edge)?;
    Ok(if e.start() == vertex {
        e.end()
    } else {
        e.start()
    })
}

/// The face around `edge` other than `face`.
fn other_face(adjacency: &AdjacencyIndex, edge: EdgeId, face: FaceId) -> Option<FaceId> {
    let around = adjacency.faces_for_edge(edge);
    (around.len() == 2 && around.contains(&face)).then(|| {
        if around[0] == face {
            around[1]
        } else {
            around[0]
        }
    })
}

/// How far a chamfer reaches into each face of each picked edge: `near`
/// into the edge's reference face and `far` into its other face, or `near`
/// into both where the edge has no reference.
struct Distances {
    near: f64,
    far: f64,
    reference: HashMap<EdgeId, FaceId>,
}

impl Distances {
    fn on(&self, edge: EdgeId, face: FaceId) -> f64 {
        match self.reference.get(&edge) {
            Some(&reference) if reference != face => self.far,
            _ => self.near,
        }
    }

    fn least(&self) -> f64 {
        self.near.min(self.far)
    }
}

/// What one connected group of selected edges reads as.
enum Reading {
    /// A chain this path builds exactly.
    Chain(Box<Plan>),
    /// A selection this path does not build, which another engine can take.
    Other,
    /// A chain that no chamfer of its distances fits, for the reason given.
    Misfit(String),
}

/// Read one connected group of selected edges off the solid as a chain.
fn plan_chain(
    topo: &Topology,
    faces: &[FaceId],
    group: &[EdgeId],
    distances: &Distances,
    far_sides: &mut FarSides,
) -> Result<Reading, OperationsError> {
    let adjacency = AdjacencyIndex::build_from_faces(topo, faces)?;
    // The faces along every edge of the group.
    let mut common: Option<Vec<FaceId>> = None;
    for &edge in group {
        let mut around: Vec<FaceId> = adjacency.faces_for_edge(edge).to_vec();
        around.dedup();
        if around.len() != 2 {
            return Ok(Reading::Other);
        }
        common = Some(match common {
            None => around,
            Some(kept) => kept.into_iter().filter(|f| around.contains(f)).collect(),
        });
    }
    let common = common.unwrap_or_default();
    // Every edge on one cap, the walls apart; a line lies on two planes and
    // either may be its cap, whatever the shell's face order. Failing that,
    // every edge along one wall, the caps apart. A reading that does not fit
    // gives way to one that builds.
    let mut misfit = None;
    let readings = common
        .iter()
        .map(|&cap| (cap, Common::Cap(cap)))
        .chain(common.iter().map(|&wall| (wall, Common::Wall(wall))));
    for (face, along) in readings {
        if matches!(along, Common::Cap(_)) && plane_normal(topo, face)?.is_none() {
            continue;
        }
        match plan_on(topo, faces, &adjacency, group, along, distances, far_sides)? {
            Reading::Chain(plan) => return Ok(Reading::Chain(plan)),
            Reading::Misfit(reason) => {
                misfit.get_or_insert(reason);
            }
            Reading::Other => {}
        }
    }
    Ok(misfit.map_or(Reading::Other, Reading::Misfit))
}

/// The face a whole chain runs along.
#[derive(Clone, Copy)]
enum Common {
    Cap(FaceId),
    Wall(FaceId),
}

/// Into the wall's material from a line along a ruling of `cylinder`, and
/// the turn about the axis to the contact a chord `d` away: the left of the
/// wall's stored traversal of the edge, about the cylinder's own normal. A
/// reversed face flips both its normal and its traversal, as `build_chain`
/// reads a face's use, so the two flips cancel.
#[allow(clippy::too_many_arguments)]
fn ruling_side(
    topo: &Topology,
    wall: FaceId,
    cylinder: &CylindricalSurface,
    cap_normal: Vec3,
    edge: EdgeId,
    start: Point3,
    end: Point3,
    d: f64,
) -> Result<Option<WallSide>, OperationsError> {
    let axis = cylinder.axis();
    let along = (end - start).normalize()?;
    let radius = cylinder.radius();
    if along.cross(axis).length() > JOIN_TOL
        || cap_normal.dot(axis).abs() > JOIN_TOL
        || d >= 2.0 * radius
    {
        return Ok(None);
    }
    let foot = |p: Point3| cylinder.origin() + axis * axis.dot(p - cylinder.origin());
    for p in [start, end] {
        if ((p - foot(p)).length() - radius).abs() > JOIN_TOL {
            return Ok(None);
        }
    }
    let Some(flag) = wire_flag(topo, wall, edge)? else {
        return Ok(None);
    };
    let radial = (start - foot(start)).normalize()?;
    let traversal = if flag { along } else { -along };
    let sign = radial.cross(traversal).dot(axis.cross(radial)).signum();
    Ok(Some(WallSide::Ruling {
        cylinder: cylinder.clone(),
        turn: 2.0 * (0.5 * d / radius).asin() * sign,
    }))
}

/// [`plan_chain`] with every edge along `common`.
fn plan_on(
    topo: &Topology,
    faces: &[FaceId],
    adjacency: &AdjacencyIndex,
    group: &[EdgeId],
    common: Common,
    distances: &Distances,
    far_sides: &mut FarSides,
) -> Result<Reading, OperationsError> {
    let Some(pieces) = read_pieces(topo, adjacency, group, common, distances)? else {
        return Ok(Reading::Other);
    };
    if let Some(reason) = misfit(topo, adjacency, &pieces, far_sides)? {
        return Ok(Reading::Misfit(reason));
    }
    Ok(match plan_pieces(topo, faces, adjacency, pieces)? {
        Some(plan) => Reading::Chain(Box::new(plan)),
        None => Reading::Other,
    })
}

/// Why no chamfer of the pieces' distances fits their chain, if none does:
/// a cap's contact past an arc's axis, where it has no circle to run on, or
/// a contact at or past the far side of its face, so that the face, and any
/// face that carries it on smoothly, ends before the chamfer does (see
/// [`FarSides::reach`]). Each face's far side is read from its box, which
/// holds the whole face, so a face that may reach farther is not refused. A
/// chain of lines between planes that meets no curved face is not read, as
/// the other engines take it.
fn misfit(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    pieces: &[Piece],
    far_sides: &mut FarSides,
) -> Result<Option<String>, OperationsError> {
    let tol = Tolerance::new().linear;
    if !meets_a_curved_face(topo, adjacency, pieces)? {
        return Ok(None);
    }
    for piece in pieces {
        let tangent = piece.tangent(piece.start_point)?;
        let cap_dir = piece.cap_normal.cross(tangent) * piece.cap_sign;
        let mut sides = Vec::with_capacity(2);
        match &piece.curve {
            EdgeCurve::Circle(c) => {
                let radial = (piece.start_point - c.center()).normalize()?;
                if c.radius() + cap_dir.dot(radial) * piece.cap_d <= tol {
                    return Ok(Some(format!(
                        "a chamfer {} into its cap runs past the axis of an arc of radius {}",
                        piece.cap_d,
                        c.radius()
                    )));
                }
            }
            EdgeCurve::Line => sides.push((piece.cap, cap_dir, piece.cap_d)),
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {}
        }
        match &piece.wall_side {
            WallSide::Plane { normal, sign } => {
                sides.push((piece.wall, normal.cross(tangent) * *sign, piece.wall_d));
            }
            WallSide::Axis(into) => sides.push((piece.wall, *into, piece.wall_d)),
            WallSide::Ruling { .. } => {}
        }
        for (face, dir, d) in sides {
            if let Some(far) = far_sides.reach(topo, adjacency, face, piece.start_point, dir)?
                && d >= far - tol
            {
                return Ok(Some(format!(
                    "a chamfer {d} into a face of its edge runs past the face's far side, \
                     {far:.6} from the edge"
                )));
            }
        }
    }
    Ok(None)
}

/// Whether a chain holds an arc or a line along a ruling, or a face that is
/// not a plane lies beside one of its ends. Each cap and wall is read once.
fn meets_a_curved_face(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    pieces: &[Piece],
) -> Result<bool, OperationsError> {
    if pieces.iter().any(|piece| {
        matches!(piece.curve, EdgeCurve::Circle(_))
            || matches!(piece.wall_side, WallSide::Ruling { .. })
    }) {
        return Ok(true);
    }
    let ends: HashSet<VertexId> = pieces.iter().flat_map(|p| [p.start, p.end]).collect();
    let faces: HashSet<FaceId> = pieces.iter().flat_map(|p| [p.cap, p.wall]).collect();
    for face in faces {
        for edge in brepkit_topology::explorer::face_edges(topo, face)? {
            let data = topo.edge(edge)?;
            if !ends.contains(&data.start()) && !ends.contains(&data.end()) {
                continue;
            }
            for &beside in adjacency.faces_for_edge(edge) {
                if plane_normal(topo, beside)?.is_none() {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// How far faces reach along directions, each run of faces that carry each
/// other on read once along each direction (see [`FarSides::reach`]).
#[derive(Default)]
struct FarSides {
    /// For each face read, what each read of its run along a direction gave.
    known: HashMap<FaceId, Vec<FarRead>>,
    /// How many face boxes have been read.
    boxes: usize,
}

/// One read of a run of faces along a direction.
#[derive(Clone, Copy)]
struct FarRead {
    /// The direction's bits.
    bits: [u64; 3],
    /// The point the run was read from.
    from: Point3,
    /// How far the run reaches past `from` along the direction, `None`
    /// where a box is not known.
    far: Option<f64>,
}

impl FarSides {
    /// How far `face` reaches from `origin` along `dir` at most, with each
    /// face that carries it on past an edge where the two faces' normals
    /// agree (a flat seam or a tangent join), where a contact can run on.
    /// Read from each face's box in a frame along `dir`; `None` where a box
    /// is not known. The run is the same from any of its faces, so each
    /// face of it keeps what was read for a read along the same direction,
    /// to its last bit: a direction near it reads a far side that can move
    /// by the angle between them times the distance to the run, which need
    /// not be small. What is kept is measured from the point it was read
    /// from, on the run, so where the solid lies does not enter it.
    fn reach(
        &mut self,
        topo: &Topology,
        adjacency: &AdjacencyIndex,
        face: FaceId,
        origin: Point3,
        dir: Vec3,
    ) -> Result<Option<f64>, OperationsError> {
        let dir = dir.normalize()?;
        let key = [dir.x().to_bits(), dir.y().to_bits(), dir.z().to_bits()];
        if let Some(read) = self
            .known
            .get(&face)
            .and_then(|reads| reads.iter().find(|read| read.bits == key))
        {
            return Ok(read.far.map(|far| far - (origin - read.from).dot(dir)));
        }
        let frame = Frame3::from_normal(origin, dir)?;
        let frame = Reach {
            origin,
            axes: [frame.x, frame.y, dir],
            lo: [0.0; 3],
            hi: [0.0; 3],
            pieces: Vec::new(),
        };
        let mut far = Some(f64::NEG_INFINITY);
        let mut seen = HashSet::from([face]);
        let mut next = vec![face];
        while let Some(at) = next.pop() {
            if let Some(so_far) = far {
                self.boxes += 1;
                far = frame.face_box(topo, at)?.map(|(_, hi)| so_far.max(hi[2]));
            }
            for edge in brepkit_topology::explorer::face_edges(topo, at)? {
                let Some(beyond) = other_face(adjacency, edge, at) else {
                    continue;
                };
                if seen.contains(&beyond) {
                    continue;
                }
                let data = topo.edge(edge)?;
                let (a, b) = (
                    topo.vertex(data.start())?.point(),
                    topo.vertex(data.end())?.point(),
                );
                let (t0, t1) = data.curve().domain_with_endpoints(a, b);
                let mid = data.curve().evaluate_with_endpoints(0.5 * (t0 + t1), a, b);
                if (outward_at(topo, at, mid)? - outward_at(topo, beyond, mid)?).length() < JOIN_TOL
                {
                    seen.insert(beyond);
                    next.push(beyond);
                }
            }
        }
        for read in seen {
            self.known.entry(read).or_default().push(FarRead {
                bits: key,
                from: origin,
                far,
            });
        }
        Ok(far)
    }
}

/// The outward normal of `face` at `p` on it.
fn outward_at(topo: &Topology, face: FaceId, p: Point3) -> Result<Vec3, OperationsError> {
    let data = topo.face(face)?;
    let (u, v) = data.surface().project_point(p).unwrap_or((0.0, 0.0));
    let normal = data.surface().normal(u, v);
    Ok(if data.is_reversed() { -normal } else { normal })
}

/// Each edge of `group` read as a piece along `common`, or `None` when one
/// is not an edge this path reads.
#[allow(clippy::too_many_lines)]
fn read_pieces(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    group: &[EdgeId],
    common: Common,
    distances: &Distances,
) -> Result<Option<Vec<Piece>>, OperationsError> {
    let tol = Tolerance::new().linear;
    let step = distances.least() * 1e-3;

    let mut pieces = Vec::with_capacity(group.len());
    for &edge in group {
        let data = topo.edge(edge)?;
        let (start, end) = (data.start(), data.end());
        let start_point = topo.vertex(start)?.point();
        let end_point = topo.vertex(end)?.point();
        let curve = read_curve(topo, edge)?;
        let (cap, wall) = match common {
            Common::Cap(cap) => {
                let Some(wall) = other_face(adjacency, edge, cap) else {
                    return Ok(None);
                };
                (cap, wall)
            }
            Common::Wall(wall) => {
                let Some(cap) = other_face(adjacency, edge, wall) else {
                    return Ok(None);
                };
                (cap, wall)
            }
        };
        let Some(cap_normal) = plane_normal(topo, cap)? else {
            return Ok(None);
        };
        let (cap_d, wall_d) = (distances.on(edge, cap), distances.on(edge, wall));
        let wall_side = match &curve {
            EdgeCurve::Line => {
                if let Some(normal) = plane_normal(topo, wall)? {
                    let tangent = (end_point - start_point).normalize()?;
                    let m = start_point + (end_point - start_point) * 0.5;
                    let across = normal.cross(tangent);
                    let Some(sign) = side_of_plane_face(topo, wall, normal, m, across, step)?
                    else {
                        return Ok(None);
                    };
                    WallSide::Plane { normal, sign }
                } else if let Some(cylinder) = cylinder_of(topo, wall)? {
                    let Some(side) = ruling_side(
                        topo,
                        wall,
                        &cylinder,
                        cap_normal,
                        edge,
                        start_point,
                        end_point,
                        wall_d,
                    )?
                    else {
                        return Ok(None);
                    };
                    side
                } else {
                    return Ok(None);
                }
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
                // Into the wall's material along its axis: the left of the
                // wall's stored traversal of the arc about the cylinder's own
                // normal, read at the arc's middle, as for a ruling. A wall
                // that rises past the rim elsewhere does not turn it.
                let Some(flag) = wire_flag(topo, wall, edge)? else {
                    return Ok(None);
                };
                let (t0, t1) = curve.domain_with_endpoints(start_point, end_point);
                let mid = circle.evaluate(0.5 * (t0 + t1));
                let tangent = circle.tangent(0.5 * (t0 + t1)).normalize()?;
                let foot = cylinder.origin() + axis * axis.dot(mid - cylinder.origin());
                let radial = (mid - foot).normalize()?;
                let traversal = if flag { tangent } else { -tangent };
                let into = axis * radial.cross(traversal).dot(axis).signum();
                // And the wall reaches past the rim that way.
                let along = |p: Point3| (p - cylinder.origin()).dot(into);
                let rim = along(start_point);
                let mut reach = 0.0_f64;
                for oe in topo.wire(topo.face(wall)?.outer_wire())?.edges() {
                    let e = topo.edge(oe.edge())?;
                    for v in [e.start(), e.end()] {
                        reach = reach.max(along(topo.vertex(v)?.point()) - rim);
                    }
                }
                if reach <= tol {
                    return Ok(None);
                }
                WallSide::Axis(into)
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
        };
        // Which way the cap's material lies, read at the edge's middle.
        let mut piece = Piece {
            edge,
            cap,
            cap_normal,
            wall,
            start,
            end,
            curve,
            start_point,
            end_point,
            cap_sign: 1.0,
            wall_side,
            cap_d,
            wall_d,
        };
        let m = piece.midpoint();
        let across = cap_normal.cross(piece.tangent(m)?);
        let Some(sign) = side_of_plane_face(topo, cap, cap_normal, m, across, step)? else {
            return Ok(None);
        };
        piece.cap_sign = sign;
        pieces.push(piece);
    }
    Ok(Some(pieces))
}

/// The plan of a chain of `pieces`, or `None` when it is not one this path
/// builds exactly or it would meet something it must not.
#[allow(clippy::too_many_lines)]
fn plan_pieces(
    topo: &Topology,
    faces: &[FaceId],
    adjacency: &AdjacencyIndex,
    pieces: Vec<Piece>,
) -> Result<Option<Plan>, OperationsError> {
    let tol = Tolerance::new().linear;
    let mut piece_sections = Vec::with_capacity(pieces.len());
    let mut uses: Vec<(VertexId, Vec<usize>)> = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        piece_sections.push([
            piece.section(piece.start_point)?,
            piece.section(piece.end_point)?,
        ]);
        for vertex in [piece.start, piece.end] {
            match uses.iter_mut().find(|(v, _)| *v == vertex) {
                Some((_, at)) => at.push(index),
                None => uses.push((vertex, vec![index])),
            }
        }
    }

    // What the chain does at each vertex: join two edges, turn a corner
    // between two, or end.
    let mut sections: Vec<(VertexId, Section)> = Vec::new();
    let mut joints = Vec::new();
    let mut ends = Vec::new();
    let mut plane_ends = Vec::new();
    let mut corners = Vec::new();
    let mut curved = pieces.iter().any(|p| {
        matches!(p.curve, EdgeCurve::Circle(_)) || matches!(p.wall_side, WallSide::Ruling { .. })
    });
    for (vertex, at) in &uses {
        let vertex = *vertex;
        match at[..] {
            [a, b] => {
                let (first, second) = (&pieces[a], &pieces[b]);
                let through = first.leaving(vertex)? + second.leaving(vertex)?;
                if through.length() > JOIN_TOL {
                    let Some(corner) = plan_corner(topo, &pieces, &piece_sections, a, b, vertex)?
                    else {
                        return Ok(None);
                    };
                    corners.push(corner);
                    continue;
                }
                let ruled = |p: &Piece| matches!(p.wall_side, WallSide::Ruling { .. });
                if first.cap != second.cap || ruled(first) || ruled(second) {
                    return Ok(None);
                }
                let (sa, sb) = (
                    piece_sections[a][first.slot(vertex)],
                    piece_sections[b][second.slot(vertex)],
                );
                if (sa.cap - sb.cap).length() > JOIN_TOL || (sa.wall - sb.wall).length() > JOIN_TOL
                {
                    return Ok(None);
                }
                sections.push((vertex, sa));
                let ruling = ruling_along(topo, first.wall, vertex, sa.wall_dir)?;
                if let Some((_, length)) = ruling
                    && length <= first.wall_d + tol
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
                let section = piece_sections[a][piece.slot(vertex)];
                sections.push((vertex, section));
                if let Some((end, beyond_curved)) =
                    plan_tangent_end(topo, adjacency, piece, &section, vertex)?
                {
                    curved |= beyond_curved;
                    ends.push(end);
                } else if let Some(end) =
                    plan_plane_end(topo, adjacency, a, piece, &piece_sections[a], vertex)?
                {
                    plane_ends.push(end);
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        }
    }

    // The solid's change: the triangle's area along its centroid's path.
    let mut volume_change = 0.0;
    for (piece, piece_ends) in pieces.iter().zip(&piece_sections) {
        let section = piece_ends[0];
        let mut area =
            0.5 * section.cap_dir.cross(section.wall_dir).length() * piece.cap_d * piece.wall_d;
        // A wall running under the cap's outward side is a convex edge, whose
        // triangle is material; above it, the triangle fills a corner.
        let cap_out = if topo.face(piece.cap)?.is_reversed() {
            -piece.cap_normal
        } else {
            piece.cap_normal
        };
        let convex = section.wall_dir.dot(cap_out) < 0.0;
        // Beside a ruling the triangle's wall side is an arc, which bulges
        // inside its circle: into the region the chamfer takes or fills where
        // that region lies inside the circle, that is, where a convex edge
        // meets a round wall or a concave edge meets a hollow one.
        if let WallSide::Ruling { cylinder, turn } = &piece.wall_side {
            let bulge = 0.5 * cylinder.radius().powi(2) * (turn.abs() - turn.abs().sin());
            if convex == topo.face(piece.wall)?.is_reversed() {
                area -= bulge;
            } else {
                area += bulge;
            }
        }
        let path = match &piece.curve {
            EdgeCurve::Line => (piece.end_point - piece.start_point).length(),
            EdgeCurve::Circle(c) => {
                let radial = (piece.start_point - c.center()).normalize()?;
                let toward = section.cap_dir.dot(radial);
                // A cap point past the arc's axis has no circle to run on.
                if c.radius() + toward * piece.cap_d <= tol {
                    return Ok(None);
                }
                let (t0, t1) = piece
                    .curve
                    .domain_with_endpoints(piece.start_point, piece.end_point);
                (t1 - t0) * (c.radius() + toward * piece.cap_d / 3.0)
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
        };
        let sign = if convex { -1.0 } else { 1.0 };
        volume_change += sign * area * path;
    }
    // The chamfers at a corner meet within a ball about it reaching its
    // farthest point, and a slanted end stops within one about its vertex.
    let mut corner_allowance = 0.0;
    for corner in &corners {
        let at = topo.vertex(corner.vertex)?.point();
        let reach = [Some(corner.meet), Some(corner.reach), corner.cut]
            .into_iter()
            .flatten()
            .map(|p| (p - at).length())
            .fold(corner_reach(&pieces, corner), f64::max);
        corner_allowance += 4.0 / 3.0 * PI * reach.powi(3);
    }
    for end in plane_ends.iter().filter(|e| e.slanted) {
        let reach = end_reach(topo, &pieces[end.piece], end)?;
        corner_allowance += 4.0 / 3.0 * PI * reach.powi(3);
    }

    let plan = Plan {
        pieces,
        piece_sections,
        sections,
        joints,
        ends,
        plane_ends,
        corners,
        volume_change,
        corner_allowance,
        curved,
    };
    if !plan_is_clear(topo, &plan)? || !wedges_are_clear(topo, &plan, faces)? {
        return Ok(None);
    }
    Ok(Some(plan))
}

/// A box's least and greatest corners in some frame.
type Box3 = ([f64; 3], [f64; 3]);

/// A box in a frame of its own: the region one piece's chamfer, or one
/// corner's, lies in. A piece's chamfer lies within `d` of its edge, and a
/// corner's region within its farthest point's distance of its vertex.
struct Reach {
    origin: Point3,
    axes: [Vec3; 3],
    lo: [f64; 3],
    hi: [f64; 3],
    /// The pieces whose chamfer faces bound the region.
    pieces: Vec<usize>,
}

impl Reach {
    fn local(&self, p: Point3) -> [f64; 3] {
        let off = p - self.origin;
        self.axes.map(|axis| off.dot(axis))
    }

    fn meets(&self, (lo, hi): Box3) -> bool {
        (0..3).all(|k| lo[k] <= self.hi[k] && hi[k] >= self.lo[k])
    }

    /// The box of `face` in this frame: its boundary edges counted exactly,
    /// which bound a plane, cylinder or cone face, and the poles or the
    /// whole extent of a surface that can bulge past them. `None` for a
    /// NURBS curve or surface with a weight that is not positive, which its
    /// poles do not bound.
    fn face_box(&self, topo: &Topology, face: FaceId) -> Result<Option<Box3>, OperationsError> {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut add = |p: [f64; 3]| {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        };
        let to_vec = |v: Vec3| {
            let [x, y, z] = self.axes.map(|axis| v.dot(axis));
            Vec3::new(x, y, z)
        };
        let at = |p: Point3| {
            let [x, y, z] = self.local(p);
            Point3::new(x, y, z)
        };
        let data = topo.face(face)?;
        for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                let (a, b) = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                add(self.local(a));
                add(self.local(b));
                let span = |curve: &EdgeCurve| {
                    if edge.start() == edge.end() {
                        (0.0, TAU)
                    } else {
                        curve.domain_with_endpoints(a, b)
                    }
                };
                let bounds = match edge.curve() {
                    EdgeCurve::Line => continue,
                    EdgeCurve::Circle(c) => {
                        let (t0, t1) = span(edge.curve());
                        Circle3D::with_axes(
                            at(c.center()),
                            to_vec(c.normal()),
                            c.radius(),
                            to_vec(c.u_axis()),
                            to_vec(c.v_axis()),
                        )?
                        .arc_aabb(t0, t1)
                    }
                    EdgeCurve::Ellipse(e) => {
                        let (t0, t1) = span(edge.curve());
                        brepkit_math::curves::Ellipse3D::with_axes(
                            at(e.center()),
                            to_vec(e.normal()),
                            e.semi_major(),
                            e.semi_minor(),
                            to_vec(e.u_axis()),
                            to_vec(e.v_axis()),
                        )?
                        .arc_aabb(t0, t1)
                    }
                    EdgeCurve::NurbsCurve(n) => {
                        if !n.weights().iter().all(|w| *w > 0.0) {
                            return Ok(None);
                        }
                        for &p in n.control_points() {
                            add(self.local(p));
                        }
                        continue;
                    }
                };
                add([bounds.min.x(), bounds.min.y(), bounds.min.z()]);
                add([bounds.max.x(), bounds.max.y(), bounds.max.z()]);
            }
        }
        let around = |center: Point3, reach: f64| {
            let c = self.local(center);
            [c.map(|v| v - reach), c.map(|v| v + reach)]
        };
        match data.surface() {
            FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {}
            FaceSurface::Sphere(sphere) => {
                for p in around(sphere.center(), sphere.radius()) {
                    add(p);
                }
            }
            FaceSurface::Torus(torus) => {
                for p in around(torus.center(), torus.major_radius() + torus.minor_radius()) {
                    add(p);
                }
            }
            FaceSurface::Nurbs(surface) => {
                if !surface.weights().iter().flatten().all(|w| *w > 0.0) {
                    return Ok(None);
                }
                for row in surface.control_points() {
                    for &p in row {
                        add(self.local(p));
                    }
                }
            }
        }
        Ok(Some((lo, hi)))
    }
}

/// Points whose convex hull holds `face`: its boundary's vertices and the
/// poles of its arcs, split to at most 60 degrees, or a NURBS curve's poles;
/// and for a surface that can bulge past its boundary, a box about a sphere
/// or a torus, or a NURBS surface's poles. A plane, cylinder or cone face
/// lies in its boundary's hull: each point of it is on a segment, or a
/// ruling, between two boundary points. `None` where a weight is not
/// positive, so the poles do not bound the curve or surface.
fn face_hull_points(topo: &Topology, face: FaceId) -> Result<Option<Vec<Point3>>, OperationsError> {
    let mut points = Vec::new();
    let data = topo.face(face)?;
    for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            points.extend([a, b]);
            let conic = match edge.curve() {
                EdgeCurve::Line => None,
                EdgeCurve::Circle(c) => {
                    Some((c.center(), c.u_axis() * c.radius(), c.v_axis() * c.radius()))
                }
                EdgeCurve::Ellipse(e) => Some((
                    e.center(),
                    e.u_axis() * e.semi_major(),
                    e.v_axis() * e.semi_minor(),
                )),
                EdgeCurve::NurbsCurve(n) => {
                    if !n.weights().iter().all(|w| *w > 0.0) {
                        return Ok(None);
                    }
                    points.extend_from_slice(n.control_points());
                    None
                }
            };
            if let Some((center, u, v)) = conic {
                let (t0, t1) = if edge.start() == edge.end() {
                    (0.0, TAU)
                } else {
                    edge.curve().domain_with_endpoints(a, b)
                };
                let count = ((t1 - t0).abs() / (PI / 3.0)).ceil().max(1.0);
                let step = (t1 - t0) / count;
                let reach = 1.0 / (0.5 * step).cos();
                let mut t = t0;
                for _ in 0..count as usize {
                    let mid = t + 0.5 * step;
                    points.push(center + u * t.cos() + v * t.sin());
                    points.push(center + (u * mid.cos() + v * mid.sin()) * reach);
                    t += step;
                }
            }
        }
    }
    let corners = |center: Point3, axes: [Vec3; 3], half: [f64; 3]| -> Vec<Point3> {
        (0..8_usize)
            .map(|k| {
                let mut p = center;
                for axis in 0..3 {
                    let sign = if (k >> axis) & 1 == 0 { -1.0 } else { 1.0 };
                    p = p + axes[axis] * (sign * half[axis]);
                }
                p
            })
            .collect()
    };
    match data.surface() {
        FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {}
        FaceSurface::Sphere(sphere) => {
            let world = [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ];
            points.extend(corners(sphere.center(), world, [sphere.radius(); 3]));
        }
        FaceSurface::Torus(torus) => {
            let outer = torus.major_radius() + torus.minor_radius();
            points.extend(corners(
                torus.center(),
                [torus.x_axis(), torus.y_axis(), torus.z_axis()],
                [outer, outer, torus.minor_radius()],
            ));
        }
        FaceSurface::Nurbs(surface) => {
            if !surface.weights().iter().flatten().all(|w| *w > 0.0) {
                return Ok(None);
            }
            points.extend(surface.control_points().iter().flatten().copied());
        }
    }
    Ok(Some(points))
}

/// Whether a face meets a reach's box: for a plane face, exactly, where its
/// region meets the polygon its plane cuts from the box; for any other face,
/// always, so that the box test alone decides it.
fn plane_meets_reach(
    topo: &Topology,
    face: FaceId,
    reach: &Reach,
) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    let FaceSurface::Plane { normal, d } = *topo.face(face)?.surface() else {
        return Ok(true);
    };
    // The box's corners, and where its edges cross the plane.
    let corner = |k: usize| {
        let pick = |axis: usize| {
            if (k >> axis) & 1 == 0 {
                reach.lo[axis]
            } else {
                reach.hi[axis]
            }
        };
        reach.origin + reach.axes[0] * pick(0) + reach.axes[1] * pick(1) + reach.axes[2] * pick(2)
    };
    let level = |p: Point3| normal.dot(to_vec(p)) - d;
    let mut cut: Vec<Point3> = Vec::new();
    for a in 0..8_usize {
        for axis in 0..3 {
            if (a >> axis) & 1 == 1 {
                continue;
            }
            let b = a | (1 << axis);
            let (p, q) = (corner(a), corner(b));
            let (lp, lq) = (level(p), level(q));
            if lp.abs() <= tol {
                cut.push(p);
            }
            if (lp < -tol && lq > tol) || (lp > tol && lq < -tol) {
                cut.push(p + (q - p) * (lp / (lp - lq)));
            }
        }
    }
    if cut.is_empty() {
        return Ok(false);
    }
    let Some(flat) = Flat::about(topo, face, cut[0])? else {
        return Ok(true);
    };
    let region = convex_hull(&cut.iter().map(|&p| flat.point(p)).collect::<Vec<_>>());
    let data = topo.face(face)?;
    for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let Some(pieces) = flat.edge(topo, oe.edge())? else {
                return Ok(true);
            };
            if pieces
                .into_iter()
                .any(|p| piece_meets_region(p, &region, tol))
            {
                return Ok(true);
            }
        }
    }
    // No edge meets the polygon: it lies wholly inside the face or outside.
    let frame = Frame3::from_normal(cut[0], normal)?;
    let Some(boundary) = face_boundary_2d(topo, face, cut[0], frame.x, frame.y)? else {
        return Ok(true);
    };
    Ok(point_in_region(&boundary, Point2::new(0.0, 0.0), tol) != Some(false))
}

/// Whether every face of the solid the chain does not rebuild stays clear
/// of the material its chamfers take or fill. A face whose box, in each
/// piece's frame and each corner's, stays out of that piece's or corner's
/// reach cannot meet it. A face that reaches in, such as a hole drilled
/// from elsewhere up into a chamfer's wedge, refuses the chain.
fn wedges_are_clear(
    topo: &Topology,
    plan: &Plan,
    faces: &[FaceId],
) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    let mut own: HashSet<FaceId> = HashSet::new();
    let mut reaches: Vec<Reach> = Vec::new();
    // Each piece's own frame, which turns with the solid.
    let mut frames: Vec<[Vec3; 3]> = Vec::with_capacity(plan.pieces.len());
    for (index, piece) in plan.pieces.iter().enumerate() {
        own.extend([piece.cap, piece.wall]);
        let reach = piece.reach() + tol;
        match &piece.curve {
            EdgeCurve::Circle(c) => {
                frames.push([c.u_axis(), c.v_axis(), c.normal()]);
                // In stretches of at most 15 degrees, each a box along its
                // chord, radially out from the axis, and along the axis.
                let (t0, t1) = piece
                    .curve
                    .domain_with_endpoints(piece.start_point, piece.end_point);
                let count = ((t1 - t0) / (PI / 12.0)).ceil().max(1.0);
                let step = (t1 - t0) / count;
                let half = 0.5 * step;
                let radius = c.radius();
                let mut mid = t0 + half;
                for _ in 0..count as usize {
                    let out = c.u_axis() * mid.cos() + c.v_axis() * mid.sin();
                    let along = c.normal().cross(out);
                    let width = radius * half.sin() + reach;
                    reaches.push(Reach {
                        origin: c.center(),
                        axes: [along, out, c.normal()],
                        lo: [-width, radius * half.cos() - reach, -reach],
                        hi: [width, radius + reach, reach],
                        pieces: vec![index],
                    });
                    mid += step;
                }
            }
            EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                let run = piece.end_point - piece.start_point;
                let along = run.normalize()?;
                let into = plan.piece_sections[index][0].cap_dir;
                frames.push([along, into, along.cross(into)]);
                reaches.push(Reach {
                    origin: piece.start_point,
                    axes: [along, into, along.cross(into)],
                    lo: [-reach; 3],
                    hi: [run.length() + reach, reach, reach],
                    pieces: vec![index],
                });
            }
        }
    }
    own.extend(plan.joints.iter().flat_map(|j| j.walls));
    own.extend(plan.ends.iter().flat_map(|e| [e.wall, e.beyond]));
    own.extend(plan.plane_ends.iter().map(|e| e.face));
    for corner in &plan.corners {
        own.extend([corner.shared, corner.faces[0], corner.faces[1]]);
        let at = topo.vertex(corner.vertex)?.point();
        let radius = [Some(corner.meet), Some(corner.reach), corner.cut]
            .into_iter()
            .flatten()
            .map(|p| (p - at).length())
            .fold(corner_reach(&plan.pieces, corner), f64::max)
            + tol;
        reaches.push(Reach {
            origin: at,
            axes: frames[corner.pieces[0]],
            lo: [-radius; 3],
            hi: [radius; 3],
            pieces: corner.pieces.to_vec(),
        });
    }
    // A slanted end's contacts can run on past its section.
    for end in plan.plane_ends.iter().filter(|e| e.slanted) {
        let radius = end_reach(topo, &plan.pieces[end.piece], end)? + tol;
        reaches.push(Reach {
            origin: topo.vertex(end.vertex)?.point(),
            axes: frames[end.piece],
            lo: [-radius; 3],
            hi: [radius; 3],
            pieces: vec![end.piece],
        });
    }
    // Each piece's chamfer face, and the sign of its level on the side it
    // takes or fills, which holds the edge, where the other side is convex:
    // either side of a plane, or the inside of a cone.
    let mut kept: Vec<Option<(Band, f64)>> = Vec::with_capacity(plan.pieces.len());
    for (index, piece) in plan.pieces.iter().enumerate() {
        let [a, b] = plan.piece_sections[index];
        let band = piece.band(&a, &b)?;
        let removed = band.level(piece.midpoint()).signum();
        let convex_kept = match band {
            Band::Plane { .. } => true,
            Band::Cone(_) => removed > 0.0,
        };
        kept.push(convex_kept.then_some((band, removed)));
    }
    for &face in faces {
        if own.contains(&face) {
            continue;
        }
        let mut hull: Option<Option<Vec<Point3>>> = None;
        for reach in &reaches {
            match reach.face_box(topo, face)? {
                Some(bounds) if !reach.meets(bounds) => continue,
                Some(_) if !plane_meets_reach(topo, face, reach)? => continue,
                _ => {}
            }
            // The face's hull lies on the kept side of every chamfer face
            // that bounds the region, where that side is convex.
            if hull.is_none() {
                hull = Some(face_hull_points(topo, face)?);
            }
            let Some(Some(points)) = &hull else {
                return Ok(false);
            };
            for &index in &reach.pieces {
                let Some((band, removed)) = &kept[index] else {
                    return Ok(false);
                };
                if points.iter().any(|&p| band.level(p) * removed >= -tol) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// A free end where the cap's next edge carries straight on: the chain ends
/// in a flat triangle across the edge, and the wall's ruling at the vertex
/// keeps its far part. Also returns whether the neighbour beyond is curved.
fn plan_tangent_end(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    piece: &Piece,
    section: &Section,
    vertex: VertexId,
) -> Result<Option<(End, bool)>, OperationsError> {
    let tol = Tolerance::new().linear;
    if matches!(piece.wall_side, WallSide::Ruling { .. }) {
        return Ok(None);
    }
    let leaving = piece.leaving(vertex)?;
    let next: Vec<EdgeId> = edges_at(topo, piece.cap, vertex)?
        .into_iter()
        .filter(|&e| e != piece.edge)
        .collect();
    let [next] = next[..] else {
        return Ok(None);
    };
    let next_data = topo.edge(next)?;
    let (na, nb) = (
        topo.vertex(next_data.start())?.point(),
        topo.vertex(next_data.end())?.point(),
    );
    let next_tangent = match read_curve(topo, next)? {
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
    let Some(beyond) = other_face(adjacency, next, piece.cap) else {
        return Ok(None);
    };
    let Some((ruling, length)) = ruling_along(topo, piece.wall, vertex, section.wall_dir)? else {
        return Ok(None);
    };
    if length <= piece.wall_d + tol
        || beyond == piece.wall
        || !adjacency.faces_for_edge(ruling).contains(&beyond)
    {
        return Ok(None);
    }
    let curved = plane_normal(topo, beyond)?.is_none();
    Ok(Some((
        End {
            vertex,
            ruling,
            wall: piece.wall,
            beyond,
        },
        curved,
    )))
}

/// Where along `edge` from `vertex` the point `p` on its curve lies: `1` on
/// the edge past the vertex (`p` splits it), `-1` on its curve run on back
/// past the vertex, `None` off the curve, at either end, or past the far
/// end.
fn side_along(
    topo: &Topology,
    edge: EdgeId,
    vertex: VertexId,
    p: Point3,
) -> Result<Option<f64>, OperationsError> {
    let tol = Tolerance::new().linear;
    let data = topo.edge(edge)?;
    let at = topo.vertex(vertex)?.point();
    let far = topo.vertex(far_end(topo, edge, vertex)?)?.point();
    match read_curve(topo, edge)? {
        EdgeCurve::Line => {
            let run = far - at;
            let length2 = run.dot(run);
            let s = (p - at).dot(run) / length2;
            if (at + run * s - p).length() > tol {
                return Ok(None);
            }
            let slack = tol / length2.sqrt();
            Ok(if s > slack && s < 1.0 - slack {
                Some(1.0)
            } else if s < -slack {
                Some(-1.0)
            } else {
                None
            })
        }
        EdgeCurve::Circle(c) => {
            if data.start() == data.end()
                || ((p - c.center()).length() - c.radius()).abs() > tol
                || c.normal().dot(p - c.center()).abs() > tol
            {
                return Ok(None);
            }
            // The arc runs counter-clockwise from its start to its end.
            let (from, to, forward) = if data.start() == vertex {
                (c.project(at), c.project(far), true)
            } else {
                (c.project(far), c.project(at), false)
            };
            let span = (to - from).rem_euclid(TAU);
            let slack = tol / c.radius();
            let off = if forward {
                (c.project(p) - from).rem_euclid(TAU)
            } else {
                (to - c.project(p)).rem_euclid(TAU)
            };
            Ok(if off > slack && off < span - slack {
                Some(1.0)
            } else if off > span + slack && off > PI {
                Some(-1.0)
            } else {
                None
            })
        }
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => Ok(None),
    }
}

/// A free end at a plane through the vertex (see [`PlaneEnd`]).
#[allow(clippy::too_many_lines)]
fn plan_plane_end(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    index: usize,
    piece: &Piece,
    sections: &[Section; 2],
    vertex: VertexId,
) -> Result<Option<PlaneEnd>, OperationsError> {
    let tol = Tolerance::new().linear;
    let section = sections[piece.slot(vertex)];
    let others = |face: FaceId| -> Result<Vec<EdgeId>, OperationsError> {
        Ok(edges_at(topo, face, vertex)?
            .into_iter()
            .filter(|&e| e != piece.edge)
            .collect())
    };
    let (cap_edges, wall_edges) = (others(piece.cap)?, others(piece.wall)?);
    let ([cap_edge], [wall_edge]) = (&cap_edges[..], &wall_edges[..]) else {
        return Ok(None);
    };
    let (Some(face), Some(also)) = (
        other_face(adjacency, *cap_edge, piece.cap),
        other_face(adjacency, *wall_edge, piece.wall),
    ) else {
        return Ok(None);
    };
    let Some(normal) = plane_normal(topo, face)? else {
        return Ok(None);
    };
    if face != also {
        return Ok(None);
    }
    let edges = [*cap_edge, *wall_edge];
    let at = piece.at(vertex);
    let slanted = normal.cross(piece.tangent(at)?).length() > JOIN_TOL;
    let (points, cross) = if slanted {
        // Each contact runs on, or back, along its track to the plane's
        // straight edge on its face.
        let tracks = piece.tracks(at, &section)?;
        let mut points = [section.cap, section.wall];
        for (side, track) in tracks.iter().enumerate() {
            if !matches!(read_curve(topo, edges[side])?, EdgeCurve::Line) {
                return Ok(None);
            }
            let far = topo.vertex(far_end(topo, edges[side], vertex)?)?.point();
            let Some(p) = track.meets_line(at, far - at, points[side], tol) else {
                return Ok(None);
            };
            points[side] = p;
        }
        let plane = Band::Plane {
            normal,
            d: normal.dot(to_vec(at)),
        };
        let band = piece.band(&sections[0], &sections[1])?;
        let Some(cross) = crossing_curve(&band, &plane, points[0], points[1]) else {
            return Ok(None);
        };
        (points, cross)
    } else {
        ([section.cap, section.wall], EdgeCurve::Line)
    };
    let (Some(cap_side), Some(wall_side)) = (
        side_along(topo, edges[0], vertex, points[0])?,
        side_along(topo, edges[1], vertex, points[1])?,
    ) else {
        return Ok(None);
    };
    let split = [cap_side > 0.0, wall_side > 0.0];
    if (!split[0] && !split[1]) || (slanted && !(split[0] && split[1])) {
        return Ok(None);
    }
    // The cross edge crosses no other edge of the plane: a square end's
    // segment here, a slanted end's curve within its region's hull later.
    if !slanted {
        let cross = Span::Segment(points[0], points[1]);
        let data = topo.face(face)?;
        for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                if edges.contains(&oe.edge()) {
                    continue;
                }
                let e = topo.edge(oe.edge())?;
                let curve = read_curve(topo, oe.edge())?;
                let Some(span) = Span::of(
                    &curve,
                    topo.vertex(e.start())?.point(),
                    topo.vertex(e.end())?.point(),
                ) else {
                    return Ok(None);
                };
                if !cross.touches(&span, tol).is_empty() {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(PlaneEnd {
        vertex,
        piece: index,
        face,
        edges,
        split,
        points,
        cross,
        slanted,
    }))
}

/// How far from its vertex a slanted end's chamfer can reach: the piece's
/// reach, the contacts' ends and the poles of the curve between them.
fn end_reach(topo: &Topology, piece: &Piece, end: &PlaneEnd) -> Result<f64, OperationsError> {
    let at = topo.vertex(end.vertex)?.point();
    let poles = match &end.cross {
        EdgeCurve::NurbsCurve(curve) => curve.control_points().to_vec(),
        EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => Vec::new(),
    };
    Ok(end
        .points
        .iter()
        .chain(&poles)
        .map(|&p| (p - at).length())
        .fold(piece.reach(), f64::max))
}

/// The farther reach of a corner's two pieces.
fn corner_reach(pieces: &[Piece], corner: &Corner) -> f64 {
    pieces[corner.pieces[0]]
        .reach()
        .max(pieces[corner.pieces[1]].reach())
}

/// A corner between pieces `a` and `b` at `vertex` (see [`Corner`]).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn plan_corner(
    topo: &Topology,
    pieces: &[Piece],
    piece_sections: &[[Section; 2]],
    a: usize,
    b: usize,
    vertex: VertexId,
) -> Result<Option<Corner>, OperationsError> {
    let tol = Tolerance::new().linear;
    let (pa, pb) = (&pieces[a], &pieces[b]);
    let shared: Vec<FaceId> = [pa.cap, pa.wall]
        .into_iter()
        .filter(|f| *f == pb.cap || *f == pb.wall)
        .collect();
    let [shared] = shared[..] else {
        return Ok(None);
    };
    let other_of = |p: &Piece| if p.cap == shared { p.wall } else { p.cap };
    let (fa, fb) = (other_of(pa), other_of(pb));
    if fa == fb {
        return Ok(None);
    }
    // The vertex's edges are the two pieces and the third, between fa and fb.
    let at_shared = edges_at(topo, shared, vertex)?;
    let (at_a, at_b) = (edges_at(topo, fa, vertex)?, edges_at(topo, fb, vertex)?);
    let thirds: Vec<EdgeId> = at_a
        .iter()
        .copied()
        .filter(|e| *e != pa.edge && at_b.contains(e))
        .collect();
    let [third] = thirds[..] else {
        return Ok(None);
    };
    if at_shared.len() != 2 || at_a.len() != 2 || at_b.len() != 2 || third == pb.edge {
        return Ok(None);
    }
    if !matches!(read_curve(topo, third)?, EdgeCurve::Line) {
        return Ok(None);
    }
    let at = topo.vertex(vertex)?.point();
    let far = topo.vertex(far_end(topo, third, vertex)?)?.point();
    let run = far - at;

    let section_at = |index: usize, piece: &Piece| piece_sections[index][piece.slot(vertex)];
    let (sa, sb) = (section_at(a, pa), section_at(b, pb));
    let tracks_a = pa.tracks(at, &sa)?;
    let tracks_b = pb.tracks(at, &sb)?;
    // Each piece's track on the shared face, then on its other face.
    let split = |p: &Piece, tracks: [Track; 2]| {
        let [cap, wall] = tracks;
        if p.cap == shared {
            (cap, wall)
        } else {
            (wall, cap)
        }
    };
    let (shared_a, other_a) = split(pa, tracks_a);
    let (shared_b, other_b) = split(pb, tracks_b);
    let Some(meet) = shared_a.meets(&shared_b, at, tol) else {
        return Ok(None);
    };
    if (meet - at).length() > 10.0 * pa.reach().max(pb.reach()) {
        return Ok(None);
    }
    let on_third = |track: &Track| -> Option<(Point3, f64)> {
        let p = track.meets_line(at, run, at, tol)?;
        let s = (p - at).dot(run) / run.dot(run);
        (s > tol / run.length() && s < 1.0).then_some((p, s))
    };
    let (Some((reach_a, s_a)), Some((reach_b, s_b))) = (on_third(&other_a), on_third(&other_b))
    else {
        return Ok(None);
    };

    let bands = [
        pa.band(&piece_sections[a][0], &piece_sections[a][1])?,
        pb.band(&piece_sections[b][0], &piece_sections[b][1])?,
    ];
    let (order, reach, cut, sliver) = if (reach_a - reach_b).length() <= JOIN_TOL {
        ([a, b], reach_a + (reach_b - reach_a) * 0.5, None, None)
    } else {
        // The farther-reaching chamfer cuts the other's contact on its face.
        let (first, second, reach, near) = if s_a > s_b {
            (a, b, reach_a, reach_b)
        } else {
            (b, a, reach_b, reach_a)
        };
        let far_band = &bands[usize::from(first == b)];
        let near_piece = &pieces[second];
        // Along the second piece's contact on its own face, from where it
        // reaches the third edge back to its other end.
        let other_end = {
            let s = piece_sections[second][1 - near_piece.slot(vertex)];
            if near_piece.cap == shared {
                s.wall
            } else {
                s.cap
            }
        };
        let near_track = split(
            near_piece,
            near_piece.tracks(at, &section_at(second, near_piece))?,
        )
        .1;
        let along = |t: f64| -> Point3 {
            match &near_track {
                Track::Line { .. } => near + (other_end - near) * t,
                Track::Circle(c) => {
                    let from = c.project(near);
                    let to = c.project(other_end);
                    let sweep = if near_piece.slot(vertex) == 0 {
                        (to - from).rem_euclid(TAU)
                    } else {
                        -(from - to).rem_euclid(TAU)
                    };
                    c.evaluate(from + sweep * t)
                }
            }
        };
        let Some(t) = root_between(|t| far_band.level(along(t))) else {
            return Ok(None);
        };
        let cut = along(t);
        let Some(normal) = plane_normal(topo, other_of(near_piece))? else {
            return Ok(None);
        };
        let face_band = Band::Plane {
            normal,
            d: normal.dot(to_vec(cut)),
        };
        let Some(sliver) = crossing_curve(far_band, &face_band, reach, cut) else {
            return Ok(None);
        };
        ([first, second], reach, Some(cut), Some(sliver))
    };
    let faces = order.map(|index| other_of(&pieces[index]));
    let from = cut.unwrap_or(reach);
    if (from - meet).length() <= tol {
        return Ok(None);
    }
    let Some(mitre) = crossing_curve(&bands[0], &bands[1], from, meet) else {
        return Ok(None);
    };
    Ok(Some(Corner {
        vertex,
        shared,
        third,
        pieces: order,
        faces,
        meet,
        reach,
        cut,
        mitre,
        sliver,
    }))
}

/// The contact piece `index` leaves on its cap (`side` 0) or on its wall
/// (`side` 1), between the points where it ends. `None` for a piece that is
/// not a line or an arc.
fn contact_span(plan: &Plan, index: usize, side: usize) -> Result<Option<Span>, OperationsError> {
    let piece = &plan.pieces[index];
    let [s0, s1] = plan.piece_sections[index];
    let pick = |(cap, wall): (Point3, Point3)| if side == 0 { cap } else { wall };
    let a = pick(plan.contact_ends(index, piece.start));
    let b = pick(plan.contact_ends(index, piece.end));
    let (at_a, at_b) = if side == 0 {
        (s0.cap, s1.cap)
    } else {
        (s0.wall, s1.wall)
    };
    Ok(match (&piece.curve, &piece.contact_curves(&s0)?[side]) {
        (EdgeCurve::Line, _) => Some(Span::Segment(a, b)),
        (EdgeCurve::Circle(_), EdgeCurve::Circle(circle)) => Some(Span::Arc {
            sweep: contact_sweep(piece, circle, (a, at_a), (b, at_b)),
            t0: circle.project(a),
            circle: circle.clone(),
        }),
        _ => None,
    })
}

/// Whether two chains, each clear on its own, take strips of one face that
/// overlap: a contact or a tangent end's cross edge of one runs into the
/// other's strip, or two contacts touch. A chain cannot clear another's
/// strip, which the other's chamfer takes. Only a span and a strip, or two
/// spans, whose boxes meet are read, so the work grows with the pieces that
/// lie near each other rather than with every pair; `tests` counts the
/// pairs read.
fn chains_overlap(plans: &[Plan], tests: &mut usize) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    // Each new span with its chain and face, and each strip with its chain,
    // piece and face.
    let mut spans: Vec<(usize, FaceId, Span)> = Vec::new();
    let mut strips: Vec<(usize, usize, FaceId)> = Vec::new();
    let mut strip_boxes: Vec<(usize, Aabb3)> = Vec::new();
    for (owner, plan) in plans.iter().enumerate() {
        for (index, piece) in plan.pieces.iter().enumerate() {
            for (side, face) in [piece.cap, piece.wall].into_iter().enumerate() {
                if let Some(span) = contact_span(plan, index, side)? {
                    spans.push((owner, face, span));
                }
                strip_boxes.push((strips.len(), plan.strip_box(index, side)?));
                strips.push((owner, index, face));
            }
        }
        for end in &plan.ends {
            if let (Some(section), Some(piece)) = (
                plan.section(end.vertex),
                plan.pieces
                    .iter()
                    .find(|p| p.start == end.vertex || p.end == end.vertex),
            ) {
                let at = piece.at(end.vertex);
                spans.push((owner, piece.cap, Span::Segment(at, section.cap)));
            }
        }
    }
    let span_boxes: Vec<(usize, Aabb3)> = spans
        .iter()
        .enumerate()
        .map(|(k, (_, _, span))| (k, span.bounds().expanded(tol)))
        .collect();
    let (strip_tree, span_tree) = (Bvh::build(&strip_boxes), Bvh::build(&span_boxes));
    let mut found = Vec::new();
    for (k, (owner, face, span)) in spans.iter().enumerate() {
        let bounds = span_boxes[k].1;
        strip_tree.query_overlap_into(&bounds, &mut found);
        let mut samples = None;
        for &s in &found {
            let (other, index, strip_face) = strips[s];
            if other == *owner || strip_face != *face {
                continue;
            }
            *tests += 1;
            for &p in samples.get_or_insert_with(|| span.samples()).iter() {
                if plans[other].takes(index, *face, p, tol)? {
                    return Ok(true);
                }
            }
        }
        span_tree.query_overlap_into(&bounds, &mut found);
        for &m in &found {
            let (other, other_face, other_span) = &spans[m];
            if m <= k || other == owner || other_face != face {
                continue;
            }
            *tests += 1;
            if !span.touches(other_span, tol).is_empty() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether a plan's new boundary meets nothing it must not, on its caps, on
/// its walls, and against itself.
#[allow(clippy::too_many_lines)]
fn plan_is_clear(topo: &Topology, plan: &Plan) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    if !contacts_keep_order(plan)? {
        return Ok(false);
    }
    // The new boundary on each cap: each contact, a sliver's chord, and each
    // tangent end's cross edge, with where each may touch the cap's edges.
    let mut new_spans: Vec<(FaceId, Span, Vec<Point3>, Option<usize>)> = Vec::new();
    for (index, piece) in plan.pieces.iter().enumerate() {
        let mut allowed = Vec::new();
        for vertex in [piece.start, piece.end] {
            if let Some(corner) = plan.corner(vertex)
                && corner.shared != piece.cap
            {
                allowed.push(plan.contact_ends(index, vertex).0);
            }
            if plan.plane_end(vertex).is_some() {
                allowed.push(plan.contact_ends(index, vertex).0);
            }
        }
        let Some(span) = contact_span(plan, index, 0)? else {
            return Ok(false);
        };
        new_spans.push((piece.cap, span, allowed, Some(index)));
    }
    for end in &plan.ends {
        let Some(section) = plan.section(end.vertex) else {
            return Ok(false);
        };
        let at = topo.vertex(end.vertex)?.point();
        let Some(piece) = plan
            .pieces
            .iter()
            .find(|p| p.start == end.vertex || p.end == end.vertex)
        else {
            return Ok(false);
        };
        new_spans.push((piece.cap, Span::Segment(at, section.cap), vec![at], None));
    }
    // Nor may it meet itself away from where its pieces join: across a narrow
    // neck, the two sides' new edges cross.
    for (i, (cap, first, _, _)) in new_spans.iter().enumerate() {
        for (other_cap, second, _, _) in &new_spans[i + 1..] {
            if cap != other_cap {
                continue;
            }
            for p in first.touches(second, tol) {
                let joins = |span: &Span| {
                    span.ends()
                        .iter()
                        .any(|&end| (p - end).length() <= 10.0 * tol)
                };
                if !(joins(first) && joins(second)) {
                    return Ok(false);
                }
            }
        }
    }

    // Each cap's other edges: what is left of a corner's third edge or a
    // plane end's edge past the point the chamfer moves its end to.
    let mut kept_from: HashMap<EdgeId, Vec<(VertexId, Point3)>> = HashMap::new();
    for corner in &plan.corners {
        kept_from
            .entry(corner.third)
            .or_default()
            .push((corner.vertex, corner.reach));
    }
    for end in &plan.plane_ends {
        if end.split[0] {
            kept_from
                .entry(end.edges[0])
                .or_default()
                .push((end.vertex, end.points[0]));
        }
    }
    let piece_edges: HashSet<EdgeId> = plan.pieces.iter().map(|p| p.edge).collect();
    let caps: Vec<FaceId> = {
        let mut caps: Vec<FaceId> = plan.pieces.iter().map(|p| p.cap).collect();
        caps.sort_by_key(|f| f.index());
        caps.dedup();
        caps
    };
    for &cap in &caps {
        let cap_data = topo.face(cap)?;
        for wid in
            std::iter::once(cap_data.outer_wire()).chain(cap_data.inner_wires().iter().copied())
        {
            for oe in topo.wire(wid)?.edges() {
                if piece_edges.contains(&oe.edge()) {
                    continue;
                }
                let edge = topo.edge(oe.edge())?;
                let mut a = topo.vertex(edge.start())?.point();
                let mut b = topo.vertex(edge.end())?.point();
                let curve = read_curve(topo, oe.edge())?;
                let Some(span) = Span::of(&curve, a, b) else {
                    return Ok(false);
                };
                for (new_cap, new_span, allowed, _) in &new_spans {
                    if *new_cap != cap {
                        continue;
                    }
                    for p in new_span.touches(&span, tol) {
                        if !allowed.iter().any(|at| (p - *at).length() <= 10.0 * tol) {
                            return Ok(false);
                        }
                    }
                }
                if let Some(cuts) = kept_from.get(&oe.edge()) {
                    if !matches!(curve, EdgeCurve::Line) {
                        return Ok(false);
                    }
                    for &(vertex, to) in cuts {
                        if vertex == edge.start() {
                            a = to;
                        } else {
                            b = to;
                        }
                    }
                }
                let cut_to: Vec<Point3> = kept_from
                    .get(&oe.edge())
                    .map(|cuts| cuts.iter().map(|&(_, to)| to).collect())
                    .unwrap_or_default();
                for p in edge_samples(&curve, a, b) {
                    // A cut end lies on the contact, the strip's edge.
                    if cut_to.iter().any(|&to| (p - to).length() <= 10.0 * tol) {
                        continue;
                    }
                    if plan.strip_holds(cap, p, tol)? {
                        return Ok(false);
                    }
                }
            }
        }
    }

    // Nor on a wall, which keeps every other edge and hole where it was.
    let mut planned: HashSet<EdgeId> = piece_edges;
    planned.extend(plan.joints.iter().map(|j| j.ruling));
    planned.extend(plan.ends.iter().map(|e| e.ruling));
    planned.extend(plan.corners.iter().map(|c| c.third));
    planned.extend(plan.plane_ends.iter().flat_map(|e| e.edges));
    for (index, piece) in plan.pieces.iter().enumerate() {
        let section = plan.piece_sections[index][0];
        let wall = topo.face(piece.wall)?;
        if !wall.inner_wires().is_empty() {
            return Ok(false);
        }
        if let WallSide::Plane { .. } = piece.wall_side {
            // A line stays clear if its ends do, and a circle on the wall's
            // cylinder runs at one height.
            for oe in topo.wire(wall.outer_wire())?.edges() {
                if planned.contains(&oe.edge()) {
                    continue;
                }
                let edge = topo.edge(oe.edge())?;
                match read_curve(topo, oe.edge())? {
                    EdgeCurve::Line => {}
                    EdgeCurve::Circle(c) => {
                        if c.normal().cross(section.wall_dir).length() > JOIN_TOL {
                            return Ok(false);
                        }
                    }
                    EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(false),
                }
                for v in [edge.start(), edge.end()] {
                    let height =
                        (topo.vertex(v)?.point() - piece.start_point).dot(section.wall_dir);
                    if height <= piece.wall_d + tol {
                        return Ok(false);
                    }
                }
            }
        } else if !cylinder_wall_clear(topo, piece, &planned)? {
            return Ok(false);
        }
    }
    // Nor may a corner's region, or a plane end's, hold anything else of
    // their faces.
    for corner in &plan.corners {
        if !corner_is_clear(topo, plan, corner, &new_spans)? {
            return Ok(false);
        }
    }
    for end in &plan.plane_ends {
        if !plane_end_is_clear(topo, plan, end, &new_spans)? {
            return Ok(false);
        }
    }
    // Nor may two strips on one wall overlap, but where their edges meet.
    for i in 0..plan.pieces.len() {
        for j in i + 1..plan.pieces.len() {
            let (a, b) = (&plan.pieces[i], &plan.pieces[j]);
            let meet = [a.start, a.end]
                .iter()
                .any(|v| *v == b.start || *v == b.end);
            if a.wall == b.wall && !meet && wall_strips_overlap(topo, plan, i, j)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// How far round `circle`, the way the arc `piece` runs, a contact turns
/// from `a` to `b`, which lie near its sections' points `at_a` and `at_b`. A
/// corner can move an end back or on from its section by up to a half turn.
fn contact_sweep(
    piece: &Piece,
    circle: &Circle3D,
    (a, at_a): (Point3, Point3),
    (b, at_b): (Point3, Point3),
) -> f64 {
    let (t0, t1) = piece
        .curve
        .domain_with_endpoints(piece.start_point, piece.end_point);
    let moved =
        |p: Point3, at: Point3| (circle.project(p) - circle.project(at) + PI).rem_euclid(TAU) - PI;
    (t1 - t0) + moved(b, at_b) - moved(a, at_a)
}

/// Whether each piece's contacts still run the way its edge runs between
/// the points where they end. A corner can move an end back or on, but a
/// contact whose ends pass each other would fold its chamfer over.
fn contacts_keep_order(plan: &Plan) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    for (index, piece) in plan.pieces.iter().enumerate() {
        let [s0, s1] = plan.piece_sections[index];
        let (cap_a, wall_a) = plan.contact_ends(index, piece.start);
        let (cap_b, wall_b) = plan.contact_ends(index, piece.end);
        let curves = piece.contact_curves(&s0)?;
        let sides = [
            ((cap_a, s0.cap), (cap_b, s1.cap)),
            ((wall_a, s0.wall), (wall_b, s1.wall)),
        ];
        for (curve, (a, b)) in curves.iter().zip(sides) {
            let keeps = match curve {
                EdgeCurve::Circle(circle) => {
                    let sweep = contact_sweep(piece, circle, a, b);
                    sweep > tol / circle.radius() && sweep < TAU
                }
                EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                    (b.0 - a.0).dot(piece.tangent(piece.start_point)?) > tol
                }
            };
            if !keeps {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// A face laid flat about a point: a plane in its own frame, or a cylinder
/// unrolled about the point's ruling, as the arc length round the axis from
/// that ruling and the height along it. Lines on the plane and the
/// cylinder's rulings and circles about its axis lie flat as segments.
enum Flat {
    Plane {
        origin: Point3,
        x: Vec3,
        y: Vec3,
        normal: Vec3,
    },
    Cylinder {
        cylinder: CylindricalSurface,
        u0: f64,
    },
}

impl Flat {
    /// `None` for a face that is neither a plane nor a cylinder.
    fn about(topo: &Topology, face: FaceId, at: Point3) -> Result<Option<Self>, OperationsError> {
        Ok(match topo.face(face)?.surface() {
            FaceSurface::Plane { normal, .. } => {
                let frame = Frame3::from_normal(at, *normal)?;
                Some(Self::Plane {
                    origin: at,
                    x: frame.x,
                    y: frame.y,
                    normal: *normal,
                })
            }
            FaceSurface::Cylinder(cylinder) => Some(Self::Cylinder {
                u0: cylinder.project_point(at).0,
                cylinder: cylinder.clone(),
            }),
            FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_) => None,
        })
    }

    fn point(&self, p: Point3) -> [f64; 2] {
        match self {
            Self::Plane { origin, x, y, .. } => [(p - *origin).dot(*x), (p - *origin).dot(*y)],
            Self::Cylinder { cylinder, u0 } => {
                let (u, h) = cylinder.project_point(p);
                [((u - u0 + PI).rem_euclid(TAU) - PI) * cylinder.radius(), h]
            }
        }
    }

    /// `edge` laid flat, or `None` where it does not lie flat as segments
    /// and arcs. A circle round a cylinder's axis lies flat as its run and
    /// that run a whole turn either way, so a region near the unrolled
    /// ruling meets it wherever it does.
    fn edge(&self, topo: &Topology, edge: EdgeId) -> Result<Option<Vec<Piece2>>, OperationsError> {
        let data = topo.edge(edge)?;
        let (a, b) = (
            topo.vertex(data.start())?.point(),
            topo.vertex(data.end())?.point(),
        );
        let curve = read_curve(topo, edge)?;
        Ok(match (self, &curve) {
            (Self::Plane { .. }, EdgeCurve::Line) => {
                Some(vec![Piece2::Segment(self.point(a), self.point(b))])
            }
            (Self::Plane { normal, .. }, EdgeCurve::Circle(c)) => {
                if c.normal().cross(*normal).length() > JOIN_TOL {
                    return Ok(None);
                }
                let (t0, t1) = if data.start() == data.end() {
                    (0.0, TAU)
                } else {
                    curve.domain_with_endpoints(a, b)
                };
                let turn = if c.normal().dot(*normal) > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let center = self.point(c.center());
                let start = self.point(a);
                Some(vec![Piece2::Arc {
                    center,
                    radius: c.radius(),
                    from: (start[1] - center[1]).atan2(start[0] - center[0]),
                    sweep: (t1 - t0) * turn,
                }])
            }
            (Self::Cylinder { cylinder, .. }, EdgeCurve::Line) => {
                let run = b - a;
                if run.length() > JOIN_TOL
                    && run.normalize()?.cross(cylinder.axis()).length() > JOIN_TOL
                {
                    return Ok(None);
                }
                Some(vec![Piece2::Segment(self.point(a), self.point(b))])
            }
            (Self::Cylinder { cylinder, .. }, EdgeCurve::Circle(c)) => {
                let axis = cylinder.axis();
                let coaxial = c.normal().cross(axis).length() < JOIN_TOL
                    && (c.center() - cylinder.origin()).cross(axis).length() < JOIN_TOL;
                if !coaxial {
                    return Ok(None);
                }
                let (t0, t1) = if data.start() == data.end() {
                    (0.0, TAU)
                } else {
                    curve.domain_with_endpoints(a, b)
                };
                let turn = if c.normal().dot(axis) > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let [x0, h] = self.point(a);
                let run = (t1 - t0) * turn * cylinder.radius();
                let whole = TAU * cylinder.radius();
                Some(
                    [-whole, 0.0, whole]
                        .iter()
                        .map(|shift| Piece2::Segment([x0 + shift, h], [x0 + shift + run, h]))
                        .collect(),
                )
            }
            _ => None,
        })
    }
}

/// A segment or a circular arc in a flat face.
#[derive(Clone, Copy)]
enum Piece2 {
    Segment([f64; 2], [f64; 2]),
    Arc {
        center: [f64; 2],
        radius: f64,
        from: f64,
        sweep: f64,
    },
}

/// The convex hull of `points`, counter-clockwise.
fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
    sorted.dedup_by(|p, q| (p[0] - q[0]).hypot(p[1] - q[1]) <= 1e-15);
    if sorted.len() < 3 {
        return sorted;
    }
    let turn = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(2 * sorted.len());
    for pass in [false, true] {
        let start = hull.len();
        let ordered: Vec<[f64; 2]> = if pass {
            sorted.iter().rev().copied().collect()
        } else {
            sorted.clone()
        };
        for p in ordered {
            while hull.len() >= start + 2
                && turn(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// The distance from `p` to the segment from `a` to `b`.
fn to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length2 = dx * dx + dy * dy;
    let t = if length2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}

/// Whether `p` lies in the convex polygon `region`, or within `tol` of it.
fn region_holds(region: &[[f64; 2]], p: [f64; 2], tol: f64) -> bool {
    let n = region.len();
    if n == 0 {
        return false;
    }
    if n < 3 {
        let b = region[n - 1];
        return to_segment(p, region[0], b) <= tol;
    }
    let inside = (0..n).all(|k| {
        let (a, b) = (region[k], region[(k + 1) % n]);
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0
    });
    inside || (0..n).any(|k| to_segment(p, region[k], region[(k + 1) % n]) <= tol)
}

/// Whether two segments come within `tol` of each other.
fn segments_meet(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2], tol: f64) -> bool {
    let cross = |o: [f64; 2], p: [f64; 2], q: [f64; 2]| {
        (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    };
    let (d1, d2) = (cross(c, d, a), cross(c, d, b));
    let (d3, d4) = (cross(a, b, c), cross(a, b, d));
    if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
        return true;
    }
    to_segment(a, c, d) <= tol
        || to_segment(b, c, d) <= tol
        || to_segment(c, a, b) <= tol
        || to_segment(d, a, b) <= tol
}

/// Whether `piece` comes within `tol` of the convex polygon `region`: an
/// end of it lies there, or it meets a side. An arc meets a side where the
/// side passes within `tol` of its circle inside its sweep.
fn piece_meets_region(piece: Piece2, region: &[[f64; 2]], tol: f64) -> bool {
    let n = region.len();
    let sides: Vec<([f64; 2], [f64; 2])> = match n {
        0 => return false,
        1 => vec![(region[0], region[0])],
        2 => vec![(region[0], region[1])],
        _ => (0..n).map(|k| (region[k], region[(k + 1) % n])).collect(),
    };
    match piece {
        Piece2::Segment(a, b) => {
            region_holds(region, a, tol)
                || region_holds(region, b, tol)
                || sides.iter().any(|&(c, d)| segments_meet(a, b, c, d, tol))
        }
        Piece2::Arc {
            center,
            radius,
            from,
            sweep,
        } => {
            let at = |t: f64| [center[0] + radius * t.cos(), center[1] + radius * t.sin()];
            if region_holds(region, at(from), tol) || region_holds(region, at(from + sweep), tol) {
                return true;
            }
            let slack = tol / radius;
            let in_sweep = |p: [f64; 2]| {
                let t = (p[1] - center[1]).atan2(p[0] - center[0]);
                let off = if sweep >= 0.0 {
                    (t - from).rem_euclid(TAU)
                } else {
                    (from - t).rem_euclid(TAU)
                };
                off <= sweep.abs() + slack || off >= TAU - slack
            };
            sides.iter().any(|&(c, d)| {
                // The points of the side nearest and farthest from the
                // circle's centre, and where it crosses the circle.
                let (dx, dy) = (d[0] - c[0], d[1] - c[1]);
                let (wx, wy) = (c[0] - center[0], c[1] - center[1]);
                let a2 = dx * dx + dy * dy;
                let mut candidates = vec![c, d];
                if a2 > 0.0 {
                    let t = (-(wx * dx + wy * dy) / a2).clamp(0.0, 1.0);
                    candidates.push([c[0] + t * dx, c[1] + t * dy]);
                    let b = 2.0 * (wx * dx + wy * dy);
                    let disc = b * b - 4.0 * a2 * (wx * wx + wy * wy - radius * radius);
                    if disc >= 0.0 {
                        for root in [
                            (-b - disc.sqrt()) / (2.0 * a2),
                            (-b + disc.sqrt()) / (2.0 * a2),
                        ] {
                            if (0.0..=1.0).contains(&root) {
                                candidates.push([c[0] + root * dx, c[1] + root * dy]);
                            }
                        }
                    }
                }
                candidates.into_iter().any(|p| {
                    ((p[0] - center[0]).hypot(p[1] - center[1]) - radius).abs() <= tol
                        && in_sweep(p)
                })
            })
        }
    }
}

/// Where the tangents to `circle` at `a` and `b` meet, on the bisector of
/// the shorter arc between them: with its ends, a hull holding that arc.
/// `None` when the arc turns a quarter or more.
fn arc_pole(circle: &Circle3D, a: Point3, b: Point3) -> Option<Point3> {
    let (ra, rb) = (a - circle.center(), b - circle.center());
    let turn = ra.cross(rb).length().atan2(ra.dot(rb));
    if turn >= 0.5 * PI {
        return None;
    }
    let bisector = (ra + rb).normalize().ok()?;
    Some(circle.center() + bisector * (circle.radius() / (0.5 * turn).cos()))
}

/// Whether every edge of `face` but `planned`, and every new boundary span
/// in `spans`, stays clear of the convex hull of `points` laid flat about
/// `at`.
fn hull_is_clear(
    topo: &Topology,
    face: FaceId,
    at: Point3,
    points: &[Point3],
    planned: &HashSet<EdgeId>,
    spans: &[&Span],
) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    let Some(flat) = Flat::about(topo, face, at)? else {
        return Ok(false);
    };
    let flat_points: Vec<[f64; 2]> = points.iter().map(|&p| flat.point(p)).collect();
    let region = convex_hull(&flat_points);
    let data = topo.face(face)?;
    for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            if planned.contains(&oe.edge()) {
                continue;
            }
            let Some(pieces) = flat.edge(topo, oe.edge())? else {
                return Ok(false);
            };
            if pieces
                .into_iter()
                .any(|p| piece_meets_region(p, &region, tol))
            {
                return Ok(false);
            }
        }
    }
    for span in spans {
        let piece = match span {
            Span::Segment(a, b) => Piece2::Segment(flat.point(*a), flat.point(*b)),
            Span::Arc { circle, t0, sweep } => {
                let Flat::Plane { normal, .. } = &flat else {
                    return Ok(false);
                };
                let turn = if circle.normal().dot(*normal) > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let center = flat.point(circle.center());
                let start = flat.point(circle.evaluate(*t0));
                Piece2::Arc {
                    center,
                    radius: circle.radius(),
                    from: (start[1] - center[1]).atan2(start[0] - center[0]),
                    sweep: sweep * turn,
                }
            }
        };
        if piece_meets_region(piece, &region, tol) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the region a corner takes from each of its three faces is clear
/// of everything else on them. On each face the region lies in the hull of
/// the vertex, each piece's section point and contact end there, the poles
/// of an arc contact between the two, the third edge's new end and, on the
/// face the sliver cuts, the sliver's poles.
fn corner_is_clear(
    topo: &Topology,
    plan: &Plan,
    corner: &Corner,
    new_spans: &[(FaceId, Span, Vec<Point3>, Option<usize>)],
) -> Result<bool, OperationsError> {
    let at = topo.vertex(corner.vertex)?.point();
    let planned: HashSet<EdgeId> = corner
        .pieces
        .iter()
        .map(|&i| plan.pieces[i].edge)
        .chain(std::iter::once(corner.third))
        .collect();
    for face in [corner.shared, corner.faces[0], corner.faces[1]] {
        let mut points = vec![at];
        for &index in &corner.pieces {
            let piece = &plan.pieces[index];
            let side = if face == piece.cap {
                0
            } else if face == piece.wall {
                1
            } else {
                continue;
            };
            let section = plan.piece_sections[index][piece.slot(corner.vertex)];
            let ends = plan.contact_ends(index, corner.vertex);
            let (from, to) = if side == 0 {
                (section.cap, ends.0)
            } else {
                (section.wall, ends.1)
            };
            points.extend([from, to]);
            if let EdgeCurve::Circle(circle) =
                &piece.contact_curves(&plan.piece_sections[index][0])?[side]
                && (to - from).length() > JOIN_TOL
            {
                let Some(pole) = arc_pole(circle, from, to) else {
                    return Ok(false);
                };
                points.push(pole);
            }
        }
        if face == corner.faces[0] || face == corner.faces[1] {
            points.push(corner.reach);
        }
        if face == corner.faces[1] {
            points.extend(corner.cut);
            if let Some(EdgeCurve::NurbsCurve(sliver)) = &corner.sliver {
                points.extend(sliver.control_points());
            }
        }
        let spans: Vec<&Span> = new_spans
            .iter()
            .filter(|(f, _, _, owner)| {
                *f == face && owner.is_none_or(|o| !corner.pieces.contains(&o))
            })
            .map(|(_, span, _, _)| span)
            .collect();
        if !hull_is_clear(topo, face, at, &points, &planned, &spans)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the region a plane end cuts from its plane, or hands it, is clear
/// of the plane's other edges: the hull of the vertex, the contacts' two
/// ends, the pole of the stretch of an arc edge between the vertex and its
/// point, and the poles of a slanted end's curve. A slanted end's contacts
/// end off its section, so the region each takes from its cap or its wall
/// between the section and the end must be clear as well, as at a corner.
fn plane_end_is_clear(
    topo: &Topology,
    plan: &Plan,
    end: &PlaneEnd,
    new_spans: &[(FaceId, Span, Vec<Point3>, Option<usize>)],
) -> Result<bool, OperationsError> {
    let at = topo.vertex(end.vertex)?.point();
    let mut points = vec![at, end.points[0], end.points[1]];
    for (edge, p) in end.edges.into_iter().zip(end.points) {
        if let EdgeCurve::Circle(circle) = read_curve(topo, edge)? {
            let Some(pole) = arc_pole(&circle, at, p) else {
                return Ok(false);
            };
            points.push(pole);
        }
    }
    if let EdgeCurve::NurbsCurve(curve) = &end.cross {
        points.extend_from_slice(curve.control_points());
    }
    let planned: HashSet<EdgeId> = end.edges.into_iter().collect();
    if !hull_is_clear(topo, end.face, at, &points, &planned, &[])? {
        return Ok(false);
    }
    if !end.slanted {
        return Ok(true);
    }
    let piece = &plan.pieces[end.piece];
    let section = plan.piece_sections[end.piece][piece.slot(end.vertex)];
    let curves = piece.contact_curves(&plan.piece_sections[end.piece][0])?;
    let planned: HashSet<EdgeId> = [piece.edge, end.edges[0], end.edges[1]]
        .into_iter()
        .collect();
    for (side, face) in [piece.cap, piece.wall].into_iter().enumerate() {
        let from = if side == 0 { section.cap } else { section.wall };
        let to = end.points[side];
        let mut points = vec![at, from, to];
        if let EdgeCurve::Circle(circle) = &curves[side]
            && (to - from).length() > JOIN_TOL
        {
            let Some(pole) = arc_pole(circle, from, to) else {
                return Ok(false);
            };
            points.push(pole);
        }
        let spans: Vec<&Span> = new_spans
            .iter()
            .filter(|(f, _, _, owner)| *f == face && *owner != Some(end.piece))
            .map(|(_, span, _, _)| span)
            .collect();
        if !hull_is_clear(topo, face, at, &points, &planned, &spans)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The angles (from `from` over `width`) and the heights along the axis
/// (from `low` to `high`) of the strip a chamfer takes from the cylinder
/// wall under `piece`.
fn cylinder_strip(cylinder: &CylindricalSurface, piece: &Piece) -> Option<(f64, f64, f64, f64)> {
    let axis = cylinder.axis();
    let height = |p: Point3| axis.dot(p - cylinder.origin());
    let angle = |p: Point3| cylinder.project_point(p).0;
    match &piece.wall_side {
        WallSide::Axis(dir) => {
            let (u0, u1, um) = (
                angle(piece.start_point),
                angle(piece.end_point),
                angle(piece.midpoint()),
            );
            let (from, width) = if (um - u0).rem_euclid(TAU) < (u1 - u0).rem_euclid(TAU) {
                (u0, (u1 - u0).rem_euclid(TAU))
            } else {
                (u1, (u0 - u1).rem_euclid(TAU))
            };
            let (rim, other) = (
                height(piece.start_point),
                height(piece.start_point + *dir * piece.wall_d),
            );
            Some((from, width, rim.min(other), rim.max(other)))
        }
        WallSide::Ruling { turn, .. } => {
            let u = angle(piece.start_point);
            let (from, width) = if *turn > 0.0 {
                (u, *turn)
            } else {
                ((u + turn).rem_euclid(TAU), -turn)
            };
            let (h0, h1) = (height(piece.start_point), height(piece.end_point));
            Some((from, width, h0.min(h1), h0.max(h1)))
        }
        WallSide::Plane { .. } => None,
    }
}

/// Whether some of the segment from `a` to `b` lies strictly inside the
/// strip a line piece's chamfer takes from its plane wall: from `origin`
/// along `along` for `length`, and across along `across` for up to `d`.
fn segment_enters_strip(
    (a, b): (Point3, Point3),
    origin: Point3,
    along: Vec3,
    length: f64,
    across: Vec3,
    d: f64,
) -> bool {
    let tol = Tolerance::new().linear;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    // Keep the parameters where `start + t * rate` lies within `(min, max)`.
    let mut clip = |start: f64, rate: f64, min: f64, max: f64| {
        if rate.abs() <= 1e-15 {
            if start <= min || start >= max {
                hi = lo - 1.0;
            }
            return;
        }
        let (t0, t1) = ((min - start) / rate, (max - start) / rate);
        lo = lo.max(t0.min(t1));
        hi = hi.min(t0.max(t1));
    };
    let (sa, sb) = ((a - origin).dot(along), (b - origin).dot(along));
    let (ha, hb) = ((a - origin).dot(across), (b - origin).dot(across));
    clip(sa, sb - sa, tol, length - tol);
    clip(ha, hb - ha, tol, d + tol);
    hi > lo + 1e-12
}

/// Whether two pieces along one wall, that share no vertex, take strips
/// from it that overlap.
fn wall_strips_overlap(
    topo: &Topology,
    plan: &Plan,
    i: usize,
    j: usize,
) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    let (a, b) = (&plan.pieces[i], &plan.pieces[j]);
    if let Some(cylinder) = cylinder_of(topo, a.wall)? {
        let (Some((fa, wa, la, ha)), Some((fb, wb, lb, hb))) =
            (cylinder_strip(&cylinder, a), cylinder_strip(&cylinder, b))
        else {
            return Ok(true);
        };
        let slack = tol / cylinder.radius();
        let angles =
            (fb - fa).rem_euclid(TAU) < wa - slack || (fa - fb).rem_euclid(TAU) < wb - slack;
        let heights = ha.min(hb) > la.max(lb) + tol;
        return Ok(angles && heights);
    }
    // On a plane wall, each piece's edge and contact stay out of the other's
    // strip.
    for (first, second) in [(i, j), (j, i)] {
        let (piece, other) = (&plan.pieces[first], &plan.pieces[second]);
        let section = plan.piece_sections[first][0];
        let run = piece.end_point - piece.start_point;
        let along = run.normalize()?;
        let [s0, s1] = plan.piece_sections[second];
        for segment in [(other.start_point, other.end_point), (s0.wall, s1.wall)] {
            if segment_enters_strip(
                segment,
                piece.start_point,
                along,
                run.length(),
                section.wall_dir,
                piece.wall_d,
            ) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether the cylinder wall under `piece` keeps every edge but `planned` out
/// of the strip the chamfer takes from it. Its rulings and its circles about
/// the axis are read exactly; any other edge fails.
fn cylinder_wall_clear(
    topo: &Topology,
    piece: &Piece,
    planned: &HashSet<EdgeId>,
) -> Result<bool, OperationsError> {
    let tol = Tolerance::new().linear;
    let Some(cylinder) = cylinder_of(topo, piece.wall)? else {
        return Ok(false);
    };
    let axis = cylinder.axis();
    let height = |p: Point3| axis.dot(p - cylinder.origin());
    let angle = |p: Point3| cylinder.project_point(p).0;
    let slack = tol / cylinder.radius();
    let Some((from, width, _, _)) = cylinder_strip(&cylinder, piece) else {
        return Ok(false);
    };
    let along_axis = match &piece.wall_side {
        WallSide::Axis(dir) => Some((height(piece.start_point), dir.dot(axis))),
        WallSide::Ruling { .. } | WallSide::Plane { .. } => None,
    };
    let (h0, h1) = (height(piece.start_point), height(piece.end_point));
    // Under an arc, from just below the rim to the contact; beside a ruling,
    // strictly between the edge's ends.
    let meets_heights = |a: f64, b: f64| match along_axis {
        Some((rim, sign)) => {
            let (lo, hi) = ((a - rim) * sign, (b - rim) * sign);
            lo.max(hi) > tol && lo.min(hi) <= piece.wall_d + tol
        }
        None => a.max(b) > h0.min(h1) + tol && a.min(b) < h0.max(h1) - tol,
    };
    // Under an arc, strictly within its angles; beside a ruling, past the
    // edge up to the contact, which lies at `from` where the turn is
    // negative.
    let contact_first = matches!(piece.wall_side, WallSide::Ruling { turn, .. } if turn < 0.0);
    let in_angles = |u: f64| {
        let off = (u - from).rem_euclid(TAU);
        match along_axis {
            Some(_) => off > slack && off < width - slack,
            None if contact_first => off < width - slack || off > TAU - slack,
            None => off > slack && off <= width + slack,
        }
    };
    let wall = topo.face(piece.wall)?;
    for oe in topo.wire(wall.outer_wire())?.edges() {
        if planned.contains(&oe.edge()) {
            continue;
        }
        let edge = topo.edge(oe.edge())?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        match read_curve(topo, oe.edge())? {
            EdgeCurve::Line => {
                let run = b - a;
                if run.length() > tol && run.normalize()?.cross(axis).length() > JOIN_TOL {
                    return Ok(false);
                }
                if in_angles(angle(a)) && meets_heights(height(a), height(b)) {
                    return Ok(false);
                }
            }
            EdgeCurve::Circle(c) => {
                let coaxial = c.normal().cross(axis).length() < JOIN_TOL
                    && (c.center() - cylinder.origin()).cross(axis).length() < JOIN_TOL;
                if !coaxial {
                    return Ok(false);
                }
                let h = height(c.center());
                if !meets_heights(h, h) {
                    continue;
                }
                if edge.start() == edge.end() {
                    return Ok(false);
                }
                let curve = EdgeCurve::Circle(c.clone());
                let (t0, t1) = curve.domain_with_endpoints(a, b);
                let (ua, ub, um) = (angle(a), angle(b), angle(c.evaluate(0.5 * (t0 + t1))));
                let (lo, span) = if (um - ua).rem_euclid(TAU) < (ub - ua).rem_euclid(TAU) {
                    (ua, (ub - ua).rem_euclid(TAU))
                } else {
                    (ub, (ua - ub).rem_euclid(TAU))
                };
                let overlap = (lo - from).rem_euclid(TAU) < width - slack
                    || (from - lo).rem_euclid(TAU) < span - slack;
                if overlap {
                    return Ok(false);
                }
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(false),
        }
    }
    Ok(true)
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

/// Each edge's and each face's image in a copy of a solid.
type Images = (HashMap<EdgeId, EdgeId>, HashMap<FaceId, FaceId>);

/// Each edge and each face of `original` paired with its image in `copy`,
/// which lists its faces, wires and edges in the same order.
fn images_in(
    topo: &Topology,
    original: SolidId,
    copy: SolidId,
) -> Result<Option<Images>, OperationsError> {
    let from = brepkit_topology::explorer::solid_faces(topo, original)?;
    let to = brepkit_topology::explorer::solid_faces(topo, copy)?;
    if from.len() != to.len() {
        return Ok(None);
    }
    let face_images: HashMap<FaceId, FaceId> =
        from.iter().copied().zip(to.iter().copied()).collect();
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
    Ok(Some((images, face_images)))
}

/// Edges with the flag each takes, in order.
type Run = Vec<(EdgeId, bool)>;

/// `run` read the other way.
fn reversed(run: &[(EdgeId, bool)]) -> Run {
    run.iter().rev().map(|&(e, f)| (e, !f)).collect()
}

/// An edge's replacement in one face, read from the edge's stored start to
/// its stored end: what comes before it, what takes its place (the edge
/// itself when `None`), and what comes after it.
#[derive(Default)]
struct Swap {
    before: Run,
    core: Option<Run>,
    after: Run,
}

/// Each rewritten face's replacements, its id first so faces rewrite in a
/// fixed order.
#[derive(Default)]
struct Swaps(BTreeMap<usize, (FaceId, HashMap<EdgeId, Swap>)>);

impl Swaps {
    fn entry(&mut self, face: FaceId, edge: EdgeId) -> &mut Swap {
        self.0
            .entry(face.index())
            .or_insert_with(|| (face, HashMap::new()))
            .1
            .entry(edge)
            .or_default()
    }

    /// Replace `edge` in `face` with `run`; false when it already has another
    /// replacement there.
    fn core(&mut self, face: FaceId, edge: EdgeId, run: Run) -> bool {
        let swap = self.entry(face, edge);
        if let Some(existing) = &swap.core {
            return *existing == run;
        }
        swap.core = Some(run);
        true
    }

    /// Insert `seq`, read the way `face` runs, where `face` leaves `edge`;
    /// `forward` is the way the face uses the edge.
    fn after(&mut self, face: FaceId, edge: EdgeId, forward: bool, seq: &[(EdgeId, bool)]) {
        let swap = self.entry(face, edge);
        if forward {
            swap.after.extend_from_slice(seq);
        } else {
            let mut before = reversed(seq);
            before.append(&mut swap.before);
            swap.before = before;
        }
    }
}

/// `wire_id` with each listed edge replaced by its run, read backward where
/// the wire traverses the edge backward.
fn rewrite_wire(
    topo: &mut Topology,
    wire_id: WireId,
    swaps: &HashMap<EdgeId, Swap>,
) -> Result<WireId, OperationsError> {
    let wire = topo.wire(wire_id)?;
    let closed = wire.is_closed();
    let mut edges = Vec::with_capacity(wire.edges().len());
    for oe in wire.edges() {
        match swaps.get(&oe.edge()) {
            None => edges.push(OrientedEdge::new(oe.edge(), oe.is_forward())),
            Some(swap) => {
                let own = [(oe.edge(), true)];
                let core: &[(EdgeId, bool)] = swap.core.as_deref().unwrap_or(&own);
                let run: Run = swap
                    .before
                    .iter()
                    .chain(core)
                    .chain(&swap.after)
                    .copied()
                    .collect();
                if oe.is_forward() {
                    edges.extend(run.iter().map(|&(e, f)| OrientedEdge::new(e, f)));
                } else {
                    edges.extend(run.iter().rev().map(|&(e, f)| OrientedEdge::new(e, !f)));
                }
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
        d: normal.dot(to_vec(p)),
    }
}

/// Whether an edge from `p0` to `p1` along `curve`, cut back to run from `q0`
/// to `q1`, keeps a stretch the way it ran: two cuts from its two ends can
/// pass each other.
fn keeps_a_stretch(curve: &EdgeCurve, p0: Point3, p1: Point3, q0: Point3, q1: Point3) -> bool {
    let tol = Tolerance::new().linear;
    match curve {
        EdgeCurve::Line => {
            let run = p1 - p0;
            (q1 - q0).dot(run) > tol * run.length()
        }
        EdgeCurve::Circle(c) => {
            let slack = tol / c.radius();
            let at = |p: Point3| {
                let off = (c.project(p) - c.project(p0)).rem_euclid(TAU);
                if off > TAU - slack { 0.0 } else { off }
            };
            let span = at(p1);
            let to = if (q1 - p1).length() <= tol {
                span
            } else {
                at(q1)
            };
            span > slack && at(q0) + slack < to && to <= span + slack
        }
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
    }
}

/// The flag with which an edge stored from `start` runs from `from`.
fn runs_from(topo: &Topology, edge: EdgeId, from: VertexId) -> Result<bool, OperationsError> {
    Ok(topo.edge(edge)?.start() == from)
}

/// Where a piece's contacts end at one of its vertices: the vertex on the
/// cap, the vertex on the wall, and the edges between them, cap to wall.
struct EndPoints {
    cap: VertexId,
    wall: VertexId,
    path: Run,
}

/// An edge to cut back: the edge, each end it loses with the vertex that
/// takes its place, and the faces along it.
type CutEnds = (EdgeId, Vec<(VertexId, VertexId)>, Vec<FaceId>);

/// A corner as built: the meeting point, the third edge's new end, the cut
/// point (the third edge's new end where nothing is cut), the mitre and the
/// sliver.
struct BuiltCorner {
    meet: VertexId,
    reach: VertexId,
    cut: VertexId,
    mitre: EdgeId,
    sliver: Option<EdgeId>,
}

/// Build one planned chain into the solid it was read from, whose caps,
/// walls and neighbours are rewritten in place. Returns the new faces, or
/// `None` when the shell's faces do not agree on the edges they share.
#[allow(clippy::too_many_lines)]
fn build_chain(topo: &mut Topology, plan: &Plan) -> Result<Option<Vec<FaceId>>, OperationsError> {
    let tol = Tolerance::new().linear;
    let mut cap_vertex: HashMap<VertexId, VertexId> = HashMap::new();
    let mut wall_vertex: HashMap<VertexId, VertexId> = HashMap::new();
    let mut across: HashMap<VertexId, EdgeId> = HashMap::new();
    for &(vertex, section) in &plan.sections {
        let (points, curve) = match plan.plane_end(vertex) {
            Some(end) => (end.points, end.cross.clone()),
            None => ([section.cap, section.wall], EdgeCurve::Line),
        };
        let c = topo.add_vertex(Vertex::new(points[0], tol));
        let w = topo.add_vertex(Vertex::new(points[1], tol));
        cap_vertex.insert(vertex, c);
        wall_vertex.insert(vertex, w);
        across.insert(vertex, topo.add_edge(Edge::new(c, w, curve)));
    }

    let mut swaps = Swaps::default();
    // Each edge cut back from one or both of its ends, with the faces along it.
    let mut trims: Vec<(EdgeId, VertexId, VertexId, [FaceId; 2])> = Vec::new();

    let mut built_corners: HashMap<VertexId, BuiltCorner> = HashMap::new();
    for corner in &plan.corners {
        let meet = topo.add_vertex(Vertex::new(corner.meet, tol));
        let reach = topo.add_vertex(Vertex::new(corner.reach, tol));
        let cut = match corner.cut {
            Some(p) => topo.add_vertex(Vertex::new(p, tol)),
            None => reach,
        };
        let mitre = topo.add_edge(Edge::new(cut, meet, corner.mitre.clone()));
        let sliver = corner
            .sliver
            .clone()
            .map(|curve| topo.add_edge(Edge::new(reach, cut, curve)));
        trims.push((corner.third, corner.vertex, reach, corner.faces));
        built_corners.insert(
            corner.vertex,
            BuiltCorner {
                meet,
                reach,
                cut,
                mitre,
                sliver,
            },
        );
    }

    // At a plane end, each side's point splits its edge or the plane takes
    // the stretch back to the vertex.
    let mut stretch: HashMap<(VertexId, usize), EdgeId> = HashMap::new();
    for end in &plan.plane_ends {
        let piece = &plan.pieces[end.piece];
        let points = [cap_vertex[&end.vertex], wall_vertex[&end.vertex]];
        for side in 0..2 {
            let face = if side == 0 { piece.cap } else { piece.wall };
            if end.split[side] {
                trims.push((end.edges[side], end.vertex, points[side], [face, end.face]));
                continue;
            }
            let curve = read_curve(topo, end.edges[side])?;
            let edge = topo.edge(end.edges[side])?;
            // The stretch runs on the edge's curve the way the edge does.
            let new = if edge.start() == end.vertex {
                Edge::new(points[side], end.vertex, curve)
            } else {
                Edge::new(end.vertex, points[side], curve)
            };
            stretch.insert((end.vertex, side), topo.add_edge(new));
        }
    }

    // Between two walls, the ruling shortens for both.
    for joint in &plan.joints {
        trims.push((
            joint.ruling,
            joint.vertex,
            wall_vertex[&joint.vertex],
            joint.walls,
        ));
    }

    // Each trimmed edge once, cut back from every end it loses.
    let mut cut_ends: BTreeMap<usize, CutEnds> = BTreeMap::new();
    for (edge, from, to, faces) in trims {
        let entry = cut_ends
            .entry(edge.index())
            .or_insert_with(|| (edge, Vec::new(), Vec::new()));
        entry.1.push((from, to));
        for face in faces {
            if !entry.2.contains(&face) {
                entry.2.push(face);
            }
        }
    }
    for (edge, cuts, faces) in cut_ends.into_values() {
        let data = topo.edge(edge)?;
        let (mut start, mut end) = (data.start(), data.end());
        if start == end {
            return Ok(None);
        }
        let (original_start, original_end) = (start, end);
        for (from, to) in cuts {
            if from == original_start && start == original_start {
                start = to;
            } else if from == original_end && end == original_end {
                end = to;
            } else {
                return Ok(None);
            }
        }
        let curve = read_curve(topo, edge)?;
        let ends = [original_start, original_end, start, end];
        let [p0, p1, q0, q1] = ends.map(|v| topo.vertex(v).map(Vertex::point));
        if !keeps_a_stretch(&curve, p0?, p1?, q0?, q1?) {
            return Ok(None);
        }
        let shorter = topo.add_edge(Edge::new(start, end, curve));
        for face in faces {
            if !swaps.core(face, edge, vec![(shorter, true)]) {
                return Ok(None);
            }
        }
    }

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
        if !swaps.core(end.wall, end.ruling, vec![(far, true)])
            || !swaps.core(end.beyond, end.ruling, vec![(first, true), (second, true)])
        {
            return Ok(None);
        }
    }

    // Where each piece's contacts end at a vertex.
    let end_points = |index: usize, vertex: VertexId| -> Option<EndPoints> {
        let piece = &plan.pieces[index];
        if let (Some(corner), Some(built)) = (plan.corner(vertex), built_corners.get(&vertex)) {
            let first = index == corner.pieces[0];
            let other = if first { built.reach } else { built.cut };
            // From the contact on the piece's own face to the one on the
            // shared face.
            let mut path: Run = Vec::new();
            if first && let Some(sliver) = built.sliver {
                path.push((sliver, true));
            }
            path.push((built.mitre, true));
            return Some(if corner.shared == piece.cap {
                EndPoints {
                    cap: built.meet,
                    wall: other,
                    path: reversed(&path),
                }
            } else {
                EndPoints {
                    cap: other,
                    wall: built.meet,
                    path,
                }
            });
        }
        Some(EndPoints {
            cap: *cap_vertex.get(&vertex)?,
            wall: *wall_vertex.get(&vertex)?,
            path: vec![(*across.get(&vertex)?, true)],
        })
    };

    // What a side's run takes on at a vertex before the contact (`at_start`)
    // or after it: a tangent end's cross edge on the cap, a corner's sliver
    // on the face it cuts, or a plane end's stretch back to the vertex.
    let extra = |topo: &Topology,
                 index: usize,
                 side: usize,
                 vertex: VertexId,
                 at_start: bool|
     -> Result<Run, OperationsError> {
        let piece = &plan.pieces[index];
        let face = if side == 0 { piece.cap } else { piece.wall };
        if side == 0
            && let Some(&cross) = crossing.get(&vertex)
        {
            return Ok(vec![(cross, at_start)]);
        }
        if let (Some(corner), Some(built)) = (plan.corner(vertex), built_corners.get(&vertex))
            && index == corner.pieces[1]
            && face == corner.faces[1]
            && let Some(sliver) = built.sliver
        {
            return Ok(vec![(sliver, at_start)]);
        }
        if let Some(&ext) = stretch.get(&(vertex, side)) {
            // From the vertex to the point at the start, back at the end.
            return Ok(vec![(ext, runs_from(topo, ext, vertex)? == at_start)]);
        }
        Ok(Vec::new())
    };

    // Each edge moves into the cap and along its wall, and a plane or a cone
    // takes the band between.
    let mut new_faces = Vec::with_capacity(plan.pieces.len() + plan.ends.len());
    let mut across_uses: HashMap<VertexId, bool> = HashMap::new();
    for (index, piece) in plan.pieces.iter().enumerate() {
        let [a, b] = plan.piece_sections[index];
        let (Some(start), Some(stop)) =
            (end_points(index, piece.start), end_points(index, piece.end))
        else {
            return Ok(None);
        };
        let (Some(cap_flag), Some(wall_flag)) = (
            wire_flag(topo, piece.cap, piece.edge)?,
            wire_flag(topo, piece.wall, piece.edge)?,
        ) else {
            return Ok(None);
        };
        let cap_reversed = topo.face(piece.cap)?.is_reversed();
        let cap_out = if cap_reversed {
            -piece.cap_normal
        } else {
            piece.cap_normal
        };
        let wall_reversed = topo.face(piece.wall)?.is_reversed();
        let cap_use = cap_flag != cap_reversed;
        let wall_use = wall_flag != wall_reversed;
        if cap_use == wall_use {
            return Ok(None);
        }

        let [cap_curve, wall_curve] = piece.contact_curves(&a)?;
        let band = match piece.band(&a, &b)? {
            Band::Cone(cone) => {
                let Some(cylinder) = cylinder_of(topo, piece.wall)? else {
                    return Ok(None);
                };
                let (u, v) = cylinder.project_point(piece.start_point);
                let wall_normal = cylinder.normal(u, v);
                let wall_out = if wall_reversed {
                    -wall_normal
                } else {
                    wall_normal
                };
                let mid = a.cap + (a.wall - a.cap) * 0.5;
                let (u, v) = cone.project_point(mid);
                let reversed = cone.normal(u, v).dot(cap_out + wall_out) < 0.0;
                Some((cone, reversed))
            }
            Band::Plane { .. } => None,
        };
        let new_cap = topo.add_edge(Edge::new(start.cap, stop.cap, cap_curve));
        let new_wall = topo.add_edge(Edge::new(start.wall, stop.wall, wall_curve));

        for (side, face, new) in [(0, piece.cap, new_cap), (1, piece.wall, new_wall)] {
            let mut run = extra(topo, index, side, piece.start, true)?;
            run.push((new, true));
            run.extend(extra(topo, index, side, piece.end, false)?);
            if !swaps.core(face, piece.edge, run) {
                return Ok(None);
            }
        }

        // The band runs its new cap edge against the cap's use of it.
        let reversed_band = band.as_ref().is_some_and(|&(_, r)| r);
        let forward = cap_use == reversed_band;
        let (uses, corners) = if forward {
            let mut uses = vec![(new_cap, true)];
            uses.extend(stop.path.iter().copied());
            uses.push((new_wall, false));
            uses.extend(reversed(&start.path));
            (uses, [a.cap, b.cap, b.wall, a.wall])
        } else {
            let mut uses = start.path.clone();
            uses.push((new_wall, true));
            uses.extend(reversed(&stop.path));
            uses.push((new_cap, false));
            (uses, [a.cap, a.wall, b.wall, b.cap])
        };
        across_uses.insert(piece.start, forward == reversed_band);
        across_uses.insert(piece.end, forward != reversed_band);
        let wire = topo.add_wire(Wire::new(
            uses.iter().map(|&(e, f)| OrientedEdge::new(e, f)).collect(),
            true,
        )?);
        let face = if let Some((cone, reversed)) = band {
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

    // A plane end's plane takes the cross edge, and the stretch on a side
    // that does not split, between its two edges at the vertex.
    for end in &plan.plane_ends {
        let points = [cap_vertex[&end.vertex], wall_vertex[&end.vertex]];
        let mut entering = None;
        for side in 0..2 {
            let Some(flag) = wire_flag(topo, end.face, end.edges[side])? else {
                return Ok(None);
            };
            let edge = topo.edge(end.edges[side])?;
            let ends_here = if flag {
                edge.end() == end.vertex
            } else {
                edge.start() == end.vertex
            };
            if ends_here {
                entering = Some((side, flag));
            }
        }
        let Some((first, flag)) = entering else {
            return Ok(None);
        };
        let second = 1 - first;
        let mut seq: Run = Vec::new();
        if let Some(&ext) = stretch.get(&(end.vertex, first)) {
            seq.push((ext, runs_from(topo, ext, end.vertex)?));
        }
        seq.push((across[&end.vertex], first == 0));
        if let Some(&ext) = stretch.get(&(end.vertex, second)) {
            seq.push((ext, runs_from(topo, ext, points[second])?));
        }
        swaps.after(end.face, end.edges[first], flag, &seq);
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

    for (face, face_swaps) in swaps.0.into_values() {
        // An edge this face no longer uses keeps no trim on it.
        for (&edge, swap) in &face_swaps {
            if swap.core.is_some() {
                topo.pcurves_mut().remove(edge, face);
            }
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

/// Chamfer `edges` of `solid` at `d` on both faces, when they form chains of
/// lines and arcs this path can build exactly.
///
/// Each chain runs along one flat cap, its lines between planes and its arcs
/// between the cap and a coaxial cylinder, or along one wall, its lines and
/// arcs on caps of their own and its lines on a cylinder wall running along
/// its rulings. An edge tangent to the next shares its section with it. An
/// edge meeting the next at an angle meets it in a mitre, where the sharp
/// edge between their other faces is left unselected. A free end stops where
/// the next edge carries straight on, or at a plane through the end that
/// holds the plane's edges along the cap and the wall: square to the edge,
/// or slanted, where the chamfer face meets the plane in a line or a conic.
/// A selection of plain lines that meet no curved neighbour is left to the
/// other engines.
///
/// Returns `None`, leaving `solid` as it was, when the selection is not one
/// of these or the chamfer would meet another edge of a cap or a wall.
///
/// # Errors
///
/// Returns an error if a topology lookup fails, or the chamfer would break
/// into one of the solid's cavities. Returns an error, and builds nothing,
/// where no chamfer of `d` fits a chain: its cap's contact runs past an arc's
/// axis, a contact runs past the far side of its face, or the strips that
/// two chains take from one face overlap.
pub fn chamfer_tangent_chains(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    d: f64,
) -> Result<Option<SolidId>, OperationsError> {
    if d <= 0.0 {
        return Ok(None);
    }
    let distances = Distances {
        near: d,
        far: d,
        reference: HashMap::new(),
    };
    chamfer_chains(topo, solid, edges, &distances)
}

/// [`chamfer_tangent_chains`] with two distances.
///
/// The chamfer reaches `d1` into each edge's reference face and `d2` into
/// its other face. Where two edges of a chain meet tangent, each face must
/// take the same distance from both.
///
/// # Errors
///
/// Returns an error if a topology lookup fails, the chamfer would break
/// into one of the solid's cavities, or no chamfer of these distances fits
/// a chain, as for [`chamfer_tangent_chains`].
pub fn chamfer_tangent_chains_on(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[(EdgeId, FaceId)],
    d1: f64,
    d2: f64,
) -> Result<Option<SolidId>, OperationsError> {
    if d1 <= 0.0 || d2 <= 0.0 {
        return Ok(None);
    }
    // Each reference one of its edge's two faces, and one for each edge.
    let outer = topo
        .shell(topo.solid(solid)?.outer_shell())?
        .faces()
        .to_vec();
    let adjacency = AdjacencyIndex::build_from_faces(topo, &outer)?;
    let mut reference = HashMap::with_capacity(edges.len());
    for &(edge, face) in edges {
        let around = adjacency.faces_for_edge(edge);
        if around.len() != 2
            || !around.contains(&face)
            || reference
                .insert(edge, face)
                .is_some_and(|other| other != face)
        {
            return Ok(None);
        }
    }
    let ids: Vec<EdgeId> = edges.iter().map(|&(e, _)| e).collect();
    let distances = Distances {
        near: d1,
        far: d2,
        reference,
    };
    chamfer_chains(topo, solid, &ids, &distances)
}

#[allow(clippy::too_many_lines)]
fn chamfer_chains(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    distances: &Distances,
) -> Result<Option<SolidId>, OperationsError> {
    if edges.is_empty() {
        return Ok(None);
    }
    let outer_faces = topo
        .shell(topo.solid(solid)?.outer_shell())?
        .faces()
        .to_vec();
    let groups = connected_groups(topo, edges)?;
    let mut plans = Vec::with_capacity(groups.len());
    let mut other = false;
    let mut far_sides = FarSides::default();
    for group in &groups {
        match plan_chain(topo, &outer_faces, group, distances, &mut far_sides)? {
            Reading::Chain(plan) => plans.push(*plan),
            Reading::Other => other = true,
            Reading::Misfit(reason) => return Err(OperationsError::InvalidInput { reason }),
        }
    }
    if other || !plans.iter().any(|p| p.curved) {
        return Ok(None);
    }
    if chains_overlap(&plans, &mut 0)? {
        return Err(OperationsError::InvalidInput {
            reason: "the chamfers of two chains take strips of one face that overlap".into(),
        });
    }
    let change: f64 = plans.iter().map(|p| p.volume_change).sum();
    let swept: f64 = plans.iter().map(|p| p.volume_change.abs()).sum();
    let allowance: f64 = plans.iter().map(|p| p.corner_allowance).sum();

    let copy = crate::copy::copy_solid(topo, solid)?;
    let Some((images, face_images)) = images_in(topo, solid, copy)? else {
        return Ok(None);
    };
    let mut reference = HashMap::with_capacity(distances.reference.len());
    for (edge, face) in &distances.reference {
        let (Some(&e), Some(&f)) = (images.get(edge), face_images.get(face)) else {
            return Ok(None);
        };
        reference.insert(e, f);
    }
    let image_distances = Distances {
        near: distances.near,
        far: distances.far,
        reference,
    };
    let data = topo.solid(copy)?;
    let (outer, inners) = (data.outer_shell(), data.inner_shells().to_vec());
    // A copy leaves its pcurves behind, and a face this chamfer never touches
    // keeps its trims.
    crate::cavities::copy_solid_pcurves(topo, solid, copy)?;
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
        // shares and leave new edges on a cap or wall it shares, which it
        // must clear. Those chains reshape faces, so the far sides are read
        // afresh.
        let mut far_sides = FarSides::default();
        let plan = match plan_chain(topo, &faces, &image, &image_distances, &mut far_sides)? {
            Reading::Chain(plan) => plan,
            Reading::Other => return Ok(None),
            // It fits the solid alone, so the chains before it leave it no room.
            Reading::Misfit(reason) => {
                return Err(OperationsError::InvalidInput {
                    reason: format!("the chamfers of two chains meet: {reason}"),
                });
            }
        };
        let Some(new_faces) = build_chain(topo, &plan)? else {
            return Ok(None);
        };
        faces.extend_from_slice(&new_faces);
        made.extend(new_faces);
    }
    let outer = topo.add_shell(Shell::new(faces)?);
    let result = topo.add_solid(Solid::new(outer, inners));
    crate::cavities::clear_of_own_cavities(topo, result, &made, "chamfer")?;

    // The closed form decides whether what was built is what was meant.
    let deflection = (distances.least() * 0.01).max(1e-4);
    let before = crate::measure::solid_volume(topo, solid, deflection)?;
    let after = crate::measure::solid_volume(topo, result, deflection)?;
    let oriented = crate::measure::oriented_solid_volume(topo, result, deflection)?;
    if oriented <= 0.0 || (after - before - change).abs() > VOLUME_MARGIN * swept + allowance {
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// A conic arc lies on both its surfaces and in the hull of its poles.
    #[test]
    fn a_conic_arc_lies_on_both_surfaces_and_in_its_poles_hull() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, -1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        // The plane y = 2 meets the cone, r = 10 - z, in a hyperbola.
        let on = |x: f64| Point3::new(x, 2.0, 10.0 - x.hypot(2.0));
        let (a, b) = (on(1.0), on(6.0));
        let curve = conic_arc(Vec3::new(0.0, 1.0, 0.0), &cone, a, b).unwrap();
        let EdgeCurve::NurbsCurve(nurbs) = &curve else {
            panic!("a conic arc is a NURBS curve");
        };
        let flat = |p: Point3| [p.x(), p.z()];
        let hull = convex_hull(
            &nurbs
                .control_points()
                .iter()
                .map(|&p| flat(p))
                .collect::<Vec<_>>(),
        );
        for k in 0..=200 {
            let p = nurbs.evaluate(f64::from(k) / 200.0);
            assert!(Band::Cone(cone.clone()).level(p).abs() < 1e-9);
            assert!((p.y() - 2.0).abs() < 1e-9);
            assert!(region_holds(&hull, flat(p), 1e-12));
        }
    }

    /// The square `[0, 1] x [0, 1]` on `z = 0` moved `shift` along `x`, the
    /// face of its plane beyond `y = 1` up to `y = 3`, and a wall down from
    /// `y = 0` to `z = -2`, with their adjacency.
    fn seam_and_crease(topo: &mut Topology, shift: f64) -> ([FaceId; 3], AdjacencyIndex) {
        let mut at = |x: f64, y: f64, z: f64| {
            topo.add_vertex(Vertex::new(Point3::new(x + shift, y, z), 1e-7))
        };
        let [p0, p1, p2, p3, p4, p5, q0, q1] = [
            at(0.0, 0.0, 0.0),
            at(1.0, 0.0, 0.0),
            at(1.0, 1.0, 0.0),
            at(0.0, 1.0, 0.0),
            at(1.0, 3.0, 0.0),
            at(0.0, 3.0, 0.0),
            at(1.0, 0.0, -2.0),
            at(0.0, 0.0, -2.0),
        ];
        let mut line = |a, b| topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let [base, right, seam, left] = [line(p0, p1), line(p1, p2), line(p2, p3), line(p3, p0)];
        let [far_right, top, far_left] = [line(p2, p4), line(p4, p5), line(p5, p3)];
        let [down_right, bottom, down_left] = [line(p1, q0), line(q0, q1), line(q1, p0)];
        let mut face = |uses: &[(EdgeId, bool)], normal: Vec3, d: f64| {
            let edges = uses.iter().map(|&(e, f)| OrientedEdge::new(e, f)).collect();
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            topo.add_face(Face::new(
                wire,
                Vec::new(),
                FaceSurface::Plane { normal, d },
            ))
        };
        let up = Vec3::new(0.0, 0.0, 1.0);
        let square = face(
            &[(base, true), (right, true), (seam, true), (left, true)],
            up,
            0.0,
        );
        let beyond = face(
            &[
                (seam, false),
                (far_right, true),
                (top, true),
                (far_left, true),
            ],
            up,
            0.0,
        );
        let wall = face(
            &[
                (base, false),
                (down_left, false),
                (bottom, false),
                (down_right, false),
            ],
            Vec3::new(0.0, -1.0, 0.0),
            0.0,
        );
        let faces = [square, beyond, wall];
        let adjacency = AdjacencyIndex::build_from_faces(topo, &faces).unwrap();
        (faces, adjacency)
    }

    /// A face's far side runs on across a flat seam to the next face of its
    /// plane, and stops at a crease (see [`seam_and_crease`]).
    #[test]
    fn a_far_side_runs_on_across_a_flat_seam_and_stops_at_a_crease() {
        let mut topo = Topology::new();
        let ([square, beyond, wall], adjacency) = seam_and_crease(&mut topo, 0.0);
        let origin = Point3::new(0.5, 0.0, 0.0);
        let mut far_sides = FarSides::default();
        let mut far = |face, from, dir| {
            far_sides
                .reach(&topo, &adjacency, face, from, dir)
                .unwrap()
                .unwrap()
        };
        let (across, down) = (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!((far(square, origin, across) - 3.0).abs() < 1e-12);
        assert!(far(square, origin, down).abs() < 1e-12);
        assert!((far(wall, origin, down) - 2.0).abs() < 1e-12);
        // The run read from the square along `across` serves the face beyond.
        assert!((far(beyond, Point3::new(0.5, 2.0, 0.0), across) - 1.0).abs() < 1e-12);
        assert_eq!(far_sides.boxes, 5);
    }

    /// With the faces 1e9 out, a read along a direction 5e-8 off one read
    /// before reads the run afresh: a far side carried over from the first
    /// direction would move by the turn times the distance out, 50.
    #[test]
    fn a_far_side_along_a_near_direction_far_out_is_read_afresh() {
        let mut topo = Topology::new();
        let ([square, _, _], adjacency) = seam_and_crease(&mut topo, 1e9);
        let origin = Point3::new(1e9 + 0.5, 0.0, 0.0);
        let (across, near) = (Vec3::new(0.0, 1.0, 0.0), Vec3::new(5e-8, 1.0, 0.0));
        let mut far_sides = FarSides::default();
        let mut far = |dir| {
            far_sides
                .reach(&topo, &adjacency, square, origin, dir)
                .unwrap()
                .unwrap()
        };
        assert!((far(across) - 3.0).abs() < 1e-6);
        let second = far(near);
        assert!((second - 3.0).abs() < 1e-6, "{second}");
        assert_eq!(far_sides.boxes, 4);
    }

    /// A rod of radius 8, 10 high, whose wall is `n` faces of one cylinder,
    /// each under one arc of the floor's rim and one of the top's, joined
    /// along rulings. Returns the rod, its floor, and the floor's and the
    /// top's arcs.
    fn split_rod(topo: &mut Topology, n: usize) -> (SolidId, FaceId, Vec<EdgeId>, Vec<EdgeId>) {
        let (r, h) = (8.0, 10.0);
        let at = |k: usize, z: f64| {
            let t = TAU * k as f64 / n as f64;
            Point3::new(r * t.cos(), r * t.sin(), z)
        };
        let mut ring = |z: f64| -> Vec<VertexId> {
            (0..n)
                .map(|k| topo.add_vertex(Vertex::new(at(k, z), 1e-7)))
                .collect()
        };
        let (low, high) = (ring(0.0), ring(h));
        let mut arcs = |ring: &[VertexId], z: f64| -> Vec<EdgeId> {
            let circle =
                Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
            (0..n)
                .map(|k| {
                    let curve = EdgeCurve::Circle(circle.clone());
                    topo.add_edge(Edge::new(ring[k], ring[(k + 1) % n], curve))
                })
                .collect()
        };
        let (floor_rim, top_rim) = (arcs(&low, 0.0), arcs(&high, h));
        let rulings: Vec<EdgeId> = (0..n)
            .map(|k| topo.add_edge(Edge::new(low[k], high[k], EdgeCurve::Line)))
            .collect();
        let cylinder =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r)
                .unwrap();
        let mut loop_of = |uses: Vec<(EdgeId, bool)>| {
            let edges = uses
                .into_iter()
                .map(|(e, f)| OrientedEdge::new(e, f))
                .collect();
            topo.add_wire(Wire::new(edges, true).unwrap())
        };
        let mut wires: Vec<WireId> = (0..n)
            .map(|k| {
                loop_of(vec![
                    (floor_rim[k], true),
                    (rulings[(k + 1) % n], true),
                    (top_rim[k], false),
                    (rulings[k], false),
                ])
            })
            .collect();
        wires.push(loop_of(
            floor_rim.iter().rev().map(|&e| (e, false)).collect(),
        ));
        wires.push(loop_of(top_rim.iter().map(|&e| (e, true)).collect()));
        let mut faces: Vec<FaceId> = wires[..n]
            .iter()
            .map(|&w| {
                let surface = FaceSurface::Cylinder(cylinder.clone());
                topo.add_face(Face::new(w, Vec::new(), surface))
            })
            .collect();
        let plane = |z: f64, up: f64| FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, up),
            d: z * up,
        };
        let floor = topo.add_face(Face::new(wires[n], Vec::new(), plane(0.0, -1.0)));
        faces.push(floor);
        faces.push(topo.add_face(Face::new(wires[n + 1], Vec::new(), plane(h, 1.0))));
        let shell = topo.add_shell(Shell::new(faces).unwrap());
        let solid = topo.add_solid(Solid::new(shell, Vec::new()));
        (solid, floor, floor_rim, top_rim)
    }

    /// A rim of 1000 arcs, each under a wall face of its own, the walls one
    /// run joined along their rulings: each arc's far side along the axis is
    /// that run's, which is read once for the chain, not once for each arc.
    #[test]
    fn a_rim_of_many_arcs_reads_its_walls_once() {
        let n = 1000;
        let mut topo = Topology::new();
        let (solid, floor, rim, _) = split_rod(&mut topo, n);
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        let adjacency = AdjacencyIndex::build_from_faces(&topo, &faces).unwrap();
        let distances = Distances {
            near: 1.0,
            far: 1.0,
            reference: HashMap::new(),
        };
        let pieces = read_pieces(&topo, &adjacency, &rim, Common::Cap(floor), &distances)
            .unwrap()
            .unwrap();
        assert_eq!(pieces.len(), n);
        let mut far_sides = FarSides::default();
        assert!(
            misfit(&topo, &adjacency, &pieces, &mut far_sides)
                .unwrap()
                .is_none()
        );
        assert_eq!(far_sides.boxes, n);
    }

    /// A complexity guard, run with `--features perf-counters` as the other
    /// `scaling_` guards are. The rod's two rims of 1000 arcs each are two
    /// chains on the one run of walls, and each chain reads that run once. A
    /// span of one chain is read against a strip or a span of the other only
    /// where their boxes meet: at 1, where the rims' strips lie 8 apart, no
    /// pair is read and both rims come off; at 5.5, where they overlap by 1,
    /// the first wall's pair finds the overlap and the chamfer is refused for
    /// it.
    #[cfg(feature = "perf-counters")]
    #[test]
    fn scaling_two_rims_of_many_arcs_read_each_wall_once() {
        let n = 1000;
        for (d, overlap) in [(1.0, false), (5.5, true)] {
            let mut topo = Topology::new();
            let (solid, _, floor_rim, top_rim) = split_rod(&mut topo, n);
            let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
            let distances = Distances {
                near: d,
                far: d,
                reference: HashMap::new(),
            };
            let mut plans = Vec::new();
            for rim in [&floor_rim, &top_rim] {
                let mut far_sides = FarSides::default();
                let Reading::Chain(plan) =
                    plan_chain(&topo, &faces, rim, &distances, &mut far_sides).unwrap()
                else {
                    panic!("d = {d}: a rim is a chain");
                };
                assert_eq!(far_sides.boxes, n, "d = {d}");
                plans.push(*plan);
            }
            let mut tests = 0;
            assert_eq!(
                chains_overlap(&plans, &mut tests).unwrap(),
                overlap,
                "d = {d}"
            );
            assert!(tests <= 8, "d = {d}: {tests} pairs read");

            let both: Vec<EdgeId> = floor_rim.iter().chain(&top_rim).copied().collect();
            let built = chamfer_tangent_chains(&mut topo, solid, &both, d);
            if overlap {
                let reason = built.unwrap_err().to_string();
                assert!(
                    reason.contains("take strips of one face that overlap"),
                    "{reason}"
                );
            } else {
                let result = built.unwrap().unwrap();
                let rod = PI * 64.0 * 10.0;
                let rings = 2.0 * PI * d * d * (8.0 - d / 3.0);
                let volume = crate::measure::solid_volume(&topo, result, 0.001).unwrap();
                assert!((volume - (rod - rings)).abs() < 1e-6, "{volume}");
            }
        }
    }

    /// An arc meets a region where it passes through it, even where no end
    /// lies inside and it crosses a side between two of its samples; it
    /// does not where its sweep turns away.
    #[test]
    fn an_arc_meets_a_region_it_passes_through() {
        let square = convex_hull(&[[-0.1, 0.95], [0.1, 0.95], [0.1, 1.05], [-0.1, 1.05]]);
        let arc = |from: f64, sweep: f64| Piece2::Arc {
            center: [0.0, 0.0],
            radius: 1.0,
            from,
            sweep,
        };
        assert!(piece_meets_region(arc(0.0, PI), &square, 1e-9));
        assert!(piece_meets_region(arc(PI, -PI), &square, 1e-9));
        assert!(!piece_meets_region(arc(PI, PI), &square, 1e-9));
        let far = Piece2::Segment([2.0, 0.0], [2.0, 3.0]);
        assert!(!piece_meets_region(far, &square, 1e-9));
        let through = Piece2::Segment([-1.0, 1.0], [1.0, 1.0]);
        assert!(piece_meets_region(through, &square, 1e-9));
    }
}
