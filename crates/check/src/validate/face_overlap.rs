//! Whether two flat faces of one solid cover the same area.
//!
//! A face can lie on a part of another face, either way round. Each edge of
//! such a solid can have two uses, so the edge checks pass, while its mesh
//! is open.
//!
//! The two faces are read in their common plane, each with its own lines,
//! arcs and B-splines. The boundary of an area that two faces share runs
//! along their edges: a stretch of an edge of one face lies inside the other
//! face, or along an edge of the other with both faces on one side of it.
//! So each edge of each face is cut where it meets the other face's
//! boundary, and the face is read just inside each stretch between two
//! cuts, nearer and nearer the edge. A point inside both faces shows that
//! they share an area.

use std::collections::{HashSet, VecDeque};
use std::f64::consts::TAU;

use brepkit_math::aabb::Aabb3;
use brepkit_math::frame::Frame3;
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::region2d::Boundary2;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{face_edges, solid_faces};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::planar::{PlanarRegion, face_boundary_2d, face_region_2d};
use brepkit_topology::solid::SolidId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// The most angle, in radians, between the normals of two faces read as on
/// one plane.
const MOST_TILT: f64 = 1e-4;

/// The stretches an edge is first cut into where its meetings with the
/// other face's boundary are not solved for: a B-spline edge, or an arc of
/// an ellipse against another arc. A B-spline's stretches keep to its knot
/// spans.
const STRETCHES: u32 = 64;

/// The most halvings of a stretch next to the other face's boundary where
/// its meetings are not solved for.
const MOST_HALVINGS: u32 = 16;

/// The most stretches read within one of an edge's first stretches where its
/// meetings are not solved for, halved breadth first, so the budget spreads
/// over the whole first stretch. The edge's other first stretches have
/// budgets of their own.
const MOST_STRETCHES: usize = 128;

/// The fractions along an edge both faces share at which it is read: more
/// than one, so a vertex of either face touching the edge at one of them
/// does not hide the side it lies on.
const SHARED_READS: [f64; 3] = [0.25, 0.5, 0.75];

/// A plane face.
struct Flat {
    face: FaceId,
    normal: Vec3,
    /// The plane's distance from the middle of the solid's plane faces,
    /// which moves with the solid, and which two faces on one plane share
    /// whichever way round they face.
    offset: f64,
    origin: Point3,
    bounds: Aabb3,
}

impl Flat {
    fn of(topo: &Topology, face_id: FaceId) -> Result<Option<Self>, CheckError> {
        let face = topo.face(face_id)?;
        let FaceSurface::Plane { normal, .. } = face.surface() else {
            return Ok(None);
        };
        let Ok(normal) = normal.normalize() else {
            return Ok(None);
        };
        let Some(first) = topo.wire(face.outer_wire())?.edges().first() else {
            return Ok(None);
        };
        let origin = topo.vertex(topo.edge(first.edge())?.start())?.point();
        let bounds = crate::util::face_aabb(topo, face_id)?;
        Ok(Some(Self {
            face: face_id,
            normal,
            offset: 0.0,
            origin,
            bounds,
        }))
    }

    fn size(&self) -> f64 {
        (self.bounds.max - self.bounds.min).length()
    }
}

/// Whether the two faces lie on one plane within `tol` and their boxes meet:
/// the second's first vertex lies within `tol` of the first's plane, and the
/// two planes part by no more than `tol` across the larger face. The faces
/// may face either way.
fn on_one_plane(a: &Flat, b: &Flat, tol: f64) -> bool {
    let tilt = (a.normal - b.normal)
        .length()
        .min((a.normal + b.normal).length());
    tilt <= MOST_TILT
        && tilt * a.size().max(b.size()) <= tol
        && a.normal.dot(b.origin - a.origin).abs() <= tol
        && a.bounds.expanded(tol).intersects(b.bounds)
}

/// An edge from `a` to `b` along `curve` in the plane of `frame`, as the
/// piece it is: `None` for a B-spline.
fn flat_piece(curve: &EdgeCurve, a: Point3, b: Point3, frame: &Frame3) -> Option<Boundary2> {
    let point = |p: Point3| {
        let d = p - frame.origin;
        Point2::new(d.dot(frame.x), d.dot(frame.y))
    };
    let direction = |v: Vec3| Vec2::new(v.dot(frame.x), v.dot(frame.y));
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    match curve {
        EdgeCurve::Line => Some(Boundary2::Segment(point(a), point(b))),
        EdgeCurve::NurbsCurve(_) => None,
        EdgeCurve::Circle(c) => Some(Boundary2::Arc {
            center: point(c.center()),
            u: direction(c.u_axis()),
            v: direction(c.v_axis()),
            a: c.radius(),
            b: c.radius(),
            t0,
            t1,
        }),
        EdgeCurve::Ellipse(e) => Some(Boundary2::Arc {
            center: point(e.center()),
            u: direction(e.u_axis()),
            v: direction(e.v_axis()),
            a: e.semi_major(),
            b: e.semi_minor(),
            t0,
            t1,
        }),
    }
}

/// The fraction along `piece`, from 0 to 1, of the point `x` on or beside
/// its line or curve, `None` past its ends by more than `tol`.
fn fraction_of(piece: &Boundary2, x: Point2, tol: f64) -> Option<f64> {
    match *piece {
        Boundary2::Segment(a, b) => {
            let d = b - a;
            let len2 = d.dot(d);
            if len2 <= 0.0 {
                return None;
            }
            let f = (x - a).dot(d) / len2;
            let slack = tol / len2.sqrt();
            (-slack..=1.0 + slack)
                .contains(&f)
                .then_some(f.clamp(0.0, 1.0))
        }
        Boundary2::Arc {
            center,
            u,
            v,
            a,
            b,
            t0,
            t1,
        } => {
            let w = x - center;
            let theta = (w.dot(v) / b).atan2(w.dot(u) / a);
            let slack = tol / a.min(b);
            let mut rel = (theta - t0).rem_euclid(TAU);
            if rel > TAU - slack {
                rel -= TAU;
            }
            let span = t1 - t0;
            (-slack..=span + slack)
                .contains(&rel)
                .then_some((rel / span).clamp(0.0, 1.0))
        }
    }
}

/// The ends of `piece`.
fn ends(piece: &Boundary2) -> [Point2; 2] {
    match *piece {
        Boundary2::Segment(a, b) => [a, b],
        Boundary2::Arc {
            center,
            u,
            v,
            a,
            b,
            t0,
            t1,
        } => [t0, t1].map(|t| center + u * (a * t.cos()) + v * (b * t.sin())),
    }
}

/// The points where the line through `p` along `d` meets the conic of
/// `arc`, as multiples of `d` from `p`; one point where it passes within
/// `tol` of the conic without meeting it.
fn line_meets_conic(p: Point2, d: Vec2, arc: &Boundary2, tol: f64) -> Vec<f64> {
    let Boundary2::Arc {
        center, u, v, a, b, ..
    } = *arc
    else {
        return Vec::new();
    };
    let w = p - center;
    let (x0, y0) = (w.dot(u) / a, w.dot(v) / b);
    let (dx, dy) = (d.dot(u) / a, d.dot(v) / b);
    let alpha = dx.madd(dx, dy * dy);
    if alpha <= 0.0 {
        return Vec::new();
    }
    let beta = 2.0 * x0.madd(dx, y0 * dy);
    let gamma = x0.madd(x0, y0.madd(y0, -1.0));
    let disc = beta.madd(beta, -4.0 * alpha * gamma);
    if disc >= 0.0 {
        let root = disc.sqrt();
        return vec![
            (-beta - root) / (2.0 * alpha),
            (-beta + root) / (2.0 * alpha),
        ];
    }
    let s = -beta / (2.0 * alpha);
    let (x, y) = (dx.madd(s, x0), dy.madd(s, y0));
    if (x.hypot(y) - 1.0).abs() * a.min(b) <= tol {
        vec![s]
    } else {
        Vec::new()
    }
}

/// The points where the carriers of two pieces meet, and the ends of a
/// stretch where they run along each other. `None` for an arc of an
/// ellipse against another arc, which are not solved for.
fn carriers_meet(p: &Boundary2, q: &Boundary2, tol: f64) -> Option<Vec<Point2>> {
    match (*p, *q) {
        (Boundary2::Segment(a, b), Boundary2::Segment(c, e)) => {
            let (d, f) = (b - a, e - c);
            let denom = d.x().madd(f.y(), -(d.y() * f.x()));
            let w = c - a;
            let off_line = w.x().madd(d.y(), -(w.y() * d.x()));
            if denom.abs() <= 1e-12 * d.length() * f.length() {
                // Parallel: along each other where `q` lies on `p`'s line.
                return Some(if off_line.abs() <= tol * d.length() {
                    vec![c, e]
                } else {
                    Vec::new()
                });
            }
            let s = w.x().madd(f.y(), -(w.y() * f.x())) / denom;
            Some(vec![a + d * s])
        }
        (Boundary2::Segment(a, b), Boundary2::Arc { .. }) => {
            let d = b - a;
            Some(
                line_meets_conic(a, d, q, tol)
                    .into_iter()
                    .map(|s| a + d * s)
                    .collect(),
            )
        }
        (Boundary2::Arc { .. }, Boundary2::Segment(c, e)) => {
            let f = e - c;
            Some(
                line_meets_conic(c, f, p, tol)
                    .into_iter()
                    .map(|s| c + f * s)
                    .collect(),
            )
        }
        (
            Boundary2::Arc {
                center: c1,
                a: a1,
                b: b1,
                ..
            },
            Boundary2::Arc {
                center: c2,
                a: a2,
                b: b2,
                ..
            },
        ) => {
            let circle = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
            if !circle(a1, b1) || !circle(a2, b2) {
                return None;
            }
            let between = c2 - c1;
            let gap = between.length();
            if gap <= tol {
                // One centre: along each other where the radii agree.
                return Some(if (a1 - a2).abs() <= tol {
                    ends(q).to_vec()
                } else {
                    Vec::new()
                });
            }
            if gap > a1 + a2 + tol || gap < (a1 - a2).abs() - tol {
                return Some(Vec::new());
            }
            let along = (a1 * a1 - a2 * a2 + gap * gap) / (2.0 * gap);
            let high = a1.madd(a1, -(along * along)).max(0.0).sqrt();
            let axis = between * (1.0 / gap);
            let foot = c1 + axis * along;
            let side = Vec2::new(-axis.y(), axis.x());
            Some(vec![foot + side * high, foot - side * high])
        }
    }
}

/// The fractions along `piece` at which it meets the pieces `others`, or
/// starts or ends a stretch along one of them. `None` where a pair is not
/// solved for.
fn meetings(piece: &Boundary2, others: &[Boundary2], tol: f64) -> Option<Vec<f64>> {
    let mut cuts = Vec::new();
    for other in others {
        for x in carriers_meet(piece, other, tol)? {
            if fraction_of(other, x, tol).is_some()
                && let Some(f) = fraction_of(piece, x, tol)
            {
                cuts.push(f);
            }
        }
    }
    Some(cuts)
}

/// The other face of a pair: its region, its boundary as lines and arcs
/// (`None` where it has a B-spline), and its edges.
struct Other<'a> {
    region: &'a PlanarRegion,
    pieces: Option<&'a [Boundary2]>,
    edges: &'a HashSet<EdgeId>,
    /// A box that holds the face, widened by the tolerance: an edge clear
    /// of it is clear of the face.
    bounds: Aabb3,
}

/// Whether some point lies inside both faces, farther than `tol` from the
/// boundary of each: read just inside `face`, beside each stretch of its
/// edges between the points where they meet the other face's boundary.
#[allow(clippy::too_many_lines)]
fn shares_area(
    topo: &Topology,
    face_id: FaceId,
    region: &PlanarRegion,
    other_face: &Other<'_>,
    frame: &Frame3,
    tol: f64,
) -> Result<bool, CheckError> {
    let (other, pieces) = (other_face.region, other_face.pieces);
    let flat = |p: Point3| {
        let d = p - frame.origin;
        Point2::new(d.dot(frame.x), d.dot(frame.y))
    };
    let face = topo.face(face_id)?;
    let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for wid in wires {
        for oe in topo.wire(wid)?.edges() {
            if crate::util::edge_aabb(topo, oe.edge())?
                .is_some_and(|bounds| !bounds.intersects(other_face.bounds))
            {
                continue;
            }
            let edge = topo.edge(oe.edge())?;
            let curve = edge.curve();
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            if matches!(curve, EdgeCurve::Line) && (b - a).length() <= tol {
                continue;
            }
            let (t0, t1) = curve.domain_with_endpoints(a, b);
            let at = |f: f64| (t1 - t0).madd(f, t0);
            let point = |f: f64| flat(curve.evaluate_with_endpoints(at(f), a, b));
            // A stretch's length read through its middle: a closed edge's
            // ends are one point.
            let through = |f0: f64, mid: f64, f1: f64| {
                (point(mid) - point(f0)).length() + (point(f1) - point(mid)).length()
            };
            // The edge's length along the chords between even steps and, on a
            // B-spline, its knots: a closed edge's ends are one point, and a
            // B-spline's parameter can run unevenly along it.
            let mut steps: Vec<f64> = (0..=STRETCHES)
                .map(|k| f64::from(k) / f64::from(STRETCHES))
                .collect();
            if let EdgeCurve::NurbsCurve(n) = curve {
                let (lo, hi) = (t0.min(t1), t0.max(t1));
                steps.extend(
                    n.knots()
                        .iter()
                        .filter(|&&k| k > lo && k < hi)
                        .map(|&k| (k - t0) / (t1 - t0)),
                );
                steps.sort_by(f64::total_cmp);
                steps.dedup();
            }
            let edge_length: f64 = steps
                .windows(2)
                .map(|w| (point(w[1]) - point(w[0])).length())
                .sum();
            if edge_length <= tol {
                continue;
            }
            // A point inside both faces beside the edge at `f`: each side of
            // the edge in turn, from the edge's length away down to the
            // tolerance, which `side` keeps clear of each face's boundary.
            let beside = |f: f64| {
                let m = point(f);
                let along = curve.tangent_with_endpoints(at(f), a, b);
                let Ok(across) = Vec2::new(-along.dot(frame.y), along.dot(frame.x)).normalize()
                else {
                    return false;
                };
                let mut step = edge_length;
                for _ in 0..128 {
                    if step <= tol {
                        break;
                    }
                    for s in [step, -step] {
                        let q = m + across * s;
                        if region.side(q, tol) == Some(true) && other.side(q, tol) == Some(true) {
                            return true;
                        }
                    }
                    step *= 0.5;
                }
                false
            };
            let cuts = flat_piece(curve, a, b, frame)
                .zip(pieces)
                .and_then(|(piece, others)| meetings(&piece, others, tol));
            if let Some(mut cuts) = cuts {
                // Each stretch between two cuts lies inside the other face,
                // outside it, or along its boundary.
                cuts.extend([0.0, 1.0]);
                cuts.sort_by(f64::total_cmp);
                for pair in cuts.windows(2) {
                    let (f0, f1) = (pair[0], pair[1]);
                    let mid = 0.5 * (f0 + f1);
                    if through(f0, mid, f1) > tol
                        && other.side(point(mid), tol) != Some(false)
                        && beside(mid)
                    {
                        return Ok(true);
                    }
                }
                continue;
            }
            // A bound on how far the stretch from `f0` to `f1` reaches from
            // `m`, its point at the middle: a line's half, an arc's half
            // length, and a B-spline piece's farthest control point, which
            // holds the piece when its weights are positive.
            let reach = |f0: f64, f1: f64, m: Point2| match curve {
                EdgeCurve::Line => 0.5 * (point(f1) - point(f0)).length(),
                EdgeCurve::Circle(c) => 0.5 * c.radius() * (at(f1) - at(f0)).abs(),
                EdgeCurve::Ellipse(e) => 0.5 * e.semi_major() * (at(f1) - at(f0)).abs(),
                EdgeCurve::NurbsCurve(n) => curve_bezier_between(n, at(f0), at(f1))
                    .ok()
                    .filter(|hull| hull.iter().all(|c| c[3] > 0.0))
                    .map_or(f64::INFINITY, |hull| {
                        hull.iter()
                            .map(|&[x, y, z, w]| {
                                (flat(Point3::new(x / w, y / w, z / w)) - m).length()
                            })
                            .fold(0.0, f64::max)
                    }),
            };
            let cuts = steps;
            // An edge the other face shares runs along its boundary, and each
            // face lies on one side of it all along it: it is read beside a
            // few points, not each first stretch. Two faces on either side
            // of it fail every read, each of them halved down to the
            // tolerance on both sides.
            if other_face.edges.contains(&oe.edge()) {
                for mid in SHARED_READS {
                    if beside(mid) {
                        return Ok(true);
                    }
                }
                continue;
            }
            // A stretch that reads outside or inside the other face by more
            // than its reach lies wholly there, and one inside is read beside
            // its middle. Any other is halved, and read beside its middle
            // where that lies inside the other face, or where it is halved no
            // more.
            for first in cuts.windows(2) {
                let mut pending = VecDeque::from([(first[0], first[1], 0_u32)]);
                let mut read = 0;
                while let Some((f0, f1, depth)) = pending.pop_front() {
                    read += 1;
                    let mid = 0.5 * (f0 + f1);
                    let m = point(mid);
                    match other.side(m, reach(f0, f1, m)) {
                        Some(false) => {}
                        Some(true) => {
                            if beside(mid) {
                                return Ok(true);
                            }
                        }
                        None => {
                            let last =
                                depth >= MOST_HALVINGS || read + pending.len() + 2 > MOST_STRETCHES;
                            let near = other.side(m, tol);
                            if (near == Some(true) || (last && near.is_none())) && beside(mid) {
                                return Ok(true);
                            }
                            if !last {
                                pending.push_back((f0, mid, depth + 1));
                                pending.push_back((mid, f1, depth + 1));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(false)
}

/// Check that no two plane faces of a solid cover the same area.
///
/// Reports each pair of plane faces that lie on one plane, either way round,
/// and share an area. Faces that meet only along an edge or at a point are
/// not reported. Only faces whose planes agree within `tolerance` and whose
/// boxes meet are read along their edges.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_faces_overlap(
    topo: &Topology,
    solid_id: SolidId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    faces_overlap(topo, solid_id, tolerance, None)
}

/// Check the pairs of plane faces of a solid with one face in `faces`.
///
/// Reports what [`check_faces_overlap`] reports, for an operation that
/// changed only those faces of a solid that was sound before.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_faces_overlap_among<S: std::hash::BuildHasher>(
    topo: &Topology,
    solid_id: SolidId,
    tolerance: f64,
    faces: &HashSet<FaceId, S>,
) -> Result<Vec<ValidationIssue>, CheckError> {
    faces_overlap(
        topo,
        solid_id,
        tolerance,
        Some(&|face| faces.contains(&face)),
    )
}

fn faces_overlap(
    topo: &Topology,
    solid_id: SolidId,
    tolerance: f64,
    among: Option<&dyn Fn(FaceId) -> bool>,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let mut flats = Vec::new();
    for face_id in solid_faces(topo, solid_id)? {
        if let Some(flat) = Flat::of(topo, face_id)? {
            flats.push(flat);
        }
    }
    let Some(first) = flats.first() else {
        return Ok(Vec::new());
    };
    let mut all = first.bounds;
    for flat in &flats {
        all = all.union(flat.bounds);
    }
    let centre = all.min + (all.max - all.min) * 0.5;
    let mut spread = 0.0_f64;
    for flat in &mut flats {
        flat.offset = flat.normal.dot(flat.origin - centre).abs();
        spread = spread.max((flat.origin - centre).length());
    }
    // Two faces on one plane: their offsets part by no more than the
    // tolerance and the planes' tilt over the faces' distance from the
    // middle. The faces fall in runs of offsets each within that of the
    // next, and within a run, each face is read against the faces whose
    // boxes meet its own along the axis the run's boxes spread most along.
    let window = MOST_TILT.madd(spread, tolerance);
    flats.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut first = 0;
    while first < flats.len() {
        let mut last = first + 1;
        while last < flats.len() && flats[last].offset - flats[last - 1].offset <= window {
            last += 1;
        }
        let mut run: Vec<usize> = (first..last).collect();
        let mut spread = [(f64::INFINITY, f64::NEG_INFINITY); 3];
        for &i in &run {
            for (k, (lo, hi)) in spread.iter_mut().enumerate() {
                let x = flats[i].bounds.min.0[k];
                *lo = lo.min(x);
                *hi = hi.max(x);
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| (spread[a].1 - spread[a].0).total_cmp(&(spread[b].1 - spread[b].0)))
            .unwrap_or(0);
        run.sort_by(|&i, &j| flats[i].bounds.min.0[axis].total_cmp(&flats[j].bounds.min.0[axis]));
        for (k, &i) in run.iter().enumerate() {
            for &j in &run[k + 1..] {
                if flats[j].bounds.min.0[axis] > flats[i].bounds.max.0[axis] + tolerance {
                    break;
                }
                if among.is_some_and(|among| !among(flats[i].face) && !among(flats[j].face)) {
                    continue;
                }
                if (flats[i].offset - flats[j].offset).abs() <= window
                    && on_one_plane(&flats[i], &flats[j], tolerance)
                {
                    pairs.push((i.min(j), i.max(j)));
                }
            }
        }
        first = last;
    }
    pairs.sort_unstable();
    let mut issues = Vec::new();
    for (i, j) in pairs {
        let (a, b) = (&flats[i], &flats[j]);
        let Ok(frame) = Frame3::from_normal(a.origin, a.normal) else {
            continue;
        };
        let region = |face| face_region_2d(topo, face, frame.origin, frame.x, frame.y, tolerance);
        let pieces = |face| face_boundary_2d(topo, face, frame.origin, frame.x, frame.y);
        let (Some(ra), Some(rb)) = (region(a.face)?, region(b.face)?) else {
            continue;
        };
        let (pa, pb) = (pieces(a.face)?, pieces(b.face)?);
        let edges = |face| -> Result<HashSet<EdgeId>, CheckError> {
            Ok(face_edges(topo, face)?.into_iter().collect())
        };
        let (ea, eb) = (edges(a.face)?, edges(b.face)?);
        let of_b = Other {
            region: &rb,
            pieces: pb.as_deref(),
            edges: &eb,
            bounds: b.bounds.expanded(tolerance),
        };
        let of_a = Other {
            region: &ra,
            pieces: pa.as_deref(),
            edges: &ea,
            bounds: a.bounds.expanded(tolerance),
        };
        if shares_area(topo, a.face, &ra, &of_b, &frame, tolerance)?
            || shares_area(topo, b.face, &rb, &of_a, &frame, tolerance)?
        {
            issues.push(ValidationIssue {
                check: CheckId::SolidFacesOverlap,
                severity: Severity::Error,
                entity: EntityRef::Face(a.face),
                description: format!(
                    "faces {} and {} lie on one plane and cover the same area",
                    a.face.index(),
                    b.face.index()
                ),
                deviation: None,
            });
        }
    }
    Ok(issues)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::HashMap;

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::edge::{Edge, EdgeId};
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::{Vertex, VertexId};
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Plane faces of lines through shared vertices and edges.
    #[derive(Default)]
    struct Builder {
        topo: Topology,
        vertices: Vec<(Point3, VertexId)>,
        edges: HashMap<(usize, usize), EdgeId>,
        faces: Vec<FaceId>,
    }

    impl Builder {
        fn vertex(&mut self, p: [f64; 3]) -> VertexId {
            let p = Point3::new(p[0], p[1], p[2]);
            if let Some(&(_, v)) = self.vertices.iter().find(|(q, _)| (*q - p).length() < 1e-9) {
                return v;
            }
            let v = self.topo.add_vertex(Vertex::new(p, 1e-7));
            self.vertices.push((p, v));
            v
        }

        fn edge(&mut self, a: VertexId, b: VertexId) -> OrientedEdge {
            let key = (a.index().min(b.index()), a.index().max(b.index()));
            let id = *self
                .edges
                .entry(key)
                .or_insert_with(|| self.topo.add_edge(Edge::new(a, b, EdgeCurve::Line)));
            OrientedEdge::new(id, self.topo.edge(id).unwrap().start() == a)
        }

        /// A face through `corners`, run counter-clockwise about `normal`.
        fn face(&mut self, corners: &[[f64; 3]], normal: [f64; 3]) -> FaceId {
            let normal = Vec3::new(normal[0], normal[1], normal[2])
                .normalize()
                .unwrap();
            let points: Vec<Point3> = corners
                .iter()
                .map(|c| Point3::new(c[0], c[1], c[2]))
                .collect();
            let mut turn = Vec3::new(0.0, 0.0, 0.0);
            for k in 0..points.len() {
                let (a, b) = (points[k], points[(k + 1) % points.len()]);
                turn += Vec3::new(a.x(), a.y(), a.z()).cross(Vec3::new(b.x(), b.y(), b.z()));
            }
            let mut ids: Vec<VertexId> = corners.iter().map(|&c| self.vertex(c)).collect();
            if turn.dot(normal) < 0.0 {
                ids.reverse();
            }
            let edges = (0..ids.len())
                .map(|k| self.edge(ids[k], ids[(k + 1) % ids.len()]))
                .collect();
            let wire = self.topo.add_wire(Wire::new(edges, true).unwrap());
            let origin = Vec3::new(corners[0][0], corners[0][1], corners[0][2]);
            let face = self.topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal,
                    d: normal.dot(origin),
                },
            ));
            self.faces.push(face);
            face
        }

        fn solid(&mut self) -> SolidId {
            let shell = self.topo.add_shell(Shell::new(self.faces.clone()).unwrap());
            self.topo.add_solid(Solid::new(shell, vec![]))
        }
    }

    /// A 10 cube less a chamfer of 1 along its edge on y = 0 and z = 10.
    /// With `patched`, each end wall keeps its whole square, and a triangle
    /// on it, facing into the cube, closes the chamfer's end.
    fn chamfered_cube(patched: bool) -> (Topology, SolidId) {
        let mut b = Builder::default();
        b.face(
            &[[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]],
            [0., 0., -1.],
        );
        b.face(
            &[
                [0., 10., 0.],
                [10., 10., 0.],
                [10., 10., 10.],
                [0., 10., 10.],
            ],
            [0., 1., 0.],
        );
        b.face(
            &[
                [0., 1., 10.],
                [10., 1., 10.],
                [10., 10., 10.],
                [0., 10., 10.],
            ],
            [0., 0., 1.],
        );
        b.face(
            &[[0., 0., 0.], [10., 0., 0.], [10., 0., 9.], [0., 0., 9.]],
            [0., -1., 0.],
        );
        b.face(
            &[[0., 0., 9.], [10., 0., 9.], [10., 1., 10.], [0., 1., 10.]],
            [0., -1., 1.],
        );
        for (x, out) in [(0.0, -1.0), (10.0, 1.0)] {
            let mut wall = vec![[x, 0., 0.], [x, 10., 0.], [x, 10., 10.], [x, 1., 10.]];
            if patched {
                wall.push([x, 0., 10.]);
                b.face(&[[x, 0., 9.], [x, 1., 10.], [x, 0., 10.]], [-out, 0., 0.]);
            }
            wall.push([x, 0., 9.]);
            b.face(&wall, [out, 0., 0.]);
        }
        let solid = b.solid();
        (b.topo, solid)
    }

    fn found(topo: &Topology, solid: SolidId) -> Vec<String> {
        check_faces_overlap(topo, solid, 1e-6)
            .unwrap()
            .into_iter()
            .map(|i| i.description)
            .collect()
    }

    #[test]
    fn a_patch_on_its_end_wall_is_reported() {
        let (topo, solid) = chamfered_cube(true);
        let got = found(&topo, solid);
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(got.iter().all(|d| d.contains("cover the same area")));
    }

    #[test]
    fn a_chamfered_cube_passes() {
        let (topo, solid) = chamfered_cube(false);
        assert!(found(&topo, solid).is_empty());
    }

    /// A box's top split in two along x = 4, each half its own face.
    #[test]
    fn two_halves_of_a_face_pass() {
        let mut b = Builder::default();
        b.face(
            &[[0., 0., 10.], [4., 0., 10.], [4., 10., 10.], [0., 10., 10.]],
            [0., 0., 1.],
        );
        b.face(
            &[
                [4., 0., 10.],
                [10., 0., 10.],
                [10., 10., 10.],
                [4., 10., 10.],
            ],
            [0., 0., 1.],
        );
        let solid = b.solid();
        assert!(found(&b.topo, solid).is_empty());
    }

    /// Two faces on one plane, facing one way, that overlap in a strip at
    /// the side of each, and a face that lies inside another.
    #[test]
    fn faces_that_overlap_facing_one_way_are_reported() {
        let mut b = Builder::default();
        b.face(
            &[[0., 0., 0.], [6., 0., 0.], [6., 10., 0.], [0., 10., 0.]],
            [0., 0., 1.],
        );
        b.face(
            &[[5., 0., 0.], [10., 0., 0.], [10., 10., 0.], [5., 10., 0.]],
            [0., 0., 1.],
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);

        let mut b = Builder::default();
        b.face(
            &[[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]],
            [0., 0., 1.],
        );
        b.face(
            &[[2., 2., 0.], [3., 2., 0.], [3., 3., 0.], [2., 3., 0.]],
            [0., 0., -1.],
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// A square face with a round hole and the disc that fills it, on one
    /// plane, share the circle and nothing more. A disc a little off the
    /// hole's centre reaches across the hole's rim.
    #[test]
    fn a_disc_in_a_rings_hole_passes_and_one_across_its_rim_is_reported() {
        for (centre, overlaps) in [(5.0, false), (5.5, true)] {
            let mut b = Builder::default();
            let ring = b.face(
                &[[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]],
                [0., 0., 1.],
            );
            let z = Vec3::new(0.0, 0.0, 1.0);
            let circle = |b: &mut Builder, c: f64| {
                let circle = Circle3D::new(Point3::new(c, 5.0, 0.0), z, 2.0).unwrap();
                let v = b.vertex([c + 2.0, 5.0, 0.0]);
                let e = b.topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
                (
                    e,
                    b.topo
                        .add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap()),
                )
            };
            let (_, hole) = circle(&mut b, 5.0);
            let (rim, _) = circle(&mut b, centre);
            b.topo.face_mut(ring).unwrap().inner_wires_mut().push(hole);
            let disc = b
                .topo
                .add_wire(Wire::new(vec![OrientedEdge::new(rim, false)], true).unwrap());
            let face = b.topo.add_face(Face::new(
                disc,
                vec![],
                FaceSurface::Plane { normal: z, d: 0.0 },
            ));
            b.faces.push(face);
            let solid = b.solid();
            assert_eq!(
                !found(&b.topo, solid).is_empty(),
                overlaps,
                "centre {centre}"
            );
        }
    }

    fn square(b: &mut Builder, (x, y): (f64, f64), side: f64, normal: [f64; 3]) {
        b.face(
            &[
                [x, y, 0.0],
                [x + side, y, 0.0],
                [x + side, y + side, 0.0],
                [x, y + side, 0.0],
            ],
            normal,
        );
    }

    /// Two squares whose corners overlap by 0.1, and two that share a strip
    /// 0.001 wide: no point at a fixed fraction of an edge finds the shared
    /// area, a point beside the edge stretch inside the other face does.
    #[test]
    fn a_small_corner_or_a_narrow_strip_in_common_is_reported() {
        for (far, normal) in [((9.9, 9.9), [0., 0., 1.]), ((9.999, 0.0), [0., 0., -1.])] {
            let mut b = Builder::default();
            square(&mut b, (0.0, 0.0), 10.0, [0., 0., 1.]);
            square(&mut b, far, 10.0, normal);
            let solid = b.solid();
            assert_eq!(found(&b.topo, solid).len(), 1, "{far:?}");
        }
        // Squares that meet along an edge, or at a corner, share no area.
        for far in [(10.0, 0.0), (10.0, 10.0), (10.0, 3.0)] {
            let mut b = Builder::default();
            square(&mut b, (0.0, 0.0), 10.0, [0., 0., 1.]);
            square(&mut b, far, 10.0, [0., 0., -1.]);
            let solid = b.solid();
            assert!(found(&b.topo, solid).is_empty(), "{far:?}");
        }
    }

    /// Two faces 0.01 across, one on the other, are read at the scale of
    /// the tolerance, not of their size.
    #[test]
    fn two_small_faces_one_on_the_other_are_reported() {
        let mut b = Builder::default();
        square(&mut b, (0.0, 0.0), 0.01, [0., 0., 1.]);
        b.face(
            &[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.01, 0.01, 0.0]],
            [0., 0., -1.],
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// Far from the origin, two faces whose planes part by 1e-8 radians lie
    /// within 1e-7 of each other: their offsets from the origin part by far
    /// more than the tolerance, and from the solid's middle they do not.
    #[test]
    fn faces_far_from_the_origin_on_one_tilted_plane_are_reported() {
        let mut b = Builder::default();
        let x0 = 1e6;
        square(&mut b, (x0, 0.0), 10.0, [0., 0., 1.]);
        let z = |x: f64| -1e-8 * (x - x0);
        b.face(
            &[
                [x0, 0.0, z(x0)],
                [x0 + 10.0, 0.0, z(x0 + 10.0)],
                [x0 + 10.0, 10.0, z(x0 + 10.0)],
                [x0, 10.0, z(x0)],
            ],
            [1e-8, 0.0, 1.0],
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// A half disc on a square, its arc across the square's edge.
    #[test]
    fn an_arc_across_an_edge_finds_the_area_beyond_it() {
        let mut b = Builder::default();
        square(&mut b, (0.0, 0.0), 10.0, [0., 0., 1.]);
        let z = Vec3::new(0.0, 0.0, 1.0);
        let circle = Circle3D::new(Point3::new(10.05, 5.0, 0.0), z, 0.1).unwrap();
        let (p, q) = (b.vertex([10.05, 5.1, 0.0]), b.vertex([10.05, 4.9, 0.0]));
        let arc = b.topo.add_edge(Edge::new(p, q, EdgeCurve::Circle(circle)));
        let chord = b.edge(q, p);
        let wire = b
            .topo
            .add_wire(Wire::new(vec![OrientedEdge::new(arc, true), chord], true).unwrap());
        let face = b.topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: -z, d: 0.0 },
        ));
        b.faces.push(face);
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// Two squares that share a strip ten tolerances wide.
    #[test]
    fn a_strip_ten_tolerances_wide_in_common_is_reported() {
        let mut b = Builder::default();
        square(&mut b, (0.0, 0.0), 10.0, [0., 0., 1.]);
        square(&mut b, (9.999_99, 0.0), 10.0, [0., 0., -1.]);
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// Two discs about one centre, each bounded by one closed circle, the
    /// smaller one on the larger.
    #[test]
    fn a_disc_on_a_larger_disc_is_reported() {
        let mut b = Builder::default();
        let z = Vec3::new(0.0, 0.0, 1.0);
        for (r, normal) in [(2.0, z), (1.0, -z)] {
            let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), z, r).unwrap();
            let v = b.vertex([r, 0.0, 0.0]);
            let e = b.topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let wire = b
                .topo
                .add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
            let face = b.topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane { normal, d: 0.0 },
            ));
            b.faces.push(face);
        }
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// Two overlapping squares on one plane whose stored normals part by
    /// 2e-9 radians. The normals run round the circle square to `d`, and the
    /// hair of tilt lies along `d`: a reading that turns each normal to the
    /// side of `d` splits the two normals of each pair across that circle.
    #[test]
    fn faces_whose_normals_part_by_a_hair_are_read_on_one_plane() {
        let d = Vec3::new(0.577_215_664_9, 0.618_033_988_7, 0.533_102_917_2)
            .normalize()
            .unwrap();
        let ring = Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), d).unwrap();
        for k in 0..8 {
            let turn = std::f64::consts::FRAC_PI_4 * f64::from(k);
            let n = ring.x * turn.cos() + ring.y * turn.sin();
            let frame = Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), n).unwrap();
            let hair = d * 1e-9;
            let mut b = Builder::default();
            let corner = |x: f64, y: f64| {
                let p = Point3::new(0.0, 0.0, 0.0) + frame.x * x + frame.y * y;
                [p.x(), p.y(), p.z()]
            };
            for ((x, y), normal) in [((0.0, 0.0), n + hair), ((5.0, 5.0), -(n - hair))] {
                b.face(
                    &[
                        corner(x, y),
                        corner(x + 10.0, y),
                        corner(x + 10.0, y + 10.0),
                        corner(x, y + 10.0),
                    ],
                    [normal.x(), normal.y(), normal.z()],
                );
            }
            let solid = b.solid();
            assert_eq!(found(&b.topo, solid).len(), 1, "turn {k}");
        }
    }

    /// Two squares seven tolerances across, one on the other.
    #[test]
    fn two_squares_seven_tolerances_across_one_on_the_other_are_reported() {
        let mut b = Builder::default();
        square(&mut b, (0.0, 0.0), 7e-6, [0., 0., 1.]);
        square(&mut b, (0.0, 0.0), 7e-6, [0., 0., -1.]);
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// A rectangle whose edges are straight B-splines, each its own edge.
    fn spline_rectangle(b: &mut Builder, (x0, y0): (f64, f64), (x1, y1): (f64, f64)) {
        let corners = [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]];
        let ids: Vec<VertexId> = corners.iter().map(|&c| b.vertex(c)).collect();
        let edges = (0..4)
            .map(|k| {
                let (p, q) = (corners[k], corners[(k + 1) % 4]);
                let line = brepkit_math::nurbs::curve::NurbsCurve::new(
                    1,
                    vec![0.0, 0.0, 1.0, 1.0],
                    vec![Point3::new(p[0], p[1], p[2]), Point3::new(q[0], q[1], q[2])],
                    vec![1.0, 1.0],
                )
                .unwrap();
                let e = b.topo.add_edge(Edge::new(
                    ids[k],
                    ids[(k + 1) % 4],
                    EdgeCurve::NurbsCurve(line),
                ));
                OrientedEdge::new(e, true)
            })
            .collect();
        let wire = b.topo.add_wire(Wire::new(edges, true).unwrap());
        let face = b.topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        b.faces.push(face);
    }

    /// Two thin rectangles of B-spline edges that cross: none of the points
    /// first read along either rectangle's edges lies inside the other.
    #[test]
    fn two_thin_spline_rectangles_that_cross_are_reported() {
        let mut b = Builder::default();
        spline_rectangle(&mut b, (0.0, 0.020), (10.0, 0.021));
        spline_rectangle(&mut b, (0.100, -5.0), (0.101, 5.0));
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);

        let mut b = Builder::default();
        spline_rectangle(&mut b, (0.0, 0.020), (10.0, 0.021));
        spline_rectangle(&mut b, (0.100, 0.03), (0.101, 5.0));
        let solid = b.solid();
        assert!(found(&b.topo, solid).is_empty());
    }

    /// The straight edge from `a` to `b` of the builder, its line made a
    /// B-spline of degree 1, so its faces are read through samples.
    fn as_spline(b: &mut Builder, from: [f64; 3], to: [f64; 3]) {
        let (va, vb) = (b.vertex(from), b.vertex(to));
        let key = (va.index().min(vb.index()), va.index().max(vb.index()));
        let id = b.edges[&key];
        let edge = b.topo.edge(id).unwrap();
        let points = [edge.start(), edge.end()].map(|v| b.topo.vertex(v).unwrap().point());
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            points.to_vec(),
            vec![1.0, 1.0],
        )
        .unwrap();
        b.topo
            .edge_mut(id)
            .unwrap()
            .set_curve(EdgeCurve::NurbsCurve(line));
    }

    /// Two faces sharing an edge, each with a B-spline edge elsewhere: the
    /// shared edge is read beside a few of its points. Faces either side
    /// of it pass, and a face on the same side as the other is reported.
    #[test]
    fn faces_sharing_an_edge_beside_splines_are_read_on_its_sides() {
        let z = [0.0, 0.0, 1.0];
        for (other, overlap) in [(20.0, false), (5.0, true)] {
            let mut b = Builder::default();
            b.face(
                &[[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]],
                z,
            );
            b.face(
                &[
                    [10., 0., 0.],
                    [other, 0., 0.],
                    [other, 10., 0.],
                    [10., 10., 0.],
                ],
                z,
            );
            as_spline(&mut b, [0., 10., 0.], [0., 0., 0.]);
            as_spline(&mut b, [other, 0., 0.], [other, 10., 0.]);
            let solid = b.solid();
            assert_eq!(
                !found(&b.topo, solid).is_empty(),
                overlap,
                "other side at {other}"
            );
        }
    }

    /// A face whose outline is one closed straight B-spline through
    /// `corners`, closing at the first corner.
    fn spline_outline(b: &mut Builder, corners: &[[f64; 2]], normal: Vec3) {
        let mut points: Vec<Point3> = corners
            .iter()
            .map(|c| Point3::new(c[0], c[1], 0.0))
            .collect();
        points.push(points[0]);
        let n = points.len();
        #[allow(clippy::cast_precision_loss)]
        let mut knots: Vec<f64> = (0..n).map(|k| k as f64).collect();
        knots.insert(0, 0.0);
        knots.push(knots[knots.len() - 1]);
        let line =
            brepkit_math::nurbs::curve::NurbsCurve::new(1, knots, points, vec![1.0; n]).unwrap();
        let v = b.vertex([corners[0][0], corners[0][1], 0.0]);
        let e = b
            .topo
            .add_edge(Edge::new(v, v, EdgeCurve::NurbsCurve(line)));
        let wire = b
            .topo
            .add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let face = b.topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal, d: 0.0 },
        ));
        b.faces.push(face);
    }

    /// Two squares side by side, each outline one closed B-spline that
    /// closes on the edge they share. A tab of the lower square reaches into
    /// the upper one. The stretches along the shared edge do not keep the
    /// rest of each outline from being read.
    #[test]
    fn a_tab_past_a_shared_edge_of_spline_outlines_is_reported() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let mut b = Builder::default();
        spline_outline(
            &mut b,
            &[
                [10.0, 0.0],
                [5.0, 0.0],
                [5.0, 1.0],
                [4.0, 1.0],
                [4.0, 0.0],
                [0.0, 0.0],
                [0.0, -10.0],
                [10.0, -10.0],
            ],
            z,
        );
        spline_outline(
            &mut b,
            &[[10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            -z,
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);

        let mut b = Builder::default();
        spline_outline(
            &mut b,
            &[[10.0, 0.0], [0.0, 0.0], [0.0, -10.0], [10.0, -10.0]],
            z,
        );
        spline_outline(
            &mut b,
            &[[10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            -z,
        );
        let solid = b.solid();
        assert!(found(&b.topo, solid).is_empty());
    }

    /// A face under a B-spline of 64 cubic spans `y = 0.02 u (u - 0.5)
    /// (u - 1)`, each span's ends and middle on `y = 0`, and a rectangle
    /// above `y = 0` along it: each span's hump above the line lies in both.
    #[test]
    fn a_spline_edge_that_crosses_a_line_between_its_samples_is_reported() {
        let spans = 64_u32;
        let mut control = Vec::new();
        for k in 0..spans {
            let x = f64::from(k);
            if k == 0 {
                control.push(Point3::new(x, 0.0, 0.0));
            }
            control.push(Point3::new(x + 1.0 / 3.0, 0.02 / 6.0, 0.0));
            control.push(Point3::new(x + 2.0 / 3.0, -0.02 / 6.0, 0.0));
            control.push(Point3::new(x + 1.0, 0.0, 0.0));
        }
        let mut knots = vec![0.0; 4];
        for k in 1..spans {
            knots.extend([f64::from(k); 3]);
        }
        knots.extend([f64::from(spans); 4]);
        let n = control.len();
        let wavy =
            brepkit_math::nurbs::curve::NurbsCurve::new(3, knots, control, vec![1.0; n]).unwrap();
        let top = f64::from(spans);
        let mut b = Builder::default();
        let [p, q, r, t] = [[0.0, 0.0], [top, 0.0], [top, -10.0], [0.0, -10.0]]
            .map(|c| b.vertex([c[0], c[1], 0.0]));
        let curve = b
            .topo
            .add_edge(Edge::new(p, q, EdgeCurve::NurbsCurve(wavy)));
        let edges = vec![
            OrientedEdge::new(curve, true),
            b.edge(q, r),
            b.edge(r, t),
            b.edge(t, p),
        ];
        let wire = b.topo.add_wire(Wire::new(edges, true).unwrap());
        let z = Vec3::new(0.0, 0.0, 1.0);
        let face = b.topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: -z, d: 0.0 },
        ));
        b.faces.push(face);
        b.face(
            &[
                [0.0, 0.0, 0.0],
                [top, 0.0, 0.0],
                [top, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            [0.0, 0.0, 1.0],
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// Two discs twenty tolerances across, one on the other, each bounded by
    /// one closed B-spline circle: each of the 64 first stretches of a rim
    /// is about a tolerance long.
    #[test]
    fn two_small_spline_discs_one_on_the_other_are_reported() {
        let r = 1e-5;
        let mut b = Builder::default();
        let z = Vec3::new(0.0, 0.0, 1.0);
        for normal in [z, -z] {
            let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), z, r).unwrap();
            let rim = brepkit_geometry::convert::curve_to_nurbs::circle_to_nurbs(&circle, 0.0, TAU)
                .unwrap();
            let at = circle.evaluate(0.0);
            let v = b.vertex([at.x(), at.y(), at.z()]);
            let e = b.topo.add_edge(Edge::new(v, v, EdgeCurve::NurbsCurve(rim)));
            let wire = b
                .topo
                .add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
            let face = b.topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane { normal, d: 0.0 },
            ));
            b.faces.push(face);
        }
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// Two squares, one on the other, each bounded by one closed straight
    /// B-spline whose knots crowd its parameter into the first fifth: its
    /// ends and middle lie within 1.25e-7 of each other.
    #[test]
    fn faces_bounded_by_a_spline_with_crowded_knots_are_reported() {
        let mut b = Builder::default();
        let z = Vec3::new(0.0, 0.0, 1.0);
        for normal in [z, -z] {
            let points = [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
                [0.0, 1e-7],
                [0.0, 0.0],
            ]
            .map(|c| Point3::new(c[0], c[1], 0.0))
            .to_vec();
            let outline = brepkit_math::nurbs::curve::NurbsCurve::new(
                1,
                vec![0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 1.0, 1.0],
                points,
                vec![1.0; 6],
            )
            .unwrap();
            let v = b.vertex([0.0, 0.0, 0.0]);
            let e = b
                .topo
                .add_edge(Edge::new(v, v, EdgeCurve::NurbsCurve(outline)));
            let wire = b
                .topo
                .add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
            let face = b.topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane { normal, d: 0.0 },
            ));
            b.faces.push(face);
        }
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// Two squares three tolerances across, one on the other: their middle
    /// lies a tolerance and a half from each edge.
    #[test]
    fn two_squares_three_tolerances_across_one_on_the_other_are_reported() {
        let mut b = Builder::default();
        square(&mut b, (0.0, 0.0), 3e-6, [0., 0., 1.]);
        square(&mut b, (0.0, 0.0), 3e-6, [0., 0., -1.]);
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }
}
