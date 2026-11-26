//! A closed rim whose outline bends tighter than the round's radius.
//!
//! A round of a plane cap's rim on a wall standing square to it touches
//! the cap the radius in from the outline. Where the outline bends on its
//! convex side tighter than the radius, that inset runs backwards and
//! crosses itself, and no ball can roll round the tip between the two
//! feet of the crossing: the ball centred over the crossing touches the
//! wall at both feet at once. The rim is rounded from one foot round to
//! the other the long way, and the tip between them is left sharp: each
//! end of the round closes on its section through the ball over the
//! crossing, which touches the cap at one point.

use std::collections::{HashMap, HashSet};

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use brepkit_math::madd::Madd;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::fitting::interpolate;
use brepkit_math::nurbs::knot_ops::curve_split;
use brepkit_math::nurbs::projection::project_point_to_curve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::kinked_rim::{KINK, ON, closure_kink, faces_using, ruling, vertex_on_wire};
use crate::radius_law::RadiusLaw;
use crate::section::CircSection;
use crate::trimmer::SplitScope;
use crate::walker::interpolate_blend_surface;

/// Samples taken round the outline to find where its inset runs back.
const SAMPLES: u32 = 4000;

/// The outline in its cap's plane and the side its material lies on.
#[derive(Clone)]
pub struct Outline {
    curve: NurbsCurve,
    /// The cap's plane normal.
    normal: Vec3,
    /// The cap's outward normal.
    up: Vec3,
    /// +1 when the material lies on `normal x tangent`'s side, else -1.
    side: f64,
    radius: f64,
}

impl Outline {
    /// The parameter a fraction `s` of the way round.
    fn at(&self, s: f64) -> f64 {
        let (u0, u1) = self.curve.domain();
        (u1 - u0).madd(s.rem_euclid(1.0), u0)
    }

    /// The point a fraction `s` of the way round.
    fn point(&self, s: f64) -> Point3 {
        self.curve.evaluate(self.at(s))
    }

    /// Where a ball standing on the wall at `s` touches the cap.
    fn inset(&self, s: f64) -> Option<Point3> {
        let tangent = self.curve.derivatives(self.at(s), 1)[1];
        let inward = self.normal.cross(tangent).normalize().ok()? * self.side;
        Some(self.point(s) + inward * self.radius)
    }
}

/// A point's coordinates in the plane through `origin` with axes `x`, `y`.
fn plane_xy(p: Point3, origin: Point3, x: Vec3, y: Vec3) -> (f64, f64) {
    let d = p - origin;
    (d.dot(x), d.dot(y))
}

/// Where segments `a0 a1` and `b0 b1` cross, as fractions along each.
fn segments_cross(
    a0: (f64, f64),
    a1: (f64, f64),
    b0: (f64, f64),
    b1: (f64, f64),
) -> Option<(f64, f64)> {
    let r = (a1.0 - a0.0, a1.1 - a0.1);
    let s = (b1.0 - b0.0, b1.1 - b0.1);
    let denom = r.0 * s.1 - r.1 * s.0;
    if denom.abs() < 1e-300 {
        return None;
    }
    let q = (b0.0 - a0.0, b0.1 - a0.1);
    let t = (q.0 * s.1 - q.1 * s.0) / denom;
    let u = (q.0 * r.1 - q.1 * r.0) / denom;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then_some((t, u))
}

/// The fractions round the outline of the two feet of each crossing of its
/// inset, the tip between them running forward from the first.
fn tips(outline: &Outline) -> Option<Vec<(f64, f64)>> {
    let n = SAMPLES as usize;
    let origin = outline.point(0.0);
    let x = outline.curve.derivatives(outline.at(0.0), 1)[1]
        .normalize()
        .ok()?;
    let y = outline.normal.cross(x).normalize().ok()?;
    #[allow(clippy::cast_precision_loss)]
    let frac = |k: usize| k as f64 / f64::from(SAMPLES);
    let mut flat = Vec::with_capacity(n);
    let mut forward = Vec::with_capacity(n);
    for k in 0..n {
        let s = frac(k);
        flat.push(plane_xy(outline.inset(s)?, origin, x, y));
        let tangent = outline.curve.derivatives(outline.at(s), 1)[1];
        let next = outline.inset(s + 0.25 / f64::from(SAMPLES))?;
        let before = outline.inset(s - 0.25 / f64::from(SAMPLES))?;
        forward.push((next - before).dot(tangent) > 0.0);
    }
    if forward.iter().all(|&f| !f) {
        return None;
    }
    // Each run of samples where the inset runs backwards.
    let start = forward.iter().position(|&f| f)?;
    let mut runs = Vec::new();
    let mut k = 0;
    while k < n {
        let i = (start + k) % n;
        if forward[i] {
            k += 1;
            continue;
        }
        let first = k;
        while k < n && !forward[(start + k) % n] {
            k += 1;
        }
        runs.push(((start + first) % n, k - first));
    }
    let mut found = Vec::new();
    for (first, len) in runs {
        // The nearest crossing of a segment before the run with one after it.
        let reach = n / 4;
        let segment = |i: usize| (flat[i % n], flat[(i + 1) % n]);
        let mut best: Option<(usize, f64, f64)> = None;
        'outer: for back in 1..reach {
            let i = (first + n - back) % n;
            let (a0, a1) = segment(i);
            for ahead in 0..reach {
                let j = (first + len + ahead) % n;
                let (b0, b1) = segment(j);
                if let Some((t, u)) = segments_cross(a0, a1, b0, b1) {
                    let a = frac(i) + t / f64::from(SAMPLES);
                    let b = frac(j) + u / f64::from(SAMPLES);
                    best = Some((back + ahead, a, b));
                    break 'outer;
                }
            }
        }
        let (_, a, b) = best?;
        found.push(refine(outline, a, b)?);
    }
    Some(found)
}

/// Newton on the two feet `a`, `b` whose insets meet.
fn refine(outline: &Outline, mut a: f64, mut b: f64) -> Option<(f64, f64)> {
    let x = outline.curve.derivatives(outline.at(a), 1)[1]
        .normalize()
        .ok()?;
    let y = outline.normal.cross(x).normalize().ok()?;
    let origin = outline.point(a);
    let gap = |a: f64, b: f64| -> Option<(f64, f64)> {
        let d = outline.inset(a)? - outline.inset(b)?;
        Some((d.dot(x), d.dot(y)))
    };
    let h = 1e-7;
    for _ in 0..50 {
        let g = gap(a, b)?;
        let ga = gap(a + h, b)?;
        let gb = gap(a, b + h)?;
        let (j00, j10) = ((ga.0 - g.0) / h, (ga.1 - g.1) / h);
        let (j01, j11) = ((gb.0 - g.0) / h, (gb.1 - g.1) / h);
        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-300 {
            return None;
        }
        let da = (g.0 * j11 - g.1 * j01) / det;
        let db = (j00 * g.1 - j10 * g.0) / det;
        a -= da;
        b -= db;
        if da.abs().max(db.abs()) < 1e-14 {
            break;
        }
    }
    let g = gap(a, b)?;
    let scale = (origin - Point3::new(0.0, 0.0, 0.0)).length().max(1.0);
    (g.0.hypot(g.1) <= 10.0 * ON * scale).then_some((a.rem_euclid(1.0), b.rem_euclid(1.0)))
}

/// Whether fraction `s` lies on the tip running forward from `a` to `b`.
fn on_tip(s: f64, (a, b): (f64, f64)) -> bool {
    if a <= b {
        s > a && s < b
    } else {
        s > a || s < b
    }
}

/// A rim cut at its tips' feet: each edge cut by its pieces, the tips'
/// feet as fractions round the outline, and the outline.
type TightRim = (HashMap<EdgeId, [EdgeId; 2]>, Vec<(f64, f64)>, Outline);

/// The tips of one picked rim: the rim's pieces, those on a tip marked.
fn split_tight_rim(
    topo: &mut Topology,
    scope: &SplitScope,
    solid: SolidId,
    edge: EdgeId,
    radius: f64,
) -> Result<Option<TightRim>, BlendError> {
    let Ok(data) = topo.edge(edge) else {
        return Ok(None);
    };
    let EdgeCurve::NurbsCurve(curve) = data.curve().clone() else {
        return Ok(None);
    };
    if data.start() != data.end() || !closure_kink(topo, edge).is_some_and(|k| k <= KINK) {
        return Ok(None);
    }
    let faces = faces_using(topo, solid, edge)?;
    let (mut cap, mut wall) = (None, None);
    for &face in &faces {
        match topo.face(face)?.surface() {
            FaceSurface::Plane { normal, .. } => cap = Some((face, *normal)),
            FaceSurface::Nurbs(_) => wall = Some(face),
            _ => {}
        }
    }
    let (Some((cap, normal)), Some(wall), 2) = (cap, wall, faces.len()) else {
        return Ok(None);
    };
    let wall_face = topo.face(wall)?;
    let (wall_surface, reversed) = (wall_face.surface().clone(), wall_face.is_reversed());
    let cap_reversed = topo.face(cap)?.is_reversed();
    let mut outline = Outline {
        curve: curve.clone(),
        normal,
        up: if cap_reversed { -normal } else { normal },
        side: 1.0,
        radius,
    };
    // The wall must stand square to the cap all round, so that the round
    // touches the cap the radius in from the outline.
    for k in 0..16 {
        let s = f64::from(k) / 16.0;
        let (p, q) = (outline.point(s), outline.point(s + 1e-3));
        let Some(r) = ruling(&wall_surface, p, q) else {
            return Ok(None);
        };
        let Ok(up) = (r.far - p).normalize() else {
            return Ok(None);
        };
        if up.cross(normal).length() > 1e-6 {
            return Ok(None);
        }
    }
    // The material lies away from the wall's outward normal.
    let p = outline.point(0.0);
    let Some((u, v)) = wall_surface.project_point(p) else {
        return Ok(None);
    };
    let mut out = wall_surface.normal(u, v);
    if reversed {
        out = -out;
    }
    let tangent = curve.derivatives(outline.at(0.0), 1)[1];
    outline.side = if normal.cross(tangent).dot(out) > 0.0 {
        -1.0
    } else {
        1.0
    };
    let Some(found) = tips(&outline) else {
        return Ok(None);
    };
    // The rim's own vertex, where the wall's seam meets it, must lie on the
    // round, not on a tip.
    if found.is_empty()
        || found
            .iter()
            .any(|&tip| on_tip(0.0, tip) || on_tip(1.0 - 1e-9, tip))
    {
        return Ok(None);
    }
    let mut cuts = HashMap::new();
    for &(a, b) in &found {
        for s in [a, b] {
            let point = outline.point(s);
            match vertex_on_wire(topo, scope, cap, point)? {
                Some((_, Some((whole, pieces)))) => {
                    cuts.insert(whole, pieces);
                }
                Some((_, None)) => {}
                None => return Ok(None),
            }
        }
    }
    Ok(Some((cuts, found, outline)))
}

/// Rounds each picked closed rim that bends tighter than its radius from
/// one foot of its inset's crossing to the other, the tip left sharp.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn split_tight_rims(
    topo: &mut Topology,
    scope: &SplitScope,
    solid: SolidId,
    edge_sets: &mut [(Vec<EdgeId>, RadiusLaw)],
) -> Result<Vec<Tip>, BlendError> {
    let mut tips_left = Vec::new();
    for (edges, law) in edge_sets.iter_mut() {
        let RadiusLaw::Constant(radius) = *law else {
            continue;
        };
        let mut picked = Vec::with_capacity(edges.len() + 2);
        for &edge in edges.iter() {
            let Some((cuts, found, outline)) = split_tight_rim(topo, scope, solid, edge, radius)?
            else {
                picked.push(edge);
                continue;
            };
            log::debug!(
                "closed rim {edge:?} bends tighter than r = {radius}: tips {found:?} left sharp"
            );
            let curve = &outline.curve;
            let (u0, u1) = curve.domain();
            let mut dropped = HashSet::new();
            let mut stack = vec![edge];
            let mut pieces = Vec::new();
            while let Some(e) = stack.pop() {
                if let Some(&[first, second]) = cuts.get(&e) {
                    stack.push(second);
                    stack.push(first);
                } else {
                    pieces.push(e);
                }
            }
            for &piece in &pieces {
                let EdgeCurve::NurbsCurve(own) = topo.edge(piece)?.curve() else {
                    continue;
                };
                let (d0, d1) = own.domain();
                let middle = own.evaluate(0.5f64.madd(d1 - d0, d0));
                let hit = project_point_to_curve(curve, middle, ON)?;
                let s = (hit.parameter - u0) / (u1 - u0);
                if found.iter().any(|&tip| on_tip(s, tip)) {
                    dropped.insert(piece);
                }
            }
            picked.extend(pieces.into_iter().filter(|p| !dropped.contains(p)));
            tips_left.extend(found.iter().map(|&(a, b)| Tip {
                outline: outline.clone(),
                a,
                b,
            }));
        }
        *edges = picked;
    }
    Ok(tips_left)
}

/// A tip left sharp by [`split_tight_rims`], to be rounded once the rest of
/// the rim is: the outline it lies on and the fractions round it of its two
/// feet.
pub struct Tip {
    outline: Outline,
    a: f64,
    b: f64,
}

impl Outline {
    /// The direction into the material across the outline at `s`.
    fn inward(&self, s: f64) -> Option<Vec3> {
        let tangent = self.curve.derivatives(self.at(s), 1)[1];
        Some(self.normal.cross(tangent).normalize().ok()? * self.side)
    }

    /// The ball's centre standing on the rim at `s`.
    fn centre(&self, s: f64) -> Option<Point3> {
        Some(self.point(s) + self.inward(s)? * self.radius - self.up * self.radius)
    }

    /// The round's point at `s`, `theta` from the cap towards the wall.
    fn band(&self, s: f64, theta: f64) -> Option<Point3> {
        let out = -self.inward(s)?;
        Some(self.centre(s)? + (self.up * theta.cos() + out * theta.sin()) * self.radius)
    }

    /// The round's section at `s`, from the cap to the wall.
    fn section(&self, s: f64) -> Option<CircSection> {
        Some(CircSection {
            p1: self.band(s, 0.0)?,
            p2: self.band(s, FRAC_PI_2)?,
            center: self.centre(s)?,
            radius: self.radius,
            uv1: (0.0, 0.0),
            uv2: (0.0, 0.0),
            t: s,
        })
    }

    /// The outline's bend radius at `s`.
    fn bend(&self, s: f64) -> f64 {
        let d = self.curve.derivatives(self.at(s), 2);
        let speed = d[1].length();
        speed * speed * speed / d[1].cross(d[2]).length().max(1e-300)
    }

    /// The same outline, the round's inset `depth` in place of its radius.
    fn inset_by(&self, depth: f64) -> Self {
        Self {
            radius: depth,
            ..self.clone()
        }
    }
}

/// Where between `a` and `b` (forward) the outline bends tightest, and its
/// radius there.
fn tightest(outline: &Outline, a: f64, b: f64) -> (f64, f64) {
    let span = (b - a).rem_euclid(1.0);
    let at = |x: f64| (a + span * x).rem_euclid(1.0);
    let n = 400;
    let mut best = (0.5, f64::MAX);
    for k in 0..=n {
        let x = f64::from(k) / f64::from(n);
        let r = outline.bend(at(x));
        if r < best.1 {
            best = (x, r);
        }
    }
    let (mut lo, mut hi) = (
        (best.0 - 1.0 / f64::from(n)).max(0.0),
        (best.0 + 1.0 / f64::from(n)).min(1.0),
    );
    for _ in 0..100 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if outline.bend(at(m1)) < outline.bend(at(m2)) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let x = 0.5 * (lo + hi);
    (at(x), outline.bend(at(x)))
}

/// The arc of the circle about `centre` from `from` to `to`, under a
/// quarter turn.
fn arc(centre: Point3, from: Point3, to: Point3) -> Result<NurbsCurve, BlendError> {
    let (a, b) = (from - centre, to - centre);
    let half = 0.5 * a.normalize()?.dot(b.normalize()?).clamp(-1.0, 1.0).acos();
    let w = half.cos();
    let mid = centre + (a + b) * (0.5 / (w * w));
    Ok(NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![from, mid, to],
        vec![1.0, w, 1.0],
    )?)
}

/// `first` followed by `second`, which starts where `first` ends.
fn join(first: &NurbsCurve, second: &NurbsCurve) -> Result<NurbsCurve, BlendError> {
    let p = first.degree();
    if second.degree() != p {
        return Err(tip_failure("pieces of one outline of two degrees"));
    }
    let k1 = first.knots();
    let k2 = second.knots();
    let shift = k1[k1.len() - 1] - k2[0];
    let mut knots = k1[..k1.len() - 1].to_vec();
    knots.extend(k2[p + 1..].iter().map(|k| k + shift));
    let w1 = first.weights();
    let w2 = second.weights();
    let scale = w1[w1.len() - 1] / w2[0];
    let mut points = first.control_points().to_vec();
    points.extend_from_slice(&second.control_points()[1..]);
    let mut weights = w1.to_vec();
    weights.extend(w2[1..].iter().map(|w| w * scale));
    Ok(NurbsCurve::new(p, knots, points, weights)?)
}

/// The outline from fraction `a` forward to `b` (past 1 when it runs across
/// the outline's start), moved by `by`.
fn piece_moved(curve: &NurbsCurve, a: f64, b: f64, by: Vec3) -> Result<NurbsCurve, BlendError> {
    let (u0, u1) = curve.domain();
    let at = |s: f64| (u1 - u0).madd(s, u0);
    let edge = 1e-12;
    // The outline from `from` to `to`, both in [0, 1].
    let part = |from: f64, to: f64| -> Result<NurbsCurve, BlendError> {
        let tail = if from <= edge {
            curve.clone()
        } else {
            curve_split(curve, at(from))?.1
        };
        Ok(if to >= 1.0 - edge {
            tail
        } else {
            curve_split(&tail, at(to))?.0
        })
    };
    let piece = if b <= 1.0 + edge {
        part(a, b.min(1.0))?
    } else if a >= 1.0 - edge {
        part(a - 1.0, b - 1.0)?
    } else {
        join(&part(a, 1.0)?, &part(0.0, b - 1.0)?)?
    };
    Ok(NurbsCurve::new(
        piece.degree(),
        piece.knots().to_vec(),
        piece.control_points().iter().map(|&p| p + by).collect(),
        piece.weights().to_vec(),
    )?)
}

/// The vertex of `solid` at `point`.
fn vertex_at(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    tol: f64,
) -> Result<VertexId, BlendError> {
    let mut found = None;
    for e in solid_edges(topo, solid)? {
        let edge = topo.edge(e)?;
        for v in [edge.start(), edge.end()] {
            if (topo.vertex(v)?.point() - point).length() <= tol {
                if found.is_some_and(|f| f != v) {
                    return Err(tip_failure("two vertices where a tip's corner lies"));
                }
                found = Some(v);
            }
        }
    }
    found.ok_or_else(|| tip_failure("no vertex where a tip's corner lies"))
}

fn tip_failure(why: &str) -> BlendError {
    BlendError::PlanningFailure {
        reason: format!("rounding a tight tip: {why}"),
    }
}

/// An edge's two ends, as a set.
fn ends(topo: &Topology, edge: EdgeId) -> Result<[VertexId; 2], BlendError> {
    let data = topo.edge(edge)?;
    Ok([data.start(), data.end()])
}

fn joins(topo: &Topology, edge: EdgeId, x: VertexId, y: VertexId) -> Result<bool, BlendError> {
    let [s, e] = ends(topo, edge)?;
    Ok((s == x && e == y) || (s == y && e == x))
}

/// The run of `face`'s outer wire from the edge joining `x0`, `x1` to the
/// edge joining `y0`, `y1`, the shorter way, as a range of positions.
fn run_between(
    topo: &Topology,
    wire: &[OrientedEdge],
    first: (VertexId, VertexId),
    last: (VertexId, VertexId),
) -> Result<(usize, usize), BlendError> {
    let n = wire.len();
    let mut i = None;
    let mut j = None;
    for (k, oe) in wire.iter().enumerate() {
        if joins(topo, oe.edge(), first.0, first.1)? {
            i = Some(k);
        }
        if joins(topo, oe.edge(), last.0, last.1)? {
            j = Some(k);
        }
    }
    let (Some(i), Some(j)) = (i, j) else {
        return Err(tip_failure("a face round a tip lacks its edges"));
    };
    // The run forward from i to j, or from j to i: the shorter.
    let forward = (j + n - i) % n;
    let backward = (i + n - j) % n;
    Ok(if forward <= backward {
        (i, forward + 1)
    } else {
        (j, backward + 1)
    })
}

/// `wire` with the `len` edges from position `start` (cyclic) replaced by
/// `with`.
fn replace_run(
    wire: &[OrientedEdge],
    start: usize,
    len: usize,
    with: &[OrientedEdge],
) -> Vec<OrientedEdge> {
    let n = wire.len();
    let mut out = with.to_vec();
    for k in len..n {
        out.push(wire[(start + k) % n]);
    }
    out
}

/// The oriented edges `edges` (each joining consecutive points of `path`)
/// running along `path`.
fn along(
    topo: &Topology,
    edges: &[EdgeId],
    path: &[VertexId],
) -> Result<Vec<OrientedEdge>, BlendError> {
    edges
        .iter()
        .zip(path)
        .map(|(&e, &from)| Ok(OrientedEdge::new(e, topo.edge(e)?.start() == from)))
        .collect()
}

/// Rounds the tips left sharp: the round carries on round each, as one face
/// from each foot to where the outline bends tightest, and the two cut each
/// other off along the crease where the round folds back over itself.
#[allow(clippy::redundant_pub_crate, clippy::too_many_lines)]
pub(crate) fn round_tips(
    topo: &mut Topology,
    solid: SolidId,
    tips: &[Tip],
) -> Result<(), BlendError> {
    for tip in tips {
        let o = &tip.outline;
        let r = o.radius;
        let scale = (o.point(tip.a) - Point3::new(0.0, 0.0, 0.0))
            .length()
            .max(1.0);
        let tol = 1e-6 * scale;
        let (q, bend) = tightest(o, tip.a, tip.b);
        if bend >= r {
            return Err(tip_failure("the outline does not bend past the radius"));
        }
        let theta_q = (1.0 - bend / r).asin();
        // The crease, from the cap (theta 0) down to where the fold ends.
        let steps = 24;
        let (mut s1, mut s2) = (tip.a, tip.b);
        let mut crease = vec![(tip.a, tip.b, 0.0)];
        for k in 1..steps {
            let x = f64::from(k) / f64::from(steps);
            let theta = theta_q * (1.0 - (1.0 - x) * (1.0 - x));
            let depth = r * (1.0 - theta.sin());
            match refine(&o.inset_by(depth), s1, s2) {
                Some((n1, n2))
                    if on_tip(n1, (tip.a, q + 1e-12)) && on_tip(n2, (q - 1e-12, tip.b)) =>
                {
                    (s1, s2) = (n1, n2);
                    crease.push((s1, s2, theta));
                }
                _ => break,
            }
        }
        let point_q = o
            .band(q, theta_q)
            .ok_or_else(|| tip_failure("the fold's end"))?;
        let mut crease_points: Vec<Point3> = crease
            .iter()
            .map(|&(s1, _, theta)| o.band(s1, theta).ok_or_else(|| tip_failure("the crease")))
            .collect::<Result<_, _>>()?;
        crease_points.push(point_q);
        let crease_curve = interpolate(&crease_points, 3)?;

        // The corners.
        let p = o
            .band(tip.a, 0.0)
            .ok_or_else(|| tip_failure("the cap's corner"))?;
        let v_p = vertex_at(topo, solid, p, tol)?;
        let v_at = vertex_at(topo, solid, o.point(tip.a), tol)?;
        let v_bt = vertex_at(topo, solid, o.point(tip.b), tol)?;
        let v_a = vertex_at(topo, solid, o.point(tip.a) - o.up * r, tol)?;
        let v_b = vertex_at(topo, solid, o.point(tip.b) - o.up * r, tol)?;

        // The faces round the tip.
        let shell_id = topo.solid(solid)?.outer_shell();
        let faces = topo.shell(shell_id)?.faces().to_vec();
        let mut tri = [None, None];
        let mut cap = None;
        let mut wall = None;
        for &f in &faces {
            let wire = topo.wire(topo.face(f)?.outer_wire())?.edges().to_vec();
            let mut vs = HashSet::new();
            for oe in &wire {
                vs.extend(ends(topo, oe.edge())?);
            }
            if wire.len() == 3 && vs.contains(&v_p) {
                if vs.contains(&v_a) && vs.contains(&v_at) {
                    tri[0] = Some(f);
                    continue;
                }
                if vs.contains(&v_b) && vs.contains(&v_bt) {
                    tri[1] = Some(f);
                    continue;
                }
            }
            if vs.contains(&v_at) && vs.contains(&v_bt) {
                if vs.contains(&v_p) {
                    cap = Some(f);
                } else if vs.contains(&v_a) && vs.contains(&v_b) {
                    wall = Some(f);
                }
            }
        }
        let (Some(tri_a), Some(tri_b), Some(cap), Some(wall)) = (tri[0], tri[1], cap, wall) else {
            return Err(tip_failure("the faces round a sharp tip are not as left"));
        };
        let arc_of = |topo: &Topology, f: FaceId, v: VertexId| -> Result<EdgeId, BlendError> {
            for oe in topo.wire(topo.face(f)?.outer_wire())?.edges() {
                if joins(topo, oe.edge(), v_p, v)? {
                    return Ok(oe.edge());
                }
            }
            Err(tip_failure("a tip's end section"))
        };
        let arc_a = arc_of(topo, tri_a, v_a)?;
        let arc_b = arc_of(topo, tri_b, v_b)?;
        // How the rounds beyond use the end sections.
        let user =
            |topo: &Topology, e: EdgeId, not: FaceId| -> Result<(FaceId, bool), BlendError> {
                for &f in &faces {
                    if f == not {
                        continue;
                    }
                    let face = topo.face(f)?;
                    // A reversed face's wire runs against its outward side.
                    for oe in topo.wire(face.outer_wire())?.edges() {
                        if oe.edge() == e {
                            return Ok((f, oe.is_forward() != face.is_reversed()));
                        }
                    }
                }
                Err(tip_failure("a tip's end section bounds one face"))
            };
        let (band_a, sense_a) = user(topo, arc_a, tri_a)?;
        let (_, sense_b) = user(topo, arc_b, tri_b)?;
        // The new edges.
        let ruler = |s: f64| o.point(s) - o.up * r;
        let v_qw = topo.add_vertex(Vertex::new(ruler(q), ON));
        let v_q = topo.add_vertex(Vertex::new(point_q, ON));
        // The fractions unwrapped to run forward from the first foot.
        let b_on = tip.a + (tip.b - tip.a).rem_euclid(1.0);
        let q_on = tip.a + (q - tip.a).rem_euclid(1.0);
        let rim_a = piece_moved(&o.curve, tip.a, q_on, -o.up * r)?;
        let rim_b = piece_moved(&o.curve, q_on, b_on, -o.up * r)?;
        let wall_a = topo.add_edge(Edge::new(v_a, v_qw, EdgeCurve::NurbsCurve(rim_a)));
        let wall_b = topo.add_edge(Edge::new(v_qw, v_b, EdgeCurve::NurbsCurve(rim_b)));
        let centre_q = o.centre(q).ok_or_else(|| tip_failure("the ball"))?;
        let middle = topo.add_edge(Edge::new(
            v_qw,
            v_q,
            EdgeCurve::NurbsCurve(arc(centre_q, ruler(q), point_q)?),
        ));
        let fold = topo.add_edge(Edge::new(v_p, v_q, EdgeCurve::NurbsCurve(crease_curve)));

        // The two faces, each a round through its sections.
        let surface = |from: f64, to: f64| -> Result<NurbsSurface, BlendError> {
            let span = (to - from).rem_euclid(1.0);
            let sections: Vec<CircSection> = (0..=48)
                .map(|k| o.section(from + span * f64::from(k) / 48.0))
                .collect::<Option<_>>()
                .ok_or_else(|| tip_failure("a section"))?;
            interpolate_blend_surface(&sections)
        };
        let first = along(topo, &[arc_a, wall_a, middle, fold], &[v_p, v_a, v_qw, v_q])?;
        let second = along(topo, &[fold, middle, wall_b, arc_b], &[v_p, v_q, v_qw, v_b])?;
        // Each loop runs round its face's outward side.
        let flip = |wire: &[OrientedEdge]| -> Vec<OrientedEdge> {
            wire.iter()
                .rev()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect()
        };
        let mut first = first;
        let mut second = second;
        if first[0].is_forward() == sense_a {
            first = flip(&first);
        }
        if second[3].is_forward() == sense_b {
            second = flip(&second);
        }
        let fold_first = first
            .iter()
            .find(|oe| oe.edge() == fold)
            .map(OrientedEdge::is_forward);
        let fold_second = second
            .iter()
            .find(|oe| oe.edge() == fold)
            .map(OrientedEdge::is_forward);
        if fold_first == fold_second {
            return Err(tip_failure("the crease's two faces run it the same way"));
        }
        // Each face's side, from the round beyond its foot.
        let band_face = topo.face(band_a)?;
        let near = o
            .band(tip.a, FRAC_PI_4)
            .ok_or_else(|| tip_failure("a section"))?;
        let outward = |surface: &FaceSurface, reversed: bool| -> Option<Vec3> {
            let (u, v) = surface.project_point(near)?;
            let n = surface.normal(u, v);
            Some(if reversed { -n } else { n })
        };
        let beyond = outward(band_face.surface(), band_face.is_reversed())
            .ok_or_else(|| tip_failure("the round beyond"))?;
        let mut made = Vec::new();
        for (from, to, wire) in [(tip.a, q, first), (q, tip.b, second)] {
            let surf = FaceSurface::Nurbs(surface(from, to)?);
            let own = outward(&surf, false).ok_or_else(|| tip_failure("a tip's round"))?;
            let reversed = own.dot(beyond) < 0.0;
            // A reversed face keeps its wire round its surface's own side.
            let wire = topo.add_wire(Wire::new(if reversed { flip(&wire) } else { wire }, true)?);
            let face = if reversed {
                Face::new_reversed(wire, Vec::new(), surf)
            } else {
                Face::new(wire, Vec::new(), surf)
            };
            made.push(topo.add_face(face));
        }

        // The wall runs on along the round's contact, and the cap passes the
        // tip by its corner alone.
        let wall_wire = topo.wire(topo.face(wall)?.outer_wire())?.edges().to_vec();
        let (start, len) = run_between(topo, &wall_wire, (v_a, v_at), (v_bt, v_b))?;
        let entry = ends(topo, wall_wire[start].edge())?;
        let from_a = entry.contains(&v_a) && !entry.contains(&v_b);
        let with = if from_a {
            along(topo, &[wall_a, wall_b], &[v_a, v_qw])?
        } else {
            along(topo, &[wall_b, wall_a], &[v_b, v_qw])?
        };
        let new_wall = replace_run(&wall_wire, start, len, &with);
        let new_wall = topo.add_wire(Wire::new(new_wall, true)?);
        topo.face_mut(wall)?.set_outer_wire(new_wall);

        let cap_wire = topo.wire(topo.face(cap)?.outer_wire())?.edges().to_vec();
        let (start, len) = run_between(topo, &cap_wire, (v_p, v_at), (v_bt, v_p))?;
        let new_cap = replace_run(&cap_wire, start, len, &[]);
        let new_cap = topo.add_wire(Wire::new(new_cap, true)?);
        topo.face_mut(cap)?.set_outer_wire(new_cap);

        let mut kept: Vec<FaceId> = faces
            .into_iter()
            .filter(|&f| f != tri_a && f != tri_b)
            .collect();
        kept.extend(made);
        *topo.shell_mut(shell_id)? = Shell::new(kept)?;
    }
    Ok(())
}
