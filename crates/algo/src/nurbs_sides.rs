//! Whether the sides of a NURBS surface's domain keep apart from each other:
//! the surface then neither closes on itself, along a side or across a
//! corner, nor shrinks to a point along a side, and each point of it near a
//! side has one place in its parameters.

use brepkit_math::aabb::Aabb3;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::Face;

use crate::error::AlgoError;

/// How many times a piece of a side is halved at most.
const MOST_DEPTH: u32 = 24;

/// How many pairs of pieces one pair of sides may take up, as found and as
/// halved.
const MOST_PAIRS: usize = 1 << 14;

/// The angle, in radians, by which the rounding of a direction's
/// coordinates can move the angle between two directions read by `acos`.
const ANGLE_SLACK: f64 = 1e-7;

/// A Bezier piece by its control points `[w x, w y, w z, w]`.
type Net = Vec<[f64; 4]>;

/// How far a face's own edges and vertices may stand off its surface: ten
/// tolerances, or more where an edge or a vertex of the face carries a
/// wider tolerance of its own.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn face_reach(topo: &Topology, face: &Face, tol: Tolerance) -> Result<f64, AlgoError> {
    let mut widest = 10.0 * tol.linear;
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            if let Some(t) = edge.tolerance() {
                widest = widest.max(t.min(crate::ds::shape_store::MAX_WIDEN));
            }
            for v in [edge.start(), edge.end()] {
                widest = widest.max(topo.vertex(v)?.tolerance());
            }
        }
    }
    Ok(widest)
}

/// Whether each side of `surface`'s domain keeps farther than `reach` from
/// every other side, but at the corner two sides share, and no side shrinks
/// to a point.
///
/// A side is the curve whose control points blend the surface's rows (or
/// columns) by their basis functions at the domain's end, which holds
/// whether or not the knots are clamped. Its Bezier pieces lie in the boxes
/// of their control points, as each weight is positive. Two pieces whose
/// boxes come within `reach` and the rounding are halved, the larger first,
/// until their boxes keep apart. Two pieces that start at the corner their
/// sides share each lie in the cone from the corner through their other
/// control points, the more so as they are halved: where those cones meet
/// only at the corner, the pieces do too, and come within a distance of each
/// other only near it. A side whose control points all lie within `reach`
/// of each other is a point.
///
/// The control points are rounded blends of the surface's, whose sizes the
/// weights scale by their ratio, and each halving blends them again: each
/// lies within [`MOST_DEPTH`] such roundings of its exact place. Pieces
/// whose boxes need more halvings to part, two sides that take up more than
/// [`MOST_PAIRS`] pairs of pieces, two pieces away from a shared corner
/// whose ends come within `reach` and the rounding, or a weight that is not
/// a positive normal float, read as not apart.
pub fn sides_are_apart(surface: &NurbsSurface, reach: f64) -> bool {
    let weights = surface.weights();
    if !weights.iter().flatten().all(|&w| w > 0.0 && w.is_normal()) {
        return false;
    }
    let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
    let (Some(u_lo), Some(u_hi), Some(v_lo), Some(v_hi)) = (
        side(surface, true, u0),
        side(surface, true, u1),
        side(surface, false, v0),
        side(surface, false, v1),
    ) else {
        return false;
    };
    let (least, most) = weights
        .iter()
        .flatten()
        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &w| {
            (lo.min(w), hi.max(w))
        });
    let size = surface
        .control_points()
        .iter()
        .flatten()
        .fold(0.0_f64, |m, p| {
            m.max(p.x().abs()).max(p.y().abs()).max(p.z().abs())
        })
        * (most / least);
    #[allow(clippy::cast_precision_loss)]
    let roundings =
        (8 * (surface.degree_u() + surface.degree_v() + 4) * (MOST_DEPTH as usize + 2)) as f64;
    let slack = roundings * f64::EPSILON * size.max(f64::MIN_POSITIVE);
    let margin = reach + slack;
    if !margin.is_finite() {
        return false;
    }
    for s in [&u_lo, &u_hi, &v_lo, &v_hi] {
        if whole_box(s).is_none_or(|b| (b.max - b.min).length() <= margin) {
            return false;
        }
    }
    // `u_lo` runs along `v` from the corner `(u0, v0)` to `(u0, v1)`, and
    // `v_lo` along `u` from `(u0, v0)` to `(u1, v0)`.
    let from_end = |s: &[Net]| -> Vec<Net> {
        s.iter()
            .rev()
            .map(|n| n.iter().rev().copied().collect())
            .collect()
    };
    apart(&u_lo, &u_hi, false, margin, slack)
        && apart(&v_lo, &v_hi, false, margin, slack)
        && apart(&u_lo, &v_lo, true, margin, slack)
        && apart(&from_end(&u_lo), &v_hi, true, margin, slack)
        && apart(&u_hi, &from_end(&v_lo), true, margin, slack)
        && apart(&from_end(&u_hi), &from_end(&v_hi), true, margin, slack)
}

/// The Bezier pieces of the side of `surface` at `at` along `u`, or along
/// `v`, in order of the other parameter. `None` where a blended weight is
/// not a positive normal float or a piece cannot be read.
fn side(surface: &NurbsSurface, along_u: bool, at: f64) -> Option<Vec<Net>> {
    let net = surface.control_points();
    let weights = surface.weights();
    let (rows, columns) = (net.len(), net.first().map_or(0, Vec::len));
    let (u, v) = (
        (surface.knots_u(), surface.degree_u(), rows),
        (surface.knots_v(), surface.degree_v(), columns),
    );
    let ((knots, p, count), (other, q, across)) = if along_u { (u, v) } else { (v, u) };
    let span = brepkit_math::nurbs::basis::find_span(count, p, at, knots);
    let basis = brepkit_math::nurbs::basis::basis_funs(span, at, p, knots);
    let first = span.checked_sub(p)?;
    let mut points = Vec::with_capacity(across);
    let mut blended = Vec::with_capacity(across);
    for j in 0..across {
        let (mut sum, mut total) = (Vec3::new(0.0, 0.0, 0.0), 0.0);
        for (k, &b) in basis.iter().enumerate() {
            let (i, j) = if along_u {
                (first + k, j)
            } else {
                (j, first + k)
            };
            let w = b * *weights.get(i)?.get(j)?;
            let c = *net.get(i)?.get(j)?;
            sum += Vec3::new(c.x(), c.y(), c.z()) * w;
            total += w;
        }
        if !(total > 0.0 && total.is_normal()) {
            return None;
        }
        points.push(Point3::new(
            sum.x() / total,
            sum.y() / total,
            sum.z() / total,
        ));
        blended.push(total);
    }
    let curve =
        brepkit_math::nurbs::curve::NurbsCurve::new(q, other.to_vec(), points, blended).ok()?;
    let (lo, hi) = curve.domain();
    let mut breaks: Vec<f64> = std::iter::once(lo)
        .chain(other.iter().copied().filter(|&k| lo < k && k < hi))
        .chain(std::iter::once(hi))
        .collect();
    breaks.dedup_by(|a, b| a <= b);
    breaks
        .windows(2)
        .map(|w| brepkit_math::nurbs::decompose::curve_bezier_between(&curve, w[0], w[1]).ok())
        .collect()
}

fn point(q: [f64; 4]) -> Point3 {
    Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3])
}

fn net_box(net: &[[f64; 4]]) -> Aabb3 {
    Aabb3::from_points(net.iter().map(|&q| point(q)))
}

fn whole_box(side: &[Net]) -> Option<Aabb3> {
    side.iter().map(|n| net_box(n)).reduce(Aabb3::union)
}

/// The distance between two boxes: zero where they meet.
fn gap(a: Aabb3, b: Aabb3) -> f64 {
    let along =
        |lo_a: f64, hi_a: f64, lo_b: f64, hi_b: f64| (lo_b - hi_a).max(lo_a - hi_b).max(0.0);
    let x = along(a.min.x(), a.max.x(), b.min.x(), b.max.x());
    let y = along(a.min.y(), a.max.y(), b.min.y(), b.max.y());
    let z = along(a.min.z(), a.max.z(), b.min.z(), b.max.z());
    x.hypot(y).hypot(z)
}

/// A Bezier piece cut in two at the middle of its parameters.
fn halves(net: &[[f64; 4]]) -> (Net, Net) {
    let mut row = net.to_vec();
    let n = row.len();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    left.push(row[0]);
    right.push(row[n - 1]);
    for level in 1..n {
        for j in 0..n - level {
            let (a, b) = (row[j], row[j + 1]);
            row[j] = [
                f64::midpoint(a[0], b[0]),
                f64::midpoint(a[1], b[1]),
                f64::midpoint(a[2], b[2]),
                f64::midpoint(a[3], b[3]),
            ];
        }
        left.push(row[0]);
        right.push(row[n - 1 - level]);
    }
    right.reverse();
    (left, right)
}

/// Whether the sides `a` and `b`, each a run of Bezier pieces, keep farther
/// than `margin` apart, but near the corner they share at their starts where
/// `shared` (see [`sides_are_apart`]).
fn apart(a: &[Net], b: &[Net], shared: bool, margin: f64, slack: f64) -> bool {
    apart_counted(a, b, shared, margin, slack).0
}

/// [`apart`], and how many pairs of pieces it took up. Each pair of pieces
/// whose boxes come within `margin`, as found and as halved, takes one of
/// [`MOST_PAIRS`]; past them the sides read as not apart. A pair borrows
/// the sides' own pieces until it halves them.
fn apart_counted(a: &[Net], b: &[Net], shared: bool, margin: f64, slack: f64) -> (bool, usize) {
    use std::borrow::Cow;
    type Pair<'a> = (Cow<'a, [[f64; 4]]>, Cow<'a, [[f64; 4]]>, bool, u32);
    if let (Some(x), Some(y)) = (whole_box(a), whole_box(b))
        && gap(x, y) > margin
    {
        return (true, 0);
    }
    let boxes_b: Vec<Aabb3> = b.iter().map(|n| net_box(n)).collect();
    let mut work: Vec<Pair<'_>> = Vec::new();
    let mut taken = 0_usize;
    for (i, x) in a.iter().enumerate() {
        let bx = net_box(x);
        for (j, y) in b.iter().enumerate() {
            if gap(bx, boxes_b[j]) <= margin {
                if taken == MOST_PAIRS {
                    return (false, taken);
                }
                taken += 1;
                work.push((
                    Cow::Borrowed(x.as_slice()),
                    Cow::Borrowed(y.as_slice()),
                    shared && i == 0 && j == 0,
                    0,
                ));
            }
        }
    }
    while let Some((x, y, corner, depth)) = work.pop() {
        let (bx, by) = (net_box(&x), net_box(&y));
        if gap(bx, by) > margin || (corner && cones_apart(&x, &y, slack)) {
            continue;
        }
        // The ends of a piece lie on its side: two ends within the margin
        // show the sides meet there, as along a closed surface's seam.
        if !corner && ends_meet(&x, &y, margin) {
            return (false, taken);
        }
        if depth >= MOST_DEPTH {
            return (false, taken);
        }
        let (dx, dy) = ((bx.max - bx.min).length(), (by.max - by.min).length());
        let xs = split(x, corner || dx >= dy);
        let ys = split(y, corner || dy > dx);
        for (i, xx) in xs.iter().enumerate() {
            for (j, yy) in ys.iter().enumerate() {
                if taken == MOST_PAIRS {
                    return (false, taken);
                }
                taken += 1;
                work.push((
                    xx.clone(),
                    yy.clone(),
                    corner && i == 0 && j == 0,
                    depth + 1,
                ));
            }
        }
    }
    (true, taken)
}

/// `piece` halved where `cut`, else whole.
fn split(
    piece: std::borrow::Cow<'_, [[f64; 4]]>,
    cut: bool,
) -> Vec<std::borrow::Cow<'_, [[f64; 4]]>> {
    use std::borrow::Cow;
    if cut {
        let (l, r) = halves(&piece);
        vec![Cow::Owned(l), Cow::Owned(r)]
    } else {
        vec![piece]
    }
}

/// Whether an end of one Bezier piece lies within `margin` of an end of
/// the other.
fn ends_meet(a: &[[f64; 4]], b: &[[f64; 4]], margin: f64) -> bool {
    let ends = |n: &[[f64; 4]]| [n.first().copied(), n.last().copied()];
    ends(a).into_iter().flatten().any(|p| {
        ends(b)
            .into_iter()
            .flatten()
            .any(|q| (point(p) - point(q)).length() <= margin)
    })
}

/// The axis and half-angle of a circular cone from a Bezier piece's first
/// control point that holds the piece: each point of the piece is the first
/// control point plus a blend of the others' offsets from it with weights
/// that are not negative. Each offset's direction is read to an angle of
/// its rounding over its length. `None` where an offset is no longer than a
/// few `slack`, whose direction is then unread: a weight can carry the piece
/// that way as far as the offset reaches, past the rounding (a side run
/// back 1e-3 along its neighbour under a weight of 1000), or where the
/// half-angle reaches a right angle.
fn cone(net: &[[f64; 4]], slack: f64) -> Option<(Vec3, f64)> {
    let apex = point(*net.first()?);
    let reaches: Vec<(Vec3, f64)> = net[1..]
        .iter()
        .map(|&q| {
            let g = point(q) - apex;
            let length = g.length();
            (length > 4.0 * slack).then(|| (g * (1.0 / length), 4.0 * slack / length))
        })
        .collect::<Option<_>>()?;
    let sum = reaches
        .iter()
        .fold(Vec3::new(0.0, 0.0, 0.0), |s, (u, _)| s + *u);
    let length = sum.length();
    if reaches.is_empty() || length.is_nan() || length <= 0.0 {
        return None;
    }
    let axis = sum * (1.0 / length);
    let half = reaches
        .iter()
        .map(|(u, err)| u.dot(axis).clamp(-1.0, 1.0).acos() + err)
        .fold(0.0_f64, f64::max);
    (half < std::f64::consts::FRAC_PI_2).then_some((axis, half))
}

/// Whether two Bezier pieces that start at one corner leave it in cones that
/// meet only there ([`cone`]).
fn cones_apart(a: &[[f64; 4]], b: &[[f64; 4]], slack: f64) -> bool {
    let (Some((x, hx)), Some((y, hy))) = (cone(a, slack), cone(b, slack)) else {
        return false;
    };
    x.dot(y).clamp(-1.0, 1.0).acos() > hx + hy + ANGLE_SLACK
}

#[cfg(test)]
pub mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::madd::Madd;

    /// A biquadratic surface on the knots `[0, 0, 0, 0.5, 1, 1, 1]` both
    /// ways. Its control points of the first three rows and columns stand
    /// at `(i + j, i j, 0)` where `folded`, the same for `(i, j)` and
    /// `(j, i)`, so the surface over the first span both ways is its own
    /// mirror across the diagonal: its sides at `u = 0` and `v = 0` are one
    /// curve there, a fold across the corner. Else they stand at `(i, j,
    /// 0)`. The last row and column run off apart.
    pub fn corner_sheet(folded: bool) -> NurbsSurface {
        let rows: Vec<Vec<Point3>> = (0..4_u32)
            .map(|i| {
                (0..4_u32)
                    .map(|j| {
                        let (x, y) = (f64::from(i), f64::from(j));
                        match (i, j) {
                            (3, 3) => Point3::new(6.0, 6.0, 0.0),
                            (3, _) => Point3::new(3.0 + y, 2.0 + y, 1.0),
                            (_, 3) => Point3::new(x - 1.0, 3.0 + x, -1.0),
                            _ if folded => Point3::new(x + y, x * y, 0.0),
                            _ => Point3::new(x, y, 0.0),
                        }
                    })
                    .collect()
            })
            .collect();
        let knots = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        NurbsSurface::new(2, 2, knots.clone(), knots, rows, vec![vec![1.0; 4]; 4]).unwrap()
    }

    /// A rational quadratic circle of radius 2 round `z`, from `z = 0` to
    /// `z = 3`, moved by `offset`: its last row of control points stands
    /// `gap` short of its first along `y`, so it closes on itself to that.
    pub fn tube(gap: f64, offset: Vec3) -> NurbsSurface {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let round = [
            (1.0, 0.0, 1.0),
            (1.0, 1.0, w),
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, w),
            (-1.0, 0.0, 1.0),
            (-1.0, -1.0, w),
            (0.0, -1.0, 1.0),
            (1.0, -1.0, w),
            (1.0, 0.0, 1.0),
        ];
        let rows = round
            .iter()
            .enumerate()
            .map(|(i, &(x, y, _))| {
                let off = if i == round.len() - 1 { gap } else { 0.0 };
                [0.0, 3.0]
                    .iter()
                    .map(|&z| Point3::new(2.0 * x, 2.0f64.madd(y, -off), z) + offset)
                    .collect()
            })
            .collect();
        NurbsSurface::new(
            2,
            1,
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            vec![0.0, 0.0, 1.0, 1.0],
            rows,
            round.iter().map(|&(_, _, w)| vec![w; 2]).collect(),
        )
        .unwrap()
    }

    /// A rational ball of radius 1 about `centre`: rows round `z`, and
    /// columns along a half circle from the south pole to the north, so
    /// its sides at `v = 0` and `v = 1` are its poles.
    pub fn ball(centre: Point3) -> NurbsSurface {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let round = [
            (1.0, 0.0, 1.0),
            (1.0, 1.0, w),
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, w),
            (-1.0, 0.0, 1.0),
            (-1.0, -1.0, w),
            (0.0, -1.0, 1.0),
            (1.0, -1.0, w),
            (1.0, 0.0, 1.0),
        ];
        let half = [
            (0.0, -1.0, 1.0),
            (1.0, -1.0, w),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, w),
            (0.0, 1.0, 1.0),
        ];
        let rows = round
            .iter()
            .map(|&(x, y, _)| {
                half.iter()
                    .map(|&(r, z, _)| centre + Vec3::new(r * x, r * y, z))
                    .collect()
            })
            .collect();
        let weights = round
            .iter()
            .map(|&(_, _, a)| half.iter().map(|&(_, _, b)| a * b).collect())
            .collect();
        NurbsSurface::new(
            2,
            2,
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            rows,
            weights,
        )
        .unwrap()
    }

    /// The least distance, read densely, between two sides of `surface`
    /// away from the corner they share, and the least spread of a side.
    fn dense_sides(surface: &NurbsSurface) -> (f64, f64) {
        const N: u32 = 400;
        let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
        let lerp = |a: f64, b: f64, k: u32| (b - a).madd(f64::from(k) / f64::from(N), a);
        // Each side from the corner `(u0, v0)` or `(u1, v1)` it runs from.
        let sides: [Vec<Point3>; 4] = [
            (0..=N)
                .map(|k| surface.evaluate(u0, lerp(v0, v1, k)))
                .collect(),
            (0..=N)
                .map(|k| surface.evaluate(lerp(u0, u1, k), v0))
                .collect(),
            (0..=N)
                .map(|k| surface.evaluate(u1, lerp(v1, v0, k)))
                .collect(),
            (0..=N)
                .map(|k| surface.evaluate(lerp(u1, u0, k), v1))
                .collect(),
        ];
        let spread = sides
            .iter()
            .map(|s| {
                let b = Aabb3::from_points(s.iter().copied());
                (b.max - b.min).length()
            })
            .fold(f64::INFINITY, f64::min);
        // Sides 0 and 1 start at one corner, as 2 and 3 do; 0 and 3 end at
        // one corner, as 1 and 2 do; 0 and 2, and 1 and 3, share none.
        let shared = |a: usize, b: usize, i: usize, j: usize| {
            let last = N as usize;
            match (a, b) {
                (0, 1) | (2, 3) => i < 20 && j < 20,
                (0, 3) | (1, 2) => i + 20 > last && j + 20 > last,
                _ => false,
            }
        };
        let mut least = f64::INFINITY;
        for a in 0..4 {
            for b in a + 1..4 {
                for (i, p) in sides[a].iter().enumerate() {
                    for (j, q) in sides[b].iter().enumerate() {
                        if !shared(a, b, i, j) {
                            least = least.min((*p - *q).length());
                        }
                    }
                }
            }
        }
        (least, spread)
    }

    /// Sides that dense evaluation finds within the reach of each other, or
    /// a side that it finds to be a point, read as not apart: a fold across
    /// a corner, tubes closed exactly, a million from the origin too, and to
    /// a hundred millionth of their size, or past their start, and a ball
    /// with its poles for sides.
    /// An open sheet, the corner sheet unfolded and a tube open by a
    /// thousandth read as apart, as dense evaluation finds them.
    #[test]
    fn sides_read_apart_only_where_they_keep_apart() {
        let reach = 1e-6;
        let far = Vec3::new(1e6, -2e6, 5e5);
        let zero = Vec3::new(0.0, 0.0, 0.0);
        let cases = [
            ("fold", corner_sheet(true), false),
            ("unfolded", corner_sheet(false), true),
            ("closed tube", tube(0.0, zero), false),
            ("closed tube far off", tube(0.0, far), false),
            ("tube closed to 1e-8", tube(2e-8, zero), false),
            ("tube open by 1e-3", tube(2e-3, zero), true),
            ("tube open by 1e-3 far off", tube(2e-3, far), true),
            ("ball", ball(Point3::new(0.0, 0.0, 0.0)), false),
        ];
        for (name, surface, want) in cases {
            let (least, spread) = dense_sides(&surface);
            let dense = least > reach && spread > reach;
            assert_eq!(dense, want, "{name}: dense reads {least:e}, {spread:e}");
            assert_eq!(sides_are_apart(&surface, reach), want, "{name}");
        }

        // A tube run past its start: its rim at `v = 0` crosses `y = 0` again
        // between two of its last dense samples, through the foot of its side
        // at `u = 0`, which the samples miss by 2e-3.
        let past = tube(-2e-3, zero);
        let (least, _) = dense_sides(&past);
        let y = |u: f64| past.evaluate(u, 0.0).y();
        let (mut lo, mut hi) = (1.0 - 1e-3, 1.0);
        assert!(y(lo) < 0.0 && y(hi) > 0.0);
        for _ in 0..60 {
            let mid = f64::midpoint(lo, hi);
            if y(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let foot = past.evaluate(0.0, 0.0);
        let through = (past.evaluate(lo, 0.0) - foot).length();
        assert!(least > 1e-3 && through < 1e-6, "{least:e}, {through:e}");
        assert!(!sides_are_apart(&past, reach));
    }

    /// A strip of `spans` knot spans along `v`, quadratic along `u`, whose
    /// sides at `u = 0` and `u = 1` stand 1e-7 apart, each running back and
    /// forth along one segment of length 1e-3, so every piece of one side
    /// comes within the reach of every piece of the other. Its middle row
    /// stands 1 off them.
    pub fn zigzag_strip(spans: usize) -> NurbsSurface {
        let side = |x: f64| -> Vec<Point3> {
            (0..=spans)
                .map(|k| Point3::new(x, if k % 2 == 0 { 0.0 } else { 1e-3 }, 0.0))
                .collect()
        };
        #[allow(clippy::cast_precision_loss)]
        let knots: Vec<f64> = std::iter::once(0.0)
            .chain((0..=spans).map(|k| k as f64 / spans as f64))
            .chain(std::iter::once(1.0))
            .collect();
        NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            knots,
            vec![side(0.0), side(1.0), side(1e-7)],
            vec![vec![1.0; spans + 1]; 3],
        )
        .unwrap()
    }

    /// Two sides of a thousand pieces each, every piece of one within the
    /// reach of every piece of the other, take up no more than
    /// [`MOST_PAIRS`] pairs of pieces before they read as not apart.
    #[test]
    fn sides_of_many_crowded_pieces_take_up_a_bounded_number_of_pairs() {
        let strip = zigzag_strip(1000);
        let ((u0, u1), _) = (strip.domain_u(), strip.domain_v());
        let (near, far) = (
            side(&strip, true, u0).unwrap(),
            side(&strip, true, u1).unwrap(),
        );
        assert_eq!(near.len(), 1000);
        let (read, taken) = apart_counted(&near, &far, false, 1e-6, 1e-13);
        assert!(!read && taken <= MOST_PAIRS, "{read} after {taken} pairs");
        assert!(!sides_are_apart(&strip, 1e-6));
    }

    /// A patch of degrees 2 and 1 a million from the origin whose side at
    /// `v = 0` starts back along its side at `u = 0`: its second control
    /// point stands 1e-3 behind the corner with a weight of 1000. Its
    /// control points' sizes times their weights' ratio give the rounding a
    /// reach of 3.2e-4, past which that side runs on along the other: the
    /// two sides meet 8.3e-4 from the corner, and do not keep apart.
    #[test]
    fn a_side_run_back_along_its_neighbour_from_their_corner_is_not_apart() {
        let o = Point3::new(1e6, 1e6, 1e6);
        let at = |x: f64, y: f64, z: f64| o + Vec3::new(x, y, z);
        let patch = NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![at(0.0, 0.0, 0.0), at(-1.0, 0.0, 0.0)],
                vec![at(-1e-3, 0.0, 0.0), at(-1.0, 1.0, 1.0)],
                vec![at(1.0, 0.0, 0.0), at(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1000.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();
        // The side at `v = 0` runs back to its least `x`, on the side at
        // `u = 0`, the segment from the corner back to `x = -1`.
        let back = (0..=20_000_u32)
            .map(|k| patch.evaluate(f64::from(k) / 20_000.0, 0.0))
            .map(|p| (p - o).x())
            .fold(0.0_f64, f64::min);
        assert!(back < -8e-4, "{back:e}");
        assert!(!sides_are_apart(&patch, 1e-6));
    }
}
