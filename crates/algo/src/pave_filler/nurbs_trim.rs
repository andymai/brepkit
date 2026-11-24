//! A NURBS face's pcurves read in its surface's parameters: on which side of
//! the face's wires a point of the surface lies.

use brepkit_math::curves2d::Curve2D;
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

use crate::error::AlgoError;

/// How many times a piece of a wire is halved before a point in its box
/// counts as on it: past this the pieces are no longer than the rounding of
/// their coordinates.
const MOST_HALVINGS: u32 = 56;

/// How many pieces one reading of a point may halve.
const MOST_PIECES: usize = 1 << 12;

/// The longest arc of a circle or an ellipse that one piece takes, in
/// radians.
const ARC_STEP: f64 = std::f64::consts::FRAC_PI_4;

/// A box in a surface's parameters.
#[derive(Clone, Copy)]
struct Box2 {
    lo: [f64; 2],
    hi: [f64; 2],
}

impl Box2 {
    fn of(points: impl IntoIterator<Item = Point2>) -> Self {
        points.into_iter().fold(
            Self {
                lo: [f64::INFINITY; 2],
                hi: [f64::NEG_INFINITY; 2],
            },
            |b, p| Self {
                lo: [b.lo[0].min(p.x()), b.lo[1].min(p.y())],
                hi: [b.hi[0].max(p.x()), b.hi[1].max(p.y())],
            },
        )
    }

    fn grown(self, by: f64) -> Self {
        Self {
            lo: [self.lo[0] - by, self.lo[1] - by],
            hi: [self.hi[0] + by, self.hi[1] + by],
        }
    }
}

/// What holds a piece of a wire between its two ends.
enum Hull {
    /// The straight segment between the ends.
    Segment,
    /// An arc of a circle or an ellipse over the parameters `t`, whose
    /// second derivative is never longer than `bend`: the arc lies within
    /// `bend Δt² / 8` of the chord between its ends.
    Conic {
        curve: Curve2D,
        t: (f64, f64),
        bend: f64,
    },
    /// A Bezier piece of a B-spline by its control points `[w x, w y, w]`,
    /// each weight positive: the piece lies in the box of the points.
    Bezier(Vec<[f64; 3]>),
}

/// A piece of a wire from `a` to `b`. Two pieces that meet share the one
/// computed point there, so the pieces of a wire run round a closed curve.
struct Piece {
    a: Point2,
    b: Point2,
    hull: Hull,
}

impl Piece {
    /// A box that holds the piece and its ends, grown by `slack`.
    fn bounds(&self, slack: f64) -> Box2 {
        let ends = Box2::of([self.a, self.b]);
        let held = match &self.hull {
            Hull::Segment => ends,
            Hull::Conic { t, bend, .. } => {
                let w = t.1 - t.0;
                ends.grown(bend * w * w / 8.0)
            }
            Hull::Bezier(net) => Box2::of(
                net.iter()
                    .map(|q| Point2::new(q[0] / q[2], q[1] / q[2]))
                    .chain([self.a, self.b]),
            ),
        };
        held.grown(slack)
    }

    /// The piece cut in two at the middle of its parameters.
    fn halves(&self) -> (Self, Self) {
        match &self.hull {
            Hull::Segment => {
                let m = Point2::new(
                    f64::midpoint(self.a.x(), self.b.x()),
                    f64::midpoint(self.a.y(), self.b.y()),
                );
                (
                    Self {
                        a: self.a,
                        b: m,
                        hull: Hull::Segment,
                    },
                    Self {
                        a: m,
                        b: self.b,
                        hull: Hull::Segment,
                    },
                )
            }
            Hull::Conic { curve, t, bend } => {
                let tm = f64::midpoint(t.0, t.1);
                let m = curve.evaluate(tm);
                (
                    Self {
                        a: self.a,
                        b: m,
                        hull: Hull::Conic {
                            curve: curve.clone(),
                            t: (t.0, tm),
                            bend: *bend,
                        },
                    },
                    Self {
                        a: m,
                        b: self.b,
                        hull: Hull::Conic {
                            curve: curve.clone(),
                            t: (tm, t.1),
                            bend: *bend,
                        },
                    },
                )
            }
            Hull::Bezier(net) => {
                let mut row = net.clone();
                let mut left = vec![row[0]];
                let mut right = vec![row[row.len() - 1]];
                for level in 1..row.len() {
                    for j in 0..row.len() - level {
                        let (p, q) = (row[j], row[j + 1]);
                        row[j] = [
                            f64::midpoint(p[0], q[0]),
                            f64::midpoint(p[1], q[1]),
                            f64::midpoint(p[2], q[2]),
                        ];
                    }
                    left.push(row[0]);
                    right.push(row[row.len() - 1 - level]);
                }
                right.reverse();
                let q = row[0];
                let m = Point2::new(q[0] / q[2], q[1] / q[2]);
                (
                    Self {
                        a: self.a,
                        b: m,
                        hull: Hull::Bezier(left),
                    },
                    Self {
                        a: m,
                        b: self.b,
                        hull: Hull::Bezier(right),
                    },
                )
            }
        }
    }
}

/// A NURBS face's wires in its surface's parameters, each the closed run of
/// its edges' pcurves ([`super::helpers::pcurve_loops`]), the end of each
/// joined to the start of the next by a segment. On a surface whose sides
/// keep apart, a point of the face is a point of the surface whose
/// parameters lie inside the loops.
///
/// A point's side is read by a ray from its parameters toward growing `v`:
/// each piece of a loop whose box lies past the point along `v` and across
/// its `u` is crossed an odd number of times where its ends lie on either
/// side of the point's `u`, and one whose box misses the ray is crossed by
/// it nowhere. A piece whose box holds the point is halved. Each box holds
/// its piece's own curve and the rounded points the pieces run through, so
/// the loops of the rounded points wind round the point as the face's wires
/// do, wherever the point lies outside the boxes.
pub(super) struct NurbsFaceTrim {
    surface: NurbsSurface,
    pieces: Vec<Piece>,
    /// How far a computed point of a piece, a control point of a half or the
    /// end of a run can stand from where it lies exactly.
    slack: f64,
}

impl NurbsFaceTrim {
    /// The trim of face `face_id` on `surface`. `None` on a periodic
    /// surface, or where [`super::helpers::pcurve_loops`] reads no loops
    /// within `reach`.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if a topology lookup fails.
    pub(super) fn new(
        topo: &Topology,
        face_id: FaceId,
        surface: &NurbsSurface,
        tol: Tolerance,
        reach: f64,
    ) -> Result<Option<Self>, AlgoError> {
        if surface.is_periodic_u() || surface.is_periodic_v() {
            return Ok(None);
        }
        let Some(loops) = super::helpers::pcurve_loops(topo, face_id, surface, tol, reach)? else {
            return Ok(None);
        };
        let runs: Vec<Vec<(&Curve2D, (f64, f64))>> = loops
            .loops
            .iter()
            .map(|wire| {
                wire.iter()
                    .map(|(pcurve, forward)| {
                        let run = if *forward {
                            (pcurve.t_start(), pcurve.t_end())
                        } else {
                            (pcurve.t_end(), pcurve.t_start())
                        };
                        (pcurve.curve(), run)
                    })
                    .collect()
            })
            .collect();
        Ok(Self::of_runs(surface, &runs))
    }

    /// The trim on `surface` of loops each of which runs along its curves
    /// over their parameters, in order. `None` where a B-spline's piece
    /// cannot be read, or the loops hold no piece.
    fn of_runs(surface: &NurbsSurface, loops: &[Vec<(&Curve2D, (f64, f64))>]) -> Option<Self> {
        let mut pieces: Vec<Piece> = Vec::new();
        let (mut size, mut degree) = (0.0_f64, 2_usize);
        for wire in loops {
            let first = pieces.len();
            for &(curve, run) in wire {
                let found = pcurve_pieces(curve, run, &mut size, &mut degree)?;
                if let (Some(last), Some(next)) =
                    (pieces.get(first..).and_then(<[_]>::last), found.first())
                {
                    pieces.push(Piece {
                        a: last.b,
                        b: next.a,
                        hull: Hull::Segment,
                    });
                }
                pieces.extend(found);
            }
            if let (Some(start), Some(end)) = (pieces.get(first), pieces.last()) {
                let (a, b) = (end.b, start.a);
                pieces.push(Piece {
                    a,
                    b,
                    hull: Hull::Segment,
                });
            }
        }
        if pieces.is_empty() || !size.is_finite() {
            return None;
        }
        // A halving rounds each coordinate of a net it blends by half a unit
        // of the net's largest; a projection divides by a weight, which a
        // net's sizes carry as the ratio of its weights; and a point of a
        // line or a conic is a few products and sums.
        #[allow(clippy::cast_precision_loss)]
        let roundings = (8 * (MOST_HALVINGS as usize + 4) * (degree + 2)) as f64;
        Some(Self {
            surface: surface.clone(),
            pieces,
            slack: roundings * f64::EPSILON * size.max(f64::MIN_POSITIVE),
        })
    }

    /// Whether `p`, a point within `near` of the face's surface, lies on the
    /// face: its parameters, where the surface's point there lies within
    /// `near` of it, read against the loops. `None` where no such parameters
    /// are found, or the parameters lie in the box of a piece of a loop
    /// halved as far as it goes. A surface that runs through one point twice
    /// away from its sides gives that point two places in its parameters,
    /// and the reading takes the one the projection finds.
    pub(super) fn holds(&self, p: Point3, near: f64, tol: Tolerance) -> Option<bool> {
        let foot =
            brepkit_math::nurbs::projection::project_point_to_surface(&self.surface, p, tol.linear)
                .ok()?;
        let off = (self.surface.evaluate(foot.u, foot.v) - p).length();
        if off.is_nan() || off > near {
            return None;
        }
        self.holds_parameters(Point2::new(foot.u, foot.v))
    }

    /// Whether the parameters `q` lie inside the loops, or `None` where they
    /// lie in the box of a piece halved as far as it goes.
    fn holds_parameters(&self, q: Point2) -> Option<bool> {
        let mut budget = MOST_PIECES;
        let mut odd = false;
        for piece in &self.pieces {
            odd ^= self.crossed(piece, q, &mut budget, 0)?;
        }
        Some(odd)
    }

    /// Whether the ray from `q` toward growing `v` crosses `piece` an odd
    /// number of times, a point read on the side of growing `u` where its
    /// `u` is that of `q`.
    fn crossed(&self, piece: &Piece, q: Point2, budget: &mut usize, depth: u32) -> Option<bool> {
        let b = piece.bounds(self.slack);
        if b.lo[0] > q.x() || b.hi[0] < q.x() || b.hi[1] < q.y() {
            return Some(false);
        }
        if b.lo[1] > q.y() {
            return Some((piece.a.x() >= q.x()) != (piece.b.x() >= q.x()));
        }
        if depth >= MOST_HALVINGS || *budget == 0 {
            return None;
        }
        *budget -= 1;
        let (left, right) = piece.halves();
        Some(
            self.crossed(&left, q, budget, depth + 1)?
                != self.crossed(&right, q, budget, depth + 1)?,
        )
    }
}

/// The pieces of `curve` from parameter `run.0` to `run.1`, in that order:
/// a line whole, a circle or an ellipse in arcs of at most [`ARC_STEP`], and
/// a B-spline in its Bezier pieces, each between two of its knots. `size`
/// takes the largest coordinate the pieces' points reach, a B-spline's
/// scaled by the ratio of its weights, and `degree` the B-spline's degree.
/// `None` where a B-spline's piece cannot be read.
fn pcurve_pieces(
    curve: &Curve2D,
    run: (f64, f64),
    size: &mut f64,
    degree: &mut usize,
) -> Option<Vec<Piece>> {
    let largest = |p: Point2| p.x().abs().max(p.y().abs());
    match curve {
        Curve2D::Line(line) => {
            *size = size.max(largest(line.origin()) + run.0.abs().max(run.1.abs()));
            Some(vec![Piece {
                a: curve.evaluate(run.0),
                b: curve.evaluate(run.1),
                hull: Hull::Segment,
            }])
        }
        Curve2D::Circle(_) | Curve2D::Ellipse(_) => {
            let (centre, bend) = match curve {
                Curve2D::Circle(c) => (c.center(), c.radius()),
                Curve2D::Ellipse(e) => (e.center(), e.semi_major().max(e.semi_minor())),
                Curve2D::Line(_) | Curve2D::Nurbs(_) => return None,
            };
            *size = size.max(largest(centre) + bend);
            let count = ((run.1 - run.0).abs() / ARC_STEP).ceil().max(1.0);
            if !count.is_finite() || count > 1e6 {
                return None;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let count = count as usize;
            #[allow(clippy::cast_precision_loss)]
            let at = |k: usize| {
                if k == count {
                    run.1
                } else {
                    (run.1 - run.0).madd(k as f64 / count as f64, run.0)
                }
            };
            let points: Vec<(f64, Point2)> = (0..=count)
                .map(|k| {
                    let t = at(k);
                    (t, curve.evaluate(t))
                })
                .collect();
            Some(
                points
                    .windows(2)
                    .map(|w| Piece {
                        a: w[0].1,
                        b: w[1].1,
                        hull: Hull::Conic {
                            curve: curve.clone(),
                            t: (w[0].0, w[1].0),
                            bend,
                        },
                    })
                    .collect(),
            )
        }
        Curve2D::Nurbs(spline) => {
            let weights = spline.weights();
            let (least, most) = weights
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &w| {
                    (lo.min(w), hi.max(w))
                });
            if !(least > 0.0 && least.is_normal()) {
                return None;
            }
            let ratio = most / least;
            for p in spline.control_points() {
                *size = size.max(largest(*p) * ratio);
            }
            *degree = (*degree).max(spline.degree());
            let lifted = brepkit_math::nurbs::curve::NurbsCurve::new(
                spline.degree(),
                spline.knots().to_vec(),
                spline
                    .control_points()
                    .iter()
                    .map(|p| Point3::new(p.x(), p.y(), 0.0))
                    .collect(),
                weights.to_vec(),
            )
            .ok()?;
            let (lo, hi) = (run.0.min(run.1), run.0.max(run.1));
            let mut breaks: Vec<f64> = std::iter::once(lo)
                .chain(spline.knots().iter().copied().filter(|&k| lo < k && k < hi))
                .chain(std::iter::once(hi))
                .collect();
            breaks.dedup_by(|a, b| a <= b);
            if run.0 > run.1 {
                breaks.reverse();
            }
            let mut pieces: Vec<Piece> = Vec::with_capacity(breaks.len());
            for w in breaks.windows(2) {
                let net: Vec<[f64; 3]> =
                    brepkit_math::nurbs::decompose::curve_bezier_between(&lifted, w[0], w[1])
                        .ok()?
                        .into_iter()
                        .map(|q| [q[0], q[1], q[3]])
                        .collect();
                let (first, last) = (net.first()?, net.last()?);
                if !net.iter().all(|q| q[2] > 0.0 && q[2].is_normal()) {
                    return None;
                }
                let a = pieces.last().map_or_else(
                    || Point2::new(first[0] / first[2], first[1] / first[2]),
                    |p| p.b,
                );
                let b = Point2::new(last[0] / last[2], last[1] / last[2]);
                pieces.push(Piece {
                    a,
                    b,
                    hull: Hull::Bezier(net),
                });
            }
            (!pieces.is_empty()).then_some(pieces)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves2d::{Circle2D, Ellipse2D, Line2D, NurbsCurve2D};
    use brepkit_math::vec::Vec2;

    fn sheet() -> NurbsSurface {
        let rows = [0.0, 1.0]
            .iter()
            .map(|&x| [0.0, 1.0].iter().map(|&y| Point3::new(x, y, 0.0)).collect())
            .collect();
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        NurbsSurface::new(1, 1, knots.clone(), knots, rows, vec![vec![1.0; 2]; 2]).unwrap()
    }

    /// The circle about `c` of radius `r` as a rational quadratic B-spline
    /// over `[0, 1]`.
    fn spline_circle(c: Point2, r: f64) -> NurbsCurve2D {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let corners = [
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 0.0),
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (1.0, 0.0),
        ];
        NurbsCurve2D::new(
            2,
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            corners
                .iter()
                .map(|&(x, y)| Point2::new(x.madd(r, c.x()), y.madd(r, c.y())))
                .collect(),
            (0..9).map(|k| if k % 2 == 1 { w } else { 1.0 }).collect(),
        )
        .unwrap()
    }

    /// The unit square less a round hole bounded by a B-spline and an
    /// elliptic hole, each run either way, read at a grid of points and at
    /// points `1e-3`, `1e-6` and `1e-9` to either side of each wire: each
    /// point is read on its own side of every wire, a point on the square's
    /// border, which the reading may put on either side, aside.
    #[test]
    fn parameters_read_inside_loops_of_lines_splines_and_ellipses_as_they_lie() {
        let (c, r) = (Point2::new(0.3, 0.3), 0.15);
        let (e, a, b, turn) = (Point2::new(0.7, 0.65), 0.2, 0.08, 0.5_f64);
        let round = Curve2D::Nurbs(spline_circle(c, r));
        let oval = Curve2D::Ellipse(Ellipse2D::new(e, a, b, turn).unwrap());
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let sides: Vec<Curve2D> = (0..4)
            .map(|k| {
                let (p, q) = (corners[k], corners[(k + 1) % 4]);
                Curve2D::Line(
                    Line2D::new(Point2::new(p.0, p.1), Vec2::new(q.0 - p.0, q.1 - p.1)).unwrap(),
                )
            })
            .collect();
        let inside_oval = |q: Point2| {
            let (s, co) = turn.sin_cos();
            let (dx, dy) = (q.x() - e.x(), q.y() - e.y());
            let (x, y) = (dx.madd(co, dy * s), dy.madd(co, -(dx * s)));
            (x / a).powi(2) + (y / b).powi(2) < 1.0
        };
        let truth = |q: Point2| {
            (0.0..1.0).contains(&q.x())
                && (0.0..1.0).contains(&q.y())
                && (q - c).length() > r
                && !inside_oval(q)
        };
        let mut points: Vec<Point2> = (0..=100)
            .flat_map(|i| (0..=100).map(move |j| (i, j)))
            .map(|(i, j)| {
                Point2::new(
                    0.011f64.madd(f64::from(i), -0.05),
                    0.011f64.madd(f64::from(j), -0.05),
                )
            })
            .collect();
        for k in 0..64 {
            let t = std::f64::consts::TAU * f64::from(k) / 64.0;
            for off in [1e-3, 1e-6, 1e-9, -1e-3, -1e-6, -1e-9] {
                points.push(Point2::new(
                    (r + off).madd(t.cos(), c.x()),
                    (r + off).madd(t.sin(), c.y()),
                ));
                let (s, co) = turn.sin_cos();
                let (x, y) = (a * t.cos(), b * t.sin());
                let normal = Vec2::new(b * t.cos(), a * t.sin());
                let n = normal * (1.0 / normal.length());
                let (nx, ny) = (n.x().madd(co, -(n.y() * s)), n.x().madd(s, n.y() * co));
                points.push(Point2::new(
                    x.madd(co, -(y * s)) + e.x() + nx * off,
                    x.madd(s, y * co) + e.y() + ny * off,
                ));
            }
        }
        let tau = std::f64::consts::TAU;
        for (round_run, oval_run) in [((0.0, 1.0), (0.0, tau)), ((1.0, 0.0), (tau, 0.0))] {
            let loops: Vec<Vec<(&Curve2D, (f64, f64))>> = vec![
                sides.iter().map(|side| (side, (0.0, 1.0))).collect(),
                vec![(&round, round_run)],
                vec![(&oval, oval_run)],
            ];
            let trim = NurbsFaceTrim::of_runs(&sheet(), &loops).unwrap();
            let misread: Vec<(Point2, Option<bool>)> = points
                .iter()
                .filter(|q| {
                    let border = q
                        .x()
                        .abs()
                        .min(q.y().abs())
                        .min((1.0 - q.x()).abs())
                        .min((1.0 - q.y()).abs())
                        < 1e-12;
                    !border
                })
                .map(|&q| (q, trim.holds_parameters(q)))
                .filter(|&(q, read)| read != Some(truth(q)))
                .collect();
            assert!(misread.is_empty(), "{round_run:?}: {misread:?}");
        }
    }

    /// A circle read over a quarter turn, then on round the rest of it, as
    /// two pcurves: the gap closer between them and the arcs' boxes leave
    /// the reading of a point at its centre and just outside it unchanged.
    #[test]
    fn a_circle_in_two_runs_reads_its_inside() {
        let circle = Curve2D::Circle(Circle2D::new(Point2::new(0.5, 0.5), 0.25).unwrap());
        let quarter = std::f64::consts::FRAC_PI_2;
        let loops = vec![vec![
            (&circle, (0.0, quarter)),
            (&circle, (quarter, std::f64::consts::TAU)),
        ]];
        let trim = NurbsFaceTrim::of_runs(&sheet(), &loops).unwrap();
        assert_eq!(trim.holds_parameters(Point2::new(0.5, 0.5)), Some(true));
        assert_eq!(
            trim.holds_parameters(Point2::new(0.75 + 1e-9, 0.5)),
            Some(false)
        );
        assert_eq!(
            trim.holds_parameters(Point2::new(0.75 - 1e-9, 0.5)),
            Some(true)
        );
        assert_eq!(
            trim.holds_parameters(Point2::new(0.5, 0.75 + 1e-9)),
            Some(false)
        );
    }
}
