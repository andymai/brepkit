//! The section of a cylinder and a B-spline extrusion along the cylinder's
//! axis: the lines along the axis through the points where the circle
//! crosses the extrusion's profile.
//!
//! Marched instead, each line came back as a fitted curve that strayed from
//! the straight ruling by up to 1.2e-5 where the walls cross at a shallow
//! angle, and the faces it bounded lost that sliver of volume.

use brepkit_math::aabb::Aabb3;
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;

use super::{RawCurve, trim_t_range_to_aabb};

/// Past this sine of the angle between them, the extrusion does not run
/// along the cylinder's axis.
const PARALLEL_SIN: f64 = 1e-12;

/// A parameter interval of the profile this much narrower than its knot
/// span is not split further.
const NARROWEST: f64 = 1e-12;

/// The most intervals one section may read before it is left to the
/// marcher.
const MOST_INTERVALS: usize = 4096;

/// A profile within rounding of the circle over more of its knot span than
/// this runs along the cylinder, and the section is left to the marcher.
const WIDEST_GRAZE: f64 = 1e-3;

/// The rulings where `cylinder` crosses the B-spline extrusion `surface`,
/// trimmed to both faces' boxes, to the extrusion's height and to the
/// cylinder face's span `v_range` along the axis, each within `tol`, the
/// caller's linear tolerance, of both surfaces. `None` where
/// `surface` is not an extrusion along the cylinder's axis to within `tol`
/// over its length, where its profile runs along the circle, so the
/// section is not a set of isolated lines, or where a ruling is not proved
/// within `tol` of both.
///
/// A ruling where the circle only touches the profile is left out, as a
/// graze splits neither face. A crossing so shallow that rounding hides the
/// profile's side about it is placed at the middle of the hidden run, or
/// left to the marcher where that run spreads wider than `tol`.
pub(super) fn cylinder_extrusion_rulings(
    cylinder: &CylindricalSurface,
    surface: &NurbsSurface,
    (bbox_a, bbox_b): (&Aabb3, &Aabb3),
    tol: f64,
    v_range: Option<(f64, f64)>,
) -> Option<Vec<RawCurve>> {
    let axis = cylinder.axis().normalize().ok()?;
    let (profile, along, slack) = extrusion(surface)?;
    let length = along.length();
    let drift = along.cross(axis).length();
    if length <= 0.0
        || drift > PARALLEL_SIN * length
        || drift + slack + 4.0 * f64::EPSILON * length > tol
    {
        return None;
    }
    let section = CircleSection {
        profile: &profile,
        origin: cylinder.origin(),
        axis,
        radius: cylinder.radius(),
        tol,
    };
    let roots = section.crossings()?;
    // Each ruling runs from the profile to its copy moved by `along`, past
    // which the extrusion has no surface: a point there projects on to the
    // patch's edge and reads as on the face.
    let rise = along.dot(axis);
    // A ruling's points lie off the wall by no more than its base lies off
    // the profile, the extrusion's drift and the slack of its poles' moves.
    let budget = tol - (drift + slack + 4.0 * f64::EPSILON * length);
    let mut lines = Vec::with_capacity(roots.len());
    for t in roots {
        let (base, off) = section.foot(t)?;
        if off > budget {
            return None;
        }
        let boxed = trim_t_range_to_aabb(base, axis, bbox_a, bbox_b);
        let mut t_range = (boxed.0.max(rise.min(0.0)), boxed.1.min(rise.max(0.0)));
        // The cylinder face spans only `v_range` along the axis, measured
        // from the cylinder's origin; the base stands at `height` there.
        if let Some((v0, v1)) = v_range {
            let height = (base - cylinder.origin()).dot(axis);
            t_range = (t_range.0.max(v0 - height), t_range.1.min(v1 - height));
        }
        if t_range.1 - t_range.0 <= tol {
            continue;
        }
        let (p0, p1) = (base + axis * t_range.0, base + axis * t_range.1);
        // What the bounds above miss, a reading of each ruling's ends and
        // middle against both surfaces catches.
        if [p0, p0 + (p1 - p0) * 0.5, p1]
            .into_iter()
            .any(|p| !on_both(cylinder, surface, p, tol))
        {
            return None;
        }
        lines.push(RawCurve {
            curve: EdgeCurve::Line,
            bbox: Aabb3::from_points([p0, p1]),
            t_range,
            p_start: p0,
            p_end: p1,
        });
    }
    Some(lines)
}

/// Whether `p` lies within `tol` of `cylinder`, by its distance from the
/// axis less the radius, and of `wall`, by the distance to a point of it
/// that a projection finds, which no nearest point is farther than.
fn on_both(cylinder: &CylindricalSurface, wall: &NurbsSurface, p: Point3, tol: f64) -> bool {
    let Ok(axis) = cylinder.axis().normalize() else {
        return false;
    };
    let w = p - cylinder.origin();
    let off_axis = (w - axis * w.dot(axis)).length();
    (off_axis - cylinder.radius()).abs() <= tol
        && brepkit_math::nurbs::projection::project_point_to_surface(wall, p, 1e-3 * tol)
            .is_ok_and(|found| found.distance <= tol)
}

/// `surface` as a profile moved along a vector: degree one with two control
/// rows (or columns) across one direction, the second row the first moved
/// by about one vector, with the same weights. With the profile and the
/// first pole's move, how far any pole's move may stray from that one, in
/// model units: each line of the patch across the rows strays from it by
/// a blend of those, so by no more.
fn extrusion(surface: &NurbsSurface) -> Option<(NurbsCurve, Vec3, f64)> {
    let cps = surface.control_points();
    let weights = surface.weights();
    let (first, first_w, second, second_w, degree, knots) =
        if surface.degree_u() == 1 && cps.len() == 2 && surface.knots_u().len() == 4 {
            (
                cps[0].clone(),
                weights[0].clone(),
                cps[1].clone(),
                weights[1].clone(),
                surface.degree_v(),
                surface.knots_v().to_vec(),
            )
        } else if surface.degree_v() == 1
            && cps.iter().all(|row| row.len() == 2)
            && surface.knots_v().len() == 4
        {
            (
                cps.iter().map(|row| row[0]).collect(),
                weights.iter().map(|row| row[0]).collect(),
                cps.iter().map(|row| row[1]).collect(),
                weights.iter().map(|row| row[1]).collect(),
                surface.degree_u(),
                surface.knots_u().to_vec(),
            )
        } else {
            return None;
        };
    let (&start, &moved) = (first.first()?, second.first()?);
    let along = moved - start;
    // Each pole's move less the first's, and what the rounding of the
    // subtractions can hide of it, coordinate by coordinate.
    let rounding = |a: f64, b: f64, c: f64, d: f64| {
        2.0 * f64::EPSILON * (a.abs() + b.abs() + c.abs() + d.abs())
    };
    let slack = first.iter().zip(&second).fold(0.0_f64, |most, (a, b)| {
        let hidden = Vec3::new(
            rounding(a.x(), b.x(), start.x(), moved.x()),
            rounding(a.y(), b.y(), start.y(), moved.y()),
            rounding(a.z(), b.z(), start.z(), moved.z()),
        );
        most.max(((*b - *a) - along).length() + hidden.length())
    });
    let same_weights = first_w
        .iter()
        .zip(&second_w)
        .all(|(a, b)| (a - b).abs() <= 8.0 * f64::EPSILON * a.abs());
    if !same_weights || first_w.iter().any(|w| *w <= 0.0) {
        return None;
    }
    let profile = NurbsCurve::new(degree, knots, first, first_w).ok()?;
    Some((profile, along, slack))
}

/// A profile curve against the circle a cylinder cuts across its axis.
struct CircleSection<'a> {
    profile: &'a NurbsCurve,
    origin: Point3,
    axis: Vec3,
    radius: f64,
    /// The caller's linear tolerance.
    tol: f64,
}

impl CircleSection<'_> {
    /// Where `point` lies across the axis from the cylinder's origin.
    fn across(&self, point: Point3) -> Vec3 {
        let w = point - self.origin;
        w - self.axis * w.dot(self.axis)
    }

    /// The squared distance of the profile's point at `t` from the axis,
    /// less the squared radius: negative inside the cylinder.
    fn reach(&self, t: f64) -> f64 {
        let w = self.across(self.profile.evaluate(t));
        w.dot(w) - self.radius * self.radius
    }

    /// The point where the profile near `t` crosses the circle, and a bound
    /// on how far it lies from the profile.
    ///
    /// The parameter places a crossing only as finely as its spacing, which
    /// on a fast stretch of a rational profile is far coarser in space than
    /// the tolerance: at a shallow crossing the profile's point there stands
    /// off the circle by that spacing times the sine of the angle. So the
    /// crossing is read in space too, where the profile's tangent meets the
    /// cylinder (see [`Self::crossing_near`]). The other choice is the point
    /// on the circle across from the profile's point, as far from the
    /// profile as the profile's point is from the circle. Of the two, the
    /// one with the smaller bound.
    fn foot(&self, t: f64) -> Option<(Point3, f64)> {
        let point = self.profile.evaluate(t);
        let w = point - self.origin;
        let height = w.dot(self.axis);
        let a = self.across(point);
        let radial = a.normalize().ok().map(|out| {
            (
                self.origin + self.axis * height + out * self.radius,
                (a.length() - self.radius).abs() + self.rounding(point),
            )
        });
        match (self.crossing_near(t), radial) {
            (Some(near), Some(radial)) => Some(if near.1 <= radial.1 { near } else { radial }),
            (near, radial) => near.or(radial),
        }
    }

    /// What rounding can move a distance read at `point` by: a few units in
    /// the last place of the largest coordinate there or at the origin.
    fn rounding(&self, point: Point3) -> f64 {
        let place = [point, self.origin]
            .iter()
            .flat_map(|p| [p.x(), p.y(), p.z()])
            .fold(self.radius, |m: f64, c| m.max(c.abs()));
        64.0 * f64::EPSILON * place
    }

    /// Where the profile's tangent meets the cylinder, from Newton steps in
    /// the parameter from `t0` until it can move no more, then one step
    /// along the tangent in space, and a bound on how far that point lies
    /// from the profile.
    ///
    /// Over the parameters from the last one to the float past the step,
    /// the profile lies in the hull of its Bezier control points there, so
    /// within the largest distance of those from the tangent line. Its
    /// shadow on that line runs from its two ends' shadows, so a point of
    /// the line between them lies within that distance of the profile.
    /// `None` where a step does not settle, the parameter strays more than
    /// 1e-6 of the domain, the step leaves the domain, or the point's
    /// shadow lies outside the ends' shadows.
    fn crossing_near(&self, t0: f64) -> Option<(Point3, f64)> {
        let (lo, hi) = self.profile.domain();
        let mut t = t0;
        let mut read = self.tangent_step(t)?;
        for _ in 0..8 {
            let next = (t + read.step).clamp(lo, hi);
            if next.to_bits() == t.to_bits() {
                break;
            }
            t = next;
            read = self.tangent_step(t)?;
        }
        let s = read.step;
        let reached = t + s;
        if (t - t0).abs() > 1e-6 * (hi - lo) || !(lo..=hi).contains(&reached) {
            return None;
        }
        let point = read.point + read.slope * s;
        let slack = self.rounding(point);
        if s == 0.0 {
            return Some((point, slack));
        }
        let (a, b) = if s > 0.0 {
            (t, reached.next_up().min(hi))
        } else {
            (reached.next_down().max(lo), t)
        };
        let dir = read.slope.normalize().ok()?;
        let shadow = |p: Point3| (p - read.point).dot(dir);
        let off_line = |p: Point3| {
            let d = p - read.point;
            (d - dir * d.dot(dir)).length()
        };
        let (from, to) = (
            shadow(self.profile.evaluate(a)),
            shadow(self.profile.evaluate(b)),
        );
        let here = shadow(point);
        if here < from.min(to) - slack || here > from.max(to) + slack {
            return None;
        }
        let mut most: f64 = 0.0;
        for q in curve_bezier_between(self.profile, a, b).ok()? {
            if q[3] <= 0.0 {
                return None;
            }
            most = most.max(off_line(Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3])));
        }
        Some((point, most + 2.0 * slack))
    }

    /// The profile's point at `t`, its derivative there, and the step in the
    /// parameter along that derivative to where the profile's tangent meets
    /// the cylinder: the root of `|a + s b|^2 = r^2` nearer zero, for `a`
    /// the point across the axis and `b` the derivative across it. The step
    /// moves along the whole derivative, so it keeps to the profile's
    /// height along the axis too. `None` where the tangent misses.
    fn tangent_step(&self, t: f64) -> Option<TangentStep> {
        let ders = self.profile.derivatives(t, 1);
        let point = self.profile.evaluate(t);
        let slope = *ders.get(1)?;
        let a = self.across(point);
        let b = slope - self.axis * slope.dot(self.axis);
        let (ab, bb) = (a.dot(b), b.dot(b));
        let c = a.dot(a) - self.radius * self.radius;
        let disc = ab.madd(ab, -(bb * c));
        if bb <= 0.0 || disc < 0.0 {
            return None;
        }
        let q = ab + ab.signum() * disc.sqrt();
        if q == 0.0 {
            return None;
        }
        Some(TangentStep {
            point,
            slope,
            step: -c / q,
        })
    }

    /// The profile's parameters where it crosses the circle, in order.
    /// `None` when the crossings are not isolated within
    /// [`MOST_INTERVALS`] intervals, where the profile runs within rounding
    /// of the circle over more than [`WIDEST_GRAZE`] of a span, where a
    /// crossing hidden by rounding is not placed within the tolerance, or where
    /// a run with no crossing found may hide some (see [`Self::quiet`]).
    ///
    /// The intervals are read in order. One with no root, or with one,
    /// proves the profile's side of the circle at its ends. The intervals
    /// where rounding hides the side make runs: a run between two sides
    /// that differ holds a crossing, read at its middle; between two sides
    /// alike, a touch, where the run is quiet. A closed profile's last run
    /// goes on into its first. An open profile's quiet run at either end
    /// gives no ruling: the line there is the extrusion's own boundary
    /// edge, which the edge and face phase meets on the cylinder.
    fn crossings(&self) -> Option<Vec<f64>> {
        let (d0, d1) = self.profile.domain();
        let mut cuts: Vec<f64> = std::iter::once(d0)
            .chain(
                self.profile
                    .knots()
                    .iter()
                    .copied()
                    .filter(|&k| k > d0 && k < d1),
            )
            .chain(std::iter::once(d1))
            .collect();
        cuts.dedup_by(|a, b| (*a - *b).abs() <= 0.0);
        let mut roots: Vec<f64> = Vec::new();
        let mut budget = MOST_INTERVALS;
        let mut walk = Walk::default();
        for span in cuts.windows(2) {
            let width = span[1] - span[0];
            let mut pending = vec![(span[0], span[1])];
            while let Some((a, b)) = pending.pop() {
                budget = budget.checked_sub(1)?;
                let read = self.bernstein(a, b)?;
                match read.reading() {
                    Reading::Apart => {
                        walk.arrive(read.side_at_start()?, self, &mut roots)?;
                    }
                    Reading::Crossing => {
                        walk.arrive(read.side_at_start()?, self, &mut roots)?;
                        roots.push(self.bisect(a, b));
                        walk.last = Some(read.side_at_end()?);
                    }
                    Reading::Graze if b - a > WIDEST_GRAZE * width => return None,
                    Reading::Unsettled if b - a > NARROWEST * width => {
                        let m = f64::midpoint(a, b);
                        pending.push((m, b));
                        pending.push((a, m));
                    }
                    Reading::Graze | Reading::Unsettled => walk.unknown(a, b, read.most()),
                }
            }
        }
        // An unclamped profile need not start and end at its first and last
        // poles: it closes where its domain's two ends meet.
        let closed = (self.profile.evaluate(d0) - self.profile.evaluate(d1)).length() <= self.tol;
        walk.finish(closed, self, &mut roots)?;
        roots.sort_by(f64::total_cmp);
        Some(roots)
    }

    /// How far apart the points of a run of parameters spread, given as one
    /// or two pieces, the second on from the first across a closed
    /// profile's seam.
    fn spread(&self, ((a, b), (c, d)): Pieces) -> f64 {
        let at = |t: f64| self.profile.evaluate(t);
        (at(b) - at(a)).length() + (at(d) - at(c)).length()
    }

    /// The middle of a run of parameters, given as one or two pieces. `None`
    /// where the run's points spread wider than the tolerance apart.
    fn middle(&self, run: Pieces) -> Option<f64> {
        let ((a, b), (c, d)) = run;
        if self.spread(run) > self.tol {
            return None;
        }
        let half = 0.5 * ((b - a) + (d - c));
        Some(if a + half <= b {
            a + half
        } else {
            c + (half - (b - a))
        })
    }

    /// Whether a run with one side at both ends, where `most` bounds
    /// `reach`, holds no crossing that counts, so that it may be read as a
    /// touch. Crossings are merged into a touch only here, and only where
    /// one of two bounds holds:
    ///
    /// - the run's points spread no wider than the tolerance, so any crossings
    ///   in it lie within the tolerance of each other;
    /// - `|reach| <= most` all along the run keeps the profile within
    ///   `most / (r + sqrt(r^2 - most))` of the circle, and that is no more
    ///   than the tolerance, so any lens between crossings in it is thinner
    ///   than the tolerance.
    ///
    /// Two crossings around a lens deeper than the tolerance fail both, and
    /// the section is left to the marcher unless they are found apart.
    fn quiet(&self, run: Pieces, most: f64) -> bool {
        let r2 = self.radius * self.radius;
        // How far `reach` within `most` of zero lets the profile stand off
        // the circle, read so that it does not cancel.
        let off = most / (self.radius + (r2 - most).max(0.0).sqrt());
        self.spread(run) <= self.tol || off <= self.tol
    }

    /// The Bernstein coefficients over `[a, b]` of `reach` times the weight
    /// function squared, in the profile's Bezier form there, and what
    /// rounding can move each by. The control
    /// points are read across the axis from the cylinder's origin, so the
    /// coefficients grow with the profile's size about the axis; the
    /// rounding of that subtraction grows with where the part sits.
    fn bernstein(&self, a: f64, b: f64) -> Option<Bernstein> {
        let homogeneous = curve_bezier_between(self.profile, a, b).ok()?;
        let p = homogeneous.len().checked_sub(1)?;
        // Each control point across the axis, times its weight, and the
        // weight.
        let mut across = Vec::with_capacity(p + 1);
        let origin = self.origin;
        let mut place = origin.x().abs().max(origin.y().abs()).max(origin.z().abs());
        let mut reach: f64 = self.radius;
        for q in &homogeneous {
            let w = q[3];
            if w <= 0.0 {
                return None;
            }
            let point = Point3::new(q[0] / w, q[1] / w, q[2] / w);
            place = place
                .max(point.x().abs())
                .max(point.y().abs())
                .max(point.z().abs());
            let v = self.across(point);
            reach = reach.max(v.length());
            across.push((v * w, w));
        }
        let max_w = across.iter().map(|&(_, w)| w).fold(0.0, f64::max);
        let min_w = across.iter().map(|&(_, w)| w).fold(f64::INFINITY, f64::min);
        let r2 = self.radius * self.radius;
        let n = 2 * p;
        let mut c = vec![0.0; n + 1];
        for (i, &(vi, wi)) in across.iter().enumerate() {
            for (j, &(vj, wj)) in across.iter().enumerate() {
                let share = binomial(p, i) * binomial(p, j) / binomial(n, i + j);
                c[i + j] += share * (vi.dot(vj) - r2 * wi * wj);
            }
        }
        let band = 64.0
            * f64::EPSILON
            * max_w
            * max_w
            * reach.madd(place, reach.madd(reach, r2))
            * (n + 1) as f64;
        Some(Bernstein {
            c,
            band,
            least_w2: min_w * min_w,
        })
    }
    /// The root of `reach` in `[a, b]`, where it is monotone and changes
    /// sign beyond rounding between the ends.
    fn bisect(&self, mut a: f64, mut b: f64) -> f64 {
        let below = self.reach(a) < 0.0;
        for _ in 0..200 {
            let m = f64::midpoint(a, b);
            if m <= a || m >= b {
                break;
            }
            if (self.reach(m) < 0.0) == below {
                a = m;
            } else {
                b = m;
            }
        }
        f64::midpoint(a, b)
    }
}

/// A run of parameters as one or two pieces, the second on from the first
/// across a closed profile's seam.
type Pieces = ((f64, f64), (f64, f64));

/// A run of parameters where rounding hides the profile's side of the
/// circle, and a bound on `reach` over it.
#[derive(Clone, Copy)]
struct Run {
    from: f64,
    to: f64,
    most: f64,
}

impl Run {
    /// The run as one piece and an empty second piece.
    const fn pieces(self) -> Pieces {
        ((self.from, self.to), (self.to, self.to))
    }
}

/// A walk along the profile: the sides of the circle it is known to lie on,
/// outside (`true`) or inside, and the runs between where rounding hides
/// the side.
#[derive(Default)]
struct Walk {
    /// The first side known, and the run before it, if any.
    first: Option<(bool, Option<Run>)>,
    /// The side last known.
    last: Option<bool>,
    /// The run since the side last known, or since the start.
    run: Option<Run>,
}

impl Walk {
    /// The side over `[a, b]` is not known, and `most` bounds `reach` there.
    fn unknown(&mut self, a: f64, b: f64, most: f64) {
        self.run = Some(self.run.map_or(
            Run {
                from: a,
                to: b,
                most,
            },
            |run| Run {
                to: b,
                most: run.most.max(most),
                ..run
            },
        ));
    }

    /// The profile is known to lie on `side` again: add to `roots` the
    /// crossing in the run since the side last known, where that side
    /// differs. `None` where the sides differ and no run is between, or
    /// the crossing is not placed, or where they are alike and the run is
    /// not quiet. A run before the first side known waits for
    /// [`Self::finish`].
    fn arrive(
        &mut self,
        side: bool,
        section: &CircleSection<'_>,
        roots: &mut Vec<f64>,
    ) -> Option<()> {
        let run = self.run.take();
        match (self.last, run) {
            (None, _) => self.first = Some((side, run)),
            (Some(before), _) if before != side => roots.push(section.middle(run?.pieces())?),
            (Some(_), Some(run)) if !section.quiet(run.pieces(), run.most) => return None,
            (Some(_), _) => {}
        }
        self.last = Some(side);
        Some(())
    }

    /// Settle the runs left at the ends of the walk. On a closed profile the
    /// last run goes on into the first: add to `roots` the crossing across
    /// the seam where the sides there differ. `None` where no side is
    /// known, where that crossing is not placed, or where a run at an end,
    /// or across the seam between sides alike, is not quiet.
    fn finish(self, closed: bool, section: &CircleSection<'_>, roots: &mut Vec<f64>) -> Option<()> {
        let (first, lead) = self.first?;
        let last = self.last?;
        let (d0, d1) = section.profile.domain();
        if !closed {
            return [lead, self.run]
                .into_iter()
                .flatten()
                .all(|run| section.quiet(run.pieces(), run.most))
                .then_some(());
        }
        let (pieces, most) = match (self.run, lead) {
            (None, None) => return (first == last).then_some(()),
            (Some(run), None) | (None, Some(run)) => (run.pieces(), run.most),
            (Some(trail), Some(lead)) => {
                (((trail.from, d1), (d0, lead.to)), trail.most.max(lead.most))
            }
        };
        if first == last {
            return section.quiet(pieces, most).then_some(());
        }
        roots.push(section.middle(pieces)?);
        Some(())
    }
}

/// The Bernstein coefficients over an interval of `reach` times the
/// square of the profile's weight function, what rounding can move each
/// by, and the square of the smallest weight of the interval's Bezier
/// form: the weight function, a blend of those weights, is no smaller.
struct Bernstein {
    c: Vec<f64>,
    band: f64,
    least_w2: f64,
}

impl Bernstein {
    /// The side of the circle at the interval's start, where it is known:
    /// outside (`true`) or inside. The first coefficient is the value there.
    fn side_at_start(&self) -> Option<bool> {
        self.side(*self.c.first()?)
    }

    /// The side of the circle at the interval's end, where it is known.
    fn side_at_end(&self) -> Option<bool> {
        self.side(*self.c.last()?)
    }

    fn side(&self, value: f64) -> Option<bool> {
        (value.abs() > self.band).then_some(value > 0.0)
    }

    /// A bound on `reach` over the interval: the coefficients, each within
    /// the band, bound it times the weight function squared, which is no
    /// less than the smallest weight squared.
    fn most(&self) -> f64 {
        (self.c.iter().fold(0.0, |m: f64, x| m.max(x.abs())) + self.band) / self.least_w2
    }

    /// What the coefficients prove, which bound `reach` and whose
    /// differences bound its slope.
    fn reading(&self) -> Reading {
        let (c, band) = (&self.c, self.band);
        if c.iter().all(|x| x.abs() <= band) {
            return Reading::Graze;
        }
        if c.iter().all(|x| *x > band) || c.iter().all(|x| *x < -band) {
            return Reading::Apart;
        }
        let rising = c.windows(2).all(|pair| pair[1] - pair[0] > 2.0 * band);
        let falling = c.windows(2).all(|pair| pair[1] - pair[0] < -2.0 * band);
        if rising || falling {
            match (self.side_at_start(), self.side_at_end()) {
                (Some(first), Some(last)) if first != last => return Reading::Crossing,
                (Some(_), Some(_)) => return Reading::Apart,
                _ => {}
            }
        }
        Reading::Unsettled
    }
}

/// A profile's point, its derivative there, and the step in its parameter
/// along that derivative to where its tangent meets the cylinder.
struct TangentStep {
    point: Point3,
    slope: Vec3,
    step: f64,
}

/// What the Bernstein coefficients over an interval prove.
enum Reading {
    /// No root.
    Apart,
    /// The profile stays within rounding of the circle: a touch, which
    /// splits neither face.
    Graze,
    /// Exactly one root, where the profile crosses the circle: the sides at
    /// the two ends differ beyond rounding, and the slope keeps one sign.
    Crossing,
    /// None of these: the interval is split.
    Unsettled,
}

fn binomial(n: usize, k: usize) -> f64 {
    let mut value = 1.0;
    for i in 0..k.min(n - k) {
        value = value * (n - i) as f64 / (i + 1) as f64;
    }
    value
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;

    use super::*;

    /// The rulings where `cylinder` crosses `wall`, each checked within the
    /// linear tolerance of both, at its two ends and its middle: off the
    /// cylinder by its distance from the axis less the radius, off the wall
    /// as [`off_wall`] reads it.
    fn rulings(
        cylinder: &CylindricalSurface,
        wall: &NurbsSurface,
        bbox_a: &Aabb3,
        bbox_b: &Aabb3,
    ) -> Option<Vec<RawCurve>> {
        rulings_within(
            cylinder,
            wall,
            (bbox_a, bbox_b),
            brepkit_math::tolerance::Tolerance::new().linear,
        )
    }

    /// [`rulings`] with `tol` for the caller's linear tolerance, each ruling
    /// checked within it.
    fn rulings_within(
        cylinder: &CylindricalSurface,
        wall: &NurbsSurface,
        boxes: (&Aabb3, &Aabb3),
        tol: f64,
    ) -> Option<Vec<RawCurve>> {
        rulings_on_face(cylinder, wall, boxes, (tol, None))
    }

    /// [`rulings_within`] for a cylinder face that spans `v_range` along
    /// the axis, each ruling checked inside that span too.
    fn rulings_on_face(
        cylinder: &CylindricalSurface,
        wall: &NurbsSurface,
        (bbox_a, bbox_b): (&Aabb3, &Aabb3),
        (tol, v_range): (f64, Option<(f64, f64)>),
    ) -> Option<Vec<RawCurve>> {
        let lines = cylinder_extrusion_rulings(cylinder, wall, (bbox_a, bbox_b), tol, v_range)?;
        let axis = cylinder.axis().normalize().unwrap();
        for line in &lines {
            for p in [
                line.p_start,
                line.p_start + (line.p_end - line.p_start) * 0.5,
                line.p_end,
            ] {
                let w = p - cylinder.origin();
                let off_cylinder = ((w - axis * w.dot(axis)).length() - cylinder.radius()).abs();
                let off_wall = off_wall(wall, p);
                assert!(
                    off_cylinder <= tol && off_wall <= tol,
                    "{p:?}: {off_cylinder:e} off the cylinder, {off_wall:e} off the wall"
                );
                if let Some((v0, v1)) = v_range {
                    let v = w.dot(axis);
                    assert!(
                        v >= v0 - tol && v <= v1 + tol,
                        "{p:?}: at {v} along the axis, off the face's {v0}..{v1}"
                    );
                }
            }
        }
        Some(lines)
    }

    /// How far `p` lies from `wall`: the distance to the point a projection
    /// finds, or, where less, to a triangle from there to the wall's points
    /// at the next parameters each way, which strays from the wall by no
    /// more than its sag over one step of the parameters. A fast stretch of
    /// a rational wall spaces the points its parameters reach wider than
    /// the tolerance, and the projection can only land on one of them.
    fn off_wall(wall: &NurbsSurface, p: Point3) -> f64 {
        let found =
            brepkit_math::nurbs::projection::project_point_to_surface(wall, p, 1e-10).unwrap();
        let (u_lo, u_hi) = wall.domain_u();
        let (v_lo, v_hi) = wall.domain_v();
        let (u, v) = (found.u, found.v);
        let mut best = found.distance;
        for u_next in [u.next_up(), u.next_down()] {
            for v_next in [v.next_up(), v.next_down()] {
                let side_u = wall.evaluate(u_next.clamp(u_lo, u_hi), v);
                let side_v = wall.evaluate(u, v_next.clamp(v_lo, v_hi));
                best = best.min(to_triangle(p, [found.point, side_u, side_v]));
            }
        }
        best
    }

    /// The distance from `p` to the triangle `[a, b, c]`.
    fn to_triangle(p: Point3, [a, b, c]: [Point3; 3]) -> f64 {
        let to_segment = |a: Point3, b: Point3| {
            let d = b - a;
            let len2 = d.dot(d);
            let s = if len2 > 0.0 {
                ((p - a).dot(d) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (p - (a + d * s)).length()
        };
        let edges = to_segment(a, b).min(to_segment(b, c)).min(to_segment(c, a));
        let normal = (b - a).cross(c - a);
        let Ok(n) = normal.normalize() else {
            return edges;
        };
        let height = (p - a).dot(n);
        let foot = p - n * height;
        let inside = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(x, y)| (y - x).cross(foot - x).dot(normal) >= 0.0);
        if inside { height.abs() } else { edges }
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            ),
            (
                "mirrored",
                Mat4::translation(1.0, 0.5, -2.0) * Mat4::scale(-1.0, 1.0, 1.0),
            ),
        ]
    }

    fn direction(pose: &Mat4, v: Vec3) -> Vec3 {
        pose.mul_point(Point3::new(v.x(), v.y(), v.z()))
            - pose.mul_point(Point3::new(0.0, 0.0, 0.0))
    }

    /// An ellipse of semi-axes `a` along x and `b` along y about `(cx, cy)`
    /// in `z = -1`, as a rational B-spline, moved up by `rise` (and across by
    /// `lean`), placed by `pose`.
    fn elliptic_extrusion(
        (cx, cy): (f64, f64),
        (a, b): (f64, f64),
        rise: f64,
        lean: f64,
        pose: &Mat4,
    ) -> NurbsSurface {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let unit = [
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
        let row = |shift: Vec3| -> Vec<Point3> {
            unit.iter()
                .map(|&(x, y)| pose.mul_point(Point3::new(cx + a * x, cy + b * y, -1.0) + shift))
                .collect()
        };
        let weights: Vec<f64> = (0..9).map(|i| if i % 2 == 0 { 1.0 } else { s }).collect();
        NurbsSurface::new(
            1,
            2,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            vec![
                row(Vec3::new(0.0, 0.0, 0.0)),
                row(Vec3::new(lean, 0.0, rise)),
            ],
            vec![weights.clone(), weights],
        )
        .unwrap()
    }

    fn rod(pose: &Mat4) -> CylindricalSurface {
        CylindricalSurface::new(
            pose.mul_point(Point3::new(5.0, 5.0, 0.0)),
            direction(pose, Vec3::new(0.0, 0.0, 1.0)),
            2.0,
        )
        .unwrap()
    }

    fn room() -> Aabb3 {
        Aabb3 {
            min: Point3::new(-100.0, -100.0, -100.0),
            max: Point3::new(100.0, 100.0, 100.0),
        }
    }

    /// The circle of radius 2 about (5, 5) and the ellipse of semi-axes 3
    /// and `sqrt(5.4)` about (4, 5) cross at (6, 5 -/+ sqrt(3)) and touch at
    /// (7, 5): two rulings, from the profile up the extrusion's height.
    #[test]
    fn a_rod_crosses_an_elliptic_extrusion_on_two_rulings_and_touches_it_on_none() {
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((4.0, 5.0), (3.0, 5.4_f64.sqrt()), 4.0, 0.0, &pose);
            let lines = rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
            assert_eq!(lines.len(), 2, "{name}");
            let up = direction(&pose, Vec3::new(0.0, 0.0, 4.0));
            let mut lows: Vec<Point3> = Vec::new();
            for line in &lines {
                let (from, to) = if (line.p_end - line.p_start).dot(up) > 0.0 {
                    (line.p_start, line.p_end)
                } else {
                    (line.p_end, line.p_start)
                };
                assert!(
                    (to - (from + up)).length() < 1e-12,
                    "{name}: {from:?} {to:?}"
                );
                lows.push(from);
            }
            for y in [5.0 - 3.0_f64.sqrt(), 5.0 + 3.0_f64.sqrt()] {
                let crossing = pose.mul_point(Point3::new(6.0, y, -1.0));
                let near = lows
                    .iter()
                    .filter(|p| (**p - crossing).length() < 1e-12)
                    .count();
                assert_eq!(near, 1, "{name}: {lows:?} against {crossing:?}");
            }
        }
    }

    #[test]
    fn an_extrusion_leaning_off_the_axis_is_left_to_the_marcher() {
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((4.0, 5.0), (3.0, 5.4_f64.sqrt()), 4.0, 1e-6, &pose);
            assert!(
                rulings(&rod(&pose), &wall, &room(), &room()).is_none(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_profile_running_along_the_circle_is_left_to_the_marcher() {
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((5.0, 5.0), (2.0, 2.0), 4.0, 0.0, &pose);
            assert!(
                rulings(&rod(&pose), &wall, &room(), &room()).is_none(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_profile_clear_of_the_circle_gives_no_ruling() {
        for (name, pose) in poses() {
            for (centre, axes) in [
                ((5.0, 5.0), (3.0, 2.5)),
                ((5.0, 5.0), (1.0, 0.5)),
                ((12.0, 5.0), (3.0, 2.0)),
            ] {
                let wall = elliptic_extrusion(centre, axes, 4.0, 0.0, &pose);
                let lines = rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
                assert!(lines.is_empty(), "{name} {centre:?} {axes:?}");
            }
        }
    }

    /// A degree-1 profile through `points` in `z = -1` over `knots`, moved
    /// up by 4, placed by `pose`.
    fn polyline_extrusion(points: &[(f64, f64)], knots: Vec<f64>, pose: &Mat4) -> NurbsSurface {
        let row = |z: f64| -> Vec<Point3> {
            points
                .iter()
                .map(|&(x, y)| pose.mul_point(Point3::new(x, y, z)))
                .collect()
        };
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            knots,
            vec![row(-1.0), row(3.0)],
            vec![vec![1.0; points.len()]; 2],
        )
        .unwrap()
    }

    fn cylinder_about(x: f64, y: f64, radius: f64, pose: &Mat4) -> CylindricalSurface {
        CylindricalSurface::new(
            pose.mul_point(Point3::new(x, y, 0.0)),
            direction(pose, Vec3::new(0.0, 0.0, 1.0)),
            radius,
        )
        .unwrap()
    }

    /// An elliptic profile whose seam lies on the rod's circle, where it
    /// crosses it: the side there is hidden by rounding at both ends of the
    /// profile's domain, and the crossing is read across the seam.
    #[test]
    fn a_crossing_on_a_closed_profiles_seam_is_one_ruling() {
        let seam = (6.0, 5.0 + 3.0_f64.sqrt());
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((seam.0 - 3.0, seam.1), (3.0, 2.0), 4.0, 0.0, &pose);
            let lines = rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
            assert_eq!(lines.len(), 2, "{name}");
            let crossing = pose.mul_point(Point3::new(seam.0, seam.1, -1.0));
            let near = lines
                .iter()
                .filter(|l| {
                    (l.p_start - crossing).length() < 1e-9 || (l.p_end - crossing).length() < 1e-9
                })
                .count();
            assert_eq!(near, 1, "{name}");
        }
    }

    /// A quarter of the elliptic profile, open, that starts on the rod's
    /// circle where the two touch: the run at its start gives no ruling, and
    /// its one crossing gives one.
    #[test]
    fn an_open_profile_that_starts_on_a_touch_gives_only_its_crossing() {
        let b = 5.4_f64.sqrt();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        for (name, pose) in poses() {
            let row = |z: f64| -> Vec<Point3> {
                [(7.0, 5.0), (7.0, 5.0 + b), (4.0, 5.0 + b)]
                    .iter()
                    .map(|&(x, y)| pose.mul_point(Point3::new(x, y, z)))
                    .collect()
            };
            let wall = NurbsSurface::new(
                1,
                2,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![row(-1.0), row(3.0)],
                vec![vec![1.0, s, 1.0]; 2],
            )
            .unwrap();
            let lines = rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
            assert_eq!(lines.len(), 1, "{name}");
            let crossing = pose.mul_point(Point3::new(6.0, 5.0 + 3.0_f64.sqrt(), -1.0));
            let line = &lines[0];
            assert!(
                (line.p_start - crossing).length() < 1e-9
                    || (line.p_end - crossing).length() < 1e-9,
                "{name}: {:?} {:?}",
                line.p_start,
                line.p_end
            );
        }
    }

    /// Two crossings two apart in space and 5e-11 apart in the profile's
    /// parameter, on a span 1e-10 wide: two rulings.
    #[test]
    fn crossings_close_in_parameter_and_apart_in_space_are_two_rulings() {
        for (name, pose) in poses() {
            let wall = polyline_extrusion(
                &[(-2.0, 0.0), (2.0, 0.0), (3.0, 1.0)],
                vec![0.0, 0.0, 1e-10, 1.0, 1.0],
                &pose,
            );
            let lines = rulings(
                &cylinder_about(0.0, 0.0, 1.0, &pose),
                &wall,
                &room(),
                &room(),
            )
            .unwrap();
            assert_eq!(
                lines.len(),
                2,
                "{name}: {:?}",
                lines
                    .iter()
                    .map(|l| (l.p_start, l.p_end))
                    .collect::<Vec<_>>()
            );
            for x in [-1.0, 1.0] {
                let crossing = pose.mul_point(Point3::new(x, 0.0, -1.0));
                assert!(
                    lines.iter().any(|l| (l.p_start - crossing).length() < 1e-9
                        || (l.p_end - crossing).length() < 1e-9),
                    "{name}: no ruling at {x}"
                );
            }
        }
    }

    /// A profile that keeps at least 1.01 from the axis of a unit cylinder,
    /// a million from the origin: no ruling.
    #[test]
    fn a_profile_clear_of_a_cylinder_far_from_the_origin_gives_no_ruling() {
        for (name, pose) in poses() {
            let wall = polyline_extrusion(
                &[
                    (1_000_002.0, 0.0),
                    (1_000_000.606, 0.808),
                    (1_000_000.0, 2.0),
                ],
                vec![0.0, 0.0, 0.5, 1.0, 1.0],
                &pose,
            );
            let lines = rulings(
                &cylinder_about(1_000_000.0, 0.0, 1.0, &pose),
                &wall,
                &room(),
                &room(),
            );
            assert!(lines.is_some_and(|l| l.is_empty()), "{name}");
        }
    }

    /// A degree-1 profile through `points` in `z = height`, moved by `along`,
    /// placed by `pose`.
    fn moved_polyline(
        points: &[(f64, f64)],
        knots: Vec<f64>,
        (height, weight): (f64, f64),
        along: Vec3,
        pose: &Mat4,
    ) -> NurbsSurface {
        let row = |shift: Vec3| -> Vec<Point3> {
            points
                .iter()
                .map(|&(x, y)| pose.mul_point(Point3::new(x, y, height) + shift))
                .collect()
        };
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            knots,
            vec![row(Vec3::new(0.0, 0.0, 0.0)), row(along)],
            vec![vec![weight; points.len()]; 2],
        )
        .unwrap()
    }

    /// An extrusion 1e8 long that leans 5e-13 off the axis drifts 5e-5
    /// across it over its length: a ruling along the axis strays from its
    /// far end, and the section is left to the marcher.
    #[test]
    fn a_long_extrusion_that_drifts_off_the_axis_is_left_to_the_marcher() {
        for (name, pose) in poses() {
            let wall = moved_polyline(
                &[(0.5, -2.0), (0.5, 2.0)],
                vec![0.0, 0.0, 1.0, 1.0],
                (0.0, 1.0),
                Vec3::new(5e-5, 0.0, 1e8),
                &pose,
            );
            let rod = cylinder_about(0.0, 0.0, 1.0, &pose);
            assert!(rulings(&rod, &wall, &room(), &room()).is_none(), "{name}");
        }
    }

    /// The line x = 0.99999 crosses a unit circle at y = -/+ 0.004472, where
    /// it keeps within 1e-5 of the circle. With the part a thousand million
    /// from the origin, or with the cylinder's origin that far along its
    /// axis, rounding hides the side there, and the run between the
    /// crossings has the same side at both ends. Either both crossings are
    /// found or the section is left to the marcher, for the line and for a
    /// closed profile whose seam lies between them, whatever one weight all
    /// the poles share.
    #[test]
    fn crossings_hidden_in_a_run_with_one_side_at_both_ends_are_not_lost() {
        type Profile<'a> = (&'a [(f64, f64)], Vec<f64>);
        let far = 1e9;
        let x: f64 = 0.99999;
        let y = (1.0 - x * x).sqrt();
        let profiles: [Profile<'_>; 2] = [
            (&[(x, -100.0), (x, 100.0)], vec![0.0, 0.0, 1.0, 1.0]),
            (
                &[
                    (x, 0.0),
                    (x, 100.0),
                    (200.0, 100.0),
                    (200.0, -100.0),
                    (x, -100.0),
                    (x, 0.0),
                ],
                vec![0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            ),
        ];
        let room = Aabb3 {
            min: Point3::new(-1e10, -1e10, -1e10),
            max: Point3::new(1e10, 1e10, 1e10),
        };
        for (name, pose) in poses() {
            // The part far away, or only the cylinder's origin along its axis.
            for (part, origin) in [(far, 0.0), (0.0, far)] {
                let pose = pose * Mat4::translation(0.0, 0.0, part);
                let rod = CylindricalSurface::new(
                    pose.mul_point(Point3::new(0.0, 0.0, origin)),
                    direction(&pose, Vec3::new(0.0, 0.0, 1.0)),
                    1.0,
                )
                .unwrap();
                for (points, knots) in &profiles {
                    let read = |weight: f64| {
                        let wall = moved_polyline(
                            points,
                            knots.clone(),
                            (0.0, weight),
                            Vec3::new(0.0, 0.0, 4.0),
                            &pose,
                        );
                        rulings(&rod, &wall, &room, &room)
                    };
                    let what = format!("{name}, part at {part}, {} poles", points.len());
                    let found = read(1.0);
                    for weight in [1e-3, 1e3] {
                        assert_eq!(
                            read(weight).map(|l| l.len()),
                            found.as_ref().map(Vec::len),
                            "{what}, weight {weight}"
                        );
                    }
                    let Some(lines) = found else {
                        continue;
                    };
                    assert_eq!(lines.len(), 2, "{what}");
                    for y in [-y, y] {
                        let crossing = pose.mul_point(Point3::new(x, y, 0.0));
                        assert!(
                            lines.iter().any(|l| (l.p_start - crossing).length() < 1e-6
                                || (l.p_end - crossing).length() < 1e-6),
                            "{what}: no ruling at {y}"
                        );
                    }
                }
            }
        }
    }

    /// A profile through (0.5, -2), (0.5, 2) and (1e12, 2), whose middle
    /// pole moves 0.001 farther across than the others as it rises by 4.
    /// The ruling at x = 0.5 would miss the wall's top by up to 0.001:
    /// the section is left to the marcher.
    #[test]
    fn an_extrusion_whose_poles_move_apart_is_left_to_the_marcher() {
        for (name, pose) in poses() {
            let points = [(0.5, -2.0), (0.5, 2.0), (1e12, 2.0)];
            let row = |shift: &dyn Fn(usize) -> Vec3| -> Vec<Point3> {
                points
                    .iter()
                    .enumerate()
                    .map(|(i, &(x, y))| pose.mul_point(Point3::new(x, y, 0.0) + shift(i)))
                    .collect()
            };
            let wall = NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.5, 1.0, 1.0],
                vec![
                    row(&|_| Vec3::new(0.0, 0.0, 0.0)),
                    row(&|i| Vec3::new(if i == 1 { 1e-3 } else { 0.0 }, 0.0, 4.0)),
                ],
                vec![vec![1.0; 3]; 2],
            )
            .unwrap();
            let rod = cylinder_about(0.0, 0.0, 1.0, &pose);
            let room = Aabb3 {
                min: Point3::new(-1e13, -1e13, -1e13),
                max: Point3::new(1e13, 1e13, 1e13),
            };
            assert!(rulings(&rod, &wall, &room, &room).is_none(), "{name}");
        }
    }

    /// A line at distance `r cos(angle)` from the axis of a cylinder of
    /// radius `r` about the origin, from y = -2r to 2r, with weights
    /// `weights` at its ends, moved by `length` along the axis, the part
    /// moved by `offset` and placed by `pose`, the cylinder's origin moved
    /// by `along_axis` along its axis.
    fn shallow_crossing(
        (r, angle): (f64, f64),
        weights: (f64, f64),
        length: f64,
        (offset, along_axis): (f64, f64),
        pose: &Mat4,
    ) -> (CylindricalSurface, NurbsSurface) {
        let pose = *pose * Mat4::translation(offset, 0.0, offset);
        let h = r * angle.cos();
        let row = |z: f64| -> Vec<Point3> {
            [(h, -2.0 * r), (h, 2.0 * r)]
                .iter()
                .map(|&(x, y)| pose.mul_point(Point3::new(x, y, z)))
                .collect()
        };
        let wall = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![row(0.0), row(length)],
            vec![vec![weights.0, weights.1]; 2],
        )
        .unwrap();
        let rod = CylindricalSurface::new(
            pose.mul_point(Point3::new(0.0, 0.0, along_axis)),
            direction(&pose, Vec3::new(0.0, 0.0, 1.0)),
            r,
        )
        .unwrap();
        (rod, wall)
    }

    /// Rulings at shallow crossings, over weights, lengths, radii and
    /// placements: each lies on both surfaces (see [`rulings`]). Where the
    /// lens between the two crossings is deeper than the tolerance, both
    /// are found or the section is left to the marcher; where it is
    /// thinner, a touch may stand for them.
    #[test]
    fn rulings_at_shallow_crossings_lie_on_both_walls() {
        for tol in [1e-9, 1e-7, 1e-5] {
            shallow_crossings_within(tol);
        }
    }

    /// [`rulings_at_shallow_crossings_lie_on_both_walls`] for the caller's
    /// linear tolerance `tol`.
    fn shallow_crossings_within(tol: f64) {
        let huge = Aabb3 {
            min: Point3::new(-1e13, -1e13, -1e13),
            max: Point3::new(1e13, 1e13, 1e13),
        };
        for (name, pose) in poses() {
            for angle in [1e-3, 1e-6] {
                for weights in [(1.0, 1.0), (1e-3, 1e3), (1e3, 1e-3)] {
                    for length in [1e-3, 1e6] {
                        for r in [1e-3, 1e6] {
                            for place in [(0.0, 0.0), (1e8, 0.0), (0.0, 1e8)] {
                                let (rod, wall) =
                                    shallow_crossing((r, angle), weights, length, place, &pose);
                                let what = format!(
                                    "tolerance {tol:e}, {name}, angle {angle:e}, weights \
                                     {weights:?}, length {length:e}, radius {r:e}, place \
                                     {place:?}"
                                );
                                let Some(lines) = rulings_within(&rod, &wall, (&huge, &huge), tol)
                                else {
                                    continue;
                                };
                                let depth = 2.0 * r * (0.5 * angle).sin().powi(2);
                                if depth > tol {
                                    assert_eq!(lines.len(), 2, "{what}");
                                } else {
                                    assert!(lines.len() % 2 == 0, "{what}: {}", lines.len());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Two crossings of a unit circle by a line 1e-6 inside it, at 1.4e-3
    /// rad: the lens between them is ten times the tolerance deep, so they
    /// are not a touch. Near the origin both are rulings; with the
    /// cylinder's origin 1e8 along its axis, where rounding hides the side
    /// between them, both are rulings or the section is left to the
    /// marcher.
    #[test]
    fn two_crossings_around_a_lens_ten_tolerances_deep_are_two_rulings() {
        let tol = brepkit_math::tolerance::Tolerance::new().linear;
        let angle = (1.0 - 10.0 * tol).acos();
        for (name, pose) in poses() {
            for along_axis in [0.0, 1e8] {
                let (rod, wall) =
                    shallow_crossing((1.0, angle), (1.0, 1.0), 4.0, (0.0, along_axis), &pose);
                let found = rulings(&rod, &wall, &room(), &room());
                if along_axis == 0.0 {
                    assert_eq!(found.map(|l| l.len()), Some(2), "{name}");
                } else {
                    assert!(found.is_none_or(|l| l.len() == 2), "{name}, far");
                }
            }
        }
    }

    /// A profile through `poles` of degree `degree` over `knots` with
    /// `weights`, moved by `along`, placed by `pose`.
    fn moved_profile(
        poles: &[Point3],
        (degree, knots): (usize, Vec<f64>),
        weights: &[f64],
        along: Vec3,
        pose: &Mat4,
    ) -> NurbsSurface {
        let row = |shift: Vec3| -> Vec<Point3> {
            poles.iter().map(|p| pose.mul_point(*p + shift)).collect()
        };
        NurbsSurface::new(
            1,
            degree,
            vec![0.0, 0.0, 1.0, 1.0],
            knots,
            vec![row(Vec3::new(0.0, 0.0, 0.0)), row(along)],
            vec![weights.to_vec(); 2],
        )
        .unwrap()
    }

    /// The cubic (0.5 + 1.5s, 0.5s^3) over a knot domain one float wide
    /// crosses a circle of radius 1.1 near (1.0993, 0.0319). The parameter
    /// cannot move between its two values, and the profile's tangent at the
    /// lower one, straight along x, meets the circle at (1.1, 0), 0.0316
    /// off the wall: the ruling is placed within the tolerance or the
    /// section is left to the marcher.
    #[test]
    fn a_crossing_on_a_span_one_float_wide_is_placed_on_the_wall_or_deferred() {
        let one_up = 1.0_f64.next_up();
        for (name, pose) in poses() {
            let wall = moved_profile(
                &[
                    Point3::new(0.5, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.5, 0.0, 0.0),
                    Point3::new(2.0, 0.5, 0.0),
                ],
                (3, vec![1.0, 1.0, 1.0, 1.0, one_up, one_up, one_up, one_up]),
                &[1.0; 4],
                Vec3::new(0.0, 0.0, 4.0),
                &pose,
            );
            let rod = cylinder_about(0.0, 0.0, 1.1, &pose);
            if let Some(lines) = rulings(&rod, &wall, &room(), &room()) {
                assert!(lines.len() <= 1, "{name}: {}", lines.len());
            }
        }
    }

    /// The line from (0.5, -2, 0) to (0.5, 2, 1e6) with weights 1e10 and 1
    /// rises along the axis as it runs: a crossing moved across the axis
    /// alone leaves its height behind, and the ruling's end overshoots the
    /// wall. The ruling is placed within the tolerance or the section is
    /// left to the marcher.
    #[test]
    fn a_crossing_on_a_profile_that_rises_keeps_its_height() {
        for (name, pose) in poses() {
            let wall = moved_profile(
                &[Point3::new(0.5, -2.0, 0.0), Point3::new(0.5, 2.0, 1e6)],
                (1, vec![0.0, 0.0, 1.0, 1.0]),
                &[1e10, 1.0],
                Vec3::new(0.0, 0.0, 4.0),
                &pose,
            );
            let rod = cylinder_about(0.0, 0.0, 1.0, &pose);
            let huge = Aabb3 {
                min: Point3::new(-1e7, -1e7, -1e7),
                max: Point3::new(1e7, 1e7, 1e7),
            };
            if let Some(lines) = rulings(&rod, &wall, &huge, &huge) {
                assert!(lines.len() <= 2, "{name}: {}", lines.len());
            }
        }
    }

    /// A line through (0.5, -2), (0.5, 0) and (0.5, 2), whose middle pole
    /// moves 5e-8 across as well as 4 up, crosses a unit cylinder at
    /// y = -/+ 0.866, where its wall's top strays 2.8e-8 off a ruling along
    /// the axis. A caller's tolerance of 1e-9 leaves the section to the
    /// marcher; 1e-7 and 1e-5 hold the rulings. Each ruling lies within the
    /// caller's tolerance of both surfaces.
    #[test]
    fn rulings_hold_to_the_callers_tolerance() {
        for (name, pose) in poses() {
            let points = [(0.5, -2.0), (0.5, 0.0), (0.5, 2.0)];
            let row = |shift: &dyn Fn(usize) -> Vec3| -> Vec<Point3> {
                points
                    .iter()
                    .enumerate()
                    .map(|(i, &(x, y))| pose.mul_point(Point3::new(x, y, 0.0) + shift(i)))
                    .collect()
            };
            let wall = NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.5, 1.0, 1.0],
                vec![
                    row(&|_| Vec3::new(0.0, 0.0, 0.0)),
                    row(&|i| Vec3::new(if i == 1 { 5e-8 } else { 0.0 }, 0.0, 4.0)),
                ],
                vec![vec![1.0; 3]; 2],
            )
            .unwrap();
            let rod = cylinder_about(0.0, 0.0, 1.0, &pose);
            for tol in [1e-9, 1e-7, 1e-5] {
                let found = rulings_within(&rod, &wall, (&room(), &room()), tol);
                if tol < 2.8e-8 {
                    assert!(found.is_none(), "{name}, tolerance {tol:e}");
                } else {
                    assert_eq!(found.map(|l| l.len()), Some(2), "{name}, tolerance {tol:e}");
                }
            }
        }
    }

    /// The rational quadratic circle of radius `radius` about `centre`, in
    /// quarters from angle `start`, with each joint knot `joint` times over
    /// (2 or 3), as poles, knots and weights.
    fn circle_profile(
        centre: (f64, f64),
        radius: f64,
        start: f64,
        joint: usize,
    ) -> (Vec<Point3>, Vec<f64>, Vec<f64>) {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let at = |angle: f64, scale: f64| {
            Point3::new(
                centre.0 + scale * radius * angle.cos(),
                centre.1 + scale * radius * angle.sin(),
                0.0,
            )
        };
        let quarter = std::f64::consts::FRAC_PI_2;
        let (mut poles, mut weights) = (vec![at(start, 1.0)], vec![1.0]);
        let mut knots = vec![0.0; 3];
        for q in 0..4 {
            let a = start + quarter * f64::from(q);
            poles.push(at(a + 0.5 * quarter, std::f64::consts::SQRT_2));
            poles.push(at(a + quarter, 1.0));
            weights.extend([s, 1.0]);
            let k = 0.25 * f64::from(q + 1);
            if q < 3 {
                knots.extend(std::iter::repeat_n(k, joint));
                if joint == 3 {
                    poles.push(at(a + quarter, 1.0));
                    weights.push(1.0);
                }
            }
        }
        knots.extend([1.0; 3]);
        (poles, knots, weights)
    }

    /// The extrusion by 4 along z of a closed profile, placed by `pose`.
    fn closed_extrusion(
        (poles, knots, weights): (Vec<Point3>, Vec<f64>, Vec<f64>),
        pose: &Mat4,
    ) -> NurbsSurface {
        moved_profile(&poles, (2, knots), &weights, Vec3::new(0.0, 0.0, 4.0), pose)
    }

    /// That `lines` are exactly one ruling at each of `crossings`, in `z = 0`
    /// before `pose`.
    fn one_ruling_at_each(lines: &[RawCurve], crossings: &[(f64, f64)], pose: &Mat4, what: &str) {
        assert_eq!(lines.len(), crossings.len(), "{what}");
        for &(x, y) in crossings {
            let at = pose.mul_point(Point3::new(x, y, 0.0));
            let near = lines
                .iter()
                .filter(|l| (l.p_start - at).length() < 1e-9 || (l.p_end - at).length() < 1e-9)
                .count();
            assert_eq!(near, 1, "{what}: at ({x}, {y})");
        }
    }

    /// A circle of radius 2.5 about (3, 5) crosses the rod at
    /// (4.5625, 5 -/+ 1.9516), with its seam on the upper crossing, just
    /// before or just after it by 1e-12 and 1e-9 in its parameter: one
    /// ruling at each crossing.
    #[test]
    fn a_closed_profiles_seam_on_or_beside_a_crossing_gives_one_ruling_each() {
        let (cx, cy, r): (f64, f64, f64) = (3.0, 5.0, 2.5);
        let along: f64 = 1.5625;
        let up = (r * r - along * along).sqrt();
        let crossings = [(cx + along, cy - up), (cx + along, cy + up)];
        let angle = up.atan2(along);
        // The angle a standard circle profile reaches at parameter `t`.
        let reached = |t: f64| {
            let (poles, knots, weights) = circle_profile((0.0, 0.0), 1.0, 0.0, 2);
            let curve = NurbsCurve::new(2, knots, poles, weights).unwrap();
            let p = curve.evaluate(t);
            p.y().atan2(p.x())
        };
        for (name, pose) in poses() {
            for shift in [0.0_f64, 1e-12, 1e-9, -1e-12, -1e-9] {
                let start = angle - reached(shift.rem_euclid(1.0));
                let wall = closed_extrusion(circle_profile((cx, cy), r, start, 2), &pose);
                let lines = rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
                one_ruling_at_each(
                    &lines,
                    &crossings,
                    &pose,
                    &format!("{name}, seam {shift:e}"),
                );
            }
        }
    }

    /// A circle of radius 2.732 about (4, 4) crosses the rod at its seam,
    /// (6.732, 4), and at its first quarter joint, (4, 6.732), whose knot
    /// is three times over, so the profile there is two pieces that meet:
    /// one ruling at each crossing.
    #[test]
    fn crossings_on_a_seam_and_a_knot_of_full_multiplicity_give_one_ruling_each() {
        let alpha = -std::f64::consts::FRAC_PI_6;
        let r = 2.0 * (alpha.cos() - alpha.sin());
        let c = 5.0 + 2.0 * alpha.sin();
        let crossings = [(c + r, c), (c, c + r)];
        for (name, pose) in poses() {
            for joint in [2, 3] {
                let wall = closed_extrusion(circle_profile((c, c), r, 0.0, joint), &pose);
                let lines = rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
                one_ruling_at_each(&lines, &crossings, &pose, &format!("{name}, joint {joint}"));
            }
        }
    }

    /// A closed polygon with its seam at a tip 1e-6 inside a unit circle,
    /// whose sides cross the circle near y = -/+ 5e-7, ten tolerances
    /// apart: one ruling at each crossing.
    #[test]
    fn a_seam_between_two_crossings_ten_tolerances_apart_gives_one_ruling_each() {
        let tip = 1.0 - 1e-6;
        let poles = [
            (tip, 0.0),
            (1.0 + 1e-6, 1e-6),
            (3.0, 1e-6),
            (3.0, -1e-6),
            (1.0 + 1e-6, -1e-6),
            (tip, 0.0),
        ];
        // Where each side from the tip meets the circle x^2 + y^2 = 1.
        let meet = |sign: f64| {
            let (dx, dy) = (1.0 + 1e-6 - tip, sign * 1e-6);
            let (a, b, c) = (dx * dx + dy * dy, 2.0 * tip * dx, tip * tip - 1.0);
            let s = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
            (tip + s * dx, s * dy)
        };
        for (name, pose) in poses() {
            let wall = moved_profile(
                &poles.map(|(x, y)| Point3::new(x, y, 0.0)),
                (1, vec![0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0]),
                &[1.0; 6],
                Vec3::new(0.0, 0.0, 4.0),
                &pose,
            );
            let rod = cylinder_about(0.0, 0.0, 1.0, &pose);
            let lines = rulings(&rod, &wall, &room(), &room()).unwrap();
            one_ruling_at_each(&lines, &[meet(1.0), meet(-1.0)], &pose, name);
        }
    }

    /// A cylinder face from z = 0 to 2 against an extrusion from z = -1 to
    /// 3: each ruling runs only over the face, 0 to 2.
    #[test]
    fn rulings_end_where_a_shorter_cylinder_face_ends() {
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((4.0, 5.0), (3.0, 5.4_f64.sqrt()), 4.0, 0.0, &pose);
            let face = CylindricalSurface::new(
                pose.mul_point(Point3::new(5.0, 5.0, 0.0)),
                direction(&pose, Vec3::new(0.0, 0.0, 1.0)),
                2.0,
            )
            .unwrap();
            let tol = brepkit_math::tolerance::Tolerance::new().linear;
            let lines =
                rulings_on_face(&face, &wall, (&room(), &room()), (tol, Some((0.0, 2.0)))).unwrap();
            assert_eq!(lines.len(), 2, "{name}");
            for line in &lines {
                assert!(
                    ((line.p_end - line.p_start).length() - 2.0).abs() < 1e-9,
                    "{name}: {:?} {:?}",
                    line.p_start,
                    line.p_end
                );
            }
        }
    }

    /// The unclamped quadratic with knots 0 to 8 and poles (0, 0), (2, 0),
    /// (2, 2), (0, 2), (0, 0), (2, 0) closes at (1, 0), though its first and
    /// last poles differ. Against a unit cylinder it crosses at its seam,
    /// (1, 0), and at (0, 1): a ruling at each.
    #[test]
    fn an_unclamped_closed_profile_keeps_its_seam_crossing() {
        let poles = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
            (2.0, 0.0),
        ];
        for (name, pose) in poses() {
            let wall = moved_profile(
                &poles.map(|(x, y)| Point3::new(x, y, 0.0)),
                (2, (0..=8).map(f64::from).collect()),
                &[1.0; 6],
                Vec3::new(0.0, 0.0, 4.0),
                &pose,
            );
            let rod = cylinder_about(0.0, 0.0, 1.0, &pose);
            let lines = rulings(&rod, &wall, &room(), &room()).unwrap();
            one_ruling_at_each(&lines, &[(1.0, 0.0), (0.0, 1.0)], &pose, name);
        }
    }
}
