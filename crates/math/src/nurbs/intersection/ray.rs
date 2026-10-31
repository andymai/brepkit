//! Every crossing of a ray with a NURBS surface.

use crate::frame::Frame3;
use crate::nurbs::decompose::surface_to_bezier_patches;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

/// Deepest subdivision of one Bezier patch.
const MAX_DEPTH: u32 = 48;

/// Most patches one ray may visit before it is reported as grazing.
const MAX_VISITS: usize = 20_000;

/// A leaf's size against the surface's: below it, Newton from the leaf's
/// middle finds the crossing the leaf holds.
const LEAF: f64 = 1e-3;

/// The least slant a hit is read at, however flat the surface.
const GRAZE_COS: f64 = 1e-9;

/// How many leaves apart a hit's second crossing may lie and still share a
/// leaf's Newton with it.
const SEPARATION: f64 = 4.0;

/// A NURBS surface made ready for ray crossings.
///
/// It holds the surface's Bezier patches, each a grid of homogeneous control
/// points. A surface whose weights are all positive lies in the hull of each
/// patch's control points, which the search relies on to pass over the
/// patches a ray misses.
#[derive(Debug, Clone)]
pub struct RaySurface {
    surface: NurbsSurface,
    patches: Vec<Patch>,
    scale: f64,
}

#[derive(Debug, Clone)]
struct Patch {
    rows: usize,
    cols: usize,
    /// `rows` rows of `cols` homogeneous points `[w x, w y, w z, w]`, a row
    /// per `u` index.
    pts: Vec<[f64; 4]>,
    u: (f64, f64),
    v: (f64, f64),
}

/// Where a ray meets a surface.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// The surface's `u` there.
    pub u: f64,
    /// The surface's `v` there.
    pub v: f64,
    /// The distance along the ray's unit direction.
    pub t: f64,
    /// The point.
    pub point: Point3,
}

/// A ray's hits on a surface, and whether any of them, or a place the ray
/// passes, is too near tangent for the count of hits to be trusted.
#[derive(Debug, Clone, Default)]
pub struct RayHits {
    /// The hits, each once.
    pub hits: Vec<RayHit>,
    /// The ray touches the surface tangentially, runs in it, or passes
    /// within the reach of it without a crossing that settles.
    pub grazed: bool,
}

impl Patch {
    fn point(&self, i: usize, j: usize) -> Point3 {
        let [x, y, z, w] = self.pts[i * self.cols + j];
        Point3::new(x / w, y / w, z / w)
    }

    /// The patch split at the middle of its `u` range (`along_u`) or its
    /// `v` range, by de Casteljau on the homogeneous points.
    fn split(&self, along_u: bool) -> (Self, Self) {
        let (n, m) = if along_u {
            (self.rows, self.cols)
        } else {
            (self.cols, self.rows)
        };
        let at = |k: usize, l: usize| {
            if along_u {
                k * self.cols + l
            } else {
                l * self.cols + k
            }
        };
        let mut low = self.pts.clone();
        let mut high = self.pts.clone();
        let mut work = vec![[0.0; 4]; n];
        for l in 0..m {
            for (k, slot) in work.iter_mut().enumerate() {
                *slot = self.pts[at(k, l)];
            }
            low[at(0, l)] = work[0];
            high[at(n - 1, l)] = work[n - 1];
            for r in 1..n {
                for k in 0..n - r {
                    for c in 0..4 {
                        work[k][c] = 0.5 * (work[k][c] + work[k + 1][c]);
                    }
                }
                low[at(r, l)] = work[0];
                high[at(n - 1 - r, l)] = work[n - 1 - r];
            }
        }
        let (mut a, mut b) = (self.clone(), self.clone());
        a.pts = low;
        b.pts = high;
        if along_u {
            let mid = f64::midpoint(self.u.0, self.u.1);
            (a.u, b.u) = ((self.u.0, mid), (mid, self.u.1));
        } else {
            let mid = f64::midpoint(self.v.0, self.v.1);
            (a.v, b.v) = ((self.v.0, mid), (mid, self.v.1));
        }
        (a, b)
    }

    /// The longest control polygon along `u` and along `v`: each bounds the
    /// length of the patch's curves that way, however far a bulge between
    /// its ends rises.
    fn spans(&self) -> (f64, f64) {
        let mut along_u = 0.0_f64;
        for j in 0..self.cols {
            let length: f64 = (1..self.rows)
                .map(|i| (self.point(i, j) - self.point(i - 1, j)).length())
                .sum();
            along_u = along_u.max(length);
        }
        let mut along_v = 0.0_f64;
        for i in 0..self.rows {
            let length: f64 = (1..self.cols)
                .map(|j| (self.point(i, j) - self.point(i, j - 1)).length())
                .sum();
            along_v = along_v.max(length);
        }
        (along_u, along_v)
    }
}

impl RaySurface {
    /// `surface` made ready, or `None` when a weight is not positive, a knot
    /// vector is not clamped, or the surface cannot be cut into Bezier
    /// patches.
    #[must_use]
    pub fn new(surface: &NurbsSurface) -> Option<Self> {
        // An unclamped knot vector's end spans are not Bezier patches of the
        // surface as decomposed, so the search would cross a different one.
        let clamped = |knots: &[f64], degree: usize| {
            let n = knots.len();
            n > 2 * degree + 1
                && knots[..=degree]
                    .iter()
                    .all(|&k| (k - knots[0]).abs() <= 0.0)
                && knots[n - degree - 1..]
                    .iter()
                    .all(|&k| (k - knots[n - 1]).abs() <= 0.0)
        };
        if surface
            .weights()
            .iter()
            .flatten()
            .any(|&w| !w.is_finite() || w <= 0.0)
            || !clamped(surface.knots_u(), surface.degree_u())
            || !clamped(surface.knots_v(), surface.degree_v())
        {
            return None;
        }
        let patches = surface_to_bezier_patches(surface)
            .ok()?
            .into_iter()
            .map(|p| {
                let cps = p.surface.control_points();
                let ws = p.surface.weights();
                let rows = cps.len();
                let cols = cps.first().map_or(0, Vec::len);
                let pts = cps
                    .iter()
                    .zip(ws)
                    .flat_map(|(row, wrow)| {
                        row.iter()
                            .zip(wrow)
                            .map(|(c, &w)| [c.x() * w, c.y() * w, c.z() * w, w])
                    })
                    .collect();
                Patch {
                    rows,
                    cols,
                    pts,
                    u: p.u_range,
                    v: p.v_range,
                }
            })
            .filter(|p| p.rows > 0 && p.cols > 0)
            .collect();
        let b = surface.aabb();
        let scale = (b.max - b.min).length().max(1e-9);
        Some(Self {
            surface: surface.clone(),
            patches,
            scale,
        })
    }

    /// The surface.
    #[must_use]
    pub const fn surface(&self) -> &NurbsSurface {
        &self.surface
    }

    /// Every point where the line through `origin` along `dir` meets the
    /// surface no farther than `reach` behind `origin`, with `t` measured
    /// along `dir` made unit. A hit within `reach` of another is the same
    /// hit.
    #[must_use]
    pub fn hits(&self, origin: Point3, dir: Vec3, reach: f64) -> RayHits {
        let mut out = RayHits::default();
        // How near another hit must lie to each hit to be it: a hit settled
        // on its parameters' spacing is known no closer than that.
        let mut radii: Vec<f64> = Vec::new();
        let Ok(frame) = Frame3::from_normal(origin, dir) else {
            out.grazed = true;
            return out;
        };
        let leaf = LEAF * self.scale;
        let mut stack: Vec<(Patch, u32)> = self.patches.iter().map(|p| (p.clone(), 0)).collect();
        let mut visits = 0_usize;
        while let Some((patch, depth)) = stack.pop() {
            visits += 1;
            if visits > MAX_VISITS {
                out.grazed = true;
                return out;
            }
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for i in 0..patch.rows {
                for j in 0..patch.cols {
                    let d = patch.point(i, j) - origin;
                    for (k, x) in [frame.x.dot(d), frame.y.dot(d), frame.z.dot(d)]
                        .into_iter()
                        .enumerate()
                    {
                        lo[k] = lo[k].min(x);
                        hi[k] = hi[k].max(x);
                    }
                }
            }
            if lo[0] > reach || hi[0] < -reach || lo[1] > reach || hi[1] < -reach || hi[2] < -reach
            {
                continue;
            }
            let (along_u, along_v) = patch.spans();
            if along_u.max(along_v) <= leaf || depth >= MAX_DEPTH {
                // A leaf the solve cannot settle is cut further, and at the
                // deepest level the ray is taken to graze it.
                if self.solve(&patch, origin, frame.z, reach, (&mut out, &mut radii)) {
                    continue;
                }
                if depth >= MAX_DEPTH {
                    out.grazed = true;
                    continue;
                }
            }
            let (a, b) = patch.split(along_u >= along_v);
            stack.push((a, depth + 1));
            stack.push((b, depth + 1));
        }
        out
    }

    /// Whether a ray along `dir` through the surface at `(u, v)` meets it
    /// again too near to be found apart: bending by its normal curvature
    /// `k` along the ray's tangential part `w`, the surface comes back to
    /// the ray `2 (n . dir) / (k |w|^2)` along it. A ray at a tangent point
    /// meets it there twice.
    fn second_crossing_near(&self, u: f64, v: f64, dir: Vec3) -> bool {
        let d = self.surface.derivatives(u, v, 2);
        let (su, sv) = (d[1][0], d[0][1]);
        let Ok(n) = su.cross(sv).normalize() else {
            return true;
        };
        let slant = n.dot(dir);
        if slant.abs() <= GRAZE_COS {
            return true;
        }
        let w = dir - n * slant;
        let (guu, guv, gvv) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let det = guu.mul_add(gvv, -guv * guv);
        if det <= 0.0 || w.length() <= GRAZE_COS {
            return false;
        }
        let (wu, wv) = (su.dot(w), sv.dot(w));
        let a = gvv.mul_add(wu, -guv * wv) / det;
        let b = guu.mul_add(wv, -guv * wu) / det;
        let bend = n.dot(d[2][0] * (a * a) + d[1][1] * (2.0 * a * b) + d[0][2] * (b * b));
        2.0 * slant.abs() <= SEPARATION * LEAF * self.scale * bend.abs()
    }

    /// Newton on `S(u, v) = origin + t dir` from the middle of `leaf`,
    /// recording the hit when it lands in the leaf. Whether it converged,
    /// in the leaf or out of it: a stall (the ray along the tangent plane
    /// at a step) or no convergence leaves the leaf unsettled.
    ///
    /// Where the surface runs fast in its parameters, a unit in the last
    /// place of `u` or `v` moves the point farther than 1e-12 of the
    /// surface's size, and Newton stalls short of that. A hit it stalls on
    /// within four such units is settled there, after Newton has run its
    /// course; it is one hit with any other that near, and when that is
    /// farther than `reach` the ray is flagged, its count in doubt.
    fn solve(
        &self,
        leaf: &Patch,
        origin: Point3,
        dir: Vec3,
        reach: f64,
        (out, radii): (&mut RayHits, &mut Vec<f64>),
    ) -> bool {
        let (mut u, mut v) = (
            f64::midpoint(leaf.u.0, leaf.u.1),
            f64::midpoint(leaf.v.0, leaf.v.1),
        );
        let (du, dv) = (self.surface.domain_u(), self.surface.domain_v());
        let mut t = dir.dot(self.surface.evaluate(u, v) - origin);
        let mut stalled = None;
        for _ in 0..32 {
            let d = self.surface.derivatives(u, v, 1);
            let (s, su, sv) = (d[0][0], d[1][0], d[0][1]);
            let f = s - Vec3::new(origin.x(), origin.y(), origin.z()) - dir * t;
            if f.length() <= 1e-12 * self.scale {
                return self.record(leaf, (u, v, t), origin, dir, reach, reach, (out, radii));
            }
            let spacing = 4.0 * f64::EPSILON * su.length().mul_add(u.abs(), sv.length() * v.abs());
            stalled = (f.length() <= spacing).then_some((u, v, t, spacing));
            let normal = su.cross(sv);
            let det = -normal.dot(dir);
            if det.abs() <= 1e-300 {
                break;
            }
            // Cramer's rule on [su sv -dir] x = -f.
            let neg_dir = -dir;
            let rhs = -f;
            let x_u = rhs.cross(sv).dot(neg_dir) / det;
            let x_v = su.cross(rhs).dot(neg_dir) / det;
            let x_t = normal.dot(rhs) / det;
            u = (u + x_u).clamp(du.0, du.1);
            v = (v + x_v).clamp(dv.0, dv.1);
            t += x_t;
        }
        let Some((u, v, t, spacing)) = stalled else {
            return false;
        };
        out.grazed |= spacing > reach;
        self.record(
            leaf,
            (u, v, t),
            origin,
            dir,
            reach,
            spacing.max(reach),
            (out, radii),
        )
    }

    /// Record the hit at `(u, v)`, `t` along the ray, when it lies in `leaf`
    /// and ahead of `origin` by no less than `-reach`, unless one already
    /// lies within `merge` of it or within its own radius in `radii`, which
    /// keeps one radius per hit: the same crossing merges whichever is found
    /// first. Always settles the leaf.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        leaf: &Patch,
        (u, v, t): (f64, f64, f64),
        origin: Point3,
        dir: Vec3,
        reach: f64,
        merge: f64,
        (out, radii): (&mut RayHits, &mut Vec<f64>),
    ) -> bool {
        let (du, dv) = (self.surface.domain_u(), self.surface.domain_v());
        let slack_u = 1e-9 * (du.1 - du.0);
        let slack_v = 1e-9 * (dv.1 - dv.0);
        if u < leaf.u.0 - slack_u
            || u > leaf.u.1 + slack_u
            || v < leaf.v.0 - slack_v
            || v > leaf.v.1 + slack_v
            || t < -reach
        {
            return true;
        }
        out.grazed |= self.second_crossing_near(u, v, dir);
        let point = origin + dir * t;
        let known = out
            .hits
            .iter()
            .zip(radii.iter())
            .any(|(h, &radius)| (h.point - point).length() <= radius.max(merge));
        if !known {
            out.hits.push(RayHit { u, v, t, point });
            radii.push(merge);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// The lid `z = 100 u (1 - u)^4 v (1 - v)` over the unit square, flat
    /// along its rim and peaking at 2.048 at `(0.2, 0.5)`.
    fn lid() -> NurbsSurface {
        let rows = (0..6)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let z = if (i, j) == (1, 1) { 10.0 } else { 0.0 };
                        Point3::new(f64::from(i) / 5.0, f64::from(j) / 2.0, z)
                    })
                    .collect()
            })
            .collect();
        NurbsSurface::new(
            5,
            2,
            [vec![0.0; 6], vec![1.0; 6]].concat(),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            rows,
            vec![vec![1.0; 3]; 6],
        )
        .unwrap()
    }

    #[test]
    fn a_ray_up_through_the_lids_peak_meets_it_there() {
        let lid = RaySurface::new(&lid()).unwrap();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let at_peak = lid.hits(Point3::new(0.2, 0.5, -1.0), up, 1e-9);
        assert!(!at_peak.grazed);
        assert_eq!(at_peak.hits.len(), 1);
        assert!((at_peak.hits[0].point.z() - 2.048).abs() < 1e-12);
        assert!((at_peak.hits[0].t - 3.048).abs() < 1e-12);
        assert!((at_peak.hits[0].u - 0.2).abs() < 1e-9 && (at_peak.hits[0].v - 0.5).abs() < 1e-9);

        let beside = lid.hits(Point3::new(1.5, 0.5, -1.0), up, 1e-9);
        assert!(beside.hits.is_empty() && !beside.grazed);
        let behind = lid.hits(Point3::new(0.2, 0.5, 3.0), up, 1e-9);
        assert!(behind.hits.is_empty());
    }

    /// A level ray at `z = 1.5` along `u` through `v = 0.5` crosses the
    /// lid's bulge twice, where `100 u (1 - u)^4 / 4 = 1.5`.
    #[test]
    fn a_level_ray_crosses_the_bulge_twice() {
        let lid = RaySurface::new(&lid()).unwrap();
        let across = lid.hits(Point3::new(-1.0, 0.5, 1.5), Vec3::new(1.0, 0.0, 0.0), 1e-9);
        assert!(!across.grazed);
        assert_eq!(across.hits.len(), 2, "{:?}", across.hits);
        for h in &across.hits {
            let u = h.point.x();
            assert!((25.0 * u * (1.0 - u).powi(4) - 1.5).abs() < 1e-9, "{h:?}");
        }
    }

    #[test]
    fn a_ray_touching_the_peak_grazes() {
        let lid = RaySurface::new(&lid()).unwrap();
        let level = lid.hits(
            Point3::new(-1.0, 0.5, 2.048),
            Vec3::new(1.0, 0.0, 0.0),
            1e-9,
        );
        assert!(level.grazed, "{level:?}");
    }

    /// A unit biquadratic patch with only its middle control point raised,
    /// to 4000: a level ray at 500 along u through v = 0.5 crosses its
    /// spike twice, where 4000 u (1 - u) = 500 at v = 0.5, so at
    /// u = (1 ± 1/√2) / 2.
    #[test]
    fn a_level_ray_crosses_a_tall_spike() {
        let rows = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let z = if (i, j) == (1, 1) { 4000.0 } else { 0.0 };
                        Point3::new(f64::from(i) / 2.0, f64::from(j) / 2.0, z)
                    })
                    .collect()
            })
            .collect();
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let spike =
            NurbsSurface::new(2, 2, knots.clone(), knots, rows, vec![vec![1.0; 3]; 3]).unwrap();
        let ray = RaySurface::new(&spike).unwrap();
        let across = ray.hits(
            Point3::new(-1.0, 0.5, 500.0),
            Vec3::new(1.0, 0.0, 0.0),
            1e-9,
        );
        assert!(!across.grazed);
        let mut xs: Vec<f64> = across.hits.iter().map(|h| h.point.x()).collect();
        xs.sort_by(f64::total_cmp);
        let root = std::f64::consts::FRAC_1_SQRT_2;
        assert_eq!(xs.len(), 2, "{xs:?}");
        assert!((xs[0] - (1.0 - root) / 2.0).abs() < 1e-9, "{xs:?}");
        assert!((xs[1] - f64::midpoint(1.0, root)).abs() < 1e-9, "{xs:?}");
    }

    /// The saddle `(u, v, (u - 1/2)(v - 1/2))` over the unit square, beside
    /// a span reaching to `x = 2000` that makes the whole saddle one leaf.
    /// A level ray from its middle at height 0.04 along `(1, 1, 0)` crosses
    /// it at `(0.7, 0.7)`, though at the leaf's middle the ray runs along
    /// the tangent plane and a solve from there stalls.
    #[test]
    fn a_ray_level_with_a_saddles_middle_crosses_it() {
        let p = Point3::new;
        let rows = vec![
            vec![p(0.0, 0.0, 0.25), p(0.0, 1.0, -0.25)],
            vec![p(1.0, 0.0, -0.25), p(1.0, 1.0, 0.25)],
            vec![p(2000.0, 0.0, 0.0), p(2000.0, 1.0, 0.0)],
        ];
        let s = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 2.0, 2.0],
            vec![0.0, 0.0, 1.0, 1.0],
            rows,
            vec![vec![1.0; 2]; 3],
        )
        .unwrap();
        let ray = RaySurface::new(&s).unwrap();
        let dir = Vec3::new(1.0, 1.0, 0.0).normalize().unwrap();
        let found = ray.hits(p(0.5, 0.5, 0.04), dir, 1e-9);
        assert_eq!(found.hits.len(), 1, "{:?}", found.grazed);
        assert!((found.hits[0].point - p(0.7, 0.7, 0.04)).length() < 1e-9);
    }

    /// A surface whose `u` knots run `[-2, -1, 0, 1, 2, 3]`, unclamped, is
    /// not taken: its end patches as decomposed are not the surface.
    #[test]
    fn an_unclamped_surface_is_not_ready() {
        let rows = (0..3)
            .map(|i| {
                (0..2)
                    .map(|j| Point3::new(0.5 * f64::from(i), f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let s = NurbsSurface::new(
            2,
            1,
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.0, 1.0, 1.0],
            rows,
            vec![vec![1.0; 2]; 3],
        )
        .unwrap();
        assert!(RaySurface::new(&s).is_none());
    }

    #[test]
    fn a_surface_with_a_negative_weight_is_not_ready() {
        let mut weights = vec![vec![1.0; 3]; 6];
        weights[2][1] = -1.0;
        let s = lid();
        let s = NurbsSurface::new(
            5,
            2,
            s.knots_u().to_vec(),
            s.knots_v().to_vec(),
            s.control_points().to_vec(),
            weights,
        )
        .unwrap();
        assert!(RaySurface::new(&s).is_none());
    }

    /// A flat unit patch crossed slant: one hit over [0, 1] in each
    /// parameter, over [0, 1e-12], and over [2^40, 2^40 + 1], each through
    /// its middle, where every leaf of its subdivision has a corner. Over
    /// [2^40, 2^40 + 1] a unit in the last place of a parameter is 2^-12:
    /// the middle lies on that grid, but (0.50003, 0.50007) does not, and
    /// its hit, settled no finer, is flagged.
    #[test]
    fn a_patch_on_a_tiny_or_far_knot_domain_is_crossed_once() {
        let p = Point3::new;
        let corners = vec![
            vec![p(0.0, 0.0, 0.0), p(0.0, 1.0, 0.0)],
            vec![p(1.0, 0.0, 0.0), p(1.0, 1.0, 0.0)],
        ];
        let dir = Vec3::new(0.3, 0.2, -1.0);
        let far = 2.0_f64.powi(40);
        let middle = p(0.5, 0.5, 0.0);
        for (lo, hi, through, flagged) in [
            (0.0, 1.0, middle, false),
            (0.0, 1e-12, middle, false),
            (far, far + 1.0, middle, false),
            (far, far + 1.0, p(0.500_03, 0.500_07, 0.0), true),
        ] {
            let knots = vec![lo, lo, hi, hi];
            let s = NurbsSurface::new(
                1,
                1,
                knots.clone(),
                knots,
                corners.clone(),
                vec![vec![1.0; 2]; 2],
            )
            .unwrap();
            let found = RaySurface::new(&s).unwrap().hits(through - dir, dir, 1e-6);
            let case = format!("knots {lo} to {hi} through {through:?}");
            assert_eq!(found.hits.len(), 1, "{case}: {:?}", found.hits);
            assert_eq!(found.grazed, flagged, "{case}");
        }
    }

    /// Two hits 1e-4 apart, one settled on a spacing of 1e-3 and one found
    /// to 1e-6, are one hit whichever is found first.
    #[test]
    fn hits_merge_on_the_wider_radius_in_either_order() {
        let p = Point3::new;
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let flat = NurbsSurface::new(
            1,
            1,
            knots.clone(),
            knots,
            vec![
                vec![p(0.0, 0.0, 0.0), p(0.0, 1.0, 0.0)],
                vec![p(1.0, 0.0, 0.0), p(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0; 2]; 2],
        )
        .unwrap();
        let ray = RaySurface::new(&flat).unwrap();
        let leaf = ray.patches[0].clone();
        let (origin, dir) = (p(0.5, 0.5, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let wide = ((0.5, 0.5, 1.0), 1e-3);
        let near = ((0.5, 0.5, 1.0 + 1e-4), 1e-6);
        for order in [[wide, near], [near, wide]] {
            let (mut out, mut radii) = (RayHits::default(), Vec::new());
            for (hit, merge) in order {
                assert!(ray.record(&leaf, hit, origin, dir, 1e-6, merge, (&mut out, &mut radii)));
            }
            assert_eq!(out.hits.len(), 1, "{order:?}: {:?}", out.hits);
        }
    }
}
