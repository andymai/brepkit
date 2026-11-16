//! Whether a spline surface folds back over itself: has a place, off its
//! collapsed edges, where its normal vanishes and turns over.
//!
//! On one Bezier patch with homogeneous control points `(H, w)`, the normal
//! `S_u x S_v` of `S = H / w` is `N / w³` with
//! `N = w (H_u x H_v) + w_v (H x H_u) + w_u (H_v x H)`, a polynomial; for a
//! polynomial patch `N = H_u x H_v`. Its Bernstein coefficients come from
//! products of the control points, and inside the patch `N` is a blend of
//! them with positive weights. If none points against their sum, `N` does
//! not vanish inside the patch. On a side of the patch only the coefficients
//! of that side have weight, and at a corner only one, so each side and each
//! corner needs a coefficient of its own along the sum. A side that is an
//! edge of the surface and has shrunk to a point is let off: the surface has
//! no normal there. A part that fails is halved both ways and read again; a
//! fold fails at every depth.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};

/// Halvings of a Bezier patch, along each parameter, before a part that still
/// cannot be proved regular counts as folded.
const DEEPEST: u32 = 14;

/// A vector this small a share of the vectors it is read against is zero:
/// they are parallel, or it is nothing, to rounding.
const PARALLEL: f64 = 1e-12;

/// The first place `surface` folds: the middle `(u, v)` of the smallest part
/// of it that could not be proved regular. `None` when it is regular
/// everywhere off its collapsed edges, and for a surface of degree zero, which
/// has no normal, or one with a weight that is not positive.
pub(super) fn first_fold(surface: &NurbsSurface) -> Option<(f64, f64)> {
    if surface.degree_u() == 0
        || surface.degree_v() == 0
        || surface
            .weights()
            .iter()
            .flatten()
            .any(|&w| w.is_nan() || w <= 0.0)
    {
        return None;
    }
    let (domain_u, domain_v) = (surface.domain_u(), surface.domain_v());
    for (net, u, v) in span_nets(surface)? {
        let on_edge = [
            u.0 <= domain_u.0,
            u.1 >= domain_u.1,
            v.0 <= domain_v.0,
            v.1 >= domain_v.1,
        ];
        if let Some(fold) = fold_in(&net, u, v, on_edge, 0) {
            return Some(fold);
        }
    }
    None
}

/// The Bezier net of each knot-span patch of `surface`, with its ranges of
/// u and v, from the blossoms of each span's de Boor points: no knot is
/// inserted, so a knot vector that is not clamped needs nothing more.
fn span_nets(surface: &NurbsSurface) -> Option<Vec<SpanNet>> {
    let (pu, pv) = (surface.degree_u(), surface.degree_v());
    let (knots_u, knots_v) = (surface.knots_u(), surface.knots_v());
    let (points, weights) = (surface.control_points(), surface.weights());
    let spans = |knots: &[f64], p: usize| -> Vec<usize> {
        (p..knots.len() - p - 1)
            .filter(|&i| knots[i] < knots[i + 1])
            .collect()
    };
    // The span `[knots[i], knots[i + 1]]` of the curve on `points`, as the
    // curve of those `p + 1` points alone over the knots that shape it.
    let piece = |points: Vec<Point3>, weights: Vec<f64>, knots: &[f64], p: usize, i: usize| {
        let local = NurbsCurve::new(p, knots[i - p..=i + p + 1].to_vec(), points, weights).ok()?;
        curve_bezier_between(&local, knots[i], knots[i + 1]).ok()
    };
    let spans_v = spans(knots_v, pv);
    let mut nets = Vec::new();
    for i in spans(knots_u, pu) {
        for &j in &spans_v {
            let rows: Vec<Vec<[f64; 4]>> = (i - pu..=i)
                .map(|r| {
                    piece(
                        points[r][j - pv..=j].to_vec(),
                        weights[r][j - pv..=j].to_vec(),
                        knots_v,
                        pv,
                        j,
                    )
                })
                .collect::<Option<_>>()?;
            let mut net = vec![vec![[0.0; 4]; pv + 1]; pu + 1];
            for c in 0..=pv {
                let (column, column_weights): (Vec<Point3>, Vec<f64>) = rows
                    .iter()
                    .map(|row| {
                        let q = row[c];
                        (Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3]), q[3])
                    })
                    .unzip();
                for (r, q) in piece(column, column_weights, knots_u, pu, i)?
                    .into_iter()
                    .enumerate()
                {
                    net[r][c] = q;
                }
            }
            nets.push((
                net,
                (knots_u[i], knots_u[i + 1]),
                (knots_v[j], knots_v[j + 1]),
            ));
        }
    }
    Some(nets)
}

/// `on_edge`: whether the part's sides at its least u, greatest u, least v
/// and greatest v lie on the surface's own edges.
fn fold_in(
    net: &[Vec<[f64; 4]>],
    u: (f64, f64),
    v: (f64, f64),
    on_edge: [bool; 4],
    depth: u32,
) -> Option<(f64, f64)> {
    if regular(net, on_edge) {
        return None;
    }
    let (mid_u, mid_v) = (f64::midpoint(u.0, u.1), f64::midpoint(v.0, v.1));
    if depth == DEEPEST {
        return Some((mid_u, mid_v));
    }
    let (low, high) = halve_rows(net);
    let halves_u = [
        (low, (u.0, mid_u), [on_edge[0], false]),
        (high, (mid_u, u.1), [false, on_edge[1]]),
    ];
    for (half, half_u, edge_u) in halves_u {
        let (left, right): (Vec<_>, Vec<_>) = half.iter().map(|row| halve(row)).unzip();
        let halves_v = [
            (left, (v.0, mid_v), [on_edge[2], false]),
            (right, (mid_v, v.1), [false, on_edge[3]]),
        ];
        for (quarter, quarter_v, edge_v) in halves_v {
            let edges = [edge_u[0], edge_u[1], edge_v[0], edge_v[1]];
            if let Some(fold) = fold_in(&quarter, half_u, quarter_v, edges, depth + 1) {
                return Some(fold);
            }
        }
    }
    None
}

/// Homogeneous control points `[w x, w y, w z, w]`, in rows along u.
type Net = Vec<Vec<[f64; 4]>>;

/// A patch's net with its ranges of u and v.
type SpanNet = (Net, (f64, f64), (f64, f64));

/// Coefficients in rows along u, `cols` to a row.
struct Grid<T> {
    rows: usize,
    cols: usize,
    values: Vec<T>,
}

impl<T: Copy> Grid<T> {
    fn at(&self, i: usize, j: usize) -> T {
        self.values[i * self.cols + j]
    }
}

/// `C(n, k)`.
fn choose(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, m| acc * (n - m) as f64 / (m + 1) as f64)
}

/// The Bernstein coefficients of the product of two tensor polynomials given
/// by theirs, each product of coefficients taken by `times`.
fn product<A: Copy, B: Copy>(a: &Grid<A>, b: &Grid<B>, times: impl Fn(A, B) -> Vec3) -> Grid<Vec3> {
    let (pa, qa) = (a.rows - 1, a.cols - 1);
    let (pb, qb) = (b.rows - 1, b.cols - 1);
    let (rows, cols) = (pa + pb + 1, qa + qb + 1);
    let mut values = vec![Vec3::new(0.0, 0.0, 0.0); rows * cols];
    for i1 in 0..=pa {
        for i2 in 0..=pb {
            let share_u = choose(pa, i1) * choose(pb, i2) / choose(pa + pb, i1 + i2);
            for j1 in 0..=qa {
                for j2 in 0..=qb {
                    let share_v = choose(qa, j1) * choose(qb, j2) / choose(qa + qb, j1 + j2);
                    let k = (i1 + i2) * cols + j1 + j2;
                    values[k] += times(a.at(i1, j1), b.at(i2, j2)) * (share_u * share_v);
                }
            }
        }
    }
    Grid { rows, cols, values }
}

/// The scalar coefficients `scale` times the vector ones `of`.
fn scaled(scale: &Grid<f64>, of: &Grid<Vec3>) -> Grid<Vec3> {
    product(scale, of, |s, v| v * s)
}

/// The coefficients of `N` (see the module notes) on a part.
fn normal_coefficients(net: &[Vec<[f64; 4]>]) -> Grid<Vec3> {
    let (rows, cols) = (net.len(), net[0].len());
    let (m, n) = ((rows - 1) as f64, (cols - 1) as f64);
    let point = |q: [f64; 4]| Vec3::new(q[0], q[1], q[2]);
    let grid = |rows: usize, cols: usize, at: &dyn Fn(usize, usize) -> [f64; 4]| {
        let mut values = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                values.push(at(i, j));
            }
        }
        Grid { rows, cols, values }
    };
    let step = |a: [f64; 4], b: [f64; 4], k: f64| std::array::from_fn(|c| k * (b[c] - a[c]));
    let h = grid(rows, cols, &|i, j| net[i][j]);
    let h_u = grid(rows - 1, cols, &|i, j| step(net[i][j], net[i + 1][j], m));
    let h_v = grid(rows, cols - 1, &|i, j| step(net[i][j], net[i][j + 1], n));
    let cross = |a: [f64; 4], b: [f64; 4]| point(a).cross(point(b));
    let along = product(&h_u, &h_v, cross);
    if net
        .iter()
        .flatten()
        .all(|q| q[3].to_bits() == 1.0_f64.to_bits())
    {
        return along;
    }
    let weight = |g: &Grid<[f64; 4]>| Grid {
        rows: g.rows,
        cols: g.cols,
        values: g.values.iter().map(|q| q[3]).collect(),
    };
    let first = scaled(&weight(&h), &along);
    let second = scaled(&weight(&h_v), &product(&h, &h_u, cross));
    let third = scaled(&weight(&h_u), &product(&h_v, &h, cross));
    Grid {
        rows: first.rows,
        cols: first.cols,
        values: first
            .values
            .iter()
            .zip(&second.values)
            .zip(&third.values)
            .map(|((a, b), c)| *a + *b + *c)
            .collect(),
    }
}

/// Whether no coefficient of `N` points against the sum of them all, and
/// each side and corner has one that points along it. A part with no
/// coefficient of any size (it has shrunk to a curve or a point) is not
/// regular.
fn regular(net: &[Vec<[f64; 4]>], on_edge: [bool; 4]) -> bool {
    let normal = normal_coefficients(net);
    let sum = normal
        .values
        .iter()
        .fold(Vec3::new(0.0, 0.0, 0.0), |s, c| s + *c);
    let length = sum.length();
    let largest = normal.values.iter().fold(0.0_f64, |m, c| m.max(c.length()));
    if largest == 0.0 || length <= PARALLEL * largest {
        return false;
    }
    if normal
        .values
        .iter()
        .any(|c| c.dot(sum) < -PARALLEL * c.length() * length)
    {
        return false;
    }
    let along =
        |c: Vec3| c.length() > PARALLEL * largest && c.dot(sum) > PARALLEL * c.length() * length;

    let euclid = |q: [f64; 4]| Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3]);
    let first = euclid(net[0][0]);
    let size = net
        .iter()
        .flatten()
        .map(|q| (euclid(*q) - first).length())
        .fold(0.0, f64::max);
    let is_point = |points: &mut dyn Iterator<Item = [f64; 4]>| {
        let mut points = points.map(euclid);
        points
            .next()
            .is_none_or(|p0| points.all(|p| (p - p0).length() <= PARALLEL * size))
    };
    let (last_row, last_col) = (net.len() - 1, net[0].len() - 1);
    let (k, l) = (normal.rows - 1, normal.cols - 1);
    let let_off = [
        on_edge[0] && is_point(&mut net[0].iter().copied()),
        on_edge[1] && is_point(&mut net[last_row].iter().copied()),
        on_edge[2] && is_point(&mut net.iter().map(|row| row[0])),
        on_edge[3] && is_point(&mut net.iter().map(|row| row[last_col])),
    ];
    let sides: [Vec<Vec3>; 4] = [
        (0..=l).map(|j| normal.at(0, j)).collect(),
        (0..=l).map(|j| normal.at(k, j)).collect(),
        (0..=k).map(|i| normal.at(i, 0)).collect(),
        (0..=k).map(|i| normal.at(i, l)).collect(),
    ];
    if sides
        .iter()
        .zip(let_off)
        .any(|(side, off)| !off && !side.iter().any(|c| along(*c)))
    {
        return false;
    }
    [
        (normal.at(0, 0), 0, 2),
        (normal.at(0, l), 0, 3),
        (normal.at(k, 0), 1, 2),
        (normal.at(k, l), 1, 3),
    ]
    .iter()
    .all(|&(c, side_u, side_v)| let_off[side_u] || let_off[side_v] || along(c))
}

/// The control points of the two halves of a Bezier curve.
fn halve(points: &[[f64; 4]]) -> (Vec<[f64; 4]>, Vec<[f64; 4]>) {
    let mut work = points.to_vec();
    let mut low = Vec::with_capacity(points.len());
    let mut high = Vec::with_capacity(points.len());
    for level in 0..points.len() {
        let last = points.len() - 1 - level;
        low.push(work[0]);
        high.push(work[last]);
        for k in 0..last {
            for c in 0..4 {
                work[k][c] = 0.5 * (work[k][c] + work[k + 1][c]);
            }
        }
    }
    high.reverse();
    (low, high)
}

/// The two halves of a patch along u, whose rows run along v.
fn halve_rows(net: &[Vec<[f64; 4]>]) -> (Net, Net) {
    let columns = net.first().map_or(0, Vec::len);
    let mut low = vec![Vec::with_capacity(columns); net.len()];
    let mut high = vec![Vec::with_capacity(columns); net.len()];
    for j in 0..columns {
        let column: Vec<[f64; 4]> = net.iter().map(|row| row[j]).collect();
        let (a, b) = halve(&column);
        for (i, (pa, pb)) in a.into_iter().zip(b).enumerate() {
            low[i].push(pa);
            high[i].push(pb);
        }
    }
    (low, high)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A strip along u over the plane cubic whose control `x` values are
    /// `xs`, with weights `weights` across, `rows` control rows along u.
    fn strip(xs: [f64; 4], weights: [f64; 4], rows: usize) -> NurbsSurface {
        let net = (0..rows)
            .map(|i| xs.iter().map(|&x| Point3::new(x, i as f64, 0.0)).collect())
            .collect();
        let mut knots_u = vec![0.0; 2];
        knots_u.extend((1..rows - 1).map(|k| k as f64 / (rows - 1) as f64));
        knots_u.extend([1.0; 2]);
        NurbsSurface::new(
            1,
            3,
            knots_u,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            net,
            vec![weights.to_vec(); rows],
        )
        .unwrap()
    }

    /// Whether `S_u x S_v` along z changes sign across v at some u, by
    /// dense samples.
    fn turns_over(surface: &NurbsSurface) -> bool {
        let sign = |u: f64, v: f64| {
            let d = surface.derivatives(u, v, 1);
            d[1][0].cross(d[0][1]).z()
        };
        let mut seen = (false, false);
        for i in 0..=20 {
            for j in 0..=400 {
                let z = sign(f64::from(i) / 20.0, f64::from(j) / 400.0);
                seen = (seen.0 || z > 1e-9, seen.1 || z < -1e-9);
            }
        }
        seen.0 && seen.1
    }

    #[test]
    fn a_section_that_runs_back_folds() {
        for weights in [[1.0; 4], [1.0, 2.0, 0.5, 1.0]] {
            let folded = strip([0.0, 12.0, -6.0, 6.0], weights, 3);
            assert!(turns_over(&folded), "{weights:?}");
            let (_, v) = first_fold(&folded).expect("a fold");
            assert!((0.05..=0.95).contains(&v), "{weights:?}: v = {v}");
        }
    }

    #[test]
    fn a_section_that_runs_one_way_does_not_fold() {
        for weights in [[1.0; 4], [1.0, 2.0, 0.5, 1.0]] {
            let plain = strip([0.0, 2.0, 4.0, 6.0], weights, 4);
            assert!(!turns_over(&plain), "{weights:?}");
            assert_eq!(first_fold(&plain), None, "{weights:?}");
        }
    }

    /// A strip whose section is a quadratic spline over the knots 0 to 8,
    /// not clamped, through control `x` values `xs`.
    fn unclamped(xs: [f64; 6]) -> NurbsSurface {
        let net = (0..2)
            .map(|i| {
                xs.iter()
                    .map(|&x| Point3::new(x, f64::from(i), 0.0))
                    .collect()
            })
            .collect();
        NurbsSurface::new(
            1,
            2,
            vec![0.0, 0.0, 1.0, 1.0],
            (0..9).map(f64::from).collect(),
            net,
            vec![vec![1.0; 6]; 2],
        )
        .unwrap()
    }

    #[test]
    fn a_spline_that_is_not_clamped_is_read_over_its_domain() {
        let one_way = unclamped([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(!turns_over_on(&one_way));
        assert_eq!(first_fold(&one_way), None);
        let back = unclamped([0.0, 1.0, 4.0, 2.0, 5.0, 6.0]);
        assert!(turns_over_on(&back));
        let (_, v) = first_fold(&back).expect("a fold");
        assert!((2.0..=6.0).contains(&v), "v = {v}");
    }

    /// As [`turns_over`], over the surface's own domain.
    fn turns_over_on(surface: &NurbsSurface) -> bool {
        let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
        let mut seen = (false, false);
        for i in 0..=10 {
            for j in 0..=800 {
                let u = u0 + (u1 - u0) * f64::from(i) / 10.0;
                let v = v0 + (v1 - v0) * f64::from(j) / 800.0;
                let d = surface.derivatives(u, v, 1);
                let z = d[1][0].cross(d[0][1]).z();
                seen = (seen.0 || z > 1e-9, seen.1 || z < -1e-9);
            }
        }
        seen.0 && seen.1
    }

    /// A quarter of a cylinder as a rational patch has a normal everywhere.
    #[test]
    fn a_rational_quarter_cylinder_does_not_fold() {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let arc = [(1.0, 0.0, 1.0), (1.0, 1.0, w), (0.0, 1.0, 1.0)];
        let net = (0..2)
            .map(|i| {
                arc.iter()
                    .map(|&(x, y, _)| Point3::new(x, y, 3.0 * f64::from(i)))
                    .collect()
            })
            .collect();
        let weights = vec![arc.iter().map(|a| a.2).collect::<Vec<_>>(); 2];
        let quarter = NurbsSurface::new(
            1,
            2,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            net,
            weights,
        )
        .unwrap();
        assert_eq!(first_fold(&quarter), None);
    }

    /// A strip whose edge at u = 1 has shrunk to a point has no normal
    /// there, and no fold.
    #[test]
    fn a_strip_that_closes_to_a_point_does_not_fold() {
        let net = vec![
            (0..4)
                .map(|k| Point3::new(f64::from(k), 0.0, 0.0))
                .collect::<Vec<_>>(),
            vec![Point3::new(1.5, 5.0, 1.0); 4],
        ];
        let closing = NurbsSurface::new(
            1,
            3,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            net,
            vec![vec![1.0; 4]; 2],
        )
        .unwrap();
        assert_eq!(first_fold(&closing), None);
    }
}
