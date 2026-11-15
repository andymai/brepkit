//! Whether a polynomial spline surface folds: has a place, off its collapsed
//! edges, where its two partials line up and its normal turns over. Proved
//! from the control points, so it shares nothing with the code under test.

use brepkit_math::nurbs::decompose::surface_to_bezier_patches;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};

/// Halvings of a Bezier patch, along each parameter, before a part that
/// still cannot be proved regular counts as folded.
const DEEPEST: u32 = 14;

/// A cross product this small a share of the two vectors' lengths is zero:
/// the vectors are parallel to rounding.
const PARALLEL: f64 = 1e-12;

/// Where a surface folds: the middle of the smallest part of it that could
/// not be proved regular.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fold {
    pub u: f64,
    pub v: f64,
}

/// The first fold of `surface`, or `None` when it is regular everywhere
/// off its collapsed edges. `None` also for a rational surface, which this
/// does not read.
///
/// On one Bezier patch each partial is a blend of the steps between
/// neighbouring control points, with weights that are positive inside the
/// patch. The normal `Su x Sv` is then a blend, with positive weights, of the
/// cross products of each u step with each v step. If none of those points
/// against their sum, the normal does not vanish inside the patch.
///
/// On a side of the patch, only the steps at that side have weight, and at a
/// corner only one step each way. So each side and each corner needs a
/// product of its own that points along the sum, or the normal can vanish on
/// the line two parts share. A side that is an edge of the surface and has
/// shrunk to a point is let off: the surface has no normal there.
///
/// A part that fails is halved and read again. A fold fails at every depth.
pub fn first_fold(surface: &NurbsSurface) -> Option<Fold> {
    if surface
        .weights()
        .iter()
        .flatten()
        .any(|&w| (w - 1.0).abs() > 1e-12)
    {
        return None;
    }
    let (domain_u, domain_v) = (surface.domain_u(), surface.domain_v());
    for patch in surface_to_bezier_patches(surface).unwrap() {
        let net: Vec<Vec<Point3>> = patch.surface.control_points().to_vec();
        let (u, v) = (patch.u_range, patch.v_range);
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

/// `on_edge`: whether the part's sides at its least u, greatest u, least v
/// and greatest v lie on the surface's own edges.
fn fold_in(
    net: &[Vec<Point3>],
    u: (f64, f64),
    v: (f64, f64),
    on_edge: [bool; 4],
    depth: u32,
) -> Option<Fold> {
    if regular(net, on_edge) {
        return None;
    }
    let (mid_u, mid_v) = (0.5 * (u.0 + u.1), 0.5 * (v.0 + v.1));
    if depth == DEEPEST {
        return Some(Fold { u: mid_u, v: mid_v });
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

/// Whether no cross product of a u step with a v step points against the
/// sum of them all, and each side and corner has one that points along it
/// (see [`first_fold`]). A patch that has no such product at all (it has
/// shrunk to a curve or a point) is not regular.
fn regular(net: &[Vec<Point3>], on_edge: [bool; 4]) -> bool {
    let (n, m) = (net.len(), net[0].len());
    // steps_u[i][j] runs from row i to row i + 1 at column j, and
    // steps_v[i][j] from column j to column j + 1 on row i.
    let steps_u: Vec<Vec<Vec3>> = (0..n - 1)
        .map(|i| (0..m).map(|j| net[i + 1][j] - net[i][j]).collect())
        .collect();
    let steps_v: Vec<Vec<Vec3>> = (0..n)
        .map(|i| (0..m - 1).map(|j| net[i][j + 1] - net[i][j]).collect())
        .collect();
    let product = |a: Vec3, b: Vec3| {
        let cross = a.cross(b);
        (cross.length() > PARALLEL * a.length() * b.length()).then_some(cross)
    };
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    let mut crosses = Vec::new();
    for a in steps_u.iter().flatten() {
        for b in steps_v.iter().flatten() {
            if let Some(cross) = product(*a, *b) {
                sum += cross;
                crosses.push(cross);
            }
        }
    }
    let length = sum.length();
    if crosses.is_empty() || length == 0.0 {
        return false;
    }
    if crosses
        .iter()
        .any(|cross| cross.dot(sum) < -PARALLEL * cross.length() * length)
    {
        return false;
    }
    let along = |a: Vec3, b: Vec3| {
        product(a, b).is_some_and(|cross| cross.dot(sum) > PARALLEL * cross.length() * length)
    };
    let any_along = |us: &[Vec3], vs: &[Vec3]| us.iter().any(|a| vs.iter().any(|b| along(*a, *b)));
    let size = net
        .iter()
        .flatten()
        .map(|p| (*p - net[0][0]).length())
        .fold(0.0, f64::max);
    let is_point = |steps: &[Vec3]| steps.iter().all(|s| s.length() <= PARALLEL * size);

    let column =
        |grid: &[Vec<Vec3>], j: usize| -> Vec<Vec3> { grid.iter().map(|r| r[j]).collect() };
    // Each side: the u steps and the v steps that have weight on it, the
    // steps that run along it, and its two corners' steps.
    let sides = [
        (steps_u[0].clone(), steps_v[0].clone(), true),
        (steps_u[n - 2].clone(), steps_v[n - 1].clone(), true),
        (column(&steps_u, 0), column(&steps_v, 0), false),
        (column(&steps_u, m - 1), column(&steps_v, m - 2), false),
    ];
    let mut let_off = [false; 4];
    for (k, (us, vs, runs_along_v)) in sides.iter().enumerate() {
        let run = if *runs_along_v { vs } else { us };
        if on_edge[k] && is_point(run) {
            let_off[k] = true;
        } else if !any_along(us, vs) {
            return false;
        }
    }
    let corners = [
        (steps_u[0][0], steps_v[0][0], 0, 2),
        (steps_u[0][m - 1], steps_v[0][m - 2], 0, 3),
        (steps_u[n - 2][0], steps_v[n - 1][0], 1, 2),
        (steps_u[n - 2][m - 1], steps_v[n - 1][m - 2], 1, 3),
    ];
    corners
        .iter()
        .all(|&(a, b, side_u, side_v)| let_off[side_u] || let_off[side_v] || along(a, b))
}

/// The control points of the two halves of a Bezier curve.
fn halve(points: &[Point3]) -> (Vec<Point3>, Vec<Point3>) {
    let mut work = points.to_vec();
    let mut low = Vec::with_capacity(points.len());
    let mut high = Vec::with_capacity(points.len());
    for level in 0..points.len() {
        let last = points.len() - 1 - level;
        low.push(work[0]);
        high.push(work[last]);
        for k in 0..last {
            work[k] = work[k] + (work[k + 1] - work[k]) * 0.5;
        }
    }
    high.reverse();
    (low, high)
}

/// The two halves of a patch along u, whose rows run along v.
fn halve_rows(net: &[Vec<Point3>]) -> (Vec<Vec<Point3>>, Vec<Vec<Point3>>) {
    let columns = net.first().map_or(0, Vec::len);
    let mut low = vec![Vec::with_capacity(columns); net.len()];
    let mut high = vec![Vec::with_capacity(columns); net.len()];
    for j in 0..columns {
        let column: Vec<Point3> = net.iter().map(|row| row[j]).collect();
        let (a, b) = halve(&column);
        for (i, (pa, pb)) in a.into_iter().zip(b).enumerate() {
            low[i].push(pa);
            high[i].push(pb);
        }
    }
    (low, high)
}
