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
/// against their sum, the normal does not vanish inside the patch. A part
/// that fails is halved and read again. A fold fails at every depth.
pub fn first_fold(surface: &NurbsSurface) -> Option<Fold> {
    if surface
        .weights()
        .iter()
        .flatten()
        .any(|&w| (w - 1.0).abs() > 1e-12)
    {
        return None;
    }
    for patch in surface_to_bezier_patches(surface).unwrap() {
        let net: Vec<Vec<Point3>> = patch.surface.control_points().to_vec();
        if let Some(fold) = fold_in(&net, patch.u_range, patch.v_range, 0) {
            return Some(fold);
        }
    }
    None
}

fn fold_in(net: &[Vec<Point3>], u: (f64, f64), v: (f64, f64), depth: u32) -> Option<Fold> {
    if regular(net) {
        return None;
    }
    let (mid_u, mid_v) = (0.5 * (u.0 + u.1), 0.5 * (v.0 + v.1));
    if depth == DEEPEST {
        return Some(Fold { u: mid_u, v: mid_v });
    }
    let (low, high) = halve_rows(net);
    for (half, half_u) in [(low, (u.0, mid_u)), (high, (mid_u, u.1))] {
        let (left, right): (Vec<_>, Vec<_>) = half.iter().map(|row| halve(row)).unzip();
        for (quarter, quarter_v) in [(left, (v.0, mid_v)), (right, (mid_v, v.1))] {
            if let Some(fold) = fold_in(&quarter, half_u, quarter_v, depth + 1) {
                return Some(fold);
            }
        }
    }
    None
}

/// Whether no cross product of a u step with a v step points against the
/// sum of them all. A patch that has no such product at all (it has shrunk
/// to a curve or a point) is not regular.
fn regular(net: &[Vec<Point3>]) -> bool {
    let steps_u: Vec<Vec3> = net
        .windows(2)
        .flat_map(|rows| rows[0].iter().zip(&rows[1]).map(|(a, b)| *b - *a))
        .collect();
    let steps_v: Vec<Vec3> = net
        .iter()
        .flat_map(|row| row.windows(2).map(|p| p[1] - p[0]))
        .collect();
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    let mut crosses = Vec::with_capacity(steps_u.len() * steps_v.len());
    for a in &steps_u {
        for b in &steps_v {
            let cross = a.cross(*b);
            if cross.length() > PARALLEL * a.length() * b.length() {
                sum += cross;
                crosses.push(cross);
            }
        }
    }
    let length = sum.length();
    if crosses.is_empty() || length == 0.0 {
        return false;
    }
    crosses
        .iter()
        .all(|cross| cross.dot(sum) >= -PARALLEL * cross.length() * length)
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
