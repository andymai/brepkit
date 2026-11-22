//! Curve-surface intersection.

use crate::MathError;
use crate::aabb::Aabb3;
use crate::madd::Madd;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::decompose::{curve_to_bezier_segments, surface_to_bezier_patches};
use crate::nurbs::span_index::{Hpoint, box_gap, halve, knot_spans, span_bezier};
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

use super::MAX_NEWTON_ITER;

/// A parameter interval.
type Interval = (f64, f64);

/// A curve-surface intersection hit with parameters on both entities.
#[derive(Debug, Clone, Copy)]
pub struct CurveSurfaceHit {
    /// 3D position of the intersection.
    pub point: Point3,
    /// Parameter on the curve.
    pub t: f64,
    /// Parameters on the surface (u, v).
    pub uv: (f64, f64),
}

/// Find all intersection points between a NURBS curve and a NURBS surface.
///
/// Pairs the curve's knot spans with the surface's by their control boxes,
/// narrows each pair by subdivision, then refines candidates with Newton
/// iteration on the coupled 3x3 system `C(t) - S(u,v) = 0`.
///
/// # Parameters
///
/// - `curve`: The NURBS curve
/// - `surface`: The NURBS surface
/// - `tolerance`: Distance tolerance for convergence (e.g. 1e-7)
///
/// # Errors
///
/// Returns an error if Bezier decomposition fails (on a surface whose spans
/// cannot be indexed).
#[allow(clippy::too_many_lines)]
pub fn intersect_curve_surface(
    curve: &NurbsCurve,
    surface: &NurbsSurface,
    tolerance: f64,
) -> Result<Vec<CurveSurfaceHit>, MathError> {
    const DEPTH: u32 = 12;
    let Some(index) = surface.span_index() else {
        return intersect_curve_surface_patches(curve, surface, tolerance);
    };
    let weights = curve.weights();
    if weights.iter().any(|&w| w <= 0.0 || !w.is_normal()) {
        return intersect_curve_surface_patches(curve, surface, tolerance);
    }
    // Use a generous dedup threshold: many seeds converge to the same root,
    // and we want to collapse them. 1e-6 in 3D space is well below any
    // meaningful geometric distinction at CAD tolerances.
    let dedup_dist = tolerance.max(1e-6) * 10.0;
    let dedup_sq = dedup_dist * dedup_dist;
    let mut hits: Vec<CurveSurfaceHit> = Vec::new();

    // Each curve span against each surface span whose control box meets
    // its own, as Bezier nets, halved in whichever of the curve and the
    // surface's two directions spans the most. Their weights being
    // positive, each piece lies in the box of its net, so a pair whose
    // boxes stand apart holds no hit and is dropped. Newton's method runs
    // from the middles of the pairs left at the finest level. A fitted
    // patch can hold tens of thousands of spans whose control boxes overlap
    // a boundary curve's: seeding every such pair by sampling ran for
    // minutes.
    let cps = curve.control_points();
    let p = curve.degree();
    let (nu, nv) = (surface.degree_u() + 1, surface.degree_v() + 1);
    let mut stack: Vec<NetPair> = Vec::new();
    for (t0, t1, last) in knot_spans(curve.knots(), p, cps.len()) {
        let local: Vec<Hpoint> = (last - p..=last)
            .map(|i| {
                let (q, w) = (cps[i], weights[i]);
                [q.x() * w, q.y() * w, q.z() * w, w]
            })
            .collect();
        let curve_net = span_bezier(&local, curve.knots(), last, (t0, t1));
        let bound = net_box(&curve_net).expanded(dedup_dist);
        for i in index.span_ids_meeting(&bound) {
            let Some((surface_net, us, vs)) = index.bezier_net(surface, i) else {
                continue;
            };
            stack.push(NetPair {
                curve: curve_net.clone(),
                ts: (t0, t1),
                surface: surface_net.to_vec(),
                us,
                vs,
                depth: 0,
            });
            while let Some(pair) = stack.pop() {
                let (curve_box, surface_box) = (net_box(&pair.curve), net_box(&pair.surface));
                if box_gap(&curve_box, &surface_box) > dedup_dist {
                    continue;
                }
                if pair.depth < DEPTH {
                    let depth = pair.depth + 1;
                    let size = |b: &Aabb3| (b.max - b.min).length();
                    let (along_u, along_v) = net_spread(&pair.surface, nu, nv);
                    if size(&curve_box) >= size(&surface_box) {
                        let (lo, hi) = halve(&pair.curve, p + 1, 1, true);
                        let tm = f64::midpoint(pair.ts.0, pair.ts.1);
                        stack.push(NetPair {
                            curve: lo,
                            ts: (pair.ts.0, tm),
                            surface: pair.surface.clone(),
                            depth,
                            ..pair
                        });
                        stack.push(NetPair {
                            curve: hi,
                            ts: (tm, pair.ts.1),
                            depth,
                            ..pair
                        });
                    } else if along_u >= along_v {
                        let (lo, hi) = halve(&pair.surface, nu, nv, true);
                        let um = f64::midpoint(pair.us.0, pair.us.1);
                        stack.push(NetPair {
                            curve: pair.curve.clone(),
                            surface: lo,
                            us: (pair.us.0, um),
                            depth,
                            ..pair
                        });
                        stack.push(NetPair {
                            surface: hi,
                            us: (um, pair.us.1),
                            depth,
                            ..pair
                        });
                    } else {
                        let (lo, hi) = halve(&pair.surface, nu, nv, false);
                        let vm = f64::midpoint(pair.vs.0, pair.vs.1);
                        stack.push(NetPair {
                            curve: pair.curve.clone(),
                            surface: lo,
                            vs: (pair.vs.0, vm),
                            depth,
                            ..pair
                        });
                        stack.push(NetPair {
                            surface: hi,
                            vs: (vm, pair.vs.1),
                            depth,
                            ..pair
                        });
                    }
                    continue;
                }
                let tm = f64::midpoint(pair.ts.0, pair.ts.1);
                let (um, vm) = (
                    f64::midpoint(pair.us.0, pair.us.1),
                    f64::midpoint(pair.vs.0, pair.vs.1),
                );
                if let Some(hit) = refine_curve_surface_point(curve, surface, tm, um, vm, tolerance)
                {
                    let dup = hits.iter().any(|h: &CurveSurfaceHit| {
                        let d = h.point - hit.point;
                        d.x().madd(d.x(), d.y().madd(d.y(), d.z() * d.z())) < dedup_sq
                    });
                    if !dup {
                        hits.push(hit);
                    }
                }
            }
        }
    }

    // Sort by curve parameter for deterministic output.
    hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    Ok(hits)
}

/// A piece of a curve span and a piece of a surface span, as Bezier nets,
/// with their parameter intervals.
struct NetPair {
    curve: Vec<Hpoint>,
    ts: Interval,
    surface: Vec<Hpoint>,
    us: Interval,
    vs: Interval,
    depth: u32,
}

/// The box of a net's points.
fn net_box(net: &[Hpoint]) -> Aabb3 {
    Aabb3::from_points(
        net.iter()
            .map(|h| Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])),
    )
}

/// How far a surface net of `nu` by `nv` points reaches along `u` and along
/// `v`: the longest distance between a line's two end points, each way.
fn net_spread(net: &[Hpoint], nu: usize, nv: usize) -> (f64, f64) {
    let at = |i: usize, j: usize| {
        let h = net[i * nv + j];
        Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
    };
    let along_u = (0..nv)
        .map(|j| (at(nu - 1, j) - at(0, j)).length())
        .fold(0.0, f64::max);
    let along_v = (0..nu)
        .map(|i| (at(i, nv - 1) - at(i, 0)).length())
        .fold(0.0, f64::max);
    (along_u, along_v)
}

/// [`intersect_curve_surface`] by Bezier decomposition, for a surface whose
/// spans cannot be indexed (a weight that is not positive).
fn intersect_curve_surface_patches(
    curve: &NurbsCurve,
    surface: &NurbsSurface,
    tolerance: f64,
) -> Result<Vec<CurveSurfaceHit>, MathError> {
    let segments = curve_to_bezier_segments(curve)?;
    let patches = surface_to_bezier_patches(surface)?;

    let mut hits = Vec::new();
    // Use a generous dedup threshold: many seeds converge to the same root,
    // and we want to collapse them. 1e-6 in 3D space is well below any
    // meaningful geometric distinction at CAD tolerances.
    let dedup_dist = tolerance.max(1e-6) * 10.0;
    let dedup_sq = dedup_dist * dedup_dist;

    for seg in &segments {
        let seg_aabb = seg.aabb();
        let (t_lo, t_hi) = seg.domain();

        for patch in &patches {
            let patch_aabb = patch.aabb();
            if !seg_aabb.intersects(patch_aabb) {
                continue;
            }

            // Sample the curve segment and find seed points where C(t) is
            // close to the surface patch.
            let n_samples = 8;
            let (u0, u1) = patch.u_range;
            let (v0, v1) = patch.v_range;

            for i in 0..=n_samples {
                let frac = i as f64 / n_samples as f64;
                let t = t_lo + (t_hi - t_lo) * frac;
                let pt = seg.evaluate(t);

                // Find closest (u, v) on this patch to pt via a quick
                // grid search over a 3x3 sub-grid of the patch.
                let mut best_u = (u0 + u1) * 0.5;
                let mut best_v = (v0 + v1) * 0.5;
                let mut best_dist = f64::MAX;
                for iu in 0..=2_usize {
                    let uf = u0 + (u1 - u0) * (iu as f64 / 2.0);
                    for iv in 0..=2_usize {
                        let vf = v0 + (v1 - v0) * (iv as f64 / 2.0);
                        let sp = surface.evaluate(uf, vf);
                        let d = (sp - pt).length();
                        if d < best_dist {
                            best_dist = d;
                            best_u = uf;
                            best_v = vf;
                        }
                    }
                }
                let u_guess = best_u;
                let v_guess = best_v;

                if let Some(hit) =
                    refine_curve_surface_point(curve, surface, t, u_guess, v_guess, tolerance)
                {
                    // Deduplicate against existing hits.
                    let dup = hits.iter().any(|h: &CurveSurfaceHit| {
                        let d = h.point - hit.point;
                        d.x().madd(d.x(), d.y().madd(d.y(), d.z() * d.z())) < dedup_sq
                    });
                    if !dup {
                        hits.push(hit);
                    }
                }
            }
        }
    }

    // Sort by curve parameter for deterministic output.
    hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    Ok(hits)
}

/// Refine a curve-surface intersection seed `(t, u, v)` via Newton iteration
/// on the 3x3 system `F(t,u,v) = C(t) - S(u,v) = 0`.
fn refine_curve_surface_point(
    curve: &NurbsCurve,
    surface: &NurbsSurface,
    t_guess: f64,
    u_guess: f64,
    v_guess: f64,
    tolerance: f64,
) -> Option<CurveSurfaceHit> {
    let mut t = t_guess;
    let mut u = u_guess;
    let mut v = v_guess;

    let (t_min, t_max) = curve.domain();
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    for _ in 0..MAX_NEWTON_ITER {
        let c_pt = curve.evaluate(t);
        let s_pt = surface.evaluate(u, v);
        let residual = c_pt - s_pt;
        let dist_sq = residual.x().madd(
            residual.x(),
            residual.y().madd(residual.y(), residual.z() * residual.z()),
        );

        if dist_sq < tolerance * tolerance {
            return Some(CurveSurfaceHit {
                point: c_pt,
                t,
                uv: (u, v),
            });
        }

        // Jacobian columns: dC/dt, -dS/du, -dS/dv
        let c_derivs = curve.derivatives(t, 1);
        let ct = if c_derivs.len() > 1 {
            c_derivs[1]
        } else {
            return None;
        };

        let s_derivs = surface.first_derivatives(u, v);
        let su = s_derivs.1;
        let sv = s_derivs.2;

        // J = [ct | -su | -sv], F = residual (as Vec3)
        // Solve J * [dt, du, dv]^T = -F
        let r = Vec3::new(residual.x(), residual.y(), residual.z());

        // 3x3 matrix: columns are ct, -su, -sv
        // Use Cramer's rule.
        let col0 = ct;
        let col1 = Vec3::new(-su.x(), -su.y(), -su.z());
        let col2 = Vec3::new(-sv.x(), -sv.y(), -sv.z());

        let det = col0.x() * (col1.y() * col2.z() - col1.z() * col2.y())
            - col1.x() * (col0.y() * col2.z() - col0.z() * col2.y())
            + col2.x() * (col0.y() * col1.z() - col0.z() * col1.y());

        if det.abs() < 1e-30 {
            // Singular -- try Tikhonov regularization.
            let lambda = 1e-6;
            let jtj = [
                [col0.dot(col0) + lambda, col0.dot(col1), col0.dot(col2)],
                [col1.dot(col0), col1.dot(col1) + lambda, col1.dot(col2)],
                [col2.dot(col0), col2.dot(col1), col2.dot(col2) + lambda],
            ];
            let jtr = [col0.dot(r), col1.dot(r), col2.dot(r)];
            if let Some((dt, du, dv)) = solve_3x3_cramer(&jtj, &jtr) {
                t -= dt;
                u -= du;
                v -= dv;
            } else {
                break;
            }
        } else {
            let inv_det = 1.0 / det;

            // Replace each column with -r to get Cramer numerators.
            let neg_r = Vec3::new(-r.x(), -r.y(), -r.z());

            let dt = inv_det
                * (neg_r.x() * (col1.y() * col2.z() - col1.z() * col2.y())
                    - col1.x() * (neg_r.y() * col2.z() - neg_r.z() * col2.y())
                    + col2.x() * (neg_r.y() * col1.z() - neg_r.z() * col1.y()));
            let du = inv_det
                * (col0.x() * (neg_r.y() * col2.z() - neg_r.z() * col2.y())
                    - neg_r.x() * (col0.y() * col2.z() - col0.z() * col2.y())
                    + col2.x() * (col0.y() * neg_r.z() - col0.z() * neg_r.y()));
            let dv = inv_det
                * (col0.x() * (col1.y() * neg_r.z() - col1.z() * neg_r.y())
                    - col1.x() * (col0.y() * neg_r.z() - col0.z() * neg_r.y())
                    + neg_r.x() * (col0.y() * col1.z() - col0.z() * col1.y()));

            t += dt;
            u += du;
            v += dv;
        }

        t = t.clamp(t_min, t_max);
        u = u.clamp(u_min, u_max);
        v = v.clamp(v_min, v_max);
    }

    // Final check.
    let c_pt = curve.evaluate(t);
    let s_pt = surface.evaluate(u, v);
    let residual = c_pt - s_pt;
    let dist_sq = residual.x().madd(
        residual.x(),
        residual.y().madd(residual.y(), residual.z() * residual.z()),
    );

    // 10x relaxation: seed points may be imprecise
    if dist_sq < (tolerance * 10.0).powi(2) {
        Some(CurveSurfaceHit {
            point: c_pt,
            t,
            uv: (u, v),
        })
    } else {
        None
    }
}

/// Solve a 3x3 system `A * x = b` using Cramer's rule.
fn solve_3x3_cramer(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<(f64, f64, f64)> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-30 {
        return None;
    }
    let inv = 1.0 / det;

    let x0 = inv
        * (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
            + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]));
    let x1 = inv
        * (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
            - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]));
    let x2 = inv
        * (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
            - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
            + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]));

    Some((x0, x1, x2))
}
