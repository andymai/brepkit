//! Curve-surface intersection.

use crate::MathError;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::decompose::{curve_to_bezier_segments, surface_to_bezier_patches};
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

use super::MAX_NEWTON_ITER;

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
/// Decomposes both into Bezier segments/patches, performs AABB filtering,
/// then refines candidates with Newton iteration on the coupled 3x3 system
/// `C(t) - S(u,v) = 0`.
///
/// # Parameters
///
/// - `curve`: The NURBS curve
/// - `surface`: The NURBS surface
/// - `tolerance`: Distance tolerance for convergence (e.g. 1e-7)
///
/// # Errors
///
/// Returns an error if Bezier decomposition fails.
#[allow(clippy::too_many_lines)]
pub fn intersect_curve_surface(
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
                        d.x().mul_add(d.x(), d.y().mul_add(d.y(), d.z() * d.z())) < dedup_sq
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
        let dist_sq = residual.x().mul_add(
            residual.x(),
            residual
                .y()
                .mul_add(residual.y(), residual.z() * residual.z()),
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

        let s_derivs = surface.derivatives(u, v, 1);
        let su = s_derivs[1][0];
        let sv = s_derivs[0][1];

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
    let dist_sq = residual.x().mul_add(
        residual.x(),
        residual
            .y()
            .mul_add(residual.y(), residual.z() * residual.z()),
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
