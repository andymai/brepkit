//! SSI marching: tracing intersection curves from seed points, including
//! branch detection, RKF45 adaptive stepping, and singular tangent analysis.

use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

use super::surface_seeding::refine_ssi_point;
use super::{IntersectionPoint, MAX_BRANCHES_PER_DIRECTION};

/// March along an intersection curve, detecting branch points.
///
/// Returns the traced points and any branch seed points found during
/// marching. Branch points are detected when the surface normals become
/// near-parallel (`|n1 x n2|` drops below threshold), confirmed via
/// second-order curvature analysis. At confirmed branch points, new
/// seeds are spawned in divergent directions.
pub(super) fn march_with_branches(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    seed: &IntersectionPoint,
    step_size: f64,
    tolerance: f64,
) -> (Vec<IntersectionPoint>, Vec<IntersectionPoint>) {
    let max_steps = 200;
    let mut branch_seeds: Vec<IntersectionPoint> = Vec::new();

    // March forward, collecting branch points.
    let (forward, fwd_branches) =
        march_direction_with_branches(s1, s2, seed, true, step_size, tolerance, max_steps);
    branch_seeds.extend(fwd_branches);

    // March backward, collecting branch points.
    let (backward, bwd_branches) =
        march_direction_with_branches(s1, s2, seed, false, step_size, tolerance, max_steps);
    branch_seeds.extend(bwd_branches);

    // Combine: backward (reversed) + seed + forward.
    let mut result: Vec<IntersectionPoint> = backward.into_iter().rev().collect();
    result.push(*seed);
    result.extend(forward);

    (result, branch_seeds)
}

/// Detect branch directions at a near-tangential point.
///
/// At a point where `|n1 x n2|` is small (surfaces nearly tangent),
/// sample perturbation directions and find viable SSI continuations
/// that differ from the current march direction by > 30 deg. Each such
/// direction produces a new seed offset slightly from the branch point.
fn find_branch_directions(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    point: &IntersectionPoint,
    current_tangent: Vec3,
    step_size: f64,
    tolerance: f64,
) -> Vec<IntersectionPoint> {
    let eps = step_size * 0.1;
    let (u1, v1) = point.param1;
    let (u2, v2) = point.param2;
    let min_branch_angle = 30.0_f64.to_radians();

    let directions: [(f64, f64); 8] = [
        (eps, 0.0),
        (-eps, 0.0),
        (0.0, eps),
        (0.0, -eps),
        (eps, eps),
        (eps, -eps),
        (-eps, eps),
        (-eps, -eps),
    ];

    let mut branch_seeds = Vec::new();

    for &(du, dv) in &directions {
        let u1p = u1 + du;
        let v1p = v1 + dv;

        if let Some(refined) = refine_ssi_point(s1, s2, u1p, v1p, u2, v2, tolerance) {
            let d = refined.point - point.point;
            let dist = d.length();
            if dist < tolerance {
                continue; // Too close, not a real branch
            }

            if let Ok(dir) = d.normalize() {
                // Check that this direction diverges from the current tangent.
                let cos_angle = dir.dot(current_tangent).abs();
                let angle = cos_angle.acos();
                if angle > min_branch_angle {
                    branch_seeds.push(refined);
                }
            }
        }
    }

    branch_seeds
}

/// March in one direction, detecting branch points where `|n1 x n2|`
/// drops below threshold. Returns traced points and branch seed points.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn march_direction_with_branches(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    seed: &IntersectionPoint,
    forward: bool,
    step_size: f64,
    tolerance: f64,
    max_steps: usize,
) -> (Vec<IntersectionPoint>, Vec<IntersectionPoint>) {
    let traced = march_direction(s1, s2, seed, forward, step_size, tolerance, max_steps);
    let mut branch_seeds: Vec<IntersectionPoint> = Vec::new();

    // Post-process: scan traced points for near-tangential locations.
    let branch_threshold = tolerance * 1000.0;

    for pt in &traced {
        if branch_seeds.len() >= MAX_BRANCHES_PER_DIRECTION {
            break;
        }

        let n1 = match s1.normal(pt.param1.0, pt.param1.1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let n2 = match s2.normal(pt.param2.0, pt.param2.1) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let cross_mag = n1.cross(n2).length();
        if cross_mag >= branch_threshold {
            continue; // Not near-tangential, no branch point
        }

        // Near-tangential point found. Use second-order analysis to confirm.
        let has_branch = {
            let d1 = s1.derivatives(pt.param1.0, pt.param1.1, 2);
            let d2 = s2.derivatives(pt.param2.0, pt.param2.1, 2);
            if d1.len() >= 3 && d1[0].len() >= 3 && d2.len() >= 3 && d2[0].len() >= 3 {
                // Check curvature difference eigenvalues.
                let s1u = d1[1][0];
                let s1v = d1[0][1];
                let n1_raw = s1u.cross(s1v);
                let n1_len = n1_raw.length();
                if n1_len > 1e-12 {
                    let n = n1_raw * (1.0 / n1_len);
                    let l1 = d1[2][0].dot(n);
                    let m1 = d1[1][1].dot(n);
                    let n1c = d1[0][2].dot(n);

                    let s2u = d2[1][0];
                    let s2v = d2[0][1];
                    let n2_raw = s2u.cross(s2v);
                    let n2_len = n2_raw.length();
                    if n2_len > 1e-12 {
                        let n2n = n2_raw * (1.0 / n2_len);
                        let l2 = d2[2][0].dot(n2n);
                        let m2 = d2[1][1].dot(n2n);
                        let n2c = d2[0][2].dot(n2n);

                        let dl = l1 - l2;
                        let dm = m1 - m2;
                        let dn = n1c - n2c;
                        let trace = dl + dn;
                        let det = dl * dn - dm * dm;
                        let disc = trace * trace - 4.0 * det;
                        let disc_sqrt = disc.max(0.0).sqrt();
                        let lambda1 = 0.5 * (trace - disc_sqrt);
                        let lambda2 = 0.5 * (trace + disc_sqrt);

                        // Both eigenvalues near zero indicates a branch point
                        // (the curvature difference vanishes in all directions).
                        lambda1.abs() < 0.1 && lambda2.abs() < 0.1
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };

        if !has_branch {
            continue;
        }

        // Confirmed branch point: find divergent directions.
        let sign = if forward { 1.0 } else { -1.0 };
        let current_tangent = {
            let t = n1.cross(n2);
            t.normalize()
                .ok()
                .map(|t| Vec3::new(t.x() * sign, t.y() * sign, t.z() * sign))
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0))
        };

        let new_seeds = find_branch_directions(s1, s2, pt, current_tangent, step_size, tolerance);
        branch_seeds.extend(new_seeds);
    }

    (traced, branch_seeds)
}

/// March along an intersection curve from a seed point.
///
/// Uses the tangent direction (cross product of surface normals) to step
/// forward, then corrects back to the intersection with Newton.
/// The `step_size` is the *initial* step size. The marcher adapts it based
/// on both RKF45 integration error and geometric curvature (angular
/// deviation between successive tangent vectors). This ensures fine
/// resolution on high-curvature portions and efficient large steps on
/// flat portions.
#[cfg(test)]
pub(super) fn march_intersection(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    seed: &IntersectionPoint,
    step_size: f64,
    tolerance: f64,
) -> Vec<IntersectionPoint> {
    let max_steps = 200;

    // March forward.
    let forward = march_direction(s1, s2, seed, true, step_size, tolerance, max_steps);
    // March backward.
    let backward = march_direction(s1, s2, seed, false, step_size, tolerance, max_steps);

    // Combine: backward (reversed) + seed + forward.
    let mut result: Vec<IntersectionPoint> = backward.into_iter().rev().collect();
    result.push(*seed);
    result.extend(forward);

    result
}

/// Compute the SSI tangent in parameter space at the given parameters.
/// Returns `(du1, dv1, du2, dv2)` or `None` if normals are degenerate.
#[allow(clippy::similar_names)]
fn ssi_tangent_params(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    u1: f64,
    v1: f64,
    u2: f64,
    v2: f64,
    sign: f64,
) -> Option<[f64; 4]> {
    let n1 = s1.normal(u1, v1).ok()?;
    let n2 = s2.normal(u2, v2).ok()?;

    let tangent_raw = n1.cross(n2);
    let tangent = if let Ok(t) = tangent_raw.normalize() {
        t
    } else {
        // Tangential intersection (normals parallel/antiparallel).
        // Try perturbation analysis to recover a direction.
        let pt = IntersectionPoint {
            point: s1.evaluate(u1, v1),
            param1: (u1, v1),
            param2: (u2, v2),
        };
        singular_tangent_direction(s1, s2, &pt)?
    };

    let t = Vec3::new(tangent.x() * sign, tangent.y() * sign, tangent.z() * sign);

    let d1 = s1.derivatives(u1, v1, 1);
    let d2 = s2.derivatives(u2, v2, 1);

    let (du1, dv1) = project_tangent_to_params(&d1, t, 1.0);
    let (du2, dv2) = project_tangent_to_params(&d2, t, 1.0);

    // Normalize so the maximum component magnitude is 1.0.
    let max_comp = du1.abs().max(dv1.abs()).max(du2.abs()).max(dv2.abs());
    if max_comp < 1e-20 {
        return None;
    }

    Some([
        du1 / max_comp,
        dv1 / max_comp,
        du2 / max_comp,
        dv2 / max_comp,
    ])
}

/// At a singular point (where surface normals are parallel/antiparallel),
/// use perturbation analysis to determine the intersection curve direction.
///
/// Compute the tangent direction at a singular (tangential) intersection point.
///
/// At a tangential point, the first-order tangent `n1 x n2` vanishes because
/// the surface normals are parallel. This function uses second-order curvature
/// analysis to determine the correct marching direction.
///
/// Algorithm (based on Patrikalakis-Maekawa):
/// 1. Compute second derivatives of both surfaces at the touch point
/// 2. Compute the curvature difference tensor in the tangent plane
/// 3. Find the principal direction of the curvature difference --
///    this is the direction along which the surfaces separate fastest
/// 4. The intersection curve follows the direction perpendicular to
///    the maximum curvature difference
///
/// Falls back to perturbation-based search if second-order analysis
/// is degenerate (e.g., surfaces are osculating to second order).
#[allow(clippy::similar_names, clippy::too_many_lines)]
fn singular_tangent_direction(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    point: &IntersectionPoint,
) -> Option<Vec3> {
    let (u1, v1) = point.param1;
    let (u2, v2) = point.param2;

    // Try second-order analysis first.
    if let Some(dir) = second_order_tangent(s1, s2, u1, v1, u2, v2) {
        return Some(dir);
    }

    // Fallback: perturbation-based search (original method).
    perturbation_tangent(s1, s2, point)
}

/// Second-order curvature analysis for tangential intersection direction.
///
/// Computes the difference of the second fundamental forms of the two
/// surfaces at the touch point, projected onto the shared tangent plane.
/// The eigenvector corresponding to the zero (or smallest) eigenvalue of
/// this difference gives the direction along which the surfaces remain
/// in contact -- i.e., the intersection curve tangent.
#[allow(clippy::similar_names)]
pub(super) fn second_order_tangent(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    u1: f64,
    v1: f64,
    u2: f64,
    v2: f64,
) -> Option<Vec3> {
    // Compute second-order derivatives for both surfaces.
    let d1 = s1.derivatives(u1, v1, 2);
    let d2 = s2.derivatives(u2, v2, 2);

    // Check we have enough derivative data.
    if d1.len() < 3 || d1[0].len() < 3 || d2.len() < 3 || d2[0].len() < 3 {
        return None;
    }

    // First derivatives (tangent vectors).
    let s1u = d1[1][0]; // dS1/du1
    let s1v = d1[0][1]; // dS1/dv1

    // Normal of surface 1.
    let n1 = s1u.cross(s1v);
    let n1_len = n1.length();
    if n1_len < 1e-12 {
        return None;
    }
    let n1 = n1 * (1.0 / n1_len);

    // Second fundamental form coefficients of surface 1:
    // L = S_uu * n, M = S_uv * n, N = S_vv * n
    let l1 = d1[2][0].dot(n1);
    let m1 = d1[1][1].dot(n1);
    let n1_coeff = d1[0][2].dot(n1);

    // Second fundamental form coefficients of surface 2:
    let s2u = d2[1][0];
    let s2v = d2[0][1];
    let n2 = s2u.cross(s2v);
    let n2_len = n2.length();
    if n2_len < 1e-12 {
        return None;
    }
    let n2 = n2 * (1.0 / n2_len);

    let l2 = d2[2][0].dot(n2);
    let m2 = d2[1][1].dot(n2);
    let n2_coeff = d2[0][2].dot(n2);

    // Curvature difference: dII = II_1 - II_2
    // In the 2x2 matrix [dL, dM; dM, dN]:
    let dl = l1 - l2;
    let dm = m1 - m2;
    let dn = n1_coeff - n2_coeff;

    // Find eigenvectors of the 2x2 symmetric matrix [dl, dm; dm, dn].
    // The eigenvector with the smaller eigenvalue gives the direction
    // where the curvature difference is minimal -> intersection continues.
    let trace = dl + dn;
    let det = dl * dn - dm * dm;
    let disc = trace * trace - 4.0 * det;

    if disc < -1e-12 {
        return None; // Complex eigenvalues (shouldn't happen for symmetric matrix)
    }
    let disc_sqrt = disc.max(0.0).sqrt();

    let lambda1 = 0.5 * (trace - disc_sqrt);
    let lambda2 = 0.5 * (trace + disc_sqrt);

    // Pick the eigenvector corresponding to the eigenvalue closest to zero.
    let target_lambda = if lambda1.abs() < lambda2.abs() {
        lambda1
    } else {
        lambda2
    };

    // Eigenvector of [dl, dm; dm, dn] for eigenvalue l:
    // (dl - l) x + dm y = 0 -> (x, y) = (-dm, dl - l) or (dn - l, -dm)
    let (ex, ey) = if (dl - target_lambda).abs() > dm.abs() {
        (-dm, dl - target_lambda)
    } else {
        (dn - target_lambda, -dm)
    };

    let e_len = (ex * ex + ey * ey).sqrt();
    if e_len < 1e-12 {
        return None; // Degenerate: curvature difference is isotropic
    }

    // Convert the 2D eigenvector (in parameter space of surface 1) back to 3D.
    // The direction in 3D is: ex * S1_u + ey * S1_v
    let tangent_3d = s1u * (ex / e_len) + s1v * (ey / e_len);

    tangent_3d.normalize().ok()
}

/// Perturbation-based tangent direction finder (fallback).
///
/// Samples 8 directions around the current point in parameter space, attempts
/// Newton refinement at each, and returns the direction to the most distant
/// successfully refined point.
fn perturbation_tangent(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    point: &IntersectionPoint,
) -> Option<Vec3> {
    let eps = 1e-4;
    let (u1, v1) = point.param1;
    let (u2, v2) = point.param2;

    let directions: [(f64, f64); 8] = [
        (eps, 0.0),
        (-eps, 0.0),
        (0.0, eps),
        (0.0, -eps),
        (eps, eps),
        (eps, -eps),
        (-eps, eps),
        (-eps, -eps),
    ];

    let mut best_dir: Option<Vec3> = None;
    let mut best_dist = 0.0_f64;

    for &(du, dv) in &directions {
        let u1p = (u1 + du).clamp(0.001, 0.999);
        let v1p = (v1 + dv).clamp(0.001, 0.999);

        if let Some(refined) = refine_ssi_point(s1, s2, u1p, v1p, u2, v2, 1e-8) {
            let d = refined.point - point.point;
            let dist = d.length();
            if dist > best_dist && dist > 1e-12 {
                if let Ok(normalized) = d.normalize() {
                    best_dist = dist;
                    best_dir = Some(normalized);
                }
            }
        }
    }

    best_dir
}

/// Constrain a parameter value to the domain, wrapping if periodic or
/// clamping if not.
///
/// For periodic parameters, values that exceed the domain are wrapped
/// modulo the period (e.g., `u = 6.5` on a `[0, 2pi]` cylinder wraps
/// to `u ~ 0.217`). For non-periodic parameters, values are clamped
/// with a 0.1% margin to avoid evaluation at exact boundaries.
pub(super) fn constrain_param(v: f64, min: f64, max: f64, periodic: bool) -> f64 {
    if periodic {
        let span = max - min;
        if span <= 0.0 {
            return min;
        }
        let wrapped = min + (v - min).rem_euclid(span);
        // Tiny margin to stay within evaluable domain.
        let margin = 1e-10 * span;
        wrapped.clamp(min + margin, max - margin)
    } else {
        let margin = 0.001 * (max - min);
        v.clamp(min + margin, max - margin)
    }
}

/// Constrain all four parameters to their respective surface domains,
/// wrapping periodic parameters instead of clamping.
pub(super) fn constrain_state(state: &[f64; 4], s1: &NurbsSurface, s2: &NurbsSurface) -> [f64; 4] {
    let (u1_min, u1_max) = s1.domain_u();
    let (v1_min, v1_max) = s1.domain_v();
    let (u2_min, u2_max) = s2.domain_u();
    let (v2_min, v2_max) = s2.domain_v();
    [
        constrain_param(state[0], u1_min, u1_max, s1.is_periodic_u()),
        constrain_param(state[1], v1_min, v1_max, s1.is_periodic_v()),
        constrain_param(state[2], u2_min, u2_max, s2.is_periodic_u()),
        constrain_param(state[3], v2_min, v2_max, s2.is_periodic_v()),
    ]
}

/// Check if a parameter state is at the domain boundary of either surface.
///
/// For periodic parameters, wrapping means we never hit a boundary -- the
/// surface seamlessly continues. Only non-periodic parameters can be "at
/// boundary" (e.g., the v-direction of a cylinder, or any NURBS surface).
fn at_boundary(state: &[f64; 4], s1: &NurbsSurface, s2: &NurbsSurface) -> bool {
    let periodic = [
        s1.is_periodic_u(),
        s1.is_periodic_v(),
        s2.is_periodic_u(),
        s2.is_periodic_v(),
    ];
    let constrained = constrain_state(state, s1, s2);
    state
        .iter()
        .zip(constrained.iter())
        .zip(periodic.iter())
        .any(|((&s, &c), &is_per)| !is_per && (s - c).abs() > f64::EPSILON)
}

/// March in one direction along the intersection curve using RKF45
/// adaptive stepping with closed-loop detection and curvature-based
/// step adaptation.
///
/// Combines two adaptation strategies:
/// - **RKF45 error control**: halves step when integration error exceeds tolerance
/// - **Angular deviation**: halves step when the tangent turns more than 10 deg per
///   step, doubles when less than 2 deg. This ensures fine resolution on
///   high-curvature regions (tight bends) and efficient large steps on
///   straight portions.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn march_direction(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    seed: &IntersectionPoint,
    forward: bool,
    step_size: f64,
    tolerance: f64,
    max_steps: usize,
) -> Vec<IntersectionPoint> {
    // Maximum number of turning points (tangent reversals) to traverse.
    // Realistic SSI curves have at most 2-3 turning points; the limit
    // prevents infinite loops on degenerate near-tangential cases.
    const MAX_TURNING_POINTS: usize = 3;

    let mut points = Vec::new();
    let mut current = *seed;

    let mut sign = if forward { 1.0 } else { -1.0 };
    let mut h = step_size;
    // Minimum step: don't go below 1/1000th of the initial step.
    // Using tolerance (1e-6) as h_min caused runaway step-halving in
    // near-tangential cases -- thousands of tiny steps each requiring
    // full RKF45 + Newton cycles.
    let h_min = (step_size * 1e-3).max(tolerance);
    let max_h = step_size * 4.0;
    // Angular thresholds in radians for curvature adaptation.
    let max_angle = 10.0_f64.to_radians(); // halve step if tangent turns > 10 deg
    let min_angle = 2.0_f64.to_radians(); // double step if tangent turns < 2 deg

    // Track previous 3D tangent for angular deviation.
    let mut prev_tangent: Option<Vec3> = {
        let n1 = s1.normal(seed.param1.0, seed.param1.1).ok();
        let n2 = s2.normal(seed.param2.0, seed.param2.1).ok();
        match (n1, n2) {
            (Some(n1), Some(n2)) => {
                let t = n1.cross(n2);
                t.normalize()
                    .ok()
                    .map(|t| Vec3::new(t.x() * sign, t.y() * sign, t.z() * sign))
            }
            _ => None,
        }
    };

    let mut total_evals = 0_usize;
    let max_evals = max_steps * 3;
    let mut turning_points_count = 0_usize;

    for _ in 0..max_steps {
        let y = [
            current.param1.0,
            current.param1.1,
            current.param2.0,
            current.param2.1,
        ];

        // Try RKF45 with adaptive step, allowing a few retries per accepted step.
        let (y4, accepted_h) = loop {
            total_evals += 1;
            if total_evals > max_evals {
                return points;
            }

            let Some(result) = rkf45_step(s1, s2, &y, h, sign) else {
                return points;
            };

            let (y4, y5) = result;

            // Compute error estimate.
            let err = ((y5[0] - y4[0]).powi(2)
                + (y5[1] - y4[1]).powi(2)
                + (y5[2] - y4[2]).powi(2)
                + (y5[3] - y4[3]).powi(2))
            .sqrt();

            if err > tolerance && h > h_min {
                // Reject step, halve h and retry.
                h = (h * 0.5).max(h_min);
                continue;
            }

            // Accept this step.
            let accepted = h;

            // Adjust h for next step based on integration error.
            if err < tolerance / 10.0 {
                h = (h * 2.0).min(max_h);
            }

            break (y4, accepted);
        };
        let _ = accepted_h;

        // Accept the 4th-order solution (more conservative).
        let next = constrain_state(&y4, s1, s2);

        // Newton-refine to stay on the intersection curve.
        if let Some(refined) =
            refine_ssi_point(s1, s2, next[0], next[1], next[2], next[3], tolerance)
        {
            // Check that we actually moved.
            if (refined.point - current.point).length() < tolerance {
                break;
            }

            // Curvature-based step adaptation: compute tangent at the new
            // point and check angular deviation from the previous tangent.
            let cur_tangent = {
                let n1 = s1.normal(refined.param1.0, refined.param1.1).ok();
                let n2 = s2.normal(refined.param2.0, refined.param2.1).ok();
                match (n1, n2) {
                    (Some(n1), Some(n2)) => {
                        let t = n1.cross(n2);
                        t.normalize()
                            .ok()
                            .map(|t| Vec3::new(t.x() * sign, t.y() * sign, t.z() * sign))
                    }
                    _ => None,
                }
            };

            if let (Some(prev_t), Some(cur_t)) = (prev_tangent, cur_tangent) {
                let cos_angle = prev_t.dot(cur_t).clamp(-1.0, 1.0);
                let angle = cos_angle.acos();

                if cos_angle < 0.0 {
                    // Turning point detected: tangent reversed direction.
                    // This happens when the intersection curve has a cusp
                    // or reversal in parameter space. Use bisection to
                    // locate the turning point precisely, then continue
                    // marching through it by flipping the sign.
                    if h > h_min * 4.0 {
                        // Retry with much smaller step to get closer to the
                        // turning point before continuing.
                        h = (h * 0.25).max(h_min);
                        // Don't add this point; we'll re-step.
                        continue;
                    }
                    // At minimum step: accept the turning point and
                    // continue marching past it. The tangent has reversed,
                    // so flip the sign to keep following the curve in the
                    // new direction.
                    points.push(refined);
                    current = refined;
                    sign = -sign;
                    // cur_tangent was computed with the old sign. After
                    // flipping, negate it so the angular deviation check
                    // on the next step sees a consistent forward direction.
                    prev_tangent = cur_tangent.map(|t| Vec3::new(-t.x(), -t.y(), -t.z()));
                    h = step_size; // Reset step size for the new direction.
                    turning_points_count += 1;
                    if turning_points_count >= MAX_TURNING_POINTS {
                        break;
                    }
                    continue;
                } else if angle > max_angle && h > h_min {
                    // High curvature -- reduce step size for next iteration.
                    h = (h * 0.5).max(h_min);
                } else if angle < min_angle {
                    // Low curvature -- increase step size.
                    h = (h * 2.0).min(max_h);
                }
                // Otherwise keep current step size.
            }

            prev_tangent = cur_tangent;

            // Check boundary.
            let ref_state = [
                refined.param1.0,
                refined.param1.1,
                refined.param2.0,
                refined.param2.1,
            ];
            if at_boundary(&ref_state, s1, s2) {
                points.push(refined);
                break;
            }

            // Closed-loop detection: check if the current 3D point is close
            // to the first traced segment (not just the seed). Using 3D
            // distance instead of 4D parameter distance avoids issues when
            // surfaces have very different parameterization scales.
            if points.len() >= 5 {
                let d_3d = (refined.point - seed.point).length();
                // Also check against the second point to detect crossing
                // the start segment, not just proximity to the seed.
                // 100x: generous loop-closure detection radius to avoid missed closures
                let near_seed = d_3d < tolerance * 100.0;
                let near_first_seg = if points.len() >= 2 {
                    point_to_segment_dist(refined.point, points[0].point, points[1].point)
                        < tolerance * 100.0
                } else {
                    false
                };

                if near_seed || near_first_seg {
                    // Close the loop by adding the seed point.
                    points.push(*seed);
                    break;
                }
            }

            points.push(refined);
            current = refined;
        } else {
            break;
        }
    }

    points
}

/// Perform one RKF45 step. Returns `(y_4th, y_5th)` or `None` if
/// tangent evaluation fails at any stage.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn rkf45_step(
    s1: &NurbsSurface,
    s2: &NurbsSurface,
    y: &[f64; 4],
    h: f64,
    sign: f64,
) -> Option<([f64; 4], [f64; 4])> {
    // Helper: evaluate f at a state, scaling by h.
    let f = |state: &[f64; 4]| -> Option<[f64; 4]> {
        let clamped = constrain_state(state, s1, s2);
        let t = ssi_tangent_params(s1, s2, clamped[0], clamped[1], clamped[2], clamped[3], sign)?;
        Some([t[0] * h, t[1] * h, t[2] * h, t[3] * h])
    };

    // Helper: y + sum of scaled k vectors.
    let add = |base: &[f64; 4], terms: &[(&[f64; 4], f64)]| -> [f64; 4] {
        let mut out = *base;
        for &(k, coeff) in terms {
            for i in 0..4 {
                out[i] += k[i] * coeff;
            }
        }
        out
    };

    let k1 = f(y)?;
    let k2 = f(&add(y, &[(&k1, 1.0 / 4.0)]))?;
    let k3 = f(&add(y, &[(&k1, 3.0 / 32.0), (&k2, 9.0 / 32.0)]))?;
    let k4 = f(&add(
        y,
        &[
            (&k1, 1932.0 / 2197.0),
            (&k2, -7200.0 / 2197.0),
            (&k3, 7296.0 / 2197.0),
        ],
    ))?;
    let k5 = f(&add(
        y,
        &[
            (&k1, 439.0 / 216.0),
            (&k2, -8.0),
            (&k3, 3680.0 / 513.0),
            (&k4, -845.0 / 4104.0),
        ],
    ))?;
    let k6 = f(&add(
        y,
        &[
            (&k1, -8.0 / 27.0),
            (&k2, 2.0),
            (&k3, -3544.0 / 2565.0),
            (&k4, 1859.0 / 4104.0),
            (&k5, -11.0 / 40.0),
        ],
    ))?;

    // 4th-order solution.
    let y4 = add(
        y,
        &[
            (&k1, 25.0 / 216.0),
            (&k3, 1408.0 / 2565.0),
            (&k4, 2197.0 / 4104.0),
            (&k5, -1.0 / 5.0),
        ],
    );

    // 5th-order solution.
    let y5 = add(
        y,
        &[
            (&k1, 16.0 / 135.0),
            (&k3, 6656.0 / 12825.0),
            (&k4, 28561.0 / 56430.0),
            (&k5, -9.0 / 50.0),
            (&k6, 2.0 / 55.0),
        ],
    );

    Some((y4, y5))
}

/// Project a 3D tangent vector onto surface parameter space.
///
/// Given surface derivatives `derivs` (from `surface.derivatives(u, v, 1)`)
/// and a 3D tangent direction scaled by `step`, compute the parameter
/// increments (du, dv) that move along the tangent on the surface.
fn project_tangent_to_params(derivs: &[Vec<Vec3>], tangent: Vec3, step: f64) -> (f64, f64) {
    let su = derivs[1][0]; // dS/du
    let sv = derivs[0][1]; // dS/dv

    let t = Vec3::new(tangent.x() * step, tangent.y() * step, tangent.z() * step);

    // Solve [su*su, su*sv; su*sv, sv*sv] [du; dv] = [su*t; sv*t]
    let a11 = su.dot(su);
    let a12 = su.dot(sv);
    let a22 = sv.dot(sv);
    let b1 = su.dot(t);
    let b2 = sv.dot(t);

    let det = a11.mul_add(a22, -(a12 * a12));
    if det.abs() < 1e-20 {
        return (0.0, 0.0);
    }

    let du = b1.mul_add(a22, -(b2 * a12)) / det;
    let dv = a11.mul_add(b2, -(a12 * b1)) / det;

    (du, dv)
}

/// Compute a Newton step to move (u, v) on the surface closer to a target
/// 3D point. Solves the 2x2 system from the surface's first derivatives.
pub(super) fn surface_newton_step(
    surface: &NurbsSurface,
    u: f64,
    v: f64,
    target: Point3,
) -> (f64, f64) {
    let pt = surface.evaluate(u, v);
    let r = target - pt;
    let r_vec = Vec3::new(r.x(), r.y(), r.z());

    let derivs = surface.derivatives(u, v, 1);
    let su = derivs[1][0];
    let sv = derivs[0][1];

    // Solve: [su*su, su*sv; su*sv, sv*sv] [du; dv] = [su*r; sv*r]
    let a11 = su.dot(su);
    let a12 = su.dot(sv);
    let a22 = sv.dot(sv);
    let b1 = su.dot(r_vec);
    let b2 = sv.dot(r_vec);

    let det = a11.mul_add(a22, -(a12 * a12));

    // Relative singularity threshold scales with the Jacobian magnitude,
    // catching singularities near surface poles/apex where derivatives
    // shrink toward zero (absolute 1e-20 would be too lenient there).
    if det.abs() < (a11 + a22).max(1e-30) * 1e-12 {
        // Near-degenerate Jacobian -- surface singularity (pole, apex, seam).
        // Apply Tikhonov regularization: add lI to the normal equations.
        // This biases toward smaller steps, preventing divergence.
        let lambda = (a11 + a22).max(1e-10) * 1e-4;
        let a11r = a11 + lambda;
        let a22r = a22 + lambda;
        let det_r = a11r.mul_add(a22r, -(a12 * a12));
        if det_r.abs() < 1e-30 {
            // Still degenerate -- try stepping along whichever derivative is non-zero.
            let su_len = su.dot(su);
            let sv_len = sv.dot(sv);
            if su_len > 1e-30 {
                return (b1 / su_len, 0.0);
            }
            if sv_len > 1e-30 {
                return (0.0, b2 / sv_len);
            }
            return (0.0, 0.0);
        }
        let du = b1.mul_add(a22r, -(b2 * a12)) / det_r;
        let dv = a11r.mul_add(b2, -(a12 * b1)) / det_r;
        return (du, dv);
    }

    let du = b1.mul_add(a22, -(b2 * a12)) / det;
    let dv = a11.mul_add(b2, -(a12 * b1)) / det;

    (du, dv)
}

/// Minimum distance from point `p` to the line segment `a`-`b`.
pub(super) fn point_to_segment_dist(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let ap = p - a;
    let len_sq = ab.dot(ab);
    if len_sq < 1e-30 {
        return ap.length();
    }
    let t = (ap.dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = Point3::new(a.x() + t * ab.x(), a.y() + t * ab.y(), a.z() + t * ab.z());
    (p - proj).length()
}

/// Check if a point is near any polyline segment in traced curves.
///
/// Uses segment distance (not point distance) to avoid false positives:
/// a seed near the *middle* of a traced curve won't be rejected just
/// because it's close to an interior point -- it must be within `dist`
/// of the actual polyline path. This prevents discarding seeds that
/// could reach a different branch.
pub(super) fn near_existing_segment(
    segments: &[Vec<IntersectionPoint>],
    point: &IntersectionPoint,
    dist: f64,
) -> bool {
    for seg in segments {
        if seg.len() < 2 {
            // Single-point segment: fallback to point distance.
            if let Some(p) = seg.first() {
                if (p.point - point.point).length() < dist {
                    return true;
                }
            }
            continue;
        }
        for w in seg.windows(2) {
            if point_to_segment_dist(point.point, w[0].point, w[1].point) < dist {
                return true;
            }
        }
    }
    false
}
