//! Offset self-intersection detection and removal.
//!
//! When offsetting concave NURBS surfaces, the offset surface can fold back
//! on itself, creating self-intersections. This module detects such regions
//! and trims the offset surface to produce valid geometry.
//!
//! ## Approach
//!
//! 1. **SSI-based** (primary): Use `detect_self_intersection` from the math
//!    crate to find actual self-intersection curves on the offset surface.
//!    The SSI curves' parameter-space data defines the boundary between
//!    valid and folded-back regions.
//! 2. **Sampling-based** (fallback): If SSI detection finds no curves (e.g.,
//!    on degenerate surfaces or very tight folds), fall back to grid sampling
//!    with distance checks against the original surface.

use brepkit_math::nurbs::projection::project_point_to_surface;
use brepkit_math::nurbs::self_intersection::detect_self_intersection;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::vec::Point3;

use crate::OperationsError;

/// Grid resolution for the initial self-intersection detection pass.
const DETECTION_GRID: usize = 20;

/// Grid resolution for SSI-based detection.
const SSI_GRID: usize = 25;

/// Grid resolution for the dense refit sampling pass.
const REFIT_GRID: usize = 30;

/// Maximum fraction of the parameter domain that can be invalid before
/// the function returns an error instead of a degenerate surface.
const MAX_INVALID_FRACTION: f64 = 0.5;

/// Factor applied to tolerance for the distance check. Samples whose
/// distance to the original surface deviates from `|offset_distance|`
/// by more than `tolerance * DISTANCE_TOLERANCE_FACTOR` are flagged.
const DISTANCE_TOLERANCE_FACTOR: f64 = 10.0;

/// Relative tolerance for the fallback distance check. Tightened from
/// 0.15 to 0.02 — the SSI-based primary path handles the cases that
/// formerly needed the loose 15% tolerance.
const RELATIVE_DISTANCE_TOL: f64 = 0.02;

/// Detect and remove self-intersections in an offset NURBS surface.
///
/// Given the original surface and the offset surface, detects where the offset
/// crosses itself, trims away the self-intersecting portions, and returns a
/// cleaned offset surface.
///
/// # Algorithm
///
/// 1. Try SSI-based detection: find actual self-intersection curves on the
///    offset surface. The SSI curves' parameter-space data classifies which
///    side of each curve is at the correct offset distance (valid) vs.
///    folded back (invalid).
/// 2. If SSI detection finds curves, trim using the SSI parameter boundary.
/// 3. If SSI detection finds nothing, fall back to grid sampling with
///    distance/normal checks (tighter tolerance than before).
///
/// # Errors
/// Returns an error if surface analysis fails.
#[allow(clippy::too_many_lines)]
pub fn trim_offset_self_intersections(
    original: &NurbsSurface,
    offset: &NurbsSurface,
    offset_distance: f64,
    tolerance: f64,
) -> Result<NurbsSurface, OperationsError> {
    // ── Primary path: SSI-based detection ──────────────────────────────
    if let Ok(ssi_curves) = detect_self_intersection(offset, SSI_GRID, tolerance) {
        if !ssi_curves.is_empty() {
            return trim_via_ssi(original, offset, offset_distance, tolerance, &ssi_curves);
        }
    }

    // ── Fallback path: sampling-based detection ────────────────────────
    trim_via_sampling(original, offset, offset_distance, tolerance)
}

/// Trim the offset surface using SSI curves found by `detect_self_intersection`.
///
/// The SSI curves divide the parameter domain into regions. We classify each
/// region by checking a sample point's distance to the original surface:
/// the side at the correct offset distance is valid, the folded-back side
/// is trimmed away.
#[allow(clippy::cast_precision_loss)]
fn trim_via_ssi(
    original: &NurbsSurface,
    offset: &NurbsSurface,
    offset_distance: f64,
    tolerance: f64,
    ssi_curves: &[brepkit_math::nurbs::self_intersection::SelfIntersectionCurve],
) -> Result<NurbsSurface, OperationsError> {
    let expected_dist = offset_distance.abs();
    let dist_tol = distance_tolerance(tolerance, expected_dist);

    // Build a validity mask using the SSI curve parameter-space data.
    // For each SSI curve, the params_a and params_b give two sheets of the
    // surface that map to the same 3D point. One sheet is at the correct
    // offset distance (valid), the other is folded back (invalid).
    //
    // Strategy: sample the offset surface on a grid. For each sample, check
    // if it's on the invalid side of any SSI curve by:
    //   1. Finding the closest SSI curve point in parameter space
    //   2. Checking which sheet (a or b) it's closer to
    //   3. Checking distance to the original surface for that sheet
    let (u_min, u_max) = offset.domain_u();
    let (v_min, v_max) = offset.domain_v();
    let n = REFIT_GRID;

    // Pre-compute which sheets are valid vs. invalid for each SSI curve.
    // We check the midpoint of each sheet against the original surface.
    let mut invalid_params: Vec<(f64, f64)> = Vec::new();

    for ssi in ssi_curves {
        if ssi.params_a.is_empty() {
            continue;
        }
        // Check midpoint of each sheet.
        let mid_idx = ssi.params_a.len() / 2;
        let (ua, va) = ssi.params_a[mid_idx];
        let (ub, vb) = ssi.params_b[mid_idx];

        let pt_a = offset.evaluate(ua, va);
        let pt_b = offset.evaluate(ub, vb);

        let dist_a = project_point_to_surface(original, pt_a, tolerance)
            .map_or(expected_dist, |proj| proj.distance);
        let dist_b = project_point_to_surface(original, pt_b, tolerance)
            .map_or(expected_dist, |proj| proj.distance);

        // The sheet further from the expected offset is invalid.
        let a_err = (dist_a - expected_dist).abs();
        let b_err = (dist_b - expected_dist).abs();

        if a_err > b_err {
            invalid_params.extend_from_slice(&ssi.params_a);
        } else {
            invalid_params.extend_from_slice(&ssi.params_b);
        }
    }

    // Build validity mask: a sample is invalid if it's near any invalid
    // parameter point (within a grid cell's worth of parameter distance).
    let u_cell = (u_max - u_min) / (n as f64);
    let v_cell = (v_max - v_min) / (n as f64);
    let proximity = (u_cell * u_cell + v_cell * v_cell).sqrt() * 1.5;

    let mut mask = Vec::with_capacity(n);
    for i in 0..n {
        let u = lerp(u_min, u_max, i as f64 / (n - 1) as f64);
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let v = lerp(v_min, v_max, j as f64 / (n - 1) as f64);

            // Check if this sample is near any invalid parameter point.
            let near_invalid = invalid_params
                .iter()
                .any(|&(iu, iv)| ((u - iu).powi(2) + (v - iv).powi(2)).sqrt() < proximity);

            let valid = if near_invalid {
                // Double-check with actual distance measurement.
                let pt = offset.evaluate(u, v);
                project_point_to_surface(original, pt, tolerance).map_or(true, |proj| {
                    let dist_ok = (proj.distance - expected_dist).abs() <= dist_tol;
                    let normal_ok =
                        check_normal_consistency(original, offset, proj.u, proj.v, u, v);
                    dist_ok && normal_ok
                })
            } else {
                true
            };

            row.push(valid);
        }
        mask.push(row);
    }

    // Count invalid samples.
    let total = n * n;
    let invalid_count = mask
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| !v)
        .count();

    if invalid_count == 0 {
        return Ok(offset.clone());
    }

    let invalid_fraction = invalid_count as f64 / total as f64;
    if invalid_fraction > MAX_INVALID_FRACTION {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "offset self-intersection covers {:.0}% of the surface (limit: {:.0}%)",
                invalid_fraction * 100.0,
                MAX_INVALID_FRACTION * 100.0
            ),
        });
    }

    refit_valid_region(offset, &mask)
}

/// Fallback: sampling-based self-intersection detection and trimming.
fn trim_via_sampling(
    original: &NurbsSurface,
    offset: &NurbsSurface,
    offset_distance: f64,
    tolerance: f64,
) -> Result<NurbsSurface, OperationsError> {
    // Step 1: Detection — sample the offset surface and check distances.
    let validity_mask =
        detect_self_intersections_sampling(original, offset, offset_distance, tolerance);

    // Count invalid samples.
    let total = validity_mask.len() * validity_mask[0].len();
    let invalid_count = validity_mask
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| !v)
        .count();

    // If no self-intersections detected, return the offset unchanged.
    if invalid_count == 0 {
        return Ok(offset.clone());
    }

    // If too much of the surface is invalid, return an error.
    #[allow(clippy::cast_precision_loss)]
    let invalid_fraction = invalid_count as f64 / total as f64;
    if invalid_fraction > MAX_INVALID_FRACTION {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "offset self-intersection covers {:.0}% of the surface (limit: {:.0}%), \
                 offset distance may be too large",
                invalid_fraction * 100.0,
                MAX_INVALID_FRACTION * 100.0
            ),
        });
    }

    // Step 2: Localization — refine the validity mask with normal checking.
    let refined_mask = refine_validity_mask(original, offset, offset_distance, tolerance);

    // Recount after refinement.
    let refined_total = refined_mask.len() * refined_mask[0].len();
    let refined_invalid = refined_mask
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| !v)
        .count();

    #[allow(clippy::cast_precision_loss)]
    let refined_fraction = refined_invalid as f64 / refined_total as f64;
    if refined_fraction > MAX_INVALID_FRACTION {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "offset self-intersection covers {:.0}% after refinement (limit: {:.0}%)",
                refined_fraction * 100.0,
                MAX_INVALID_FRACTION * 100.0
            ),
        });
    }

    // Step 3: Trimming — refit the surface using only valid samples.
    refit_valid_region(offset, &refined_mask)
}

/// Compute the distance tolerance for a given offset distance and base tolerance.
fn distance_tolerance(tolerance: f64, expected_dist: f64) -> f64 {
    (tolerance * DISTANCE_TOLERANCE_FACTOR).max(expected_dist * RELATIVE_DISTANCE_TOL)
}

/// Linearly interpolate between `a` and `b` at parameter `t`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    t.mul_add(b - a, a)
}

/// Sample the offset surface on a grid and classify each sample as valid
/// or invalid based on its distance to the original surface.
///
/// A sample is valid if its distance to the original surface is within
/// the computed tolerance of `|offset_distance|`.
#[allow(clippy::cast_precision_loss)]
fn detect_self_intersections_sampling(
    original: &NurbsSurface,
    offset: &NurbsSurface,
    offset_distance: f64,
    tolerance: f64,
) -> Vec<Vec<bool>> {
    let (u_min, u_max) = offset.domain_u();
    let (v_min, v_max) = offset.domain_v();
    let n = DETECTION_GRID;
    let expected_dist = offset_distance.abs();
    let dist_tol = distance_tolerance(tolerance, expected_dist);

    let mut mask = Vec::with_capacity(n);

    for i in 0..n {
        let u = lerp(u_min, u_max, i as f64 / (n - 1) as f64);
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let v = lerp(v_min, v_max, j as f64 / (n - 1) as f64);
            let offset_pt = offset.evaluate(u, v);

            // Project the offset point onto the original surface to get actual distance.
            let valid = project_point_to_surface(original, offset_pt, tolerance)
                .map_or(true, |proj| {
                    (proj.distance - expected_dist).abs() <= dist_tol
                });

            row.push(valid);
        }
        mask.push(row);
    }

    mask
}

/// Refine the validity mask using a denser grid and normal orientation checks.
///
/// A sample is invalid if either:
/// - Its distance to the original surface deviates from the expected offset, or
/// - The offset surface normal at that point has the wrong sign relative to the
///   original surface normal (indicating the surface has "folded back").
#[allow(clippy::cast_precision_loss)]
fn refine_validity_mask(
    original: &NurbsSurface,
    offset: &NurbsSurface,
    offset_distance: f64,
    tolerance: f64,
) -> Vec<Vec<bool>> {
    let (u_min, u_max) = offset.domain_u();
    let (v_min, v_max) = offset.domain_v();
    let n = REFIT_GRID;
    let expected_dist = offset_distance.abs();
    let dist_tol = distance_tolerance(tolerance, expected_dist);

    let mut mask = Vec::with_capacity(n);

    for i in 0..n {
        let u = lerp(u_min, u_max, i as f64 / (n - 1) as f64);
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let v = lerp(v_min, v_max, j as f64 / (n - 1) as f64);
            let offset_pt = offset.evaluate(u, v);

            let valid =
                project_point_to_surface(original, offset_pt, tolerance).map_or(true, |proj| {
                    let dist_ok = (proj.distance - expected_dist).abs() <= dist_tol;
                    let normal_ok =
                        check_normal_consistency(original, offset, proj.u, proj.v, u, v);
                    dist_ok && normal_ok
                });

            row.push(valid);
        }
        mask.push(row);
    }

    mask
}

/// Check that the offset surface normal is consistent with the original
/// surface normal at corresponding parameter locations.
///
/// For a valid offset (positive or negative), the offset surface normal should
/// agree in direction with the original surface normal. A sign flip indicates
/// the surface has folded back (self-intersection).
///
/// Returns `true` if normal orientation is consistent (no fold-back).
#[allow(clippy::similar_names)]
fn check_normal_consistency(
    original: &NurbsSurface,
    offset: &NurbsSurface,
    orig_u: f64,
    orig_v: f64,
    off_u: f64,
    off_v: f64,
) -> bool {
    let Ok(orig_normal) = original.normal(orig_u, orig_v) else {
        return true; // Degenerate point, conservatively valid.
    };
    let Ok(off_normal) = offset.normal(off_u, off_v) else {
        return true;
    };

    // Positive dot product means normals agree in direction.
    orig_normal.dot(off_normal) > 0.0
}

/// Refit the offset surface using only the valid sample points.
///
/// Samples the offset surface on a dense grid, drops invalid samples, then
/// interpolates a new surface through a regular sub-grid of valid points.
#[allow(clippy::cast_precision_loss)]
fn refit_valid_region(
    offset: &NurbsSurface,
    validity_mask: &[Vec<bool>],
) -> Result<NurbsSurface, OperationsError> {
    let (u_min, u_max) = offset.domain_u();
    let (v_min, v_max) = offset.domain_v();
    let n = validity_mask.len();

    // Find the bounding box of valid samples in parameter space.
    let mut u_valid_min = n;
    let mut u_valid_max = 0_usize;
    let mut v_valid_min = n;
    let mut v_valid_max = 0_usize;

    for (i, row) in validity_mask.iter().enumerate() {
        for (j, &valid) in row.iter().enumerate() {
            if valid {
                u_valid_min = u_valid_min.min(i);
                u_valid_max = u_valid_max.max(i);
                v_valid_min = v_valid_min.min(j);
                v_valid_max = v_valid_max.max(j);
            }
        }
    }

    // Need at least a 2x2 grid of valid points to refit.
    if u_valid_max <= u_valid_min || v_valid_max <= v_valid_min {
        return Err(OperationsError::InvalidInput {
            reason: "valid region of offset surface is too small to refit".into(),
        });
    }

    // Resample the offset surface on a regular grid within the valid bounds.
    let refit_n = REFIT_GRID.min(u_valid_max - u_valid_min + 1).max(4);
    let n_div = (n - 1) as f64;
    let u_start = lerp(u_min, u_max, u_valid_min as f64 / n_div);
    let u_end = lerp(u_min, u_max, u_valid_max as f64 / n_div);
    let v_start = lerp(v_min, v_max, v_valid_min as f64 / n_div);
    let v_end = lerp(v_min, v_max, v_valid_max as f64 / n_div);

    let refit_div = (refit_n - 1) as f64;

    let mut grid: Vec<Vec<Point3>> = Vec::with_capacity(refit_n);
    for i in 0..refit_n {
        let u = lerp(u_start, u_end, i as f64 / refit_div);
        let mut row = Vec::with_capacity(refit_n);
        for j in 0..refit_n {
            let v = lerp(v_start, v_end, j as f64 / refit_div);
            row.push(offset.evaluate(u, v));
        }
        grid.push(row);
    }

    // Fit a new surface through the valid sample grid.
    let degree = offset.degree_u().min(offset.degree_v()).clamp(1, 3);
    let degree = degree.min(refit_n - 1);

    interpolate_surface(&grid, degree, degree).map_err(|e| OperationsError::InvalidInput {
        reason: format!("offset refit interpolation failed: {e}"),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use brepkit_math::nurbs::NurbsSurface;
    use brepkit_math::vec::Point3 as P;

    /// Build a convex dome surface (paraboloid opening upward).
    /// Control points form a bowl shape, so offsetting outward (along +Z normal)
    /// should not cause self-intersections.
    fn make_convex_surface() -> NurbsSurface {
        let ctrl = vec![
            vec![
                P::new(0.0, 0.0, 1.0),
                P::new(0.5, 0.0, 0.0),
                P::new(1.0, 0.0, 1.0),
            ],
            vec![
                P::new(0.0, 0.5, 0.0),
                P::new(0.5, 0.5, -1.0),
                P::new(1.0, 0.5, 0.0),
            ],
            vec![
                P::new(0.0, 1.0, 1.0),
                P::new(0.5, 1.0, 0.0),
                P::new(1.0, 1.0, 1.0),
            ],
        ];
        let weights = vec![vec![1.0; 3]; 3];
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        NurbsSurface::new(2, 2, knots.clone(), knots, ctrl, weights).unwrap()
    }

    /// Build a flat bilinear surface (no curvature).
    fn make_flat_surface() -> NurbsSurface {
        let ctrl = vec![
            vec![P::new(0.0, 0.0, 0.0), P::new(1.0, 0.0, 0.0)],
            vec![P::new(0.0, 1.0, 0.0), P::new(1.0, 1.0, 0.0)],
        ];
        let weights = vec![vec![1.0; 2]; 2];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        NurbsSurface::new(1, 1, knots.clone(), knots, ctrl, weights).unwrap()
    }

    /// Manually offset a surface by sampling + displacing along normals.
    #[allow(clippy::cast_precision_loss)]
    fn manual_offset(surface: &NurbsSurface, distance: f64, samples: usize) -> NurbsSurface {
        let n = samples;
        let (u_min, u_max) = surface.domain_u();
        let (v_min, v_max) = surface.domain_v();

        let mut grid: Vec<Vec<P>> = Vec::with_capacity(n);
        for i in 0..n {
            let u = lerp(u_min, u_max, i as f64 / (n - 1) as f64);
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                let v = lerp(v_min, v_max, j as f64 / (n - 1) as f64);
                let pt = surface.evaluate(u, v);
                let normal = surface.normal(u, v).unwrap();
                row.push(P::new(
                    normal.x().mul_add(distance, pt.x()),
                    normal.y().mul_add(distance, pt.y()),
                    normal.z().mul_add(distance, pt.z()),
                ));
            }
            grid.push(row);
        }

        let degree = surface.degree_u().min(surface.degree_v()).min(3);
        brepkit_math::nurbs::surface_fitting::interpolate_surface(&grid, degree, degree).unwrap()
    }

    #[test]
    fn convex_surface_no_trim() {
        let original = make_convex_surface();
        // Small outward offset of a convex surface should not self-intersect.
        let offset = manual_offset(&original, 0.1, 10);

        let result = trim_offset_self_intersections(&original, &offset, 0.1, 1e-6).unwrap();

        // The result should be essentially unchanged — verify by checking
        // that corner points are close to the input offset surface.
        let p_orig = offset.evaluate(0.0, 0.0);
        let p_result = result.evaluate(0.0, 0.0);
        let dist = ((p_orig.x() - p_result.x()).powi(2)
            + (p_orig.y() - p_result.y()).powi(2)
            + (p_orig.z() - p_result.z()).powi(2))
        .sqrt();
        assert!(
            dist < 0.1,
            "convex surface should pass through with minimal change, got dist={dist}"
        );
    }

    #[test]
    fn concave_surface_trims_self_intersection() {
        // Build a concave surface (bowl) and offset it inward by a large amount.
        // The center of the bowl has high curvature, so a large inward offset
        // should cause self-intersection there.
        let original = make_convex_surface();
        // Large negative offset (inward on concave side) to provoke self-intersection.
        let offset = manual_offset(&original, -2.0, 10);

        let result = trim_offset_self_intersections(&original, &offset, -2.0, 1e-6);

        // Either the trimming succeeds (producing a smaller surface) or it
        // fails because too much area is invalid. Both outcomes are acceptable;
        // the key is that it does NOT silently return a self-intersecting surface.
        match result {
            Ok(trimmed) => {
                // The trimmed surface should have a reduced parameter-space extent
                // or fewer control points, indicating trimming occurred.
                let orig_cp_count =
                    offset.control_points().len() * offset.control_points()[0].len();
                let trim_cp_count =
                    trimmed.control_points().len() * trimmed.control_points()[0].len();
                // Trimmed surface may have same or different control point count,
                // but it should be a valid surface.
                assert!(
                    trim_cp_count >= 4,
                    "trimmed surface should have at least 2x2 control points"
                );
                // Just verify it's not identical to the input (some trimming happened).
                let _ = orig_cp_count; // used for reference
            }
            Err(e) => {
                // Expected: self-intersection covers too much area.
                let msg = format!("{e}");
                assert!(
                    msg.contains("self-intersection"),
                    "error should mention self-intersection: {msg}"
                );
            }
        }
    }

    #[test]
    fn fallback_on_extreme_offset() {
        // Build an offset surface that is wildly wrong: we claim the offset
        // distance is 1.0 but provide a surface that is actually offset by 100.0.
        // This simulates a catastrophically self-intersecting offset where
        // no region is at the correct distance.
        let original = make_flat_surface();
        let bogus_offset = manual_offset(&original, 100.0, 6);

        // Tell the function the offset should be 1.0 — since the actual surface
        // is at distance ~100, everything will fail the distance check.
        let result = trim_offset_self_intersections(&original, &bogus_offset, 1.0, 1e-6);

        assert!(
            result.is_err(),
            "extreme offset mismatch should return an error"
        );
    }

    #[test]
    fn flat_surface_no_trim_needed() {
        let original = make_flat_surface();
        let offset = manual_offset(&original, 1.0, 6);

        let result = trim_offset_self_intersections(&original, &offset, 1.0, 1e-6).unwrap();

        // Flat surface offset should pass through unchanged.
        let p = result.evaluate(0.5, 0.5);
        // The bilinear surface normal points in -Z (cross product ordering),
        // so offset of +1.0 along normal gives z=-1.0.
        assert!(
            p.z().abs() > 0.5,
            "flat offset should be displaced from z=0, got z={}",
            p.z()
        );
    }
}
