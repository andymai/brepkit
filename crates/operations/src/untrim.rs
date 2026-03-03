//! NURBS face untrimming: convert a trimmed NURBS face to an untrimmed
//! tensor-product patch.
//!
//! A trimmed NURBS face has an underlying surface that extends beyond its
//! trim boundary. Untrimming creates a new surface whose natural domain
//! matches the trimmed region, eliminating the need for trim curves.
//!
//! The approach follows Massarwi & Elber (2018), simplified to a
//! sampling-based pipeline:
//! 1. Sample the trim boundary in parameter space
//! 2. Compute the parameter-space bounding box
//! 3. Build a regular grid inside the trim loop, evaluating the original
//!    surface at each interior point
//! 4. Fit a new NURBS surface through the sampled grid

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::option_if_let_else,
    clippy::tuple_array_conversions
)]

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::nurbs::surface_fitting::{approximate_surface_lspia, interpolate_surface};
use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::OperationsError;

/// A trim curve in the parameter space of a surface.
///
/// Represented as a polyline approximation of the 2D curve in (u, v) space.
#[derive(Debug, Clone)]
pub struct TrimCurve {
    /// The 2D curve in (u, v) parameter space (polyline approximation).
    pub curve: Vec<Point2>,
}

/// Convert a trimmed NURBS face to an untrimmed tensor-product patch.
///
/// Samples the original surface along the trim boundary and interior,
/// then fits a new bounded surface whose domain matches the trimmed region.
/// The result is a surface that doesn't need trim curves.
///
/// # Arguments
/// * `surface` - The underlying NURBS surface
/// * `trim_curves` - Parameter-space trim curves defining the face boundary
/// * `samples_per_curve` - Number of sample points per trim curve segment
/// * `interior_samples` - Grid density for interior sampling (N x N grid)
///
/// # Errors
/// Returns an error if the trim loop is empty or surface fitting fails.
#[allow(clippy::too_many_lines)]
pub fn untrim_face(
    surface: &NurbsSurface,
    trim_curves: &[TrimCurve],
    samples_per_curve: usize,
    interior_samples: usize,
) -> Result<NurbsSurface, OperationsError> {
    // Step 1: Collect all trim boundary points into a single polyline loop.
    let trim_loop = collect_trim_loop(trim_curves, samples_per_curve)?;

    // Step 2: Compute the bounding box in parameter space.
    let (u_min, u_max, v_min, v_max) = param_bounding_box(&trim_loop)?;

    // Step 3: Build a regular grid inside the trim loop.
    let n = interior_samples.max(4);
    #[allow(clippy::cast_precision_loss)]
    let n_f = (n - 1) as f64;

    let mut grid: Vec<Vec<Point3>> = Vec::with_capacity(n);

    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let u = u_min + (u_max - u_min) * (i as f64 / n_f);
        let mut row: Vec<Point3> = Vec::with_capacity(n);

        for j in 0..n {
            let v = v_min + (v_max - v_min) * (j as f64 / n_f);
            let uv = Point2::new(u, v);

            // If the point is inside the trim loop, evaluate the original
            // surface. Otherwise, use the closest boundary point projected
            // through the surface to avoid holes in the fitting grid.
            let pt = if point_in_trim_loop(uv, &trim_loop) {
                surface.evaluate(u, v)
            } else {
                // Project to the nearest boundary point and evaluate there.
                let closest = closest_point_on_polyline(uv, &trim_loop);
                surface.evaluate(closest.x(), closest.y())
            };
            row.push(pt);
        }
        grid.push(row);
    }

    // Step 4: Fit a new NURBS surface through the grid.
    let degree = surface.degree_u().min(surface.degree_v()).clamp(1, 3);
    let grid_rows = grid.len();
    let grid_cols = grid[0].len();
    let fitted = if grid_rows * grid_cols > 100 {
        // Use LSPIA for large grids — better performance.
        approximate_surface_lspia(
            &grid,
            degree,
            degree,
            grid_rows.min(20),
            grid_cols.min(20),
            1e-6,
            50,
        )
        .map_err(|e| OperationsError::InvalidInput {
            reason: format!("untrim LSPIA surface fitting failed: {e}"),
        })?
    } else {
        interpolate_surface(&grid, degree, degree).map_err(|e| OperationsError::InvalidInput {
            reason: format!("untrim surface interpolation failed: {e}"),
        })?
    };

    Ok(fitted)
}

/// Convenience: untrim a face directly from topology.
///
/// Extracts the NURBS surface and PCurves, then calls [`untrim_face`].
///
/// # Errors
/// Returns an error if the face is not NURBS or has no PCurves.
pub fn untrim_topology_face(
    topo: &Topology,
    face: FaceId,
    samples_per_curve: usize,
    interior_samples: usize,
) -> Result<NurbsSurface, OperationsError> {
    let face_ref = topo.face(face)?;
    let surface = match face_ref.surface() {
        FaceSurface::Nurbs(s) => s.clone(),
        _ => {
            return Err(OperationsError::InvalidInput {
                reason: "untrim requires a NURBS face surface".into(),
            });
        }
    };

    // Collect PCurves for all edges on this face.
    let pcurve_entries = topo.pcurves.pcurves_for_face(face);
    if pcurve_entries.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "face has no PCurves; cannot determine trim boundary".into(),
        });
    }

    // Sample each PCurve into a TrimCurve polyline.
    let n_samples = samples_per_curve.max(2);
    let trim_curves: Vec<TrimCurve> = pcurve_entries
        .iter()
        .map(|(_, pc)| {
            let t0 = pc.t_start();
            let t1 = pc.t_end();
            #[allow(clippy::cast_precision_loss)]
            let dt = (t1 - t0) / (n_samples - 1) as f64;
            let curve: Vec<Point2> = (0..n_samples)
                .map(|k| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = t0 + dt * k as f64;
                    pc.evaluate(t)
                })
                .collect();
            TrimCurve { curve }
        })
        .collect();

    untrim_face(&surface, &trim_curves, samples_per_curve, interior_samples)
}

/// Collect all trim curve segments into a single closed polyline loop.
fn collect_trim_loop(
    trim_curves: &[TrimCurve],
    _samples_per_curve: usize,
) -> Result<Vec<Point2>, OperationsError> {
    if trim_curves.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "no trim curves provided".into(),
        });
    }

    let mut loop_pts: Vec<Point2> = Vec::new();
    for tc in trim_curves {
        // Append all points, skipping the first point of subsequent
        // curves to avoid duplicates at junctions.
        if loop_pts.is_empty() {
            loop_pts.extend_from_slice(&tc.curve);
        } else {
            // Skip the first point if it duplicates the last appended point.
            let start_idx = if tc.curve.is_empty() {
                0
            } else {
                let last = loop_pts[loop_pts.len() - 1];
                let first = tc.curve[0];
                let dist = ((last.x() - first.x()).powi(2) + (last.y() - first.y()).powi(2)).sqrt();
                if dist < 1e-10 { 1 } else { 0 }
            };
            if start_idx < tc.curve.len() {
                loop_pts.extend_from_slice(&tc.curve[start_idx..]);
            }
        }
    }

    if loop_pts.len() < 3 {
        return Err(OperationsError::InvalidInput {
            reason: "trim loop has fewer than 3 points".into(),
        });
    }

    Ok(loop_pts)
}

/// Compute the (u, v) bounding box of a set of parameter-space points.
fn param_bounding_box(points: &[Point2]) -> Result<(f64, f64, f64, f64), OperationsError> {
    if points.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "empty point set for bounding box".into(),
        });
    }

    let mut u_min = f64::MAX;
    let mut u_max = f64::MIN;
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;

    for p in points {
        if p.x() < u_min {
            u_min = p.x();
        }
        if p.x() > u_max {
            u_max = p.x();
        }
        if p.y() < v_min {
            v_min = p.y();
        }
        if p.y() > v_max {
            v_max = p.y();
        }
    }

    Ok((u_min, u_max, v_min, v_max))
}

/// Ray-casting point-in-polygon test for 2D parameter space.
///
/// Returns `true` if `point` is inside the closed polyline `trim`.
fn point_in_trim_loop(point: Point2, trim: &[Point2]) -> bool {
    let mut crossings = 0u32;
    let n = trim.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (yi, yj) = (trim[i].y(), trim[j].y());
        if (yi <= point.y() && yj > point.y()) || (yj <= point.y() && yi > point.y()) {
            let t = (point.y() - yi) / (yj - yi);
            let x_cross = trim[i].x() + t * (trim[j].x() - trim[i].x());
            if point.x() < x_cross {
                crossings += 1;
            }
        }
    }
    crossings % 2 == 1
}

/// Find the closest point on a polyline to a given 2D point.
fn closest_point_on_polyline(point: Point2, polyline: &[Point2]) -> Point2 {
    let mut best = polyline[0];
    let mut best_dist = f64::MAX;

    let n = polyline.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let a = polyline[i];
        let b = polyline[j];

        // Project point onto segment [a, b].
        let ab_x = b.x() - a.x();
        let ab_y = b.y() - a.y();
        let len_sq = ab_x.mul_add(ab_x, ab_y * ab_y);

        let candidate = if len_sq < 1e-30 {
            a
        } else {
            let t = ((point.x() - a.x()) * ab_x + (point.y() - a.y()) * ab_y) / len_sq;
            let t_clamped = t.clamp(0.0, 1.0);
            Point2::new(
                ab_x.mul_add(t_clamped, a.x()),
                ab_y.mul_add(t_clamped, a.y()),
            )
        };

        let dx = point.x() - candidate.x();
        let dy = point.y() - candidate.y();
        let dist = dx.mul_add(dx, dy * dy);
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point2, Point3};

    use super::*;

    /// Build a flat bilinear surface on the XY plane (z=0), domain [0,1]x[0,1].
    fn flat_bilinear_surface() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid bilinear surface")
    }

    /// Build a rectangular trim loop matching the full surface domain [0,1]x[0,1].
    fn rectangular_trim() -> Vec<TrimCurve> {
        vec![TrimCurve {
            curve: vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(1.0, 1.0),
                Point2::new(0.0, 1.0),
            ],
        }]
    }

    /// Build a circular trim loop (centered at 0.5, 0.5, radius 0.3).
    fn circular_trim(n_points: usize) -> Vec<TrimCurve> {
        let cx = 0.5;
        let cy = 0.5;
        let r: f64 = 0.3;
        let mut pts = Vec::with_capacity(n_points);
        #[allow(clippy::cast_precision_loss)]
        for k in 0..n_points {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) / (n_points as f64);
            pts.push(Point2::new(
                r.mul_add(angle.cos(), cx),
                r.mul_add(angle.sin(), cy),
            ));
        }
        vec![TrimCurve { curve: pts }]
    }

    // ── Point-in-polygon tests ──────────────────────────────────────────────

    #[test]
    fn pip_inside_square() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(point_in_trim_loop(Point2::new(0.5, 0.5), &square));
    }

    #[test]
    fn pip_outside_square() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(!point_in_trim_loop(Point2::new(2.0, 0.5), &square));
    }

    // ── Untrim tests ────────────────────────────────────────────────────────

    #[test]
    fn untrim_rectangular_trim() {
        let surface = flat_bilinear_surface();
        let trims = rectangular_trim();

        let result = untrim_face(&surface, &trims, 10, 8).unwrap();

        // The fitted surface should closely match the original at interior points.
        let p_orig = surface.evaluate(0.5, 0.5);
        let p_new = result.evaluate(0.5, 0.5);
        let dist = ((p_orig.x() - p_new.x()).powi(2)
            + (p_orig.y() - p_new.y()).powi(2)
            + (p_orig.z() - p_new.z()).powi(2))
        .sqrt();
        assert!(
            dist < 0.1,
            "untrimmed surface should match original at center, dist={dist}"
        );
    }

    #[test]
    fn untrim_circular_trim_on_plane() {
        let surface = flat_bilinear_surface();
        let trims = circular_trim(32);

        let result = untrim_face(&surface, &trims, 10, 10).unwrap();

        // The result should be a valid NURBS surface.
        // Check that evaluating at the center gives approximately z=0
        // (since the original surface is the XY plane).
        let p = result.evaluate(0.5, 0.5);
        assert!(
            p.z().abs() < 0.1,
            "circular trim on flat plane should give z~0 at center, got z={}",
            p.z()
        );
    }

    #[test]
    fn untrim_preserves_geometry() {
        // Use a curved surface to test that geometry is preserved.
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.5)],
                vec![Point3::new(0.0, 1.0, 0.5), Point3::new(1.0, 1.0, 1.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid surface");

        let trims = rectangular_trim();
        let result = untrim_face(&surface, &trims, 10, 10).unwrap();

        // Sample at several interior points and check deviation.
        let test_params = [
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
        ];

        for &(u, v) in &test_params {
            let p_orig = surface.evaluate(u, v);
            let p_new = result.evaluate(u, v);
            let dist = ((p_orig.x() - p_new.x()).powi(2)
                + (p_orig.y() - p_new.y()).powi(2)
                + (p_orig.z() - p_new.z()).powi(2))
            .sqrt();
            assert!(
                dist < 0.2,
                "deviation at ({u}, {v}) = {dist}, expected < 0.2"
            );
        }
    }

    #[test]
    fn untrim_empty_trims_returns_error() {
        let surface = flat_bilinear_surface();
        let result = untrim_face(&surface, &[], 10, 8);
        assert!(result.is_err());
    }

    #[test]
    fn untrim_too_few_points_returns_error() {
        let surface = flat_bilinear_surface();
        let trims = vec![TrimCurve {
            curve: vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
        }];
        let result = untrim_face(&surface, &trims, 10, 8);
        assert!(result.is_err());
    }
}
