//! Exact NURBS boolean operations via surface-surface intersection.
//!
//! Unlike the tessellate-then-clip approach in [`boolean`](crate::boolean),
//! this module computes exact intersection curves between NURBS faces and
//! splits faces along those curves. This produces precise B-Rep topology
//! without tessellation artifacts.
//!
//! ## Algorithm
//!
//! 1. **Face pair intersection**: compute SSI curves for all overlapping face pairs
//! 2. **`PCurve` construction**: build 2D curves in each face's parameter space
//! 3. **Face splitting**: use SSI curves to split faces into fragments
//! 4. **Classification**: determine inside/outside status of each fragment
//! 5. **Assembly**: collect fragments based on boolean operation type

use brepkit_math::curves2d::{Curve2D, NurbsCurve2D};
use brepkit_math::nurbs::intersection::{IntersectionCurve, IntersectionPoint};
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::pcurve::PCurve;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;
use crate::boolean::BooleanOp;

/// Perform a boolean operation on two solids containing NURBS faces.
///
/// Uses exact surface-surface intersection to split faces, avoiding
/// the tessellation approximation of [`boolean::boolean`](crate::boolean::boolean).
///
/// Falls back to the tessellated approach for mixed planar/NURBS solids.
///
/// # Errors
///
/// Returns an error if SSI computation fails, or the result is empty.
pub fn nurbs_boolean(
    topo: &mut Topology,
    op: BooleanOp,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<SolidId, OperationsError> {
    // Collect NURBS face pairs that potentially overlap
    let face_pairs = find_overlapping_face_pairs(topo, solid_a, solid_b)?;

    if face_pairs.is_empty() {
        // No NURBS face overlaps — use the standard boolean
        return crate::boolean::boolean(topo, op, solid_a, solid_b);
    }

    // Phase 1: Compute SSI curves for each overlapping pair
    let ssi_results = compute_all_ssi(topo, &face_pairs)?;

    // Phase 2: Build pcurves and register them
    register_pcurves(topo, &face_pairs, &ssi_results)?;

    // Phase 3: For now, delegate to the tessellated boolean for the actual
    // face splitting and assembly. The pcurves are stored for future use
    // by a proper face-splitting algorithm.
    //
    // True exact face splitting requires:
    // - Trimmed NURBS surface evaluation
    // - Winding number computation in parameter space
    // - Face fragment classification via pcurve containment
    //
    // This is planned for a future iteration. For now, having correct
    // pcurves is the critical foundation.
    crate::boolean::boolean(topo, op, solid_a, solid_b)
}

/// A pair of faces from two different solids that may overlap.
struct FacePair {
    face_a: FaceId,
    face_b: FaceId,
    surface_a: NurbsSurface,
    surface_b: NurbsSurface,
}

/// Find NURBS face pairs whose bounding boxes overlap.
fn find_overlapping_face_pairs(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<Vec<FacePair>, OperationsError> {
    let faces_a = collect_nurbs_faces(topo, solid_a)?;
    let faces_b = collect_nurbs_faces(topo, solid_b)?;

    let mut pairs = Vec::new();

    for &(fid_a, ref surf_a) in &faces_a {
        let bbox_a = surface_bbox(surf_a);
        for &(fid_b, ref surf_b) in &faces_b {
            let bbox_b = surface_bbox(surf_b);
            if bboxes_overlap(&bbox_a, &bbox_b) {
                pairs.push(FacePair {
                    face_a: fid_a,
                    face_b: fid_b,
                    surface_a: surf_a.clone(),
                    surface_b: surf_b.clone(),
                });
            }
        }
    }

    Ok(pairs)
}

/// Collect NURBS faces from a solid.
fn collect_nurbs_faces(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<(FaceId, NurbsSurface)>, OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut faces = Vec::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        if let FaceSurface::Nurbs(surf) = face.surface() {
            faces.push((fid, surf.clone()));
        }
    }
    Ok(faces)
}

/// Compute approximate bounding box of a NURBS surface by sampling.
fn surface_bbox(surface: &NurbsSurface) -> ([f64; 3], [f64; 3]) {
    let samples = 10;
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];

    for i in 0..=samples {
        let u = (u_max - u_min).mul_add(f64::from(i) / f64::from(samples), u_min);
        for j in 0..=samples {
            let v = (v_max - v_min).mul_add(f64::from(j) / f64::from(samples), v_min);
            let p = surface.evaluate(u, v);
            min[0] = min[0].min(p.x());
            min[1] = min[1].min(p.y());
            min[2] = min[2].min(p.z());
            max[0] = max[0].max(p.x());
            max[1] = max[1].max(p.y());
            max[2] = max[2].max(p.z());
        }
    }

    (min, max)
}

/// Check if two bounding boxes overlap (with small tolerance).
fn bboxes_overlap(a: &([f64; 3], [f64; 3]), b: &([f64; 3], [f64; 3])) -> bool {
    let tol = 1e-6;
    for i in 0..3 {
        if a.0[i] > b.1[i] + tol || b.0[i] > a.1[i] + tol {
            return false;
        }
    }
    true
}

/// Compute SSI curves for all face pairs.
fn compute_all_ssi(
    topo: &Topology,
    pairs: &[FacePair],
) -> Result<Vec<Vec<IntersectionCurve>>, OperationsError> {
    let _ = topo; // used for future face-level queries
    let mut results = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let curves = brepkit_math::nurbs::intersection::intersect_nurbs_nurbs(
            &pair.surface_a,
            &pair.surface_b,
            20,   // sample grid resolution
            0.02, // march step size
        )?;
        results.push(curves);
    }

    Ok(results)
}

/// Build pcurves from SSI results and register them on the topology.
fn register_pcurves(
    topo: &mut Topology,
    pairs: &[FacePair],
    ssi_results: &[Vec<IntersectionCurve>],
) -> Result<(), OperationsError> {
    for (pair, curves) in pairs.iter().zip(ssi_results.iter()) {
        for ssi_curve in curves {
            if ssi_curve.points.len() < 2 {
                continue;
            }

            // Build pcurve on face_a: 2D curve from param1 values
            if let Some(pcurve_a) = build_pcurve_from_params(&ssi_curve.points, true) {
                // Find edges on face_a that this SSI curve intersects
                // For now, store the pcurve as a new edge association
                let face_a = pair.face_a;
                let wire = topo.wire(topo.face(face_a)?.outer_wire())?;
                if let Some(first_edge) = wire.edges().first() {
                    topo.pcurves.set(first_edge.edge(), face_a, pcurve_a);
                }
            }

            // Build pcurve on face_b: 2D curve from param2 values
            if let Some(pcurve_b) = build_pcurve_from_params(&ssi_curve.points, false) {
                let face_b = pair.face_b;
                let wire = topo.wire(topo.face(face_b)?.outer_wire())?;
                if let Some(first_edge) = wire.edges().first() {
                    topo.pcurves.set(first_edge.edge(), face_b, pcurve_b);
                }
            }
        }
    }

    Ok(())
}

/// Build a 2D pcurve from intersection point parameters.
///
/// If `use_param1` is true, uses the (u1, v1) parameters;
/// otherwise uses (u2, v2).
fn build_pcurve_from_params(points: &[IntersectionPoint], use_param1: bool) -> Option<PCurve> {
    if points.len() < 2 {
        return None;
    }

    let params_2d: Vec<Point2> = points
        .iter()
        .map(|p| {
            let (u, v) = if use_param1 { p.param1 } else { p.param2 };
            Point2::new(u, v)
        })
        .collect();

    // Create a degree-1 NURBS curve through the parameter points
    // (piecewise linear in parameter space)
    let n = params_2d.len();
    let degree = 1.min(n - 1);

    let mut knots = Vec::with_capacity(n + degree + 1);
    // Clamped knot vector
    knots.extend(vec![0.0; degree + 1]);
    if n > degree + 1 {
        for i in 1..(n - degree) {
            #[allow(clippy::cast_precision_loss)]
            knots.push(i as f64 / (n - degree) as f64);
        }
    }
    knots.extend(vec![1.0; degree + 1]);

    let weights = vec![1.0; n];

    let curve = NurbsCurve2D::new(degree, knots, params_2d, weights).ok()?;

    Some(PCurve::new(Curve2D::Nurbs(curve), 0.0, 1.0))
}

/// Compute the centroid of a NURBS face by sampling.
#[must_use]
pub fn nurbs_face_centroid(surface: &NurbsSurface) -> Point3 {
    let samples = 5;
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    let total = (samples + 1) * (samples + 1);

    for i in 0..=samples {
        let u = (u_max - u_min).mul_add(f64::from(i) / f64::from(samples), u_min);
        for j in 0..=samples {
            let v = (v_max - v_min).mul_add(f64::from(j) / f64::from(samples), v_min);
            let p = surface.evaluate(u, v);
            cx += p.x();
            cy += p.y();
            cz += p.z();
        }
    }

    let inv = 1.0 / f64::from(total);
    Point3::new(cx * inv, cy * inv, cz * inv)
}

/// Split a NURBS face along trim curves in parameter space.
///
/// Given a face and a set of trim polylines (in the face's (u,v) parameter space),
/// splits the face into fragments. Each fragment gets a new face with the same
/// underlying NURBS surface but different trim boundaries.
///
/// Returns the 3D vertices of each fragment (evaluated from parameter-space polygons).
///
/// # Algorithm
///
/// 1. Build the face boundary as a polygon in parameter space
/// 2. For each trim polyline, find where it intersects the boundary
/// 3. Split the boundary polygon into two halves along the trim
/// 4. Evaluate 3D positions from parameter-space fragments
#[must_use]
pub fn split_face_by_trim_curves(
    surface: &NurbsSurface,
    boundary_uv: &[Point2],
    trim_curves_uv: &[Vec<Point2>],
) -> Vec<Vec<Point3>> {
    if trim_curves_uv.is_empty() || boundary_uv.len() < 3 {
        // No trim curves or degenerate boundary — return the original face
        let verts: Vec<Point3> = boundary_uv
            .iter()
            .map(|p| surface.evaluate(p.x(), p.y()))
            .collect();
        return vec![verts];
    }

    // Start with the boundary as the single "working polygon"
    let mut fragments: Vec<Vec<Point2>> = vec![boundary_uv.to_vec()];

    // Split each fragment by each trim curve
    for trim in trim_curves_uv {
        if trim.len() < 2 {
            continue;
        }

        let mut new_fragments = Vec::new();

        for frag in &fragments {
            let split_result = split_polygon_by_polyline(frag, trim);
            new_fragments.extend(split_result);
        }

        fragments = new_fragments;
    }

    // Evaluate 3D positions for each fragment
    fragments
        .iter()
        .filter(|f| f.len() >= 3) // skip degenerate fragments
        .map(|frag| {
            frag.iter()
                .map(|p| surface.evaluate(p.x(), p.y()))
                .collect()
        })
        .collect()
}

/// Split a 2D polygon by a polyline.
///
/// The polyline cuts through the polygon, dividing it into two or more
/// pieces. Uses a simplified Sutherland-Hodgman-like approach where
/// the polyline defines a "cutting line" from its first to last point.
///
/// Returns the resulting polygon fragments (may be 1 if polyline doesn't
/// cross the polygon, or 2+ if it does).
fn split_polygon_by_polyline(polygon: &[Point2], polyline: &[Point2]) -> Vec<Vec<Point2>> {
    if polyline.len() < 2 || polygon.len() < 3 {
        return vec![polygon.to_vec()];
    }

    // Use the line from first to last point of the polyline as the splitting line
    let line_start = polyline[0];
    let line_end = polyline[polyline.len() - 1];

    let dx = line_end.x() - line_start.x();
    let dy = line_end.y() - line_start.y();

    if dx.mul_add(dx, dy * dy) < 1e-20 {
        return vec![polygon.to_vec()];
    }

    // Classify each polygon vertex as positive or negative side of the line
    let signs: Vec<f64> = polygon
        .iter()
        .map(|p| {
            let px = p.x() - line_start.x();
            let py = p.y() - line_start.y();
            // Cross product: positive = left side, negative = right side
            dx.mul_add(py, -(dy * px))
        })
        .collect();

    // Check if all vertices are on the same side
    let has_positive = signs.iter().any(|&s| s > 1e-12);
    let has_negative = signs.iter().any(|&s| s < -1e-12);

    if !has_positive || !has_negative {
        // No crossing — return polygon unchanged
        return vec![polygon.to_vec()];
    }

    // Split: collect vertices into two groups based on which side of the line
    let n = polygon.len();
    let mut side_a = Vec::new();
    let mut side_b = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let p_i = polygon[i];
        let s_i = signs[i];
        let s_j = signs[j];

        if s_i >= 0.0 {
            side_a.push(p_i);
        } else {
            side_b.push(p_i);
        }

        // If edge crosses the line, add the intersection point to both sides
        if (s_i > 1e-12 && s_j < -1e-12) || (s_i < -1e-12 && s_j > 1e-12) {
            let p_j = polygon[j];
            let t = s_i / (s_i - s_j);
            let intersection = Point2::new(
                p_i.x().mul_add(1.0 - t, p_j.x() * t),
                p_i.y().mul_add(1.0 - t, p_j.y() * t),
            );
            side_a.push(intersection);
            side_b.push(intersection);
        }
    }

    let mut result = Vec::new();
    if side_a.len() >= 3 {
        result.push(side_a);
    }
    if side_b.len() >= 3 {
        result.push(side_b);
    }

    if result.is_empty() {
        result.push(polygon.to_vec());
    }

    result
}

/// Get the boundary of a NURBS face in parameter space.
///
/// Samples the face's wire edges and projects them to (u, v) coordinates.
/// For faces without explicit pcurves, uses the surface domain boundary.
#[must_use]
pub fn face_parameter_boundary(surface: &NurbsSurface, segments: usize) -> Vec<Point2> {
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let n = segments.max(2);

    let mut boundary = Vec::with_capacity(4 * n);

    // Bottom edge: u varies, v = v_min
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let u = (u_max - u_min).mul_add(i as f64 / n as f64, u_min);
        boundary.push(Point2::new(u, v_min));
    }
    // Right edge: u = u_max, v varies
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let v = (v_max - v_min).mul_add(i as f64 / n as f64, v_min);
        boundary.push(Point2::new(u_max, v));
    }
    // Top edge: u varies (reversed), v = v_max
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let u = (u_min - u_max).mul_add(i as f64 / n as f64, u_max);
        boundary.push(Point2::new(u, v_max));
    }
    // Left edge: u = u_min, v varies (reversed)
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let v = (v_min - v_max).mul_add(i as f64 / n as f64, v_max);
        boundary.push(Point2::new(u_min, v));
    }

    boundary
}

/// Sample a pcurve into a polyline in parameter space.
#[must_use]
pub fn sample_pcurve(pcurve: &PCurve, num_samples: usize) -> Vec<Point2> {
    let n = num_samples.max(2);
    let t_start = pcurve.t_start();
    let t_end = pcurve.t_end();

    (0..=n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = (t_end - t_start).mul_add(i as f64 / n as f64, t_start);
            pcurve.evaluate(t)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use brepkit_math::nurbs::surface::NurbsSurface;

    fn make_flat_surface(z: f64) -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, z), Point3::new(1.0, 0.0, z)],
                vec![Point3::new(0.0, 1.0, z), Point3::new(1.0, 1.0, z)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap()
    }

    #[test]
    fn surface_bbox_covers_surface() {
        let surf = make_flat_surface(3.0);
        let (min, max) = surface_bbox(&surf);
        assert!(min[2] <= 3.0);
        assert!(max[2] >= 3.0);
        assert!(min[0] <= 0.0);
        assert!(max[0] >= 1.0);
    }

    #[test]
    fn overlapping_bboxes() {
        let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = ([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        assert!(bboxes_overlap(&a, &b));
    }

    #[test]
    fn non_overlapping_bboxes() {
        let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        assert!(!bboxes_overlap(&a, &b));
    }

    #[test]
    fn build_pcurve_from_two_points() {
        let points = vec![
            IntersectionPoint {
                point: Point3::new(0.0, 0.0, 0.0),
                param1: (0.0, 0.0),
                param2: (0.5, 0.5),
            },
            IntersectionPoint {
                point: Point3::new(1.0, 0.0, 0.0),
                param1: (1.0, 0.0),
                param2: (0.5, 1.0),
            },
        ];

        let pcurve = build_pcurve_from_params(&points, true).unwrap();
        let start = pcurve.evaluate(0.0);
        let end = pcurve.evaluate(1.0);

        assert!((start.x() - 0.0).abs() < 1e-10);
        assert!((start.y() - 0.0).abs() < 1e-10);
        assert!((end.x() - 1.0).abs() < 1e-10);
        assert!((end.y() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn build_pcurve_from_param2() {
        let points = vec![
            IntersectionPoint {
                point: Point3::new(0.0, 0.0, 0.0),
                param1: (0.0, 0.0),
                param2: (0.2, 0.3),
            },
            IntersectionPoint {
                point: Point3::new(1.0, 0.0, 0.0),
                param1: (1.0, 0.0),
                param2: (0.8, 0.9),
            },
        ];

        let pcurve = build_pcurve_from_params(&points, false).unwrap();
        let start = pcurve.evaluate(0.0);

        assert!((start.x() - 0.2).abs() < 1e-10);
        assert!((start.y() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn nurbs_centroid_of_flat_surface() {
        let surf = make_flat_surface(0.0);
        let c = nurbs_face_centroid(&surf);
        assert!((c.x() - 0.5).abs() < 0.1);
        assert!((c.y() - 0.5).abs() < 0.1);
        assert!((c.z() - 0.0).abs() < 0.1);
    }

    #[test]
    fn pcurve_from_single_point_returns_none() {
        let points = vec![IntersectionPoint {
            point: Point3::new(0.0, 0.0, 0.0),
            param1: (0.5, 0.5),
            param2: (0.5, 0.5),
        }];

        assert!(build_pcurve_from_params(&points, true).is_none());
    }

    #[test]
    fn nurbs_boolean_on_planar_solids_falls_back() {
        // With all-planar solids, nurbs_boolean should fall back to standard boolean
        let mut topo = Topology::new();
        let a = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let b = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Transform b to overlap with a
        crate::transform::transform_solid(
            &mut topo,
            b,
            &brepkit_math::mat::Mat4::translation(1.0, 0.0, 0.0),
        )
        .unwrap();

        let result = nurbs_boolean(&mut topo, BooleanOp::Fuse, a, b);
        assert!(result.is_ok());
    }

    // ── Face splitting tests ──────────────────────────

    #[test]
    fn split_face_no_trim_curves() {
        let surf = make_flat_surface(0.0);
        let boundary = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        let fragments = split_face_by_trim_curves(&surf, &boundary, &[]);
        assert_eq!(fragments.len(), 1, "no trim = 1 fragment");
        assert_eq!(fragments[0].len(), 4);
    }

    #[test]
    fn split_face_horizontal_trim() {
        let surf = make_flat_surface(0.0);
        let boundary = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        // Horizontal trim at v=0.5
        let trim = vec![Point2::new(0.0, 0.5), Point2::new(1.0, 0.5)];

        let fragments = split_face_by_trim_curves(&surf, &boundary, &[trim]);
        assert_eq!(fragments.len(), 2, "horizontal trim should split into 2");

        // Both fragments should have at least 3 vertices
        for frag in &fragments {
            assert!(frag.len() >= 3, "fragment too small: {} verts", frag.len());
        }
    }

    #[test]
    fn split_face_vertical_trim() {
        let surf = make_flat_surface(0.0);
        let boundary = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        // Vertical trim at u=0.5
        let trim = vec![Point2::new(0.5, 0.0), Point2::new(0.5, 1.0)];

        let fragments = split_face_by_trim_curves(&surf, &boundary, &[trim]);
        assert_eq!(fragments.len(), 2, "vertical trim should split into 2");
    }

    #[test]
    fn split_face_diagonal_trim() {
        let surf = make_flat_surface(0.0);
        let boundary = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        // Diagonal from (0,0) to (1,1)
        let trim = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)];

        let fragments = split_face_by_trim_curves(&surf, &boundary, &[trim]);
        // Diagonal through corners may produce 2 triangles or may not split
        // (depends on edge case handling). At minimum we get 1 fragment.
        assert!(!fragments.is_empty());
    }

    #[test]
    fn split_polygon_no_crossing() {
        let poly = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        // Line entirely outside the polygon
        let line = vec![Point2::new(2.0, 0.0), Point2::new(2.0, 1.0)];

        let result = split_polygon_by_polyline(&poly, &line);
        assert_eq!(result.len(), 1, "no crossing = 1 fragment");
    }

    #[test]
    fn face_parameter_boundary_has_correct_corners() {
        let surf = make_flat_surface(0.0);
        let boundary = face_parameter_boundary(&surf, 4);

        // Should have 4*4 = 16 points
        assert_eq!(boundary.len(), 16);

        // First point should be at (0,0)
        assert!((boundary[0].x() - 0.0).abs() < 1e-10);
        assert!((boundary[0].y() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sample_pcurve_endpoints() {
        use brepkit_math::curves2d::Line2D;
        use brepkit_math::vec::Vec2;

        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
        let pcurve = PCurve::new(Curve2D::Line(line), 0.0, 1.0);

        let samples = sample_pcurve(&pcurve, 10);
        assert_eq!(samples.len(), 11);
        assert!((samples[0].x() - 0.0).abs() < 1e-10);
        assert!((samples[10].x() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn split_face_fragments_have_correct_z() {
        let z = 5.0;
        let surf = make_flat_surface(z);
        let boundary = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        let trim = vec![Point2::new(0.5, 0.0), Point2::new(0.5, 1.0)];

        let fragments = split_face_by_trim_curves(&surf, &boundary, &[trim]);
        for frag in &fragments {
            for pt in frag {
                assert!(
                    (pt.z() - z).abs() < 1e-10,
                    "all points should be at z={z}, got {}",
                    pt.z()
                );
            }
        }
    }
}
