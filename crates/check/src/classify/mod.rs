//! Point-in-solid classification (ray casting + winding numbers).
//!
//! The primary entry point is [`classify_point`], which uses analytic ray
//! casting with UV boundary containment to determine whether a 3D point
//! lies inside, outside, or on the boundary of a B-Rep solid.

pub(crate) mod boundary;
pub(crate) mod ray_surface;
pub(crate) mod winding;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::CheckError;

/// Result of classifying a point relative to a solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointClassification {
    /// The point is inside the solid.
    Inside,
    /// The point is outside the solid.
    Outside,
    /// The point is on the boundary (within tolerance).
    OnBoundary,
}

/// Options controlling the classification algorithm.
#[derive(Debug, Clone)]
pub struct ClassifyOptions {
    /// Distance threshold for "on boundary" detection.
    pub tolerance: f64,
    /// Minimum scalar product for ray quality (default 0.2).
    pub ray_quality_threshold: f64,
    /// Maximum recovery attempts when ray hits face boundary.
    pub max_recovery_attempts: usize,
}

impl Default for ClassifyOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            ray_quality_threshold: 0.2,
            max_recovery_attempts: 10,
        }
    }
}

/// Classify a point relative to a solid using analytic ray casting.
///
/// Uses three irrational ray directions for majority-vote consensus.
/// If the first two agree, the third is skipped. If all three disagree
/// (very rare — indicates grazing rays), perturbed recovery directions
/// are tried.
///
/// # Errors
///
/// Returns an error if the solid or its faces contain invalid topology references.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn classify_point(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    options: &ClassifyOptions,
) -> Result<PointClassification, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    // Boundary check: project point onto each face, check distance.
    if is_on_boundary(topo, shell.faces(), point, options.tolerance)? {
        return Ok(PointClassification::OnBoundary);
    }

    // Three irrational ray directions for majority-vote consensus.
    // If the first two agree, the third breaks no tie and we exit early.
    let base_dirs = [
        Vec3::new(
            0.573_576_436_351_046,
            0.740_535_693_464_567_5,
            0.350_889_803_483_932_2,
        ),
        Vec3::new(
            -0.350_889_803_483_932_2,
            0.573_576_436_351_046,
            0.740_535_693_464_567_5,
        ),
        Vec3::new(
            0.740_535_693_464_567_5,
            -0.350_889_803_483_932_2,
            0.573_576_436_351_046,
        ),
    ];

    let mut inside_votes = 0u32;
    let mut outside_votes = 0u32;

    for &dir in &base_dirs {
        let crossings = count_ray_crossings(topo, shell.faces(), point, dir)?;
        if crossings % 2 == 1 {
            inside_votes += 1;
        } else {
            outside_votes += 1;
        }
        // Early exit: if 2 rays agree, that's the answer.
        if inside_votes >= 2 {
            return Ok(PointClassification::Inside);
        }
        if outside_votes >= 2 {
            return Ok(PointClassification::Outside);
        }
    }

    // All three disagreed (very rare). Try perturbed directions as recovery.
    for attempt in 0..options.max_recovery_attempts {
        // Generate a pseudo-random direction from attempt index using golden ratio.
        let seed = (attempt as f64 + 1.0) * 0.618_033_988_749_895;
        let theta = seed * std::f64::consts::TAU;
        let phi = (seed * std::f64::consts::E).fract() * std::f64::consts::PI;
        let dir = Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());

        let crossings = count_ray_crossings(topo, shell.faces(), point, dir)?;
        if crossings % 2 == 1 {
            inside_votes += 1;
        } else {
            outside_votes += 1;
        }
        let remaining = options.max_recovery_attempts as u32 - attempt as u32;
        if inside_votes > outside_votes + remaining {
            return Ok(PointClassification::Inside);
        }
        if outside_votes > inside_votes + remaining {
            return Ok(PointClassification::Outside);
        }
    }

    // Majority vote from all attempts.
    if inside_votes > outside_votes {
        Ok(PointClassification::Inside)
    } else {
        Ok(PointClassification::Outside)
    }
}

/// Checks if a point is within `tolerance` of any face boundary.
///
/// Uses analytic point-to-surface distance for all surface types, then
/// verifies the projection falls within the face polygon.
fn is_on_boundary(
    topo: &Topology,
    faces: &[FaceId],
    point: Point3,
    tolerance: f64,
) -> Result<bool, CheckError> {
    for &fid in faces {
        let face = topo.face(fid)?;
        let dist = match face.surface() {
            FaceSurface::Plane { normal, d } => {
                let pv = Vec3::new(point.x(), point.y(), point.z());
                (normal.dot(pv) - d).abs()
            }
            FaceSurface::Cylinder(cyl) => {
                let (u, v) = cyl.project_point(point);
                let on_surface = cyl.evaluate(u, v);
                (point - on_surface).length()
            }
            FaceSurface::Cone(cone) => {
                let (u, v) = cone.project_point(point);
                let on_surface = cone.evaluate(u, v);
                (point - on_surface).length()
            }
            FaceSurface::Sphere(sph) => {
                let (u, v) = sph.project_point(point);
                let on_surface = sph.evaluate(u, v);
                (point - on_surface).length()
            }
            FaceSurface::Torus(tor) => {
                let (u, v) = tor.project_point(point);
                let on_surface = tor.evaluate(u, v);
                (point - on_surface).length()
            }
            FaceSurface::Nurbs(nurbs) => {
                match brepkit_math::nurbs::projection::project_point_to_surface(
                    nurbs, point, tolerance,
                ) {
                    Ok(proj) => proj.distance,
                    Err(_) => f64::INFINITY,
                }
            }
        };
        if dist < tolerance {
            // Also check if point projects inside the face boundary.
            let polygon = crate::util::face_polygon(topo, fid)?;
            if polygon.len() >= 3 {
                let normal = boundary::polygon_normal(&polygon);
                if crate::util::point_in_polygon_3d(&point, &polygon, &normal) {
                    return Ok(true);
                }
            } else {
                // Full-surface face (like torus with seam edges only).
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Classify a point relative to a solid using generalized winding numbers.
///
/// More robust than ray casting for imperfect geometry (small gaps,
/// T-junctions). Sums the signed solid angles of triangulated faces and
/// classifies based on the resulting winding number.
///
/// # Errors
///
/// Returns an error if the solid or its faces contain invalid topology references.
pub fn classify_point_winding(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    options: &ClassifyOptions,
) -> Result<PointClassification, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    if is_on_boundary(topo, shell.faces(), point, options.tolerance)? {
        return Ok(PointClassification::OnBoundary);
    }

    let w = winding::winding_number(topo, solid, point)?;
    if w > 0.5 {
        Ok(PointClassification::Inside)
    } else {
        Ok(PointClassification::Outside)
    }
}

/// Robust classification combining winding numbers and ray casting.
///
/// Uses winding numbers first, falling back to ray casting when the
/// winding number is ambiguous (between 0.4 and 0.6). This provides the
/// best accuracy for both clean and imperfect geometry.
///
/// # Errors
///
/// Returns an error if the solid or its faces contain invalid topology references.
pub fn classify_point_robust(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    options: &ClassifyOptions,
) -> Result<PointClassification, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    if is_on_boundary(topo, shell.faces(), point, options.tolerance)? {
        return Ok(PointClassification::OnBoundary);
    }

    let w = winding::winding_number(topo, solid, point)?;
    if w > 0.6 {
        return Ok(PointClassification::Inside);
    }
    if w < 0.4 {
        return Ok(PointClassification::Outside);
    }
    // Ambiguous — fall back to ray casting.
    classify_point(topo, solid, point, options)
}

/// Count total ray crossings across all faces of a shell.
///
/// Builds a BVH over face AABBs to skip faces whose bounding box
/// the ray does not intersect.
fn count_ray_crossings(
    topo: &Topology,
    faces: &[FaceId],
    origin: Point3,
    direction: Vec3,
) -> Result<u32, CheckError> {
    use brepkit_math::bvh::Bvh;

    // Build face AABBs and BVH for broad-phase filtering.
    let face_aabbs: Vec<(usize, brepkit_math::aabb::Aabb3)> = faces
        .iter()
        .enumerate()
        .filter_map(|(i, &fid)| crate::util::face_aabb(topo, fid).ok().map(|aabb| (i, aabb)))
        .collect();
    let bvh = Bvh::build(&face_aabbs);

    // query_ray returns the primitive IDs (the `i` values), which are
    // indices into the original `faces` slice.
    let candidates = bvh.query_ray(origin, direction);

    let mut crossings = 0u32;
    for face_idx in candidates {
        crossings += boundary::count_face_ray_crossings(topo, faces[face_idx], origin, direction)?;
    }
    Ok(crossings)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::winding;
    use super::*;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    #[test]
    fn point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let center = Point3::new(0.5, 0.5, 0.5);
        let opts = ClassifyOptions::default();

        let result = classify_point(&topo, solid, center, &opts).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let far = Point3::new(5.0, 5.0, 5.0);
        let opts = ClassifyOptions::default();

        let result = classify_point(&topo, solid, far, &opts).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_on_boundary_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        // Center of the top face (z=1).
        let on_face = Point3::new(0.5, 0.5, 1.0);
        let opts = ClassifyOptions::default();

        let result = classify_point(&topo, solid, on_face, &opts).unwrap();
        assert_eq!(result, PointClassification::OnBoundary);
    }

    #[test]
    fn point_near_edge_outside() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        // Just outside the box along the x-axis.
        let outside = Point3::new(1.001, 0.5, 0.5);
        let opts = ClassifyOptions::default();

        let result = classify_point(&topo, solid, outside, &opts).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_at_corner_boundary() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        // Very close to a vertex of the box.
        let near_corner = Point3::new(0.0, 0.0, 0.0);
        let opts = ClassifyOptions::default();

        let result = classify_point(&topo, solid, near_corner, &opts).unwrap();
        assert_eq!(result, PointClassification::OnBoundary);
    }

    #[test]
    fn winding_inside_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let center = Point3::new(0.5, 0.5, 0.5);

        let w = winding::winding_number(&topo, solid, center).unwrap();
        assert!(
            w > 0.5,
            "winding number for interior point should be > 0.5, got {w}"
        );
    }

    #[test]
    fn winding_outside_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let far = Point3::new(5.0, 5.0, 5.0);

        let w = winding::winding_number(&topo, solid, far).unwrap();
        assert!(
            w < 0.5,
            "winding number for exterior point should be < 0.5, got {w}"
        );
    }

    #[test]
    fn classify_winding_matches_ray() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let center = Point3::new(0.5, 0.5, 0.5);
        let opts = ClassifyOptions::default();

        let ray_result = classify_point(&topo, solid, center, &opts).unwrap();
        let winding_result = classify_point_winding(&topo, solid, center, &opts).unwrap();
        assert_eq!(ray_result, winding_result);
    }

    #[test]
    fn point_negative_quadrant_outside() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let neg = Point3::new(-1.0, -1.0, -1.0);
        let opts = ClassifyOptions::default();

        let result = classify_point(&topo, solid, neg, &opts).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }
}
