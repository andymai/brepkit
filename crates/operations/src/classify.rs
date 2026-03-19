//! Point-in-solid classification via ray casting and generalized winding numbers.
//!
//! Determines whether a 3D point is inside, outside, or on the boundary
//! of a solid.
//!
//! Three classifiers are provided:
//! - [`classify_point`]: analytic ray casting (fast, no tessellation for analytic faces)
//! - [`classify_point_winding`]: generalized winding numbers (robust to gaps, uses tessellation)
//! - [`classify_point_robust`]: winding numbers with ray-casting fallback

use brepkit_math::predicates::point_in_polygon;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use std::f64::consts::PI;

use crate::OperationsError;
use crate::boolean::face_polygon;
use crate::distance::{point_in_polygon_3d, point_to_face_distance};

// ── Tolerance constants ─────────────────────────────────────────────────
// Grouped here so they can be tuned together. These are near-zero guards
// for floating-point arithmetic, NOT geometric tolerance (use `Tolerance`
// struct for that).

/// Near-zero threshold for floating-point denominators and discriminants.
const NEAR_ZERO: f64 = 1e-15;

/// Minimum positive ray parameter to count as a forward hit (avoids self-intersection).
const RAY_T_MIN: f64 = 1e-12;

/// Threshold for half-space sign test (negative side rejection).
const HALF_SPACE_EPS: f64 = 1e-10;

/// Near-zero threshold for degenerate vector length (e.g. polygon normal).
const DEGENERATE_LEN: f64 = 1e-30;

/// Threshold for coincident vertex detection (squared distance).
const COINCIDENT_SQ: f64 = 1e-12;

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

/// Classifies a point relative to a solid using analytic ray casting.
///
/// Shoots a ray from `point` and counts crossings with the solid's
/// boundary faces. Uses direct ray-surface intersection for analytic
/// faces (plane, cylinder, cone, sphere, torus) and tessellation
/// only for NURBS faces.
///
/// `deflection` controls tessellation quality for NURBS faces.
/// `tolerance` is the distance threshold for "on boundary" classification.
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    if is_on_boundary(topo, shell.faces(), point, tolerance)? {
        return Ok(PointClassification::OnBoundary);
    }

    // Two perpendicular irrational ray directions for dual-ray consensus.
    let ray_dirs = [
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
    ];

    let mut inside_votes = 0u32;
    for &dir in &ray_dirs {
        let crossings = count_ray_crossings(topo, shell.faces(), point, dir, deflection)?;
        if crossings % 2 == 1 {
            inside_votes += 1;
        }
    }

    if inside_votes >= 2 {
        Ok(PointClassification::Inside)
    } else {
        Ok(PointClassification::Outside)
    }
}

/// Classifies a point relative to a solid using generalized winding numbers.
///
/// For each triangle on the solid's boundary, computes the signed solid angle
/// subtended at the query point. The sum divided by 4pi gives the winding
/// number: > 0.5 means inside, < 0.5 means outside.
///
/// This method is inherently robust to mesh defects (small gaps, non-manifold
/// edges) because it integrates a continuous function rather than counting
/// discrete crossings.
///
/// `deflection` controls tessellation quality.
/// `tolerance` is the distance threshold for "on boundary" classification.
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point_winding(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    let (winding, on_boundary) = compute_winding_number(topo, solid, point, deflection, tolerance)?;
    if on_boundary {
        return Ok(PointClassification::OnBoundary);
    }
    if winding > 0.5 {
        Ok(PointClassification::Inside)
    } else {
        Ok(PointClassification::Outside)
    }
}

/// Robust point classification combining winding numbers and ray casting.
///
/// Tries generalized winding numbers first (more robust to mesh defects),
/// then falls back to analytic ray casting if the winding number is ambiguous
/// (within 0.1 of the 0.5 threshold).
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point_robust(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    let (winding, on_boundary) = compute_winding_number(topo, solid, point, deflection, tolerance)?;
    if on_boundary {
        return Ok(PointClassification::OnBoundary);
    }

    if winding > 0.6 {
        return Ok(PointClassification::Inside);
    }
    if winding < 0.4 {
        return Ok(PointClassification::Outside);
    }

    // Ambiguous region (0.4..=0.6): fall back to ray casting
    classify_point(topo, solid, point, deflection, tolerance)
}

// ── Boundary check (analytic, no tessellation) ──────────────────────────

/// Checks if a point is within `tolerance` of any face boundary.
///
/// Uses analytic point-to-surface distance for all surface types.
fn is_on_boundary(
    topo: &Topology,
    faces: &[FaceId],
    point: Point3,
    tolerance: f64,
) -> Result<bool, OperationsError> {
    let tol = Tolerance::new();
    for &fid in faces {
        if let Some((dist, _)) = point_to_face_distance(topo, point, fid, tol)? {
            if dist < tolerance {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

// ── Ray crossing (analytic for analytic faces, tessellation for NURBS) ──

/// Counts the number of times a ray crosses the solid's boundary.
fn count_ray_crossings(
    topo: &Topology,
    faces: &[FaceId],
    origin: Point3,
    direction: Vec3,
    deflection: f64,
) -> Result<u32, OperationsError> {
    let mut crossings = 0u32;
    for &fid in faces {
        crossings += count_face_ray_crossings(topo, fid, origin, direction, deflection)?;
    }
    Ok(crossings)
}

/// Count ray crossings for a single face, dispatching by surface type.
#[allow(clippy::too_many_lines)]
fn count_face_ray_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    _deflection: f64,
) -> Result<u32, OperationsError> {
    let face = topo.face(face_id)?;
    match face.surface() {
        FaceSurface::Plane { normal, d } => {
            ray_plane_crossings(topo, face_id, origin, direction, *normal, *d)
        }
        FaceSurface::Cylinder(cyl) => {
            let cyl = cyl.clone();
            let roots = ray_cylinder_roots(origin, direction, &cyl);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| cyl.project_point(p),
                false,
            )
        }
        FaceSurface::Cone(cone) => {
            let cone = cone.clone();
            let roots = ray_cone_roots(origin, direction, &cone);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| cone.project_point(p),
                false,
            )
        }
        FaceSurface::Sphere(sph) => {
            // Sphere boundaries are planar (equator, small circles), so
            // point_in_polygon_3d works. UV projection fails at poles.
            let sph = sph.clone();
            let roots = ray_sphere_roots(origin, direction, &sph);
            count_3d_polygon_crossings(topo, face_id, origin, direction, &roots)
        }
        FaceSurface::Torus(tor) => {
            let tor = tor.clone();
            let roots = ray_torus_roots(origin, direction, &tor);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| tor.project_point(p),
                true,
            )
        }
        FaceSurface::Nurbs(surface) => {
            ray_crossings_nurbs(topo, face_id, origin, direction, surface)
        }
    }
}

/// Ray-plane intersection with point-in-polygon boundary test.
fn ray_plane_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    normal: Vec3,
    d: f64,
) -> Result<u32, OperationsError> {
    let denom = normal.dot(direction);
    if denom.abs() < NEAR_ZERO {
        return Ok(0);
    }

    let t = (d - normal.dot(Vec3::new(origin.x(), origin.y(), origin.z()))) / denom;
    if t <= RAY_T_MIN {
        return Ok(0);
    }

    let hit = origin + direction * t;
    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        return Ok(0);
    }

    if point_in_polygon_3d(&hit, &verts, &normal) {
        Ok(1)
    } else {
        Ok(0)
    }
}

/// Count crossings using 3D polygon containment (for faces with planar boundaries,
/// e.g. sphere hemispheres where UV projection has pole singularities).
///
/// The polygon normal (from Newell's method) indicates which side of the boundary
/// plane the face extends into. A hit point must be on that side AND project
/// inside the boundary polygon.
fn count_3d_polygon_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    roots: &[f64],
) -> Result<u32, OperationsError> {
    if roots.is_empty() {
        return Ok(0);
    }

    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        return Ok(0);
    }
    let mut normal = polygon_normal(&verts);
    // If the face is reversed, the surface normal is flipped — the face
    // extends into the opposite side of the boundary plane.
    let face = topo.face(face_id)?;
    if face.is_reversed() {
        normal = -normal;
    }
    // A reference point on the boundary plane.
    let ref_pt = verts[0];

    let mut crossings = 0u32;
    for &t in roots {
        if t <= RAY_T_MIN {
            continue;
        }
        let hit = origin + direction * t;

        // The hit must be on the face's side of the boundary plane.
        // The polygon normal (from wire winding) points toward the face interior.
        let side = (hit - ref_pt).dot(normal);
        if side < -HALF_SPACE_EPS {
            continue;
        }

        if point_in_polygon_3d(&hit, &verts, &normal) {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Count crossings for analytic (non-planar) faces using UV containment.
///
/// Given ray parameter roots (where the ray hits the infinite surface),
/// checks whether each hit point falls within the face's trimming boundary
/// by projecting to the surface's (u,v) parameter space.
///
/// If the face boundary is degenerate (all vertices coincide, as in a full
/// torus face with seam edges), every positive-t root is counted as a crossing.
fn count_analytic_crossings<F>(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    roots: &[f64],
    project: F,
    v_periodic: bool,
) -> Result<u32, OperationsError>
where
    F: Fn(Point3) -> (f64, f64),
{
    if roots.is_empty() {
        return Ok(0);
    }

    // Build UV boundary polygon from face wire
    let verts = face_polygon(topo, face_id)?;

    // Detect degenerate boundary: a "full-surface" face whose wire has fewer than
    // 3 distinct vertices (e.g. a torus with only seam edges, where all boundary
    // vertices project to the same point). Every positive-t root is a crossing.
    let is_full_surface = verts.len() < 3 || {
        let ref_pt = verts[0];
        verts
            .iter()
            .all(|v| (*v - ref_pt).length_squared() < COINCIDENT_SQ)
    };
    if is_full_surface {
        return Ok(roots.iter().filter(|&&t| t > RAY_T_MIN).count() as u32);
    }

    let uv_boundary = build_uv_boundary(&verts, &project, v_periodic);

    let mut crossings = 0u32;
    for &t in roots {
        if t <= RAY_T_MIN {
            continue;
        }
        let hit = origin + direction * t;
        let (hit_u, hit_v) = project(hit);

        if point_in_uv_boundary(hit_u, hit_v, &uv_boundary, v_periodic) {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Unwrap a step in a periodic (angular) coordinate.
///
/// Given the previous unwrapped value `prev` and the next raw value `next`,
/// returns the next value adjusted so the step lies in `[-PI, PI)`.
/// This keeps a sequence of angular coordinates continuous (no ±TAU jumps).
#[inline]
fn unwrap_angle(prev: f64, next: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let diff = next - prev;
    prev + diff - tau * ((diff + PI) / tau).floor()
}

/// Build a UV boundary polygon from 3D face boundary vertices,
/// with proper unwrapping of periodic coordinates.
///
/// `v_periodic`: whether the v-coordinate is periodic (e.g. torus). Cylinder
/// and cone have linear v (height / distance), so only u is unwrapped for them.
fn build_uv_boundary<F>(verts: &[Point3], project: &F, v_periodic: bool) -> Vec<(f64, f64)>
where
    F: Fn(Point3) -> (f64, f64),
{
    let mut uv: Vec<(f64, f64)> = verts.iter().map(|&p| project(p)).collect();

    for i in 1..uv.len() {
        // u is always periodic (angular coordinate for all analytic surfaces).
        uv[i].0 = unwrap_angle(uv[i - 1].0, uv[i].0);

        // v is periodic only for doubly-periodic surfaces (torus).
        if v_periodic {
            uv[i].1 = unwrap_angle(uv[i - 1].1, uv[i].1);
        }
    }

    uv
}

/// Test if a (u,v) point is inside the UV boundary polygon.
///
/// Adjusts the test point's u coordinate (and v when periodic) to lie within
/// the unwrapped polygon's coordinate range before testing.
fn point_in_uv_boundary(
    hit_u: f64,
    hit_v: f64,
    uv_boundary: &[(f64, f64)],
    v_periodic: bool,
) -> bool {
    // Find the u range of the unwrapped boundary.
    let u_min = uv_boundary
        .iter()
        .map(|(u, _)| *u)
        .fold(f64::INFINITY, f64::min);
    let u_max = uv_boundary
        .iter()
        .map(|(u, _)| *u)
        .fold(f64::NEG_INFINITY, f64::max);
    let u_center = (u_min + u_max) * 0.5;

    // Shift hit_u to be closest to the polygon's u center.
    let hu = unwrap_angle(u_center, hit_u);

    // For doubly-periodic surfaces (torus), also shift hit_v.
    let hv = if v_periodic {
        let v_min = uv_boundary
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::INFINITY, f64::min);
        let v_max = uv_boundary
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        let v_center = (v_min + v_max) * 0.5;
        unwrap_angle(v_center, hit_v)
    } else {
        hit_v
    };

    let poly: Vec<Point2> = uv_boundary
        .iter()
        .map(|(u, v)| Point2::new(*u, *v))
        .collect();
    let test = Point2::new(hu, hv);
    point_in_polygon(test, &poly)
}

// ── Ray-surface intersection roots ──────────────────────────────────────

/// Compute ray-cylinder intersection parameters.
fn ray_cylinder_roots(
    origin: Point3,
    direction: Vec3,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Vec<f64> {
    let ov = origin - cyl.origin();
    let axis = cyl.axis();

    // Project origin and direction onto plane perpendicular to axis.
    let ov_perp = ov - axis * ov.dot(axis);
    let d_perp = direction - axis * direction.dot(axis);

    let a = d_perp.dot(d_perp);
    let b = 2.0 * ov_perp.dot(d_perp);
    let c = ov_perp.dot(ov_perp) - cyl.radius() * cyl.radius();

    solve_quadratic(a, b, c)
}

/// Compute ray-cone intersection parameters.
fn ray_cone_roots(
    origin: Point3,
    direction: Vec3,
    cone: &brepkit_math::surfaces::ConicalSurface,
) -> Vec<f64> {
    let ov = origin - cone.apex();
    let axis = cone.axis();
    let cos_a = cone.half_angle().cos();
    let cos2 = cos_a * cos_a;

    let d_dot_a = direction.dot(axis);
    let ov_dot_a = ov.dot(axis);

    // Cone equation: (P·axis)² cos²θ = |P|² sin²θ
    // Rearranged: (P·axis)² - |P|² tan²θ = 0
    // Or equivalently: (d·a)²·t² + 2(d·a)(ov·a)·t + (ov·a)² - (d·d·t² + 2·ov·d·t + ov·ov)·tan²θ
    // = (cos²θ(d·a)² - (d·d)(1-cos²θ))·t² + ...
    // Simplify: a = cos²(d·a)² - d·d·sin², etc.
    let sin2 = 1.0 - cos2;

    let a = cos2 * d_dot_a * d_dot_a - sin2 * (direction.dot(direction) - d_dot_a * d_dot_a);
    let half_b = cos2 * d_dot_a * ov_dot_a - sin2 * (direction.dot(ov) - d_dot_a * ov_dot_a);
    let c = cos2 * ov_dot_a * ov_dot_a - sin2 * (ov.dot(ov) - ov_dot_a * ov_dot_a);

    solve_quadratic(a, 2.0 * half_b, c)
}

/// Compute ray-sphere intersection parameters.
fn ray_sphere_roots(
    origin: Point3,
    direction: Vec3,
    sph: &brepkit_math::surfaces::SphericalSurface,
) -> Vec<f64> {
    let ov = origin - sph.center();

    let a = direction.dot(direction);
    let b = 2.0 * ov.dot(direction);
    let c = ov.dot(ov) - sph.radius() * sph.radius();

    solve_quadratic(a, b, c)
}

/// Compute ray-torus intersection parameters (quartic).
fn ray_torus_roots(
    origin: Point3,
    direction: Vec3,
    tor: &brepkit_math::surfaces::ToroidalSurface,
) -> Vec<f64> {
    // Transform to torus-local coordinates.
    let z_axis = tor.z_axis();
    let x_axis = tor.x_axis();
    let y_axis = z_axis.cross(x_axis);

    let ov = origin - tor.center();
    let o = Vec3::new(ov.dot(x_axis), ov.dot(y_axis), ov.dot(z_axis));
    let d = Vec3::new(
        direction.dot(x_axis),
        direction.dot(y_axis),
        direction.dot(z_axis),
    );

    let big_r = tor.major_radius();
    let small_r = tor.minor_radius();

    let sum_d_sq = d.dot(d);
    let sum_od = o.dot(d);
    let sum_o_sq = o.dot(o);
    let k = sum_o_sq - small_r * small_r - big_r * big_r;

    let c4 = sum_d_sq * sum_d_sq;
    let c3 = 4.0 * sum_d_sq * sum_od;
    let c2 = 2.0 * sum_d_sq * k + 4.0 * sum_od * sum_od + 4.0 * big_r * big_r * d.z() * d.z();
    let c1 = 4.0 * k * sum_od + 8.0 * big_r * big_r * o.z() * d.z();
    let c0 = k * k - 4.0 * big_r * big_r * (small_r * small_r - o.z() * o.z());

    solve_quartic(c4, c3, c2, c1, c0)
}

/// Count ray crossings for a NURBS face using ray-surface intersection.
///
/// Uses `intersect_line_nurbs` to find ray-surface hits, then tests each
/// hit against the face's UV boundary polygon.
fn ray_crossings_nurbs(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<u32, OperationsError> {
    use brepkit_math::nurbs::intersection::intersect_line_nurbs;

    let hits = intersect_line_nurbs(surface, origin, direction, 20)?;
    if hits.is_empty() {
        return Ok(0);
    }

    // Build UV boundary from face wire vertices.
    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        // Full-surface face — every forward hit is a crossing.
        return Ok(hits
            .iter()
            .filter(|h| {
                let diff = h.point - origin;
                let t = Vec3::new(diff.x(), diff.y(), diff.z()).dot(direction);
                t > RAY_T_MIN
            })
            .count() as u32);
    }

    let project = |p: Point3| -> (f64, f64) { surface.project_point(p) };
    let uv_boundary = build_uv_boundary(&verts, &project, false);

    let mut crossings = 0u32;
    for hit in &hits {
        // Check ray parameter is positive (forward hit).
        let diff = hit.point - origin;
        let t = Vec3::new(diff.x(), diff.y(), diff.z()).dot(direction) / direction.dot(direction);
        if t <= RAY_T_MIN {
            continue;
        }

        // Use the UV parameters from the intersection result.
        let (hit_u, hit_v) = hit.param1;
        if point_in_uv_boundary(hit_u, hit_v, &uv_boundary, false) {
            crossings += 1;
        }
    }

    Ok(crossings)
}

// ── Winding number (still tessellation-based) ───────────────────────────

/// Computes the generalized winding number of a point relative to a solid.
///
/// Returns `(winding_number, is_on_boundary)`.
///
/// Uses ray casting to determine inside/outside classification. Counts
/// total ray crossings across all faces using the same analytic + NURBS
/// dispatch as `count_face_ray_crossings`.
#[allow(clippy::similar_names)]
fn compute_winding_number(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<(f64, bool), OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    if is_on_boundary(topo, shell.faces(), point, tolerance)? {
        return Ok((0.0, true));
    }

    // Use ray casting: count crossings in a fixed direction.
    let direction = Vec3::new(1.0, 0.3, 0.1); // avoid axis-aligned rays
    let mut crossings = 0u32;
    for &fid in shell.faces() {
        crossings += count_face_ray_crossings(topo, fid, point, direction, deflection)?;
    }

    // Odd crossings = inside (winding ~1.0), even = outside (winding ~0.0).
    let winding = if crossings % 2 == 1 { 1.0 } else { 0.0 };
    Ok((winding, false))
}

// ── Math helpers ─────────────────────────────────────────────────────────

/// Compute the normal of a polygon via Newell's method.
fn polygon_normal(verts: &[Point3]) -> Vec3 {
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    let n = verts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = verts[i];
        let vj = verts[j];
        nx += (vi.y() - vj.y()) * (vi.z() + vj.z());
        ny += (vi.z() - vj.z()) * (vi.x() + vj.x());
        nz += (vi.x() - vj.x()) * (vi.y() + vj.y());
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < DEGENERATE_LEN {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(nx / len, ny / len, nz / len)
    }
}

/// Solve a·t² + b·t + c = 0, returning real roots.
fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < NEAR_ZERO {
        if b.abs() < NEAR_ZERO {
            return Vec::new();
        }
        return vec![-c / b];
    }

    let disc = b * b - 4.0 * a * c;
    if disc < -RAY_T_MIN {
        return Vec::new();
    }
    if disc < RAY_T_MIN {
        return vec![-b / (2.0 * a)];
    }

    let sqrt_disc = disc.sqrt();
    let q = if b >= 0.0 {
        -0.5 * (b + sqrt_disc)
    } else {
        -0.5 * (b - sqrt_disc)
    };

    let mut roots = Vec::with_capacity(2);
    roots.push(q / a);
    if q.abs() > NEAR_ZERO {
        roots.push(c / q);
    }
    roots
}

/// Solve a cubic a·t³ + b·t² + c·t + d = 0, returning all real roots.
fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a.abs() < NEAR_ZERO {
        return solve_quadratic(b, c, d);
    }

    let p = b / a;
    let q = c / a;
    let r = d / a;

    // Depressed cubic u³ + dp·u + dq = 0, where t = u - p/3.
    let p2 = p * p;
    let dp = q - p2 / 3.0;
    let dq = r - p * q / 3.0 + 2.0 * p2 * p / 27.0;

    let disc = dq * dq / 4.0 + dp * dp * dp / 27.0;
    let shift = -p / 3.0;

    if disc > RAY_T_MIN {
        // One real root.
        let sqrt_disc = disc.sqrt();
        let u = (-dq / 2.0 + sqrt_disc).cbrt() + (-dq / 2.0 - sqrt_disc).cbrt();
        vec![u + shift]
    } else if disc < -RAY_T_MIN {
        // Three real roots (casus irreducibilis — trigonometric method).
        // dp < 0 here, so -dp/3 > 0.
        // m = sqrt(-dp/3) = cbrt(sqrt(-dp³/27)); factor in each root is 2·m.
        let m = (-dp / 3.0).sqrt();
        let theta = (-dq / (2.0 * m * m * m)).clamp(-1.0, 1.0).acos();
        let tau = std::f64::consts::TAU;
        vec![
            2.0 * m * (theta / 3.0).cos() + shift,
            2.0 * m * ((theta + tau) / 3.0).cos() + shift,
            2.0 * m * ((theta + 2.0 * tau) / 3.0).cos() + shift,
        ]
    } else {
        // Repeated root.
        if dq.abs() < NEAR_ZERO {
            vec![shift] // triple root
        } else {
            let u = (dq / 2.0).cbrt();
            vec![-2.0 * u + shift, u + shift]
        }
    }
}

/// Solve a quartic equation via Ferrari's method.
#[allow(clippy::many_single_char_names)]
fn solve_quartic(a: f64, b: f64, c: f64, d: f64, e: f64) -> Vec<f64> {
    if a.abs() < NEAR_ZERO * NEAR_ZERO {
        return solve_cubic(b, c, d, e);
    }

    let p = b / a;
    let q = c / a;
    let r = d / a;
    let s = e / a;

    // Resolvent cubic: y³ - q·y² + (p·r - 4s)·y - (p²s - 4qs + r²) = 0
    let rc_b = -q;
    let rc_c = p * r - 4.0 * s;
    let rc_d = -(p * p * s - 4.0 * q * s + r * r);

    let y = solve_cubic_one_real(rc_b, rc_c, rc_d);

    let disc1 = p * p * 0.25 - q + y;
    if disc1 < -HALF_SPACE_EPS {
        return Vec::new();
    }
    let disc1 = disc1.max(0.0).sqrt();

    let disc2a = y * y * 0.25 - s;
    if disc2a < -HALF_SPACE_EPS {
        return Vec::new();
    }
    let disc2 = disc2a.max(0.0).sqrt();

    let sign = if p * y * 0.5 - r >= 0.0 { 1.0 } else { -1.0 };

    let mut roots = Vec::with_capacity(4);
    roots.extend(solve_quadratic(
        1.0,
        p * 0.5 + disc1,
        y * 0.5 + sign * disc2,
    ));
    roots.extend(solve_quadratic(
        1.0,
        p * 0.5 - disc1,
        y * 0.5 - sign * disc2,
    ));
    roots
}

/// Find one real root of t³ + b·t² + c·t + d = 0 via Cardano.
fn solve_cubic_one_real(b: f64, c: f64, d: f64) -> f64 {
    let p = c - b * b / 3.0;
    let q = d - b * c / 3.0 + 2.0 * b * b * b / 27.0;

    let disc = q * q / 4.0 + p * p * p / 27.0;

    let u = if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        (-q / 2.0 + sqrt_disc).cbrt() + (-q / 2.0 - sqrt_disc).cbrt()
    } else {
        let r = (-p * p * p / 27.0).sqrt();
        let theta = (-q / (2.0 * r)).clamp(-1.0, 1.0).acos();
        2.0 * r.cbrt() * (theta / 3.0).cos()
    };

    u - b / 3.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};

    #[test]
    fn point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_on_boundary_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(1.0, 1.0, 2.0), 0.1, 1e-3).unwrap();
        assert_eq!(result, PointClassification::OnBoundary);
    }

    #[test]
    fn point_outside_negative_direction() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point(&topo, solid, Point3::new(-5.0, -5.0, -5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_near_corner() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(0.9, 0.9, 0.9), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    // ── Analytic surface tests ───────────────────────────

    #[test]
    fn point_inside_cylinder() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 2.0, 5.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_cylinder() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 2.0, 5.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(10.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_inside_sphere() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 3.0, 32).unwrap();

        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_sphere() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 3.0, 32).unwrap();

        let result = classify_point(&topo, solid, Point3::new(5.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_inside_cone() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 5.0).unwrap();

        // Point on the axis, inside the cone
        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_cone() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 5.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(10.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_inside_torus() {
        let mut topo = Topology::new();
        // major=3, minor=1 → tube center at distance 3 from origin
        let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

        // Point inside the tube (on the x-axis at distance 3 from origin)
        let result = classify_point(&topo, solid, Point3::new(3.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_torus() {
        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

        // Point at origin — in the hole of the torus
        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_outside_torus_far() {
        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

        // Point far from torus
        let result = classify_point(&topo, solid, Point3::new(10.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    // ── Solver tests ────────────────────────────────────────

    #[test]
    fn cubic_three_roots() {
        // (t-1)(t-2)(t-3) = t³ - 6t² + 11t - 6
        let mut roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(roots.len(), 3, "expected 3 roots, got {}", roots.len());
        assert!((roots[0] - 1.0).abs() < 1e-8);
        assert!((roots[1] - 2.0).abs() < 1e-8);
        assert!((roots[2] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn cubic_one_root() {
        // t³ + t + 2 = 0 has one real root at t = -1
        // (t+1)(t² - t + 2) = t³ + 2  ... no that's not right
        // Let's use t³ + 3t + 4: discriminant = 16 + 4 > 0 → one real root
        let roots = solve_cubic(1.0, 0.0, 3.0, 4.0);
        assert_eq!(roots.len(), 1, "expected 1 root, got {}", roots.len());
    }

    #[test]
    fn quartic_degenerate_to_cubic() {
        // a ≈ 0: effectively t³ - 6t² + 11t - 6 = 0 → roots 1,2,3
        let mut roots = solve_quartic(0.0, 1.0, -6.0, 11.0, -6.0);
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(roots.len(), 3, "expected 3 roots, got {}", roots.len());
        assert!((roots[0] - 1.0).abs() < 1e-8);
        assert!((roots[1] - 2.0).abs() < 1e-8);
        assert!((roots[2] - 3.0).abs() < 1e-8);
    }

    // ── Winding number tests ─────────────────────────────

    #[test]
    fn winding_point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_winding(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn winding_point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_winding(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    // ── Robust classifier tests ──────────────────────────

    #[test]
    fn robust_point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_robust(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn robust_point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_robust(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    // ── Math helper tests ────────────────────────────────

    #[test]
    fn quadratic_two_roots() {
        let mut roots = solve_quadratic(1.0, -5.0, 6.0);
        assert_eq!(roots.len(), 2);
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sorted = roots;
        assert!((sorted[0] - 2.0).abs() < 1e-10);
        assert!((sorted[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn quadratic_no_roots() {
        let roots = solve_quadratic(1.0, 0.0, 1.0);
        assert!(roots.is_empty());
    }

    #[test]
    fn quartic_known_roots() {
        let mut roots = solve_quartic(1.0, -10.0, 35.0, -50.0, 24.0);
        assert!(roots.len() >= 4, "expected 4 roots, got {}", roots.len());
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sorted = roots;
        for (i, &expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!(
                (sorted[i] - expected).abs() < 1e-6,
                "root {} expected {}, got {}",
                i,
                expected,
                sorted[i]
            );
        }
    }
}
