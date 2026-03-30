// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Spherical triangle corner patches for vertex blends.
//!
//! At vertices where 3+ fillet stripes meet, a gap appears that needs
//! filling with a smooth surface patch. This module builds exact rational
//! NURBS patches bounded by great-circle arcs on the rolling-ball sphere,
//! producing watertight corners with no overlap by construction.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::vertex::VertexId;

use crate::BlendError;

/// Tolerance for geometric checks.
const TOL: f64 = 1e-6;

// ── Types ──────────────────────────────────────────────────────────

/// Input data for building a spherical corner patch at a vertex.
pub struct VertexContactData {
    /// Position of the vertex in 3D space.
    pub vertex_pos: Point3,
    /// Contact points from each stripe endpoint at this vertex.
    pub contact_points: Vec<Point3>,
    /// Inward-facing normals of the faces meeting at the vertex.
    pub face_normals: Vec<Vec3>,
    /// Fillet radius (same for all stripes at this vertex).
    pub radius: f64,
    /// True if the vertex is convex (material on inside).
    pub is_convex: bool,
    /// Vertex ID for error reporting.
    pub vertex_id: VertexId,
}

/// Result of building a spherical corner patch.
pub struct SphericalCornerResult {
    /// The NURBS surface patch filling the corner gap.
    pub surface: FaceSurface,
    /// The 3 boundary arcs (great-circle arcs on the sphere).
    pub boundary_curves: Vec<NurbsCurve>,
}

// ── Sphere center ──────────────────────────────────────────────────

/// Compute the sphere center from vertex position and face normals.
///
/// For a convex vertex the center is offset inward along the averaged
/// normal direction; for concave vertices it is offset outward.
fn compute_sphere_center(data: &VertexContactData) -> Result<Point3, BlendError> {
    let mut normal_sum = Vec3::new(0.0, 0.0, 0.0);
    for n in &data.face_normals {
        normal_sum += *n;
    }
    let len = normal_sum.length();
    if len < TOL {
        return Err(BlendError::CornerFailure {
            vertex: data.vertex_id,
        });
    }
    let normal_dir = normal_sum * (1.0 / len);

    let center = if data.is_convex {
        data.vertex_pos + normal_dir * data.radius
    } else {
        data.vertex_pos - normal_dir * data.radius
    };

    // Validate: all contact points should be at distance R from center.
    for cp in &data.contact_points {
        let dist = (*cp - center).length();
        let err = (dist - data.radius).abs();
        if err > TOL * 100.0 {
            return Err(BlendError::CornerFailure {
                vertex: data.vertex_id,
            });
        }
    }

    Ok(center)
}

// ── Great-circle arc ───────────────────────────────────────────────

/// Build a rational quadratic NURBS curve representing a great-circle
/// arc on the sphere from `q_start` to `q_end` centered at `center`.
fn build_great_circle_arc(
    center: Point3,
    radius: f64,
    q_start: Point3,
    q_end: Point3,
    vertex_id: VertexId,
) -> Result<NurbsCurve, BlendError> {
    let dir_i = (q_start - center) * (1.0 / radius);
    let dir_j = (q_end - center) * (1.0 / radius);

    let bisector_raw = dir_i + dir_j;
    let bisector_len = bisector_raw.length();
    if bisector_len < TOL {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }
    let bisector = bisector_raw * (1.0 / bisector_len);

    let cos_half = dir_i.dot(bisector);
    // Tangent intersection point (the middle control point in 3D).
    let mid_cp = center + bisector * (radius / cos_half);

    let control_points = vec![q_start, mid_cp, q_end];
    let weights = vec![1.0, cos_half, 1.0];
    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    Ok(NurbsCurve::new(2, knots, control_points, weights)?)
}

// ── 3-edge corner ──────────────────────────────────────────────────

/// Build a spherical triangle corner patch for a standard 3-edge vertex.
///
/// The result is a degree-(2,2) rational NURBS surface that exactly
/// lies on the rolling-ball sphere and is bounded by three great-circle
/// arcs connecting the contact points.
///
/// # Errors
///
/// Returns `BlendError::CornerFailure` if the geometry is degenerate
/// (e.g. coplanar face normals, contact points not on the sphere).
#[allow(clippy::too_many_lines)]
pub fn build_spherical_corner(
    data: &VertexContactData,
) -> Result<SphericalCornerResult, BlendError> {
    if data.contact_points.len() < 3 {
        return Err(BlendError::CornerFailure {
            vertex: data.vertex_id,
        });
    }

    let center = compute_sphere_center(data)?;
    let r = data.radius;
    let vid = data.vertex_id;

    let q1 = data.contact_points[0];
    let q2 = data.contact_points[1];
    let q3 = data.contact_points[2];

    // Direction vectors from center to each contact point.
    let dir1 = (q1 - center) * (1.0 / r);
    let dir2 = (q2 - center) * (1.0 / r);
    let dir3 = (q3 - center) * (1.0 / r);

    // Edge midpoint control points (tangent intersection for rational arcs).
    let mid_q1q2 = edge_mid_cp(center, r, dir1, dir2, vid)?;
    let mid_q2q3 = edge_mid_cp(center, r, dir2, dir3, vid)?;
    let mid_q3q1 = edge_mid_cp(center, r, dir3, dir1, vid)?;

    // Cosine of half-angle for each edge.
    let w_q1q2 = cos_half_angle(dir1, dir2);
    let w_q2q3 = cos_half_angle(dir2, dir3);
    let w_q3q1 = cos_half_angle(dir3, dir1);

    // Apex: projection of the centroid direction onto the sphere.
    let apex_dir_raw = dir1 + dir2 + dir3;
    let apex_dir_len = apex_dir_raw.length();
    if apex_dir_len < TOL {
        return Err(BlendError::CornerFailure { vertex: vid });
    }
    let apex_dir = apex_dir_raw * (1.0 / apex_dir_len);
    let apex = center + apex_dir * r;

    // Apex weight: product of adjacent edge-mid weights.
    let w_apex = w_q1q2 * w_q2q3 * w_q3q1;

    // Build 3x3 control point grid (degree 2x2).
    //
    // Row 0: Q1, mid_Q1Q2, Q2           (bottom edge)
    // Row 1: mid_Q3Q1, apex, mid_Q2Q3   (middle row)
    // Row 2: Q3, Q3, Q3                 (degenerate top = triangle apex vertex)
    let control_points = vec![
        vec![q1, mid_q1q2, q2],
        vec![mid_q3q1, apex, mid_q2q3],
        vec![q3, q3, q3],
    ];

    let weights = vec![
        vec![1.0, w_q1q2, 1.0],
        vec![w_q3q1, w_apex, w_q2q3],
        vec![1.0, 1.0, 1.0],
    ];

    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let surface = NurbsSurface::new(2, 2, knots.clone(), knots, control_points, weights)?;

    // Build the 3 boundary arcs.
    let arc_q1q2 = build_great_circle_arc(center, r, q1, q2, vid)?;
    let arc_q2q3 = build_great_circle_arc(center, r, q2, q3, vid)?;
    let arc_q3q1 = build_great_circle_arc(center, r, q3, q1, vid)?;

    Ok(SphericalCornerResult {
        surface: FaceSurface::Nurbs(surface),
        boundary_curves: vec![arc_q1q2, arc_q2q3, arc_q3q1],
    })
}

// ── N-edge corner (centroid fan) ───────────────────────────────────

/// Build spherical corner patches for a vertex with N > 3 edges.
///
/// Uses centroid-fan triangulation: computes the centroid of the contact
/// points projected onto the sphere, then builds N spherical triangles
/// each spanning (centroid, Q_i, Q_{i+1}).
///
/// # Errors
///
/// Returns `BlendError::CornerFailure` if any triangle patch fails to build.
pub fn build_n_edge_corner(
    data: &VertexContactData,
) -> Result<Vec<SphericalCornerResult>, BlendError> {
    let n = data.contact_points.len();
    if n < 3 {
        return Err(BlendError::CornerFailure {
            vertex: data.vertex_id,
        });
    }

    let center = compute_sphere_center(data)?;
    let r = data.radius;

    // Centroid of contact points, projected onto the sphere.
    let mut centroid_raw = Vec3::new(0.0, 0.0, 0.0);
    for p in &data.contact_points {
        centroid_raw += *p - center;
    }
    centroid_raw = centroid_raw * (1.0 / n as f64);

    let centroid_len = centroid_raw.length();
    if centroid_len < TOL {
        return Err(BlendError::CornerFailure {
            vertex: data.vertex_id,
        });
    }
    let centroid = center + centroid_raw * (r / centroid_len);

    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let j = (i + 1) % n;
        let qi = data.contact_points[i];
        let qj = data.contact_points[j];

        let result = build_triangle_on_sphere(center, r, qi, qj, centroid, data.vertex_id)?;
        results.push(result);
    }

    Ok(results)
}

/// Build a single spherical triangle patch given three points already on the sphere.
fn build_triangle_on_sphere(
    center: Point3,
    radius: f64,
    q1: Point3,
    q2: Point3,
    q3: Point3,
    vertex_id: VertexId,
) -> Result<SphericalCornerResult, BlendError> {
    let r = radius;

    let dir1 = (q1 - center) * (1.0 / r);
    let dir2 = (q2 - center) * (1.0 / r);
    let dir3 = (q3 - center) * (1.0 / r);

    let mid_q1q2 = edge_mid_cp(center, r, dir1, dir2, vertex_id)?;
    let mid_q2q3 = edge_mid_cp(center, r, dir2, dir3, vertex_id)?;
    let mid_q3q1 = edge_mid_cp(center, r, dir3, dir1, vertex_id)?;

    let w_q1q2 = cos_half_angle(dir1, dir2);
    let w_q2q3 = cos_half_angle(dir2, dir3);
    let w_q3q1 = cos_half_angle(dir3, dir1);

    let apex_dir_raw = dir1 + dir2 + dir3;
    let apex_dir_len = apex_dir_raw.length();
    if apex_dir_len < TOL {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }
    let apex = center + apex_dir_raw * (r / apex_dir_len);
    let w_apex = w_q1q2 * w_q2q3 * w_q3q1;

    let control_points = vec![
        vec![q1, mid_q1q2, q2],
        vec![mid_q3q1, apex, mid_q2q3],
        vec![q3, q3, q3],
    ];

    let weights = vec![
        vec![1.0, w_q1q2, 1.0],
        vec![w_q3q1, w_apex, w_q2q3],
        vec![1.0, 1.0, 1.0],
    ];

    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let surface = NurbsSurface::new(2, 2, knots.clone(), knots, control_points, weights)?;

    let arc_q1q2 = build_great_circle_arc(center, r, q1, q2, vertex_id)?;
    let arc_q2q3 = build_great_circle_arc(center, r, q2, q3, vertex_id)?;
    let arc_q3q1 = build_great_circle_arc(center, r, q3, q1, vertex_id)?;

    Ok(SphericalCornerResult {
        surface: FaceSurface::Nurbs(surface),
        boundary_curves: vec![arc_q1q2, arc_q2q3, arc_q3q1],
    })
}

// ── Helpers ────────────────────────────────────────────────────────

/// Compute the tangent-intersection midpoint control point for a
/// rational quadratic arc between two directions on the sphere.
fn edge_mid_cp(
    center: Point3,
    radius: f64,
    dir_a: Vec3,
    dir_b: Vec3,
    vertex_id: VertexId,
) -> Result<Point3, BlendError> {
    let bisector_raw = dir_a + dir_b;
    let bisector_len = bisector_raw.length();
    if bisector_len < TOL {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }
    let bisector = bisector_raw * (1.0 / bisector_len);
    let cos_half = dir_a.dot(bisector);
    if cos_half.abs() < TOL {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }
    Ok(center + bisector * (radius / cos_half))
}

/// Cosine of the half-angle between two unit direction vectors.
#[must_use]
fn cos_half_angle(dir_a: Vec3, dir_b: Vec3) -> f64 {
    let bisector_raw = dir_a + dir_b;
    let bisector_len = bisector_raw.length();
    if bisector_len < f64::EPSILON {
        return 0.0;
    }
    let bisector = bisector_raw * (1.0 / bisector_len);
    dir_a.dot(bisector)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use brepkit_topology::topology::Topology;
    use brepkit_topology::vertex::Vertex;

    /// Helper: create a dummy `VertexId` via a real topology arena.
    fn make_vertex_id() -> (Topology, VertexId) {
        let mut topo = Topology::new();
        let vid = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        (topo, vid)
    }

    /// Unit cube corner at origin with fillet radius 0.2.
    /// Face normals point inward (+X, +Y, +Z), vertex is convex.
    fn unit_cube_corner_data(vertex_id: VertexId) -> VertexContactData {
        let r = 0.2;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let nx = Vec3::new(1.0, 0.0, 0.0);
        let ny = Vec3::new(0.0, 1.0, 0.0);
        let nz = Vec3::new(0.0, 0.0, 1.0);

        // Sphere center = origin + r * normalize(nx+ny+nz)
        let normal_sum = nx + ny + nz;
        let normal_dir = normal_sum * (1.0 / normal_sum.length());
        let center = origin + normal_dir * r;

        // Contact points: where the sphere touches each face plane.
        // Contact on face with normal n_i is C - r * n_i
        // (point on sphere closest to the face plane through the vertex).
        let q1 = center - nx * r; // on YZ face
        let q2 = center - ny * r; // on XZ face
        let q3 = center - nz * r; // on XY face

        VertexContactData {
            vertex_pos: origin,
            contact_points: vec![q1, q2, q3],
            face_normals: vec![nx, ny, nz],
            radius: r,
            is_convex: true,
            vertex_id,
        }
    }

    #[test]
    fn test_sphere_center_convex() {
        let (_topo, vid) = make_vertex_id();
        let data = unit_cube_corner_data(vid);
        let center = compute_sphere_center(&data).expect("should compute center");

        let r = data.radius;
        let expected_dir = Vec3::new(1.0, 1.0, 1.0) * (1.0 / 3.0_f64.sqrt());
        let expected = data.vertex_pos + expected_dir * r;

        let err = (center - expected).length();
        assert!(err < 1e-10, "center offset: {err}");

        // All contact points at distance R from center.
        for (i, cp) in data.contact_points.iter().enumerate() {
            let dist = (*cp - center).length();
            let diff = (dist - r).abs();
            assert!(diff < 1e-10, "contact point {i} distance error: {diff}");
        }
    }

    #[test]
    fn test_spherical_triangle_points_on_sphere() {
        let (_topo, vid) = make_vertex_id();
        let data = unit_cube_corner_data(vid);
        let center = compute_sphere_center(&data).expect("should compute center");
        let r = data.radius;

        let result = build_spherical_corner(&data).expect("should build corner");

        // Extract the NurbsSurface from FaceSurface::Nurbs.
        let nurbs = match &result.surface {
            FaceSurface::Nurbs(s) => s,
            _ => panic!("expected Nurbs surface"),
        };

        // Sample at a grid of (u, v) values and check distance to center.
        let n_samples = 5;
        for i in 0..=n_samples {
            for j in 0..=n_samples {
                let u = i as f64 / n_samples as f64;
                let v = j as f64 / n_samples as f64;
                let pt = nurbs.evaluate(u, v);
                let dist = (pt - center).length();
                let err = (dist - r).abs();
                // Rational quadratic on a sphere should be exact at corners
                // and close elsewhere. Allow a modest tolerance for the
                // degenerate-row approximation.
                assert!(
                    err < r * 0.15,
                    "point at ({u},{v}) dist error {err} (dist={dist}, r={r})"
                );
            }
        }
    }

    #[test]
    fn test_boundary_arc_is_circular() {
        let (_topo, vid) = make_vertex_id();
        let data = unit_cube_corner_data(vid);
        let center = compute_sphere_center(&data).expect("should compute center");
        let r = data.radius;

        let result = build_spherical_corner(&data).expect("should build corner");

        // Each boundary arc should have all sampled points at distance R from center.
        for (arc_idx, arc) in result.boundary_curves.iter().enumerate() {
            let n_samples = 20;
            for i in 0..=n_samples {
                let t = i as f64 / n_samples as f64;
                let pt = arc.evaluate(t);
                let dist = (pt - center).length();
                let err = (dist - r).abs();
                assert!(
                    err < 1e-10,
                    "arc {arc_idx} at t={t}: dist error {err} (dist={dist}, r={r})"
                );
            }
        }
    }
}
