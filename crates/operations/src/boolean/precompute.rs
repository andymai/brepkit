//! Phase 0 precompute and shared helpers for boolean operations.
//!
//! Functions in this module are used during the initial setup of a boolean
//! operation: collecting face data, computing bounding boxes, and checking
//! for containment or disjointness shortcuts.

use std::collections::HashSet;

use brepkit_math::aabb::Aabb3;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::classify::try_build_analytic_classifier;
use super::fragments::sample_edge_curve;
use super::types::{BooleanOp, CLASSIFIER_CYL_SEGMENTS, CLOSED_CURVE_SAMPLES, FaceClass, FaceData};

// ---------------------------------------------------------------------------
// Shared helpers (used across multiple phases)
// ---------------------------------------------------------------------------

/// Deduplicate 3D points by quantized position (spatial hashing).
///
/// Resolution is derived from the tolerance: `1.0 / tol.linear`.
pub(super) fn dedup_points_by_position(pts: &mut Vec<Point3>, tol: Tolerance) {
    let scale = 1.0 / tol.linear;
    let mut seen = HashSet::new();
    pts.retain(|p| {
        #[allow(clippy::cast_possible_truncation)]
        let key = (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        );
        seen.insert(key)
    });
}

/// Compute a representative normal and d-value for a face from its surface type.
///
/// For planar faces, returns the plane normal/d directly. For analytic surfaces
/// (cylinder, sphere, cone, torus), computes the normal from the surface
/// definition and a sample vertex — avoiding expensive tessellation.
pub(super) fn analytic_face_normal_d(surface: &FaceSurface, verts: &[Point3]) -> (Vec3, f64) {
    match surface {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        FaceSurface::Cylinder(cyl) => {
            // Cylinder axis direction as representative normal.
            let n = cyl.axis();
            let d = if verts.is_empty() {
                0.0
            } else {
                crate::dot_normal_point(n, verts[0])
            };
            (n, d)
        }
        FaceSurface::Sphere(sph) => {
            // Outward radial from center through first vertex.
            if let Some(&v) = verts.first() {
                let dir = v - sph.center();
                let n = dir.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                (n, crate::dot_normal_point(n, v))
            } else {
                (Vec3::new(0.0, 0.0, 1.0), 0.0)
            }
        }
        FaceSurface::Cone(cone) => {
            let n = cone.axis();
            let d = if verts.is_empty() {
                0.0
            } else {
                crate::dot_normal_point(n, verts[0])
            };
            (n, d)
        }
        FaceSurface::Torus(tor) => {
            let n = tor.z_axis();
            let d = if verts.is_empty() {
                0.0
            } else {
                crate::dot_normal_point(n, verts[0])
            };
            (n, d)
        }
        FaceSurface::Nurbs(_) => {
            // For NURBS, use polygon normal from vertices.
            if verts.len() >= 3 {
                let e1 = verts[1] - verts[0];
                let e2 = verts[2] - verts[0];
                let n = e1.cross(e2).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                (n, crate::dot_normal_point(n, verts[0]))
            } else {
                (Vec3::new(0.0, 0.0, 1.0), 0.0)
            }
        }
    }
}

/// Compute the v-range hint for an analytic surface based on face vertices.
///
/// Returns `Some((v_min, v_max))` if the surface has a non-trivial v
/// parameterization that depends on the face extent (cylinder, cone).
/// Returns `None` for surfaces where the default v_range is correct
/// (sphere, torus).
pub(super) fn compute_v_range_hint(surface: &FaceSurface, verts: &[Point3]) -> Option<(f64, f64)> {
    match surface {
        FaceSurface::Cylinder(cyl) => {
            // v = axial distance from origin. Compute from face vertices
            // with padding to ensure intersections near the boundary are found.
            cylinder_v_extent(cyl, verts).map(|(lo, hi)| {
                let pad = (hi - lo) * 0.1;
                (lo - pad, hi + pad)
            })
        }
        FaceSurface::Cone(cone) => {
            // v = distance from apex along the axis-radial direction.
            // Compute from face vertices.
            let axis = cone.axis();
            let apex = cone.apex();
            let half = cone.half_angle();
            let (sin_a, cos_a) = half.sin_cos();
            let mut v_min = f64::MAX;
            let mut v_max = f64::MIN;
            for &p in verts {
                let d = p - apex;
                let axial = d.dot(axis);
                let radial_sq = (d.dot(d) - axial * axial).max(0.0);
                // v = sqrt(axial^2 + radial_sq) with correct sign
                let v = axial * sin_a + radial_sq.sqrt() * cos_a;
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
            // Degenerate v-extent guard: if the cone's v-parameter range is
            // thinner than 1e-10, the face has no meaningful axial span.
            // Matches build_v_levels dedup tolerance (1e-10).
            if (v_max - v_min).abs() < 1e-10 {
                None
            } else {
                let pad = (v_max - v_min) * 0.1;
                // Clamp the lower bound away from the apex singularity (v=0),
                // but allow negative v if the geometry requires it.
                let lo = if v_min > 0.0 {
                    (v_min - pad).max(0.001)
                } else {
                    v_min - pad
                };
                Some((lo, v_max + pad))
            }
        }
        // Sphere, torus, plane, and NURBS have fixed or irrelevant v-ranges.
        FaceSurface::Sphere(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_) => None,
    }
}

/// Compute the axial extent (v-range) of points projected onto a cylinder axis.
///
/// Returns `None` if the extent is degenerate (< 1e-10).
///
/// 1e-10: parametric-space dedup tolerance for axial projection. The v-parameter
/// is in model-space units (meters), so 1e-10 m = 0.1 nm. This matches the
/// `build_v_levels` dedup tolerance used throughout the fragment builders.
pub fn cylinder_v_extent(
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    points: &[Point3],
) -> Option<(f64, f64)> {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for &p in points {
        let v = cyl.axis().dot(p - cyl.origin());
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    if (v_max - v_min).abs() < 1e-10 {
        None
    } else {
        Some((v_min, v_max))
    }
}

/// Compute the v-extent of points projected onto a cone's generator direction.
///
/// Returns `None` if the extent is degenerate (< 1e-10).
///
/// 1e-10: parametric-space dedup tolerance — same as `cylinder_v_extent`.
pub fn cone_v_extent(
    cone: &brepkit_math::surfaces::ConicalSurface,
    points: &[Point3],
) -> Option<(f64, f64)> {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for &p in points {
        let (_, v) = cone.project_point(p);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    if (v_max - v_min).abs() < 1e-10 {
        None
    } else {
        Some((v_min, v_max))
    }
}

/// Compute the v-extent (latitude range) of points projected onto a sphere.
///
/// Returns `None` if the extent is degenerate (< 1e-10).
#[allow(dead_code)] // used by boolean_v2 (upcoming)
pub fn sphere_v_extent(
    sph: &brepkit_math::surfaces::SphericalSurface,
    points: &[Point3],
) -> Option<(f64, f64)> {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for &p in points {
        let (_, v) = sph.project_point(p);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    if (v_max - v_min).abs() < 1e-10 {
        None
    } else {
        Some((v_min, v_max))
    }
}

#[allow(dead_code)] // used by boolean_v2 (upcoming)
/// Compute the v-extent (minor angle range) of points projected onto a torus.
///
/// Returns `None` if the extent is degenerate (< 1e-10).
pub fn torus_v_extent(
    tor: &brepkit_math::surfaces::ToroidalSurface,
    points: &[Point3],
) -> Option<(f64, f64)> {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for &p in points {
        let (_, v) = tor.project_point(p);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    if (v_max - v_min).abs() < 1e-10 {
        None
    } else {
        Some((v_min, v_max))
    }
}

// ---------------------------------------------------------------------------
// Phase 0 helpers
// ---------------------------------------------------------------------------

/// Collect face polygons and plane data for a solid.
///
/// For planar faces, returns the face polygon directly.
/// For NURBS faces, tessellates into triangles and returns each triangle
/// as a separate planar "face" entry. This allows the existing planar
/// boolean clipping algorithm to handle NURBS solids.
pub(super) fn collect_face_data(
    topo: &Topology,
    solid_id: SolidId,
    _deflection: f64,
) -> Result<FaceData, crate::OperationsError> {
    let solid = topo.solid(solid_id)?;
    let shell = topo.shell(solid.outer_shell())?;
    let mut result = Vec::with_capacity(shell.faces().len());

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let verts = face_polygon(topo, fid)?;
        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                result.push((fid, verts, *normal, *d));
            }
            FaceSurface::Cylinder(cyl) => {
                // Approximate the cylinder barrel as planar quads for the
                // classifier. Much faster than full tessellation (~16 quads
                // vs ~800 triangles per band) while giving correct crossing
                // parity for ray-casting.
                let Some((v_min, v_max)) = cylinder_v_extent(cyl, &verts) else {
                    continue;
                };
                #[allow(clippy::cast_precision_loss)]
                for i in 0..CLASSIFIER_CYL_SEGMENTS {
                    let u0 = std::f64::consts::TAU * (i as f64) / (CLASSIFIER_CYL_SEGMENTS as f64);
                    let u1 =
                        std::f64::consts::TAU * ((i + 1) as f64) / (CLASSIFIER_CYL_SEGMENTS as f64);
                    let b0 = cyl.evaluate(u0, v_min);
                    let b1 = cyl.evaluate(u1, v_min);
                    let t0 = cyl.evaluate(u0, v_max);
                    let t1 = cyl.evaluate(u1, v_max);

                    // Two triangles per quad.
                    for tri in &[[b0, b1, t1], [b0, t1, t0]] {
                        let edge1 = tri[1] - tri[0];
                        let edge2 = tri[2] - tri[0];
                        let cross = edge1.cross(edge2);
                        let Ok(n) = cross.normalize() else { continue };
                        let d_val = crate::dot_normal_point(n, tri[0]);
                        result.push((fid, tri.to_vec(), n, d_val));
                    }
                }
            }
            FaceSurface::Sphere(sph) => {
                // Approximate sphere face as a (u,v) grid of planar quads.
                // Use compute_sphere_v_range to correctly handle hemisphere caps
                // (boundary vertices are at the equator, face extends to pole).
                let face_data = topo.face(fid)?;
                let sign = if face_data.is_reversed() { -1.0 } else { 1.0 };
                let center = sph.center();

                let n_u = CLASSIFIER_CYL_SEGMENTS;
                let n_v = 8;
                let (v_min, v_max) =
                    crate::tessellate::compute_sphere_v_range(topo, face_data, sph);

                #[allow(clippy::cast_precision_loss)]
                for j in 0..n_v {
                    let vp0 = v_min + (v_max - v_min) * j as f64 / n_v as f64;
                    let vp1 = v_min + (v_max - v_min) * (j + 1) as f64 / n_v as f64;
                    for i in 0..n_u {
                        let u0 = std::f64::consts::TAU * i as f64 / n_u as f64;
                        let u1 = std::f64::consts::TAU * (i + 1) as f64 / n_u as f64;
                        let b0 = sph.evaluate(u0, vp0);
                        let b1 = sph.evaluate(u1, vp0);
                        let t0 = sph.evaluate(u0, vp1);
                        let t1 = sph.evaluate(u1, vp1);

                        for tri in &[[b0, b1, t1], [b0, t1, t0]] {
                            // Radial normal from sphere center.
                            let cx = (tri[0].x() + tri[1].x() + tri[2].x()) / 3.0;
                            let cy = (tri[0].y() + tri[1].y() + tri[2].y()) / 3.0;
                            let cz = (tri[0].z() + tri[1].z() + tri[2].z()) / 3.0;
                            let dir = Vec3::new(cx - center.x(), cy - center.y(), cz - center.z());
                            let len = dir.length();
                            if len < 1e-15 {
                                continue;
                            }
                            let n = dir * (sign / len);
                            let d_val = crate::dot_normal_point(n, tri[0]);
                            result.push((fid, tri.to_vec(), n, d_val));
                        }
                    }
                }
            }
            FaceSurface::Cone(cone) => {
                // Approximate cone face as a (u,v) grid of planar quads.
                let face_data = topo.face(fid)?;
                let sign = if face_data.is_reversed() { -1.0 } else { 1.0 };
                let Some((v_min, v_max)) = cone_v_extent(cone, &verts) else {
                    continue;
                };
                let n_u = CLASSIFIER_CYL_SEGMENTS;

                #[allow(clippy::cast_precision_loss)]
                for i in 0..n_u {
                    let u0 = std::f64::consts::TAU * i as f64 / n_u as f64;
                    let u1 = std::f64::consts::TAU * (i + 1) as f64 / n_u as f64;
                    let b0 = cone.evaluate(u0, v_min);
                    let b1 = cone.evaluate(u1, v_min);
                    let t0 = cone.evaluate(u0, v_max);
                    let t1 = cone.evaluate(u1, v_max);

                    for tri in &[[b0, b1, t1], [b0, t1, t0]] {
                        let edge1 = tri[1] - tri[0];
                        let edge2 = tri[2] - tri[0];
                        let cross = edge1.cross(edge2);
                        let Ok(mut n) = cross.normalize() else {
                            continue;
                        };
                        // Ensure normal points outward (radial from cone axis).
                        let mid_u = f64::midpoint(u0, u1);
                        let mid_v = f64::midpoint(v_min, v_max);
                        let surf_n = cone.normal(mid_u, mid_v);
                        if n.dot(surf_n) * sign < 0.0 {
                            n = -n;
                        }
                        let d_val = crate::dot_normal_point(n, tri[0]);
                        result.push((fid, tri.to_vec(), n, d_val));
                    }
                }
            }
            FaceSurface::Torus(tor) => {
                // Approximate torus face as a (u,v) grid of planar quads.
                let face_data = topo.face(fid)?;
                let sign = if face_data.is_reversed() { -1.0 } else { 1.0 };
                let n_u = CLASSIFIER_CYL_SEGMENTS;
                let n_v = CLASSIFIER_CYL_SEGMENTS;

                let v_vals: Vec<f64> = verts.iter().map(|p| tor.project_point(*p).1).collect();
                let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
                let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                #[allow(clippy::cast_precision_loss)]
                for j in 0..n_v {
                    let vp0 = v_min + (v_max - v_min) * j as f64 / n_v as f64;
                    let vp1 = v_min + (v_max - v_min) * (j + 1) as f64 / n_v as f64;
                    for i in 0..n_u {
                        let u0 = std::f64::consts::TAU * i as f64 / n_u as f64;
                        let u1 = std::f64::consts::TAU * (i + 1) as f64 / n_u as f64;
                        let b0 = tor.evaluate(u0, vp0);
                        let b1 = tor.evaluate(u1, vp0);
                        let t0 = tor.evaluate(u0, vp1);
                        let t1 = tor.evaluate(u1, vp1);

                        for tri in &[[b0, b1, t1], [b0, t1, t0]] {
                            let edge1 = tri[1] - tri[0];
                            let edge2 = tri[2] - tri[0];
                            let cross = edge1.cross(edge2);
                            let Ok(mut n) = cross.normalize() else {
                                continue;
                            };
                            let mid_u = f64::midpoint(u0, u1);
                            let mid_v = f64::midpoint(vp0, vp1);
                            let surf_n = tor.normal(mid_u, mid_v);
                            if n.dot(surf_n) * sign < 0.0 {
                                n = -n;
                            }
                            let d_val = crate::dot_normal_point(n, tri[0]);
                            result.push((fid, tri.to_vec(), n, d_val));
                        }
                    }
                }
            }
            FaceSurface::Nurbs(surface) => {
                // Approximate NURBS face as a coarse (u,v) grid.
                let face_data = topo.face(fid)?;
                let sign = if face_data.is_reversed() { -1.0 } else { 1.0 };
                let n_u = 8;
                let n_v = 8;
                let (u_min, u_max) = surface.domain_u();
                let (v_min, v_max) = surface.domain_v();

                #[allow(clippy::cast_precision_loss)]
                for j in 0..n_v {
                    let vp0 = v_min + (v_max - v_min) * j as f64 / n_v as f64;
                    let vp1 = v_min + (v_max - v_min) * (j + 1) as f64 / n_v as f64;
                    for i in 0..n_u {
                        let up0 = u_min + (u_max - u_min) * i as f64 / n_u as f64;
                        let up1 = u_min + (u_max - u_min) * (i + 1) as f64 / n_u as f64;
                        let b0 = surface.evaluate(up0, vp0);
                        let b1 = surface.evaluate(up1, vp0);
                        let t0 = surface.evaluate(up0, vp1);
                        let t1 = surface.evaluate(up1, vp1);

                        for tri in &[[b0, b1, t1], [b0, t1, t0]] {
                            let edge1 = tri[1] - tri[0];
                            let edge2 = tri[2] - tri[0];
                            let cross = edge1.cross(edge2);
                            let Ok(mut n) = cross.normalize() else {
                                continue;
                            };
                            let mid_u = f64::midpoint(up0, up1);
                            let mid_v = f64::midpoint(vp0, vp1);
                            let Ok(surf_n) = surface.normal(mid_u, mid_v) else {
                                continue;
                            };
                            if n.dot(surf_n) * sign < 0.0 {
                                n = -n;
                            }
                            let d_val = crate::dot_normal_point(n, tri[0]);
                            result.push((fid, tri.to_vec(), n, d_val));
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Get a polygon approximation of a face by sampling curved edges.
///
/// Samples circle/ellipse edges into 32 points so faces with a
/// single closed-curve edge (e.g. cylinder caps) get a proper polygon.
///
/// # Errors
///
/// Returns an error if the face or its wire cannot be resolved.
pub fn face_polygon(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut pts = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let curve = edge.curve();
        // Sample closed parametric edges (start == end vertex).
        // Partial arcs fall through to the vertex-based path.
        let start_vid = edge.start();
        let end_vid = edge.end();
        let is_closed_edge = start_vid == end_vid
            && matches!(
                curve,
                EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
            );
        if is_closed_edge {
            // Must use CLOSED_CURVE_SAMPLES (not a larger value) — vertex count
            // must match create_band_fragments and inner-wire dedup for sharing.
            let mut sampled = sample_edge_curve(curve, CLOSED_CURVE_SAMPLES);
            if !oe.is_forward() {
                sampled.reverse();
            }
            pts.extend(sampled);
        } else {
            let vid = oe.oriented_start(edge);
            pts.push(topo.vertex(vid)?.point());
        }
    }

    Ok(pts)
}

/// Compute a conservative AABB for a face using only wire vertex positions.
///
/// Unlike `face_polygon()` which samples closed curves (32 points per circle),
/// this function only collects the start/end vertex positions of each edge.
/// For analytic surfaces (cylinder, sphere, cone, torus, NURBS), the AABB is
/// expanded to account for surface curvature via `expand_aabb_for_surface`.
///
/// This is much cheaper than `face_polygon()` and is used for early rejection:
/// if the wire AABB doesn't overlap the tool's AABB, the face cannot intersect
/// any tool face.
pub(super) fn face_wire_aabb(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Aabb3, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut points = Vec::with_capacity(wire.edges().len() * 4);
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        points.push(topo.vertex(edge.start())?.point());
        points.push(topo.vertex(edge.end())?.point());
        // For closed curve edges (start == end), the two vertex positions
        // are a single point — not enough to capture the curve extent.
        // Sample 4 cardinal points to get a proper AABB.
        if edge.start() == edge.end() {
            let samples = sample_edge_curve(edge.curve(), 4);
            points.extend(samples);
        }
    }
    let mut aabb = Aabb3::try_from_points(points.into_iter()).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "face has no vertices".into(),
        }
    })?;
    crate::measure::expand_aabb_for_surface(&mut aabb, face.surface());
    Ok(aabb)
}

/// Compute AABB encompassing all face vertices, expanded for surface curvature.
///
/// For analytic surfaces (sphere, cylinder, cone, torus), the tessellated
/// vertices may not reach surface extremes. We call `expand_aabb_for_surface`
/// on each face to produce a conservative bounding box.
pub(super) fn solid_aabb(
    topo: &Topology,
    faces: &FaceData,
    tol: Tolerance,
) -> Result<Aabb3, crate::OperationsError> {
    let mut aabb = Aabb3::try_from_points(
        faces
            .iter()
            .flat_map(|(_, verts, _, _)| verts.iter().copied()),
    )
    .map(|bb| bb.expanded(tol.linear))
    .ok_or_else(|| crate::OperationsError::InvalidInput {
        reason: "solid has no vertices".into(),
    })?;

    for (fid, _, _, _) in faces {
        let face = topo.face(*fid)?;
        crate::measure::expand_aabb_for_surface(&mut aabb, face.surface());
    }

    Ok(aabb)
}

/// Check if one solid is entirely contained in the other and short-circuit
/// the boolean operation without expensive face intersection computation.
///
/// Uses analytic classifiers (box, sphere, cylinder) for O(1) per-vertex
/// containment tests. Returns `None` if classifiers can't be built or
/// containment isn't detected.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_containment_shortcut(
    topo: &mut Topology,
    op: BooleanOp,
    _a: SolidId,
    _b: SolidId,
    faces_a: &FaceData,
    faces_b: &FaceData,
    tol: Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let classifier_a = try_build_analytic_classifier(topo, _a);
    let classifier_b = try_build_analytic_classifier(topo, _b);

    let extract = |data: &FaceData| -> Vec<(Vec<Point3>, Vec3, f64)> {
        data.iter()
            .map(|(_, verts, normal, d)| (verts.clone(), *normal, *d))
            .collect()
    };

    // Check: is A entirely inside B?
    if let Some(ref cb) = classifier_b {
        let all_a_inside_b = faces_a.iter().all(|(_, verts, _, _)| {
            verts
                .iter()
                .all(|v| matches!(cb.classify(*v, tol), Some(FaceClass::Inside)))
        });
        if all_a_inside_b {
            log::debug!("boolean {op:?}: A fully inside B, shortcut");
            return match op {
                // A - B: A is inside B → nothing remains
                BooleanOp::Cut => Err(crate::OperationsError::InvalidInput {
                    reason: "cut: first solid is fully inside second".into(),
                }),
                // A ∩ B: result is A (since A ⊂ B)
                BooleanOp::Intersect => {
                    Ok(Some(super::assemble_solid(topo, &extract(faces_a), tol)?))
                }
                // A ∪ B: result is B (since A ⊂ B)
                BooleanOp::Fuse => Ok(Some(super::assemble_solid(topo, &extract(faces_b), tol)?)),
            };
        }
    }

    // Check: is B entirely inside A?
    if let Some(ref ca) = classifier_a {
        let all_b_inside_a = faces_b.iter().all(|(_, verts, _, _)| {
            verts
                .iter()
                .all(|v| matches!(ca.classify(*v, tol), Some(FaceClass::Inside)))
        });
        if all_b_inside_a {
            log::debug!("boolean {op:?}: B fully inside A, shortcut");
            return match op {
                // A - B: B is inside A → result is A with B-shaped hole
                // Can't shortcut — need actual face splitting.
                BooleanOp::Cut => Ok(None),
                // A ∩ B: result is B (since B ⊂ A)
                BooleanOp::Intersect => {
                    Ok(Some(super::assemble_solid(topo, &extract(faces_b), tol)?))
                }
                // A ∪ B: result is A (since B ⊂ A)
                BooleanOp::Fuse => Ok(Some(super::assemble_solid(topo, &extract(faces_a), tol)?)),
            };
        }
    }

    Ok(None)
}

/// Handle the case where two solids' AABBs don't overlap.
pub(super) fn handle_disjoint(
    topo: &mut Topology,
    op: BooleanOp,
    faces_a: &FaceData,
    faces_b: &FaceData,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    let extract = |data: &FaceData| -> Vec<(Vec<Point3>, Vec3, f64)> {
        data.iter()
            .map(|(_, verts, normal, d)| (verts.clone(), *normal, *d))
            .collect()
    };

    match op {
        BooleanOp::Fuse => {
            let mut selected = extract(faces_a);
            selected.extend(extract(faces_b));
            super::assemble_solid(topo, &selected, tol)
        }
        BooleanOp::Cut => {
            // A - B with no overlap → A unchanged.
            super::assemble_solid(topo, &extract(faces_a), tol)
        }
        BooleanOp::Intersect => Err(crate::OperationsError::InvalidInput {
            reason: "intersection of disjoint solids is empty".into(),
        }),
    }
}
