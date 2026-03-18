#![allow(dead_code)]
//! Phase 4: Classification — point-in-solid tests for boolean fragment labeling.
//!
//! Determines whether each face fragment lies inside, outside, or on the
//! boundary of the opposing solid. Supports analytic O(1) classifiers for
//! simple shapes (box, sphere, cylinder, cone, convex polyhedron) and falls
//! back to multi-ray casting with BVH acceleration for complex solids.

use super::intersect::point_along_line;
use super::types::{AnalyticClassifier, FaceClass, FaceData, FaceFragment};

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::predicates::point_in_polygon;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use crate::dot_normal_point;

// ---------------------------------------------------------------------------
// Analytic classifier impl
// ---------------------------------------------------------------------------

impl AnalyticClassifier {
    /// Classify a centroid as Inside, Outside, or None (on boundary → fall back).
    pub(super) fn classify(&self, centroid: Point3, tol: Tolerance) -> Option<FaceClass> {
        match self {
            Self::Sphere { center, radius } => {
                let dx = centroid.x() - center.x();
                let dy = centroid.y() - center.y();
                let dz = centroid.z() - center.z();
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq < (radius - tol.linear) * (radius - tol.linear) {
                    Some(FaceClass::Inside)
                } else if dist_sq > (radius + tol.linear) * (radius + tol.linear) {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::Cylinder {
                origin,
                axis,
                radius,
                z_min,
                z_max,
            } => {
                let diff = centroid - *origin;
                let axial = diff.dot(*axis);
                // Check axial bounds (cap faces).
                if axial < *z_min - tol.linear || axial > *z_max + tol.linear {
                    return Some(FaceClass::Outside);
                }
                // Radial distance from the axis.
                let projected = *axis * axial;
                let radial_vec = diff - projected;
                let radial_dist_sq = radial_vec.x() * radial_vec.x()
                    + radial_vec.y() * radial_vec.y()
                    + radial_vec.z() * radial_vec.z();
                if radial_dist_sq < (radius - tol.linear) * (radius - tol.linear)
                    && axial > *z_min + tol.linear
                    && axial < *z_max - tol.linear
                {
                    Some(FaceClass::Inside)
                } else if radial_dist_sq > (radius + tol.linear) * (radius + tol.linear) {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::Cone {
                origin,
                axis,
                z_min,
                z_max,
                r_at_z_min,
                r_at_z_max,
            } => {
                let diff = centroid - *origin;
                let axial = diff.dot(*axis);
                // Check axial bounds.
                if axial < *z_min - tol.linear || axial > *z_max + tol.linear {
                    return Some(FaceClass::Outside);
                }
                // Radial distance from axis.
                let projected = *axis * axial;
                let radial_vec = diff - projected;
                let radial_dist_sq = radial_vec.x() * radial_vec.x()
                    + radial_vec.y() * radial_vec.y()
                    + radial_vec.z() * radial_vec.z();
                // Linearly interpolate expected radius between z_min and z_max.
                let dz = z_max - z_min;
                // Numerical-zero guard: if the cone's axial height is < 1e-12,
                // the radius interpolation parameter is undefined — fall back to
                // midpoint (t = 0.5). 1e-12 is ~1 picometer, well below any
                // meaningful cone frustum height.
                let t = if dz.abs() > 1e-12 {
                    (axial - z_min) / dz
                } else {
                    0.5
                };
                let expected_r = r_at_z_min + t * (r_at_z_max - r_at_z_min);
                if radial_dist_sq < (expected_r - tol.linear).max(0.0).powi(2)
                    && axial > *z_min + tol.linear
                    && axial < *z_max - tol.linear
                {
                    Some(FaceClass::Inside)
                } else if radial_dist_sq > (expected_r + tol.linear) * (expected_r + tol.linear) {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::Box { min, max } => {
                // 6 comparisons — the fastest possible classifier.
                let tl = tol.linear;
                if centroid.x() > min.x() + tl
                    && centroid.x() < max.x() - tl
                    && centroid.y() > min.y() + tl
                    && centroid.y() < max.y() - tl
                    && centroid.z() > min.z() + tl
                    && centroid.z() < max.z() - tl
                {
                    Some(FaceClass::Inside)
                } else if centroid.x() < min.x() - tl
                    || centroid.x() > max.x() + tl
                    || centroid.y() < min.y() - tl
                    || centroid.y() > max.y() + tl
                    || centroid.z() < min.z() - tl
                    || centroid.z() > max.z() + tl
                {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::ConvexPolyhedron { planes } => {
                // For convex polyhedra, a point is inside iff it's on the
                // interior side of every face plane. Outward normal convention:
                // `signed_dist = normal · centroid - d` → positive means outside.
                let tl = tol.linear;
                let mut max_signed_dist = f64::NEG_INFINITY;
                for &(normal, d) in planes {
                    let cv = Vec3::new(centroid.x(), centroid.y(), centroid.z());
                    let signed_dist = normal.dot(cv) - d;
                    max_signed_dist = max_signed_dist.max(signed_dist);
                }
                if max_signed_dist < -tl {
                    Some(FaceClass::Inside)
                } else if max_signed_dist > tl {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Analytic classifier construction
// ---------------------------------------------------------------------------

/// Try to build an analytic classifier for a solid.
///
/// Returns `Some` when the solid is a simple convex analytic shape (e.g. sphere)
/// that supports O(1) point-in-solid tests. Falls back to `None` for complex or
/// non-analytic solids.
#[allow(clippy::too_many_lines)]
pub(super) fn try_build_analytic_classifier(
    topo: &Topology,
    solid: SolidId,
) -> Option<AnalyticClassifier> {
    let s = topo.solid(solid).ok()?;
    let shell = topo.shell(s.outer_shell()).ok()?;
    let tol = Tolerance::new();

    // Complex solids (>20 faces) can't be simple analytic shapes (box,
    // cylinder+caps, sphere). Skip the face-by-face scan to avoid O(F)
    // overhead on large fused/boolean intermediate results.
    if shell.faces().len() > 20 {
        return None;
    }

    let mut sphere_info: Option<(Point3, f64)> = None;
    let mut cylinder_info: Option<(Point3, Vec3, f64)> = None;
    let mut cone_info: Option<(Point3, Vec3, f64)> = None;
    let mut has_planar = false;
    let mut has_sphere = false;
    let mut has_cylinder = false;
    let mut has_cone = false;

    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Sphere(sph) => {
                has_sphere = true;
                if let Some((c, r)) = sphere_info {
                    let dc = (c - sph.center()).length();
                    if dc > tol.linear || (r - sph.radius()).abs() > tol.linear {
                        return None;
                    }
                } else {
                    sphere_info = Some((sph.center(), sph.radius()));
                }
            }
            FaceSurface::Cylinder(cyl) => {
                has_cylinder = true;
                if let Some((o, a, r)) = cylinder_info {
                    let do_ = (o - cyl.origin()).length();
                    let da = 1.0 - a.dot(cyl.axis()).abs();
                    if do_ > tol.linear || da > tol.angular || (r - cyl.radius()).abs() > tol.linear
                    {
                        return None;
                    }
                } else {
                    cylinder_info = Some((cyl.origin(), cyl.axis(), cyl.radius()));
                }
            }
            FaceSurface::Cone(con) => {
                has_cone = true;
                if let Some((a, ax, ha)) = cone_info {
                    let da = (a - con.apex()).length();
                    let dax = 1.0 - ax.dot(con.axis()).abs();
                    if da > tol.linear
                        || dax > tol.angular
                        || (ha - con.half_angle()).abs() > tol.angular
                    {
                        return None;
                    }
                } else {
                    cone_info = Some((con.apex(), con.axis(), con.half_angle()));
                }
            }
            FaceSurface::Plane { .. } => {
                has_planar = true;
            }
            // Torus, NURBS, and sphere (without a dedicated classifier)
            // fall back to ray-casting.
            FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => return None,
        }
    }

    // Pure planar solid (all faces are planes) — likely a box.
    // Detect axis-aligned boxes by checking that all faces are axis-aligned planes
    // that form 3 opposite-normal pairs.
    if has_planar && !has_sphere && !has_cylinder {
        let faces = shell.faces();
        if faces.len() == 6 {
            // Collect all plane normals and d values.
            let mut planes: Vec<(Vec3, f64)> = Vec::with_capacity(6);
            let mut is_box = true;
            for &fid in faces {
                if let Ok(face) = topo.face(fid) {
                    if let FaceSurface::Plane { normal, d } = face.surface() {
                        // Check axis-alignment: exactly one component should be ±1.
                        let ax = normal.x().abs();
                        let ay = normal.y().abs();
                        let az = normal.z().abs();
                        if (ax > 1.0 - tol.angular && ay < tol.angular && az < tol.angular)
                            || (ay > 1.0 - tol.angular && ax < tol.angular && az < tol.angular)
                            || (az > 1.0 - tol.angular && ax < tol.angular && ay < tol.angular)
                        {
                            planes.push((*normal, *d));
                        } else {
                            is_box = false;
                            break;
                        }
                    } else {
                        is_box = false;
                        break;
                    }
                } else {
                    is_box = false;
                    break;
                }
            }
            if is_box && planes.len() == 6 {
                // Extract min/max from the 3 axis pairs.
                let mut x_vals = Vec::new();
                let mut y_vals = Vec::new();
                let mut z_vals = Vec::new();
                for &(normal, d) in &planes {
                    if normal.x().abs() > 0.5 {
                        // d = normal · point, so the plane is at x = d/normal.x()
                        x_vals.push(d / normal.x());
                    } else if normal.y().abs() > 0.5 {
                        y_vals.push(d / normal.y());
                    } else {
                        z_vals.push(d / normal.z());
                    }
                }
                if x_vals.len() == 2 && y_vals.len() == 2 && z_vals.len() == 2 {
                    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    z_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    return Some(AnalyticClassifier::Box {
                        min: Point3::new(x_vals[0], y_vals[0], z_vals[0]),
                        max: Point3::new(x_vals[1], y_vals[1], z_vals[1]),
                    });
                }
            }
        }

        // All-planar solid that isn't an axis-aligned box: treat as convex
        // polyhedron (half-plane classifier). Works for hex prisms, wedges,
        // arbitrary extruded polygons, etc.
        // Only valid for CONVEX solids — verify by checking that the vertex
        // centroid is on the interior side of every face plane.
        if has_planar && !has_sphere && !has_cylinder && !has_cone {
            let faces = shell.faces();
            let mut planes = Vec::with_capacity(faces.len());
            // First pass: collect all planes.
            for &fid in faces {
                let face = topo.face(fid).ok()?;
                if let FaceSurface::Plane { normal, d } = face.surface() {
                    let (n, dv) = if face.is_reversed() {
                        (-*normal, -*d)
                    } else {
                        (*normal, *d)
                    };
                    planes.push((n, dv));
                } else {
                    return None;
                }
            }
            // Convexity check: every vertex must be on the interior side of
            // every face plane. This is O(V×F) but only runs for small all-planar
            // solids (hex prisms: V=12, F=8 → 96 dot products).
            let mut all_verts: Vec<Vec3> = Vec::new();
            for &fid in faces {
                let face = topo.face(fid).ok()?;
                let wire = topo.wire(face.outer_wire()).ok()?;
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge()).ok()?;
                    let v = topo.vertex(edge.start()).ok()?;
                    let pv = Vec3::new(v.point().x(), v.point().y(), v.point().z());
                    all_verts.push(pv);
                }
            }
            let convex_tol = tol.linear * 10.0; // Generous tolerance for vertex-on-face
            let is_convex = planes
                .iter()
                .all(|&(n, d)| all_verts.iter().all(|&v| n.dot(v) <= d + convex_tol));
            if is_convex {
                return Some(AnalyticClassifier::ConvexPolyhedron { planes });
            }
        }
    }

    // Pure sphere solid (all faces are sphere).
    if has_sphere && !has_planar && !has_cylinder {
        let (center, radius) = sphere_info?;
        return Some(AnalyticClassifier::Sphere { center, radius });
    }

    // Cylinder solid (1 cylinder face + 2 planar caps).
    if has_cylinder && has_planar && !has_sphere {
        let (origin, axis, radius) = cylinder_info?;
        // Compute axial extent from planar cap faces.
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for &fid in shell.faces() {
            let face = topo.face(fid).ok()?;
            if let FaceSurface::Plane { normal, d } = face.surface() {
                // The plane's d value along the cylinder axis gives the cap position.
                // For a cylinder cap, the plane normal is parallel to the axis.
                let dot = normal.dot(axis);
                if dot.abs() > 0.5 {
                    // Project the plane onto the axis to find the cap position.
                    // d = normal · point, so point along axis = d / dot for this axis.
                    // d/dot gives position from global origin; subtract
                    // the cylinder origin's projection to get axis-relative z.
                    let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());
                    let z = *d / dot - axis.dot(origin_vec);
                    z_min = z_min.min(z);
                    z_max = z_max.max(z);
                }
            }
        }
        if z_min < z_max {
            return Some(AnalyticClassifier::Cylinder {
                origin,
                axis,
                radius,
                z_min,
                z_max,
            });
        }
    }

    // Cone solid (1 cone face + 1 or 2 planar caps).
    if has_cone && has_planar && !has_sphere && !has_cylinder {
        let (apex, axis, _half_angle) = cone_info?;
        // Use the cone's axis as the reference axis, but measure z from a fixed
        // origin (the apex position) rather than relying on the apex being the
        // geometric virtual apex (which may be wrong for frustums).
        let origin = apex;
        let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());

        // Compute axial extent and radii from planar cap faces.
        // z values are measured from `origin` along `axis`.
        let mut caps: Vec<(f64, f64)> = Vec::new(); // (z_position, radius)
        for &fid in shell.faces() {
            let face = topo.face(fid).ok()?;
            if let FaceSurface::Plane { normal, d } = face.surface() {
                let dot = normal.dot(axis);
                if dot.abs() > 0.5 {
                    let z = *d / dot - axis.dot(origin_vec);
                    // Compute the cap radius from face vertices.
                    let wire = topo.wire(face.outer_wire()).ok()?;
                    let mut max_r_sq = 0.0_f64;
                    for oe in wire.edges() {
                        let edge = topo.edge(oe.edge()).ok()?;
                        for vid in [edge.start(), edge.end()] {
                            let v = topo.vertex(vid).ok()?;
                            let diff = v.point() - origin;
                            let axial_comp = axis * diff.dot(axis);
                            let radial = diff - axial_comp;
                            let r_sq = radial.x() * radial.x()
                                + radial.y() * radial.y()
                                + radial.z() * radial.z();
                            max_r_sq = max_r_sq.max(r_sq);
                        }
                    }
                    caps.push((z, max_r_sq.sqrt()));
                }
            }
        }

        // Sort caps by z and extract z_min/z_max with their radii.
        caps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let (mut z_min, mut z_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut r_at_z_min, mut r_at_z_max) = (0.0, 0.0);
        for &(z, r) in &caps {
            if z < z_min {
                z_min = z;
                r_at_z_min = r;
            }
            if z > z_max {
                z_max = z;
                r_at_z_max = r;
            }
        }

        // For pointed cones with only one cap.
        if z_min == f64::INFINITY {
            z_min = 0.0;
            r_at_z_min = 0.0;
        }
        if z_max == f64::NEG_INFINITY {
            z_max = 0.0;
            r_at_z_max = 0.0;
        }
        if (z_max - z_min).abs() > tol.linear {
            return Some(AnalyticClassifier::Cone {
                origin,
                axis,
                z_min,
                z_max,
                r_at_z_min,
                r_at_z_max,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// BVH construction
// ---------------------------------------------------------------------------

/// Build a BVH over face data for accelerated ray-cast classification.
///
/// Returns `None` when the face count is small enough that linear scan is
/// faster than BVH construction + traversal overhead.
pub(super) fn build_face_bvh(faces: &FaceData) -> Option<Bvh> {
    // Only worth building for ≥32 faces (BVH overhead vs linear scan).
    if faces.len() < 32 {
        return None;
    }
    let aabbs: Vec<(usize, Aabb3)> = faces
        .iter()
        .enumerate()
        .map(|(i, (_, verts, _, _))| {
            let bb = Aabb3::from_points(verts.iter().copied());
            (i, bb)
        })
        .collect();
    Some(Bvh::build(&aabbs))
}

// ---------------------------------------------------------------------------
// Fragment classification
// ---------------------------------------------------------------------------

/// Classify a face fragment relative to the opposite solid.
///
/// When `bvh` is `Some`, uses BVH-accelerated ray queries instead of
/// linearly scanning all faces. This reduces classification from O(F) to
/// O(log F) per fragment.
pub(super) fn classify_fragment(
    frag: &FaceFragment,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    let centroid = polygon_centroid(&frag.vertices);
    let class = classify_point(centroid, frag.normal, opposite, bvh, tol);
    guard_tangent_coplanar(class, &frag.vertices, frag.normal, opposite, bvh, tol)
}

/// Guard against false coplanar classifications at tangent touch points.
///
/// When a face touches a curved surface tangentially, the centroid may lie
/// exactly on the opposing surface with aligned normals, causing a false
/// Coplanar classification. This function verifies coplanar results by
/// checking whether most fragment vertices are also on the opposing face
/// planes. If fewer than half are, it's a tangent touch — re-classify via
/// ray-casting from a vertex that's clearly off the opposing surface.
pub(super) fn guard_tangent_coplanar(
    class: FaceClass,
    vertices: &[Point3],
    normal: Vec3,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    if !matches!(class, FaceClass::CoplanarSame | FaceClass::CoplanarOpposite) || vertices.len() < 3
    {
        return class;
    }

    // Check how many vertices are on the same plane as any opposing face.
    // Use plane distance only (not polygon containment) to avoid false
    // negatives from vertices on polygon edges.
    let mut on_plane_count = 0usize;
    for v in vertices {
        let on_any_plane = opposite.iter().any(|(_, _verts, n_opp, d_opp)| {
            let dist = dot_normal_point(*n_opp, *v) - d_opp;
            dist.abs() < tol.linear * 10.0
        });
        if on_any_plane {
            on_plane_count += 1;
        }
    }

    // If most vertices are on an opposing face plane, this is likely a true
    // coplanar situation — keep the original classification. Only override
    // when very few vertices (at most 1) are on any opposing plane, which
    // indicates a tangent touch at a point or line.
    if on_plane_count <= 1 {
        // Find a vertex that's NOT on any opposing face for reliable ray-cast.
        for v in vertices {
            let on_any = opposite.iter().any(|(_, _verts, n_opp, d_opp)| {
                let dist = dot_normal_point(*n_opp, *v) - d_opp;
                dist.abs() < tol.linear * 10.0
            });
            if !on_any {
                return classify_point(*v, normal, opposite, bvh, tol);
            }
        }
        // All vertices near some opposing plane — use centroid ray-cast.
        let centroid = polygon_centroid(vertices);
        return multiray_classify(centroid, normal, opposite, bvh, tol);
    }

    class
}

/// Classify a point (centroid with a ray direction) relative to an opposite solid.
///
/// This is the core classification logic, separated from `FaceFragment` to avoid
/// unnecessary cloning when the centroid/normal are already known.
pub(super) fn classify_point(
    centroid: Point3,
    normal: Vec3,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    // First check for coplanar faces — must scan candidates only.
    // For coplanar test we need faces near the centroid's plane, so use
    // BVH point-containment if available, otherwise linear scan.
    let coplanar_indices: Vec<usize> = if let Some(bvh) = bvh {
        let probe = Aabb3 {
            min: centroid + Vec3::new(-tol.linear, -tol.linear, -tol.linear),
            max: centroid + Vec3::new(tol.linear, tol.linear, tol.linear),
        };
        bvh.query_overlap(&probe)
    } else {
        (0..opposite.len()).collect()
    };

    for &i in &coplanar_indices {
        let (_, ref verts, n_opp, d_opp) = opposite[i];
        // Skip if the centroid coincides with a vertex of the opposing face.
        // This prevents false coplanar matches when the centroid is at a
        // singular point (e.g. cone apex, sphere pole) that is a vertex of
        // tessellated face data. At such vertices, the centroid lies on the
        // triangle plane (distance = 0) but the face is NOT truly coplanar.
        // Use a tight threshold (10× tolerance) to avoid interfering with
        // legitimate near-touching geometry.
        let near_vertex = verts.iter().any(|v| {
            let dx = centroid.x() - v.x();
            let dy = centroid.y() - v.y();
            let dz = centroid.z() - v.z();
            dx * dx + dy * dy + dz * dz < tol.linear * 10.0 * tol.linear * 10.0
        });
        if near_vertex {
            continue;
        }
        let dist = dot_normal_point(n_opp, centroid) - d_opp;
        if dist.abs() < tol.linear && point_in_face_3d(centroid, verts, &n_opp) {
            let dot = normal.dot(n_opp);
            return if dot > tol.angular {
                FaceClass::CoplanarSame
            } else if dot < -tol.angular {
                FaceClass::CoplanarOpposite
            } else {
                // Normals are nearly perpendicular — treat as non-coplanar.
                // Fall through to ray-cast classification.
                continue;
            };
        }
    }

    multiray_classify(centroid, normal, opposite, bvh, tol)
}

// ---------------------------------------------------------------------------
// Ray-casting helpers
// ---------------------------------------------------------------------------

/// Test a single face against a ray for crossing parity.
///
/// Returns +1 for a front-to-back crossing, -1 for back-to-front, or 0 for
/// no intersection (parallel, behind origin, or outside polygon).
#[inline]
fn ray_face_crossing(
    centroid: Point3,
    ray_dir: Vec3,
    verts: &[Point3],
    n_opp: Vec3,
    d_opp: f64,
    tol: Tolerance,
) -> i32 {
    let denom = n_opp.dot(ray_dir);
    if denom.abs() < tol.angular {
        return 0;
    }
    let numer = d_opp - dot_normal_point(n_opp, centroid);
    let t = numer / denom;
    if t <= tol.linear {
        return 0;
    }
    let hit = point_along_line(&centroid, &ray_dir, t);
    if point_in_face_3d(hit, verts, &n_opp) {
        if denom > 0.0 { -1 } else { 1 }
    } else {
        0
    }
}

/// Multi-ray inside/outside classification via majority vote.
///
/// Casts 3 rays (the given normal + two ~55° rotations) and counts boundary
/// crossings. Returns Inside if 2+ of 3 rays report an odd crossing count.
/// This is the shared implementation used by both `classify_point` and the
/// tangent-coplanar guard's fallback.
fn multiray_classify(
    point: Point3,
    normal: Vec3,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    let ray_dirs = {
        let perp = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let cross_vec = normal.cross(perp);
        let axis_len = cross_vec.length();
        // Numerical-zero guard: if the cross product has near-zero length,
        // the fragment normal is nearly parallel to the chosen perpendicular
        // vector — fall back to using the normal itself for all 3 ray
        // directions (degenerate but safe; the majority vote still works).
        if axis_len < 1e-12 {
            [normal, normal, normal]
        } else {
            let inv = 1.0 / axis_len;
            let axis = Vec3::new(
                cross_vec.x() * inv,
                cross_vec.y() * inv,
                cross_vec.z() * inv,
            );
            let rodrigues = |cos_a: f64, sin_a: f64| -> Vec3 {
                let dot = axis.dot(normal);
                let cross = axis.cross(normal);
                Vec3::new(
                    normal.x().mul_add(
                        cos_a,
                        cross.x().mul_add(sin_a, axis.x() * dot * (1.0 - cos_a)),
                    ),
                    normal.y().mul_add(
                        cos_a,
                        cross.y().mul_add(sin_a, axis.y() * dot * (1.0 - cos_a)),
                    ),
                    normal.z().mul_add(
                        cos_a,
                        cross.z().mul_add(sin_a, axis.z() * dot * (1.0 - cos_a)),
                    ),
                )
            };
            // cos(55°) ≈ 0.574, sin(55°) ≈ 0.819
            [normal, rodrigues(0.574, 0.819), rodrigues(0.574, -0.819)]
        }
    };

    // Reuse a single candidate buffer across all 3 ray queries to avoid
    // per-ray Vec allocation. The buffer grows on the first query and is
    // cleared+reused on subsequent queries.
    let mut inside_votes = 0u8;
    let mut candidates = Vec::new();
    for ray_dir in &ray_dirs {
        let mut crossings = 0i32;
        if let Some(bvh) = bvh {
            bvh.query_ray_into(point, *ray_dir, &mut candidates);
            for &idx in &candidates {
                let (_, ref verts, n_opp, d_opp) = opposite[idx];
                crossings += ray_face_crossing(point, *ray_dir, verts, n_opp, d_opp, tol);
            }
        } else {
            for &(_, ref verts, n_opp, d_opp) in opposite {
                crossings += ray_face_crossing(point, *ray_dir, verts, n_opp, d_opp, tol);
            }
        }
        if crossings != 0 {
            inside_votes += 1;
        }
    }

    if inside_votes >= 2 {
        FaceClass::Inside
    } else {
        FaceClass::Outside
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Compute the centroid of a polygon.
///
/// Returns the origin if the polygon is empty (should not happen in
/// practice since fragments are filtered for `len >= 3`).
#[inline]
pub(super) fn polygon_centroid(vertices: &[Point3]) -> Point3 {
    if vertices.is_empty() {
        return Point3::new(0.0, 0.0, 0.0);
    }
    #[allow(clippy::cast_precision_loss)] // polygon vertex counts fit in f64
    let inv_n = 1.0 / vertices.len() as f64;
    let (sx, sy, sz) = vertices.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), v| {
        (ax + v.x(), ay + v.y(), az + v.z())
    });
    Point3::new(sx * inv_n, sy * inv_n, sz * inv_n)
}

/// Test if a 3D point lies inside a planar face polygon by projecting to 2D.
#[inline]
pub(super) fn point_in_face_3d(point: Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    // Project to 2D by dropping the coordinate corresponding to the largest
    // normal component. This avoids degenerate projections.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let (project_point, project_polygon): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
        (
            Point2::new(point.x(), point.y()),
            polygon.iter().map(|p| Point2::new(p.x(), p.y())).collect(),
        )
    } else if ay >= ax {
        (
            Point2::new(point.x(), point.z()),
            polygon.iter().map(|p| Point2::new(p.x(), p.z())).collect(),
        )
    } else {
        (
            Point2::new(point.y(), point.z()),
            polygon.iter().map(|p| Point2::new(p.y(), p.z())).collect(),
        )
    };

    point_in_polygon(project_point, &project_polygon)
}
