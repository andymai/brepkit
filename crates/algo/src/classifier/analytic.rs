//! Analytic O(1) point-in-solid classification (canonical implementation).
//!
//! For convex solids composed entirely of analytic surfaces (plane,
//! cylinder, cone, sphere), a point can be classified by testing
//! the signed distance to each face constraint. Originally ported from
//! `operations/boolean/classify.rs`.
//!
//! NOTE: `operations/boolean/classify.rs` contains a duplicate of this
//! logic. Bug fixes should be applied here first; the operations copy
//! will be deleted during the GFA step 5 switchover.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;

// ---------------------------------------------------------------------------
// Analytic classifier enum
// ---------------------------------------------------------------------------

/// Analytic classifier for simple convex solids.
///
/// Instead of ray-casting against tessellated triangles, uses exact
/// geometric predicates to classify points inside/outside a solid.
pub enum AnalyticClassifier {
    /// Point-in-sphere: `|p - center| <= radius`.
    Sphere {
        /// Sphere center.
        center: Point3,
        /// Sphere radius.
        radius: f64,
    },
    /// Point-in-cylinder: radial distance from axis <= radius AND axial
    /// position within [z_min, z_max].
    Cylinder {
        /// Cylinder axis origin.
        origin: Point3,
        /// Cylinder axis direction (unit).
        axis: Vec3,
        /// Cylinder radius.
        radius: f64,
        /// Minimum axial position.
        z_min: f64,
        /// Maximum axial position.
        z_max: f64,
    },
    /// Point-in-cone-frustum: radial distance from axis <= interpolated radius
    /// AND axial position within [z_min, z_max].
    Cone {
        /// Cone apex (axis origin).
        origin: Point3,
        /// Cone axis direction (unit).
        axis: Vec3,
        /// Minimum axial position.
        z_min: f64,
        /// Maximum axial position.
        z_max: f64,
        /// Radius at `z_min`.
        r_at_z_min: f64,
        /// Radius at `z_max`.
        r_at_z_max: f64,
    },
    /// Point-in-box: axis-aligned bounding box test.
    Box {
        /// Box minimum corner.
        min: Point3,
        /// Box maximum corner.
        max: Point3,
    },
    /// Point-in-convex-polyhedron: half-plane test against each face.
    ConvexPolyhedron {
        /// Outward-pointing normals and signed distances.
        planes: Vec<(Vec3, f64)>,
    },
    /// General convex analytic solid: intersection of half-planes, cylinders,
    /// and cone frustums.
    ConvexAnalytic {
        /// Half-plane constraints: `normal . p < d` means inside.
        planes: Vec<(Vec3, f64)>,
        /// Cylinder constraints: `(origin, axis, radius, z_min, z_max)`.
        cylinders: Vec<(Point3, Vec3, f64, f64, f64)>,
        /// Cone frustum constraints: `(origin, axis, z_min, z_max, r_min, r_max)`.
        cones: Vec<(Point3, Vec3, f64, f64, f64, f64)>,
    },
    /// Composite classifier for shelled/hollow solids.
    Composite {
        /// Outer boundary classifier.
        outer: std::boxed::Box<Self>,
        /// Inner cavity classifier.
        inner: std::boxed::Box<Self>,
    },
}

impl AnalyticClassifier {
    /// Classify a point as Inside, Outside, or On (within tolerance of the
    /// boundary).
    #[must_use]
    pub fn classify(&self, centroid: Point3, tol: Tolerance) -> Option<FaceClass> {
        match self {
            Self::Sphere { center, radius } => {
                let dx = centroid.x() - center.x();
                let dy = centroid.y() - center.y();
                let dz = centroid.z() - center.z();
                let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                if dist_sq < (radius - tol.linear) * (radius - tol.linear) {
                    Some(FaceClass::Inside)
                } else if dist_sq > (radius + tol.linear) * (radius + tol.linear) {
                    Some(FaceClass::Outside)
                } else {
                    None
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
                if axial < *z_min - tol.linear || axial > *z_max + tol.linear {
                    return Some(FaceClass::Outside);
                }
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
                    None
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
                if axial < *z_min - tol.linear || axial > *z_max + tol.linear {
                    return Some(FaceClass::Outside);
                }
                let projected = *axis * axial;
                let radial_vec = diff - projected;
                let radial_dist_sq = radial_vec.x() * radial_vec.x()
                    + radial_vec.y() * radial_vec.y()
                    + radial_vec.z() * radial_vec.z();
                let dz = z_max - z_min;
                let t = if dz.abs() > tol.linear {
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
                    None
                }
            }
            Self::Box { min, max } => {
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
                    None
                }
            }
            Self::ConvexPolyhedron { planes } => {
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
                    None
                }
            }
            Self::ConvexAnalytic {
                planes,
                cylinders,
                cones,
            } => Some(classify_convex_analytic(
                centroid, tol, planes, cylinders, cones,
            )),
            Self::Composite { outer, inner } => {
                let outer_class = outer.classify(centroid, tol);
                match outer_class {
                    Some(FaceClass::Outside) => Some(FaceClass::Outside),
                    Some(FaceClass::Inside) => {
                        let inner_class = inner.classify(centroid, tol);
                        match inner_class {
                            Some(FaceClass::Inside) => Some(FaceClass::Outside),
                            Some(FaceClass::Outside) => Some(FaceClass::Inside),
                            // Inner boundary → on the boundary of the composite
                            None => None,
                            _ => None,
                        }
                    }
                    // Outer boundary → on the boundary of the composite
                    None => None,
                    _ => None,
                }
            }
        }
    }
}

/// Classify against combined plane + cylinder + cone constraints.
fn classify_convex_analytic(
    centroid: Point3,
    tol: Tolerance,
    planes: &[(Vec3, f64)],
    cylinders: &[(Point3, Vec3, f64, f64, f64)],
    cones: &[(Point3, Vec3, f64, f64, f64, f64)],
) -> FaceClass {
    let tl = tol.linear;
    let cv = Vec3::new(centroid.x(), centroid.y(), centroid.z());

    let mut max_plane_dist = f64::NEG_INFINITY;
    for &(normal, d) in planes {
        let signed_dist = normal.dot(cv) - d;
        max_plane_dist = max_plane_dist.max(signed_dist);
    }

    let mut max_cyl_excess = f64::NEG_INFINITY;
    for &(origin, axis, radius, z_min, z_max) in cylinders {
        let diff = centroid - origin;
        let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
        let axial = diff_v.dot(axis);
        if axial < z_min - tl || axial > z_max + tl {
            return FaceClass::Outside;
        }
        let projected = axis * axial;
        let radial_vec = diff_v - projected;
        let radial_dist = radial_vec.length();
        max_cyl_excess = max_cyl_excess.max(radial_dist - radius);
    }

    let mut max_cone_excess = f64::NEG_INFINITY;
    for &(origin, axis, z_min, z_max, r_min, r_max) in cones {
        let diff = centroid - origin;
        let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
        let axial = diff_v.dot(axis);
        if axial < z_min - tl || axial > z_max + tl {
            return FaceClass::Outside;
        }
        let dz = z_max - z_min;
        let t = if dz.abs() > tol.linear {
            (axial - z_min) / dz
        } else {
            0.5
        };
        let expected_r = r_min + t * (r_max - r_min);
        let projected = axis * axial;
        let radial_vec = diff_v - projected;
        let radial_dist = radial_vec.length();
        max_cone_excess = max_cone_excess.max(radial_dist - expected_r);
    }

    let max_excess = max_plane_dist.max(max_cyl_excess).max(max_cone_excess);
    if max_excess < -tl {
        FaceClass::Inside
    } else if max_excess > tl {
        FaceClass::Outside
    } else {
        FaceClass::On
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Try to classify a point using analytic geometry.
///
/// Returns `Some(FaceClass)` if the solid is a convex analytic solid
/// and the point can be classified without tessellation. Returns `None`
/// if the solid is not suitable for analytic classification.
#[must_use]
pub fn classify_analytic(topo: &Topology, solid: SolidId, point: Point3) -> Option<FaceClass> {
    let classifier = try_build_analytic_classifier(topo, solid)?;
    let tol = Tolerance::new();
    classifier.classify(point, tol)
}

// ---------------------------------------------------------------------------
// Analytic classifier construction
// ---------------------------------------------------------------------------

/// Try to build an analytic classifier for a solid.
///
/// Returns `Some` when the solid is a simple convex analytic shape
/// that supports O(1) point-in-solid tests. Falls back to `None` for
/// complex or non-analytic solids.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn try_build_analytic_classifier(
    topo: &Topology,
    solid: SolidId,
) -> Option<AnalyticClassifier> {
    let s = topo.solid(solid).ok()?;
    let shell = topo.shell(s.outer_shell()).ok()?;
    let tol = Tolerance::new();

    if shell.faces().len() > 50 {
        return None;
    }

    // Detect shelled/hollow solids via reversed faces.
    let has_reversed = shell
        .faces()
        .iter()
        .any(|&fid| topo.face(fid).ok().is_some_and(Face::is_reversed));
    if has_reversed {
        return try_build_composite_classifier(topo, solid);
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
            FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => return None,
        }
    }

    // Pure planar solid — try axis-aligned box or convex polyhedron.
    if has_planar && !has_sphere && !has_cylinder && !has_cone {
        return try_build_planar_classifier(topo, solid, shell.faces(), &tol);
    }

    // Pure sphere.
    if has_sphere && !has_planar && !has_cylinder {
        let (center, radius) = sphere_info?;
        return Some(AnalyticClassifier::Sphere { center, radius });
    }

    // Cylinder + plane caps.
    if has_cylinder && has_planar && !has_sphere {
        if let Some(c) = try_build_cylinder_classifier(topo, shell.faces(), cylinder_info?, &tol) {
            return Some(c);
        }
    }

    // Cone + plane caps.
    if has_cone && has_planar && !has_sphere && !has_cylinder {
        if let Some(c) = try_build_cone_classifier(topo, shell.faces(), cone_info?, &tol) {
            return Some(c);
        }
    }

    // Mixed plane+cone/cylinder: try ConvexAnalytic.
    if has_planar && (has_cone || has_cylinder) && !has_sphere {
        return try_build_convex_analytic(topo, solid);
    }

    None
}

// ---------------------------------------------------------------------------
// Sub-builders
// ---------------------------------------------------------------------------

/// Try to build a classifier for an all-planar solid.
#[allow(clippy::too_many_lines)]
fn try_build_planar_classifier(
    topo: &Topology,
    solid: SolidId,
    faces: &[brepkit_topology::face::FaceId],
    tol: &Tolerance,
) -> Option<AnalyticClassifier> {
    // Try axis-aligned box (exactly 6 faces).
    if faces.len() == 6 {
        if let Some(c) = try_build_box_classifier(topo, faces, tol) {
            return Some(c);
        }
    }

    // Try convex polyhedron.
    let mut planes = Vec::with_capacity(faces.len());
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

    // Convexity check: every vertex must be on the interior side of every plane.
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
    let convex_tol = tol.linear * 10.0;
    let is_convex = planes
        .iter()
        .all(|&(n, d)| all_verts.iter().all(|&v| n.dot(v) <= d + convex_tol));
    if is_convex {
        return Some(AnalyticClassifier::ConvexPolyhedron { planes });
    }

    // Non-convex all-planar solid — try composite.
    try_build_composite_classifier(topo, solid)
}

/// Try to build an axis-aligned box classifier from 6 plane faces.
fn try_build_box_classifier(
    topo: &Topology,
    faces: &[brepkit_topology::face::FaceId],
    tol: &Tolerance,
) -> Option<AnalyticClassifier> {
    let mut planes: Vec<(Vec3, f64)> = Vec::with_capacity(6);
    for &fid in faces {
        let face = topo.face(fid).ok()?;
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let ax = normal.x().abs();
            let ay = normal.y().abs();
            let az = normal.z().abs();
            if (ax > 1.0 - tol.angular && ay < tol.angular && az < tol.angular)
                || (ay > 1.0 - tol.angular && ax < tol.angular && az < tol.angular)
                || (az > 1.0 - tol.angular && ax < tol.angular && ay < tol.angular)
            {
                planes.push((*normal, *d));
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    if planes.len() != 6 {
        return None;
    }

    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut z_vals = Vec::new();
    for &(normal, d) in &planes {
        if normal.x().abs() > 0.5 {
            x_vals.push(d / normal.x());
        } else if normal.y().abs() > 0.5 {
            y_vals.push(d / normal.y());
        } else {
            z_vals.push(d / normal.z());
        }
    }
    if x_vals.len() != 2 || y_vals.len() != 2 || z_vals.len() != 2 {
        return None;
    }
    let sort =
        |v: &mut Vec<f64>| v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sort(&mut x_vals);
    sort(&mut y_vals);
    sort(&mut z_vals);
    Some(AnalyticClassifier::Box {
        min: Point3::new(x_vals[0], y_vals[0], z_vals[0]),
        max: Point3::new(x_vals[1], y_vals[1], z_vals[1]),
    })
}

/// Try to build a cylinder classifier from cylinder + plane caps.
fn try_build_cylinder_classifier(
    topo: &Topology,
    faces: &[brepkit_topology::face::FaceId],
    (origin, axis, radius): (Point3, Vec3, f64),
    _tol: &Tolerance,
) -> Option<AnalyticClassifier> {
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for &fid in faces {
        let face = topo.face(fid).ok()?;
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let dot = normal.dot(axis);
            if dot.abs() > 0.5 {
                let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());
                let z = *d / dot - axis.dot(origin_vec);
                z_min = z_min.min(z);
                z_max = z_max.max(z);
            }
        }
    }
    if z_min < z_max {
        Some(AnalyticClassifier::Cylinder {
            origin,
            axis,
            radius,
            z_min,
            z_max,
        })
    } else {
        None
    }
}

/// Try to build a cone classifier from cone + plane caps.
#[allow(clippy::too_many_lines)]
fn try_build_cone_classifier(
    topo: &Topology,
    faces: &[brepkit_topology::face::FaceId],
    (apex, axis, _half_angle): (Point3, Vec3, f64),
    tol: &Tolerance,
) -> Option<AnalyticClassifier> {
    let origin = apex;
    let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());

    let mut caps: Vec<(f64, f64)> = Vec::new();
    for &fid in faces {
        let face = topo.face(fid).ok()?;
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let dot = normal.dot(axis);
            if dot.abs() > 0.5 {
                let z = *d / dot - axis.dot(origin_vec);
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

    if !z_min.is_finite() {
        z_min = 0.0;
        r_at_z_min = 0.0;
    }
    if !z_max.is_finite() {
        z_max = 0.0;
        r_at_z_max = 0.0;
    }

    if (z_max - z_min).abs() > tol.linear {
        Some(AnalyticClassifier::Cone {
            origin,
            axis,
            z_min,
            z_max,
            r_at_z_min,
            r_at_z_max,
        })
    } else {
        None
    }
}

/// Build a `ConvexAnalytic` classifier from a convex solid with mixed surface types.
#[allow(clippy::too_many_lines)]
fn try_build_convex_analytic(topo: &Topology, solid: SolidId) -> Option<AnalyticClassifier> {
    let s = topo.solid(solid).ok()?;
    let shell = topo.shell(s.outer_shell()).ok()?;
    let tol = Tolerance::new();

    let mut planes: Vec<(Vec3, f64)> = Vec::new();
    let mut cylinders: Vec<(Point3, Vec3, f64, f64, f64)> = Vec::new();
    let mut cones: Vec<(Point3, Vec3, f64, f64, f64, f64)> = Vec::new();

    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                let (n, dv) = if face.is_reversed() {
                    (-*normal, -*d)
                } else {
                    (*normal, *d)
                };
                planes.push((n, dv));
            }
            FaceSurface::Cylinder(cyl) => {
                let origin = cyl.origin();
                let axis = cyl.axis();
                let r = cyl.radius();
                let origin_v = Vec3::new(origin.x(), origin.y(), origin.z());
                let wire = topo.wire(face.outer_wire()).ok()?;
                let (z_min, z_max) = wire_axial_extent(topo, wire, origin_v, axis)?;
                cylinders.push((origin, axis, r, z_min, z_max));
            }
            FaceSurface::Cone(con) => {
                let apex = con.apex();
                let axis = con.axis();
                let apex_v = Vec3::new(apex.x(), apex.y(), apex.z());
                let wire = topo.wire(face.outer_wire()).ok()?;
                let (z_min, z_max, r_min, r_max) = wire_cone_extent(topo, wire, apex_v, axis)?;
                cones.push((apex, axis, z_min, z_max, r_min, r_max));
            }
            // Sphere, Torus, and NURBS faces are not supported by the
            // ConvexAnalytic classifier — bail out to ray-cast.
            FaceSurface::Sphere(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => return None,
        }
    }

    if planes.is_empty() {
        return None;
    }

    // Convexity check: vertex centroid must be inside all constraints.
    let mut centroid = Vec3::new(0.0, 0.0, 0.0);
    let mut vert_count = 0u32;
    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        let wire = topo.wire(face.outer_wire()).ok()?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).ok()?;
            let v = topo.vertex(edge.start()).ok()?;
            let p = v.point();
            centroid += Vec3::new(p.x(), p.y(), p.z());
            vert_count += 1;
        }
    }
    if vert_count == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let centroid = centroid * (1.0 / vert_count as f64);
    let centroid_pt = Point3::new(centroid.x(), centroid.y(), centroid.z());

    for &(normal, d) in &planes {
        if normal.dot(centroid) - d > tol.linear {
            return None;
        }
    }
    for &(origin, axis, radius, z_min, z_max) in &cylinders {
        let diff = centroid_pt - origin;
        let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
        let axial = diff_v.dot(axis);
        if axial < z_min - tol.linear || axial > z_max + tol.linear {
            return None;
        }
        let projected = axis * axial;
        if (diff_v - projected).length() > radius + tol.linear {
            return None;
        }
    }
    for &(origin, axis, z_min, z_max, r_min, r_max) in &cones {
        let diff = centroid_pt - origin;
        let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
        let axial = diff_v.dot(axis);
        if axial < z_min - tol.linear || axial > z_max + tol.linear {
            return None;
        }
        let dz = z_max - z_min;
        let t = if dz.abs() > tol.linear {
            (axial - z_min) / dz
        } else {
            0.5
        };
        let expected_r = r_min + t * (r_max - r_min);
        let projected = axis * axial;
        if (diff_v - projected).length() > expected_r + tol.linear {
            return None;
        }
    }

    Some(AnalyticClassifier::ConvexAnalytic {
        planes,
        cylinders,
        cones,
    })
}

// ---------------------------------------------------------------------------
// Composite classifier
// ---------------------------------------------------------------------------

/// Try to build a composite classifier for a shelled/hollow solid.
#[allow(clippy::too_many_lines)]
fn try_build_composite_classifier(topo: &Topology, solid: SolidId) -> Option<AnalyticClassifier> {
    let s = topo.solid(solid).ok()?;
    let shell = topo.shell(s.outer_shell()).ok()?;
    let tol = Tolerance::new();

    // Compute vertex centroid for inner/outer classification.
    let centroid = {
        let mut c = Vec3::new(0.0, 0.0, 0.0);
        let mut count = 0u32;
        for &fid in shell.faces() {
            let face = topo.face(fid).ok()?;
            let wire = topo.wire(face.outer_wire()).ok()?;
            for oe in wire.edges() {
                let e = topo.edge(oe.edge()).ok()?;
                let p = topo.vertex(e.start()).ok()?.point();
                c += Vec3::new(p.x(), p.y(), p.z());
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0 / count as f64;
        Point3::new(c.x() * inv, c.y() * inv, c.z() * inv)
    };

    let mut outer_planes: Vec<(Vec3, f64)> = Vec::new();
    let mut inner_planes: Vec<(Vec3, f64)> = Vec::new();
    let mut outer_cylinders: Vec<(Point3, Vec3, f64, f64, f64)> = Vec::new();
    let mut inner_cylinders: Vec<(Point3, Vec3, f64, f64, f64)> = Vec::new();
    let mut outer_cones: Vec<(Point3, Vec3, f64, f64, f64, f64)> = Vec::new();
    let mut inner_cones: Vec<(Point3, Vec3, f64, f64, f64, f64)> = Vec::new();

    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                let (n, dv) = if face.is_reversed() {
                    (-*normal, -*d)
                } else {
                    (*normal, *d)
                };
                let cv = Vec3::new(centroid.x(), centroid.y(), centroid.z());
                let signed_dist = n.dot(cv) - dv;
                if signed_dist < 0.0 {
                    outer_planes.push((n, dv));
                } else {
                    inner_planes.push((n, dv));
                }
            }
            FaceSurface::Cylinder(cyl) => {
                let origin = cyl.origin();
                let axis = cyl.axis();
                let r = cyl.radius();
                let origin_v = Vec3::new(origin.x(), origin.y(), origin.z());
                let wire = topo.wire(face.outer_wire()).ok()?;
                let (z_min, z_max) = wire_axial_extent(topo, wire, origin_v, axis)?;
                let diff = centroid - origin;
                let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
                let projected = axis * diff_v.dot(axis);
                let radial_dist = (diff_v - projected).length();
                if radial_dist < r {
                    outer_cylinders.push((origin, axis, r, z_min, z_max));
                } else {
                    inner_cylinders.push((origin, axis, r, z_min, z_max));
                }
            }
            FaceSurface::Cone(con) => {
                let apex = con.apex();
                let axis = con.axis();
                let apex_v = Vec3::new(apex.x(), apex.y(), apex.z());
                let wire = topo.wire(face.outer_wire()).ok()?;
                let (z_min, z_max, r_min, r_max) = wire_cone_extent(topo, wire, apex_v, axis)?;
                let diff = centroid - apex;
                let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
                let axial = diff_v.dot(axis);
                let dz = z_max - z_min;
                let t = if dz.abs() > tol.linear {
                    ((axial - z_min) / dz).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let expected_r = r_min + t * (r_max - r_min);
                let projected = axis * axial;
                let radial_dist = (diff_v - projected).length();
                if radial_dist < expected_r {
                    outer_cones.push((apex, axis, z_min, z_max, r_min, r_max));
                } else {
                    inner_cones.push((apex, axis, z_min, z_max, r_min, r_max));
                }
            }
            // Sphere, Torus, NURBS — skip for composite classifier
            FaceSurface::Sphere(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => {}
        }
    }

    let build_box = |planes: &[(Vec3, f64)]| -> Option<AnalyticClassifier> {
        if planes.len() < 4 {
            return None;
        }
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut z_vals = Vec::new();
        for &(normal, d) in planes {
            if normal.x().abs() > 0.5 {
                x_vals.push(d / normal.x());
            } else if normal.y().abs() > 0.5 {
                y_vals.push(d / normal.y());
            } else if normal.z().abs() > 0.5 {
                z_vals.push(d / normal.z());
            }
        }
        if x_vals.is_empty() || y_vals.is_empty() || z_vals.is_empty() {
            return None;
        }
        let sort = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        };
        sort(&mut x_vals);
        sort(&mut y_vals);
        sort(&mut z_vals);
        let x_min = *x_vals.first()?;
        let x_max = if x_vals.len() >= 2 {
            *x_vals.last()?
        } else {
            x_min + 1e6
        };
        let y_min = *y_vals.first()?;
        let y_max = if y_vals.len() >= 2 {
            *y_vals.last()?
        } else {
            y_min + 1e6
        };
        let z_min = *z_vals.first()?;
        let z_max = if z_vals.len() >= 2 {
            *z_vals.last()?
        } else {
            z_min + 1e6
        };
        Some(AnalyticClassifier::Box {
            min: Point3::new(x_min, y_min, z_min),
            max: Point3::new(x_max, y_max, z_max),
        })
    };

    let build_classifier = |planes: &[(Vec3, f64)],
                            cylinders: &[(Point3, Vec3, f64, f64, f64)],
                            cones: &[(Point3, Vec3, f64, f64, f64, f64)]|
     -> Option<AnalyticClassifier> {
        if (!cylinders.is_empty() || !cones.is_empty()) && planes.len() >= 2 {
            Some(AnalyticClassifier::ConvexAnalytic {
                planes: planes.to_vec(),
                cylinders: cylinders.to_vec(),
                cones: cones.to_vec(),
            })
        } else {
            build_box(planes)
        }
    };

    let outer = build_classifier(&outer_planes, &outer_cylinders, &outer_cones)?;
    let inner = build_classifier(&inner_planes, &inner_cylinders, &inner_cones)?;

    Some(AnalyticClassifier::Composite {
        outer: std::boxed::Box::new(outer),
        inner: std::boxed::Box::new(inner),
    })
}

// ---------------------------------------------------------------------------
// Wire geometry helpers
// ---------------------------------------------------------------------------

/// Compute axial extent (z_min, z_max) of a wire's vertices along an axis.
fn wire_axial_extent(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
    origin: Vec3,
    axis: Vec3,
) -> Option<(f64, f64)> {
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        for vid in [edge.start(), edge.end()] {
            let v = topo.vertex(vid).ok()?;
            let diff = v.point() - Point3::new(origin.x(), origin.y(), origin.z());
            let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
            let z = diff_v.dot(axis);
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }
    }
    if z_min.is_finite() && z_max.is_finite() {
        Some((z_min, z_max))
    } else {
        None
    }
}

/// Compute axial extent + radius range for a cone face's wire.
fn wire_cone_extent(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
    apex: Vec3,
    axis: Vec3,
) -> Option<(f64, f64, f64, f64)> {
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut r_at_zmin = 0.0_f64;
    let mut r_at_zmax = 0.0_f64;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        for vid in [edge.start(), edge.end()] {
            let v = topo.vertex(vid).ok()?;
            let diff = v.point() - Point3::new(apex.x(), apex.y(), apex.z());
            let diff_v = Vec3::new(diff.x(), diff.y(), diff.z());
            let z = diff_v.dot(axis);
            let projected = axis * z;
            let radial = diff_v - projected;
            let r = radial.length();
            if z < z_min {
                z_min = z;
                r_at_zmin = r;
            }
            if z > z_max {
                z_max = z;
                r_at_zmax = r;
            }
        }
    }
    if z_min.is_finite() && z_max.is_finite() {
        Some((z_min, z_max, r_at_zmin, r_at_zmax))
    } else {
        None
    }
}
