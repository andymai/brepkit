//! Measurement operations for B-rep geometry: bounding boxes, areas,
//! volumes, and centers of mass.

use std::collections::HashSet;

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::tessellate;

// ── Bounding box ──────────────────────────────────────────────────

/// Compute the axis-aligned bounding box of a solid.
///
/// For planar solids, uses vertex positions (exact). For solids with
/// analytic surfaces (sphere, cylinder, cone, torus), expands the AABB
/// to include surface extremes. For NURBS, uses control-point hulls.
///
/// # Errors
///
/// Returns an error if the solid has no vertices or a topology lookup fails.
pub fn solid_bounding_box(
    topo: &Topology,
    solid: SolidId,
) -> Result<Aabb3, crate::OperationsError> {
    let points = collect_solid_vertex_points(topo, solid)?;
    let mut aabb = Aabb3::try_from_points(points.iter().copied()).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "solid has no vertices".into(),
        }
    })?;

    // Expand AABB for analytic surfaces whose extremes lie beyond vertices.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    for &fid in shell.faces() {
        if let Ok(face) = topo.face(fid) {
            expand_aabb_for_surface(&mut aabb, face.surface());
        }
    }

    Ok(aabb)
}

/// Expand an AABB to include a point.
fn aabb_include(aabb: &mut Aabb3, p: Point3) {
    *aabb = aabb.union(Aabb3 { min: p, max: p });
}

/// Expand an AABB to include surface-specific extremes that vertices miss.
pub(crate) fn expand_aabb_for_surface(aabb: &mut Aabb3, surface: &FaceSurface) {
    match surface {
        FaceSurface::Sphere(s) => {
            let c = s.center();
            let r = s.radius();
            aabb_include(aabb, Point3::new(c.x() - r, c.y() - r, c.z() - r));
            aabb_include(aabb, Point3::new(c.x() + r, c.y() + r, c.z() + r));
        }
        FaceSurface::Cylinder(c) => {
            let origin = c.origin();
            let axis = c.axis();
            let r = c.radius();
            // Expand by ±r only in radial directions (perpendicular to the
            // cylinder axis). The axial extent is already covered by vertices.
            // For each world axis i, the maximum radial reach is r * sqrt(1 - axis_i²).
            let rx = r * (1.0 - axis.x() * axis.x()).max(0.0).sqrt();
            let ry = r * (1.0 - axis.y() * axis.y()).max(0.0).sqrt();
            let rz = r * (1.0 - axis.z() * axis.z()).max(0.0).sqrt();
            for corner in [aabb.min, aabb.max] {
                let rel = Vec3::new(
                    corner.x() - origin.x(),
                    corner.y() - origin.y(),
                    corner.z() - origin.z(),
                );
                let t = axis.dot(rel);
                let coa = Point3::new(
                    origin.x() + axis.x() * t,
                    origin.y() + axis.y() * t,
                    origin.z() + axis.z() * t,
                );
                aabb_include(aabb, Point3::new(coa.x() - rx, coa.y() - ry, coa.z() - rz));
                aabb_include(aabb, Point3::new(coa.x() + rx, coa.y() + ry, coa.z() + rz));
            }
        }
        FaceSurface::Torus(t) => {
            // Use the torus's actual axis to compute correct AABB extents.
            let c = t.center();
            let outer_r = t.major_radius() + t.minor_radius();
            let axis = t.z_axis();
            // Axial extent: minor_radius along the torus axis.
            let axial_offset = Vec3::new(
                axis.x() * t.minor_radius(),
                axis.y() * t.minor_radius(),
                axis.z() * t.minor_radius(),
            );
            // Radial extent: outer_r in the equatorial plane (perpendicular to axis).
            // Worst case for each AABB axis is outer_r * sqrt(1 - axis_component^2),
            // but conservatively use outer_r for all axes.
            aabb_include(
                aabb,
                Point3::new(
                    c.x() - outer_r + axial_offset.x().min(0.0),
                    c.y() - outer_r + axial_offset.y().min(0.0),
                    c.z() - outer_r + axial_offset.z().min(0.0),
                ),
            );
            aabb_include(
                aabb,
                Point3::new(
                    c.x() + outer_r + axial_offset.x().max(0.0),
                    c.y() + outer_r + axial_offset.y().max(0.0),
                    c.z() + outer_r + axial_offset.z().max(0.0),
                ),
            );
        }
        FaceSurface::Nurbs(nurbs) => {
            for row in nurbs.control_points() {
                for pt in row {
                    aabb_include(aabb, *pt);
                }
            }
        }
        FaceSurface::Plane { .. } | FaceSurface::Cone(_) => {}
    }
}

// ── Face area ─────────────────────────────────────────────────────

/// Compute the area of a single face.
///
/// For planar faces, uses Newell's method (exact, no tessellation).
/// For NURBS faces, tessellates and sums triangle areas.
///
/// # Errors
///
/// Returns an error if the face is missing or tessellation fails.
pub fn face_area(
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;

    match face.surface() {
        FaceSurface::Plane { .. } => planar_face_area(topo, face_id),
        FaceSurface::Cylinder(cyl) => {
            // Cylinder lateral area: integrate r * du * dv over the face domain.
            // Use face_polygon to sample curved edges (circle caps give 32 points).
            let r = cyl.radius();
            let positions = crate::boolean::face_polygon(topo, face_id)?;
            if positions.len() >= 2 {
                // Project boundary to get v-range (axial extent)
                let axis = cyl.axis();
                let origin = cyl.origin();
                let v_vals: Vec<f64> = positions
                    .iter()
                    .map(|p| {
                        axis.dot(Vec3::new(
                            p.x() - origin.x(),
                            p.y() - origin.y(),
                            p.z() - origin.z(),
                        ))
                    })
                    .collect();
                let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
                let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let height = (v_max - v_min).abs();
                // Compute angular sweep from boundary points projected onto the
                // circular cross-section. For full cylinders this gives 2π; for
                // partial cylinders it gives the actual angular extent.
                let u_vals: Vec<f64> = positions
                    .iter()
                    .map(|p| {
                        let rel =
                            Vec3::new(p.x() - origin.x(), p.y() - origin.y(), p.z() - origin.z());
                        let along = axis.dot(rel);
                        let radial = rel - axis * along;
                        radial.y().atan2(radial.x())
                    })
                    .collect();
                let u_min = u_vals.iter().copied().fold(f64::INFINITY, f64::min);
                let u_max = u_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let angular_span = u_max - u_min;
                // If the angular span covers most of a full circle (> 350°),
                // treat it as a full revolution — boundary sampling may not
                // reach exactly ±π.
                let sweep = if angular_span > 330.0_f64.to_radians() {
                    std::f64::consts::TAU
                } else {
                    angular_span
                };
                Ok(sweep * r * height)
            } else {
                let mesh = tessellate::tessellate(topo, face_id, deflection)?;
                Ok(triangle_mesh_area(&mesh))
            }
        }
        FaceSurface::Sphere(sph) => {
            // Spherical zone area = 2πr² * (sin(v_max) - sin(v_min))
            // where v is the latitude parameter (-π/2 to π/2).
            let r = sph.radius();
            let positions = crate::boolean::face_polygon(topo, face_id)?;
            if positions.len() >= 3 {
                let v_vals: Vec<f64> = positions.iter().map(|p| sph.project_point(*p).1).collect();
                let avg_v: f64 = v_vals.iter().sum::<f64>() / v_vals.len() as f64;
                let signed_area = newell_signed_z_area(&positions);
                let (v_min, v_max) = if signed_area > 0.0 {
                    (avg_v, std::f64::consts::FRAC_PI_2)
                } else {
                    (-std::f64::consts::FRAC_PI_2, avg_v)
                };
                Ok(2.0 * std::f64::consts::PI * r * r * (v_max.sin() - v_min.sin()))
            } else {
                // Full sphere fallback
                Ok(4.0 * std::f64::consts::PI * r * r)
            }
        }
        FaceSurface::Cone(_) => analytic_cone_face_area(topo, face_id),
        FaceSurface::Torus(_) => analytic_torus_face_area(topo, face_id),
        FaceSurface::Nurbs(_) => {
            let mesh = tessellate::tessellate(topo, face_id, deflection)?;
            Ok(triangle_mesh_area(&mesh))
        }
    }
}

/// Compute the area of a conical face analytically.
///
/// For a cone parameterised as
///   `P(u,v) = apex + v*(cos(α)*radial(u) + sin(α)*axis)`
/// the surface element is `dA = v * cos(α) * du * dv`.
///
/// Integrating over `u ∈ [u0,u1], v ∈ [v0,v1]`:
///   `area = cos(α) * (u1-u0) * (v1²-v0²) / 2`
///
/// This equals `π*(r0+r1)*slant*angle_frac` (standard frustum lateral area)
/// when verified: `r0=v0*cos(α)`, `r1=v1*cos(α)`, slant=|v1-v0|,
/// angle_frac=(u1-u0)/TAU.
fn analytic_cone_face_area(
    topo: &Topology,
    face_id: FaceId,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let cone = match face.surface() {
        FaceSurface::Cone(c) => c,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_cone_face_area requires a cone face".into(),
            });
        }
    };
    let wire = topo.wire(face.outer_wire())?;

    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = cone.project_point(vtx.point());
                    u_vals.push(u);
                    v_vals.push(v);
                }
            }
            if !edge.is_closed() {
                if let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        let ts = circle.project(sv.point());
                        let te = circle.project(ev.point());
                        let fwd = (te - ts).rem_euclid(std::f64::consts::TAU);
                        let mid_t = if fwd <= std::f64::consts::PI {
                            ts + fwd * 0.5
                        } else {
                            ts - (std::f64::consts::TAU - fwd) * 0.5
                        };
                        let mid = circle.evaluate(mid_t);
                        let (u, _) = cone.project_point(mid);
                        u_vals.push(u);
                    }
                }
            }
        }
    }

    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (v_max - v_min).abs() < 1e-15 {
        return Ok(0.0);
    }

    let u_range = compute_angular_range(&mut u_vals);
    let (u0, u1) = u_range;

    let cos_a = cone.half_angle().cos();
    let area = cos_a * (u1 - u0) * (v_max * v_max - v_min * v_min) / 2.0;
    Ok(area.abs())
}

/// Compute the area of a toroidal face analytically.
///
/// For a torus parameterised as
///   `P(u,v) = C + (R + r*cos(v))*(cos(u)*x + sin(u)*y) + r*sin(v)*z`
/// the surface element is `dA = r * (R + r*cos(v)) * du * dv`.
///
/// Integrating over `u ∈ [u0,u1], v ∈ [v0,v1]`:
///   `area = r * (u1-u0) * [R*(v1-v0) + r*(sin(v1)-sin(v0))]`
///
/// For a full torus: `area = r * 2π * (R*2π + r*0) = 4π²Rr` ✓
fn analytic_torus_face_area(
    topo: &Topology,
    face_id: FaceId,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let tor = match face.surface() {
        FaceSurface::Torus(t) => t,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_torus_face_area requires a torus face".into(),
            });
        }
    };
    let wire = topo.wire(face.outer_wire())?;

    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = tor.project_point(vtx.point());
                    u_vals.push(u);
                    v_vals.push(v);
                }
            }
            if !edge.is_closed() {
                if let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        let ts = circle.project(sv.point());
                        let te = circle.project(ev.point());
                        let fwd = (te - ts).rem_euclid(std::f64::consts::TAU);
                        let mid_t = if fwd <= std::f64::consts::PI {
                            ts + fwd * 0.5
                        } else {
                            ts - (std::f64::consts::TAU - fwd) * 0.5
                        };
                        let mid = circle.evaluate(mid_t);
                        let (u, _) = tor.project_point(mid);
                        u_vals.push(u);
                    }
                }
            }
        }
    }

    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (v_max - v_min).abs() < 1e-15 {
        // Full torus: v wraps from 0 to 2π, all boundary v-vals are the same.
        // Use full v-range.
        let u_range = compute_angular_range(&mut u_vals);
        let (u0, u1) = u_range;
        let big_r = tor.major_radius();
        let small_r = tor.minor_radius();
        let dv = std::f64::consts::TAU;
        let area = small_r * (u1 - u0) * (big_r * dv + small_r * 0.0);
        return Ok(area.abs());
    }

    let u_range = compute_angular_range(&mut u_vals);
    let (u0, u1) = u_range;

    let big_r = tor.major_radius();
    let small_r = tor.minor_radius();
    let area =
        small_r * (u1 - u0) * (big_r * (v_max - v_min) + small_r * (v_max.sin() - v_min.sin()));
    Ok(area.abs())
}

/// Newell's method: compute the area of a planar polygon from its
/// boundary vertices, subtracting inner wire (hole) areas.
fn planar_face_area(topo: &Topology, face_id: FaceId) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let outer_wire = topo.wire(face.outer_wire())?;
    let outer_positions = collect_wire_positions(topo, outer_wire)?;

    let outer_area = newell_area(&outer_positions);

    // Subtract hole areas.
    let mut hole_area = 0.0;
    for &inner_wid in face.inner_wires() {
        let inner_wire = topo.wire(inner_wid)?;
        let inner_positions = collect_wire_positions(topo, inner_wire)?;
        hole_area += newell_area(&inner_positions);
    }

    Ok((outer_area - hole_area).abs())
}

/// Compute the area of a polygon using Newell's method.
fn newell_area(positions: &[Point3]) -> f64 {
    let n = positions.len();
    if n < 3 {
        return 0.0;
    }

    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = positions[i];
        let vj = positions[j];
        sx = vi.z().mul_add(-vj.y(), vi.y().mul_add(vj.z(), sx));
        sy = vi.x().mul_add(-vj.z(), vi.z().mul_add(vj.x(), sy));
        sz = vi.y().mul_add(-vj.x(), vi.x().mul_add(vj.y(), sz));
    }

    0.5 * sz.mul_add(sz, sx.mul_add(sx, sy * sy)).sqrt()
}

/// Signed area of a polygon projected onto the XY plane.
/// Positive = CCW from +Z, negative = CW.
fn newell_signed_z_area(pts: &[Point3]) -> f64 {
    let n = pts.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].x() * pts[j].y() - pts[j].x() * pts[i].y();
    }
    area * 0.5
}

/// Sum of triangle areas from a tessellated mesh.
fn triangle_mesh_area(mesh: &tessellate::TriangleMesh) -> f64 {
    let mut area = 0.0;
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    let tri_count = idx.len() / 3;

    for t in 0..tri_count {
        let i0 = idx[t * 3] as usize;
        let i1 = idx[t * 3 + 1] as usize;
        let i2 = idx[t * 3 + 2] as usize;

        let a = pos[i1] - pos[i0];
        let b = pos[i2] - pos[i0];
        area += 0.5 * a.cross(b).length();
    }

    area
}

// ── Solid surface area ────────────────────────────────────────────

/// Compute the total surface area of a solid.
///
/// Sums `face_area()` over every face in every shell.
///
/// # Errors
///
/// Returns an error if a topology lookup or tessellation fails.
pub fn solid_surface_area(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let mut total = 0.0;
    for fid in collect_solid_face_ids(topo, solid)? {
        total += face_area(topo, fid, deflection)?;
    }
    Ok(total)
}

// ── Volume ────────────────────────────────────────────────────────

/// Try to compute the volume of a solid analytically by detecting known
/// primitive shapes (sphere, cylinder, cone/frustum, torus).
///
/// Returns `None` if the solid is not a recognized pure primitive, in which
/// case the caller should fall back to tessellation.
///
/// Detection rules (single pass over shell faces):
/// - Any `Nurbs` face → `None` (fall back)
/// - All faces are `Sphere` → sphere formula `(4/3)πr³`
/// - Exactly 1 `Cylinder` + ≥1 `Plane` caps, 0 other analytic → `πr²h`
/// - Exactly 1 `Cone` + ≤2 `Plane` caps, 0 other analytic → cone/frustum formula
///   (cap radii are read from the `Circle3D` edges of the cap faces)
/// - Exactly 1 `Torus` + 0 planes, 0 other analytic → `2π²Rr²`
fn try_analytic_solid_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use std::f64::consts::PI;

    let solid_data = topo.solid(solid).ok()?;
    let shell = topo.shell(solid_data.outer_shell()).ok()?;

    // Classify all faces by surface type.
    let mut sphere_r: Option<f64> = None;
    let mut cyl: Option<(Point3, Vec3, f64)> = None; // (origin, axis, radius)
    let mut cone_params: Option<(Point3, Vec3)> = None; // (apex, axis)
    let mut torus_params: Option<(f64, f64)> = None; // (major_r, minor_r)
    let mut planes: Vec<(Vec3, f64)> = Vec::new();
    let mut plane_face_ids: Vec<FaceId> = Vec::new();

    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Nurbs(_) => return None,
            FaceSurface::Plane { normal, d } => {
                planes.push((*normal, *d));
                plane_face_ids.push(fid);
            }
            FaceSurface::Sphere(s) => {
                let r = s.radius();
                match sphere_r {
                    None => sphere_r = Some(r),
                    // Multiple sphere faces must all share the same radius.
                    Some(existing) if (r - existing).abs() > existing * 1e-6 => return None,
                    Some(_) => {}
                }
            }
            FaceSurface::Cylinder(c) => {
                if cyl.is_some() {
                    return None;
                }
                cyl = Some((c.origin(), c.axis(), c.radius()));
            }
            FaceSurface::Cone(c) => {
                if cone_params.is_some() {
                    return None;
                }
                cone_params = Some((c.apex(), c.axis()));
            }
            FaceSurface::Torus(t) => {
                if torus_params.is_some() {
                    return None;
                }
                torus_params = Some((t.major_radius(), t.minor_radius()));
            }
        }
    }

    // ── Sphere: all faces are sphere faces (e.g. two hemispheres) ─────
    if let Some(r) = sphere_r {
        if cyl.is_none() && cone_params.is_none() && torus_params.is_none() && planes.is_empty() {
            // Verify actual vertex distances match the stored radius.
            // A non-uniform scale transforms vertices but leaves the sphere
            // surface radius unchanged, making the analytic formula wrong.
            let sphere_faces: Vec<_> = shell.faces().to_vec();
            let center = if let Ok(f) = topo.face(sphere_faces[0]) {
                if let FaceSurface::Sphere(s) = f.surface() {
                    s.center()
                } else {
                    return None;
                }
            } else {
                return None;
            };
            let mut max_dist = 0.0_f64;
            let mut min_dist = f64::INFINITY;
            for &fid in &sphere_faces {
                if let Ok(face) = topo.face(fid) {
                    if let Ok(wire) = topo.wire(face.outer_wire()) {
                        for oe in wire.edges() {
                            if let Ok(e) = topo.edge(oe.edge()) {
                                if let Ok(v) = topo.vertex(e.start()) {
                                    let d = (v.point() - center).length();
                                    max_dist = max_dist.max(d);
                                    min_dist = min_dist.min(d);
                                }
                            }
                        }
                    }
                }
            }
            // If all vertices are equidistant (within 1%), use analytic formula
            if (max_dist - min_dist).abs() < r * 0.01 {
                return Some(4.0 / 3.0 * PI * r * r * r);
            }
            // Non-uniform scale detected — fall through to tessellation
            return None;
        }
    }

    // ── Cylinder: 1 cylindrical face + planar caps ────────────────────
    //
    // A pure cylinder has exactly 1 cylindrical face and 2 planar caps.
    // If there are more than 2 planes the solid is compound (e.g. a box
    // with a drilled hole has 1 cylindrical hole-wall + 6 box faces).
    // In the compound case the cylindrical face is a concave inner surface
    // and the formula πr²h would compute the cylinder volume, not the solid.
    if let Some((origin, axis, r)) = cyl {
        if cone_params.is_none()
            && torus_params.is_none()
            && sphere_r.is_none()
            && planes.len() == 2
        {
            let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());
            let mut ts = cap_t_values(origin_vec, axis, &planes);
            if ts.len() >= 2 {
                ts.sort_by(f64::total_cmp);
                if let (Some(&t_min), Some(&t_max)) = (ts.first(), ts.last()) {
                    return Some(PI * r * r * (t_max - t_min));
                }
            }
        }
    }

    // ── Cone / frustum: 1 conical face + planar caps ──────────────────
    //
    // Cap radii are read directly from the Circle3D edges of the cap faces,
    // bypassing the ConicalSurface parameterization entirely. Heights are
    // derived from the circle centers projected onto the cone axis.
    if let Some((apex, axis)) = cone_params {
        if cyl.is_none() && torus_params.is_none() && sphere_r.is_none() {
            let apex_vec = Vec3::new(apex.x(), apex.y(), apex.z());

            // Collect (circle_center, radius) from each plane cap face.
            let mut cap_circles: Vec<(Point3, f64)> = Vec::new();
            for &fid in &plane_face_ids {
                if let Some(cap) = find_cap_circle(topo, fid) {
                    cap_circles.push(cap);
                }
            }

            // If any cap face did not yield a circle, the cone is degenerate or
            // unsupported — fall back to tessellation rather than silently wrong answer.
            if cap_circles.len() != plane_face_ids.len() {
                return None;
            }

            match cap_circles.as_slice() {
                [(c, r)] => {
                    // Pointed cone: h = distance from apex to cap center along axis.
                    let c_vec = Vec3::new(c.x(), c.y(), c.z());
                    let h = (c_vec - apex_vec).dot(axis).abs();
                    return Some(PI / 3.0 * r * r * h);
                }
                [(c1, r1), (c2, r2)] => {
                    // Frustum: h = distance between cap centers projected onto axis.
                    let c1_vec = Vec3::new(c1.x(), c1.y(), c1.z());
                    let c2_vec = Vec3::new(c2.x(), c2.y(), c2.z());
                    let h = (c2_vec - c1_vec).dot(axis).abs();
                    return Some(PI * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2));
                }
                _ => {}
            }
        }
    }

    // ── Torus: 1 toroidal face, no planar caps ────────────────────────
    if let Some((r_major, r_minor)) = torus_params {
        if cyl.is_none() && cone_params.is_none() && sphere_r.is_none() && planes.is_empty() {
            return Some(2.0 * PI * PI * r_major * r_minor * r_minor);
        }
    }

    None
}

/// Minimum |n · axis| for a plane to be considered a perpendicular cap face
/// (i.e. the plane normal is within ~8° of the axis direction).
const AXIS_PARALLEL_MIN_DOT: f64 = 0.99;

/// Compute signed distances along `axis` from `ref_pt` to cap planes that are
/// roughly perpendicular to the axis (`|n · axis| > AXIS_PARALLEL_MIN_DOT`).
///
/// For a plane `n · P = d`, the intersection with the line `ref_pt + t * axis`
/// satisfies `t = (d − n · ref_pt) / (n · axis)`.
fn cap_t_values(ref_pt: Vec3, axis: Vec3, planes: &[(Vec3, f64)]) -> Vec<f64> {
    let mut ts = Vec::new();
    for &(n, d) in planes {
        let nd = n.dot(axis);
        if nd.abs() > AXIS_PARALLEL_MIN_DOT {
            ts.push((d - n.dot(ref_pt)) / nd);
        }
    }
    ts
}

/// Search a face's outer wire for a `Circle3D` edge and return its `(center, radius)`.
///
/// Used by the cone volume formula to read cap radii directly from the geometry
/// rather than inferring them from the `ConicalSurface` parameterization.
fn find_cap_circle(topo: &Topology, face_id: FaceId) -> Option<(Point3, f64)> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    for oe in wire.edges() {
        // Use let-else so a missing edge skips to the next iteration
        // rather than returning None for the whole face.
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        if let brepkit_topology::edge::EdgeCurve::Circle(c) = edge.curve() {
            return Some((c.center(), c.radius()));
        }
    }
    None
}

/// Compute the volume of a solid using the signed tetrahedra method
/// (divergence theorem on a surface tessellation).
///
/// For each triangle `(v0, v1, v2)`, the signed volume of the
/// tetrahedron it forms with the origin is `v0 · (v1 × v2) / 6`.
///
/// For pure-primitive solids (sphere, cylinder, cone, torus), uses exact
/// analytic formulas instead of tessellation.
///
/// # Errors
///
/// Returns an error if tessellation or topology lookups fail.
pub fn solid_volume(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    // Fast path: exact analytic formula for known primitives.
    if let Some(v) = try_analytic_solid_volume(topo, solid) {
        return Ok(v);
    }

    // Fast path: for solids made entirely of planar triangular faces
    // (e.g. mesh imports), compute volume directly from face geometry.
    // This avoids re-tessellation which has known WASM winding issues.
    if let Ok(v) = solid_volume_from_faces(topo, solid, deflection) {
        return Ok(v);
    }

    // For solids with faces that have inner wires (holes from boolean ops)
    // or reversed non-planar faces (inner walls from shell/boolean operations),
    // use direct per-face tessellation with signed-volume summation.
    // tessellate() handles face reversal (flips winding + normals), so raw
    // signed tets are correct even without a globally watertight mesh.
    let needs_direct_tessellation = {
        let s = topo.solid(solid)?;
        let sh = topo.shell(s.outer_shell())?;
        sh.faces().iter().any(|&fid| {
            topo.face(fid).is_ok_and(|f| {
                !f.inner_wires().is_empty()
                    || (f.is_reversed() && !matches!(f.surface(), FaceSurface::Plane { .. }))
            })
        })
    };
    if needs_direct_tessellation {
        return volume_from_direct_face_tessellation(topo, solid, deflection);
    }

    // Try watertight tessellation first — this gives correct volume
    // via signed tetrahedra since the mesh is closed.
    let mesh = tessellate::tessellate_solid(topo, solid, deflection)?;
    if !mesh.indices.is_empty() {
        let vol = signed_volume_from_mesh(&mesh);
        // If watertight mesh volume is non-trivial, use it.
        // Near-zero result indicates inconsistent winding (e.g. shelled solids
        // where inner face normals cancel outer face contributions).
        if vol > 1e-12 {
            return Ok(vol);
        }
    }

    // For all-planar solids (e.g. chamfered boxes with non-manifold topology
    // where tessellate_solid gives wrong winding), use polygon-based volume.
    if let Ok(v) = volume_from_planar_polygons(topo, solid, deflection) {
        return Ok(v);
    }

    // Fallback: per-face tessellation with centroid-based winding correction.
    volume_from_per_face_tessellation(topo, solid, deflection)
}

/// Compute signed volume from a watertight triangle mesh using
/// the divergence theorem (signed tetrahedra method).
fn signed_volume_from_mesh(mesh: &tessellate::TriangleMesh) -> f64 {
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    let tri_count = idx.len() / 3;

    let mut total = 0.0;
    for t in 0..tri_count {
        let v0 = pos[idx[t * 3] as usize];
        let v1 = pos[idx[t * 3 + 1] as usize];
        let v2 = pos[idx[t * 3 + 2] as usize];

        let a = Vec3::new(v0.x(), v0.y(), v0.z());
        let b = Vec3::new(v1.x(), v1.y(), v1.z());
        let c = Vec3::new(v2.x(), v2.y(), v2.z());

        total += a.dot(b.cross(c));
    }

    (total / 6.0).abs()
}

/// Compute volume by tessellating each face independently, then
/// applying winding correction using the solid's approximate centroid.
///
/// For each face, we check whether the tessellated triangle normals
/// point away from the solid centroid (outward). If most triangles'
/// normals point inward, we flip the sign for that face's volume
/// contribution.
fn volume_from_per_face_tessellation(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    // Compute approximate solid centroid from vertex positions.
    let vertex_points = collect_solid_vertex_points(topo, solid)?;
    let centroid = if vertex_points.is_empty() {
        Point3::new(0.0, 0.0, 0.0)
    } else {
        let n = vertex_points.len() as f64;
        let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
        for p in &vertex_points {
            sx += p.x();
            sy += p.y();
            sz += p.z();
        }
        Point3::new(sx / n, sy / n, sz / n)
    };

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total: f64 = 0.0;
    for &fid in shell.faces() {
        let mesh = tessellate::tessellate(topo, fid, deflection)?;
        let idx = &mesh.indices;
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;
        if tri_count == 0 {
            continue;
        }

        // Use the face's stored normal for planar faces (more reliable
        // than centroid-based heuristic for chamfer bevels and other
        // angled faces).
        let sign = if let Ok(face) = topo.face(fid) {
            if let FaceSurface::Plane { normal, .. } = face.surface() {
                face_winding_sign_from_normal(&mesh, *normal)
            } else {
                face_winding_sign_centroid(&mesh, centroid)
            }
        } else {
            face_winding_sign_centroid(&mesh, centroid)
        };

        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());

            total += sign * a.dot(b.cross(c));
        }
    }

    Ok((total / 6.0).abs())
}

/// Exact signed volume contribution of a cylindrical face via the
/// divergence theorem: `V = (1/3) ∫∫ P·n dA`.
///
/// For a cylinder parameterised as
///   `P(u,v) = O + r*(cos u · ex + sin u · ey) + v · a`
/// the outward normal is `n = cos u · ex + sin u · ey`, dA = r du dv.
///
/// Integrating analytically over `u ∈ [u1,u2], v ∈ [v1,v2]`:
///   `V = (r/3) · h · [ ox·(sin u2 − sin u1) + oy·(−cos u2 + cos u1) + r·(u2 − u1) ]`
/// where `ox = O·ex`, `oy = O·ey`, `h = v2 − v1`.
///
/// For a reversed face the contribution is negated.
fn analytic_cylinder_signed_volume(
    topo: &Topology,
    face_id: FaceId,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let cyl = match face.surface() {
        FaceSurface::Cylinder(c) => c,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_cylinder_signed_volume requires a cylinder face".into(),
            });
        }
    };

    // Collect boundary vertex (u,v) parameters on the cylinder.
    let wire = topo.wire(face.outer_wire())?;
    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = cyl.project_point(vtx.point());
                    u_vals.push(u);
                    v_vals.push(v);
                }
            }
            // Sample circle-edge midpoints for angular coverage.
            if !edge.is_closed() {
                if let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        let ts = circle.project(sv.point());
                        let te = circle.project(ev.point());
                        // Choose the shorter arc for the midpoint.
                        let fwd = (te - ts).rem_euclid(std::f64::consts::TAU);
                        let mid_t = if fwd <= std::f64::consts::PI {
                            ts + fwd * 0.5
                        } else {
                            ts - (std::f64::consts::TAU - fwd) * 0.5
                        };
                        let mid = circle.evaluate(mid_t);
                        let (u, _) = cyl.project_point(mid);
                        u_vals.push(u);
                    }
                }
            }
        }
    }

    // Determine v-range (axial).
    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let h = v_max - v_min;
    if h.abs() < 1e-15 {
        return Ok(0.0);
    }

    let u_range = compute_angular_range(&mut u_vals);

    let r = cyl.radius();
    let x_axis = cyl.x_axis();
    let y_axis = cyl.y_axis();

    let o_vec = Vec3::new(cyl.origin().x(), cyl.origin().y(), cyl.origin().z());
    let ox = o_vec.dot(x_axis);
    let oy = o_vec.dot(y_axis);

    let (u1, u2) = u_range;
    let (sin1, cos1) = u1.sin_cos();
    let (sin2, cos2) = u2.sin_cos();

    let vol = (r / 3.0) * h * (ox * (sin2 - sin1) + oy * (-cos2 + cos1) + r * (u2 - u1));

    Ok(if face.is_reversed() { -vol } else { vol })
}

/// Exact signed volume contribution of a conical face via the divergence
/// theorem: `V = (1/3) ∫∫ P·n dA`.
///
/// For a cone parameterised as
///   `P(u,v) = apex + v*(cos_a*(cos u · ex + sin u · ey) + sin_a · axis)`
/// the outward normal is `n = sin_a*(cos u · ex + sin u · ey) - cos_a · axis`,
/// and `dA = v · cos_a · du dv`.
///
/// The integrand `P·n · dA` simplifies to closed form over `[u1,u2] × [v1,v2]`.
fn analytic_cone_signed_volume(
    topo: &Topology,
    face_id: FaceId,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let cone = match face.surface() {
        FaceSurface::Cone(c) => c,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_cone_signed_volume requires a cone face".into(),
            });
        }
    };

    let wire = topo.wire(face.outer_wire())?;
    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = cone.project_point(vtx.point());
                    u_vals.push(u);
                    v_vals.push(v);
                }
            }
            if !edge.is_closed() {
                if let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        let ts = circle.project(sv.point());
                        let te = circle.project(ev.point());
                        let fwd = (te - ts).rem_euclid(std::f64::consts::TAU);
                        let mid_t = if fwd <= std::f64::consts::PI {
                            ts + fwd * 0.5
                        } else {
                            ts - (std::f64::consts::TAU - fwd) * 0.5
                        };
                        let mid = circle.evaluate(mid_t);
                        let (u, _) = cone.project_point(mid);
                        u_vals.push(u);
                    }
                }
            }
        }
    }

    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (v_max - v_min).abs() < 1e-15 {
        return Ok(0.0);
    }

    let u_range = compute_angular_range(&mut u_vals);

    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let x_axis = cone.x_axis();
    let y_axis = cone.y_axis();
    let axis = cone.axis();
    let apex = cone.apex();
    let a_vec = Vec3::new(apex.x(), apex.y(), apex.z());

    // Compute the divergence-theorem integral analytically.
    //
    // P(u,v) = apex + v*(cos_a*radial(u) + sin_a*axis)
    // n(u) = sin_a*radial(u) - cos_a*axis   (outward normal direction)
    // dA = v * cos_a * du * dv
    //
    // P·n = apex·(sin_a*radial - cos_a*axis)
    //     + v*(cos_a*sin_a*(radial·radial) + sin_a²*(axis·radial) - cos_a²*(radial·axis) - cos_a*sin_a*(axis·axis))
    //     = apex·(sin_a*radial - cos_a*axis) + v*(cos_a*sin_a - cos_a*sin_a)
    //     = apex·(sin_a*radial(u) - cos_a*axis)
    //
    // The v-dependent terms cancel: cos_a*sin_a - cos_a*sin_a = 0, so P·n is v-independent.
    //
    // Full integrand = (1/3) * P·n * dA = (1/3) * [a_vec·(sin_a*radial(u) - cos_a*axis)] * v*cos_a * du * dv
    //
    // ∫∫ = (cos_a/3) * [(v²/2)|v1..v2] * ∫[sin_a*(ax*cos_u + ay*sin_u) - cos_a*az] du
    // where ax = a_vec·x_axis, ay = a_vec·y_axis, az = a_vec·axis
    let ax = a_vec.dot(x_axis);
    let ay = a_vec.dot(y_axis);
    let az = a_vec.dot(axis);

    let v2_half = (v_max * v_max - v_min * v_min) / 2.0;

    let (u1, u2) = u_range;
    let (sin1, cos1) = u1.sin_cos();
    let (sin2, cos2) = u2.sin_cos();

    let u_integral = sin_a * (ax * (sin2 - sin1) + ay * (-cos2 + cos1)) - cos_a * az * (u2 - u1);

    let vol = (cos_a / 3.0) * v2_half * u_integral;

    Ok(if face.is_reversed() { -vol } else { vol })
}

/// Exact signed volume contribution of a spherical face via the divergence
/// theorem: `V = (1/3) ∫∫ P·n dA`.
///
/// For a sphere parameterised as
///   `P(u,v) = C + r*(cos_v*cos_u*ex + cos_v*sin_u*ey + sin_v*ez)`
/// the outward normal equals the unit radial direction, and `dA = r²*cos_v * du dv`.
///
/// `P·n = C·n + r`, so the integrand is `(1/3)*(C·n + r)*r²*cos_v du dv`.
fn analytic_sphere_signed_volume(
    topo: &Topology,
    face_id: FaceId,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let sph = match face.surface() {
        FaceSurface::Sphere(s) => s,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_sphere_signed_volume requires a sphere face".into(),
            });
        }
    };

    let wire = topo.wire(face.outer_wire())?;
    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = sph.project_point(vtx.point());
                    u_vals.push(u);
                    v_vals.push(v);
                }
            }
            if !edge.is_closed() {
                if let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        let ts = circle.project(sv.point());
                        let te = circle.project(ev.point());
                        let fwd = (te - ts).rem_euclid(std::f64::consts::TAU);
                        let mid_t = if fwd <= std::f64::consts::PI {
                            ts + fwd * 0.5
                        } else {
                            ts - (std::f64::consts::TAU - fwd) * 0.5
                        };
                        let mid = circle.evaluate(mid_t);
                        let (u, _) = sph.project_point(mid);
                        u_vals.push(u);
                    }
                }
            }
        }
    }

    let mut v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mut v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // For sphere caps (single circle boundary at one latitude), the boundary
    // vertices all share approximately the same v, so v_max ≈ v_min.
    // Determine which pole the face covers by checking a face interior point.
    if (v_max - v_min).abs() < 0.01 {
        let v_boundary = f64::midpoint(v_min, v_max);
        let positions = crate::boolean::face_polygon(topo, face_id)?;
        if positions.is_empty() {
            return Ok(0.0);
        }
        let n = positions.len() as f64;
        let avg = Point3::new(
            positions.iter().map(|p| p.x()).sum::<f64>() / n,
            positions.iter().map(|p| p.y()).sum::<f64>() / n,
            positions.iter().map(|p| p.z()).sum::<f64>() / n,
        );
        let (_, v_interior) = sph.project_point(avg);
        if v_interior > v_boundary {
            v_min = v_boundary;
            v_max = std::f64::consts::FRAC_PI_2;
        } else {
            v_min = -std::f64::consts::FRAC_PI_2;
            v_max = v_boundary;
        }
    }

    let u_range = compute_angular_range(&mut u_vals);

    let r = sph.radius();
    let x_axis = sph.x_axis();
    let y_axis = sph.y_axis();
    let z_axis = sph.z_axis();
    let c = sph.center();
    let c_vec = Vec3::new(c.x(), c.y(), c.z());

    // P·n = C·(cos_v*cos_u*ex + cos_v*sin_u*ey + sin_v*ez) + r
    // dA = r² * cos_v * du * dv
    //
    // Integrand = (1/3) * (cx*cos_v*cos_u + cy*cos_v*sin_u + cz*sin_v + r) * r² * cos_v
    // where cx = C·ex, cy = C·ey, cz = C·ez
    let cx = c_vec.dot(x_axis);
    let cy = c_vec.dot(y_axis);
    let cz = c_vec.dot(z_axis);

    let (u1, u2) = u_range;
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();
    let du = u2 - u1;

    // ∫cos_v*cos_v dv = v/2 + sin(2v)/4
    let vv_integral = |v: f64| -> f64 { v / 2.0 + (2.0 * v).sin() / 4.0 };
    let cos2_v = vv_integral(v_max) - vv_integral(v_min);

    // ∫cos_v dv = sin_v
    let cos_v_int = v_max.sin() - v_min.sin();

    // ∫sin_v*cos_v dv = sin²(v)/2
    let sincos_v = (v_max.sin().powi(2) - v_min.sin().powi(2)) / 2.0;

    // Full integral:
    // cx * cos2_v * (sin_u2 - sin_u1)
    // + cy * cos2_v * (-cos_u2 + cos_u1)
    // + cz * sincos_v * du
    // + r * cos_v_int * du
    let vol = (r * r / 3.0)
        * (cx * cos2_v * (sin_u2 - sin_u1)
            + cy * cos2_v * (-cos_u2 + cos_u1)
            + cz * sincos_v * du
            + r * cos_v_int * du);

    Ok(if face.is_reversed() { -vol } else { vol })
}

/// Exact signed volume contribution of a toroidal face via the divergence
/// theorem: `V = (1/3) ∫∫ P·n dA`.
///
/// For a torus parameterised as
///   `P(u,v) = C + (R + r*cos_v)*(cos_u*ex + sin_u*ey) + r*sin_v*ez`
/// the outward normal `n = cos_v*(cos_u*ex + sin_u*ey) + sin_v*ez`,
/// and `dA = r*(R + r*cos_v) du dv`.
fn analytic_torus_signed_volume(
    topo: &Topology,
    face_id: FaceId,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let tor = match face.surface() {
        FaceSurface::Torus(t) => t,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_torus_signed_volume requires a torus face".into(),
            });
        }
    };

    let wire = topo.wire(face.outer_wire())?;
    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = tor.project_point(vtx.point());
                    u_vals.push(u);
                    v_vals.push(v);
                }
            }
            if !edge.is_closed() {
                if let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        let ts = circle.project(sv.point());
                        let te = circle.project(ev.point());
                        let fwd = (te - ts).rem_euclid(std::f64::consts::TAU);
                        let mid_t = if fwd <= std::f64::consts::PI {
                            ts + fwd * 0.5
                        } else {
                            ts - (std::f64::consts::TAU - fwd) * 0.5
                        };
                        let mid = circle.evaluate(mid_t);
                        let (u, _) = tor.project_point(mid);
                        u_vals.push(u);
                    }
                }
            }
        }
    }

    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (v_max - v_min).abs() < 1e-15 {
        return Ok(0.0);
    }

    let u_range = compute_angular_range(&mut u_vals);

    let big_r = tor.major_radius();
    let small_r = tor.minor_radius();
    let x_axis = tor.x_axis();
    let y_axis = tor.y_axis();
    let z_axis = tor.z_axis();
    let c = tor.center();
    let c_vec = Vec3::new(c.x(), c.y(), c.z());

    // P·n = [C + (R+r*cos_v)*radial_u + r*sin_v*ez] · [cos_v*radial_u + sin_v*ez]
    //     = C·(cos_v*radial_u + sin_v*ez) + (R+r*cos_v)*cos_v + r*sin²_v
    //     = cos_v*(cx*cos_u + cy*sin_u) + sin_v*cz + (R+r*cos_v)*cos_v + r*sin²_v
    //     = cos_v*(cx*cos_u + cy*sin_u) + sin_v*cz + R*cos_v + r*cos²_v + r*sin²_v
    //     = cos_v*(cx*cos_u + cy*sin_u) + sin_v*cz + R*cos_v + r
    //
    // dA = r*(R + r*cos_v) du dv
    //
    // Full integrand = (1/3) * P·n * dA
    let cx = c_vec.dot(x_axis);
    let cy = c_vec.dot(y_axis);
    let cz = c_vec.dot(z_axis);

    let (u1, u2) = u_range;
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();
    let du = u2 - u1;

    // We need to integrate over v:
    // ∫ [cos_v*(cx*cos_u + cy*sin_u) + cz*sin_v + R*cos_v + r] * r*(R + r*cos_v) dv
    //
    // Expand the product with (R + r*cos_v):
    // = r * ∫ [cos_v*(cx*cos_u+cy*sin_u)*(R+r*cos_v)
    //        + cz*sin_v*(R+r*cos_v)
    //        + R*cos_v*(R+r*cos_v)
    //        + r*(R+r*cos_v)] dv
    //
    // This is a sum of standard trigonometric integrals.
    // Let S = cx*cos_u + cy*sin_u (depends on u, integrated separately)

    // Standard integrals over [v1,v2]:
    let sv1 = v_min.sin();
    let sv2 = v_max.sin();
    let cv1 = v_min.cos();
    let cv2 = v_max.cos();
    let dv = v_max - v_min;

    // ∫cos_v dv = sin_v
    let i_cos = sv2 - sv1;
    // ∫cos²_v dv = v/2 + sin(2v)/4
    let i_cos2 =
        (v_max / 2.0 + (2.0 * v_max).sin() / 4.0) - (v_min / 2.0 + (2.0 * v_min).sin() / 4.0);
    // ∫sin_v dv = -cos_v
    let i_sin = -cv2 + cv1;
    // ∫sin_v*cos_v dv = sin²(v)/2
    let i_sincos = (sv2 * sv2 - sv1 * sv1) / 2.0;
    // Group terms by u-dependence:
    // Terms with S (= cx*cos_u + cy*sin_u):
    //   r*[R*i_cos + r*i_cos2] * ∫S du
    let s_u_integral = cx * (sin_u2 - sin_u1) + cy * (-cos_u2 + cos_u1);
    let s_coeff = small_r * (big_r * i_cos + small_r * i_cos2);

    // Terms with cz*sin_v:
    //   r*cz*[R*i_sin + r*i_sincos] * du
    let cz_coeff = small_r * cz * (big_r * i_sin + small_r * i_sincos);

    // Terms with R*cos_v:
    //   r*R*[R*i_cos + r*i_cos2] * du
    let rcos_coeff = small_r * big_r * (big_r * i_cos + small_r * i_cos2);

    // Terms with r (constant in v):
    //   r*r*[R*dv + r*i_cos] * du
    let const_coeff = small_r * small_r * (big_r * dv + small_r * i_cos);

    let vol = (1.0 / 3.0) * (s_coeff * s_u_integral + (cz_coeff + rcos_coeff + const_coeff) * du);

    Ok(if face.is_reversed() { -vol } else { vol })
}

/// Compute the angular range `(u_start, u_end)` from a set of projected u values.
///
/// Detects the largest angular gap and treats it as the boundary between the
/// face's angular extent. For full revolutions (no significant gap), returns
/// `(0, 2π)`.
fn compute_angular_range(u_vals: &mut Vec<f64>) -> (f64, f64) {
    use std::f64::consts::TAU;
    let tol_lin = brepkit_math::tolerance::Tolerance::default().linear;

    u_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    u_vals.dedup_by(|a, b| (*a - *b).abs() < tol_lin);

    if u_vals.len() < 3 {
        return (0.0, TAU);
    }

    let mut max_gap = 0.0_f64;
    let mut gap_end_idx = 0_usize;
    for i in 0..u_vals.len() {
        let j = (i + 1) % u_vals.len();
        let gap = if j > i {
            u_vals[j] - u_vals[i]
        } else {
            u_vals[j] + TAU - u_vals[i]
        };
        if gap > max_gap {
            max_gap = gap;
            gap_end_idx = j;
        }
    }
    let n_angles = u_vals.len() as f64;
    let even_gap = TAU / n_angles;
    let gap_threshold = (2.5 * even_gap).min(TAU / 3.0);
    if max_gap < gap_threshold {
        (0.0, TAU)
    } else {
        let u_start = u_vals[gap_end_idx];
        let gap_start_idx = if gap_end_idx == 0 {
            u_vals.len() - 1
        } else {
            gap_end_idx - 1
        };
        let u_end = u_vals[gap_start_idx];
        if u_end > u_start {
            (u_start, u_end)
        } else {
            (u_start, u_end + TAU)
        }
    }
}

/// Compute volume by tessellating each face and summing signed tetrahedra
/// WITHOUT winding correction. Relies on `tessellate()` already handling
/// face reversal (via `is_reversed` flag) to produce correctly oriented
/// triangles. For analytic surface faces (cylinder, cone, sphere, torus),
/// uses exact analytical integration via the divergence theorem instead
/// of tessellation.
pub(crate) fn volume_from_direct_face_tessellation(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total: f64 = 0.0;
    for &fid in shell.faces() {
        let face = topo.face(fid)?;

        // Use exact analytical volume for analytic surface faces.
        match face.surface() {
            FaceSurface::Cylinder(_) => {
                total += analytic_cylinder_signed_volume(topo, fid)? * 6.0;
                continue;
            }
            FaceSurface::Cone(_) => {
                total += analytic_cone_signed_volume(topo, fid)? * 6.0;
                continue;
            }
            FaceSurface::Sphere(_) => {
                total += analytic_sphere_signed_volume(topo, fid)? * 6.0;
                continue;
            }
            FaceSurface::Torus(_) => {
                total += analytic_torus_signed_volume(topo, fid)? * 6.0;
                continue;
            }
            FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => {}
        }

        let mesh = tessellate::tessellate(topo, fid, deflection)?;
        let idx = &mesh.indices;
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;

        let mut face_total = 0.0;
        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());

            face_total += a.dot(b.cross(c));
        }

        total += face_total;
    }

    Ok((total / 6.0).abs())
}

/// Compute volume for all-planar solids using face polygon triangulation
/// with winding correction based on the stored face normal.
///
/// For each planar face, fan-triangulates the polygon and sums the signed
/// tetrahedra volumes, correcting winding so that the face normal points
/// outward. This works even for non-manifold topology since it only needs
/// each face's vertices and normal.
fn volume_from_planar_polygons(
    topo: &Topology,
    solid: SolidId,
    _deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    // Use the divergence theorem: V = (1/3) Σ d_i × A_i
    // where d_i = n_i · p (signed distance from origin to face plane)
    // and A_i is the polygon area.
    let mut total = 0.0_f64;
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let face_normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "planar polygon volume requires all-planar faces".into(),
                });
            }
        };

        // Get face polygon vertices from wire edges.
        let wire = topo.wire(face.outer_wire())?;
        let mut verts = Vec::with_capacity(wire.edges().len());
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            verts.push(topo.vertex(vid)?.point());
        }

        if verts.len() < 3 {
            continue;
        }

        // Signed distance from origin to the face plane.
        let d = crate::dot_normal_point(face_normal, verts[0]);

        // Polygon area via Newell method: A = |Σ (vi × vi+1)| / 2
        let n = verts.len();
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut cz = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            let vi = &verts[i];
            let vj = &verts[j];
            cx += vi.y() * vj.z() - vi.z() * vj.y();
            cy += vi.z() * vj.x() - vi.x() * vj.z();
            cz += vi.x() * vj.y() - vi.y() * vj.x();
        }
        let area_vec = Vec3::new(cx, cy, cz);
        let area = area_vec.length() / 2.0;

        total += d * area;
    }

    Ok((total / 3.0).abs())
}

/// Compute the volume of a solid directly from its face vertex
/// positions, bypassing tessellation. Only valid for solids composed
/// entirely of planar triangular faces (e.g. mesh imports).
///
/// Returns an error if the solid contains non-planar or
/// non-triangular faces.
///
/// # Errors
///
/// Returns [`OperationsError`] if topology lookups fail or if the
/// solid contains non-planar/non-triangular faces.
pub fn solid_volume_from_faces(
    topo: &Topology,
    solid: SolidId,
    _deflection: f64,
) -> Result<f64, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total = 0.0;
    let mut all_planar_triangles = true;

    for &fid in shell.faces() {
        let face = topo.face(fid)?;

        // Only use the fast path for planar faces with exactly 3 line edges.
        if !matches!(face.surface(), FaceSurface::Plane { .. }) {
            all_planar_triangles = false;
            break;
        }

        let wire = topo.wire(face.outer_wire())?;
        let edges = wire.edges();
        if edges.len() != 3 {
            all_planar_triangles = false;
            break;
        }

        // Check all edges are lines.
        let mut pts = Vec::with_capacity(3);
        for oe in edges {
            let edge = topo.edge(oe.edge())?;
            if !matches!(edge.curve(), EdgeCurve::Line) {
                all_planar_triangles = false;
                break;
            }
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            pts.push(topo.vertex(vid)?.point());
        }
        if !all_planar_triangles {
            break;
        }

        let a = Vec3::new(pts[0].x(), pts[0].y(), pts[0].z());
        let b = Vec3::new(pts[1].x(), pts[1].y(), pts[1].z());
        let c = Vec3::new(pts[2].x(), pts[2].y(), pts[2].z());

        total += a.dot(b.cross(c));
    }

    if all_planar_triangles {
        Ok((total / 6.0).abs())
    } else {
        Err(crate::OperationsError::InvalidInput {
            reason: "solid contains non-planar or non-triangular faces".to_string(),
        })
    }
}

// ── Center of mass ────────────────────────────────────────────────

/// Compute the center of mass of a solid, assuming uniform density.
///
/// Uses the same signed-tetrahedra decomposition as `solid_volume`,
/// accumulating the centroid contribution of each tetrahedron:
/// `centroid += signed_vol * (a + b + c)`, then divides by
/// `4 * total_volume`.
///
/// # Errors
///
/// Returns an error if the solid has zero volume or tessellation fails.
pub fn solid_center_of_mass(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<Point3, crate::OperationsError> {
    // Fast path: for all-planar-triangle solids, compute directly
    // from face geometry (avoids re-tessellation winding issues).
    if let Ok(com) = center_of_mass_from_faces(topo, solid) {
        return Ok(com);
    }

    // Compute approximate solid centroid from vertex positions.
    let vertex_points = collect_solid_vertex_points(topo, solid)?;
    let approx_centroid = if vertex_points.is_empty() {
        Point3::new(0.0, 0.0, 0.0)
    } else {
        let n = vertex_points.len() as f64;
        let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
        for p in &vertex_points {
            sx += p.x();
            sy += p.y();
            sz += p.z();
        }
        Point3::new(sx / n, sy / n, sz / n)
    };

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total_vol: f64 = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for &fid in shell.faces() {
        let mesh = tessellate::tessellate(topo, fid, deflection)?;
        let idx = &mesh.indices;
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;
        if tri_count == 0 {
            continue;
        }

        let sign = face_winding_sign_centroid(&mesh, approx_centroid);

        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());

            let signed_vol = sign * a.dot(b.cross(c));
            total_vol += signed_vol;
            cx += signed_vol * (v0.x() + v1.x() + v2.x());
            cy += signed_vol * (v0.y() + v1.y() + v2.y());
            cz += signed_vol * (v0.z() + v1.z() + v2.z());
        }
    }

    if total_vol.abs() < 1e-15 {
        // Volume too small to compute weighted CoM — fall back to vertex centroid.
        return Ok(approx_centroid);
    }

    let denom = 4.0 * total_vol;
    Ok(Point3::new(cx / denom, cy / denom, cz / denom))
}

/// Compute center of mass directly from face vertex positions for
/// solids composed entirely of planar triangular faces.
fn center_of_mass_from_faces(
    topo: &Topology,
    solid: SolidId,
) -> Result<Point3, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total_vol = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        if !matches!(face.surface(), FaceSurface::Plane { .. }) {
            return Err(crate::OperationsError::InvalidInput {
                reason: "non-planar face".to_string(),
            });
        }
        let wire = topo.wire(face.outer_wire())?;
        let edges = wire.edges();
        if edges.len() != 3 {
            return Err(crate::OperationsError::InvalidInput {
                reason: "non-triangular face".to_string(),
            });
        }

        let mut pts = Vec::with_capacity(3);
        for oe in edges {
            let edge = topo.edge(oe.edge())?;
            if !matches!(edge.curve(), EdgeCurve::Line) {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "non-line edge".to_string(),
                });
            }
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            pts.push(topo.vertex(vid)?.point());
        }

        let a = Vec3::new(pts[0].x(), pts[0].y(), pts[0].z());
        let b = Vec3::new(pts[1].x(), pts[1].y(), pts[1].z());
        let c = Vec3::new(pts[2].x(), pts[2].y(), pts[2].z());

        let signed_vol = a.dot(b.cross(c));
        total_vol += signed_vol;
        cx += signed_vol * (pts[0].x() + pts[1].x() + pts[2].x());
        cy += signed_vol * (pts[0].y() + pts[1].y() + pts[2].y());
        cz += signed_vol * (pts[0].z() + pts[1].z() + pts[2].z());
    }

    if total_vol.abs() < 1e-15 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "solid has zero volume, center of mass is undefined".into(),
        });
    }

    let denom = 4.0 * total_vol;
    Ok(Point3::new(cx / denom, cy / denom, cz / denom))
}

// ── Helpers ───────────────────────────────────────────────────────

/// Determine winding sign correction for a tessellated face using
/// the topology face's stored surface normal rather than mesh normals.
/// This is more reliable across platforms (native vs WASM).
/// Determine the winding sign for volume computation of a face.
///
/// For planar faces, uses the face's stored normal. For non-planar faces,
/// uses the solid's approximate centroid to determine the expected outward
/// direction, since the surface's geometric normal may not match the
/// topological outward normal.
fn face_winding_sign_centroid(mesh: &tessellate::TriangleMesh, solid_centroid: Point3) -> f64 {
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    let tri_count = idx.len() / 3;
    if tri_count == 0 {
        return 1.0;
    }

    // For each triangle, check whether its geometric normal points
    // away from the solid centroid. Use majority vote.
    let mut agree: i32 = 0;
    for t in 0..tri_count {
        let i0 = idx[t * 3] as usize;
        let i1 = idx[t * 3 + 1] as usize;
        let i2 = idx[t * 3 + 2] as usize;
        let tri_normal = (pos[i1] - pos[i0]).cross(pos[i2] - pos[i0]);
        if tri_normal.length() < 1e-20 {
            continue;
        }
        // Vector from centroid to triangle midpoint
        let mid = Point3::new(
            (pos[i0].x() + pos[i1].x() + pos[i2].x()) / 3.0,
            (pos[i0].y() + pos[i1].y() + pos[i2].y()) / 3.0,
            (pos[i0].z() + pos[i1].z() + pos[i2].z()) / 3.0,
        );
        let outward = mid - solid_centroid;
        // If tri_normal points same way as outward, winding is correct
        agree += if tri_normal.dot(outward) >= 0.0 {
            1
        } else {
            -1
        };
    }

    if agree >= 0 { 1.0 } else { -1.0 }
}

/// Determine winding sign by comparing the mesh triangle normals against
/// the face's known surface normal. More reliable than centroid-based
/// heuristic for angled faces (e.g. chamfer bevels).
fn face_winding_sign_from_normal(mesh: &tessellate::TriangleMesh, face_normal: Vec3) -> f64 {
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    let tri_count = idx.len() / 3;
    if tri_count == 0 {
        return 1.0;
    }

    let mut agree: i32 = 0;
    for t in 0..tri_count {
        let i0 = idx[t * 3] as usize;
        let i1 = idx[t * 3 + 1] as usize;
        let i2 = idx[t * 3 + 2] as usize;
        let tri_normal = (pos[i1] - pos[i0]).cross(pos[i2] - pos[i0]);
        if tri_normal.length() < 1e-20 {
            continue;
        }
        agree += if tri_normal.dot(face_normal) >= 0.0 {
            1
        } else {
            -1
        };
    }

    if agree >= 0 { 1.0 } else { -1.0 }
}

/// Collect deduplicated vertex positions from a solid.
fn collect_solid_vertex_points(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let solid_data = topo.solid(solid)?;

    for shell_id in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id)?;
        for &fid in shell.faces() {
            let face = topo.face(fid)?;
            for wire_id in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge())?;
                    vertex_ids.insert(edge.start());
                    vertex_ids.insert(edge.end());
                }
            }
        }
    }

    let mut points = Vec::with_capacity(vertex_ids.len());
    for vid in vertex_ids {
        points.push(topo.vertex(vid)?.point());
    }
    Ok(points)
}

/// Collect all face IDs from a solid's shells.
fn collect_solid_face_ids(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<FaceId>, crate::OperationsError> {
    let mut face_ids = Vec::new();
    let solid_data = topo.solid(solid)?;

    for shell_id in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id)?;
        face_ids.extend_from_slice(shell.faces());
    }
    Ok(face_ids)
}

/// Collect ordered vertex positions from a wire.
fn collect_wire_positions(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
) -> Result<Vec<Point3>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    let mut positions = Vec::new();
    let n_samples = 256_usize;
    let tol = 1e-10;

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        match edge.curve() {
            EdgeCurve::Line => {
                let vid = if oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                let pt = topo.vertex(vid)?.point();
                if positions
                    .last()
                    .is_none_or(|p: &Point3| (*p - pt).length() > tol)
                {
                    positions.push(pt);
                }
            }
            EdgeCurve::Circle(c) => {
                let (t0, t1) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    let ts = c.project(sp);
                    let mut te = c.project(ep);
                    if te <= ts {
                        te += std::f64::consts::TAU;
                    }
                    (ts, te)
                };
                sample_edge_curve(
                    &|t| c.evaluate(t),
                    t0,
                    t1,
                    n_samples,
                    oe.is_forward(),
                    tol,
                    &mut positions,
                );
            }
            EdgeCurve::Ellipse(e) => {
                let (t0, t1) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    let ts = e.project(sp);
                    let mut te = e.project(ep);
                    if te <= ts {
                        te += std::f64::consts::TAU;
                    }
                    (ts, te)
                };
                sample_edge_curve(
                    &|t| e.evaluate(t),
                    t0,
                    t1,
                    n_samples,
                    oe.is_forward(),
                    tol,
                    &mut positions,
                );
            }
            EdgeCurve::NurbsCurve(nc) => {
                let (u0, u1) = nc.domain();
                sample_edge_curve(
                    &|t| nc.evaluate(t),
                    u0,
                    u1,
                    n_samples,
                    oe.is_forward(),
                    tol,
                    &mut positions,
                );
            }
        }
    }
    Ok(positions)
}

/// Sample points along a parametric curve for area/distance calculations.
#[allow(clippy::cast_precision_loss)]
fn sample_edge_curve(
    evaluate: &dyn Fn(f64) -> Point3,
    t0: f64,
    t1: f64,
    n_samples: usize,
    forward: bool,
    tol: f64,
    positions: &mut Vec<Point3>,
) {
    let indices: Box<dyn Iterator<Item = usize>> = if forward {
        Box::new(0..n_samples)
    } else {
        Box::new((0..n_samples).rev())
    };
    for i in indices {
        let t = t0 + (t1 - t0) * (i as f64) / (n_samples as f64);
        let pt = evaluate(t);
        if positions
            .last()
            .is_none_or(|p: &Point3| (*p - pt).length() > tol)
        {
            positions.push(pt);
        }
    }
}

// ── Edge and wire length ──────────────────────────────────────────

/// Compute the length of a single edge.
///
/// For line edges, returns the Euclidean distance between endpoints.
/// For NURBS curve edges, uses numerical integration (Simpson's rule).
///
/// # Errors
///
/// Returns an error if the edge lookup fails.
pub fn edge_length(
    topo: &Topology,
    edge_id: brepkit_topology::edge::EdgeId,
) -> Result<f64, crate::OperationsError> {
    let edge = topo.edge(edge_id)?;
    match edge.curve() {
        brepkit_topology::edge::EdgeCurve::Line => {
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            Ok((end - start).length())
        }
        brepkit_topology::edge::EdgeCurve::NurbsCurve(curve) => Ok(curve.arc_length(50)),
        brepkit_topology::edge::EdgeCurve::Circle(circle) => {
            if edge.is_closed() {
                Ok(circle.circumference())
            } else {
                let start = topo.vertex(edge.start())?.point();
                let end = topo.vertex(edge.end())?.point();
                let t0 = circle.project(start);
                let t1 = circle.project(end);
                let mut angle = t1 - t0;
                if angle < 0.0 {
                    angle += std::f64::consts::TAU;
                }
                Ok(angle * circle.radius())
            }
        }
        brepkit_topology::edge::EdgeCurve::Ellipse(ellipse) => {
            if edge.is_closed() {
                Ok(ellipse.approximate_circumference())
            } else {
                // Approximate arc length via sampling
                let start = topo.vertex(edge.start())?.point();
                let end = topo.vertex(edge.end())?.point();
                let t0 = ellipse.project(start);
                let t1 = ellipse.project(end);
                let mut angle = t1 - t0;
                if angle < 0.0 {
                    angle += std::f64::consts::TAU;
                }
                let n = 50;
                let dt = angle / n as f64;
                let mut length = 0.0;
                let mut prev = ellipse.evaluate(t0);
                for i in 1..=n {
                    let t = t0 + dt * i as f64;
                    let curr = ellipse.evaluate(t);
                    length += (curr - prev).length();
                    prev = curr;
                }
                Ok(length)
            }
        }
    }
}

/// Compute the total length (perimeter) of a wire.
///
/// Sums the length of all edges in the wire.
///
/// # Errors
///
/// Returns an error if any edge lookup fails.
pub fn wire_length(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<f64, crate::OperationsError> {
    let wire = topo.wire(wire_id)?;
    let mut total = 0.0;
    for oe in wire.edges() {
        total += edge_length(topo, oe.edge())?;
    }
    Ok(total)
}

/// Compute the perimeter of a face (outer wire length).
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn face_perimeter(topo: &Topology, face_id: FaceId) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    wire_length(topo, face.outer_wire())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;

    // ── Helper: assert relative error within tolerance ────────────
    //
    // For analytic primitives (box, cylinder, sphere, cone, torus) we
    // expect 1e-8 relative error. For NURBS-involving operations
    // (fillet, boolean, tessellation-based) we accept 1e-4.

    fn assert_rel(actual: f64, expected: f64, rel_tol: f64, label: &str) {
        let rel_err = if expected.abs() < 1e-15 {
            actual.abs()
        } else {
            (actual - expected).abs() / expected.abs()
        };
        assert!(
            rel_err < rel_tol,
            "{label}: expected {expected:.8}, got {actual:.8}, \
             rel_err={rel_err:.2e} (tolerance={rel_tol:.0e})"
        );
    }

    /// Assert that `aabb` contains the box `[min_x..max_x, min_y..max_y, min_z..max_z]`
    /// within a 1e-6 slack on each bound.
    fn assert_aabb_contains(
        aabb: &brepkit_math::aabb::Aabb3,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) {
        let slack = 1e-6;
        assert!(
            aabb.min.x() <= min_x + slack,
            "min.x={} > {min_x}",
            aabb.min.x()
        );
        assert!(
            aabb.min.y() <= min_y + slack,
            "min.y={} > {min_y}",
            aabb.min.y()
        );
        assert!(
            aabb.min.z() <= min_z + slack,
            "min.z={} > {min_z}",
            aabb.min.z()
        );
        assert!(
            aabb.max.x() >= max_x - slack,
            "max.x={} < {max_x}",
            aabb.max.x()
        );
        assert!(
            aabb.max.y() >= max_y - slack,
            "max.y={} < {max_y}",
            aabb.max.y()
        );
        assert!(
            aabb.max.z() >= max_z - slack,
            "max.z={} < {max_z}",
            aabb.max.z()
        );
    }

    // ── Bounding box ─────────────────────────────────────────────

    #[test]
    fn unit_cube_bounding_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        let tol = Tolerance::new();

        assert!(tol.approx_eq(aabb.min.x(), 0.0));
        assert!(tol.approx_eq(aabb.min.y(), 0.0));
        assert!(tol.approx_eq(aabb.min.z(), 0.0));
        assert!(tol.approx_eq(aabb.max.x(), 1.0));
        assert!(tol.approx_eq(aabb.max.y(), 1.0));
        assert!(tol.approx_eq(aabb.max.z(), 1.0));
    }

    /// AABB for a sphere must include the full radius extent in all axes.
    /// Sphere at origin with r=5: AABB should be [-5,-5,-5] to [5,5,5].
    #[test]
    fn sphere_bounding_box() {
        use crate::primitives::make_sphere;

        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 5.0, 8).unwrap();

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        // Sphere AABB is expanded by expand_aabb_for_surface to include ±r.
        assert_aabb_contains(&aabb, -5.0, -5.0, -5.0, 5.0, 5.0, 5.0);
    }

    /// AABB for a cylinder at origin, r=3, h=10 (z=0..10).
    /// Must include the full circular extent: x,y ∈ [-3,3].
    #[test]
    fn cylinder_bounding_box() {
        use crate::primitives::make_cylinder;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        assert_aabb_contains(&aabb, -3.0, -3.0, 0.0, 3.0, 3.0, 10.0);
    }

    /// AABB for a torus at origin, R=10, r=3, axis along Z.
    /// Radial extent: ±(R+r) = ±13 in x,y.
    /// Axial extent: ±r = ±3 in z.
    #[test]
    fn torus_bounding_box() {
        use crate::primitives::make_torus;

        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 10.0, 3.0, 16).unwrap();

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        // Radial: ±(R+r) = ±13
        assert!(aabb.min.x() <= -13.0 + 1e-6, "min.x={}", aabb.min.x());
        assert!(aabb.min.y() <= -13.0 + 1e-6, "min.y={}", aabb.min.y());
        assert!(aabb.max.x() >= 13.0 - 1e-6, "max.x={}", aabb.max.x());
        assert!(aabb.max.y() >= 13.0 - 1e-6, "max.y={}", aabb.max.y());
        // Axial: ±r = ±3
        assert!(aabb.min.z() <= -3.0 + 1e-6, "min.z={}", aabb.min.z());
        assert!(aabb.max.z() >= 3.0 - 1e-6, "max.z={}", aabb.max.z());
    }

    // ── Volume: analytic primitives (1e-8 tolerance) ─────────────

    #[test]
    fn unit_cube_volume() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        // Unit cube is all-planar — volume is computed via polygon method,
        // which should be exact to floating-point precision.
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        assert_rel(vol, 1.0, 1e-8, "unit cube volume");
    }

    /// make_box(10,10,10) → volume = 1000.0 exactly.
    /// Previously tested with 5% tolerance (|vol-1000| < 50) — absurdly loose
    /// for a pure-planar box that uses no tessellation.
    #[test]
    fn box_volume_exact() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        // All-planar solid → exact polygon-based volume.
        assert_rel(vol, 1000.0, 1e-8, "10×10×10 box volume");
    }

    /// Non-cube rectangular box: volume = dx × dy × dz.
    #[test]
    fn rectangular_box_volume_exact() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.5, 7.3, 4.1).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        // 2.5 × 7.3 × 4.1 = 74.825
        assert_rel(vol, 2.5 * 7.3 * 4.1, 1e-8, "2.5×7.3×4.1 box volume");
    }

    #[test]
    fn sphere_volume_analytic_exact() {
        use crate::primitives::make_sphere;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        // Low segment count — analytic path must not depend on tessellation quality.
        let solid = make_sphere(&mut topo, 3.0, 4).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = (4/3)πr³ = (4/3)π(27) ≈ 113.097
        let expected = 4.0 / 3.0 * PI * 27.0;
        assert_rel(vol, expected, 1e-10, "sphere r=3 volume");
    }

    #[test]
    fn cylinder_volume_analytic_exact() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 5.0, 20.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = πr²h = π(25)(20) ≈ 1570.796
        let expected = PI * 25.0 * 20.0;
        assert_rel(vol, expected, 1e-10, "cylinder r=5 h=20 volume");
    }

    #[test]
    fn cone_pointed_volume_analytic_exact() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 5.0, 0.0, 15.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = (π/3)r²h = (π/3)(25)(15) ≈ 392.699
        let expected = PI / 3.0 * 25.0 * 15.0;
        assert_rel(vol, expected, 1e-10, "pointed cone r=5 h=15 volume");
    }

    #[test]
    fn cone_frustum_volume_analytic_exact() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = (πh/3)(r₁² + r₁r₂ + r₂²) = (π·3/3)(4+2+1) ≈ 21.991
        let expected = PI / 3.0 * 3.0 * (4.0 + 2.0 + 1.0);
        assert_rel(vol, expected, 1e-10, "frustum r1=2 r2=1 h=3 volume");
    }

    #[test]
    fn torus_volume_analytic_exact() {
        use crate::primitives::make_torus;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 10.0, 3.0, 32).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = 2π²Rr² = 2π²(10)(9) ≈ 1776.529
        let expected = 2.0 * PI * PI * 10.0 * 9.0;
        assert_rel(vol, expected, 1e-10, "torus R=10 r=3 volume");
    }

    // ── Volume: tessellation-based (1e-4 tolerance) ──────────────

    /// Ellipsoid via non-uniform scale of a unit sphere.
    /// V = (4/3)π·a·b·c where a,b,c are the semi-axes.
    ///
    /// Non-uniform scale defeats the analytic sphere detector (vertex
    /// distances no longer match stored radius), so this goes through
    /// tessellation. A fine deflection (0.01) is needed because the NURBS
    /// ellipsoid has high curvature variation across its surface.
    #[test]
    fn ellipsoid_volume() {
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, 1.0, 16).unwrap();
        let mat = brepkit_math::mat::Mat4::scale(5.0, 3.0, 2.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        // Use fine deflection (0.01) for NURBS ellipsoid — the adaptive
        // tessellator needs small chord tolerance to refine the high-curvature
        // regions near the minor axis (z semi-axis = 2).
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        // V = (4/3)π·5·3·2 = 40π ≈ 125.664
        let expected = 4.0 / 3.0 * PI * 5.0 * 3.0 * 2.0;
        assert_rel(vol, expected, 0.01, "ellipsoid 5×3×2 volume");
    }

    /// Extruded 2×3 rectangle by 4 → volume = 2×3×4 = 24.0 exactly.
    #[test]
    fn extruded_box_volume() {
        use brepkit_math::vec::{Point3, Vec3 as V};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-10));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 0.0, 0.0), 1e-10));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 3.0, 0.0), 1e-10));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 3.0, 0.0), 1e-10));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);
        let face_id = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: V::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let solid =
            crate::extrude::extrude(&mut topo, face_id, V::new(0.0, 0.0, 1.0), 4.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        // All-planar extrusion: 2 × 3 × 4 = 24.0 exactly.
        assert_rel(vol, 24.0, 1e-8, "extruded 2×3×4 box volume");
    }

    // ── Volume: operations that involve NURBS (1e-4 tolerance) ───

    /// Fillet on one edge of a 20³ box.
    ///
    /// A rolling-ball fillet of radius r on one edge of length L removes
    /// a prismatic quarter-cylinder notch:
    ///   V_removed = (1 - π/4) × r² × L
    ///
    /// For r=2, L=20:
    ///   V_removed = (1 - π/4) × 4 × 20 = (1 - 0.7854) × 80 ≈ 17.168
    ///   V_expected = 8000 - 17.168 ≈ 7982.83
    ///
    /// Previously: bounds were 7000 < vol < 8000 (12.5% tolerance window).
    /// Now: 1% tolerance around the derived value.
    #[test]
    fn fillet_single_edge_volume() {
        use brepkit_topology::explorer;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        let edges: Vec<_> = explorer::solid_edges(&topo, solid).unwrap();
        let filleted =
            crate::fillet::fillet_rolling_ball(&mut topo, solid, &[edges[0]], 2.0).unwrap();
        let vol = solid_volume(&topo, filleted, 0.01).unwrap();
        // V = 20³ - (1 - π/4) × r² × L = 8000 - (1-π/4)×4×20 ≈ 7982.83
        let expected = 8000.0 - (1.0 - PI / 4.0) * 4.0 * 20.0;
        assert_rel(vol, expected, 0.01, "fillet r=2 on 20³ box, one edge");
    }

    // ── Surface area ─────────────────────────────────────────────

    #[test]
    fn unit_cube_surface_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let area = solid_surface_area(&topo, solid, 0.1).unwrap();
        // 6 faces × 1.0 each = 6.0 exactly for all-planar solid.
        assert_rel(area, 6.0, 1e-8, "unit cube surface area");
    }

    #[test]
    fn unit_cube_face_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let area = face_area(&topo, fid, 0.1).unwrap();
            assert_rel(area, 1.0, 1e-8, "unit cube face area");
        }
    }

    /// Box surface area = 2(ab + bc + ac).
    /// 10×10×10 → SA = 6 × 100 = 600.
    #[test]
    fn box_surface_area_exact() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let area = solid_surface_area(&topo, solid, 0.1).unwrap();
        assert_rel(area, 600.0, 1e-8, "10³ box surface area");
    }

    /// Cylinder total surface area = 2πr² + 2πrh (two caps + lateral).
    /// r=3, h=10 → SA = 2π(9) + 2π(3)(10) = 18π + 60π = 78π ≈ 245.04.
    #[test]
    fn cylinder_surface_area() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let area = solid_surface_area(&topo, solid, 0.01).unwrap();
        // SA = 2πr² + 2πrh = 2π(9 + 30) = 78π ≈ 245.04
        let expected = 2.0 * PI * (9.0 + 30.0);
        assert_rel(area, expected, 1e-4, "cylinder r=3 h=10 surface area");
    }

    /// Sphere surface area = 4πr². r=5 → SA = 100π ≈ 314.16.
    #[test]
    fn sphere_surface_area() {
        use crate::primitives::make_sphere;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 5.0, 16).unwrap();
        let area = solid_surface_area(&topo, solid, 0.01).unwrap();
        // SA = 4πr² = 4π(25) = 100π ≈ 314.159
        let expected = 4.0 * PI * 25.0;
        assert_rel(area, expected, 1e-4, "sphere r=5 surface area");
    }

    // ── Center of mass ───────────────────────────────────────────

    #[test]
    fn unit_cube_center_of_mass() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        // Symmetric solid centered at (0.5, 0.5, 0.5).
        assert_rel(com.x(), 0.5, 1e-8, "cube CoM x");
        assert_rel(com.y(), 0.5, 1e-8, "cube CoM y");
        assert_rel(com.z(), 0.5, 1e-8, "cube CoM z");
    }

    /// Cylinder r=3, h=10, base at z=0. CoM is at (0, 0, h/2) = (0, 0, 5).
    #[test]
    fn cylinder_center_of_mass() {
        use crate::primitives::make_cylinder;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.01).unwrap();
        // By symmetry: x=0, y=0. By uniform density: z = h/2 = 5.
        assert_rel(com.x().abs(), 0.0, 1e-4, "cylinder CoM x");
        assert_rel(com.y().abs(), 0.0, 1e-4, "cylinder CoM y");
        assert_rel(com.z(), 5.0, 1e-4, "cylinder CoM z");
    }

    /// Pointed cone, r_bottom=4, h=12, base at z=0.
    /// CoM_z = h/4 = 3.0 (standard formula for solid cone).
    #[test]
    fn cone_center_of_mass() {
        use crate::primitives::make_cone;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 4.0, 0.0, 12.0).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.01).unwrap();
        // Cone CoM is at h/4 from the base.
        assert_rel(com.x().abs(), 0.0, 1e-4, "cone CoM x");
        assert_rel(com.y().abs(), 0.0, 1e-4, "cone CoM y");
        assert_rel(com.z(), 3.0, 1e-4, "cone CoM z = h/4 = 3");
    }

    /// Non-symmetric box: 2×3×4, origin at (0,0,0).
    /// CoM = (1, 1.5, 2) — midpoint of each dimension.
    #[test]
    fn rectangular_box_center_of_mass() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        assert_rel(com.x(), 1.0, 1e-8, "rect box CoM x");
        assert_rel(com.y(), 1.5, 1e-8, "rect box CoM y");
        assert_rel(com.z(), 2.0, 1e-8, "rect box CoM z");
    }

    // ── Edge & wire lengths ──────────────────────────────────────

    #[test]
    fn edge_length_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                let len = edge_length(&topo, oe.edge()).unwrap();
                assert!(
                    tol.approx_eq(len, 1.0),
                    "unit cube edge should have length ~1.0, got {len}"
                );
            }
        }
    }

    /// Cylinder circumference edge: 2πr.
    /// r=3 → 6π ≈ 18.8496.
    #[test]
    fn edge_length_circle() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();

        // Find a circular edge (cap boundary).
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found_circle = false;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            if matches!(face.surface(), FaceSurface::Plane { .. }) {
                let wire = topo.wire(face.outer_wire()).unwrap();
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_)) {
                        let len = edge_length(&topo, oe.edge()).unwrap();
                        // Circumference = 2πr = 6π ≈ 18.8496
                        assert_rel(len, 2.0 * PI * 3.0, 1e-8, "circle edge length");
                        found_circle = true;
                    }
                }
            }
        }
        assert!(found_circle, "should have found at least one circular edge");
    }

    #[test]
    fn face_perimeter_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let perim = face_perimeter(&topo, fid).unwrap();
            assert_rel(perim, 4.0, 1e-8, "unit cube face perimeter");
        }
    }

    #[test]
    fn wire_length_rectangle() {
        use brepkit_topology::builder::make_rectangle_face;

        let mut topo = Topology::new();
        let fid = make_rectangle_face(&mut topo, 3.0, 5.0).unwrap();

        let face = topo.face(fid).unwrap();
        let len = wire_length(&topo, face.outer_wire()).unwrap();
        // Perimeter = 2(3+5) = 16.0 exactly.
        assert_rel(len, 16.0, 1e-8, "3×5 rectangle perimeter");
    }

    // ── Boolean volume ───────────────────────────────────────────

    /// Boolean cut must reduce volume: cut(box, cylinder) < box volume.
    ///
    /// Regression test for the cylinder band classification bug.
    #[test]
    fn cut_box_cylinder_volume_decreases() {
        use crate::boolean::{BooleanOp, boolean};
        use crate::primitives::{make_box, make_cylinder};
        use crate::transform::transform_solid;
        use brepkit_math::mat::Mat4;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let bx = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let cyl = make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        transform_solid(&mut topo, cyl, &Mat4::translation(5.0, 5.0, 0.0)).unwrap();
        let cut = boolean(&mut topo, BooleanOp::Cut, bx, cyl).unwrap();

        let box_vol = solid_volume(&topo, bx, 0.01).unwrap();
        let cut_vol = solid_volume(&topo, cut, 0.01).unwrap();
        // V = 10³ - πr²h = 1000 - π(9)(10) = 1000 - 90π ≈ 717.35
        let expected = 1000.0 - PI * 9.0 * 10.0;

        let s = topo.solid(cut).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            7,
            "expected 7 faces (6 plane + 1 cylinder bore)"
        );

        assert!(
            cut_vol < box_vol,
            "cut volume ({cut_vol:.2}) must be less than box volume ({box_vol:.2})"
        );
        assert_rel(cut_vol, expected, 0.02, "cut(box, cylinder) volume");
    }

    // ── Edge case: degenerate and boundary inputs ────────────────

    /// Volume and AABB of a very thin box (one dimension near-zero).
    /// 10 × 10 × 0.001 → V = 0.1, SA = 2(100 + 0.01 + 0.01) = 200.04.
    #[test]
    fn thin_box_volume_and_area() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 0.001).unwrap();
        let vol = solid_volume(&topo, solid, 0.001).unwrap();
        let area = solid_surface_area(&topo, solid, 0.001).unwrap();
        // V = 10 × 10 × 0.001 = 0.1
        assert_rel(vol, 0.1, 1e-8, "thin box volume");
        // SA = 2(10×10 + 10×0.001 + 10×0.001) = 2(100+0.01+0.01) = 200.04
        assert_rel(area, 200.04, 1e-8, "thin box surface area");
    }

    /// Very large box to check numerical stability at scale.
    /// 1000 × 1000 × 1000 → V = 1e9.
    #[test]
    fn large_box_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1000.0, 1000.0, 1000.0).unwrap();
        let vol = solid_volume(&topo, solid, 1.0).unwrap();
        assert_rel(vol, 1e9, 1e-8, "1000³ box volume");
    }

    /// Very small box to check numerical stability at micro-scale.
    /// 0.001 × 0.001 × 0.001 → V = 1e-9.
    #[test]
    fn tiny_box_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 0.001, 0.001, 0.001).unwrap();
        let vol = solid_volume(&topo, solid, 0.0001).unwrap();
        assert_rel(vol, 1e-9, 1e-8, "0.001³ box volume");
    }

    /// Cylinder with very small radius: r=0.01, h=10.
    /// V = π(0.0001)(10) = 0.001π ≈ 0.003142.
    #[test]
    fn thin_cylinder_volume() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 0.01, 10.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * 0.0001 * 10.0;
        assert_rel(vol, expected, 1e-8, "thin cylinder r=0.01 h=10 volume");
    }

    /// Flat cylinder (disc-like): r=10, h=0.01.
    /// V = π(100)(0.01) = π ≈ 3.1416.
    #[test]
    fn flat_cylinder_volume() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 10.0, 0.01).unwrap();
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * 100.0 * 0.01;
        assert_rel(vol, expected, 1e-8, "flat cylinder r=10 h=0.01 volume");
    }

    /// Nearly-pointed frustum: r_bottom=5, r_top=0.001, h=10.
    /// V = (πh/3)(r₁² + r₁r₂ + r₂²) = (10π/3)(25 + 0.005 + 0.000001) ≈ 261.80.
    #[test]
    fn near_pointed_frustum_volume() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 5.0, 0.001, 10.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        let r1 = 5.0_f64;
        let r2 = 0.001_f64;
        let expected = PI * 10.0 / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2);
        assert_rel(vol, expected, 1e-8, "near-pointed frustum volume");
    }

    // ── Composition: measure after operations ────────────────────

    /// Volume after transform: uniform scale by 2 triples each dimension.
    /// Unit cube → 2×2×2 cube → V = 8.
    #[test]
    fn volume_after_uniform_scale() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mat = brepkit_math::mat::Mat4::scale(2.0, 2.0, 2.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        assert_rel(vol, 8.0, 1e-8, "unit cube scaled ×2 volume");
    }

    /// CoM shifts correctly under translation.
    /// Box (0,0,0)-(1,1,1) translated by (10,20,30) → CoM = (10.5, 20.5, 30.5).
    #[test]
    fn center_of_mass_after_translation() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(10.0, 20.0, 30.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        assert_rel(com.x(), 10.5, 1e-8, "translated CoM x");
        assert_rel(com.y(), 20.5, 1e-8, "translated CoM y");
        assert_rel(com.z(), 30.5, 1e-8, "translated CoM z");
    }

    /// Volume is invariant under rotation.
    #[test]
    fn volume_invariant_under_rotation() {
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 3.0, 4.0, 5.0).unwrap();
        let vol_before = solid_volume(&topo, solid, 0.1).unwrap();

        // Rotate 45° around Z axis.
        let mat = brepkit_math::mat::Mat4::rotation_z(PI / 4.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        let vol_after = solid_volume(&topo, solid, 0.1).unwrap();

        // V = 3×4×5 = 60, should be unchanged.
        assert_rel(vol_before, 60.0, 1e-8, "box volume before rotation");
        assert_rel(vol_after, 60.0, 1e-8, "box volume after rotation");
    }

    // ── Error path validation ────────────────────────────────────

    /// Cone total surface area = π*r*l + π*r² (lateral + base cap).
    /// r=1, h=1: slant l=√2, lateral = π*1*√2 = π√2 ≈ 4.4429.
    /// Total = π√2 + π ≈ 7.584.
    #[test]
    fn cone_face_area_analytic() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 1.0, 0.0, 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        let mut lateral_area = 0.0;
        let mut cap_area = 0.0;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let area = face_area(&topo, fid, 0.01).unwrap();
            match face.surface() {
                FaceSurface::Cone(_) => lateral_area += area,
                FaceSurface::Plane { .. } => cap_area += area,
                _ => panic!("unexpected surface type in cone"),
            }
        }

        // Lateral area = π*r*l = π*1*√2 ≈ 4.4429
        let slant = 2.0_f64.sqrt();
        assert_rel(lateral_area, PI * slant, 1e-8, "cone r=1 h=1 lateral area");
        // Cap (base disk) = π*r² = π ≈ 3.1416
        assert_rel(cap_area, PI, 2e-4, "cone r=1 h=1 base cap area");
    }

    /// Torus total surface area = 4π²*R*r.
    /// R=2, r=0.5: area = 4π²*2*0.5 = 4π² ≈ 39.478.
    #[test]
    fn torus_face_area_analytic() {
        use crate::primitives::make_torus;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        // Use 32 segments for a decent tessellation of boundary
        let solid = make_torus(&mut topo, 2.0, 0.5, 32).unwrap();

        let area = solid_surface_area(&topo, solid, 0.01).unwrap();
        // SA = 4π²Rr = 4π²*2*0.5 = 4π² ≈ 39.478
        let expected = 4.0 * PI * PI * 2.0 * 0.5;
        assert_rel(area, expected, 1e-8, "torus R=2 r=0.5 surface area");
    }

    /// face_area with a non-planar face and zero deflection should still
    /// return a reasonable result (tessellation at maximum detail).
    #[test]
    fn cylinder_face_area_analytic() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        let mut lateral_area = 0.0;
        let mut cap_area = 0.0;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let area = face_area(&topo, fid, 0.01).unwrap();
            match face.surface() {
                FaceSurface::Cylinder(_) => lateral_area += area,
                FaceSurface::Plane { .. } => cap_area += area,
                _ => panic!("unexpected surface type in cylinder"),
            }
        }

        // Lateral area = 2πrh = 2π(3)(10) = 60π ≈ 188.496
        assert_rel(
            lateral_area,
            2.0 * PI * 3.0 * 10.0,
            1e-4,
            "cylinder lateral area",
        );
        // Two caps = 2πr² = 2π(9) = 18π ≈ 56.549
        // Cap area uses Newell's method on 256-sample polygon of circle edge,
        // so discretization error is O(1/n²) ≈ 2e-5 per cap.
        assert_rel(cap_area, 2.0 * PI * 9.0, 2e-4, "cylinder cap area");
    }
}
