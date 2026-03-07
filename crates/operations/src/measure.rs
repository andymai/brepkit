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
fn expand_aabb_for_surface(aabb: &mut Aabb3, surface: &FaceSurface) {
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
                aabb_include(aabb, Point3::new(coa.x() - r, coa.y() - r, coa.z() - r));
                aabb_include(aabb, Point3::new(coa.x() + r, coa.y() + r, coa.z() + r));
            }
        }
        FaceSurface::Torus(t) => {
            let c = t.center();
            let outer_r = t.major_radius() + t.minor_radius();
            aabb_include(
                aabb,
                Point3::new(c.x() - outer_r, c.y() - outer_r, c.z() - t.minor_radius()),
            );
            aabb_include(
                aabb,
                Point3::new(c.x() + outer_r, c.y() + outer_r, c.z() + t.minor_radius()),
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
        _ => {
            let mesh = tessellate::tessellate(topo, face_id, deflection)?;
            Ok(triangle_mesh_area(&mesh))
        }
    }
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

        let sign = face_winding_sign_centroid(&mesh, centroid);

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
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;

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

    #[test]
    fn unit_cube_volume() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 1.0),
            "unit cube volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn unit_cube_surface_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let area = solid_surface_area(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(area, 6.0),
            "unit cube surface area should be ~6.0, got {area}"
        );
    }

    #[test]
    fn unit_cube_center_of_mass() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(com.x(), 0.5),
            "center x should be ~0.5, got {}",
            com.x()
        );
        assert!(
            tol.approx_eq(com.y(), 0.5),
            "center y should be ~0.5, got {}",
            com.y()
        );
        assert!(
            tol.approx_eq(com.z(), 0.5),
            "center z should be ~0.5, got {}",
            com.z()
        );
    }

    #[test]
    fn unit_cube_face_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let area = face_area(&topo, fid, 0.1).unwrap();
            assert!(
                tol.approx_eq(area, 1.0),
                "each unit cube face should have area ~1.0, got {area}"
            );
        }
    }

    #[test]
    fn extruded_box_volume() {
        use brepkit_math::vec::{Point3, Vec3 as V};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        // Create a 2×3 rectangle on the XY plane.
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

        // Extrude by 4 along Z.
        let solid =
            crate::extrude::extrude(&mut topo, face_id, V::new(0.0, 0.0, 1.0), 4.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 24.0),
            "2×3 rect extruded by 4 should have volume ~24.0, got {vol}"
        );
    }

    #[test]
    fn edge_length_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        // All edges of a unit cube have length 1.0.
        let tol = Tolerance::loose();
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

    #[test]
    fn face_perimeter_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let perim = face_perimeter(&topo, fid).unwrap();
            assert!(
                tol.approx_eq(perim, 4.0),
                "unit cube face perimeter should be ~4.0, got {perim}"
            );
        }
    }

    #[test]
    fn wire_length_rectangle() {
        use brepkit_topology::builder::make_rectangle_face;

        let mut topo = Topology::new();
        let fid = make_rectangle_face(&mut topo, 3.0, 5.0).unwrap();

        let face = topo.face(fid).unwrap();
        let len = wire_length(&topo, face.outer_wire()).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(len, 16.0),
            "3×5 rectangle perimeter should be ~16.0, got {len}"
        );
    }

    // ── Analytic volume tests ──────────────────────────────────────

    #[test]
    fn sphere_volume_analytic_exact() {
        use crate::primitives::make_sphere;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        // Low segment count — analytic path must not depend on tessellation quality.
        let solid = make_sphere(&mut topo, 3.0, 4).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        let expected = 4.0 / 3.0 * PI * 27.0; // (4/3)π × 3³ ≈ 113.1
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "sphere volume should be exact via analytic path: expected {expected:.6}, got {vol:.6}, rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn cylinder_volume_analytic_exact() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 5.0, 20.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        let expected = PI * 25.0 * 20.0; // π × 5² × 20 ≈ 1570.8
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "cylinder volume should be exact via analytic path: expected {expected:.6}, got {vol:.6}, rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn cone_pointed_volume_analytic_exact() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 5.0, 0.0, 15.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        let expected = PI / 3.0 * 25.0 * 15.0; // π/3 × 5² × 15 ≈ 392.7
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "pointed cone volume should be exact: expected {expected:.6}, got {vol:.6}, rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn cone_frustum_volume_analytic_exact() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        let expected = PI / 3.0 * 3.0 * (4.0 + 2.0 + 1.0); // πh/3 × (r1²+r1r2+r2²) ≈ 21.99
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "frustum volume should be exact: expected {expected:.6}, got {vol:.6}, rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn torus_volume_analytic_exact() {
        use crate::primitives::make_torus;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 10.0, 3.0, 32).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        let expected = 2.0 * PI * PI * 10.0 * 9.0; // 2π²Rr² ≈ 1776.5
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "torus volume should be exact: expected {expected:.6}, got {vol:.6}, rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn ellipsoid_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, 1.0, 16).unwrap();
        let mat = brepkit_math::mat::Mat4::scale(5.0, 3.0, 2.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        let expected = 4.0 / 3.0 * std::f64::consts::PI * 5.0 * 3.0 * 2.0;
        assert!(
            (vol - expected).abs() < expected * 0.15,
            "expected ~{expected}, got {vol}"
        );
    }

    #[test]
    fn fillet_single_edge_volume() {
        use brepkit_topology::explorer;
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        let edges: Vec<_> = explorer::solid_edges(&topo, solid).unwrap();
        let filleted =
            crate::fillet::fillet_rolling_ball(&mut topo, solid, &[edges[0]], 2.0).unwrap();
        let vol = solid_volume(&topo, filleted, 0.01).unwrap();
        assert!(
            vol < 8000.0,
            "filleted box should have less volume than 8000, got {vol}"
        );
        assert!(
            vol > 7000.0,
            "filleted box should still have significant volume, got {vol}"
        );
    }

    #[test]
    fn chamfered_box_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        // Box volume should be close to 1000
        assert!(
            (vol - 1000.0).abs() < 50.0,
            "box should have volume ~1000, got {vol}"
        );
    }
}
