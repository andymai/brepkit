//! Face and solid surface area computation.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::tessellate;

use super::helpers::{collect_solid_face_ids, collect_wire_positions, compute_angular_range};

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
                // circular cross-section. For full cylinders this gives 2pi; for
                // partial cylinders it gives the actual angular extent.
                let u_vals: Vec<f64> = positions
                    .iter()
                    .map(|p| {
                        let rel = *p - origin;
                        let along = axis.dot(rel);
                        let radial = rel - axis * along;
                        radial.y().atan2(radial.x())
                    })
                    .collect();
                let u_min = u_vals.iter().copied().fold(f64::INFINITY, f64::min);
                let u_max = u_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let angular_span = u_max - u_min;
                // If the angular span covers most of a full circle (> 350 deg),
                // treat it as a full revolution -- boundary sampling may not
                // reach exactly +/-pi.
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
            // Spherical zone area = 2*pi*r^2 * (sin(v_max) - sin(v_min))
            // where v is the latitude parameter (-pi/2 to pi/2).
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
///   `P(u,v) = apex + v*(cos(a)*radial(u) + sin(a)*axis)`
/// the surface element is `dA = v * cos(a) * du * dv`.
///
/// Integrating over `u in [u0,u1], v in [v0,v1]`:
///   `area = cos(a) * (u1-u0) * (v1^2-v0^2) / 2`
///
/// This equals `pi*(r0+r1)*slant*angle_frac` (standard frustum lateral area)
/// when verified: `r0=v0*cos(a)`, `r1=v1*cos(a)`, slant=|v1-v0|,
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

    if v_vals.is_empty() {
        return Ok(0.0);
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
/// Integrating over `u in [u0,u1], v in [v0,v1]`:
///   `area = r * (u1-u0) * [R*(v1-v0) + r*(sin(v1)-sin(v0))]`
///
/// For a full torus: `area = r * 2pi * (R*2pi + r*0) = 4pi^2*Rr`
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

    if v_vals.is_empty() {
        return Ok(0.0);
    }
    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (v_max - v_min).abs() < 1e-15 {
        // Full torus: v wraps from 0 to 2pi, all boundary v-vals are the same.
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
