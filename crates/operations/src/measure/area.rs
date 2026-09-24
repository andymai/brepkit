//! Face and solid surface area computation.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::tessellate;

use super::helpers::{
    collect_solid_face_ids, collect_wire_positions, compute_angular_range, planar_wire_signed_area2,
};

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
            if let Some(area) = cylinder_face_uv_area(topo, face_id, cyl)? {
                return Ok(area);
            }
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
                let sweep = if let Some(s) = cylinder_arc_sweep(topo, face_id, axis, origin)? {
                    s
                } else {
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
                    if angular_span > 330.0_f64.to_radians() {
                        std::f64::consts::TAU
                    } else {
                        angular_span
                    }
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
                let mut holes = 0.0;
                for &wire in face.inner_wires() {
                    holes += sphere_hole_area(topo, sph, wire)?;
                }
                Ok(2.0 * std::f64::consts::PI * r * r * (v_max.sin() - v_min.sin()) - holes)
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

/// The area a hole takes from a sphere cap: `R² |∮ sin v du|` inside a loop
/// that winds none of the sphere's u (a drill's entry), and for a loop around
/// the cap's pole the cap beyond it, `R² (2π − |∮ sin v du|)` (a bore's rim).
/// The integral runs along the loop by midpoint sums at two resolutions and
/// a Richardson step.
fn sphere_hole_area(
    topo: &Topology,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<f64, crate::OperationsError> {
    use std::f64::consts::{PI, TAU};
    let wrap = |d: f64| (d + PI).rem_euclid(TAU) - PI;
    let (mut sweep, mut progress) = (0.0, 0.0);
    for oe in topo.wire(wire_id)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        let (from, to) = if oe.is_forward() { (t0, t1) } else { (t1, t0) };
        let at = |t: f64| sphere.project_point(edge.curve().evaluate_with_endpoints(t, sp, ep));
        let sums = |n: usize| {
            let (mut sweep, mut progress) = (0.0, 0.0);
            #[allow(clippy::cast_precision_loss)]
            let step = (to - from) / n as f64;
            let mut u_prev = at(from).0;
            for k in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let tk = from + step * k as f64;
                let (_, vm) = at(tk + 0.5 * step);
                let (un, _) = at(tk + step);
                sweep += vm.sin() * wrap(un - u_prev);
                progress += wrap(un - u_prev);
                u_prev = un;
            }
            (sweep, progress)
        };
        let (coarse, _) = sums(128);
        let (fine, turned) = sums(256);
        sweep += (4.0 * fine - coarse) / 3.0;
        progress += turned;
    }
    let r2 = sphere.radius() * sphere.radius();
    Ok(if progress.abs() > PI {
        r2 * (TAU - sweep.abs())
    } else {
        r2 * sweep.abs()
    })
}

/// The area of a cylinder face whose boundary is not a rectangle in
/// `(u, v)` (a wall trimmed by an oblique plane's ellipse, say): `r` times
/// the region's area in `(u, v)`, by Green's theorem `A = ∮ u dv` along
/// each wire, less its holes. `Ok(None)` for a face bounded only by rulings
/// and rims, which the rectangle below measures exactly, or when a wire's
/// unwrapped `u` does not close.
fn cylinder_face_uv_area(
    topo: &Topology,
    face_id: FaceId,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    let face = topo.face(face_id)?;
    let wires: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    let axis = cyl.axis();
    let mut rectangular = true;
    for &wid in &wires {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            rectangular &= match edge.curve() {
                EdgeCurve::Line => {
                    let a = topo.vertex(edge.start())?.point();
                    let b = topo.vertex(edge.end())?.point();
                    (b - a).cross(axis).length() <= 1e-9 * (b - a).length().max(1.0)
                }
                EdgeCurve::Circle(c) => c.normal().cross(axis).length() <= 1e-9,
                EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
            };
        }
    }
    if rectangular {
        return Ok(None);
    }
    let mut total = 0.0;
    for (k, &wid) in wires.iter().enumerate() {
        let Some(signed) = wire_uv_area_on_cylinder(topo, wid, cyl)? else {
            return Ok(None);
        };
        if k == 0 {
            total += signed.abs();
        } else {
            total -= signed.abs();
        }
    }
    Ok(Some(cyl.radius() * total))
}

/// `∮ u dv` along one wire on a cylinder, with `u` unwrapped continuously,
/// by Gauss-Legendre quadrature in each edge's own parameter; `None` when
/// the unwrapped `u` does not return to its start.
fn wire_uv_area_on_cylinder(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::{PI, TAU};
    const SEGMENTS: usize = 16;
    const ORDER: usize = 8;
    let axis = cyl.axis();
    let origin = cyl.origin();
    let v_of = |p: Point3| axis.dot(p - origin);
    let u_near = |p: Point3, near: Option<f64>| {
        let (u, _) = cyl.project_point(p);
        near.map_or(u, |n| u - ((u - n + PI) / TAU).floor() * TAU)
    };
    let points = brepkit_math::quadrature::gauss_legendre_points(ORDER);
    let mut sum = 0.0;
    let mut first_u = None;
    let mut last_u: Option<f64> = None;
    for oe in topo.wire(wire_id)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let start = topo.vertex(edge.start())?.point();
        let end = topo.vertex(edge.end())?.point();
        let curve = edge.curve();
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        // A closed conic's domain starts at its frame's origin; walk it from
        // its vertex so the wire's u stays continuous.
        let (t0, t1) = match curve {
            EdgeCurve::Circle(c) if edge.start() == edge.end() => {
                let tv = c.project(start);
                (tv, tv + (t1 - t0))
            }
            EdgeCurve::Ellipse(e) if edge.start() == edge.end() => {
                let tv = e.project(start);
                (tv, tv + (t1 - t0))
            }
            _ => (t0, t1),
        };
        let (ta, tb) = if oe.is_forward() { (t0, t1) } else { (t1, t0) };
        let at = |t: f64| curve.evaluate_with_endpoints(t, start, end);
        let mut u_prev = u_near(at(ta), last_u);
        if first_u.is_none() {
            first_u = Some(u_prev);
        }
        #[allow(clippy::cast_precision_loss)]
        for seg in 0..SEGMENTS {
            let a = ta + (tb - ta) * seg as f64 / SEGMENTS as f64;
            let b = ta + (tb - ta) * (seg + 1) as f64 / SEGMENTS as f64;
            let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
            let h = 1e-6 * half.abs().max(1e-12);
            for gp in points {
                let t = mid + half * gp.x;
                let u = u_near(at(t), Some(u_prev));
                let dv = match curve {
                    EdgeCurve::Line => axis.dot(end - start),
                    EdgeCurve::Circle(c) => c.radius() * axis.dot(c.tangent(t)),
                    EdgeCurve::Ellipse(e) => axis.dot(e.tangent(t)),
                    EdgeCurve::NurbsCurve(_) => (v_of(at(t + h)) - v_of(at(t - h))) / (2.0 * h),
                };
                sum += gp.w * half * u * dv;
                u_prev = u;
            }
        }
        last_u = Some(u_near(at(tb), Some(u_prev)));
    }
    Ok(match (first_u, last_u) {
        (Some(a), Some(b)) if (a - b).abs() <= 1e-6 => Some(sum),
        _ => None,
    })
}

/// Angular sweep of a cylindrical face derived from its boundary arc edges.
///
/// `face_polygon` only contributes endpoint vertices for partial arcs, so a
/// point-based angular range misreads spans that cross the atan2 branch cut
/// (e.g. a 90-degree corner arc straddling u=pi reads as 270 degrees). The
/// stored `Circle` arcs carry the true span (CCW start→end around their own
/// axis): sum the spans per axial level and take the widest level.
///
/// Returns `None` when the boundary has no circle arcs (chord-polygon faces),
/// letting the caller fall back to point-based estimation.
fn cylinder_arc_sweep(
    topo: &Topology,
    face_id: FaceId,
    axis: Vec3,
    origin: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    // Axial-level merge guard. `v` is `axis · (center − origin)`, which
    // accumulates more rounding error than a single distance comparison, so
    // this stays looser than the 1e-7 default linear tolerance to keep arcs at
    // the same axial height from splitting into distinct levels.
    const LEVEL_MERGE_TOL: f64 = 1e-6;

    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut levels: Vec<(f64, f64)> = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let EdgeCurve::Circle(circle) = edge.curve() else {
            continue;
        };
        if edge.start() == edge.end() {
            return Ok(Some(std::f64::consts::TAU));
        }
        let sp = topo.vertex(edge.start())?.point();
        let ep = topo.vertex(edge.end())?.point();
        let ts = circle.project(sp);
        let mut te = circle.project(ep);
        if te <= ts {
            te += std::f64::consts::TAU;
        }
        let span = te - ts;
        let v = axis.dot(circle.center() - origin);
        if let Some(entry) = levels
            .iter_mut()
            .find(|(lv, _)| (*lv - v).abs() < LEVEL_MERGE_TOL)
        {
            entry.1 += span;
        } else {
            levels.push((v, span));
        }
    }
    Ok(levels
        .iter()
        .map(|&(_, span)| span.min(std::f64::consts::TAU))
        .fold(None, |acc: Option<f64>, span| {
            Some(acc.map_or(span, |a| a.max(span)))
        }))
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
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
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
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
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

    if v_vals.is_empty() {
        return Ok(0.0);
    }
    let mut v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mut v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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

    // A toroidal band (e.g. a rim fillet) is bounded by two rims at distinct v,
    // and v is periodic: the raw [v_min, v_max] may be the long (bulge) arc
    // rather than the short fillet arc. If the naive span exceeds π, take the
    // complementary (wrapped) arc instead.
    if v_max - v_min > std::f64::consts::PI {
        let new_min = v_max;
        let new_max = v_min + std::f64::consts::TAU;
        v_min = new_min;
        v_max = new_max;
    }

    let u_range = compute_angular_range(&mut u_vals);
    let (u0, u1) = u_range;

    let big_r = tor.major_radius();
    let small_r = tor.minor_radius();
    let area =
        small_r * (u1 - u0) * (big_r * (v_max - v_min) + small_r * (v_max.sin() - v_min.sin()));
    Ok(area.abs())
}

/// The area of a planar face, less its holes: exact by Green's theorem when
/// every edge is a line or a circle, else by Newell's method over the
/// sampled boundary.
fn planar_face_area(topo: &Topology, face_id: FaceId) -> Result<f64, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    let face = topo.face(face_id)?;
    let mut lines_and_circles = true;
    for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire)?.edges() {
            lines_and_circles &= matches!(
                topo.edge(oe.edge())?.curve(),
                EdgeCurve::Line | EdgeCurve::Circle(_)
            );
        }
    }
    if lines_and_circles
        && let FaceSurface::Plane { normal, .. } = face.surface()
        && let Ok(frame) =
            brepkit_math::frame::Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), *normal)
    {
        let mut exact = Some(0.0);
        for (k, wire) in std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .enumerate()
        {
            exact = match (
                exact,
                planar_wire_signed_area2(topo, wire, frame.x, frame.y)?,
            ) {
                (Some(total), Some((area2, _))) if k == 0 => Some(total + area2.abs() / 2.0),
                (Some(total), Some((area2, _))) => Some(total - area2.abs() / 2.0),
                _ => None,
            };
        }
        if let Some(area) = exact {
            return Ok(area.abs());
        }
    }
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
