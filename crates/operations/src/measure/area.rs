//! Face and solid surface area computation.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::tessellate;

use super::helpers::{
    collect_solid_face_ids, collect_wire_positions, compute_angular_range,
    planar_wire_signed_area2, torus_band_v_range, torus_sector_u_range, traversal_spans,
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
            let mut holes = 0.0;
            for &wire in face.inner_wires() {
                holes += sphere_hole_area(topo, sph, wire)?;
            }
            // A face bounded by one circle is the cap on the side its boundary
            // winds about, and a cap `h` high has `2 pi r h` of the sphere,
            // however the circle tilts.
            if let [oe] = topo.wire(face.outer_wire())?.edges()
                && let edge = topo.edge(oe.edge())?
                && edge.start() == edge.end()
                && let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve()
                && let Ok(winding) = newell_normal(&positions).normalize()
            {
                let rise = winding.dot(circle.center() - sph.center());
                return Ok(2.0 * std::f64::consts::PI * r * (r - rise) - holes);
            }
            if let Some(area) = sphere_face_uv_area(topo, face_id, sph, holes, deflection)? {
                return Ok(area);
            }
            if positions.len() >= 3 {
                let v_vals: Vec<f64> = positions.iter().map(|p| sph.project_point(*p).1).collect();
                let avg_v: f64 = v_vals.iter().sum::<f64>() / v_vals.len() as f64;
                let signed_area = newell_signed_z_area(&positions);
                let (v_min, v_max) = if signed_area > 0.0 {
                    (avg_v, std::f64::consts::FRAC_PI_2)
                } else {
                    (-std::f64::consts::FRAC_PI_2, avg_v)
                };
                Ok(2.0 * std::f64::consts::PI * r * r * (v_max.sin() - v_min.sin()) - holes)
            } else {
                // Full sphere fallback
                Ok(4.0 * std::f64::consts::PI * r * r)
            }
        }
        FaceSurface::Cone(cone) => match cone_face_uv_area(topo, face_id, cone)? {
            Some(area) => Ok(area),
            None => analytic_cone_face_area(topo, face_id),
        },
        FaceSurface::Torus(tor) => match torus_face_uv_area(topo, face_id, tor)? {
            Some(area) => Ok(area),
            None => analytic_torus_face_area(topo, face_id),
        },
        FaceSurface::Nurbs(_) => {
            let mesh = tessellate::tessellate(topo, face_id, deflection)?;
            Ok(triangle_mesh_area(&mesh))
        }
    }
}

/// `∮ u w(v) dv` along a wire on a sphere, `w = R² cos v`, with u unwrapped
/// and free to jump at a pole: plus or minus the area of one of the two
/// regions the wire bounds. `None` when the wire winds the axis.
pub(super) fn sphere_wire_signed_area(
    topo: &Topology,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<Option<f64>, crate::OperationsError> {
    let (axis, radius) = (sphere.z_axis(), sphere.radius());
    let project = |p: Point3| sphere.project_point(p);
    // v = asin((P - C)·z / R), so along the surface dv = z·dP / (R cos v).
    let grad_v = |p: Point3| {
        let (_, v) = sphere.project_point(p);
        axis * (1.0 / (radius * v.cos()))
    };
    let weight = |v: f64| radius * radius * v.cos();
    let centre = sphere.center();
    let poles = [centre + axis * radius, centre - axis * radius];
    let metric = RevolutionMetric {
        project: &project,
        grad_v: &grad_v,
        weight: &weight,
        poles: &poles,
        v_periodic: false,
    };
    wire_uv_area(topo, wire_id, &metric)
}

/// Whether a sphere face is the region of area `patch` its outer loop bounds
/// rather than the sphere past it (`past`), by the face's own mesh. `None`
/// when the two are too close to tell apart (a patch near half the sphere).
pub(super) fn sphere_face_is_patch(
    topo: &Topology,
    face_id: FaceId,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    patch: f64,
    past: f64,
) -> Result<Option<bool>, crate::OperationsError> {
    let whole = 4.0 * std::f64::consts::PI * sphere.radius() * sphere.radius();
    if (patch - past).abs() < 0.05 * whole {
        return Ok(None);
    }
    let mesh = tessellate::tessellate(topo, face_id, 1e-2 * sphere.radius())?;
    let meshed = triangle_mesh_area(&mesh);
    Ok(Some((meshed - patch).abs() <= (meshed - past).abs()))
}

/// The area of a sphere face whose outer loop closes in u (through a pole if
/// it meets one), where the area element is `R² cos v du dv`: the region that
/// loop bounds or the sphere past it, less the `holes`. A face too near half
/// the sphere to tell which takes its mesh's area. `None` when the outer loop
/// winds the axis.
fn sphere_face_uv_area(
    topo: &Topology,
    face_id: FaceId,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    holes: f64,
    deflection: f64,
) -> Result<Option<f64>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let Some(outer) = sphere_wire_signed_area(topo, sphere, face.outer_wire())? else {
        return Ok(None);
    };
    let whole = 4.0 * std::f64::consts::PI * sphere.radius() * sphere.radius();
    let (patch, past) = (outer.abs() - holes, whole - outer.abs() - holes);
    Ok(Some(
        match sphere_face_is_patch(topo, face_id, sphere, patch, past)? {
            Some(true) => patch,
            Some(false) => past,
            None => triangle_mesh_area(&tessellate::tessellate(topo, face_id, deflection)?),
        },
    ))
}

/// The area a hole takes from a sphere face: the smaller region inside a loop
/// that winds none of the sphere's u (a drill's entry, a pocket through a
/// pole), and for a loop around the axis the smaller cap beyond it,
/// `R² (2π − |∮ sin v du|)` (a bore's rim), by midpoint sums at two
/// resolutions and a Richardson step.
fn sphere_hole_area(
    topo: &Topology,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<f64, crate::OperationsError> {
    use std::f64::consts::{PI, TAU};
    let r2 = sphere.radius() * sphere.radius();
    if let Some(signed) = sphere_wire_signed_area(topo, sphere, wire_id)? {
        return Ok(signed.abs().min(2.0 * TAU * r2 - signed.abs()));
    }
    let wrap = |d: f64| (d + PI).rem_euclid(TAU) - PI;
    let mut sweep = 0.0;
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
            let mut sweep = 0.0;
            #[allow(clippy::cast_precision_loss)]
            let step = (to - from) / n as f64;
            let mut u_prev = at(from).0;
            for k in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let tk = from + step * k as f64;
                let (_, vm) = at(tk + 0.5 * step);
                let (un, _) = at(tk + step);
                sweep += vm.sin() * wrap(un - u_prev);
                u_prev = un;
            }
            sweep
        };
        sweep += (4.0 * sums(256) - sums(128)) / 3.0;
    }
    Ok(r2 * (TAU - sweep.abs()))
}

/// The parameters strictly inside `(ta, tb)`, in traversal order, where a
/// curve passes through one of `poles` (within `1e-7`): each local minimum of
/// the distance over 64 probes that the curve could bring to the pole within
/// its neighbouring probes, refined by ternary search.
fn pole_crossings(at: &dyn Fn(f64) -> Point3, (ta, tb): (f64, f64), poles: &[Point3]) -> Vec<f64> {
    const PROBES: usize = 64;
    let mut found: Vec<f64> = Vec::new();
    #[allow(clippy::cast_precision_loss)]
    let ts: Vec<f64> = (0..=PROBES)
        .map(|k| ta + (tb - ta) * k as f64 / PROBES as f64)
        .collect();
    let points: Vec<Point3> = ts.iter().map(|&t| at(t)).collect();
    for &pole in poles {
        let gap = |t: f64| (at(t) - pole).length();
        let gaps: Vec<f64> = points.iter().map(|&p| (p - pole).length()).collect();
        for k in 0..=PROBES {
            let (before, after) = (k.saturating_sub(1), (k + 1).min(PROBES));
            if gaps[k] > gaps[before] || gaps[k] > gaps[after] {
                continue;
            }
            // Within the neighbouring probes the distance falls by at most
            // the arc between them, about the chords' length.
            let reach =
                (points[before] - points[k]).length() + (points[after] - points[k]).length();
            if gaps[k] > 1.5 * reach + 1e-7 {
                continue;
            }
            let (mut lo, mut hi) = (ts[before], ts[after]);
            for _ in 0..80 {
                let (m1, m2) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
                if gap(m1) < gap(m2) {
                    hi = m2;
                } else {
                    lo = m1;
                }
            }
            let t = 0.5 * (lo + hi);
            let p = at(t);
            if gap(t) <= 1e-7
                && (p - at(ta)).length() > 1e-7
                && (p - at(tb)).length() > 1e-7
                && !found.iter().any(|&f| (at(f) - p).length() <= 1e-7)
            {
                found.push(t);
            }
        }
    }
    if tb < ta {
        found.sort_by(|a, b| b.total_cmp(a));
    } else {
        found.sort_by(f64::total_cmp);
    }
    found
}

/// A lateral of revolution read in its `(u, v)` parameters, where the area
/// element is `weight(v) du dv`.
struct RevolutionMetric<'a> {
    project: &'a dyn Fn(Point3) -> (f64, f64),
    /// `∇v` along the surface at a point on it, so a boundary curve's
    /// `dv/dt` is `grad_v · P'(t)`.
    grad_v: &'a dyn Fn(Point3) -> Vec3,
    weight: &'a dyn Fn(f64) -> f64,
    /// Where the `u` lines collapse (a cone's apex, a sphere's poles): a loop
    /// through one may jump in `u` there, since the weight vanishes on it.
    poles: &'a [Point3],
    /// Whether `v` wraps too (a torus's tube angle), so a loop must also
    /// return to its start in `v`.
    v_periodic: bool,
}

/// The area of a cylinder face whose boundary is not a rectangle in
/// `(u, v)` (a wall trimmed by an oblique plane's ellipse, say): `r` times
/// the region's area in `(u, v)`. `Ok(None)` for a face bounded only by
/// rulings and rims, which the rectangle below measures exactly, or when a
/// wire's unwrapped `u` does not close.
fn cylinder_face_uv_area(
    topo: &Topology,
    face_id: FaceId,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    let axis = cyl.axis();
    let r = cyl.radius();
    let rectangular = face_edges_all(topo, face_id, |edge, a, b| {
        Ok(match edge.curve() {
            EdgeCurve::Line => (b - a).cross(axis).length() <= 1e-9 * (b - a).length().max(1.0),
            EdgeCurve::Circle(c) => c.normal().cross(axis).length() <= 1e-9,
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
        })
    })?;
    // A box in (u, v) measures in closed form; a rectilinear outline with a
    // notch (a window straddling the seam, carried by the outer wire) or a
    // hole is no box.
    let face = topo.face(face_id)?;
    if rectangular
        && face.inner_wires().is_empty()
        && topo.wire(face.outer_wire())?.edges().len() <= 4
    {
        return Ok(None);
    }
    let project = |p: Point3| cyl.project_point(p);
    let grad_v = |_: Point3| axis;
    let weight = |_: f64| r;
    face_uv_area(
        topo,
        face_id,
        &RevolutionMetric {
            project: &project,
            grad_v: &grad_v,
            weight: &weight,
            poles: &[],
            v_periodic: false,
        },
    )
}

/// The area of a cone face whose boundary is not a rectangle in `(u, v)`
/// (a wall trimmed by an oblique plane's ellipse), where the area element
/// is `v cos(a) du dv`. `Ok(None)` for a box of rulings and coaxial rims,
/// which [`analytic_cone_face_area`] measures exactly, or when a wire's
/// unwrapped `u` does not close.
fn cone_face_uv_area(
    topo: &Topology,
    face_id: FaceId,
    cone: &brepkit_math::surfaces::ConicalSurface,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    let axis = cone.axis();
    let apex = cone.apex();
    let rectangular = face_edges_all(topo, face_id, |edge, a, b| {
        Ok(match edge.curve() {
            EdgeCurve::Line => {
                let far = if (a - apex).length() > (b - apex).length() {
                    a
                } else {
                    b
                };
                (b - a).cross(far - apex).length()
                    <= 1e-9 * (b - a).length() * (far - apex).length()
            }
            EdgeCurve::Circle(c) => c.normal().cross(axis).length() <= 1e-9,
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
        })
    })?;
    // As for a cylinder: a notched or holed rectilinear outline is no box.
    let face = topo.face(face_id)?;
    if rectangular
        && face.inner_wires().is_empty()
        && topo.wire(face.outer_wire())?.edges().len() <= 4
    {
        return Ok(None);
    }
    let cos_a = cone.half_angle().cos();
    let project = |p: Point3| cone.project_point(p);
    // P = apex + v g(u) with a unit generator g normal to g'(u), so v
    // changes along the surface as g does.
    let grad_v = |p: Point3| {
        let (u, _) = cone.project_point(p);
        cone.evaluate(u, 1.0) - apex
    };
    let weight = |v: f64| v * cos_a;
    face_uv_area(
        topo,
        face_id,
        &RevolutionMetric {
            project: &project,
            grad_v: &grad_v,
            weight: &weight,
            poles: &[apex],
            v_periodic: false,
        },
    )
}

/// The area of a torus face trimmed by a free-form curve (a plane's loop
/// around the tube), where the area element is `r (R + r cos v) du dv`.
/// `Ok(None)` for a face bounded only by circles and seam placeholders,
/// which [`analytic_torus_face_area`] measures, or when a wire's unwrapped
/// `u` does not close.
fn torus_face_uv_area(
    topo: &Topology,
    face_id: FaceId,
    torus: &brepkit_math::surfaces::ToroidalSurface,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    let circular = face_edges_all(topo, face_id, |edge, _, _| {
        Ok(matches!(
            edge.curve(),
            EdgeCurve::Line | EdgeCurve::Circle(_)
        ))
    })?;
    if circular {
        return Ok(None);
    }
    let (big, small) = (torus.major_radius(), torus.minor_radius());
    let (ex, ey, ez) = (torus.x_axis(), torus.y_axis(), torus.z_axis());
    let project = |p: Point3| torus.project_point(p);
    // P_v = r (−sin v ρ̂(u) + cos v ẑ), so ∇v is P_v / r².
    let grad_v = |p: Point3| {
        let (u, v) = torus.project_point(p);
        let ((sin_u, cos_u), (sin_v, cos_v)) = (u.sin_cos(), v.sin_cos());
        ((ex * cos_u + ey * sin_u) * -sin_v + ez * cos_v) * (1.0 / small)
    };
    let weight = |v: f64| small * small.mul_add(v.cos(), big);
    let metric = RevolutionMetric {
        project: &project,
        grad_v: &grad_v,
        weight: &weight,
        poles: &[],
        v_periodic: true,
    };
    // A whole ring's outer wire is its seam placeholders collapsed onto one
    // vertex, which encloses nothing in (u, v): the ring less its holes.
    let face = topo.face(face_id)?;
    let mut collapsed = true;
    for oe in topo.wire(face.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        collapsed &= matches!(edge.curve(), EdgeCurve::Line)
            && (topo.vertex(edge.start())?.point() - topo.vertex(edge.end())?.point()).length()
                < 1e-9;
    }
    if collapsed {
        let mut area = 4.0 * std::f64::consts::PI * std::f64::consts::PI * big * small;
        for &wid in face.inner_wires() {
            let Some(hole) = wire_uv_area(topo, wid, &metric)? else {
                return Ok(None);
            };
            area -= hole.abs();
        }
        return Ok(Some(area));
    }
    face_uv_area(topo, face_id, &metric)
}

/// Whether `test` holds for every edge of a face, given the edge and its
/// start and end points.
fn face_edges_all(
    topo: &Topology,
    face_id: FaceId,
    mut test: impl FnMut(
        &brepkit_topology::edge::Edge,
        Point3,
        Point3,
    ) -> Result<bool, crate::OperationsError>,
) -> Result<bool, crate::OperationsError> {
    let face = topo.face(face_id)?;
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let a = topo.vertex(edge.start())?.point();
            let b = topo.vertex(edge.end())?.point();
            if !test(edge, a, b)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// A face's area from its `(u, v)` region by Green's theorem,
/// `A = ∮ u w(v) dv` along each wire, less its holes.
fn face_uv_area(
    topo: &Topology,
    face_id: FaceId,
    metric: &RevolutionMetric<'_>,
) -> Result<Option<f64>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wires: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    let mut total = 0.0;
    for (k, &wid) in wires.iter().enumerate() {
        let Some(signed) = wire_uv_area(topo, wid, metric)? else {
            return Ok(None);
        };
        if k == 0 {
            total += signed.abs();
        } else {
            total -= signed.abs();
        }
    }
    Ok(Some(total))
}

/// `∮ u w(v) dv` along one wire, with `u` unwrapped continuously, by
/// Gauss-Legendre quadrature in each edge's own parameter; `None` when the
/// unwrapped `u` does not return to its start away from a pole.
fn wire_uv_area(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    metric: &RevolutionMetric<'_>,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::{PI, TAU};
    const SEGMENTS: usize = 16;
    const ORDER: usize = 8;
    let at_pole = |p: Point3| metric.poles.iter().any(|&q| (p - q).length() <= 1e-7);
    let mut touches_pole = false;
    let mut u_near = |p: Point3, near: Option<f64>| {
        if at_pole(p) {
            touches_pole = true;
            if let Some(n) = near {
                return n;
            }
        }
        let (u, _) = (metric.project)(p);
        near.map_or(u, |n| u - ((u - n + PI) / TAU).floor() * TAU)
    };
    let points = brepkit_math::quadrature::gauss_legendre_points(ORDER);
    let mut sum = 0.0;
    let mut first_u = None;
    let mut last_u: Option<f64> = None;
    // The loop's v, unwrapped along it, for a surface whose v wraps.
    let mut first_v: Option<f64> = None;
    let mut v_walk: Option<f64> = None;
    let mut walk_v = |p: Point3| {
        let (_, v) = (metric.project)(p);
        let next = v_walk.map_or(v, |w| v - ((v - w + PI) / TAU).floor() * TAU);
        first_v.get_or_insert(next);
        v_walk = Some(next);
    };
    // The region's u has to jump by a turn somewhere on the loop, which is
    // free only across the pole (the weight vanishes there): walk from it.
    // An arc over a pole between its vertices is cut there, so its jump
    // falls on a span end too.
    let wire_edges = topo.wire(wire_id)?.edges().to_vec();
    let mut spans: Vec<(usize, (f64, f64), bool)> = Vec::new();
    for (k, oe) in wire_edges.iter().enumerate() {
        let edge = topo.edge(oe.edge())?;
        let start = topo.vertex(edge.start())?.point();
        let end = topo.vertex(edge.end())?.point();
        let at = |t: f64| edge.curve().evaluate_with_endpoints(t, start, end);
        for (ta, tb) in traversal_spans(edge, oe.is_forward(), start, end) {
            let mut from = ta;
            for t in pole_crossings(&at, (ta, tb), metric.poles) {
                spans.push((k, (from, t), at_pole(at(from))));
                from = t;
            }
            spans.push((k, (from, tb), at_pole(at(from))));
        }
    }
    if let Some(k) = spans.iter().position(|&(_, _, leaves)| leaves) {
        spans.rotate_left(k);
    }
    for &(k, (ta, tb), _) in &spans {
        let edge = topo.edge(wire_edges[k].edge())?;
        let start = topo.vertex(edge.start())?.point();
        let end = topo.vertex(edge.end())?.point();
        let curve = edge.curve();
        let at = |t: f64| curve.evaluate_with_endpoints(t, start, end);
        let mut u_prev = u_near(at(ta), last_u);
        if first_u.is_none() {
            first_u = Some(u_prev);
        }
        walk_v(at(ta));
        // A NURBS edge integrates knot span by knot span, where it is
        // smooth; other curves in even segments.
        #[allow(clippy::cast_precision_loss)]
        let mut cuts: Vec<f64> = (0..=SEGMENTS)
            .map(|seg| ta + (tb - ta) * seg as f64 / SEGMENTS as f64)
            .collect();
        if let EdgeCurve::NurbsCurve(nc) = curve {
            let (lo, hi) = (ta.min(tb), ta.max(tb));
            cuts = std::iter::once(ta)
                .chain(nc.knots().iter().copied().filter(|&k| k > lo && k < hi))
                .chain(std::iter::once(tb))
                .collect();
            cuts.dedup();
            if tb < ta {
                let last = cuts.len() - 1;
                cuts[1..last].reverse();
            }
        }
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
            for gp in points {
                let t = mid + half * gp.x;
                let p = at(t);
                let u = u_near(p, Some(u_prev));
                let tangent = match curve {
                    EdgeCurve::Line => end - start,
                    EdgeCurve::Circle(c) => c.tangent(t) * c.radius(),
                    EdgeCurve::Ellipse(e) => e.tangent(t),
                    EdgeCurve::NurbsCurve(nc) => nc.derivatives(t, 1)[1],
                };
                let (_, v) = (metric.project)(p);
                let dv = (metric.grad_v)(p).dot(tangent);
                sum += gp.w * half * u * (metric.weight)(v) * dv;
                u_prev = u;
                walk_v(p);
            }
        }
        last_u = Some(u_near(at(tb), Some(u_prev)));
        walk_v(at(tb));
    }
    let v_closes = !metric.v_periodic
        || matches!((first_v, v_walk), (Some(a), Some(b)) if (a - b).abs() <= 1e-6);
    Ok(match (first_u, last_u) {
        (Some(a), Some(b)) if v_closes && (touches_pole || (a - b).abs() <= 1e-6) => Some(sum),
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
    // A sector seamed along a latitude says by its seam which way round the
    // ring it runs, and covers the whole tube.
    if let Some((u0, u1)) = torus_sector_u_range(topo, face, tor)? {
        let tube = std::f64::consts::TAU * tor.major_radius();
        return Ok(tor.minor_radius() * (u1 - u0) * tube);
    }
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

    // A band seamed along a meridian says by its seam which side it covers.
    if let Some((v0, v1)) = torus_band_v_range(topo, face, tor)? {
        v_min = v0;
        v_max = v1;
    }
    let big_r = tor.major_radius();
    let small_r = tor.minor_radius();
    let area =
        small_r * (u1 - u0) * (big_r * (v_max - v_min) + small_r * (v_max.sin() - v_min.sin()));
    Ok(area.abs())
}

/// The area of a planar face, less its holes: exact by Green's theorem when
/// every edge is a line, a circle or an ellipse, else by Newell's method over
/// the sampled boundary.
fn planar_face_area(topo: &Topology, face_id: FaceId) -> Result<f64, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    let face = topo.face(face_id)?;
    let mut lines_and_conics = true;
    for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire)?.edges() {
            lines_and_conics &= matches!(
                topo.edge(oe.edge())?.curve(),
                EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)
            );
        }
    }
    if lines_and_conics
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

/// Newell's normal of a closed polygon: twice its vector area.
fn newell_normal(pts: &[Point3]) -> Vec3 {
    let n = pts.len();
    let mut normal = Vec3::new(0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        normal += Vec3::new(
            (a.y() - b.y()) * (a.z() + b.z()),
            (a.z() - b.z()) * (a.x() + b.x()),
            (a.x() - b.x()) * (a.y() + b.y()),
        );
    }
    normal
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
