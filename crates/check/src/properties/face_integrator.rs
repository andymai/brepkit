//! Per-face Gauss quadrature integration for area, volume, CoM, and inertia.
//!
//! Provides numerical integration of geometric properties over individual
//! faces. Planar faces use polygon fan triangulation; parametric faces
//! (cylinder, cone, sphere, torus, NURBS) use tensor-product Gauss-Legendre
//! quadrature over the UV domain.

use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::CheckError;

/// Contribution of a single face to global geometric properties.
#[derive(Debug, Clone)]
pub struct FaceContribution {
    /// Face area.
    pub area: f64,
    /// Volume contribution: (1/3) integral of P dot N dA.
    pub volume: f64,
    /// Volume-weighted x-moment: (1/2) integral of x^2 * n_x dA (divergence theorem).
    pub volume_moment_x: f64,
    /// Volume-weighted y-moment: (1/2) integral of y^2 * n_y dA (divergence theorem).
    pub volume_moment_y: f64,
    /// Volume-weighted z-moment: (1/2) integral of z^2 * n_z dA (divergence theorem).
    pub volume_moment_z: f64,
    /// Area-weighted centroid x-component (for surface centroid, not solid CoM).
    pub centroid_x: f64,
    /// Area-weighted centroid y-component (for surface centroid, not solid CoM).
    pub centroid_y: f64,
    /// Area-weighted centroid z-component (for surface centroid, not solid CoM).
    pub centroid_z: f64,
}

/// Integrate a face's geometric contribution using Gauss quadrature.
///
/// For planar faces, evaluates via polygon fan triangulation. For
/// parametric surfaces (analytic and NURBS), evaluates the surface and its
/// partial derivatives on a Gauss-point grid over the UV domain derived
/// from the face's boundary vertices.
///
/// # Errors
///
/// Returns an error if topology entities are missing or the face has
/// insufficient geometry for integration.
pub fn integrate_face(
    topo: &Topology,
    face_id: FaceId,
    gauss_order: usize,
) -> Result<FaceContribution, CheckError> {
    integrate_face_about(topo, face_id, gauss_order, Vec3::new(0.0, 0.0, 0.0))
}

/// [`integrate_face`] with its volume term `(1/3)∫(P − about)·N dA` taken
/// about a point other than the origin. A solid's faces summed about the
/// same point give its volume wherever that point is, but a face whose
/// coordinates are millions of units out multiplies them into its flux, and
/// the trimmed quadrature's small closure errors grow with them.
#[allow(clippy::too_many_lines)]
pub(crate) fn integrate_face_about(
    topo: &Topology,
    face_id: FaceId,
    gauss_order: usize,
    about: Vec3,
) -> Result<FaceContribution, CheckError> {
    let face = topo.face(face_id)?;
    let reversed = face.is_reversed();
    let sign = if reversed { -1.0 } else { 1.0 };

    match face.surface() {
        FaceSurface::Plane { normal, .. } => {
            let effective_normal = if reversed { -*normal } else { *normal };
            integrate_planar_face(topo, face_id, effective_normal, about)
        }
        FaceSurface::Cylinder(s) => {
            let full = (
                (0.0, std::f64::consts::TAU),
                (f64::NEG_INFINITY, f64::INFINITY),
            );
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, false, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, s, true, false)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
                false,
                &[],
                about,
            ))
        }
        FaceSurface::Cone(s) => {
            let full = (
                (0.0, std::f64::consts::TAU),
                (f64::NEG_INFINITY, f64::INFINITY),
            );
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, false, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, s, true, false)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
                false,
                &[],
                about,
            ))
        }
        FaceSurface::Sphere(s) => {
            let full = (
                (0.0, std::f64::consts::TAU),
                (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
            );
            let bounds = face_uv_bounds(topo, face_id, s, true, false, full)?;
            // A turned sphere's equator projects to a rounding sliver of v
            // rather than one value, and bounds no more than it does upright.
            let (u_range, v_range) =
                if (bounds.1.1 - bounds.1.0) * s.radius() <= Tolerance::new().linear {
                    full
                } else {
                    bounds
                };
            let uv_boundary = build_face_uv_boundary(topo, face_id, s, true, false)?;
            let hole_vs = full_revolution_hole_vs(topo, face_id, s);
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
                false,
                &hole_vs,
                about,
            ))
        }
        FaceSurface::Torus(s) => {
            let full = ((0.0, std::f64::consts::TAU), (0.0, std::f64::consts::TAU));
            let (u_range, v_range) = face_uv_bounds(topo, face_id, s, true, true, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, s, true, true)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                true,
                true,
                &[],
                about,
            ))
        }
        FaceSurface::Nurbs(s) => {
            let full = (s.domain_u(), s.domain_v());
            let periodic_u = s.is_periodic_u();
            let periodic_v = s.is_periodic_v();
            let (u_range, v_range) =
                face_uv_bounds(topo, face_id, s, periodic_u, periodic_v, full)?;
            let uv_boundary = build_face_uv_boundary(topo, face_id, s, periodic_u, periodic_v)?;
            Ok(integrate_with_trimming(
                s,
                u_range,
                v_range,
                gauss_order,
                sign,
                &uv_boundary,
                periodic_u,
                periodic_v,
                &[],
                about,
            ))
        }
    }
}

/// UV domain bounds as `((u_min, u_max), (v_min, v_max))`.
type UvBounds = ((f64, f64), (f64, f64));

/// The v-positions of a face's full-revolution inner wires (holes) on a
/// surface periodic in u.
///
/// A boolean that drills a cylinder through a sphere leaves each spherical
/// band bounded by a latitude circle hole (the tunnel rim). Such a hole wraps
/// the full u-period and sits at a single v, so the band runs from its outer
/// latitude to the hole — not on to the pole. Collecting these lets the
/// integrator clip the band instead of over-integrating the polar cap the hole
/// removed. Each entry is the mean projected v of one full-revolution hole.
fn full_revolution_hole_vs<S: ParametricSurface>(
    topo: &Topology,
    face_id: FaceId,
    surface: &S,
) -> Vec<f64> {
    use std::f64::consts::TAU;
    let Ok(face) = topo.face(face_id) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &wid in face.inner_wires() {
        let Ok(wire) = topo.wire(wid) else { continue };
        let mut us = Vec::new();
        let mut vs = Vec::new();
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            // Oriented traversal: the wire-ordered start vertex is the edge's
            // end when the oriented edge is reversed.
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            let Ok(v) = topo.vertex(vid) else {
                continue;
            };
            let (u, vv) = surface.project_point(v.point());
            us.push(u);
            vs.push(vv);
        }
        if vs.is_empty() {
            continue;
        }
        let v_min = vs.iter().copied().fold(f64::INFINITY, f64::min);
        let v_max = vs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Constant-v latitude circle.
        if v_max - v_min > 1e-6 {
            continue;
        }
        // Full revolution in u: the unwrapped per-vertex deltas around the
        // CLOSED loop (including the closing step back to the first vertex) sum
        // to ≈ TAU. A single-edge closed circle has one vertex, so also accept
        // holes whose sole edge is a closed circle curve.
        let unwrapped_span = {
            let n = us.len();
            let mut acc = 0.0;
            for i in 0..n {
                let d = us[(i + 1) % n] - us[i];
                acc += d - TAU * ((d + std::f64::consts::PI) / TAU).floor();
            }
            acc.abs()
        };
        let single_closed_circle = wire.edges().len() == 1
            && wire.edges().first().is_some_and(|oe| {
                topo.edge(oe.edge())
                    .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Circle(_)))
            });
        if unwrapped_span >= TAU - 1e-3 || single_closed_circle {
            out.push(0.5 * (v_min + v_max));
        }
    }
    out
}

/// Compute UV bounds for a parametric face by projecting its outer wire's
/// vertices, and points along each curved edge, onto the surface and taking
/// the min/max of the resulting parameters.
///
/// For surfaces with periodic u or v coordinates (cylinders, cones, spheres,
/// tori), sequentially unwraps the angular coordinates so that faces straddling
/// the 0/2pi seam produce correct ranges.
///
/// When all projected vertices coincide (e.g. a full-revolution face),
/// `full_domain` is returned instead.
///
/// **Limitation:** Only the outer wire is used for UV bounds. Inner wires
/// (holes) are handled during Gauss integration by the UV containment check
/// in `integrate_parametric_trimmed`, but the current containment only tests
/// against the outer boundary. Faces with holes will over-integrate the hole
/// region. A proper fix requires multi-polygon UV containment (outer minus
/// holes).
fn face_uv_bounds<S: ParametricSurface>(
    topo: &Topology,
    face_id: FaceId,
    surface: &S,
    periodic_u: bool,
    periodic_v: bool,
    full_domain: UvBounds,
) -> Result<UvBounds, CheckError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;

    // Each edge's start and, for a curved edge, points along its own span in
    // traversal order: an arc past half a turn between two vertices would
    // otherwise unwrap the short way below.
    let mut uvs = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        let pt = topo.vertex(vid)?.point();
        uvs.push(surface.project_point(pt));
        if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) && !edge.is_closed() {
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            // A NURBS edge's span is its whole knot range, which may run from
            // the edge's end back to its start.
            let first = edge.curve().evaluate_with_endpoints(t0, sp, ep);
            let from_start = (first - sp).length() <= (first - ep).length();
            for k in 1..4 {
                let f = f64::from(k) / 4.0;
                let f = if oe.is_forward() == from_start {
                    f
                } else {
                    1.0 - f
                };
                let p = edge
                    .curve()
                    .evaluate_with_endpoints((t1 - t0).mul_add(f, t0), sp, ep);
                uvs.push(surface.project_point(p));
            }
        }
    }

    if uvs.is_empty() {
        return Err(CheckError::IntegrationFailed(
            "face wire has no edges".into(),
        ));
    }

    // Unwrap periodic coordinates sequentially so seam-straddling faces
    // produce a contiguous range instead of the full [0, 2pi). A point where
    // the surface has no `u` of its own (a sphere's pole, a cone's apex)
    // projects to an arbitrary `u`: it stays out of the range, the walk
    // starts just after the first such point, and the point after any later
    // one is unwrapped toward the middle of the range so far, which picks its
    // representative nearest that range.
    if periodic_u {
        let mut singular: Vec<bool> = uvs.iter().map(|&uv| singular_in_u(surface, uv)).collect();
        if let Some(first) = singular.iter().position(|&s| s) {
            uvs.rotate_left(first + 1);
            singular.rotate_left(first + 1);
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut prev: Option<f64> = None;
        let mut after_singular = false;
        for (uv, &at_singular) in uvs.iter_mut().zip(&singular) {
            if at_singular {
                after_singular = true;
                continue;
            }
            let u = match prev {
                None => uv.0,
                Some(p) if !after_singular => unwrap_angle(p, uv.0),
                Some(_) => unwrap_angle(f64::midpoint(lo, hi), uv.0),
            };
            uv.0 = u;
            lo = lo.min(u);
            hi = hi.max(u);
            prev = Some(u);
            after_singular = false;
        }
        if let Some(p) = prev {
            for (uv, _) in uvs.iter_mut().zip(&singular).filter(|(_, s)| **s) {
                uv.0 = p;
            }
        }
    }
    if periodic_v {
        for i in 1..uvs.len() {
            uvs[i].1 = unwrap_angle(uvs[i - 1].1, uvs[i].1);
        }
    }

    // Check for coincident vertices (all project to same point) — use full domain.
    let coincident = uvs.len() < 3 || {
        let ref_uv = uvs[0];
        uvs.iter()
            .all(|uv| (uv.0 - ref_uv.0).abs() < 1e-6 && (uv.1 - ref_uv.1).abs() < 1e-6)
    };
    if coincident {
        return Ok(full_domain);
    }

    let u_min = uvs.iter().map(|uv| uv.0).fold(f64::INFINITY, f64::min);
    let mut u_max = uvs.iter().map(|uv| uv.0).fold(f64::NEG_INFINITY, f64::max);
    let v_min = uvs.iter().map(|uv| uv.1).fold(f64::INFINITY, f64::min);
    let mut v_max = uvs.iter().map(|uv| uv.1).fold(f64::NEG_INFINITY, f64::max);

    // All boundary vertices on the seam of a periodic axis (e.g. a
    // full-revolution lateral face whose circles start/end at the seam)
    // collapse that axis's range to zero — the face actually spans the
    // full period.
    if periodic_u && u_max - u_min < 1e-9 {
        u_max = u_min + (full_domain.0.1 - full_domain.0.0);
    }
    if periodic_v && v_max - v_min < 1e-9 {
        v_max = v_min + (full_domain.1.1 - full_domain.1.0);
    }

    if u_min >= u_max || v_min >= v_max {
        // A degenerate projection (e.g. all boundary vertices on a sphere's
        // pole seam) does not mean an empty face — it means the boundary failed
        // to bound a sub-region, so the face spans the full analytic domain.
        return Ok(full_domain);
    }

    Ok(((u_min, u_max), (v_min, v_max)))
}

/// Unwrap a step in a periodic (angular) coordinate to avoid discontinuities.
///
/// Adjusts `next` so that `next - prev` lies in `(-pi, pi]`, keeping the
/// sequence monotonic through the 0/2pi seam.
fn unwrap_angle(prev: f64, next: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let diff = next - prev;
    prev + diff - tau * ((diff + std::f64::consts::PI) / tau).floor()
}

/// Integrate a planar face using polygon fan triangulation.
///
/// Inner wires (holes) are integrated the same way and subtracted from the
/// outer-wire contribution.
fn integrate_planar_face(
    topo: &Topology,
    face_id: FaceId,
    normal: Vec3,
    about: Vec3,
) -> Result<FaceContribution, CheckError> {
    if let Some(contrib) = planar_face_by_edges(topo, face_id, normal, about)? {
        return Ok(contrib);
    }
    let polygon = crate::util::face_polygon(topo, face_id)?;
    let mut contrib = integrate_planar_polygon(&polygon, normal, about);

    let face = topo.face(face_id)?;
    let inner: Vec<_> = face.inner_wires().to_vec();
    for wid in inner {
        let hole = crate::util::wire_polygon(topo, wid)?;
        let h = integrate_planar_polygon(&hole, normal, about);
        contrib.area -= h.area;
        contrib.volume -= h.volume;
        contrib.volume_moment_x -= h.volume_moment_x;
        contrib.volume_moment_y -= h.volume_moment_y;
        contrib.volume_moment_z -= h.volume_moment_z;
        contrib.centroid_x -= h.centroid_x;
        contrib.centroid_y -= h.centroid_y;
        contrib.centroid_z -= h.centroid_z;
    }

    Ok(contrib)
}

/// A plane face's contribution integrated along its edges' own curves by
/// Green's theorem: each wire gives the area and the first and second
/// moments of the region it bounds in the face's plane, and the face is the
/// outer wire's region less its holes'. A curved edge is read exactly rather
/// than through chords. `None` for a face with a wire that does not chain
/// into a loop.
fn planar_face_by_edges(
    topo: &Topology,
    face_id: FaceId,
    normal: Vec3,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    let face = topo.face(face_id)?;
    let outer = topo.wire(face.outer_wire())?;
    let Some(first) = outer.edges().first() else {
        return Ok(None);
    };
    let first_edge = topo.edge(first.edge())?;
    let origin = topo.vertex(first.oriented_start(first_edge))?.point();
    let Ok(frame) = brepkit_math::frame::Frame3::from_normal(origin, normal) else {
        return Ok(None);
    };
    let mut m = [0.0; 6];
    for (k, wid) in std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .enumerate()
    {
        let Some(w) = wire_plane_moments(topo, wid, &frame)? else {
            return Ok(None);
        };
        // Each wire counts its own region, whichever way it runs; holes
        // are taken away.
        let sign = w[0].signum() * if k == 0 { 1.0 } else { -1.0 };
        for (total, part) in m.iter_mut().zip(w) {
            *total = sign.mul_add(part, *total);
        }
    }
    let [area, ix, iy, ixx, ixy, iyy] = m;
    let (o, e1, e2) = (frame.origin, frame.x, frame.y);
    // The integral over the region of the square of one coordinate
    // `oc + c1 x + c2 y`.
    let square = |oc: f64, c1: f64, c2: f64| {
        (oc * oc).mul_add(
            area,
            (2.0 * oc).mul_add(
                c1.mul_add(ix, c2 * iy),
                (c1 * c1).mul_add(ixx, (2.0 * c1 * c2).mul_add(ixy, c2 * c2 * iyy)),
            ),
        )
    };
    let reach = Vec3::new(o.x(), o.y(), o.z()) - about;
    Ok(Some(FaceContribution {
        area,
        volume: reach.dot(normal) * area / 3.0,
        volume_moment_x: 0.5 * normal.x() * square(o.x(), e1.x(), e2.x()),
        volume_moment_y: 0.5 * normal.y() * square(o.y(), e1.y(), e2.y()),
        volume_moment_z: 0.5 * normal.z() * square(o.z(), e1.z(), e2.z()),
        centroid_x: o.x().mul_add(area, e1.x().mul_add(ix, e2.x() * iy)),
        centroid_y: o.y().mul_add(area, e1.y().mul_add(ix, e2.y() * iy)),
        centroid_z: o.z().mul_add(area, e1.z().mul_add(ix, e2.z() * iy)),
    }))
}

/// The integrals over the region a wire bounds in `frame`'s plane of `1`,
/// `x`, `y`, `x²`, `xy` and `y²`, signed by the way the wire runs, each a
/// line integral along the wire (`½∮(x dy - y dx)`, `½∮x² dy`, `-½∮y² dx`,
/// `⅓∮x³ dy`, `½∮x²y dy`, `-⅓∮y³ dx`) taken on the edges' own curves by
/// Gauss-Legendre quadrature. Edges are chained by their vertices, as
/// `wire_polygon` chains them. `None` for a wire that does not close.
fn wire_plane_moments(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    frame: &brepkit_math::frame::Frame3,
) -> Result<Option<[f64; 6]>, CheckError> {
    use std::f64::consts::FRAC_PI_8;
    let wire = topo.wire(wire_id)?;
    let gauss = gauss_legendre_points(8);
    let mut m = [0.0; 6];
    let mut prev: Option<(brepkit_topology::vertex::VertexId, Point3)> = None;
    let mut first: Option<Point3> = None;
    let mut scale = 0.0_f64;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let (s, e) = (edge.start(), edge.end());
        let (sp, ep) = (topo.vertex(s)?.point(), topo.vertex(e)?.point());
        scale = scale.max((sp - frame.origin).length());
        let forward = match prev {
            Some((pe, _)) if s == pe && e != pe => true,
            Some((pe, _)) if e == pe && s != pe => false,
            _ if s == e => oe.is_forward(),
            Some((_, last)) => (sp - last).length() <= (ep - last).length(),
            None => oe.is_forward(),
        };
        let (from, to) = if forward { (sp, ep) } else { (ep, sp) };
        first.get_or_insert(from);
        prev = Some((if forward { e } else { s }, to));
        let curve = edge.curve();
        let (t0, t1) = curve.domain_with_endpoints(sp, ep);
        // The curve's own direction runs from `sp`, or (a NURBS edge's span)
        // from `ep` back; a closed edge runs its way from its vertex.
        let natural = s == e || {
            let at = curve.evaluate_with_endpoints(t0, sp, ep);
            (at - sp).length() <= (at - ep).length()
        };
        // Breaks between which the integrand is smooth: a NURBS curve's
        // knots within the span, and pieces no wider than an eighth of a turn
        // on a conic.
        let mut breaks = vec![t0];
        match curve {
            EdgeCurve::Line => {}
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                let n = ((t1 - t0).abs() / FRAC_PI_8).ceil().max(1.0) as u32;
                breaks.extend((1..n).map(|k| (t1 - t0).mul_add(f64::from(k) / f64::from(n), t0)));
            }
            EdgeCurve::NurbsCurve(c) => {
                let (lo, hi) = (t0.min(t1), t0.max(t1));
                breaks.extend(c.knots().iter().copied().filter(|&k| k > lo && k < hi));
                breaks.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (hi - lo).abs());
            }
        }
        breaks.push(t1);
        if forward != natural {
            breaks.reverse();
        }
        for piece in breaks.windows(2) {
            let (step, mid) = (piece[1] - piece[0], f64::midpoint(piece[0], piece[1]));
            for gp in gauss {
                let t = (0.5 * step).mul_add(gp.x, mid);
                let w = 0.5 * step * gp.w;
                let p = curve.evaluate_with_endpoints(t, sp, ep) - frame.origin;
                let d = match curve {
                    EdgeCurve::Line => ep - sp,
                    EdgeCurve::Circle(c) => c.tangent(t) * c.radius(),
                    EdgeCurve::Ellipse(c) => c.tangent(t),
                    EdgeCurve::NurbsCurve(c) => c.derivatives(t, 1)[1],
                };
                let (x, y) = (p.dot(frame.x), p.dot(frame.y));
                let (dx, dy) = (d.dot(frame.x), d.dot(frame.y));
                m[0] += w * 0.5 * x.mul_add(dy, -(y * dx));
                m[1] += w * 0.5 * x * x * dy;
                m[2] -= w * 0.5 * y * y * dx;
                m[3] += w * x * x * x * dy / 3.0;
                m[4] += w * 0.5 * x * x * y * dy;
                m[5] -= w * y * y * y * dx / 3.0;
            }
        }
    }
    let closes = match (first, prev) {
        (Some(a), Some((_, b))) => (a - b).length() <= 1e-6 * scale.max(1.0),
        _ => false,
    };
    Ok(closes.then_some(m))
}

/// Integrate a planar polygon's contribution via fan triangulation.
fn integrate_planar_polygon(polygon: &[Point3], normal: Vec3, about: Vec3) -> FaceContribution {
    if polygon.len() < 3 {
        return FaceContribution {
            area: 0.0,
            volume: 0.0,
            volume_moment_x: 0.0,
            volume_moment_y: 0.0,
            volume_moment_z: 0.0,
            centroid_x: 0.0,
            centroid_y: 0.0,
            centroid_z: 0.0,
        };
    }

    // Fan triangulation from vertex 0 with SIGNED triangle areas (projected
    // onto the face normal): a fan over a NON-CONVEX polygon (a notched
    // boolean cap) sweeps triangles across the notch, and an unsigned fan
    // counts those positively — the notch region then adds instead of
    // cancelling, over-counting by an amount that depends on where vertex 0
    // happens to sit. Signed accumulation makes the fan exact for any
    // simple planar polygon; a globally CW polygon nets negative and is
    // flipped wholesale below.
    let mut area = 0.0;
    let mut vol = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for i in 1..polygon.len() - 1 {
        let (a, b, c) = (polygon[0], polygon[i], polygon[i + 1]);
        let ab = b - a;
        let ac = c - a;
        let cross = Vec3::new(
            ab.y() * ac.z() - ab.z() * ac.y(),
            ab.z() * ac.x() - ab.x() * ac.z(),
            ab.x() * ac.y() - ab.y() * ac.x(),
        );
        let tri_area = cross.dot(normal) * 0.5;
        area += tri_area;

        // Volume contribution: (1/3) * centroid dot normal * area
        let centroid = Point3::new(
            (a.x() + b.x() + c.x()) / 3.0,
            (a.y() + b.y() + c.y()) / 3.0,
            (a.z() + b.z() + c.z()) / 3.0,
        );
        let pv = Vec3::new(centroid.x(), centroid.y(), centroid.z()) - about;
        vol += pv.dot(normal) * tri_area / 3.0;

        // Volume moments via divergence theorem: (1/2) integral of x^2 * n_x dA
        // For a planar triangle with constant normal, integral of x^2 over triangle
        // = (area/3) * (x_a^2 + x_b^2 + x_c^2 + x_a*x_b + x_a*x_c + x_b*x_c) / 2
        // Simplified: use (x_a^2 + x_b^2 + x_c^2 + x_a*x_b + x_a*x_c + x_b*x_c)/6
        let avg_x2 = (a.x() * a.x()
            + b.x() * b.x()
            + c.x() * c.x()
            + a.x() * b.x()
            + a.x() * c.x()
            + b.x() * c.x())
            / 6.0;
        let avg_y2 = (a.y() * a.y()
            + b.y() * b.y()
            + c.y() * c.y()
            + a.y() * b.y()
            + a.y() * c.y()
            + b.y() * c.y())
            / 6.0;
        let avg_z2 = (a.z() * a.z()
            + b.z() * b.z()
            + c.z() * c.z()
            + a.z() * b.z()
            + a.z() * c.z()
            + b.z() * c.z())
            / 6.0;
        mx += 0.5 * avg_x2 * normal.x() * tri_area;
        my += 0.5 * avg_y2 * normal.y() * tri_area;
        mz += 0.5 * avg_z2 * normal.z() * tri_area;

        cx += centroid.x() * tri_area;
        cy += centroid.y() * tri_area;
        cz += centroid.z() * tri_area;
    }

    // A polygon wound CW about `normal` nets a negative signed area; flip
    // every accumulated quantity so callers keep the historical positive-
    // area contract (hole handling in `integrate_planar_face` subtracts).
    let flip = if area < 0.0 { -1.0 } else { 1.0 };
    FaceContribution {
        area: area * flip,
        volume: vol * flip,
        volume_moment_x: mx * flip,
        volume_moment_y: my * flip,
        volume_moment_z: mz * flip,
        centroid_x: cx * flip,
        centroid_y: cy * flip,
        centroid_z: cz * flip,
    }
}

/// Integrate a parametric surface using Gauss quadrature over the UV domain.
#[allow(clippy::cast_precision_loss)]
fn integrate_parametric<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
    about: Vec3,
) -> FaceContribution {
    // Composite quadrature: tile the domain into patches no larger than ~PI/4
    // so one Gauss rule resolves curved and periodic integrands. A single patch
    // over a torus's full 2*PI period in both u and v under-resolves it (~0.5%
    // error); several patches per period converge to machine precision. The
    // patch count is capped so a long *linear* axis (e.g. a tall cylinder/cone
    // whose v is axial distance) cannot make integration cost scale with model
    // size — its integrand is low-degree, so a bounded number of patches stays
    // exact. Angular axes never exceed 2*PI (= 8 patches), well under the cap.
    const MAX_PATCHES: usize = 16;

    let gauss_pts = gauss_legendre_points(gauss_order);
    let patch = std::f64::consts::FRAC_PI_4;
    let nu = (((u_range.1 - u_range.0).abs() / patch).ceil() as usize).clamp(1, MAX_PATCHES);
    let nv = (((v_range.1 - v_range.0).abs() / patch).ceil() as usize).clamp(1, MAX_PATCHES);
    let du_patch = (u_range.1 - u_range.0) / nu as f64;
    let dv_patch = (v_range.1 - v_range.0) / nv as f64;
    let u_scale = du_patch / 2.0;
    let v_scale = dv_patch / 2.0;

    let mut area = 0.0;
    let mut vol = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for iu in 0..nu {
        let u_mid = du_patch.mul_add(iu as f64, u_range.0) + u_scale;
        for iv in 0..nv {
            let v_mid = dv_patch.mul_add(iv as f64, v_range.0) + v_scale;
            for gpu in gauss_pts {
                let u = u_scale.mul_add(gpu.x, u_mid);
                for gpv in gauss_pts {
                    let v = v_scale.mul_add(gpv.x, v_mid);
                    let w = gpu.w * gpv.w * u_scale * v_scale;

                    let p = surface.evaluate(u, v);
                    let du = surface.partial_u(u, v);
                    let dv = surface.partial_v(u, v);

                    // Normal = du x dv (unnormalized, includes Jacobian)
                    let n = Vec3::new(
                        du.y() * dv.z() - du.z() * dv.y(),
                        du.z() * dv.x() - du.x() * dv.z(),
                        du.x() * dv.y() - du.y() * dv.x(),
                    );
                    let n_len = n.length();

                    area += w * n_len;

                    // Volume: (1/3) P dot N (unnormalized N includes Jacobian)
                    let pv = Vec3::new(p.x(), p.y(), p.z()) - about;
                    vol += w * pv.dot(n) / 3.0;

                    // Volume moments via divergence theorem:
                    // CoM_x = (1/2V) surface_integral(x^2 * n_x dA)
                    // n already includes Jacobian, so n.x() = N_x * |J|
                    mx += w * 0.5 * p.x() * p.x() * n.x();
                    my += w * 0.5 * p.y() * p.y() * n.y();
                    mz += w * 0.5 * p.z() * p.z() * n.z();

                    cx += w * p.x() * n_len;
                    cy += w * p.y() * n_len;
                    cz += w * p.z() * n_len;
                }
            }
        }
    }

    FaceContribution {
        area,
        volume: vol * sign,
        volume_moment_x: mx * sign,
        volume_moment_y: my * sign,
        volume_moment_z: mz * sign,
        centroid_x: cx,
        centroid_y: cy,
        centroid_z: cz,
    }
}

/// Absolute shoelace area of a UV polygon. Near-zero means the boundary has
/// collapsed onto a line or point (a degenerate seam/pole projection).
fn polygon_area(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % n];
        a += x0 * y1 - x1 * y0;
    }
    (a * 0.5).abs()
}

/// Dispatch to trimmed or untrimmed parametric integration based on whether
/// a UV boundary polygon is available.
#[allow(clippy::too_many_arguments)]
fn integrate_with_trimming<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
    uv_boundary: &[(f64, f64)],
    u_periodic: bool,
    v_periodic: bool,
    hole_vs: &[f64],
    about: Vec3,
) -> FaceContribution {
    if uv_boundary.len() < 3 {
        return integrate_parametric(surface, u_range, v_range, gauss_order, sign, about);
    }

    // The dense boundary polygon is the reliable signal for a face's true
    // parametric extent: `face_uv_bounds` samples only sparse edge endpoints and
    // under-spans full-revolution faces (a cone's lateral face reports a narrow
    // u-range though its boundary wraps the full 2pi). A face that wraps the
    // full period in u, or whose boundary collapses onto a seam or pole, cannot
    // be trimmed by a UV polygon — the apex/pole/seam folds the polygon and the
    // point-in-polygon test rejects valid interior samples. Integrate the
    // analytic surface untrimmed over its true domain in those cases.
    let u_min = uv_boundary
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min);
    let v_min = uv_boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min);
    let v_max = uv_boundary
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    // Winding number of the boundary around the periodic u-axis: ±TAU for a
    // face that wraps a full revolution, ~0 for a partially-trimmed face. The
    // polygon is already unwrapped (a pole's side may run past half a turn),
    // so its steps telescope and only the closing step is wrapped.
    let tau = std::f64::consts::TAU;
    let winding: f64 = {
        let (first, last) = (uv_boundary[0].0, uv_boundary[uv_boundary.len() - 1].0);
        let close = first - last;
        last - first + close - tau * ((close + std::f64::consts::PI) / tau).floor()
    };
    let full_revolution = u_periodic && winding.abs() >= tau - 1e-3;
    let v_degenerate = (v_max - v_min) <= 1e-9;

    if full_revolution && v_degenerate {
        // Polar cap (e.g. a sphere hemisphere bounded only by one latitude
        // circle): the cap runs from that latitude to a pole. The winding sign
        // (CCW vs CW boundary) selects which pole — the boundary's interior
        // side — so the two hemispheres do not both integrate the whole sphere.
        let v_pole = if winding >= 0.0 { v_range.1 } else { v_range.0 };
        // A full-revolution hole at a latitude between the outer circle and the
        // pole (the drilled-tunnel rim) clips the cap into a band: integrate
        // only from the outer latitude to the hole, not on to the pole.
        let v_far = hole_vs
            .iter()
            .copied()
            // Same side of v_min as the pole (strict same sign → positive
            // product), and not coincident with v_min.
            .filter(|&hv| (hv - v_min) * (v_pole - v_min) > 0.0 && (hv - v_min).abs() > 1e-9)
            .min_by(|a, b| (a - v_min).abs().total_cmp(&(b - v_min).abs()))
            .unwrap_or(v_pole);
        let v_dom = (v_min.min(v_far), v_min.max(v_far));
        integrate_parametric(
            surface,
            (u_min, u_min + tau),
            v_dom,
            gauss_order,
            sign,
            about,
        )
    } else if full_revolution {
        // Full-revolution band (cone/cylinder): integrate the whole revolution
        // over the band's v-extent.
        integrate_parametric(
            surface,
            (u_min, u_min + tau),
            (v_min, v_max),
            gauss_order,
            sign,
            about,
        )
    } else if polygon_area(uv_boundary) <= 1e-12 {
        // Collapsed polygon (e.g. a closed torus whose seam projects to a
        // point): trust the analytic full-domain range from `face_uv_bounds`.
        integrate_parametric(surface, u_range, v_range, gauss_order, sign, about)
    } else {
        integrate_parametric_trimmed(
            surface,
            u_range,
            v_range,
            gauss_order,
            sign,
            uv_boundary,
            u_periodic,
            v_periodic,
            about,
        )
    }
}

/// Integrate a parametric surface with UV boundary trimming.
///
/// At each Gauss point, checks if the (u,v) coordinate falls inside the
/// face's UV boundary polygon. Points outside are skipped (zero contribution).
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]
fn integrate_parametric_trimmed<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
    gauss_order: usize,
    sign: f64,
    uv_boundary: &[(f64, f64)],
    u_periodic: bool,
    v_periodic: bool,
    about: Vec3,
) -> FaceContribution {
    use brepkit_math::predicates::point_in_polygon;
    use brepkit_math::vec::Point2;

    // The same patches as `integrate_parametric`: one Gauss rule over a whole
    // period misses the periodic terms of a moment or of a flux about a point
    // off the axis.
    const MAX_PATCHES: usize = 16;
    let gauss_pts = gauss_legendre_points(gauss_order);
    let patch = std::f64::consts::FRAC_PI_4;
    let nu = (((u_range.1 - u_range.0).abs() / patch).ceil() as usize).clamp(1, MAX_PATCHES);
    let nv = (((v_range.1 - v_range.0).abs() / patch).ceil() as usize).clamp(1, MAX_PATCHES);
    let du_patch = (u_range.1 - u_range.0) / nu as f64;
    let dv_patch = (v_range.1 - v_range.0) / nv as f64;
    let u_scale = du_patch / 2.0;
    let v_scale = dv_patch / 2.0;

    let uv_poly: Vec<Point2> = uv_boundary
        .iter()
        .map(|(u, v)| Point2::new(*u, *v))
        .collect();

    let u_bcenter = if u_periodic {
        let bmin = uv_boundary
            .iter()
            .map(|(bu, _)| *bu)
            .fold(f64::INFINITY, f64::min);
        let bmax = uv_boundary
            .iter()
            .map(|(bu, _)| *bu)
            .fold(f64::NEG_INFINITY, f64::max);
        (bmin + bmax) * 0.5
    } else {
        0.0
    };
    // The polygon's v can sit a period off the bounds' (a torus sector whose
    // boundary walk starts on its top arc).
    let v_bcenter = {
        let bmin = uv_boundary
            .iter()
            .map(|(_, bv)| *bv)
            .fold(f64::INFINITY, f64::min);
        let bmax = uv_boundary
            .iter()
            .map(|(_, bv)| *bv)
            .fold(f64::NEG_INFINITY, f64::max);
        (bmin + bmax) * 0.5
    };

    let mut area = 0.0;
    let mut vol = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    let points = (0..nu)
        .flat_map(|iu| (0..nv).map(move |iv| (iu, iv)))
        .flat_map(|(iu, iv)| {
            let u_mid = du_patch.mul_add(iu as f64, u_range.0) + u_scale;
            let v_mid = dv_patch.mul_add(iv as f64, v_range.0) + v_scale;
            gauss_pts.iter().flat_map(move |gpu| {
                gauss_pts.iter().map(move |gpv| {
                    (
                        u_scale.mul_add(gpu.x, u_mid),
                        v_scale.mul_add(gpv.x, v_mid),
                        gpu.w * gpv.w * u_scale * v_scale,
                    )
                })
            })
        });
    for (u, v, w) in points {
        let test_u = if u_periodic {
            let tau = std::f64::consts::TAU;
            let diff = u - u_bcenter;
            u_bcenter + diff - tau * ((diff + std::f64::consts::PI) / tau).floor()
        } else {
            u
        };

        let test_v = if v_periodic {
            let tau = std::f64::consts::TAU;
            let diff = v - v_bcenter;
            v_bcenter + diff - tau * ((diff + std::f64::consts::PI) / tau).floor()
        } else {
            v
        };

        if !point_in_polygon(Point2::new(test_u, test_v), &uv_poly) {
            continue;
        }

        let p = surface.evaluate(u, v);
        let du = surface.partial_u(u, v);
        let dv = surface.partial_v(u, v);
        let n = Vec3::new(
            du.y() * dv.z() - du.z() * dv.y(),
            du.z() * dv.x() - du.x() * dv.z(),
            du.x() * dv.y() - du.y() * dv.x(),
        );
        let n_len = n.length();

        area += w * n_len;

        let pv = Vec3::new(p.x(), p.y(), p.z()) - about;
        vol += w * pv.dot(n) / 3.0;

        mx += w * 0.5 * p.x() * p.x() * n.x();
        my += w * 0.5 * p.y() * p.y() * n.y();
        mz += w * 0.5 * p.z() * p.z() * n.z();

        cx += w * p.x() * n_len;
        cy += w * p.y() * n_len;
        cz += w * p.z() * n_len;
    }

    FaceContribution {
        area,
        volume: vol * sign,
        volume_moment_x: mx * sign,
        volume_moment_y: my * sign,
        volume_moment_z: mz * sign,
        centroid_x: cx,
        centroid_y: cy,
        centroid_z: cz,
    }
}

/// Build a UV boundary polygon from a face's outer wire.
///
/// Projects each boundary vertex onto the surface to obtain (u, v) coordinates,
/// then unwraps periodic u-coordinates to avoid seam discontinuities.
fn build_face_uv_boundary<S: ParametricSurface>(
    topo: &Topology,
    face_id: FaceId,
    surface: &S,
    u_periodic: bool,
    v_periodic: bool,
) -> Result<Vec<(f64, f64)>, CheckError> {
    let polygon = crate::util::face_polygon(topo, face_id)?;
    if polygon.len() < 3 {
        return Ok(vec![]);
    }

    let mut uv: Vec<(f64, f64)> = polygon.iter().map(|&p| surface.project_point(p)).collect();

    if u_periodic && let Some(first) = uv.iter().position(|&p| singular_in_u(surface, p)) {
        // A point with no `u` of its own (a pole, an apex) is a side of the
        // polygon along its `v`, from the `u` before it to the `u` after it.
        // The walk starts just after the first such point and closes along
        // it; the point after any later one is unwrapped toward the middle
        // of the range so far, as in `face_uv_bounds`.
        uv.rotate_left(first + 1);
        let mut out: Vec<(f64, f64)> = Vec::with_capacity(uv.len() + 2);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut pole: Option<f64> = None;
        for &(u, v) in &uv {
            if singular_in_u(surface, (u, v)) {
                if let Some(&(last, _)) = out.last() {
                    out.push((last, v));
                }
                pole = Some(v);
                continue;
            }
            let u = match (out.last(), pole) {
                (None, _) => u,
                (Some(&(last, _)), None) => unwrap_angle(last, u),
                (Some(_), Some(_)) => unwrap_angle(f64::midpoint(lo, hi), u),
            };
            if let (Some(pv), false) = (pole, out.is_empty()) {
                out.push((u, pv));
            }
            pole = None;
            lo = lo.min(u);
            hi = hi.max(u);
            out.push((u, v));
        }
        if let (Some(pv), Some(&(u0, _))) = (pole, out.first()) {
            out.push((u0, pv));
        }
        uv = out;
    } else if u_periodic {
        for i in 1..uv.len() {
            uv[i].0 = unwrap_angle(uv[i - 1].0, uv[i].0);
        }
    }
    // A band running over a v-periodic surface's v seam (a torus's v = 0
    // line) keeps a contiguous v only when v is unwrapped too.
    if v_periodic {
        for i in 1..uv.len() {
            uv[i].1 = unwrap_angle(uv[i - 1].1, uv[i].1);
        }
    }

    Ok(uv)
}

/// Whether the surface has no `u` of its own at `(u, v)` (a sphere's pole, a
/// cone's apex), where a point projects to an arbitrary `u`.
fn singular_in_u<S: ParametricSurface>(surface: &S, (u, v): (f64, f64)) -> bool {
    surface.partial_u(u, v).length() <= 1e-6 * surface.partial_v(u, v).length()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::{Point3, Vec3};

    #[test]
    fn planar_fan_is_signed_on_nonconvex_polygons() {
        // An L-shape (10x10 square minus a 5x5 corner notch, area 75). The
        // fan pivot at (0,0) sweeps triangles across the notch; an unsigned
        // fan counted them positively and measured 87.5.
        let poly = [
            Point3::new(0.0, 0.0, 2.0),
            Point3::new(10.0, 0.0, 2.0),
            Point3::new(10.0, 5.0, 2.0),
            Point3::new(5.0, 5.0, 2.0),
            Point3::new(5.0, 10.0, 2.0),
            Point3::new(0.0, 10.0, 2.0),
        ];
        let up = Vec3::new(0.0, 0.0, 1.0);
        let c = integrate_planar_polygon(&poly, up, Vec3::new(0.0, 0.0, 0.0));
        assert!((c.area - 75.0).abs() < 1e-9, "area {}", c.area);
        assert!(
            (c.volume - 2.0 * 75.0 / 3.0).abs() < 1e-9,
            "vol {}",
            c.volume
        );

        // The same polygon wound CW nets negative and must flip wholesale,
        // preserving the positive-area contract the hole subtraction relies on.
        let rev: Vec<Point3> = poly.iter().rev().copied().collect();
        let c2 = integrate_planar_polygon(&rev, up, Vec3::new(0.0, 0.0, 0.0));
        assert!((c2.area - 75.0).abs() < 1e-9, "rev area {}", c2.area);
    }

    /// The part of a unit ball's upper hemisphere 270 degrees wide, its wire
    /// (up a meridian to the pole, down another, along the equator) started
    /// at each of its edges and run either way, its pole vertex on the axis
    /// or 1e-7 off it: the pole's `u` is arbitrary, and the face reads its
    /// area `3π/2` however its wire runs.
    #[test]
    fn a_wedge_through_a_pole_reads_its_area_from_any_start() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let origin = Point3::new(0.0, 0.0, 0.0);
        let turn = 1.5 * std::f64::consts::PI;
        let (east, west) = (
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(turn.cos(), turn.sin(), 0.0),
        );
        let arc = |a: Point3, b: Point3| {
            let normal = (a - origin).cross(b - origin);
            EdgeCurve::Circle(Circle3D::new(origin, normal, 1.0).unwrap())
        };
        for (first, reversed, off) in (0..3).flat_map(|f| {
            [(false, 0.0), (true, 0.0), (false, 1e-7), (true, 1e-7)].map(|(r, o)| (f, r, o))
        }) {
            let pole = Point3::new(off, 0.0, (1.0_f64 - off * off).sqrt());
            let mut topo = Topology::new();
            let [ve, vp, vw] = [east, pole, west].map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let equator = Circle3D::new(origin, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
            let mut edges = vec![
                topo.add_edge(Edge::new(vw, vp, arc(west, pole))),
                topo.add_edge(Edge::new(vp, ve, arc(pole, east))),
                topo.add_edge(Edge::new(ve, vw, EdgeCurve::Circle(equator))),
            ];
            edges.rotate_left(first);
            let edges = if reversed {
                edges
                    .into_iter()
                    .rev()
                    .map(|e| OrientedEdge::new(e, false))
                    .collect()
            } else {
                edges
                    .into_iter()
                    .map(|e| OrientedEdge::new(e, true))
                    .collect()
            };
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            let ball = SphericalSurface::new(origin, 1.0).unwrap();
            let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Sphere(ball)));
            let c = integrate_face(&topo, face, 5).unwrap();
            assert!(
                (c.area - turn).abs() < 1e-6,
                "wire started at edge {first}, reversed {reversed}, pole {off} off: area {}, truth {turn}",
                c.area
            );
        }
    }

    /// A disc of radius 2 at `z = 3` with a hole of radius 1, bounded by
    /// closed circles: its area, flux and first moments come out exact, where
    /// 32 chords a circle hold 0.64% less.
    #[test]
    fn a_plane_face_reads_its_circles_exactly() {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let centre = Point3::new(1.0, -2.0, 3.0);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let mut ring = |r: f64, forward: bool| {
            let v = topo.add_vertex(Vertex::new(Point3::new(1.0 + r, -2.0, 3.0), 1e-7));
            let c = Circle3D::new(centre, up, r).unwrap();
            let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(c)));
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e, forward)], true).unwrap())
        };
        let (outer, hole) = (ring(2.0, true), ring(1.0, false));
        let plane = FaceSurface::Plane { normal: up, d: 3.0 };
        let face = topo.add_face(Face::new(outer, vec![hole], plane));
        let c = integrate_face(&topo, face, 5).unwrap();
        let area = 3.0 * std::f64::consts::PI;
        assert!((c.area - area).abs() < 1e-12, "area {}", c.area);
        assert!((c.volume - area).abs() < 1e-12, "volume {}", c.volume);
        assert!(
            (c.centroid_x - area).abs() < 1e-12,
            "x moment {}",
            c.centroid_x
        );
        assert!(
            (c.centroid_y + 2.0 * area).abs() < 1e-12,
            "y moment {}",
            c.centroid_y
        );
        assert!(
            (c.volume_moment_z - 4.5 * area).abs() < 1e-12,
            "z moment {}",
            c.volume_moment_z
        );
    }

    /// A unit cylinder's wall turning 270 degrees and 1 tall, its lower rim a
    /// NURBS arc stored either way round: the wall reads its area whichever
    /// end the rim's knot span starts at.
    #[test]
    fn a_rim_stored_end_to_start_reads_its_span() {
        use brepkit_geometry::convert::circle_to_nurbs;
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let turn = 1.5 * std::f64::consts::PI;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let at = |a: f64, z: f64| Point3::new(a.cos(), a.sin(), z);
        for end_to_start in [false, true] {
            let mut topo = Topology::new();
            let [a0, b0, a1, b1] = [(0.0, 0.0), (turn, 0.0), (0.0, 1.0), (turn, 1.0)]
                .map(|(a, z)| topo.add_vertex(Vertex::new(at(a, z), 1e-7)));
            let up = Circle3D::new(origin, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
            let rim = if end_to_start {
                let down = Circle3D::new(origin, Vec3::new(0.0, 0.0, -1.0), 1.0).unwrap();
                let t0 = down.project(at(turn, 0.0));
                circle_to_nurbs(&down, t0, t0 + turn).unwrap()
            } else {
                let t0 = up.project(at(0.0, 0.0));
                circle_to_nurbs(&up, t0, t0 + turn).unwrap()
            };
            let top = Circle3D::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), 1.0);
            let edges = [
                (
                    topo.add_edge(Edge::new(a0, b0, EdgeCurve::NurbsCurve(rim))),
                    true,
                ),
                (topo.add_edge(Edge::new(b0, b1, EdgeCurve::Line)), true),
                (
                    topo.add_edge(Edge::new(a1, b1, EdgeCurve::Circle(top.unwrap()))),
                    false,
                ),
                (topo.add_edge(Edge::new(a1, a0, EdgeCurve::Line)), true),
            ]
            .map(|(e, forward)| OrientedEdge::new(e, forward));
            let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
            let cylinder = CylindricalSurface::new(origin, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
            let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Cylinder(cylinder)));
            let c = integrate_face(&topo, face, 5).unwrap();
            assert!(
                (c.area - turn).abs() < 1e-9,
                "end to start {end_to_start}: area {}, truth {turn}",
                c.area
            );
        }
    }
}
