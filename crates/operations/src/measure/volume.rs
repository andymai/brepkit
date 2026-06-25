//! Volume, center of mass, and related computations for B-rep solids.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::tessellate;

use super::helpers::{collect_solid_vertex_points, compute_angular_range};

/// Volume of a solid that contains a bored quadric — a sphere (or torus) face
/// carrying a full-revolution latitude-circle hole (a drilled tunnel rim) — via
/// exact per-face Gauss quadrature on the analytic surfaces.
///
/// The tessellation paths below cannot bound such an annular band: both of its
/// boundary loops are constant-v latitude circles, so the band's UV outline is
/// degenerate and the mesh fills the removed polar cap, over-counting. The
/// per-face analytic integrator (orientation-aware, hole-clipped) is exact.
///
/// Scope is deliberately narrow — only solids whose tessellated volume is known
/// to be wrong — so every other analytic solid keeps its existing
/// tessellation-based volume. Returns `None` (defer to tessellation) when no
/// bored quadric is present, when any face is NURBS, or when a face fails to
/// integrate.
/// Whether a sphere face's outer wire lies on a single constant-`v` latitude
/// (the simple bored-quadric band) rather than a scalloped, varying-`v` collar
/// floor. Projects the outer wire's vertices to `(u, v)` and tests the `v`
/// spread.
fn sphere_outer_wire_constant_v(
    topo: &Topology,
    face_id: FaceId,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return false;
    };
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            return false;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            return false;
        };
        let (sp, ep) = (sv.point(), ev.point());
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        // Sample ALONG each edge, not just its start vertex: a great-circle arc
        // has both endpoints on the seam latitude yet bulges away from it, so
        // endpoint-only sampling would mis-read a scalloped collar floor as a
        // constant-v band and wrongly take the analytic fast path.
        for i in 0..=8 {
            let t = t0 + (t1 - t0) * (f64::from(i) / 8.0);
            let (_, v) = sphere.project_point(edge.curve().evaluate_with_endpoints(t, sp, ep));
            v_min = v_min.min(v);
            v_max = v_max.max(v);
        }
    }
    // Latitude-flatness threshold sized to the linear-tolerance magnitude: a
    // real band's v-spread is ~fp-noise; a collar's is a large fraction of a
    // radian.
    (v_max - v_min) <= 1e-7
}

/// Whether the solid has at least one sphere face that is a scalloped collar
/// (a bored quadric whose outer wire varies in `v`, e.g. a box ∩ sphere patch).
fn solid_has_scalloped_sphere_collar(topo: &Topology, solid: SolidId) -> bool {
    let Ok(faces) = brepkit_topology::explorer::solid_faces(topo, solid) else {
        return false;
    };
    faces.iter().any(|&fid| {
        topo.face(fid).is_ok_and(|f| match f.surface() {
            FaceSurface::Sphere(s) => {
                !f.inner_wires().is_empty() && !sphere_outer_wire_constant_v(topo, fid, s)
            }
            _ => false,
        })
    })
}

/// Count mesh edges incident to a number of triangles other than 2 (boundary or
/// non-manifold edges). Zero means a closed 2-manifold.
fn mesh_boundary_edge_count(mesh: &tessellate::TriangleMesh) -> usize {
    use brepkit_math::det_hash::DetHashMap;
    let mut counts: DetHashMap<(u32, u32), usize> = DetHashMap::default();
    for tri in mesh.indices.chunks_exact(3) {
        for &(i, j) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if i < j { (i, j) } else { (j, i) };
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts.values().filter(|&&c| c != 2).count()
}

fn analytic_faces_solid_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use brepkit_topology::explorer::solid_faces;

    let faces = solid_faces(topo, solid).ok()?;
    if faces.is_empty() {
        return None;
    }

    // The Steinmetz lens fuse — two mutually-trimmed equal cylinders, whose
    // walls keep the lens ellipses as holes — has an EXACT closed-form volume
    // (computed directly below). The hole-unaware tessellation paths over-count
    // the lens, and a general holed-cylinder integrator was too broad to be
    // correct; the closed form is exact and needs no special integration.
    if solid_is_steinmetz_lens_fuse(topo, &faces) {
        return steinmetz_lens_fuse_volume(topo, &faces);
    }

    let mut has_bored_quadric = false;
    for &fid in &faces {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Nurbs(_) => return None,
            // Sphere only: the per-face integrator's hole-clipping is wired up
            // for spheres. A bored torus would pass `hole_vs = []` and
            // over-integrate, so defer it to tessellation until torus
            // hole-clipping lands (with the torus−box analytic split).
            FaceSurface::Sphere(s) if !face.inner_wires().is_empty() => {
                // The integrator's hole-clipping models a band between two
                // constant-v latitudes. A collar whose OUTER wire varies in v
                // (great-circle/seam arcs, e.g. a box ∩ sphere patch) is not
                // that shape — its scalloped floor and lune bites would be
                // mis-integrated, so defer the whole solid to tessellation.
                if !sphere_outer_wire_constant_v(topo, fid, s) {
                    return None;
                }
                has_bored_quadric = true;
            }
            // A holed cylinder/cone wall that is NOT the Steinmetz lens fuse
            // (handled above) cannot be integrated correctly here — the
            // integrator does not subtract its holes — so defer the whole solid
            // to tessellation.
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) if !face.inner_wires().is_empty() => {
                return None;
            }
            FaceSurface::Torus(_) if !face.inner_wires().is_empty() => return None,
            _ => {}
        }
    }
    if !has_bored_quadric {
        return None;
    }

    let gauss_order = brepkit_check::properties::PropertiesOptions::default().gauss_order;
    let mut total = 0.0;
    for &fid in &faces {
        total += brepkit_check::properties::face_integrator::integrate_face(topo, fid, gauss_order)
            .ok()?
            .volume;
    }
    Some(total.abs())
}

/// Exact volume of the STEINMETZ LENS FUSE — two equal-radius `r` cylinders with
/// perpendicular intersecting axes, fused.
///
/// `V = π·r²·(h₁ + h₂) − (16/3)·r³`: the two cylinder volumes (heights `h₁`,
/// `h₂` are each wall's cap-to-cap extent along its axis) minus their Steinmetz
/// intersection `16·r³/3`. Reads `r` and the two heights from the two holed
/// cylindrical walls (already verified to exist by
/// [`solid_is_steinmetz_lens_fuse`]). Returns `None` only on a topology lookup
/// failure or a malformed wall.
fn steinmetz_lens_fuse_volume(topo: &Topology, faces: &[FaceId]) -> Option<f64> {
    use std::f64::consts::PI;

    let mut r: Option<f64> = None;
    let mut heights: Vec<f64> = Vec::new();
    for &fid in faces {
        let face = topo.face(fid).ok()?;
        let FaceSurface::Cylinder(cyl) = face.surface() else {
            continue;
        };
        if face.inner_wires().is_empty() {
            continue; // Only the two holed walls.
        }
        // Equal radii: confirm the second wall matches the first.
        match r {
            None => r = Some(cyl.radius()),
            Some(r0) if (r0 - cyl.radius()).abs() > 1e-6 * r0.max(1.0) => return None,
            Some(_) => {}
        }
        // Cap-to-cap height = the axial (v) extent of the wall's outer wire.
        let wire = topo.wire(face.outer_wire()).ok()?;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        for oe in wire.edges() {
            let e = topo.edge(oe.edge()).ok()?;
            for vid in [e.start(), e.end()] {
                let p = topo.vertex(vid).ok()?.point();
                let (_, v) = cyl.project_point(p);
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
        if !v_min.is_finite() || !v_max.is_finite() || v_max <= v_min {
            return None;
        }
        heights.push(v_max - v_min);
    }
    let r = r?;
    if heights.len() != 2 {
        return None;
    }
    let v_cyls = PI * r * r * (heights[0] + heights[1]);
    let v_steinmetz = 16.0 / 3.0 * r * r * r;
    Some(v_cyls - v_steinmetz)
}

/// Whether the solid is the STEINMETZ LENS FUSE — two equal-radius cylinders
/// with PERPENDICULAR, INTERSECTING axes, fused — for which the volume has the
/// EXACT closed form [`steinmetz_lens_fuse_volume`].
///
/// Validated by both topology AND geometry, so a different equal-radius two-
/// cylinder fuse with the same topology (e.g. oblique or parallel-offset axes,
/// whose intersection is NOT `16r³/3`) is rejected and defers to tessellation:
///   * exactly two cylindrical faces, each carrying inner wires (the two seam
///     ellipses as holes); every other face planar (the four end caps);
///   * the two holed walls SHARE their inner-wire edges (the same seam ellipses
///     bound both);
///   * the two cylinders are EQUAL RADIUS, their axes PERPENDICULAR
///     (`|a₁·a₂| ≈ 0`) and INTERSECTING (closest-approach of the two axis lines
///     ≈ 0). The closed form holds only for that right-angle configuration.
///
/// An ordinary drilled cylinder has ONE holed wall (its bore rim is not shared
/// with a second cylindrical wall), so it returns `false`.
fn solid_is_steinmetz_lens_fuse(topo: &Topology, faces: &[FaceId]) -> bool {
    use std::collections::HashSet;

    let mut holed_cyl_walls: Vec<FaceId> = Vec::new();
    for &fid in faces {
        let Ok(face) = topo.face(fid) else {
            return false;
        };
        match face.surface() {
            FaceSurface::Cylinder(_) if !face.inner_wires().is_empty() => holed_cyl_walls.push(fid),
            FaceSurface::Cylinder(_) | FaceSurface::Plane { .. } => {}
            // Any sphere/cone/torus/NURBS face, or a holed non-cylinder, is not
            // the cyl∪cyl lens signature.
            _ => return false,
        }
    }
    if holed_cyl_walls.len() != 2 {
        return false;
    }
    // The two holed walls must SHARE their inner-wire edges (the seam ellipses).
    let inner_edges = |fid: FaceId| -> HashSet<usize> {
        let mut s = HashSet::new();
        if let Ok(face) = topo.face(fid) {
            for &wid in face.inner_wires() {
                if let Ok(wire) = topo.wire(wid) {
                    for oe in wire.edges() {
                        s.insert(oe.edge().index());
                    }
                }
            }
        }
        s
    };
    let a = inner_edges(holed_cyl_walls[0]);
    let b = inner_edges(holed_cyl_walls[1]);
    if a.is_empty() || a != b {
        return false;
    }

    // Geometry: equal radius, perpendicular + intersecting axes.
    let (Ok(f0), Ok(f1)) = (topo.face(holed_cyl_walls[0]), topo.face(holed_cyl_walls[1])) else {
        return false;
    };
    let (FaceSurface::Cylinder(c0), FaceSurface::Cylinder(c1)) = (f0.surface(), f1.surface())
    else {
        return false;
    };
    cylinders_perpendicular_and_intersecting(c0, c1)
}

/// Whether two cylinders are equal-radius with perpendicular, intersecting axes
/// — the geometric precondition for the right-angle Steinmetz closed form.
fn cylinders_perpendicular_and_intersecting(
    c0: &brepkit_math::surfaces::CylindricalSurface,
    c1: &brepkit_math::surfaces::CylindricalSurface,
) -> bool {
    let r0 = c0.radius();
    if (r0 - c1.radius()).abs() > 1e-6 * r0.max(1.0) {
        return false; // Unequal radius.
    }
    let a0 = c0.axis();
    let a1 = c1.axis();
    if a0.dot(a1).abs() > 1e-6 {
        return false; // Not perpendicular.
    }
    // Closest approach of the two axis lines (perpendicular ⇒ the system
    // decouples): s* = −(w0·a0), t* = w0·a1, where w0 = o0 − o1.
    let o0 = c0.origin();
    let o1 = c1.origin();
    let w0 = Vec3::new(o0.x() - o1.x(), o0.y() - o1.y(), o0.z() - o1.z());
    let s = -w0.dot(a0);
    let t = w0.dot(a1);
    let p0 = Point3::new(
        o0.x() + a0.x() * s,
        o0.y() + a0.y() * s,
        o0.z() + a0.z() * s,
    );
    let p1 = Point3::new(
        o1.x() + a1.x() * t,
        o1.y() + a1.y() * t,
        o1.z() + a1.z() * t,
    );
    (p0 - p1).length() <= 1e-6 * r0.max(1.0)
}

/// Try to compute the volume of a solid analytically by detecting known
/// primitive shapes (sphere, cylinder, cone/frustum, torus).
///
/// Returns `None` if the solid is not a recognized pure primitive, in which
/// case the caller should fall back to tessellation.
///
/// Detection rules (single pass over shell faces):
/// - Any `Nurbs` face -> `None` (fall back)
/// - All faces are `Sphere` -> sphere formula `(4/3)pi*r^3`
/// - Exactly 1 `Cylinder` + >=1 `Plane` caps, 0 other analytic -> `pi*r^2*h`
/// - Exactly 1 `Cone` + <=2 `Plane` caps, 0 other analytic -> cone/frustum formula
///   (cap radii are read from the `Circle3D` edges of the cap faces)
/// - Exactly 1 `Torus` + 0 planes, 0 other analytic -> `2*pi^2*R*r^2`
#[allow(clippy::too_many_lines)]
fn try_analytic_solid_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use std::f64::consts::PI;

    let solid_data = topo.solid(solid).ok()?;
    let shell = topo.shell(solid_data.outer_shell()).ok()?;

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

    if let Some(r) = sphere_r
        && cyl.is_none()
        && cone_params.is_none()
        && torus_params.is_none()
        && planes.is_empty()
    {
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
            if let Ok(face) = topo.face(fid)
                && let Ok(wire) = topo.wire(face.outer_wire())
            {
                for oe in wire.edges() {
                    if let Ok(e) = topo.edge(oe.edge())
                        && let Ok(v) = topo.vertex(e.start())
                    {
                        let d = (v.point() - center).length();
                        max_dist = max_dist.max(d);
                        min_dist = min_dist.min(d);
                    }
                }
            }
        }
        // If all vertices are equidistant (within 1%), use analytic formula
        if (max_dist - min_dist).abs() < r * 0.01 {
            return Some(4.0 / 3.0 * PI * r * r * r);
        }
        // Non-uniform scale detected -- fall through to tessellation
        return None;
    }

    // A pure cylinder has exactly 1 cylindrical face and 2 planar caps.
    // If there are more than 2 planes the solid is compound (e.g. a box
    // with a drilled hole has 1 cylindrical hole-wall + 6 box faces).
    // In the compound case the cylindrical face is a concave inner surface
    // and the formula pi*r^2*h would compute the cylinder volume, not the solid.
    if let Some((origin, axis, r)) = cyl
        && cone_params.is_none()
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

    // Cap radii are read directly from the Circle3D edges of the cap faces,
    // bypassing the ConicalSurface parameterization entirely. Heights are
    // derived from the circle centers projected onto the cone axis.
    if let Some((apex, axis)) = cone_params
        && cyl.is_none()
        && torus_params.is_none()
        && sphere_r.is_none()
    {
        let apex_vec = Vec3::new(apex.x(), apex.y(), apex.z());

        // Collect (circle_center, radius) from each plane cap face.
        let mut cap_circles: Vec<(Point3, f64)> = Vec::new();
        for &fid in &plane_face_ids {
            if let Some(cap) = find_cap_circle(topo, fid) {
                cap_circles.push(cap);
            }
        }

        // If any cap face did not yield a circle, the cone is degenerate or
        // unsupported -- fall back to tessellation rather than silently wrong answer.
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

    if let Some((r_major, r_minor)) = torus_params
        && cyl.is_none()
        && cone_params.is_none()
        && sphere_r.is_none()
        && planes.is_empty()
    {
        return Some(2.0 * PI * PI * r_major * r_minor * r_minor);
    }

    None
}

/// Minimum |n . axis| for a plane to be considered a perpendicular cap face
/// (i.e. the plane normal is within ~8 deg of the axis direction).
const AXIS_PARALLEL_MIN_DOT: f64 = 0.99;

/// Compute signed distances along `axis` from `ref_pt` to cap planes that are
/// roughly perpendicular to the axis (`|n . axis| > AXIS_PARALLEL_MIN_DOT`).
///
/// For a plane `n . P = d`, the intersection with the line `ref_pt + t * axis`
/// satisfies `t = (d - n . ref_pt) / (n . axis)`.
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

/// Clamp the tessellation deflection used for volume so curved faces are
/// sampled finely enough for an accurate boundary integral.
///
/// A coarse preview deflection inscribes too few facets in a curved face and
/// under-counts its volume; volume is a precise query, so cap the deflection at
/// a small fraction of the solid's extent, estimated as the diagonal of the
/// **vertex** bounding box. (Curvature can bulge slightly beyond the vertices,
/// so this under-estimates the true AABB — but only by making the cap
/// conservatively finer, which is safe.) Never coarsens a finer request, and
/// falls back to `requested` if the extent cannot be determined.
///
/// A solid-extent scale (rather than per-curved-face curvature radius) is used
/// deliberately: it keeps the deflection consistent between a sub-solid and a
/// boolean result containing it, preserving the `volume(A ∪ B) == volume(A) +
/// volume(B)` invariant for coincident-contact fuses. A curvature-radius cap
/// would tessellate a shared face differently in each context and break it.
fn volume_tessellation_deflection(topo: &Topology, solid: SolidId, requested: f64) -> f64 {
    let Ok(pts) = collect_solid_vertex_points(topo, solid) else {
        return requested;
    };
    let Some((&first, rest)) = pts.split_first() else {
        return requested;
    };
    let (mut lo, mut hi) = (first, first);
    for p in rest {
        lo = Point3::new(lo.x().min(p.x()), lo.y().min(p.y()), lo.z().min(p.z()));
        hi = Point3::new(hi.x().max(p.x()), hi.y().max(p.y()), hi.z().max(p.z()));
    }
    let diag = (hi - lo).length();
    if !diag.is_finite() || diag <= 0.0 {
        return requested;
    }
    requested.min((diag * 5e-5).max(1e-9))
}

/// Compute the volume of a solid using the signed tetrahedra method
/// (divergence theorem on a surface tessellation).
///
/// For each triangle `(v0, v1, v2)`, the signed volume of the
/// tetrahedron it forms with the origin is `v0 . (v1 x v2) / 6`.
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

    // Fast path: a solid whose faces are ALL analytic (planes + quadrics,
    // no NURBS) integrates exactly via per-face Gauss quadrature on the
    // analytic surfaces — orientation-aware and immune to the inscribed-mesh
    // undercount and the degenerate-UV annular-band over-count that the
    // tessellation paths below suffer on bored quadrics (e.g. a cylinder
    // drilled through a sphere).
    if let Some(v) = analytic_faces_solid_volume(topo, solid) {
        return Ok(v);
    }

    // Volume integrates the boundary, so curved faces must be tessellated
    // finely or the inscribed mesh under-counts them (a swept cylinder or a
    // box with a cylindrical hole measures ~1-2% low at a coarse preview
    // deflection). Clamp the deflection to a small fraction of the solid's
    // extent — never coarsening a finer request — so the volume is accurate
    // regardless of the (preview-tuned) deflection the caller passes.
    let deflection = volume_tessellation_deflection(topo, solid, deflection);

    // A scalloped sphere collar (box ∩ sphere) cannot be per-face tessellated
    // watertight (its band path needs the solid's shared boundary vertices), and
    // its analytic integral is the hard u-dependent lune trim we defer. The
    // whole-solid mesh IS watertight, so take the divergence-theorem volume off
    // that closed mesh.
    if solid_has_scalloped_sphere_collar(topo, solid) {
        let mesh = tessellate::tessellate_solid(topo, solid, deflection)?;
        if !mesh.indices.is_empty() && mesh_boundary_edge_count(&mesh) == 0 {
            return Ok(signed_volume_from_mesh(&mesh));
        }
        // Non-watertight mesh: fall through to the generic paths below rather
        // than return a leaky volume.
    }

    // Fast path: for solids made entirely of planar triangular faces
    // (e.g. mesh imports), compute volume directly from face geometry.
    // This avoids re-tessellation which has known WASM winding issues.
    if let Ok(v) = solid_volume_from_faces(topo, solid, deflection) {
        return Ok(v);
    }

    // Planar polygon volume (Newell area) is disabled: GFA boolean results
    // go through merge_duplicate_edges which can create crossed polygon
    // winding, making Newell area wrong. Always use tessellation-based
    // volume which handles all cases correctly.

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

    // Try watertight tessellation -- gives correct volume via signed tetrahedra
    // since the mesh is closed.
    let mesh = tessellate::tessellate_solid(topo, solid, deflection)?;
    if !mesh.indices.is_empty() {
        let vol = signed_volume_from_mesh(&mesh);
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

/// Compute volume by tessellating each face independently and summing
/// signed tetrahedra contributions (divergence theorem).
///
/// `tessellate()` already handles face reversal (flipping triangle
/// winding for reversed faces), so the raw signed tetrahedra sum
/// produces the correct result without any winding heuristic.
fn volume_from_per_face_tessellation(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total: f64 = 0.0;
    for &fid in shell.faces() {
        let mesh = tessellate::tessellate(topo, fid, deflection)?;
        let idx = &mesh.indices;
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;

        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());

            total += a.dot(b.cross(c));
        }
    }

    let signed_volume = total / 6.0;
    if signed_volume < 0.0 {
        log::debug!(
            "volume_from_per_face_tessellation: raw signed volume is negative ({signed_volume:.6}), \
             possible face orientation issue"
        );
    }
    Ok(signed_volume.abs())
}

/// Exact signed volume contribution of a cylindrical face via the
/// divergence theorem: `V = (1/3) integral P.n dA`.
///
/// For a cylinder parameterised as
///   `P(u,v) = O + r*(cos u * ex + sin u * ey) + v * a`
/// the outward normal is `n = cos u * ex + sin u * ey`, dA = r du dv.
///
/// Integrating analytically over `u in [u1,u2], v in [v1,v2]`:
///   `V = (r/3) * h * [ ox*(sin u2 - sin u1) + oy*(-cos u2 + cos u1) + r*(u2 - u1) ]`
/// where `ox = O.ex`, `oy = O.ey`, `h = v2 - v1`.
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
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
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
            // A revolution-band boundary is a rational NURBS arc, not an
            // `EdgeCurve::Circle`. Sample its domain midpoint too, or a partial
            // (sub-2π) band has only its two endpoint angles, `compute_angular_range`
            // falls back to the full 2π, and the band over-counts (gh #968).
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) = edge.curve()
            {
                let (t0, t1) = nc.domain();
                let (u, _) = cyl.project_point(nc.evaluate(f64::midpoint(t0, t1)));
                u_vals.push(u);
            }
        }
    }

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
/// theorem: `V = (1/3) integral P.n dA`.
///
/// For a cone parameterised as
///   `P(u,v) = apex + v*(cos_a*(cos u * ex + sin u * ey) + sin_a * axis)`
/// the outward normal is `n = sin_a*(cos u * ex + sin u * ey) - cos_a * axis`,
/// and `dA = v * cos_a * du dv`.
///
/// The integrand `P.n * dA` simplifies to closed form over `[u1,u2] x [v1,v2]`.
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
            // Sample NURBS revolution-band arcs too (see the cylinder case, #968).
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) = edge.curve()
            {
                let (t0, t1) = nc.domain();
                let (u, _) = cone.project_point(nc.evaluate(f64::midpoint(t0, t1)));
                u_vals.push(u);
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
    // P.n = apex.(sin_a*radial - cos_a*axis)
    //     + v*(cos_a*sin_a*(radial.radial) + sin_a^2*(axis.radial) - cos_a^2*(radial.axis) - cos_a*sin_a*(axis.axis))
    //     = apex.(sin_a*radial - cos_a*axis) + v*(cos_a*sin_a - cos_a*sin_a)
    //     = apex.(sin_a*radial(u) - cos_a*axis)
    //
    // The v-dependent terms cancel: cos_a*sin_a - cos_a*sin_a = 0, so P.n is v-independent.
    //
    // Full integrand = (1/3) * P.n * dA = (1/3) * [a_vec.(sin_a*radial(u) - cos_a*axis)] * v*cos_a * du * dv
    //
    // integral = (cos_a/3) * [(v^2/2)|v1..v2] * integral[sin_a*(ax*cos_u + ay*sin_u) - cos_a*az] du
    // where ax = a_vec.x_axis, ay = a_vec.y_axis, az = a_vec.axis
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
/// theorem: `V = (1/3) integral P.n dA`.
///
/// For a sphere parameterised as
///   `P(u,v) = C + r*(cos_v*cos_u*ex + cos_v*sin_u*ey + sin_v*ez)`
/// the outward normal equals the unit radial direction, and `dA = r^2*cos_v * du dv`.
///
/// `P.n = C.n + r`, so the integrand is `(1/3)*(C.n + r)*r^2*cos_v du dv`.
#[allow(clippy::too_many_lines)]
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
                let (u, _) = sph.project_point(mid);
                u_vals.push(u);
            }
        }
    }

    let mut v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mut v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // For sphere caps (single circle boundary at one latitude), the boundary
    // vertices all share approximately the same v, so v_max ~ v_min.
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

    // P.n = C.(cos_v*cos_u*ex + cos_v*sin_u*ey + sin_v*ez) + r
    // dA = r^2 * cos_v * du * dv
    //
    // Integrand = (1/3) * (cx*cos_v*cos_u + cy*cos_v*sin_u + cz*sin_v + r) * r^2 * cos_v
    // where cx = C.ex, cy = C.ey, cz = C.ez
    let cx = c_vec.dot(x_axis);
    let cy = c_vec.dot(y_axis);
    let cz = c_vec.dot(z_axis);

    let (u1, u2) = u_range;
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();
    let du = u2 - u1;

    // integral cos_v*cos_v dv = v/2 + sin(2v)/4
    let vv_integral = |v: f64| -> f64 { v / 2.0 + (2.0 * v).sin() / 4.0 };
    let cos2_v = vv_integral(v_max) - vv_integral(v_min);

    // integral cos_v dv = sin_v
    let cos_v_int = v_max.sin() - v_min.sin();

    // integral sin_v*cos_v dv = sin^2(v)/2
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
/// theorem: `V = (1/3) integral P.n dA`.
///
/// For a torus parameterised as
///   `P(u,v) = C + (R + r*cos_v)*(cos_u*ex + sin_u*ey) + r*sin_v*ez`
/// the outward normal `n = cos_v*(cos_u*ex + sin_u*ey) + sin_v*ez`,
/// and `dA = r*(R + r*cos_v) du dv`.
#[allow(clippy::too_many_lines)]
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

    // P.n = [C + (R+r*cos_v)*radial_u + r*sin_v*ez] . [cos_v*radial_u + sin_v*ez]
    //     = C.(cos_v*radial_u + sin_v*ez) + (R+r*cos_v)*cos_v + r*sin^2_v
    //     = cos_v*(cx*cos_u + cy*sin_u) + sin_v*cz + (R+r*cos_v)*cos_v + r*sin^2_v
    //     = cos_v*(cx*cos_u + cy*sin_u) + sin_v*cz + R*cos_v + r*cos^2_v + r*sin^2_v
    //     = cos_v*(cx*cos_u + cy*sin_u) + sin_v*cz + R*cos_v + r
    //
    // dA = r*(R + r*cos_v) du dv
    //
    // Full integrand = (1/3) * P.n * dA
    let cx = c_vec.dot(x_axis);
    let cy = c_vec.dot(y_axis);
    let cz = c_vec.dot(z_axis);

    let (u1, u2) = u_range;
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();
    let du = u2 - u1;

    // We need to integrate over v:
    // integral [cos_v*(cx*cos_u + cy*sin_u) + cz*sin_v + R*cos_v + r] * r*(R + r*cos_v) dv
    //
    // Expand the product with (R + r*cos_v):
    // = r * integral [cos_v*(cx*cos_u+cy*sin_u)*(R+r*cos_v)
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

    // integral cos_v dv = sin_v
    let i_cos = sv2 - sv1;
    // integral cos^2_v dv = v/2 + sin(2v)/4
    let i_cos2 =
        (v_max / 2.0 + (2.0 * v_max).sin() / 4.0) - (v_min / 2.0 + (2.0 * v_min).sin() / 4.0);
    // integral sin_v dv = -cos_v
    let i_sin = -cv2 + cv1;
    // integral sin_v*cos_v dv = sin^2(v)/2
    let i_sincos = (sv2 * sv2 - sv1 * sv1) / 2.0;
    // Group terms by u-dependence:
    // Terms with S (= cx*cos_u + cy*sin_u):
    //   r*[R*i_cos + r*i_cos2] * integral S du
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

/// Compute volume by tessellating each face and summing signed tetrahedra
/// WITHOUT winding correction. Relies on `tessellate()` already handling
/// face reversal (via `is_reversed` flag) to produce correctly oriented
/// triangles. For analytic surface faces (cylinder, cone, sphere, torus),
/// uses exact analytical integration via the divergence theorem instead
/// of tessellation.
pub fn volume_from_direct_face_tessellation(
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

/// Compute the volume of a solid directly from its face vertex
/// positions, bypassing tessellation. Only valid for solids composed
/// entirely of planar triangular faces (e.g. mesh imports).
///
/// Returns an error if the solid contains non-planar or
/// non-triangular faces.
///
/// # Errors
///
/// Returns [`crate::OperationsError`] if topology lookups fail or if the
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

    // tessellate() already handles face reversal (flips winding),
    // so signed tetrahedra sum is correct without winding heuristics.
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

        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());

            let signed_vol = a.dot(b.cross(c));
            total_vol += signed_vol;
            cx += signed_vol * (v0.x() + v1.x() + v2.x());
            cy += signed_vol * (v0.y() + v1.y() + v2.y());
            cz += signed_vol * (v0.z() + v1.z() + v2.z());
        }
    }

    if total_vol.abs() < 1e-15 {
        // Volume too small to compute weighted CoM -- fall back to vertex centroid.
        let vertex_points = collect_solid_vertex_points(topo, solid)?;
        let n = vertex_points.len().max(1) as f64;
        let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
        for p in &vertex_points {
            sx += p.x();
            sy += p.y();
            sz += p.z();
        }
        return Ok(Point3::new(sx / n, sy / n, sz / n));
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

#[cfg(test)]
mod regression_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};
    use brepkit_topology::face::FaceSurface;

    fn unit_square_extrude_volume() -> (f64, bool) {
        let mut topo = Topology::new();
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let wire = make_polygon_wire(&mut topo, &pts, 1e-7).unwrap();
        let face = make_face_from_wire(&mut topo, wire).unwrap();
        let cap_is_plane = matches!(
            topo.face(face).unwrap().surface(),
            FaceSurface::Plane { .. }
        );
        let solid =
            crate::extrude::extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        (vol, cap_is_plane)
    }

    #[test]
    fn unit_square_extrude_volume_is_one() {
        let (vol, cap_is_plane) = unit_square_extrude_volume();
        assert!(
            cap_is_plane,
            "axis-aligned square cap must be a planar face"
        );
        assert!((vol - 1.0).abs() < 1e-6, "expected 1.0, got {vol}");
    }

    #[test]
    fn rectangle_extrude_volume_matches_box() {
        // A non-square, non-unit axis-aligned rectangle whose winding-derived
        // cap normal must still come out correct.
        let mut topo = Topology::new();
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(5.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];
        let wire = make_polygon_wire(&mut topo, &pts, 1e-7).unwrap();
        let face = make_face_from_wire(&mut topo, wire).unwrap();
        let solid =
            crate::extrude::extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        assert!((vol - 30.0).abs() < 1e-6, "expected 30.0, got {vol}");
    }

    /// Build the census Steinmetz fuse: two equal r=3, h=20 cylinders with
    /// perpendicular intersecting axes (one along z, one along x), fused.
    fn steinmetz_fuse_census() -> (Topology, SolidId) {
        use brepkit_math::mat::Mat4;
        let mut topo = Topology::new();
        let c1 = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        crate::transform::transform_solid(&mut topo, c1, &Mat4::translation(0.0, 0.0, -10.0))
            .unwrap();
        let c2 = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            c2,
            &Mat4::rotation_y(std::f64::consts::FRAC_PI_2),
        )
        .unwrap();
        crate::transform::transform_solid(&mut topo, c2, &Mat4::translation(-10.0, 0.0, 0.0))
            .unwrap();
        let res =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, c1, c2).unwrap();
        (topo, res)
    }

    #[test]
    fn steinmetz_lens_fuse_closed_form_volume() {
        let (topo, res) = steinmetz_fuse_census();
        // The gate fires, and the closed form gives the EXACT volume:
        // V = π·9·(20+20) − (16/3)·27 = 1130.97 − 144 = 986.97.
        let faces = brepkit_topology::explorer::solid_faces(&topo, res).unwrap();
        assert!(
            solid_is_steinmetz_lens_fuse(&topo, &faces),
            "the perpendicular cyl∪cyl fuse must be detected as the lens fuse"
        );
        let v = steinmetz_lens_fuse_volume(&topo, &faces).expect("closed form");
        let expect = std::f64::consts::PI * 9.0 * 40.0 - 16.0 / 3.0 * 27.0;
        assert!(
            (v - expect).abs() < 1e-9,
            "closed-form lens volume {v} should equal {expect} (986.97)"
        );
        // The public `solid_volume` returns the same exact value.
        let vol = solid_volume(&topo, res, 0.01).unwrap();
        assert!(
            (vol - expect).abs() < 1e-6,
            "solid_volume {vol} should match closed form {expect}"
        );
    }

    #[test]
    fn steinmetz_gate_does_not_fire_on_plain_or_coaxial_cylinders() {
        use brepkit_math::mat::Mat4;
        // A plain cylinder (one wall, no holes) is NOT the lens fuse.
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, cyl).unwrap();
        assert!(
            !solid_is_steinmetz_lens_fuse(&topo, &faces),
            "a plain cylinder is not the lens fuse"
        );

        // A coaxial cyl∩cyl (two collinear cylinders, no mutually-trimmed lens
        // walls) is NOT the lens fuse.
        let mut topo2 = Topology::new();
        let a = crate::primitives::make_cylinder(&mut topo2, 5.0, 20.0).unwrap();
        let b = crate::primitives::make_cylinder(&mut topo2, 5.0, 20.0).unwrap();
        crate::transform::transform_solid(&mut topo2, b, &Mat4::translation(0.0, 0.0, 10.0))
            .unwrap();
        let inter = crate::boolean::boolean(&mut topo2, crate::boolean::BooleanOp::Intersect, a, b)
            .unwrap();
        let f2 = brepkit_topology::explorer::solid_faces(&topo2, inter).unwrap();
        assert!(
            !solid_is_steinmetz_lens_fuse(&topo2, &f2),
            "coaxial cyl∩cyl is not the lens fuse"
        );
    }

    #[test]
    fn cyl_perp_intersecting_predicate() {
        use brepkit_math::surfaces::CylindricalSurface;
        let cyl = |o: Point3, a: Vec3, r: f64| CylindricalSurface::new(o, a, r).unwrap();
        let z = cyl(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0);

        // The census config: z⊥x, axes meet at the origin, equal r → TRUE.
        let x = cyl(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 3.0);
        assert!(cylinders_perpendicular_and_intersecting(&z, &x));

        // Unequal radius → false.
        let x_big = cyl(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 4.0);
        assert!(!cylinders_perpendicular_and_intersecting(&z, &x_big));

        // Non-perpendicular (45°), intersecting, equal r → false (its
        // intersection is NOT 16r³/3, so the closed form would be wrong).
        let diag = cyl(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 1.0), 3.0);
        assert!(!cylinders_perpendicular_and_intersecting(&z, &diag));

        // Parallel-offset equal r (both along z) → false (not perpendicular).
        let z_off = cyl(Point3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0);
        assert!(!cylinders_perpendicular_and_intersecting(&z, &z_off));

        // Perpendicular but SKEW (x-axis shifted in y so its line never meets
        // the z-axis) → false (not intersecting).
        let x_skew = cyl(Point3::new(0.0, 5.0, 8.0), Vec3::new(1.0, 0.0, 0.0), 3.0);
        assert!(!cylinders_perpendicular_and_intersecting(&z, &x_skew));
    }
}
