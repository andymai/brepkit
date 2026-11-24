//! Volume, center of mass, and related computations for B-rep solids.

use brepkit_math::madd::Madd;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::tessellate;

use super::helpers::{
    angular_range_from_wire_arcs, collect_solid_vertex_points, compute_angular_range,
    on_torus_axis, planar_wire_signed_area2, torus_band_v_range, torus_sector_u_range,
    traversal_spans,
};

/// Whether a sphere face's outer wire lies on a single constant-`v` latitude
/// (the simple bored-quadric band) rather than a scalloped, varying-`v` collar
/// floor. Projects the outer wire's vertices to `(u, v)` and tests the `v`
/// spread.
fn sphere_outer_wire_constant_v(
    topo: &Topology,
    face_id: FaceId,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> bool {
    topo.face(face_id)
        .is_ok_and(|face| sphere_wire_constant_v(topo, face.outer_wire(), sphere))
}

/// Whether a wire on a sphere lies on a single constant-`v` latitude.
fn sphere_wire_constant_v(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> bool {
    let Ok(wire) = topo.wire(wire_id) else {
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
/// (a bored quadric whose outer wire varies in `v` and winds the sphere's
/// axis, e.g. a box ∩ sphere patch). A holed patch that does not wind the
/// axis (an octant with a corner cut from it) keeps the per-face path.
fn solid_has_scalloped_sphere_collar(topo: &Topology, solid: SolidId) -> bool {
    let Ok(faces) = brepkit_topology::explorer::solid_faces(topo, solid) else {
        return false;
    };
    faces.iter().any(|&fid| {
        topo.face(fid).is_ok_and(|f| match f.surface() {
            FaceSurface::Sphere(s) => {
                !f.inner_wires().is_empty()
                    && !sphere_outer_wire_constant_v(topo, fid, s)
                    && sphere_wire_u_progress(topo, f.outer_wire(), s)
                        .is_ok_and(|p| p.abs() > std::f64::consts::PI)
            }
            _ => false,
        })
    })
}

/// Whether the solid has a torus notch band (torus − box: a kept toroidal patch
/// bounded by two tube-angle(`v`)-WRAPPING seam-arc loops, NOT constant-`v`
/// latitude circles). Its per-face analytic tessellation can't be sampled
/// watertight in isolation (the band wraps `v` and shares vertices with the
/// notch walls), and there is no closed-form volume — so the volume is taken off
/// the (watertight) whole-solid mesh instead, like the box ∩ sphere collar.
fn solid_has_torus_notch_band(topo: &Topology, solid: SolidId) -> bool {
    let Ok(faces) = brepkit_topology::explorer::solid_faces(topo, solid) else {
        return false;
    };
    faces.iter().any(|&fid| {
        topo.face(fid).is_ok_and(|f| match f.surface() {
            // Both of a notch band's wires wrap the tube; a sector whose one
            // hole is a lobe keeps the exact per-face flux.
            FaceSurface::Torus(t) => {
                f.inner_wires().len() == 1
                    && torus_wire_wraps_tube(topo, f.outer_wire(), t)
                    && torus_wire_wraps_tube(topo, f.inner_wires()[0], t)
            }
            _ => false,
        })
    })
}

/// True when a torus face wire's vertices span the full tube angle `v` (a
/// `v`-wrapping seam loop), as opposed to a constant-`v` latitude circle. Sampled
/// from the wire's edge endpoints projected to the torus `v` parameter.
fn torus_wire_wraps_tube(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    torus: &brepkit_math::surfaces::ToroidalSurface,
) -> bool {
    let Ok(wire) = topo.wire(wire_id) else {
        return false;
    };
    let mut vs: Vec<f64> = Vec::new();
    for oe in wire.edges() {
        let Ok(e) = topo.edge(oe.edge()) else {
            return false;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(e.start()), topo.vertex(e.end())) else {
            return false;
        };
        let (sp, ep) = (sv.point(), ev.point());
        // Sample ALONG each edge (not just endpoints): a wrapping seam arc bows
        // far in v between its endpoints, so endpoint-only sampling can miss the
        // wrap and misclassify a notch band as a constant-v circle.
        let (t0, t1) = e.curve().domain_with_endpoints(sp, ep);
        for k in 0..=8 {
            let f = f64::from(k) / 8.0;
            let p = e
                .curve()
                .evaluate_with_endpoints((t1 - t0).madd(f, t0), sp, ep);
            vs.push(torus.project_point(p).1);
        }
    }
    if vs.len() < 3 {
        return false;
    }
    vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Largest v-gap (incl. wrap); a wrapping loop's gap is well under a full turn,
    // a constant-v circle collapses to ~one v value (gap ≈ 2π).
    let max_gap = vs
        .windows(2)
        .map(|w| w[1] - w[0])
        .chain(std::iter::once(
            vs[0] + std::f64::consts::TAU - vs[vs.len() - 1],
        ))
        .fold(0.0_f64, f64::max);
    max_gap < std::f64::consts::PI
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

/// The box around a solid's vertices, `None` for a solid without any.
fn vertex_box(topo: &Topology, solid: SolidId) -> Option<(Point3, Point3)> {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for f in brepkit_topology::explorer::solid_faces(topo, solid).ok()? {
        let Ok(face) = topo.face(f) else { continue };
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let Ok(wire) = topo.wire(wid) else { continue };
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                let Ok(v) = topo.vertex(edge.start()) else {
                    continue;
                };
                let p = v.point();
                for (k, c) in [p.x(), p.y(), p.z()].into_iter().enumerate() {
                    lo[k] = lo[k].min(c);
                    hi[k] = hi[k].max(c);
                }
            }
        }
    }
    (lo[0] <= hi[0]).then(|| {
        (
            Point3::new(lo[0], lo[1], lo[2]),
            Point3::new(hi[0], hi[1], hi[2]),
        )
    })
}

/// The centre of a solid's vertex box, or the origin for a solid without
/// vertices.
fn solid_centre(topo: &Topology, solid: SolidId) -> Point3 {
    vertex_box(topo, solid).map_or(Point3::new(0.0, 0.0, 0.0), |(lo, hi)| lo + (hi - lo) * 0.5)
}

/// The point the volume paths sum each face's flux about. A closed surface's
/// total does not depend on it, but a meshed face can leave slivers open
/// against its exact neighbours and a faulty solid's faces need not close, so
/// near the origin it stays the origin, as it always has; far from it, the
/// solid's centre, without which the far-off coordinates cancel the volume
/// away.
fn sum_anchor(topo: &Topology, solid: SolidId) -> Point3 {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let Some((lo, hi)) = vertex_box(topo, solid) else {
        return origin;
    };
    let centre = lo + (hi - lo) * 0.5;
    let reach = (hi - lo).length() * 0.5;
    if (centre - origin).length() <= 10.0 * reach.max(1.0) {
        origin
    } else {
        centre
    }
}

/// Exact volume for a solid bounded entirely by planar faces with straight
/// (`Line`) edges — boxes, box booleans, extruded polygons. On a plane the
/// divergence integrand `x·n` is the constant plane offset, so the integral
/// collapses to `d·A` per face with `A` the exact polygon area: no
/// tessellation, no quadrature. Orientation comes from the face's effective
/// normal alone (`|outer| − Σ|holes|` per face), matching the tessellation
/// path's tolerance of mis-wound wires.
fn all_planar_line_solid_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::explorer::solid_faces;

    let faces = solid_faces(topo, solid).ok()?;
    if faces.is_empty() {
        return None;
    }
    let about = sum_anchor(topo, solid) - Point3::new(0.0, 0.0, 0.0);
    let mut vol6 = 0.0;
    for &fid in &faces {
        let face = topo.face(fid).ok()?;
        let FaceSurface::Plane { normal, d } = face.surface() else {
            return None;
        };
        let n_unit = normal.normalize().ok()?;
        let d_unit = d / normal.length();
        let sign = if face.is_reversed() { -1.0 } else { 1.0 };
        // A skewed quad stored as a Plane (miter-sweep elbows) has vertices
        // off the stored plane; x·n is then not constant and d·A is wrong.
        // Bail to the tessellation pipeline on any planarity residual.
        let planar_eps = 1e-6 * (1.0 + d_unit.abs());
        let ring_area2 = |wid| -> Option<f64> {
            let wire = topo.wire(wid).ok()?;
            let edges = wire.edges();
            if edges.len() < 3 {
                return None;
            }
            let mut area2 = Vec3::new(0.0, 0.0, 0.0);
            let mut first: Option<Vec3> = None;
            let mut prev: Option<Vec3> = None;
            for oe in edges {
                let e = topo.edge(oe.edge()).ok()?;
                if !matches!(e.curve(), EdgeCurve::Line) {
                    return None;
                }
                let vid = if oe.is_forward() { e.start() } else { e.end() };
                let p = topo.vertex(vid).ok()?.point();
                let p = Vec3::new(p.x(), p.y(), p.z());
                if (p.dot(n_unit) - d_unit).abs() > planar_eps {
                    return None;
                }
                // About the ring's first vertex: products of coordinates far
                // from the origin would cancel away the area.
                if let (Some(q), Some(f)) = (prev, first) {
                    area2 += (q - f).cross(p - f);
                } else {
                    first = Some(p);
                }
                prev = Some(p);
            }
            Some(area2.dot(n_unit).abs())
        };
        let mut face_area2 = ring_area2(face.outer_wire())?;
        for &iw in face.inner_wires() {
            face_area2 -= ring_area2(iw)?;
        }
        if face_area2 < 0.0 {
            return None;
        }
        vol6 += sign * (d_unit - n_unit.dot(about)) * face_area2;
    }
    Some((vol6 / 6.0).abs())
}

/// Volume of the Steinmetz lens fuse in closed form, or of a solid with a
/// bored sphere (a sphere face whose wires are all latitudes, such as a drilled
/// tunnel's rims) through the check crate's per-face integrator.
///
/// That integrator reads a sphere face along its wires, and a face its wire
/// path declines falls back to a band between latitudes, which is exact only
/// for latitude wires. So any other holed sphere, cylinder, cone or torus face
/// returns `None` here and is left to
/// [`brepkit_check::properties::exact_solid_volume`], which declines rather
/// than approximating. `None` too for a NURBS face or a solid with no bored
/// sphere.
fn analytic_faces_solid_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use brepkit_topology::explorer::solid_faces;

    let faces = solid_faces(topo, solid).ok()?;
    if faces.is_empty() {
        return None;
    }

    let mut has_bored_quadric = false;
    for &fid in &faces {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Nurbs(_) => return None,
            FaceSurface::Sphere(s) if !face.inner_wires().is_empty() => {
                // A collar whose outer wire varies in v (a box ∩ sphere
                // patch) or a hole that is not a latitude (a drill's entry
                // loop) would be mis-read by the latitude-band fallback.
                if !sphere_outer_wire_constant_v(topo, fid, s)
                    || !face
                        .inner_wires()
                        .iter()
                        .all(|&wire| sphere_wire_constant_v(topo, wire, s))
                {
                    return None;
                }
                has_bored_quadric = true;
            }
            // The fallback for a holed cylinder, cone or torus face does not
            // subtract its holes.
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

    let opts = brepkit_check::properties::PropertiesOptions::default();
    brepkit_check::properties::solid_volume(topo, solid, &opts)
        .ok()
        .map(f64::abs)
}

/// Exact volume of a fully-analytic SURFACE-OF-REVOLUTION solid (cone / cylinder
/// / torus walls + concentric circular planar disc caps) via the per-face
/// divergence-theorem integrators — no tessellation, so it is immune to the
/// inscribed-mesh undercount.
///
/// Returns `None` (defer to tessellation) unless EVERY face fits the revolution
/// signature, so this does NOT fire for boolean results that merely have an
/// arc-bounded planar face (a rounded-rect cap, an arc-frame lip):
///   * no NURBS face; inner wires only on PLANAR caps (an annulus cap
///     subtracts its holes; a bored solid is left to
///     [`analytic_faces_solid_volume`] or
///     [`brepkit_check::properties::exact_solid_volume`]);
///   * at least one quadric wall (cylinder/cone/torus) — it is a revolution;
///   * every cylinder/cone/torus shares ONE axis line;
///   * every planar face is a circular disc/annulus/sector whose bounding
///     arc(s) are centred ON that shared axis, or, when every wall is a
///     cylinder, a face parallel to the axis (it meets the walls in rulings).
fn analytic_revolution_solid_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use brepkit_topology::explorer::solid_faces;

    let about = sum_anchor(topo, solid);

    let faces = solid_faces(topo, solid).ok()?;
    if faces.is_empty() {
        return None;
    }

    // Establish the shared revolution axis from the first quadric wall.
    let mut axis: Option<(Point3, Vec3)> = None;
    let mut has_wall = false;
    let (mut has_cone, mut has_torus) = (false, false);
    let axis_tol = 1e-7;

    let set_or_check_axis = |axis: &mut Option<(Point3, Vec3)>, o: Point3, d: Vec3| -> bool {
        let d = match d.normalize() {
            Ok(d) => d,
            Err(_) => return false,
        };
        match axis {
            None => {
                *axis = Some((o, d));
                true
            }
            Some((o0, d0)) => {
                // Same axis LINE: parallel directions and the origins' offset is
                // along the axis (no perpendicular component).
                if d0.cross(d).length() > 1e-6 {
                    return false;
                }
                let off = o - *o0;
                (off - *d0 * off.dot(*d0)).length() <= axis_tol * off.length().max(1.0)
            }
        }
    };

    for &fid in &faces {
        let face = topo.face(fid).ok()?;
        if !face.inner_wires().is_empty() {
            // Only a holed PLANAR cap is integrable here: an annulus cap (a
            // washer-style revolve, a coaxially bored tube) subtracts its holes
            // inside `planar_cap_signed_volume`, and the second pass still
            // requires every arc — inner wires included — to be centred on the
            // shared axis. A holed quadric wall has no hole-aware integrator.
            if !matches!(face.surface(), FaceSurface::Plane { .. }) {
                return None;
            }
        }
        match face.surface() {
            FaceSurface::Sphere(_) => return None,
            // NURBS faces are validated in the second pass (they must be the
            // degenerate on-axis band the revolve leaves when a profile touches
            // the axis); they need the axis, established here.
            FaceSurface::Nurbs(_) => {}
            FaceSurface::Cylinder(c) => {
                has_wall = true;
                if !set_or_check_axis(&mut axis, c.origin(), c.axis()) {
                    return None;
                }
            }
            FaceSurface::Cone(c) => {
                has_wall = true;
                has_cone = true;
                if !set_or_check_axis(&mut axis, c.apex(), c.axis()) {
                    return None;
                }
            }
            FaceSurface::Torus(t) => {
                has_wall = true;
                has_torus = true;
                if !set_or_check_axis(&mut axis, t.center(), t.z_axis()) {
                    return None;
                }
            }
            FaceSurface::Plane { .. } => {} // checked below, once the axis is known
        }
    }
    let (axis_o, axis_d) = axis?;
    if !has_wall {
        return None;
    }

    // Second pass (axis known): every planar face must be a circular
    // disc/annulus/sector centred on the shared axis or a side face along
    // cylinder walls' rulings, and every NURBS face must
    // be the degenerate on-axis band (zero radial extent). Cache each planar
    // cap's analytic volume here so the summation below reuses it rather than
    // re-traversing the wire and re-running arc recognition a second time.
    let mut cap_volumes: std::collections::HashMap<FaceId, f64> = std::collections::HashMap::new();
    let mut has_side_face = false;
    for &fid in &faces {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                let unit = normal.normalize().ok()?;
                let v = if unit.cross(axis_d).length() <= 1e-6 {
                    if !planar_face_arcs_centered_on_axis(topo, fid, axis_o, axis_d) {
                        return None;
                    }
                    // Must be analytically integrable (a circular-arc-bounded cap).
                    planar_cap_signed_volume(topo, fid, about).ok()??
                } else {
                    // A face parallel to the axis of cylinder walls (a flat on
                    // a shaft, the cut face of a half rod) meets them in
                    // rulings, and one through the axis meets cone walls in
                    // rulings through the apex, so the walls stay rectangles
                    // in (u, v).
                    let o = axis_o - Point3::new(0.0, 0.0, 0.0);
                    let off_axis = (unit.dot(o) - d / normal.length()).abs();
                    if unit.dot(axis_d).abs() > 1e-9
                        || has_torus
                        || (has_cone && off_axis > 1e-7_f64.max(1e-12 * o.length()))
                    {
                        return None;
                    }
                    has_side_face = true;
                    planar_face_signed_volume(topo, fid, about).ok()??.0
                };
                cap_volumes.insert(fid, v);
            }
            FaceSurface::Nurbs(_) if !nurbs_band_is_on_axis(topo, fid, axis_o, axis_d) => {
                return None;
            }
            _ => {}
        }
    }

    // A side face that stops short of a wall's full height leaves that wall
    // an L in (u, v), which the rectangle integrator would over-count.
    if has_side_face {
        for &fid in &faces {
            if matches!(
                topo.face(fid).ok()?.surface(),
                FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
            ) && !ruled_wall_is_rectangle(topo, fid)?
            {
                return None;
            }
        }
    }

    // All faces fit — sum the exact per-face divergence-theorem contributions
    // (quadric walls + analytic planar disc caps; the on-axis NURBS bands are
    // zero-area and contribute nothing). No tessellation occurs.
    let mut total = 0.0;
    for &fid in &faces {
        let face = topo.face(fid).ok()?;
        let c = match face.surface() {
            FaceSurface::Cylinder(_) => analytic_cylinder_signed_volume(topo, fid, about).ok()?,
            FaceSurface::Cone(_) => analytic_cone_signed_volume(topo, fid, about).ok()?,
            FaceSurface::Torus(_) => analytic_torus_signed_volume(topo, fid, about).ok()??,
            FaceSurface::Plane { .. } => *cap_volumes.get(&fid)?,
            FaceSurface::Nurbs(_) => 0.0, // degenerate on-axis band
            FaceSurface::Sphere(_) => return None,
        };
        total += c;
    }
    Some(total.abs())
}

/// Whether a cylinder or cone face is a rectangle in `(u, v)`: every line a
/// ruling (along the axis, or through a cone's apex) running its full height,
/// every other edge a rim arc at its bottom or top spanning at most a half
/// turn (the cone's angular-range reader takes an arc's shorter side, and a
/// cylinder is held to the same bound).
fn ruled_wall_is_rectangle(topo: &Topology, face_id: FaceId) -> Option<bool> {
    use brepkit_topology::edge::EdgeCurve;
    let face = topo.face(face_id).ok()?;
    let (axis, apex) = match face.surface() {
        FaceSurface::Cylinder(cyl) => (cyl.axis(), None),
        FaceSurface::Cone(cone) => (cone.axis(), Some(cone.apex())),
        _ => return Some(false),
    };
    let surface = face.surface();
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut ends = Vec::with_capacity(wire.edges().len());
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let (start, end) = (
            topo.vertex(edge.start()).ok()?.point(),
            topo.vertex(edge.end()).ok()?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        if matches!(edge.curve(), EdgeCurve::Circle(_))
            && (t1 - t0).abs() > std::f64::consts::PI + 1e-9
        {
            return Some(false);
        }
        if matches!(edge.curve(), EdgeCurve::Line) {
            let along = match apex {
                None => axis,
                Some(apex) if (start - apex).length() > (end - apex).length() => start - apex,
                Some(apex) => end - apex,
            };
            let off = (end - start).cross(along).length() / along.length().max(1e-300);
            if off > 1e-9 * (end - start).length().max(1.0) {
                return Some(false);
            }
        }
        let (Some((_, v_start)), Some((_, v_end))) =
            (surface.project_point(start), surface.project_point(end))
        else {
            return None;
        };
        ends.push((edge.curve().clone(), v_start, v_end));
    }
    let v_min = ends
        .iter()
        .map(|e| e.1.min(e.2))
        .fold(f64::INFINITY, f64::min);
    let v_max = ends
        .iter()
        .map(|e| e.1.max(e.2))
        .fold(f64::NEG_INFINITY, f64::max);
    let tol = 1e-9 * (v_max - v_min).abs().max(1.0);
    let at = |v: f64, level: f64| (v - level).abs() <= tol;
    Some(ends.iter().all(|(curve, a, b)| match curve {
        EdgeCurve::Line => (at(*a, v_min) && at(*b, v_max)) || (at(*a, v_max) && at(*b, v_min)),
        EdgeCurve::Circle(c) => {
            c.normal().cross(axis).length() <= 1e-9
                && ((at(*a, v_min) && at(*b, v_min)) || (at(*a, v_max) && at(*b, v_max)))
        }
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
    }))
}

/// Whether a NURBS face is a degenerate revolution band on the axis — all its
/// boundary vertices lie on the axis line (zero radial extent), so it bounds no
/// volume. This is the band a revolve leaves when the profile touches the axis.
fn nurbs_band_is_on_axis(topo: &Topology, face_id: FaceId, axis_o: Point3, axis_d: Vec3) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let tol = 1e-7;
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return false;
    };
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            return false;
        };
        for vid in [edge.start(), edge.end()] {
            let Ok(v) = topo.vertex(vid) else {
                return false;
            };
            let off = v.point() - axis_o;
            let radial = off - axis_d * off.dot(axis_d);
            if radial.length() > tol {
                return false;
            }
        }
    }
    true
}

/// Whether every circular-arc edge of a planar face is centred on the given axis
/// line — the test that distinguishes a revolution disc/annulus cap (arcs about
/// the axis) from an arbitrary arc-bounded planar face (e.g. a rounded rectangle,
/// whose corner arcs are centred at the corners, off the axis).
fn planar_face_arcs_centered_on_axis(
    topo: &Topology,
    face_id: FaceId,
    axis_o: Point3,
    axis_d: Vec3,
) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let tol = 1e-6;
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let Ok(wire) = topo.wire(wire_id) else {
            return false;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                return false;
            };
            let center = match edge.curve() {
                brepkit_topology::edge::EdgeCurve::Circle(c) => Some(c.center()),
                brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => {
                    let rtol = brepkit_math::tolerance::Tolerance::default().linear * 100.0;
                    match brepkit_geometry::convert::recognize_curve(nc, rtol) {
                        brepkit_geometry::convert::RecognizedCurve::Circle { center, .. } => {
                            Some(center)
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(center) = center {
                // Distance from the arc centre to the axis line must be ~0.
                let off = center - axis_o;
                let perp = off - axis_d * off.dot(axis_d);
                if perp.length() > tol * off.length().max(1.0) {
                    return false;
                }
            }
        }
    }
    true
}

/// A ball less caps cut off by planes, each plane face a full disc and no two
/// caps meeting (a shell holding two separate pieces of one ball has full
/// discs whose caps overlap). About the centre, a sphere face adds `r/3` of
/// its area and
/// a disc `d/3` of its own, `d` the centre's distance to the disc's plane
/// along its outward normal; a cap `r - d` high takes `2 pi r (r - d)` of the
/// sphere.
fn ball_less_caps_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use std::f64::consts::PI;

    let solid_data = topo.solid(solid).ok()?;
    if !solid_data.inner_shells().is_empty() {
        return None;
    }
    let mut ball: Option<(Point3, f64)> = None;
    // A piece of a ball faces out of the sphere, so its sphere faces carry
    // the shell's orientation: all reversed means the whole shell is.
    let mut inverted: Option<bool> = None;
    // Each disc's outward unit normal and its plane's offset along it.
    let mut discs: Vec<(Vec3, f64)> = Vec::new();
    let mut on_sphere: Vec<Point3> = Vec::new();
    for &fid in topo.shell(solid_data.outer_shell()).ok()?.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Sphere(s) => {
                if *inverted.get_or_insert_with(|| face.is_reversed()) != face.is_reversed() {
                    return None;
                }
                if let Some((c, r)) = ball {
                    if (s.center() - c).length() > 1e-9 * r || (s.radius() - r).abs() > 1e-9 * r {
                        return None;
                    }
                } else {
                    ball = Some((s.center(), s.radius()));
                }
                let wires =
                    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
                for wid in wires {
                    for oe in topo.wire(wid).ok()?.edges() {
                        let edge = topo.edge(oe.edge()).ok()?;
                        on_sphere.push(topo.vertex(edge.start()).ok()?.point());
                    }
                }
            }
            FaceSurface::Plane { normal, d } => {
                let wire = topo.wire(face.outer_wire()).ok()?;
                let [oe] = wire.edges() else {
                    return None;
                };
                let edge = topo.edge(oe.edge()).ok()?;
                if !face.inner_wires().is_empty()
                    || edge.start() != edge.end()
                    || !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
                {
                    return None;
                }
                let len = normal.length();
                let sign = if face.is_reversed() { -1.0 } else { 1.0 };
                discs.push((*normal * (sign / len), sign * d / len));
                on_sphere.push(topo.vertex(edge.start()).ok()?.point());
            }
            _ => return None,
        }
    }
    let (center, r) = ball?;
    if inverted == Some(true) {
        for disc in &mut discs {
            *disc = (-disc.0, -disc.1);
        }
    }
    // A non-uniform scale moves the vertices off the stored sphere.
    if discs.is_empty()
        || on_sphere
            .iter()
            .any(|p| ((*p - center).length() - r).abs() > 1e-6 * r)
    {
        return None;
    }
    let origin = Vec3::new(center.x(), center.y(), center.z());
    let caps: Vec<(Vec3, f64)> = discs
        .into_iter()
        .map(|(n, offset)| (n, offset - n.dot(origin)))
        .collect();
    if caps.iter().any(|&(_, d)| d.abs() >= r) {
        return None;
    }
    // Each cap spans `acos(d / r)` about its normal; two are apart when their
    // normals are further apart than the sum.
    for (i, &(ni, di)) in caps.iter().enumerate() {
        for &(nj, dj) in &caps[i + 1..] {
            let apart = ni.dot(nj).clamp(-1.0, 1.0).acos();
            if apart < (di / r).acos() + (dj / r).acos() - 1e-9 {
                return None;
            }
        }
    }
    let mut sphere_area = 4.0 * PI * r * r;
    let mut flux = 0.0;
    for (_, d) in caps {
        sphere_area -= 2.0 * PI * r * (r - d);
        flux += d * PI * (r * r - d * d);
    }
    Some((r * sphere_area + flux) / 3.0)
}

/// Try to compute the volume of a solid analytically by detecting known
/// primitive shapes (sphere, cylinder, cone/frustum, torus).
///
/// Returns `None` if the solid is not a recognized pure primitive, in which
/// case the caller should fall back to tessellation.
///
/// Detection rules (single pass over shell faces):
/// - Any `Nurbs` face -> `None` (fall back)
/// - All faces on one `Sphere` (one centre, one radius) -> sphere formula `(4/3)pi*r^3`
/// - Exactly 1 `Cylinder` + >=1 `Plane` caps, 0 other analytic -> `pi*r^2*h`
/// - Exactly 1 `Cone` + <=2 `Plane` caps, 0 other analytic -> cone/frustum formula
///   (cap radii are read from the `Circle3D` edges of the cap faces)
/// - Exactly 1 `Torus` + 0 planes, 0 other analytic -> `2*pi^2*R*r^2`
#[allow(clippy::too_many_lines)]
fn try_analytic_solid_volume(topo: &Topology, solid: SolidId) -> Option<f64> {
    use std::f64::consts::PI;

    let solid_data = topo.solid(solid).ok()?;
    // A closed-form primitive formula describes the outer shell alone, so a
    // solid carrying a cavity has to go the long way round.
    if !solid_data.inner_shells().is_empty() {
        return None;
    }
    let shell = topo.shell(solid_data.outer_shell()).ok()?;

    let mut sphere_r: Option<f64> = None;
    let mut sphere_center: Option<Point3> = None;
    let mut cyl: Option<(Point3, Vec3, f64)> = None; // (origin, axis, radius)
    let mut cone_params: Option<(Point3, Vec3)> = None; // (apex, axis)
    let mut torus_params: Option<(f64, f64)> = None; // (major_r, minor_r)
    let mut torus_face_id: Option<FaceId> = None;
    let mut planes: Vec<(Vec3, f64)> = Vec::new();
    let mut plane_face_ids: Vec<FaceId> = Vec::new();

    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        // A holed analytic face means the solid is bored/pocketed; the closed-form
        // primitive volumes below integrate the surface as if the hole were filled.
        // Defer the whole solid to the hole-aware paths the caller tries after this
        // returns `None`.
        if !face.inner_wires().is_empty() {
            return None;
        }
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
                // ...and the same centre: two balls of one radius (touching,
                // or a lens of two) are no one ball.
                match sphere_center {
                    None => sphere_center = Some(s.center()),
                    Some(c) if (s.center() - c).length() > r * 1e-9 => return None,
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
                torus_face_id = Some(fid);
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
        // A cap bounded only by circles and ellipses crosses the whole wall,
        // however far it tilts, and the solid between two such caps holds
        // pi r^2 times their separation along the axis (the height over the
        // disc is linear, so its mean is the height at the axis).
        let full_caps = plane_face_ids.iter().all(|&fid| {
            topo.face(fid)
                .and_then(|face| topo.wire(face.outer_wire()))
                .is_ok_and(|wire| {
                    wire.edges().iter().all(|oe| {
                        topo.edge(oe.edge()).is_ok_and(|e| {
                            matches!(
                                e.curve(),
                                brepkit_topology::edge::EdgeCurve::Circle(_)
                                    | brepkit_topology::edge::EdgeCurve::Ellipse(_)
                            )
                        })
                    })
                })
        });
        let mut ts = if full_caps {
            planes
                .iter()
                .filter_map(|&(n, d)| {
                    let nd = n.dot(axis);
                    (nd.abs() > 1e-9).then(|| (d - n.dot(origin_vec)) / nd)
                })
                .collect()
        } else {
            cap_t_values(origin_vec, axis, &planes)
        };
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

        // A cap along one closed ellipse is a whole tilted section of the
        // cone, and the cone between the apex and that plane holds a third of
        // the section's area times the apex's distance to the plane.
        if cap_circles.len() != plane_face_ids.len() {
            let tips = plane_face_ids
                .iter()
                .zip(&planes)
                .map(|(&fid, &(n, d))| {
                    let area = cap_section_area(topo, fid)?;
                    Some(area * (n.dot(apex_vec) - d).abs() / n.length() / 3.0)
                })
                .collect::<Option<Vec<f64>>>()?;
            return match tips.as_slice() {
                [tip] => Some(*tip),
                [a, b] => Some((a - b).abs()),
                _ => None,
            };
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
    {
        if planes.is_empty() {
            return Some(2.0 * PI * PI * r_major * r_minor * r_minor);
        }
        // A partial-revolve torus sector: one trimmed band + two planar disc
        // caps that both contain the torus axis.
        if planes.len() == 2
            && let Some(tid) = torus_face_id
            && let Some(v) =
                partial_torus_sector_volume(topo, tid, r_major, r_minor, &planes, &plane_face_ids)
        {
            return Some(v);
        }
        return None;
    }

    None
}

/// Exact volume of a partial-revolve torus sector: one `Torus` band trimmed to
/// a sweep `Δu` plus two planar disc caps whose planes contain the torus axis,
/// each bounded by a single closed tube circle (radius = minor, centre at
/// major distance from the axis). `V = (Δu / 2π) · 2π²·R·r² = π·R·r²·Δu`.
///
/// `Δu` is read from the band's axis-centred seam arc, whose CCW start→end
/// span is the swept angle by the codebase-wide arc convention, and
/// cross-checked against the dihedral between the cap planes. Returns `None`
/// (defer to tessellation) unless every guard pins the structure.
fn partial_torus_sector_volume(
    topo: &Topology,
    torus_face: FaceId,
    r_major: f64,
    r_minor: f64,
    planes: &[(Vec3, f64)],
    plane_face_ids: &[FaceId],
) -> Option<f64> {
    use std::f64::consts::{PI, TAU};

    let face = topo.face(torus_face).ok()?;
    let FaceSurface::Torus(t) = face.surface() else {
        return None;
    };
    let center = t.center();
    let center_vec = Vec3::new(center.x(), center.y(), center.z());
    let axis_d = t.z_axis().normalize().ok()?;
    let tol = 1e-7 * r_major.max(1.0);

    // Both cap planes must contain the torus axis line.
    for &(n, d) in planes {
        let n_unit = n.normalize().ok()?;
        if n_unit.dot(axis_d).abs() > 1e-9 {
            return None;
        }
        if (n.dot(center_vec) - d).abs() > tol {
            return None;
        }
    }

    // Each cap must be a single closed tube circle: radius = minor, centre at
    // major distance from the axis.
    for &fid in plane_face_ids {
        let cap = topo.face(fid).ok()?;
        if !cap.inner_wires().is_empty() {
            return None;
        }
        let wire = topo.wire(cap.outer_wire()).ok()?;
        let edges = wire.edges();
        if edges.len() != 1 {
            return None;
        }
        let edge = topo.edge(edges[0].edge()).ok()?;
        if edge.start() != edge.end() {
            return None;
        }
        let brepkit_topology::edge::EdgeCurve::Circle(c) = edge.curve() else {
            return None;
        };
        if (c.radius() - r_minor).abs() > tol {
            return None;
        }
        let off = c.center() - center;
        let perp = off - axis_d * off.dot(axis_d);
        if (perp.length() - r_major).abs() > tol {
            return None;
        }
    }

    // The band's seam: non-closed Circle edges centred ON the axis, run up
    // and back. Their CCW start→end spans add up to the swept angle.
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut sweep: Option<f64> = None;
    let mut seen: Vec<brepkit_topology::edge::EdgeId> = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        if edge.start() == edge.end() || seen.contains(&oe.edge()) {
            continue;
        }
        seen.push(oe.edge());
        let brepkit_topology::edge::EdgeCurve::Circle(c) = edge.curve() else {
            continue;
        };
        let off = c.center() - center;
        let perp = off - axis_d * off.dot(axis_d);
        // The seam must be centred on the axis AND wound about +axis — an
        // antiparallel normal would make the CCW span read the complement.
        if perp.length() > tol
            || c.normal().cross(axis_d).length() > 1e-9
            || c.normal().dot(axis_d) < 0.0
        {
            continue;
        }
        let sp = topo.vertex(edge.start()).ok()?.point();
        let ep = topo.vertex(edge.end()).ok()?.point();
        let delta = (c.project(ep) - c.project(sp)).rem_euclid(TAU);
        sweep = Some(sweep.unwrap_or(0.0) + delta);
    }
    let du = sweep?;
    if !(1e-12..=TAU - 1e-12).contains(&du) {
        return None;
    }

    // Cross-check: the outward cap normals of a Δu sector satisfy
    // n₀·n₁ = −cos(Δu) (n₀ = −t̂, n₁ = t̂ rotated by Δu about the axis).
    let n0 = planes[0].0.normalize().ok()?;
    let n1 = planes[1].0.normalize().ok()?;
    if (n0.dot(n1) + du.cos()).abs() > 1e-6 {
        return None;
    }

    Some(PI * r_major * r_minor * r_minor * du)
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

/// Area of the conic a planar cap runs along, when every edge of its outer
/// wire is an arc of one circle or ellipse.
fn cap_section_area(topo: &Topology, face_id: FaceId) -> Option<f64> {
    use std::f64::consts::PI;

    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut area: Option<f64> = None;
    for oe in wire.edges() {
        let a = match topo.edge(oe.edge()).ok()?.curve() {
            brepkit_topology::edge::EdgeCurve::Circle(c) => PI * c.radius() * c.radius(),
            brepkit_topology::edge::EdgeCurve::Ellipse(e) => PI * e.semi_major() * e.semi_minor(),
            brepkit_topology::edge::EdgeCurve::Line
            | brepkit_topology::edge::EdgeCurve::NurbsCurve(_) => return None,
        };
        match area {
            Some(prev) if (prev - a).abs() > 1e-9 * prev => return None,
            _ => area = Some(a),
        }
    }
    area
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

/// Whether `BK_VOL_TRACE` diagnostics are enabled, checked once per process.
fn vol_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("BK_VOL_TRACE").is_ok())
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
    if let Some(v) = ball_less_caps_volume(topo, solid) {
        if vol_trace_enabled() {
            log::debug!("VOL_TRACE ball_less_caps -> {v}");
        }
        return Ok(v);
    }

    // Fast path: exact analytic formula for known primitives.
    if let Some(v) = try_analytic_solid_volume(topo, solid) {
        if vol_trace_enabled() {
            log::debug!("VOL_TRACE try_analytic -> {v}");
        }
        return Ok(v);
    }

    // Fast path: all faces planar with straight edges — exact polygon
    // divergence integral, no tessellation.
    if let Some(v) = all_planar_line_solid_volume(topo, solid) {
        if vol_trace_enabled() {
            log::debug!("VOL_TRACE all_planar_line -> {v}");
        }
        return Ok(v);
    }

    // The Steinmetz lens fuse and bored spheres (a cylinder drilled through a
    // ball), where the tessellation paths below fill a removed polar cap.
    if let Some(v) = analytic_faces_solid_volume(topo, solid) {
        if vol_trace_enabled() {
            log::debug!("VOL_TRACE analytic_faces -> {v}");
        }
        return Ok(v);
    }

    // A surface-of-revolution solid that is fully analytic (cone / cylinder /
    // torus walls with concentric circular disc-cap planes, no NURBS) integrates
    // with NO tessellation — exact, immune to the inscribed-mesh undercount. The
    // recogniser is deliberately narrow (concentric caps about one axis) so it
    // does NOT catch boolean results that merely happen to have arc-bounded
    // planar faces (rounded-rect caps, arc-frame lips).
    if let Some(v) = analytic_revolution_solid_volume(topo, solid) {
        if vol_trace_enabled() {
            log::debug!("VOL_TRACE revolution -> {v}");
        }
        return Ok(v);
    }

    // A solid of planes, cylinders, cones and spheres integrates exactly
    // along its faces' own boundaries, with no mesh to under-count its
    // curved faces or to mis-trim a face whose rim winds its pole.
    if let Some(v) = brepkit_check::properties::exact_solid_volume(topo, solid)? {
        if vol_trace_enabled() {
            log::debug!("VOL_TRACE exact boundary -> {v}");
        }
        return Ok(v.abs());
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
    // that closed mesh. Not with NURBS faces: they are integrated on their
    // surfaces below, and their inscribed mesh reads the solid short.
    let has_nurbs = brepkit_topology::explorer::solid_faces(topo, solid)?
        .into_iter()
        .any(|f| {
            topo.face(f)
                .is_ok_and(|f| matches!(f.surface(), FaceSurface::Nurbs(_)))
        });
    if !has_nurbs && solid_has_scalloped_sphere_collar(topo, solid) {
        let mesh = tessellate::tessellate_solid(topo, solid, deflection)?;
        if !mesh.indices.is_empty() && mesh_boundary_edge_count(&mesh) == 0 {
            if vol_trace_enabled() {
                log::debug!("VOL_TRACE scalloped sphere collar mesh");
            }
            return Ok(signed_volume_from_mesh(&mesh));
        }
        // Non-watertight mesh: fall through to the generic paths below rather
        // than return a leaky volume.
    }

    // A torus notch band (torus − box) likewise has no closed-form volume and
    // can't be per-face tessellated watertight (the band wraps the tube and
    // shares vertices with the notch walls). The whole-solid mesh IS watertight
    // (the structured notch-band tessellator), so take the divergence-theorem
    // volume off it. Per-face summation would under-count (the band's own per-
    // face mesh isn't closed).
    if !has_nurbs && solid_has_torus_notch_band(topo, solid) {
        let mesh = tessellate::tessellate_solid(topo, solid, deflection)?;
        if !mesh.indices.is_empty() && mesh_boundary_edge_count(&mesh) == 0 {
            if vol_trace_enabled() {
                log::debug!("VOL_TRACE torus notch band mesh");
            }
            return Ok(signed_volume_from_mesh(&mesh));
        }
        // Non-watertight mesh: fall through rather than return a leaky volume.
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
    // So does a torus, cylinder or cone face trimmed by a free-form curve (a
    // wall's hyperbola or ellipse), whose flux follows exactly along its
    // boundary where a mesh would only inscribe it, a whole ring (around a
    // cavity, say), whose flux is its volume, a sphere face whose outer
    // loop winds none of its u (a rod's end cap), and a NURBS face, whose
    // flux is integrated on the surface rather than on its inscribed mesh.
    let needs_direct_tessellation = brepkit_topology::explorer::solid_faces(topo, solid)?
        .into_iter()
        .any(|fid| {
            topo.face(fid).is_ok_and(|f| {
                !f.inner_wires().is_empty()
                    || matches!(f.surface(), FaceSurface::Nurbs(_))
                    || (f.is_reversed() && !matches!(f.surface(), FaceSurface::Plane { .. }))
                    || (matches!(f.surface(), FaceSurface::Torus(_)) && is_whole_ring(topo, f))
                    || (matches!(
                        f.surface(),
                        FaceSurface::Torus(_) | FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
                    ) && has_free_form_boundary(topo, fid).unwrap_or(false))
                    || sphere_patch_outer_area(topo, fid).is_ok_and(|area| area.is_some())
            })
        });
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

/// Divergence-theorem volume of a solid WITHOUT the absolute value, so the sign
/// reports shell orientation: positive for an outward-oriented (material-inside)
/// solid, negative for an inverted one.
///
/// [`solid_volume`] deliberately reports a magnitude, which makes it blind to a
/// globally inverted shell — the defect class that drops a boolean to the mesh
/// fallback with no volume error to show for it.
///
/// # Errors
///
/// Returns an error if the solid cannot be tessellated.
pub fn oriented_solid_volume(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let mesh = tessellate::tessellate_solid(topo, solid, deflection)?;
    Ok(oriented_mesh_volume(&mesh))
}

/// Divergence-theorem volume of a mesh WITHOUT the absolute value: the value
/// [`oriented_solid_volume`] reads from the mesh of a solid.
#[must_use]
pub fn oriented_mesh_volume(mesh: &tessellate::TriangleMesh) -> f64 {
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    // A mesh may be open, and an open mesh's sum depends on the point it is
    // taken about: meshes stay comparable only about one fixed point.
    let p0 = Point3::new(0.0, 0.0, 0.0);
    let mut total = 0.0;
    for t in 0..idx.len() / 3 {
        let v0 = pos[idx[t * 3] as usize];
        let v1 = pos[idx[t * 3 + 1] as usize];
        let v2 = pos[idx[t * 3 + 2] as usize];
        let (a, b, c) = (v0 - p0, v1 - p0, v2 - p0);
        total += a.dot((b - a).cross(c - a));
    }
    total / 6.0
}

/// Compute signed volume from a watertight triangle mesh using
/// the divergence theorem (signed tetrahedra method).
fn signed_volume_from_mesh(mesh: &tessellate::TriangleMesh) -> f64 {
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    let tri_count = idx.len() / 3;

    // A mesh may be open, and an open mesh's sum depends on the point it is
    // taken about: meshes stay comparable only about one fixed point.
    let p0 = Point3::new(0.0, 0.0, 0.0);
    let mut total = 0.0;
    for t in 0..tri_count {
        let v0 = pos[idx[t * 3] as usize];
        let v1 = pos[idx[t * 3 + 1] as usize];
        let v2 = pos[idx[t * 3 + 2] as usize];

        let (a, b, c) = (v0 - p0, v1 - p0, v2 - p0);

        total += a.dot((b - a).cross(c - a));
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
    // Per-face meshes need not close up; see `oriented_solid_volume`.
    let p0 = Point3::new(0.0, 0.0, 0.0);
    let mut total: f64 = 0.0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let mesh = tessellate::tessellate(topo, fid, deflection)?;
        let idx = &mesh.indices;
        if vol_trace_enabled() {
            log::debug!("VOL_TRACE direct plane face {fid:?} tris={}", idx.len() / 3);
        }
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;

        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let (a, b, c) = (v0 - p0, v1 - p0, v2 - p0);

            total += a.dot((b - a).cross(c - a));
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

/// Divergence-theorem flux `(1/3) ∫ P·N dA` of a cylinder or cone face over
/// its trimmed domain, holes included, or `None` when a wire winds around the
/// axis (a seamless band), which Green's theorem in the unwrapped `(u, v)`
/// plane cannot close.
///
/// On both surfaces the flux density `g(u, v)` in the face's own `(u, v)` has
/// a closed-form u-antiderivative `G`, so Green's theorem turns the area
/// integral into `∮ G dv` along every wire. `u` is unwrapped continuously
/// along each wire, which is what makes a full band come out right: its seam
/// is walked up at `u + 2π` and down at `u`. Each wire is oriented by its own
/// signed `(u, v)` area rather than trusted to be stored counter-clockwise:
/// cavity walls flipped by a cut keep their wires, so a hole cut into one
/// later can run the same way as its outer wire.
///
/// A pointed cone's apex has no `u`. A wire through it is walked from the
/// apex and the unwrap restarts at every apex visit: each run between
/// visits starts and ends at `v = 0`, where `G` vanishes, so its flux does
/// not depend on which `2π` branch it lands on.
///
/// - Cylinder `O + r·n(u) + v·a`: `g = r (r + O·n(u)) / 3`, independent of v.
/// - Cone `A + v (cos α n(u) + sin α a)`:
///   `g = v cos α (sin α A·n(u) − cos α A·a) / 3`.
#[allow(clippy::too_many_lines)]
fn developable_face_flux(
    topo: &Topology,
    face_id: FaceId,
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    use std::f64::consts::{PI, TAU};

    let face = topo.face(face_id)?;
    let origin = about;
    let surface = face.surface().clone();
    let antiderivative: Box<dyn Fn(f64, f64) -> f64> = match &surface {
        FaceSurface::Cylinder(c) => {
            let (r, o) = (c.radius(), c.origin() - origin);
            let (ox, oy) = (o.dot(c.x_axis()), o.dot(c.y_axis()));
            Box::new(move |u: f64, _v: f64| r * (r * u + u.sin() * ox - u.cos() * oy) / 3.0)
        }
        FaceSurface::Cone(c) => {
            let (sin_a, cos_a) = c.half_angle().sin_cos();
            let a = c.apex() - origin;
            let (ax, ay, az) = (a.dot(c.x_axis()), a.dot(c.y_axis()), a.dot(c.axis()));
            Box::new(move |u: f64, v: f64| {
                v * cos_a * (sin_a * (u.sin() * ax - u.cos() * ay) - cos_a * az * u) / 3.0
            })
        }
        FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "developable_face_flux requires a cylinder or cone face".into(),
            });
        }
    };
    let apex = if let FaceSurface::Cone(c) = &surface {
        Some(c.apex())
    } else {
        None
    };
    let tol = brepkit_math::tolerance::Tolerance::new().linear;
    let at_apex = |p: Point3| apex.is_some_and(|a| (p - a).length() <= tol);
    let project = |p: Point3| -> (f64, f64) { surface.project_point(p).unwrap_or((0.0, 0.0)) };
    // Unwrap `u` against the previous sample; an apex sample has no `u` and
    // leaves the anchor unset, so the next sample picks its own branch.
    let unwrap = |prev: &mut Option<f64>, p: Point3, u: f64| -> f64 {
        if at_apex(p) {
            *prev = None;
            return u;
        }
        let u = prev.map_or(u, |pu| u - ((u - pu) / TAU).round() * TAU);
        *prev = Some(u);
        u
    };

    let mut flux = 0.0;
    for (wire_index, wire_id) in std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .enumerate()
    {
        let wire = topo.wire(wire_id)?;
        let mut edges = wire.edges().to_vec();
        let traversal_start = |oe: &brepkit_topology::wire::OrientedEdge| {
            topo.edge(oe.edge()).and_then(|e| {
                let vid = if oe.is_forward() { e.start() } else { e.end() };
                topo.vertex(vid)
                    .map(brepkit_topology::vertex::Vertex::point)
            })
        };
        let mut through_apex = false;
        if apex.is_some() {
            for k in 0..edges.len() {
                if at_apex(traversal_start(&edges[k])?) {
                    edges.rotate_left(k);
                    through_apex = true;
                    break;
                }
            }
        }

        let (mut wire_flux, mut wire_area) = (0.0, 0.0);
        let mut prev_u: Option<f64> = None;
        let mut first_u: Option<f64> = None;
        for oe in &edges {
            let edge = topo.edge(oe.edge())?;
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let at = |t: f64| edge.curve().evaluate_with_endpoints(t, sp, ep);
            // Midpoint Stieltjes sums of G and of u against dv at `n` steps.
            // A straight edge on either surface is a ruling (constant u), so
            // one step is exact; curved edges take two resolutions and a
            // Richardson step.
            let stieltjes = |(from, to): (f64, f64), n: usize, prev: &mut Option<f64>| {
                let (mut g_sum, mut u_sum) = (0.0, 0.0);
                #[allow(clippy::cast_precision_loss)]
                let step = (to - from) / n as f64;
                let p0 = at(from);
                let (u0, mut v_prev) = project(p0);
                unwrap(prev, p0, u0);
                for k in 0..n {
                    #[allow(clippy::cast_precision_loss)]
                    let tk = from + step * k as f64;
                    let pm = at(tk + 0.5 * step);
                    let (um, vm) = project(pm);
                    let um = unwrap(prev, pm, um);
                    let pn = at(tk + step);
                    let (un, vn) = project(pn);
                    unwrap(prev, pn, un);
                    g_sum += antiderivative(um, vm) * (vn - v_prev);
                    u_sum += um * (vn - v_prev);
                    v_prev = vn;
                }
                (g_sum, u_sum)
            };
            for span in traversal_spans(edge, oe.is_forward(), sp, ep) {
                if first_u.is_none() && !through_apex {
                    let p0 = at(span.0);
                    first_u = Some(unwrap(&mut prev_u, p0, project(p0).0));
                }
                if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                    let (g, a) = stieltjes(span, 1, &mut prev_u);
                    wire_flux += g;
                    wire_area += a;
                    continue;
                }
                // Richardson assumes a smooth integrand, which a NURBS edge
                // is only between its knots: each knot span gets its own.
                let pieces = knot_pieces(edge.curve(), span);
                let coarse = (128 / pieces.len()).max(16);
                for piece in pieces {
                    let mut coarse_prev = prev_u;
                    let (gc, ac) = stieltjes(piece, coarse, &mut coarse_prev);
                    let (gf, af) = stieltjes(piece, 2 * coarse, &mut prev_u);
                    wire_flux += (4.0 * gf - gc) / 3.0;
                    wire_area += (4.0 * af - ac) / 3.0;
                }
            }
        }
        if let (Some(start), Some(end)) = (first_u, prev_u)
            && (end - start).abs() > PI
        {
            return Ok(None);
        }
        // `wire_area` is `∮ u dv`: positive for a counter-clockwise loop.
        let region = if wire_area < 0.0 {
            -wire_flux
        } else {
            wire_flux
        };
        flux += if wire_index == 0 { region } else { -region };
    }
    Ok(Some(if face.is_reversed() { -flux } else { flux }))
}

/// `span` cut at the interior knots of a NURBS curve, in the span's own
/// direction; any other curve's span whole.
fn knot_pieces(curve: &brepkit_topology::edge::EdgeCurve, span: (f64, f64)) -> Vec<(f64, f64)> {
    let brepkit_topology::edge::EdgeCurve::NurbsCurve(nurbs) = curve else {
        return vec![span];
    };
    let (lo, hi) = (span.0.min(span.1), span.0.max(span.1));
    let eps = 1e-12 * (hi - lo).max(1.0);
    let mut cuts: Vec<f64> = nurbs
        .knots()
        .iter()
        .copied()
        .filter(|&k| k > lo + eps && k < hi - eps)
        .collect();
    cuts.dedup_by(|a, b| (*a - *b).abs() <= eps);
    if span.1 < span.0 {
        cuts.reverse();
    }
    let mut pieces = Vec::with_capacity(cuts.len() + 1);
    let mut from = span.0;
    for cut in cuts {
        pieces.push((from, cut));
        from = cut;
    }
    pieces.push((from, span.1));
    pieces
}

/// Divergence-theorem flux `(1/3) ∫ P·N dA` of a planar face: its plane's
/// offset along the outward normal times its area, over three. The area is
/// Green's `½ ∮ (x dy − y dx)` along every wire in the plane's frame, in
/// closed form on lines, circles and ellipses and by Gauss quadrature per knot
/// span on NURBS edges, so a curved boundary is never chorded. Holes subtract
/// by magnitude: a boolean can emit one wound like the outer wire. `None` for
/// a face that is not a plane or whose holes outweigh its outer wire.
fn planar_face_flux(
    topo: &Topology,
    face_id: FaceId,
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::FRAC_PI_2;

    let face = topo.face(face_id)?;
    let FaceSurface::Plane { normal, d } = face.surface() else {
        return Ok(None);
    };
    let (normal, d) = (*normal, *d - normal.dot(about - Point3::new(0.0, 0.0, 0.0)));
    let len = normal.length();
    let origin = Point3::new(0.0, 0.0, 0.0);
    let Ok(frame) = brepkit_math::frame::Frame3::from_normal(origin, normal) else {
        return Ok(None);
    };
    let (ex, ey) = (frame.x, frame.y);
    let flat = |v: Vec3| (v.dot(ex), v.dot(ey));
    let cross = |(ax, ay): (f64, f64), (bx, by): (f64, f64)| ax.madd(by, -(ay * bx));
    // `∫ (P − Q) × P′ dt` over `[t0, t1]` for `P(t) = c + A cos t + B sin t`.
    let conic = |c: Point3, a: Vec3, b: Vec3, t0: f64, t1: f64, q: Point3| {
        let (c, a, b) = (flat(c - q), flat(a), flat(b));
        cross(c, a).madd(
            t1.cos() - t0.cos(),
            cross(c, b).madd(t1.sin() - t0.sin(), cross(a, b) * (t1 - t0)),
        )
    };
    let gauss = brepkit_math::quadrature::gauss_legendre_points(16);
    // A closed loop's `∮ (P − Q) × dP` does not depend on Q; taking Q on the
    // loop keeps products of far-off coordinates from cancelling its area.
    let wire_area2 = |wire_id| -> Result<f64, crate::OperationsError> {
        let mut sum = 0.0;
        let wire = topo.wire(wire_id)?;
        let q = match wire.edges().first() {
            Some(oe) => topo.vertex(topo.edge(oe.edge())?.start())?.point(),
            None => origin,
        };
        let q_vec = q - origin;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            for (t0, t1) in traversal_spans(edge, oe.is_forward(), sp, ep) {
                sum += match edge.curve() {
                    EdgeCurve::Line => {
                        let at = |t: f64| edge.curve().evaluate_with_endpoints(t, sp, ep) - q;
                        cross(flat(at(t0)), flat(at(t1)))
                    }
                    EdgeCurve::Circle(c) => conic(
                        c.center(),
                        c.evaluate(0.0) - c.center(),
                        c.evaluate(FRAC_PI_2) - c.center(),
                        t0,
                        t1,
                        q,
                    ),
                    EdgeCurve::Ellipse(e) => conic(
                        e.center(),
                        e.evaluate(0.0) - e.center(),
                        e.evaluate(FRAC_PI_2) - e.center(),
                        t0,
                        t1,
                        q,
                    ),
                    EdgeCurve::NurbsCurve(n) => {
                        let (lo, hi) = (t0.min(t1), t0.max(t1));
                        let mut cuts: Vec<f64> = std::iter::once(lo)
                            .chain(n.knots().iter().copied().filter(|&k| k > lo && k < hi))
                            .chain(std::iter::once(hi))
                            .collect();
                        cuts.dedup();
                        let mut part = 0.0;
                        for w in cuts.windows(2) {
                            let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
                            for g in gauss {
                                let ders = n.derivatives(half.madd(g.x, mid), 1);
                                part += g.w * half * cross(flat(ders[0] - q_vec), flat(ders[1]));
                            }
                        }
                        if t1 < t0 { -part } else { part }
                    }
                };
            }
        }
        Ok(sum)
    };
    // The fast path reads x·n as the plane offset everywhere on the face, so
    // a boundary off its stored plane (a skewed miter-sweep quad) keeps the
    // mesh. So does a NURBS edge with a weight that is not positive: its
    // denominator can pass near zero, too sharp for the fixed quadrature,
    // and it can leave the control hull the nesting boxes rely on.
    let d_unit = d / len;
    let normal_unit = normal * (1.0 / len);
    let planar_eps = 1e-6 * (1.0 + d_unit.abs());
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oe.edge())?;
            if let EdgeCurve::NurbsCurve(n) = edge.curve()
                && n.weights().iter().any(|&w| w <= 0.0)
            {
                return Ok(None);
            }
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            for (t0, t1) in traversal_spans(edge, true, sp, ep) {
                for k in 0..=4 {
                    let t = t0 + (t1 - t0) * f64::from(k) / 4.0;
                    let p = edge.curve().evaluate_with_endpoints(t, sp, ep) - about;
                    if (p.dot(normal_unit) - d_unit).abs() > planar_eps {
                        return Ok(None);
                    }
                }
            }
        }
    }
    let mut area2 = wire_area2(face.outer_wire())?.abs();
    // Holes subtract only while no two can nest: an island inside a hole is
    // material again, which the mesher's odd-depth rule handles.
    if face.inner_wires().len() > 1 {
        let boxes = face
            .inner_wires()
            .iter()
            .map(|&iw| wire_box(topo, iw, &flat))
            .collect::<Result<Vec<_>, _>>()?;
        for (k, a) in boxes.iter().enumerate() {
            if boxes[k + 1..]
                .iter()
                .any(|b| a.0 <= b.2 && b.0 <= a.2 && a.1 <= b.3 && b.1 <= a.3)
            {
                return Ok(None);
            }
        }
    }
    for &iw in face.inner_wires() {
        area2 -= wire_area2(iw)?.abs();
    }
    if area2 < 0.0 || len <= 0.0 {
        return Ok(None);
    }
    let d_out = if face.is_reversed() { -d_unit } else { d_unit };
    let flux = d_out * area2 / 6.0;
    // A rational edge whose weights vanish on its span has a pole there.
    Ok(flux.is_finite().then_some(flux))
}

/// A `(min x, min y, max x, max y)` box of a wire in a plane's frame that
/// never falls inside the wire: line endpoints, a conic arc's endpoints and
/// every axis extreme inside its span, and a NURBS curve's control points
/// (the curve stays in their hull while its weights are positive).
fn wire_box(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    flat: &dyn Fn(Vec3) -> (f64, f64),
) -> Result<(f64, f64, f64, f64), crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    let origin = Point3::new(0.0, 0.0, 0.0);
    let mut b = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut include = |(x, y): (f64, f64)| {
        b = (b.0.min(x), b.1.min(y), b.2.max(x), b.3.max(y));
    };
    for oe in topo.wire(wire_id)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let frame = match edge.curve() {
            EdgeCurve::Line => {
                include(flat(sp - origin));
                include(flat(ep - origin));
                continue;
            }
            EdgeCurve::NurbsCurve(n) => {
                for &p in n.control_points() {
                    include(flat(p - origin));
                }
                continue;
            }
            EdgeCurve::Circle(c) => (
                c.center(),
                c.evaluate(0.0) - c.center(),
                c.evaluate(FRAC_PI_2) - c.center(),
            ),
            EdgeCurve::Ellipse(e) => (
                e.center(),
                e.evaluate(0.0) - e.center(),
                e.evaluate(FRAC_PI_2) - e.center(),
            ),
        };
        // `P(t) = c + A cos t + B sin t`: along x it peaks at
        // `atan2(Bx, Ax)` and bottoms out half a turn later; likewise y.
        let ((cx, cy), (ax, ay), (bx, by)) = (flat(frame.0 - origin), flat(frame.1), flat(frame.2));
        let at = |t: f64| {
            (
                cx + ax * t.cos() + bx * t.sin(),
                cy + ay * t.cos() + by * t.sin(),
            )
        };
        for (t0, t1) in traversal_spans(edge, true, sp, ep) {
            let (lo, hi) = (t0.min(t1), t0.max(t1));
            include(at(lo));
            include(at(hi));
            for extreme in [
                bx.atan2(ax),
                bx.atan2(ax) + PI,
                by.atan2(ay),
                by.atan2(ay) + PI,
            ] {
                #[allow(clippy::cast_possible_truncation)]
                let (first, last) = (
                    ((lo - extreme) / TAU).ceil() as i64,
                    ((hi - extreme) / TAU).floor() as i64,
                );
                for turn in first..=last {
                    #[allow(clippy::cast_precision_loss)]
                    include(at(extreme + turn as f64 * TAU));
                }
            }
        }
    }
    Ok(b)
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
    about: Point3,
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
            // Sample circle-edge midpoints (along the edge's own span) for
            // angular coverage.
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::Circle(_) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
                let (sp, ep) = (sv.point(), ev.point());
                let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
                let mid = edge
                    .curve()
                    .evaluate_with_endpoints(f64::midpoint(t0, t1), sp, ep);
                let (u, _) = cyl.project_point(mid);
                u_vals.push(u);
            }
            // A revolution-band boundary is a rational NURBS arc, not an
            // `EdgeCurve::Circle`. Sample its domain midpoint too, or a partial
            // (sub-2π) band has only its two endpoint angles, `compute_angular_range`
            // falls back to the full 2π, and the band over-counts (gh #968).
            // Use the endpoint-trimmed sub-span: a boolean-split edge shares its
            // parent curve, whose full-domain midpoint lies outside the edge and
            // widens the angular range (gh #1499).
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
                let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
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

    let u_range = angular_range_from_wire_arcs(topo, wire, |p| cyl.project_point(p).0)
        .unwrap_or_else(|| compute_angular_range(&mut u_vals));

    let r = cyl.radius();
    let x_axis = cyl.x_axis();
    let y_axis = cyl.y_axis();

    let o_vec = cyl.origin() - about;
    let ox = o_vec.dot(x_axis);
    let oy = o_vec.dot(y_axis);

    let (u1, u2) = u_range;
    let (sin1, cos1) = u1.sin_cos();
    let (sin2, cos2) = u2.sin_cos();

    if vol_trace_enabled() {
        log::debug!(
            "VOL_TRACE cyl {face_id:?} u_vals={u_vals:?} u_range=({u1},{u2}) h={h} r={r} ox={ox} oy={oy}"
        );
    }
    let vol = (r / 3.0) * h * (ox * (sin2 - sin1) + oy * (-cos2 + cos1) + r * (u2 - u1));

    Ok(if face.is_reversed() { -vol } else { vol })
}

/// Exact signed volume contribution of a PLANAR face whose boundary is made of
/// straight and circular-arc edges (a revolve cap: a disc, an annulus, or an
/// angular sector of either), via the divergence theorem.
///
/// For a planar face every point satisfies `P·n = d` (the plane offset), so the
/// volume integral `(1/3)∮∮ P·n dA` reduces to `(1/3)·d·A`, where `A` is the
/// signed area in the plane oriented by the face normal. `A` is computed
/// EXACTLY by Green's theorem — `A = (1/2)∮(x dy − y dx)` — summing each edge's
/// chord term plus the exact circular-segment bulge for arc edges (including a
/// full closed circle, whose 2π sweep gives the disc area πρ²), so a circular
/// boundary is not chorded (the reason the tessellation path under-counts it).
///
/// Returns `Ok(None)` when an edge is neither a line nor a circular arc (e.g. a
/// general spline boundary), so the caller falls back to tessellation.
fn planar_cap_signed_volume(
    topo: &Topology,
    face_id: FaceId,
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    // Only claim a circular CAP (disc / annulus / sector) — a face with no arc
    // edge is an ordinary polygon (e.g. a box face), which the tessellation path
    // already integrates exactly; deferring it keeps this analytic path scoped to
    // genuine revolve caps and out of arbitrary planar-faced solids.
    Ok(planar_face_signed_volume(topo, face_id, about)?
        .filter(|&(_, arcs)| arcs > 0)
        .map(|(volume, _)| volume))
}

/// The exact divergence-theorem contribution of a planar face bounded by
/// lines and circular arcs, with its arc-edge count. `Ok(None)` when an edge
/// is anything else.
fn planar_face_signed_volume(
    topo: &Topology,
    face_id: FaceId,
    about: Point3,
) -> Result<Option<(f64, usize)>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let FaceSurface::Plane { normal, d } = face.surface() else {
        return Ok(None);
    };
    let normal = *normal;
    let d = *d - normal.dot(about - Point3::new(0.0, 0.0, 0.0));

    // Right-handed in-plane frame: ex × ey = normal, so a boundary wound CCW as
    // seen from +normal yields a positive signed area.
    let frame = match brepkit_math::frame::Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), normal) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let (ex, ey) = (frame.x, frame.y);

    // Hole areas subtract by MAGNITUDE, not by trusting the stored hole-wire
    // winding: boolean results can emit an inner rim wound the SAME way as the
    // outer (the #1045 hole-winding class), which would ADD the hole's disc.
    // A hole is inside the outer by definition, so |outer| − Σ|hole| is the
    // geometric area either way.
    let (outer_area2, mut arc_edges) =
        match planar_wire_signed_area2(topo, face.outer_wire(), ex, ey)? {
            Some(r) => r,
            None => return Ok(None),
        };
    let mut area_mag2 = outer_area2.abs();
    for &iw in face.inner_wires() {
        let Some((hole_area2, hole_arcs)) = planar_wire_signed_area2(topo, iw, ex, ey)? else {
            return Ok(None);
        };
        arc_edges += hole_arcs;
        area_mag2 -= hole_area2.abs();
    }
    if area_mag2 < 0.0 {
        return Ok(None); // holes exceed the outer boundary — not a sane cap
    }

    // `area_mag2/2` is the geometric area. The divergence-theorem contribution
    // is (1/3)·(p·n̂_out)·|A|, where the outward normal is `+normal` for a
    // forward face and `−normal` for a reversed one, so `p·n̂_out = ±d` (the
    // plane offset along that outward normal). The sign therefore comes only
    // from the outward offset, not from the wire winding.
    let area = area_mag2 / 2.0;
    let d_out = if face.is_reversed() { -d } else { d };
    Ok(Some((d_out * area / 3.0, arc_edges)))
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
    about: Point3,
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
                    // The apex (v ≈ 0) lies on the axis where u is undefined;
                    // its arbitrary projected u corrupts the angular-range gap
                    // detection (a per-segment band touching the apex would read
                    // a 2× span). Keep its v for the v-range, but omit its u.
                    if v.abs() > 1e-9 {
                        u_vals.push(u);
                    }
                    v_vals.push(v);
                }
            }
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
                // The arc's own domain says which way round it runs: taken the
                // shorter way, the two halves of a split circle can both sample
                // one side, and the band reads half its turn or all of it.
                let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
                let mid = circle.evaluate(f64::midpoint(t0, t1));
                let (u, _) = cone.project_point(mid);
                u_vals.push(u);
            }
            // Sample NURBS revolution-band arcs too (see the cylinder case, #968).
            // Skip an arc that degenerates to the apex (v ≈ 0), where u is
            // undefined. Endpoint-trimmed sub-span for boolean-split edges
            // sharing a parent curve (gh #1499).
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
                let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
                let (u, v) = cone.project_point(nc.evaluate(f64::midpoint(t0, t1)));
                if v.abs() > 1e-9 {
                    u_vals.push(u);
                }
            }
        }
    }

    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (v_max - v_min).abs() < 1e-15 {
        return Ok(0.0);
    }

    // Read the turn off the wire's arcs as the cylinder does: a wall
    // bounded by two whole circles and a seam has only the seam's angle
    // among its vertices, and the samples alone read a part turn.
    let u_range = angular_range_from_wire_arcs(topo, wire, |p| cone.project_point(p).0)
        .unwrap_or_else(|| compute_angular_range(&mut u_vals));

    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let x_axis = cone.x_axis();
    let y_axis = cone.y_axis();
    let axis = cone.axis();
    let apex = cone.apex();
    let a_vec = apex - about;

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
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let sph = match face.surface() {
        FaceSurface::Sphere(s) => s,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_sphere_signed_volume requires a sphere face".into(),
            });
        }
    };
    // The closed form below spans the outer wire's (u, v) box, which is the
    // face only when that wire runs along latitudes, meridians and chords.
    if crate::tessellate::sphere_face_is_off_grid(topo, face, sph)? {
        return Ok(None);
    }

    let wire = topo.wire(face.outer_wire())?;
    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = sph.project_point(vtx.point());
                    // A pole's u is arbitrary: only its v bounds the box.
                    if v.abs() < std::f64::consts::FRAC_PI_2 - 1e-6 {
                        u_vals.push(u);
                    }
                    v_vals.push(v);
                }
            }
            if !edge.is_closed()
                && let brepkit_topology::edge::EdgeCurve::Circle(circle) = edge.curve()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
                // The arc's own domain says which way round it runs: taken the
                // shorter way, the two halves of a split circle can both sample
                // one side, and the band reads half its turn or all of it.
                let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
                let mid = circle.evaluate(f64::midpoint(t0, t1));
                let (u, _) = sph.project_point(mid);
                u_vals.push(u);
            }
        }
    }

    let mut v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mut v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // For sphere caps (single circle boundary at one latitude), the boundary
    // vertices all share approximately the same v, so v_max ~ v_min. The
    // wire runs counter-clockwise about the outward normal, so a loop turning
    // toward +u has the north pole on its left (the loop's centroid lies on
    // the axis, which says nothing about the side).
    if (v_max - v_min).abs() < 0.01 {
        let v_boundary = f64::midpoint(v_min, v_max);
        if sphere_wire_u_progress(topo, face.outer_wire(), sph)? > 0.0 {
            v_min = v_boundary;
            v_max = std::f64::consts::FRAC_PI_2;
        } else {
            v_min = -std::f64::consts::FRAC_PI_2;
            v_max = v_boundary;
        }
    }

    let u_range = compute_angular_range(&mut u_vals);
    // That box is the face only when each edge runs along one of its sides:
    // a latitude or meridian inside it (a notch's step) leaves the face
    // short of the box.
    if !sphere_wire_on_box_sides(topo, face.outer_wire(), sph, u_range, (v_min, v_max))? {
        return Ok(None);
    }

    let r = sph.radius();
    let x_axis = sph.x_axis();
    let y_axis = sph.y_axis();
    let z_axis = sph.z_axis();
    let c = sph.center();
    let c_vec = c - about;

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
    let mut vol = vol;
    for &wire in face.inner_wires() {
        // A hole left out would count its cap as material: the face's own
        // mesh measures it instead.
        let Some(hole) = sphere_hole_flux(topo, sph, wire, about)? else {
            return Ok(None);
        };
        vol -= hole;
    }

    Ok(Some(if face.is_reversed() { -vol } else { vol }))
}

/// The net turn in u a sphere wire makes along its traversal, pole samples
/// skipped.
fn sphere_wire_u_progress(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> Result<f64, crate::OperationsError> {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let wrap = |d: f64| (d + PI).rem_euclid(TAU) - PI;
    let (mut progress, mut u_last) = (0.0, None);
    for oe in topo.wire(wire_id)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        for (from, to) in traversal_spans(edge, oe.is_forward(), sp, ep) {
            for i in 0..=16 {
                let t = (to - from).madd(f64::from(i) / 16.0, from);
                let (u, v) = sphere.project_point(edge.curve().evaluate_with_endpoints(t, sp, ep));
                if v.abs() >= FRAC_PI_2 - 1e-6 {
                    continue;
                }
                if let Some(last) = u_last {
                    progress += wrap(u - last);
                }
                u_last = Some(u);
            }
        }
    }
    Ok(progress)
}

/// Whether a sphere wire bounds the `(u, v)` box `u_range` x `v_range`:
/// every edge runs at `v` equal to one of its ends (a latitude, or a chord of
/// the equator), or, when the box spans less than a turn, at `u` equal to one
/// of its ends (a meridian, whose u means nothing at a pole), or is a seam the
/// wire runs both ways; and a wire clear of the poles winds the axis exactly
/// when the box spans the turn (a notch's outline runs along a box's sides
/// too, but around the rest).
fn sphere_wire_on_box_sides(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    (u1, u2): (f64, f64),
    (v1, v2): (f64, f64),
) -> Result<bool, crate::OperationsError> {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let wrap = |d: f64| (d + PI).rem_euclid(TAU) - PI;
    let full_turn = u2 - u1 >= TAU - 1e-9;
    let (mut progress, mut u_last, mut at_pole) = (0.0, None, false);
    let edges = topo.wire(wire_id)?.edges();
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let mut samples: Vec<(f64, f64)> = Vec::new();
        for (from, to) in traversal_spans(edge, oe.is_forward(), sp, ep) {
            samples.extend((0..=8).map(|i| {
                let t = (to - from).madd(f64::from(i) / 8.0, from);
                sphere.project_point(edge.curve().evaluate_with_endpoints(t, sp, ep))
            }));
        }
        for &(u, v) in &samples {
            if v.abs() >= FRAC_PI_2 - 1e-6 {
                at_pole = true;
            } else {
                if let Some(last) = u_last {
                    progress += wrap(u - last);
                }
                u_last = Some(u);
            }
        }
        let at_v = |side: f64| samples.iter().all(|&(_, v)| (v - side).abs() < 1e-7);
        let seam = edges
            .iter()
            .any(|other| other.edge() == oe.edge() && other.is_forward() != oe.is_forward());
        if at_v(v1) || at_v(v2) || seam {
            continue;
        }
        let at_u = |side: f64| {
            samples
                .iter()
                .filter(|&&(_, v)| v.abs() < FRAC_PI_2 - 1e-6)
                .all(|&(u, _)| wrap(u - side).abs() < 1e-7)
        };
        if !full_turn && (at_u(u1) || at_u(u2)) {
            continue;
        }
        return Ok(false);
    }
    Ok(at_pole || (progress.abs() > PI) == full_turn)
}

/// Divergence flux `(1/3) ∫ (P − about)·N dA` of the region of a sphere that
/// a hole's loop, winding none of its u, cuts out (a drill's entry, a
/// pocket's bite through a pole): the smaller of the two regions the loop
/// bounds. `None` for a loop around the axis.
fn sphere_hole_flux(
    topo: &Topology,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    wire_id: brepkit_topology::wire::WireId,
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    Ok(sphere_hole_area_and_flux(topo, sphere, wire_id, about)?.map(|(_, flux)| flux))
}

/// A hole's area and flux, as [`sphere_hole_flux`] reads them.
fn sphere_hole_area_and_flux(
    topo: &Topology,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    wire_id: brepkit_topology::wire::WireId,
    about: Point3,
) -> Result<Option<(f64, f64)>, crate::OperationsError> {
    use std::f64::consts::PI;
    let radius = sphere.radius();
    Ok(
        sphere_loop_area_and_flux(topo, sphere, wire_id, about)?.map(|(area, flux)| {
            if area > 2.0 * PI * radius * radius {
                (
                    4.0 * PI * radius * radius - area,
                    4.0 / 3.0 * PI * radius.powi(3) - flux,
                )
            } else {
                (area, flux)
            }
        }),
    )
}

/// One of the two regions a loop winding none of a sphere's u bounds, as its
/// area and its flux under the sphere's outward normal, `R·area + (C −
/// about)·∫N dA` over three. The area is `∮ u w(v) dv`, free at the poles,
/// whose sign orients the loop; `∫N dA` is `½ ∮ (P − C) × dP` taken the same
/// way (midpoint sums at two resolutions and a Richardson step). The other
/// region has the rest of the area and the ball's volume less the flux.
/// `None` for a loop around the axis.
fn sphere_loop_area_and_flux(
    topo: &Topology,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    wire_id: brepkit_topology::wire::WireId,
    about: Point3,
) -> Result<Option<(f64, f64)>, crate::OperationsError> {
    let Some(signed) = super::area::sphere_wire_signed_area(topo, sphere, wire_id)? else {
        return Ok(None);
    };
    sphere_loop_flux(topo, sphere, wire_id, signed, about).map(Some)
}

/// [`sphere_loop_area_and_flux`] for a loop whose signed area is known.
fn sphere_loop_flux(
    topo: &Topology,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    wire_id: brepkit_topology::wire::WireId,
    signed: f64,
    about: Point3,
) -> Result<(f64, f64), crate::OperationsError> {
    let centre = sphere.center();
    let mut sweep = Vec3::new(0.0, 0.0, 0.0);
    for oe in topo.wire(wire_id)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let at = |t: f64| edge.curve().evaluate_with_endpoints(t, sp, ep);
        let sums = |(from, to): (f64, f64), n: usize| {
            let mut sweep = Vec3::new(0.0, 0.0, 0.0);
            #[allow(clippy::cast_precision_loss)]
            let step = (to - from) / n as f64;
            let mut p_prev = at(from);
            for k in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let tk = from + step * k as f64;
                let pm = at(tk + 0.5 * step);
                let pn = at(tk + step);
                sweep += (pm - centre).cross(pn - p_prev) * 0.5;
                p_prev = pn;
            }
            sweep
        };
        for span in traversal_spans(edge, oe.is_forward(), sp, ep) {
            sweep += (sums(span, 256) * 4.0 - sums(span, 128)) * (1.0 / 3.0);
        }
    }
    let (area, sweep) = if signed < 0.0 {
        (-signed, sweep * -1.0)
    } else {
        (signed, sweep)
    };
    let radius = sphere.radius();
    Ok((area, (radius * area + (centre - about).dot(sweep)) / 3.0))
}

/// The signed area of a sphere face's outer loop when the face can take
/// [`sphere_patch_flux`]: a loop with no chords that winds none of the
/// sphere's u.
fn sphere_patch_outer_area(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Option<f64>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let FaceSurface::Sphere(sphere) = face.surface() else {
        return Ok(None);
    };
    for oe in topo.wire(face.outer_wire())?.edges() {
        if matches!(
            topo.edge(oe.edge())?.curve(),
            brepkit_topology::edge::EdgeCurve::Line
        ) {
            return Ok(None);
        }
    }
    super::area::sphere_wire_signed_area(topo, sphere, face.outer_wire())
}

/// The flux of a sphere face whose outer loop, with no chords, winds none
/// of the sphere's u: the region inside the loop or the sphere past it (the
/// ball's volume less the patch's flux), as the face's mesh tells them
/// apart, less its holes. `None` for any other sphere face, or one too near
/// half the sphere to tell.
fn sphere_patch_flux(
    topo: &Topology,
    face_id: FaceId,
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    use std::f64::consts::PI;
    let Some(signed) = sphere_patch_outer_area(topo, face_id)? else {
        return Ok(None);
    };
    let face = topo.face(face_id)?;
    let FaceSurface::Sphere(sphere) = face.surface() else {
        return Ok(None);
    };
    let (loop_area, loop_flux) = sphere_loop_flux(topo, sphere, face.outer_wire(), signed, about)?;
    let (mut hole_area, mut hole_flux) = (0.0, 0.0);
    for &wid in face.inner_wires() {
        let Some((area, flux)) = sphere_hole_area_and_flux(topo, sphere, wid, about)? else {
            return Ok(None);
        };
        hole_area += area;
        hole_flux += flux;
    }
    let radius = sphere.radius();
    let patch = loop_area - hole_area;
    let past = 4.0 * PI * radius * radius - loop_area - hole_area;
    let Some(inside) = super::area::sphere_face_is_patch(topo, face_id, sphere, patch, past)?
    else {
        return Ok(None);
    };
    let flux = if inside {
        loop_flux - hole_flux
    } else {
        4.0 / 3.0 * PI * radius.powi(3) - loop_flux - hole_flux
    };
    Ok(Some(if face.is_reversed() { -flux } else { flux }))
}

/// Exact signed volume contribution of a toroidal face via the divergence
/// theorem: `V = (1/3) integral P.n dA`.
///
/// For a torus parameterised as
///   `P(u,v) = C + (R + r*cos_v)*(cos_u*ex + sin_u*ey) + r*sin_v*ez`
/// the outward normal `n = cos_v*(cos_u*ex + sin_u*ey) + sin_v*ez`,
/// and `dA = r*(R + r*cos_v) du dv`.
///
/// `None` for a face with a hole that winds either angle (one of two loops
/// round the tube), which bounds no region of its own to take away.
#[allow(clippy::too_many_lines)]
fn analytic_torus_signed_volume(
    topo: &Topology,
    face_id: FaceId,
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let tor = match face.surface() {
        FaceSurface::Torus(t) => t,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "analytic_torus_signed_volume requires a torus face".into(),
            });
        }
    };

    let mut holes = 0.0;
    for &wire in face.inner_wires() {
        let Some(hole) = torus_hole_flux(topo, tor, wire, about)? else {
            return Ok(None);
        };
        holes += hole;
    }
    let wire = topo.wire(face.outer_wire())?;
    // The whole ring's flux is its volume about any point.
    if is_whole_ring(topo, face) {
        let ring = 2.0
            * std::f64::consts::PI
            * std::f64::consts::PI
            * tor.major_radius()
            * tor.minor_radius()
            * tor.minor_radius();
        let vol = ring - holes;
        return Ok(Some(if face.is_reversed() { -vol } else { vol }));
    }
    // A face trimmed by a free-form curve (a plane's loop around the tube) is
    // no box in (u, v); its boundary walk up one side and down the other
    // winds neither angle, so its flux follows as a hole's does.
    if has_free_form_boundary(topo, face_id)?
        && let Some(outer) = torus_hole_flux(topo, tor, face.outer_wire(), about)?
    {
        let vol = outer - holes;
        return Ok(Some(if face.is_reversed() { -vol } else { vol }));
    }
    let mut u_vals = Vec::new();
    let mut v_vals = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            for &vid in &[edge.start(), edge.end()] {
                if let Ok(vtx) = topo.vertex(vid) {
                    let (u, v) = tor.project_point(vtx.point());
                    if !on_torus_axis(tor, vtx.point()) {
                        u_vals.push(u);
                    }
                    v_vals.push(v);
                }
            }
            // Sample each arc edge's midpoint to widen the angular ranges past
            // the two endpoint angles — without this a partial (sub-2π) band has
            // only its corner angles, `compute_angular_range` falls back to the
            // full 2π, and the band over-counts (gh #968). Both the major (u) and
            // minor (v) ranges are captured, since an arc may run in either
            // direction. A revolution-band boundary is a rational `NurbsCurve`
            // (the swept circle) or a `Circle` (the profile arc copy).
            if !edge.is_closed()
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
                let mid = match edge.curve() {
                    brepkit_topology::edge::EdgeCurve::Circle(circle) => {
                        // The arc's own domain says which way round it runs (see
                        // the cone's samples).
                        let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
                        Some(circle.evaluate(f64::midpoint(t0, t1)))
                    }
                    brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => {
                        let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
                        Some(nc.evaluate(f64::midpoint(t0, t1)))
                    }
                    brepkit_topology::edge::EdgeCurve::Line => {
                        let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
                        let t_mid = f64::midpoint(t0, t1);
                        Some(
                            edge.curve()
                                .evaluate_with_endpoints(t_mid, sv.point(), ev.point()),
                        )
                    }
                    brepkit_topology::edge::EdgeCurve::Ellipse(_) => None,
                };
                if let Some(mid) = mid {
                    let (u, v) = tor.project_point(mid);
                    u_vals.push(u);
                    v_vals.push(v);
                }
            }
        }
    }

    // A sector between two tube cross-sections covers the whole tube, over
    // the ring angle its seam runs along.
    let sector = torus_sector_u_range(topo, face, tor)?;
    let v_min = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let v_max = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if sector.is_none() && (v_max - v_min).abs() < 1e-15 {
        return Ok(Some(0.0));
    }

    // The minor (v) angle is periodic, so the raw [min, max] span is wrong for
    // a band whose samples straddle the v = 0/2π seam: a rim at the outer
    // equator can project to v = 2π−ε from float noise, turning a [0, π] band
    // into a phantom [π/2, 2π] complement (a −27% "exact" volume). With enough
    // samples (≥3 distinct), pick the range gap-wise like `u` does — the band
    // is the complement of the largest angular gap. A band given by just two
    // distinct minor angles spanning MORE than π stays ambiguous either way
    // (e.g. a fillet's concave quarter rim, whose endpoints are 270° apart but
    // whose band is the 90° short side), so decline and fall back to
    // tessellation rather than integrate the wrong portion.
    // A band seamed along a meridian says by its seam which side it covers.
    let seamed = torus_band_v_range(topo, face, tor)?;
    let (v_min, v_max) = if sector.is_some() {
        (0.0, std::f64::consts::TAU)
    } else if let Some(range) = seamed {
        range
    } else {
        let mut sorted = v_vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        if sorted.len() < 3 {
            if (v_max - v_min) > std::f64::consts::PI + 1e-9 {
                return Err(crate::OperationsError::InvalidInput {
                    reason:
                        "torus band minor range is ambiguous (seam-straddling, no interior sample)"
                            .into(),
                });
            }
            (v_min, v_max)
        } else {
            compute_angular_range(&mut v_vals)
        }
    };

    // A band bounded by a whole parallel (a closed circle round the torus's
    // axis, as a fillet round a shaft's end has) runs the whole ring; its
    // vertices give only the seam's angle, which the samples read as a part
    // turn.
    let whole_parallel = wire.edges().iter().any(|oe| {
        topo.edge(oe.edge()).is_ok_and(|edge| match edge.curve() {
            brepkit_topology::edge::EdgeCurve::Circle(circle) if edge.is_closed() => {
                let off = circle.center() - tor.center();
                circle.normal().cross(tor.z_axis()).length() < 1e-9
                    && (off - tor.z_axis() * off.dot(tor.z_axis())).length()
                        <= 1e-7 * tor.major_radius().max(1.0)
            }
            _ => false,
        })
    });
    let u_range = sector.unwrap_or_else(|| {
        if whole_parallel {
            (0.0, std::f64::consts::TAU)
        } else {
            compute_angular_range(&mut u_vals)
        }
    });

    let big_r = tor.major_radius();
    let small_r = tor.minor_radius();
    let x_axis = tor.x_axis();
    let y_axis = tor.y_axis();
    let z_axis = tor.z_axis();
    let c = tor.center();
    let c_vec = c - about;

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

    let vol =
        (1.0 / 3.0) * (s_coeff * s_u_integral + (cz_coeff + rcos_coeff + const_coeff) * du) - holes;

    Ok(Some(if face.is_reversed() { -vol } else { vol }))
}

/// Divergence flux `(1/3) ∫ (P − about)·N dA` of the region of a torus that
/// a loop winding neither of its angles bounds (a drill's entry hole), under
/// the torus's outward normal. By Green's theorem it is `∮ G dv` along the
/// loop, counterclockwise in `(u, v)`, where `G` is the flux density's
/// antiderivative in `u`:
/// `(r/3)(R + r cos v)[cos v (cx sin u − cy cos u) + (cz sin v + R cos v + r) u]`
/// (midpoint sums at two resolutions and a Richardson step). `None` for a
/// loop that winds either angle.
fn torus_hole_flux(
    topo: &Topology,
    torus: &brepkit_math::surfaces::ToroidalSurface,
    wire_id: brepkit_topology::wire::WireId,
    about: Point3,
) -> Result<Option<f64>, crate::OperationsError> {
    use std::f64::consts::{PI, TAU};
    let wrap = |d: f64| (d + PI).rem_euclid(TAU) - PI;
    let (big_r, small_r) = (torus.major_radius(), torus.minor_radius());
    let offset = torus.center() - about;
    let (cx, cy, cz) = (
        offset.dot(torus.x_axis()),
        offset.dot(torus.y_axis()),
        offset.dot(torus.z_axis()),
    );
    let antiderivative = |u: f64, v: f64| {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_v, cos_v) = v.sin_cos();
        small_r / 3.0
            * (big_r + small_r * cos_v)
            * (cos_v * (cx * sin_u - cy * cos_u) + (cz * sin_v + big_r * cos_v + small_r) * u)
    };
    // The loop's (u, v) walk, unwrapped continuously from its first sample.
    let step_to = |state: Option<(f64, f64)>, p: Point3| {
        let (u, v) = torus.project_point(p);
        state.map_or((u, v), |(lu, lv)| (lu + wrap(u - lu), lv + wrap(v - lv)))
    };
    let mut first: Option<(f64, f64)> = None;
    let mut last: Option<(f64, f64)> = None;
    let (mut flux, mut area) = (0.0, 0.0);
    for oe in topo.wire(wire_id)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let at = |t: f64| edge.curve().evaluate_with_endpoints(t, sp, ep);
        // Midpoint Stieltjes sums of G and of u against dv at `n` steps,
        // returning the walk's state at the span's end.
        let walk = |(from, to): (f64, f64), n: usize, state: Option<(f64, f64)>| {
            let (mut g, mut a) = (0.0, 0.0);
            #[allow(clippy::cast_precision_loss)]
            let step = (to - from) / n as f64;
            let mut state = Some(step_to(state, at(from)));
            let start = state;
            let mut v_prev = state.map_or(0.0, |(_, v)| v);
            for k in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let tk = from + step * k as f64;
                let (um, vm) = step_to(state, at(tk + 0.5 * step));
                let next = step_to(Some((um, vm)), at(tk + step));
                g += antiderivative(um, vm) * (next.1 - v_prev);
                a += um * (next.1 - v_prev);
                v_prev = next.1;
                state = Some(next);
            }
            (g, a, start, state)
        };
        for span in traversal_spans(edge, oe.is_forward(), sp, ep) {
            let (gc, ac, _, _) = walk(span, 128, last);
            let (gf, af, start, end) = walk(span, 256, last);
            first = first.or(start);
            last = end;
            flux += (4.0 * gf - gc) / 3.0;
            area += (4.0 * af - ac) / 3.0;
        }
    }
    if let (Some((u0, v0)), Some((u1, v1))) = (first, last)
        && ((u1 - u0).abs() > PI || (v1 - v0).abs() > PI)
    {
        return Ok(None);
    }
    Ok(Some(if area < 0.0 { -flux } else { flux }))
}

/// Whether a torus face's outer wire is only zero-length lines: the seam
/// pair collapsed onto one vertex, bounding the whole ring.
fn is_whole_ring(topo: &Topology, face: &brepkit_topology::face::Face) -> bool {
    topo.wire(face.outer_wire()).is_ok_and(|wire| {
        wire.edges().iter().all(|oe| {
            topo.edge(oe.edge()).is_ok_and(|edge| {
                matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line)
                    && topo
                        .vertex(edge.start())
                        .ok()
                        .zip(topo.vertex(edge.end()).ok())
                        .is_some_and(|(a, b)| (a.point() - b.point()).length() < 1e-9)
            })
        })
    })
}

/// Whether a face's outer wire holds an edge that is neither a line nor a
/// circle.
fn has_free_form_boundary(topo: &Topology, fid: FaceId) -> Result<bool, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    let face = topo.face(fid)?;
    for oe in topo.wire(face.outer_wire())?.edges() {
        if matches!(
            topo.edge(oe.edge())?.curve(),
            EdgeCurve::NurbsCurve(_) | EdgeCurve::Ellipse(_)
        ) {
            return Ok(true);
        }
    }
    Ok(false)
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
    // Each face is integrated on its own, and where two faces' readings of
    // their shared edge differ (a chord against the curve it sags from) the
    // gap's flux grows with its distance from the anchor: take the solid's
    // own centre, not the origin a part may sit far from.
    let about = solid_centre(topo, solid);
    // Each face's term is worked out on its own (on rayon's pool off wasm)
    // and the terms are summed in the faces' order, as one loop would.
    let face_term = |fid: FaceId| -> Result<f64, crate::OperationsError> {
        let face = topo.face(fid)?;

        // A sphere or torus face is integrated along its wires, which holds
        // for every outline: the closed forms below read most faces as a
        // box in the surface's two angles, which a collar round a box's
        // corners or a band between two loops round a tube is not.
        if matches!(
            face.surface(),
            FaceSurface::Sphere(_) | FaceSurface::Torus(_)
        ) && let Some(volume) = brepkit_check::properties::exact_face_volume(topo, fid, about)?
        {
            return Ok(volume * 6.0);
        }

        // Use exact analytical volume for analytic surface faces. The
        // closed forms below take a cylinder or cone face as a band between
        // two iso-circles, which a face with holes, one trimmed by a
        // free-form curve (a bore's saddle-shaped entry loop) or one stepped
        // along its rim is not: any face that does not wind its axis is
        // integrated along its boundary instead, which holds for every
        // outline.
        if matches!(
            face.surface(),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
        ) && let Some(flux) = developable_face_flux(topo, fid, about)?
        {
            if vol_trace_enabled() {
                log::debug!("VOL_TRACE holed developable face {fid:?} -> {flux}");
            }
            return Ok(flux * 6.0);
        }
        match face.surface() {
            FaceSurface::Cylinder(_) => {
                let v = analytic_cylinder_signed_volume(topo, fid, about)? * 6.0;
                if vol_trace_enabled() {
                    log::debug!("VOL_TRACE direct cyl face {:?} -> {}", fid, v / 6.0);
                }
                return Ok(v);
            }
            FaceSurface::Cone(_) => {
                return Ok(analytic_cone_signed_volume(topo, fid, about)? * 6.0);
            }
            FaceSurface::Sphere(_) => {
                let flux = match sphere_patch_flux(topo, fid, about)? {
                    Some(flux) => Some(flux),
                    None => analytic_sphere_signed_volume(topo, fid, about)?,
                };
                // Neither closed form fits: the face's own mesh below.
                if let Some(flux) = flux {
                    return Ok(flux * 6.0);
                }
            }
            FaceSurface::Torus(_) => {
                // A hole the closed form cannot read: the face's own mesh
                // below.
                if let Some(flux) = analytic_torus_signed_volume(topo, fid, about)? {
                    return Ok(flux * 6.0);
                }
            }
            FaceSurface::Plane { .. } => {
                if let Some(flux) = planar_face_flux(topo, fid, about)? {
                    return Ok(flux * 6.0);
                }
            }
            // Another face of the solid that the exact integral declines
            // leaves this one to it all the same: only a face it declines
            // takes its mesh below.
            FaceSurface::Nurbs(_) => {
                if let Some(volume) =
                    brepkit_check::properties::exact_face_volume(topo, fid, about)?
                {
                    return Ok(volume * 6.0);
                }
            }
        }

        let uv_mesh = tessellate::tessellate_with_uvs(topo, fid, deflection)?;
        let nurbs = match face.surface() {
            FaceSurface::Nurbs(s) => Some(s),
            _ => None,
        };
        let (face_total, _) = surface_flux_over_mesh(&uv_mesh, nurbs, deflection, about);
        let tri_count = uv_mesh.mesh.indices.len() / 3;

        if vol_trace_enabled() {
            log::debug!(
                "VOL_TRACE direct planar face {:?} -> {} (tris={})",
                fid,
                face_total / 6.0,
                tri_count
            );
        }
        Ok(face_total)
    };
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    #[cfg(not(target_arch = "wasm32"))]
    let terms: Vec<Result<f64, crate::OperationsError>> = {
        use rayon::prelude::*;
        faces.par_iter().map(|&fid| face_term(fid)).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let terms: Vec<Result<f64, crate::OperationsError>> =
        faces.iter().map(|&fid| face_term(fid)).collect();
    let mut total: f64 = 0.0;
    for term in terms {
        total += term?;
    }

    Ok((total / 6.0).abs())
}

/// A big face's mesh is integrated in chunks of this many triangle
/// indices (1024 triangles) on rayon's pool.
#[cfg(not(target_arch = "wasm32"))]
const TRIANGLE_CHUNK: usize = 3 * 1024;

/// Twice the flux `∫ (P − about)·N dA` of a face over its mesh, the
/// signed-tetrahedra sum's unit, and the matching first moment: the sum of
/// each tetrahedron's six volumes times its three free corners, which is
/// six times the flux of `r (r·N)` for `r = P − about`.
///
/// On a NURBS face each triangle is integrated on the surface itself, by
/// Gauss quadrature over the triangle's own `(u, v)`: the flat triangles
/// inscribe a curved wall, and an extruded spline profile's side wall came
/// out short of its caps (which are exact) by about two thirds of the sag
/// times the profile's length. The surface is shifted by its corners'
/// offsets from the mesh, interpolated across the triangle, so the patch
/// runs through the mesh's own corners: a boundary vertex sits on its edge,
/// which an imported face's surface can miss by a good deal more than the
/// sag, and the triangles' `(u, v)` alone would then trim the face short.
///
/// A triangle keeps its flat contribution where the surface does not
/// reproduce its corners (a `u` unwrapped past a period, say), where a
/// corner sits at a pole, or where its `(u, v)` winding disagrees with its
/// own.
fn surface_flux_over_mesh(
    uv_mesh: &tessellate::TriangleMeshUV,
    nurbs: Option<&brepkit_math::nurbs::surface::NurbsSurface>,
    deflection: f64,
    about: Point3,
) -> (f64, Vec3) {
    use brepkit_math::traits::ParametricSurface;
    let mesh = &uv_mesh.mesh;
    let (idx, pos, uvs) = (&mesh.indices, &mesh.positions, &uv_mesh.uvs);
    let flat_flux = |i0: usize, i1: usize, i2: usize| {
        let (a, b, c) = (pos[i0] - about, pos[i1] - about, pos[i2] - about);
        let six_vol = a.dot((b - a).cross(c - a));
        (six_vol, (a + b + c) * six_vol)
    };
    let add = |(f, m): (f64, Vec3), (g, n): (f64, Vec3)| (f + g, m + n);
    let zero = (0.0, Vec3::new(0.0, 0.0, 0.0));
    let Some(s) = nurbs.filter(|_| uvs.len() == pos.len()) else {
        return idx
            .chunks_exact(3)
            .map(|t| flat_flux(t[0] as usize, t[1] as usize, t[2] as usize))
            .fold(zero, add);
    };
    if let Some(whole) = whole_patch_flux(uv_mesh, s, about) {
        return whole;
    }
    // A closed surface's mesh can run its (u, v) on past the knot range
    // (a seam unwrapped across): the surface there is the same a period back.
    let period = |range: Option<(f64, f64)>| range.map(|(lo, hi)| (lo, hi - lo));
    let periods = (period(s.closed_u()), period(s.closed_v()));
    let wrap = move |u: f64, v: f64| {
        let back =
            |x: f64, p: Option<(f64, f64)>| p.map_or(x, |(lo, len)| lo + (x - lo).rem_euclid(len));
        (back(u, periods.0), back(v, periods.1))
    };
    let lines = KnotLines::of(s, periods);
    // Each vertex's offset from the surface at its (u, v), past what a sag
    // can explain, or at a pole. A pole (a cone's apex, where a ruled patch
    // runs to a point) is one point for a whole row of the domain, so its
    // vertex has no (u, v) of its own: the mesh gives it one point of that
    // row, and a triangle from it to the rim, read over its (u, v), sweeps
    // round the surface towards that point instead of running up the strip
    // it spans. A vertex whose whole row across the domain is shorter than
    // the deflection lies at a pole (a cone's wall in B-spline form whose
    // domain stops 5e-9 short of its apex).
    let mut ev = s.evaluator();
    let tol = (10.0 * deflection).max(1e-6);
    let ((u_lo, u_hi), (v_lo, v_hi)) = (s.domain_u(), s.domain_v());
    let corners: Vec<Corner> = pos
        .iter()
        .zip(uvs)
        .map(|(p, uv)| {
            let (u, v) = wrap(uv[0], uv[1]);
            let Some((at, s_u, s_v)) = ev.point_and_partials(u, v) else {
                return Corner::Far;
            };
            // A slow speed at one point (a row that starts slowly) is only
            // a lead: the whole row must lie within the deflection.
            let (lu, lv) = (s_u.length(), s_v.length());
            if (lu * (u_hi - u_lo) <= deflection || lu <= 1e-9 * lv)
                && row_collapses(s, 0, v, deflection)
            {
                return Corner::Pole(0);
            }
            if (lv * (v_hi - v_lo) <= deflection || lv <= 1e-9 * lu)
                && row_collapses(s, 1, u, deflection)
            {
                return Corner::Pole(1);
            }
            let d = *p - at;
            if d.length() <= tol {
                Corner::Offset(d)
            } else {
                Corner::Far
            }
        })
        .collect();
    // The strip from the pole's row up to the side `a`-`b` of a triangle
    // whose third corner `p` lies at a pole: the quadrilateral from the
    // pole's row under `a`, through `a` and `b`, to the pole's row under
    // `b`, read as two triangles.
    let pole_strip = |ev: &mut brepkit_math::nurbs::evaluator::SurfaceEvaluator<'_>,
                      read: &ReadTriangle<'_>,
                      (p, axis): (usize, usize),
                      (a, da): (usize, Vec3),
                      (b, db): (usize, Vec3)| {
        let under = |i: usize| {
            let mut uv = uvs[i];
            uv[1 - axis] = uvs[p][1 - axis];
            uv
        };
        let (pa, pb) = (under(a), under(b));
        let mut off = |uv: [f64; 2]| -> Option<Vec3> {
            let (u, v) = wrap(uv[0], uv[1]);
            Some(pos[p] - ev.point_and_partials(u, v)?.0)
        };
        let (dpa, dpb) = (off(pa)?, off(pb)?);
        let first = read(
            ev,
            [
                (pa, pos[p], dpa),
                (uvs[a], pos[a], da),
                (uvs[b], pos[b], db),
            ],
        )?;
        // Under `a` and `b` at one point, the strip is the first triangle.
        if (pb[axis] - pa[axis]).abs() <= f64::EPSILON * pa[axis].abs().max(1.0) {
            return Some(first);
        }
        let second = read(
            ev,
            [(pa, pos[p], dpa), (uvs[b], pos[b], db), (pb, pos[p], dpb)],
        )?;
        Some((first.0 + second.0, first.1 + second.1))
    };
    // Each triangle's term, worked out on its own: in chunks on rayon's
    // pool (each with its own evaluator) for a big mesh, and summed in the
    // triangles' order, as one loop would.
    let term = |ev: &mut brepkit_math::nurbs::evaluator::SurfaceEvaluator<'_>, tri: &[u32]| {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let flat = flat_flux(i0, i1, i2);
        let flat_normal = (pos[i1] - pos[i0]).cross(pos[i2] - pos[i0]);
        let read = |ev: &mut brepkit_math::nurbs::evaluator::SurfaceEvaluator<'_>,
                    three: [([f64; 2], Point3, Vec3); 3]| {
            triangle_flux_on_surface(ev, &wrap, &lines, three, flat_normal, deflection, about)
        };
        let at = |i: usize, d: Vec3| (uvs[i], pos[i], d);
        match [corners[i0], corners[i1], corners[i2]] {
            [Corner::Offset(d0), Corner::Offset(d1), Corner::Offset(d2)] => {
                read(ev, [at(i0, d0), at(i1, d1), at(i2, d2)]).unwrap_or(flat)
            }
            // One corner at a pole: the triangle spans the strip between the
            // pole's row, under its other two corners, and the side between
            // them. The pole's vertex stands for each point of its row.
            [Corner::Pole(axis), Corner::Offset(da), Corner::Offset(db)] => {
                pole_strip(ev, &read, (i0, axis), (i1, da), (i2, db)).unwrap_or(flat)
            }
            [Corner::Offset(db), Corner::Pole(axis), Corner::Offset(da)] => {
                pole_strip(ev, &read, (i1, axis), (i2, da), (i0, db)).unwrap_or(flat)
            }
            [Corner::Offset(da), Corner::Offset(db), Corner::Pole(axis)] => {
                pole_strip(ev, &read, (i2, axis), (i0, da), (i1, db)).unwrap_or(flat)
            }
            _ => flat,
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    if idx.len() > TRIANGLE_CHUNK {
        use rayon::prelude::*;
        let terms: Vec<Vec<(f64, Vec3)>> = idx
            .par_chunks(TRIANGLE_CHUNK)
            .map(|chunk| {
                let mut ev = s.evaluator();
                chunk
                    .chunks_exact(3)
                    .map(|tri| term(&mut ev, tri))
                    .collect()
            })
            .collect();
        return terms.into_iter().flatten().fold(zero, add);
    }
    idx.chunks_exact(3)
        .map(|tri| term(&mut ev, tri))
        .fold(zero, add)
}

/// Twice the flux and six times the moment's flux (as
/// [`surface_flux_over_mesh`]) of a face that is its surface's whole
/// patch, integrated over the parameter domain by Gauss quadrature on each
/// knot span; `None` for a trimmed face. A revolved spline's band is one:
/// its mesh triangles, read on the surface, still left the solid a few
/// parts in a million short (an ellipse revolved, 1.2e-6).
///
/// The face's mesh tells: its `(u, v)` all lie in the domain and cover all
/// of it, which a face trimmed by any curve does not (its loop's inscribed
/// `(u, v)` polygon is smaller), and its vertices lie on the surface. The
/// mesh's winding in `(u, v)` says which
/// way the face turns from the surface's own normal.
fn whole_patch_flux(
    uv_mesh: &tessellate::TriangleMeshUV,
    s: &brepkit_math::nurbs::surface::NurbsSurface,
    about: Point3,
) -> Option<(f64, Vec3)> {
    let ((u0, u1), (v0, v1)) = (s.domain_u(), s.domain_v());
    let domain = (u1 - u0) * (v1 - v0);
    let uvs = &uv_mesh.uvs;
    if uvs.len() != uv_mesh.mesh.positions.len() || domain <= 0.0 {
        return None;
    }
    let (du, dv) = (1e-9 * (u1 - u0), 1e-9 * (v1 - v0));
    if uvs
        .iter()
        .any(|uv| uv[0] < u0 - du || uv[0] > u1 + du || uv[1] < v0 - dv || uv[1] > v1 + dv)
    {
        return None;
    }
    let signed: f64 = uv_mesh
        .mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            let (a, b, c) = (uvs[t[0] as usize], uvs[t[1] as usize], uvs[t[2] as usize]);
            0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        })
        .sum();
    // The polygons of a trimmed face's loops fall short of the domain by
    // their sag, far more than this.
    if (signed.abs() - domain).abs() > 1e-9 * domain {
        return None;
    }
    // Nor is it the patch when its edges miss the surface (an imported
    // face's can): the mesh follows the edges.
    let mut ev = s.evaluator();
    if uv_mesh
        .mesh
        .positions
        .iter()
        .zip(uvs)
        .any(|(p, uv)| (*p - ev.point(uv[0], uv[1])).length() > 1e-7)
    {
        return None;
    }
    let spans = |knots: &[f64], lo: f64, hi: f64| {
        let mut ends = vec![lo];
        ends.extend(knots.iter().copied().filter(|&k| k > lo && k < hi));
        ends.push(hi);
        ends.dedup();
        ends
    };
    let (us, vs) = (spans(s.knots_u(), u0, u1), spans(s.knots_v(), v0, v1));
    let gauss =
        brepkit_math::quadrature::gauss_legendre_points(brepkit_math::quadrature::MAX_ORDER);
    let (mut flux, mut moment) = (0.0, Vec3::new(0.0, 0.0, 0.0));
    for u_span in us.windows(2) {
        let (hu, mu) = (0.5 * (u_span[1] - u_span[0]), 0.5 * (u_span[1] + u_span[0]));
        for v_span in vs.windows(2) {
            let (hv, mv) = (0.5 * (v_span[1] - v_span[0]), 0.5 * (v_span[1] + v_span[0]));
            for gu in gauss {
                for gv in gauss {
                    let (p, s_u, s_v) =
                        ev.point_and_partials(hu.madd(gu.x, mu), hv.madd(gv.x, mv))?;
                    let r = p - about;
                    let w = gu.w * gv.w * hu * hv * r.dot(s_u.cross(s_v));
                    flux += w;
                    moment += r * w;
                }
            }
        }
    }
    let sign = signed.signum();
    Some((2.0 * sign * flux, moment * (6.0 * sign)))
}

/// Whether the row of `s` along `u` (`axis` 0) at `v = at`, or along `v`
/// at `u = at`, lies within `reach` of one point: its control points there,
/// whose hull holds it where its weights are all positive, lie within a box
/// no wider than `reach` across.
fn row_collapses(
    s: &brepkit_math::nurbs::surface::NurbsSurface,
    axis: usize,
    at: f64,
    reach: f64,
) -> bool {
    use brepkit_math::nurbs::basis::{basis_funs, find_span};
    let (cps, ws) = (s.control_points(), s.weights());
    let (rows, cols) = (cps.len(), cps.first().map_or(0, Vec::len));
    if rows == 0 || cols == 0 || ws.iter().flatten().any(|&w| w <= 0.0) {
        return false;
    }
    let (degree, knots, count, along) = if axis == 0 {
        (s.degree_v(), s.knots_v(), cols, rows)
    } else {
        (s.degree_u(), s.knots_u(), rows, cols)
    };
    let span = find_span(count, degree, at, knots);
    let basis = basis_funs(span, at, degree, knots);
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for k in 0..along {
        let (mut point, mut weight) = (Vec3::new(0.0, 0.0, 0.0), 0.0);
        for (b, &n) in basis.iter().enumerate() {
            let other = span - degree + b;
            let (i, j) = if axis == 0 { (k, other) } else { (other, k) };
            let w = n * ws[i][j];
            point += Vec3::new(cps[i][j].x(), cps[i][j].y(), cps[i][j].z()) * w;
            weight += w;
        }
        if weight <= 0.0 {
            return false;
        }
        let q = point * (1.0 / weight);
        for (c, x) in [q.x(), q.y(), q.z()].into_iter().enumerate() {
            lo[c] = lo[c].min(x);
            hi[c] = hi[c].max(x);
        }
    }
    let wide = (0..3).map(|c| (hi[c] - lo[c]).powi(2)).sum::<f64>().sqrt();
    wide <= reach
}

/// A closed direction's start and period, or `None` for an open one.
type Period = Option<(f64, f64)>;

/// A triangle's term read on a surface from its corners' `(u, v)`,
/// positions and offsets from the surface (see
/// [`triangle_flux_on_surface`]).
type ReadTriangle<'a> = dyn Fn(
        &mut brepkit_math::nurbs::evaluator::SurfaceEvaluator<'_>,
        [([f64; 2], Point3, Vec3); 3],
    ) -> Option<(f64, Vec3)>
    + 'a;

/// How a mesh vertex stands to a NURBS surface: its offset from the surface
/// at its `(u, v)`, at a pole where the row along `u` (0) or `v` (1)
/// shrinks to a point, or farther off than a sag can explain.
#[derive(Clone, Copy)]
enum Corner {
    Offset(Vec3),
    Pole(usize),
    Far,
}

/// The lines of a NURBS surface's `(u, v)` across which it may bend at once:
/// its knots inside its domain, and on a surface that closes, its seam, each
/// repeated every period along a closed direction.
struct KnotLines {
    u: Vec<f64>,
    v: Vec<f64>,
    periods: (Option<f64>, Option<f64>),
}

impl KnotLines {
    fn of(s: &brepkit_math::nurbs::surface::NurbsSurface, periods: (Period, Period)) -> Self {
        let inner = |knots: &[f64], (lo, hi): (f64, f64), period: Option<(f64, f64)>| {
            let mut lines: Vec<f64> = knots
                .iter()
                .copied()
                .filter(|&k| k > lo && k < hi)
                .collect();
            if period.is_some() {
                lines.push(lo);
            }
            lines.sort_by(f64::total_cmp);
            lines.dedup();
            lines
        };
        Self {
            u: inner(s.knots_u(), s.domain_u(), periods.0),
            v: inner(s.knots_v(), s.domain_v(), periods.1),
            periods: (periods.0.map(|p| p.1), periods.1.map(|p| p.1)),
        }
    }

    /// The lines along `axis` (0 for `u`) strictly between `lo` and `hi`.
    fn between(&self, axis: usize, lo: f64, hi: f64) -> Vec<f64> {
        let (lines, period) = if axis == 0 {
            (&self.u, self.periods.0)
        } else {
            (&self.v, self.periods.1)
        };
        let mut found = Vec::new();
        if lines.is_empty() {
            return found;
        }
        for &line in lines {
            match period {
                Some(p) if p > 0.0 => {
                    #[allow(clippy::cast_possible_truncation)]
                    let first = ((lo - line) / p).floor() as i64;
                    #[allow(clippy::cast_possible_truncation)]
                    let last = ((hi - line) / p).ceil() as i64;
                    for m in first..=last {
                        #[allow(clippy::cast_precision_loss)]
                        let x = p.madd(m as f64, line);
                        if x > lo && x < hi {
                            found.push(x);
                        }
                    }
                }
                _ => {
                    if line > lo && line < hi {
                        found.push(line);
                    }
                }
            }
        }
        found
    }

    /// The triangle `a, b, c` in `(u, v)` cut along the lines that cross it,
    /// as triangles that keep its way round; `None` where none crosses it.
    fn cut(&self, a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<Vec<[[f64; 2]; 3]>> {
        let span = |axis: usize| {
            (
                a[axis].min(b[axis]).min(c[axis]),
                a[axis].max(b[axis]).max(c[axis]),
            )
        };
        let crossing: [Vec<f64>; 2] = [0, 1].map(|axis| {
            let (lo, hi) = span(axis);
            self.between(axis, lo, hi)
        });
        if crossing.iter().all(Vec::is_empty) {
            return None;
        }
        let mut pieces: Vec<Vec<[f64; 2]>> = vec![vec![a, b, c]];
        for (axis, lines) in crossing.into_iter().enumerate() {
            for line in lines {
                let mut next = Vec::with_capacity(pieces.len() + 1);
                for piece in pieces {
                    for keep_below in [true, false] {
                        let part = clip(&piece, axis, line, keep_below);
                        if part.len() >= 3 {
                            next.push(part);
                        }
                    }
                }
                pieces = next;
            }
        }
        let mut triangles = Vec::new();
        for piece in pieces {
            for k in 1..piece.len() - 1 {
                triangles.push([piece[0], piece[k], piece[k + 1]]);
            }
        }
        Some(triangles)
    }
}

/// The part of the convex polygon `points` in `(u, v)` on one side of the
/// line where coordinate `axis` is `at`: below it, or above it.
fn clip(points: &[[f64; 2]], axis: usize, at: f64, keep_below: bool) -> Vec<[f64; 2]> {
    let inside = |p: &[f64; 2]| {
        if keep_below {
            p[axis] <= at
        } else {
            p[axis] >= at
        }
    };
    let mut out = Vec::with_capacity(points.len() + 1);
    for k in 0..points.len() {
        let (p, q) = (points[k], points[(k + 1) % points.len()]);
        if inside(&p) {
            out.push(p);
        }
        if (p[axis] - at) * (q[axis] - at) < 0.0 {
            let f = (at - p[axis]) / (q[axis] - p[axis]);
            let mut x = [(q[0] - p[0]).madd(f, p[0]), (q[1] - p[1]).madd(f, p[1])];
            x[axis] = at;
            out.push(x);
        }
    }
    out
}

/// Twice the flux about `about` of the surface patch over one mesh
/// triangle, and its first moment (see [`surface_flux_over_mesh`]), or
/// `None` where the flat triangle is the better reading. A triangle across a
/// knot line of the surface, where it may bend at once, is read in the
/// pieces the line cuts it into: the rule is smooth.
fn triangle_flux_on_surface(
    ev: &mut brepkit_math::nurbs::evaluator::SurfaceEvaluator<'_>,
    wrap: &dyn Fn(f64, f64) -> (f64, f64),
    lines: &KnotLines,
    corners: [([f64; 2], Point3, Vec3); 3],
    flat_normal: Vec3,
    deflection: f64,
    about: Point3,
) -> Option<(f64, Vec3)> {
    // Dunavant's degree-4 rule: barycentric (a, a, b) in each order.
    const RULE: [(f64, f64, f64); 2] = [
        (
            0.445_948_490_915_965,
            0.108_103_018_168_070,
            0.223_381_589_678_011,
        ),
        (
            0.091_576_213_509_771,
            0.816_847_572_980_459,
            0.109_951_743_655_322,
        ),
    ];
    let [(ua, p0, d0), (ub, p1, d1), (uc, p2, d2)] = corners;
    let uv_area2 = (ub[0] - ua[0]) * (uc[1] - ua[1]) - (uc[0] - ua[0]) * (ub[1] - ua[1]);
    if uv_area2.abs() < 1e-300 {
        return None;
    }
    let centroid = |p: Point3, q: Point3, r: Point3| {
        Point3::new(
            (p.x() + q.x() + r.x()) / 3.0,
            (p.y() + q.y() + r.y()) / 3.0,
            (p.z() + q.z() + r.z()) / 3.0,
        )
    };
    // Corners a period apart on a closed surface each map right, and the
    // triangle between them spans the surface the long way round.
    let longest = (p1 - p0)
        .length()
        .max((p2 - p1).length())
        .max((p0 - p2).length());
    let third = 1.0 / 3.0;
    let (mu, mv) = wrap(
        (ua[0] + ub[0] + uc[0]) * third,
        (ua[1] + ub[1] + uc[1]) * third,
    );
    let mid = ev.point(mu, mv) + (d0 + d1 + d2) * third;
    if (mid - centroid(p0, p1, p2)).length() > (10.0 * deflection).max(0.5 * longest) {
        return None;
    }
    // The interpolated offset's own slope in u and v, and each point's
    // weights of the corners.
    let (b1u, b1v) = ((uc[1] - ua[1]) / uv_area2, -(uc[0] - ua[0]) / uv_area2);
    let (b2u, b2v) = (-(ub[1] - ua[1]) / uv_area2, (ub[0] - ua[0]) / uv_area2);
    let off_u = (d1 - d0) * b1u + (d2 - d0) * b2u;
    let off_v = (d1 - d0) * b1v + (d2 - d0) * b2v;
    let (mut sum, mut moment, mut agrees) = (0.0, Vec3::new(0.0, 0.0, 0.0), true);
    // The triangle's normal over its `(u, v)` area is the size of the
    // surface's normal across it.
    let faint = 1e-6 * flat_normal.length() / uv_area2.abs();
    let whole = [[ua, ub, uc]];
    let cut = lines.cut(ua, ub, uc);
    for &[sa, sb, sc] in cut.as_deref().unwrap_or(&whole) {
        let area2 = (sb[0] - sa[0]) * (sc[1] - sa[1]) - (sc[0] - sa[0]) * (sb[1] - sa[1]);
        let (mut part, mut part_moment) = (0.0, Vec3::new(0.0, 0.0, 0.0));
        for &(p, q, w) in &RULE {
            for (w0, w1, w2) in [(p, p, q), (p, q, p), (q, p, p)] {
                let u = w0 * sa[0] + w1 * sb[0] + w2 * sc[0];
                let v = w0 * sa[1] + w1 * sb[1] + w2 * sc[1];
                let (du, dv) = (u - ua[0], v - ua[1]);
                let l1 = b1u.madd(du, b1v * dv);
                let l2 = b2u.madd(du, b2v * dv);
                let l0 = 1.0 - l1 - l2;
                let (wu, wv) = wrap(u, v);
                let (p_uv, s_u, s_v) = ev.point_and_partials(wu, wv)?;
                let n = (s_u + off_u).cross(s_v + off_v);
                // A point on a pole's row has next to no normal, whose
                // way is rounding: it weighs nothing either way.
                agrees &= n.dot(flat_normal) * uv_area2 >= 0.0 || n.length() <= faint;
                let r = (p_uv - about) + d0 * l0 + d1 * l1 + d2 * l2;
                part += w * r.dot(n);
                part_moment += r * (w * r.dot(n));
            }
        }
        // The rule's weights sum to one; `area2` is twice the piece's
        // signed `(u, v)` area, which makes this twice its flux, and three
        // times it six times the moment's flux.
        sum += part * area2;
        moment += part_moment * area2;
    }
    agrees.then_some((sum, moment * 3.0))
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

    let p0 = sum_anchor(topo, solid);
    let mut total = 0.0;
    let mut all_planar_triangles = true;

    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
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

        let (a, b, c) = (pts[0] - p0, pts[1] - p0, pts[2] - p0);

        total += a.dot((b - a).cross(c - a));
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
    // Per-face meshes need not close up, so the sums are taken about a
    // point on the solid, where a gap between two faces' meshes weighs
    // least (about the origin, a part far from it read its centre off by
    // its gaps times its distance). A NURBS face is integrated on its
    // surface, as `solid_volume` does.
    let p0 =
        vertex_box(topo, solid).map_or(Point3::new(0.0, 0.0, 0.0), |(lo, hi)| lo + (hi - lo) * 0.5);
    let mut total_vol: f64 = 0.0;
    let mut moment = Vec3::new(0.0, 0.0, 0.0);

    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let uv_mesh = tessellate::tessellate_with_uvs(topo, fid, deflection)?;
        let face = topo.face(fid)?;
        let nurbs = match face.surface() {
            FaceSurface::Nurbs(s) => Some(s),
            _ => None,
        };
        let (six_vol, face_moment) = surface_flux_over_mesh(&uv_mesh, nurbs, deflection, p0);
        total_vol += six_vol;
        moment += face_moment;
    }
    let (cx, cy, cz) = (moment.x(), moment.y(), moment.z());

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
    Ok(p0 + Vec3::new(cx / denom, cy / denom, cz / denom))
}

/// Compute center of mass directly from face vertex positions for
/// solids composed entirely of planar triangular faces.
fn center_of_mass_from_faces(
    topo: &Topology,
    solid: SolidId,
) -> Result<Point3, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;

    let p0 = sum_anchor(topo, solid);
    let mut total_vol = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
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

        let (a, b, c) = (pts[0] - p0, pts[1] - p0, pts[2] - p0);

        let signed_vol = a.dot((b - a).cross(c - a));
        total_vol += signed_vol;
        cx += signed_vol * (a.x() + b.x() + c.x());
        cy += signed_vol * (a.y() + b.y() + c.y());
        cz += signed_vol * (a.z() + b.z() + c.z());
    }

    if total_vol.abs() < 1e-15 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "solid has zero volume, center of mass is undefined".into(),
        });
    }

    let denom = 4.0 * total_vol;
    Ok(p0 + Vec3::new(cx / denom, cy / denom, cz / denom))
}

#[cfg(test)]
mod regression_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};
    use brepkit_topology::face::FaceSurface;

    /// A fully enclosed void makes the cavity a SEPARATE inner shell, which a
    /// volume path walking `outer_shell()` alone silently skips — the result is
    /// closed, manifold and correctly bounded, so only the volume shows it.
    #[test]
    fn enclosed_cavity_is_subtracted_from_solid_volume() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let outer = crate::primitives::make_box(&mut topo, 40.0, 40.0, 10.0).unwrap();
        // Off-centre on every axis, so the cavity moves the centre of mass too
        // and a CoM that ignores it cannot pass by symmetry.
        let void = crate::primitives::make_box(&mut topo, 20.0, 20.0, 4.0).unwrap();
        crate::transform::transform_solid(&mut topo, void, &Mat4::translation(5.0, 5.0, 2.0))
            .unwrap();
        let hollow =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, outer, void)
                .unwrap();

        assert_eq!(
            topo.solid(hollow).unwrap().inner_shells().len(),
            1,
            "the enclosed void must come back as an inner shell"
        );

        let v = solid_volume(&topo, hollow, 0.05).unwrap();
        assert!(
            (v - 14_400.0).abs() < 1.0,
            "40x40x10 minus an enclosed 20x20x4 void is 14400, got {v}"
        );

        // The tessellating paths are reached whenever the analytic ones above
        // decline, so they need the cavity too — they are not a slower route to
        // the same answer if they integrate a different set of faces.
        let direct = volume_from_direct_face_tessellation(&topo, hollow, 0.05).unwrap();
        assert!(
            (direct - 14_400.0).abs() < 1.0,
            "direct face tessellation must see the cavity, got {direct}"
        );

        // `check` measures the same solid through its own accumulator.
        let opts = brepkit_check::properties::PropertiesOptions::default();
        let checked = brepkit_check::properties::solid_volume(&topo, hollow, &opts).unwrap();
        assert!(
            (checked - 14_400.0).abs() < 1.0,
            "check::properties must see the cavity, got {checked}"
        );

        // 16000 at (20,20,5) minus 1600 at (15,15,4).
        let com = solid_center_of_mass(&topo, hollow, 0.05).unwrap();
        let want = Point3::new(20.555_555, 20.555_555, 5.111_111);
        assert!(
            (com - want).length() < 1e-3,
            "centre of mass must shift with the cavity, got {com:?}"
        );

        // Surface area is the whole boundary, cavity walls included:
        // 2*40*40 + 4*40*10 = 4800 outside, 2*20*20 + 4*20*4 = 1120 inside.
        let area = brepkit_check::properties::solid_area(&topo, hollow, &opts).unwrap();
        assert!(
            (area - 5920.0).abs() < 1.0,
            "surface area must include the cavity walls, got {area}"
        );
    }

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

    #[test]
    fn drilled_cylinder_volume_subtracts_the_bore() {
        // A coaxial tube (cylinder r=5 h=20 with a coaxial r=2 bore cut through
        // it). Its holed analytic faces (annular planar caps with inner wires +
        // two cylinder walls) must NOT hit a hole-FILLING analytic fast-path —
        // they route to the hole-aware revolution integrator (which subtracts
        // cap holes by MAGNITUDE, immune to a boolean's same-wound inner rims),
        // giving the bore-subtracted V = π·(5²−2²)·20 = π·420, not the
        // solid-cylinder π·500.
        use std::f64::consts::PI;
        let mut topo = Topology::new();
        let outer = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let bore = crate::primitives::make_cylinder(&mut topo, 2.0, 20.0).unwrap();
        let tube = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, outer, bore)
            .unwrap();

        // Neither analytic fast-path may claim this holed solid.
        assert!(
            try_analytic_solid_volume(&topo, tube).is_none(),
            "the holed tube must not hit the whole-solid analytic primitive path"
        );
        assert!(
            analytic_faces_solid_volume(&topo, tube).is_none(),
            "the holed tube must not hit the per-face analytic path (it ignores holes)"
        );

        let expect = PI * (25.0 - 4.0) * 20.0; // bore subtracted
        let vol = solid_volume(&topo, tube, 0.005).unwrap();
        let solid_cyl = PI * 25.0 * 20.0; // hole-FILLED (the wrong answer)
        assert!(
            (vol - expect).abs() < expect * 0.01,
            "drilled tube volume {vol} should be the bore-subtracted {expect}, \
             not the hole-filled {solid_cyl}"
        );
        assert!(
            (vol - solid_cyl).abs() > solid_cyl * 0.05,
            "drilled tube volume {vol} must be clearly LESS than the solid cylinder \
             {solid_cyl} (the bore is really removed)"
        );
    }

    #[test]
    fn plain_primitives_still_use_the_analytic_fast_path() {
        // The Finding-3 hole guard must not over-gate: a plain (hole-less)
        // cylinder, cone, sphere and torus must STILL hit the closed-form
        // analytic fast-path (no tessellation perf regression).
        use std::f64::consts::PI;
        let mut t = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut t, 3.0, 10.0).unwrap();
        let v = try_analytic_solid_volume(&t, cyl).expect("plain cylinder fast-path");
        assert!((v - PI * 9.0 * 10.0).abs() < 1e-9);

        let mut t = Topology::new();
        let cone = crate::primitives::make_cone(&mut t, 4.0, 0.0, 9.0).unwrap();
        let v = try_analytic_solid_volume(&t, cone).expect("plain cone fast-path");
        assert!((v - PI / 3.0 * 16.0 * 9.0).abs() < 1e-6);

        let mut t = Topology::new();
        let sph = crate::primitives::make_sphere(&mut t, 5.0, 32).unwrap();
        let v = try_analytic_solid_volume(&t, sph).expect("plain sphere fast-path");
        assert!((v - 4.0 / 3.0 * PI * 125.0).abs() < 1e-6);

        let mut t = Topology::new();
        let tor = crate::primitives::make_torus(&mut t, 6.0, 2.0, 32).unwrap();
        let v = try_analytic_solid_volume(&t, tor).expect("plain torus fast-path");
        assert!((v - 2.0 * PI * PI * 6.0 * 4.0).abs() < 1e-6);
    }

    /// A full-disc cap bounded by a SINGLE closed circle (`v→v`) must integrate to
    /// the exact disc area `πρ²` — its 2π sweep, not a dropped zero. Exercised via
    /// a TWO-section revolution (two stacked cone frustums → two cone walls + two
    /// closed-circle disc caps), which the single-primitive volume path declines
    /// (two cones), forcing `analytic_revolution_solid_volume`. The volume must be
    /// exact AND deflection-independent (analytic, not the inscribed mesh).
    #[test]
    fn closed_circle_disc_cap_volume_is_exact() {
        use brepkit_topology::explorer::solid_faces;
        use std::f64::consts::{PI, TAU};
        let mut topo = Topology::new();
        // (6,0)→(4,6) cone, (4,6)→(2,12) cone, (2,12)→(0,12) top disc cap,
        // (0,12)→(0,0) on-axis (no face); bottom (0,0)→(6,0) disc cap.
        let wire = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(6.0, 0.0, 0.0),
                Point3::new(4.0, 0.0, 6.0),
                Point3::new(2.0, 0.0, 12.0),
                Point3::new(0.0, 0.0, 12.0),
                Point3::new(0.0, 0.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let face = topo.add_face(brepkit_topology::face::Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        let solid = crate::revolve::revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            TAU,
        )
        .unwrap();

        // The merged solid has 2 cone walls + 2 closed-circle disc caps; the top
        // cap's closed circle must contribute (1/3)·12·π·2² to the volume.
        let fr = |rb: f64, rt: f64, h: f64| PI * h / 3.0 * rb.madd(rb, rb.madd(rt, rt * rt));
        let expected = fr(6.0, 4.0, 6.0) + fr(4.0, 2.0, 6.0);

        let v_fine = solid_volume(&topo, solid, 0.0001).unwrap();
        let v_coarse = solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            (v_fine - expected).abs() / expected < 1e-9,
            "two-section revolve volume {expected}, got {v_fine}"
        );
        assert!(
            (v_fine - v_coarse).abs() < 1e-9,
            "volume must be analytic (deflection-independent): {v_fine} vs {v_coarse}"
        );

        // Direct check: the top disc cap (a single closed circle, radius 2 at
        // z=12) contributes exactly (1/3)·12·π·4 via `planar_cap_signed_volume`.
        let top_cap = solid_faces(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&fid| {
                let f = topo.face(fid).unwrap();
                matches!(f.surface(), FaceSurface::Plane { .. })
                    && topo.wire(f.outer_wire()).unwrap().edges().iter().all(|oe| {
                        topo.vertex(topo.edge(oe.edge()).unwrap().start())
                            .unwrap()
                            .point()
                            .z()
                            > 11.0
                    })
            })
            .expect("top disc cap");
        let cap_v = planar_cap_signed_volume(&topo, top_cap, Point3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert!(
            (cap_v.abs() - 12.0 * PI * 4.0 / 3.0).abs() < 1e-9,
            "closed-circle disc cap contribution should be (1/3)·12·π·4, got {cap_v}"
        );
    }

    /// An ANNULAR planar cap (outer rim + a reversed inner-rim hole) must
    /// SUBTRACT the inner circular segment, giving area `π(R²−r²)`. The arc-bulge
    /// sweep must respect the oriented (reversed) inner rim — otherwise the inner
    /// segment is added (inflated area `π(R²+r²)`).
    #[test]
    fn annular_cap_volume_is_exact() {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        use std::f64::consts::PI;

        let (r_out, r_in, h) = (7.0_f64, 5.0_f64, 4.0_f64);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let mut topo = Topology::new();

        // Outer rim: a closed CCW circle at z = h, radius r_out.
        let v_out = topo.add_vertex(Vertex::new(Point3::new(r_out, 0.0, h), 1e-7));
        let outer_c = Circle3D::new(Point3::new(0.0, 0.0, h), axis, r_out).unwrap();
        let e_out = topo.add_edge(Edge::new(v_out, v_out, EdgeCurve::Circle(outer_c)));
        let outer_wire =
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e_out, true)], true).unwrap());

        // Inner rim (the hole): a closed circle at the same z, radius r_in, wound
        // OPPOSITE the outer wire — here via a reversed `OrientedEdge`.
        let v_in = topo.add_vertex(Vertex::new(Point3::new(r_in, 0.0, h), 1e-7));
        let inner_c = Circle3D::new(Point3::new(0.0, 0.0, h), axis, r_in).unwrap();
        let e_in = topo.add_edge(Edge::new(v_in, v_in, EdgeCurve::Circle(inner_c)));
        let inner_wire =
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e_in, false)], true).unwrap());

        let cap = topo.add_face(Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal: axis, d: h },
        ));

        let cap_v = planar_cap_signed_volume(&topo, cap, Point3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        // Exact annulus contribution: (1/3)·h·π·(R²−r²) (outward normal +axis).
        let expected = h * PI * (r_out * r_out - r_in * r_in) / 3.0;
        assert!(
            (cap_v.abs() - expected).abs() < 1e-9,
            "annular cap contribution should subtract the inner segment: \
             expected {expected}, got {cap_v} (inflated would be {})",
            h * PI * (r_out * r_out + r_in * r_in) / 3.0
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::primitives::{make_cone, make_cylinder};
    use brepkit_math::curves::{Circle3D, Ellipse3D};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// The patch `(0.01 u + 0.99 u², v, u²)` over the unit square, whose
    /// speed along `u` is 0.01 at `u = 0` though its rows run √2 long, read
    /// over the one triangle `(0, 0), (1, 0), (1, 1)`: no corner of it lies
    /// at a pole, and it matches the patch's flux over the triangle.
    #[test]
    fn a_slow_start_along_a_row_is_no_pole() {
        let net = |b: f64, c: f64| vec![Point3::new(b, 0.0, c), Point3::new(b, 1.0, c)];
        let surface = brepkit_math::nurbs::surface::NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![net(0.0, 0.0), net(0.005, 0.0), net(1.0, 1.0)],
            vec![vec![1.0; 2]; 3],
        )
        .unwrap();
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let positions: Vec<Point3> = uvs
            .iter()
            .map(|uv| surface.evaluate(uv[0], uv[1]))
            .collect();
        let mesh = tessellate::TriangleMeshUV {
            mesh: tessellate::TriangleMesh {
                normals: vec![Vec3::new(0.0, 0.0, 0.0); 3],
                positions,
                indices: vec![0, 1, 2],
            },
            uvs,
        };
        let about = Point3::new(0.0, 0.0, 0.0);
        let (read, _) = surface_flux_over_mesh(&mesh, Some(&surface), 0.05, about);
        // Twice the flux over `0 <= v <= u <= 1`, the integrand a polynomial
        // the rule takes exactly.
        let rule = brepkit_math::quadrature::gauss_legendre_points(16);
        let mut twice = 0.0;
        for gu in rule {
            let u = 0.5 * (gu.x + 1.0);
            for gv in rule {
                let v = 0.5 * u * (gv.x + 1.0);
                let (p, s_u, s_v) = surface.first_derivatives(u, v);
                twice += 2.0 * 0.25 * u * gu.w * gv.w * p.dot(s_u.cross(s_v));
            }
        }
        assert!(
            (read - twice).abs() < 1e-12,
            "read {read}, twice the flux {twice}"
        );
    }

    /// A frustum 3 to 1 across and 4 high, its wall in B-spline form (its
    /// rows doubled where the rational quarter circles join, so it is only
    /// C0 across those knot lines), its bottom rim chamfered. Each B-spline
    /// face of it read over a coarse mesh, triangle by triangle on the
    /// surface, matches its exact term: a triangle across a knot line is
    /// cut along it.
    #[test]
    fn a_coarse_mesh_across_a_b_spline_walls_knot_lines_reads_it_exactly() {
        use brepkit_heal::custom::convert_to_bspline::face_bspline_surface;
        let mut topo = Topology::new();
        let frustum = make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap();
        let rim = brepkit_topology::explorer::solid_edges(&topo, frustum)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(),
                    EdgeCurve::Circle(c) if c.center().z().abs() < 1e-9)
            })
            .unwrap();
        for face in brepkit_topology::explorer::solid_faces(&topo, frustum).unwrap() {
            if topo.face(face).unwrap().surface().is_planar() {
                continue;
            }
            let wall = face_bspline_surface(&topo, face).unwrap().unwrap();
            topo.face_mut(face)
                .unwrap()
                .set_surface(FaceSurface::Nurbs(wall));
        }
        let built = crate::blend_ops::chamfer_v2(&mut topo, frustum, &[rim], 0.4, 0.25).unwrap();
        let about = Point3::new(0.3, -0.2, 1.1);
        let mut read = 0;
        for face in brepkit_topology::explorer::solid_faces(&topo, built.solid).unwrap() {
            let FaceSurface::Nurbs(surface) = topo.face(face).unwrap().surface() else {
                continue;
            };
            let exact = brepkit_check::properties::exact_face_volume(&topo, face, about)
                .unwrap()
                .unwrap();
            let mesh = tessellate::tessellate_with_uvs(&topo, face, 0.05).unwrap();
            let (six, _) = surface_flux_over_mesh(&mesh, Some(surface), 0.05, about);
            assert!(
                (six / 6.0 - exact).abs() < 1e-7 * exact.abs(),
                "face {}: {}, exact {exact}",
                face.index(),
                six / 6.0
            );
            read += 1;
        }
        assert!(read >= 1);
    }

    /// The upper hemisphere less a notch over `x > 0`, `y > 0`, `z < 1`, with
    /// an exact equator: its outline runs along the sides of the `(u, v)` box
    /// of its vertices, but the face is the rest of the hemisphere around
    /// that box, so the closed form must decline it.
    #[test]
    fn sphere_box_declines_a_notched_hemisphere() {
        let mut topo = Topology::new();
        let r = 3.0_f64;
        let rim = (r * r - 1.0).sqrt();
        let o = Point3::new(0.0, 0.0, 0.0);
        let [d, a, b, c] = [
            Point3::new(0.0, r, 0.0),
            Point3::new(r, 0.0, 0.0),
            Point3::new(rim, 0.0, 1.0),
            Point3::new(0.0, rim, 1.0),
        ]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let arc = |topo: &mut Topology, from, to, centre, normal, radius| {
            let circle = Circle3D::new(centre, normal, radius).unwrap();
            topo.add_edge(Edge::new(from, to, EdgeCurve::Circle(circle)))
        };
        let up = Vec3::new(0.0, 0.0, 1.0);
        let edges = [
            arc(&mut topo, d, a, o, up, r),
            arc(&mut topo, a, b, o, Vec3::new(0.0, -1.0, 0.0), r),
            arc(&mut topo, b, c, Point3::new(0.0, 0.0, 1.0), up, rim),
            arc(&mut topo, c, d, o, Vec3::new(-1.0, 0.0, 0.0), r),
        ];
        let wire = Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let sphere = brepkit_math::surfaces::SphericalSurface::new(o, r).unwrap();
        let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Sphere(sphere)));
        assert!(
            analytic_sphere_signed_volume(&topo, face, o)
                .unwrap()
                .is_none()
        );
    }

    /// A hemisphere bounded the way a file's seamed sphere is, by its equator
    /// and a meridian seam the wire runs up and back down: the box of its
    /// full turn is the face, `2 pi R³ / 3` of flux about the centre.
    #[test]
    fn sphere_box_takes_a_seamed_hemisphere() {
        let mut topo = Topology::new();
        let r = 3.0_f64;
        let o = Point3::new(0.0, 0.0, 0.0);
        let rim = topo.add_vertex(Vertex::new(Point3::new(r, 0.0, 0.0), 1e-7));
        let pole = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, r), 1e-7));
        let equator = Circle3D::new(o, Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        let meridian = Circle3D::new(o, Vec3::new(0.0, -1.0, 0.0), r).unwrap();
        let equator = topo.add_edge(Edge::new(rim, rim, EdgeCurve::Circle(equator)));
        let seam = topo.add_edge(Edge::new(rim, pole, EdgeCurve::Circle(meridian)));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(equator, true),
                OrientedEdge::new(seam, true),
                OrientedEdge::new(seam, false),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let sphere = brepkit_math::surfaces::SphericalSurface::new(o, r).unwrap();
        let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Sphere(sphere)));
        let flux = analytic_sphere_signed_volume(&topo, face, o)
            .unwrap()
            .unwrap();
        let truth = 2.0 * std::f64::consts::PI * r.powi(3) / 3.0;
        assert!(
            (flux - truth).abs() < 1e-9 * truth,
            "flux {flux}, truth {truth}"
        );
    }

    /// The two sides of the latitude `z = 1` on a ball of radius 3, each a
    /// face bounded by that one circle run its own way: the cap above has
    /// `2 pi R (R - 1)` of the sphere and the rest `2 pi R (R + 1)`, so about
    /// the centre their fluxes are `R` times those areas over three. The
    /// loop's centroid sits on the axis above the centre for both, so only
    /// the loop's turn tells them apart. A reversed face (a pocket a Cut
    /// leaves) keeps its wire about the surface's outward normal and turns
    /// only its flux.
    #[test]
    fn sphere_box_reads_a_caps_side_from_its_turn() {
        use std::f64::consts::PI;
        let mut topo = Topology::new();
        let r = 3.0_f64;
        let o = Point3::new(0.0, 0.0, 0.0);
        let rim = (r * r - 1.0).sqrt();
        let v = topo.add_vertex(Vertex::new(Point3::new(rim, 0.0, 1.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), rim).unwrap();
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let sphere = brepkit_math::surfaces::SphericalSurface::new(o, r).unwrap();
        for (forward, side) in [
            (true, 2.0 * PI * r * (r - 1.0) * r / 3.0),
            (false, 2.0 * PI * r * (r + 1.0) * r / 3.0),
        ] {
            for reversed in [false, true] {
                let wire = Wire::new(vec![OrientedEdge::new(e, forward)], true).unwrap();
                let wid = topo.add_wire(wire);
                let surface = FaceSurface::Sphere(sphere.clone());
                let face = topo.add_face(if reversed {
                    Face::new_reversed(wid, vec![], surface)
                } else {
                    Face::new(wid, vec![], surface)
                });
                let flux = analytic_sphere_signed_volume(&topo, face, o)
                    .unwrap()
                    .unwrap();
                let truth = if reversed { -side } else { side };
                assert!(
                    (flux - truth).abs() < 1e-9 * side,
                    "forward {forward} reversed {reversed}: flux {flux}, truth {truth}"
                );
            }
        }
    }

    fn wall_flux(topo: &Topology, solid: SolidId) -> f64 {
        let wall = brepkit_topology::explorer::solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&f| !topo.face(f).unwrap().surface().is_planar())
            .unwrap();
        developable_face_flux(topo, wall, Point3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap()
    }

    /// The base disc sits at z = 0 and adds no flux, so the wall carries the
    /// whole volume. The seam is walked through the apex, which has no `u`.
    #[test]
    fn pointed_cone_wall_flux_is_its_volume() {
        let mut topo = Topology::new();
        let cone = make_cone(&mut topo, 3.0, 0.0, 6.0).unwrap();
        let flux = wall_flux(&topo, cone);
        let truth = std::f64::consts::PI * 9.0 * 6.0 / 3.0;
        assert!(
            (flux - truth).abs() < 1e-6 * truth,
            "flux {flux}, truth {truth}"
        );
    }

    fn planar_face(topo: &mut Topology, wires: Vec<Vec<OrientedEdge>>, d: f64) -> FaceId {
        let mut ids = wires
            .into_iter()
            .map(|edges| topo.add_wire(Wire::new(edges, true).unwrap()));
        let outer = ids.next().unwrap();
        let face = Face::new(
            outer,
            ids.collect(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d,
            },
        );
        topo.add_face(face)
    }

    fn closed_edge(topo: &mut Topology, curve: EdgeCurve, start: Point3) -> OrientedEdge {
        let v = topo.add_vertex(Vertex::new(start, 1e-7));
        OrientedEdge::new(topo.add_edge(Edge::new(v, v, curve)), true)
    }

    /// A whole ellipse, and half of one closed by its major axis: πab and
    /// πab / 2, times the plane offset over three.
    #[test]
    fn planar_flux_integrates_ellipses_exactly() {
        let mut topo = Topology::new();
        let e = Ellipse3D::new(
            Point3::new(1.0, -2.0, 3.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            2.0,
        )
        .unwrap();
        let rim = closed_edge(&mut topo, EdgeCurve::Ellipse(e.clone()), e.evaluate(0.0));
        let whole = planar_face(&mut topo, vec![vec![rim]], 3.0);
        // d / 3 = 1, so the flux equals the want.
        let want = std::f64::consts::PI * 6.0;
        let got = planar_face_flux(&topo, whole, Point3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert!(
            (got - want).abs() < 1e-12 * want,
            "whole ellipse {got}, want {want}"
        );

        let (a, b) = (
            topo.add_vertex(Vertex::new(e.evaluate(0.0), 1e-7)),
            topo.add_vertex(Vertex::new(e.evaluate(std::f64::consts::PI), 1e-7)),
        );
        let arc = topo.add_edge(Edge::new(a, b, EdgeCurve::Ellipse(e)));
        let chord = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        // The arc walked backwards and the chord forwards: the same half.
        let half = planar_face(
            &mut topo,
            vec![vec![
                OrientedEdge::new(chord, true),
                OrientedEdge::new(arc, false),
            ]],
            3.0,
        );
        let got = planar_face_flux(&topo, half, Point3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert!(
            (got - want / 2.0).abs() < 1e-12 * want,
            "half ellipse {got}, want {}",
            want / 2.0
        );
    }

    /// A four-span rational circle bounding a disc with an off-centre
    /// elliptical hole: πr² less πab, on a plane below the origin.
    #[test]
    fn planar_flux_integrates_nurbs_rims_and_holes() {
        let mut topo = Topology::new();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
        let nurbs = brepkit_geometry::convert::circle_to_nurbs_with_segments(
            &circle,
            0.0,
            std::f64::consts::TAU,
            4,
        )
        .unwrap();
        let start = nurbs.evaluate(nurbs.domain().0);
        let rim = closed_edge(&mut topo, EdgeCurve::NurbsCurve(nurbs), start);
        let e = Ellipse3D::new(
            Point3::new(1.0, 0.5, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.5,
            1.0,
        )
        .unwrap();
        let hole = closed_edge(&mut topo, EdgeCurve::Ellipse(e.clone()), e.evaluate(0.0));
        let face = planar_face(&mut topo, vec![vec![rim], vec![hole]], -1.0);
        let want = -(std::f64::consts::PI * (16.0 - 1.5)) / 3.0;
        let got = planar_face_flux(&topo, face, Point3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert!(
            (got - want).abs() < 1e-10 * want.abs(),
            "annulus {got}, want {want}"
        );
    }

    /// An island in a hole is material again; subtracting both would lose
    /// it, so the face keeps the mesh.
    #[test]
    fn planar_flux_leaves_nested_holes_to_the_mesh() {
        let mut topo = Topology::new();
        let ring = |r: f64| {
            Circle3D::new(Point3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap()
        };
        let wires = [5.0, 3.0, 1.0]
            .map(|r| {
                let c = ring(r);
                let start = c.evaluate(0.0);
                vec![closed_edge(&mut topo, EdgeCurve::Circle(c), start)]
            })
            .to_vec();
        let face = planar_face(&mut topo, wires, 2.0);
        assert!(
            planar_face_flux(&topo, face, Point3::new(0.0, 0.0, 0.0))
                .unwrap()
                .is_none()
        );
    }

    fn polygon_wire(topo: &mut Topology, points: &[Point3]) -> Vec<OrientedEdge> {
        let vs: Vec<_> = points
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        (0..vs.len())
            .map(|k| {
                let e = topo.add_edge(Edge::new(vs[k], vs[(k + 1) % vs.len()], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect()
    }

    /// A half disc: the arc of `circle` from angle `from` through a half turn,
    /// closed by its diameter.
    fn half_disc(topo: &mut Topology, circle: &Circle3D, from: f64) -> Vec<OrientedEdge> {
        let to = from + std::f64::consts::PI;
        let (a, b) = (
            topo.add_vertex(Vertex::new(circle.evaluate(from), 1e-7)),
            topo.add_vertex(Vertex::new(circle.evaluate(to), 1e-7)),
        );
        let arc = topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle.clone())));
        let chord = topo.add_edge(Edge::new(b, a, EdgeCurve::Line));
        vec![OrientedEdge::new(arc, true), OrientedEdge::new(chord, true)]
    }

    /// Two half-disc holes whose full circles overlap but whose arcs do not:
    /// the nesting guard must bound the arcs, not the circles, and keep the
    /// exact flux.
    #[test]
    fn planar_flux_bounds_arcs_by_their_span() {
        let mut topo = Topology::new();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let outer = polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 3.0),
                Point3::new(10.0, 0.0, 3.0),
                Point3::new(10.0, 10.0, 3.0),
                Point3::new(0.0, 10.0, 3.0),
            ],
        );
        let x = Vec3::new(1.0, 0.0, 0.0);
        let left = Circle3D::new_with_ref(Point3::new(3.0, 5.0, 3.0), z, 2.0, x).unwrap();
        let right = Circle3D::new_with_ref(Point3::new(4.0, 5.0, 3.0), z, 2.0, x).unwrap();
        let hole_left = half_disc(&mut topo, &left, std::f64::consts::FRAC_PI_2);
        let hole_right = half_disc(&mut topo, &right, -std::f64::consts::FRAC_PI_2);
        let face = planar_face(&mut topo, vec![outer, hole_left, hole_right], 3.0);
        let want = 3.0 * (100.0 - 4.0 * std::f64::consts::PI) / 3.0;
        let got = planar_face_flux(&topo, face, Point3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert!((got - want).abs() < 1e-12 * want, "flux {got}, want {want}");
    }

    /// A quad stored as a plane with one corner off it (a miter-sweep elbow)
    /// keeps the mesh: x·n is not the plane offset across it.
    #[test]
    fn planar_flux_leaves_skewed_faces_to_the_mesh() {
        let mut topo = Topology::new();
        let quad = polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 1.0),
                Point3::new(4.0, 0.0, 1.0),
                Point3::new(4.0, 4.0, 1.3),
                Point3::new(0.0, 4.0, 1.0),
            ],
        );
        let face = planar_face(&mut topo, vec![quad], 1.0);
        assert!(
            planar_face_flux(&topo, face, Point3::new(0.0, 0.0, 0.0))
                .unwrap()
                .is_none()
        );
    }

    /// A rational edge with a negative weight keeps the mesh, on the outer
    /// wire and on a hole alike.
    #[test]
    fn planar_flux_leaves_negative_weights_to_the_mesh() {
        let mut topo = Topology::new();
        let outer = polygon_wire(
            &mut topo,
            &[
                Point3::new(-5.0, -5.0, 0.0),
                Point3::new(15.0, -5.0, 0.0),
                Point3::new(15.0, 15.0, 0.0),
                Point3::new(-5.0, 15.0, 0.0),
            ],
        );
        let (a, b) = (Point3::new(0.0, 0.0, 0.0), Point3::new(4.0, 0.0, 0.0));
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![a, Point3::new(2.0, 3.0, 0.0), b],
            vec![1.0, -0.5, 1.0],
        )
        .unwrap();
        let (va, vb) = (
            topo.add_vertex(Vertex::new(a, 1e-7)),
            topo.add_vertex(Vertex::new(b, 1e-7)),
        );
        let arc = topo.add_edge(Edge::new(va, vb, EdgeCurve::NurbsCurve(curve)));
        let chord = topo.add_edge(Edge::new(vb, va, EdgeCurve::Line));
        let bulge = vec![OrientedEdge::new(arc, true), OrientedEdge::new(chord, true)];
        let square = polygon_wire(
            &mut topo,
            &[
                Point3::new(8.0, 8.0, 0.0),
                Point3::new(9.0, 8.0, 0.0),
                Point3::new(9.0, 9.0, 0.0),
                Point3::new(8.0, 9.0, 0.0),
            ],
        );
        let face = planar_face(&mut topo, vec![outer, bulge.clone(), square], 0.0);
        assert!(
            planar_face_flux(&topo, face, Point3::new(0.0, 0.0, 0.0))
                .unwrap()
                .is_none()
        );
        let alone = planar_face(&mut topo, vec![bulge], 0.0);
        assert!(
            planar_face_flux(&topo, alone, Point3::new(0.0, 0.0, 0.0))
                .unwrap()
                .is_none()
        );
    }

    /// The top cap at z = h carries a third of the volume; the wall the rest.
    #[test]
    fn cylinder_wall_flux_is_two_thirds_of_its_volume() {
        let mut topo = Topology::new();
        let cyl = make_cylinder(&mut topo, 1.5, 4.0).unwrap();
        let flux = wall_flux(&topo, cyl);
        let truth = 2.0 / 3.0 * std::f64::consts::PI * 1.5 * 1.5 * 4.0;
        assert!(
            (flux - truth).abs() < 1e-9 * truth,
            "flux {flux}, truth {truth}"
        );
    }

    /// A quarter wall of the unit cylinder, one unit tall, whose bottom rim is
    /// a spline running a quarter turn past each vertex with the vertices
    /// 3e-5 above it: the rim's span cannot be found, so the wall's extent
    /// comes from its vertices and it keeps a quarter turn.
    #[test]
    fn a_spline_rim_off_its_vertices_keeps_its_wall_a_quarter_turn() {
        use brepkit_math::surfaces::CylindricalSurface;
        use std::f64::consts::{FRAC_PI_2, PI};
        let mut topo = Topology::new();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let cyl = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z, 1.0).unwrap();
        let bottom = Circle3D::new(Point3::new(0.0, 0.0, 0.0), z, 1.0).unwrap();
        let top = Circle3D::new(Point3::new(0.0, 0.0, 1.0), z, 1.0).unwrap();
        let a0 = bottom.project(Point3::new(1.0, 0.0, 0.0));
        let spline =
            brepkit_geometry::convert::circle_to_nurbs(&bottom, a0 - FRAC_PI_2, a0 + PI).unwrap();
        let [b0, b90, t0, t90] = [
            Point3::new(1.0, 0.0, 3e-5),
            Point3::new(0.0, 1.0, 3e-5),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-4)));
        let rim = topo.add_edge(Edge::new(b0, b90, EdgeCurve::NurbsCurve(spline)));
        let right = topo.add_edge(Edge::new(b90, t90, EdgeCurve::Line));
        let top_arc = topo.add_edge(Edge::new(t0, t90, EdgeCurve::Circle(top)));
        let left = topo.add_edge(Edge::new(t0, b0, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(rim, true),
                OrientedEdge::new(right, true),
                OrientedEdge::new(top_arc, false),
                OrientedEdge::new(left, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let fid = topo.add_face(Face::new(wid, vec![], FaceSurface::Cylinder(cyl)));
        let vol = analytic_cylinder_signed_volume(&topo, fid, Point3::new(0.0, 0.0, 0.0)).unwrap();
        assert!(
            (vol - PI / 6.0).abs() < 1e-3,
            "flux {vol}, truth {}",
            PI / 6.0
        );
    }

    /// A band of revolution whose rim is split opposite its seam (the rim it
    /// shares with a neighbour whose own seam lies on the far side, as files
    /// write them) has two half arcs, and their midpoints, each taken the
    /// shorter way round, can both land on one side: the band then read as
    /// half a turn. Its whole circle says it runs the whole turn.
    fn split_rim_at_antipode(
        topo: &mut Topology,
        face_id: FaceId,
        rim: brepkit_topology::edge::EdgeId,
    ) -> FaceId {
        let face = topo.face(face_id).unwrap().clone();
        let wire = topo.wire(face.outer_wire()).unwrap().clone();
        let edge = topo.edge(rim).unwrap().clone();
        let EdgeCurve::Circle(circle) = edge.curve().clone() else {
            unreachable!("the rim is a circle");
        };
        // A hair past the antipode, as a file's coordinates leave it: one
        // half then runs a little over half a turn and the other a little
        // under.
        let start = topo.vertex(edge.start()).unwrap().point();
        let opposite = circle.evaluate(circle.project(start) + std::f64::consts::PI + 1e-6);
        let far = topo.add_vertex(Vertex::new(opposite, 1e-7));
        let out = topo.add_edge(Edge::new(
            edge.start(),
            far,
            EdgeCurve::Circle(circle.clone()),
        ));
        let back = topo.add_edge(Edge::new(far, edge.start(), EdgeCurve::Circle(circle)));
        let mut edges = Vec::new();
        for oe in wire.edges() {
            if oe.edge() != rim {
                edges.push(*oe);
            } else if oe.is_forward() {
                edges.extend([OrientedEdge::new(out, true), OrientedEdge::new(back, true)]);
            } else {
                edges.extend([
                    OrientedEdge::new(back, false),
                    OrientedEdge::new(out, false),
                ]);
            }
        }
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        topo.add_face(Face::new(wire, vec![], face.surface().clone()))
    }

    #[test]
    fn cone_band_with_a_rim_split_opposite_its_seam_reads_the_whole_turn() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 6.0, 4.0, 3.0).unwrap();
        let band = brepkit_topology::explorer::solid_faces(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
            .unwrap();
        let rim = topo
            .wire(topo.face(band).unwrap().outer_wire())
            .unwrap()
            .edges()
            .iter()
            .map(OrientedEdge::edge)
            .find(|&e| topo.edge(e).unwrap().is_closed())
            .unwrap();
        // About the base's centre the base adds nothing and the top a third
        // of its height times its area: the band has the rest of the
        // frustum, pi h (R^2 + R r) / 3.
        let about = Point3::new(0.0, 0.0, 0.0);
        let expected = std::f64::consts::PI * 3.0 * (36.0 + 24.0) / 3.0;
        let split = split_rim_at_antipode(&mut topo, band, rim);
        for face in [band, split] {
            let read = analytic_cone_signed_volume(&topo, face, about).unwrap();
            assert!(
                (read.abs() - expected).abs() <= 1e-9 * expected,
                "band reads {read}, expected {expected}"
            );
        }
    }

    /// A hole round the ring (the tube's outer equator) bounds no region on
    /// its own: the closed form declines the face rather than read the
    /// whole ring with the hole dropped.
    #[test]
    fn torus_closed_form_declines_a_hole_round_the_ring() {
        let (big, small) = (4.0, 1.0);
        let mut topo = Topology::new();
        let torus = crate::primitives::make_torus(&mut topo, big, small, 32).unwrap();
        let face = brepkit_topology::explorer::solid_faces(&topo, torus).unwrap()[0];
        let equator = Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            big + small,
        )
        .unwrap();
        let at = topo.add_vertex(Vertex::new(Point3::new(big + small, 0.0, 0.0), 1e-7));
        let edge = topo.add_edge(Edge::new(at, at, EdgeCurve::Circle(equator)));
        let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
        topo.face_mut(face).unwrap().inner_wires_mut().push(hole);
        let read = analytic_torus_signed_volume(&topo, face, Point3::new(0.0, 0.0, 0.0)).unwrap();
        assert!(read.is_none(), "read {read:?}");
    }

    #[test]
    fn torus_band_with_a_rim_split_opposite_its_seam_reads_the_whole_turn() {
        use brepkit_math::surfaces::ToroidalSurface;
        let mut topo = Topology::new();
        let (big, small, (v1, v2)) = (5.0, 1.0, (-1.0_f64, 1.0_f64));
        let centre = Point3::new(0.0, 0.0, 0.0);
        let torus = ToroidalSurface::new(centre, big, small).unwrap();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let parallel = |v: f64| {
            Circle3D::new(centre + z * (small * v.sin()), z, big + small * v.cos()).unwrap()
        };
        let [a1, a2] = [v1, v2].map(|v| topo.add_vertex(Vertex::new(torus.evaluate(0.0, v), 1e-7)));
        let low = topo.add_edge(Edge::new(a1, a1, EdgeCurve::Circle(parallel(v1))));
        let high = topo.add_edge(Edge::new(a2, a2, EdgeCurve::Circle(parallel(v2))));
        // The meridian at u = 0, run from v1 to v2 through the tube's outside.
        let meridian =
            Circle3D::new(Point3::new(big, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), small).unwrap();
        let seam = topo.add_edge(Edge::new(a1, a2, EdgeCurve::Circle(meridian)));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(low, true),
                    OrientedEdge::new(seam, true),
                    OrientedEdge::new(high, false),
                    OrientedEdge::new(seam, false),
                ],
                true,
            )
            .unwrap(),
        );
        let band = topo.add_face(Face::new(wire, vec![], FaceSurface::Torus(torus)));
        // About the torus's centre, P·N = R cos v + r and the area element
        // is r (R + r cos v) du dv.
        let about = centre;
        let expected = 2.0 * std::f64::consts::PI * small / 3.0
            * ((big * big + small * small) * (v2.sin() - v1.sin())
                + big * small * (0.5 * (v2 - v1) + 0.25 * ((2.0 * v2).sin() - (2.0 * v1).sin()))
                + big * small * (v2 - v1));
        let split = split_rim_at_antipode(&mut topo, band, low);
        for face in [band, split] {
            let read = analytic_torus_signed_volume(&topo, face, about)
                .unwrap()
                .unwrap();
            assert!(
                (read.abs() - expected).abs() <= 1e-9 * expected,
                "band reads {read}, expected {expected}"
            );
        }
    }

    /// A quarter cylinder as a rational NURBS patch, meshed on a coarse
    /// `(u, v)` grid whose bottom corners sit `drop` below the surface's own
    /// edge (as an imported face's edge can miss its surface).
    fn quarter_cylinder_mesh(
        radius: f64,
        height: f64,
        drop: f64,
    ) -> (
        brepkit_math::nurbs::surface::NurbsSurface,
        tessellate::TriangleMeshUV,
    ) {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let column = |x: f64, y: f64| vec![Point3::new(x, y, 0.0), Point3::new(x, y, height)];
        let surface = brepkit_math::nurbs::surface::NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                column(radius, 0.0),
                column(radius, radius),
                column(0.0, radius),
            ],
            vec![vec![1.0, 1.0], vec![w, w], vec![1.0, 1.0]],
        )
        .unwrap();
        let n = 4_u32;
        let mut mesh = tessellate::TriangleMeshUV::default();
        for j in 0..2_u32 {
            for i in 0..=n {
                let (u, v) = (f64::from(i) / f64::from(n), f64::from(j));
                let lift = if j == 0 { -drop } else { 0.0 };
                mesh.mesh
                    .positions
                    .push(surface.evaluate(u, v) + Vec3::new(0.0, 0.0, lift));
                mesh.mesh.normals.push(Vec3::new(0.0, 0.0, 0.0));
                mesh.uvs.push([u, v]);
            }
        }
        for i in 0..n {
            let (a, b, c, d) = (i, i + 1, n + 2 + i, n + 1 + i);
            mesh.mesh.indices.extend([a, b, c, a, c, d]);
        }
        (surface, mesh)
    }

    /// About a point on its axis a cylinder's flux density is its radius, so
    /// the patch's twice-flux is twice its radius times its area. The flat
    /// triangles inscribe it and fall short; the patch integrated on the
    /// surface does not, and it follows the mesh's corners where they leave
    /// the surface.
    #[test]
    fn nurbs_face_flux_is_integrated_on_its_surface() {
        let (radius, height) = (2.0, 3.0);
        let about = Point3::new(0.0, 0.0, 0.0);
        for (drop, on_surface) in [(0.0, true), (0.05, false)] {
            let (surface, mesh) = quarter_cylinder_mesh(radius, height, drop);
            let exact = 2.0 * radius * (0.5 * std::f64::consts::PI * radius) * (height + drop);
            let (flux, _) = surface_flux_over_mesh(&mesh, None, 0.05, about);
            assert!(
                (flux - exact).abs() > 1e-2 * exact,
                "the flat mesh reads {flux}, {exact}"
            );
            let (flux, moment) = surface_flux_over_mesh(&mesh, Some(&surface), 0.05, about);
            assert!(
                (flux - exact).abs() < 1e-5 * exact,
                "drop {drop}: patch reads {flux}, expected {exact}"
            );
            if on_surface {
                // Six times the flux of r (r·N) = R r: 6 R ∫ r dA.
                let (r, h) = (radius, height);
                let expected = Vec3::new(
                    r * r * h,
                    r * r * h,
                    0.25 * std::f64::consts::PI * r * h * h,
                ) * (6.0 * r);
                assert!(
                    (moment - expected).length() < 1e-5 * expected.length(),
                    "moment {moment:?}, expected {expected:?}"
                );
            }
        }
    }

    /// The cap under the cubic Bezier through (0, 0), (5, 20), (25, 20) and
    /// (30, 0), closed by its chord, holds 330, so extruded `h` it holds
    /// 330 h, read along its curved side's boundary however long it runs.
    #[test]
    fn a_bezier_caps_extrusion_holds_its_exact_volume() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        for h in [1.0, 100.0] {
            let mut topo = Topology::new();
            let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
            let a = topo.add_vertex(Vertex::new(p(0.0, 0.0), 1e-7));
            let b = topo.add_vertex(Vertex::new(p(30.0, 0.0), 1e-7));
            let bezier = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![p(0.0, 0.0), p(5.0, 20.0), p(25.0, 20.0), p(30.0, 0.0)],
                vec![1.0; 4],
            )
            .unwrap();
            let arch = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(bezier)));
            let chord = topo.add_edge(Edge::new(b, a, EdgeCurve::Line));
            let wire = topo.add_wire(
                Wire::new(
                    vec![
                        OrientedEdge::new(arch, true),
                        OrientedEdge::new(chord, true),
                    ],
                    true,
                )
                .unwrap(),
            );
            let cap = brepkit_topology::builder::make_face_from_wire(&mut topo, wire).unwrap();
            let solid =
                crate::extrude::extrude(&mut topo, cap, Vec3::new(0.0, 0.0, 1.0), h).unwrap();
            let volume = solid_volume(&topo, solid, 0.01).unwrap();
            assert!((volume - 330.0 * h).abs() < 1e-9 * 330.0 * h, "{volume}");
        }
    }

    /// A 10 square lofted smoothly through 17 copies turned an eighth of a
    /// turn over 40 holds 4000, every slice a 10 square: its lofted sides
    /// lie within a micron of the twisted square's, and their volume reads
    /// within 1e-8 of it.
    #[test]
    fn a_twisted_square_loft_holds_its_volume() {
        let mut topo = Topology::new();
        let faces: Vec<_> = (0..17)
            .map(|i| {
                let a = (45.0_f64 * f64::from(i) / 16.0).to_radians();
                let z = 40.0 * f64::from(i) / 16.0;
                let pts: Vec<Point3> = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
                    .iter()
                    .map(|&(x, y): &(f64, f64)| {
                        Point3::new(
                            x.madd(a.cos(), -y * a.sin()),
                            x.madd(a.sin(), y * a.cos()),
                            z,
                        )
                    })
                    .collect();
                brepkit_topology::builder::make_planar_face(&mut topo, &pts, 1e-7).unwrap()
            })
            .collect();
        let solid = crate::loft::loft_smooth(&mut topo, &faces).unwrap();
        let volume = solid_volume(&topo, solid, 0.01).unwrap();
        assert!((volume - 4000.0).abs() < 1e-8 * 4000.0, "{volume}");
    }
}
