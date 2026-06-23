//! Loft operation: create a solid by interpolating between profile faces.
//!
//! The loft connects two or more planar profiles by creating ruled (linear)
//! surfaces between corresponding profile edges.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::boolean::face_polygon;
use crate::dot_normal_point;
use crate::winding::ensure_ccw_profiles;

/// Resample a closed polygon to `target_count` evenly spaced points.
///
/// Distributes `target_count` points at equal arc-length intervals
/// along the polygon boundary.
#[allow(clippy::cast_precision_loss)]
fn resample_closed_polygon(points: &[Point3], target_count: usize) -> Vec<Point3> {
    let n = points.len();
    if n == 0 || target_count == 0 {
        return Vec::new();
    }
    // Compute cumulative arc lengths (closed: last segment wraps to first point)
    let mut cum_len = Vec::with_capacity(n + 1);
    cum_len.push(0.0);
    for i in 0..n {
        let next = (i + 1) % n;
        let seg = (points[next] - points[i]).length();
        cum_len.push(cum_len[i] + seg);
    }
    let total = *cum_len.last().unwrap_or(&0.0);
    if total < 1e-15 {
        return vec![points[0]; target_count];
    }

    let mut result = Vec::with_capacity(target_count);
    for i in 0..target_count {
        let target_len = total * (i as f64) / (target_count as f64);
        // Binary search for the segment containing target_len
        let seg = cum_len
            .partition_point(|&l| l < target_len)
            .saturating_sub(1)
            .min(n - 1);
        let seg_start = cum_len[seg];
        let seg_end = cum_len[seg + 1];
        let seg_len = seg_end - seg_start;
        let t = if seg_len > 1e-15 {
            (target_len - seg_start) / seg_len
        } else {
            0.0
        };
        let a = points[seg];
        let b = points[(seg + 1) % n];
        result.push(Point3::new(
            a.x() + t * (b.x() - a.x()),
            a.y() + t * (b.y() - a.y()),
            a.z() + t * (b.z() - a.z()),
        ));
    }
    result
}

/// Outward cap normal from the corrected (CCW-relative-to-stack) profile
/// vertices.
///
/// The stored profile-face plane normal cannot be trusted: callers may build
/// profiles with an arbitrary normal (e.g. always +Z) regardless of vertex
/// winding. Newell on the post-correction vertices points along the stacking
/// direction by construction, so the start cap (`inward = true`) is its
/// negation and the end cap is the normal itself.
fn cap_normal_from_verts(verts: &[Point3], inward: bool) -> Result<Vec3, crate::OperationsError> {
    let unit = crate::winding::newell_normal(verts).normalize()?;
    Ok(if inward { unit * -1.0 } else { unit })
}

/// Loft two or more profiles into a solid.
///
/// Each profile is a face; its surface may be planar or curved — only the
/// profile's boundary is used, so a section sketched on a cylinder, sphere,
/// cone, torus, or NURBS surface is supported. The loft connects corresponding
/// boundary vertices between adjacent profiles with ruled surfaces and closes
/// the first and last sections as end caps: a planar section boundary gets an
/// exact `Plane` cap, while a non-planar boundary is filled by a bilinear patch
/// through its corners. Profiles are resampled to a common vertex count when
/// they differ. Inner wires (holes) in profiles are ignored.
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 2 profiles are provided
/// - Profiles resample to fewer than 3 vertices
/// - A section boundary is non-planar with more than 4 edges (unsupported cap)
#[allow(clippy::too_many_lines)]
pub fn loft(topo: &mut Topology, profiles: &[FaceId]) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if profiles.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft requires at least 2 profiles".into(),
        });
    }

    // Fast path: lofting a stack of coaxial circles (incl. NURBS-recognized
    // circles produced by brepjs `sketchCircle`) collapses to an exact
    // sequence of cylinder / cone / frustum bands. The general path
    // tessellates the circles into N line segments, losing 0.6-1% of the
    // true π·r²·h (or frustum) volume per band.
    if let Some(stack_solid) = try_loft_coaxial_circle_stack(topo, profiles)? {
        return Ok(stack_solid);
    }

    // Curve-preserving path: when all profiles share the same boundary edge
    // structure (same edge count and per-edge curve type) and wind CCW, build
    // ruled surfaces per corresponding edge pair so curved edges stay curved —
    // a rounded-rect loft becomes a true rounded-rect frustum (arc corners),
    // not an octagon. The general polygon path below handles everything else.
    if let Some(curved_solid) = try_loft_matching_curved_profiles(topo, profiles)? {
        return Ok(curved_solid);
    }

    let mut profile_verts: Vec<Vec<Point3>> = Vec::with_capacity(profiles.len());
    for &fid in profiles {
        let verts = face_polygon(topo, fid)?;
        profile_verts.push(verts);
    }

    // Resample all profiles to the maximum vertex count so that lofting
    // between different-resolution profiles (e.g. rectangle ↔ circle) works.
    let n = profile_verts.iter().map(Vec::len).max().unwrap_or(0);
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft profiles must have at least 3 vertices".into(),
        });
    }
    for verts in &mut profile_verts {
        if verts.len() != n {
            *verts = resample_closed_polygon(verts, n);
        }
    }

    // Ensure profile vertex winding is CCW relative to the stacking direction.
    // The side normal formula `edge_dir.cross(connect_dir)` gives outward normals
    // only when vertices go CCW from the stacking direction.
    let _ = ensure_ccw_profiles(&mut profile_verts);

    let num_profiles = profile_verts.len();
    let num_sections = num_profiles - 1;

    let ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> = profile_verts
        .iter()
        .map(|verts| {
            verts
                .iter()
                .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
                .collect()
        })
        .collect();

    let ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = ring_verts
        .iter()
        .map(|ring| {
            (0..n)
                .map(|i| {
                    let next = (i + 1) % n;
                    topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                })
                .collect()
        })
        .collect();

    let connect_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = (0..num_sections)
        .map(|s| {
            (0..n)
                .map(|i| {
                    topo.add_edge(Edge::new(
                        ring_verts[s][i],
                        ring_verts[s + 1][i],
                        EdgeCurve::Line,
                    ))
                })
                .collect()
        })
        .collect();

    let mut all_faces = Vec::new();

    // Start cap: reversed first profile (outward normal pointing away from loft).
    {
        let cap_normal = cap_normal_from_verts(&profile_verts[0], true)?;
        all_faces.push(crate::cap::build_cap_face(
            topo,
            &ring_edges[0],
            vec![],
            &profile_verts[0],
            cap_normal,
            true,
        )?);
    }

    // Side faces: one quad per profile-edge × section.
    for s in 0..num_sections {
        for i in 0..n {
            let next_i = (i + 1) % n;

            // Quad: ring[s][i] → ring[s][next_i] → ring[s+1][next_i] → ring[s+1][i]
            let p0 = profile_verts[s][i];
            let p1 = profile_verts[s][next_i];
            let p_next = profile_verts[s + 1][i];
            let edge_dir = p1 - p0;
            let connect_dir = p_next - p0;
            let side_normal = edge_dir
                .cross(connect_dir)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let side_d = dot_normal_point(side_normal, p0);

            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(ring_edges[s][i], true),
                    OrientedEdge::new(connect_edges[s][next_i], true),
                    OrientedEdge::new(ring_edges[s + 1][i], false),
                    OrientedEdge::new(connect_edges[s][i], false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?;

            let side_wire_id = topo.add_wire(side_wire);
            let side_face = topo.add_face(Face::new(
                side_wire_id,
                vec![],
                FaceSurface::Plane {
                    normal: side_normal,
                    d: side_d,
                },
            ));
            all_faces.push(side_face);
        }
    }

    // End cap: last profile with forward orientation.
    {
        let last = num_profiles - 1;
        let cap_normal = cap_normal_from_verts(&profile_verts[last], false)?;
        all_faces.push(crate::cap::build_cap_face(
            topo,
            &ring_edges[last],
            vec![],
            &profile_verts[last],
            cap_normal,
            false,
        )?);
    }

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Detect "loft across a stack of coaxial circles" and produce an exact
/// chain of cylinder / cone / frustum bands. Returns `Ok(None)` when any
/// profile is not a recognized circle, when the circles don't share a
/// common axis, or when adjacent centers coincide — the general loft then
/// handles the case.
///
/// All circle centers must lie on one line whose direction is parallel to
/// every circle's plane normal, and consecutive centers must be ordered
/// monotonically along that line (no zero-height bands).
fn try_loft_coaxial_circle_stack(
    topo: &mut Topology,
    profiles: &[FaceId],
) -> Result<Option<SolidId>, crate::OperationsError> {
    let tol = Tolerance::new();

    let mut circles: Vec<(Point3, Vec3, f64)> = Vec::with_capacity(profiles.len());
    for &fid in profiles {
        match face_recognized_circle(topo, fid) {
            Some(c) => circles.push(c),
            None => return Ok(None),
        }
    }

    let (center_0, _, _) = circles[0];
    let (center_1, _, _) = circles[1];
    let axis = center_1 - center_0;
    let axis_len = axis.length();
    if axis_len < tol.linear {
        return Ok(None);
    }
    let axis_unit = axis * (1.0 / axis_len);

    // Every circle's normal must be parallel to the stacking axis, every
    // center must lie on the axis line, and the signed axial positions must
    // be strictly increasing (so each band has positive height).
    let mut axial = Vec::with_capacity(circles.len());
    let mut prev_t = f64::NEG_INFINITY;
    for &(center, normal, _) in &circles {
        if normal.dot(axis_unit).abs() <= 1.0 - tol.angular {
            return Ok(None);
        }
        let rel = center - center_0;
        let t = rel.dot(axis_unit);
        // Reject lateral offset from the axis line.
        let radial = rel - axis_unit * t;
        if radial.length() > tol.linear {
            return Ok(None);
        }
        if t <= prev_t + tol.linear {
            return Ok(None);
        }
        prev_t = t;
        axial.push(t);
    }

    let radii: Vec<f64> = circles.iter().map(|&(_, _, r)| r).collect();
    let solid = build_coaxial_band_stack(topo, &axial, &radii)?;

    let z_axis = Vec3::new(0.0, 0.0, 1.0);
    if (z_axis - axis_unit).length() > tol.linear {
        let rot_axis = z_axis.cross(axis_unit);
        let rot_axis_len = rot_axis.length();
        let mat = if rot_axis_len < tol.linear {
            brepkit_math::mat::Mat4::rotation_x(std::f64::consts::PI)
        } else {
            let angle = z_axis.dot(axis_unit).clamp(-1.0, 1.0).acos();
            rodrigues_rotation(rot_axis * (1.0 / rot_axis_len), angle)
        };
        crate::transform::transform_solid(topo, solid, &mat)?;
    }
    if center_0.x().abs() > tol.linear
        || center_0.y().abs() > tol.linear
        || center_0.z().abs() > tol.linear
    {
        let xform = brepkit_math::mat::Mat4::translation(center_0.x(), center_0.y(), center_0.z());
        crate::transform::transform_solid(topo, solid, &xform)?;
    }
    Ok(Some(solid))
}

/// Per-edge geometry of a profile's outer wire in traversal order: the curve
/// and its wire-oriented start/end points (so `start`→`end` follows the wire).
struct ProfileEdgeGeom {
    curve: EdgeCurve,
    start: Point3,
    end: Point3,
}

/// Merge consecutive arcs lying on the same circle into a single arc.
///
/// A 90° rounded-rect corner can arrive as one arc or as several co-circular
/// sub-arcs — drawn profiles (`drawRoundedRectangle`) split corners
/// inconsistently with size (e.g. one section's corner is one arc, another's is
/// two). Left as-is, the loft profiles get mismatched edge counts and the
/// curve-preserving path bails to the faceted polygon loft. Normalizing each
/// profile to canonical one-arc-per-corner lets split- and single-arc corners
/// align so multi-section lips/sockets keep analytic corners.
fn merge_cocircular_arcs(edges: Vec<ProfileEdgeGeom>) -> Vec<ProfileEdgeGeom> {
    use brepkit_math::curves::Circle3D;
    let tol = Tolerance::new();
    let same_circle = |c0: &Circle3D, c1: &Circle3D| -> bool {
        let (Ok(n0), Ok(n1)) = (c0.normal().normalize(), c1.normal().normalize()) else {
            return false;
        };
        n0.dot(n1).abs() >= 1.0 - tol.angular
            && (c0.center() - c1.center()).length() <= tol.linear
            && (c0.radius() - c1.radius()).abs() <= tol.linear
    };
    let mut out: Vec<ProfileEdgeGeom> = Vec::with_capacity(edges.len());
    for e in edges {
        let merge = matches!(&e.curve, EdgeCurve::Circle(_))
            && out.last().is_some_and(|last| {
                matches!(
                    (&last.curve, &e.curve),
                    (EdgeCurve::Circle(c0), EdgeCurve::Circle(c1)) if same_circle(c0, c1)
                )
            });
        if merge && let Some(last) = out.last_mut() {
            last.end = e.end;
        } else {
            out.push(e);
        }
    }
    // Wrap-around: a corner split across the closing seam (leading & trailing
    // arcs share a circle). Fold the trailing arc back into the leading one.
    if out.len() > 2 {
        let wrap = matches!(
            (&out[0].curve, &out[out.len() - 1].curve),
            (EdgeCurve::Circle(cf), EdgeCurve::Circle(cl)) if same_circle(cf, cl)
        );
        if wrap
            && let Some(last) = out.pop()
            && let Some(first) = out.first_mut()
        {
            first.start = last.start;
        }
    }
    out
}

/// Extract a planar profile's outer-wire edges in traversal order, each
/// oriented so `start`→`end` follows the wire. Returns `None` if the face is
/// not planar or has inner wires (holes) — the general loft handles those.
fn profile_oriented_edges(topo: &Topology, fid: FaceId) -> Option<Vec<ProfileEdgeGeom>> {
    let face = topo.face(fid).ok()?;
    if !matches!(face.surface(), FaceSurface::Plane { .. }) || !face.inner_wires().is_empty() {
        return None;
    }
    let wire = topo.wire(face.outer_wire()).ok()?;
    let oes = wire.edges();
    if oes.len() < 3 {
        return None;
    }
    let mut out = Vec::with_capacity(oes.len());
    for oe in oes {
        let edge = topo.edge(oe.edge()).ok()?;
        let s = topo.vertex(edge.start()).ok()?.point();
        let e = topo.vertex(edge.end()).ok()?.point();
        let (start, end) = if oe.is_forward() { (s, e) } else { (e, s) };
        out.push(ProfileEdgeGeom {
            curve: edge.curve().clone(),
            start,
            end,
        });
    }
    Some(merge_cocircular_arcs(out))
}

/// When two corner arcs are coaxial (same axis direction, centers differing
/// only along that axis), the ruled surface between them is an exact analytic
/// `Cylinder` (equal radii) or `Cone` (unequal radii) — which downstream
/// booleans handle far more robustly than a generic ruled NURBS. Mirrors the
/// reference `BRepFill_Generator`. Returns `None` for non-coaxial arcs (caller
/// falls back to a ruled NURBS).
fn coaxial_corner_surface(
    c0: &brepkit_math::curves::Circle3D,
    c1: &brepkit_math::curves::Circle3D,
    tol: Tolerance,
) -> Option<FaceSurface> {
    let n = c0.normal().normalize().ok()?;
    // Both arcs must share the same axis direction.
    if (c1.normal().normalize().ok()?.dot(n)).abs() < 1.0 - tol.angular {
        return None;
    }
    let (o0, o1) = (c0.center(), c1.center());
    let (r0, r1) = (c0.radius(), c1.radius());
    let d = o1 - o0;
    let height = d.dot(n);
    // Centers must be coaxial (no lateral offset) and the band non-degenerate.
    if (d - n * height).length() > tol.linear || height.abs() < tol.linear {
        return None;
    }
    if (r0 - r1).abs() < tol.linear {
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(o0, n, r0).ok()?;
        return Some(FaceSurface::Cylinder(cyl));
    }
    // Cone: apex on the axis through the corner center where the generator
    // radius reaches zero (same construction as `build_coaxial_band_stack`,
    // offset to the corner center instead of the global axis).
    let s_apex = -r0 * height / (r1 - r0);
    let apex = o0 + n * s_apex;
    let (axis_sign, r_ref, axial_to_ref) = if r1 > r0 {
        (1.0_f64, r1, height - s_apex)
    } else {
        (-1.0_f64, r0, s_apex)
    };
    let half_angle = axial_to_ref.abs().atan2(r_ref);
    let cone = brepkit_math::surfaces::ConicalSurface::new(apex, n * axis_sign, half_angle).ok()?;
    Some(FaceSurface::Cone(cone))
}

/// Build a degree-(1, p) ruled NURBS surface between two circular arcs, each
/// reconstructed from its oriented endpoints (via the same `domain_with_endpoints`
/// the ring edges use, so the surface matches them exactly). Returns `None` if
/// the two arcs convert to incompatible NURBS (different degree or
/// control-point count) — the caller then falls back to the polygon loft.
fn ruled_arc_surface(
    c0: &brepkit_math::curves::Circle3D,
    p0s: Point3,
    p0e: Point3,
    c1: &brepkit_math::curves::Circle3D,
    p1s: Point3,
    p1e: Point3,
) -> Option<NurbsSurface> {
    let (a0, a0e) = EdgeCurve::Circle(c0.clone()).domain_with_endpoints(p0s, p0e);
    let (a1, a1e) = EdgeCurve::Circle(c1.clone()).domain_with_endpoints(p1s, p1e);
    let nc0 = brepkit_geometry::convert::circle_to_nurbs(c0, a0, a0e).ok()?;
    let nc1 = brepkit_geometry::convert::circle_to_nurbs(c1, a1, a1e).ok()?;
    if nc0.degree() != nc1.degree() || nc0.control_points().len() != nc1.control_points().len() {
        return None;
    }
    NurbsSurface::new(
        1,
        nc0.degree(),
        vec![0.0, 0.0, 1.0, 1.0],
        nc0.knots().to_vec(),
        vec![nc0.control_points().to_vec(), nc1.control_points().to_vec()],
        vec![nc0.weights().to_vec(), nc1.weights().to_vec()],
    )
    .ok()
}

/// Loft profiles that share the same boundary edge structure while preserving
/// curved edges (arcs stay arcs via ruled NURBS side faces) instead of
/// faceting them. Returns `None` (fall back to the polygon path) unless every
/// profile is planar, hole-free, winds CCW about the stacking direction, has
/// the same edge count, the i-th edge has the same curve type in every profile,
/// at least one edge is curved, and all edges are `Line`/`Circle`.
#[allow(clippy::too_many_lines)]
fn try_loft_matching_curved_profiles(
    topo: &mut Topology,
    profiles: &[FaceId],
) -> Result<Option<SolidId>, crate::OperationsError> {
    let tol = Tolerance::new();
    let num_profiles = profiles.len();

    // Curve-preserve any chain of N >= 2 profiles. The section loops below
    // (steps 3b/4/6) already build one ruled band per adjacent profile pair, so
    // a multi-section gridfinity lip/socket keeps its arc corners analytic
    // (Cylinder/Cone/ruled-NURBS) instead of faceting to a polygon. An analytic
    // operand is exactly what the downstream cut/fuse needs to stay watertight —
    // a faceted multi-section lip is what drove the boolean to its non-manifold
    // mesh fallback. `loft()` already guarantees >= 2 profiles.
    if num_profiles < 2 {
        return Ok(None);
    }

    // 1. Extract every profile's ordered oriented edges.
    let mut profs: Vec<Vec<ProfileEdgeGeom>> = Vec::with_capacity(num_profiles);
    for &fid in profiles {
        match profile_oriented_edges(topo, fid) {
            Some(edges) => profs.push(edges),
            None => return Ok(None),
        }
    }

    // 2. Same edge count + matching per-edge curve type across all profiles.
    let n = profs[0].len();
    let kind = |c: &EdgeCurve| match c {
        EdgeCurve::Line => 0u8,
        EdgeCurve::Circle(_) => 1,
        EdgeCurve::Ellipse(_) => 2,
        EdgeCurve::NurbsCurve(_) => 3,
    };
    for p in &profs {
        if p.len() != n || (0..n).any(|i| kind(&p[i].curve) != kind(&profs[0][i].curve)) {
            return Ok(None);
        }
    }
    // Only Line/Circle edges are handled, and only worth it if some edge is
    // curved (an all-Line profile gives the identical polygon result).
    let all_line = profs[0].iter().all(|e| matches!(e.curve, EdgeCurve::Line));
    let unsupported = profs[0]
        .iter()
        .any(|e| !matches!(e.curve, EdgeCurve::Line | EdgeCurve::Circle(_)));
    if all_line || unsupported {
        return Ok(None);
    }

    // 3. Stacking direction + CCW gate. Each profile's junction points (edge
    //    starts) must wind CCW about the stacking axis so corner arcs stay
    //    convex and cap normals point outward.
    let junctions = |p: &[ProfileEdgeGeom]| -> Vec<Point3> { p.iter().map(|e| e.start).collect() };
    #[allow(clippy::cast_precision_loss)]
    let centroid = |pts: &[Point3]| -> Point3 {
        let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
        for p in pts {
            x += p.x();
            y += p.y();
            z += p.z();
        }
        let inv = 1.0 / pts.len() as f64;
        Point3::new(x * inv, y * inv, z * inv)
    };
    let axis = centroid(&junctions(&profs[num_profiles - 1])) - centroid(&junctions(&profs[0]));
    if axis.length() < tol.linear {
        return Ok(None);
    }
    for p in &profs {
        if crate::winding::newell_normal(&junctions(p)).dot(axis) <= 0.0 {
            return Ok(None);
        }
    }

    // 3b. Pre-compute every side face's surface BEFORE mutating `topo`. A
    //     Circle-pair whose arcs convert to incompatible NURBS makes the whole
    //     loft fall back to the polygon path; doing it here means that `None`
    //     return leaves no orphaned vertices/edges/faces behind.
    let mut side_surfaces: Vec<(FaceSurface, bool)> = Vec::with_capacity((num_profiles - 1) * n);
    for s in 0..num_profiles - 1 {
        for i in 0..n {
            let (p0s, p0e) = (profs[s][i].start, profs[s][i].end);
            let p1s = profs[s + 1][i].start;
            let outward = (p0e - p0s).cross(p1s - p0s);
            let entry = match (&profs[s][i].curve, &profs[s + 1][i].curve) {
                (EdgeCurve::Line, EdgeCurve::Line) => {
                    let normal = outward.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                    (
                        FaceSurface::Plane {
                            normal,
                            d: dot_normal_point(normal, p0s),
                        },
                        false,
                    )
                }
                (EdgeCurve::Circle(c0), EdgeCurve::Circle(c1)) => {
                    if let Some(surface) = coaxial_corner_surface(c0, c1, tol) {
                        // Cylinder/Cone share `build_coaxial_band_stack`'s
                        // radial-outward normal convention with this wire
                        // winding, so no reversal is needed.
                        (surface, false)
                    } else {
                        let Some(surf) =
                            ruled_arc_surface(c0, p0s, p0e, c1, p1s, profs[s + 1][i].end)
                        else {
                            return Ok(None);
                        };
                        let reversed = surf
                            .normal(0.5, 0.5)
                            .map(|nrm| nrm.dot(outward) < 0.0)
                            .unwrap_or(false);
                        (FaceSurface::Nurbs(surf), reversed)
                    }
                }
                _ => return Ok(None),
            };
            side_surfaces.push(entry);
        }
    }

    // 4. Ring vertices, ring edges (curve-preserving), connecting edges.
    let ring_vids: Vec<Vec<VertexId>> = profs
        .iter()
        .map(|p| {
            p.iter()
                .map(|e| topo.add_vertex(Vertex::new(e.start, tol.linear)))
                .collect()
        })
        .collect();
    let ring_eids: Vec<Vec<EdgeId>> = (0..num_profiles)
        .map(|s| {
            (0..n)
                .map(|i| {
                    topo.add_edge(Edge::new(
                        ring_vids[s][i],
                        ring_vids[s][(i + 1) % n],
                        profs[s][i].curve.clone(),
                    ))
                })
                .collect()
        })
        .collect();
    let conn_eids: Vec<Vec<EdgeId>> = (0..num_profiles - 1)
        .map(|s| {
            (0..n)
                .map(|i| {
                    topo.add_edge(Edge::new(
                        ring_vids[s][i],
                        ring_vids[s + 1][i],
                        EdgeCurve::Line,
                    ))
                })
                .collect()
        })
        .collect();

    let mut all_faces = Vec::new();

    // 5. Start cap (reversed first profile → outward normal away from stack).
    {
        let jn = junctions(&profs[0]);
        let cap_normal = cap_normal_from_verts(&jn, true)?;
        let edges: Vec<OrientedEdge> = (0..n)
            .rev()
            .map(|i| OrientedEdge::new(ring_eids[0][i], false))
            .collect();
        let wid = topo.add_wire(Wire::new(edges, true).map_err(crate::OperationsError::Topology)?);
        let cap_d = dot_normal_point(cap_normal, jn[0]);
        all_faces.push(topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        )));
    }

    // 6. Side faces: one per corresponding edge pair per section, using the
    //    surfaces pre-computed in step 3b.
    for s in 0..num_profiles - 1 {
        for i in 0..n {
            let next_i = (i + 1) % n;
            let side_wire_id = topo.add_wire(
                Wire::new(
                    vec![
                        OrientedEdge::new(ring_eids[s][i], true),
                        OrientedEdge::new(conn_eids[s][next_i], true),
                        OrientedEdge::new(ring_eids[s + 1][i], false),
                        OrientedEdge::new(conn_eids[s][i], false),
                    ],
                    true,
                )
                .map_err(crate::OperationsError::Topology)?,
            );
            let (surface, reversed) = side_surfaces[s * n + i].clone();
            let mut face = Face::new(side_wire_id, vec![], surface);
            if reversed {
                face.set_reversed(true);
            }
            all_faces.push(topo.add_face(face));
        }
    }

    // 7. End cap (last profile, forward).
    {
        let jn = junctions(&profs[num_profiles - 1]);
        let cap_normal = cap_normal_from_verts(&jn, false)?;
        let edges: Vec<OrientedEdge> = (0..n)
            .map(|i| OrientedEdge::new(ring_eids[num_profiles - 1][i], true))
            .collect();
        let wid = topo.add_wire(Wire::new(edges, true).map_err(crate::OperationsError::Topology)?);
        let cap_d = dot_normal_point(cap_normal, jn[0]);
        all_faces.push(topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        )));
    }

    // 8. Assemble.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(Some(topo.add_solid(Solid::new(shell_id, vec![]))))
}

/// Rotation matrix around an arbitrary unit axis by `angle` radians
/// (Rodrigues' formula). Duplicates `pattern::rotation_matrix` so loft
/// stays self-contained.
fn rodrigues_rotation(axis: Vec3, angle: f64) -> brepkit_math::mat::Mat4 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let omc = 1.0 - cos_a;
    let (ax, ay, az) = (axis.x(), axis.y(), axis.z());
    brepkit_math::mat::Mat4([
        [
            omc.mul_add(ax * ax, cos_a),
            ax.mul_add(ay * omc, -(sin_a * az)),
            ax.mul_add(az * omc, sin_a * ay),
            0.0,
        ],
        [
            ax.mul_add(ay * omc, sin_a * az),
            omc.mul_add(ay * ay, cos_a),
            ay.mul_add(az * omc, -(sin_a * ax)),
            0.0,
        ],
        [
            ax.mul_add(az * omc, -(sin_a * ay)),
            ay.mul_add(az * omc, sin_a * ax),
            omc.mul_add(az * az, cos_a),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// Build a watertight stack of analytic cylinder/cone bands along +Z.
///
/// `axial[k]` is the z-height of ring `k`, `radii[k]` its circle radius.
/// Adjacent rings are connected by a cylindrical patch (equal radii) or a
/// conical patch (differing radii). Ring circle edges are shared between
/// neighbouring bands (and the two end caps) for watertight topology.
#[allow(clippy::too_many_lines)]
fn build_coaxial_band_stack(
    topo: &mut Topology,
    axial: &[f64],
    radii: &[f64],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    let z_axis = Vec3::new(0.0, 0.0, 1.0);

    // One shared circle edge per ring (degenerate seam vertex at angle 0).
    let mut ring_edges = Vec::with_capacity(axial.len());
    let mut ring_seam_verts = Vec::with_capacity(axial.len());
    for (&z, &r) in axial.iter().zip(radii.iter()) {
        let circle = brepkit_math::curves::Circle3D::new(Point3::new(0.0, 0.0, z), z_axis, r)
            .map_err(crate::OperationsError::Math)?;
        let seam_v = topo.add_vertex(Vertex::new(Point3::new(r, 0.0, z), tol.linear));
        let e = topo.add_edge(Edge::new(seam_v, seam_v, EdgeCurve::Circle(circle)));
        ring_edges.push(e);
        ring_seam_verts.push(seam_v);
    }

    let mut faces = Vec::new();

    for band in 0..axial.len() - 1 {
        let (z0, z1) = (axial[band], axial[band + 1]);
        let (r0, r1) = (radii[band], radii[band + 1]);
        let height = z1 - z0;

        let seam = topo.add_edge(Edge::new(
            ring_seam_verts[band],
            ring_seam_verts[band + 1],
            EdgeCurve::Line,
        ));

        let surface = if (r0 - r1).abs() < tol.linear {
            let cyl = brepkit_math::surfaces::CylindricalSurface::new(
                Point3::new(0.0, 0.0, z0),
                z_axis,
                r0,
            )
            .map_err(crate::OperationsError::Math)?;
            FaceSurface::Cylinder(cyl)
        } else {
            // Virtual apex where the band's generator reaches radius zero.
            let z_apex = z0 - r0 * height / (r1 - r0);
            // Axis points apex → larger-radius end so the v>0 generator
            // sweeps outward; half-angle is measured from the radial plane.
            let (apex_z, axis_sign, r_ref, axial_to_ref) = if r1 > r0 {
                (z_apex, 1.0_f64, r1, z1 - z_apex)
            } else {
                (z_apex, -1.0_f64, r0, z_apex - z0)
            };
            let half_angle = axial_to_ref.abs().atan2(r_ref);
            let cone = brepkit_math::surfaces::ConicalSurface::new(
                Point3::new(0.0, 0.0, apex_z),
                Vec3::new(0.0, 0.0, axis_sign),
                half_angle,
            )
            .map_err(crate::OperationsError::Math)?;
            FaceSurface::Cone(cone)
        };

        let lateral_wire = Wire::new(
            vec![
                OrientedEdge::new(ring_edges[band], true),
                OrientedEdge::new(seam, true),
                OrientedEdge::new(ring_edges[band + 1], false),
                OrientedEdge::new(seam, false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;
        let lateral_wid = topo.add_wire(lateral_wire);
        faces.push(topo.add_face(Face::new(lateral_wid, vec![], surface)));
    }

    // Bottom cap (reversed first ring edge → outward normal -Z).
    let bot_wire = Wire::new(vec![OrientedEdge::new(ring_edges[0], false)], true)
        .map_err(crate::OperationsError::Topology)?;
    let bot_wid = topo.add_wire(bot_wire);
    faces.push(topo.add_face(Face::new(
        bot_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -axial[0],
        },
    )));

    // Top cap (forward last ring edge → outward normal +Z).
    let last = axial.len() - 1;
    let top_wire = Wire::new(vec![OrientedEdge::new(ring_edges[last], true)], true)
        .map_err(crate::OperationsError::Topology)?;
    let top_wid = topo.add_wire(top_wire);
    faces.push(topo.add_face(Face::new(
        top_wid,
        vec![],
        FaceSurface::Plane {
            normal: z_axis,
            d: axial[last],
        },
    )));

    let shell = Shell::new(faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Recognize a planar face as a circle (center, plane normal, radius).
/// Returns `None` when the face's outer wire isn't a single closed
/// circular edge (or NURBS-recognized circular edge), or when the
/// surface isn't planar.
fn face_recognized_circle(topo: &Topology, face_id: FaceId) -> Option<(Point3, Vec3, f64)> {
    let face = topo.face(face_id).ok()?;
    if !face.inner_wires().is_empty() {
        return None;
    }
    let normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return None,
    };
    let wire = topo.wire(face.outer_wire()).ok()?;
    let edges = wire.edges();
    if edges.len() != 1 {
        return None;
    }
    let edge = topo.edge(edges[0].edge()).ok()?;
    if edge.start() != edge.end() {
        return None; // not a closed-loop circle
    }
    match edge.curve() {
        brepkit_topology::edge::EdgeCurve::Circle(c) => Some((c.center(), normal, c.radius())),
        brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => {
            let tol = Tolerance::new().linear;
            match brepkit_geometry::convert::recognize_curve(nc, tol * 100.0) {
                brepkit_geometry::convert::RecognizedCurve::Circle { center, radius, .. } => {
                    Some((center, normal, radius))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Loft profiles into a solid with smooth NURBS side surfaces.
///
/// Like [`loft`], but produces smooth NURBS surfaces for the side faces
/// instead of piecewise-planar quads. When 3+ profiles are provided,
/// the side surfaces interpolate smoothly through all profiles using
/// tensor-product surface fitting, giving C1+ continuity across sections.
///
/// For 2 profiles, the result is equivalent to the basic [`loft`] (ruled
/// surfaces). For 3+ profiles, the result is a smooth blend. Profiles may have
/// planar or curved surfaces (only the boundary is used) and are resampled to a
/// common vertex count when they differ; end caps are filled like [`loft`].
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 2 profiles are provided
/// - Profiles resample to fewer than 3 vertices
/// - A section boundary is non-planar with more than 4 edges (unsupported cap)
/// - Surface interpolation fails
#[allow(clippy::too_many_lines)]
pub fn loft_smooth(
    topo: &mut Topology,
    profiles: &[FaceId],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if profiles.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft requires at least 2 profiles".into(),
        });
    }

    // For 2 profiles, delegate to the basic loft (ruled surfaces are optimal).
    if profiles.len() == 2 {
        return loft(topo, profiles);
    }

    let mut profile_verts: Vec<Vec<Point3>> = Vec::with_capacity(profiles.len());
    for &fid in profiles {
        let verts = face_polygon(topo, fid)?;
        profile_verts.push(verts);
    }

    let n = profile_verts.iter().map(Vec::len).max().unwrap_or(0);
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft profiles must have at least 3 vertices".into(),
        });
    }
    for verts in &mut profile_verts {
        if verts.len() != n {
            *verts = resample_closed_polygon(verts, n);
        }
    }

    // Ensure profile vertex winding is CCW relative to the stacking direction.
    let _ = ensure_ccw_profiles(&mut profile_verts);

    let num_profiles = profile_verts.len();

    let ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> = profile_verts
        .iter()
        .map(|verts| {
            verts
                .iter()
                .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
                .collect()
        })
        .collect();

    let ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = ring_verts
        .iter()
        .map(|ring| {
            (0..n)
                .map(|i| {
                    let next = (i + 1) % n;
                    topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                })
                .collect()
        })
        .collect();

    let mut all_faces = Vec::new();

    // Start cap: reversed first profile.
    {
        let cap_normal = cap_normal_from_verts(&profile_verts[0], true)?;
        all_faces.push(crate::cap::build_cap_face(
            topo,
            &ring_edges[0],
            vec![],
            &profile_verts[0],
            cap_normal,
            true,
        )?);
    }

    // NURBS side faces: one surface per edge index, spanning ALL profiles.
    // Degree in u (across profiles): min(P-1, 3) for smooth interpolation.
    // Degree in v (along edge): 1 (linear between adjacent vertices).
    let degree_u = (num_profiles - 1).min(3);
    let degree_v = 1;

    for i in 0..n {
        let next_i = (i + 1) % n;

        // Build the interpolation grid: rows = profiles, cols = 2 (edge endpoints).
        let grid: Vec<Vec<Point3>> = (0..num_profiles)
            .map(|k| vec![profile_verts[k][i], profile_verts[k][next_i]])
            .collect();

        let surface =
            interpolate_surface(&grid, degree_u, degree_v).map_err(crate::OperationsError::Math)?;

        // The wire goes around the edge of the NURBS patch:
        // bottom edge → right rail → top edge (reversed) → left rail (reversed)
        let last = num_profiles - 1;

        // Bottom edge: ring_edges[0][i] (first profile, edge i)
        // Top edge: ring_edges[last][i] (last profile, edge i)
        // Left rail: connects vertex i across all profiles
        // Right rail: connects vertex next_i across all profiles

        // For the multi-section case, we need edges spanning ALL profiles.
        // Create single edges from first to last profile for the rails.
        let e_left_rail = topo.add_edge(Edge::new(
            ring_verts[0][i],
            ring_verts[last][i],
            EdgeCurve::Line,
        ));
        let e_right_rail = topo.add_edge(Edge::new(
            ring_verts[0][next_i],
            ring_verts[last][next_i],
            EdgeCurve::Line,
        ));

        let side_wire = Wire::new(
            vec![
                OrientedEdge::new(ring_edges[0][i], true),     // bottom
                OrientedEdge::new(e_right_rail, true),         // right
                OrientedEdge::new(ring_edges[last][i], false), // top (reversed)
                OrientedEdge::new(e_left_rail, false),         // left (reversed)
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;

        let side_wire_id = topo.add_wire(side_wire);
        let side_face = topo.add_face(Face::new(side_wire_id, vec![], FaceSurface::Nurbs(surface)));
        all_faces.push(side_face);
    }

    // End cap: last profile with forward orientation.
    {
        let last = num_profiles - 1;
        let cap_normal = cap_normal_from_verts(&profile_verts[last], false)?;
        all_faces.push(crate::cap::build_cap_face(
            topo,
            &ring_edges[last],
            vec![],
            &profile_verts[last],
            cap_normal,
            false,
        )?);
    }

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

#[cfg(test)]
mod tests;
