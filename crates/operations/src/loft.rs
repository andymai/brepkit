//! Loft operation: create a solid by interpolating between profile faces.
//!
//! The loft connects two or more planar profiles by creating ruled (linear)
//! surfaces between corresponding profile edges.

use brepkit_math::nurbs::surface::NurbsSurface;
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
            let side_surface = ruled_quad(
                profile_verts[s][i],
                profile_verts[s][next_i],
                profile_verts[s + 1][i],
                profile_verts[s + 1][next_i],
            )?;

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
            let side_face = topo.add_face(Face::new(side_wire_id, vec![], side_surface));
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
#[derive(Clone)]
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
///
/// NURBS edges are recognized back to their analytic form (a brepjs sketch
/// delivers rounded-rect corner arcs and even straight runs as NURBS): without
/// recognition the curve-preserving loft bails on every sketch-built profile
/// and the polygon path facets the corners — the gridfinity socket loses
/// ~1-2.5% of its volume to chord wedges and every downstream boolean sees
/// all-plane socket operands.
fn profile_oriented_edges(topo: &Topology, fid: FaceId) -> Option<Vec<ProfileEdgeGeom>> {
    let face = topo.face(fid).ok()?;
    if !matches!(face.surface(), FaceSurface::Plane { .. }) || !face.inner_wires().is_empty() {
        return None;
    }
    let wire = topo.wire(face.outer_wire()).ok()?;
    let oes = wire.edges();
    if oes.is_empty() {
        return None;
    }
    let recog_tol = Tolerance::new().linear * 100.0;
    let mut out = Vec::with_capacity(oes.len());
    for oe in oes {
        let edge = topo.edge(oe.edge()).ok()?;
        let s = topo.vertex(edge.start()).ok()?.point();
        let e = topo.vertex(edge.end()).ok()?.point();
        let (start, end) = if oe.is_forward() { (s, e) } else { (e, s) };
        let curve = match edge.curve() {
            EdgeCurve::NurbsCurve(nc) => {
                match brepkit_geometry::convert::recognize_curve(nc, recog_tol) {
                    brepkit_geometry::convert::RecognizedCurve::Circle {
                        center,
                        normal,
                        radius,
                    } => brepkit_math::curves::Circle3D::new(center, normal, radius)
                        .map_or_else(|_| edge.curve().clone(), EdgeCurve::Circle),
                    brepkit_geometry::convert::RecognizedCurve::Line { .. } => EdgeCurve::Line,
                    _ => edge.curve().clone(),
                }
            }
            c => c.clone(),
        };
        // A circle runs counterclockwise about its normal from `start`: turn
        // it over when the edge runs the other way, or its arc between the
        // ends reads as the complement. The way it runs is read off the
        // edge's tangent where its traversal begins, whatever its parameter's
        // speed, against the circle's own tangent at that point; a closed
        // edge's parameter need not begin at its vertex, but a circle turns
        // one way all round.
        let curve = match curve {
            EdgeCurve::Circle(c) => {
                let (t0, t1) = edge.curve().domain_with_endpoints(s, e);
                let (from, run) = if oe.is_forward() {
                    (t0, t1 - t0)
                } else {
                    (t1, t0 - t1)
                };
                let at = edge.curve().evaluate_with_endpoints(from, s, e);
                // A NURBS edge's parameter can stand still at an end, where
                // its tangent fails; the shared tangent would put +x there.
                let tangent = match edge.curve() {
                    EdgeCurve::NurbsCurve(nc) => {
                        nc.tangent(from).unwrap_or(Vec3::new(0.0, 0.0, 0.0))
                    }
                    other => other.tangent_with_endpoints(from, s, e),
                };
                let mut leaves = tangent * run.signum();
                if leaves.length() <= 1e-12 * run.abs().max(1.0) {
                    leaves = edge
                        .curve()
                        .evaluate_with_endpoints(run.mul_add(1e-3, from), s, e)
                        - at;
                }
                if leaves.dot(c.normal().cross(at - c.center())) >= 0.0 {
                    EdgeCurve::Circle(c)
                } else {
                    EdgeCurve::Circle(
                        brepkit_math::curves::Circle3D::new(
                            c.center(),
                            c.normal() * -1.0,
                            c.radius(),
                        )
                        .ok()?,
                    )
                }
            }
            c => c,
        };
        out.push(ProfileEdgeGeom { curve, start, end });
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

/// The ruled surface between the segments `a0 → a1` and `b0 → b1`, `u`
/// running along them and `v` from one to the other, its normal
/// `(a1 − a0) × (b0 − a0)` at `a0`: their plane when `b1` lies in the plane
/// of the other three corners, and otherwise the bilinear patch on all four,
/// since a plane through three of a twisted quad's corners misses the fourth.
fn ruled_quad(
    a0: Point3,
    a1: Point3,
    b0: Point3,
    b1: Point3,
) -> Result<FaceSurface, crate::OperationsError> {
    let Ok(normal) = (a1 - a0).cross(b0 - a0).normalize() else {
        let normal = Vec3::new(1.0, 0.0, 0.0);
        return Ok(FaceSurface::Plane {
            normal,
            d: dot_normal_point(normal, a0),
        });
    };
    if normal.dot(b1 - a0).abs() <= Tolerance::default().linear {
        return Ok(FaceSurface::Plane {
            normal,
            d: dot_normal_point(normal, a0),
        });
    }
    let knots = vec![0.0, 0.0, 1.0, 1.0];
    Ok(FaceSurface::Nurbs(NurbsSurface::new(
        1,
        1,
        knots.clone(),
        knots,
        vec![vec![a0, b0], vec![a1, b1]],
        vec![vec![1.0; 2]; 2],
    )?))
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
    let mut nc0 = brepkit_geometry::convert::circle_to_nurbs(c0, a0, a0e).ok()?;
    let mut nc1 = brepkit_geometry::convert::circle_to_nurbs(c1, a1, a1e).ok()?;
    if nc0.control_points().len() != nc1.control_points().len() {
        // Same-shape arcs can still segment differently when their spans sit
        // on opposite sides of the π/2-multiple ceil boundary (float jitter);
        // re-convert BOTH arcs at the larger segment count so the ruled
        // pairing survives. Splitting finer never violates the ≤ π/2
        // per-segment invariant; the conversion itself rejects a count too
        // small for either span.
        let segs = ((nc0.control_points().len() - 1) / 2).max((nc1.control_points().len() - 1) / 2);
        nc0 = brepkit_geometry::convert::circle_to_nurbs_with_segments(c0, a0, a0e, segs).ok()?;
        nc1 = brepkit_geometry::convert::circle_to_nurbs_with_segments(c1, a1, a1e, segs).ok()?;
    }
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

/// The ruled surface between two profile edges, a line and a line or an arc,
/// from `r0` to `r1`. A line runs in equal steps while an arc is cut into
/// rational quadratic pieces of at most π/16, so each ruling meets the arc
/// near the angle its step reaches: the pairing a ruled loft makes between a
/// segment and an arc. `None` if the arc cannot be converted.
fn ruled_line_surface(r0: &ProfileEdgeGeom, r1: &ProfileEdgeGeom) -> Option<NurbsSurface> {
    const PIECE: f64 = std::f64::consts::PI / 16.0;
    let arc_pieces = |e: &ProfileEdgeGeom| match &e.curve {
        EdgeCurve::Circle(c) => {
            let (t0, t1) = e.curve.domain_with_endpoints(e.start, e.end);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pieces = ((t1 - t0).abs() / PIECE).ceil().max(1.0) as usize;
            brepkit_geometry::convert::circle_to_nurbs_with_segments(c, t0, t1, pieces).ok()
        }
        _ => None,
    };
    let row = |e: &ProfileEdgeGeom, pieces: usize| -> Option<(Vec<Point3>, Vec<f64>)> {
        if let Some(arc) = arc_pieces(e) {
            return Some((arc.control_points().to_vec(), arc.weights().to_vec()));
        }
        let steps = 2 * pieces;
        #[allow(clippy::cast_precision_loss)]
        let points = (0..=steps)
            .map(|k| e.start + (e.end - e.start) * (k as f64 / steps as f64))
            .collect();
        Some((points, vec![1.0; steps + 1]))
    };
    let arc = arc_pieces(r0).or_else(|| arc_pieces(r1));
    let (degree, knots, pieces) = arc.as_ref().map_or_else(
        || (1, vec![0.0, 0.0, 1.0, 1.0], 0),
        |a| {
            (
                a.degree(),
                a.knots().to_vec(),
                (a.control_points().len() - 1) / 2,
            )
        },
    );
    let ((c0, w0), (c1, w1)) = if pieces == 0 {
        (
            (vec![r0.start, r0.end], vec![1.0; 2]),
            (vec![r1.start, r1.end], vec![1.0; 2]),
        )
    } else {
        (row(r0, pieces)?, row(r1, pieces)?)
    };
    NurbsSurface::new(
        1,
        degree,
        vec![0.0, 0.0, 1.0, 1.0],
        knots,
        vec![c0, c1],
        vec![w0, w1],
    )
    .ok()
}

/// The normal shared by every profile's plane, or `None` when a profile is
/// not planar or the planes are not parallel.
fn common_plane_normal(
    topo: &Topology,
    profiles: &[FaceId],
) -> Result<Option<Vec3>, crate::OperationsError> {
    let mut common: Option<Vec3> = None;
    for &fid in profiles {
        let FaceSurface::Plane { normal, .. } = topo.face(fid)?.surface() else {
            return Ok(None);
        };
        let n = normal.normalize()?;
        match common {
            None => common = Some(n),
            Some(c) if c.dot(n).abs() >= 1.0 - 1e-9 => {}
            Some(_) => return Ok(None),
        }
    }
    Ok(common)
}

/// A profile's centre for matching it to another by angle: a lone circle's
/// own centre, else the centre of its boundary's length.
fn profile_centre(p: &[ProfileEdgeGeom]) -> Option<Point3> {
    if let [only] = p
        && let EdgeCurve::Circle(c) = &only.curve
    {
        return Some(c.center());
    }
    let origin = p.first()?.start;
    let (mut sum, mut total) = (Vec3::new(0.0, 0.0, 0.0), 0.0);
    for e in p {
        let (middle, length) = match &e.curve {
            EdgeCurve::Line => (
                e.start + (e.end - e.start) * 0.5,
                (e.end - e.start).length(),
            ),
            EdgeCurve::Circle(c) => {
                let (t0, t1) = e.curve.domain_with_endpoints(e.start, e.end);
                let half = 0.5 * (t1 - t0);
                let bisector = c.evaluate(t0 + half) - c.center();
                (
                    c.center() + bisector * (half.sin() / half),
                    2.0 * half * c.radius(),
                )
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return None,
        };
        sum += (middle - origin) * length;
        total += length;
    }
    (total > 0.0).then(|| origin + sum * (1.0 / total))
}

/// Where the ray from `from` along `dir` meets the profile `p` nearest
/// `near`: the edge it meets and the point.
fn ray_meets_profile(
    p: &[ProfileEdgeGeom],
    from: Point3,
    dir: Vec3,
    near: Point3,
) -> Option<(usize, Point3)> {
    const SLACK: f64 = 1e-9;
    let mut best: Option<(usize, Point3, f64)> = None;
    for (i, e) in p.iter().enumerate() {
        let mut hits = Vec::new();
        match &e.curve {
            EdgeCurve::Line => {
                // from + s dir = start + t (end - start), in their plane.
                let ab = e.end - e.start;
                let n = dir.cross(ab);
                let nn = n.length_squared();
                if nn > 0.0 {
                    let w = e.start - from;
                    let (s, t) = (w.cross(ab).dot(n) / nn, w.cross(dir).dot(n) / nn);
                    if s > 0.0 && (-SLACK..=1.0 + SLACK).contains(&t) {
                        hits.push(e.start + ab * t.clamp(0.0, 1.0));
                    }
                }
            }
            EdgeCurve::Circle(c) => {
                let f = from - c.center();
                let (a, b, k) = (
                    dir.length_squared(),
                    2.0 * f.dot(dir),
                    f.length_squared() - c.radius() * c.radius(),
                );
                let disc = b.mul_add(b, -4.0 * a * k);
                if a > 0.0 && disc >= 0.0 {
                    let (t0, t1) = e.curve.domain_with_endpoints(e.start, e.end);
                    for s in [
                        (-b - disc.sqrt()) / (2.0 * a),
                        (-b + disc.sqrt()) / (2.0 * a),
                    ] {
                        // A ray off the circle's plane meets its sphere,
                        // not the circle: the cut is taken on the circle.
                        let at = c.project(from + dir * s);
                        let along = (at - t0).rem_euclid(std::f64::consts::TAU);
                        if s > 0.0 && along <= (t1 - t0) + SLACK {
                            hits.push(c.evaluate(at));
                        }
                    }
                }
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return None,
        }
        for q in hits {
            let d = (q - near).length();
            if best.is_none_or(|b| d < b.2) {
                best = Some((i, q, d));
            }
        }
    }
    best.map(|(i, q, _)| (i, q))
}

/// Cuts each profile at the others' vertices, matched by angle: a vertex,
/// carried to another profile by the offset between their centres, casts a
/// ray from that profile's centre within its plane and cuts its boundary
/// where the ray meets it, unless a vertex is already there. Carried from
/// the last profile to the first and back, every profile ends with a vertex
/// at each angle any of them has. The profiles' centres, or `None` when a
/// ray misses a profile.
fn match_profiles_by_angle(
    profs: &mut [Vec<ProfileEdgeGeom>],
    normal: Vec3,
    tol: f64,
) -> Option<Vec<Point3>> {
    let centres: Vec<Point3> = profs
        .iter()
        .map(|p| profile_centre(p))
        .collect::<Option<_>>()?;
    let m = profs.len();
    let order = (1..m)
        .rev()
        .map(|i| (i, i - 1))
        .chain((0..m - 1).map(|i| (i, i + 1)));
    for (from, to) in order {
        let corners: Vec<Point3> = profs[from].iter().map(|e| e.start).collect();
        for v in corners {
            let carried = centres[to] + (v - centres[from]);
            let mut dir = carried - centres[to];
            dir = dir - normal * dir.dot(normal);
            if dir.length() <= tol {
                continue;
            }
            let (i, q) = ray_meets_profile(&profs[to], centres[to], dir, carried)?;
            if profs[to].iter().any(|e| (e.start - q).length() <= tol) {
                continue;
            }
            let e = profs[to][i].clone();
            if (e.end - q).length() <= tol {
                continue;
            }
            profs[to].splice(
                i..=i,
                [
                    ProfileEdgeGeom {
                        curve: e.curve.clone(),
                        start: e.start,
                        end: q,
                    },
                    ProfileEdgeGeom {
                        curve: e.curve,
                        start: q,
                        end: e.end,
                    },
                ],
            );
        }
    }
    Some(centres)
}

/// Turns each profile after the first to start at the vertex whose angle
/// about its centre lies nearest the first profile's start. Whether every
/// vertex then lies at its counterpart's angle: a profile a ray from its
/// centre meets more than once can run back in angle where the others do
/// not, and its vertices pair across angles.
fn start_at_matching_angles(
    profs: &mut [Vec<ProfileEdgeGeom>],
    centres: &[Point3],
    normal: Vec3,
    tol: f64,
) -> bool {
    let flat = |v: Vec3| v - normal * v.dot(normal);
    let reference = flat(profs[0][0].start - centres[0]);
    for (p, &c) in profs.iter_mut().zip(centres).skip(1) {
        let turn = |e: &ProfileEdgeGeom| {
            let d = flat(e.start - c);
            reference.cross(d).dot(normal).atan2(reference.dot(d)).abs()
        };
        if let Some(k) = (0..p.len()).min_by(|&a, &b| turn(&p[a]).total_cmp(&turn(&p[b]))) {
            p.rotate_left(k);
        }
    }
    profs.iter().zip(centres).skip(1).all(|(p, &c)| {
        p.iter().zip(&profs[0]).all(|(e, first)| {
            let (d, along) = (
                flat(e.start - c),
                flat(first.start - centres[0]).normalize(),
            );
            along.is_ok_and(|a| d.dot(a) > 0.0 && d.cross(a).length() <= 10.0 * tol)
        })
    })
}

/// Whether arcs on coaxial circles `c0` and `c1` end at matching angles
/// about their shared axis, start with start and end with end: only then do
/// the lines joining their ends rule the cylinder or cone through both.
fn ends_meet_radially(
    c0: &brepkit_math::curves::Circle3D,
    ends0: (Point3, Point3),
    c1: &brepkit_math::curves::Circle3D,
    ends1: (Point3, Point3),
    tol: Tolerance,
) -> bool {
    let Ok(n) = c0.normal().normalize() else {
        return false;
    };
    let radial = |p: Point3, o: Point3| {
        let d = p - o;
        (d - n * d.dot(n)).normalize().ok()
    };
    let reach = c0.radius().max(c1.radius());
    [(ends0.0, ends1.0), (ends0.1, ends1.1)]
        .into_iter()
        .all(
            |(a, b)| match (radial(a, c0.center()), radial(b, c1.center())) {
                (Some(ra), Some(rb)) => {
                    ra.dot(rb) > 0.0 && ra.cross(rb).length() * reach <= tol.linear
                }
                _ => false,
            },
        )
}

/// Loft profiles of lines and arcs while preserving curved edges (arcs stay
/// arcs via ruled NURBS side faces) instead of faceting them. Profiles of
/// differing edge counts are first cut at one another's vertices by angle.
/// Returns `None` (fall back to the polygon path) unless every profile is
/// planar, hole-free, winds the same way about the stacking direction, ends
/// with as many edges as the others (at least three), at least one edge is
/// curved, and all edges are `Line`/`Circle`.
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

    // 2. Only Line/Circle edges are handled, and only worth it if some edge
    //    is curved (all-Line profiles give the identical polygon result).
    let edges = || profs.iter().flatten();
    if edges().all(|e| matches!(e.curve, EdgeCurve::Line))
        || edges().any(|e| !matches!(e.curve, EdgeCurve::Line | EdgeCurve::Circle(_)))
    {
        return Ok(None);
    }
    // Profiles of differing edge counts are cut at one another's vertices,
    // matched by their angle about each profile's centre.
    let centres = if profs.iter().any(|p| p.len() != profs[0].len()) {
        let Some(normal) = common_plane_normal(topo, profiles)? else {
            return Ok(None);
        };
        let Some(centres) = match_profiles_by_angle(&mut profs, normal, tol.linear) else {
            return Ok(None);
        };
        Some((centres, normal))
    } else {
        None
    };
    let n = profs[0].len();
    if n < 3 || profs.iter().any(|p| p.len() != n) {
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
    // Profiles wound opposite the stacking axis are reversed, not rejected: a
    // loft stacked DOWNWARD from CCW-sketched sections (the gridfinity socket,
    // whose top face sits at z=0 and lofts to −height) winds every profile CW
    // about the axis. Bailing here sent every such loft to the faceting
    // polygon path even though its arcs were pristine. Mixed windings stay a
    // bail — that is degenerate input, not a convention difference.
    // Compare as a cosine (unit normal vs unit axis) so the reliability gate
    // is dimensionless — a raw Newell·axis dot is area-scaled and a machine-
    // epsilon test on it would pass geometrically meaningless windings on
    // near-degenerate profiles straight into the reversal below.
    let axis_unit = axis * (1.0 / axis.length());
    let windings: Vec<f64> = profs
        .iter()
        .map(|p| {
            let n = crate::winding::newell_normal(&junctions(p));
            let len = n.length();
            if len < tol.linear * tol.linear {
                0.0
            } else {
                n.dot(axis_unit) / len
            }
        })
        .collect();
    if windings.iter().any(|&w| w.abs() < 1e-6) {
        return Ok(None);
    }
    if windings.iter().all(|&w| w < 0.0) {
        for p in &mut profs {
            p.reverse();
            for e in p.iter_mut() {
                std::mem::swap(&mut e.start, &mut e.end);
                // Reversing an arc's endpoints alone selects the complementary
                // span (the natural circle direction now runs the long way
                // around); flip the circle normal so the same minor arc is
                // traced in the opposite direction.
                if let EdgeCurve::Circle(c) = &e.curve {
                    match brepkit_math::curves::Circle3D::new(
                        c.center(),
                        c.normal() * -1.0,
                        c.radius(),
                    ) {
                        Ok(flipped) => e.curve = EdgeCurve::Circle(flipped),
                        Err(_) => return Ok(None),
                    }
                }
            }
        }
    } else if windings.iter().any(|&w| w < 0.0) {
        return Ok(None);
    }
    if let Some((centres, normal)) = &centres
        && !start_at_matching_angles(&mut profs, centres, *normal, tol.linear)
    {
        return Ok(None);
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
            let p1e = profs[s + 1][i].end;
            let entry = match (&profs[s][i].curve, &profs[s + 1][i].curve) {
                (EdgeCurve::Line, EdgeCurve::Line) => (ruled_quad(p0s, p0e, p1s, p1e)?, false),
                (EdgeCurve::Circle(_), EdgeCurve::Line)
                | (EdgeCurve::Line, EdgeCurve::Circle(_)) => {
                    let (r0, r1) = (&profs[s][i], &profs[s + 1][i]);
                    let Some(surf) = ruled_line_surface(r0, r1) else {
                        return Ok(None);
                    };
                    let inward = surf
                        .normal(0.5, 0.5)
                        .map(|nrm| nrm.dot(outward) < 0.0)
                        .unwrap_or(false);
                    if inward {
                        match ruled_line_surface(r1, r0) {
                            Some(flipped) => (FaceSurface::Nurbs(flipped), false),
                            None => (FaceSurface::Nurbs(surf), true),
                        }
                    } else {
                        (FaceSurface::Nurbs(surf), false)
                    }
                }
                (EdgeCurve::Circle(c0), EdgeCurve::Circle(c1)) => {
                    if ends_meet_radially(c0, (p0s, p0e), c1, (p1s, p1e), tol)
                        && let Some(surface) = coaxial_corner_surface(c0, c1, tol)
                    {
                        // Radial-outward equals solid-outward only for a
                        // CONVEX corner arc (material inside the arc). A
                        // concave rounding puts the material outside the
                        // cylinder: the stored normal points into the solid
                        // and the face must be reversed (with the
                        // reversed-winding wire below), or every neighbour
                        // disagrees with it in the directed mesh while edge
                        // senses still pair. The chord-cross `outward` used
                        // by the other arms cannot discriminate here (a
                        // concave traversal flips the chord AND the radial
                        // normal together): profiles are CCW about the
                        // stacking direction, so material-outward is the
                        // TRAVERSAL TANGENT crossed with the connect
                        // direction, sampled at the arc midpoint.
                        let curve = EdgeCurve::Circle(c0.clone());
                        let (t0, t1) = curve.domain_with_endpoints(p0s, p0e);
                        let t_mid = f64::midpoint(t0, t1);
                        let mid = curve.evaluate_with_endpoints(t_mid, p0s, p0e);
                        let tan = curve.tangent_with_endpoints(t_mid, p0s, p0e);
                        // The connect direction along a COAXIAL band is the
                        // shared axis (the radial-difference part is
                        // orthogonal to the tested radial normal), so use
                        // the axis oriented from this profile to the next —
                        // exact at every arc parameter, no per-vertex
                        // connect vector needed.
                        let inward = c0
                            .normal()
                            .normalize()
                            .ok()
                            .and_then(|axis| {
                                let height = (c1.center() - c0.center()).dot(axis);
                                let up = axis * height.signum();
                                let outward_true = tan.cross(up);
                                surface
                                    .project_point(mid)
                                    .map(|(u, v)| surface.normal(u, v).dot(outward_true) < 0.0)
                            })
                            .unwrap_or(false);
                        (surface, inward)
                    } else {
                        let Some(surf) = ruled_arc_surface(c0, p0s, p0e, c1, p1s, p1e) else {
                            return Ok(None);
                        };
                        let inward = surf
                            .normal(0.5, 0.5)
                            .map(|nrm| nrm.dot(outward) < 0.0)
                            .unwrap_or(false);
                        if inward {
                            // Swapping the rails negates ∂S/∂u and flips the
                            // normal outward, so the face needs no reversal
                            // flag (and downstream consumers never see a
                            // reversed-wound loft wall). Fall back to the
                            // reversal flag if the swapped build fails.
                            match ruled_arc_surface(c1, p1s, p1e, c0, p0s, p0e) {
                                Some(flipped) => (FaceSurface::Nurbs(flipped), false),
                                None => (FaceSurface::Nurbs(surf), true),
                            }
                        } else {
                            (FaceSurface::Nurbs(surf), false)
                        }
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
            let (surface, reversed) = side_surfaces[s * n + i].clone();
            // A reversed face must carry the reversed-winding wire so the
            // effective edge senses (is_forward XOR is_reversed) still
            // oppose its neighbours'.
            let edges = if reversed {
                vec![
                    OrientedEdge::new(conn_eids[s][i], true),
                    OrientedEdge::new(ring_eids[s + 1][i], true),
                    OrientedEdge::new(conn_eids[s][next_i], false),
                    OrientedEdge::new(ring_eids[s][i], false),
                ]
            } else {
                vec![
                    OrientedEdge::new(ring_eids[s][i], true),
                    OrientedEdge::new(conn_eids[s][next_i], true),
                    OrientedEdge::new(ring_eids[s + 1][i], false),
                    OrientedEdge::new(conn_eids[s][i], false),
                ]
            };
            let side_wire_id =
                topo.add_wire(Wire::new(edges, true).map_err(crate::OperationsError::Topology)?);
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

    // One rail per vertex index through that vertex of every profile, all
    // sharing the mean chord-length parameters and so one knot vector. Each
    // side face is the ruled surface between two neighbouring rails, so its
    // boundary IS the rails and the ring edges, which both neighbours share.
    // u runs along the ring edge and v up the rails: for profiles wound CCW
    // about the stacking direction the normal Su × Sv points outward.
    let last = num_profiles - 1;
    let degree = (num_profiles - 1).min(3);
    let rail_points: Vec<Vec<Point3>> = (0..n)
        .map(|i| profile_verts.iter().map(|verts| verts[i]).collect())
        .collect();
    let params = mean_chord_params(&rail_points);
    let rails = rail_points
        .iter()
        .map(|points| interpolate_at(points, degree, &params))
        .collect::<Result<Vec<_>, _>>()?;
    let rail_edges: Vec<EdgeId> = rails
        .iter()
        .enumerate()
        .map(|(i, rail)| {
            topo.add_edge(Edge::new(
                ring_verts[0][i],
                ring_verts[last][i],
                EdgeCurve::NurbsCurve(rail.clone()),
            ))
        })
        .collect();

    for i in 0..n {
        let next_i = (i + 1) % n;
        let (a, b) = (&rails[i], &rails[next_i]);
        let surface = NurbsSurface::new(
            1,
            a.degree(),
            vec![0.0, 0.0, 1.0, 1.0],
            a.knots().to_vec(),
            vec![a.control_points().to_vec(), b.control_points().to_vec()],
            vec![vec![1.0; a.control_points().len()]; 2],
        )
        .map_err(crate::OperationsError::Math)?;

        let side_wire = Wire::new(
            vec![
                OrientedEdge::new(ring_edges[0][i], true),
                OrientedEdge::new(rail_edges[next_i], true),
                OrientedEdge::new(ring_edges[last][i], false),
                OrientedEdge::new(rail_edges[i], false),
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

/// Interpolate a non-rational B-spline through `points` at the increasing
/// `params`, with knots averaged from the parameters (The NURBS Book, eq.
/// 9.8): curves fitted at the same parameters share one knot vector.
pub(crate) fn interpolate_at(
    points: &[Point3],
    degree: usize,
    params: &[f64],
) -> Result<brepkit_math::nurbs::curve::NurbsCurve, crate::OperationsError> {
    use brepkit_math::nurbs::basis::{basis_funs, find_span};

    let n = points.len();
    if n < 2 || params.len() != n || params.windows(2).any(|w| w[1] <= w[0]) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "rail interpolation needs increasing parameters, one per point".into(),
        });
    }
    let p = degree.clamp(1, n - 1);
    let mut knots = vec![params[0]; p + 1];
    for j in 1..n - p {
        #[allow(clippy::cast_precision_loss)]
        knots.push(params[j..j + p].iter().sum::<f64>() / p as f64);
    }
    knots.extend(std::iter::repeat_n(params[n - 1], p + 1));

    // Collocation: row i holds N_{j,p}(params[i]), nonzero only across its
    // span's p + 1 columns. `last[c]` is the lowest row reaching column c.
    let mut rows = vec![vec![0.0; n]; n];
    let mut last: Vec<usize> = (0..n).collect();
    for (i, (row, &u)) in rows.iter_mut().zip(params).enumerate() {
        let span = find_span(n, p, u, &knots);
        for (k, value) in basis_funs(span, u, p, &knots).into_iter().enumerate() {
            row[span - p + k] = value;
            last[span - p + k] = last[span - p + k].max(i);
        }
    }
    for c in 1..n {
        last[c] = last[c].max(last[c - 1]);
    }
    let mut rhs: Vec<[f64; 3]> = points.iter().map(|q| [q.x(), q.y(), q.z()]).collect();
    // Gaussian elimination with partial pivoting. Spans never decrease with
    // the parameter, so no row below `last[col]` reaches column `col`, and
    // eliminating (or swapping) within that window only fills columns whose
    // windows reach at least as far: the band holds, and the solve stays
    // quadratic in the profile count.
    for col in 0..n {
        let pivot = (col..=last[col])
            .max_by(|&a, &b| rows[a][col].abs().total_cmp(&rows[b][col].abs()))
            .unwrap_or(col);
        if rows[pivot][col].abs() < 1e-14 {
            return Err(crate::OperationsError::InvalidInput {
                reason: "rail interpolation system is singular".into(),
            });
        }
        rows.swap(col, pivot);
        rhs.swap(col, pivot);
        for r in col + 1..=last[col] {
            let f = rows[r][col] / rows[col][col];
            for c in col..n {
                rows[r][c] -= f * rows[col][c];
            }
            for k in 0..3 {
                rhs[r][k] -= f * rhs[col][k];
            }
        }
    }
    let mut control = vec![[0.0; 3]; n];
    for r in (0..n).rev() {
        for k in 0..3 {
            let tail: f64 = (r + 1..n).map(|c| rows[r][c] * control[c][k]).sum();
            control[r][k] = (rhs[r][k] - tail) / rows[r][r];
        }
    }
    brepkit_math::nurbs::curve::NurbsCurve::new(
        p,
        knots,
        control
            .iter()
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect(),
        vec![1.0; n],
    )
    .map_err(crate::OperationsError::Math)
}

/// Chord-length parameters in `[0, 1]` averaged over several point rows of
/// equal length. A row whose points all coincide carries no spacing and is
/// left out; if every row does, the parameters are uniform.
pub(crate) fn mean_chord_params(rows: &[Vec<Point3>]) -> Vec<f64> {
    let len = rows.first().map_or(0, Vec::len);
    let mut sum = vec![0.0; len];
    let mut counted = 0_u32;
    for row in rows {
        let steps: Vec<f64> = row.windows(2).map(|w| (w[1] - w[0]).length()).collect();
        let total: f64 = steps.iter().sum();
        if total <= 0.0 {
            continue;
        }
        let mut acc = 0.0;
        for (k, step) in std::iter::once(0.0).chain(steps).enumerate() {
            acc += step;
            sum[k] += acc / total;
        }
        counted += 1;
    }
    if counted == 0 {
        #[allow(clippy::cast_precision_loss)]
        return (0..len)
            .map(|k| k as f64 / (len.max(2) - 1) as f64)
            .collect();
    }
    sum.iter().map(|s| s / f64::from(counted)).collect()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod rounded_l_loft_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn rounded_l_face(topo: &mut Topology, inset: f64, r: f64, z: f64) -> FaceId {
        let i = inset;
        let vs = [
            (i, i, false),
            (126.0 - i, i, false),
            (126.0 - i, 42.0 - i, false),
            (42.0 - i, 42.0 - i, true),
            (42.0 - i, 126.0 - i, false),
            (i, 126.0 - i, false),
        ];
        let n = vs.len();
        let mut t_in = Vec::new();
        let mut t_out = Vec::new();
        let mut centers = Vec::new();
        for k in 0..n {
            let (px, py, _) = vs[(k + n - 1) % n];
            let (vx, vy, concave) = vs[k];
            let (nx, ny, _) = vs[(k + 1) % n];
            let din = ((vx - px).signum(), (vy - py).signum());
            let dout = ((nx - vx).signum(), (ny - vy).signum());
            let ti = (vx - din.0 * r, vy - din.1 * r);
            let to = (vx + dout.0 * r, vy + dout.1 * r);
            let c = if concave {
                (
                    vx + dout.0.abs() * r + if din.0 == 0.0 { 0.0 } else { -din.0 * r },
                    vy + if din.1 == 0.0 { 0.0 } else { -din.1 * r } + dout.1.abs() * r,
                )
            } else {
                (ti.0 + (to.0 - vx), ti.1 + (to.1 - vy))
            };
            t_in.push(ti);
            t_out.push(to);
            centers.push(c);
        }
        let mut oes = Vec::new();
        let vid_at = |topo: &mut Topology, p: (f64, f64)| {
            topo.add_vertex(Vertex::new(Point3::new(p.0, p.1, z), 1e-7))
        };
        let mut prev_out = vid_at(topo, t_out[n - 1]);
        for k in 0..n {
            let in_vid = vid_at(topo, t_in[k]);
            let line = topo.add_edge(Edge::new(prev_out, in_vid, EdgeCurve::Line));
            oes.push(OrientedEdge::new(line, true));
            let out_vid = vid_at(topo, t_out[k]);
            let c = Circle3D::new(
                Point3::new(centers[k].0, centers[k].1, z),
                Vec3::new(0.0, 0.0, 1.0),
                r,
            )
            .unwrap();
            if vs[k].2 {
                let arc = topo.add_edge(Edge::new(out_vid, in_vid, EdgeCurve::Circle(c)));
                oes.push(OrientedEdge::new(arc, false));
            } else {
                let arc = topo.add_edge(Edge::new(in_vid, out_vid, EdgeCurve::Circle(c)));
                oes.push(OrientedEdge::new(arc, true));
            }
            prev_out = out_vid;
        }
        let wire = topo.add_wire(Wire::new(oes, true).unwrap());
        topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: z,
            },
        ))
    }

    /// A multi-section rounded-L loft (the gridfinity custom-shape lip
    /// frustum) must stay curve-preserving. The concave corner's arc centres
    /// drift with the inset, so its bands are ruled NURBS — and the two
    /// semicircle-span conversions used to segment differently on float
    /// jitter at the pi/2-multiple ceil boundary, failing the pairing and
    /// faceting the whole lip (the 22-scenario custom-shape export family).
    #[test]
    fn rounded_l_multi_section_loft_stays_curve_preserving() {
        let mut topo = Topology::new();
        let f0 = rounded_l_face(&mut topo, 2.15, 3.75 - 2.15, 0.0);
        let f1 = rounded_l_face(&mut topo, 1.6, 3.75 - 1.6, 0.7);
        let f2 = rounded_l_face(&mut topo, 1.0, 3.75 - 1.0, 4.4);
        let solid = loft(&mut topo, &[f0, f1, f2]).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        // The fixture's straight sides run askew from one section to the
        // next, so their bands are twisted too; the corners' bands are the
        // ones bounded by arcs.
        let curved = faces
            .iter()
            .filter(|&&f| {
                let face = topo.face(f).unwrap();
                face.surface().type_tag() != "plane"
                    && topo
                        .wire(face.outer_wire())
                        .unwrap()
                        .edges()
                        .iter()
                        .any(|oe| {
                            matches!(topo.edge(oe.edge()).unwrap().curve(), EdgeCurve::Circle(_))
                        })
            })
            .count();
        assert_eq!(
            curved,
            12,
            "6 corners x 2 bands must stay curved, got {curved} of {}",
            faces.len()
        );
    }
}

#[cfg(test)]
mod square_to_circle_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// A 40 x 40 square at z = 0 lofted to a circle of radius 10 at z = 40,
    /// either way up. The circle is cut at the square's corners' angles and
    /// the square at the circle's seam, each side ruled to its arc meeting it
    /// at even angles, which holds 35125.497. Pairing the square's corners
    /// with the circle's samples from wherever each starts gave 34 planar
    /// faces holding 19274.751, or 2599.671 with the circle first.
    #[test]
    fn a_square_lofts_to_a_circle_meeting_it_at_matching_angles() {
        let mut topo = Topology::new();
        let p = Point3::new;
        let wire = brepkit_topology::builder::make_polygon_wire(
            &mut topo,
            &[
                p(-20.0, -20.0, 0.0),
                p(20.0, -20.0, 0.0),
                p(20.0, 20.0, 0.0),
                p(-20.0, 20.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let square = brepkit_topology::builder::make_face_from_wire(&mut topo, wire).unwrap();
        let rim = brepkit_topology::builder::make_circle_edge(
            &mut topo,
            p(0.0, 0.0, 40.0),
            Vec3::new(0.0, 0.0, 1.0),
            10.0,
            1e-7,
        )
        .unwrap();
        let rim = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let circle = brepkit_topology::builder::make_face_from_wire(&mut topo, rim).unwrap();
        for profiles in [[square, circle], [circle, square]] {
            let solid = loft(&mut topo, &profiles).unwrap();
            let kinds: Vec<_> = brepkit_topology::explorer::solid_faces(&topo, solid)
                .unwrap()
                .iter()
                .map(|&f| topo.face(f).unwrap().surface().type_tag())
                .collect();
            assert_eq!(
                kinds.iter().filter(|&&k| k == "plane").count(),
                2,
                "{kinds:?}"
            );
            assert_eq!(kinds.len(), 7, "{kinds:?}");
            assert!(
                crate::validate::validate_solid(&topo, solid)
                    .unwrap()
                    .is_valid()
            );
            let volume = crate::measure::solid_volume(&topo, solid, 1e-4).unwrap();
            assert!((volume - 35125.497).abs() < 1e-4 * 35125.497, "{volume}");
        }
    }
    /// Every vertex of `solid` lies on each of its edges' curves, and each
    /// edge of a cylinder or cone face on that surface, to 1e-7.
    fn lies_on_its_geometry(topo: &Topology, solid: SolidId) -> bool {
        let faces = brepkit_topology::explorer::solid_faces(topo, solid).unwrap();
        faces.iter().all(|&f| {
            let face = topo.face(f).unwrap();
            let surface = face.surface();
            std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .flat_map(|w| topo.wire(w).unwrap().edges().to_vec())
                .all(|oe| {
                    let e = topo.edge(oe.edge()).unwrap();
                    let (s, t) = (
                        topo.vertex(e.start()).unwrap().point(),
                        topo.vertex(e.end()).unwrap().point(),
                    );
                    let (t0, t1) = e.curve().domain_with_endpoints(s, t);
                    let at = |x: f64| e.curve().evaluate_with_endpoints(x, s, t);
                    let ends = (at(t0) - s).length() <= 1e-7 && (at(t1) - t).length() <= 1e-7;
                    let on_surface = match surface {
                        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {
                            let mid = at(f64::midpoint(t0, t1));
                            surface.project_point(mid).is_some_and(|(u, v)| {
                                surface
                                    .evaluate(u, v)
                                    .is_some_and(|q| (q - mid).length() <= 1e-7)
                            })
                        }
                        _ => true,
                    };
                    ends && on_surface
                })
        })
    }

    /// A profile piece: the point it runs to, and the centre it runs round
    /// when it is an arc.
    type Piece = ((f64, f64), Option<(f64, f64)>);

    /// A planar profile at height `z` from `pieces`: each runs from the
    /// previous one's end to its point, straight or round the given centre.
    fn profile(topo: &mut Topology, z: f64, pieces: &[Piece]) -> FaceId {
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        let at = |(x, y): (f64, f64)| Point3::new(x, y, z);
        let vertices: Vec<_> = pieces
            .iter()
            .map(|&(q, _)| topo.add_vertex(Vertex::new(at(q), 1e-7)))
            .collect();
        let n = pieces.len();
        let edges = (0..n)
            .map(|k| {
                let (from, to) = (vertices[(k + n - 1) % n], vertices[k]);
                let curve = pieces[k].1.map_or(EdgeCurve::Line, |c| {
                    let radius = (at(pieces[k].0) - at(c)).length();
                    EdgeCurve::Circle(
                        brepkit_math::curves::Circle3D::new(
                            at(c),
                            Vec3::new(0.0, 0.0, 1.0),
                            radius,
                        )
                        .unwrap(),
                    )
                });
                OrientedEdge::new(topo.add_edge(Edge::new(from, to, curve)), true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: z,
            },
        ))
    }

    /// A square of half-size `h` with corners rounded to radius `r`, run
    /// counterclockwise, its bottom side cut at `x = cut` when given.
    fn rounded_square(h: f64, r: f64, cut: Option<f64>) -> Vec<Piece> {
        let k = h - r;
        let mut pieces = vec![((-k, -h), Some((-k, -k)))];
        pieces.extend(cut.map(|x| ((x, -h), None)));
        pieces.extend([
            ((k, -h), None),
            ((h, -k), Some((k, -k))),
            ((h, k), None),
            ((k, h), Some((k, k))),
            ((-k, h), None),
            ((-h, k), Some((-k, k))),
            ((-h, -k), None),
        ]);
        pieces
    }

    /// A 40 square lofted to a circle of radius 10 whose plane tilts 1e-5
    /// about x: rays in the square's plane meet the circle's sphere up to
    /// 7e-5 off the circle, and the cuts are taken on the circle, so the
    /// loft keeps its 7 faces rather than falling back to a polygon.
    #[test]
    fn a_square_lofts_to_a_slightly_tilted_circle_with_its_cuts_on_it() {
        let mut topo = Topology::new();
        let square = profile(
            &mut topo,
            0.0,
            &[
                ((20.0, -20.0), None),
                ((20.0, 20.0), None),
                ((-20.0, 20.0), None),
                ((-20.0, -20.0), None),
            ],
        );
        let normal = Vec3::new(0.0, -(1e-5_f64).sin(), (1e-5_f64).cos());
        let rim = brepkit_topology::builder::make_circle_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 40.0),
            normal,
            10.0,
            1e-7,
        )
        .unwrap();
        let rim = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let circle = topo.add_face(brepkit_topology::face::Face::new(
            rim,
            vec![],
            FaceSurface::Plane {
                normal,
                d: 40.0 * normal.z(),
            },
        ));
        let solid = loft(&mut topo, &[square, circle]).unwrap();
        let kinds: Vec<_> = brepkit_topology::explorer::solid_faces(&topo, solid)
            .unwrap()
            .iter()
            .map(|&f| topo.face(f).unwrap().surface().type_tag())
            .collect();
        assert_eq!(kinds.len(), 7, "{kinds:?}");
        assert!(lies_on_its_geometry(&topo, solid));
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid()
        );
    }

    /// Rounded squares of half-size 10 and corner radius 2 at z = 0, one side
    /// cut in two, and of half-size 12 and corner radius 4 at z = 10. Their
    /// corners share axes, but matching by angle cuts the second's corner
    /// arcs at the first's corners' angles about the squares' centre, so the
    /// arcs it pairs no longer end at matching angles about their own axis:
    /// each such band is ruled between them, every edge on its surface.
    #[test]
    fn coaxial_corner_arcs_cut_at_other_angles_are_ruled() {
        let mut topo = Topology::new();
        let low = profile(&mut topo, 0.0, &rounded_square(10.0, 2.0, Some(0.3)));
        let high = profile(&mut topo, 10.0, &rounded_square(12.0, 4.0, None));
        let solid = loft(&mut topo, &[low, high]).unwrap();
        assert!(lies_on_its_geometry(&topo, solid));
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid()
        );
    }

    /// Arcs read the way their edges leave their start. A quarter disc of
    /// radius 10 whose arc is a rational quadratic quarter circle weighted
    /// 1, 10/√2 and 100, its parameter 70° round by a quarter of its span,
    /// stored either way round, lofts to a circle of radius 3 above the
    /// middle of its boundary as the quarter disc with a circle arc does. A
    /// disc of radius 10 whose rim starts at its frame's angle π lofts to a
    /// square of half-size 4 above and beside it, cut at the square's corners
    /// and ruled to its sides, as the disc starting at 0 does: a closed rim's
    /// parameter starts at its frame's origin, wherever its vertex lies.
    /// Every loft keeps its arcs.
    #[test]
    fn arcs_read_the_way_they_leave_their_start() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        let plane = |z: f64| FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z,
        };
        let top = |topo: &mut Topology, x: f64, y: f64| {
            let rim = brepkit_topology::builder::make_circle_edge(
                topo,
                Point3::new(x, y, 10.0),
                Vec3::new(0.0, 0.0, 1.0),
                3.0,
                1e-7,
            )
            .unwrap();
            let rim = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
            brepkit_topology::builder::make_face_from_wire(topo, rim).unwrap()
        };
        let loft_volume = |topo: &mut Topology, low: FaceId, high: FaceId, label: &str| {
            let solid = try_loft_matching_curved_profiles(topo, &[low, high]).unwrap();
            assert!(solid.is_some(), "{label}: not lofted along its arcs");
            let solid = solid.unwrap();
            assert!(lies_on_its_geometry(topo, solid), "{label}");
            assert!(
                crate::validate::validate_solid(topo, solid)
                    .unwrap()
                    .is_valid(),
                "{label}"
            );
            crate::measure::solid_volume(topo, solid, 1e-4).unwrap()
        };

        // `None` for a circle arc, else the NURBS arc stored either way.
        let quarter = |topo: &mut Topology, nurbs: Option<bool>| {
            let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
            let [o, a, b] = [at(0.0, 0.0), at(10.0, 0.0), at(0.0, 10.0)]
                .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let w = 10.0 * std::f64::consts::FRAC_1_SQRT_2;
            let arc = match nurbs {
                None => {
                    let circle =
                        Circle3D::new(at(0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();
                    OrientedEdge::new(
                        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
                        true,
                    )
                }
                Some(stored_reversed) => {
                    let (points, weights) = if stored_reversed {
                        (
                            vec![at(0.0, 10.0), at(10.0, 10.0), at(10.0, 0.0)],
                            vec![100.0, w, 1.0],
                        )
                    } else {
                        (
                            vec![at(10.0, 0.0), at(10.0, 10.0), at(0.0, 10.0)],
                            vec![1.0, w, 100.0],
                        )
                    };
                    let curve =
                        NurbsCurve::new(2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], points, weights)
                            .unwrap();
                    let (from, to) = if stored_reversed { (b, a) } else { (a, b) };
                    OrientedEdge::new(
                        topo.add_edge(Edge::new(from, to, EdgeCurve::NurbsCurve(curve))),
                        !stored_reversed,
                    )
                }
            };
            let edges = vec![
                OrientedEdge::new(topo.add_edge(Edge::new(o, a, EdgeCurve::Line)), true),
                arc,
                OrientedEdge::new(topo.add_edge(Edge::new(b, o, EdgeCurve::Line)), true),
            ];
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            topo.add_face(Face::new(wire, vec![], plane(0.0)))
        };
        let mut volumes = Vec::new();
        for nurbs in [None, Some(false), Some(true)] {
            let mut topo = Topology::new();
            let low = quarter(&mut topo, nurbs);
            let high = top(&mut topo, 4.2, 4.2);
            volumes.push(loft_volume(&mut topo, low, high, &format!("{nurbs:?}")));
        }
        assert!(
            volumes
                .iter()
                .all(|v| (v - volumes[0]).abs() < 1e-4 * volumes[0]),
            "{volumes:?}"
        );

        let disc = |topo: &mut Topology, seam: f64| {
            let circle =
                Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();
            let v = topo.add_vertex(Vertex::new(circle.evaluate(seam), 1e-7));
            let rim = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
            topo.add_face(Face::new(wire, vec![], plane(0.0)))
        };
        let mut volumes = Vec::new();
        for seam in [0.0, std::f64::consts::PI] {
            let mut topo = Topology::new();
            let low = disc(&mut topo, seam);
            let high = profile(
                &mut topo,
                10.0,
                &[
                    ((7.0, -3.0), None),
                    ((7.0, 5.0), None),
                    ((-1.0, 5.0), None),
                    ((-1.0, -3.0), None),
                ],
            );
            volumes.push(loft_volume(&mut topo, low, high, &format!("seam {seam}")));
        }
        assert!(
            (volumes[1] - volumes[0]).abs() < 1e-4 * volumes[0],
            "{volumes:?}"
        );
    }

    /// A quarter disc of radius 10 whose arc runs counterclockwise from
    /// (0, 10) to (-10, 0) as the quarter circle R(t²): with w = 1/√2, the
    /// rational quartic over (0, 10), (0, 10), (2 P₀ + w P₁) / (2 + w),
    /// (-10, 10) and (-10, 0) weighted 1, 1, (2 + w) / 3, w and 1, whose
    /// derivative vanishes where it starts. Stored either way round, it lofts
    /// along its arcs to a circle of radius 3 above the middle of its
    /// boundary as the quarter disc with a circle arc does.
    #[test]
    fn an_arc_standing_still_where_it_starts_reads_its_way() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        let quarter = |topo: &mut Topology, nurbs: Option<bool>| {
            let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
            let [o, a, b] = [at(0.0, 0.0), at(0.0, 10.0), at(-10.0, 0.0)]
                .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let arc = match nurbs {
                None => {
                    let circle =
                        Circle3D::new(at(0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();
                    OrientedEdge::new(
                        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
                        true,
                    )
                }
                Some(stored_reversed) => {
                    let w = std::f64::consts::FRAC_1_SQRT_2;
                    let (p0, p1, p2) = (at(0.0, 10.0), at(-10.0, 10.0), at(-10.0, 0.0));
                    let mut points = vec![p0, p0, p0 + (p1 - p0) * (w / (2.0 + w)), p1, p2];
                    let mut weights = vec![1.0, 1.0, (2.0 + w) / 3.0, w, 1.0];
                    if stored_reversed {
                        points.reverse();
                        weights.reverse();
                    }
                    let curve = NurbsCurve::new(
                        4,
                        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        points,
                        weights,
                    )
                    .unwrap();
                    let still = if stored_reversed { 1.0 } else { 0.0 };
                    assert!(curve.derivatives(still, 1)[1].length() < 1e-12);
                    let halfway = curve.evaluate(0.5) - at(0.0, 0.0);
                    assert!((halfway.length() - 10.0).abs() < 1e-9);
                    let (from, to) = if stored_reversed { (b, a) } else { (a, b) };
                    OrientedEdge::new(
                        topo.add_edge(Edge::new(from, to, EdgeCurve::NurbsCurve(curve))),
                        !stored_reversed,
                    )
                }
            };
            let edges = vec![
                OrientedEdge::new(topo.add_edge(Edge::new(o, a, EdgeCurve::Line)), true),
                arc,
                OrientedEdge::new(topo.add_edge(Edge::new(b, o, EdgeCurve::Line)), true),
            ];
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: 0.0,
                },
            ))
        };
        let mut volumes = Vec::new();
        for nurbs in [None, Some(false), Some(true)] {
            let mut topo = Topology::new();
            let low = quarter(&mut topo, nurbs);
            let rim = brepkit_topology::builder::make_circle_edge(
                &mut topo,
                Point3::new(-4.2, 4.2, 10.0),
                Vec3::new(0.0, 0.0, 1.0),
                3.0,
                1e-7,
            )
            .unwrap();
            let rim = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
            let high = brepkit_topology::builder::make_face_from_wire(&mut topo, rim).unwrap();
            let solid = try_loft_matching_curved_profiles(&mut topo, &[low, high]).unwrap();
            assert!(solid.is_some(), "{nurbs:?}: not lofted along its arcs");
            let solid = solid.unwrap();
            assert!(lies_on_its_geometry(&topo, solid), "{nurbs:?}");
            assert!(
                crate::validate::validate_solid(&topo, solid)
                    .unwrap()
                    .is_valid(),
                "{nurbs:?}"
            );
            volumes.push(crate::measure::solid_volume(&topo, solid, 1e-4).unwrap());
        }
        assert!(
            volumes
                .iter()
                .all(|v| (v - volumes[0]).abs() < 1e-4 * volumes[0]),
            "{volumes:?}"
        );
    }

    /// A quarter disc of radius 10, its arc a circle edge stored either way
    /// round, lofted to a copy 10 above it and to a circle of radius 3 above
    /// the middle of its boundary: stored either way, each loft is the same
    /// valid solid, the prism holding 250π.
    #[test]
    fn a_reversed_arc_edge_lofts_as_the_arc_it_bounds() {
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        let quarter = |topo: &mut Topology, z: f64, stored_reversed: bool| {
            let at = |x: f64, y: f64| Point3::new(x, y, z);
            let [o, a, b] = [at(0.0, 0.0), at(10.0, 0.0), at(0.0, 10.0)]
                .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let up = Vec3::new(0.0, 0.0, if stored_reversed { -1.0 } else { 1.0 });
            let circle = brepkit_math::curves::Circle3D::new(at(0.0, 0.0), up, 10.0).unwrap();
            let arc = if stored_reversed {
                OrientedEdge::new(
                    topo.add_edge(Edge::new(b, a, EdgeCurve::Circle(circle))),
                    false,
                )
            } else {
                OrientedEdge::new(
                    topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
                    true,
                )
            };
            let edges = vec![
                OrientedEdge::new(topo.add_edge(Edge::new(o, a, EdgeCurve::Line)), true),
                arc,
                OrientedEdge::new(topo.add_edge(Edge::new(b, o, EdgeCurve::Line)), true),
            ];
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: z,
                },
            ))
        };
        let mut volumes = Vec::new();
        for stored_reversed in [false, true] {
            let mut topo = Topology::new();
            let low = quarter(&mut topo, 0.0, stored_reversed);
            let high = quarter(&mut topo, 10.0, stored_reversed);
            let rim = brepkit_topology::builder::make_circle_edge(
                &mut topo,
                Point3::new(4.2, 4.2, 10.0),
                Vec3::new(0.0, 0.0, 1.0),
                3.0,
                1e-7,
            )
            .unwrap();
            let rim = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
            let circle = brepkit_topology::builder::make_face_from_wire(&mut topo, rim).unwrap();
            let mut found = Vec::new();
            for (profiles, label) in [([low, high], "prism"), ([low, circle], "to a circle")] {
                let solid = loft(&mut topo, &profiles).unwrap();
                assert!(
                    lies_on_its_geometry(&topo, solid),
                    "{label}, reversed {stored_reversed}"
                );
                assert!(
                    crate::validate::validate_solid(&topo, solid)
                        .unwrap()
                        .is_valid(),
                    "{label}, reversed {stored_reversed}"
                );
                found.push(crate::measure::solid_volume(&topo, solid, 1e-4).unwrap());
            }
            let prism = 250.0 * std::f64::consts::PI;
            assert!((found[0] - prism).abs() < 1e-4 * prism, "{found:?}");
            volumes.push(found);
        }
        assert!(
            (volumes[0][1] - volumes[1][1]).abs() < 1e-9 * volumes[0][1],
            "{volumes:?}"
        );
    }

    /// A square notched at its top, lofted to a circle: a ray from its centre
    /// meets it more than once, so its cuts do not run round it in order of
    /// angle as the circle's do. Such a loft is not paired by angle; it
    /// keeps the polygon loft, valid, the circle faceted so no edge of it
    /// stays a circle.
    #[test]
    fn a_notched_square_is_not_paired_by_angle() {
        let mut topo = Topology::new();
        let notched = profile(
            &mut topo,
            0.0,
            &[
                ((3.0, -3.0), None),
                ((3.0, 3.0), None),
                ((1.0, 3.0), None),
                ((1.0, 1.0), None),
                ((-1.0, 1.0), None),
                ((-1.0, 3.0), None),
                ((-3.0, 3.0), None),
                ((-3.0, -3.0), None),
            ],
        );
        let rim = brepkit_topology::builder::make_circle_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            1e-7,
        )
        .unwrap();
        let rim = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let circle = brepkit_topology::builder::make_face_from_wire(&mut topo, rim).unwrap();
        let solid = loft(&mut topo, &[notched, circle]).unwrap();
        assert!(lies_on_its_geometry(&topo, solid));
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid()
        );
        let arcs = brepkit_topology::explorer::solid_edges(&topo, solid)
            .unwrap()
            .iter()
            .filter(|&&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_)))
            .count();
        assert_eq!(arcs, 0);
    }
}
