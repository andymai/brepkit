//! Revolution of a planar profile around an axis to create solids of revolution.
//!
//! Revolve rotates a planar face around an arbitrary axis to produce cylinders,
//! cones, torus-like shapes, and other bodies of revolution. The swept side
//! surfaces are represented as rational NURBS (degree-2 in the circular
//! direction), which exactly represent circular arcs.

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;

/// Minimum radial distance threshold for non-degenerate arcs.
const MIN_RADIAL_LEN: f64 = 1e-12;

/// A partial-revolve profile boundary is treated as planar when its deviation
/// from the best-fit plane is below this fraction of the boundary's size.
const PLANARITY_REL_TOL: f64 = 1e-6;

/// Rotate a point around an axis (origin + unit direction) by angle θ.
///
/// Uses Rodrigues' rotation formula:
///   P' = P·cos θ + (k × P)·sin θ + k·(k · P)·(1 − cos θ)
/// where P is the vector from origin to point, k is the unit axis.
fn rotate_point(point: Point3, origin: Point3, axis: Vec3, angle: f64) -> Point3 {
    let v = point - origin;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let k_dot_v = axis.dot(v);
    let k_cross_v = axis.cross(v);
    let rotated = v * cos_a + k_cross_v * sin_a + axis * (k_dot_v * (1.0 - cos_a));
    origin + rotated
}

/// Rotate a direction vector around an axis by angle θ (no translation).
fn rotate_vec(dir: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let k_dot_v = axis.dot(dir);
    let k_cross_v = axis.cross(dir);
    dir * cos_a + k_cross_v * sin_a + axis * (k_dot_v * (1.0 - cos_a))
}

/// Compute the number of arc segments needed and the angle per segment.
///
/// Each segment spans at most π/2 (90°). Returns `(num_segments, segment_angle)`.
fn arc_segmentation(total_angle: f64) -> (usize, f64) {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let num_segs = ((total_angle / FRAC_PI_2).ceil() as usize).max(1);
    #[allow(clippy::cast_precision_loss)]
    let seg_angle = total_angle / (num_segs as f64);
    (num_segs, seg_angle)
}

/// Compute the middle NURBS control point for a circular arc segment.
///
/// For a degree-2 rational Bézier arc with half-angle `half` and weight
/// `cos(half)`, the middle control point sits at distance `r / cos(half)`
/// from the axis, along the half-angle radial direction. If the point lies
/// on the axis (zero radius), the degenerate midpoint is returned as-is.
fn arc_mid_control_point(
    start: Point3,
    origin: Point3,
    axis: Vec3,
    half: f64,
    w_mid: f64,
) -> Point3 {
    let mid_on_arc = rotate_point(start, origin, axis, half);
    let r_vec = mid_on_arc - origin;
    let proj = axis * axis.dot(r_vec);
    let radial = r_vec - proj;
    if radial.length() > MIN_RADIAL_LEN {
        origin + proj + radial * (1.0 / w_mid)
    } else {
        mid_on_arc
    }
}

/// Create a degree-2 rational NURBS curve representing a circular arc.
///
/// Control points: `[start, mid, end]` with weights `[1, cos(θ/2), 1]`.
fn make_arc_curve(
    start: Point3,
    end: Point3,
    origin: Point3,
    axis: Vec3,
    angle: f64,
) -> Result<NurbsCurve, brepkit_math::MathError> {
    let half = angle / 2.0;
    let w_mid = half.cos();
    let mid_ctrl = arc_mid_control_point(start, origin, axis, half, w_mid);

    NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![start, mid_ctrl, end],
        vec![1.0, w_mid, 1.0],
    )
}

/// Build a NURBS surface of revolution for one profile edge and one arc segment.
///
/// - u-direction: profile edge (degree 1, 2 control points)
/// - v-direction: circular arc (degree 2, 3 control points)
///
/// The result is a 2×3 tensor-product surface.
fn make_revolution_surface(
    p0_start: Point3,
    p0_end: Point3,
    p1_start: Point3,
    p1_end: Point3,
    origin: Point3,
    axis: Vec3,
    seg_angle: f64,
) -> Result<NurbsSurface, brepkit_math::MathError> {
    let half = seg_angle / 2.0;
    let w_mid = half.cos();

    let mid0 = arc_mid_control_point(p0_start, origin, axis, half, w_mid);
    let mid1 = arc_mid_control_point(p1_start, origin, axis, half, w_mid);

    NurbsSurface::new(
        1,                                  // degree_u (profile — linear)
        2,                                  // degree_v (arc — quadratic rational)
        vec![0.0, 0.0, 1.0, 1.0],           // knots_u
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], // knots_v
        vec![vec![p0_start, mid0, p0_end], vec![p1_start, mid1, p1_end]],
        vec![vec![1.0, w_mid, 1.0], vec![1.0, w_mid, 1.0]],
    )
}

/// Decompose a point into `(radial_distance, axial_coordinate)` relative to the
/// revolution axis (a line through `axis_origin` with unit direction `axis`).
fn radial_axial(p: Point3, axis_origin: Point3, axis: Vec3) -> (f64, f64) {
    let v = p - axis_origin;
    let z = v.dot(axis);
    ((v - axis * z).length(), z)
}

/// Build the surface for one revolution band.
///
/// Returns the **exact** analytic surface of revolution so the band integrates
/// exactly instead of inscribing the swept arc as a NURBS band (~2% / ~0.04%
/// deficit, gh #968):
/// - axis-parallel line edge → `Cylinder`
/// - oblique line edge → `Cone` (the line, extended, meets the axis at the apex)
/// - perpendicular line edge → `Plane` (a flat annular disk normal to the axis)
/// - circular-arc edge clearing the axis → `Torus` band
///
/// The returned `reversed` flag orients the analytic surface to agree with the
/// correctly-wound NURBS band normal.
///
/// A general spline profile edge, a spindle/self-intersecting torus arc (arc
/// radius ≥ its centre's axis distance), and degenerate on-axis bands keep the
/// NURBS band. (A circular arc whose centre is ON the axis sweeps a sphere; that
/// case also falls back to NURBS here — recognising it as a `Sphere` band is a
/// follow-up.)
#[allow(clippy::too_many_arguments)]
fn revolution_band_surface(
    profile_curve: &EdgeCurve,
    p0_start: Point3,
    p0_end: Point3,
    p1_start: Point3,
    p1_end: Point3,
    axis_origin: Point3,
    axis: Vec3,
    seg_angle: f64,
) -> Result<(FaceSurface, bool), brepkit_math::MathError> {
    let nurbs = make_revolution_surface(
        p0_start,
        p0_end,
        p1_start,
        p1_end,
        axis_origin,
        axis,
        seg_angle,
    )?;

    // A circular-arc profile edge sweeps an exact torus band. (`p0_start` and
    // `p1_start` are the arc's endpoints on this segment's ring.)
    if let Some((center, radius)) = profile_arc_center_radius(profile_curve, p0_start, p1_start) {
        if let Some(result) =
            revolution_torus_band(center, radius, axis_origin, axis, &nurbs, seg_angle)?
        {
            return Ok(result);
        }
        return Ok((FaceSurface::Nurbs(nurbs), false));
    }

    if !matches!(profile_curve, EdgeCurve::Line) {
        return Ok((FaceSurface::Nurbs(nurbs), false));
    }

    let tol = 1e-9;
    // The profile edge runs `p0_start → p1_start` (vertex i to vertex i+1 on the
    // SAME segment ring); `p0_start → p0_end` is the swept-arc direction. Decompose
    // both edge endpoints into (radial, axial) coordinates about the axis.
    let (r0, z0) = radial_axial(p0_start, axis_origin, axis);
    let (r1, z1) = radial_axial(p1_start, axis_origin, axis);
    let dr = r1 - r0;
    let dz = z1 - z0;

    // A profile edge on the axis (both endpoints at r ≈ 0) sweeps a degenerate
    // zero-area band; keep the NURBS form (and avoid evaluating its degenerate
    // normal below).
    if r0 < tol && r1 < tol {
        return Ok((FaceSurface::Nurbs(nurbs), false));
    }

    // Orientation reference: match the analytic surface's winding to the NURBS
    // band normal (its du×dv, pointing into the swept material), so every face
    // of the result winds consistently. The volume integrator takes the absolute
    // value of the total, so consistency — not outward-vs-inward per se — matters.
    let band_normal = nurbs.normal(0.5, 0.5)?;

    // The radial-outward direction at the band's mid-arc. Both the cylinder and
    // cone faces have a natural radially-outward normal, so they are reversed
    // exactly when this outward direction opposes the consistent band normal.
    let mid = rotate_point(p0_start, axis_origin, axis, seg_angle / 2.0);
    let mid_radial = mid - (axis_origin + axis * (mid - axis_origin).dot(axis));
    let natural_outward = mid_radial.normalize().unwrap_or(axis);
    let outward_reversed = natural_outward.dot(band_normal) < 0.0;

    // Axis-parallel edge (Δr ≈ 0, radius > 0) → cylinder wall.
    if dr.abs() < tol {
        let surface = brepkit_math::surfaces::CylindricalSurface::new(axis_origin, axis, r0)?;
        return Ok((FaceSurface::Cylinder(surface), outward_reversed));
    }

    // Perpendicular edge (Δz ≈ 0) → flat annular disk normal to the axis. The
    // analytic volume path integrates a circular-arc-bounded planar cap exactly
    // (`planar_cap_signed_volume`), so the disc no longer suffers the chorded-
    // boundary under-count that previously kept these caps NURBS.
    if dz.abs() < tol {
        let plane_d = dot_normal_point(axis, p0_start);
        // Stored normal is `+axis`; reverse when the consistent band normal
        // points along `−axis`.
        return Ok((
            FaceSurface::Plane {
                normal: axis,
                d: plane_d,
            },
            band_normal.dot(axis) < 0.0,
        ));
    }

    // Oblique edge (both Δr and Δz non-negligible) → cone. The line, extended to
    // r = 0, meets the axis at the apex; the half-angle is the angle from the
    // radial plane to the generator (matching `make_cone`).
    let half_angle = dz.abs().atan2(dr.abs());
    if half_angle <= tol || half_angle >= FRAC_PI_2 - tol {
        // Numerically parallel/perpendicular despite the guards above — keep the
        // exact NURBS band rather than build an invalid cone.
        return Ok((FaceSurface::Nurbs(nurbs), false));
    }
    let apex = revolution_cone_apex(axis_origin, axis, r0, z0, dr, dz);
    // The cone axis points apex → widening end (radius grows from the apex). Pick
    // the axial direction from the apex toward the larger-radius endpoint.
    let wide_z = if r1 >= r0 { z1 } else { z0 };
    let apex_z = axis.dot(apex - axis_origin);
    let cone_axis = if wide_z >= apex_z { axis } else { -axis };
    let surface = brepkit_math::surfaces::ConicalSurface::new(apex, cone_axis, half_angle)?;
    Ok((FaceSurface::Cone(surface), outward_reversed))
}

/// Axial position (along the revolution axis) where the profile line, extended,
/// crosses the axis (r = 0) — the cone apex. Built from one endpoint and the
/// (Δr, Δz) slope so it is exact for the infinite line through the edge.
fn revolution_cone_apex(
    axis_origin: Point3,
    axis: Vec3,
    r_ref: f64,
    z_ref: f64,
    dr: f64,
    dz: f64,
) -> Point3 {
    // Parametrize the line in (r, z): at r = 0, the axial coordinate is
    // z_apex = z_ref - r_ref * (dz / dr). Place the apex on the axis at z_apex.
    let z_apex = z_ref - r_ref * (dz / dr);
    axis_origin + axis * z_apex
}

/// Recognise a profile edge as a circular arc, returning its `(center, radius)`.
///
/// A `Circle` edge reports them directly; a rational-quadratic `NurbsCurve`
/// (how circles are often constructed) is run through curve recognition. Returns
/// `None` for lines, ellipses, and unrecognised splines. `p_start`/`p_end` are
/// the arc endpoints, used only as the recognition tolerance scale.
fn profile_arc_center_radius(
    curve: &EdgeCurve,
    p_start: Point3,
    p_end: Point3,
) -> Option<(Point3, f64)> {
    match curve {
        EdgeCurve::Circle(c) => Some((c.center(), c.radius())),
        EdgeCurve::NurbsCurve(nc) => {
            let scale = (p_end - p_start).length().max(1.0);
            let tol = Tolerance::new().linear * 100.0 * scale;
            match brepkit_geometry::convert::recognize_curve(nc, tol) {
                brepkit_geometry::convert::RecognizedCurve::Circle { center, radius, .. } => {
                    Some((center, radius))
                }
                _ => None,
            }
        }
        EdgeCurve::Line | EdgeCurve::Ellipse(_) => None,
    }
}

/// Build the `Torus` band swept by a circular-arc profile edge with the given
/// `(center, radius)` about the revolution axis.
///
/// The torus center is the arc centre projected onto the axis; the major radius
/// is the arc centre's perpendicular distance to the axis; the minor radius is
/// the arc radius. Returns `Ok(None)` (caller keeps the NURBS band) when the arc
/// does not clear the axis (`major ≤ minor`, a spindle/degenerate torus that
/// would self-intersect) or the arc centre lies on the axis (a sphere band).
fn revolution_torus_band(
    arc_center: Point3,
    arc_radius: f64,
    axis_origin: Point3,
    axis: Vec3,
    nurbs: &NurbsSurface,
    seg_angle: f64,
) -> Result<Option<(FaceSurface, bool)>, brepkit_math::MathError> {
    let tol = 1e-9;
    // Decompose the arc centre into its axial position and radial offset.
    let to_center = arc_center - axis_origin;
    let axial = axis * to_center.dot(axis);
    let radial = to_center - axial;
    let major = radial.length();
    // Centre on the axis → sphere band (a different surface type); spindle /
    // horn torus (major ≤ minor) self-intersects. Both keep the NURBS band.
    if major <= arc_radius + tol {
        return Ok(None);
    }
    let torus_center = axis_origin + axial;
    let surface =
        brepkit_math::surfaces::ToroidalSurface::with_axis(torus_center, major, arc_radius, axis)?;
    // Match the band's winding to the NURBS band normal (consistent winding
    // across all faces; the volume integrator uses the absolute total). Compare
    // the toroidal surface's natural normal to the NURBS one at the SAME 3D
    // point — the band centre, projected onto the torus for its (u, v).
    let _ = seg_angle;
    let band_center = brepkit_math::traits::ParametricSurface::evaluate(nurbs, 0.5, 0.5);
    let band_normal = nurbs.normal(0.5, 0.5)?;
    let (tu, tv) = surface.project_point(band_center);
    let outward = surface.normal(tu, tv);
    Ok(Some((
        FaceSurface::Torus(surface),
        outward.dot(band_normal) < 0.0,
    )))
}

/// Index of the next ring for a given segment, wrapping to 0 for the last
/// segment of a full revolution.
const fn next_ring_index(seg: usize, num_segs: usize, is_full: bool) -> usize {
    if is_full && seg == num_segs - 1 {
        0
    } else {
        seg + 1
    }
}

/// Data produced by revolving a single wire (outer or inner).
struct WireRevolveData {
    ring_verts: Vec<Vec<VertexId>>,
    arc_edges: Vec<Vec<brepkit_topology::edge::EdgeId>>,
    ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>>,
    input_oriented: Vec<OrientedEdge>,
    n: usize,
}

/// Fast exact path for revolving a single circular profile a full turn: the
/// result is a torus, built as one analytic [`FaceSurface::Torus`] face.
///
/// The general revolve splits the circle into line chords and revolves each
/// into a NURBS band, which inscribes the circle and undershoots the analytic
/// volume by ~2% (gh #968). A `Torus` face is integrated exactly.
///
/// One profile edge classified for the analytic full-revolution fast path.
enum RevEdge {
    /// Axis-parallel line at radius `r`, axial span `[z0, z1]` → cylinder wall.
    Cylinder { r: f64 },
    /// Oblique line → cone wall. `apex` on the axis, `half_angle` from `make_cone`.
    Cone { apex: Point3, half_angle: f64 },
    /// Perpendicular line → flat disc/annulus cap (normal ±axis).
    Plane,
    /// Circular arc clearing the axis → torus band.
    Torus {
        center: Point3,
        major: f64,
        minor: f64,
    },
    /// Degenerate edge on the axis (both endpoints at r≈0) → no face.
    OnAxis,
}

/// One profile edge of the analytic full-revolution fast path, in wire
/// traversal order (`sp → ep`).
struct ProfileEdge {
    sp: Point3,
    ep: Point3,
    /// The traversal-start vertex — reused as the rim-circle seam vertex so
    /// walls can reuse the original profile edge as their seam.
    sv: VertexId,
    edge: brepkit_topology::edge::EdgeId,
    forward: bool,
    class: RevEdge,
    /// Traversal-ordered interior samples (quarter/mid/three-quarter) for arc
    /// edges, used for the winding shoelace and torus-wall orientation.
    interior: Vec<Point3>,
}

/// Classify one profile edge for the analytic full-revolution path, or `None`
/// when the edge has no unambiguous closed-form revolution face (spline,
/// axis-touching arc, sphere/spindle arc, degenerate cone angle).
fn classify_profile_edge(
    curve: &EdgeCurve,
    sp: Point3,
    ep: Point3,
    axis_origin: Point3,
    axis: Vec3,
    lin: f64,
) -> Option<RevEdge> {
    let (r0, z0) = radial_axial(sp, axis_origin, axis);
    let (r1, z1) = radial_axial(ep, axis_origin, axis);
    let s_on_axis = r0 < lin;
    let e_on_axis = r1 < lin;
    // Arc edge → torus band (both ends must clear the axis; a closed circle
    // inside a multi-edge wire has no seam-able span).
    if let Some((center, radius)) = profile_arc_center_radius(curve, sp, ep) {
        if s_on_axis || e_on_axis || (sp - ep).length() < lin {
            return None;
        }
        let to_c = center - axis_origin;
        let major = (to_c - axis * to_c.dot(axis)).length();
        if major <= radius + lin {
            return None; // sphere/spindle — defer
        }
        return Some(RevEdge::Torus {
            center,
            major,
            minor: radius,
        });
    }
    if !matches!(curve, EdgeCurve::Line) {
        return None; // spline — defer
    }
    let dr = r1 - r0;
    let dz = z1 - z0;
    if s_on_axis && e_on_axis {
        Some(RevEdge::OnAxis)
    } else if dr.abs() < lin {
        // Axis-parallel wall: both ends share radius r0 (> lin here).
        Some(RevEdge::Cylinder { r: r0 })
    } else if dz.abs() < lin {
        // Perpendicular cap: a disc (one end on the axis) or an annulus.
        Some(RevEdge::Plane)
    } else {
        // Oblique wall → cone. One endpoint may sit ON the axis (a pointed
        // apex); the wall then gets `make_cone`'s degenerate seam wire.
        let apex = revolution_cone_apex(axis_origin, axis, r0, z0, dr, dz);
        let half_angle = dz.abs().atan2(dr.abs());
        if half_angle <= lin || half_angle >= FRAC_PI_2 - lin {
            return None;
        }
        Some(RevEdge::Cone { apex, half_angle })
    }
}

/// Fast exact path: a FULL revolution of a fully-analytic planar profile (every
/// edge a line or circular arc) built as ONE periodic face per profile edge —
/// matching `make_cylinder`/`make_cone`/`make_torus` — instead of the segmented
/// NURBS path's 4×90° bands.
///
/// Each profile vertex off the axis becomes a shared rim `Circle3D` edge (on
/// the ORIGINAL wire vertex); each non-degenerate profile edge becomes one
/// periodic wall (cylinder/cone/torus, closed by reusing the profile edge
/// itself as the seam) or one planar disc/annulus cap (bounded by its rim
/// circle(s), an annulus keeping the smaller rim as a hole). Adjacent faces
/// reuse the same rim circle, so the shell is watertight with no segment seams.
///
/// Face orientation comes from the profile's traversal winding in the
/// (radial, axial) chart, so inward-facing walls (a washer's bore) and cap
/// normals are exact for both CCW- and CW-wound profiles.
///
/// Returns `Ok(None)` (fall back to the segmented revolve) unless the
/// revolution is full, the profile is planar with the axis in its plane, has no
/// inner wires, and every edge classifies analytically (no spline, no
/// axis-touching arc). The closed-circle→torus and degenerate cases defer too.
fn try_analytic_full_revolution(
    topo: &mut Topology,
    face: FaceId,
    axis_origin: Point3,
    axis: Vec3,
    is_full: bool,
) -> Result<Option<SolidId>, crate::OperationsError> {
    if !is_full {
        return Ok(None);
    }
    let face_data = topo.face(face)?;
    if !face_data.inner_wires().is_empty() {
        return Ok(None);
    }
    // Planar profile with the axis lying in its plane (else not a clean
    // surface of revolution about this axis).
    let normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return Ok(None),
    };
    if normal.dot(axis).abs() > 1e-9 {
        return Ok(None);
    }

    let tol = Tolerance::new();
    let lin = tol.linear;

    let wire = topo.wire(face_data.outer_wire())?;
    let oriented: Vec<OrientedEdge> = wire.edges().to_vec();
    if oriented.len() < 2 {
        // A single closed circle is the torus fast path's job; a single open
        // edge cannot bound a face. Defer.
        return Ok(None);
    }

    let mut profile: Vec<ProfileEdge> = Vec::with_capacity(oriented.len());
    for oe in &oriented {
        let edge = topo.edge(oe.edge())?;
        let (svid, evid) = if oe.is_forward() {
            (edge.start(), edge.end())
        } else {
            (edge.end(), edge.start())
        };
        let sp = topo.vertex(svid)?.point();
        let ep = topo.vertex(evid)?.point();
        let curve = edge.curve().clone();
        let ns = topo.vertex(edge.start())?.point();
        let ne = topo.vertex(edge.end())?.point();
        let Some(class) = classify_profile_edge(&curve, sp, ep, axis_origin, axis, lin) else {
            return Ok(None);
        };
        // Interior samples for arcs, taken in the curve's NATURAL start→end
        // direction (the stored arc span) and then reversed to traversal order
        // — sampling with traversal endpoints directly would pick the CCW
        // complement of a reversed arc.
        let interior = if matches!(class, RevEdge::Torus { .. }) {
            let (t0, t1) = curve.domain_with_endpoints(ns, ne);
            let mut pts: Vec<Point3> = [0.25, 0.5, 0.75]
                .iter()
                .map(|f| curve.evaluate_with_endpoints((t1 - t0).mul_add(*f, t0), ns, ne))
                .collect();
            if !oe.is_forward() {
                pts.reverse();
            }
            pts
        } else {
            Vec::new()
        };
        profile.push(ProfileEdge {
            sp,
            ep,
            sv: svid,
            edge: oe.edge(),
            forward: oe.is_forward(),
            class,
            interior,
        });
    }

    // The (radial, axial) chart: an isometric 2D coordinate system of the
    // profile plane, with the radial basis taken from the farthest-off-axis
    // profile point. The shoelace sign over the chart polygon (vertices plus
    // arc interior samples, so arc-dominated profiles don't degenerate) gives
    // the traversal winding, which fixes every face's material-outward side.
    let mut e_r: Option<Vec3> = None;
    let mut best_r = 0.0_f64;
    for pe in &profile {
        for &p in std::iter::once(&pe.sp).chain(pe.interior.iter()) {
            let v = p - axis_origin;
            let radial = v - axis * v.dot(axis);
            let r = radial.length();
            if r > best_r {
                best_r = r;
                e_r = radial.normalize().ok();
            }
        }
    }
    let Some(e_r) = e_r else {
        return Ok(None); // whole profile on the axis — degenerate
    };
    let chart = |p: Point3| {
        let v = p - axis_origin;
        (v.dot(e_r), v.dot(axis))
    };
    let mut pts2: Vec<(f64, f64)> = Vec::new();
    for pe in &profile {
        pts2.push(chart(pe.sp));
        for &q in &pe.interior {
            pts2.push(chart(q));
        }
    }
    let mut area2 = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..pts2.len() {
        let (x0, y0) = pts2[i];
        let (x1, y1) = pts2[(i + 1) % pts2.len()];
        area2 += x0.mul_add(y1, -(x1 * y0));
        scale = scale.max(x0.abs()).max(y0.abs());
    }
    if area2.abs() <= scale * scale * 1e-9 {
        return Ok(None); // degenerate (zero-area) profile — defer
    }
    let ccw = area2 > 0.0;

    Some(build_analytic_revolution(
        topo,
        axis_origin,
        axis,
        e_r,
        ccw,
        &profile,
    ))
    .transpose()
}

/// Build the periodic faces for a classified analytic full revolution (see
/// [`try_analytic_full_revolution`]). Returns the assembled solid.
#[allow(clippy::too_many_lines)]
fn build_analytic_revolution(
    topo: &mut Topology,
    axis_origin: Point3,
    axis: Vec3,
    e_r: Vec3,
    ccw: bool,
    profile: &[ProfileEdge],
) -> Result<SolidId, crate::OperationsError> {
    use brepkit_topology::edge::EdgeId;
    let lin = Tolerance::new().linear;
    let n = profile.len();
    let s = if ccw { 1.0 } else { -1.0 };
    let chart = |p: Point3| {
        let v = p - axis_origin;
        (v.dot(e_r), v.dot(axis))
    };

    // One shared rim per off-axis profile vertex: the ORIGINAL wire vertex plus
    // a full `Circle3D` edge (v→v). Reusing the input vertices lets every wall
    // reuse its original profile edge as the seam — the only orientation-safe
    // way to seam a torus band with the exact profile arc. On-axis vertices
    // have no rim (radius 0); an apex-touching cone wall uses `make_cone`'s
    // degenerate seam wire instead.
    let mut rim_circle: Vec<Option<EdgeId>> = vec![None; n];
    for (i, pe) in profile.iter().enumerate() {
        let (r, z) = radial_axial(pe.sp, axis_origin, axis);
        if r < lin {
            continue;
        }
        let center = axis_origin + axis * z;
        let circle = brepkit_math::curves::Circle3D::new(center, axis, r)
            .map_err(crate::OperationsError::Math)?;
        rim_circle[i] = Some(topo.add_edge(Edge::new(pe.sv, pe.sv, EdgeCurve::Circle(circle))));
    }

    let mut faces: Vec<FaceId> = Vec::with_capacity(n);

    for (idx, pe) in profile.iter().enumerate() {
        let next = (idx + 1) % n;
        let (x0, y0) = chart(pe.sp);
        let (x1, y1) = chart(pe.ep);
        // Material-outward direction of this edge's revolved face, from the
        // profile winding: the interior lies left of travel for a CCW chart
        // traversal, so outward is the right-hand perpendicular of the chord
        // (sign-flipped for CW).
        let (dx, dy) = (x1 - x0, y1 - y0);
        let out2 = (s * dy, -(s * dx));

        match pe.class {
            RevEdge::OnAxis => {} // no face
            RevEdge::Plane => {
                // A disc (one end on the axis) or an annulus (both ends off the
                // axis, the smaller rim kept as a hole wound opposite the outer).
                let (outer_i, inner_i) = if x0 >= x1 { (idx, next) } else { (next, idx) };
                let Some(outer_e) = rim_circle[outer_i] else {
                    continue; // no rim (degenerate) — skip
                };
                let cap_normal = if out2.1 >= 0.0 { axis } else { -axis };
                let d = dot_normal_point(cap_normal, pe.sp);
                // The rim circles are CCW about +axis, bounding a disc whose
                // outward normal is +axis when wound forward.
                let outer_fwd = cap_normal.dot(axis) > 0.0;
                let cap_wire = Wire::new(vec![OrientedEdge::new(outer_e, outer_fwd)], true)
                    .map_err(crate::OperationsError::Topology)?;
                let cap_wid = topo.add_wire(cap_wire);
                let mut inner_wids = Vec::new();
                if let Some(inner_e) = rim_circle[inner_i] {
                    let hole_wire = Wire::new(vec![OrientedEdge::new(inner_e, !outer_fwd)], true)
                        .map_err(crate::OperationsError::Topology)?;
                    inner_wids.push(topo.add_wire(hole_wire));
                }
                faces.push(topo.add_face(Face::new(
                    cap_wid,
                    inner_wids,
                    FaceSurface::Plane {
                        normal: cap_normal,
                        d,
                    },
                )));
            }
            RevEdge::Cylinder { .. } | RevEdge::Cone { .. } | RevEdge::Torus { .. } => {
                // A periodic wall seamed by the ORIGINAL profile edge. The
                // orientation probe is a point ON the wall surface mid-edge plus
                // the material-outward direction there (for an arc, the local
                // tangent from the interior samples replaces the chord).
                let (probe, outm) = if let RevEdge::Torus { .. } = pe.class {
                    let m = pe.interior[1];
                    let a = chart(pe.interior[0]);
                    let b = chart(pe.interior[2]);
                    let (tx, ty) = (b.0 - a.0, b.1 - a.1);
                    (m, (s * ty, -(s * tx)))
                } else {
                    let m = Point3::new(
                        f64::midpoint(pe.sp.x(), pe.ep.x()),
                        f64::midpoint(pe.sp.y(), pe.ep.y()),
                        f64::midpoint(pe.sp.z(), pe.ep.z()),
                    );
                    (m, out2)
                };
                let outward = e_r * outm.0 + axis * outm.1;
                let (surface, reversed) =
                    revolution_wall_surface(&pe.class, axis_origin, axis, probe, outward)?;
                let wall_wire = match (rim_circle[idx], rim_circle[next]) {
                    (Some(bot_e), Some(top_e)) => Wire::new(
                        vec![
                            OrientedEdge::new(bot_e, true),
                            OrientedEdge::new(pe.edge, pe.forward),
                            OrientedEdge::new(top_e, false),
                            OrientedEdge::new(pe.edge, !pe.forward),
                        ],
                        true,
                    ),
                    // Pointed-cone apex: the on-axis end has no rim; the wire
                    // follows `make_cone`'s degenerate pattern (rim + seam +
                    // seam⁻¹), the seam running rim → apex.
                    (Some(rim_e), None) => Wire::new(
                        vec![
                            OrientedEdge::new(rim_e, true),
                            OrientedEdge::new(pe.edge, pe.forward),
                            OrientedEdge::new(pe.edge, !pe.forward),
                        ],
                        true,
                    ),
                    (None, Some(rim_e)) => Wire::new(
                        vec![
                            OrientedEdge::new(rim_e, true),
                            OrientedEdge::new(pe.edge, !pe.forward),
                            OrientedEdge::new(pe.edge, pe.forward),
                        ],
                        true,
                    ),
                    (None, None) => {
                        return Err(crate::OperationsError::InvalidInput {
                            reason: "analytic revolution wall has no rim circle".into(),
                        });
                    }
                }
                .map_err(crate::OperationsError::Topology)?;
                let wall_wid = topo.add_wire(wall_wire);
                faces.push(if reversed {
                    topo.add_face(Face::new_reversed(wall_wid, vec![], surface))
                } else {
                    topo.add_face(Face::new(wall_wid, vec![], surface))
                });
            }
        }
    }

    if faces.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "analytic revolution produced no faces".into(),
        });
    }
    let shell = Shell::new(faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Build the analytic wall surface (cylinder/cone/torus) for one classified
/// profile edge, plus the `reversed` flag that aligns its natural normal with
/// `outward` (the material-outward direction) at `probe`, a point ON the wall.
fn revolution_wall_surface(
    class: &RevEdge,
    axis_origin: Point3,
    axis: Vec3,
    probe: Point3,
    outward: Vec3,
) -> Result<(FaceSurface, bool), crate::OperationsError> {
    match class {
        RevEdge::Cylinder { r } => {
            let sfc = brepkit_math::surfaces::CylindricalSurface::new(axis_origin, axis, *r)
                .map_err(crate::OperationsError::Math)?;
            let (u, v) = sfc.project_point(probe);
            let nat = sfc.normal(u, v);
            Ok((FaceSurface::Cylinder(sfc), nat.dot(outward) < 0.0))
        }
        RevEdge::Cone { apex, half_angle } => {
            // The cone axis points apex → widening end; the probe sits on the
            // widening side, so its axial offset from the apex gives the sign.
            let apex_z = axis.dot(*apex - axis_origin);
            let probe_z = axis.dot(probe - axis_origin);
            let cone_axis = if probe_z >= apex_z { axis } else { -axis };
            let sfc = brepkit_math::surfaces::ConicalSurface::new(*apex, cone_axis, *half_angle)
                .map_err(crate::OperationsError::Math)?;
            let (u, v) = sfc.project_point(probe);
            let nat = sfc.normal(u, v);
            Ok((FaceSurface::Cone(sfc), nat.dot(outward) < 0.0))
        }
        RevEdge::Torus {
            center,
            major,
            minor,
        } => {
            let to_c = *center - axis_origin;
            let torus_center = axis_origin + axis * to_c.dot(axis);
            let sfc = brepkit_math::surfaces::ToroidalSurface::with_axis(
                torus_center,
                *major,
                *minor,
                axis,
            )
            .map_err(crate::OperationsError::Math)?;
            let (u, v) = sfc.project_point(probe);
            let nat = sfc.normal(u, v);
            Ok((FaceSurface::Torus(sfc), nat.dot(outward) < 0.0))
        }
        RevEdge::Plane | RevEdge::OnAxis => Err(crate::OperationsError::InvalidInput {
            reason: "revolution_wall_surface called on a non-wall edge".into(),
        }),
    }
}

/// Returns `Ok(None)` — fall back to the general revolve — unless the profile
/// is a single closed circle, the axis lies in the profile plane, and the
/// circle clears the axis (`major > minor`, so the torus does not
/// self-intersect; the sphere and spindle cases fall back).
///
/// A full revolution builds one doubly-periodic `Torus` face like
/// `primitives::make_torus`. A partial revolution builds one `Torus` band
/// trimmed to the swept angle — bounded by the profile circle and its rotated
/// copy, seamed by the arc the profile vertex sweeps — plus two planar disc
/// caps, instead of chord-splitting the circle into segmented bands.
#[allow(clippy::too_many_lines)]
fn try_circle_revolution_torus(
    topo: &mut Topology,
    face: FaceId,
    axis_origin: Point3,
    axis: Vec3,
    angle: f64,
    is_full: bool,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let face_data = topo.face(face)?;
    if !face_data.inner_wires().is_empty() {
        return Ok(None);
    }
    let normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return Ok(None),
    };

    let wire = topo.wire(face_data.outer_wire())?;
    let oriented = wire.edges();
    if oriented.len() != 1 {
        return Ok(None);
    }
    let profile_eid = oriented[0].edge();
    let edge = topo.edge(profile_eid)?;
    if edge.start() != edge.end() {
        return Ok(None);
    }
    let profile_vid = edge.start();
    let (center, radius, circ_normal) = match edge.curve() {
        EdgeCurve::Circle(c) => (c.center(), c.radius(), c.normal()),
        _ => return Ok(None),
    };

    let tol = Tolerance::new();
    // The axis must lie in the profile plane (perpendicular to its normal),
    // else the swept surface is not a torus of revolution.
    if normal.dot(axis).abs() > 1e-9 {
        return Ok(None);
    }

    // Major radius = perpendicular distance from the circle center to the axis.
    let to_center = center - axis_origin;
    let along = to_center.dot(axis);
    let major_radius = (to_center - axis * along).length();
    // The circle must clear the axis; otherwise the revolution is a sphere
    // (center on axis) or a self-intersecting spindle — both fall back.
    if major_radius <= radius + tol.linear {
        return Ok(None);
    }

    let torus_center = axis_origin + axis * along;
    let surface = brepkit_math::surfaces::ToroidalSurface::with_axis(
        torus_center,
        major_radius,
        radius,
        axis,
    )
    .map_err(crate::OperationsError::Math)?;

    if is_full {
        // One doubly-periodic torus face, like `primitives::make_torus`: a
        // single seam vertex with two degenerate seam edges forming the
        // fundamental polygon a → b → a⁻¹ → b⁻¹.
        let seam = surface.evaluate(0.0, 0.0);
        let v0 = topo.add_vertex(Vertex::new(seam, tol.linear));
        let ea = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Line));
        let eb = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Line));
        let wid = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(ea, true),
                    OrientedEdge::new(eb, true),
                    OrientedEdge::new(ea, false),
                    OrientedEdge::new(eb, false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?,
        );
        let face_id = topo.add_face(Face::new(wid, vec![], FaceSurface::Torus(surface)));
        let shell_id =
            topo.add_shell(Shell::new(vec![face_id]).map_err(crate::OperationsError::Topology)?);
        return Ok(Some(topo.add_solid(Solid::new(shell_id, vec![]))));
    }

    // Partial turn: the input circle bounds the start; its rotated copy bounds
    // the end; the seam is the arc the profile vertex sweeps (an axis-centred
    // `Circle3D` arc whose CCW start→end span is exactly `angle`).
    let p0 = topo.vertex(profile_vid)?.point();
    let (seam_r, seam_z) = radial_axial(p0, axis_origin, axis);
    if seam_r < tol.linear {
        return Ok(None); // profile vertex on the axis — cannot seam
    }
    let p1 = rotate_point(p0, axis_origin, axis, angle);
    let v1 = topo.add_vertex(Vertex::new(p1, tol.linear));
    let end_center = rotate_point(center, axis_origin, axis, angle);
    let end_normal = rotate_vec(circ_normal, axis, angle);
    let end_circle = brepkit_math::curves::Circle3D::new(end_center, end_normal, radius)
        .map_err(crate::OperationsError::Math)?;
    let end_eid = topo.add_edge(Edge::new(v1, v1, EdgeCurve::Circle(end_circle)));
    let seam_circle =
        brepkit_math::curves::Circle3D::new(axis_origin + axis * seam_z, axis, seam_r)
            .map_err(crate::OperationsError::Math)?;
    let seam_eid = topo.add_edge(Edge::new(profile_vid, v1, EdgeCurve::Circle(seam_circle)));

    let wall_wire = Wire::new(
        vec![
            OrientedEdge::new(profile_eid, true),
            OrientedEdge::new(seam_eid, true),
            OrientedEdge::new(end_eid, false),
            OrientedEdge::new(seam_eid, false),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let wall_wid = topo.add_wire(wall_wire);
    // Orient the band outward at the tube's outer equator, mid-sweep.
    let radial_c = (center - torus_center)
        .normalize()
        .map_err(crate::OperationsError::Math)?;
    let outer0 = center + radial_c * radius;
    let probe = rotate_point(outer0, axis_origin, axis, angle / 2.0);
    let (pu, pv) = surface.project_point(probe);
    let nat = surface.normal(pu, pv);
    let out_probe = probe - (axis_origin + axis * axis.dot(probe - axis_origin));
    let wall_reversed = nat.dot(out_probe) < 0.0;
    let mut faces = Vec::with_capacity(3);
    faces.push(if wall_reversed {
        topo.add_face(Face::new_reversed(
            wall_wid,
            vec![],
            FaceSurface::Torus(surface),
        ))
    } else {
        topo.add_face(Face::new(wall_wid, vec![], FaceSurface::Torus(surface)))
    });

    // End caps: planar discs whose outward normals oppose (start) and follow
    // (end) the sweep direction; each is bounded by its full profile circle,
    // wound CCW about the cap normal.
    let sweep0 = axis
        .cross(center - torus_center)
        .normalize()
        .map_err(crate::OperationsError::Math)?;
    let start_normal = -sweep0;
    let start_fwd = circ_normal.dot(start_normal) > 0.0;
    let start_wire = Wire::new(vec![OrientedEdge::new(profile_eid, start_fwd)], true)
        .map_err(crate::OperationsError::Topology)?;
    let start_wid = topo.add_wire(start_wire);
    faces.push(topo.add_face(Face::new(
        start_wid,
        vec![],
        FaceSurface::Plane {
            normal: start_normal,
            d: dot_normal_point(start_normal, p0),
        },
    )));
    let final_normal = rotate_vec(sweep0, axis, angle);
    let end_fwd = end_normal.dot(final_normal) > 0.0;
    let end_wire = Wire::new(vec![OrientedEdge::new(end_eid, end_fwd)], true)
        .map_err(crate::OperationsError::Topology)?;
    let end_wid = topo.add_wire(end_wire);
    faces.push(topo.add_face(Face::new(
        end_wid,
        vec![],
        FaceSurface::Plane {
            normal: final_normal,
            d: dot_normal_point(final_normal, p1),
        },
    )));

    let shell_id = topo.add_shell(Shell::new(faces).map_err(crate::OperationsError::Topology)?);
    Ok(Some(topo.add_solid(Solid::new(shell_id, vec![]))))
}

/// Revolve a face around an axis to produce a solid of revolution.
///
/// The profile surface may be planar or curved — only its boundary is used. A
/// full revolution (2π) has no caps, so it accepts any boundary; a partial
/// revolution closes its ends with planar caps and therefore requires a planar
/// profile boundary.
///
/// # Parameters
///
/// - `face` — a face whose outer wire defines the profile
/// - `axis_origin` — a point on the rotation axis
/// - `axis_direction` — direction of the rotation axis (will be normalized)
/// - `angle_radians` — rotation angle in radians, must be in (0, 2π]
///
/// When the input face has inner wires (holes), they are propagated:
/// inner wire edges generate inward-facing revolution surfaces, and
/// start/end cap faces include the inner wires as holes.
///
/// # Errors
///
/// Returns an error if the axis is zero-length, the angle is out of range, or a
/// partial revolution is requested for a non-planar profile boundary.
#[allow(clippy::too_many_lines)]
pub fn revolve(
    topo: &mut Topology,
    face: FaceId,
    axis_origin: Point3,
    axis_direction: Vec3,
    angle_radians: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if tol.approx_eq(axis_direction.length_squared(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "revolve axis direction is zero-length".into(),
        });
    }
    let axis = axis_direction.normalize()?;

    if angle_radians <= 0.0 || angle_radians > 2.0f64.mul_add(PI, tol.angular) {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("revolve angle must be in (0, 2π], got {angle_radians}"),
        });
    }

    let is_full = angle_radians >= 2.0f64.mul_add(PI, -tol.angular);
    let angle = if is_full { 2.0 * PI } else { angle_radians };

    let face_data = topo.face(face)?;
    // A planar profile keeps its stored plane normal; a non-planar surface has
    // its profile normal derived from the boundary below. The gate that rejected
    // every non-planar face is gone.
    let stored_plane_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Nurbs(_) => None,
    };
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

    // Fast exact path: a revolution of a single circular profile that clears
    // the axis is a torus (full turn: one doubly-periodic face; partial turn:
    // one trimmed band + two disc caps). Build it analytically instead of
    // faceting the circle into chords, which undershoots the analytic volume
    // by ~2% (gh #968).
    if let Some(solid) = try_circle_revolution_torus(topo, face, axis_origin, axis, angle, is_full)?
    {
        return Ok(solid);
    }

    // Fast exact path: a full revolution of a fully-analytic profile builds one
    // periodic face per profile edge (cylinder/cone/torus walls + planar disc
    // caps), matching the primitives — instead of the segmented NURBS bands.
    if let Some(solid) = try_analytic_full_revolution(topo, face, axis_origin, axis, is_full)? {
        return Ok(solid);
    }

    // Profile-plane normal for cap orientation and side-face winding. A planar
    // profile keeps its stored normal, corrected to the CCW convention (the
    // outer wire may be CW-wound, e.g. from brepjs). A non-planar surface
    // derives the normal from the boundary's Newell normal. Partial revolutions
    // close the ends with planar caps, so a non-planar boundary is only allowed
    // for a full revolution.
    let input_normal = {
        let wire = topo.wire(input_wire_id)?;
        let oes: Vec<_> = wire.edges().to_vec();
        let wire_positions: Vec<Point3> = oes
            .iter()
            .map(|oe| -> Result<Point3, crate::OperationsError> {
                let edge = topo.edge(oe.edge())?;
                let vid = if oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                Ok(topo.vertex(vid)?.point())
            })
            .collect::<Result<_, _>>()?;
        if let Some(normal) = stored_plane_normal {
            // Planar profile: keep the stored normal, corrected to CCW.
            if crate::winding::is_cw_winding(&wire_positions, &normal) {
                -normal
            } else {
                normal
            }
        } else if is_full {
            // A full revolution has no caps, so the profile normal is unused;
            // accept any boundary (including a single closed edge). Derive a
            // best-effort normal, falling back to the axis when degenerate.
            crate::winding::newell_normal(&wire_positions)
                .normalize()
                .unwrap_or(axis)
        } else {
            // A partial revolution closes its ends with planar caps, so the
            // boundary must be a planar polygon.
            if wire_positions.len() < 3 {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "partial revolve of a non-planar profile requires a polygonal boundary"
                        .into(),
                });
            }
            let normal = crate::winding::newell_normal(&wire_positions).normalize()?;
            let plane_pt = wire_positions[0];
            let max_dev = wire_positions
                .iter()
                .map(|p| (*p - plane_pt).dot(normal).abs())
                .fold(0.0, f64::max);
            let scale = wire_positions
                .iter()
                .map(|p| (*p - plane_pt).length())
                .fold(0.0, f64::max);
            if max_dev > PLANARITY_REL_TOL * scale {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "partial revolve of a non-planar profile boundary is not supported"
                        .into(),
                });
            }
            normal
        }
    };

    let (num_segs, seg_angle) = arc_segmentation(angle);
    let num_boundaries = if is_full { num_segs } else { num_segs + 1 };

    let revolve_wire = |topo: &mut Topology,
                        wire_id: brepkit_topology::wire::WireId|
     -> Result<WireRevolveData, crate::OperationsError> {
        let wire = topo.wire(wire_id)?;
        let original_oriented: Vec<_> = wire.edges().to_vec();

        // Split closed edges (e.g. full circles) into line segments.
        let input_oriented = crate::extrude::maybe_split_closed_wire(
            topo,
            &original_oriented,
            tol.linear,
            crate::extrude::DEFAULT_DEFLECTION,
        )?;
        let n = input_oriented.len();

        let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
        for oe in &input_oriented {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            input_verts.push(vid);
        }

        let input_positions: Vec<Point3> = input_verts
            .iter()
            .map(|&vid| {
                topo.vertex(vid)
                    .map(brepkit_topology::vertex::Vertex::point)
            })
            .collect::<Result<_, _>>()?;

        let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_boundaries);
        ring_verts.push(input_verts.clone());

        for k in 1..num_boundaries {
            #[allow(clippy::cast_precision_loss)]
            let theta = seg_angle * (k as f64);
            let ring: Vec<VertexId> = input_positions
                .iter()
                .map(|&pos| {
                    let rotated = rotate_point(pos, axis_origin, axis, theta);
                    topo.add_vertex(Vertex::new(rotated, tol.linear))
                })
                .collect();
            ring_verts.push(ring);
        }

        let mut arc_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segs);

        for seg in 0..num_segs {
            let next = next_ring_index(seg, num_segs, is_full);
            let mut seg_edges = Vec::with_capacity(n);
            for (&start_vid, &end_vid) in ring_verts[seg].iter().zip(&ring_verts[next]) {
                let start_pos = topo.vertex(start_vid)?.point();
                let end_pos = topo.vertex(end_vid)?.point();
                let curve = make_arc_curve(start_pos, end_pos, axis_origin, axis, seg_angle)?;
                seg_edges.push(topo.add_edge(Edge::new(
                    start_vid,
                    end_vid,
                    EdgeCurve::NurbsCurve(curve),
                )));
            }
            arc_edges.push(seg_edges);
        }

        let input_edge_ids: Vec<_> = input_oriented
            .iter()
            .map(brepkit_topology::wire::OrientedEdge::edge)
            .collect();

        let mut ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
            Vec::with_capacity(num_boundaries);
        ring_edges.push(input_edge_ids);

        for ring in ring_verts.iter().skip(1) {
            let edges: Vec<_> = (0..n)
                .map(|i| {
                    let next_i = (i + 1) % n;
                    topo.add_edge(Edge::new(ring[i], ring[next_i], EdgeCurve::Line))
                })
                .collect();
            ring_edges.push(edges);
        }

        Ok(WireRevolveData {
            ring_verts,
            arc_edges,
            ring_edges,
            input_oriented,
            n,
        })
    };

    let outer = revolve_wire(topo, input_wire_id)?;

    let mut inner_data: Vec<WireRevolveData> = Vec::new();
    for &iw_id in &inner_wire_ids {
        inner_data.push(revolve_wire(topo, iw_id)?);
    }

    // Collect input positions for outer wire (needed for cap face normals).
    let input_positions: Vec<Point3> = outer.ring_verts[0]
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    let mut all_faces = Vec::new();

    // Start cap (bottom): reversed copy of input face for partial revolution.
    if !is_full {
        let reversed_edges: Vec<OrientedEdge> = outer
            .input_oriented
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        let wire = Wire::new(reversed_edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.add_wire(wire);

        // Create inner wire holes for the bottom cap.
        let mut bottom_inner_wires = Vec::new();
        for iwd in &inner_data {
            let inner_reversed: Vec<OrientedEdge> = iwd
                .input_oriented
                .iter()
                .rev()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect();
            let iw = Wire::new(inner_reversed, true).map_err(crate::OperationsError::Topology)?;
            bottom_inner_wires.push(topo.add_wire(iw));
        }

        let bottom_normal = -input_normal;
        let bottom_d = dot_normal_point(bottom_normal, input_positions[0]);
        let fid = topo.add_face(Face::new(
            wid,
            bottom_inner_wires,
            FaceSurface::Plane {
                normal: bottom_normal,
                d: bottom_d,
            },
        ));
        all_faces.push(fid);
    }

    // Outer side NURBS faces.
    for seg in 0..num_segs {
        let next = next_ring_index(seg, num_segs, is_full);

        for i in 0..outer.n {
            let next_i = (i + 1) % outer.n;

            let fwd_seg = if seg == 0 {
                outer.input_oriented[i].is_forward()
            } else {
                true
            };
            let fwd_next = if next == 0 {
                outer.input_oriented[i].is_forward()
            } else {
                true
            };

            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(outer.ring_edges[seg][i], fwd_seg),
                    OrientedEdge::new(outer.arc_edges[seg][next_i], true),
                    OrientedEdge::new(outer.ring_edges[next][i], !fwd_next),
                    OrientedEdge::new(outer.arc_edges[seg][i], false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?;

            let side_wire_id = topo.add_wire(side_wire);

            let p0_start = topo.vertex(outer.ring_verts[seg][i])?.point();
            let p0_end = topo.vertex(outer.ring_verts[next][i])?.point();
            let p1_start = topo.vertex(outer.ring_verts[seg][next_i])?.point();
            let p1_end = topo.vertex(outer.ring_verts[next][next_i])?.point();

            let profile_curve = topo.edge(outer.input_oriented[i].edge())?.curve().clone();
            let (surface, reversed) = revolution_band_surface(
                &profile_curve,
                p0_start,
                p0_end,
                p1_start,
                p1_end,
                axis_origin,
                axis,
                seg_angle,
            )?;

            let fid = if reversed {
                topo.add_face(Face::new_reversed(side_wire_id, vec![], surface))
            } else {
                topo.add_face(Face::new(side_wire_id, vec![], surface))
            };
            all_faces.push(fid);
        }
    }

    // Inner side NURBS faces (reversed winding for inward-facing normals).
    for iwd in &inner_data {
        for seg in 0..num_segs {
            let next = next_ring_index(seg, num_segs, is_full);

            for i in 0..iwd.n {
                let next_i = (i + 1) % iwd.n;

                let fwd_seg = if seg == 0 {
                    iwd.input_oriented[i].is_forward()
                } else {
                    true
                };
                let fwd_next = if next == 0 {
                    iwd.input_oriented[i].is_forward()
                } else {
                    true
                };

                // Reversed winding: swap the order so normals point inward.
                let side_wire = Wire::new(
                    vec![
                        OrientedEdge::new(iwd.arc_edges[seg][i], true),
                        OrientedEdge::new(iwd.ring_edges[next][i], fwd_next),
                        OrientedEdge::new(iwd.arc_edges[seg][next_i], false),
                        OrientedEdge::new(iwd.ring_edges[seg][i], !fwd_seg),
                    ],
                    true,
                )
                .map_err(crate::OperationsError::Topology)?;

                let side_wire_id = topo.add_wire(side_wire);

                let p0_start = topo.vertex(iwd.ring_verts[seg][i])?.point();
                let p0_end = topo.vertex(iwd.ring_verts[next][i])?.point();
                let p1_start = topo.vertex(iwd.ring_verts[seg][next_i])?.point();
                let p1_end = topo.vertex(iwd.ring_verts[next][next_i])?.point();

                let surface = make_revolution_surface(
                    p0_start,
                    p0_end,
                    p1_start,
                    p1_end,
                    axis_origin,
                    axis,
                    seg_angle,
                )?;

                let fid =
                    topo.add_face(Face::new(side_wire_id, vec![], FaceSurface::Nurbs(surface)));
                all_faces.push(fid);
            }
        }
    }

    // End cap (top): rotated copy of the profile for partial revolution.
    if !is_full {
        let last_ring = num_boundaries - 1;
        let top_wire = Wire::new(
            outer.ring_edges[last_ring]
                .iter()
                .map(|&eid| OrientedEdge::new(eid, true))
                .collect(),
            true,
        )
        .map_err(crate::OperationsError::Topology)?;
        let top_wire_id = topo.add_wire(top_wire);

        // Create inner wire holes for the top cap.
        let mut top_inner_wires = Vec::new();
        for iwd in &inner_data {
            let inner_top_edges: Vec<OrientedEdge> = iwd.ring_edges[last_ring]
                .iter()
                .map(|&eid| OrientedEdge::new(eid, true))
                .collect();
            let iw = Wire::new(inner_top_edges, true).map_err(crate::OperationsError::Topology)?;
            top_inner_wires.push(topo.add_wire(iw));
        }

        let rotated_normal = rotate_vec(input_normal, axis, angle);
        let top_pos = topo.vertex(outer.ring_verts[last_ring][0])?.point();
        let top_d = dot_normal_point(rotated_normal, top_pos);

        let fid = topo.add_face(Face::new(
            top_wire_id,
            top_inner_wires,
            FaceSurface::Plane {
                normal: rotated_normal,
                d: top_d,
            },
        ));
        all_faces.push(fid);
    }

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = topo.add_solid(Solid::new(shell_id, vec![]));

    Ok(solid)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::f64::consts::PI;

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_square_face;

    use crate::test_helpers::{assert_euler_genus0, euler_characteristic};

    use super::*;

    /// Count mesh edges used by exactly one triangle (open boundary edges); 0 ⇒
    /// watertight.
    fn mesh_boundary_edges(mesh: &crate::tessellate::TriangleMesh) -> usize {
        use std::collections::HashMap;
        let mut ec: HashMap<(u32, u32), i32> = HashMap::new();
        for t in mesh.indices.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                *ec.entry(key).or_insert(0) += 1;
            }
        }
        ec.values().filter(|&&c| c == 1).count()
    }

    #[test]
    fn revolve_square_full_circle() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        // Unit square at (0,0,0)→(1,1,0), revolved 360° around the Y axis.
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // The unit square revolved about its x=0 edge is a unit cylinder. The
        // periodic merge yields 3 faces (1 `Cylinder` wall + 2 `Plane` disc caps)
        // like `make_cylinder`; the on-axis edge contributes no face.
        assert_eq!(
            shell.faces().len(),
            3,
            "merges to 3 faces like make_cylinder"
        );
        let cyl_count = shell
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(cyl_count, 1, "one periodic cylinder wall");
        let plane_count = shell
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Plane { .. }))
            .count();
        assert_eq!(plane_count, 2, "two planar disc caps");

        // The periodic cylinder seam must tessellate watertight (gh #696).
        for defl in [0.1_f64, 0.05, 0.02] {
            let mesh = crate::tessellate::tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                mesh_boundary_edges(&mesh),
                0,
                "watertight at deflection {defl}"
            );
        }

        // Exact unit-cylinder volume V = π (analytic walls + analytic disc caps).
        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        assert!(
            (vol - PI).abs() / PI < 1e-9,
            "expected exact unit cylinder volume π, got {vol}"
        );

        // A solid cylinder (the profile touches the axis) is genus-0 (χ=2). The
        // segmented path's degenerate on-axis bands previously faked a genus-1
        // χ=0; the periodic merge drops them, giving the correct genus-0.
        let chi = euler_characteristic(&topo, solid);
        assert_eq!(chi, 2, "solid cylinder is genus-0 (χ=2), got {chi}");
    }

    #[test]
    fn revolve_circle_full_turn_is_exact_torus() {
        // gh #968: revolving a circle (r=2, center x=10) a full turn around the
        // Y axis is a torus (R=10, r=2). It must be one analytic Torus face with
        // the exact analytic volume 2π²Rr², not a faceted ~2%-low approximation.
        use brepkit_topology::builder::make_circle_edge;

        let mut topo = Topology::new();
        let circle = make_circle_edge(
            &mut topo,
            Point3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            1e-7,
        )
        .unwrap();
        let wid = topo.add_wire(Wire::new(vec![OrientedEdge::new(circle, true)], true).unwrap());
        let profile = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let solid = revolve(
            &mut topo,
            profile,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();

        let shell = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            1,
            "torus is a single doubly-periodic face"
        );
        assert!(matches!(
            topo.face(shell.faces()[0]).unwrap().surface(),
            FaceSurface::Torus(_)
        ));

        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let expected = 2.0 * PI * PI * 10.0 * 4.0;
        assert!(
            (vol - expected).abs() / expected < 1e-6,
            "expected exact torus volume {expected}, got {vol}"
        );
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid()
        );
    }

    #[test]
    fn revolve_washer_walls_are_exact_cylinders() {
        // gh #968: a rectangular cross-section revolved a full turn (a washer)
        // has axis-parallel inner/outer walls that become exact analytic
        // cylinders (the inner wall reversed, facing the hole) and perpendicular
        // top/bottom edges that become annular `Plane` discs. With the analytic
        // disc-area volume the whole washer is exact (no disc-cap chording).
        use brepkit_topology::builder::make_polygon_wire;

        let mut topo = Topology::new();
        let wire = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(5.0, 0.0, 0.0),
                Point3::new(7.0, 0.0, 0.0),
                Point3::new(7.0, 0.0, 5.0),
                Point3::new(5.0, 0.0, 5.0),
            ],
            1e-7,
        )
        .unwrap();
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0 * PI,
        )
        .unwrap();

        let shell = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            4,
            "washer merges to 2 periodic cylinder walls + 2 annulus caps"
        );
        let cyl_count = shell
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(cyl_count, 2, "inner+outer walls are periodic cylinders");
        let plane_count = shell
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Plane { .. }))
            .count();
        assert_eq!(plane_count, 2, "top+bottom annulus caps are planar");
        // Each annulus cap keeps its smaller rim as a hole wire.
        let holed_caps = shell
            .faces()
            .iter()
            .filter(|&&fid| {
                let f = topo.face(fid).unwrap();
                matches!(f.surface(), FaceSurface::Plane { .. }) && f.inner_wires().len() == 1
            })
            .count();
        assert_eq!(holed_caps, 2, "both annulus caps carry an inner rim hole");

        // The merged shell must tessellate watertight.
        for defl in [0.1_f64, 0.02] {
            let mesh = crate::tessellate::tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                mesh_boundary_edges(&mesh),
                0,
                "washer mesh must be watertight at deflection {defl}"
            );
        }

        // Cylinder walls + analytic annular-sector disc caps ⇒ EXACT volume,
        // independent of deflection (the annular sectors must SUBTRACT their inner
        // arc segment — a reversed-inner-rim orientation bug would inflate it).
        let expected = PI * (49.0 - 25.0) * 5.0;
        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let vol_fine = crate::measure::solid_volume(&topo, solid, 0.0001).unwrap();
        assert!(
            (vol - expected).abs() / expected < 1e-6,
            "washer volume {expected}, got {vol}"
        );
        assert!(
            (vol - vol_fine).abs() < 1e-9,
            "washer volume must be analytic (deflection-independent): {vol} vs {vol_fine}"
        );
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid()
        );
    }

    #[test]
    fn revolve_frustum_walls_are_exact_cones() {
        // A full revolution of a fully-analytic frustum profile builds ONE
        // periodic cone wall + two planar disc caps — exactly matching
        // `make_cone` — instead of the segmented NURBS bands. Volume is exact.
        use brepkit_topology::builder::make_polygon_wire;
        use brepkit_topology::explorer::solid_faces;

        let mut topo = Topology::new();
        let (r_bot, r_top, h) = (6.0_f64, 2.0_f64, 12.0_f64);
        let wire = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(r_bot, 0.0, 0.0),
                Point3::new(r_top, 0.0, h),
                Point3::new(0.0, 0.0, h),
                Point3::new(0.0, 0.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0 * PI,
        )
        .unwrap();

        // Periodic merge → 3 faces (1 Cone + 2 Plane), matching make_cone(6,2,12).
        let faces = solid_faces(&topo, solid).unwrap();
        let count = |pred: fn(&FaceSurface) -> bool| {
            faces
                .iter()
                .filter(|&&fid| pred(topo.face(fid).unwrap().surface()))
                .count()
        };
        assert_eq!(faces.len(), 3, "frustum merges to 3 faces like make_cone");
        assert_eq!(
            count(|s| matches!(s, FaceSurface::Cone(_))),
            1,
            "one cone wall"
        );
        assert_eq!(
            count(|s| matches!(s, FaceSurface::Plane { .. })),
            2,
            "two planar disc caps"
        );
        assert_eq!(
            count(|s| matches!(s, FaceSurface::Nurbs(_))),
            0,
            "no NURBS bands"
        );

        // The periodic cone face's seam must tessellate watertight (gh #696).
        for defl in [0.1_f64, 0.05, 0.02] {
            let mesh = crate::tessellate::tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                mesh_boundary_edges(&mesh),
                0,
                "frustum mesh must be watertight at deflection {defl}"
            );
        }

        // One periodic cone wall + analytic planar disc caps ⇒ exact volume.
        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * h / 3.0 * r_bot.mul_add(r_bot, r_bot.mul_add(r_top, r_top * r_top));
        assert!(
            (vol - expected).abs() / expected < 1e-9,
            "frustum volume {expected}, got {vol}"
        );
    }

    #[test]
    fn revolve_arc_profile_edge_is_torus_band() {
        // A circular-ARC profile edge revolved a full turn sweeps an exact torus
        // band. A half-disc profile (semicircle arc bulging away from the axis,
        // closed by its diameter on an axis-parallel line) makes the arc bands
        // `Torus`; Pappus gives the exact revolved volume.
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        // Semicircle centre at radius D on the axis-parallel line x = D, radius ρ,
        // in the XZ plane (axis = Z). Arc from (D,0,−ρ) up through (D+ρ,0,0) to
        // (D,0,ρ); diameter line closes it along x = D.
        let (d, rho) = (10.0_f64, 3.0_f64);
        let circ = Circle3D::new(Point3::new(d, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), rho).unwrap();
        // Circle3D in XZ plane (normal +Y): param 0 → +x, so endpoints at the
        // bottom/top of the bulge are at angles −π/2 and +π/2.
        let p_bot = circ.evaluate(-std::f64::consts::FRAC_PI_2); // (D,0,−ρ)
        let p_top = circ.evaluate(std::f64::consts::FRAC_PI_2); // (D,0,+ρ)
        let v_bot = topo.add_vertex(Vertex::new(p_bot, 1e-7));
        let v_top = topo.add_vertex(Vertex::new(p_top, 1e-7));
        // Arc edge (bulging, the +x half) and the closing diameter line.
        let e_arc = topo.add_edge(Edge::new(v_bot, v_top, EdgeCurve::Circle(circ)));
        let e_dia = topo.add_edge(Edge::new(v_top, v_bot, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(e_arc, true),
                OrientedEdge::new(e_dia, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let face = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0 * PI,
        )
        .unwrap();

        let shell = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            2,
            "half-disc merges to 1 periodic torus band + 1 annulus cap"
        );
        let torus_count = shell
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Torus(_)))
            .count();
        assert_eq!(torus_count, 1, "the arc edge is ONE periodic torus band");
        assert_eq!(
            shell
                .faces()
                .iter()
                .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cone(_)))
                .count(),
            0,
            "an arc edge must not be chorded into cone bands"
        );

        // The torus band's seam is the profile arc itself; the whole shell must
        // tessellate watertight.
        for defl in [0.1_f64, 0.02] {
            let mesh = crate::tessellate::tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                mesh_boundary_edges(&mesh),
                0,
                "half-disc mesh must be watertight at deflection {defl}"
            );
        }

        // The revolved solid is the +z half of the torus tube, with the exact
        // closed-form volume `π²·R·ρ²` (R = D, the arc centroid's radial
        // distance). Because the solid is fully analytic (torus walls + planar
        // disc caps, no NURBS), the volume is integrated analytically — exact and
        // deflection-independent, not the inscribed-mesh approximation.
        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * PI * d * rho * rho;
        assert!(
            (vol - expected).abs() / expected < 1e-9,
            "half-torus tube volume {expected}, got {vol}"
        );
    }

    #[test]
    fn revolve_arc_profile_reversed_edge_torus_band() {
        // Same half-disc class as `revolve_arc_profile_edge_is_torus_band`, but
        // the arc edge is stored top→bottom and traversed REVERSED in the wire:
        // the analytic path's arc interior samples come from the curve's
        // natural direction and are flipped to traversal order — sampling with
        // traversal endpoints directly would pick the complementary arc and
        // invert the winding.
        use brepkit_math::curves::Circle3D;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let (d, rho) = (10.0_f64, 3.0_f64);
        let circ = Circle3D::new(Point3::new(d, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), rho).unwrap();
        let p_bot = circ.evaluate(-std::f64::consts::FRAC_PI_2); // (13,0,0)
        let p_top = circ.evaluate(std::f64::consts::FRAC_PI_2); // (7,0,0)
        let v_bot = topo.add_vertex(Vertex::new(p_bot, 1e-7));
        let v_top = topo.add_vertex(Vertex::new(p_top, 1e-7));
        // Stored top→bottom (the natural arc bulging through z > 0); the wire
        // walks it bottom→top via a reversed OrientedEdge, then closes with the
        // radial diameter at z = 0. The profile is the UPPER half-disc.
        let e_arc = topo.add_edge(Edge::new(v_top, v_bot, EdgeCurve::Circle(circ)));
        let e_dia = topo.add_edge(Edge::new(v_top, v_bot, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(e_arc, false),
                OrientedEdge::new(e_dia, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let face = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0 * PI,
        )
        .unwrap();

        let shell = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            2,
            "reversed-arc half-disc merges to 1 torus band + 1 annulus cap"
        );
        for defl in [0.1_f64, 0.02] {
            let mesh = crate::tessellate::tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                mesh_boundary_edges(&mesh),
                0,
                "reversed-arc mesh must be watertight at deflection {defl}"
            );
        }
        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * PI * d * rho * rho;
        assert!(
            (vol - expected).abs() / expected < 1e-9,
            "reversed-arc half-torus volume {expected}, got {vol}"
        );
    }

    #[test]
    fn revolve_pointed_cone_apex_band_volume_is_exact() {
        // A pointed cone (apex on the axis) exercises the apex-singularity guard
        // in `analytic_cone_signed_volume`: the apex vertex has no defined angular
        // parameter, so a per-segment band touching it must not corrupt the
        // band's angular range (which would 2× the lateral integral, +50% volume).
        use brepkit_topology::builder::make_polygon_wire;

        let mut topo = Topology::new();
        let (r, h) = (5.0_f64, 12.0_f64);
        // Triangle profile: base (r,0) → apex (0,h) → axis (0,0).
        let wire = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(r, 0.0, 0.0),
                Point3::new(0.0, 0.0, h),
                Point3::new(0.0, 0.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0 * PI,
        )
        .unwrap();

        // The apex-touching wall merges to ONE periodic cone (with `make_cone`'s
        // degenerate seam wire) + one disc cap — no 4×90° segmentation.
        let shell = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            2,
            "pointed cone merges to 1 periodic cone wall + 1 disc cap"
        );
        assert_eq!(
            shell
                .faces()
                .iter()
                .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cone(_)))
                .count(),
            1,
            "the apex wall is ONE periodic cone"
        );
        for defl in [0.1_f64, 0.02] {
            let mesh = crate::tessellate::tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                mesh_boundary_edges(&mesh),
                0,
                "pointed cone mesh must be watertight at deflection {defl}"
            );
        }

        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * r * r * h / 3.0;
        assert!(
            (vol - expected).abs() / expected < 1e-9,
            "pointed cone volume {expected}, got {vol}"
        );
    }

    #[test]
    fn revolve_circle_partial_turn_is_trimmed_torus() {
        // A circle profile clearing the axis, revolved a PARTIAL turn, is one
        // trimmed `Torus` band + two planar disc caps — not segmented patches.
        // Exact sector volume: `V = π·R·ρ²·Δu`.
        use brepkit_math::curves::Circle3D;

        let (big_r, rho, angle) = (6.0_f64, 2.0_f64, 2.0 * PI / 3.0);
        let mut topo = Topology::new();
        let circ =
            Circle3D::new(Point3::new(big_r, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), rho).unwrap();
        let p0 = circ.evaluate(0.0);
        let v0 = topo.add_vertex(Vertex::new(p0, 1e-7));
        let eid = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(circ)));
        let wire = Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        let face = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            angle,
        )
        .unwrap();

        let shell = topo
            .shell(topo.solid(solid).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            3,
            "partial-turn circle revolve is 1 torus band + 2 disc caps"
        );
        assert_eq!(
            shell
                .faces()
                .iter()
                .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Torus(_)))
                .count(),
            1,
            "the swept band is ONE trimmed torus face"
        );
        assert_eq!(
            shell
                .faces()
                .iter()
                .filter(|&&fid| matches!(
                    topo.face(fid).unwrap().surface(),
                    FaceSurface::Plane { .. }
                ))
                .count(),
            2,
            "the sweep ends are planar disc caps"
        );

        let report = crate::validate::validate_solid(&topo, solid).unwrap();
        assert!(report.is_valid(), "partial torus invalid: {report:?}");
        // Material check: the watertight mesh's SIGNED volume (positive ⇒
        // outward-wound) must approach the closed form from below — a band
        // swept on the wrong side of the tube, or an inside-out shell, lands
        // nowhere near it. (Ray-cast classification is not usable here: the
        // band's wire has only 2 distinct vertices, so the UV-containment
        // classifiers degrade it to a full-surface face and misclassify.)
        let expected = PI * big_r * rho * rho * angle;
        for defl in [0.1_f64, 0.02] {
            let mesh = crate::tessellate::tessellate_solid(&topo, solid, defl).unwrap();
            assert_eq!(
                mesh_boundary_edges(&mesh),
                0,
                "partial torus mesh must be watertight at deflection {defl}"
            );
            let mut vol6 = 0.0_f64;
            for t in mesh.indices.chunks(3) {
                let a = mesh.positions[t[0] as usize];
                let b = mesh.positions[t[1] as usize];
                let c = mesh.positions[t[2] as usize];
                let av = Vec3::new(a.x(), a.y(), a.z());
                let bv = Vec3::new(b.x(), b.y(), b.z());
                let cv = Vec3::new(c.x(), c.y(), c.z());
                vol6 += av.dot(bv.cross(cv));
            }
            let mesh_vol = vol6 / 6.0;
            assert!(
                mesh_vol > 0.0 && (expected - mesh_vol) / expected < 0.02 && mesh_vol < expected,
                "inscribed mesh volume {mesh_vol} must approach {expected} from below \
                 at deflection {defl}"
            );
        }

        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        assert!(
            (vol - expected).abs() / expected < 1e-9,
            "torus sector volume {expected}, got {vol}"
        );
    }

    #[test]
    fn revolve_square_half_circle() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            PI,
        )
        .unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 180° = 2 segments × 4 profile edges + 2 planar end caps = 10 faces.
        // The unit square revolved about the Y axis has: an axis-parallel edge
        // (x=1 wall → Cylinder), two perpendicular edges (the z-faces → annular
        // Plane discs), and an on-axis edge (x=0 → degenerate NURBS).
        assert_eq!(shell.faces().len(), 10);

        let mut plane_count = 0;
        let mut cyl_count = 0;
        let mut nurbs_count = 0;
        for &fid in shell.faces() {
            match topo.face(fid).unwrap().surface() {
                FaceSurface::Plane { .. } => plane_count += 1,
                FaceSurface::Cylinder(_) => cyl_count += 1,
                FaceSurface::Nurbs(_) => nurbs_count += 1,
                _ => {}
            }
        }
        // 4 perpendicular-edge disc bands + 2 end caps.
        assert_eq!(plane_count, 6, "perpendicular bands + end caps are planar");
        assert_eq!(
            cyl_count, 2,
            "the axis-parallel wall's 2 bands are cylinders"
        );
        assert_eq!(
            nurbs_count, 2,
            "only the degenerate on-axis bands stay NURBS"
        );

        // Half revolution of a rectangle → genus-0 solid (χ=2).
        assert_euler_genus0(&topo, solid);
    }

    #[test]
    fn revolve_zero_angle_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let result = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn revolve_zero_axis_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let result = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            PI,
        );
        assert!(result.is_err());
    }

    /// Verify that revolving a square and then tessellating produces valid meshes.
    #[test]
    fn revolve_and_tessellate_roundtrip() {
        use crate::tessellate::tessellate;

        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            PI,
        )
        .unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let tol = Tolerance::new();

        for &fid in shell.faces() {
            let mesh = tessellate(&topo, fid, 0.25).unwrap();
            assert!(!mesh.positions.is_empty());
            assert!(!mesh.indices.is_empty());
            assert_eq!(mesh.positions.len(), mesh.normals.len());

            for normal in &mesh.normals {
                let len = normal.length();
                assert!(
                    tol.approx_eq(len, 1.0) || tol.approx_eq(len, 0.0),
                    "normal length should be ~1.0, got {len}"
                );
            }
        }
    }

    /// Helper: create a square face with a smaller square hole.
    fn make_face_with_hole(topo: &mut Topology) -> FaceId {
        // Outer: 2×1 rectangle at x=1..3, y=0..1 (offset from Y axis).
        let outer_pts = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(3.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let outer_wire =
            brepkit_topology::builder::make_polygon_wire(topo, &outer_pts, 1e-7).unwrap();

        // Inner: small 0.5×0.5 hole (CW winding).
        let inner_pts = vec![
            Point3::new(1.5, 0.25, 0.0),
            Point3::new(1.5, 0.75, 0.0),
            Point3::new(2.5, 0.75, 0.0),
            Point3::new(2.5, 0.25, 0.0),
        ];
        let inner_wire =
            brepkit_topology::builder::make_polygon_wire(topo, &inner_pts, 1e-7).unwrap();

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 0.0;
        let face = Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal, d },
        );
        topo.add_face(face)
    }

    #[test]
    fn revolve_face_with_hole_full_circle() {
        let mut topo = Topology::new();
        let face = make_face_with_hole(&mut topo);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // Outer: 4 edges × 4 segments = 16 faces.
        // Inner: 4 edges × 4 segments = 16 faces.
        // No caps for full revolution. Total = 32.
        assert_eq!(
            shell.faces().len(),
            32,
            "full revolve with hole: 16 outer + 16 inner = 32 faces"
        );

        // Full revolution of a face with a hole creates a genus-1 solid
        // (torus-like, outer + inner passage). χ = 0.
        let chi = euler_characteristic(&topo, solid);
        assert_eq!(chi, 0, "genus-1 revolve should have χ=0, got {chi}");
    }

    #[test]
    fn revolve_face_with_hole_partial() {
        let mut topo = Topology::new();
        let face = make_face_with_hole(&mut topo);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            PI, // half revolution
        )
        .unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // Outer: 4 edges × 2 segments = 8 NURBS side faces.
        // Inner: 4 edges × 2 segments = 8 NURBS side faces.
        // 2 planar caps (start + end) = 2.
        // Total = 18.
        assert_eq!(
            shell.faces().len(),
            18,
            "half revolve with hole: 8 outer + 8 inner + 2 caps = 18 faces"
        );

        // Caps should have inner wires (holes).
        let faces_with_holes = shell
            .faces()
            .iter()
            .filter(|&&fid| !topo.face(fid).unwrap().inner_wires().is_empty())
            .count();
        assert_eq!(
            faces_with_holes, 2,
            "start and end caps should both have inner wire holes"
        );
    }

    #[test]
    fn revolve_face_with_hole_positive_volume() {
        let mut topo = Topology::new();
        let face = make_face_with_hole(&mut topo);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();

        // By Pappus' centroid theorem: V = 2π × centroid_distance × area.
        // Outer: 2×1 rect at x=1..3, y=0..1. Centroid_x = 2.0, area = 2.0.
        // Inner: 1×0.5 rect at x=1.5..2.5, y=0.25..0.75. Centroid_x = 2.0, area = 0.5.
        // Net: V = 2π × (2.0×2.0 - 2.0×0.5) = 2π × 3.0 = 6π ≈ 18.85.
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let expected = 6.0 * PI;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "revolved hollow annular volume should be ~{expected:.2}, got {vol:.2} (rel_err={rel_err:.2e})"
        );
    }

    /// Revolve a unit square 360° around Y-axis → annular solid.
    ///
    /// By Pappus' centroid theorem:
    ///   V = 2π × centroid_distance_from_axis × area
    ///
    /// Unit square at (0,0)-(1,1) on XY plane, revolved around Y-axis.
    /// Centroid distance from Y-axis (x-axis distance) = 0.5.
    /// Area = 1.0.
    /// V = 2π × 0.5 × 1.0 = π ≈ 3.1416.
    #[test]
    fn revolve_square_full_volume() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        // V = 2π × 0.5 × 1.0 = π ≈ 3.1416
        let expected = PI;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "full revolution of unit square should have volume π ≈ {expected:.4}, \
             got {vol:.4} (rel_err={rel_err:.2e})"
        );
    }

    /// Revolve a unit square 180° → half-annular solid.
    /// V = π × centroid_distance × area = π × 0.5 × 1.0 = π/2 ≈ 1.5708.
    #[test]
    fn revolve_square_half_volume() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            PI,
        )
        .unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        // Half revolution: V = (angle/2π) × 2π × centroid × area
        //                    = π × 0.5 × 1.0 = π/2 ≈ 1.5708
        let expected = PI / 2.0;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "half revolution of unit square should have volume π/2 ≈ {expected:.4}, \
             got {vol:.4} (rel_err={rel_err:.2e})"
        );
    }

    /// Revolve a rectangle with offset from axis → larger annulus.
    ///
    /// Rectangle at x=2..4, y=0..3 (offset 2 units from Y-axis).
    /// Centroid_x = 3.0, area = 6.0.
    /// V = 2π × 3.0 × 6.0 = 36π ≈ 113.097.
    #[test]
    fn revolve_offset_rectangle_volume() {
        let mut topo = Topology::new();
        // Build manually: 2×3 rectangle at x=2..4, y=0..3.
        let pts = vec![
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(4.0, 3.0, 0.0),
            Point3::new(2.0, 3.0, 0.0),
        ];
        let wire = brepkit_topology::builder::make_polygon_wire(&mut topo, &pts, 1e-7).unwrap();
        let face = topo.add_face(brepkit_topology::face::Face::new(
            wire,
            vec![],
            brepkit_topology::face::FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        // V = 2π × centroid_x × area = 2π × 3.0 × 6.0 = 36π ≈ 113.097
        let expected = 36.0 * PI;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "revolved offset rectangle volume should be 36π ≈ {expected:.2}, \
             got {vol:.2} (rel_err={rel_err:.2e})"
        );
    }

    /// Revolve a CW-wound profile and verify the result has correct volume.
    #[test]
    fn revolve_cw_profile_produces_correct_solid() {
        use brepkit_math::vec::Vec3;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let tol_val = 1e-7;

        // CW-wound rectangle at x=2..3, y=0..1, z=0 (offset from Y axis).
        // CW order when viewed from +Z: (2,0)→(2,1)→(3,1)→(3,0)
        let v0 = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 0.0), tol_val));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(2.0, 1.0, 0.0), tol_val));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(3.0, 1.0, 0.0), tol_val));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(3.0, 0.0, 0.0), tol_val));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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
        let wid = topo.add_wire(wire);

        // CW winding → Newell normal is -Z
        let face = topo.add_face(Face::new(
            wid,
            vec![],
            brepkit_topology::face::FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        ));

        // Revolve 360° around Y axis
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        // Pappus: V = 2π × centroid_x × area = 2π × 2.5 × 1.0 = 5π ≈ 15.708
        let expected = 5.0 * PI;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "CW profile revolve volume should be 5π ≈ {expected:.2}, \
             got {vol:.2} (rel_err={rel_err:.2e})"
        );
    }

    /// A 2×3 rectangle at x=2..4, y=0..3 (planar boundary) whose *surface* is a
    /// cylinder — previously rejected by the planar-only gate. The revolve uses
    /// only the boundary, so the result matches the planar-profile case.
    fn cylinder_surface_rect(topo: &mut Topology) -> FaceId {
        let pts = vec![
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(4.0, 3.0, 0.0),
            Point3::new(2.0, 3.0, 0.0),
        ];
        let wire = brepkit_topology::builder::make_polygon_wire(topo, &pts, 1e-7).unwrap();
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        topo.add_face(brepkit_topology::face::Face::new(
            wire,
            vec![],
            FaceSurface::Cylinder(cyl),
        ))
    }

    #[test]
    fn revolve_nonplanar_surface_full_turn_volume() {
        let mut topo = Topology::new();
        let face = cylinder_surface_rect(&mut topo);
        assert!(
            !topo.face(face).unwrap().surface().is_planar(),
            "profile surface is non-planar (a cylinder)"
        );
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();
        // Pappus: V = 2π × centroid_x × area = 2π × 3 × 6 = 36π.
        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        let expected = 36.0 * PI;
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "non-planar-surface revolve volume should be 36π, got {vol}"
        );
    }

    #[test]
    fn revolve_nonplanar_surface_partial_volume() {
        // Partial revolve closes the ends with planar caps; the planar boundary
        // makes that exact.
        let mut topo = Topology::new();
        let face = cylinder_surface_rect(&mut topo);
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            PI,
        )
        .unwrap();
        // Half revolution: V = π × centroid_x × area = π × 3 × 6 = 18π.
        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        let expected = 18.0 * PI;
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "non-planar-surface partial revolve volume should be 18π, got {vol}"
        );
    }

    #[test]
    fn revolve_partial_nonplanar_boundary_is_rejected() {
        // A genuinely non-planar boundary (corners lifted off z=0) can't be
        // closed by a planar cap, so a partial revolve is rejected.
        let mut topo = Topology::new();
        let pts = vec![
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.6),
            Point3::new(4.0, 3.0, 0.0),
            Point3::new(2.0, 3.0, 0.6),
        ];
        let wire = brepkit_topology::builder::make_polygon_wire(&mut topo, &pts, 1e-7).unwrap();
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        let face = topo.add_face(brepkit_topology::face::Face::new(
            wire,
            vec![],
            FaceSurface::Cylinder(cyl),
        ));
        let result = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            PI,
        );
        assert!(
            result.is_err(),
            "partial revolve of a non-planar boundary must be rejected"
        );
    }

    #[test]
    fn revolve_full_nonplanar_boundary_is_accepted() {
        // A full revolution has no caps, so a genuinely non-planar boundary is
        // accepted (each boundary point traces a circle); the profile normal is
        // unused, so the boundary-vertex-count guard must not reject it.
        let mut topo = Topology::new();
        let pts = vec![
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.6),
            Point3::new(4.0, 3.0, 0.0),
            Point3::new(2.0, 3.0, 0.6),
        ];
        let wire = brepkit_topology::builder::make_polygon_wire(&mut topo, &pts, 1e-7).unwrap();
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        let face = topo.add_face(brepkit_topology::face::Face::new(
            wire,
            vec![],
            FaceSurface::Cylinder(cyl),
        ));
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * PI,
        )
        .unwrap();
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "full revolve of a non-planar boundary should have positive volume, got {vol}"
        );
    }
}
