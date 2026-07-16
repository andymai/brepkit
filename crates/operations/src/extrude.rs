//! Linear extrusion of faces along a direction vector.
//!
//! Supports both planar and NURBS profile faces. For NURBS faces, the
//! extrusion translates all control points, preserving the exact surface
//! representation for both caps.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;

/// Default tessellation deflection for splitting closed edges.
///
/// Used when the operation's public API does not expose a deflection parameter.
/// Matches `DEFAULT_BOOLEAN_DEFLECTION` in `boolean/types.rs`.
pub const DEFAULT_DEFLECTION: f64 = 0.1;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::WireId;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;

/// Data from extruding a single inner wire, needed for creating side faces.
struct InnerWireData {
    positions: Vec<Point3>,
    oriented: Vec<OrientedEdge>,
    edge_ids: Vec<EdgeId>,
    top_edge_ids: Vec<EdgeId>,
    vertical_edge_ids: Vec<EdgeId>,
}

/// Split closed single-edge wires (e.g. a full circle represented as one
/// NURBS edge with start==end) into multiple edges so that downstream
/// extrusion logic can create proper side faces.
///
/// The `deflection` parameter controls the maximum chord-height deviation
/// for circular/elliptical edges. For NURBS curves, the control polygon
/// bounding box is used to estimate the characteristic radius.
///
/// If no splitting is needed, returns the original edges unchanged.
///
/// # Errors
///
/// Returns an error if edge lookup fails.
pub fn maybe_split_closed_wire(
    topo: &mut Topology,
    oriented: &[OrientedEdge],
    tol: f64,
    deflection: f64,
) -> Result<Vec<OrientedEdge>, crate::OperationsError> {
    maybe_split_closed_wire_with(
        topo, oriented, tol, deflection, /*pass_through_circles=*/ false,
    )
}

/// Internal variant that lets the caller opt in to passing closed circle /
/// ellipse / NURBS-recognized-as-circle edges through unsplit. The basic
/// extrude path uses this so the side face can be built as a true analytic
/// cylinder (matching the exact π·r²·h), while sweep / complexExtrude /
/// twist still split — they apply per-edge profiles that don't work on a
/// single closed-curve edge.
pub(crate) fn maybe_split_closed_wire_with(
    topo: &mut Topology,
    oriented: &[OrientedEdge],
    tol: f64,
    deflection: f64,
    pass_through_circles: bool,
) -> Result<Vec<OrientedEdge>, crate::OperationsError> {
    let mut result = Vec::with_capacity(oriented.len() * 4);
    for oe in oriented {
        let edge = topo.edge(oe.edge())?;
        if edge.start() == edge.end()
            && !(pass_through_circles && curve_is_analytic_circle(edge.curve()))
        {
            let n = closed_edge_segments(edge.curve(), deflection);
            let split_edges = split_closed_edge(topo, oe.edge(), n, tol)?;
            for se in split_edges {
                result.push(OrientedEdge::new(se, oe.is_forward()));
            }
        } else {
            result.push(*oe);
        }
    }
    Ok(result)
}

/// Whether a closed edge's curve is a Circle, Ellipse, or a rational
/// NURBS recognized as one of those — i.e., something the extrude path
/// can convert into an exact analytic cylinder/NURBS-of-revolution side
/// face without polyline-ing.
fn curve_is_analytic_circle(curve: &EdgeCurve) -> bool {
    let tol = Tolerance::new().linear;
    match curve {
        EdgeCurve::Line => false,
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => true,
        EdgeCurve::NurbsCurve(nc) => matches!(
            brepkit_geometry::convert::recognize_curve(nc, tol * 100.0),
            brepkit_geometry::convert::RecognizedCurve::Circle { .. }
                | brepkit_geometry::convert::RecognizedCurve::Ellipse { .. }
        ),
    }
}

/// Whether an inner (hole) wire is a single closed *circle* — a true circle or
/// a rational NURBS recognized as one. This is the only hole the extrude path
/// turns into a single exact cylinder wall (with a known inward orientation).
///
/// Ellipses and generic closed curves are deliberately excluded: their
/// single-face pass-through wall is a ruled NURBS whose orientation can't be
/// derived from the degenerate `start==end` endpoints, so they keep the
/// chord-split path, which stays correct (if faceted).
fn inner_wire_is_single_circle(topo: &Topology, wire_id: WireId) -> bool {
    let Ok(wire) = topo.wire(wire_id) else {
        return false;
    };
    let edges = wire.edges();
    if edges.len() != 1 {
        return false;
    }
    let Ok(edge) = topo.edge(edges[0].edge()) else {
        return false;
    };
    if edge.start() != edge.end() {
        return false; // not a closed loop
    }
    let tol = Tolerance::new().linear;
    match edge.curve() {
        EdgeCurve::Circle(_) => true,
        EdgeCurve::NurbsCurve(nc) => matches!(
            brepkit_geometry::convert::recognize_curve(nc, tol * 100.0),
            brepkit_geometry::convert::RecognizedCurve::Circle { .. }
        ),
        _ => false,
    }
}

/// Compute the number of segments for splitting a closed edge based on
/// the chord-height deviation bound.
///
/// For circles, uses the exact radius. For ellipses, uses the semi-major
/// axis (conservative — highest curvature). For NURBS curves, estimates
/// a characteristic radius from the control polygon bounding box diagonal.
fn closed_edge_segments(curve: &EdgeCurve, deflection: f64) -> usize {
    use std::f64::consts::TAU;
    match curve {
        EdgeCurve::Circle(c) => {
            // Constant curvature: the chord formula is exact, so skip the
            // curvature floor that the convenience wrapper applies.
            brepkit_math::chord::segments_for_chord_deviation_with_angle(
                c.radius(),
                TAU,
                deflection,
                brepkit_math::chord::DEFAULT_ANGULAR_TOL,
                0.0,
                false,
            )
        }
        EdgeCurve::Ellipse(e) => {
            brepkit_math::chord::segments_for_chord_deviation(e.semi_major(), TAU, deflection)
        }
        EdgeCurve::NurbsCurve(nc) => {
            // Estimate characteristic radius from control polygon bounding box.
            let pts = nc.control_points();
            if pts.len() < 2 {
                return 8;
            }
            let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
            let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);
            for p in pts {
                min_x = min_x.min(p.x());
                min_y = min_y.min(p.y());
                min_z = min_z.min(p.z());
                max_x = max_x.max(p.x());
                max_y = max_y.max(p.y());
                max_z = max_z.max(p.z());
            }
            let dx = max_x - min_x;
            let dy = max_y - min_y;
            let dz = max_z - min_z;
            let diag = (dx * dx + dy * dy + dz * dz).sqrt();
            // Use half the diagonal as a conservative radius estimate.
            let radius_est = diag / 2.0;
            brepkit_math::chord::segments_for_chord_deviation(radius_est, TAU, deflection)
        }
        // Lines can't be closed with start==end in a meaningful way;
        // split_closed_edge already returns early for them.
        EdgeCurve::Line => 8,
    }
}

/// Split a closed edge (start==end) into `n` sub-edges by evaluating the
/// curve at evenly-spaced parameter values and creating new vertices/edges.
///
/// # Errors
///
/// Returns an error if edge lookup fails.
pub fn split_closed_edge(
    topo: &mut Topology,
    edge_id: EdgeId,
    n: usize,
    tol: f64,
) -> Result<Vec<EdgeId>, crate::OperationsError> {
    let edge = topo.edge(edge_id)?;
    let start_vid = edge.start();
    let curve = edge.curve().clone();

    let (u0, u1) = match &curve {
        EdgeCurve::NurbsCurve(nc) => nc.domain(),
        EdgeCurve::Circle(_) => (0.0, std::f64::consts::TAU),
        EdgeCurve::Ellipse(_) => (0.0, std::f64::consts::TAU),
        EdgeCurve::Line => {
            // Lines can't be closed with start==end in a meaningful way.
            return Ok(vec![edge_id]);
        }
    };

    let evaluate = |u: f64| -> Point3 {
        match &curve {
            EdgeCurve::NurbsCurve(nc) => nc.evaluate(u),
            EdgeCurve::Circle(c) => c.evaluate(u),
            EdgeCurve::Ellipse(e) => e.evaluate(u),
            // Line was handled above (early return).
            EdgeCurve::Line => Point3::new(0.0, 0.0, 0.0),
        }
    };

    let mut new_vids = Vec::with_capacity(n);
    new_vids.push(start_vid);
    for i in 1..n {
        #[allow(clippy::cast_precision_loss)]
        let u = u0 + (u1 - u0) * (i as f64) / (n as f64);
        let pt = evaluate(u);
        let vid = topo.add_vertex(Vertex::new(pt, tol));
        new_vids.push(vid);
    }

    // Create sub-edges, each as a Line between adjacent split vertices.
    // (The extrusion side faces only need vertex positions; the curve
    // representation for the side quad doesn't need to be exact.)
    let mut edge_ids = Vec::with_capacity(n);
    for i in 0..n {
        let v_start = new_vids[i];
        let v_end = new_vids[(i + 1) % n];
        // For the first segment start vertex, reuse the original.
        // For the last segment end vertex, wrap to the original start.
        let v_end_actual = if i == n - 1 { start_vid } else { v_end };
        let eid = topo.add_edge(Edge::new(v_start, v_end_actual, EdgeCurve::Line));
        edge_ids.push(eid);
    }

    Ok(edge_ids)
}

/// Sample points along a wire's oriented edges for winding detection.
///
/// Walking only the wire vertices is unreliable for wires with fewer than
/// three distinct vertices — notably a two-edge loop of one curved arc plus
/// one closing line (a half-ellipse or half-circle), whose two endpoints are
/// collinear and give the Newell normal zero signed area. Sampling the
/// interior of each curved edge (via its trimmed domain) yields a
/// non-degenerate polygon that captures the true winding, so the side-face
/// outward-normal heuristic in [`side_face_surface`] orients correctly.
fn winding_sample_points(
    topo: &Topology,
    oriented: &[OrientedEdge],
) -> Result<Vec<Point3>, crate::OperationsError> {
    let mut pts = Vec::with_capacity(oriented.len() * 3);
    for oe in oriented {
        let edge = topo.edge(oe.edge())?;
        let p_start = topo.vertex(edge.start())?.point();
        let p_end = topo.vertex(edge.end())?.point();
        let curve = edge.curve();
        // Sample fractions along the natural edge direction. Reverse them
        // when the edge is traversed backward so the polygon is walked in
        // wire order. Lines add only their endpoint; curved edges add
        // interior samples so the arc's bulge contributes signed area.
        let fracs: &[f64] = match curve {
            EdgeCurve::Line => &[0.0],
            _ => &[0.0, 0.25, 0.5, 0.75],
        };
        let (d0, d1) = curve.domain_with_endpoints(p_start, p_end);
        for &f in fracs {
            let f = if oe.is_forward() { f } else { 1.0 - f };
            let t = d0 + (d1 - d0) * f;
            pts.push(curve.evaluate_with_endpoints(t, p_start, p_end));
        }
    }
    Ok(pts)
}

/// Extract vertices, create offset (top) vertices and edges for a wire.
///
/// Returns: `(input_verts, input_positions, input_oriented, input_edge_ids,
///            top_verts, top_edge_ids, vertical_edge_ids)`
#[allow(clippy::type_complexity)]
fn extrude_wire_vertices_with(
    topo: &mut Topology,
    wire_id: WireId,
    offset: Vec3,
    pass_through_circles: bool,
) -> Result<
    (
        Vec<VertexId>,
        Vec<Point3>,
        Vec<OrientedEdge>,
        Vec<EdgeId>,
        Vec<VertexId>,
        Vec<EdgeId>,
        Vec<EdgeId>,
    ),
    crate::OperationsError,
> {
    let tol = Tolerance::new();
    let wire = topo.wire(wire_id)?;
    let original_oriented: Vec<_> = wire.edges().to_vec();

    // Check for closed single-edge wires (e.g. a full circle) and split them
    // into multiple edges so that the extrusion can create proper side faces.
    // Closed circles/ellipses (incl. NURBS-recognized ones) optionally pass
    // through unsplit when the caller can build analytic side faces: the outer
    // wire always does, and a single-circle inner (hole) wire does too (one
    // exact cylinder wall — see `inner_wire_is_single_circle`).
    let oriented = maybe_split_closed_wire_with(
        topo,
        &original_oriented,
        tol.linear,
        DEFAULT_DEFLECTION,
        pass_through_circles,
    )?;

    let mut verts: Vec<VertexId> = Vec::with_capacity(oriented.len());
    for oe in &oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        verts.push(vid);
    }

    let n = verts.len();

    let positions: Vec<Point3> = verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    let top_verts: Vec<VertexId> = positions
        .iter()
        .map(|p| {
            let top_point = *p + offset;
            topo.add_vertex(Vertex::new(top_point, tol.linear))
        })
        .collect();

    let edge_ids: Vec<EdgeId> = oriented
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect();

    let mut top_edge_ids = Vec::with_capacity(n);
    for i in 0..n {
        let next = (i + 1) % n;
        let bottom_curve = topo.edge(edge_ids[i])?.curve().clone();
        let mut top_curve = translate_edge_curve(&bottom_curve, offset)?;
        // The top edge's stored vertices (top_verts[i], top_verts[next]) follow
        // wire-traversal order. For a reversed edge those are swapped relative
        // to the curve's parameterization, so a periodic curve (ellipse/circle)
        // would resolve to the complementary sub-arc — and a NURBS curve's
        // endpoints would not match the stored vertices. Reverse the
        // translated curve's parameterization so it stays consistent.
        if !oriented[i].is_forward() {
            top_curve = reverse_edge_curve(&top_curve)?;
        }
        let top_edge = topo.add_edge(Edge::new(top_verts[i], top_verts[next], top_curve));
        top_edge_ids.push(top_edge);
    }

    let mut vertical_edge_ids = Vec::with_capacity(n);
    for i in 0..n {
        let vert_edge = topo.add_edge(Edge::new(verts[i], top_verts[i], EdgeCurve::Line));
        vertical_edge_ids.push(vert_edge);
    }

    Ok((
        verts,
        positions,
        oriented,
        edge_ids,
        top_verts,
        top_edge_ids,
        vertical_edge_ids,
    ))
}

/// Build the appropriate `FaceSurface` for a side face created by extruding
/// the given edge curve along the offset direction.
///
/// - `Line` edges produce planar faces.
/// - `Circle` edges produce cylindrical faces (the cylinder axis is the
///   extrusion direction, passing through the circle center).
/// - `NurbsCurve` and `Ellipse` edges produce ruled NURBS surfaces
///   interpolating between the bottom and translated-top curves.
///
/// Returns `(surface, is_reversed)` — `is_reversed` is `true` when the
/// surface's native normal points inward and the face needs flipping.
fn side_face_surface(
    curve: &EdgeCurve,
    p0: Point3,
    p1: Point3,
    // The edge's STORED start/end (its natural orientation), used to pick the
    // arc span. `p0`/`p1` are wire-traversal order and may be swapped (reversed
    // edge), which would select the complementary arc for ellipses/circles.
    curve_start: Point3,
    curve_end: Point3,
    offset: Vec3,
    outer_is_cw: bool,
) -> Result<(FaceSurface, bool), crate::OperationsError> {
    match curve {
        EdgeCurve::Line => {
            let edge_dir = p1 - p0;
            let normal = if outer_is_cw {
                offset.cross(edge_dir)
            } else {
                edge_dir.cross(offset)
            }
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let d = dot_normal_point(normal, p0);
            Ok((FaceSurface::Plane { normal, d }, false))
        }
        EdgeCurve::Circle(circle) => {
            // The extruded circle sweeps out a cylinder whose axis is the
            // extrusion direction, passing through the circle center.
            let cyl = brepkit_math::surfaces::CylindricalSurface::new(
                circle.center(),
                offset.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
                circle.radius(),
            )
            .map_err(crate::OperationsError::Math)?;
            // Check whether the cylinder's natural radial normal agrees with
            // the expected outward direction (same logic as NURBS reversal).
            let edge_dir = p1 - p0;
            let expected = if outer_is_cw {
                offset.cross(edge_dir)
            } else {
                edge_dir.cross(offset)
            };
            // Cylinder natural normal at p0: radial direction from axis.
            let to_pt = Vec3::new(
                p0.x() - circle.center().x(),
                p0.y() - circle.center().y(),
                p0.z() - circle.center().z(),
            );
            let along_axis = cyl.axis() * cyl.axis().dot(to_pt);
            let radial = to_pt - along_axis;
            let reversed = radial.dot(expected) < 0.0;
            Ok((FaceSurface::Cylinder(cyl), reversed))
        }
        EdgeCurve::NurbsCurve(nc) => {
            // brepjs (and other callers) often construct circles as rational
            // quadratic NURBS via `makeCircleEdge`. Extruding those as ruled
            // NURBS surfaces leaves a ~0.12% volume deficit vs the exact
            // π·r²·h answer. Recognize the NURBS as a
            // circle and use a true `Cylinder` surface for the side face
            // when possible — the recognition is geometrically exact for
            // the rational-quadratic circle construction.
            let tol = brepkit_math::tolerance::Tolerance::new().linear;
            if let brepkit_geometry::convert::RecognizedCurve::Circle {
                center,
                normal: _,
                radius,
            } = brepkit_geometry::convert::recognize_curve(nc, tol * 100.0)
            {
                let cyl = brepkit_math::surfaces::CylindricalSurface::new(
                    center,
                    offset.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
                    radius,
                )
                .map_err(crate::OperationsError::Math)?;
                let edge_dir = p1 - p0;
                let expected = if outer_is_cw {
                    offset.cross(edge_dir)
                } else {
                    edge_dir.cross(offset)
                };
                let to_pt = p0 - center;
                let along_axis = cyl.axis() * cyl.axis().dot(to_pt);
                let radial = to_pt - along_axis;
                let reversed = radial.dot(expected) < 0.0;
                return Ok((FaceSurface::Cylinder(cyl), reversed));
            }
            let surface = ruled_nurbs_surface(nc, offset)?;
            let reversed = nurbs_needs_reversal(&surface, p0, p1, offset, outer_is_cw);
            Ok((FaceSurface::Nurbs(surface), reversed))
        }
        EdgeCurve::Ellipse(ell) => {
            // Build the swept side from the edge's TRIMMED arc, not the full
            // ellipse. `ellipse_to_nurbs(full)` ignores the edge's start/end
            // trim, so a ruled surface over it sweeps the whole ellipse and
            // over-counts volume (#869). Recover the arc's angular domain from
            // the edge endpoints and build the rational-quadratic NURBS over
            // just that span (geometry's arc converter is exact at the arc
            // endpoints, so the side and planar cap share the boundary).
            let (t_start, t_end) = curve.domain_with_endpoints(curve_start, curve_end);
            let nc =
                brepkit_geometry::convert::ellipse_to_nurbs(ell, t_start, t_end).map_err(|e| {
                    crate::OperationsError::InvalidInput {
                        reason: format!("ellipse_to_nurbs failed: {e}"),
                    }
                })?;
            let surface = ruled_nurbs_surface(&nc, offset)?;
            let reversed = nurbs_needs_reversal(&surface, p0, p1, offset, outer_is_cw);
            Ok((FaceSurface::Nurbs(surface), reversed))
        }
    }
}

/// Check if the ruled NURBS surface's native normal disagrees with the
/// expected outward direction for this side face.
fn nurbs_needs_reversal(
    surface: &NurbsSurface,
    p0: Point3,
    p1: Point3,
    offset: Vec3,
    outer_is_cw: bool,
) -> bool {
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();
    let u_mid = 0.5 * (u_lo + u_hi);
    let v_mid = 0.5 * (v_lo + v_hi);
    let Ok(native) = surface.normal(u_mid, v_mid) else {
        return false;
    };

    // For an open arc the original "edge_dir.cross(offset)" formula gives the
    // expected outward direction. For a closed loop p0 == p1 — edge_dir
    // collapses to zero, the cross product is zero, and the dot test below
    // becomes useless (always false). Sample the bottom curve at a second
    // point and use the secant as the local edge direction instead, so
    // closed ellipse / NURBS-circle extrusions get the correct outward
    // orientation.
    let edge_dir = p1 - p0;
    let edge_dir = if edge_dir.length_squared() > 1e-20 {
        edge_dir
    } else {
        // Sample two points on the bottom curve a short parameter apart
        // and take their difference as the tangent direction.
        let v_next = if v_mid + 0.01 * (v_hi - v_lo) < v_hi {
            v_mid + 0.01 * (v_hi - v_lo)
        } else {
            v_mid - 0.01 * (v_hi - v_lo)
        };
        let p_mid = surface.evaluate(u_lo, v_mid);
        let p_next = surface.evaluate(u_lo, v_next);
        p_next - p_mid
    };
    let expected = if outer_is_cw {
        offset.cross(edge_dir)
    } else {
        edge_dir.cross(offset)
    };
    if expected.length_squared() < 1e-20 {
        return false;
    }
    native.dot(expected) < 0.0
}

/// Build a ruled NURBS surface by linearly interpolating between a bottom
/// NURBS curve and its translated copy (top curve).
///
/// The result is a degree `(curve_degree, 1)` surface with two rows of
/// control points: the original curve at `v=0` and the translated curve
/// at `v=1`.
fn ruled_nurbs_surface(
    nc: &brepkit_math::nurbs::curve::NurbsCurve,
    offset: Vec3,
) -> Result<NurbsSurface, crate::OperationsError> {
    let bottom_cps: Vec<Point3> = nc.control_points().to_vec();
    let top_cps: Vec<Point3> = bottom_cps.iter().map(|&p| p + offset).collect();
    let weights_row: Vec<f64> = nc.weights().to_vec();

    // Surface rows = u-direction (extrusion: 2 rows, degree 1)
    // Surface cols = v-direction (curve: n control points, original degree)
    NurbsSurface::new(
        1,                        // linear in extrusion direction (u = rows)
        nc.degree(),              // curve degree (v = columns)
        vec![0.0, 0.0, 1.0, 1.0], // clamped linear knot vector for 2 rows
        nc.knots().to_vec(),      // curve knot vector for columns
        vec![bottom_cps, top_cps],
        vec![weights_row.clone(), weights_row],
    )
    .map_err(crate::OperationsError::Math)
}

/// Recognize spline-encoded profile edges back to their analytic form,
/// in place.
///
/// 2D drawing pipelines ship corner-treated profiles as B-splines: a chamfer
/// segment and the corner-trimmed remainders of straight runs arrive as
/// degree-N NURBS even though they are exact lines, and fillet arcs can
/// arrive as rational splines. Extruding those emits ruled-NURBS side walls
/// for geometry that is exactly planar/cylindrical, and every downstream
/// boolean against the prism degrades (the snap-clip relief cut fell to a
/// mesh fallback purely because its clip walls were spline-encoded). The
/// caps' wires and the wall builder all read the same edges, so rewriting
/// the curve in place fixes every consumer at once.
fn normalize_profile_wire_curves(
    topo: &mut Topology,
    wire_id: WireId,
    tol: f64,
) -> Result<(), crate::OperationsError> {
    let edge_ids: Vec<brepkit_topology::edge::EdgeId> = topo
        .wire(wire_id)?
        .edges()
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect();
    for eid in edge_ids {
        let edge = topo.edge(eid)?;
        let EdgeCurve::NurbsCurve(nc) = edge.curve() else {
            continue;
        };
        let s3 = topo.vertex(edge.start())?.point();
        let e3 = topo.vertex(edge.end())?.point();
        let (d0, d1) = nc.domain();
        let a3 = nc.evaluate(d0);
        let b3 = nc.evaluate(d1);
        let mid3 = nc.evaluate(f64::midpoint(d0, d1));
        // Only convert when the spline's own endpoints coincide with the edge
        // vertices (either order) — anything else is a malformed profile this
        // pass must not reinterpret.
        let fwd = (a3 - s3).length() <= tol && (b3 - e3).length() <= tol;
        let rev = (a3 - e3).length() <= tol && (b3 - s3).length() <= tol;
        if !fwd && !rev {
            continue;
        }
        let new_curve = match brepkit_geometry::convert::recognize_curve(nc, tol * 100.0) {
            brepkit_geometry::convert::RecognizedCurve::Line { .. } if (s3 - e3).length() > tol => {
                Some(EdgeCurve::Line)
            }
            brepkit_geometry::convert::RecognizedCurve::Circle {
                center,
                normal,
                radius,
            } => {
                // Edges store arcs in the CCW convention: the span from the
                // START vertex to the END vertex runs counter-clockwise about
                // the curve normal. Pick the normal sign that puts the spline's
                // own midpoint on the CCW span from the start VERTEX (using the
                // curve-ordered endpoints, which lie exactly on the curve), or
                // an open arc would flip to its complement.
                let is_closed = (s3 - e3).length() <= tol;
                let n = if is_closed {
                    normal
                } else {
                    let (from, to) = if fwd { (a3, b3) } else { (b3, a3) };
                    let u = from - center;
                    let v = to - center;
                    let m = mid3 - center;
                    let ang = |w: Vec3| -> f64 {
                        u.cross(w)
                            .dot(normal)
                            .atan2(u.dot(w))
                            .rem_euclid(std::f64::consts::TAU)
                    };
                    let span = ang(v);
                    let mid_a = ang(m);
                    if span > 1e-12 && mid_a <= span {
                        normal
                    } else {
                        -normal
                    }
                };
                brepkit_math::curves::Circle3D::new(center, n, radius)
                    .ok()
                    .map(EdgeCurve::Circle)
            }
            _ => None,
        };
        if let Some(c) = new_curve {
            topo.edge_mut(eid)?.set_curve(c);
        }
    }
    Ok(())
}

/// Extrude a planar face along a direction to produce a solid.
///
/// The extrusion creates a prism-like solid from the face. A reversed copy of
/// the original face becomes the bottom (outward normal pointing opposite to
/// the extrusion direction), an offset copy becomes the top, and rectangular
/// side faces connect them.
///
/// When the input face has inner wires (holes), they are propagated:
/// - Both bottom and top cap faces include the inner wires as holes.
/// - Additional inward-facing side faces are created for each inner wire
///   edge, forming the interior walls of the hollow extrusion.
///
/// # Errors
///
/// Returns an error if the direction is zero-length, the face is not found,
/// or the face surface is not a plane.
#[allow(clippy::too_many_lines)]
pub fn extrude(
    topo: &mut Topology,
    face: FaceId,
    direction: Vec3,
    distance: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if tol.approx_eq(direction.length_squared(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "extrusion direction is zero-length".into(),
        });
    }

    if tol.approx_eq(distance, 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "extrusion distance is zero".into(),
        });
    }

    let face_data = topo.face(face)?;
    let mut input_surface = face_data.surface().clone();
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids: Vec<WireId> = face_data.inner_wires().to_vec();

    normalize_profile_wire_curves(topo, input_wire_id, tol.linear)?;
    for &iw in &inner_wire_ids {
        normalize_profile_wire_curves(topo, iw, tol.linear)?;
    }

    let offset = Vec3::new(
        direction.x() * distance,
        direction.y() * distance,
        direction.z() * distance,
    );

    let (
        input_verts,
        input_positions,
        input_oriented,
        input_edge_ids,
        _top_verts,
        top_edge_ids,
        vertical_edge_ids,
    ) = extrude_wire_vertices_with(
        topo,
        input_wire_id,
        offset,
        /*pass_through_circles=*/ true,
    )?;
    let n = input_verts.len();

    // Detect CW-wound outer wire (e.g. from brepjs polygon approximations).
    // CW winding makes `edge_dir.cross(offset)` point inward instead of outward;
    // the side-face surface builder uses this to flip side normals. Sample
    // along the edges (not just vertices) so a two-edge arc+line loop, whose
    // two endpoints alone give zero signed area, still winds correctly.
    let outer_winding_pts = winding_sample_points(topo, &input_oriented)?;
    let outer_is_cw = crate::winding::is_cw_winding(&outer_winding_pts, &offset);

    // Orient the cap-deriving normal by the extrusion direction, NOT the wire
    // winding. The two caps are at the input plane F and the swept plane F+offset;
    // the cap at F must face away from the sweep (−offset side) and the cap at
    // F+offset must face along it, regardless of how the profile wire was wound.
    // `bottom_surface`/`top_surface` below build the F-cap as −normal and the
    // F+offset-cap as +normal, so set `normal` to −normal whenever the extrusion
    // runs opposite the face normal (`normal·offset < 0`). This makes both cap
    // normals point outward for any winding (a CW profile extruded down — common
    // for brepjs dovetail tongues — previously got inside-out caps that broke the
    // downstream fuse). The cap wires stay consistent with the chosen normals
    // because the F-cap wire is the reversed profile and the F+offset-cap wire
    // keeps the profile winding.
    if let FaceSurface::Plane {
        ref mut normal,
        ref mut d,
    } = input_surface
        && normal.dot(offset) < 0.0
    {
        *normal = -*normal;
        *d = -*d;
    }

    let mut all_faces = Vec::with_capacity(n + 2 + inner_wire_ids.len() * 4);

    // Use the (possibly-split) edges so the bottom cap shares vertices
    // with the side faces, keeping the shell manifold.
    let reversed_bottom_edges: Vec<OrientedEdge> = input_oriented
        .iter()
        .rev()
        .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
        .collect();
    let bottom_wire =
        Wire::new(reversed_bottom_edges, true).map_err(crate::OperationsError::Topology)?;
    let bottom_wire_id = topo.add_wire(bottom_wire);

    let mut bottom_inner_wire_ids = Vec::with_capacity(inner_wire_ids.len());
    let mut top_inner_wire_ids = Vec::with_capacity(inner_wire_ids.len());

    let mut inner_wire_data: Vec<InnerWireData> = Vec::new();

    for &iw_id in &inner_wire_ids {
        let (
            _iw_verts,
            iw_positions,
            iw_oriented,
            iw_edge_ids,
            _iw_top_verts,
            iw_top_edge_ids,
            iw_vert_edge_ids,
        ) = extrude_wire_vertices_with(
            topo,
            iw_id,
            offset,
            inner_wire_is_single_circle(topo, iw_id),
        )?;

        // Bottom inner wire: reversed winding (same as outer wire reversal).
        let reversed_inner_edges: Vec<OrientedEdge> = iw_oriented
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        let bottom_inner_wire =
            Wire::new(reversed_inner_edges, true).map_err(crate::OperationsError::Topology)?;
        bottom_inner_wire_ids.push(topo.add_wire(bottom_inner_wire));

        // Top inner wire: same winding as bottom inner (reversed from original).
        let top_inner_edges: Vec<OrientedEdge> = iw_top_edge_ids
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let top_inner_wire =
            Wire::new(top_inner_edges, true).map_err(crate::OperationsError::Topology)?;
        top_inner_wire_ids.push(topo.add_wire(top_inner_wire));

        inner_wire_data.push(InnerWireData {
            positions: iw_positions,
            oriented: iw_oriented,
            edge_ids: iw_edge_ids,
            top_edge_ids: iw_top_edge_ids,
            vertical_edge_ids: iw_vert_edge_ids,
        });
    }

    let bottom_surface = match &input_surface {
        FaceSurface::Plane { normal, .. } => {
            let bottom_normal = -*normal;
            let bottom_d = dot_normal_point(bottom_normal, input_positions[0]);
            FaceSurface::Plane {
                normal: bottom_normal,
                d: bottom_d,
            }
        }
        FaceSurface::Nurbs(nurbs) => FaceSurface::Nurbs(nurbs.clone()),
        other => other.clone(),
    };
    let bottom_face = topo.add_face(Face::new(
        bottom_wire_id,
        bottom_inner_wire_ids,
        bottom_surface,
    ));
    all_faces.push(bottom_face);

    for i in 0..n {
        let next = (i + 1) % n;

        let side_wire = Wire::new(
            vec![
                OrientedEdge::new(input_edge_ids[i], input_oriented[i].is_forward()),
                OrientedEdge::new(vertical_edge_ids[next], true),
                OrientedEdge::new(top_edge_ids[i], false),
                OrientedEdge::new(vertical_edge_ids[i], false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;

        let side_wire_id = topo.add_wire(side_wire);

        let p0 = input_positions[i];
        let p1 = input_positions[next];
        let (edge_curve, e_start, e_end) = {
            let e = topo.edge(input_edge_ids[i])?;
            (e.curve().clone(), e.start(), e.end())
        };
        let cs = topo.vertex(e_start)?.point();
        let ce = topo.vertex(e_end)?.point();
        let (surface, reversed) =
            side_face_surface(&edge_curve, p0, p1, cs, ce, offset, outer_is_cw)?;

        let side_face = if reversed {
            topo.add_face(Face::new_reversed(side_wire_id, vec![], surface))
        } else {
            topo.add_face(Face::new(side_wire_id, vec![], surface))
        };
        all_faces.push(side_face);
    }

    // --- Inner wire side faces ---
    for iwd in &inner_wire_data {
        let iw_n = iwd.positions.len();

        // Detect inner wire winding direction relative to the face normal.
        // CW (negative signed area) is the standard B-Rep hole convention;
        // CCW (positive signed area) occurs when callers use math-convention
        // circle generation.  We support both.
        let is_cw = crate::winding::inner_wire_is_cw(&iwd.positions, &offset);

        for i in 0..iw_n {
            let next = (i + 1) % iw_n;

            let side_edges = if is_cw {
                // CW inner wire: traverse the quad in the reversed pattern
                // (up, across, down, back) so that the face normal points
                // into the hole (away from solid material).
                vec![
                    OrientedEdge::new(iwd.vertical_edge_ids[i], true),
                    OrientedEdge::new(iwd.top_edge_ids[i], true),
                    OrientedEdge::new(iwd.vertical_edge_ids[next], false),
                    OrientedEdge::new(iwd.edge_ids[i], !iwd.oriented[i].is_forward()),
                ]
            } else {
                // CCW inner wire: use the same winding pattern as outer
                // side faces (bottom-edge forward, right up, top back,
                // left down) which produces inward-pointing normals for
                // CCW inner geometry.
                vec![
                    OrientedEdge::new(iwd.edge_ids[i], iwd.oriented[i].is_forward()),
                    OrientedEdge::new(iwd.vertical_edge_ids[next], true),
                    OrientedEdge::new(iwd.top_edge_ids[i], false),
                    OrientedEdge::new(iwd.vertical_edge_ids[i], false),
                ]
            };

            let side_wire =
                Wire::new(side_edges, true).map_err(crate::OperationsError::Topology)?;
            let side_wire_id = topo.add_wire(side_wire);

            let p0 = iwd.positions[i];
            let p1 = iwd.positions[next];
            let (edge_curve, e_start, e_end) = {
                let e = topo.edge(iwd.edge_ids[i])?;
                (e.curve().clone(), e.start(), e.end())
            };
            let cs = topo.vertex(e_start)?.point();
            let ce = topo.vertex(e_end)?.point();
            // Inner wires have flipped winding relative to outer
            let inner_is_cw = !is_cw;
            let (surface, reversed) =
                side_face_surface(&edge_curve, p0, p1, cs, ce, offset, inner_is_cw)?;

            // A full closed circle passed through unsplit becomes one exact
            // cylinder wall, but its start==end vertex makes `edge_dir` zero, so
            // `side_face_surface` cannot derive the outward direction. A hole
            // wall always faces into the hole (toward the axis), i.e. opposite
            // the cylinder's natural outward radial normal, so force the flip.
            let reversed =
                reversed || (e_start == e_end && matches!(surface, FaceSurface::Cylinder(_)));

            let side_face = if reversed {
                topo.add_face(Face::new_reversed(side_wire_id, vec![], surface))
            } else {
                topo.add_face(Face::new(side_wire_id, vec![], surface))
            };
            all_faces.push(side_face);
        }
    }

    // --- Top face ---
    // Always use the split top_edge_ids so that the top cap shares vertices
    // and edges with the side faces, ensuring a closed (manifold) shell.
    let top_wire = Wire::new(
        top_edge_ids
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect(),
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let top_wire_id = topo.add_wire(top_wire);

    let top_surface = match &input_surface {
        FaceSurface::Plane { normal, .. } => {
            let top_d = dot_normal_point(*normal, input_positions[0] + offset);
            FaceSurface::Plane {
                normal: *normal,
                d: top_d,
            }
        }
        FaceSurface::Nurbs(nurbs) => {
            let translated_cps: Vec<Vec<Point3>> = nurbs
                .control_points()
                .iter()
                .map(|row| row.iter().map(|&p| p + offset).collect())
                .collect();
            let translated_surface = brepkit_math::nurbs::surface::NurbsSurface::new(
                nurbs.degree_u(),
                nurbs.degree_v(),
                nurbs.knots_u().to_vec(),
                nurbs.knots_v().to_vec(),
                translated_cps,
                nurbs.weights().to_vec(),
            )
            .map_err(crate::OperationsError::Math)?;
            FaceSurface::Nurbs(translated_surface)
        }
        FaceSurface::Cylinder(cyl) => FaceSurface::Cylinder(cyl.translated(offset)),
        FaceSurface::Cone(cone) => FaceSurface::Cone(cone.translated(offset)),
        FaceSurface::Sphere(sph) => FaceSurface::Sphere(sph.translated(offset)),
        FaceSurface::Torus(tor) => FaceSurface::Torus(tor.translated(offset)),
    };
    let top_face = topo.add_face(Face::new(top_wire_id, top_inner_wire_ids, top_surface));
    all_faces.push(top_face);

    // Assemble shell and solid.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = topo.add_solid(Solid::new(shell_id, vec![]));

    Ok(solid)
}

/// Translate an `EdgeCurve` by an offset vector.
///
/// # Errors
///
/// Returns an error if constructing the translated curve fails (should
/// never happen when translating a valid curve).
fn translate_edge_curve(
    curve: &EdgeCurve,
    offset: Vec3,
) -> Result<EdgeCurve, crate::OperationsError> {
    Ok(match curve {
        EdgeCurve::Line => EdgeCurve::Line,
        EdgeCurve::Circle(c) => {
            let new_center = c.center() + offset;
            EdgeCurve::Circle(
                brepkit_math::curves::Circle3D::with_axes(
                    new_center,
                    c.normal(),
                    c.radius(),
                    c.u_axis(),
                    c.v_axis(),
                )
                .map_err(crate::OperationsError::Math)?,
            )
        }
        EdgeCurve::Ellipse(e) => {
            let new_center = e.center() + offset;
            EdgeCurve::Ellipse(
                brepkit_math::curves::Ellipse3D::with_axes(
                    new_center,
                    e.normal(),
                    e.semi_major(),
                    e.semi_minor(),
                    e.u_axis(),
                    e.v_axis(),
                )
                .map_err(crate::OperationsError::Math)?,
            )
        }
        EdgeCurve::NurbsCurve(nc) => {
            let translated_cps: Vec<Point3> =
                nc.control_points().iter().map(|&p| p + offset).collect();
            EdgeCurve::NurbsCurve(
                brepkit_math::nurbs::curve::NurbsCurve::new(
                    nc.degree(),
                    nc.knots().to_vec(),
                    translated_cps,
                    nc.weights().to_vec(),
                )
                .map_err(crate::OperationsError::Math)?,
            )
        }
    })
}

/// Reverse a curve's intrinsic parameterization while preserving its geometry
/// (same point set, opposite direction).
///
/// For a periodic analytic curve (circle/ellipse) the parameter direction is
/// set by the binormal frame, so negating `normal` and `v_axis` flips
/// `project` to `-θ`; for a NURBS curve the control net, weights, and (mirrored)
/// knot vector are reversed. Used when copying an edge whose wire-traversal
/// order is opposite its stored direction, so the copied edge's endpoints stay
/// consistent with its curve.
fn reverse_edge_curve(curve: &EdgeCurve) -> Result<EdgeCurve, crate::OperationsError> {
    Ok(match curve {
        EdgeCurve::Line => EdgeCurve::Line,
        EdgeCurve::Circle(c) => EdgeCurve::Circle(
            brepkit_math::curves::Circle3D::with_axes(
                c.center(),
                -c.normal(),
                c.radius(),
                c.u_axis(),
                -c.v_axis(),
            )
            .map_err(crate::OperationsError::Math)?,
        ),
        EdgeCurve::Ellipse(e) => EdgeCurve::Ellipse(
            brepkit_math::curves::Ellipse3D::with_axes(
                e.center(),
                -e.normal(),
                e.semi_major(),
                e.semi_minor(),
                e.u_axis(),
                -e.v_axis(),
            )
            .map_err(crate::OperationsError::Math)?,
        ),
        EdgeCurve::NurbsCurve(nc) => {
            let knots = nc.knots();
            let span = knots[0] + knots[knots.len() - 1];
            let new_knots: Vec<f64> = knots.iter().rev().map(|&k| span - k).collect();
            let new_cps: Vec<Point3> = nc.control_points().iter().rev().copied().collect();
            let new_weights: Vec<f64> = nc.weights().iter().rev().copied().collect();
            EdgeCurve::NurbsCurve(
                brepkit_math::nurbs::curve::NurbsCurve::new(
                    nc.degree(),
                    new_knots,
                    new_cps,
                    new_weights,
                )
                .map_err(crate::OperationsError::Math)?,
            )
        }
    })
}

#[cfg(test)]
mod tests;
