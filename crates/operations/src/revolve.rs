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
/// For a straight (line) profile edge, returns the **exact** analytic surface
/// of revolution — a `Cylinder` (edge parallel to the axis) or a `Plane` (edge
/// perpendicular to it) — so the band integrates exactly instead of inscribing
/// the swept arc as a NURBS band (~2% / ~0.04% deficit, gh #968). The surface is
/// oriented to agree with the correctly-wound NURBS band normal.
///
/// Oblique line edges (cones), curved profile edges, and degenerate on-axis
/// bands keep the NURBS band — they have no simpler exact form here.
#[allow(clippy::too_many_arguments)]
fn revolution_band_surface(
    profile_is_line: bool,
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
    if !profile_is_line {
        return Ok((FaceSurface::Nurbs(nurbs), false));
    }

    let tol = 1e-9;
    let (r0, _z0) = radial_axial(p0_start, axis_origin, axis);
    let (r1, _z1) = radial_axial(p1_start, axis_origin, axis);

    // Only the axis-parallel case (a cylinder wall) is converted: it has the
    // dominant inscription error and integrates exactly via the analytic
    // cylinder formula. A perpendicular edge would become a flat annular Plane,
    // but the planar tessellator samples its circular boundary more coarsely
    // than the rational-NURBS band does, so those caps stay NURBS.
    if r0 < tol || (r0 - r1).abs() >= tol {
        return Ok((FaceSurface::Nurbs(nurbs), false));
    }

    let surface = brepkit_math::surfaces::CylindricalSurface::new(axis_origin, axis, r0)?;
    // Match the NURBS bands' orientation so every face of the result winds the
    // same way (the volume integrator takes the absolute value of the total, so
    // consistency — not outward-vs-inward per se — is what matters). The NURBS
    // band `normal` (its du×dv) points into the swept material; the cylinder's
    // natural normal is radial-outward, so reverse the wall exactly when that
    // outward radial opposes the NURBS band normal.
    let center = rotate_point(p0_start, axis_origin, axis, seg_angle / 2.0);
    let radial = center - (axis_origin + axis * (center - axis_origin).dot(axis));
    let natural_outward = radial.normalize().unwrap_or(axis);
    Ok((
        FaceSurface::Cylinder(surface),
        natural_outward.dot(nurbs.normal(0.5, 0.5)?) < 0.0,
    ))
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
/// Returns `Ok(None)` — fall back to the general revolve — unless the profile
/// is a single closed circle, the revolution is full, the axis lies in the
/// profile plane, and the circle clears the axis (`major > minor`, so the
/// torus does not self-intersect; the sphere and spindle cases fall back).
fn try_circle_revolution_torus(
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
    let normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return Ok(None),
    };

    let wire = topo.wire(face_data.outer_wire())?;
    let oriented = wire.edges();
    if oriented.len() != 1 {
        return Ok(None);
    }
    let edge = topo.edge(oriented[0].edge())?;
    let (center, radius) = match edge.curve() {
        EdgeCurve::Circle(c) => (c.center(), c.radius()),
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

    // One doubly-periodic torus face, like `primitives::make_torus`: a single
    // seam vertex with two degenerate seam edges forming the fundamental
    // polygon a → b → a⁻¹ → b⁻¹.
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
    Ok(Some(topo.add_solid(Solid::new(shell_id, vec![]))))
}

/// Revolve a planar face around an axis to produce a solid of revolution.
///
/// # Parameters
///
/// - `face` — a planar face whose outer wire defines the profile
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
/// Returns an error if the axis is zero-length, the angle is out of range,
/// or the face surface is not a plane.
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
    let mut input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "revolve of non-planar faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

    // Fast exact path: a full revolution of a single circular profile that
    // clears the axis is a torus. Build it as one analytic `ToroidalSurface`
    // face instead of faceting the circle into chords, which undershoots the
    // analytic volume by ~2% (gh #968).
    if let Some(solid) = try_circle_revolution_torus(topo, face, axis_origin, axis, is_full)? {
        return Ok(solid);
    }

    // Ensure input_normal agrees with actual vertex winding (CCW convention).
    // If the outer wire is CW-wound (e.g. from brepjs), the Newell normal
    // opposes the stored normal. Correct by negating so that cap normals,
    // side face winding, and all downstream logic produce outward-facing faces.
    {
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
        if crate::winding::is_cw_winding(&wire_positions, &input_normal) {
            input_normal = -input_normal;
        }
    }

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

            let profile_is_line = matches!(
                topo.edge(outer.input_oriented[i].edge())?.curve(),
                EdgeCurve::Line
            );
            let (surface, reversed) = revolution_band_surface(
                profile_is_line,
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

        // 4 profile edges × 4 arc segments = 16 side faces. Revolving the unit
        // square around its x=0 edge yields a unit cylinder, so the x=1 edge's
        // four bands are exact analytic cylinders (gh #968); the perpendicular
        // caps and the degenerate on-axis edge stay NURBS.
        assert_eq!(shell.faces().len(), 16);
        let cyl_count = shell
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(cyl_count, 4, "the axis-parallel wall's bands are cylinders");

        // Revolving the unit square around its edge is a unit cylinder: V = π.
        // The exact-cylinder walls leave only the NURBS disc caps' boundary
        // tessellation residual (~0.006%, vs the ~2% of an all-faceted revolve).
        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        assert!(
            (vol - PI).abs() / PI < 1e-3,
            "expected unit cylinder volume π, got {vol}"
        );

        // Full 360° revolution of a rectangle → torus-like topology (genus-1, χ=0).
        // The profile sweeps fully around, merging start/end into a closed ring.
        let chi = euler_characteristic(&topo, solid);
        assert_eq!(chi, 0, "full revolve should have χ=0 (genus-1), got {chi}");
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
        // has axis-parallel inner/outer walls that must become exact analytic
        // cylinders — the inner wall reversed (faces the hole), the outer not.
        // The all-faceted revolve undershot by ~0.04%; the walls are now exact,
        // leaving only the NURBS disc-cap residual.
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
        let cyl_count = shell
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(cyl_count, 8, "inner+outer walls × 4 segments are cylinders");

        let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * (49.0 - 25.0) * 5.0;
        assert!(
            (vol - expected).abs() / expected < 1e-3,
            "washer volume {expected}, got {vol}"
        );
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid()
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

        // 180° = 2 segments × 4 edges = 8 NURBS + 2 planar caps = 10 faces.
        assert_eq!(shell.faces().len(), 10);

        let mut planar_count = 0;
        let mut nurbs_count = 0;
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            match f.surface() {
                FaceSurface::Plane { .. } => planar_count += 1,
                _ => nurbs_count += 1,
            }
        }
        assert_eq!(planar_count, 2, "should have 2 planar end caps");
        assert_eq!(nurbs_count, 8, "should have 8 NURBS side faces");

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
}
