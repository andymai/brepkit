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

    // --- Validation ---

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

    // Read input face.
    let face_data = topo.face(face)?;
    let input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "revolve of non-planar faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

    // --- Arc segmentation ---

    let (num_segs, seg_angle) = arc_segmentation(angle);
    let num_boundaries = if is_full { num_segs } else { num_segs + 1 };

    // --- Revolve a single wire, returning ring data ---

    let revolve_wire = |topo: &mut Topology,
                        wire_id: brepkit_topology::wire::WireId|
     -> Result<WireRevolveData, crate::OperationsError> {
        let wire = topo.wire(wire_id)?;
        let original_oriented: Vec<_> = wire.edges().to_vec();

        // Split closed edges (e.g. full circles) into line segments.
        let input_oriented =
            crate::extrude::maybe_split_closed_wire(topo, &original_oriented, tol.linear)?;
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

        // Create rotated vertices.
        let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_boundaries);
        ring_verts.push(input_verts.clone());

        for k in 1..num_boundaries {
            #[allow(clippy::cast_precision_loss)]
            let theta = seg_angle * (k as f64);
            let ring: Vec<VertexId> = input_positions
                .iter()
                .map(|&pos| {
                    let rotated = rotate_point(pos, axis_origin, axis, theta);
                    topo.vertices.alloc(Vertex::new(rotated, tol.linear))
                })
                .collect();
            ring_verts.push(ring);
        }

        // Create arc edges.
        let mut arc_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segs);

        for seg in 0..num_segs {
            let next = next_ring_index(seg, num_segs, is_full);
            let mut seg_edges = Vec::with_capacity(n);
            for (&start_vid, &end_vid) in ring_verts[seg].iter().zip(&ring_verts[next]) {
                let start_pos = topo.vertex(start_vid)?.point();
                let end_pos = topo.vertex(end_vid)?.point();
                let curve = make_arc_curve(start_pos, end_pos, axis_origin, axis, seg_angle)?;
                seg_edges.push(topo.edges.alloc(Edge::new(
                    start_vid,
                    end_vid,
                    EdgeCurve::NurbsCurve(curve),
                )));
            }
            arc_edges.push(seg_edges);
        }

        // Create ring edges.
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
                    topo.edges
                        .alloc(Edge::new(ring[i], ring[next_i], EdgeCurve::Line))
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

    // --- Revolve outer wire ---

    let outer = revolve_wire(topo, input_wire_id)?;

    // --- Revolve inner wires ---

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

    // --- Build faces ---

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
        let wid = topo.wires.alloc(wire);

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
            bottom_inner_wires.push(topo.wires.alloc(iw));
        }

        let bottom_normal = -input_normal;
        let bottom_d = dot_normal_point(bottom_normal, input_positions[0]);
        let fid = topo.faces.alloc(Face::new(
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

            let side_wire_id = topo.wires.alloc(side_wire);

            let p0_start = topo.vertex(outer.ring_verts[seg][i])?.point();
            let p0_end = topo.vertex(outer.ring_verts[next][i])?.point();
            let p1_start = topo.vertex(outer.ring_verts[seg][next_i])?.point();
            let p1_end = topo.vertex(outer.ring_verts[next][next_i])?.point();

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
                topo.faces
                    .alloc(Face::new(side_wire_id, vec![], FaceSurface::Nurbs(surface)));
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

                let side_wire_id = topo.wires.alloc(side_wire);

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
                    topo.faces
                        .alloc(Face::new(side_wire_id, vec![], FaceSurface::Nurbs(surface)));
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
        let top_wire_id = topo.wires.alloc(top_wire);

        // Create inner wire holes for the top cap.
        let mut top_inner_wires = Vec::new();
        for iwd in &inner_data {
            let inner_top_edges: Vec<OrientedEdge> = iwd.ring_edges[last_ring]
                .iter()
                .map(|&eid| OrientedEdge::new(eid, true))
                .collect();
            let iw = Wire::new(inner_top_edges, true).map_err(crate::OperationsError::Topology)?;
            top_inner_wires.push(topo.wires.alloc(iw));
        }

        let rotated_normal = rotate_vec(input_normal, axis, angle);
        let top_pos = topo.vertex(outer.ring_verts[last_ring][0])?.point();
        let top_d = dot_normal_point(rotated_normal, top_pos);

        let fid = topo.faces.alloc(Face::new(
            top_wire_id,
            top_inner_wires,
            FaceSurface::Plane {
                normal: rotated_normal,
                d: top_d,
            },
        ));
        all_faces.push(fid);
    }

    // Assemble shell and solid.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    let solid = topo.solids.alloc(Solid::new(shell_id, vec![]));

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

        // 4 profile edges × 4 arc segments = 16 NURBS faces, no planar caps.
        assert_eq!(shell.faces().len(), 16);

        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            assert!(
                matches!(f.surface(), FaceSurface::Nurbs(_)),
                "full revolution should only have NURBS faces"
            );
        }
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
        let outer_wire = brepkit_topology::builder::make_polygon_wire(topo, &outer_pts).unwrap();

        // Inner: small 0.5×0.5 hole (CW winding).
        let inner_pts = vec![
            Point3::new(1.5, 0.25, 0.0),
            Point3::new(1.5, 0.75, 0.0),
            Point3::new(2.5, 0.75, 0.0),
            Point3::new(2.5, 0.25, 0.0),
        ];
        let inner_wire = brepkit_topology::builder::make_polygon_wire(topo, &inner_pts).unwrap();

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 0.0;
        let face = Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal, d },
        );
        topo.faces.alloc(face)
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

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "revolved hollow solid should have positive volume"
        );
    }
}
