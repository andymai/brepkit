//! Path sweep: sweep a profile along a NURBS curve.
//!
//! Creates a solid by moving a planar profile along an arbitrary NURBS curve
//! path, keeping the profile perpendicular to the path tangent at each sample
//! point. Uses rotation-minimizing frames (double-reflection method) to avoid
//! Frenet-frame singularities on straight segments and inflection points.

use brepkit_math::nurbs::curve::NurbsCurve;
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

/// A coordinate frame at a point along the path.
struct Frame {
    origin: Point3,
    tangent: Vec3,
    up: Vec3,
    right: Vec3,
}

/// Compute rotation-minimizing frames along a NURBS path.
///
/// Samples the path at `num_segments + 1` evenly-spaced parameter values and
/// propagates the initial up-vector using the double-reflection method to
/// produce smooth, twist-free frames.
fn compute_frames(
    path: &NurbsCurve,
    num_segments: usize,
    initial_up: Vec3,
) -> Result<Vec<Frame>, crate::OperationsError> {
    let mut frames = Vec::with_capacity(num_segments + 1);

    let t0 = path.tangent(0.0)?;
    let up0 = orthogonalize(initial_up, t0);
    let right0 = t0.cross(up0);
    frames.push(Frame {
        origin: path.evaluate(0.0),
        tangent: t0,
        up: up0,
        right: right0,
    });

    // Propagate frames using the double-reflection method (Wang et al. 2008).
    //
    // Two reflections per step:
    //   1. Reflect across the plane bisecting consecutive origins (position change).
    //   2. Reflect across the plane bisecting the reflected tangent and new tangent.
    for k in 1..=num_segments {
        #[allow(clippy::cast_precision_loss)]
        let t_param = (k as f64) / (num_segments as f64);

        let origin = path.evaluate(t_param);
        let tangent = path.tangent(t_param)?;

        let prev = &frames[k - 1];

        // Reflection 1: across the plane bisecting the two consecutive origins.
        let v1 = origin - prev.origin;
        let c1 = v1.dot(v1);
        let (up_l, tangent_l) = if c1 < 1e-30 {
            (prev.up, prev.tangent)
        } else {
            let up_r = prev.up - v1 * (2.0 * v1.dot(prev.up) / c1);
            let t_r = prev.tangent - v1 * (2.0 * v1.dot(prev.tangent) / c1);
            (up_r, t_r)
        };

        // Reflection 2: across the plane bisecting the reflected tangent
        // and the actual tangent at the new sample.
        let v2 = tangent - tangent_l;
        let c2 = v2.dot(v2);
        let up = if c2 < 1e-30 {
            orthogonalize(up_l, tangent)
        } else {
            let reflected = up_l - v2 * (2.0 * v2.dot(up_l) / c2);
            orthogonalize(reflected, tangent)
        };

        let right = tangent.cross(up);
        frames.push(Frame {
            origin,
            tangent,
            up,
            right,
        });
    }

    Ok(frames)
}

/// Project `v` to be perpendicular to `tangent`, then normalize.
///
/// Falls back to a world-axis-based vector if the projection is degenerate.
fn orthogonalize(v: Vec3, tangent: Vec3) -> Vec3 {
    let projected = v - tangent * tangent.dot(v);
    projected.normalize().unwrap_or_else(|_| {
        // Fallback: pick a world axis that isn't parallel to the tangent.
        let candidate = if tangent.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let proj2 = candidate - tangent * tangent.dot(candidate);
        // This should always succeed since candidate is chosen to not be
        // parallel to tangent.
        proj2.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
    })
}

/// Transform a profile vertex from its original position to a frame location.
///
/// The vertex's offset from the profile centroid is decomposed into the
/// initial frame's coordinate system (right, up, tangent), then
/// reconstructed in the target frame. Including the tangent component
/// ensures correct geometry even when the profile plane is not
/// perpendicular to the initial path tangent.
fn transform_point(
    point: Point3,
    centroid: Point3,
    initial_right: Vec3,
    initial_up: Vec3,
    initial_tangent: Vec3,
    frame: &Frame,
) -> Point3 {
    let offset = point - centroid;
    let local_r = initial_right.dot(offset);
    let local_u = initial_up.dot(offset);
    let local_t = initial_tangent.dot(offset);
    frame.origin + frame.right * local_r + frame.up * local_u + frame.tangent * local_t
}

/// Sweep a face along a path curve to produce a solid.
///
/// Creates a solid by moving a planar profile along a NURBS curve, with the
/// profile oriented perpendicular to the path tangent at each sample point.
/// Side faces are planar quads connecting consecutive profile rings.
///
/// # Errors
///
/// Returns an error if the profile is not planar, has inner wires (holes),
/// the path has fewer than 2 control points, or a degenerate tangent is
/// encountered.
#[allow(clippy::too_many_lines)]
pub fn sweep(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    // --- Validation ---

    if path.control_points().len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path must have at least 2 control points".into(),
        });
    }

    let face_data = topo.face(profile)?;
    let input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep of non-planar faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();

    if !face_data.inner_wires().is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep of faces with holes is not supported".into(),
        });
    }

    // Validate path has non-zero length.
    if tol.approx_eq(
        (path.evaluate(1.0) - path.evaluate(0.0)).length_squared(),
        0.0,
    ) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path has zero length (start and end coincide)".into(),
        });
    }

    // Collect profile vertices and positions.
    let input_wire = topo.wire(input_wire_id)?;
    let input_oriented: Vec<_> = input_wire.edges().to_vec();
    let n = input_oriented.len();

    if n == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }

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

    // Compute profile centroid.
    let (cx, cy, cz) = input_positions
        .iter()
        .fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
            (ax + p.x(), ay + p.y(), az + p.z())
        });
    #[allow(clippy::cast_precision_loss)]
    let centroid = Point3::new(cx / n as f64, cy / n as f64, cz / n as f64);

    // --- Compute frames along the path ---

    let num_segments = (path.control_points().len() * 2).max(4);

    // Seed the first frame's up-vector from the profile normal, projected
    // perpendicular to the path tangent at t=0.
    let up_hint = orthogonalize(input_normal, path.tangent(0.0)?);

    let frames = compute_frames(path, num_segments, up_hint)?;

    // The first frame's basis vectors define the local coordinate system
    // in which profile vertex offsets are expressed.
    let initial_right = frames[0].right;
    let initial_up = frames[0].up;
    let initial_tangent = frames[0].tangent;

    // --- Create ring vertices ---
    //
    // ring_verts[k][i] = vertex at path sample k, profile vertex i.

    let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_segments + 1);

    for frame in &frames {
        let ring: Vec<VertexId> = input_positions
            .iter()
            .map(|&pos| {
                let transformed = transform_point(
                    pos,
                    centroid,
                    initial_right,
                    initial_up,
                    initial_tangent,
                    frame,
                );
                topo.vertices.alloc(Vertex::new(transformed, tol.linear))
            })
            .collect();
        ring_verts.push(ring);
    }

    // --- Create profile edges within each ring ---
    //
    // ring_edges[k][i] = edge from ring_verts[k][i] to ring_verts[k][(i+1)%n].

    let mut ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
        Vec::with_capacity(num_segments + 1);
    for ring in &ring_verts {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.edges
                    .alloc(Edge::new(ring[i], ring[next], EdgeCurve::Line))
            })
            .collect();
        ring_edges.push(edges);
    }

    // --- Create path edges between consecutive rings ---
    //
    // path_edges[seg][i] = edge from ring_verts[seg][i] to ring_verts[seg+1][i].

    let mut path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segments);
    for seg in 0..num_segments {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                topo.edges.alloc(Edge::new(
                    ring_verts[seg][i],
                    ring_verts[seg + 1][i],
                    EdgeCurve::Line,
                ))
            })
            .collect();
        path_edges.push(edges);
    }

    // --- Build faces ---

    let mut all_faces = Vec::with_capacity(num_segments * n + 2);

    // Start cap: reversed first ring (outward normal pointing opposite to
    // path direction at the start).
    let start_reversed_edges: Vec<OrientedEdge> = (0..n)
        .rev()
        .map(|i| OrientedEdge::new(ring_edges[0][i], false))
        .collect();
    let start_wire =
        Wire::new(start_reversed_edges, true).map_err(crate::OperationsError::Topology)?;
    let start_wire_id = topo.wires.alloc(start_wire);

    let start_normal = -frames[0].tangent;
    let start_d = dot_normal_point(start_normal, topo.vertex(ring_verts[0][0])?.point());
    let start_face = topo.faces.alloc(Face::new(
        start_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: start_normal,
            d: start_d,
        },
    ));
    all_faces.push(start_face);

    // Side faces: one quad per profile-edge × path-segment.
    // Winding: ring_edge[seg][i](fwd) → path_edge[seg][next_i](fwd) →
    //          ring_edge[seg+1][i](rev) → path_edge[seg][i](rev).
    for seg in 0..num_segments {
        for i in 0..n {
            let next_i = (i + 1) % n;

            // Compute side normal from edge direction and path direction.
            let p0 = topo.vertex(ring_verts[seg][i])?.point();
            let p1 = topo.vertex(ring_verts[seg][next_i])?.point();
            let p_next = topo.vertex(ring_verts[seg + 1][i])?.point();
            let edge_dir = p1 - p0;
            let path_dir = p_next - p0;
            let side_normal = edge_dir
                .cross(path_dir)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let side_d = dot_normal_point(side_normal, p0);

            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(ring_edges[seg][i], true),
                    OrientedEdge::new(path_edges[seg][next_i], true),
                    OrientedEdge::new(ring_edges[seg + 1][i], false),
                    OrientedEdge::new(path_edges[seg][i], false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?;

            let side_wire_id = topo.wires.alloc(side_wire);
            let side_face = topo.faces.alloc(Face::new(
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

    // End cap: last ring with forward orientation (outward normal along
    // path tangent at the end).
    let end_edges: Vec<OrientedEdge> = (0..n)
        .map(|i| OrientedEdge::new(ring_edges[num_segments][i], true))
        .collect();
    let end_wire = Wire::new(end_edges, true).map_err(crate::OperationsError::Topology)?;
    let end_wire_id = topo.wires.alloc(end_wire);

    let end_normal = frames[num_segments].tangent;
    let end_d = dot_normal_point(
        end_normal,
        topo.vertex(ring_verts[num_segments][0])?.point(),
    );
    let end_face = topo.faces.alloc(Face::new(
        end_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: end_normal,
            d: end_d,
        },
    ));
    all_faces.push(end_face);

    // Assemble shell and solid.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    let solid = topo.solids.alloc(Solid::new(shell_id, vec![]));

    Ok(solid)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::HashMap;

    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::Point3;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    /// Helper: create a straight-line NURBS path from origin along +Z by `length`.
    fn straight_z_path(length: f64) -> NurbsCurve {
        NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, length)],
            vec![1.0, 1.0],
        )
        .unwrap()
    }

    /// Helper: create a quarter-circle NURBS path in the XZ plane.
    fn quarter_circle_xz_path(radius: f64) -> NurbsCurve {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(radius, 0.0, 0.0),
                Point3::new(radius, 0.0, radius),
            ],
            vec![1.0, w, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn sweep_square_along_line() {
        // Sweeping along a straight line should produce a box-like solid,
        // similar to extrude.
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = straight_z_path(2.0);

        let solid = sweep(&mut topo, face, &path).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 4 segments (from max(2*2, 4)) × 4 profile edges = 16 side faces
        // + 2 caps = 18 faces total.
        let num_segs = (path.control_points().len() * 2).max(4);
        let expected_faces = num_segs * 4 + 2;
        assert_eq!(shell.faces().len(), expected_faces);

        // Verify all faces are planar.
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            assert!(
                matches!(f.surface(), FaceSurface::Plane { .. }),
                "all sweep V1 faces should be planar"
            );
        }
    }

    #[test]
    fn sweep_square_along_quarter_circle() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = quarter_circle_xz_path(5.0);

        let solid = sweep(&mut topo, face, &path).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 6 segments (max(3*2, 4)) × 4 edges + 2 caps = 26 faces.
        let num_segs = (path.control_points().len() * 2).max(4);
        let expected_faces = num_segs * 4 + 2;
        assert_eq!(shell.faces().len(), expected_faces);

        // Verify manifold: every edge shared by exactly 2 faces.
        let mut edge_counts: HashMap<usize, usize> = HashMap::new();
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            let wire = topo.wire(f.outer_wire()).unwrap();
            for oe in wire.edges() {
                *edge_counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
        for (&edge_idx, &count) in &edge_counts {
            assert_eq!(
                count, 2,
                "edge {edge_idx} shared by {count} faces, expected 2"
            );
        }
    }

    #[test]
    fn sweep_insufficient_control_points_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        // A path with only 1 control point is invalid.
        let path = NurbsCurve::new(
            0,
            vec![0.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0)],
            vec![1.0],
        )
        .unwrap();

        let result = sweep(&mut topo, face, &path);
        assert!(result.is_err());
    }

    #[test]
    fn sweep_zero_path_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        // A path where start == end (zero length).
        let path = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(1.0, 2.0, 3.0), Point3::new(1.0, 2.0, 3.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        let result = sweep(&mut topo, face, &path);
        assert!(result.is_err());
    }

    #[test]
    fn sweep_and_tessellate_roundtrip() {
        use crate::tessellate::tessellate;

        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = quarter_circle_xz_path(5.0);

        let solid = sweep(&mut topo, face, &path).unwrap();

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
}
