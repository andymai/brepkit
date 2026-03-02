//! Pipe sweep: sweep a profile along a path with optional scaling guide.
//!
//! Extends the basic sweep by allowing a guide curve that controls
//! profile scaling along the path. Equivalent to
//! `BRepOffsetAPI_MakePipeShell` in `OpenCascade`.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;

/// Sweep a profile along a path with scaling controlled by a guide curve.
///
/// The guide curve defines how the profile scales at each point along
/// the path. The scale factor at parameter `t` is the distance from
/// the guide curve to the path curve at that parameter, divided by
/// the initial distance (at `t=0`). This means the profile starts at
/// its original size and scales proportionally along the path.
///
/// If no guide is provided, this behaves identically to a regular sweep.
///
/// # Errors
///
/// Returns an error if the profile is not planar, the path is too short,
/// or the guide curve produces degenerate scaling.
#[allow(clippy::too_many_lines)]
pub fn pipe(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
    guide: Option<&NurbsCurve>,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    // Validate path.
    if path.control_points().len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "pipe path must have at least 2 control points".into(),
        });
    }

    let face_data = topo.face(profile)?;
    let input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Nurbs(_) => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "pipe of NURBS faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();

    if !face_data.inner_wires().is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "pipe of faces with holes is not supported".into(),
        });
    }

    // Collect profile vertices.
    let input_wire = topo.wire(input_wire_id)?;
    let input_oriented: Vec<_> = input_wire.edges().to_vec();
    let n = input_oriented.len();

    if n == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "pipe profile has no edges".into(),
        });
    }

    let mut input_verts = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        input_verts.push(topo.vertex(vid)?.point());
    }

    // Compute profile centroid.
    let (cx, cy, cz) = input_verts.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
        (ax + p.x(), ay + p.y(), az + p.z())
    });
    #[allow(clippy::cast_precision_loss)]
    let centroid = Point3::new(cx / n as f64, cy / n as f64, cz / n as f64);

    // Compute scaling factors from guide curve.
    let num_segments = (path.control_points().len() * 2).max(4);
    let scale_factors = compute_scale_factors(path, guide, num_segments, tol)?;

    // Compute frames along the path (reusing sweep's frame logic).
    let initial_tangent = path.tangent(0.0)?;
    let initial_up = orthogonalize(input_normal, initial_tangent);
    let initial_right = initial_tangent.cross(initial_up);

    // Build ring vertices with scaling.
    let mut ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> =
        Vec::with_capacity(num_segments + 1);

    for (k, &scale) in scale_factors.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let t_param = (k as f64) / (num_segments as f64);

        let origin = path.evaluate(t_param);
        let tangent = path.tangent(t_param)?;
        let up = orthogonalize(initial_up, tangent);
        let right = tangent.cross(up);

        let ring: Vec<_> = input_verts
            .iter()
            .map(|&pos| {
                let offset = pos - centroid;
                let local_r = initial_right.dot(offset) * scale;
                let local_u = initial_up.dot(offset) * scale;
                let local_t = initial_tangent.dot(offset);
                let transformed = origin + right * local_r + up * local_u + tangent * local_t;
                topo.vertices.alloc(Vertex::new(transformed, tol.linear))
            })
            .collect();
        ring_verts.push(ring);
    }

    // Build edges and faces (same topology as regular sweep).
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

    let mut all_faces = Vec::with_capacity(num_segments * n + 2);

    // Start cap.
    let start_reversed: Vec<OrientedEdge> = (0..n)
        .rev()
        .map(|i| OrientedEdge::new(ring_edges[0][i], false))
        .collect();
    let start_wire = Wire::new(start_reversed, true).map_err(crate::OperationsError::Topology)?;
    let start_wire_id = topo.wires.alloc(start_wire);
    let start_normal = -(path.tangent(0.0)?);
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

    // Side faces.
    for seg in 0..num_segments {
        for i in 0..n {
            let next_i = (i + 1) % n;
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

    // End cap.
    let end_edges: Vec<OrientedEdge> = (0..n)
        .map(|i| OrientedEdge::new(ring_edges[num_segments][i], true))
        .collect();
    let end_wire = Wire::new(end_edges, true).map_err(crate::OperationsError::Topology)?;
    let end_wire_id = topo.wires.alloc(end_wire);
    let end_normal = path.tangent(1.0)?;
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

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

/// Compute scale factors along the path from the guide curve.
///
/// At each sample point, the scale is the ratio of the guide-to-path
/// distance at that point versus the initial distance at t=0.
fn compute_scale_factors(
    path: &NurbsCurve,
    guide: Option<&NurbsCurve>,
    num_segments: usize,
    tol: Tolerance,
) -> Result<Vec<f64>, crate::OperationsError> {
    let Some(guide) = guide else {
        // No guide: uniform scale of 1.0.
        return Ok(vec![1.0; num_segments + 1]);
    };

    let initial_dist = (guide.evaluate(0.0) - path.evaluate(0.0)).length();
    if initial_dist < tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: "guide curve coincides with path at t=0 (zero initial distance)".into(),
        });
    }

    let mut factors = Vec::with_capacity(num_segments + 1);
    for k in 0..=num_segments {
        #[allow(clippy::cast_precision_loss)]
        let t = (k as f64) / (num_segments as f64);
        let dist = (guide.evaluate(t) - path.evaluate(t)).length();
        factors.push(dist / initial_dist);
    }

    Ok(factors)
}

/// Project `v` to be perpendicular to `tangent`, then normalize.
fn orthogonalize(v: Vec3, tangent: Vec3) -> Vec3 {
    let projected = v - tangent * tangent.dot(v);
    projected.normalize().unwrap_or_else(|_| {
        let candidate = if tangent.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let proj2 = candidate - tangent * tangent.dot(candidate);
        proj2.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::vec::Point3;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    fn straight_z_path(length: f64) -> NurbsCurve {
        NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, length)],
            vec![1.0, 1.0],
        )
        .unwrap()
    }

    fn diverging_guide(start_dist: f64, end_dist: f64) -> NurbsCurve {
        NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                Point3::new(start_dist, 0.0, 0.0),
                Point3::new(end_dist, 0.0, 2.0),
            ],
            vec![1.0, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn pipe_without_guide_is_like_sweep() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = straight_z_path(2.0);

        let solid = pipe(&mut topo, face, &path, None).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert!(!sh.faces().is_empty(), "pipe should produce faces");

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(vol > 0.5, "pipe should have positive volume, got {vol}");
    }

    #[test]
    fn pipe_with_expanding_guide() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = straight_z_path(2.0);
        let guide = diverging_guide(1.0, 2.0); // Scale doubles along path.

        let solid = pipe(&mut topo, face, &path, Some(&guide)).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert!(!sh.faces().is_empty());

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.5,
            "guided pipe should have positive volume, got {vol}"
        );
    }

    #[test]
    fn pipe_with_contracting_guide() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = straight_z_path(2.0);
        let guide = diverging_guide(2.0, 1.0); // Scale halves along path.

        let solid = pipe(&mut topo, face, &path, Some(&guide)).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.1,
            "contracting pipe should have positive volume, got {vol}"
        );
    }

    #[test]
    fn pipe_guide_at_origin_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = straight_z_path(2.0);

        // Guide starts at the same point as the path (distance = 0).
        let guide = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 2.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        assert!(pipe(&mut topo, face, &path, Some(&guide)).is_err());
    }

    #[test]
    fn pipe_short_path_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let path = NurbsCurve::new(
            0,
            vec![0.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0)],
            vec![1.0],
        )
        .unwrap();

        assert!(pipe(&mut topo, face, &path, None).is_err());
    }
}
