//! Deep copy of topological entities.
//!
//! Creates independent copies of solids and all their sub-entities
//! (shells, faces, wires, edges, vertices) in the arena.

use std::collections::HashMap;

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

// ── Snapshot types (read phase) ────────────────────────────────────

struct VertexSnap {
    old_index: usize,
    point: Point3,
    tol: f64,
}

struct EdgeSnap {
    old_index: usize,
    start_index: usize,
    end_index: usize,
    curve: EdgeCurve,
}

struct WireSnap {
    old_index: usize,
    edges: Vec<(usize, bool)>, // (edge_old_index, forward)
    closed: bool,
}

struct FaceSnap {
    outer_wire_index: usize,
    inner_wire_indices: Vec<usize>,
    surface: FaceSurface,
    reversed: bool,
}

struct ShellSnap {
    faces: Vec<FaceSnap>,
}

/// Create a deep copy of a solid and all its topology.
///
/// Returns a new `SolidId` for the copy. The original solid is not modified.
/// All vertices, edges, wires, faces, and shells are duplicated.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
#[allow(clippy::too_many_lines)]
pub fn copy_solid(
    topo: &mut Topology,
    solid_id: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    // ── Read phase: snapshot all data ──────────────────────────────

    let solid = topo.solid(solid_id)?;
    let outer_shell_id = solid.outer_shell();
    let inner_shell_ids: Vec<_> = solid.inner_shells().to_vec();

    let all_shell_ids: Vec<_> = std::iter::once(outer_shell_id)
        .chain(inner_shell_ids.iter().copied())
        .collect();

    let mut vertex_snaps: Vec<VertexSnap> = Vec::new();
    let mut edge_snaps: Vec<EdgeSnap> = Vec::new();
    let mut wire_snaps: Vec<WireSnap> = Vec::new();
    let mut shell_snaps: Vec<ShellSnap> = Vec::new();

    let mut seen_vertices = std::collections::HashSet::new();
    let mut seen_edges = std::collections::HashSet::new();
    let mut seen_wires = std::collections::HashSet::new();

    for &shell_id in &all_shell_ids {
        let shell = topo.shell(shell_id)?;
        let mut face_snaps = Vec::new();

        for &face_id in shell.faces() {
            let face = topo.face(face_id)?;
            let surface = face.surface().clone();
            let outer_wire_index = face.outer_wire().index();
            let inner_wire_indices: Vec<usize> =
                face.inner_wires().iter().map(|w| w.index()).collect();

            // Snapshot wires.
            for wire_id_val in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if !seen_wires.insert(wire_id_val.index()) {
                    continue;
                }
                let wire = topo.wire(wire_id_val)?;
                let mut edge_refs = Vec::new();

                for oe in wire.edges() {
                    let edge_idx = oe.edge().index();
                    edge_refs.push((edge_idx, oe.is_forward()));

                    if !seen_edges.insert(edge_idx) {
                        continue;
                    }
                    let edge = topo.edge(oe.edge())?;
                    let start_idx = edge.start().index();
                    let end_idx = edge.end().index();

                    // Snapshot vertices.
                    for &vid_idx in &[start_idx, end_idx] {
                        if seen_vertices.insert(vid_idx) {
                            let vid = if vid_idx == start_idx {
                                edge.start()
                            } else {
                                edge.end()
                            };
                            let v = topo.vertex(vid)?;
                            vertex_snaps.push(VertexSnap {
                                old_index: vid_idx,
                                point: v.point(),
                                tol: v.tolerance(),
                            });
                        }
                    }

                    edge_snaps.push(EdgeSnap {
                        old_index: edge_idx,
                        start_index: start_idx,
                        end_index: end_idx,
                        curve: edge.curve().clone(),
                    });
                }

                wire_snaps.push(WireSnap {
                    old_index: wire_id_val.index(),
                    edges: edge_refs,
                    closed: wire.is_closed(),
                });
            }

            face_snaps.push(FaceSnap {
                outer_wire_index,
                inner_wire_indices,
                surface,
                reversed: face.is_reversed(),
            });
        }

        shell_snaps.push(ShellSnap { faces: face_snaps });
    }

    // ── Write phase: allocate new entities ─────────────────────────

    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_vid = topo.vertices.alloc(Vertex::new(vsnap.point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    let mut edge_map: HashMap<usize, brepkit_topology::edge::EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let copied_edge = topo
            .edges
            .alloc(Edge::new(new_start, new_end, esnap.curve.clone()));
        edge_map.insert(esnap.old_index, copied_edge);
    }

    let mut wire_map: HashMap<usize, WireId> = HashMap::new();
    for wsnap in &wire_snaps {
        let new_edges: Vec<OrientedEdge> = wsnap
            .edges
            .iter()
            .map(|&(edge_idx, fwd)| OrientedEdge::new(edge_map[&edge_idx], fwd))
            .collect();
        let new_wire =
            Wire::new(new_edges, wsnap.closed).map_err(crate::OperationsError::Topology)?;
        wire_map.insert(wsnap.old_index, topo.wires.alloc(new_wire));
    }

    let mut new_shell_ids = Vec::new();
    for ssnap in &shell_snaps {
        let mut new_face_ids = Vec::new();
        for fsnap in &ssnap.faces {
            let new_outer = wire_map[&fsnap.outer_wire_index];
            let new_inner: Vec<WireId> = fsnap
                .inner_wire_indices
                .iter()
                .map(|idx| wire_map[idx])
                .collect();
            let new_face = if fsnap.reversed {
                Face::new_reversed(new_outer, new_inner, fsnap.surface.clone())
            } else {
                Face::new(new_outer, new_inner, fsnap.surface.clone())
            };
            let new_fid = topo.faces.alloc(new_face);
            new_face_ids.push(new_fid);
        }
        let new_shell = Shell::new(new_face_ids).map_err(crate::OperationsError::Topology)?;
        new_shell_ids.push(topo.shells.alloc(new_shell));
    }

    let new_outer = new_shell_ids[0];
    let new_inner: Vec<_> = new_shell_ids[1..].to_vec();

    Ok(topo.solids.alloc(Solid::new(new_outer, new_inner)))
}

/// Create a deep copy of a solid with a simultaneous affine transform.
///
/// Equivalent to `copy_solid` followed by `transform_solid`, but performs both
/// in a single traversal — applying the matrix during the write phase instead
/// of allocating untransformed entities and then mutating them.
///
/// # Errors
///
/// Returns an error if any topology lookup fails or the matrix is degenerate.
#[allow(clippy::too_many_lines)]
pub fn copy_and_transform_solid(
    topo: &mut Topology,
    solid_id: SolidId,
    matrix: &brepkit_math::mat::Mat4,
) -> Result<SolidId, crate::OperationsError> {
    use brepkit_math::nurbs::{NurbsCurve, NurbsSurface};
    use brepkit_math::vec::Vec3;

    let tol = brepkit_math::tolerance::Tolerance::new();
    if tol.approx_eq(matrix.determinant(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "transform matrix is degenerate (zero determinant)".into(),
        });
    }
    let normal_matrix = matrix.inverse()?.transpose();

    // Helper: transform a direction vector by applying the matrix.
    let transform_dir = |dir: Vec3| -> Result<Vec3, crate::OperationsError> {
        let origin = matrix.mul_point(Point3::new(0.0, 0.0, 0.0));
        let tip = matrix.mul_point(Point3::new(dir.x(), dir.y(), dir.z()));
        let raw = Vec3::new(
            tip.x() - origin.x(),
            tip.y() - origin.y(),
            tip.z() - origin.z(),
        );
        Ok(raw.normalize()?)
    };

    // Helper: transform a plane normal via inverse transpose.
    let transform_normal = |n: Vec3| -> Result<Vec3, crate::OperationsError> {
        let transformed = normal_matrix.mul_point(Point3::new(n.x(), n.y(), n.z()));
        let origin = normal_matrix.mul_point(Point3::new(0.0, 0.0, 0.0));
        let raw = Vec3::new(
            transformed.x() - origin.x(),
            transformed.y() - origin.y(),
            transformed.z() - origin.z(),
        );
        Ok(raw.normalize()?)
    };

    // ── Read phase: snapshot all data (identical to copy_solid) ────

    let solid = topo.solid(solid_id)?;
    let outer_shell_id = solid.outer_shell();
    let inner_shell_ids: Vec<_> = solid.inner_shells().to_vec();

    let all_shell_ids: Vec<_> = std::iter::once(outer_shell_id)
        .chain(inner_shell_ids.iter().copied())
        .collect();

    let mut vertex_snaps: Vec<VertexSnap> = Vec::new();
    let mut edge_snaps: Vec<EdgeSnap> = Vec::new();
    let mut wire_snaps: Vec<WireSnap> = Vec::new();
    let mut shell_snaps: Vec<ShellSnap> = Vec::new();

    let mut seen_vertices = std::collections::HashSet::new();
    let mut seen_edges = std::collections::HashSet::new();
    let mut seen_wires = std::collections::HashSet::new();

    for &shell_id in &all_shell_ids {
        let shell = topo.shell(shell_id)?;
        let mut face_snaps = Vec::new();

        for &face_id in shell.faces() {
            let face = topo.face(face_id)?;
            let surface = face.surface().clone();
            let outer_wire_index = face.outer_wire().index();
            let inner_wire_indices: Vec<usize> =
                face.inner_wires().iter().map(|w| w.index()).collect();

            for wire_id_val in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if !seen_wires.insert(wire_id_val.index()) {
                    continue;
                }
                let wire = topo.wire(wire_id_val)?;
                let mut edge_refs = Vec::new();

                for oe in wire.edges() {
                    let edge_idx = oe.edge().index();
                    edge_refs.push((edge_idx, oe.is_forward()));

                    if !seen_edges.insert(edge_idx) {
                        continue;
                    }
                    let edge = topo.edge(oe.edge())?;
                    let start_idx = edge.start().index();
                    let end_idx = edge.end().index();

                    for &vid_idx in &[start_idx, end_idx] {
                        if seen_vertices.insert(vid_idx) {
                            let vid = if vid_idx == start_idx {
                                edge.start()
                            } else {
                                edge.end()
                            };
                            let v = topo.vertex(vid)?;
                            vertex_snaps.push(VertexSnap {
                                old_index: vid_idx,
                                point: v.point(),
                                tol: v.tolerance(),
                            });
                        }
                    }

                    edge_snaps.push(EdgeSnap {
                        old_index: edge_idx,
                        start_index: start_idx,
                        end_index: end_idx,
                        curve: edge.curve().clone(),
                    });
                }

                wire_snaps.push(WireSnap {
                    old_index: wire_id_val.index(),
                    edges: edge_refs,
                    closed: wire.is_closed(),
                });
            }

            face_snaps.push(FaceSnap {
                outer_wire_index,
                inner_wire_indices,
                surface,
                reversed: face.is_reversed(),
            });
        }

        shell_snaps.push(ShellSnap { faces: face_snaps });
    }

    // ── Write phase: allocate transformed entities ─────────────────

    // Vertices: apply matrix during allocation.
    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_point = matrix.mul_point(vsnap.point);
        let new_vid = topo.vertices.alloc(Vertex::new(new_point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    // Edges: transform NURBS control points inline.
    let mut edge_map: HashMap<usize, brepkit_topology::edge::EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let new_curve = match &esnap.curve {
            EdgeCurve::Line => EdgeCurve::Line,
            EdgeCurve::NurbsCurve(c) => {
                let new_cps: Vec<_> = c
                    .control_points()
                    .iter()
                    .map(|pt| matrix.mul_point(*pt))
                    .collect();
                EdgeCurve::NurbsCurve(NurbsCurve::new(
                    c.degree(),
                    c.knots().to_vec(),
                    new_cps,
                    c.weights().to_vec(),
                )?)
            }
            EdgeCurve::Circle(c) => {
                let new_center = matrix.mul_point(c.center());
                let origin = matrix.mul_point(Point3::new(0.0, 0.0, 0.0));
                let transform_dir = |d: brepkit_math::vec::Vec3| -> brepkit_math::vec::Vec3 {
                    matrix.mul_point(Point3::new(d.x(), d.y(), d.z())) - origin
                };
                let new_u = transform_dir(c.u_axis());
                let new_v = transform_dir(c.v_axis());
                let su = new_u.length();
                let sv = new_v.length();
                let new_normal = new_u.cross(new_v).normalize()?;
                if (su - sv).abs() < 1e-12 * su.max(sv).max(1.0) {
                    EdgeCurve::Circle(Circle3D::with_axes(
                        new_center,
                        new_normal,
                        c.radius() * su,
                        new_u.normalize()?,
                        new_v.normalize()?,
                    )?)
                } else {
                    let (semi_major, semi_minor, u_dir, v_dir) = if su >= sv {
                        (
                            c.radius() * su,
                            c.radius() * sv,
                            new_u.normalize()?,
                            new_v.normalize()?,
                        )
                    } else {
                        (
                            c.radius() * sv,
                            c.radius() * su,
                            new_v.normalize()?,
                            new_u.normalize()?,
                        )
                    };
                    EdgeCurve::Ellipse(Ellipse3D::with_axes(
                        new_center, new_normal, semi_major, semi_minor, u_dir, v_dir,
                    )?)
                }
            }
            EdgeCurve::Ellipse(e) => {
                let new_center = matrix.mul_point(e.center());
                let origin = matrix.mul_point(Point3::new(0.0, 0.0, 0.0));
                let transform_dir = |d: brepkit_math::vec::Vec3| -> brepkit_math::vec::Vec3 {
                    matrix.mul_point(Point3::new(d.x(), d.y(), d.z())) - origin
                };
                let new_u = transform_dir(e.u_axis());
                let new_v = transform_dir(e.v_axis());
                let new_normal = new_u.cross(new_v).normalize()?;
                EdgeCurve::Ellipse(Ellipse3D::with_axes(
                    new_center,
                    new_normal,
                    e.semi_major() * new_u.length(),
                    e.semi_minor() * new_v.length(),
                    new_u.normalize()?,
                    new_v.normalize()?,
                )?)
            }
        };
        let copied_edge = topo.edges.alloc(Edge::new(new_start, new_end, new_curve));
        edge_map.insert(esnap.old_index, copied_edge);
    }

    // Wires: no geometry to transform.
    let mut wire_map: HashMap<usize, WireId> = HashMap::new();
    for wsnap in &wire_snaps {
        let new_edges: Vec<OrientedEdge> = wsnap
            .edges
            .iter()
            .map(|&(edge_idx, fwd)| OrientedEdge::new(edge_map[&edge_idx], fwd))
            .collect();
        let new_wire =
            Wire::new(new_edges, wsnap.closed).map_err(crate::OperationsError::Topology)?;
        wire_map.insert(wsnap.old_index, topo.wires.alloc(new_wire));
    }

    // Shells + faces: transform surfaces inline.
    let mut new_shell_ids = Vec::new();
    for ssnap in &shell_snaps {
        let mut new_face_ids = Vec::new();
        for fsnap in &ssnap.faces {
            let new_outer = wire_map[&fsnap.outer_wire_index];
            let new_inner: Vec<WireId> = fsnap
                .inner_wire_indices
                .iter()
                .map(|idx| wire_map[idx])
                .collect();

            let new_surface = match &fsnap.surface {
                FaceSurface::Plane { normal, .. } => {
                    let new_normal = transform_normal(*normal)?;
                    // Recompute d from a transformed vertex on this face.
                    let wire_ow = &wire_snaps
                        .iter()
                        .find(|w| w.old_index == fsnap.outer_wire_index);
                    let first_edge_idx = wire_ow.map(|w| w.edges[0].0).ok_or_else(|| {
                        crate::OperationsError::InvalidInput {
                            reason: "face has no outer wire edges".into(),
                        }
                    })?;
                    let esnap = edge_snaps.iter().find(|e| e.old_index == first_edge_idx);
                    let ref_old_vertex = esnap.map(|e| e.start_index).ok_or_else(|| {
                        crate::OperationsError::InvalidInput {
                            reason: "wire references unknown edge".into(),
                        }
                    })?;
                    let ref_vid = vertex_map[&ref_old_vertex];
                    let ref_point = topo.vertex(ref_vid)?.point();
                    let new_d =
                        new_normal.dot(Vec3::new(ref_point.x(), ref_point.y(), ref_point.z()));
                    FaceSurface::Plane {
                        normal: new_normal,
                        d: new_d,
                    }
                }
                FaceSurface::Nurbs(s) => {
                    let new_cps: Vec<Vec<_>> = s
                        .control_points()
                        .iter()
                        .map(|row| row.iter().map(|pt| matrix.mul_point(*pt)).collect())
                        .collect();
                    FaceSurface::Nurbs(NurbsSurface::new(
                        s.degree_u(),
                        s.degree_v(),
                        s.knots_u().to_vec(),
                        s.knots_v().to_vec(),
                        new_cps,
                        s.weights().to_vec(),
                    )?)
                }
                FaceSurface::Cylinder(cyl) => {
                    let new_origin = matrix.mul_point(cyl.origin());
                    let new_axis = transform_dir(cyl.axis())?;
                    FaceSurface::Cylinder(brepkit_math::surfaces::CylindricalSurface::new(
                        new_origin,
                        new_axis,
                        cyl.radius(),
                    )?)
                }
                FaceSurface::Cone(cone) => {
                    let new_apex = matrix.mul_point(cone.apex());
                    let new_axis = transform_dir(cone.axis())?;
                    FaceSurface::Cone(brepkit_math::surfaces::ConicalSurface::new(
                        new_apex,
                        new_axis,
                        cone.half_angle(),
                    )?)
                }
                FaceSurface::Sphere(sph) => {
                    let new_center = matrix.mul_point(sph.center());
                    FaceSurface::Sphere(brepkit_math::surfaces::SphericalSurface::new(
                        new_center,
                        sph.radius(),
                    )?)
                }
                FaceSurface::Torus(tor) => {
                    let new_center = matrix.mul_point(tor.center());
                    FaceSurface::Torus(brepkit_math::surfaces::ToroidalSurface::new(
                        new_center,
                        tor.major_radius(),
                        tor.minor_radius(),
                    )?)
                }
            };

            let new_face = if fsnap.reversed {
                Face::new_reversed(new_outer, new_inner, new_surface)
            } else {
                Face::new(new_outer, new_inner, new_surface)
            };
            let new_fid = topo.faces.alloc(new_face);
            new_face_ids.push(new_fid);
        }
        let new_shell = Shell::new(new_face_ids).map_err(crate::OperationsError::Topology)?;
        new_shell_ids.push(topo.shells.alloc(new_shell));
    }

    let new_outer = new_shell_ids[0];
    let new_inner: Vec<_> = new_shell_ids[1..].to_vec();

    Ok(topo.solids.alloc(Solid::new(new_outer, new_inner)))
}

/// Create a deep copy of a wire and all its sub-entities.
///
/// Returns a new `WireId` for the copy. The original wire is not modified.
/// All vertices and edges are duplicated.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn copy_wire(topo: &mut Topology, wire_id: WireId) -> Result<WireId, crate::OperationsError> {
    // ── Read phase ─────────────────────────────────────────────────
    let wire = topo.wire(wire_id)?;
    let closed = wire.is_closed();

    let mut vertex_snaps: Vec<VertexSnap> = Vec::new();
    let mut edge_snaps: Vec<EdgeSnap> = Vec::new();
    let mut edge_refs: Vec<(usize, bool)> = Vec::new();

    let mut seen_vertices = std::collections::HashSet::new();
    let mut seen_edges = std::collections::HashSet::new();

    for oe in wire.edges() {
        let edge_idx = oe.edge().index();
        edge_refs.push((edge_idx, oe.is_forward()));

        if !seen_edges.insert(edge_idx) {
            continue;
        }
        let edge = topo.edge(oe.edge())?;
        let start_idx = edge.start().index();
        let end_idx = edge.end().index();

        for &vid_idx in &[start_idx, end_idx] {
            if seen_vertices.insert(vid_idx) {
                let vid = if vid_idx == start_idx {
                    edge.start()
                } else {
                    edge.end()
                };
                let v = topo.vertex(vid)?;
                vertex_snaps.push(VertexSnap {
                    old_index: vid_idx,
                    point: v.point(),
                    tol: v.tolerance(),
                });
            }
        }

        edge_snaps.push(EdgeSnap {
            old_index: edge_idx,
            start_index: start_idx,
            end_index: end_idx,
            curve: edge.curve().clone(),
        });
    }

    // ── Write phase ────────────────────────────────────────────────
    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_vid = topo.vertices.alloc(Vertex::new(vsnap.point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    let mut edge_map: HashMap<usize, brepkit_topology::edge::EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let copied_edge = topo
            .edges
            .alloc(Edge::new(new_start, new_end, esnap.curve.clone()));
        edge_map.insert(esnap.old_index, copied_edge);
    }

    let new_edges: Vec<OrientedEdge> = edge_refs
        .iter()
        .map(|&(edge_idx, fwd)| OrientedEdge::new(edge_map[&edge_idx], fwd))
        .collect();
    let new_wire = Wire::new(new_edges, closed).map_err(crate::OperationsError::Topology)?;
    Ok(topo.wires.alloc(new_wire))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    #[test]
    fn copy_creates_new_solid() {
        let mut topo = Topology::new();
        let orig = make_unit_cube_manifold(&mut topo);
        let copy = copy_solid(&mut topo, orig).unwrap();
        assert_ne!(orig.index(), copy.index());
    }

    #[test]
    fn copy_preserves_face_count() {
        let mut topo = Topology::new();
        let orig = make_unit_cube_manifold(&mut topo);
        let copy = copy_solid(&mut topo, orig).unwrap();

        let orig_faces = topo
            .shell(topo.solid(orig).unwrap().outer_shell())
            .unwrap()
            .faces()
            .len();
        let copy_faces = topo
            .shell(topo.solid(copy).unwrap().outer_shell())
            .unwrap()
            .faces()
            .len();

        assert_eq!(orig_faces, copy_faces);
    }

    #[test]
    fn copy_preserves_volume() {
        let mut topo = Topology::new();
        let orig = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let copy = copy_solid(&mut topo, orig).unwrap();

        let vol_orig = crate::measure::solid_volume(&topo, orig, 0.1).unwrap();
        let vol_copy = crate::measure::solid_volume(&topo, copy, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol_orig, vol_copy),
            "copy should preserve volume: {vol_orig} vs {vol_copy}"
        );
    }

    #[test]
    fn copy_is_independent() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let orig = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let copy = copy_solid(&mut topo, orig).unwrap();

        // Transform the copy; original should be unchanged.
        crate::transform::transform_solid(&mut topo, copy, &Mat4::translation(10.0, 0.0, 0.0))
            .unwrap();

        let bbox_orig = crate::measure::solid_bounding_box(&topo, orig).unwrap();
        let bbox_copy = crate::measure::solid_bounding_box(&topo, copy).unwrap();

        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(bbox_orig.min.x(), 0.0),
            "original should be unchanged, min_x = {}",
            bbox_orig.min.x()
        );
        assert!(
            tol.approx_eq(bbox_copy.min.x(), 10.0),
            "copy should be shifted, min_x = {}",
            bbox_copy.min.x()
        );
    }
}
