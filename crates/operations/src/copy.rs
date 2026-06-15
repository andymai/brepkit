//! Deep copy of topological entities.
//!
//! Creates independent copies of solids and all their sub-entities
//! (shells, faces, wires, edges, vertices) in the arena.

use std::collections::HashMap;

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

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
    tolerance: Option<f64>,
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
                        tolerance: edge.tolerance(),
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

    topo.reserve(
        vertex_snaps.len(),
        edge_snaps.len(),
        wire_snaps.len(),
        shell_snaps
            .iter()
            .map(|s| s.faces.len())
            .fold(0usize, usize::saturating_add),
        shell_snaps.len(),
        1,
    );

    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_vid = topo.add_vertex(Vertex::new(vsnap.point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    let mut edge_map: HashMap<usize, brepkit_topology::edge::EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let copied_edge = topo.add_edge(Edge::with_tolerance(
            new_start,
            new_end,
            esnap.curve.clone(),
            esnap.tolerance,
        ));
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
        wire_map.insert(wsnap.old_index, topo.add_wire(new_wire));
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
            let new_fid = topo.add_face(new_face);
            new_face_ids.push(new_fid);
        }
        let new_shell = Shell::new(new_face_ids).map_err(crate::OperationsError::Topology)?;
        new_shell_ids.push(topo.add_shell(new_shell));
    }

    let new_outer = new_shell_ids[0];
    let new_inner: Vec<_> = new_shell_ids[1..].to_vec();

    Ok(topo.add_solid(Solid::new(new_outer, new_inner)))
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

    // Transform a plane normal via inverse transpose.
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

    // Read phase mirrors copy_solid.
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
                        tolerance: edge.tolerance(),
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

    topo.reserve(
        vertex_snaps.len(),
        edge_snaps.len(),
        wire_snaps.len(),
        shell_snaps
            .iter()
            .map(|s| s.faces.len())
            .fold(0usize, usize::saturating_add),
        shell_snaps.len(),
        1,
    );

    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_point = matrix.mul_point(vsnap.point);
        let new_vid = topo.add_vertex(Vertex::new(new_point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

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
        let copied_edge = topo.add_edge(Edge::with_tolerance(
            new_start,
            new_end,
            new_curve,
            esnap.tolerance,
        ));
        edge_map.insert(esnap.old_index, copied_edge);
    }

    // Wires carry no geometry to transform.
    let mut wire_map: HashMap<usize, WireId> = HashMap::new();
    for wsnap in &wire_snaps {
        let new_edges: Vec<OrientedEdge> = wsnap
            .edges
            .iter()
            .map(|&(edge_idx, fwd)| OrientedEdge::new(edge_map[&edge_idx], fwd))
            .collect();
        let new_wire =
            Wire::new(new_edges, wsnap.closed).map_err(crate::OperationsError::Topology)?;
        wire_map.insert(wsnap.old_index, topo.add_wire(new_wire));
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
            let new_fid = topo.add_face(new_face);
            new_face_ids.push(new_fid);
        }
        let new_shell = Shell::new(new_face_ids).map_err(crate::OperationsError::Topology)?;
        new_shell_ids.push(topo.add_shell(new_shell));
    }

    let new_outer = new_shell_ids[0];
    let new_inner: Vec<_> = new_shell_ids[1..].to_vec();

    Ok(topo.add_solid(Solid::new(new_outer, new_inner)))
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
            tolerance: edge.tolerance(),
        });
    }

    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_vid = topo.add_vertex(Vertex::new(vsnap.point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    let mut edge_map: HashMap<usize, brepkit_topology::edge::EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let copied_edge = topo.add_edge(Edge::with_tolerance(
            new_start,
            new_end,
            esnap.curve.clone(),
            esnap.tolerance,
        ));
        edge_map.insert(esnap.old_index, copied_edge);
    }

    let new_edges: Vec<OrientedEdge> = edge_refs
        .iter()
        .map(|&(edge_idx, fwd)| OrientedEdge::new(edge_map[&edge_idx], fwd))
        .collect();
    let new_wire = Wire::new(new_edges, closed).map_err(crate::OperationsError::Topology)?;
    Ok(topo.add_wire(new_wire))
}

/// Create a deep copy of a single face and all its sub-entities.
///
/// Returns a new [`FaceId`] for the copy. The original face and any shape that
/// shares its sub-entities are left untouched, so the copy can be translated to
/// form a pocket or boss profile without corrupting the donor solid. The
/// surface carrier and orientation (`reversed`) flag are cloned verbatim — no
/// geometry is recomputed.
///
/// # Errors
///
/// Returns an error if any topology lookup fails.
pub fn copy_face(topo: &mut Topology, face_id: FaceId) -> Result<FaceId, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let outer_wire_index = face.outer_wire().index();
    let inner_wire_indices: Vec<usize> = face.inner_wires().iter().map(|w| w.index()).collect();

    let mut vertex_snaps: Vec<VertexSnap> = Vec::new();
    let mut edge_snaps: Vec<EdgeSnap> = Vec::new();
    let mut wire_snaps: Vec<WireSnap> = Vec::new();

    let mut seen_vertices = std::collections::HashSet::new();
    let mut seen_edges = std::collections::HashSet::new();
    let mut seen_wires = std::collections::HashSet::new();

    for wire_id_val in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
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
                tolerance: edge.tolerance(),
            });
        }

        wire_snaps.push(WireSnap {
            old_index: wire_id_val.index(),
            edges: edge_refs,
            closed: wire.is_closed(),
        });
    }

    topo.reserve(
        vertex_snaps.len(),
        edge_snaps.len(),
        wire_snaps.len(),
        1,
        0,
        0,
    );

    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_vid = topo.add_vertex(Vertex::new(vsnap.point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    let mut edge_map: HashMap<usize, brepkit_topology::edge::EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let copied_edge = topo.add_edge(Edge::with_tolerance(
            new_start,
            new_end,
            esnap.curve.clone(),
            esnap.tolerance,
        ));
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
        wire_map.insert(wsnap.old_index, topo.add_wire(new_wire));
    }

    let new_outer = wire_map[&outer_wire_index];
    let new_inner: Vec<WireId> = inner_wire_indices.iter().map(|idx| wire_map[idx]).collect();
    let new_face = if reversed {
        Face::new_reversed(new_outer, new_inner, surface)
    } else {
        Face::new(new_outer, new_inner, surface)
    };
    Ok(topo.add_face(new_face))
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

    #[test]
    fn copy_wire_creates_new_wire() {
        use brepkit_math::vec::Point3;
        use brepkit_topology::builder::make_polygon_wire;

        let mut topo = Topology::new();
        let orig = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let copy = copy_wire(&mut topo, orig).unwrap();

        assert_ne!(orig.index(), copy.index());
    }

    #[test]
    fn copy_wire_preserves_edge_count() {
        use brepkit_math::vec::Point3;
        use brepkit_topology::builder::make_polygon_wire;

        let mut topo = Topology::new();
        let orig = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let copy = copy_wire(&mut topo, orig).unwrap();

        let orig_edges = topo.wire(orig).unwrap().edges().len();
        let copy_edges = topo.wire(copy).unwrap().edges().len();
        assert_eq!(orig_edges, copy_edges);
    }

    #[test]
    fn copy_wire_is_independent() {
        use brepkit_math::mat::Mat4;
        use brepkit_math::vec::Point3;
        use brepkit_topology::builder::make_polygon_wire;

        let mut topo = Topology::new();
        let orig = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let copy = copy_wire(&mut topo, orig).unwrap();

        // Transform the copy; original should be unchanged.
        crate::transform::transform_wire(&mut topo, copy, &Mat4::translation(10.0, 0.0, 0.0))
            .unwrap();

        // Check that original vertex positions are unchanged.
        let tol = Tolerance::new();
        let orig_wire = topo.wire(orig).unwrap();
        let first_edge = orig_wire.edges().first().unwrap();
        let start = topo.edge(first_edge.edge()).unwrap().start();
        let pos = topo.vertex(start).unwrap().point();
        assert!(
            tol.approx_eq(pos.x(), 0.0),
            "original wire should be unchanged, x = {}",
            pos.x()
        );
    }

    #[test]
    fn copy_wire_with_circle_edge() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        // Create a closed circular wire (single circle edge, start == end).
        let v = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
        let wid = topo.add_wire(wire);

        let copy_wid = copy_wire(&mut topo, wid).unwrap();
        assert_ne!(wid.index(), copy_wid.index());

        // Verify the copied wire has a circle edge.
        let copy_wire = topo.wire(copy_wid).unwrap();
        let copy_edge = topo.edge(copy_wire.edges()[0].edge()).unwrap();
        assert!(
            matches!(copy_edge.curve(), EdgeCurve::Circle(_)),
            "copied edge should be a Circle"
        );
    }

    fn make_plane_quad(topo: &mut Topology) -> brepkit_topology::face::FaceId {
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::builder::make_polygon_wire;
        use brepkit_topology::face::{Face, FaceSurface};

        let outer = make_polygon_wire(
            topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
                Point3::new(4.0, 4.0, 0.0),
                Point3::new(0.0, 4.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        topo.add_face(Face::new(outer, Vec::new(), surface))
    }

    fn distinct_vertex_count(topo: &Topology, face_id: brepkit_topology::face::FaceId) -> usize {
        let face = topo.face(face_id).unwrap();
        let mut seen = std::collections::HashSet::new();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                seen.insert(edge.start().index());
                seen.insert(edge.end().index());
            }
        }
        seen.len()
    }

    fn total_edge_count(topo: &Topology, face_id: brepkit_topology::face::FaceId) -> usize {
        let face = topo.face(face_id).unwrap();
        let mut seen = std::collections::HashSet::new();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                seen.insert(oe.edge().index());
            }
        }
        seen.len()
    }

    fn loop_count(topo: &Topology, face_id: brepkit_topology::face::FaceId) -> usize {
        let face = topo.face(face_id).unwrap();
        1 + face.inner_wires().len()
    }

    #[test]
    fn copy_face_creates_new_face() {
        let mut topo = Topology::new();
        let orig = make_plane_quad(&mut topo);
        let copy = copy_face(&mut topo, orig).unwrap();
        assert_ne!(orig.index(), copy.index());
    }

    #[test]
    fn copy_face_preserves_topology_counts() {
        use brepkit_topology::explorer::solid_faces;

        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let box_face = *solid_faces(&topo, solid).unwrap().first().unwrap();
        let copy = copy_face(&mut topo, box_face).unwrap();

        assert_eq!(loop_count(&topo, box_face), loop_count(&topo, copy));
        assert_eq!(
            total_edge_count(&topo, box_face),
            total_edge_count(&topo, copy)
        );
        assert_eq!(
            distinct_vertex_count(&topo, box_face),
            distinct_vertex_count(&topo, copy)
        );

        // Face with one inner loop (hole).
        let holed = make_holed_face(&mut topo);
        let holed_copy = copy_face(&mut topo, holed).unwrap();
        assert_eq!(loop_count(&topo, holed), 2);
        assert_eq!(loop_count(&topo, holed_copy), 2);
        assert_eq!(
            total_edge_count(&topo, holed),
            total_edge_count(&topo, holed_copy)
        );
        assert_eq!(
            distinct_vertex_count(&topo, holed),
            distinct_vertex_count(&topo, holed_copy)
        );
    }

    fn make_holed_face(topo: &mut Topology) -> brepkit_topology::face::FaceId {
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::builder::make_polygon_wire;
        use brepkit_topology::face::{Face, FaceSurface};

        let outer = make_polygon_wire(
            topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(0.0, 10.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let inner = make_polygon_wire(
            topo,
            &[
                Point3::new(3.0, 3.0, 0.0),
                Point3::new(7.0, 3.0, 0.0),
                Point3::new(7.0, 7.0, 0.0),
                Point3::new(3.0, 7.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        topo.add_face(Face::new(outer, vec![inner], surface))
    }

    #[test]
    fn copy_face_is_independent() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let orig = make_plane_quad(&mut topo);

        let orig_first_vertex = {
            let face = topo.face(orig).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            topo.edge(wire.edges()[0].edge()).unwrap().start()
        };
        let orig_start_x = topo.vertex(orig_first_vertex).unwrap().point().x();

        let copy = copy_face(&mut topo, orig).unwrap();
        crate::transform::transform_face(&mut topo, copy, &Mat4::translation(10.0, 0.0, 0.0))
            .unwrap();

        let tol = Tolerance::new();
        let orig_x_after = topo.vertex(orig_first_vertex).unwrap().point().x();
        assert!(
            tol.approx_eq(orig_x_after, orig_start_x),
            "original face vertex should be unchanged, x = {orig_x_after}"
        );

        let copy_first_vertex = {
            let face = topo.face(copy).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            topo.edge(wire.edges()[0].edge()).unwrap().start()
        };
        let copy_x = topo.vertex(copy_first_vertex).unwrap().point().x();
        assert!(
            tol.approx_eq(copy_x, orig_start_x + 10.0),
            "copy face vertex should be shifted by 10, x = {copy_x}"
        );
    }

    #[test]
    fn copy_face_preserves_orientation() {
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::builder::make_polygon_wire;
        use brepkit_topology::face::{Face, FaceSurface};

        let mut topo = Topology::new();
        let outer = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            1e-7,
        )
        .unwrap();
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let orig = topo.add_face(Face::new_reversed(outer, Vec::new(), surface));

        let copy = copy_face(&mut topo, orig).unwrap();
        assert!(
            topo.face(copy).unwrap().is_reversed(),
            "copied face should preserve the reversed orientation flag"
        );
    }

    #[test]
    fn copy_face_with_circle_edge() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let v = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let orig = topo.add_face(Face::new(wid, Vec::new(), surface));

        let copy = copy_face(&mut topo, orig).unwrap();
        assert_eq!(distinct_vertex_count(&topo, copy), 1);

        let copy_face_data = topo.face(copy).unwrap();
        let copy_wire = topo.wire(copy_face_data.outer_wire()).unwrap();
        let copy_edge = topo.edge(copy_wire.edges()[0].edge()).unwrap();
        assert!(
            matches!(copy_edge.curve(), EdgeCurve::Circle(_)),
            "copied edge should be a Circle"
        );
    }
}
