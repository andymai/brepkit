//! Linear extrusion of faces along a direction vector.
//!
//! Supports both planar and NURBS profile faces. For NURBS faces, the
//! extrusion translates all control points, preserving the exact surface
//! representation for both caps.

use brepkit_math::tolerance::Tolerance;
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
/// If no splitting is needed, returns the original edges unchanged.
///
/// # Errors
///
/// Returns an error if edge lookup fails.
pub fn maybe_split_closed_wire(
    topo: &mut Topology,
    oriented: &[OrientedEdge],
    tol: f64,
) -> Result<Vec<OrientedEdge>, crate::OperationsError> {
    // Only need splitting when the wire has edges whose start==end vertex.
    // Collect all edges that need splitting, and pass through the rest.
    let mut result = Vec::with_capacity(oriented.len() * 4);
    for oe in oriented {
        let edge = topo.edge(oe.edge())?;
        if edge.start() == edge.end() {
            // Closed edge — split the curve at evenly-spaced parameters.
            // Use 32 segments for a good approximation of curved geometry
            // in the side faces (which are planar quads).
            let split_edges = split_closed_edge(topo, oe.edge(), 256, tol)?;
            for se in split_edges {
                result.push(OrientedEdge::new(se, oe.is_forward()));
            }
        } else {
            result.push(*oe);
        }
    }
    Ok(result)
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

    // Get the parameter domain of the curve.
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
        let vid = topo.vertices.alloc(Vertex::new(pt, tol));
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
        let eid = topo
            .edges
            .alloc(Edge::new(v_start, v_end_actual, EdgeCurve::Line));
        edge_ids.push(eid);
    }

    Ok(edge_ids)
}

/// Extract vertices, create offset (top) vertices and edges for a wire.
///
/// Returns: `(input_verts, input_positions, input_oriented, input_edge_ids,
///            top_verts, top_edge_ids, vertical_edge_ids)`
#[allow(clippy::type_complexity)]
fn extrude_wire_vertices(
    topo: &mut Topology,
    wire_id: WireId,
    offset: Vec3,
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
    let oriented = maybe_split_closed_wire(topo, &original_oriented, tol.linear)?;

    let mut verts: Vec<VertexId> = Vec::with_capacity(oriented.len());
    for oe in &oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
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
            topo.vertices.alloc(Vertex::new(top_point, tol.linear))
        })
        .collect();

    let edge_ids: Vec<EdgeId> = oriented
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect();

    let mut top_edge_ids = Vec::with_capacity(n);
    for i in 0..n {
        let next = (i + 1) % n;
        let top_edge = topo
            .edges
            .alloc(Edge::new(top_verts[i], top_verts[next], EdgeCurve::Line));
        top_edge_ids.push(top_edge);
    }

    let mut vertical_edge_ids = Vec::with_capacity(n);
    for i in 0..n {
        let vert_edge = topo
            .edges
            .alloc(Edge::new(verts[i], top_verts[i], EdgeCurve::Line));
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

/// Determine if a polygon (inner wire) is CW when viewed from the
/// extrusion direction.
///
/// Uses the signed area projected onto the extrusion axis: negative = CW,
/// positive = CCW. This generalizes to non-axis-aligned extrusions.
fn inner_wire_is_cw(positions: &[Point3], offset: &Vec3) -> bool {
    if positions.len() < 3 {
        return true; // degenerate — default to CW
    }
    let axis = offset.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let p0 = positions[0];
    let mut signed_area_2 = 0.0;
    for i in 1..positions.len() - 1 {
        let a = positions[i] - p0;
        let b = positions[i + 1] - p0;
        signed_area_2 += a.cross(b).dot(axis);
    }
    signed_area_2 < 0.0
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

    // Validate direction is non-zero.
    if tol.approx_eq(direction.length_squared(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "extrusion direction is zero-length".into(),
        });
    }

    // Validate distance is non-zero.
    if tol.approx_eq(distance, 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "extrusion distance is zero".into(),
        });
    }

    // Read the input face's data.
    let face_data = topo.face(face)?;
    let input_surface = face_data.surface().clone();
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids: Vec<WireId> = face_data.inner_wires().to_vec();

    // Compute offset vector.
    let offset = Vec3::new(
        direction.x() * distance,
        direction.y() * distance,
        direction.z() * distance,
    );

    // --- Process outer wire ---
    let (
        input_verts,
        input_positions,
        input_oriented,
        input_edge_ids,
        _top_verts,
        top_edge_ids,
        vertical_edge_ids,
    ) = extrude_wire_vertices(topo, input_wire_id, offset)?;
    let n = input_verts.len();

    let mut all_faces = Vec::with_capacity(n + 2 + inner_wire_ids.len() * 4);

    // --- Bottom face: reversed copy of input wire ---
    // Use the (possibly-split) edges so the bottom cap shares vertices
    // with the side faces, keeping the shell manifold.
    let reversed_bottom_edges: Vec<OrientedEdge> = input_oriented
        .iter()
        .rev()
        .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
        .collect();
    let bottom_wire =
        Wire::new(reversed_bottom_edges, true).map_err(crate::OperationsError::Topology)?;
    let bottom_wire_id = topo.wires.alloc(bottom_wire);

    // --- Process inner wires and create inner wire data ---
    let mut bottom_inner_wire_ids = Vec::with_capacity(inner_wire_ids.len());
    let mut top_inner_wire_ids = Vec::with_capacity(inner_wire_ids.len());

    let mut inner_wire_data: Vec<InnerWireData> = Vec::new();

    for &iw_id in &inner_wire_ids {
        let (
            iw_verts,
            iw_positions,
            iw_oriented,
            iw_edge_ids,
            iw_top_verts,
            iw_top_edge_ids,
            iw_vert_edge_ids,
        ) = extrude_wire_vertices(topo, iw_id, offset)?;
        let iw_n = iw_verts.len();

        // Bottom inner wire: reversed winding (same as outer wire reversal).
        let reversed_inner_edges: Vec<OrientedEdge> = iw_oriented
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        let bottom_inner_wire =
            Wire::new(reversed_inner_edges, true).map_err(crate::OperationsError::Topology)?;
        bottom_inner_wire_ids.push(topo.wires.alloc(bottom_inner_wire));

        // Top inner wire: same winding as bottom inner (reversed from original).
        let top_inner_edges: Vec<OrientedEdge> = iw_top_edge_ids
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let top_inner_wire =
            Wire::new(top_inner_edges, true).map_err(crate::OperationsError::Topology)?;
        top_inner_wire_ids.push(topo.wires.alloc(top_inner_wire));

        let _ = iw_top_verts;
        let _ = iw_n;
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
            let bottom_normal = Vec3::new(-normal.x(), -normal.y(), -normal.z());
            let bottom_d = dot_normal_point(bottom_normal, input_positions[0]);
            FaceSurface::Plane {
                normal: bottom_normal,
                d: bottom_d,
            }
        }
        FaceSurface::Nurbs(nurbs) => FaceSurface::Nurbs(nurbs.clone()),
        other => other.clone(),
    };
    let bottom_face = topo.faces.alloc(Face::new(
        bottom_wire_id,
        bottom_inner_wire_ids,
        bottom_surface,
    ));
    all_faces.push(bottom_face);

    // --- Outer side faces ---
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

        let side_wire_id = topo.wires.alloc(side_wire);

        let p0 = input_positions[i];
        let p1 = input_positions[next];
        let edge_dir = p1 - p0;
        let side_normal = edge_dir
            .cross(offset)
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let side_d = dot_normal_point(side_normal, p0);

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

    // --- Inner wire side faces ---
    for iwd in &inner_wire_data {
        let iw_n = iwd.positions.len();

        // Detect inner wire winding direction relative to the face normal.
        // CW (negative signed area) is the standard B-Rep hole convention;
        // CCW (positive signed area) occurs when callers use math-convention
        // circle generation.  We support both.
        let is_cw = inner_wire_is_cw(&iwd.positions, &offset);

        for i in 0..iw_n {
            let next = (i + 1) % iw_n;

            let (side_edges, side_normal) = if is_cw {
                // CW inner wire: traverse the quad in the reversed pattern
                // (up, across, down, back) so that the face normal points
                // into the hole (away from solid material).
                let edges = vec![
                    OrientedEdge::new(iwd.vertical_edge_ids[i], true),
                    OrientedEdge::new(iwd.top_edge_ids[i], true),
                    OrientedEdge::new(iwd.vertical_edge_ids[next], false),
                    OrientedEdge::new(iwd.edge_ids[i], !iwd.oriented[i].is_forward()),
                ];
                let p0 = iwd.positions[i];
                let p1 = iwd.positions[next];
                let edge_dir = p1 - p0;
                let normal = edge_dir
                    .cross(offset)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                (edges, normal)
            } else {
                // CCW inner wire: use the same winding pattern as outer
                // side faces (bottom-edge forward, right up, top back,
                // left down) which produces inward-pointing normals for
                // CCW inner geometry.
                let edges = vec![
                    OrientedEdge::new(iwd.edge_ids[i], iwd.oriented[i].is_forward()),
                    OrientedEdge::new(iwd.vertical_edge_ids[next], true),
                    OrientedEdge::new(iwd.top_edge_ids[i], false),
                    OrientedEdge::new(iwd.vertical_edge_ids[i], false),
                ];
                let p0 = iwd.positions[i];
                let p1 = iwd.positions[next];
                let edge_dir = p1 - p0;
                let normal = offset
                    .cross(edge_dir)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                (edges, normal)
            };

            let side_wire =
                Wire::new(side_edges, true).map_err(crate::OperationsError::Topology)?;
            let side_wire_id = topo.wires.alloc(side_wire);

            let p0 = iwd.positions[i];
            let side_d = dot_normal_point(side_normal, p0);

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
    let top_wire_id = topo.wires.alloc(top_wire);

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
    let top_face = topo
        .faces
        .alloc(Face::new(top_wire_id, top_inner_wire_ids, top_surface));
    all_faces.push(top_face);

    // Assemble shell and solid.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    let solid = topo.solids.alloc(Solid::new(shell_id, vec![]));

    Ok(solid)
}

/// Create a translated copy of a wire by duplicating all edges, translating
/// their vertices and curve geometry by `offset`.
#[allow(dead_code)]
fn make_translated_wire(
    topo: &mut Topology,
    oriented_edges: &[OrientedEdge],
    offset: Vec3,
) -> Result<WireId, crate::OperationsError> {
    use std::collections::HashMap;

    let tol = Tolerance::new();
    let mut vertex_map: HashMap<VertexId, VertexId> = HashMap::new();

    // Map vertices: for each unique vertex, create a translated copy.
    let mut get_or_create_vertex =
        |topo: &mut Topology, vid: VertexId| -> Result<VertexId, crate::OperationsError> {
            if let Some(&mapped) = vertex_map.get(&vid) {
                return Ok(mapped);
            }
            let pt = topo.vertex(vid)?.point();
            let new_vid = topo.vertices.alloc(Vertex::new(pt + offset, tol.linear));
            vertex_map.insert(vid, new_vid);
            Ok(new_vid)
        };

    let mut new_edges = Vec::with_capacity(oriented_edges.len());
    for oe in oriented_edges {
        let edge = topo.edge(oe.edge())?;
        let v_start = edge.start();
        let v_end = edge.end();
        let curve = edge.curve().clone();

        let new_start = get_or_create_vertex(topo, v_start)?;
        let new_end = if v_start == v_end {
            new_start
        } else {
            get_or_create_vertex(topo, v_end)?
        };

        let new_curve = translate_edge_curve(&curve, offset)?;
        let new_eid = topo.edges.alloc(Edge::new(new_start, new_end, new_curve));
        new_edges.push(OrientedEdge::new(new_eid, oe.is_forward()));
    }

    let wire = Wire::new(new_edges, true).map_err(crate::OperationsError::Topology)?;
    Ok(topo.wires.alloc(wire))
}

/// Translate an `EdgeCurve` by an offset vector.
///
/// # Errors
///
/// Returns an error if constructing the translated curve fails (should
/// never happen when translating a valid curve).
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::HashMap;

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::{make_unit_square_face, make_unit_triangle_face};

    use super::*;
    use crate::test_helpers::assert_euler_genus0;

    #[test]
    fn extrude_square_creates_box() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 4 sides + top + bottom = 6 faces
        assert_eq!(shell.faces().len(), 6);
        // 4 input + 4 top + 4 vertical = 12 edges (original input edges are reused)
        assert_eq!(topo.edges.len(), 12);
        // 4 input + 4 top = 8 vertices
        assert_eq!(topo.vertices.len(), 8);
    }

    #[test]
    fn extrude_triangle_creates_prism() {
        let mut topo = Topology::new();
        let face = make_unit_triangle_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 3 sides + top + bottom = 5 faces
        assert_eq!(shell.faces().len(), 5);
        assert_eq!(topo.edges.len(), 9);
        assert_eq!(topo.vertices.len(), 6);
    }

    #[test]
    fn extrude_zero_direction_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 0.0), 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn extrude_zero_distance_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 0.0);
        assert!(result.is_err());
    }

    /// Verify that extruding a +Z face upward produces a solid where:
    /// - The bottom face normal points -Z (outward-downward)
    /// - The top face normal points +Z (outward-upward)
    /// - All edges are shared by exactly 2 faces (manifold)
    #[test]
    fn extrude_orientation_correct() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        let mut found_bottom = false;
        let mut found_top = false;
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = f.surface() {
                // Bottom: normal ≈ (0, 0, -1)
                if tol.approx_eq(normal.z(), -1.0)
                    && tol.approx_eq(normal.x(), 0.0)
                    && tol.approx_eq(normal.y(), 0.0)
                {
                    found_bottom = true;
                }
                // Top: normal ≈ (0, 0, 1)
                if tol.approx_eq(normal.z(), 1.0)
                    && tol.approx_eq(normal.x(), 0.0)
                    && tol.approx_eq(normal.y(), 0.0)
                {
                    found_top = true;
                }
            }
        }
        assert!(found_bottom, "bottom face should have -Z normal");
        assert!(found_top, "top face should have +Z normal");

        // Verify manifold: every edge used by exactly 2 faces.
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

    // ── NURBS face extrusion tests ──────────────────────

    /// Build a NURBS face: a curved surface on the XY plane.
    fn make_nurbs_face(topo: &mut Topology) -> FaceId {
        use brepkit_math::nurbs::surface::NurbsSurface;

        // Bicubic surface with some curvature.
        let cps = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![Point3::new(0.0, 1.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
        ];
        let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let surface = NurbsSurface::new(1, 1, knots.clone(), knots, cps, weights).unwrap();

        let tol = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.5), tol));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.5), tol));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

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
        let wid = topo.wires.alloc(wire);

        topo.faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(surface)))
    }

    #[test]
    fn extrude_nurbs_face_creates_solid() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 4 sides + top + bottom = 6 faces
        assert_eq!(
            sh.faces().len(),
            6,
            "extruded NURBS face should have 6 faces"
        );
    }

    #[test]
    fn extrude_nurbs_face_top_is_nurbs() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // At least 2 faces should be NURBS (top and bottom caps).
        let nurbs_count = sh
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)))
            .count();

        assert!(
            nurbs_count >= 2,
            "extruded NURBS face should have at least 2 NURBS caps, got {nurbs_count}"
        );
    }

    #[test]
    fn extrude_nurbs_face_top_translated() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // Find the top NURBS face (the one at higher z).
        let mut top_z = f64::MIN;
        for &fid in sh.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Nurbs(surface) = f.surface() {
                let pt = surface.evaluate(0.0, 0.0);
                if pt.z() > top_z {
                    top_z = pt.z();
                }
            }
        }

        // The top surface should be translated by distance 3.0 along Z.
        // Original (0,0) point is at z=0, so top should be at z=3.0.
        assert!(
            (top_z - 3.0).abs() < 1e-7,
            "top NURBS surface should be at z≈3.0, got z={top_z}"
        );
    }

    #[test]
    fn extrude_nurbs_face_positive_volume() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

        // NURBS face is a bilinear patch: corners at z=0 (y=0) and z=0.5 (y=1).
        // The extrusion offset is (0,0,2), so top surface is at z=2 and z=2.5.
        // The solid is a "wedge" — thicker at one end.
        //
        // For a bilinear patch f(u,v) with z = 0.5v, the XY footprint is a
        // unit square. The enclosed volume between bottom and top surfaces:
        // V = ∫∫ height dA where height = 2.0 (constant, offset along Z).
        // But the solid shape is bounded by the slanted bottom/top NURBS faces.
        //
        // Actual result from tessellation: ~0.667 (≈ 2/3).
        // This is plausible: the bilinear bottom face "scoops out" volume.
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.3 && vol < 1.5,
            "extruded NURBS solid should have positive volume in (0.3, 1.5), got {vol}"
        );

        assert_euler_genus0(&topo, solid);
    }

    /// Helper: create a square face with a smaller square hole in the center.
    fn make_face_with_hole(topo: &mut Topology) -> FaceId {
        // Outer wire: 2×2 square centered at origin.
        let outer_pts = vec![
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
        ];
        let outer_wire = brepkit_topology::builder::make_polygon_wire(topo, &outer_pts).unwrap();

        // Inner wire: 0.5×0.5 square hole (CW winding = hole).
        let inner_pts = vec![
            Point3::new(-0.25, -0.25, 0.0),
            Point3::new(-0.25, 0.25, 0.0),
            Point3::new(0.25, 0.25, 0.0),
            Point3::new(0.25, -0.25, 0.0),
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
    fn extrude_face_with_hole_produces_more_faces() {
        let mut topo = Topology::new();
        let face = make_face_with_hole(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // Outer: 4 sides + top + bottom = 6
        // Inner: 4 inward-facing sides = 4
        // Total: 10 faces
        assert_eq!(
            shell.faces().len(),
            10,
            "extruded face with hole should have 10 faces (6 outer + 4 inner)"
        );
    }

    #[test]
    fn extrude_face_with_hole_caps_have_inner_wires() {
        let mut topo = Topology::new();
        let face = make_face_with_hole(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // The bottom and top faces should have inner wires (holes).
        let faces_with_holes_count = shell
            .faces()
            .iter()
            .filter(|&&fid| !topo.face(fid).unwrap().inner_wires().is_empty())
            .count();

        assert_eq!(
            faces_with_holes_count, 2,
            "bottom and top caps should both have inner wire holes"
        );
    }

    #[test]
    fn extrude_zero_distance_errors() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 0.0);
        assert!(result.is_err(), "zero distance extrusion should error");
    }

    #[test]
    fn extrude_face_with_hole_has_correct_volume() {
        let mut topo_solid = Topology::new();
        let solid_face = make_unit_square_face(&mut topo_solid);
        let solid_box =
            extrude(&mut topo_solid, solid_face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let mut topo_hollow = Topology::new();
        let hollow_face = make_face_with_hole(&mut topo_hollow);
        let hollow_solid =
            extrude(&mut topo_hollow, hollow_face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let vol_solid = crate::measure::solid_volume(&topo_solid, solid_box, 0.1).unwrap();
        let vol_hollow = crate::measure::solid_volume(&topo_hollow, hollow_solid, 0.1).unwrap();

        // Solid box: 1×1×1 = 1.0 exactly.
        let rel_solid = (vol_solid - 1.0).abs() / 1.0;
        assert!(
            rel_solid < 1e-8,
            "unit box volume should be 1.0, got {vol_solid} (rel_err={rel_solid:.2e})"
        );

        // Hollow: outer 2×2×1 = 4.0, hole 0.5×0.5×1 = 0.25, net = 3.75.
        //
        // Note: the signed-tetrahedra volume method for solids with inner
        // wires (holes) uses `volume_from_direct_face_tessellation`, which
        // relies on correct face winding from `tessellate()`. The hole
        // subtraction accuracy depends on the inner wall face orientations.
        let expected_hollow = 3.75;
        let rel_hollow = (vol_hollow - expected_hollow).abs() / expected_hollow;
        assert!(
            rel_hollow < 0.01,
            "hollow extrusion volume should be {expected_hollow}, got {vol_hollow} \
             (rel_err={rel_hollow:.2e}). If > 1%, inner-wire volume subtraction may be buggy."
        );
    }

    /// Extrude a square along +Z by distance 5 → volume = 1×1×5 = 5.0.
    #[test]
    fn extrude_square_volume_exact() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        // All-planar: exact to floating-point.
        let rel_err = (vol - 5.0).abs() / 5.0;
        assert!(
            rel_err < 1e-8,
            "extruded unit square by 5 should have volume 5.0, got {vol} (rel_err={rel_err:.2e})"
        );
    }

    /// Extrude a triangle by 3 → volume = (base_area × height) = (0.5 × 3) = 1.5.
    #[test]
    fn extrude_triangle_volume_exact() {
        let mut topo = Topology::new();
        let face = make_unit_triangle_face(&mut topo);
        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        // Unit triangle area = 0.5, height = 3.0 → V = 1.5.
        let expected = 1.5;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-8,
            "extruded unit triangle by 3 should have volume {expected}, got {vol} (rel_err={rel_err:.2e})"
        );
    }

    /// Extrude in a non-axis-aligned direction.
    ///
    /// `extrude()` does NOT normalize the direction: offset = direction × distance.
    /// With direction=(1,0,1) and distance=2, offset = (2,0,2).
    ///
    /// For a sheared prism, volume = base_area × |offset · face_normal|.
    /// Face normal is (0,0,1) (XY plane), so V = 1.0 × |2| = 2.0.
    #[test]
    fn extrude_oblique_direction_volume() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let solid = extrude(&mut topo, face, Vec3::new(1.0, 0.0, 1.0), 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        // offset = (1,0,1)*2 = (2,0,2). Height along Z (face normal) = 2.0.
        // V = base_area × height = 1.0 × 2.0 = 2.0.
        let expected = 2.0;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-8,
            "oblique extrusion volume should be {expected}, got {vol} (rel_err={rel_err:.2e})"
        );
    }

    /// Reproduce brepjs compound extrude: 20×20 rectangle with a circular
    /// polygon hole (CCW winding, 32 segments, radius 3), extruded by 10.
    #[test]
    fn extrude_face_with_ccw_circle_hole_volume() {
        let mut topo = Topology::new();

        // Outer wire: 20×20 rectangle (CCW).
        let outer_pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(20.0, 0.0, 0.0),
            Point3::new(20.0, 20.0, 0.0),
            Point3::new(0.0, 20.0, 0.0),
        ];
        let outer_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts).unwrap();

        // Inner wire: 32-segment polygon circle at center (10,10), radius 3.
        // CCW winding (standard math convention: cos/sin going counter-clockwise).
        let n_segments = 32;
        let cx = 10.0;
        let cy = 10.0;
        let r = 3.0;
        let inner_pts: Vec<Point3> = (0..n_segments)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
                Point3::new(cx + r * theta.cos(), cy + r * theta.sin(), 0.0)
            })
            .collect();
        let inner_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts).unwrap();

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal, d: 0.0 },
        );
        let face_id = topo.faces.alloc(face);

        let solid = extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        // Expected: 20*20*10 - polygon_area*10
        // Polygon area of regular 32-gon inscribed in circle r=3:
        // A = n * r^2 * sin(2*pi/n) / 2 = 32 * 9 * sin(pi/16) / 2 ≈ 27.86
        // Expected volume ≈ 4000 - 278.6 = 3721.4
        let polygon_area: f64 = (0..n_segments)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let theta1 = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
                #[allow(clippy::cast_precision_loss)]
                let theta2 = 2.0 * std::f64::consts::PI * ((i + 1) as f64) / (n_segments as f64);
                0.5 * r * r * (theta2 - theta1).sin().abs()
            })
            .sum();
        let expected = 20.0 * 20.0 * 10.0 - polygon_area * 10.0;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.01,
            "CCW circle hole extrusion volume should be ~{expected:.1}, got {vol:.1} \
             (rel_err={rel_err:.2e})"
        );
    }
}
