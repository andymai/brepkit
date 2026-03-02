//! Measurement operations for B-rep geometry: bounding boxes, areas,
//! volumes, and centers of mass.

use std::collections::HashSet;

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::tessellate;

// ── Bounding box ──────────────────────────────────────────────────

/// Compute the axis-aligned bounding box of a solid from its vertices.
///
/// This is exact for planar solids and a tight approximation for NURBS
/// solids (NURBS control point hulls may extend slightly beyond
/// the actual surface but vertices lie on the surface boundary).
///
/// # Errors
///
/// Returns an error if the solid has no vertices or a topology lookup fails.
pub fn solid_bounding_box(
    topo: &Topology,
    solid: SolidId,
) -> Result<Aabb3, crate::OperationsError> {
    let points = collect_solid_vertex_points(topo, solid)?;
    Aabb3::try_from_points(points.iter().copied()).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "solid has no vertices".into(),
        }
    })
}

// ── Face area ─────────────────────────────────────────────────────

/// Compute the area of a single face.
///
/// For planar faces, uses Newell's method (exact, no tessellation).
/// For NURBS faces, tessellates and sums triangle areas.
///
/// # Errors
///
/// Returns an error if the face is missing or tessellation fails.
pub fn face_area(
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;

    if matches!(face.surface(), FaceSurface::Plane { .. }) {
        planar_face_area(topo, face_id)
    } else {
        let mesh = tessellate::tessellate(topo, face_id, deflection)?;
        Ok(triangle_mesh_area(&mesh))
    }
}

/// Newell's method: compute the area of a planar polygon from its
/// boundary vertices.
fn planar_face_area(topo: &Topology, face_id: FaceId) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let positions = collect_wire_positions(topo, wire)?;

    let n = positions.len();
    if n < 3 {
        return Ok(0.0);
    }

    // Newell's method: sum cross products of consecutive edge pairs.
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = positions[i];
        let vj = positions[j];
        sx = vi.z().mul_add(-vj.y(), vi.y().mul_add(vj.z(), sx));
        sy = vi.x().mul_add(-vj.z(), vi.z().mul_add(vj.x(), sy));
        sz = vi.y().mul_add(-vj.x(), vi.x().mul_add(vj.y(), sz));
    }

    let area = 0.5 * sz.mul_add(sz, sx.mul_add(sx, sy * sy)).sqrt();
    Ok(area)
}

/// Sum of triangle areas from a tessellated mesh.
fn triangle_mesh_area(mesh: &tessellate::TriangleMesh) -> f64 {
    let mut area = 0.0;
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    let tri_count = idx.len() / 3;

    for t in 0..tri_count {
        let i0 = idx[t * 3] as usize;
        let i1 = idx[t * 3 + 1] as usize;
        let i2 = idx[t * 3 + 2] as usize;

        let a = pos[i1] - pos[i0];
        let b = pos[i2] - pos[i0];
        area += 0.5 * a.cross(b).length();
    }

    area
}

// ── Solid surface area ────────────────────────────────────────────

/// Compute the total surface area of a solid.
///
/// Sums `face_area()` over every face in every shell.
///
/// # Errors
///
/// Returns an error if a topology lookup or tessellation fails.
pub fn solid_surface_area(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let mut total = 0.0;
    for fid in collect_solid_face_ids(topo, solid)? {
        total += face_area(topo, fid, deflection)?;
    }
    Ok(total)
}

// ── Volume ────────────────────────────────────────────────────────

/// Compute the volume of a solid using the signed tetrahedra method
/// (divergence theorem on a surface tessellation).
///
/// For each triangle `(v0, v1, v2)`, the signed volume of the
/// tetrahedron it forms with the origin is `v0 · (v1 × v2) / 6`.
///
/// # Errors
///
/// Returns an error if tessellation or topology lookups fail.
pub fn solid_volume(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<f64, crate::OperationsError> {
    let mut total = 0.0;

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    for &fid in shell.faces() {
        let mesh = tessellate::tessellate(topo, fid, deflection)?;
        let idx = &mesh.indices;
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;
        if tri_count == 0 {
            continue;
        }

        // Determine winding sign: compare the tessellated triangle normal
        // with the stored mesh normal (which reflects the face's intended
        // outward direction). If they disagree, flip the contribution.
        let sign = face_winding_sign(&mesh);

        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());

            total += sign * a.dot(b.cross(c));
        }
    }

    Ok((total / 6.0).abs())
}

// ── Center of mass ────────────────────────────────────────────────

/// Compute the center of mass of a solid, assuming uniform density.
///
/// Uses the same signed-tetrahedra decomposition as `solid_volume`,
/// accumulating the centroid contribution of each tetrahedron:
/// `centroid += signed_vol * (a + b + c)`, then divides by
/// `4 * total_volume`.
///
/// # Errors
///
/// Returns an error if the solid has zero volume or tessellation fails.
pub fn solid_center_of_mass(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<Point3, crate::OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total_vol = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;

    for &fid in shell.faces() {
        let mesh = tessellate::tessellate(topo, fid, deflection)?;
        let idx = &mesh.indices;
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;
        if tri_count == 0 {
            continue;
        }

        let sign = face_winding_sign(&mesh);

        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];

            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());

            let signed_vol = sign * a.dot(b.cross(c));
            total_vol += signed_vol;
            cx += signed_vol * (v0.x() + v1.x() + v2.x());
            cy += signed_vol * (v0.y() + v1.y() + v2.y());
            cz += signed_vol * (v0.z() + v1.z() + v2.z());
        }
    }

    if total_vol.abs() < 1e-15 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "solid has zero volume, center of mass is undefined".into(),
        });
    }

    let denom = 4.0 * total_vol;
    Ok(Point3::new(cx / denom, cy / denom, cz / denom))
}

// ── Helpers ───────────────────────────────────────────────────────

/// Determine winding sign correction for a tessellated face.
///
/// Compares the geometric triangle normal (from cross product of first
/// triangle's edges) with the stored mesh normal. Returns `1.0` if they
/// agree, `-1.0` if the winding is reversed.
fn face_winding_sign(mesh: &tessellate::TriangleMesh) -> f64 {
    let idx = &mesh.indices;
    let pos = &mesh.positions;
    if idx.len() < 3 {
        return 1.0;
    }
    let i0 = idx[0] as usize;
    let i1 = idx[1] as usize;
    let i2 = idx[2] as usize;
    let edge1 = pos[i1] - pos[i0];
    let edge2 = pos[i2] - pos[i0];
    let tri_normal = edge1.cross(edge2);
    let mesh_normal = mesh.normals[i0];
    if tri_normal.dot(mesh_normal) >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Collect deduplicated vertex positions from a solid.
fn collect_solid_vertex_points(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let solid_data = topo.solid(solid)?;

    for shell_id in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id)?;
        for &fid in shell.faces() {
            let face = topo.face(fid)?;
            for wire_id in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge())?;
                    vertex_ids.insert(edge.start());
                    vertex_ids.insert(edge.end());
                }
            }
        }
    }

    let mut points = Vec::with_capacity(vertex_ids.len());
    for vid in vertex_ids {
        points.push(topo.vertex(vid)?.point());
    }
    Ok(points)
}

/// Collect all face IDs from a solid's shells.
fn collect_solid_face_ids(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<FaceId>, crate::OperationsError> {
    let mut face_ids = Vec::new();
    let solid_data = topo.solid(solid)?;

    for shell_id in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id)?;
        face_ids.extend_from_slice(shell.faces());
    }
    Ok(face_ids)
}

/// Collect ordered vertex positions from a wire.
fn collect_wire_positions(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let mut positions = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        positions.push(topo.vertex(vid)?.point());
    }
    Ok(positions)
}

// ── Edge and wire length ──────────────────────────────────────────

/// Compute the length of a single edge.
///
/// For line edges, returns the Euclidean distance between endpoints.
/// For NURBS curve edges, uses numerical integration (Simpson's rule).
///
/// # Errors
///
/// Returns an error if the edge lookup fails.
pub fn edge_length(
    topo: &Topology,
    edge_id: brepkit_topology::edge::EdgeId,
) -> Result<f64, crate::OperationsError> {
    let edge = topo.edge(edge_id)?;
    match edge.curve() {
        brepkit_topology::edge::EdgeCurve::Line => {
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            Ok((end - start).length())
        }
        brepkit_topology::edge::EdgeCurve::NurbsCurve(curve) => Ok(curve.arc_length(50)),
    }
}

/// Compute the total length (perimeter) of a wire.
///
/// Sums the length of all edges in the wire.
///
/// # Errors
///
/// Returns an error if any edge lookup fails.
pub fn wire_length(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<f64, crate::OperationsError> {
    let wire = topo.wire(wire_id)?;
    let mut total = 0.0;
    for oe in wire.edges() {
        total += edge_length(topo, oe.edge())?;
    }
    Ok(total)
}

/// Compute the perimeter of a face (outer wire length).
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn face_perimeter(topo: &Topology, face_id: FaceId) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    wire_length(topo, face.outer_wire())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;

    #[test]
    fn unit_cube_bounding_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        let tol = Tolerance::new();

        assert!(tol.approx_eq(aabb.min.x(), 0.0));
        assert!(tol.approx_eq(aabb.min.y(), 0.0));
        assert!(tol.approx_eq(aabb.min.z(), 0.0));
        assert!(tol.approx_eq(aabb.max.x(), 1.0));
        assert!(tol.approx_eq(aabb.max.y(), 1.0));
        assert!(tol.approx_eq(aabb.max.z(), 1.0));
    }

    #[test]
    fn unit_cube_volume() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 1.0),
            "unit cube volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn unit_cube_surface_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let area = solid_surface_area(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(area, 6.0),
            "unit cube surface area should be ~6.0, got {area}"
        );
    }

    #[test]
    fn unit_cube_center_of_mass() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(com.x(), 0.5),
            "center x should be ~0.5, got {}",
            com.x()
        );
        assert!(
            tol.approx_eq(com.y(), 0.5),
            "center y should be ~0.5, got {}",
            com.y()
        );
        assert!(
            tol.approx_eq(com.z(), 0.5),
            "center z should be ~0.5, got {}",
            com.z()
        );
    }

    #[test]
    fn unit_cube_face_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let area = face_area(&topo, fid, 0.1).unwrap();
            assert!(
                tol.approx_eq(area, 1.0),
                "each unit cube face should have area ~1.0, got {area}"
            );
        }
    }

    #[test]
    fn extruded_box_volume() {
        use brepkit_math::vec::{Point3, Vec3 as V};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        // Create a 2×3 rectangle on the XY plane.
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-10));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 0.0, 0.0), 1e-10));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 3.0, 0.0), 1e-10));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 3.0, 0.0), 1e-10));

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
        let face_id = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: V::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Extrude by 4 along Z.
        let solid =
            crate::extrude::extrude(&mut topo, face_id, V::new(0.0, 0.0, 1.0), 4.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 24.0),
            "2×3 rect extruded by 4 should have volume ~24.0, got {vol}"
        );
    }

    #[test]
    fn edge_length_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        // All edges of a unit cube have length 1.0.
        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                let len = edge_length(&topo, oe.edge()).unwrap();
                assert!(
                    tol.approx_eq(len, 1.0),
                    "unit cube edge should have length ~1.0, got {len}"
                );
            }
        }
    }

    #[test]
    fn face_perimeter_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let perim = face_perimeter(&topo, fid).unwrap();
            assert!(
                tol.approx_eq(perim, 4.0),
                "unit cube face perimeter should be ~4.0, got {perim}"
            );
        }
    }

    #[test]
    fn wire_length_rectangle() {
        use brepkit_topology::builder::make_rectangle_face;

        let mut topo = Topology::new();
        let fid = make_rectangle_face(&mut topo, 3.0, 5.0).unwrap();

        let face = topo.face(fid).unwrap();
        let len = wire_length(&topo, face.outer_wire()).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(len, 16.0),
            "3×5 rectangle perimeter should be ~16.0, got {len}"
        );
    }
}
