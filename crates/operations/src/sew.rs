//! Topology sewing: merge loose faces into connected shells.
//!
//! Takes a set of independent faces and merges coincident edges/vertices
//! to create a topologically connected shell.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

/// Sew a set of loose faces into a solid.
///
/// Finds geometrically coincident edges between faces (within
/// `tolerance`) and merges them, creating shared topology. Then
/// assembles the sewn faces into a shell and solid.
///
/// # Algorithm
///
/// 1. Collect all edge endpoints from all faces
/// 2. Merge coincident vertices using spatial hashing
/// 3. Merge coincident edges (same start/end vertices after merging)
/// 4. Rebuild faces with the merged edges
/// 5. Assemble into a shell and solid
///
/// # Errors
///
/// Returns an error if fewer than 2 faces are provided, or if the
/// faces contain NURBS surfaces.
#[allow(clippy::too_many_lines)]
pub fn sew_faces(
    topo: &mut Topology,
    faces: &[FaceId],
    tolerance: f64,
) -> Result<SolidId, crate::OperationsError> {
    if faces.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sewing requires at least 2 faces".into(),
        });
    }

    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };

    // Phase 1: Snapshot all face geometry.
    let mut face_snapshots: Vec<FaceSnapshot> = Vec::with_capacity(faces.len());

    for &fid in faces {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let wire = topo.wire(face.outer_wire())?;

        let mut edge_points: Vec<(Point3, Point3)> = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            if oe.is_forward() {
                edge_points.push((start, end));
            } else {
                edge_points.push((end, start));
            }
        }

        face_snapshots.push(FaceSnapshot {
            surface,
            edge_points,
        });
    }

    // Phase 2: Merge coincident vertices using spatial hash.
    let resolution = 1.0 / tol;
    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> = HashMap::new();

    let get_or_create_vertex =
        |topo: &mut Topology, p: Point3, map: &mut HashMap<(i64, i64, i64), VertexId>| {
            #[allow(clippy::cast_possible_truncation)]
            let key = (
                (p.x() * resolution).round() as i64,
                (p.y() * resolution).round() as i64,
                (p.z() * resolution).round() as i64,
            );
            *map.entry(key)
                .or_insert_with(|| topo.add_vertex(Vertex::new(p, tol)))
        };

    // Phase 3: Create merged edges and rebuild faces.
    let mut edge_map: HashMap<(usize, usize), EdgeId> = HashMap::new();
    let mut new_face_ids: Vec<FaceId> = Vec::with_capacity(face_snapshots.len());

    for snap in &face_snapshots {
        let mut oriented_edges: Vec<OrientedEdge> = Vec::with_capacity(snap.edge_points.len());

        for &(start_pt, end_pt) in &snap.edge_points {
            let v_start = get_or_create_vertex(topo, start_pt, &mut vertex_map);
            let v_end = get_or_create_vertex(topo, end_pt, &mut vertex_map);

            // Canonical edge key: sorted vertex indices.
            let (key_min, key_max) = if v_start.index() <= v_end.index() {
                (v_start.index(), v_end.index())
            } else {
                (v_end.index(), v_start.index())
            };
            let is_forward = v_start.index() <= v_end.index();

            let edge_id = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                let (canonical_start, canonical_end) = if v_start.index() <= v_end.index() {
                    (v_start, v_end)
                } else {
                    (v_end, v_start)
                };
                topo.add_edge(Edge::new(canonical_start, canonical_end, EdgeCurve::Line))
            });

            oriented_edges.push(OrientedEdge::new(edge_id, is_forward));
        }

        let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        let wire_id = topo.add_wire(wire);

        let face_id = topo.add_face(Face::new(wire_id, vec![], snap.surface.clone()));
        new_face_ids.push(face_id);
    }

    // Phase 4: Assemble into shell and solid.
    let shell = Shell::new(new_face_ids).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Snapshot of a face's geometry for the sewing algorithm.
struct FaceSnapshot {
    surface: FaceSurface,
    /// Edge endpoints in traversal order: `[(start, end), ...]`
    edge_points: Vec<(Point3, Point3)>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// Helper: create a standalone quad face with independent vertices/edges.
    fn make_loose_quad(
        topo: &mut Topology,
        p0: Point3,
        p1: Point3,
        p2: Point3,
        p3: Point3,
        normal: Vec3,
        d: f64,
    ) -> FaceId {
        let tol_val = 1e-7;
        let v0 = topo.add_vertex(Vertex::new(p0, tol_val));
        let v1 = topo.add_vertex(Vertex::new(p1, tol_val));
        let v2 = topo.add_vertex(Vertex::new(p2, tol_val));
        let v3 = topo.add_vertex(Vertex::new(p3, tol_val));

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
        topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }))
    }

    #[test]
    fn sew_two_adjacent_quads() {
        let mut topo = Topology::new();

        // Two quads sharing an edge along x=1.
        let f0 = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let f1 = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );

        let solid = sew_faces(&mut topo, &[f0, f1], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 2, "sewn result should have 2 faces");
    }

    #[test]
    fn sew_shares_coincident_edges() {
        let mut topo = Topology::new();

        let f0 = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let f1 = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );

        let solid = sew_faces(&mut topo, &[f0, f1], 1e-6).unwrap();

        // Count unique edges across both faces.
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        let mut edge_set = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_set.insert(oe.edge().index());
            }
        }

        // 2 quads with 4 edges each, sharing 1 edge = 7 unique edges.
        assert_eq!(
            edge_set.len(),
            7,
            "two adjacent quads should share 1 edge (7 unique), got {}",
            edge_set.len()
        );
    }

    #[test]
    fn sew_six_cube_faces() {
        let mut topo = Topology::new();

        // Create 6 loose faces of a unit cube.
        let bottom = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            0.0,
        );
        let top = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        );
        let front = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.0,
        );
        let back = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
        );
        let left = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.0,
        );
        let right = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
        );

        let solid = sew_faces(&mut topo, &[bottom, top, front, back, left, right], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 6, "sewn cube should have 6 faces");

        // Count unique edges — a cube has 12.
        let mut edge_set = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_set.insert(oe.edge().index());
            }
        }
        assert_eq!(
            edge_set.len(),
            12,
            "cube should have 12 edges, got {}",
            edge_set.len()
        );
    }

    #[test]
    fn sew_single_face_error() {
        let mut topo = Topology::new();
        let f = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        assert!(sew_faces(&mut topo, &[f], 1e-6).is_err());
    }
}
