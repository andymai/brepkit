//! Loft operation: create a solid by interpolating between profile faces.
//!
//! Equivalent to `BRepOffsetAPI_ThruSections` in `OpenCascade`. The loft
//! connects two or more planar profiles by creating ruled (linear)
//! surfaces between corresponding profile edges.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::boolean::face_vertices;
use crate::dot_normal_point;

/// Loft two or more planar profiles into a solid.
///
/// Each profile is a planar face. All profiles must have the same
/// number of boundary vertices. The loft connects corresponding
/// vertices between adjacent profiles with ruled (linear) surfaces,
/// and caps the first and last profiles as the solid's end faces.
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 2 profiles are provided
/// - Profiles have different vertex counts
/// - Any profile is not a planar face
#[allow(clippy::too_many_lines)]
pub fn loft(topo: &mut Topology, profiles: &[FaceId]) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if profiles.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft requires at least 2 profiles".into(),
        });
    }

    // Collect vertex positions for each profile.
    let mut profile_verts: Vec<Vec<Point3>> = Vec::with_capacity(profiles.len());
    for &fid in profiles {
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { .. } => {}
            FaceSurface::Nurbs(_) => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "loft of NURBS faces is not supported".into(),
                });
            }
        }
        let verts = face_vertices(topo, fid)?;
        profile_verts.push(verts);
    }

    // Validate all profiles have the same vertex count.
    let n = profile_verts[0].len();
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft profiles must have at least 3 vertices".into(),
        });
    }
    for (i, verts) in profile_verts.iter().enumerate() {
        if verts.len() != n {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!(
                    "profile {} has {} vertices, but profile 0 has {n}",
                    i,
                    verts.len()
                ),
            });
        }
    }

    let num_profiles = profile_verts.len();
    let num_sections = num_profiles - 1;

    // Create all vertices.
    let ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> = profile_verts
        .iter()
        .map(|verts| {
            verts
                .iter()
                .map(|&p| topo.vertices.alloc(Vertex::new(p, tol.linear)))
                .collect()
        })
        .collect();

    // Create profile edges for each ring.
    let ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = ring_verts
        .iter()
        .map(|ring| {
            (0..n)
                .map(|i| {
                    let next = (i + 1) % n;
                    topo.edges
                        .alloc(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                })
                .collect()
        })
        .collect();

    // Create connecting edges between adjacent profiles.
    let connect_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = (0..num_sections)
        .map(|s| {
            (0..n)
                .map(|i| {
                    topo.edges.alloc(Edge::new(
                        ring_verts[s][i],
                        ring_verts[s + 1][i],
                        EdgeCurve::Line,
                    ))
                })
                .collect()
        })
        .collect();

    let mut all_faces = Vec::new();

    // Start cap: reversed first profile (outward normal pointing away from loft).
    {
        let face_data = topo.face(profiles[0])?;
        let cap_normal = match face_data.surface() {
            FaceSurface::Plane { normal, .. } => -*normal,
            FaceSurface::Nurbs(_) => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "unexpected NURBS face".into(),
                });
            }
        };
        let reversed_edges: Vec<OrientedEdge> = (0..n)
            .rev()
            .map(|i| OrientedEdge::new(ring_edges[0][i], false))
            .collect();
        let wire = Wire::new(reversed_edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.wires.alloc(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[0][0]);
        let fid = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        ));
        all_faces.push(fid);
    }

    // Side faces: one quad per profile-edge × section.
    for s in 0..num_sections {
        for i in 0..n {
            let next_i = (i + 1) % n;

            // Quad: ring[s][i] → ring[s][next_i] → ring[s+1][next_i] → ring[s+1][i]
            let p0 = profile_verts[s][i];
            let p1 = profile_verts[s][next_i];
            let p_next = profile_verts[s + 1][i];
            let edge_dir = p1 - p0;
            let connect_dir = p_next - p0;
            let side_normal = edge_dir
                .cross(connect_dir)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let side_d = dot_normal_point(side_normal, p0);

            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(ring_edges[s][i], true),
                    OrientedEdge::new(connect_edges[s][next_i], true),
                    OrientedEdge::new(ring_edges[s + 1][i], false),
                    OrientedEdge::new(connect_edges[s][i], false),
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

    // End cap: last profile with forward orientation.
    {
        let face_data = topo.face(profiles[num_profiles - 1])?;
        let cap_normal = match face_data.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            FaceSurface::Nurbs(_) => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "unexpected NURBS face".into(),
                });
            }
        };
        let edges: Vec<OrientedEdge> = (0..n)
            .map(|i| OrientedEdge::new(ring_edges[num_profiles - 1][i], true))
            .collect();
        let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.wires.alloc(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[num_profiles - 1][0]);
        let fid = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        ));
        all_faces.push(fid);
    }

    // Assemble.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// Helper: make a square face at z=offset with given size.
    fn make_square_at(topo: &mut Topology, size: f64, z: f64) -> FaceId {
        let hs = size / 2.0;
        let tol_val = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(-hs, -hs, z), tol_val));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(hs, -hs, z), tol_val));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(hs, hs, z), tol_val));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(-hs, hs, z), tol_val));

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

        topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: z,
            },
        ))
    }

    #[test]
    fn loft_two_identical_squares_makes_box() {
        let mut topo = Topology::new();
        let bottom = make_square_at(&mut topo, 1.0, 0.0);
        let top = make_square_at(&mut topo, 1.0, 1.0);

        let solid = loft(&mut topo, &[bottom, top]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 2 caps + 4 sides = 6 faces
        assert_eq!(sh.faces().len(), 6, "lofted box should have 6 faces");

        // Volume should be 1.0 (unit cube)
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 1.0),
            "lofted box volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn loft_tapered_frustum() {
        let mut topo = Topology::new();
        let bottom = make_square_at(&mut topo, 2.0, 0.0);
        let top = make_square_at(&mut topo, 1.0, 3.0);

        let solid = loft(&mut topo, &[bottom, top]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        assert_eq!(sh.faces().len(), 6);

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        // Frustum of a square pyramid: V = h/3 * (A1 + A2 + sqrt(A1*A2))
        // A1 = 4.0, A2 = 1.0, h = 3.0
        // V = 3/3 * (4 + 1 + 2) = 7.0
        let expected = 7.0;
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "tapered frustum volume should be ~{expected}, got {vol} (error: {:.1}%)",
            (vol - expected).abs() / expected * 100.0
        );
    }

    #[test]
    fn loft_three_profiles() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 2.0, 0.0);
        let p1 = make_square_at(&mut topo, 1.0, 1.5);
        let p2 = make_square_at(&mut topo, 2.0, 3.0);

        let solid = loft(&mut topo, &[p0, p1, p2]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 2 caps + 2 sections × 4 edges = 10 faces
        assert_eq!(sh.faces().len(), 10);

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(vol > 0.0, "lofted solid should have positive volume");
    }

    #[test]
    fn loft_single_profile_error() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 1.0, 0.0);

        assert!(loft(&mut topo, &[p0]).is_err());
    }

    #[test]
    fn loft_mismatched_vertex_count_error() {
        let mut topo = Topology::new();
        let square = make_square_at(&mut topo, 1.0, 0.0);

        // Create a triangle profile.
        let tol_val = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 1.0), tol_val));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 1.0), tol_val));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.5, 1.0, 1.0), tol_val));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);
        let tri = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 1.0,
            },
        ));

        assert!(
            loft(&mut topo, &[square, tri]).is_err(),
            "mismatched vertex counts should fail"
        );
    }
}
