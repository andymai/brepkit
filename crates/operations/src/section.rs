//! Sectioning (slicing) solids with planes.
//!
//! Computes the cross-section of a solid at a given cutting plane,
//! producing face(s) representing the intersection. This is the
//! equivalent of `BRepAlgoAPI_Section` in `OpenCascade`.

#![allow(clippy::too_many_lines, clippy::doc_markdown)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::boolean::face_vertices;
use crate::dot_normal_point;

/// A cross-section result: one or more planar faces on the cutting plane.
#[derive(Debug)]
pub struct Section {
    /// The face IDs of the cross-section faces in the topology.
    pub faces: Vec<FaceId>,
}

/// Compute the cross-section of a solid with a plane.
///
/// The cutting plane is defined by a point on the plane and its normal.
/// Returns the cross-section as one or more planar faces lying on the
/// cutting plane. For a simple convex solid this is typically one face;
/// for solids with holes or disconnected volumes there may be multiple.
///
/// # Algorithm
///
/// 1. For each face of the solid, compute the intersection with the
///    cutting plane (line segments for planar faces).
/// 2. Collect all intersection segments.
/// 3. Assemble segments into closed wires.
/// 4. Create planar faces from the wires.
///
/// # Errors
///
/// Returns an error if no intersection exists (plane doesn't cut
/// the solid), or if NURBS intersection computation fails.
pub fn section(
    topo: &mut Topology,
    solid: SolidId,
    plane_point: Point3,
    plane_normal: Vec3,
) -> Result<Section, crate::OperationsError> {
    let tol = Tolerance::new();

    // Normalize the cutting plane normal.
    let normal = plane_normal.normalize()?;
    let d = dot_normal_point(normal, plane_point);

    // Collect all intersection segments from the solid's faces.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut segments: Vec<(Point3, Point3)> = Vec::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane {
                normal: face_normal,
                d: face_d,
            } => {
                let face_normal = *face_normal;
                let face_d = *face_d;

                let verts = face_vertices(topo, fid)?;
                if let Some(seg) =
                    intersect_planar_face_with_plane(&verts, face_normal, face_d, normal, d, tol)
                {
                    segments.push(seg);
                }
            }
            FaceSurface::Nurbs(nurbs) => {
                // Use surface-plane intersection to find crossing points.
                let intersection_curves =
                    brepkit_math::nurbs::intersection::intersect_plane_nurbs(nurbs, normal, d, 50)?;
                for curve in &intersection_curves {
                    if curve.points.len() >= 2 {
                        let first = curve.points.first().map(|p| p.point);
                        let last = curve.points.last().map(|p| p.point);
                        if let (Some(a), Some(b)) = (first, last) {
                            segments.push((a, b));
                        }
                    }
                }
            }
            FaceSurface::Cylinder(cyl) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_cylinder(cyl, normal, d)?;
                for curve in &curves {
                    if curve.points.len() >= 2 {
                        if let (Some(a), Some(b)) = (
                            curve.points.first().map(|p| p.point),
                            curve.points.last().map(|p| p.point),
                        ) {
                            segments.push((a, b));
                        }
                    }
                }
            }
            FaceSurface::Cone(cone) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_cone(cone, normal, d)?;
                for curve in &curves {
                    if curve.points.len() >= 2 {
                        if let (Some(a), Some(b)) = (
                            curve.points.first().map(|p| p.point),
                            curve.points.last().map(|p| p.point),
                        ) {
                            segments.push((a, b));
                        }
                    }
                }
            }
            FaceSurface::Sphere(sphere) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_sphere(sphere, normal, d)?;
                for curve in &curves {
                    if curve.points.len() >= 2 {
                        if let (Some(a), Some(b)) = (
                            curve.points.first().map(|p| p.point),
                            curve.points.last().map(|p| p.point),
                        ) {
                            segments.push((a, b));
                        }
                    }
                }
            }
            FaceSurface::Torus(torus) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_torus(torus, normal, d)?;
                for curve in &curves {
                    if curve.points.len() >= 2 {
                        if let (Some(a), Some(b)) = (
                            curve.points.first().map(|p| p.point),
                            curve.points.last().map(|p| p.point),
                        ) {
                            segments.push((a, b));
                        }
                    }
                }
            }
        }
    }

    if segments.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "cutting plane does not intersect the solid".into(),
        });
    }

    // Assemble segments into closed wires.
    let wires = assemble_wires(topo, &segments, normal, d, tol)?;

    if wires.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no closed cross-section could be assembled".into(),
        });
    }

    // Create faces from wires.
    let mut result_faces = Vec::with_capacity(wires.len());
    for wid in wires {
        let face = topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));
        result_faces.push(face);
    }

    Ok(Section {
        faces: result_faces,
    })
}

/// Intersect a planar face polygon with a cutting plane.
///
/// Returns the line segment (if any) where the cutting plane crosses
/// the face polygon. Uses the Sutherland-Hodgman approach: classify
/// each vertex as above/below the plane, find edge crossings.
fn intersect_planar_face_with_plane(
    verts: &[Point3],
    _face_normal: Vec3,
    _face_d: f64,
    cut_normal: Vec3,
    cut_d: f64,
    tol: Tolerance,
) -> Option<(Point3, Point3)> {
    let n = verts.len();
    if n < 3 {
        return None;
    }

    // Classify each vertex relative to the cutting plane.
    let dists: Vec<f64> = verts
        .iter()
        .map(|v| dot_normal_point(cut_normal, *v) - cut_d)
        .collect();

    // If all vertices lie on the cutting plane the face is coplanar —
    // the cross-section boundary comes from adjacent faces, not this one.
    if dists.iter().all(|d| d.abs() < tol.linear) {
        return None;
    }

    // Collect intersection points where edges cross the plane.
    let mut crossings = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let di = dists[i];
        let dj = dists[j];

        // Check if vertex is on the plane.
        if di.abs() < tol.linear {
            crossings.push(verts[i]);
            continue;
        }

        // Check for sign change (edge crosses plane).
        if (di > tol.linear && dj < -tol.linear) || (di < -tol.linear && dj > tol.linear) {
            let t = di / (di - dj);
            let pi = verts[i];
            let pj = verts[j];
            let ix = Point3::new(
                (pj.x() - pi.x()).mul_add(t, pi.x()),
                (pj.y() - pi.y()).mul_add(t, pi.y()),
                (pj.z() - pi.z()).mul_add(t, pi.z()),
            );
            crossings.push(ix);
        }
    }

    // Deduplicate nearby crossings.
    let mut unique = Vec::new();
    for p in &crossings {
        if !unique
            .iter()
            .any(|q: &Point3| (*p - *q).length_squared() < tol.linear * tol.linear)
        {
            unique.push(*p);
        }
    }

    if unique.len() >= 2 {
        Some((unique[0], unique[1]))
    } else {
        None
    }
}

/// Assemble intersection segments into closed wires.
///
/// Chains segments endpoint-to-endpoint using spatial proximity,
/// producing one or more closed wires.
fn assemble_wires(
    topo: &mut Topology,
    segments: &[(Point3, Point3)],
    _normal: Vec3,
    _d: f64,
    tol: Tolerance,
) -> Result<Vec<WireId>, crate::OperationsError> {
    if segments.is_empty() {
        return Ok(vec![]);
    }

    // Convert segments to a mutable list for chaining.
    let mut remaining: Vec<(Point3, Point3)> = segments.to_vec();
    let mut wires = Vec::new();

    while !remaining.is_empty() {
        // Start a new chain with the first remaining segment.
        let first = remaining.remove(0);
        let mut chain: Vec<Point3> = vec![first.0, first.1];

        // Iteratively find segments that connect to the chain end.
        let mut changed = true;
        while changed {
            changed = false;
            let chain_end = chain[chain.len() - 1];

            for i in 0..remaining.len() {
                let (a, b) = remaining[i];
                let dist_a = (a - chain_end).length_squared();
                let dist_b = (b - chain_end).length_squared();
                let threshold_sq = (tol.linear * 1000.0) * (tol.linear * 1000.0);

                if dist_a < threshold_sq {
                    chain.push(b);
                    remaining.remove(i);
                    changed = true;
                    break;
                } else if dist_b < threshold_sq {
                    chain.push(a);
                    remaining.remove(i);
                    changed = true;
                    break;
                }
            }
        }

        // Check if the chain forms a closed loop.
        if chain.len() < 3 {
            continue;
        }

        let start = chain[0];
        let end = chain[chain.len() - 1];
        let close_tol = tol.linear * 1000.0;
        let closed = (start - end).length_squared() < close_tol * close_tol;

        if !closed {
            // Not a closed wire — skip (partial section).
            continue;
        }

        // Remove the duplicate closing point if present.
        if chain.len() > 3 {
            chain.pop();
        }

        // Build topology: vertices, edges, wire.
        let n = chain.len();
        let vert_ids: Vec<_> = chain
            .iter()
            .map(|&p| topo.vertices.alloc(Vertex::new(p, tol.linear)))
            .collect();

        let mut oriented_edges = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i + 1) % n;
            let edge = topo
                .edges
                .alloc(Edge::new(vert_ids[i], vert_ids[j], EdgeCurve::Line));
            oriented_edges.push(OrientedEdge::new(edge, true));
        }

        let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        wires.push(topo.wires.alloc(wire));
    }

    Ok(wires)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    #[test]
    fn section_cube_at_half_height() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Cut with a horizontal plane at z=0.5.
        let result = section(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        assert_eq!(
            result.faces.len(),
            1,
            "should produce one cross-section face"
        );

        // The cross section of a unit cube at z=0.5 should be a unit square.
        let area = crate::measure::face_area(&topo, result.faces[0], 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(area, 1.0),
            "cross-section area should be ~1.0, got {area}"
        );
    }

    #[test]
    fn section_cube_at_quarter_height() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let result = section(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 0.25),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        assert_eq!(result.faces.len(), 1);

        // Still a 1×1 square at any height for a cube.
        let area = crate::measure::face_area(&topo, result.faces[0], 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(area, 1.0),
            "cross-section area should be ~1.0, got {area}"
        );
    }

    #[test]
    fn section_cube_along_x() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Cut with a vertical plane at x=0.5.
        let result = section(
            &mut topo,
            cube,
            Point3::new(0.5, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        assert_eq!(result.faces.len(), 1);

        // Cross-section of unit cube at x=0.5 is a 1×1 square in YZ.
        let area = crate::measure::face_area(&topo, result.faces[0], 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(area, 1.0),
            "cross-section area should be ~1.0, got {area}"
        );
    }

    #[test]
    fn section_plane_misses_solid() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Plane above the cube.
        let result = section(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(result.is_err(), "plane above cube should produce error");
    }

    #[test]
    fn section_extruded_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        // Box extends from (0,0,0) to (2,3,4). Cut at z=2 (middle of the box).
        let result = section(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        assert_eq!(result.faces.len(), 1);

        let area = crate::measure::face_area(&topo, result.faces[0], 0.1).unwrap();
        let tol = Tolerance::loose();
        // 2×3 = 6
        assert!(
            tol.approx_eq(area, 6.0),
            "cross-section area should be ~6.0, got {area}"
        );
    }

    #[test]
    fn section_diagonal_plane() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Diagonal cutting plane x + y = 0.8, which avoids edges/vertices.
        // It passes through the interior of 4 faces:
        // front(y=0): x=0.8 → (0.8,0,0)-(0.8,0,1)
        // back(y=1): x=-0.2 → misses (x not in [0,1])
        // left(x=0): y=0.8 → (0,0.8,0)-(0,0.8,1)
        // right(x=1): y=-0.2 → misses
        // top(z=1): line x+y=0.8 → (0.8,0,1)-(0,0.8,1)
        // bottom(z=0): line x+y=0.8 → (0.8,0,0)-(0,0.8,0)
        let result = section(
            &mut topo,
            cube,
            Point3::new(0.4, 0.4, 0.5),
            Vec3::new(1.0, 1.0, 0.0),
        );

        // Should produce a rectangular cross-section.
        assert!(
            result.is_ok(),
            "diagonal plane should intersect cube: {:?}",
            result.err()
        );
        let sec = result.unwrap();
        assert_eq!(sec.faces.len(), 1);
    }
}
