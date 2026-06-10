//! Sectioning (slicing) solids with planes.
//!
//! Computes the cross-section of a solid at a given cutting plane,
//! producing face(s) representing the intersection.

#![allow(clippy::too_many_lines, clippy::doc_markdown)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use brepkit_math::nurbs::intersection::IntersectionPoint;

use crate::boolean::face_polygon;
use crate::dot_normal_point;

/// Chain intersection curve points into consecutive segment pairs.
///
/// Instead of connecting only the first and last point (chord approximation),
/// this chains all intermediate points as individual segments, faithfully
/// tracing the actual intersection curve.
fn chain_curve_points(points: &[IntersectionPoint], segments: &mut Vec<(Point3, Point3)>) {
    if points.len() < 2 {
        return;
    }
    for pair in points.windows(2) {
        segments.push((pair[0].point, pair[1].point));
    }
}

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
/// A plane that misses the solid entirely yields an empty face list
/// (a successful empty result, not an error).
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
/// Returns an error if NURBS intersection computation fails, or if
/// intersection segments exist but cannot be assembled into a closed
/// cross-section wire.
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

    // Collect all intersection segments from all shells (outer + inner).
    let solid_data = topo.solid(solid)?;
    let all_shell_ids: Vec<_> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();

    let mut face_ids: Vec<FaceId> = Vec::new();
    for shell_id in &all_shell_ids {
        let shell = topo.shell(*shell_id)?;
        face_ids.extend_from_slice(shell.faces());
    }

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

                let verts = face_polygon(topo, fid)?;
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
                    chain_curve_points(&curve.points, &mut segments);
                }
            }
            FaceSurface::Cylinder(cyl) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_cylinder(cyl, normal, d)?;
                for curve in &curves {
                    chain_curve_points(&curve.points, &mut segments);
                }
            }
            FaceSurface::Cone(cone) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_cone(cone, normal, d)?;
                for curve in &curves {
                    chain_curve_points(&curve.points, &mut segments);
                }
            }
            FaceSurface::Sphere(sphere) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_sphere(sphere, normal, d)?;
                for curve in &curves {
                    chain_curve_points(&curve.points, &mut segments);
                }
            }
            FaceSurface::Torus(torus) => {
                let curves =
                    brepkit_math::analytic_intersection::intersect_plane_torus(torus, normal, d)?;
                for curve in &curves {
                    chain_curve_points(&curve.points, &mut segments);
                }
            }
        }
    }

    // If no crossing segments were found, the cutting plane may be exactly
    // coplanar with one or more faces. In that case, extract the boundary
    // edges of those coplanar faces as the cross-section.
    if segments.is_empty() {
        let coplanar_segs = extract_coplanar_boundary(topo, &face_ids, normal, d, tol)?;
        segments = coplanar_segs;
    }

    // No crossing segments and no coplanar boundary across outer + inner
    // shells means the cutting plane lies wholly on one side of the solid:
    // a genuine miss. The empty set is a valid section, not an error.
    if segments.is_empty() {
        return Ok(Section { faces: Vec::new() });
    }

    // Assemble segments into closed wires.
    let mut wires = assemble_wires(topo, &segments, normal, d, tol)?;

    // Fallback: if crossing-based segments didn't form closed wires,
    // try using coplanar face boundaries instead (handles the case where
    // the cutting plane exactly coincides with a face of the solid).
    if wires.is_empty() {
        let coplanar_segs = extract_coplanar_boundary(topo, &face_ids, normal, d, tol)?;
        if !coplanar_segs.is_empty() {
            wires = assemble_wires(topo, &coplanar_segs, normal, d, tol)?;
        }
    }

    if wires.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no closed cross-section could be assembled".into(),
        });
    }

    // Group wires by containment: outer wires get their inner wires as holes.
    // Without this, a hollow shape's section would produce two separate faces
    // (outer rectangle + inner hole rectangle) instead of one face-with-hole.
    let groups = group_wires_by_containment(topo, &wires, normal);

    let mut result_faces = Vec::with_capacity(groups.len());
    for (outer, inners) in groups {
        let face = topo.add_face(Face::new(outer, inners, FaceSurface::Plane { normal, d }));
        result_faces.push(face);
    }

    Ok(Section {
        faces: result_faces,
    })
}

/// Group section wires by containment: each outer wire collects all wires that
/// are spatially inside it as inner (hole) wires. Returns `(outer, [inner])` tuples.
///
/// A wire is "inner" when it sits inside another wire (in the section plane) and
/// is not itself inside any wire that's already inside another. Two-level nesting
/// (holes within holes within holes) is collapsed into the outermost containment;
/// CAD sections don't typically produce deeper nesting and OCCT treats them the
/// same way.
fn group_wires_by_containment(
    topo: &Topology,
    wires: &[WireId],
    normal: Vec3,
) -> Vec<(WireId, Vec<WireId>)> {
    if wires.len() <= 1 {
        return wires.iter().map(|&w| (w, vec![])).collect();
    }

    // Collect each wire's vertex polygon (used for both bounding-box and
    // point-in-polygon checks below).
    let polygons: Vec<Vec<Point3>> = wires.iter().map(|&wid| wire_vertices(topo, wid)).collect();

    // For each wire, find the smallest wire that strictly contains it.
    // A wire with no strict container is itself an outer wire.
    let mut parent: Vec<Option<usize>> = vec![None; wires.len()];
    for (i, poly_i) in polygons.iter().enumerate() {
        if poly_i.is_empty() {
            continue;
        }
        let sample = poly_i[0];
        let mut best_parent: Option<usize> = None;
        let mut best_area = f64::INFINITY;
        for (j, poly_j) in polygons.iter().enumerate() {
            if i == j || poly_j.len() < 3 {
                continue;
            }
            if crate::distance::point_in_polygon_3d(&sample, poly_j, &normal) {
                let area = polygon_area_3d(poly_j, normal);
                if area < best_area {
                    best_area = area;
                    best_parent = Some(j);
                }
            }
        }
        parent[i] = best_parent;
    }

    // Build groups: each outer wire (no parent) collects its direct children.
    let mut groups: Vec<(WireId, Vec<WireId>)> = Vec::new();
    let mut group_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (i, &wid) in wires.iter().enumerate() {
        if parent[i].is_none() {
            group_of.insert(i, groups.len());
            groups.push((wid, vec![]));
        }
    }
    for (i, &wid) in wires.iter().enumerate() {
        if let Some(p) = parent[i] {
            // Walk up to the outermost ancestor (in case of nested holes).
            let mut top = p;
            while let Some(next) = parent[top] {
                top = next;
            }
            if let Some(&group_idx) = group_of.get(&top) {
                groups[group_idx].1.push(wid);
            }
        }
    }

    groups
}

fn wire_vertices(topo: &Topology, wire_id: WireId) -> Vec<Point3> {
    let Ok(wire) = topo.wire(wire_id) else {
        return vec![];
    };
    let mut pts = Vec::with_capacity(wire.edges().len());
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        if let Ok(v) = topo.vertex(vid) {
            pts.push(v.point());
        }
    }
    pts
}

/// Unsigned area of a 3D planar polygon (projected to dominant axis plane).
fn polygon_area_3d(polygon: &[Point3], normal: Vec3) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();
    let to_2d = |p: Point3| -> (f64, f64) {
        if az >= ax && az >= ay {
            (p.x(), p.y())
        } else if ay >= ax {
            (p.x(), p.z())
        } else {
            (p.y(), p.z())
        }
    };
    let mut sum = 0.0;
    let n = polygon.len();
    for i in 0..n {
        let (x0, y0) = to_2d(polygon[i]);
        let (x1, y1) = to_2d(polygon[(i + 1) % n]);
        sum += x0 * y1 - x1 * y0;
    }
    (sum * 0.5).abs()
}

type EdgeKey = ((i64, i64, i64), (i64, i64, i64));

/// Extract boundary edges of faces coplanar with the cutting plane.
///
/// For each coplanar face, its edges are boundary edges if they are not shared
/// with another coplanar face. These boundary edges form the cross-section
/// outline where the solid meets the cutting plane.
fn extract_coplanar_boundary(
    topo: &Topology,
    face_ids: &[FaceId],
    cut_normal: Vec3,
    cut_d: f64,
    tol: Tolerance,
) -> Result<Vec<(Point3, Point3)>, crate::OperationsError> {
    use std::collections::HashMap;

    // Use a relaxed tolerance for coplanar detection to handle
    // floating-point precision differences across platforms (e.g. WASM).
    let coplanar_tol = tol.linear * 100.0;

    // Identify coplanar face indices.
    let mut coplanar_faces = Vec::new();
    for &fid in face_ids {
        let verts = face_polygon(topo, fid)?;
        if verts.len() >= 3
            && verts
                .iter()
                .all(|v| (dot_normal_point(cut_normal, *v) - cut_d).abs() < coplanar_tol)
        {
            coplanar_faces.push(fid);
        }
    }

    if coplanar_faces.is_empty() {
        return Ok(Vec::new());
    }

    // Collect all edges of coplanar faces. Each edge is represented by a pair
    // of quantized endpoint coordinates (to handle floating-point matching).
    // An edge shared by two coplanar faces appears twice and is internal.
    // An edge appearing once is a boundary edge.
    let quantize = |p: Point3| -> (i64, i64, i64) {
        let scale = 1.0 / (tol.linear * 10.0);
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    let edge_key = |a: Point3, b: Point3| -> EdgeKey {
        let qa = quantize(a);
        let qb = quantize(b);
        if qa <= qb { (qa, qb) } else { (qb, qa) }
    };

    // Count edge occurrences and store the actual points.
    let mut edge_counts: HashMap<EdgeKey, (Point3, Point3, usize)> = HashMap::new();

    for &fid in &coplanar_faces {
        let verts = face_polygon(topo, fid)?;
        let n = verts.len();
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            let key = edge_key(a, b);
            edge_counts
                .entry(key)
                .and_modify(|e| e.2 += 1)
                .or_insert((a, b, 1));
        }
    }

    // Boundary edges appear exactly once.
    let boundary: Vec<(Point3, Point3)> = edge_counts
        .into_values()
        .filter(|(_, _, count)| *count == 1)
        .map(|(a, b, _)| (a, b))
        .collect();

    Ok(boundary)
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
    // Use a relaxed tolerance (100× linear) for cross-platform consistency.
    let coplanar_tol = tol.linear * 100.0;
    if dists.iter().all(|d| d.abs() < coplanar_tol) {
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

    // Chaining tolerance must be tight enough to keep separate loops apart
    // (e.g., the outer perimeter and the inner hole of a hollow shape) while
    // still tolerating the small endpoint drift produced by tessellated NURBS
    // intersections. `tol.linear * 1000` (≈ 1e-4) is OCCT-compatible for both.
    //
    // Earlier this defaulted to 50% of the average segment length, which for
    // a hollow 20×20 box section was ≈ 15 — wildly generous, bridging the
    // outer and inner wires into one self-intersecting polygon.
    let chain_tol = tol.linear * 1000.0;

    while !remaining.is_empty() {
        // Start a new chain with the first remaining segment.
        let first = remaining.remove(0);
        let mut chain: Vec<Point3> = vec![first.0, first.1];

        // Iteratively find the closest segment that connects to the chain end.
        let mut changed = true;
        while changed {
            changed = false;
            let chain_end = chain[chain.len() - 1];
            let threshold_sq = chain_tol * chain_tol;

            // Find the closest matching segment endpoint.
            let mut best_idx = None;
            let mut best_dist = threshold_sq;
            let mut best_forward = true;

            for i in 0..remaining.len() {
                let (a, b) = remaining[i];
                let dist_a = (a - chain_end).length_squared();
                let dist_b = (b - chain_end).length_squared();

                if dist_a < best_dist {
                    best_dist = dist_a;
                    best_idx = Some(i);
                    best_forward = true;
                }
                if dist_b < best_dist {
                    best_dist = dist_b;
                    best_idx = Some(i);
                    best_forward = false;
                }
            }

            if let Some(idx) = best_idx {
                let (a, b) = remaining.remove(idx);
                chain.push(if best_forward { b } else { a });
                changed = true;
            }
        }

        // Check if the chain forms a closed loop.
        if chain.len() < 3 {
            continue;
        }

        let start = chain[0];
        let end = chain[chain.len() - 1];
        let closed = (start - end).length_squared() < chain_tol * chain_tol;

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
            .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
            .collect();

        let mut oriented_edges = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i + 1) % n;
            let edge = topo.add_edge(Edge::new(vert_ids[i], vert_ids[j], EdgeCurve::Line));
            oriented_edges.push(OrientedEdge::new(edge, true));
        }

        let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        wires.push(topo.add_wire(wire));
    }

    Ok(wires)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

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
        assert!(
            (area - 1.0).abs() < 1e-6,
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
        assert!(
            (area - 1.0).abs() < 1e-6,
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
        assert!(
            (area - 1.0).abs() < 1e-6,
            "cross-section area should be ~1.0, got {area}"
        );
    }

    /// Section a box-minus-sphere at z=0.
    ///
    /// Box is (0,0,0)-(20,20,20), sphere at origin r=12.
    /// At z=0, the sphere intersects the box creating a circular cutout.
    /// Cross-section = 20×20 rectangle minus circle of radius 12
    ///   (but sphere center is at origin and box starts at 0, so the
    ///   sphere only clips a quarter-circle at the z=0 face corner).
    ///
    /// The exact area depends on how much of the sphere lies within the
    /// box at this plane. At minimum, we verify the section succeeds and
    /// produces a face with positive area less than the full 20×20 = 400.
    #[test]
    #[ignore = "section through the cut region returns the full 400 box-face area — the cut cavity is missing from the section plane"]
    fn section_after_boolean_cut() {
        let mut topo = Topology::new();
        let b = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        let s = crate::primitives::make_sphere(&mut topo, 12.0, 16).unwrap();

        let solid =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, b, s).unwrap();

        let sec = section(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        assert!(!sec.faces.is_empty(), "should produce at least one face");

        // The cross-section area should be positive and less than the full
        // box face (400). The sphere removes a quarter-disc of radius 12
        // from the corner, so area ≈ 400 - π(144)/4 ≈ 400 - 113.1 ≈ 286.9.
        let total_area: f64 = sec
            .faces
            .iter()
            .map(|&fid| crate::measure::face_area(&topo, fid, 0.1).unwrap())
            .sum();
        assert!(
            total_area > 200.0,
            "section area should be > 200 (box face minus sphere), got {total_area:.2}"
        );
        assert!(
            total_area < 400.0,
            "section area should be < 400 (full box face), got {total_area:.2}"
        );
    }

    #[test]
    fn section_plane_misses_solid() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Plane above the cube — misses entirely.
        let result = section(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();
        assert!(
            result.faces.is_empty(),
            "plane above cube should produce an empty section"
        );
    }

    #[test]
    fn section_plane_flush_with_top_face() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Plane coincident with the cube's top face at z=1 — the coplanar
        // boundary fallback must yield the face outline, not a miss.
        let result = section(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();
        assert_eq!(
            result.faces.len(),
            1,
            "flush plane should yield the face outline"
        );
        let area = crate::measure::face_area(&topo, result.faces[0], 0.1).unwrap();
        assert!(
            (area - 1.0).abs() < 1e-6,
            "flush-face section area should be ~1.0, got {area}"
        );
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
        // 2×3 = 6
        assert!(
            (area - 6.0).abs() < 1e-6,
            "cross-section area should be ~6.0, got {area}"
        );
    }

    /// Diagonal section: plane x + y = 0.8, normal = (1,1,0).
    ///
    /// This plane intersects the unit cube at:
    ///   front (y=0): x=0.8 → line (0.8,0,0)-(0.8,0,1)
    ///   left (x=0): y=0.8 → line (0,0.8,0)-(0,0.8,1)
    ///   top (z=1): x+y=0.8 → line (0.8,0,1)-(0,0.8,1)
    ///   bottom (z=0): x+y=0.8 → line (0.8,0,0)-(0,0.8,0)
    ///
    /// The cross-section is a rectangle with:
    ///   width = distance between the two vertical lines
    ///         = |(0.8,0)-(0,0.8)| = √(0.64+0.64) = 0.8√2
    ///   height = 1.0 (z extent)
    ///   area = 0.8√2 ≈ 1.1314
    #[test]
    fn section_diagonal_plane() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let result = section(
            &mut topo,
            cube,
            Point3::new(0.4, 0.4, 0.5),
            Vec3::new(1.0, 1.0, 0.0),
        );

        assert!(
            result.is_ok(),
            "diagonal plane should intersect cube: {:?}",
            result.err()
        );
        let sec = result.unwrap();
        assert_eq!(sec.faces.len(), 1);

        // Area = width × height = 0.8√2 × 1.0 ≈ 1.1314
        let area = crate::measure::face_area(&topo, sec.faces[0], 0.01).unwrap();
        let expected = 0.8 * std::f64::consts::SQRT_2;
        let rel_err = (area - expected).abs() / expected;
        assert!(
            rel_err < 1e-4,
            "diagonal section area should be 0.8√2 ≈ {expected:.4}, got {area:.4} \
             (rel_err={rel_err:.2e})"
        );
    }

    /// Section of a cylinder at mid-height → circular cross-section.
    /// Cylinder r=5, h=10, section at z=5 → circle area = πr² = 25π ≈ 78.54.
    #[test]
    fn section_cylinder_at_midheight() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 5.0, 10.0).unwrap();

        let result = section(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        assert!(
            result.is_ok(),
            "section of cylinder should succeed: {:?}",
            result.err()
        );
        let sec = result.unwrap();
        assert!(!sec.faces.is_empty(), "should produce at least one face");

        let total_area: f64 = sec
            .faces
            .iter()
            .map(|&fid| crate::measure::face_area(&topo, fid, 0.01).unwrap())
            .sum();
        // Circle area = πr² = 25π ≈ 78.54
        let expected = std::f64::consts::PI * 25.0;
        let rel_err = (total_area - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "cylinder section area should be πr² = {expected:.2}, got {total_area:.2} \
             (rel_err={rel_err:.2e})"
        );
    }
}
