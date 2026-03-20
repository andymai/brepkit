//! Edge and vertex convexity classification.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_geometry::extrema::point_surface::point_to_surface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use crate::data::{EdgeClass, OffsetData, VertexClass};
use crate::error::OffsetError;

/// Squared length below which a vector is considered degenerate (zero).
const ZERO_LENGTH_SQ: f64 = 1e-30;

/// Squared norm below which two anti-parallel normals are considered
/// indistinguishable (their sum is zero).
const ANTI_PARALLEL_SQ: f64 = 1e-20;

/// Classify every edge of the solid as Convex, Concave, or Tangent,
/// and derive vertex classifications from incident edges.
///
/// # Errors
///
/// Returns [`OffsetError::AnalysisFailed`] if an edge cannot be classified.
#[allow(clippy::too_many_lines)]
pub fn analyse_edges(
    topo: &Topology,
    solid: SolidId,
    data: &mut OffsetData,
) -> Result<(), OffsetError> {
    let edge_face_map = brepkit_topology::explorer::edge_to_face_map(topo, solid)?;

    // Tolerance angle: 4 * arcsin(min(tol / (|offset| * 0.5), 1.0))
    let abs_offset = data.distance.abs();
    let tol = data.options.tolerance.linear;
    let tol_angle = if abs_offset > f64::EPSILON {
        4.0 * (tol / (abs_offset * 0.5)).min(1.0).asin()
    } else {
        // Zero offset — treat everything as tangent
        PI
    };

    // Track which vertices are incident to which edge classifications
    let mut vertex_edges: BTreeMap<usize, Vec<EdgeClass>> = BTreeMap::new();

    for (&edge_idx, face_ids) in &edge_face_map {
        // Only classify edges shared by exactly 2 faces (manifold edges)
        if face_ids.len() != 2 {
            continue;
        }

        let edge_id =
            topo.edge_id_from_index(edge_idx)
                .ok_or_else(|| OffsetError::InvalidInput {
                    reason: format!("edge index {edge_idx} not found in arena"),
                })?;

        let class = classify_edge(topo, edge_id, face_ids[0], face_ids[1], tol_angle)?;

        data.edge_class.insert(edge_idx, class);

        // Record edge class for start and end vertices
        let edge_data = topo.edge(edge_id)?;
        vertex_edges
            .entry(edge_data.start().index())
            .or_default()
            .push(class);
        vertex_edges
            .entry(edge_data.end().index())
            .or_default()
            .push(class);
    }

    // Classify vertices based on incident edge classifications
    for (&vid_idx, classes) in &vertex_edges {
        let has_convex = classes
            .iter()
            .any(|c| matches!(c, EdgeClass::Convex { .. }));
        let has_concave = classes
            .iter()
            .any(|c| matches!(c, EdgeClass::Concave { .. }));

        let vclass = match (has_convex, has_concave) {
            (true, true) => VertexClass::Mixed,
            (false, true) => VertexClass::Concave,
            _ => VertexClass::Convex, // all convex or all tangent
        };
        data.vertex_class.insert(vid_idx, vclass);
    }

    Ok(())
}

/// Classify a single edge based on the dihedral angle between its two faces.
fn classify_edge(
    topo: &Topology,
    edge_id: EdgeId,
    face_a: brepkit_topology::face::FaceId,
    face_b: brepkit_topology::face::FaceId,
    tol_angle: f64,
) -> Result<EdgeClass, OffsetError> {
    let edge = topo.edge(edge_id)?;
    let start = topo.vertex(edge.start())?.point();
    let end = topo.vertex(edge.end())?.point();

    let midpoint = Point3::new(
        (start.x() + end.x()) * 0.5,
        (start.y() + end.y()) * 0.5,
        (start.z() + end.z()) * 0.5,
    );

    let tangent_vec = Vec3::new(
        end.x() - start.x(),
        end.y() - start.y(),
        end.z() - start.z(),
    );
    let len = (tangent_vec.x() * tangent_vec.x()
        + tangent_vec.y() * tangent_vec.y()
        + tangent_vec.z() * tangent_vec.z())
    .sqrt();

    if len * len < ZERO_LENGTH_SQ {
        return Err(OffsetError::AnalysisFailed {
            edge: edge_id,
            reason: "degenerate edge (zero length)".to_string(),
        });
    }

    let n_a = face_outward_normal(topo, face_a, midpoint, edge_id)?;
    let n_b = face_outward_normal(topo, face_b, midpoint, edge_id)?;

    // Angle between the outward normals.
    let dot_normals = (n_a.x() * n_b.x() + n_a.y() * n_b.y() + n_a.z() * n_b.z()).clamp(-1.0, 1.0);
    let cross = n_a.cross(n_b);
    let cross_len = (cross.x() * cross.x() + cross.y() * cross.y() + cross.z() * cross.z()).sqrt();
    let angle_between_normals = cross_len.atan2(dot_normals); // [0, π]

    // Convexity test: sample a point slightly inside each face from
    // the edge midpoint, compute their average, and check whether the
    // average outward normal points away from that interior point.
    //
    // For a convex edge, the material is "below" both face normals,
    // so (n_a + n_b) · (interior_point - midpoint) < 0.
    //
    // To get points inside each face we offset the midpoint along the
    // in-face direction perpendicular to the edge. For face_a this is
    // the component of n_a × tangent that points toward the face interior.
    // We use a small epsilon offset and check the dot product sign.
    let n_avg = Vec3::new(n_a.x() + n_b.x(), n_a.y() + n_b.y(), n_a.z() + n_b.z());
    let n_avg_len_sq = n_avg.x() * n_avg.x() + n_avg.y() * n_avg.y() + n_avg.z() * n_avg.z();

    if angle_between_normals < tol_angle || (PI - angle_between_normals).abs() < tol_angle {
        return Ok(EdgeClass::Tangent);
    }

    if n_avg_len_sq < ANTI_PARALLEL_SQ {
        // Normals are anti-parallel — ambiguous, treat as tangent.
        return Ok(EdgeClass::Tangent);
    }

    // For a convex edge on a closed solid, a point offset from the
    // edge midpoint along the *negative* average normal direction should
    // be inside the solid. Equivalently, the average normal points
    // outward (away from the material) at a convex edge.
    //
    // We verify this by checking that both face normals have a positive
    // component along n_avg. At a convex edge both normals "fan out",
    // so n_a · n_avg > 0 and n_b · n_avg > 0. At a concave edge the
    // normals converge past each other and at least one will have a
    // negative component.
    //
    // Actually: n_a · (n_a + n_b) = 1 + n_a·n_b which is always ≥ 0
    // when angle_between_normals ≤ π/2, and can still be positive up
    // to angle = π. This doesn't discriminate.
    //
    // Correct approach: use the face centroid. Sample the centroid of
    // each face and check which side of the edge it's on relative to
    // the average normal.
    //
    // SIMPLEST CORRECT TEST: at a convex edge, the dot product
    // n_a · n_b is positive when the normals open at less than π/2
    // (acute convex), zero at right angles, and negative when they
    // open past π/2. But ALL box edges are convex regardless of this
    // dot product sign.
    //
    // The actual discriminant: sample a third point. Take the centroid
    // of face_a's outer wire. The vector from edge midpoint to that
    // centroid should have a NEGATIVE dot with n_b for a convex edge
    // (the face interior is on the material side, below n_b).
    let centroid_a = face_centroid(topo, face_a)?;
    let to_centroid = Vec3::new(
        centroid_a.x() - midpoint.x(),
        centroid_a.y() - midpoint.y(),
        centroid_a.z() - midpoint.z(),
    );
    // At a convex edge, face_a's interior (centroid) is on the
    // *material side* of face_b, so to_centroid · n_b < 0.
    let discriminant =
        to_centroid.x() * n_b.x() + to_centroid.y() * n_b.y() + to_centroid.z() * n_b.z();

    if discriminant < 0.0 {
        Ok(EdgeClass::Convex {
            angle: angle_between_normals,
        })
    } else {
        Ok(EdgeClass::Concave {
            angle: angle_between_normals,
        })
    }
}

/// Compute the centroid of a face's outer wire vertices.
fn face_centroid(
    topo: &Topology,
    face_id: brepkit_topology::face::FaceId,
) -> Result<Point3, OffsetError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut count = 0_usize;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let p = topo.vertex(edge.start())?.point();
        sum_x += p.x();
        sum_y += p.y();
        sum_z += p.z();
        count += 1;
    }
    if count == 0 {
        return Err(OffsetError::InvalidInput {
            reason: "face has empty outer wire".into(),
        });
    }
    let n = count as f64;
    Ok(Point3::new(sum_x / n, sum_y / n, sum_z / n))
}

/// Compute the outward-facing surface normal of a face at a given 3D point.
fn face_outward_normal(
    topo: &Topology,
    face_id: brepkit_topology::face::FaceId,
    point: Point3,
    edge_id: EdgeId,
) -> Result<Vec3, OffsetError> {
    let face = topo.face(face_id)?;
    let reversed = face.is_reversed();

    let raw_normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Cylinder(cyl) => {
            let (u, v) = cyl.project_point(point);
            cyl.normal(u, v)
        }
        FaceSurface::Cone(cone) => {
            let (u, v) = cone.project_point(point);
            cone.normal(u, v)
        }
        FaceSurface::Sphere(sph) => {
            let (u, v) = sph.project_point(point);
            sph.normal(u, v)
        }
        FaceSurface::Torus(tor) => {
            let (u, v) = tor.project_point(point);
            tor.normal(u, v)
        }
        FaceSurface::Nurbs(nurbs) => {
            let proj = point_to_surface(point, nurbs, nurbs.domain_u(), nurbs.domain_v());
            nurbs
                .normal(proj.u, proj.v)
                .map_err(|_| OffsetError::AnalysisFailed {
                    edge: edge_id,
                    reason: "NURBS normal evaluation failed".to_string(),
                })?
        }
    };

    if reversed {
        Ok(Vec3::new(-raw_normal.x(), -raw_normal.y(), -raw_normal.z()))
    } else {
        Ok(raw_normal)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::data::{EdgeClass, OffsetData, OffsetOptions, VertexClass};
    use brepkit_topology::Topology;

    fn analyse_primitive(topo: &Topology, solid: SolidId) -> OffsetData {
        let mut data = OffsetData::new(1.0, OffsetOptions::default(), vec![]);
        analyse_edges(topo, solid, &mut data).unwrap();
        data
    }

    #[test]
    fn box_all_edges_convex() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = analyse_primitive(&topo, solid);
        assert_eq!(data.edge_class.len(), 12);
        for class in data.edge_class.values() {
            assert!(matches!(class, EdgeClass::Convex { .. }), "got {class:?}");
        }
    }

    #[test]
    fn box_dihedral_angle_approx_half_pi() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = analyse_primitive(&topo, solid);
        for class in data.edge_class.values() {
            if let EdgeClass::Convex { angle } = class {
                let err = (angle - std::f64::consts::FRAC_PI_2).abs();
                assert!(err < 0.2, "expected approx pi/2, got {angle:.4}");
            }
        }
    }

    #[test]
    fn box_vertices_all_convex() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = analyse_primitive(&topo, solid);
        assert_eq!(data.vertex_class.len(), 8);
        for class in data.vertex_class.values() {
            assert_eq!(*class, VertexClass::Convex);
        }
    }
}
