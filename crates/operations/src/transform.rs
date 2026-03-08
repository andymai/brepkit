//! Affine transforms applied to topological shapes.

use std::collections::HashSet;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::WireId;

/// Apply an affine transform to a solid, modifying vertex positions and
/// face surface geometry in place.
///
/// The transform matrix must be non-degenerate (non-zero determinant).
/// All unique vertices reachable from the solid's shells are transformed,
/// NURBS edge curves and face surfaces have their control points updated,
/// and all planar face normals are updated using the inverse transpose.
///
/// # Errors
///
/// Returns an error if the matrix is degenerate or a referenced entity is missing.
#[allow(clippy::too_many_lines)]
pub fn transform_solid(
    topo: &mut Topology,
    solid: SolidId,
    matrix: &Mat4,
) -> Result<(), crate::OperationsError> {
    let tol = Tolerance::new();
    if tol.approx_eq(matrix.determinant(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "transform matrix is degenerate (zero determinant)".into(),
        });
    }

    // Collect all unique vertex IDs, edge IDs, and face IDs in a read phase.
    let (vertex_ids, edge_ids, face_ids) = collect_solid_entities(topo, solid)?;

    // Mutate phase 1: transform each vertex.
    for vid in vertex_ids {
        let vertex = topo.vertex_mut(vid)?;
        let new_point = matrix.mul_point(vertex.point());
        vertex.set_point(new_point);
    }

    // Mutate phase 2: transform edge curves (NURBS, Circle, Ellipse).
    transform_edges(topo, &edge_ids, matrix)?;

    // Mutate phase 3: transform face surface geometry.
    // For plane normals, use the inverse transpose: n' = (M⁻¹)ᵀ · n
    let normal_matrix = matrix.inverse()?.transpose();

    for fid in face_ids {
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { normal, .. } => {
                let n = *normal;
                // Transform the normal via the inverse transpose (treating it as
                // a direction, so we use mul_point on a point at (nx, ny, nz)
                // and subtract the translation component).
                let transformed =
                    normal_matrix.mul_point(brepkit_math::vec::Point3::new(n.x(), n.y(), n.z()));
                // Extract direction only (ignore any translation component from
                // the inverse transpose by subtracting the origin transform).
                let origin = normal_matrix.mul_point(brepkit_math::vec::Point3::new(0.0, 0.0, 0.0));
                let raw = Vec3::new(
                    transformed.x() - origin.x(),
                    transformed.y() - origin.y(),
                    transformed.z() - origin.z(),
                );
                let new_normal = raw.normalize()?;

                // Recompute d from a transformed vertex on this face. We use
                // the first vertex of the outer wire.
                let wire = topo.wire(face.outer_wire())?;
                let first_oe = &wire.edges()[0];
                let edge = topo.edge(first_oe.edge())?;
                let ref_vid = if first_oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                let ref_point = topo.vertex(ref_vid)?.point();
                let new_d = new_normal.dot(Vec3::new(ref_point.x(), ref_point.y(), ref_point.z()));

                // Now mutate the face.
                let face_mut = topo.face_mut(fid)?;
                face_mut.set_surface(FaceSurface::Plane {
                    normal: new_normal,
                    d: new_d,
                });
            }
            FaceSurface::Nurbs(s) => {
                let new_control_points: Vec<Vec<_>> = s
                    .control_points()
                    .iter()
                    .map(|row| row.iter().map(|pt| matrix.mul_point(*pt)).collect())
                    .collect();
                let new_surface = NurbsSurface::new(
                    s.degree_u(),
                    s.degree_v(),
                    s.knots_u().to_vec(),
                    s.knots_v().to_vec(),
                    new_control_points,
                    s.weights().to_vec(),
                );
                topo.face_mut(fid)?
                    .set_surface(FaceSurface::Nurbs(new_surface?));
            }
            FaceSurface::Cylinder(cyl) => {
                let new_origin = matrix.mul_point(cyl.origin());
                let new_axis = transform_direction(matrix, cyl.axis())?;
                // Scale radius: measure how the matrix scales a direction perpendicular to axis
                let new_radius = scaled_radius(matrix, cyl.axis(), cyl.radius());
                let new_cyl = brepkit_math::surfaces::CylindricalSurface::new(
                    new_origin, new_axis, new_radius,
                )?;
                topo.face_mut(fid)?
                    .set_surface(FaceSurface::Cylinder(new_cyl));
            }
            FaceSurface::Cone(cone) => {
                let new_apex = matrix.mul_point(cone.apex());
                let new_axis = transform_direction(matrix, cone.axis())?;
                let new_cone = brepkit_math::surfaces::ConicalSurface::new(
                    new_apex,
                    new_axis,
                    cone.half_angle(),
                )?;
                topo.face_mut(fid)?.set_surface(FaceSurface::Cone(new_cone));
            }
            FaceSurface::Sphere(sph) => {
                if is_uniform_scale(matrix) {
                    let new_center = matrix.mul_point(sph.center());
                    // Extract uniform scale factor from column magnitudes
                    let m = &matrix.0;
                    let sx = (m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0]).sqrt();
                    let new_sph = brepkit_math::surfaces::SphericalSurface::new(
                        new_center,
                        sph.radius() * sx,
                    )?;
                    topo.face_mut(fid)?
                        .set_surface(FaceSurface::Sphere(new_sph));
                } else {
                    // Non-uniform scale: sample the face's v-range of the
                    // sphere and refit as NURBS.
                    let (v_min, v_max) = sphere_face_v_range(topo, fid, sph)?;
                    let sph_clone = sph.clone();
                    let nurbs = sphere_to_transformed_nurbs(&sph_clone, matrix, v_min, v_max)?;
                    topo.face_mut(fid)?.set_surface(FaceSurface::Nurbs(nurbs));
                }
            }
            FaceSurface::Torus(tor) => {
                let new_center = matrix.mul_point(tor.center());
                let m = &matrix.0;
                let sx = (m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0]).sqrt();
                let new_tor = brepkit_math::surfaces::ToroidalSurface::new(
                    new_center,
                    tor.major_radius() * sx,
                    tor.minor_radius() * sx,
                )?;
                topo.face_mut(fid)?.set_surface(FaceSurface::Torus(new_tor));
            }
        }
    }

    Ok(())
}

/// Determine the v-range (latitude) of a sphere face from its boundary.
///
/// Projects boundary vertices onto the sphere to find their latitudes,
/// then uses the sign of the average vertex Z offset from center to
/// determine which hemisphere the face covers.
fn sphere_face_v_range(
    topo: &Topology,
    face_id: FaceId,
    sph: &brepkit_math::surfaces::SphericalSurface,
) -> Result<(f64, f64), crate::OperationsError> {
    use std::f64::consts::FRAC_PI_2;

    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut v_vals = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let pt = topo.vertex(edge.start())?.point();
        let (_u, v) = sph.project_point(pt);
        v_vals.push(v);
    }

    if v_vals.is_empty() {
        // Full sphere with no boundary → full range
        return Ok((-FRAC_PI_2, FRAC_PI_2));
    }

    // All boundary vertices should be at roughly the same v (equator).
    // Determine hemisphere by checking whether face is above or below boundary.
    let boundary_v = v_vals.iter().copied().sum::<f64>() / v_vals.len() as f64;

    // Check which side: sample a face interior point. A simpler heuristic:
    // if any inner wire exists, check it. Otherwise, examine the face's
    // Newell normal direction relative to the sphere center.
    //
    // For brepkit's make_sphere: south hemisphere has normals pointing
    // away from center with v ∈ [-π/2, boundary_v], north hemisphere
    // v ∈ [boundary_v, π/2].
    //
    // Use a heuristic: compute the average Z of boundary relative to center
    // and compare with the face's position hints.
    let center = sph.center();
    let avg_boundary_z: f64 = {
        let mut sum = 0.0;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let pt = topo.vertex(edge.start())?.point();
            sum += pt.z() - center.z();
        }
        sum / wire.edges().len() as f64
    };

    // If the boundary is near the equator (avg_z ≈ 0), we need another way.
    // Try to detect hemisphere by checking if the face has a pole vertex
    // (a degenerate edge with a pole at v = ±π/2).
    // Simpler approach: this is called before the transform, and make_sphere
    // creates two faces. Just check if boundary_v ≈ 0 and pick hemispheres.
    if boundary_v.abs() < 0.1 {
        // Near equator: use face ordering. Check if this face has vertices
        // near the north pole (z > center.z) or south pole (z < center.z).
        // If avg_boundary_z is near 0, look for a degenerate pole vertex.
        let mut has_pole_north = false;
        let mut has_pole_south = false;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            if edge.start() == edge.end() {
                let pt = topo.vertex(edge.start())?.point();
                let dz = pt.z() - center.z();
                if dz > 0.0 {
                    has_pole_north = true;
                } else {
                    has_pole_south = true;
                }
            }
        }
        if has_pole_north {
            return Ok((boundary_v, FRAC_PI_2));
        }
        if has_pole_south {
            return Ok((-FRAC_PI_2, boundary_v));
        }
        // Default: use the winding direction. If first edge goes "forward" in
        // parameter space, it's the north hemisphere.
        // Fallback: just check avg Z of all edge midpoints would require
        // curve evaluation. Use a simpler heuristic based on face ordering.
        // The first face in make_sphere is south, second is north.
        // This is fragile, but works for this specific case.
        if avg_boundary_z >= 0.0 {
            return Ok((boundary_v, FRAC_PI_2));
        }
        return Ok((-FRAC_PI_2, boundary_v));
    }

    if boundary_v > 0.0 {
        Ok((boundary_v, FRAC_PI_2))
    } else {
        Ok((-FRAC_PI_2, boundary_v))
    }
}

/// Check whether a transform matrix has uniform scaling (all axis scale
/// factors are approximately equal). Non-uniform scaling distorts spheres
/// into ellipsoids, so analytic representations must be converted to NURBS.
/// Compute the scaled radius of a circle perpendicular to `axis` after transform.
fn scaled_radius(matrix: &Mat4, axis: Vec3, radius: f64) -> f64 {
    // Pick a direction perpendicular to the axis
    let perp = if axis.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
            .cross(axis)
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0))
    } else {
        Vec3::new(0.0, 1.0, 0.0)
            .cross(axis)
            .normalize()
            .unwrap_or(Vec3::new(0.0, 1.0, 0.0))
    };
    // Transform the perpendicular direction and measure its length
    let origin = brepkit_math::vec::Point3::new(0.0, 0.0, 0.0);
    let end =
        brepkit_math::vec::Point3::new(perp.x() * radius, perp.y() * radius, perp.z() * radius);
    let t_origin = matrix.mul_point(origin);
    let t_end = matrix.mul_point(end);
    let diff = t_end - t_origin;
    diff.length()
}

fn is_uniform_scale(matrix: &Mat4) -> bool {
    let m = &matrix.0;
    // Column vector magnitudes of the upper-left 3×3
    let sx = (m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0]).sqrt();
    let sy = (m[0][1] * m[0][1] + m[1][1] * m[1][1] + m[2][1] * m[2][1]).sqrt();
    let sz = (m[0][2] * m[0][2] + m[1][2] * m[1][2] + m[2][2] * m[2][2]).sqrt();
    let avg = (sx + sy + sz) / 3.0;
    let rel = 0.01; // 1% tolerance
    (sx - avg).abs() < avg * rel && (sy - avg).abs() < avg * rel && (sz - avg).abs() < avg * rel
}

/// Sample a spherical surface over a given v-range, transform the points
/// with a matrix, and refit as a NURBS surface. This preserves the correct
/// geometry when a non-uniform scale is applied (sphere → ellipsoid).
#[allow(clippy::cast_precision_loss)]
fn sphere_to_transformed_nurbs(
    sph: &brepkit_math::surfaces::SphericalSurface,
    matrix: &Mat4,
    v_min: f64,
    v_max: f64,
) -> Result<NurbsSurface, crate::OperationsError> {
    use std::f64::consts::TAU;

    let n_u = 33; // Longitude samples (0 to 2π)
    let n_v = 17; // Latitude samples

    let mut rows: Vec<Vec<brepkit_math::vec::Point3>> = Vec::with_capacity(n_v);
    for iv in 0..n_v {
        let v = v_min + (v_max - v_min) * (iv as f64) / ((n_v - 1) as f64);
        let mut row = Vec::with_capacity(n_u);
        for iu in 0..n_u {
            let u = TAU * (iu as f64) / ((n_u - 1) as f64);
            let pt = sph.evaluate(u, v);
            row.push(matrix.mul_point(pt));
        }
        rows.push(row);
    }

    let nurbs = brepkit_math::nurbs::surface_fitting::interpolate_surface(&rows, 3, 3)?;
    Ok(nurbs)
}

/// Transforms a direction vector by applying the matrix and subtracting the
/// translation component, then normalizing.
fn transform_direction(matrix: &Mat4, dir: Vec3) -> Result<Vec3, crate::OperationsError> {
    let origin = matrix.mul_point(brepkit_math::vec::Point3::new(0.0, 0.0, 0.0));
    let tip = matrix.mul_point(brepkit_math::vec::Point3::new(dir.x(), dir.y(), dir.z()));
    let raw = Vec3::new(
        tip.x() - origin.x(),
        tip.y() - origin.y(),
        tip.z() - origin.z(),
    );
    Ok(raw.normalize()?)
}

/// Transform a set of edge curves in place.
///
/// Line edges need no update — their geometry is defined by vertices.
#[allow(clippy::too_many_lines)]
fn transform_edges(
    topo: &mut Topology,
    edge_ids: &HashSet<EdgeId>,
    matrix: &Mat4,
) -> Result<(), crate::OperationsError> {
    let origin = matrix.mul_point(brepkit_math::vec::Point3::new(0.0, 0.0, 0.0));
    let transform_dir = |d: Vec3| -> Vec3 {
        matrix.mul_point(brepkit_math::vec::Point3::new(d.x(), d.y(), d.z())) - origin
    };
    for &eid in edge_ids {
        let edge = topo.edge(eid)?;
        let new_curve = match edge.curve() {
            EdgeCurve::Line => None,
            EdgeCurve::NurbsCurve(c) => {
                let new_control_points: Vec<_> = c
                    .control_points()
                    .iter()
                    .map(|pt| matrix.mul_point(*pt))
                    .collect();
                Some(EdgeCurve::NurbsCurve(NurbsCurve::new(
                    c.degree(),
                    c.knots().to_vec(),
                    new_control_points,
                    c.weights().to_vec(),
                )?))
            }
            EdgeCurve::Circle(c) => {
                let new_center = matrix.mul_point(c.center());
                let new_u = transform_dir(c.u_axis());
                let new_v = transform_dir(c.v_axis());
                let su = new_u.length();
                let sv = new_v.length();
                let new_normal = new_u.cross(new_v).normalize()?;
                if (su - sv).abs() < 1e-12 * su.max(sv).max(1.0) {
                    Some(EdgeCurve::Circle(
                        brepkit_math::curves::Circle3D::with_axes(
                            new_center,
                            new_normal,
                            c.radius() * su,
                            new_u.normalize()?,
                            new_v.normalize()?,
                        )?,
                    ))
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
                    Some(EdgeCurve::Ellipse(
                        brepkit_math::curves::Ellipse3D::with_axes(
                            new_center, new_normal, semi_major, semi_minor, u_dir, v_dir,
                        )?,
                    ))
                }
            }
            EdgeCurve::Ellipse(e) => {
                let new_center = matrix.mul_point(e.center());
                let new_u = transform_dir(e.u_axis());
                let new_v = transform_dir(e.v_axis());
                let new_normal = new_u.cross(new_v).normalize()?;
                Some(EdgeCurve::Ellipse(
                    brepkit_math::curves::Ellipse3D::with_axes(
                        new_center,
                        new_normal,
                        e.semi_major() * new_u.length(),
                        e.semi_minor() * new_v.length(),
                        new_u.normalize()?,
                        new_v.normalize()?,
                    )?,
                ))
            }
        };
        if let Some(curve) = new_curve {
            topo.edge_mut(eid)?.set_curve(curve);
        }
    }
    Ok(())
}

/// Apply an affine transform to a wire, modifying vertex positions and
/// edge curve geometry in place.
///
/// # Errors
///
/// Returns an error if the matrix is degenerate or a referenced entity is missing.
pub fn transform_wire(
    topo: &mut Topology,
    wire_id: WireId,
    matrix: &Mat4,
) -> Result<(), crate::OperationsError> {
    let tol = Tolerance::new();
    if tol.approx_eq(matrix.determinant(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "transform matrix is degenerate (zero determinant)".into(),
        });
    }

    let (vertex_ids, edge_ids) = collect_wire_entities(topo, wire_id)?;

    // Transform vertices.
    for vid in vertex_ids {
        let vertex = topo.vertex_mut(vid)?;
        let new_point = matrix.mul_point(vertex.point());
        vertex.set_point(new_point);
    }

    // Transform edge curves.
    transform_edges(topo, &edge_ids, matrix)?;

    Ok(())
}

/// Traverses wire → edges → vertices and returns deduplicated sets.
fn collect_wire_entities(
    topo: &Topology,
    wire_id: WireId,
) -> Result<(HashSet<VertexId>, HashSet<EdgeId>), crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let mut edge_ids = HashSet::new();
    let wire = topo.wire(wire_id)?;
    for oe in wire.edges() {
        let eid = oe.edge();
        edge_ids.insert(eid);
        let edge = topo.edge(eid)?;
        vertex_ids.insert(edge.start());
        vertex_ids.insert(edge.end());
    }
    Ok((vertex_ids, edge_ids))
}

/// Traverses solid → shells → faces → wires → edges → vertices and
/// returns deduplicated sets of vertex IDs, edge IDs, and face IDs.
#[allow(clippy::type_complexity)]
fn collect_solid_entities(
    topo: &Topology,
    solid: SolidId,
) -> Result<(HashSet<VertexId>, HashSet<EdgeId>, HashSet<FaceId>), crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let mut edge_ids = HashSet::new();
    let mut face_ids = HashSet::new();
    let solid_data = topo.solid(solid)?;
    let shell_ids: Vec<_> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();

    for shell_id in shell_ids {
        let shell = topo.shell(shell_id)?;
        let fids: Vec<_> = shell.faces().to_vec();

        for face_id in fids {
            face_ids.insert(face_id);
            let face = topo.face(face_id)?;
            let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .collect();

            for wire_id in wire_ids {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let eid = oe.edge();
                    edge_ids.insert(eid);
                    let edge = topo.edge(eid)?;
                    vertex_ids.insert(edge.start());
                    vertex_ids.insert(edge.end());
                }
            }
        }
    }

    Ok((vertex_ids, edge_ids, face_ids))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;

    #[test]
    fn translate_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);
        let matrix = Mat4::translation(1.0, 0.0, 0.0);

        transform_solid(&mut topo, solid, &matrix).unwrap();

        // All vertices should have x shifted by 1.0.
        let tol = Tolerance::new();
        for (_id, v) in topo.vertices.iter() {
            let x = v.point().x();
            assert!(
                tol.approx_eq(x, 1.0) || tol.approx_eq(x, 2.0),
                "unexpected x = {x}"
            );
        }
    }

    #[test]
    fn identity_transform_no_change() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let before: Vec<_> = topo.vertices.iter().map(|(_, v)| v.point()).collect();

        transform_solid(&mut topo, solid, &Mat4::identity()).unwrap();

        let tol = Tolerance::new();
        for (i, (_, v)) in topo.vertices.iter().enumerate() {
            assert!(tol.approx_eq(v.point().x(), before[i].x()));
            assert!(tol.approx_eq(v.point().y(), before[i].y()));
            assert!(tol.approx_eq(v.point().z(), before[i].z()));
        }
    }

    #[test]
    fn degenerate_matrix_error() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);
        let matrix = Mat4::scale(0.0, 1.0, 1.0);

        let result = transform_solid(&mut topo, solid, &matrix);
        assert!(result.is_err());
    }

    /// Rotating a cube 90 degrees around the Z axis should update face normals.
    #[test]
    fn rotation_updates_face_normals() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        // 90-degree rotation around Z: +X face normal → +Y, -X → -Y, etc.
        let matrix = Mat4::rotation_z(std::f64::consts::FRAC_PI_2);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // Collect all plane normals.
        let mut normals: Vec<Vec3> = Vec::new();
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = f.surface() {
                normals.push(*normal);
            }
        }

        // Original cube had normals along ±X, ±Y, ±Z.
        // After 90° Z-rotation: ±X → ±Y, ±Y → ∓X, ±Z unchanged.
        // So we should still have 6 normals, each approximately axis-aligned.
        assert_eq!(normals.len(), 6);

        // Check that we still have a +Z and -Z normal (unchanged by Z rotation).
        let has_pos_z = normals
            .iter()
            .any(|n| tol.approx_eq(n.z(), 1.0) && tol.approx_eq(n.x(), 0.0));
        let has_neg_z = normals
            .iter()
            .any(|n| tol.approx_eq(n.z(), -1.0) && tol.approx_eq(n.x(), 0.0));
        assert!(has_pos_z, "should have +Z normal after Z rotation");
        assert!(has_neg_z, "should have -Z normal after Z rotation");
    }

    // ── Analytic surface transform helpers ───────────────────────────────────

    /// Build a minimal solid containing a single face with the given surface.
    ///
    /// The wire is a unit square in XY; only the face surface type varies.
    fn make_single_face_solid(
        topo: &mut Topology,
        surface: FaceSurface,
    ) -> brepkit_topology::solid::SolidId {
        use brepkit_math::vec::Point3;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::shell::Shell;
        use brepkit_topology::solid::Solid;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let tol = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol));

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
        let fid = topo.faces.alloc(Face::new(wid, vec![], surface));
        let shell = Shell::new(vec![fid]).unwrap();
        let shell_id = topo.shells.alloc(shell);
        topo.solids.alloc(Solid::new(shell_id, vec![]))
    }

    // ── Cylinder surface transform ────────────────────────────────────────────

    #[test]
    fn translate_cylinder_face_updates_origin() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Point3;

        let mut topo = Topology::new();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();
        let solid = make_single_face_solid(&mut topo, FaceSurface::Cylinder(cyl));

        let matrix = Mat4::translation(5.0, 3.0, 1.0);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        // Find the (now-transformed) cylinder face.
        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found = false;
        for &fid in shell.faces() {
            if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
                assert!(
                    tol.approx_eq(c.origin().x(), 5.0),
                    "cylinder origin x should be 5.0, got {}",
                    c.origin().x()
                );
                assert!(
                    tol.approx_eq(c.origin().y(), 3.0),
                    "cylinder origin y should be 3.0, got {}",
                    c.origin().y()
                );
                assert!(
                    tol.approx_eq(c.origin().z(), 1.0),
                    "cylinder origin z should be 1.0, got {}",
                    c.origin().z()
                );
                // The axis (0,0,1) should be unchanged by a pure translation.
                assert!(
                    tol.approx_eq(c.axis().z(), 1.0),
                    "cylinder axis z should still be 1.0"
                );
                assert!(
                    tol.approx_eq(c.radius(), 2.0),
                    "cylinder radius should be unchanged"
                );
                found = true;
            }
        }
        assert!(found, "cylinder face not found after transform");
    }

    #[test]
    fn rotate_cylinder_face_updates_axis() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Point3;

        let mut topo = Topology::new();
        // Cylinder with axis along +Z.
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let solid = make_single_face_solid(&mut topo, FaceSurface::Cylinder(cyl));

        // 90° rotation around Y: Z-axis → X-axis
        let matrix = Mat4::rotation_y(std::f64::consts::FRAC_PI_2);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found = false;
        for &fid in shell.faces() {
            if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
                // After 90° Y rotation, original Z-axis should point along +X.
                assert!(
                    tol.approx_eq(c.axis().x().abs(), 1.0),
                    "cylinder axis should be along X after Y rotation, got {:?}",
                    c.axis()
                );
                found = true;
            }
        }
        assert!(found, "cylinder face not found after rotation");
    }

    // ── Cone surface transform ────────────────────────────────────────────────

    #[test]
    fn translate_cone_face_updates_apex() {
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_math::vec::Point3;

        let mut topo = Topology::new();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let solid = make_single_face_solid(&mut topo, FaceSurface::Cone(cone));

        let matrix = Mat4::translation(2.0, 4.0, 6.0);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found = false;
        for &fid in shell.faces() {
            if let FaceSurface::Cone(c) = topo.face(fid).unwrap().surface() {
                assert!(
                    tol.approx_eq(c.apex().x(), 2.0),
                    "cone apex x should be 2.0, got {}",
                    c.apex().x()
                );
                assert!(
                    tol.approx_eq(c.apex().y(), 4.0),
                    "cone apex y should be 4.0"
                );
                assert!(
                    tol.approx_eq(c.apex().z(), 6.0),
                    "cone apex z should be 6.0"
                );
                // Axis should be unchanged by a translation.
                assert!(
                    tol.approx_eq(c.axis().z(), 1.0),
                    "cone axis z should still be 1.0"
                );
                found = true;
            }
        }
        assert!(found, "cone face not found after transform");
    }

    // ── Sphere surface transform ──────────────────────────────────────────────

    #[test]
    fn translate_sphere_face_updates_center() {
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_math::vec::Point3;

        let mut topo = Topology::new();
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
        let solid = make_single_face_solid(&mut topo, FaceSurface::Sphere(sphere));

        let matrix = Mat4::translation(-1.0, 2.0, 5.0);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found = false;
        for &fid in shell.faces() {
            if let FaceSurface::Sphere(s) = topo.face(fid).unwrap().surface() {
                assert!(
                    tol.approx_eq(s.center().x(), -1.0),
                    "sphere center x should be -1.0"
                );
                assert!(
                    tol.approx_eq(s.center().y(), 2.0),
                    "sphere center y should be 2.0"
                );
                assert!(
                    tol.approx_eq(s.center().z(), 5.0),
                    "sphere center z should be 5.0"
                );
                assert!(
                    tol.approx_eq(s.radius(), 3.0),
                    "sphere radius should be unchanged"
                );
                found = true;
            }
        }
        assert!(found, "sphere face not found after transform");
    }

    // ── Torus surface transform ───────────────────────────────────────────────

    #[test]
    fn translate_torus_face_updates_center() {
        use brepkit_math::surfaces::ToroidalSurface;
        use brepkit_math::vec::Point3;

        let mut topo = Topology::new();
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.5).unwrap();
        let solid = make_single_face_solid(&mut topo, FaceSurface::Torus(torus));

        let matrix = Mat4::translation(10.0, -3.0, 0.5);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found = false;
        for &fid in shell.faces() {
            if let FaceSurface::Torus(t) = topo.face(fid).unwrap().surface() {
                assert!(
                    tol.approx_eq(t.center().x(), 10.0),
                    "torus center x should be 10.0"
                );
                assert!(
                    tol.approx_eq(t.center().y(), -3.0),
                    "torus center y should be -3.0"
                );
                assert!(
                    tol.approx_eq(t.center().z(), 0.5),
                    "torus center z should be 0.5"
                );
                assert!(
                    tol.approx_eq(t.major_radius(), 5.0),
                    "torus major radius should be unchanged"
                );
                assert!(
                    tol.approx_eq(t.minor_radius(), 1.5),
                    "torus minor radius should be unchanged"
                );
                found = true;
            }
        }
        assert!(found, "torus face not found after transform");
    }

    // ── transform_direction degenerate case ───────────────────────────────────

    #[test]
    fn transform_direction_zero_vector_is_error() {
        // A zero direction vector cannot be normalized and must return an error.
        // This exercises the normalize() error branch in transform_direction.
        let result = super::transform_direction(&Mat4::identity(), Vec3::new(0.0, 0.0, 0.0));
        assert!(
            result.is_err(),
            "transform_direction with zero vector should return an error"
        );
    }

    #[test]
    fn transform_direction_unit_z_identity_unchanged() {
        // Identity matrix should leave a unit direction unchanged.
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let result = super::transform_direction(&Mat4::identity(), dir).unwrap();
        let tol = Tolerance::new();
        assert!(tol.approx_eq(result.z(), 1.0), "z should remain 1.0");
        assert!(tol.approx_eq(result.x(), 0.0), "x should remain 0.0");
        assert!(tol.approx_eq(result.y(), 0.0), "y should remain 0.0");
    }

    /// Revolving a face produces NURBS surfaces; translating the result
    /// should move both vertices and NURBS control points.
    #[test]
    fn transform_nurbs_solid() {
        use brepkit_math::vec::Point3;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        // Create a small rectangular face at x=[2,3], z=[0,1], y=0.
        // This is offset from the Y axis so the revolve doesn't self-intersect.
        let tol_val = 1e-10;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 0.0, 0.0), tol_val));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(3.0, 0.0, 0.0), tol_val));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(3.0, 0.0, 1.0), tol_val));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 0.0, 1.0), tol_val));

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

        let normal = brepkit_math::vec::Vec3::new(0.0, -1.0, 0.0);
        let rect = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane { normal, d: 0.0 },
        ));

        // Revolve 90° around the Y axis to produce NURBS faces.
        let solid = crate::revolve::revolve(
            &mut topo,
            rect,
            Point3::new(0.0, 0.0, 0.0),
            brepkit_math::vec::Vec3::new(0.0, 1.0, 0.0),
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap();

        // Record a NURBS surface control point before the transform.
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut original_nurbs_cp = None;
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Nurbs(s) = f.surface() {
                original_nurbs_cp = Some(s.control_points()[0][0]);
                break;
            }
        }
        let original_cp = original_nurbs_cp.unwrap();

        // Translate by (10, 0, 0).
        let matrix = Mat4::translation(10.0, 0.0, 0.0);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        // Verify NURBS control points have shifted.
        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found = false;
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Nurbs(s) = f.surface() {
                let cp = s.control_points()[0][0];
                assert!(
                    tol.approx_eq(cp.x(), original_cp.x() + 10.0),
                    "NURBS control point x should shift by 10, got {} (was {})",
                    cp.x(),
                    original_cp.x()
                );
                assert!(
                    tol.approx_eq(cp.y(), original_cp.y()),
                    "NURBS control point y should be unchanged"
                );
                assert!(
                    tol.approx_eq(cp.z(), original_cp.z()),
                    "NURBS control point z should be unchanged"
                );
                found = true;
                break;
            }
        }
        assert!(found, "should still have NURBS faces after transform");
    }

    #[test]
    fn translate_wire() {
        use brepkit_math::vec::Point3;
        use brepkit_topology::builder::make_polygon_wire;

        let mut topo = Topology::new();
        let wire = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
        )
        .unwrap();

        transform_wire(&mut topo, wire, &Mat4::translation(5.0, 0.0, 0.0)).unwrap();

        // All vertices should have x shifted by 5.
        let w = topo.wire(wire).unwrap();
        for oe in w.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let start_pos = topo.vertex(edge.start()).unwrap().point();
            assert!(
                start_pos.x() >= 4.9,
                "vertex x should be shifted: got {}",
                start_pos.x()
            );
        }
    }

    #[test]
    fn degenerate_matrix_errors_for_wire() {
        use brepkit_math::vec::Point3;
        use brepkit_topology::builder::make_polygon_wire;

        let mut topo = Topology::new();
        let wire = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
        )
        .unwrap();

        let result = transform_wire(&mut topo, wire, &Mat4::scale(0.0, 1.0, 1.0));
        assert!(result.is_err());
    }
}
