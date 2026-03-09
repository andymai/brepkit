//! Shell (hollow/offset) operation for creating thin-walled solids.
//!
//! Equivalent to `BRepOffsetAPI_MakeThickSolid` in `OpenCascade`.
//! Offsets faces of a solid inward to create a hollow shell with
//! uniform wall thickness. Optionally removes specified faces to
//! create openings.

use std::collections::HashSet;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::boolean::{FaceSpec, assemble_solid_mixed};
use crate::dot_normal_point;

/// Create a hollow shell from a solid by offsetting faces inward.
///
/// Each face is offset inward by `thickness` along its outward normal.
/// Supports planar, NURBS, and analytic surface faces.
/// If `open_faces` is non-empty, those faces are removed from both the
/// outer and inner shells, creating openings.
///
/// # Errors
///
/// Returns an error if:
/// - `thickness` is non-positive
/// - Any face in `open_faces` is not part of the solid
/// - Face offset fails (e.g., negative radius for curved surfaces)
/// - The resulting shell is degenerate
#[allow(clippy::too_many_lines)]
pub fn shell(
    topo: &mut Topology,
    solid: SolidId,
    thickness: f64,
    open_faces: &[FaceId],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if thickness <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("shell thickness must be positive, got {thickness}"),
        });
    }

    let solid_data = topo.solid(solid)?;
    let shell_data = topo.shell(solid_data.outer_shell())?;
    let all_face_ids: Vec<FaceId> = shell_data.faces().to_vec();

    let open_set: HashSet<usize> = open_faces.iter().map(|f| f.index()).collect();

    // Validate open_faces belong to the solid.
    let solid_face_set: HashSet<usize> = all_face_ids.iter().map(|f| f.index()).collect();
    for &of in open_faces {
        if !solid_face_set.contains(&of.index()) {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!("face {} is not part of the solid", of.index()),
            });
        }
    }

    // Collect face vertex data (samples curved edges for proper polygons).
    let mut face_verts: Vec<(FaceId, Vec<Point3>)> = Vec::new();
    for &fid in &all_face_ids {
        let verts = crate::boolean::face_polygon(topo, fid)?;
        face_verts.push((fid, verts));
    }

    let mut result_specs: Vec<FaceSpec> = Vec::new();

    // Outer faces: keep the original faces that are not open.
    for &(fid, ref verts) in &face_verts {
        if open_set.contains(&fid.index()) {
            continue;
        }
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                result_specs.push(FaceSpec::Planar {
                    vertices: verts.clone(),
                    normal: *normal,
                    d: *d,
                });
            }
            other => {
                result_specs.push(FaceSpec::Surface {
                    vertices: verts.clone(),
                    surface: other.clone(),
                });
            }
        }
    }

    // Inner faces: offset each non-open face inward using offset_face.
    for &(fid, ref _verts) in &face_verts {
        if open_set.contains(&fid.index()) {
            continue;
        }

        // Use offset_face for the inner surface (handles all surface types).
        let inner_fid = crate::offset_face::offset_face(topo, fid, -thickness, 8)?;
        let inner_face = topo.face(inner_fid)?;

        // Get inner vertices (reversed winding for inward-facing normal).
        let inner_verts: Vec<Point3> = {
            let inner_wire = topo.wire(inner_face.outer_wire())?;
            let mut pts = Vec::new();
            for oe in inner_wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let vid = if oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                pts.push(topo.vertex(vid)?.point());
            }
            pts.into_iter().rev().collect()
        };

        match inner_face.surface() {
            FaceSurface::Plane { normal, .. } => {
                let inner_normal = -*normal;
                let inner_d = dot_normal_point(inner_normal, inner_verts[0]);
                result_specs.push(FaceSpec::Planar {
                    vertices: inner_verts,
                    normal: inner_normal,
                    d: inner_d,
                });
            }
            other => {
                result_specs.push(FaceSpec::Surface {
                    vertices: inner_verts,
                    surface: other.clone(),
                });
            }
        }
    }

    // Compute approximate solid center from face vertices for rim orientation.
    let solid_center = {
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        let mut count = 0.0;
        for (_, verts) in &face_verts {
            for v in verts {
                cx += v.x();
                cy += v.y();
                cz += v.z();
                count += 1.0;
            }
        }
        if count > 0.0 {
            Point3::new(cx / count, cy / count, cz / count)
        } else {
            Point3::new(0.0, 0.0, 0.0)
        }
    };

    // Rim faces: for each edge of an open face, connect outer to inner.
    for (fid, verts) in &face_verts {
        if !open_set.contains(&fid.index()) {
            continue;
        }

        let face = topo.face(*fid)?;
        let normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            FaceSurface::Cylinder(cyl) => cyl.axis(),
            FaceSurface::Cone(cone) => cone.axis(),
            FaceSurface::Sphere(sph) => {
                // Use direction from sphere center to face centroid as normal proxy.
                let centroid = verts.iter().fold(Vec3::new(0.0, 0.0, 0.0), |acc, v| {
                    acc + Vec3::new(v.x(), v.y(), v.z())
                });
                #[allow(clippy::cast_precision_loss)]
                let inv = 1.0 / verts.len() as f64;
                let centroid_pt =
                    Point3::new(centroid.x() * inv, centroid.y() * inv, centroid.z() * inv);
                (centroid_pt - sph.center())
                    .normalize()
                    .unwrap_or(Vec3::new(0.0, 0.0, 1.0))
            }
            FaceSurface::Torus(tor) => tor.z_axis(),
            FaceSurface::Nurbs(nurbs) => {
                // Evaluate surface normal at domain center.
                let (u0, u1) = nurbs.domain_u();
                let (v0, v1) = nurbs.domain_v();
                let mid_u = (u0 + u1) * 0.5;
                let mid_v = (v0 + v1) * 0.5;
                nurbs
                    .normal(mid_u, mid_v)
                    .and_then(Vec3::normalize)
                    .unwrap_or(Vec3::new(0.0, 0.0, 1.0))
            }
        };

        let n = verts.len();
        let offset = Vec3::new(
            -normal.x() * thickness,
            -normal.y() * thickness,
            -normal.z() * thickness,
        );

        for i in 0..n {
            let j = (i + 1) % n;
            let outer_a = verts[i];
            let outer_b = verts[j];
            let inner_a = outer_a + offset;
            let inner_b = outer_b + offset;

            let rim_verts = vec![outer_a, outer_b, inner_b, inner_a];
            let edge1 = outer_b - outer_a;
            let edge2 = inner_a - outer_a;
            let mut rim_normal = edge1
                .cross(edge2)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));

            // Ensure rim normal points outward (away from solid center).
            let rim_center = Point3::new(
                (outer_a.x() + outer_b.x() + inner_a.x() + inner_b.x()) / 4.0,
                (outer_a.y() + outer_b.y() + inner_a.y() + inner_b.y()) / 4.0,
                (outer_a.z() + outer_b.z() + inner_a.z() + inner_b.z()) / 4.0,
            );
            let to_center = solid_center - rim_center;
            if rim_normal.dot(to_center) > 0.0 {
                rim_normal = -rim_normal;
            }

            let rim_d = dot_normal_point(rim_normal, outer_a);
            result_specs.push(FaceSpec::Planar {
                vertices: rim_verts,
                normal: rim_normal,
                d: rim_d,
            });
        }
    }

    if result_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "shell operation produced no faces".into(),
        });
    }

    assemble_solid_mixed(topo, &result_specs, tol)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    /// Helper: get face IDs matching a given normal direction.
    fn find_faces_by_normal(topo: &Topology, solid: SolidId, target_normal: Vec3) -> Vec<FaceId> {
        let tol = Tolerance::loose();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let mut result = Vec::new();
        for &fid in sh.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = f.surface() {
                if tol.approx_eq(normal.x(), target_normal.x())
                    && tol.approx_eq(normal.y(), target_normal.y())
                    && tol.approx_eq(normal.z(), target_normal.z())
                {
                    result.push(fid);
                }
            }
        }
        result
    }

    #[test]
    fn shell_closed_box() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Shell with no open faces: creates a solid shell.
        let result = shell(&mut topo, cube, 0.1, &[]).unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 6 outer + 6 inner = 12 faces (no rim faces since no openings).
        assert_eq!(sh.faces().len(), 12, "closed shell should have 12 faces");
    }

    #[test]
    fn shell_open_top() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Find the top face (+Z normal).
        let top_faces = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(top_faces.len(), 1, "should find exactly one +Z face");

        let result = shell(&mut topo, cube, 0.1, &top_faces).unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 5 outer + 5 inner + 4 rim = 14 faces
        assert_eq!(sh.faces().len(), 14, "open-top shell should have 14 faces");
    }

    #[test]
    fn shell_volume_decrease() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let original_vol = crate::measure::solid_volume(&topo, cube, 0.1).unwrap();

        let result = shell(&mut topo, cube, 0.1, &[]).unwrap();
        let shell_vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

        // The shelled solid should have less volume than the original
        // (we removed the interior).
        assert!(
            shell_vol < original_vol,
            "shell volume ({shell_vol}) should be less than original ({original_vol})"
        );
        assert!(
            shell_vol > 0.0,
            "shell volume should be positive, got {shell_vol}"
        );
    }

    #[test]
    fn shell_zero_thickness_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(shell(&mut topo, cube, 0.0, &[]).is_err());
    }

    #[test]
    fn shell_negative_thickness_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(shell(&mut topo, cube, -0.1, &[]).is_err());
    }

    #[test]
    fn shell_two_open_faces_volume() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Find both +Z and -Z faces
        let top = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
        let bot = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, -1.0));
        let mut open_faces = top;
        open_faces.extend(bot);
        assert_eq!(open_faces.len(), 2);

        let result = shell(&mut topo, cube, 0.1, &open_faces).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.01).unwrap();
        // Expected: 1.0 - 0.8*0.8*1.0 = 0.36
        assert!(vol > 0.1, "tube shell volume should be positive, got {vol}");
        assert!(
            vol < 1.0,
            "tube shell volume should be < original 1.0, got {vol}"
        );
    }
}
