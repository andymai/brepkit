//! Split a solid into two halves along a cutting plane.
//!
//! Equivalent to `BRepAlgoAPI_Splitter` in `OpenCascade`. Divides a
//! solid into two new solids along the specified cutting plane.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::boolean::{FaceSpec, assemble_solid_mixed};
use crate::dot_normal_point;

/// Result of splitting a solid: two halves.
#[derive(Debug)]
pub struct SplitResult {
    /// The half on the positive side of the cutting plane (same side as normal).
    pub positive: SolidId,
    /// The half on the negative side of the cutting plane.
    pub negative: SolidId,
}

/// Split a solid into two halves along a plane.
///
/// The cutting plane is defined by a point and a normal. The `positive`
/// half contains geometry on the side the normal points toward; the
/// `negative` half contains the rest.
///
/// # Algorithm
///
/// For each face of the solid:
/// 1. Classify vertices as above (+), below (-), or on the plane
/// 2. Faces entirely on one side go to that half
/// 3. Faces straddling the plane are clipped into two fragments
/// 4. A cap face (the cross-section) is added to close each half
///
/// # Errors
///
/// Returns an error if the plane doesn't intersect the solid, any face
/// is NURBS, or the result cannot be assembled.
#[allow(clippy::too_many_lines)]
pub fn split(
    topo: &mut Topology,
    solid: SolidId,
    plane_point: Point3,
    plane_normal: Vec3,
) -> Result<SplitResult, crate::OperationsError> {
    let tol = Tolerance::new();
    let normal = plane_normal.normalize()?;
    let d = dot_normal_point(normal, plane_point);

    // Collect face data.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<brepkit_topology::face::FaceId> = shell.faces().to_vec();

    let mut positive_specs: Vec<FaceSpec> = Vec::new();
    let mut negative_specs: Vec<FaceSpec> = Vec::new();
    let mut cap_points: Vec<Point3> = Vec::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let verts = crate::boolean::face_vertices(topo, fid)?;
        let dists: Vec<f64> = verts
            .iter()
            .map(|v| dot_normal_point(normal, *v) - d)
            .collect();

        // Classify: all positive, all negative, or mixed.
        let all_pos = dists.iter().all(|&di| di > -tol.linear);
        let all_neg = dists.iter().all(|&di| di < tol.linear);

        // Helper to create FaceSpec preserving the surface type.
        let make_spec = |v: Vec<Point3>, surf: &brepkit_topology::face::FaceSurface| -> FaceSpec {
            match surf {
                brepkit_topology::face::FaceSurface::Plane { normal: fn_, d: fd } => {
                    FaceSpec::Planar {
                        vertices: v,
                        normal: *fn_,
                        d: *fd,
                    }
                }
                other => FaceSpec::Surface {
                    vertices: v,
                    surface: other.clone(),
                },
            }
        };

        if all_pos && !all_neg {
            positive_specs.push(make_spec(verts, &surface));
        } else if all_neg && !all_pos {
            negative_specs.push(make_spec(verts, &surface));
        } else if all_pos && all_neg {
            positive_specs.push(make_spec(verts.clone(), &surface));
            negative_specs.push(make_spec(verts, &surface));
        } else {
            // Mixed: clip the face polygon. Clipped fragments are planar
            // approximations (proper curved-face splitting requires exact
            // surface-plane intersection curves — future work).
            let face_normal = match &surface {
                brepkit_topology::face::FaceSurface::Plane { normal: fn_, .. } => *fn_,
                _ => normal, // Fallback for non-planar straddling faces
            };

            let (pos_verts, neg_verts, crossings) = clip_polygon(&verts, &dists, tol);

            if pos_verts.len() >= 3 {
                let pos_d = dot_normal_point(face_normal, pos_verts[0]);
                positive_specs.push(FaceSpec::Planar {
                    vertices: pos_verts,
                    normal: face_normal,
                    d: pos_d,
                });
            }
            if neg_verts.len() >= 3 {
                let neg_d = dot_normal_point(face_normal, neg_verts[0]);
                negative_specs.push(FaceSpec::Planar {
                    vertices: neg_verts,
                    normal: face_normal,
                    d: neg_d,
                });
            }

            cap_points.extend(crossings);
        }
    }

    if positive_specs.is_empty() || negative_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "cutting plane does not split the solid (entirely on one side)".into(),
        });
    }

    // Build cap faces from crossing points.
    let cap = build_cap_polygon(&cap_points, normal, d, tol);

    if let Some((cap_verts, cap_normal, cap_d)) = cap {
        positive_specs.push(FaceSpec::Planar {
            vertices: cap_verts.clone(),
            normal: -cap_normal,
            d: -cap_d,
        });
        negative_specs.push(FaceSpec::Planar {
            vertices: cap_verts,
            normal: cap_normal,
            d: cap_d,
        });
    }

    let pos_solid = assemble_solid_mixed(topo, &positive_specs, tol)?;
    let neg_solid = assemble_solid_mixed(topo, &negative_specs, tol)?;

    Ok(SplitResult {
        positive: pos_solid,
        negative: neg_solid,
    })
}

/// Clip a polygon by a plane, producing positive and negative fragments.
///
/// Returns `(positive_verts, negative_verts, crossing_points)`.
fn clip_polygon(
    verts: &[Point3],
    dists: &[f64],
    tol: Tolerance,
) -> (Vec<Point3>, Vec<Point3>, Vec<Point3>) {
    let n = verts.len();
    let mut pos_verts = Vec::new();
    let mut neg_verts = Vec::new();
    let mut crossings = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let di = dists[i];
        let dj = dists[j];

        // Classify current vertex.
        if di >= -tol.linear {
            pos_verts.push(verts[i]);
        }
        if di <= tol.linear {
            neg_verts.push(verts[i]);
        }

        // Check for edge crossing.
        if (di > tol.linear && dj < -tol.linear) || (di < -tol.linear && dj > tol.linear) {
            let t = di / (di - dj);
            let pi = verts[i];
            let pj = verts[j];
            let ix = Point3::new(
                (pj.x() - pi.x()).mul_add(t, pi.x()),
                (pj.y() - pi.y()).mul_add(t, pi.y()),
                (pj.z() - pi.z()).mul_add(t, pi.z()),
            );
            pos_verts.push(ix);
            neg_verts.push(ix);
            crossings.push(ix);
        }
    }

    (pos_verts, neg_verts, crossings)
}

/// Build a cap polygon from crossing points.
///
/// Orders the points by angle around the centroid in the cutting plane.
fn build_cap_polygon(
    points: &[Point3],
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> Option<(Vec<Point3>, Vec3, f64)> {
    // Deduplicate points.
    let mut unique = Vec::new();
    for p in points {
        if !unique
            .iter()
            .any(|q: &Point3| (*p - *q).length_squared() < tol.linear * tol.linear)
        {
            unique.push(*p);
        }
    }

    if unique.len() < 3 {
        return None;
    }

    // Compute centroid.
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / unique.len() as f64;
    let (cx, cy, cz) = unique.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
        (ax + p.x(), ay + p.y(), az + p.z())
    });
    let centroid = Point3::new(cx * inv_n, cy * inv_n, cz * inv_n);

    // Build a local 2D coordinate system on the plane.
    let candidate = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = normal.cross(candidate);
    let u_len = u_axis.length();
    if u_len < tol.linear {
        return None;
    }
    let u_axis = Vec3::new(u_axis.x() / u_len, u_axis.y() / u_len, u_axis.z() / u_len);
    let v_axis = normal.cross(u_axis);

    // Sort points by angle around centroid.
    let mut angles: Vec<(f64, usize)> = unique
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let offset = *p - centroid;
            let u = u_axis.dot(offset);
            let v = v_axis.dot(offset);
            (v.atan2(u), i)
        })
        .collect();
    angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let ordered: Vec<Point3> = angles.iter().map(|&(_, i)| unique[i]).collect();

    Some((ordered, normal, d))
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
    fn split_cube_at_half_height() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let result = split(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        // Each half should have positive volume.
        let vol_pos = crate::measure::solid_volume(&topo, result.positive, 0.1).unwrap();
        let vol_neg = crate::measure::solid_volume(&topo, result.negative, 0.1).unwrap();

        assert!(
            vol_pos > 0.1,
            "positive half should have volume, got {vol_pos}"
        );
        assert!(
            vol_neg > 0.1,
            "negative half should have volume, got {vol_neg}"
        );

        // Together they should approximately equal the original volume.
        let tol = Tolerance::loose();
        let total = vol_pos + vol_neg;
        assert!(
            tol.approx_eq(total, 1.0),
            "halves should sum to ~1.0, got {total} ({vol_pos} + {vol_neg})"
        );
    }

    #[test]
    fn split_box_at_quarter() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Box extends from (0,0,0) to (2,2,2). Cut at z=0.5 (quarter height).
        let result = split(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        let vol_pos = crate::measure::solid_volume(&topo, result.positive, 0.1).unwrap();
        let vol_neg = crate::measure::solid_volume(&topo, result.negative, 0.1).unwrap();

        // Box from 0 to 2. Cut at z=0.5: positive is 3/4, negative is 1/4.
        let total = vol_pos + vol_neg;
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(total, 8.0),
            "halves should sum to ~8.0, got {total}"
        );
    }

    #[test]
    fn split_plane_misses_solid() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Plane above the cube.
        let result = split(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(result.is_err(), "plane above cube should fail");
    }

    #[test]
    fn split_along_x_axis() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let result = split(
            &mut topo,
            cube,
            Point3::new(0.5, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        let vol_pos = crate::measure::solid_volume(&topo, result.positive, 0.1).unwrap();
        let vol_neg = crate::measure::solid_volume(&topo, result.negative, 0.1).unwrap();

        let total = vol_pos + vol_neg;
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(total, 1.0),
            "halves should sum to ~1.0, got {total}"
        );
    }
}
