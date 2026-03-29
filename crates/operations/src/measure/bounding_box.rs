//! Bounding box computation for B-rep solids.

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use super::helpers::collect_solid_vertex_points;

/// Compute the axis-aligned bounding box of a solid.
///
/// For planar solids, uses vertex positions (exact). For solids with
/// analytic surfaces (sphere, cylinder, cone, torus), expands the AABB
/// to include surface extremes. For NURBS, uses control-point hulls.
///
/// # Errors
///
/// Returns an error if the solid has no vertices or a topology lookup fails.
pub fn solid_bounding_box(
    topo: &Topology,
    solid: SolidId,
) -> Result<Aabb3, crate::OperationsError> {
    let points = collect_solid_vertex_points(topo, solid)?;
    let mut aabb = Aabb3::try_from_points(points.iter().copied()).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "solid has no vertices".into(),
        }
    })?;

    // Expand AABB for analytic surfaces whose extremes lie beyond vertices.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    for &fid in shell.faces() {
        if let Ok(face) = topo.face(fid) {
            expand_aabb_for_surface(&mut aabb, face.surface());
        }
    }

    Ok(aabb)
}

/// Expand an AABB to include a point.
fn aabb_include(aabb: &mut Aabb3, p: Point3) {
    *aabb = aabb.union(Aabb3 { min: p, max: p });
}

/// Expand an AABB to include surface-specific extremes that vertices miss.
pub fn expand_aabb_for_surface(aabb: &mut Aabb3, surface: &FaceSurface) {
    match surface {
        FaceSurface::Sphere(s) => {
            let c = s.center();
            let r = s.radius();
            aabb_include(aabb, Point3::new(c.x() - r, c.y() - r, c.z() - r));
            aabb_include(aabb, Point3::new(c.x() + r, c.y() + r, c.z() + r));
        }
        FaceSurface::Cylinder(c) => {
            let origin = c.origin();
            let axis = c.axis();
            let r = c.radius();
            // Expand by +/-r only in radial directions (perpendicular to the
            // cylinder axis). The axial extent is already covered by vertices.
            // For each world axis i, the maximum radial reach is r * sqrt(1 - axis_i^2).
            let rx = r * (1.0 - axis.x() * axis.x()).max(0.0).sqrt();
            let ry = r * (1.0 - axis.y() * axis.y()).max(0.0).sqrt();
            let rz = r * (1.0 - axis.z() * axis.z()).max(0.0).sqrt();
            for corner in [aabb.min, aabb.max] {
                let rel = brepkit_math::vec::Vec3::new(
                    corner.x() - origin.x(),
                    corner.y() - origin.y(),
                    corner.z() - origin.z(),
                );
                let t = axis.dot(rel);
                let coa = Point3::new(
                    origin.x() + axis.x() * t,
                    origin.y() + axis.y() * t,
                    origin.z() + axis.z() * t,
                );
                aabb_include(aabb, Point3::new(coa.x() - rx, coa.y() - ry, coa.z() - rz));
                aabb_include(aabb, Point3::new(coa.x() + rx, coa.y() + ry, coa.z() + rz));
            }
        }
        FaceSurface::Torus(t) => {
            // Use the torus's actual axis to compute correct AABB extents.
            let c = t.center();
            let outer_r = t.major_radius() + t.minor_radius();
            let axis = t.z_axis();
            // Axial extent: minor_radius along the torus axis.
            let axial_offset = brepkit_math::vec::Vec3::new(
                axis.x() * t.minor_radius(),
                axis.y() * t.minor_radius(),
                axis.z() * t.minor_radius(),
            );
            // Radial extent: outer_r in the equatorial plane (perpendicular to axis).
            // Worst case for each AABB axis is outer_r * sqrt(1 - axis_component^2),
            // but conservatively use outer_r for all axes.
            aabb_include(
                aabb,
                Point3::new(
                    c.x() - outer_r + axial_offset.x().min(0.0),
                    c.y() - outer_r + axial_offset.y().min(0.0),
                    c.z() - outer_r + axial_offset.z().min(0.0),
                ),
            );
            aabb_include(
                aabb,
                Point3::new(
                    c.x() + outer_r + axial_offset.x().max(0.0),
                    c.y() + outer_r + axial_offset.y().max(0.0),
                    c.z() + outer_r + axial_offset.z().max(0.0),
                ),
            );
        }
        FaceSurface::Nurbs(nurbs) => {
            // Evaluate the actual surface at a parameter grid instead of using
            // control points. NURBS control polyhedra can overshoot the surface,
            // especially for fillet blend surfaces where interpolation fitting
            // pushes control points beyond the data extent.
            let (u_min, u_max) = nurbs.domain_u();
            let (v_min, v_max) = nurbs.domain_v();
            let n_samples = 8;
            #[allow(clippy::cast_precision_loss)]
            for iu in 0..=n_samples {
                let u = u_min + (u_max - u_min) * (iu as f64) / (n_samples as f64);
                for iv in 0..=n_samples {
                    let v = v_min + (v_max - v_min) * (iv as f64) / (n_samples as f64);
                    aabb_include(aabb, nurbs.evaluate(u, v));
                }
            }
        }
        FaceSurface::Plane { .. } | FaceSurface::Cone(_) => {}
    }
}
