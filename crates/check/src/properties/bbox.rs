//! Axis-aligned bounding box computation for solids.

use brepkit_math::aabb::Aabb3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::CheckError;
use crate::util::expand_aabb_for_surface;

/// Compute the axis-aligned bounding box of a solid.
///
/// Collects all vertex positions from the outer shell, then expands the
/// box for surface curvature (spheres, cylinders, tori, NURBS).
///
/// # Errors
///
/// Returns an error if any topology entity referenced by the solid is missing,
/// or if the solid has no vertices.
pub fn bounding_box(topo: &Topology, solid: SolidId) -> Result<Aabb3, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    // Collect all vertex positions
    let mut points = Vec::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            points.push(topo.vertex(edge.start())?.point());
            points.push(topo.vertex(edge.end())?.point());
        }
    }

    let mut aabb = Aabb3::try_from_points(points.iter().copied())
        .ok_or_else(|| CheckError::ClassificationFailed("solid has no vertices".into()))?;

    // Expand for surface curvature
    for &fid in shell.faces() {
        if let Ok(face) = topo.face(fid) {
            expand_aabb_for_surface(&mut aabb, face.surface());
        }
    }

    Ok(aabb)
}
