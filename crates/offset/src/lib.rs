//! # brepkit-offset
//!
//! Solid offset engine for brepkit.
//!
//! This is layer L2, depending on `brepkit-math`, `brepkit-topology`,
//! and `brepkit-geometry`.
//!
//! # Pipeline
//!
//! The offset algorithm follows a 9-phase pipeline:
//!
//! 1. **Analyse** — classify edges as convex/concave/tangent, derive vertex
//!    classes.
//! 2. **Offset** — construct the offset surface for each face (translate
//!    planes, adjust cylinder radii, etc.).
//! 3. **Intersect 3D** — intersect adjacent offset faces in 3D to find new
//!    edge curves.
//! 4. **Intersect 2D** — intersect offset PCurves in parameter space to find
//!    edge split points.
//! 5. **Split edges** — split original edges at intersection parameters.
//! 6. **Arc joints** — optionally insert rolling-ball arc fillets at convex
//!    edges.
//! 7. **Build loops** — assemble trimmed edges into closed wire loops for each
//!    offset face.
//! 8. **Assemble** — build the final shell and solid from offset faces and
//!    wire loops.
//! 9. **Self-intersection removal** — detect and excise global
//!    self-intersections if enabled.

pub(crate) mod analyse;
pub(crate) mod arc_joint;
pub(crate) mod assemble;
pub(crate) mod data;
pub mod error;
pub(crate) mod image;
pub(crate) mod inter2d;
pub(crate) mod inter3d;
pub(crate) mod loops;
pub(crate) mod offset;
pub(crate) mod self_int;

pub use data::{JointType, OffsetOptions};
pub use error::OffsetError;

use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::data::OffsetData;

/// Offset all faces of a solid by the given signed distance.
///
/// Positive distance offsets outward (enlarges), negative inward (shrinks).
///
/// # Errors
///
/// Returns [`OffsetError`] if the offset collapses the solid, any
/// intersection fails, or the result cannot be assembled into a valid solid.
pub fn offset_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    options: OffsetOptions,
) -> Result<SolidId, OffsetError> {
    if !distance.is_finite() || distance.abs() < options.tolerance.linear {
        return Err(OffsetError::InvalidInput {
            reason: "offset distance must be non-zero and finite".into(),
        });
    }
    if options.joint == JointType::Intersection
        && let Some(result) = image::offset_solid(topo, solid, distance, options.tolerance.linear)?
    {
        return Ok(result);
    }
    offset_by_intersection(topo, solid, distance, &[], options)
}

/// Offset a solid while excluding specific faces, producing a thick
/// (hollowed) solid.
///
/// Excluded faces are left at their original positions, and side walls
/// connect them to the offset faces.
///
/// # Errors
///
/// Returns [`OffsetError`] if the offset collapses the solid, any
/// intersection fails, or the result cannot be assembled into a valid solid.
#[allow(clippy::too_many_lines)]
pub fn thick_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    exclude: &[FaceId],
    options: OffsetOptions,
) -> Result<SolidId, OffsetError> {
    if !distance.is_finite() || distance.abs() < options.tolerance.linear {
        return Err(OffsetError::InvalidInput {
            reason: "offset distance must be non-zero and finite".into(),
        });
    }
    if let Some(result) =
        image::thick_solid(topo, solid, distance, exclude, options.tolerance.linear)?
    {
        return Ok(result);
    }
    offset_by_intersection(topo, solid, distance, exclude, options)
}

/// The thick solid of [`thick_solid`] built exact, as images of the input's
/// faces, or `None` when that does not apply.
///
/// It does not apply when the wall would not keep the input's topology (a
/// face collapses, an edge turns round, loops meet), or to NURBS faces,
/// edges other than lines and circles, open faces that are curved, holed,
/// or touch each other, and solids with cavities.
///
/// # Errors
///
/// Returns [`OffsetError`] if a topology lookup fails.
pub fn thick_solid_by_image(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    exclude: &[FaceId],
    tolerance: f64,
) -> Result<Option<SolidId>, OffsetError> {
    image::thick_solid(topo, solid, distance, exclude, tolerance)
}

/// The phased pipeline: offset faces intersected pairwise and rebuilt into
/// loops.
fn offset_by_intersection(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    exclude: &[FaceId],
    options: OffsetOptions,
) -> Result<SolidId, OffsetError> {
    let mut data = OffsetData::new(distance, options, exclude.to_vec());

    analyse::analyse_edges(topo, solid, &mut data)?;

    offset::build_offset_faces(topo, solid, &mut data)?;

    inter3d::intersect_faces_3d(topo, solid, &mut data)?;

    inter2d::intersect_pcurves_2d(topo, solid, &mut data)?;

    // Edge splitting (phase 5) is integrated into inter2d for now.

    if data.options.joint == JointType::Arc {
        arc_joint::build_arc_joints(topo, &mut data)?;
    }

    loops::build_wire_loops(topo, &mut data)?;

    let result = assemble::assemble_solid(topo, &data)?;

    if data.options.remove_self_intersections {
        return self_int::remove_self_intersections(topo, result);
    }

    Ok(result)
}
