//! Imprint faces and edges on a solid, and split a solid by faces.
//!
//! [`crate::split::split`] reads a whole plane. These take the tool itself, so
//! a face dipping into a solid marks only what it touches.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// The relative volume an imprint or split may read differently: curved faces
/// read coarsely, and the two readings cut them differently.
const VOLUME_MARGIN: f64 = 1e-3;

/// What to mark a solid with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImprintTool {
    /// A face, of any surface, marking the solid where it meets its faces.
    Face(FaceId),
    /// An edge, marking the faces it runs in.
    Edge(EdgeId),
}

/// A copy of `solid` with its faces split along where `tools` meet them.
///
/// The copy has the volume of `solid`. A mark that cannot separate two regions
/// of a face (it stops inside the face, or joins its outline to a hole just
/// once) leaves the face whole, though the edges of its neighbours may gain
/// vertices. Tools that miss the solid return a plain copy.
///
/// # Errors
///
/// Returns an error if a handle does not resolve, the solid is empty, a face
/// cannot be split along its marks, or the result would not be a valid solid
/// of the same volume.
pub fn imprint_faces(
    topo: &mut Topology,
    solid: SolidId,
    tools: &[ImprintTool],
) -> Result<SolidId, OperationsError> {
    refuse_empty(topo, solid)?;
    let mut faces = Vec::new();
    let mut edges = Vec::new();
    for &tool in tools {
        match tool {
            ImprintTool::Face(face) => {
                topo.face(face)?;
                faces.push(face);
            }
            ImprintTool::Edge(edge) => {
                topo.edge(edge)?;
                edges.push(edge);
            }
        }
    }
    let result = brepkit_algo::gfa::imprint(topo, solid, &faces, &edges)?;
    check_pieces(topo, solid, &[result])?;
    Ok(result)
}

/// `solid` in the pieces `tool_faces` separate it into.
///
/// A piece of a tool strictly inside the solid and bounded all around, by the
/// solid's faces or by other such pieces, is a wall between two regions, and
/// each region comes back as a solid. A tool that walls nothing off imprints
/// instead, as [`imprint_faces`] does, so a face that only dips in returns one
/// solid. Any surface can be a tool, and tools may cross each other.
///
/// # Errors
///
/// Returns an error if a handle does not resolve, the solid is empty, a face
/// cannot be split along its marks, or the pieces would not be valid solids
/// summing to the volume of `solid`.
pub fn split_by_faces(
    topo: &mut Topology,
    solid: SolidId,
    tool_faces: &[FaceId],
) -> Result<Vec<SolidId>, OperationsError> {
    refuse_empty(topo, solid)?;
    for &face in tool_faces {
        topo.face(face)?;
    }
    let pieces = brepkit_algo::gfa::split_by_faces(topo, solid, tool_faces)?;
    check_pieces(topo, solid, &pieces)?;
    Ok(pieces)
}

fn refuse_empty(topo: &Topology, solid: SolidId) -> Result<(), OperationsError> {
    topo.solid(solid)?;
    if topo.is_empty_solid(solid) {
        return Err(OperationsError::InvalidInput {
            reason: "cannot imprint on or split an empty solid".into(),
        });
    }
    Ok(())
}

/// Nothing in an imprint is classified or cut away, so the volume and validity
/// of the pieces decide whether the sections went where they should.
fn check_pieces(
    topo: &Topology,
    solid: SolidId,
    pieces: &[SolidId],
) -> Result<(), OperationsError> {
    for &piece in pieces {
        if !crate::validate::validate_solid(topo, piece)?.is_valid() {
            return Err(OperationsError::NonManifoldResult);
        }
    }
    let bounds = crate::measure::solid_outer_bounds(topo, solid)?;
    let deflection = ((bounds.max - bounds.min).length() * 1e-3).max(Tolerance::new().linear);
    let before = crate::measure::solid_volume(topo, solid, deflection)?;
    let mut after = 0.0;
    for &piece in pieces {
        after += crate::measure::solid_volume(topo, piece, deflection)?;
    }
    if (after - before).abs() > VOLUME_MARGIN * before {
        return Err(OperationsError::InvalidInput {
            reason: format!("the pieces hold a volume of {after}, the solid {before}"),
        });
    }
    Ok(())
}
