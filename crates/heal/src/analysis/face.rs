//! Face analysis — small faces, degeneracy, wire count.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

use crate::HealError;
use crate::status::Status;

/// Result of analyzing a single face.
#[derive(Debug, Clone)]
pub struct FaceAnalysis {
    /// Whether the face's bounding box diagonal is below tolerance.
    pub is_small: bool,
    /// Diagonal length of the axis-aligned bounding box of the face's vertices.
    pub bbox_diagonal: f64,
    /// Total number of wires (outer + inner).
    pub wire_count: usize,
    /// Whether the face is degenerate (all vertices collapse to a point).
    pub is_degenerate: bool,
    /// Outcome status flags.
    pub status: Status,
}

/// Analyze a face for size, degeneracy, and wire structure.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn analyze_face(
    topo: &Topology,
    face_id: FaceId,
    tolerance: &Tolerance,
) -> Result<FaceAnalysis, HealError> {
    let face = topo.face(face_id)?;
    let wire_count = 1 + face.inner_wires().len();

    // Collect all vertex positions from all wires.
    let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();

    let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut vertex_count = 0usize;

    for wid in &wire_ids {
        let wire = topo.wire(*wid)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for vid in [edge.start(), edge.end()] {
                let p = topo.vertex(vid)?.point();
                min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
                max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
                vertex_count += 1;
            }
        }
    }

    let bbox_diagonal = if vertex_count > 0 {
        (max - min).length()
    } else {
        0.0
    };

    let is_small = bbox_diagonal < tolerance.linear;
    let is_degenerate = is_small && vertex_count > 0;

    let mut status = Status::OK;
    if is_small {
        status = status.merge(Status::DONE1);
    }
    if is_degenerate {
        status = status.merge(Status::DONE2);
    }

    Ok(FaceAnalysis {
        is_small,
        bbox_diagonal,
        wire_count,
        is_degenerate,
        status,
    })
}
