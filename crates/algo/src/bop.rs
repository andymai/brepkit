//! Boolean operation selection — filters sub-faces by classification.
//!
//! Given the classified sub-faces from the Builder, selects which faces
//! to keep for each operation type (fuse, cut, intersect).

use crate::builder::{FaceClass, SubFace};
use crate::ds::Rank;

/// Boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Union: keep exterior faces from both solids.
    Fuse,
    /// Subtraction: keep exterior of A and interior of B (reversed).
    Cut,
    /// Intersection: keep interior faces from both solids.
    Intersect,
}

/// Select sub-faces to keep based on the boolean operation type.
///
/// The truth table:
/// - **Fuse**: A-Outside + B-Outside + CoplanarSame from either
/// - **Cut**: A-Outside + B-Inside (reversed normals) + CoplanarOpposite from B
/// - **Intersect**: A-Inside + B-Inside + CoplanarSame from either
#[must_use]
pub(crate) fn select_faces(sub_faces: &[SubFace], op: BooleanOp) -> Vec<SelectedFace> {
    sub_faces
        .iter()
        .filter_map(|sf| {
            let keep = match op {
                BooleanOp::Fuse => matches!(
                    (&sf.rank, &sf.classification),
                    (Rank::A | Rank::B, FaceClass::Outside) | (_, FaceClass::CoplanarSame)
                ),
                BooleanOp::Cut => matches!(
                    (&sf.rank, &sf.classification),
                    (Rank::A, FaceClass::Outside)
                        | (Rank::B, FaceClass::Inside | FaceClass::CoplanarOpposite)
                ),
                BooleanOp::Intersect => matches!(
                    (&sf.rank, &sf.classification),
                    (Rank::A | Rank::B, FaceClass::Inside) | (_, FaceClass::CoplanarSame)
                ),
            };

            if keep {
                Some(SelectedFace {
                    face_id: sf.face_id,
                    reversed: op == BooleanOp::Cut && sf.rank == Rank::B,
                })
            } else {
                None
            }
        })
        .collect()
}

/// A face selected for the boolean result.
#[derive(Debug, Clone)]
pub(crate) struct SelectedFace {
    /// The topology face to include.
    pub face_id: brepkit_topology::face::FaceId,
    /// Whether to reverse this face's orientation in the result.
    pub reversed: bool,
}
