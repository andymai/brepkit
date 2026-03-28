//! Boolean operation selection — filters sub-faces by classification.
//!
//! Given the classified sub-faces from the Builder, selects which faces
//! to keep for each operation type (fuse, cut, intersect).
//!
//! Same-domain (SD) faces are handled separately from the IN/OUT truth
//! table. For each SD pair, the operation determines which face to keep
//! and which to discard, using orientation comparison:
//!
//! - **Fuse** (`isSameOriNeeded = true`): keep the same-oriented face (A),
//!   discard the other.
//! - **Cut** (`isSameOriNeeded = false`): keep the differently-oriented
//!   face (B, reversed). Discard A's overlapping sub-face.
//! - **Intersect** (`isSameOriNeeded = true`): keep the same-oriented face (A).

use std::collections::HashSet;

use crate::builder::same_domain::SameDomainPair;
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
/// The base truth table (non-SD faces):
/// - **Fuse**: A-Outside + B-Outside + A-On
/// - **Cut**: A-Outside + A-On + B-Inside
/// - **Intersect**: A-Inside + B-Inside + A-On
///
/// SD faces are handled by [`apply_sd_selection`] which overrides the
/// base selection for faces involved in same-domain pairs.
#[must_use]
pub(crate) fn select_faces(
    sub_faces: &[SubFace],
    op: BooleanOp,
    sd_pairs: &[SameDomainPair],
) -> Vec<SelectedFace> {
    // Step 1: Identify which sub-face indices are part of SD pairs
    let sd_indices: HashSet<usize> = sd_pairs.iter().flat_map(|p| [p.idx_a, p.idx_b]).collect();

    // Step 2: Select non-SD faces via the standard truth table
    let mut selected: Vec<SelectedFace> = sub_faces
        .iter()
        .enumerate()
        .filter_map(|(idx, sf)| {
            // Skip SD faces — handled separately below
            if sd_indices.contains(&idx) {
                return None;
            }

            let keep = match op {
                BooleanOp::Fuse => matches!(
                    (&sf.rank, &sf.classification),
                    (Rank::A | Rank::B, FaceClass::Outside) | (Rank::A, FaceClass::On)
                ),
                BooleanOp::Cut => matches!(
                    (&sf.rank, &sf.classification),
                    (Rank::A, FaceClass::Outside | FaceClass::On) | (Rank::B, FaceClass::Inside)
                ),
                BooleanOp::Intersect => matches!(
                    (&sf.rank, &sf.classification),
                    (Rank::A | Rank::B, FaceClass::Inside) | (Rank::A, FaceClass::On)
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
        .collect();

    // Step 3: Apply SD-specific selection
    apply_sd_selection(sub_faces, op, sd_pairs, &mut selected);

    selected
}

/// Apply same-domain face selection rules.
///
/// For each SD pair, decide which face(s) to include based on the operation
/// and orientation. This mirrors the reference implementation's approach:
/// `isSameOriNeeded = (objState == toolState)`.
fn apply_sd_selection(
    sub_faces: &[SubFace],
    op: BooleanOp,
    sd_pairs: &[SameDomainPair],
    selected: &mut Vec<SelectedFace>,
) {
    // For Fuse/Intersect, we want same-oriented faces.
    // For Cut, we want differently-oriented faces.
    let same_ori_needed = match op {
        BooleanOp::Fuse | BooleanOp::Intersect => true,
        BooleanOp::Cut => false,
    };

    for pair in sd_pairs {
        let sf_a = &sub_faces[pair.idx_a];

        // Distinguish touching (face on boundary) from overlapping
        // (face inside opposing solid). Uses AABB containment check
        // from same_domain (deterministic).
        let is_touching = !pair.b_contained_in_a;

        if same_ori_needed == pair.same_orientation {
            // Orientations match what the operation needs:
            // - Fuse + same-ori: keep A (representative) if on exterior, discard if internal
            // - Intersect + same-ori: keep A
            // - Cut + opposite-ori: depends on A's classification
            if op == BooleanOp::Cut {
                if is_touching {
                    // Touching: A's face is on the exterior — keep it
                    selected.push(SelectedFace {
                        face_id: sf_a.face_id,
                        reversed: false,
                    });
                }
                // Overlapping: both faces cancel — discard both
                continue;
            }
            // For Fuse/Intersect: if A is classified as Inside the opposing
            // solid, this SD pair is an internal overlap (e.g., cylinder cap
            // coincides with box face disc sub-face) — discard both faces.
            if (op == BooleanOp::Fuse || op == BooleanOp::Intersect)
                && sf_a.classification == FaceClass::Inside
            {
                continue;
            }
            selected.push(SelectedFace {
                face_id: sf_a.face_id,
                reversed: false,
            });
        } else {
            // Orientations DON'T match what the operation needs:
            // - Fuse + opposite-ori: internal faces — discard both
            // - Intersect + opposite-ori: discard both
            // - Cut + same-ori: discard both — the overlap region is
            //   cut away from A, and B's cap is provided by the
            //   B-Inside non-SD face from the regular selection.
            // For Fuse/Intersect/Cut with mismatched orientation: discard both
        }
    }

    if !sd_pairs.is_empty() {
        log::debug!(
            "apply_sd_selection: {} SD pairs processed for {:?}",
            sd_pairs.len(),
            op
        );
    }
}

/// A face selected for the boolean result.
#[derive(Debug, Clone)]
pub(crate) struct SelectedFace {
    /// The topology face to include.
    pub face_id: brepkit_topology::face::FaceId,
    /// Whether to reverse this face's orientation in the result.
    pub reversed: bool,
}
