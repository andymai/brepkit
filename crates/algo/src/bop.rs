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
//! - **Intersect** (`isSameOriNeeded = true`): keep the same-oriented face.
//!   For a geometric-containment pair this is the smaller (contained) face,
//!   whose footprint is bounded by the interior of both solids.

use std::collections::HashSet;

use crate::builder::same_domain::{SameDomainPair, WithinRankDuplicate};
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
    within_rank_dups: &[WithinRankDuplicate],
) -> Vec<SelectedFace> {
    // Step 1: Filter SD pairs to only those with valid (in-bounds) indices.
    // Invalid pairs are logged and excluded so they don't cause valid faces
    // to be incorrectly excluded from the normal truth-table selection.
    let valid_sd_pairs: Vec<&SameDomainPair> = sd_pairs
        .iter()
        .filter(|p| {
            // `representative` is always idx_a or idx_b, so its bound is
            // implied — but check it explicitly since `apply_sd_selection`
            // indexes `sub_faces` with it and an out-of-bounds value would
            // panic.
            let a_ok = p.idx_a < sub_faces.len();
            let b_ok = p.idx_b < sub_faces.len();
            let rep_ok = p.representative < sub_faces.len();
            if !a_ok || !b_ok || !rep_ok {
                log::warn!(
                    "SD pair ({}, {}) rep {} has out-of-bounds index (len={}), skipping",
                    p.idx_a,
                    p.idx_b,
                    p.representative,
                    sub_faces.len()
                );
            }
            a_ok && b_ok && rep_ok
        })
        .collect();

    // Step 1b: Within-rank duplicates (issue #696) — same-domain faces from
    // the same input solid. No operation-specific logic: the duplicate is
    // just removed from the selection. Filter by valid index in case the
    // upstream pipeline produces stale records after a split.
    let dup_indices: HashSet<usize> = within_rank_dups
        .iter()
        .filter(|d| d.duplicate < sub_faces.len() && d.representative < sub_faces.len())
        .map(|d| d.duplicate)
        .collect();

    // Step 2: Identify which sub-face indices are part of valid SD pairs
    let sd_indices: HashSet<usize> = valid_sd_pairs
        .iter()
        .flat_map(|p| [p.idx_a, p.idx_b])
        .collect();

    // Step 3: Select non-SD faces via the standard truth table
    let mut selected: Vec<SelectedFace> = sub_faces
        .iter()
        .enumerate()
        .filter_map(|(idx, sf)| {
            // Skip SD faces — handled separately below
            if sd_indices.contains(&idx) {
                return None;
            }
            // Skip within-rank SD duplicates (#696). Their representative
            // (the lowest-indexed face in the group) goes through the normal
            // truth-table selection; the duplicate is dropped here.
            if dup_indices.contains(&idx) {
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

    // Step 4: Apply SD-specific selection on valid pairs only
    apply_sd_selection(sub_faces, op, &valid_sd_pairs, &mut selected);

    selected
}

/// Apply same-domain face selection rules.
///
/// For each SD pair, decide which face(s) to include based on the operation
/// and orientation. This mirrors the reference implementation's approach:
/// `isSameOriNeeded = (objState == toolState)`.
///
/// Caller must ensure all pairs have valid (in-bounds) indices.
fn apply_sd_selection(
    sub_faces: &[SubFace],
    op: BooleanOp,
    sd_pairs: &[&SameDomainPair],
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
            // - Fuse + same-ori: keep the larger (containing) face for a
            //   containment pair, A for a coextensive pair
            // - Intersect + same-ori: keep the smaller (contained) face for a
            //   containment pair, A for a coextensive pair
            // - Cut + opposite-ori: keep A only when the faces merely touch
            //   on the boundary; discard both when A overlaps B's interior
            if op == BooleanOp::Cut {
                if is_touching {
                    // Touching: A's face is on the exterior — keep it. Cut is
                    // asymmetric (A is the minuend), so A is always the correct
                    // exterior representative — no geometry-based pick needed.
                    selected.push(SelectedFace {
                        face_id: sf_a.face_id,
                        reversed: false,
                    });
                }
                // Overlapping: both faces cancel — discard both
                continue;
            }
            // Fuse/Intersect with matching orientation: a same-oriented
            // coincident pair always lies on the result's exterior (both
            // solids' material is on the same side of the shared plane), so
            // keep exactly one face. Genuinely-internal coincident faces have
            // OPPOSITE orientation and fall into the else-branch.
            //
            // Which face is correct depends on the operation when the pair is a
            // geometric-containment pair (the two faces have different extent):
            //  - Fuse keeps the LARGER (containing) face — it covers the whole
            //    union boundary on the shared plane. `pair.representative` is
            //    that larger face (picked by area, so order-independent).
            //  - Intersect keeps the SMALLER face — the intersection footprint
            //    on the shared plane is bounded by the interior of BOTH solids,
            //    i.e. the contained face. Keeping the larger face would include
            //    area outside one operand and can leave the shell non-manifold.
            // For coextensive pairs both faces span the same domain, so larger
            // and smaller coincide and either choice is order-independent.
            let keep = if op == BooleanOp::Intersect {
                // The smaller face is the endpoint that is NOT the
                // (area-larger) representative.
                if pair.representative == pair.idx_a {
                    pair.idx_b
                } else {
                    pair.idx_a
                }
            } else {
                pair.representative
            };
            selected.push(SelectedFace {
                face_id: sub_faces[keep].face_id,
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::builder::same_domain::SameDomainPair;
    use brepkit_math::vec::Point3;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Create a dummy face in the topology to get a valid FaceId.
    fn dummy_face_id(topo: &mut Topology) -> brepkit_topology::face::FaceId {
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let eid = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let wire_id = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        topo.add_face(Face::new(
            wire_id,
            vec![],
            FaceSurface::Plane {
                normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    /// Helper to create a SubFace for testing.
    fn make_sub_face(topo: &mut Topology, rank: Rank, classification: FaceClass) -> SubFace {
        SubFace {
            face_id: dummy_face_id(topo),
            classification,
            rank,
            interior_point: None,
        }
    }

    #[test]
    fn select_faces_no_sd_pairs() {
        let mut topo = Topology::new();
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Outside),
        ];
        let selected = select_faces(&sub_faces, BooleanOp::Fuse, &[], &[]);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn sd_pair_out_of_bounds_skips_gracefully() {
        let mut topo = Topology::new();
        let sub_faces = vec![make_sub_face(&mut topo, Rank::A, FaceClass::Outside)];
        // SD pair references idx 5, which is out of bounds
        let sd_pairs = vec![SameDomainPair {
            idx_a: 5,
            idx_b: 10,
            same_orientation: true,
            b_contained_in_a: false,
            representative: 5,
        }];
        // Should not panic — out-of-bounds pairs are skipped
        let selected = select_faces(&sub_faces, BooleanOp::Fuse, &sd_pairs, &[]);
        // The non-SD face should still be selected
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn fuse_keeps_outside_faces() {
        let mut topo = Topology::new();
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::A, FaceClass::Inside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Inside),
        ];
        let selected = select_faces(&sub_faces, BooleanOp::Fuse, &[], &[]);
        assert_eq!(selected.len(), 2, "Fuse keeps only Outside faces");
    }

    #[test]
    fn cut_keeps_a_outside_and_b_inside() {
        let mut topo = Topology::new();
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::A, FaceClass::Inside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Inside),
        ];
        let selected = select_faces(&sub_faces, BooleanOp::Cut, &[], &[]);
        assert_eq!(selected.len(), 2, "Cut keeps A-Outside + B-Inside");
        assert!(
            selected.iter().any(|s| s.reversed),
            "B-Inside should be reversed"
        );
    }

    #[test]
    fn intersect_keeps_inside_faces() {
        let mut topo = Topology::new();
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::A, FaceClass::Inside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Inside),
        ];
        let selected = select_faces(&sub_faces, BooleanOp::Intersect, &[], &[]);
        assert_eq!(selected.len(), 2, "Intersect keeps only Inside faces");
    }

    /// A same-oriented geometric-containment SD pair: idx_a is the larger
    /// (containing) face, idx_b the smaller (contained) face. `representative`
    /// is the larger face (area-based pick). Fuse must keep the larger face so
    /// the full union boundary is covered; Intersect must keep the SMALLER
    /// face, whose footprint is bounded by the interior of both solids.
    #[test]
    fn sd_containment_pair_fuse_larger_intersect_smaller() {
        let mut topo = Topology::new();
        // idx 0 = A (larger / representative), idx 1 = B (smaller). Both On so
        // they only enter the result via SD selection, not the truth table.
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::On),
            make_sub_face(&mut topo, Rank::B, FaceClass::On),
        ];
        let larger = sub_faces[0].face_id;
        let smaller = sub_faces[1].face_id;
        let sd_pairs = vec![SameDomainPair {
            idx_a: 0,
            idx_b: 1,
            same_orientation: true,
            b_contained_in_a: true,
            representative: 0,
        }];

        let fused = select_faces(&sub_faces, BooleanOp::Fuse, &sd_pairs, &[]);
        assert_eq!(fused.len(), 1, "Fuse keeps exactly one face of the SD pair");
        assert_eq!(
            fused[0].face_id, larger,
            "Fuse keeps the larger (containing) face"
        );

        let intersected = select_faces(&sub_faces, BooleanOp::Intersect, &sd_pairs, &[]);
        assert_eq!(
            intersected.len(),
            1,
            "Intersect keeps exactly one face of the SD pair"
        );
        assert_eq!(
            intersected[0].face_id, smaller,
            "Intersect keeps the smaller (contained) face"
        );
    }

    /// The smaller-for-Intersect pick is independent of which operand is A:
    /// swap so idx_a is the smaller face and `representative` (the larger) is
    /// idx_b. Intersect must still keep the smaller face (idx_a here).
    #[test]
    fn sd_containment_intersect_smaller_when_representative_is_b() {
        let mut topo = Topology::new();
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::On),
            make_sub_face(&mut topo, Rank::B, FaceClass::On),
        ];
        let smaller = sub_faces[0].face_id;
        let larger = sub_faces[1].face_id;
        let sd_pairs = vec![SameDomainPair {
            idx_a: 0,
            idx_b: 1,
            same_orientation: true,
            b_contained_in_a: true,
            representative: 1,
        }];

        let fused = select_faces(&sub_faces, BooleanOp::Fuse, &sd_pairs, &[]);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].face_id, larger, "Fuse keeps the larger face");

        let intersected = select_faces(&sub_faces, BooleanOp::Intersect, &sd_pairs, &[]);
        assert_eq!(intersected.len(), 1);
        assert_eq!(
            intersected[0].face_id, smaller,
            "Intersect keeps the smaller face regardless of which operand it is"
        );
    }

    #[test]
    fn within_rank_dup_drops_duplicate_keeps_representative() {
        // idx 0 (representative) + idx 1 (duplicate): both rank A, Outside —
        // would normally both be selected for Fuse. The dup record should
        // remove idx 1, keep idx 0.
        let mut topo = Topology::new();
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::A, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Outside),
        ];
        let dups = vec![WithinRankDuplicate {
            representative: 0,
            duplicate: 1,
        }];
        let selected = select_faces(&sub_faces, BooleanOp::Fuse, &[], &dups);
        assert_eq!(
            selected.len(),
            2,
            "Fuse should keep representative + B-Outside, drop within-rank duplicate"
        );
    }

    #[test]
    fn within_rank_dup_out_of_bounds_skips_gracefully() {
        let mut topo = Topology::new();
        let sub_faces = vec![
            make_sub_face(&mut topo, Rank::A, FaceClass::Outside),
            make_sub_face(&mut topo, Rank::B, FaceClass::Outside),
        ];
        let dups = vec![WithinRankDuplicate {
            representative: 5,
            duplicate: 10,
        }];
        let selected = select_faces(&sub_faces, BooleanOp::Fuse, &[], &dups);
        assert_eq!(
            selected.len(),
            2,
            "Out-of-bounds within-rank-dup record should be ignored, not panic"
        );
    }
}
