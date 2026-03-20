//! Chamfer builder: orchestrates the full chamfer pipeline.
//!
//! Supports symmetric, asymmetric, and distance-angle chamfer modes on
//! planar face pairs (v1). Reuses the analytic fast path and face trimming
//! infrastructure from the fillet pipeline.

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

use crate::analytic;
use crate::builder_utils::{create_blend_face, sample_nurbs_endpoints};
use crate::spine::Spine;
use crate::stripe::StripeResult;
use crate::trimmer::{self, TrimSide};
use crate::{BlendError, BlendResult};

/// Internal representation of a chamfer edge set with its distance parameters.
enum ChamferEdgeSet {
    /// Two explicit distances (d1 on face 1, d2 on face 2).
    TwoDistance {
        /// Edges to chamfer.
        edges: Vec<EdgeId>,
        /// Distance on face 1.
        d1: f64,
        /// Distance on face 2.
        d2: f64,
    },
    /// Distance on face 1 plus angle from face 1 toward face 2.
    DistanceAngle {
        /// Edges to chamfer.
        edges: Vec<EdgeId>,
        /// Distance on face 1.
        distance: f64,
        /// Angle from face 1 (radians).
        angle: f64,
    },
}

/// Builder for chamfer (bevel) operations on solid edges.
///
/// Collects edge sets with their distance parameters, then computes and
/// assembles the chamfered solid in a single `build()` call.
pub struct ChamferBuilder<'a> {
    topo: &'a mut Topology,
    solid: SolidId,
    edge_sets: Vec<ChamferEdgeSet>,
}

impl<'a> ChamferBuilder<'a> {
    /// Create a new chamfer builder for the given solid.
    #[must_use]
    pub fn new(topo: &'a mut Topology, solid: SolidId) -> Self {
        Self {
            topo,
            solid,
            edge_sets: Vec::new(),
        }
    }

    /// Add edges with symmetric chamfer distance (d1 = d2 = d).
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_edges_symmetric(&mut self, edges: &[EdgeId], d: f64) -> &mut Self {
        self.edge_sets.push(ChamferEdgeSet::TwoDistance {
            edges: edges.to_vec(),
            d1: d,
            d2: d,
        });
        self
    }

    /// Add edges with asymmetric chamfer distances.
    ///
    /// `d1` is the distance on face 1, `d2` on face 2.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_edges_asymmetric(&mut self, edges: &[EdgeId], d1: f64, d2: f64) -> &mut Self {
        self.edge_sets.push(ChamferEdgeSet::TwoDistance {
            edges: edges.to_vec(),
            d1,
            d2,
        });
        self
    }

    /// Add edges with distance-angle chamfer.
    ///
    /// `distance` is measured on face 1; `angle` (radians) determines
    /// the depth on face 2 as `distance * tan(angle)`.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_edges_distance_angle(
        &mut self,
        edges: &[EdgeId],
        distance: f64,
        angle: f64,
    ) -> &mut Self {
        self.edge_sets.push(ChamferEdgeSet::DistanceAngle {
            edges: edges.to_vec(),
            distance,
            angle,
        });
        self
    }

    /// Compute and build the chamfered solid.
    ///
    /// # Algorithm
    ///
    /// 1. Build adjacency index for the solid.
    /// 2. For each target edge, find the two adjacent faces.
    /// 3. Build single-edge spines (no chain propagation in v1).
    /// 4. Compute stripes via analytic fast path or record failure.
    /// 5. Trim adjacent faces along contact curves.
    /// 6. Assemble new solid from trimmed faces, blend faces, and untouched
    ///    original faces.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if no edges were specified, or if topology
    /// lookups fail. Individual edge failures are recorded in
    /// [`BlendResult::failed`] rather than aborting the whole operation.
    #[allow(clippy::too_many_lines)]
    pub fn build(self) -> Result<BlendResult, BlendError> {
        // ── Validate input ──────────────────────────────────────────────
        let all_edges: Vec<(EdgeId, f64, f64)> = self
            .edge_sets
            .into_iter()
            .flat_map(|set| {
                let (edges, d1, d2) = match set {
                    ChamferEdgeSet::TwoDistance { edges, d1, d2 } => (edges, d1, d2),
                    ChamferEdgeSet::DistanceAngle {
                        edges,
                        distance,
                        angle,
                    } => {
                        let d2 = distance * angle.tan();
                        (edges, distance, d2)
                    }
                };
                edges.into_iter().map(move |eid| (eid, d1, d2))
            })
            .collect();

        if all_edges.is_empty() {
            return Err(BlendError::Topology(
                brepkit_topology::TopologyError::Empty {
                    entity: "chamfer edge set",
                },
            ));
        }

        let topo = self.topo;

        // ── Build adjacency ─────────────────────────────────────────────
        let adjacency = topo.build_adjacency(self.solid)?;

        // Collect all original face IDs.
        let shell_id = topo.solid(self.solid)?.outer_shell();
        let original_faces: Vec<FaceId> = topo.shell(shell_id)?.faces().to_vec();

        // Track which faces are touched (adjacent to a chamfer edge).
        let mut touched_faces: HashSet<FaceId> = HashSet::new();

        // ── Phase 1: Compute stripes ────────────────────────────────────
        let mut succeeded: Vec<EdgeId> = Vec::new();
        let mut failed: Vec<(EdgeId, BlendError)> = Vec::new();
        let mut stripe_results: Vec<StripeResult> = Vec::new();

        for (edge_id, d1, d2) in &all_edges {
            let result = compute_chamfer_stripe(topo, &adjacency, *edge_id, *d1, *d2);
            match result {
                Ok(sr) => {
                    touched_faces.insert(sr.stripe.face1);
                    touched_faces.insert(sr.stripe.face2);
                    stripe_results.push(sr);
                    succeeded.push(*edge_id);
                }
                Err(e) => {
                    failed.push((*edge_id, e));
                }
            }
        }

        // If no stripes succeeded, return the original solid with all failures.
        if stripe_results.is_empty() {
            return Ok(BlendResult {
                solid: self.solid,
                succeeded: Vec::new(),
                failed,
                is_partial: false,
            });
        }

        // ── Phase 2: Trim faces ─────────────────────────────────────────
        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();

        for sr in &stripe_results {
            let stripe = &sr.stripe;

            let contact1_pts = sample_nurbs_endpoints(&stripe.contact1);
            let contact2_pts = sample_nurbs_endpoints(&stripe.contact2);

            // Compute trim side from chamfer geometry.
            let keep_side1 =
                if let (Some(sec), Ok(face)) = (stripe.sections.first(), topo.face(stripe.face1)) {
                    let n = face.surface().normal(0.0, 0.0);
                    if n.dot(sec.center - sec.p1) > 0.0 {
                        TrimSide::Right
                    } else {
                        TrimSide::Left
                    }
                } else {
                    TrimSide::Right
                };
            let keep_side2 =
                if let (Some(sec), Ok(face)) = (stripe.sections.first(), topo.face(stripe.face2)) {
                    let n = face.surface().normal(0.0, 0.0);
                    if n.dot(sec.center - sec.p2) > 0.0 {
                        TrimSide::Right
                    } else {
                        TrimSide::Left
                    }
                } else {
                    TrimSide::Right
                };

            // Trim face 1.
            let current_face1 = face_replacements
                .get(&stripe.face1)
                .copied()
                .unwrap_or(stripe.face1);
            let trim1 = trimmer::trim_face(
                topo,
                current_face1,
                &contact1_pts,
                &[(0.0, 0.0), (1.0, 0.0)],
                keep_side1,
            );

            match trim1 {
                Ok(tr) if tr.trimmed_face != current_face1 => {
                    face_replacements.insert(stripe.face1, tr.trimmed_face);
                }
                Ok(_) => {}
                Err(e) => {
                    log::warn!("chamfer trimming failed on face {:?}: {e}", stripe.face1);
                }
            }

            // Trim face 2.
            let current_face2 = face_replacements
                .get(&stripe.face2)
                .copied()
                .unwrap_or(stripe.face2);
            let trim2 = trimmer::trim_face(
                topo,
                current_face2,
                &contact2_pts,
                &[(0.0, 0.0), (1.0, 0.0)],
                keep_side2,
            );

            match trim2 {
                Ok(tr) if tr.trimmed_face != current_face2 => {
                    face_replacements.insert(stripe.face2, tr.trimmed_face);
                }
                Ok(_) => {}
                Err(e) => {
                    log::warn!("chamfer trimming failed on face {:?}: {e}", stripe.face2);
                }
            }
        }

        // ── Phase 3: Create blend faces ─────────────────────────────────
        let mut blend_face_ids: Vec<FaceId> = Vec::new();

        for sr in &stripe_results {
            let blend_face_id = create_blend_face(topo, &sr.stripe)?;
            blend_face_ids.push(blend_face_id);
        }

        // ── Phase 4: Assemble solid ─────────────────────────────────────
        let mut result_faces: Vec<FaceId> = Vec::new();

        // Add untouched original faces.
        for &fid in &original_faces {
            if !touched_faces.contains(&fid) {
                result_faces.push(fid);
            }
        }

        // Add trimmed replacements (or originals if not replaced).
        for &fid in &touched_faces {
            let replacement = face_replacements.get(&fid).copied();
            result_faces.push(replacement.unwrap_or(fid));
        }

        // Add blend faces.
        result_faces.extend(&blend_face_ids);

        // Build shell and solid.
        let new_shell = Shell::new(result_faces)?;
        let new_shell_id = topo.add_shell(new_shell);
        let new_solid = Solid::new(new_shell_id, Vec::new());
        let new_solid_id = topo.add_solid(new_solid);

        let is_partial = !failed.is_empty();
        Ok(BlendResult {
            solid: new_solid_id,
            succeeded,
            failed,
            is_partial,
        })
    }
}

/// Compute a chamfer stripe for a single edge using the adjacency index.
///
/// # Errors
///
/// Returns [`BlendError`] if the edge is non-manifold, if topology lookups
/// fail, or if the analytic path cannot produce a result.
fn compute_chamfer_stripe(
    topo: &Topology,
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    edge_id: EdgeId,
    d1: f64,
    d2: f64,
) -> Result<StripeResult, BlendError> {
    // Find the two adjacent faces.
    let adj_faces = adjacency.faces_for_edge(edge_id);
    if adj_faces.len() != 2 {
        // Non-manifold (3+ faces) or boundary (0-1 faces) edge cannot be chamfered.
        log::warn!(
            "edge {edge_id:?} has {} adjacent faces (expected 2) — cannot chamfer non-manifold or boundary edges",
            adj_faces.len()
        );
        return Err(BlendError::StartSolutionFailure {
            edge: edge_id,
            t: 0.0,
        });
    }
    let face1 = adj_faces[0];
    let face2 = adj_faces[1];

    // Snapshot surface data.
    let surf1 = topo.face(face1)?.surface().clone();
    let surf2 = topo.face(face2)?.surface().clone();

    // Build a single-edge spine.
    let spine = Spine::from_single_edge(topo, edge_id)?;

    // Try analytic fast path.
    if let Some(result) =
        analytic::try_analytic_chamfer(&surf1, &surf2, &spine, topo, d1, d2, face1, face2)?
    {
        return Ok(result);
    }

    // v1: no walker fallback for non-analytic surface pairs.
    Err(BlendError::UnsupportedSurface {
        face: face1,
        surface_tag: format!(
            "{}+{} (walker not yet integrated)",
            surf1.type_tag(),
            surf2.type_tag()
        ),
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use brepkit_topology::adjacency::AdjacencyIndex;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    /// Find the first manifold edge of the solid (shared by exactly 2 faces).
    fn find_manifold_edge(topo: &Topology, solid: SolidId) -> EdgeId {
        let adjacency = AdjacencyIndex::build(topo, solid).unwrap();
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let faces = topo.shell(shell_id).unwrap().faces().to_vec();

        for &fid in &faces {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                let adj = adjacency.faces_for_edge(oe.edge());
                if adj.len() == 2 {
                    return oe.edge();
                }
            }
        }
        panic!("cube should have manifold edges");
    }

    #[test]
    fn chamfer_builder_symmetric() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let target_edge = find_manifold_edge(&topo, solid);

        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let original_face_count = topo.shell(shell_id).unwrap().faces().len();

        let mut builder = ChamferBuilder::new(&mut topo, solid);
        builder.add_edges_symmetric(&[target_edge], 0.1);
        let result = builder.build().expect("chamfer build should succeed");

        // The result solid should exist and have more faces.
        let result_solid = topo.solid(result.solid).unwrap();
        let result_shell = topo.shell(result_solid.outer_shell()).unwrap();

        assert!(
            result_shell.faces().len() > original_face_count,
            "expected more faces after chamfer: got {}, original {}",
            result_shell.faces().len(),
            original_face_count,
        );

        assert!(result.succeeded.contains(&target_edge));
        assert!(result.failed.is_empty());
        assert!(!result.is_partial);

        // The blend surface should be a plane (plane-plane chamfer).
        let mut found_chamfer_plane = false;
        for &fid in result_shell.faces() {
            let face = topo.face(fid).unwrap();
            if matches!(face.surface(), FaceSurface::Plane { .. }) {
                found_chamfer_plane = true;
            }
        }
        assert!(
            found_chamfer_plane,
            "chamfer should produce a planar blend surface"
        );
    }

    #[test]
    fn chamfer_builder_distance_angle() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let target_edge = find_manifold_edge(&topo, solid);

        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let original_face_count = topo.shell(shell_id).unwrap().faces().len();

        // 45-degree angle means d2 = distance * tan(45deg) = distance.
        let distance = 0.15;
        let angle = std::f64::consts::FRAC_PI_4;

        let mut builder = ChamferBuilder::new(&mut topo, solid);
        builder.add_edges_distance_angle(&[target_edge], distance, angle);
        let result = builder.build().expect("chamfer build should succeed");

        let result_solid = topo.solid(result.solid).unwrap();
        let result_shell = topo.shell(result_solid.outer_shell()).unwrap();

        assert!(
            result_shell.faces().len() > original_face_count,
            "expected more faces after distance-angle chamfer"
        );
        assert!(result.succeeded.contains(&target_edge));
        assert!(result.failed.is_empty());
    }

    #[test]
    fn chamfer_builder_empty_edges_error() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let builder = ChamferBuilder::new(&mut topo, solid);
        // No edges added — should error.
        let result = builder.build();
        assert!(result.is_err(), "empty edge set should produce an error");
    }
}
