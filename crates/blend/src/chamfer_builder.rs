//! Chamfer builder: orchestrates the full chamfer pipeline.
//!
//! Supports symmetric, asymmetric, and distance-angle chamfer modes on
//! planar face pairs (v1). Reuses the analytic fast path and face trimming
//! infrastructure from the fillet pipeline.

use std::collections::HashSet;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::analytic;
use crate::builder_utils::sample_nurbs_endpoints;
use crate::spine::Spine;
use crate::stripe::StripeResult;
use crate::trimmer::{self, TrimKeep};
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

        let adjacency = topo.build_adjacency(self.solid)?;

        let shell_id = topo.solid(self.solid)?.outer_shell();
        let original_faces: Vec<FaceId> = topo.shell(shell_id)?.faces().to_vec();

        let mut touched_faces: HashSet<FaceId> = HashSet::new();

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

        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();

        let mut stripe_contact_edges: Vec<(
            Option<brepkit_topology::edge::EdgeId>,
            Option<brepkit_topology::edge::EdgeId>,
        )> = Vec::new();
        for sr in &stripe_results {
            let stripe = &sr.stripe;
            stripe_contact_edges.push((None, None));

            let contact1_pts = sample_nurbs_endpoints(&stripe.contact1);
            let contact2_pts = sample_nurbs_endpoints(&stripe.contact2);

            // Keep the side of the contact line AWAY from the spine edge
            // (mirrors the fillet builder): the strip between the contact
            // line and the old edge is what the chamfer face replaces. The
            // side is resolved inside the trimmer, whose Left/Right frame
            // follows each face's wire traversal and cannot be predicted
            // here — a surface-normal side test against the section centre
            // reads the same for both traversals and picks the wrong chain
            // on one of them (the concave notch on a canonically-wound
            // prism kept the ridge strip and grew the solid).
            let spine_pt = stripe.spine.evaluate(topo, 0.0)?;
            let keep = TrimKeep::AwayFrom(spine_pt);

            let current_face1 = face_replacements
                .get(&stripe.face1)
                .copied()
                .unwrap_or(stripe.face1);
            let trim1 = trimmer::trim_face(
                topo,
                current_face1,
                &contact1_pts,
                &[(0.0, 0.0), (1.0, 0.0)],
                keep,
            );

            match trim1 {
                Ok(tr) if tr.trimmed_face != current_face1 => {
                    if let Some(slot) = stripe_contact_edges.last_mut() {
                        slot.0 = tr.contact_edge;
                    }
                    face_replacements.insert(stripe.face1, tr.trimmed_face);
                }
                Ok(_) => {}
                Err(e) => {
                    log::warn!("chamfer trimming failed on face {:?}: {e}", stripe.face1);
                }
            }

            let current_face2 = face_replacements
                .get(&stripe.face2)
                .copied()
                .unwrap_or(stripe.face2);
            let trim2 = trimmer::trim_face(
                topo,
                current_face2,
                &contact2_pts,
                &[(0.0, 0.0), (1.0, 0.0)],
                keep,
            );

            match trim2 {
                Ok(tr) if tr.trimmed_face != current_face2 => {
                    if let Some(slot) = stripe_contact_edges.last_mut() {
                        slot.1 = tr.contact_edge;
                    }
                    face_replacements.insert(stripe.face2, tr.trimmed_face);
                }
                Ok(_) => {}
                Err(e) => {
                    log::warn!("chamfer trimming failed on face {:?}: {e}", stripe.face2);
                }
            }
        }

        let mut blend_face_ids: Vec<FaceId> = Vec::new();
        let mut cross_edges: Vec<(EdgeId, VertexId, VertexId, FaceId)> = Vec::new();

        for (si, sr) in stripe_results.iter().enumerate() {
            // Reuse the trimmed neighbours' contact edges (mirrors the fillet
            // builder): a freshly minted duplicate leaves both copies use-1
            // and opens the shell along the chamfer flanks.
            let (c1, c2) = stripe_contact_edges
                .get(si)
                .copied()
                .unwrap_or((None, None));
            let info =
                crate::builder_utils::create_chamfer_face_with_contacts(topo, &sr.stripe, c1, c2)?;
            orient_chamfer_face(topo, info.face, &sr.stripe)?;
            // A closed rim has no end faces to take its cross edges.
            if !sr.stripe.spine.is_closed() {
                for (edge, from, to) in info.cross_end.into_iter().chain(info.cross_start) {
                    cross_edges.push((edge, from, to, info.face));
                }
            }
            blend_face_ids.push(info.face);
        }

        let mut result_faces: Vec<FaceId> = Vec::new();

        for &fid in &original_faces {
            if !touched_faces.contains(&fid) {
                result_faces.push(fid);
            }
        }

        for &fid in &touched_faces {
            let replacement = face_replacements.get(&fid).copied();
            result_faces.push(replacement.unwrap_or(fid));
        }

        close_chamfer_ends(topo, &result_faces, &cross_edges)?;
        result_faces.extend(&blend_face_ids);

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

/// A chamfer surface's normal follows the spine tangent and the order of the
/// contacts, not the material, so on a concave edge it points into the
/// solid. On either kind of edge the face's outward side leans the way its
/// two neighbours face, which sets the face's flag.
fn orient_chamfer_face(
    topo: &mut Topology,
    face: FaceId,
    stripe: &crate::stripe::Stripe,
) -> Result<(), BlendError> {
    let normal_at = |surface: &FaceSurface, p: Point3| {
        surface
            .project_point(p)
            .map_or_else(|| surface.normal(0.0, 0.0), |(u, v)| surface.normal(u, v))
    };
    let middle = |c: &brepkit_math::nurbs::curve::NurbsCurve| {
        let (t0, t1) = c.domain();
        c.evaluate(f64::midpoint(t0, t1))
    };
    let (p1, p2) = (middle(&stripe.contact1), middle(&stripe.contact2));
    let outward = |fid: FaceId, p: Point3| -> Result<Vec3, BlendError> {
        let f = topo.face(fid)?;
        let n = normal_at(f.surface(), p);
        Ok(if f.is_reversed() { -n } else { n })
    };
    let lean = outward(stripe.face1, p1)? + outward(stripe.face2, p2)?;
    let own = normal_at(topo.face(face)?.surface(), p1 + (p2 - p1) * 0.5);
    if own.dot(lean) < 0.0 {
        topo.face_mut(face)?.set_reversed(true);
    }
    Ok(())
}

/// A chamfer's end faces still run through the corner it cut off (or
/// filled): their wires step from one contact point to the old vertex and on
/// to the other contact point. Each such detour is replaced by the chamfer's
/// cross edge between the two points, which the end face then shares with
/// the chamfer face. That drops the corner triangle from the end face at a
/// convex edge and adds it at a concave one. A cross edge with no detour to
/// take it (two chamfers meeting at the vertex, or a contact edge that
/// could not be matched to the trimmer's) would leave the shell open, so
/// the chamfer is refused.
fn close_chamfer_ends(
    topo: &mut Topology,
    faces: &[FaceId],
    cross_edges: &[(EdgeId, VertexId, VertexId, FaceId)],
) -> Result<(), BlendError> {
    let ends = |topo: &Topology, oe: &OrientedEdge| -> Result<(VertexId, VertexId), BlendError> {
        let e = topo.edge(oe.edge())?;
        Ok(if oe.is_forward() {
            (e.start(), e.end())
        } else {
            (e.end(), e.start())
        })
    };
    for &(cross, from, to, chamfer) in cross_edges {
        let mut spliced_one = false;
        'faces: for &fid in faces {
            let face = topo.face(fid)?;
            let wires: Vec<WireId> = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .collect();
            for wid in wires {
                let edges = topo.wire(wid)?.edges().to_vec();
                let n = edges.len();
                if n < 3 {
                    continue;
                }
                for i in 0..n {
                    let (a, corner) = ends(topo, &edges[i])?;
                    let (corner2, b) = ends(topo, &edges[(i + 1) % n])?;
                    if corner != corner2 || !((a == from && b == to) || (a == to && b == from)) {
                        continue;
                    }
                    let mut spliced: Vec<OrientedEdge> =
                        (0..n).map(|k| edges[(i + 2 + k) % n]).take(n - 2).collect();
                    spliced.push(OrientedEdge::new(cross, a == from));
                    let new_wire = topo.add_wire(Wire::new(spliced, true)?);
                    let face = topo.face_mut(fid)?;
                    if face.outer_wire() == wid {
                        face.set_outer_wire(new_wire);
                    } else if let Some(slot) =
                        face.inner_wires_mut().iter_mut().find(|w| **w == wid)
                    {
                        *slot = new_wire;
                    }
                    spliced_one = true;
                    break 'faces;
                }
            }
        }
        if !spliced_one {
            return Err(BlendError::TrimmingFailure { face: chamfer });
        }
    }
    Ok(())
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
    let adj_faces = adjacency.faces_for_edge(edge_id);
    if adj_faces.len() != 2 {
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

    let surf1 = topo.face(face1)?.surface().clone();
    let surf2 = topo.face(face2)?.surface().clone();

    let spine = Spine::from_single_edge(topo, edge_id)?;

    if let Some(result) =
        analytic::try_analytic_chamfer(&surf1, &surf2, &spine, topo, d1, d2, face1, face2)?
    {
        return Ok(result);
    }

    log::debug!(
        target: "brepkit_approx",
        "chamfer: analytic path unavailable for {}+{} — v1 has no walker fallback, returning UnsupportedSurface",
        surf1.type_tag(),
        surf2.type_tag()
    );
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
        let result = builder.build();
        assert!(result.is_err(), "empty edge set should produce an error");
    }
}
