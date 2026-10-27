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
    /// Two distances, `d1` on each edge's given reference face and `d2` on
    /// its other face.
    Referenced {
        /// Edges to chamfer, each with its reference face.
        edges: Vec<(EdgeId, FaceId)>,
        /// Distance on the reference face.
        d1: f64,
        /// Distance on the other face.
        d2: f64,
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

    /// Add edges with two distances, `d1` on each edge's reference face and
    /// `d2` on its other face (a distance-and-angle chamfer measured from a
    /// chosen face is `d2 = d1 * tan(angle)`). A reference face that does
    /// not border its edge fails the build.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_edges_asymmetric_on(
        &mut self,
        edges: &[(EdgeId, FaceId)],
        d1: f64,
        d2: f64,
    ) -> &mut Self {
        self.edge_sets.push(ChamferEdgeSet::Referenced {
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
    /// 5. Trim adjacent faces along contact curves, recording a stripe that
    ///    cannot trim both its faces as a failure.
    /// 6. Assemble new solid from trimmed faces, blend faces, untouched
    ///    original faces, and the input's inner shells.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if no edges were specified, an edge lies on an
    /// inner shell, or topology lookups fail. Individual edge failures are
    /// recorded in [`BlendResult::failed`] rather than aborting the whole
    /// operation.
    #[allow(clippy::too_many_lines)]
    pub fn build(self) -> Result<BlendResult, BlendError> {
        let adjacency = self.topo.build_adjacency(self.solid)?;
        let solid = self.topo.solid(self.solid)?;
        let (shell_id, inner_shells) = (solid.outer_shell(), solid.inner_shells().to_vec());
        // The chamfer rebuilds the outer shell only and carries the inner
        // shells over as they are, so it cannot cut a cavity's edge.
        let mut cavity_edges: HashSet<EdgeId> = HashSet::new();
        for &shell in &inner_shells {
            for &face in self.topo.shell(shell)?.faces() {
                cavity_edges.extend(brepkit_topology::explorer::face_edges(self.topo, face)?);
            }
        }
        let on_outer_shell = |eid: EdgeId| {
            if cavity_edges.contains(&eid) {
                Err(BlendError::PlanningFailure {
                    reason: format!("chamfer edge {eid:?} is on an inner shell"),
                })
            } else {
                Ok(eid)
            }
        };
        let mut all_edges: Vec<(EdgeId, f64, f64)> = Vec::new();
        for set in self.edge_sets {
            match set {
                ChamferEdgeSet::TwoDistance { edges, d1, d2 } => {
                    for eid in edges {
                        all_edges.push((on_outer_shell(eid)?, d1, d2));
                    }
                }
                ChamferEdgeSet::DistanceAngle {
                    edges,
                    distance,
                    angle,
                } => {
                    let d2 = distance * angle.tan();
                    for eid in edges {
                        all_edges.push((on_outer_shell(eid)?, distance, d2));
                    }
                }
                // Each stripe takes d1 on the edge's first adjacent face.
                ChamferEdgeSet::Referenced { edges, d1, d2 } => {
                    for (eid, reference) in edges {
                        match adjacency.faces_for_edge(on_outer_shell(eid)?) {
                            [first, _] if *first == reference => all_edges.push((eid, d1, d2)),
                            [_, second] if *second == reference => all_edges.push((eid, d2, d1)),
                            _ => {
                                return Err(BlendError::Topology(
                                    brepkit_topology::TopologyError::Empty {
                                        entity: "chamfer reference face on its edge",
                                    },
                                ));
                            }
                        }
                    }
                }
            }
        }

        if all_edges.is_empty() {
            return Err(BlendError::Topology(
                brepkit_topology::TopologyError::Empty {
                    entity: "chamfer edge set",
                },
            ));
        }

        let topo = self.topo;

        let original_faces: Vec<FaceId> = topo.shell(shell_id)?.faces().to_vec();

        let mut succeeded: Vec<EdgeId> = Vec::new();
        let mut failed: Vec<(EdgeId, BlendError)> = Vec::new();
        let mut stripe_results: Vec<(EdgeId, StripeResult)> = Vec::new();

        for &(edge_id, d1, d2) in &all_edges {
            match compute_chamfer_stripe(topo, &adjacency, edge_id, d1, d2) {
                Ok(sr) => stripe_results.push((edge_id, sr)),
                Err(e) => failed.push((edge_id, e)),
            }
        }

        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();

        // Each stripe that trims both its faces, with the contact edges the
        // trims left on them.
        let mut trimmed: Vec<(EdgeId, StripeResult, [Option<EdgeId>; 2])> = Vec::new();
        for (edge_id, sr) in stripe_results {
            let stripe = &sr.stripe;

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

            // The stripe trims both its faces or neither: a face left whole
            // (the trimmer refused it, or cannot cut its surface) would leave
            // the chamfer face's contact edge on it unshared and the shell
            // open, so the edge is left unchamfered and recorded as failed.
            let faces = [stripe.face1, stripe.face2];
            let before = faces.map(|f| face_replacements.get(&f).copied());
            let mut contacts = [None, None];
            let mut failure = None;
            for (slot, (face, contact)) in faces
                .into_iter()
                .zip([&stripe.contact1, &stripe.contact2])
                .enumerate()
            {
                let current = face_replacements.get(&face).copied().unwrap_or(face);
                match trimmer::trim_face(
                    topo,
                    current,
                    &sample_nurbs_endpoints(contact),
                    &[(0.0, 0.0), (1.0, 0.0)],
                    keep,
                ) {
                    Ok(tr) if tr.trimmed_face != current => {
                        contacts[slot] = tr.contact_edge;
                        face_replacements.insert(face, tr.trimmed_face);
                    }
                    Ok(_) => {
                        failure = Some(BlendError::TrimmingFailure { face: current });
                        break;
                    }
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
            }
            if let Some(error) = failure {
                for (face, prior) in faces.into_iter().zip(before) {
                    match prior {
                        Some(f) => face_replacements.insert(face, f),
                        None => face_replacements.remove(&face),
                    };
                }
                failed.push((edge_id, error));
                continue;
            }
            succeeded.push(edge_id);
            trimmed.push((edge_id, sr, contacts));
        }

        // If no stripe trimmed, return the original solid with all failures.
        if trimmed.is_empty() {
            return Ok(BlendResult {
                solid: self.solid,
                succeeded: Vec::new(),
                failed,
                is_partial: false,
            });
        }

        let mut blend_face_ids: Vec<FaceId> = Vec::new();
        let mut cross_edges: Vec<CrossEdge> = Vec::new();

        for (edge_id, sr, [c1, c2]) in &trimmed {
            // Reuse the trimmed neighbours' contact edges (mirrors the fillet
            // builder): a freshly minted duplicate leaves both copies use-1
            // and opens the shell along the chamfer flanks.
            let info = crate::builder_utils::create_chamfer_face_with_contacts(
                topo, &sr.stripe, *c1, *c2,
            )?;
            orient_chamfer_face(topo, info.face, &sr.stripe)?;
            // A closed rim has no end faces to take its cross edges. The
            // spine runs the chamfered edge forward, so the cross edge at its
            // end cuts off the edge's end vertex and the one at its start,
            // the edge's start vertex.
            if !sr.stripe.spine.is_closed() {
                let chamfered = topo.edge(*edge_id)?;
                let ends = [
                    info.cross_end.map(|c| (c, chamfered.end())),
                    info.cross_start.map(|c| (c, chamfered.start())),
                ];
                for ((edge, from, to), corner) in ends.into_iter().flatten() {
                    cross_edges.push(CrossEdge {
                        edge,
                        from,
                        to,
                        corner,
                        chamfer: info.face,
                    });
                }
            }
            blend_face_ids.push(info.face);
        }

        let mut result_faces: Vec<FaceId> = original_faces
            .iter()
            .map(|fid| face_replacements.get(fid).copied().unwrap_or(*fid))
            .collect();

        close_chamfer_ends(topo, &result_faces, &cross_edges)?;
        result_faces.extend(&blend_face_ids);

        let new_shell = Shell::new(result_faces)?;
        let new_shell_id = topo.add_shell(new_shell);
        let new_solid = Solid::new(new_shell_id, inner_shells);
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

/// A chamfer's cross edge at one end: it runs `from` one contact point `to`
/// the other on the chamfer face `chamfer`, across the corner vertex
/// `corner` of the chamfered edge.
struct CrossEdge {
    edge: EdgeId,
    from: VertexId,
    to: VertexId,
    corner: VertexId,
    chamfer: FaceId,
}

/// A chamfer's end faces still run through the corner it cut off (or
/// filled): their wires step from one contact point to the old vertex and on
/// to the other contact point. Each such detour is replaced by the chamfer's
/// cross edge between the two points, which the end face then shares with
/// the chamfer face. That drops the corner triangle from the end face at a
/// convex edge and adds it at a concave one. The detour can pass more
/// vertices than the corner, where an end face's straight edge was left in
/// pieces and the chamfer reaches past the piece at the corner: the detour
/// is the run between the points that passes the corner, which can have
/// more edges than the face's other run. A cross edge with no detour to
/// take it (two chamfers meeting at the vertex, or a contact edge that
/// could not be matched to the trimmer's) would leave the shell open, so
/// the chamfer is refused.
fn close_chamfer_ends(
    topo: &mut Topology,
    faces: &[FaceId],
    cross_edges: &[CrossEdge],
) -> Result<(), BlendError> {
    for cross in cross_edges {
        let Some(detour) = corner_detour(topo, faces, cross)? else {
            return Err(BlendError::TrimmingFailure {
                face: cross.chamfer,
            });
        };
        let edges = topo.wire(detour.wire)?.edges().to_vec();
        let n = edges.len();
        let (i, k) = (detour.first, detour.len);
        let mut spliced: Vec<OrientedEdge> = (0..n - k).map(|j| edges[(i + k + j) % n]).collect();
        spliced.push(OrientedEdge::new(cross.edge, detour.starts_at_from));
        let new_wire = topo.add_wire(Wire::new(spliced, true)?);
        let face = topo.face_mut(detour.face)?;
        if face.outer_wire() == detour.wire {
            face.set_outer_wire(new_wire);
        } else if let Some(slot) = face
            .inner_wires_mut()
            .iter_mut()
            .find(|w| **w == detour.wire)
        {
            *slot = new_wire;
        }
    }
    Ok(())
}

/// A run of edges in an end face's wire: `len` edges from position `first`.
struct Detour {
    face: FaceId,
    wire: WireId,
    first: usize,
    len: usize,
    starts_at_from: bool,
}

/// The run in an end face's wire from one of `cross`'s contact points to
/// the other through its corner.
fn corner_detour(
    topo: &Topology,
    faces: &[FaceId],
    cross: &CrossEdge,
) -> Result<Option<Detour>, BlendError> {
    let ends = |oe: &OrientedEdge| -> Result<(VertexId, VertexId), BlendError> {
        let e = topo.edge(oe.edge())?;
        Ok(if oe.is_forward() {
            (e.start(), e.end())
        } else {
            (e.end(), e.start())
        })
    };
    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let edges = topo.wire(wid)?.edges();
            let n = edges.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, _) = ends(&edges[i])?;
                if a != cross.from && a != cross.to {
                    continue;
                }
                let other = if a == cross.from {
                    cross.to
                } else {
                    cross.from
                };
                let mut past_corner = false;
                for k in 1..n {
                    let (_, b) = ends(&edges[(i + k - 1) % n])?;
                    if b == cross.from || b == cross.to {
                        if b == other && past_corner {
                            return Ok(Some(Detour {
                                face: fid,
                                wire: wid,
                                first: i,
                                len: k,
                                starts_at_from: a == cross.from,
                            }));
                        }
                        break;
                    }
                    past_corner |= b == cross.corner;
                }
            }
        }
    }
    Ok(None)
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

    /// A wall and a cap on a closed circular rim of `radius` at z = 0, laid
    /// out as the cylinder and cone primitives lay out their bottom rims:
    /// the wall first, running the rim forward, then the cap (outward -z)
    /// running it backward.
    fn bottom_rim(topo: &mut Topology, radius: f64, wall: FaceSurface) -> (SolidId, EdgeId) {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;

        let v = topo.add_vertex(Vertex::new(Point3::new(radius, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), radius).unwrap();
        let rim = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let cap = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        };
        let mut faces = Vec::new();
        for (forward, surface) in [(true, wall), (false, cap)] {
            let wire =
                topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, forward)], true).unwrap());
            faces.push(topo.add_face(Face::new(wire, Vec::new(), surface)));
        }
        let shell = topo.add_shell(Shell::new(faces).unwrap());
        (topo.add_solid(Solid::new(shell, Vec::new())), rim)
    }

    /// Each contact circle's radius and height, in either order.
    fn assert_contacts(stripe: &crate::stripe::Stripe, want: [(f64, f64); 2]) {
        let at = |c: &brepkit_math::nurbs::curve::NurbsCurve| {
            let p = c.evaluate(c.domain().0);
            (p.x().hypot(p.y()), p.z())
        };
        let got = [at(&stripe.contact1), at(&stripe.contact2)];
        let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() + (a.1 - b.1).abs() < 1e-9;
        assert!(
            (near(got[0], want[0]) && near(got[1], want[1]))
                || (near(got[0], want[1]) && near(got[1], want[0])),
            "contacts {got:?}, want {want:?}"
        );
    }

    /// The trimmer cannot cut a face along a closed rim, so the builder
    /// records the rim as failed and hands the solid back as it was.
    fn assert_rim_refused(topo: &mut Topology, solid: SolidId, rim: EdgeId, d: f64) {
        let mut builder = ChamferBuilder::new(topo, solid);
        builder.add_edges_symmetric(&[rim], d);
        let result = builder.build().unwrap();
        assert_eq!(result.solid, solid);
        assert!(result.succeeded.is_empty());
        assert!(
            matches!(result.failed[..], [(e, BlendError::TrimmingFailure { .. })] if e == rim),
            "{:?}",
            result.failed
        );
    }

    /// A cylinder's bottom rim: the stripe is an exact 45-degree cone from
    /// radius `r - d` on the cap to height `d` up the wall (not below the
    /// cap).
    #[test]
    fn a_cylinder_rims_stripe_is_an_exact_cone() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::traits::ParametricSurface;

        let (r, d) = (2.0, 0.4);
        let mut topo = Topology::new();
        let wall = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r)
            .unwrap();
        let (solid, rim) = bottom_rim(&mut topo, r, FaceSurface::Cylinder(wall));
        let adjacency = topo.build_adjacency(solid).unwrap();
        let stripe = compute_chamfer_stripe(&topo, &adjacency, rim, d, d)
            .unwrap()
            .stripe;
        let FaceSurface::Cone(cone) = &stripe.surface else {
            panic!("expected a cone, got {}", stripe.surface.type_tag());
        };
        let alpha = std::f64::consts::FRAC_PI_4;
        assert!((cone.half_angle() - alpha).abs() < 1e-12);
        // At u = 0 the cone's radial direction is +y, the wall's frame.
        for (v, want) in [
            ((r - d) / alpha.cos(), Point3::new(0.0, r - d, 0.0)),
            (r / alpha.cos(), Point3::new(0.0, r, d)),
        ] {
            let got = ParametricSurface::evaluate(cone, 0.0, v);
            assert!(
                (got - want).length() < 1e-9,
                "at v={v}: {got:?}, want {want:?}"
            );
        }
        assert_contacts(&stripe, [(r - d, 0.0), (r, d)]);
        assert_rim_refused(&mut topo, solid, rim, d);
    }

    /// A frustum's bottom rim (radius 3, wall at `alpha = atan2(6, 3)` from
    /// the cap): the stripe is an exact cone at `pi/2 - alpha/2` from the
    /// cap's contact at `r - d` to the wall's, `d` along the wall.
    #[test]
    fn a_frustum_rims_stripe_is_an_exact_cone() {
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_math::traits::ParametricSurface;

        let (r, d) = (3.0, 0.4);
        let alpha = 6.0_f64.atan2(3.0);
        let mut topo = Topology::new();
        let wall =
            ConicalSurface::new(Point3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -1.0), alpha)
                .unwrap();
        let (solid, rim) = bottom_rim(&mut topo, r, FaceSurface::Cone(wall));
        let adjacency = topo.build_adjacency(solid).unwrap();
        let stripe = compute_chamfer_stripe(&topo, &adjacency, rim, d, d)
            .unwrap()
            .stripe;
        let FaceSurface::Cone(cone) = &stripe.surface else {
            panic!("expected a cone, got {}", stripe.surface.type_tag());
        };
        let beta = std::f64::consts::FRAC_PI_2 - alpha * 0.5;
        assert!((cone.half_angle() - beta).abs() < 1e-9);
        // The wall's frame about -z puts the cone's u = 0 direction at -y.
        let (wall_r, wall_z) = (r - d * alpha.cos(), d * alpha.sin());
        for (v, want) in [
            ((r - d) / beta.cos(), Point3::new(0.0, -(r - d), 0.0)),
            (wall_r / beta.cos(), Point3::new(0.0, -wall_r, wall_z)),
        ] {
            let got = ParametricSurface::evaluate(cone, 0.0, v);
            assert!(
                (got - want).length() < 1e-9,
                "at v={v}: {got:?}, want {want:?}"
            );
        }
        assert_contacts(&stripe, [(r - d, 0.0), (wall_r, wall_z)]);
        assert_rim_refused(&mut topo, solid, rim, d);
    }
}
