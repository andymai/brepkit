//! Chamfer builder: orchestrates the full chamfer pipeline.
//!
//! Supports symmetric, asymmetric, and distance-angle chamfer modes on
//! planar face pairs (v1). Reuses the analytic fast path and face trimming
//! infrastructure from the fillet pipeline.

use brepkit_math::madd::Madd;
use std::collections::{HashMap, HashSet};

use brepkit_math::curves2d::{Curve2D, Line2D, NurbsCurve2D};
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::pcurve::PCurve;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::analytic;
use crate::blend_func::{BlendParams, ChamferBlend};
use crate::builder_utils::{SolidSnapshot, sample_nurbs_endpoints};
use crate::spine::Spine;
use crate::stripe::StripeResult;
use crate::trimmer::{self, TrimKeep};
use crate::walker::{Walker, WalkerConfig, interpolate_chamfer_band};
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
    ///    cannot trim both its faces as a failure. A second edge that trims
    ///    at a vertex of one trimmed before ends the build.
    /// 6. Assemble new solid from trimmed faces, blend faces, untouched
    ///    original faces, and the input's inner shells.
    ///
    /// The input solid comes through unchanged, whether the build succeeds
    /// or fails.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if no edges were specified, an edge lies on an
    /// inner shell, or topology lookups fail. Returns
    /// [`BlendError::CornerFailure`] at the first vertex where two chamfered
    /// edges that trimmed meet. Other edge failures are recorded in
    /// [`BlendResult::failed`] rather than aborting the whole operation.
    pub fn build(self) -> Result<BlendResult, BlendError> {
        let Self {
            topo,
            solid,
            edge_sets,
        } = self;
        // Trimming splits edges in every wire that uses them and closing a
        // chamfer's ends rewrites its end faces' wires, all in place on faces
        // the input shares with the result.
        let snapshot = SolidSnapshot::capture(topo, solid)?;
        let result = ChamferBuilder {
            topo: &mut *topo,
            solid,
            edge_sets,
        }
        .build_in_place();
        match result {
            Ok(result) => {
                let preserve = snapshot.clone_result_source_faces(topo, result.solid);
                let restore = snapshot.restore(topo);
                match (preserve, restore) {
                    (Ok(()), Ok(())) => Ok(result),
                    (Err(error), _) | (_, Err(error)) => Err(error),
                }
            }
            Err(error) => {
                snapshot.restore(topo)?;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn build_in_place(self) -> Result<BlendResult, BlendError> {
        let scope = trimmer::SplitScope::of_solid(self.topo, self.solid)?;
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
            let stripe =
                compute_chamfer_stripe(topo, &adjacency, edge_id, d1, d2).and_then(|mut sr| {
                    if sr.stripe.spine.is_closed() {
                        start_at_seam(topo, edge_id, &mut sr.stripe)?;
                    }
                    Ok(sr)
                });
            match stripe {
                Ok(mut sr) => {
                    run_to_end_planes(topo, &original_faces, edge_id, &mut sr.stripe)?;
                    stripe_results.push((edge_id, sr));
                }
                Err(e) => failed.push((edge_id, e)),
            }
        }

        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();

        // Each stripe that trims both its faces, with the contact edges the
        // trims left on them.
        let mut trimmed: Vec<(EdgeId, StripeResult, [Option<EdgeId>; 2])> = Vec::new();
        let mut consumed: HashSet<FaceId> = HashSet::new();
        // The end vertex of each open edge that trimmed. Two chamfers that
        // meet at a vertex need a mitre between their planes, which this
        // builder does not make: the second one to trim there ends the build
        // before the work on the rest.
        let mut ends_taken: HashMap<VertexId, EdgeId> = HashMap::new();
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
            let mut slivers = Vec::new();
            let chamfered = topo.edge(edge_id)?;
            let corners = [chamfered.start(), chamfered.end()];
            // A stripe on a curved face is trimmed along its contact curves
            // in each face's own parameters, as the fillet builder trims; an
            // open one ends on planes that hold its end sections.
            let curved = matches!(stripe.surface, FaceSurface::Nurbs(_))
                || faces
                    .iter()
                    .any(|&f| topo.face(f).is_ok_and(|face| !face.surface().is_planar()));
            if curved
                && !stripe.spine.is_closed()
                && !ends_on_section_planes(topo, &original_faces, edge_id, stripe)?
            {
                failed.push((
                    edge_id,
                    BlendError::UnsupportedSurface {
                        face: stripe.face1,
                        surface_tag: "a curved chamfer ending off a plane square to its edge"
                            .to_owned(),
                    },
                ));
                continue;
            }
            // Faces beside the two that a split of a shared edge hands a copy,
            // with what each mapped to before.
            let mut incident: Vec<(FaceId, Option<FaceId>)> = Vec::new();
            for (slot, (face, contact)) in faces
                .into_iter()
                .zip([&stripe.contact1, &stripe.contact2])
                .enumerate()
            {
                let current = face_replacements.get(&face).copied().unwrap_or(face);
                if curved {
                    match trim_along(topo, &scope, current, contact, keep, stripe.spine_edges()) {
                        Ok(batch) => {
                            contacts[slot] = batch.contact_edges.first().copied();
                            face_replacements.insert(face, batch.trimmed_face);
                            for &(source, copy) in &batch.incident_replacements {
                                let original = face_replacements
                                    .iter()
                                    .find(|&(_, &now)| now == source)
                                    .map_or(source, |(&original, _)| original);
                                incident
                                    .push((original, face_replacements.get(&original).copied()));
                                face_replacements.insert(original, copy);
                            }
                        }
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                    }
                    continue;
                }
                match trimmer::trim_face_to_corners(
                    topo,
                    &scope,
                    current,
                    &sample_nurbs_endpoints(contact),
                    &[(0.0, 0.0), (1.0, 0.0)],
                    keep,
                    &corners,
                ) {
                    Ok(tr) if tr.trimmed_face != current => {
                        contacts[slot] = tr.contact_edge;
                        // A distance that reaches exactly across the face
                        // (a side as high as the chamfer is wide) leaves it
                        // its far edge and the contact along it, with no
                        // area: the face goes, and the chamfer meets its
                        // neighbour across that edge.
                        if let Some(far) =
                            consumed_face_edge(topo, tr.trimmed_face, tr.contact_edge)?
                        {
                            contacts[slot] = Some(far);
                            slivers.push(tr.trimmed_face);
                        }
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
                for (face, prior) in incident
                    .into_iter()
                    .rev()
                    .chain(faces.into_iter().zip(before))
                {
                    match prior {
                        Some(f) => face_replacements.insert(face, f),
                        None => face_replacements.remove(&face),
                    };
                }
                failed.push((edge_id, error));
                continue;
            }
            if corners[0] != corners[1] {
                for corner in corners {
                    if ends_taken
                        .insert(corner, edge_id)
                        .is_some_and(|other| other != edge_id)
                    {
                        return Err(BlendError::CornerFailure { vertex: corner });
                    }
                }
            }
            consumed.extend(slivers);
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
            if sr.stripe.spine.is_closed() {
                let supports = [sr.stripe.face1, sr.stripe.face2]
                    .map(|f| face_replacements.get(&f).copied().unwrap_or(f));
                let face = closed_chamfer_face(topo, &sr.stripe, [*c1, *c2], supports)?;
                blend_face_ids.push(face);
                continue;
            }
            let info = crate::builder_utils::create_chamfer_face_with_contacts(
                topo, &sr.stripe, *c1, *c2,
            )?;
            orient_chamfer_face(topo, info.face, &sr.stripe)?;
            if let (
                FaceSurface::Nurbs(_) | FaceSurface::Cone(_) | FaceSurface::Cylinder(_),
                [Some(c1), Some(c2)],
            ) = (&sr.stripe.surface, [*c1, *c2])
            {
                let supports = [sr.stripe.face1, sr.stripe.face2]
                    .map(|f| face_replacements.get(&f).copied().unwrap_or(f));
                run_against_supports(topo, info.face, [(supports[0], c1), (supports[1], c2)])?;
            }
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
            .filter(|fid| !consumed.contains(fid))
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

/// The far edge of a trimmed face left as that edge and the contact along
/// it: a two-edge loop whose edges share their ends and both lie on one
/// line. `None` for any other face.
fn consumed_face_edge(
    topo: &Topology,
    face: FaceId,
    contact: Option<EdgeId>,
) -> Result<Option<EdgeId>, BlendError> {
    let Some(contact) = contact else {
        return Ok(None);
    };
    let edges = topo.wire(topo.face(face)?.outer_wire())?.edges();
    let [a, b] = edges else {
        return Ok(None);
    };
    let far = if a.edge() == contact {
        b.edge()
    } else {
        a.edge()
    };
    let (c, f) = (topo.edge(contact)?, topo.edge(far)?);
    if !matches!(c.curve(), brepkit_topology::edge::EdgeCurve::Line)
        || !matches!(f.curve(), brepkit_topology::edge::EdgeCurve::Line)
    {
        return Ok(None);
    }
    let same_ends = (c.start() == f.start() && c.end() == f.end())
        || (c.start() == f.end() && c.end() == f.start());
    Ok(same_ends.then_some(far))
}

/// A flat chamfer's contacts end in the section square to its edge at each
/// end, which is the end face's plane only where that face stands square to
/// the edge. Where one plane face closes the edge's end at a slant (a prism
/// whose next side turns off at an angle), each contact line is run on, or
/// back, to that face's plane: the chamfer plane then ends on its line in
/// the end face, which the end face takes as it does at a square end.
fn run_to_end_planes(
    topo: &Topology,
    faces: &[FaceId],
    edge_id: EdgeId,
    stripe: &mut crate::stripe::Stripe,
) -> Result<(), BlendError> {
    if !matches!(stripe.surface, FaceSurface::Plane { .. }) || stripe.spine.is_closed() {
        return Ok(());
    }
    let line_ends = |c: &brepkit_math::nurbs::curve::NurbsCurve| {
        (c.degree() == 1 && c.control_points().len() == 2)
            .then(|| (c.control_points()[0], c.control_points()[1]))
    };
    let (Some(mut c1), Some(mut c2)) = (line_ends(&stripe.contact1), line_ends(&stripe.contact2))
    else {
        return Ok(());
    };
    let edge = topo.edge(edge_id)?;
    let spine_start = stripe.spine.evaluate(topo, 0.0)?;
    let (sp, ep) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let ends = if (sp - spine_start).length() <= (ep - spine_start).length() {
        [edge.start(), edge.end()]
    } else {
        [edge.end(), edge.start()]
    };
    let mut changed = false;
    for (slot, corner) in ends.into_iter().enumerate() {
        // The faces through the corner other than the chamfer's two.
        let mut others = Vec::new();
        for &fid in faces {
            if fid == stripe.face1 || fid == stripe.face2 {
                continue;
            }
            let face = topo.face(fid)?;
            let mut through = false;
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wid)?.edges() {
                    let e = topo.edge(oe.edge())?;
                    through |= e.start() == corner || e.end() == corner;
                }
            }
            if through {
                others.push(fid);
            }
        }
        let [end_face] = others[..] else {
            continue;
        };
        let FaceSurface::Plane { normal, d } = *topo.face(end_face)?.surface() else {
            continue;
        };
        let to_plane = |(a, b): (Point3, Point3)| -> Option<Point3> {
            let dir = b - a;
            let along = normal.dot(dir);
            if along.abs() <= 1e-9 * dir.length() {
                return None;
            }
            let from = if slot == 0 { b } else { a };
            let s = (d - normal.dot(Vec3::new(from.x(), from.y(), from.z()))) / along;
            Some(from + dir * s)
        };
        let (Some(q1), Some(q2)) = (to_plane(c1), to_plane(c2)) else {
            continue;
        };
        let moved = |(a, b): (Point3, Point3), q: Point3| if slot == 0 { (q, b) } else { (a, q) };
        c1 = moved(c1, q1);
        c2 = moved(c2, q2);
        changed = true;
    }
    if !changed {
        return Ok(());
    }
    let line = |(a, b): (Point3, Point3)| {
        brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![a, b],
            vec![1.0, 1.0],
        )
    };
    stripe.contact1 = line(c1)?;
    stripe.contact2 = line(c2)?;
    for section in &mut stripe.sections {
        let (p1, p2) = if section.t < 0.5 {
            (c1.0, c2.0)
        } else {
            (c1.1, c2.1)
        };
        section.p1 = p1;
        section.p2 = p2;
        section.center = p1 + (p2 - p1) * 0.5;
    }
    Ok(())
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

/// Starts a closed stripe's contacts where the contact on its seamed face
/// crosses that face's seam: the seam runs through the chamfered edge's
/// vertex, and the trim shortens it at the contact's vertex. A closed spine
/// starts at its curve's first parameter, which need not be that vertex.
/// Both contacts are cut at one parameter and rejoined, and a walked band is
/// rebuilt from its rotated sides, so the band's straight row there stays
/// the band's seam. A face without a seam leaves the stripe as it is; two
/// seamed faces are refused.
fn start_at_seam(
    topo: &Topology,
    edge_id: EdgeId,
    stripe: &mut crate::stripe::Stripe,
) -> Result<(), BlendError> {
    const SAMPLES: usize = 256;
    let periodic = |face: FaceId| -> Result<Option<(f64, f64)>, BlendError> {
        let surface = topo.face(face)?.surface();
        Ok(match surface {
            FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => Some((0.0, std::f64::consts::TAU)),
            FaceSurface::Nurbs(n) if n.is_periodic_u() => Some(n.domain_u()),
            FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => None,
        })
    };
    let (face, side, (u0, u1)) = match (periodic(stripe.face1)?, periodic(stripe.face2)?) {
        (Some(range), None) => (stripe.face1, 0, range),
        (None, Some(range)) => (stripe.face2, 1, range),
        (None, None) => return Ok(()),
        (Some(_), Some(_)) => {
            return Err(BlendError::UnsupportedSurface {
                face: stripe.face1,
                surface_tag: "a closed chamfer between two seamed faces".to_owned(),
            });
        }
    };
    let period = u1 - u0;
    let surface = topo.face(face)?.surface().clone();
    let u_of = |p: Point3| surface.project_point(p).map_or(0.0, |(u, _)| u);
    let vertex = topo.vertex(topo.edge(edge_id)?.start())?.point();
    let seam = u_of(vertex);
    let contact = if side == 0 {
        &stripe.contact1
    } else {
        &stripe.contact2
    };
    let (c0, c1) = contact.domain();
    let at = |f: f64| c0 + f * (c1 - c0);
    // The contact's turn from the seam, in (-period/2, period/2].
    let off = |t: f64| {
        let d = (u_of(contact.evaluate(t)) - seam).rem_euclid(period);
        if d > 0.5 * period { d - period } else { d }
    };
    if off(c0).abs() <= 1e-12 * period {
        return Ok(());
    }
    // Where the turn crosses zero between two samples, not where it wraps.
    let mut bracket = None;
    for k in 0..SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let (a, b) = (
            at(k as f64 / SAMPLES as f64),
            at((k + 1) as f64 / SAMPLES as f64),
        );
        let (fa, fb) = (off(a), off(b));
        if fa == 0.0 {
            bracket = Some((a, a));
            break;
        }
        if fa.signum() != fb.signum() && (fa - fb).abs() < 0.5 * period {
            bracket = Some((a, b));
            break;
        }
    }
    let Some((mut lo, mut hi)) = bracket else {
        return Err(BlendError::TrimmingFailure { face });
    };
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if off(mid).signum() == off(lo).signum() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    stripe.contact1 = rotated(&stripe.contact1, t)?;
    stripe.contact2 = rotated(&stripe.contact2, t)?;
    if let FaceSurface::Nurbs(band) = &stripe.surface {
        stripe.surface = FaceSurface::Nurbs(brepkit_math::nurbs::surface::NurbsSurface::new(
            band.degree_u(),
            stripe.contact1.degree(),
            band.knots_u().to_vec(),
            stripe.contact1.knots().to_vec(),
            vec![
                stripe.contact1.control_points().to_vec(),
                stripe.contact2.control_points().to_vec(),
            ],
            vec![
                stripe.contact1.weights().to_vec(),
                stripe.contact2.weights().to_vec(),
            ],
        )?);
    }
    Ok(())
}

/// The closed curve `curve` starting at its parameter `t`: its run from `t`
/// to its end, then from its start to `t`, joined where the curve closes.
fn rotated(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    t: f64,
) -> Result<brepkit_math::nurbs::curve::NurbsCurve, BlendError> {
    let (t0, t1) = curve.domain();
    if t - t0 <= 1e-12 * (t1 - t0) || t1 - t <= 1e-12 * (t1 - t0) {
        return Ok(curve.clone());
    }
    let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(curve, t)?;
    let p = curve.degree();
    let (lk, rk) = (left.knots(), right.knots());
    let (lw, rw) = (left.weights(), right.weights());
    let (Some(&end_weight), Some(&start_weight)) = (rw.last(), lw.first()) else {
        return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
    };
    if rk.len() < p + 1 || lk.len() < p + 1 || start_weight <= 0.0 {
        return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
    }
    let shift = t1 - t0;
    let mut knots = rk[..rk.len() - (p + 1)].to_vec();
    knots.extend(std::iter::repeat_n(t1, p));
    knots.extend(lk[p + 1..].iter().map(|k| k + shift));
    let mut points = right.control_points().to_vec();
    points.extend_from_slice(&left.control_points()[1..]);
    let scale = end_weight / start_weight;
    let mut weights = rw.to_vec();
    weights.extend(lw[1..].iter().map(|w| w * scale));
    Ok(brepkit_math::nurbs::curve::NurbsCurve::new(
        p, knots, points, weights,
    )?)
}

/// Trims `face` along a stripe's `contact` in the face's own parameters,
/// keeping the side `keep` names. The contact's pcurve runs with the curve
/// that the contact edge stores: on a plane, a NURBS contact's control
/// points map into the plane's frame, and a circle or a contact on a curved
/// face takes [`pcurve_along`].
fn trim_along(
    topo: &mut Topology,
    scope: &trimmer::SplitScope,
    face: FaceId,
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
    keep: TrimKeep,
    spine_edges: &[EdgeId],
) -> Result<trimmer::BatchTrimResult, BlendError> {
    // The trimmer's weld distance: ends this close make one closed contact.
    const CLOSED: f64 = 1e-7;
    let surface = topo.face(face)?.surface().clone();
    let mut adapter = None;
    let support = crate::builder_utils::surface_ref_or_adapter(&surface, &mut adapter);
    let ends = sample_nurbs_endpoints(contact);
    // A contact that is a circle (an analytic band's) is stored as one, as
    // the faces' own rims are, so the meshes of the faces along it agree.
    let curve = as_circle(contact)
        .map_or_else(|| EdgeCurve::NurbsCurve(contact.clone()), EdgeCurve::Circle);
    let pcurve = match &curve {
        EdgeCurve::NurbsCurve(nurbs) if surface.is_planar() => {
            let (pcurve, start, end) =
                crate::fillet_builder::build_planar_pcurve_from_contact(support, nurbs)?;
            PCurve::new(pcurve, start, end)
        }
        EdgeCurve::Line
        | EdgeCurve::Circle(_)
        | EdgeCurve::Ellipse(_)
        | EdgeCurve::NurbsCurve(_) => {
            let start = ends[0];
            let end = if (ends[1] - start).length() <= CLOSED {
                start
            } else {
                ends[1]
            };
            let periods = [
                crate::fillet_builder::face_u_period(&surface),
                face_v_period(&surface),
            ];
            pcurve_along(support, periods, &curve, (start, end))
                .ok_or(BlendError::TrimmingFailure { face })?
        }
    };
    let mut restriction = trimmer::ParametricRestriction::new(ends, keep);
    restriction.source_edges = spine_edges.to_vec();
    restriction.curve = Some(curve);
    restriction.pcurve = Some(pcurve);
    trimmer::trim_parametric_face_batch(
        topo,
        scope,
        face,
        &[restriction],
        trimmer::BoundarySplitPolicy::All,
    )
}

/// The period of `surface` in v: the torus's, and a NURBS surface's that
/// closes round in v.
fn face_v_period(surface: &FaceSurface) -> Option<f64> {
    match surface {
        FaceSurface::Torus(_) => Some(std::f64::consts::TAU),
        FaceSurface::Nurbs(nurbs) if nurbs.is_periodic_v() => {
            let (v0, v1) = nurbs.domain_v();
            (v1 > v0).then_some(v1 - v0)
        }
        FaceSurface::Plane { .. }
        | FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Nurbs(_) => None,
    }
}

/// The pcurve on `support` of `curve` run from `start` to `end`, over the
/// curve's own parameters between them: one fraction of each span gives one
/// point. The image of the curve in the surface's parameters is read along
/// the curve, across the seams of `periods` (u, v). Where that image is a
/// straight line run at an even rate (a circle round a cylinder's or a
/// cone's axis), the pcurve is that line. Elsewhere a cubic interpolates the
/// image at more and more points, until between them it lies within `FIT`
/// of the image, in pieces that meet at the curve's corners. `None` when no
/// such cubic comes within it.
fn pcurve_along(
    support: &dyn ParametricSurface,
    periods: [Option<f64>; 2],
    curve: &EdgeCurve,
    (start, end): (Point3, Point3),
) -> Option<PCurve> {
    const FIT: f64 = 1e-7;
    const FIRST: u32 = 16;
    const LAST: u32 = 4096;
    let (t0, t1) = curve.domain_with_endpoints(start, end);
    let fit = FIT * (1.0 + start.x().abs().max(start.y().abs()).max(start.z().abs()));
    let on = |uv: Point2| support.evaluate(uv.x(), uv.y());
    let mut pieces: Vec<f64> = corners(curve)
        .into_iter()
        .map(|t| (t - t0) / (t1 - t0))
        .filter(|f| *f > 1e-9 && *f < 1.0 - 1e-9)
        .collect();
    pieces.sort_by(f64::total_cmp);
    pieces.insert(0, 0.0);
    pieces.push(1.0);
    let mut steps = FIRST;
    while steps <= LAST {
        let count = 2 * steps;
        let step = |k: u32| f64::from(k) / f64::from(count);
        let mut fractions = vec![0.0];
        for piece in pieces.windows(2) {
            fractions.extend((1..=count).map(|k| (piece[1] - piece[0]).madd(step(k), piece[0])));
        }
        let mut image: Vec<Point2> = Vec::with_capacity(fractions.len());
        for &f in &fractions {
            let point = curve.evaluate_with_endpoints((t1 - t0).madd(f, t0), start, end);
            let (mut u, mut v) = support.project_point(point);
            if let Some(last) = image.last() {
                for (value, (period, reference)) in [
                    (&mut u, (periods[0], last.x())),
                    (&mut v, (periods[1], last.y())),
                ] {
                    if let Some(period) = period {
                        *value -= period * ((*value - reference) / period).round();
                    }
                }
            }
            image.push(Point2::new(u, v));
        }
        let near = |at: Point2, k: usize| (on(at) - on(image[k])).length() <= fit;
        let (first, chord) = (image[0], image[image.len() - 1] - image[0]);
        if fractions
            .iter()
            .enumerate()
            .all(|(k, &f)| near(first + chord * f, k))
        {
            let line = Line2D::new(first, chord).ok()?;
            return Some(PCurve::new(Curve2D::Line(line), 0.0, chord.length()));
        }
        let params: Vec<f64> = (0..=steps)
            .map(|k| f64::from(k) / f64::from(steps))
            .collect();
        let flat = |p: Point3| Point2::new(p.x(), p.y());
        let mut knots = vec![0.0; 4];
        let mut points: Vec<Point2> = Vec::new();
        let mut within = true;
        for (index, piece) in pieces.windows(2).enumerate() {
            let base = index * count as usize;
            let run = &image[base..=base + count as usize];
            let nodes: Vec<Point3> = run
                .iter()
                .step_by(2)
                .map(|uv| Point3::new(uv.x(), uv.y(), 0.0))
                .collect();
            let cubic = brepkit_math::nurbs::fitting::interpolate_at(&nodes, &params, 3).ok()?;
            within &= (1..count)
                .step_by(2)
                .all(|k| near(flat(cubic.evaluate(step(k))), base + k as usize));
            let inner = &cubic.knots()[4..cubic.knots().len() - 4];
            knots.extend(
                inner
                    .iter()
                    .map(|&k| (piece[1] - piece[0]).madd(k, piece[0])),
            );
            knots.extend(std::iter::repeat_n(piece[1], 3));
            let skip = usize::from(index > 0);
            points.extend(cubic.control_points()[skip..].iter().copied().map(flat));
        }
        if within {
            knots.push(1.0);
            let weights = vec![1.0; points.len()];
            let pcurve = NurbsCurve2D::new(3, knots, points, weights).ok()?;
            return Some(PCurve::new(Curve2D::Nurbs(pcurve), 0.0, 1.0));
        }
        steps *= 2;
    }
    None
}

/// The parameters of `curve` where it may turn a corner: the knots of a
/// NURBS curve that reach its degree in number, inside its domain.
fn corners(curve: &EdgeCurve) -> Vec<f64> {
    let EdgeCurve::NurbsCurve(nurbs) = curve else {
        return Vec::new();
    };
    let (d0, d1) = nurbs.domain();
    let same = 1e-12 * (d1 - d0).abs();
    let knots = nurbs.knots();
    let mut found = Vec::new();
    let mut i = 0;
    while i < knots.len() {
        let knot = knots[i];
        let run = knots[i..]
            .iter()
            .take_while(|&&k| (k - knot).abs() <= same)
            .count();
        if run >= nurbs.degree() && knot - d0 > same && d1 - knot > same {
            found.push(knot);
        }
        i += run;
    }
    found
}

/// Turns `face`'s outer wire round where it runs its contact edges the way
/// their support faces do: each edge must run one way in its support and
/// the other in the chamfer face, read about each face's outward side.
fn run_against_supports(
    topo: &mut Topology,
    face: FaceId,
    contacts: [(FaceId, EdgeId); 2],
) -> Result<(), BlendError> {
    let runs = |topo: &Topology, face: FaceId, edge: EdgeId| -> Result<Option<bool>, BlendError> {
        let f = topo.face(face)?;
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            if let Some(oe) = topo.wire(wid)?.edges().iter().find(|oe| oe.edge() == edge) {
                return Ok(Some(oe.is_forward() != f.is_reversed()));
            }
        }
        Ok(None)
    };
    let mut against = [false; 2];
    for (slot, (support, edge)) in contacts.into_iter().enumerate() {
        let (Some(theirs), Some(ours)) = (runs(topo, support, edge)?, runs(topo, face, edge)?)
        else {
            return Err(BlendError::TrimmingFailure { face });
        };
        against[slot] = theirs != ours;
    }
    match against {
        [true, true] => Ok(()),
        [false, false] => {
            let wire = topo.wire(topo.face(face)?.outer_wire())?;
            let reversed: Vec<OrientedEdge> = wire
                .edges()
                .iter()
                .rev()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect();
            let wire = topo.add_wire(Wire::new(reversed, true)?);
            topo.face_mut(face)?.set_outer_wire(wire);
            Ok(())
        }
        _ => Err(BlendError::TrimmingFailure { face }),
    }
}

/// The circle `contact` runs on, starting at its start and running its way,
/// when every point of it lies on that circle; `None` for any other curve.
fn as_circle(
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
) -> Option<brepkit_math::curves::Circle3D> {
    const SAMPLES: u32 = 32;
    let (t0, t1) = contact.domain();
    let at = |f: f64| contact.evaluate(t0 + f * (t1 - t0));
    let (a, b, c) = (at(0.0), at(1.0 / 3.0), at(2.0 / 3.0));
    // The circumcircle of three of its points.
    let (ab, ac) = (b - a, c - a);
    let normal = ab.cross(ac);
    let n2 = normal.dot(normal);
    if n2 <= 1e-24 * ab.dot(ab) * ac.dot(ac) {
        return None;
    }
    let center =
        a + (normal.cross(ab) * ac.dot(ac) + ac.cross(normal) * ab.dot(ab)) * (1.0 / (2.0 * n2));
    let radius = (a - center).length();
    let tol = 1e-9 * radius.max(1.0);
    let axis = normal.normalize().ok()?;
    for k in 0..=SAMPLES {
        let p = at(f64::from(k) / f64::from(SAMPLES));
        if ((p - center).length() - radius).abs() > tol || axis.dot(p - center).abs() > tol {
            return None;
        }
    }
    // The circle runs about its normal from `a` the way the contact does.
    let ahead = contact.derivatives(t0, 1)[1];
    let normal = if (a - center).cross(ahead).dot(axis) >= 0.0 {
        axis
    } else {
        -axis
    };
    brepkit_math::curves::Circle3D::new_with_ref(
        center,
        normal,
        radius,
        (a - center).normalize().ok()?,
    )
    .ok()
}

/// The chamfer face of a closed stripe: its two closed contact edges and one
/// seam between their vertices, the band's straight row there, used once
/// each way. The band's outward side leans the way its two neighbours face,
/// and each contact runs against its neighbour's use of it, both read about
/// each face's outward side.
fn closed_chamfer_face(
    topo: &mut Topology,
    stripe: &crate::stripe::Stripe,
    contacts: [Option<EdgeId>; 2],
    supports: [FaceId; 2],
) -> Result<FaceId, BlendError> {
    const ON: f64 = 1e-6;
    let refuse = || BlendError::TrimmingFailure { face: stripe.face1 };
    let [Some(c1), Some(c2)] = contacts else {
        return Err(refuse());
    };
    let mut vertices = [None, None];
    for (slot, (edge, contact)) in [(c1, &stripe.contact1), (c2, &stripe.contact2)]
        .into_iter()
        .enumerate()
    {
        let e = topo.edge(edge)?;
        let at = topo.vertex(e.start())?.point();
        if e.start() != e.end() || (at - contact.evaluate(contact.domain().0)).length() > ON {
            return Err(refuse());
        }
        vertices[slot] = Some(e.start());
    }
    let [Some(v1), Some(v2)] = vertices else {
        return Err(refuse());
    };
    let seam = topo.add_edge(Edge::new(v1, v2, brepkit_topology::edge::EdgeCurve::Line));
    let loop_of = |flags: [bool; 2]| {
        vec![
            OrientedEdge::new(c1, flags[0]),
            OrientedEdge::new(seam, true),
            OrientedEdge::new(c2, flags[1]),
            OrientedEdge::new(seam, false),
        ]
    };
    let wire = topo.add_wire(Wire::new(loop_of([true, false]), true)?);
    let face = topo.add_face(Face::new(wire, Vec::new(), stripe.surface.clone()));
    orient_chamfer_face(topo, face, stripe)?;
    let band_reversed = topo.face(face)?.is_reversed();
    let mut flags = [true, true];
    for (slot, (support, edge)) in supports.into_iter().zip([c1, c2]).enumerate() {
        let f = topo.face(support)?;
        let mut theirs = None;
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            if let Some(oe) = topo.wire(wid)?.edges().iter().find(|oe| oe.edge() == edge) {
                theirs = Some(oe.is_forward() != f.is_reversed());
            }
        }
        let Some(theirs) = theirs else {
            return Err(refuse());
        };
        flags[slot] = theirs == band_reversed;
    }
    let wire = topo.add_wire(Wire::new(loop_of(flags), true)?);
    topo.face_mut(face)?.set_outer_wire(wire);
    Ok(face)
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
            if splice_inside_corner(topo, faces, cross)? {
                continue;
            }
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

/// A chamfer whose edge ends at an inside corner of one of its faces (an
/// L's floor, the edge along the L's arm) takes off a wedge whose end
/// bares a triangle of the end face beyond: the corner, the contact point
/// on that face, and the contact point on the other face. The trim left
/// that face running from the corner to its contact point, and the end
/// face from the corner up to its own, each edge now used once. The end
/// face, which holds the cross edge, swaps its edge for the face's edge and
/// the cross edge, taking the triangle. Returns whether it spliced.
fn splice_inside_corner(
    topo: &mut Topology,
    faces: &[FaceId],
    cross: &CrossEdge,
) -> Result<bool, BlendError> {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let joins = |eid: EdgeId, a: VertexId, b: VertexId| -> Result<bool, BlendError> {
        let e = topo.edge(eid)?;
        Ok((e.start() == a && e.end() == b) || (e.start() == b && e.end() == a))
    };
    // The edges from the corner to the two contact points, each used once:
    // one the face's (beside the triangle), one the end face's.
    let mut to_point: [Option<EdgeId>; 2] = [None, None];
    for (&eid, &n) in &uses {
        if n != 1 {
            continue;
        }
        for (slot, point) in [cross.from, cross.to].into_iter().enumerate() {
            if joins(eid, cross.corner, point)? {
                if to_point[slot].is_some() {
                    return Ok(false);
                }
                to_point[slot] = Some(eid);
            }
        }
    }
    let [Some(to_from), Some(to_to)] = to_point else {
        return Ok(false);
    };
    for (beside, near, far, end_edge) in [
        (to_from, cross.from, cross.to, to_to),
        (to_to, cross.to, cross.from, to_from),
    ] {
        if splice_end_face(topo, faces, cross, beside, near, far, end_edge)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Splices the triangle into the end face whose wire holds `end_edge`
/// (from the corner to `far`), on a plane holding the cross edge: the edge
/// gives way to `beside` (from the corner to `near`) and the cross edge.
fn splice_end_face(
    topo: &mut Topology,
    faces: &[FaceId],
    cross: &CrossEdge,
    beside: EdgeId,
    near: VertexId,
    far: VertexId,
    end_edge: EdgeId,
) -> Result<bool, BlendError> {
    // The end face's edge from the corner to the other point, used once,
    // on a plane that holds the cross edge.
    for &fid in faces {
        let face = topo.face(fid)?;
        let on_plane = |p: Point3| match face.surface() {
            FaceSurface::Plane { normal, d } => {
                let scale = 1.0 + p.x().abs().max(p.y().abs()).max(p.z().abs());
                (normal
                    .x()
                    .madd(p.x(), normal.y().madd(p.y(), normal.z() * p.z()))
                    - d)
                    .abs()
                    <= 1e-6 * scale
            }
            FaceSurface::Nurbs(_)
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => false,
        };
        if !on_plane(topo.vertex(cross.from)?.point()) || !on_plane(topo.vertex(cross.to)?.point())
        {
            continue;
        }
        let wires: Vec<WireId> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wid in wires {
            let edges = topo.wire(wid)?.edges().to_vec();
            let Some(i) = edges.iter().position(|oe| oe.edge() == end_edge) else {
                continue;
            };
            let e = topo.edge(edges[i].edge())?;
            let from_corner = edges[i].oriented_start(e) == cross.corner;
            let b = topo.edge(beside)?;
            let beside_out = OrientedEdge::new(beside, b.start() == cross.corner);
            let beside_in = OrientedEdge::new(beside, b.end() == cross.corner);
            // The cross edge runs `from` to `to`.
            let cross_from = |start: VertexId| OrientedEdge::new(cross.edge, start == cross.from);
            let replacement = if from_corner {
                [beside_out, cross_from(near)]
            } else {
                [cross_from(far), beside_in]
            };
            let mut spliced = edges[..i].to_vec();
            spliced.extend(replacement);
            spliced.extend_from_slice(&edges[i + 1..]);
            let new_wire = topo.add_wire(Wire::new(spliced, true)?);
            let face = topo.face_mut(fid)?;
            if face.outer_wire() == wid {
                face.set_outer_wire(new_wire);
            } else if let Some(slot) = face.inner_wires_mut().iter_mut().find(|w| **w == wid) {
                *slot = new_wire;
            }
            return Ok(true);
        }
    }
    Ok(false)
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

    if face1 == face2 {
        return Err(BlendError::UnsupportedSurface {
            face: face1,
            surface_tag: format!("{} seam (one face on both sides)", surf1.type_tag()),
        });
    }
    log::debug!(
        target: "brepkit_approx",
        "chamfer: analytic path unavailable for {}+{} — walking an approximate NURBS band",
        surf1.type_tag(),
        surf2.type_tag()
    );
    walk_chamfer_stripe(topo, edge_id, [face1, face2], spine, d1, d2)
}

/// A chamfer stripe walked along its edge where no closed form holds (a
/// slanted plane's ellipse on a cylinder): at each station of the spine,
/// each contact lies in the section square to the spine, on its face's
/// surface, its distance from the spine point (see [`ChamferBlend`]). The
/// band is ruled, straight across from contact to contact, and passes
/// through every walked section; its two sides are the contacts.
///
/// # Errors
///
/// Returns [`BlendError`] when the walk does not start or does not reach the
/// spine's end, or the band cannot be fitted.
fn walk_chamfer_stripe(
    topo: &Topology,
    edge_id: EdgeId,
    [face1, face2]: [FaceId; 2],
    spine: Spine,
    d1: f64,
    d2: f64,
) -> Result<StripeResult, BlendError> {
    let surf1 = topo.face(face1)?.surface().clone();
    let surf2 = topo.face(face2)?.surface().clone();
    let (mut adapter1, mut adapter2) = (None, None);
    let ps1 = crate::builder_utils::surface_ref_or_adapter(&surf1, &mut adapter1);
    let ps2 = crate::builder_utils::surface_ref_or_adapter(&surf2, &mut adapter2);
    let blend = ChamferBlend { d1, d2 };
    let walker = Walker::new(&blend, ps1, ps2, &spine, topo, WalkerConfig::default());

    // A chord meets a surface's section on either side of the spine: the
    // start takes the one into the face.
    let length = spine.length();
    let start_point = spine.evaluate(topo, 0.0)?;
    let tangent = spine.tangent(topo, 0.0)?;
    let guess =
        |ps: &dyn ParametricSurface, face: FaceId, d: f64| -> Result<(f64, f64), BlendError> {
            let into = into_face_at(topo, face, edge_id, &spine, start_point, tangent)?;
            Ok(ps.project_point(start_point + into * d))
        };
    let (u1, v1) = guess(ps1, face1, d1)?;
    let (u2, v2) = guess(ps2, face2, d2)?;
    let (start, _) = walker
        .solve_section(0.0, Some(BlendParams { u1, v1, u2, v2 }))
        .ok_or(BlendError::StartSolutionFailure {
            edge: edge_id,
            t: 0.0,
        })?;
    let walk = walker.walk(start, 0.0, length)?;
    let band = interpolate_chamfer_band(&walk.sections)?;
    let contact1 = crate::fillet_builder::band_side(&band, 0)?;
    let contact2 = crate::fillet_builder::band_side(&band, 1)?;
    let pcurve = |ps: &dyn ParametricSurface,
                  surf: &FaceSurface,
                  contact: &brepkit_math::nurbs::curve::NurbsCurve|
     -> Result<brepkit_math::curves2d::Curve2D, BlendError> {
        if surf.is_planar() {
            Ok(crate::fillet_builder::build_planar_pcurve_from_contact(ps, contact)?.0)
        } else {
            crate::fillet_builder::build_pcurve_from_contact(
                ps,
                contact,
                crate::fillet_builder::face_u_period(surf),
            )
        }
    };
    let pcurve1 = pcurve(ps1, &surf1, &contact1)?;
    let pcurve2 = pcurve(ps2, &surf2, &contact2)?;
    Ok(StripeResult {
        stripe: crate::stripe::Stripe {
            spine,
            surface: FaceSurface::Nurbs(band),
            pcurve1,
            pcurve2,
            contact1,
            contact2,
            face1,
            face2,
            sections: walk.sections,
        },
        new_edges: Vec::new(),
    })
}

/// The direction into `face` from the spine at `point`, square to the
/// spine's `tangent` in the face's tangent plane: the side
/// [`crate::builder_utils::into_face_from_edge`] reads at the edge's middle,
/// carried along the spine.
fn into_face_at(
    topo: &Topology,
    face: FaceId,
    edge: EdgeId,
    spine: &Spine,
    point: Point3,
    tangent: Vec3,
) -> Result<Vec3, BlendError> {
    let surface = topo.face(face)?.surface();
    let across = |p: Point3, t: Vec3| -> Result<Vec3, BlendError> {
        let (u, v) = surface.project_point(p).unwrap_or((0.0, 0.0));
        Ok(surface.normal(u, v).cross(t).normalize()?)
    };
    let half = 0.5 * spine.length();
    let middle = across(spine.evaluate(topo, half)?, spine.tangent(topo, half)?)?;
    let Some(into) = crate::builder_utils::into_face_from_edge(topo, face, edge)? else {
        return Err(BlendError::TrimmingFailure { face });
    };
    let here = across(point, tangent)?;
    Ok(if middle.dot(into) < 0.0 { -here } else { here })
}

/// Whether each end of an open stripe on a curved face lies on a plane that
/// holds its end section, which takes the chamfer's cross edge there: the
/// plane stands square to the edge.
fn ends_on_section_planes(
    topo: &Topology,
    faces: &[FaceId],
    edge_id: EdgeId,
    stripe: &crate::stripe::Stripe,
) -> Result<bool, BlendError> {
    let (Some(first), Some(last)) = (stripe.sections.first(), stripe.sections.last()) else {
        return Ok(false);
    };
    let start_point = stripe.spine.evaluate(topo, 0.0)?;
    let edge = topo.edge(edge_id)?;
    let near_start = (topo.vertex(edge.start())?.point() - start_point).length()
        <= (topo.vertex(edge.end())?.point() - start_point).length();
    let vertices = if near_start {
        [edge.start(), edge.end()]
    } else {
        [edge.end(), edge.start()]
    };
    for (vertex, section) in vertices.into_iter().zip([first, last]) {
        if !end_plane_holds(topo, faces, [stripe.face1, stripe.face2], vertex, section)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the one face through `vertex` other than the stripe's two is a
/// plane through both of `section`'s contacts.
fn end_plane_holds(
    topo: &Topology,
    faces: &[FaceId],
    stripe_faces: [FaceId; 2],
    vertex: VertexId,
    section: &crate::section::CircSection,
) -> Result<bool, BlendError> {
    let mut found = None;
    for &fid in faces {
        if stripe_faces.contains(&fid) {
            continue;
        }
        let face = topo.face(fid)?;
        let mut through = false;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                let e = topo.edge(oe.edge())?;
                through |= e.start() == vertex || e.end() == vertex;
            }
        }
        if through {
            if found.is_some() {
                return Ok(false);
            }
            found = Some(fid);
        }
    }
    let Some(end) = found else {
        return Ok(false);
    };
    let FaceSurface::Plane { normal, d } = *topo.face(end)?.surface() else {
        return Ok(false);
    };
    let on = |p: Point3| {
        let scale = 1.0 + p.x().abs().max(p.y().abs()).max(p.z().abs());
        (normal
            .x()
            .madd(p.x(), normal.y().madd(p.y(), normal.z() * p.z()))
            - d)
            .abs()
            <= 1e-7 * scale
    };
    Ok(on(section.p1) && on(section.p2))
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

    /// A closed curve started again at an inner parameter is the same curve,
    /// from the point there round to it.
    #[test]
    fn a_closed_curve_rotated_is_the_same_curve() {
        use brepkit_math::curves::Circle3D;
        let circle =
            Circle3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.6, 0.8), 2.5).unwrap();
        let curve =
            crate::analytic::circle_arc_to_nurbs(&circle, 0.0, std::f64::consts::TAU).unwrap();
        let (t0, t1) = curve.domain();
        for f in [0.13, 0.5, 0.77] {
            let t = t0 + f * (t1 - t0);
            let turned = rotated(&curve, t).unwrap();
            let (s0, s1) = turned.domain();
            assert!((turned.evaluate(s0) - curve.evaluate(t)).length() < 1e-12);
            assert!((turned.evaluate(s1) - curve.evaluate(t)).length() < 1e-12);
            for k in 0..=64 {
                let p = turned.evaluate(s0 + (s1 - s0) * f64::from(k) / 64.0);
                let off = ((p - circle.center()).length() - circle.radius()).abs()
                    + (p - circle.center()).dot(circle.normal()).abs();
                assert!(off < 1e-12, "{off:e} off the circle at {k}");
            }
            // Its points run the way the curve's do.
            let ahead =
                |c: &brepkit_math::nurbs::curve::NurbsCurve, at: f64| c.derivatives(at, 1)[1];
            assert!(ahead(&turned, s0).dot(ahead(&curve, t)) > 0.0);
        }
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
    }
}
