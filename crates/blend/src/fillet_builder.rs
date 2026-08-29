//! Fillet builder: orchestrates the full fillet pipeline.
//!
//! Spine construction, analytic/walking stripe computation, face trimming,
//! and solid assembly. Supports constant and variable radius fillets on
//! planar face pairs (v1).

use std::collections::HashSet;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::{Shell, ShellId};
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::analytic;
use crate::blend_func::{ConstRadBlend, EvolRadBlend};
use crate::boundary_registry::{BoundaryKey, BoundaryOwner, BoundaryRegistry, PlannedVertex};
use crate::builder_utils::{FlippedNormalSurface, sample_nurbs_endpoints, surface_ref_or_adapter};

use crate::corner;
use crate::fillet_plan::FilletPlan;
use crate::radius_law::RadiusLaw;
use crate::spine::Spine;
use crate::stripe::{Stripe, StripeResult};
use crate::trimmer;
use crate::walker::{Walker, WalkerConfig, approximate_blend_surface};
use crate::{BlendError, BlendResult};
type BlendCrossEdge = (EdgeId, VertexId, VertexId);
type BlendCrossPair = (Option<BlendCrossEdge>, Option<BlendCrossEdge>);
type BlendCrossHandlePair = (
    Option<crate::boundary_registry::BoundaryHandle>,
    Option<crate::boundary_registry::BoundaryHandle>,
);
/// Builder for fillet (rounding) operations on solid edges.
///
/// Collects edge sets with their radius laws, then computes and assembles
/// the filleted solid in a single `build()` call.
pub struct FilletBuilder<'a> {
    topo: &'a mut Topology,
    solid: SolidId,
    /// Edge sets to fillet, each with their radius/law.
    edge_sets: Vec<(Vec<EdgeId>, RadiusLaw)>,
}

impl<'a> FilletBuilder<'a> {
    /// Create a new fillet builder for the given solid.
    #[must_use]
    pub fn new(topo: &'a mut Topology, solid: SolidId) -> Self {
        Self {
            topo,
            solid,
            edge_sets: Vec::new(),
        }
    }

    /// Add edges to fillet with a constant radius.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_edges(&mut self, edges: &[EdgeId], radius: f64) -> &mut Self {
        self.edge_sets
            .push((edges.to_vec(), RadiusLaw::Constant(radius)));
        self
    }

    /// Add edges with variable radius law.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_edges_with_law(&mut self, edges: &[EdgeId], law: RadiusLaw) -> &mut Self {
        self.edge_sets.push((edges.to_vec(), law));
        self
    }

    /// Compute and build the filleted solid.
    ///
    /// # Algorithm
    ///
    /// 1. Build adjacency index for the solid.
    /// 2. Derive ordered multi-edge spines from the immutable G1 contour plan.
    /// 3. Compute each contour stripe via analytic fast path or walking engine.
    /// 4. Trim adjacent faces along contact curves.
    /// 5. Assemble new solid from trimmed faces, blend faces, and untouched
    ///    original faces.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] if no edges were specified, or if topology
    /// lookups fail. Individual edge failures are recorded in
    /// [`BlendResult::failed`] rather than aborting the whole operation.
    pub fn build(self) -> Result<BlendResult, BlendError> {
        if self.edge_sets.iter().all(|(edges, _)| edges.is_empty()) {
            return Err(BlendError::Topology(
                brepkit_topology::TopologyError::Empty {
                    entity: "fillet edge set",
                },
            ));
        }
        let Self {
            topo,
            solid,
            edge_sets,
        } = self;
        let snapshot = SolidSnapshot::capture(topo, solid)?;
        let result = build_in_place(topo, solid, edge_sets);
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

    fn build_in_place(self) -> Result<BlendResult, BlendError> {
        self.build_in_place_with_forced_postassembly_failure(false)
    }

    #[cfg(test)]
    fn build_with_forced_postassembly_failure(self) -> Result<BlendResult, BlendError> {
        self.build_in_place_with_forced_postassembly_failure(true)
    }

    #[allow(clippy::too_many_lines)]
    fn build_in_place_with_forced_postassembly_failure(
        self,
        force_postassembly_failure: bool,
    ) -> Result<BlendResult, BlendError> {
        // Build the immutable plan before any result topology is allocated.
        // Every valid request is represented by one ordered contour; the
        // contour spine is then used for one coherent stripe computation.
        let plan = match FilletPlan::build(self.topo, self.solid, &self.edge_sets) {
            Ok(plan) => Some(plan),
            Err(BlendError::PlanningFailure { reason })
                if reason.starts_with("unsupported ordered junction valence") =>
            {
                return Err(BlendError::PlanningFailure { reason });
            }
            Err(BlendError::PlanningFailure { .. }) => None,
            Err(error) => return Err(error),
        };

        // Keep actual RadiusLaw values only for the recoverable invalid-edge
        // fallback. Valid geometry always consumes the immutable plan's law.
        let mut all_edges: Vec<(EdgeId, usize)> = Vec::new();
        let mut laws: Vec<RadiusLaw> = Vec::with_capacity(self.edge_sets.len());
        for (law_idx, (edges, law)) in self.edge_sets.into_iter().enumerate() {
            for eid in edges {
                all_edges.push((eid, law_idx));
            }
            laws.push(law);
        }
        if all_edges.is_empty() {
            return Err(BlendError::Topology(
                brepkit_topology::TopologyError::Empty {
                    entity: "fillet edge set",
                },
            ));
        }

        let topo = self.topo;
        let adjacency = topo.build_adjacency(self.solid)?;
        let shell_id = topo.solid(self.solid)?.outer_shell();
        let original_faces: Vec<FaceId> = topo.shell(shell_id)?.faces().to_vec();
        let mut touched_faces: HashSet<FaceId> = HashSet::new();
        let mut failed: Vec<(EdgeId, BlendError)> = Vec::new();
        let mut stripe_results: Vec<StripeResult> = Vec::new();

        if let Some(plan) = plan.as_ref() {
            for contour in &plan.contours {
                match compute_stripe_for_contour(topo, &adjacency, contour) {
                    Ok(sr) => {
                        touched_faces.insert(sr.stripe.face1);
                        touched_faces.insert(sr.stripe.face2);
                        stripe_results.push(sr);
                    }
                    Err(error) => {
                        let message = error.to_string();
                        for &edge_id in &contour.edges {
                            failed.push((
                                edge_id,
                                BlendError::PlanningFailure {
                                    reason: format!("contour stripe failed: {message}"),
                                },
                            ));
                        }
                    }
                }
            }
        } else {
            let mut seen = HashSet::new();
            let duplicate_edges: HashSet<usize> = all_edges
                .iter()
                .filter(|(edge_id, _)| {
                    all_edges
                        .iter()
                        .filter(|(other, _)| other.index() == edge_id.index())
                        .nth(1)
                        .is_some()
                })
                .map(|(edge_id, _)| edge_id.index())
                .collect();
            for &(edge_id, law_idx) in &all_edges {
                if duplicate_edges.contains(&edge_id.index()) {
                    if seen.insert(edge_id.index()) {
                        failed.push((
                            edge_id,
                            BlendError::PlanningFailure {
                                reason: "edge requested more than once".to_owned(),
                            },
                        ));
                    }
                    continue;
                }
                if !seen.insert(edge_id.index()) {
                    continue;
                }
                match compute_stripe_for_edge(topo, &adjacency, edge_id, &laws[law_idx]) {
                    Ok(sr) => {
                        touched_faces.insert(sr.stripe.face1);
                        touched_faces.insert(sr.stripe.face2);
                        stripe_results.push(sr);
                    }
                    Err(error) => failed.push((edge_id, error)),
                }
            }
        }
        if stripe_results.is_empty() {
            return Ok(BlendResult {
                solid: self.solid,
                succeeded: Vec::new(),
                failed,
                is_partial: false,
            });
        }

        // Partition out closed-revolution rim stripes (a full circular rim
        // between a bounded disc cap and a cylinder/cone wall). These need an
        // annular assembly that rebuilds the cap, shortens the wall, and emits
        // a toroidal band — all sharing the two contact-circle edges — which
        // the per-face line-based trimmer cannot produce (a closed interior
        // contact circle crosses no boundary edge). Regular stripes still flow
        // through the trim + corner + blend-face path below.
        let mut boundary_registry = BoundaryRegistry::new();
        let mut blend_face_ids: Vec<FaceId> = Vec::new();
        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();
        let mut regular_results: Vec<&StripeResult> = Vec::new();

        for (stripe_index, sr) in stripe_results.iter().enumerate() {
            if let Some(rim) = closed_rim_info(topo, &sr.stripe)? {
                let contour_id =
                    plan.as_ref()
                        .and_then(|fillet_plan| {
                            fillet_plan.contours.iter().position(|contour| {
                                contour.spine.edges() == sr.stripe.spine_edges()
                            })
                        })
                        .unwrap_or(stripe_index);
                match assemble_closed_rim(
                    topo,
                    &sr.stripe,
                    &rim,
                    contour_id,
                    &mut boundary_registry,
                    &mut face_replacements,
                ) {
                    Ok(band) => blend_face_ids.push(band),
                    Err(e) => {
                        // Closed-rim geometry is authoritative; do not hide
                        // an assembly failure behind a sequential fallback.
                        return Err(e);
                    }
                }
            } else {
                regular_results.push(sr);
            }
        }

        // Planned contours are reconstructed once by the mapped batch
        // trimmer. The legacy loop builder remains only for the recoverable
        // unplanned fallback until the production cutover.
        let (rim_contact_edges, rim_notches) = if plan.is_some() {
            (std::collections::HashMap::new(), Vec::new())
        } else {
            rebuild_closed_rim_loop_faces(topo, &regular_results, &mut face_replacements)?
        };

        let stripes: Vec<Stripe> = regular_results.iter().map(|sr| sr.stripe.clone()).collect();
        let contour_to_stripe: Vec<Option<usize>> = plan
            .as_ref()
            .map(|fillet_plan| {
                fillet_plan
                    .contours
                    .iter()
                    .map(|contour| {
                        regular_results
                            .iter()
                            .position(|sr| sr.stripe.spine_edges() == contour.spine.edges())
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut corner_face_ids: Vec<FaceId> = Vec::new();

        let mut stripe_contact_handles: Vec<(
            Option<crate::boundary_registry::BoundaryHandle>,
            Option<crate::boundary_registry::BoundaryHandle>,
        )> = regular_results.iter().map(|_| (None, None)).collect();
        let mut stripe_contact_edges: Vec<(
            Option<brepkit_topology::edge::EdgeId>,
            Option<brepkit_topology::edge::EdgeId>,
        )> = regular_results
            .iter()
            .enumerate()
            .map(|(si, sr)| {
                (
                    rim_contact_edges.get(&(sr.stripe.face1, si)).copied(),
                    rim_contact_edges.get(&(sr.stripe.face2, si)).copied(),
                )
            })
            .collect();

        // Collect all parametric restrictions first. Each support face is then
        // rebuilt exactly once, independent of stripe input order.
        let mut restrictions_by_face: std::collections::HashMap<
            FaceId,
            Vec<(usize, trimmer::ParametricRestriction)>,
        > = std::collections::HashMap::new();
        for (si, sr) in regular_results.iter().enumerate() {
            let stripe = &sr.stripe;
            let spine_pt = stripe.spine.evaluate(topo, 0.0)?;
            let keep = trimmer::TrimKeep::AwayFrom(spine_pt);
            for (face_id, contact) in [
                (stripe.face1, stripe.contact1.clone()),
                (stripe.face2, stripe.contact2.clone()),
            ] {
                if rim_contact_edges.contains_key(&(face_id, si)) {
                    continue;
                }
                let mut restriction =
                    trimmer::ParametricRestriction::new(sample_nurbs_endpoints(&contact), keep);
                restriction.source_edges = stripe.spine_edges().to_vec();
                let source_surface = topo.face(face_id)?.surface().clone();
                let mut adapter = None;
                let support = surface_ref_or_adapter(&source_surface, &mut adapter);
                let analytic_pcurve = if face_id == stripe.face1 {
                    &stripe.pcurve1
                } else {
                    &stripe.pcurve2
                };
                let (pcurve, pcurve_start, pcurve_end) = if source_surface.is_planar() {
                    build_planar_pcurve_from_contact(support, &contact)?
                } else {
                    match build_pcurve_from_contact(
                        support,
                        &contact,
                        face_u_period(&source_surface),
                    ) {
                        Ok(pcurve) => (pcurve, 0.0, 1.0),
                        Err(BlendError::Math(brepkit_math::MathError::ZeroVector)) => {
                            match analytic_pcurve {
                                brepkit_math::curves2d::Curve2D::Circle(_)
                                | brepkit_math::curves2d::Curve2D::Ellipse(_) => {
                                    (analytic_pcurve.clone(), 0.0, std::f64::consts::TAU)
                                }
                                brepkit_math::curves2d::Curve2D::Nurbs(curve) => {
                                    let (start, end) = curve.domain();
                                    (analytic_pcurve.clone(), start, end)
                                }
                                brepkit_math::curves2d::Curve2D::Line(_) => {
                                    return Err(BlendError::Math(
                                        brepkit_math::MathError::ZeroVector,
                                    ));
                                }
                            }
                        }
                        Err(error) => return Err(error),
                    }
                };
                restriction.curve = Some(EdgeCurve::NurbsCurve(contact));
                restriction.pcurve = Some(brepkit_topology::pcurve::PCurve::new(
                    pcurve,
                    pcurve_start,
                    pcurve_end,
                ));
                restrictions_by_face
                    .entry(face_id)
                    .or_default()
                    .push((si, restriction));
            }
        }
        let mut restriction_faces: Vec<FaceId> = restrictions_by_face.keys().copied().collect();
        restriction_faces.sort_unstable_by_key(|face| face.index());
        for face_id in restriction_faces {
            let restrictions = restrictions_by_face
                .remove(&face_id)
                .ok_or(BlendError::TrimmingFailure { face: face_id })?;
            let restriction_plan: Vec<trimmer::ParametricRestriction> = restrictions
                .iter()
                .map(|(_, restriction)| restriction.clone())
                .collect();
            let batch = trimmer::trim_parametric_face_batch(topo, face_id, &restriction_plan)?;
            face_replacements.insert(face_id, batch.trimmed_face);
            for &(source, replacement) in &batch.incident_replacements {
                face_replacements.entry(source).or_insert(replacement);
            }
            for (index, &(si, _)) in restrictions.iter().enumerate() {
                let Some(&contact_edge) = batch.contact_edges.get(index) else {
                    return Err(BlendError::TrimmingFailure { face: face_id });
                };
                let stripe = &regular_results[si].stripe;
                let support_forward = std::iter::once(topo.face(batch.trimmed_face)?.outer_wire())
                    .chain(topo.face(batch.trimmed_face)?.inner_wires().iter().copied())
                    .find_map(|wire_id| {
                        topo.wire(wire_id)
                            .ok()?
                            .edges()
                            .iter()
                            .find_map(|oriented| {
                                (oriented.edge() == contact_edge).then_some(oriented.is_forward())
                            })
                    })
                    .ok_or(BlendError::TrimmingFailure { face: face_id })?;
                let contour_id = plan
                    .as_ref()
                    .and_then(|fillet_plan| {
                        fillet_plan
                            .contours
                            .iter()
                            .position(|contour| contour.spine.edges() == stripe.spine_edges())
                    })
                    .unwrap_or(si);
                let side = if stripe.face1 == face_id { 0 } else { 1 };
                let edge = topo.edge(contact_edge)?;
                let start = edge.start();
                let end = edge.end();
                let curve = restrictions[index]
                    .1
                    .curve
                    .clone()
                    .unwrap_or(EdgeCurve::Line);
                let start_point = topo.vertex(start)?.point();
                let end_point = topo.vertex(end)?.point();
                let periodic_support = matches!(
                    topo.face(face_id)?.surface(),
                    FaceSurface::Cylinder(_)
                        | FaceSurface::Cone(_)
                        | FaceSurface::Torus(_)
                        | FaceSurface::Sphere(_)
                ) || matches!(
                    topo.face(face_id)?.surface(),
                    FaceSurface::Nurbs(surface) if surface.is_periodic_u()
                );
                let start_vertex = if periodic_support {
                    PlannedVertex::periodic(start, (contour_id as u64) << 32 | (side as u64) << 8)
                } else {
                    PlannedVertex::new(start)
                };
                let end_vertex = if periodic_support {
                    PlannedVertex::periodic(end, (contour_id as u64) << 32 | (side as u64) << 8 | 1)
                } else {
                    PlannedVertex::new(end)
                };
                let handle = boundary_registry.register(
                    BoundaryKey::contact(contour_id, 0, side),
                    start_vertex,
                    end_vertex,
                    curve.clone(),
                    curve.domain_with_endpoints(start_point, end_point),
                    [
                        BoundaryOwner::planned(
                            format!("support face {face_id:?} contour {contour_id}"),
                            support_forward,
                        ),
                        BoundaryOwner::planned(
                            format!("blend contour {contour_id} contact {side}"),
                            side == 0,
                        ),
                    ],
                )?;
                boundary_registry.bind_existing_edge(topo, handle, contact_edge)?;
                if side == 0 {
                    stripe_contact_handles[si].0 = Some(handle);
                    stripe_contact_edges[si].0 = Some(contact_edge);
                } else {
                    stripe_contact_handles[si].1 = Some(handle);
                    stripe_contact_edges[si].1 = Some(contact_edge);
                }
            }
        }

        let mut blend_cross_edges: Vec<BlendCrossEdge> = Vec::new();
        let mut blend_cross_by_stripe: Vec<BlendCrossPair> =
            vec![(None, None); regular_results.len()];
        let mut blend_cross_handles: Vec<BlendCrossHandlePair> =
            vec![(None, None); regular_results.len()];
        for (si, sr) in regular_results.iter().enumerate() {
            let stripe = &sr.stripe;

            // Every batched planar contact is already registered and bound to
            // the support replacement. Build cross-section boundaries from
            // the same registry whenever both contacts belong to this
            // stripe. An unresolved second owner is intentional here: BK7's
            // junction/runout stage will attach it without minting a twin.
            let info = if let (Some(contact1), Some(contact2)) = stripe_contact_handles[si] {
                let contour_id = plan
                    .as_ref()
                    .and_then(|fillet_plan| {
                        fillet_plan
                            .contours
                            .iter()
                            .position(|contour| contour.spine.edges() == stripe.spine_edges())
                    })
                    .unwrap_or(si);
                let support_faces = [
                    face_replacements
                        .get(&stripe.face1)
                        .copied()
                        .unwrap_or(stripe.face1),
                    face_replacements
                        .get(&stripe.face2)
                        .copied()
                        .unwrap_or(stripe.face2),
                ];
                let terminals = ordered_spine_endpoints(topo, &stripe.spine)?;
                if let Some(forward) = contact_forward_for_spine(
                    topo,
                    &boundary_registry,
                    contact1,
                    support_faces[0],
                    terminals,
                    true,
                )? {
                    boundary_registry.set_owner_forward(contact1, 1, forward)?;
                }
                if let Some(forward) = contact_forward_for_spine(
                    topo,
                    &boundary_registry,
                    contact2,
                    support_faces[1],
                    terminals,
                    false,
                )? {
                    boundary_registry.set_owner_forward(contact2, 1, forward)?;
                }
                let contact_vertices =
                    |handle: crate::boundary_registry::BoundaryHandle| -> Result<
                        (VertexId, VertexId),
                        BlendError,
                    > {
                        let entry =
                            boundary_registry.entry(handle).ok_or_else(|| {
                                BlendError::PlanningFailure {
                                    reason: format!("unknown contact boundary {handle}"),
                                }
                            })?;
                        let (start, end) = (entry.start.vertex, entry.end.vertex);
                        Ok(if entry.owners[1].forward {
                            (start, end)
                        } else {
                            (end, start)
                        })
                    };
                let (p1_start, p1_end) = contact_vertices(contact1)?;
                let (p2_start, p2_end) = contact_vertices(contact2)?;
                let mut section_curve = |section: Option<&crate::section::CircSection>,
                                         start: VertexId,
                                         end: VertexId|
                 -> Result<
                    Option<crate::boundary_registry::BoundaryHandle>,
                    BlendError,
                > {
                    if start == end {
                        return Ok(None);
                    }
                    let start_point = topo.vertex(start)?.point();
                    let end_point = topo.vertex(end)?.point();
                    let curve = section
                        .and_then(|section| {
                            crate::builder_utils::cross_section_curve(
                                section,
                                start_point,
                                end_point,
                            )
                        })
                        .unwrap_or(EdgeCurve::Line);
                    let key = BoundaryKey::cross_section(
                        contour_id,
                        usize::from(start == p2_start || start == p2_end),
                        0,
                    );
                    let handle = boundary_registry.register(
                        key,
                        PlannedVertex::new(start),
                        PlannedVertex::new(end),
                        curve.clone(),
                        curve.domain_with_endpoints(start_point, end_point),
                        [
                            BoundaryOwner::planned(
                                format!("blend contour {contour_id} cross-section"),
                                true,
                            ),
                            BoundaryOwner::planned(
                                format!("junction/runout contour {contour_id}"),
                                false,
                            ),
                        ],
                    )?;
                    boundary_registry.defer_owner(handle, 1)?;
                    Ok(Some(handle))
                };
                if stripe.spine.is_closed() {
                    // Periodic closed-rim stripes: the two cross-sections are
                    // the same seam traversed twice. Build the band with one
                    // shared seam edge (canonical representation) instead of
                    // two twin arcs that would each keep a single face use.
                    let info = crate::builder_utils::create_periodic_blend_face(
                        topo,
                        stripe,
                        &mut boundary_registry,
                        contact1,
                        contact2,
                    )?;
                    boundary_registry.set_owner_face(contact1, 1, info.face)?;
                    boundary_registry.set_owner_face(contact2, 1, info.face)?;
                    blend_cross_handles[si] = (None, None);
                    info
                } else {
                    let cross_start = section_curve(stripe.sections.first(), p2_end, p1_start)?;
                    let cross_end = section_curve(stripe.sections.last(), p1_end, p2_start)?;
                    let info = crate::builder_utils::create_blend_face_from_registry(
                        topo,
                        stripe,
                        &mut boundary_registry,
                        (contact1, 1),
                        (contact2, 1),
                        cross_end.map(|handle| (handle, 0)),
                        cross_start.map(|handle| (handle, 0)),
                    )?;
                    for handle in [cross_end, cross_start].into_iter().flatten() {
                        boundary_registry.set_owner_face(handle, 0, info.face)?;
                    }
                    boundary_registry.set_owner_face(contact1, 1, info.face)?;
                    boundary_registry.set_owner_face(contact2, 1, info.face)?;
                    blend_cross_handles[si] = (cross_end, cross_start);
                    info
                }
            } else {
                let (c1, c2) = stripe_contact_edges
                    .get(si)
                    .copied()
                    .unwrap_or((None, None));
                crate::builder_utils::create_blend_face_with_contacts(topo, stripe, c1, c2)?
            };
            // Adopt loop-rebuild edges into the registry. This is needed for
            // curved/variable stripes where the batch trimmer did not emit a
            // contact handle, and avoids any positional closure repair.
            let contour_id = plan
                .as_ref()
                .and_then(|fillet_plan| {
                    fillet_plan
                        .contours
                        .iter()
                        .position(|contour| contour.spine.edges() == stripe.spine_edges())
                })
                .unwrap_or(si);
            let support_faces = [
                face_replacements
                    .get(&stripe.face1)
                    .copied()
                    .unwrap_or(stripe.face1),
                face_replacements
                    .get(&stripe.face2)
                    .copied()
                    .unwrap_or(stripe.face2),
            ];
            for (side, contact_edge) in [stripe_contact_edges[si].0, stripe_contact_edges[si].1]
                .into_iter()
                .enumerate()
            {
                let existing_handle = match side {
                    0 => stripe_contact_handles[si].0,
                    1 => stripe_contact_handles[si].1,
                    _ => unreachable!("support side is always 0 or 1"),
                };
                if existing_handle.is_some() {
                    continue;
                }
                let Some(contact_edge) = contact_edge else {
                    continue;
                };
                let edge_data = topo.edge(contact_edge)?.clone();
                let support_forward = face_edge_forward(topo, support_faces[side], contact_edge)
                    .ok_or(BlendError::TrimmingFailure {
                        face: support_faces[side],
                    })?;
                let blend_forward =
                    face_edge_forward(topo, info.face, contact_edge).unwrap_or(true);
                let handle = boundary_registry.register(
                    BoundaryKey::contact(contour_id, 0, side as u8),
                    PlannedVertex::new(edge_data.start()),
                    PlannedVertex::new(edge_data.end()),
                    edge_data.curve().clone(),
                    edge_data.curve().domain_with_endpoints(
                        topo.vertex(edge_data.start())?.point(),
                        topo.vertex(edge_data.end())?.point(),
                    ),
                    [
                        BoundaryOwner::new(
                            support_faces[side],
                            "rim support contact",
                            support_forward,
                        ),
                        BoundaryOwner::new(info.face, "rim blend contact", blend_forward),
                    ],
                )?;
                boundary_registry.bind_existing_edge(topo, handle, contact_edge)?;
                let _ = boundary_registry.oriented_edge(topo, handle, 1)?;
                match side {
                    0 => stripe_contact_handles[si].0 = Some(handle),
                    1 => stripe_contact_handles[si].1 = Some(handle),
                    _ => unreachable!("support side is always 0 or 1"),
                }
            }
            for (cross_side, cross) in [info.cross_end, info.cross_start].into_iter().enumerate() {
                let existing_handle = match cross_side {
                    0 => blend_cross_handles[si].0,
                    1 => blend_cross_handles[si].1,
                    _ => unreachable!("cross side is always 0 or 1"),
                };
                if existing_handle.is_some() {
                    continue;
                }
                let Some((cross_edge, _from, _to)) = cross else {
                    continue;
                };
                let edge_data = topo.edge(cross_edge)?.clone();
                let blend_forward = face_edge_forward(topo, info.face, cross_edge).unwrap_or(true);
                let handle = boundary_registry.register(
                    BoundaryKey::cross_section(contour_id, cross_side, 0),
                    PlannedVertex::new(edge_data.start()),
                    PlannedVertex::new(edge_data.end()),
                    edge_data.curve().clone(),
                    edge_data.curve().domain_with_endpoints(
                        topo.vertex(edge_data.start())?.point(),
                        topo.vertex(edge_data.end())?.point(),
                    ),
                    [
                        BoundaryOwner::new(info.face, "rim blend cross-section", blend_forward),
                        BoundaryOwner::planned("ordered junction or runout", false),
                    ],
                )?;
                boundary_registry.defer_owner(handle, 1)?;
                boundary_registry.bind_existing_edge(topo, handle, cross_edge)?;
                let _ = boundary_registry.oriented_edge(topo, handle, 0)?;
                match cross_side {
                    0 => blend_cross_handles[si].0 = Some(handle),
                    1 => blend_cross_handles[si].1 = Some(handle),
                    _ => unreachable!("cross side is always 0 or 1"),
                }
            }
            blend_face_ids.push(info.face);
            blend_cross_by_stripe[si] = (info.cross_end, info.cross_start);
            blend_cross_edges.extend(info.cross_end);
            blend_cross_edges.extend(info.cross_start);
        }

        // The ordered junction solver needs the final copy-on-write support
        // faces so residual terminal edges can be registered as runout
        // boundaries instead of being closed by positional repair.
        let stripe_support_faces: Vec<(FaceId, FaceId)> = stripes
            .iter()
            .map(|stripe| {
                (
                    face_replacements
                        .get(&stripe.face1)
                        .copied()
                        .unwrap_or(stripe.face1),
                    face_replacements
                        .get(&stripe.face2)
                        .copied()
                        .unwrap_or(stripe.face2),
                )
            })
            .collect();
        if let Some(fillet_plan) = plan.as_ref() {
            let corner_results = corner::compute_ordered_corners(
                topo,
                &stripes,
                &fillet_plan.junctions,
                &contour_to_stripe,
                &blend_cross_handles,
                &stripe_support_faces,
                &mut boundary_registry,
            )?;
            corner_face_ids.extend(corner_results.iter().map(|result| result.face_id));
        }
        // Faces using each vertex, over the ORIGINAL shell: a stripe end
        // whose outline vertex belongs to a third face (a perpendicular
        // end face) is closed by the notch-arc path, not by a cap.

        // Each abrupt stripe end that produced notch edges on BOTH adjacent
        // faces gets a cap whose terminal arc reuses the exact cross edge
        // already consumed by the blend wall. This keeps both uses identical
        // without a positional weld.
        for (si, sr) in regular_results.iter().enumerate() {
            let stripe = &sr.stripe;
            let ends: [Option<&crate::section::CircSection>; 2] =
                [stripe.sections.first(), stripe.sections.last()];
            for (end_index, sec) in ends
                .into_iter()
                .enumerate()
                .filter_map(|(index, section)| section.map(|section| (index, section)))
            {
                let pair: Vec<&NotchRecord> = rim_notches
                    .iter()
                    .filter(|nr| {
                        nr.stripe == si
                            && ((nr.contact_pt - sec.p1).length() < 1e-6
                                || (nr.contact_pt - sec.p2).length() < 1e-6)
                    })
                    .collect();
                if pair.len() != 2 || pair[0].outline_vid != pair[1].outline_vid {
                    continue;
                }
                let (na, nb) = if (pair[0].contact_pt - sec.p1).length() < 1e-6 {
                    (pair[0], pair[1])
                } else {
                    (pair[1], pair[0])
                };
                let outline_pt = topo.vertex(na.outline_vid)?.point();
                let n_raw = (sec.p1 - outline_pt).cross(sec.p2 - outline_pt);
                let Ok(plane_n) = n_raw.normalize() else {
                    continue;
                };
                let fwd_of = |topo: &Topology,
                              eid: EdgeId,
                              from: brepkit_topology::vertex::VertexId|
                 -> Result<bool, BlendError> {
                    Ok(topo.edge(eid)?.start() == from)
                };
                let existing_arc = match end_index {
                    0 => blend_cross_by_stripe[si].1,
                    1 => blend_cross_by_stripe[si].0,
                    _ => None,
                }
                .filter(|(_, from, to)| {
                    (*from == na.contact_vid && *to == nb.contact_vid)
                        || (*from == nb.contact_vid && *to == na.contact_vid)
                });
                let potential_registry_arc = match end_index {
                    0 => blend_cross_handles[si].1,
                    1 => blend_cross_handles[si].0,
                    _ => None,
                };
                if potential_registry_arc.is_some_and(|handle| {
                    boundary_registry
                        .entry(handle)
                        .is_some_and(|entry| entry.owners[1].face.is_some())
                }) {
                    continue;
                }
                let registry_arc = potential_registry_arc.filter(|&handle| {
                    boundary_registry
                        .entry(handle)
                        .and_then(crate::boundary_registry::BoundaryEntry::edge_id)
                        .and_then(|edge| topo.edge(edge).ok())
                        .is_some_and(|edge| {
                            (edge.start() == na.contact_vid && edge.end() == nb.contact_vid)
                                || (edge.start() == nb.contact_vid && edge.end() == na.contact_vid)
                        })
                });
                let mut registry_arc_use = None;
                let (arc, arc_forward) = if let Some(handle) = registry_arc {
                    let oriented = boundary_registry.oriented_edge(topo, handle, 1)?;
                    let edge = topo.edge(oriented.edge())?;
                    registry_arc_use = Some(handle);
                    (
                        oriented.edge(),
                        (edge.start() == na.contact_vid && edge.end() == nb.contact_vid)
                            == oriented.is_forward(),
                    )
                } else if let Some((arc, from, to)) = existing_arc {
                    (arc, from == na.contact_vid && to == nb.contact_vid)
                } else {
                    let Ok(circle_n) = (sec.p1 - sec.center).cross(sec.p2 - sec.center).normalize()
                    else {
                        continue;
                    };
                    let Ok(circle) = Circle3D::new(sec.center, circle_n, sec.radius) else {
                        continue;
                    };
                    (
                        topo.add_edge(Edge::new(
                            na.contact_vid,
                            nb.contact_vid,
                            EdgeCurve::Circle(circle),
                        )),
                        true,
                    )
                };
                let oe1 = OrientedEdge::new(na.edge, fwd_of(topo, na.edge, na.outline_vid)?);
                let oe2 = OrientedEdge::new(arc, arc_forward);
                let oe3 = OrientedEdge::new(nb.edge, fwd_of(topo, nb.edge, nb.contact_vid)?);
                let Ok(wire) = Wire::new(vec![oe1, oe2, oe3], true) else {
                    continue;
                };
                let wid = topo.add_wire(wire);
                let d = plane_n.dot(Vec3::new(outline_pt.x(), outline_pt.y(), outline_pt.z()));
                let cap = topo.add_face(Face::new(
                    wid,
                    Vec::new(),
                    FaceSurface::Plane { normal: plane_n, d },
                ));
                if let Some(handle) = registry_arc_use {
                    boundary_registry.set_owner_face(handle, 1, cap)?;
                }
                blend_face_ids.push(cap);
                log::debug!("stripe {si} end cap built at {outline_pt:?}");
            }
        }

        // Notch the fillet's end cross-section arcs out of the faces that
        // still cover the scooped corner (the untouched end caps): replace
        // each cap's two-edge corner path with the blend's own cross edge so
        // both sides share one edge entity.

        for arc in &blend_cross_edges {
            let corner_owned = boundary_registry
                .handle_for_edge(arc.0)
                .and_then(|handle| boundary_registry.entry(handle))
                .and_then(|entry| entry.owners[1].face)
                .is_some();
            if corner_owned {
                continue;
            }
            let candidates: Vec<(FaceId, FaceId)> = original_faces
                .iter()
                .map(|&f| (f, face_replacements.get(&f).copied().unwrap_or(f)))
                .collect();
            for (orig, fid) in candidates {
                if let Some(nf) = crate::builder_utils::notch_face_corner_with_arc(topo, fid, *arc)?
                {
                    face_replacements.insert(orig, nf);
                    break;
                }
            }
        }

        // Terminal cross-section boundaries not consumed by a corner patch
        // belong to the support face created by notch surgery. Attach that
        // planned owner now instead of relying on positional welding.
        for handles in &blend_cross_handles {
            for handle in [handles.0, handles.1].into_iter().flatten() {
                let Some(edge) = boundary_registry
                    .entry(handle)
                    .and_then(crate::boundary_registry::BoundaryEntry::edge_id)
                else {
                    continue;
                };
                let owner_attached = boundary_registry
                    .entry(handle)
                    .and_then(|entry| entry.owners[1].face)
                    .is_some();
                if owner_attached {
                    continue;
                }
                let support_face = original_faces.iter().find_map(|&original| {
                    let replacement = face_replacements
                        .get(&original)
                        .copied()
                        .unwrap_or(original);
                    face_edge_forward(topo, replacement, edge).map(|_| replacement)
                });
                let Some(support_face) = support_face else {
                    continue;
                };
                boundary_registry.set_owner_face(handle, 1, support_face)?;
                let _ = boundary_registry.oriented_edge(topo, handle, 1)?;
            }
        }
        // Runout support edges are discovered from the final support wires.
        // Complete their deferred support owners now that notch surgery has
        // selected the copy-on-write face IDs.
        let runout_handles: Vec<_> = boundary_registry
            .entries()
            .enumerate()
            .filter_map(|(handle, entry)| {
                (matches!(
                    entry.key.kind,
                    crate::boundary_registry::BoundaryKind::Runout
                ) && entry.owners[0].face.is_none())
                .then_some(handle)
            })
            .collect();
        for handle in runout_handles {
            let edge = boundary_registry
                .entry(handle)
                .and_then(crate::boundary_registry::BoundaryEntry::edge_id)
                .ok_or_else(|| BlendError::PlanningFailure {
                    reason: format!("runout boundary {handle} was not materialized"),
                })?;
            let support = original_faces.iter().find_map(|&original| {
                let replacement = face_replacements
                    .get(&original)
                    .copied()
                    .unwrap_or(original);
                face_edge_forward(topo, replacement, edge).map(|forward| (replacement, forward))
            });
            let Some((support_face, support_forward)) = support else {
                return Err(BlendError::PlanningFailure {
                    reason: format!(
                        "runout boundary {:?} has no result support owner",
                        boundary_registry.entry(handle).map(|entry| entry.key)
                    ),
                });
            };
            boundary_registry.set_owner_forward(handle, 0, support_forward)?;
            boundary_registry.set_owner_face(handle, 0, support_face)?;
            let _ = boundary_registry.oriented_edge(topo, handle, 0)?;
        }

        // Commit support-side uses after notch surgery has selected the final
        // replacement face IDs. The blend-side uses are recorded by the
        // registry-backed blend constructor below.
        for (si, sr) in regular_results.iter().enumerate() {
            let stripe = &sr.stripe;
            let support_faces = [
                face_replacements
                    .get(&stripe.face1)
                    .copied()
                    .unwrap_or(stripe.face1),
                face_replacements
                    .get(&stripe.face2)
                    .copied()
                    .unwrap_or(stripe.face2),
            ];
            for (side, support_face) in support_faces.into_iter().enumerate() {
                let handle = match side {
                    0 => stripe_contact_handles[si].0,
                    1 => stripe_contact_handles[si].1,
                    _ => unreachable!("support side is always 0 or 1"),
                };
                let Some(handle) = handle else {
                    continue;
                };
                boundary_registry.set_owner_face(handle, 0, support_face)?;
                let _ = boundary_registry.oriented_edge(topo, handle, 0)?;
            }
        }
        boundary_registry.preassembly_audit()?;

        let mut result_faces: Vec<FaceId> = Vec::new();

        for &fid in &original_faces {
            if !touched_faces.contains(&fid) {
                // An untouched face may still have been rebuilt by the
                // end-cap notch pass.
                result_faces.push(face_replacements.get(&fid).copied().unwrap_or(fid));
            }
        }

        // Iterate touched faces in deterministic index order: HashSet order
        // varies run-to-run and would make the result shell's face order
        // (and thus STEP export entity numbering) nondeterministic.
        let mut touched: Vec<FaceId> = touched_faces.iter().copied().collect();
        touched.sort_unstable_by_key(|f| f.index());
        for &fid in &touched {
            let replacement = face_replacements.get(&fid).copied();
            result_faces.push(replacement.unwrap_or(fid));
        }

        result_faces.extend(&blend_face_ids);
        result_faces.extend(&corner_face_ids);

        // Faces carried over from the input solid keep their (correct)
        // orientation: they seed the sense propagation and calibrate the
        // boundary-walk convention. Only faces built by THIS pass — walls,
        // corner patches, bands, fills, and rebuilt originals — are
        // eligible for repair; a previous fillet's NURBS wall arriving as
        // an input face must never be re-judged.
        let original_set: std::collections::HashSet<FaceId> =
            original_faces.iter().copied().collect();
        let seeds: Vec<FaceId> = result_faces
            .iter()
            .filter(|f| original_set.contains(f))
            .copied()
            .collect();
        let new_faces: Vec<FaceId> = result_faces
            .iter()
            .filter(|f| !original_set.contains(f))
            .copied()
            .collect();
        // Registry-backed boundaries are the sole closure mechanism for the
        // authoritative fillet path. Positional welding and residual-loop
        // filling are intentionally not available here.
        brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
        brepkit_topology::orientation::normalize_face_normals(topo, &new_faces, &seeds)?;
        brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
        boundary_registry.refresh_owner_orientations(topo)?;
        // The post-assembly shell may come out globally inverted (every
        // face's effective normal pointing inward) while still passing the
        // incidence and closure gates. Detect that via the signed volume of
        // the result shell and flip the whole shell when negative, then
        // re-run the orientation passes so consistency is restored.
        let signed_volume = crate::signed_volume::signed_shell_volume(topo, &result_faces)?;
        if signed_volume < 0.0 {
            log::debug!("orientation flip: signed shell volume {signed_volume} < 0");
            for &face_id in &result_faces {
                let face = topo.face_mut(face_id)?;
                face.set_reversed(!face.is_reversed());
            }
            brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
            brepkit_topology::orientation::normalize_face_normals(topo, &new_faces, &seeds)?;
            brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
            boundary_registry.refresh_owner_orientations(topo)?;
        }
        let postassembly = if force_postassembly_failure {
            Err("forced postassembly incidence gate failure".to_owned())
        } else {
            boundary_registry
                .postassembly_audit(topo, &result_faces)
                .map(|_| ())
                .map_err(|error| error.to_string())
        };
        if let Err(diagnostic) = postassembly {
            for edge_id in succeeded_candidates(plan.as_ref(), &stripe_results, &all_edges) {
                failed.push((
                    edge_id,
                    BlendError::PlanningFailure {
                        reason: format!("assembly incidence gate failed: {diagnostic}"),
                    },
                ));
            }
            return Ok(BlendResult {
                solid: self.solid,
                succeeded: Vec::new(),
                failed,
                is_partial: true,
            });
        }
        let new_shell = match Shell::new(result_faces) {
            Ok(shell) => shell,
            Err(error) => {
                let diagnostic = error.to_string();
                for edge_id in succeeded_candidates(plan.as_ref(), &stripe_results, &all_edges) {
                    failed.push((
                        edge_id,
                        BlendError::PlanningFailure {
                            reason: format!("assembly shell gate failed: {diagnostic}"),
                        },
                    ));
                }
                return Ok(BlendResult {
                    solid: self.solid,
                    succeeded: Vec::new(),
                    failed,
                    is_partial: true,
                });
            }
        };
        let new_shell_id = topo.add_shell(new_shell);
        let new_solid = Solid::new(new_shell_id, Vec::new());
        let new_solid_id = topo.add_solid(new_solid);
        // A requested edge is successful only after every support, blend, and
        // shell-incidence step above has completed. Geometry-only success is
        // deliberately not reported.
        let succeeded = succeeded_candidates(plan.as_ref(), &stripe_results, &all_edges);
        let is_partial = !failed.is_empty();
        Ok(BlendResult {
            solid: new_solid_id,
            succeeded,
            failed,
            is_partial,
        })
    }
}
fn face_edge_forward(topo: &Topology, face_id: FaceId, edge_id: EdgeId) -> Option<bool> {
    let face = topo.face(face_id).ok()?;
    std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .find_map(|wire| {
            topo.wire(wire)
                .ok()?
                .edges()
                .iter()
                .find_map(|oriented| (oriented.edge() == edge_id).then_some(oriented.is_forward()))
        })
}
fn ordered_spine_endpoints(
    topo: &Topology,
    spine: &Spine,
) -> Result<Option<(VertexId, VertexId)>, BlendError> {
    let edges = spine.edges();
    if edges.is_empty() {
        return Ok(None);
    }
    let mut directions = vec![true; edges.len()];
    if edges.len() > 1 {
        let first = topo.edge(edges[0])?;
        let next = topo.edge(edges[1])?;
        if first.end() != next.start()
            && first.end() != next.end()
            && (first.start() == next.start() || first.start() == next.end())
        {
            directions[0] = false;
        }
    }
    for index in 1..edges.len() {
        let previous = topo.edge(edges[index - 1])?;
        let previous_end = if directions[index - 1] {
            previous.end()
        } else {
            previous.start()
        };
        let edge = topo.edge(edges[index])?;
        directions[index] = if edge.start() == previous_end {
            true
        } else {
            edge.end() != previous_end
        };
    }
    let first = topo.edge(edges[0])?;
    let last = topo.edge(edges[edges.len() - 1])?;
    let start = if directions[0] {
        first.start()
    } else {
        first.end()
    };
    let end = if directions[edges.len() - 1] {
        last.end()
    } else {
        last.start()
    };
    if start == end {
        Ok(None)
    } else {
        Ok(Some((start, end)))
    }
}

fn contact_source_vertex(
    topo: &Topology,
    support_face: FaceId,
    contact_vertex: VertexId,
    terminals: (VertexId, VertexId),
) -> Option<VertexId> {
    if contact_vertex == terminals.0 || contact_vertex == terminals.1 {
        return Some(contact_vertex);
    }
    let face = topo.face(support_face).ok()?;
    let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for wire_id in wires {
        let wire = topo.wire(wire_id).ok()?;
        for oriented in wire.edges() {
            let edge = topo.edge(oriented.edge()).ok()?;
            let other = if edge.start() == contact_vertex {
                edge.end()
            } else if edge.end() == contact_vertex {
                edge.start()
            } else {
                continue;
            };
            if other == terminals.0 || other == terminals.1 {
                return Some(other);
            }
        }
    }
    None
}

fn contact_forward_for_spine(
    topo: &Topology,
    registry: &BoundaryRegistry,
    handle: crate::boundary_registry::BoundaryHandle,
    support_face: FaceId,
    terminals: Option<(VertexId, VertexId)>,
    starts_at_spine_start: bool,
) -> Result<Option<bool>, BlendError> {
    let Some((spine_start, spine_end)) = terminals else {
        return Ok(None);
    };
    let entry = registry
        .entry(handle)
        .ok_or_else(|| BlendError::PlanningFailure {
            reason: format!("unknown contact boundary {handle}"),
        })?;
    let direct_vertices = (
        contact_source_vertex(
            topo,
            support_face,
            entry.start.vertex,
            (spine_start, spine_end),
        ),
        contact_source_vertex(
            topo,
            support_face,
            entry.end.vertex,
            (spine_start, spine_end),
        ),
    );
    let (Some(start), Some(end)) = direct_vertices else {
        return Ok(None);
    };
    let desired = if starts_at_spine_start {
        (spine_start, spine_end)
    } else {
        (spine_end, spine_start)
    };
    if (start, end) == desired {
        Ok(Some(true))
    } else if (end, start) == desired {
        Ok(Some(false))
    } else {
        Ok(None)
    }
}

fn build_in_place(
    topo: &mut Topology,
    solid: SolidId,
    edge_sets: Vec<(Vec<EdgeId>, RadiusLaw)>,
) -> Result<BlendResult, BlendError> {
    FilletBuilder {
        topo,
        solid,
        edge_sets,
    }
    .build_in_place()
}

struct SolidGraph {
    solid_id: SolidId,
    shell_ids: Vec<ShellId>,
    face_ids: Vec<FaceId>,
    wire_ids: Vec<WireId>,
    edge_ids: Vec<EdgeId>,
    vertex_ids: Vec<VertexId>,
    pcurves: Vec<(EdgeId, FaceId, brepkit_topology::pcurve::PCurve)>,
}

impl SolidGraph {
    fn capture(topo: &Topology, solid: SolidId) -> Result<Self, BlendError> {
        let source = topo.solid(solid)?;
        let mut shell_ids = vec![source.outer_shell()];
        shell_ids.extend_from_slice(source.inner_shells());
        shell_ids.sort_unstable();
        shell_ids.dedup();

        let mut face_ids = Vec::new();
        let mut seen_faces = HashSet::new();
        for &shell_id in &shell_ids {
            for &face_id in topo.shell(shell_id)?.faces() {
                if seen_faces.insert(face_id) {
                    face_ids.push(face_id);
                }
            }
        }

        let mut wire_ids = Vec::new();
        let mut seen_wires = HashSet::new();
        for &face_id in &face_ids {
            let face = topo.face(face_id)?;
            for wire_id in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if seen_wires.insert(wire_id) {
                    wire_ids.push(wire_id);
                }
            }
        }

        let mut edge_ids = Vec::new();
        let mut seen_edges = HashSet::new();
        for &wire_id in &wire_ids {
            for oriented_edge in topo.wire(wire_id)?.edges() {
                if seen_edges.insert(oriented_edge.edge()) {
                    edge_ids.push(oriented_edge.edge());
                }
            }
        }

        let mut vertex_ids = Vec::new();
        let mut seen_vertices = HashSet::new();
        for &edge_id in &edge_ids {
            let edge = topo.edge(edge_id)?;
            for vertex_id in [edge.start(), edge.end()] {
                if seen_vertices.insert(vertex_id) {
                    vertex_ids.push(vertex_id);
                }
            }
        }

        let mut pcurves = Vec::new();
        for &face_id in &face_ids {
            for (edge_id, pcurve) in topo.pcurves().pcurves_for_face(face_id) {
                if seen_edges.contains(&edge_id) {
                    pcurves.push((edge_id, face_id, pcurve.clone()));
                }
            }
        }

        Ok(Self {
            solid_id: solid,
            shell_ids,
            face_ids,
            wire_ids,
            edge_ids,
            vertex_ids,
            pcurves,
        })
    }
}
struct SolidSnapshot {
    graph: SolidGraph,
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    wires: Vec<Wire>,
    faces: Vec<Face>,
    shells: Vec<Shell>,
    solid: Solid,
}

impl SolidSnapshot {
    fn capture(topo: &Topology, solid: SolidId) -> Result<Self, BlendError> {
        let graph = SolidGraph::capture(topo, solid)?;
        let vertices = graph
            .vertex_ids
            .iter()
            .map(|&id| topo.vertex(id).cloned())
            .collect::<Result<_, _>>()?;
        let edges = graph
            .edge_ids
            .iter()
            .map(|&id| topo.edge(id).cloned())
            .collect::<Result<_, _>>()?;
        let wires = graph
            .wire_ids
            .iter()
            .map(|&id| topo.wire(id).cloned())
            .collect::<Result<_, _>>()?;
        let faces = graph
            .face_ids
            .iter()
            .map(|&id| topo.face(id).cloned())
            .collect::<Result<_, _>>()?;
        let shells = graph
            .shell_ids
            .iter()
            .map(|&id| topo.shell(id).cloned())
            .collect::<Result<_, _>>()?;
        let solid = topo.solid(solid)?.clone();
        Ok(Self {
            graph,
            vertices,
            edges,
            wires,
            faces,
            shells,
            solid,
        })
    }

    fn restore(&self, topo: &mut Topology) -> Result<(), BlendError> {
        for (&id, value) in self.graph.vertex_ids.iter().zip(&self.vertices) {
            *topo.vertex_mut(id)? = value.clone();
        }
        for (&id, value) in self.graph.edge_ids.iter().zip(&self.edges) {
            *topo.edge_mut(id)? = value.clone();
        }
        for (&id, value) in self.graph.wire_ids.iter().zip(&self.wires) {
            *topo.wire_mut(id)? = value.clone();
        }
        for (&id, value) in self.graph.face_ids.iter().zip(&self.faces) {
            *topo.face_mut(id)? = value.clone();
        }
        for (&id, value) in self.graph.shell_ids.iter().zip(&self.shells) {
            *topo.shell_mut(id)? = value.clone();
        }
        *topo.solid_mut(self.graph.solid_id)? = self.solid.clone();

        for &edge_id in &self.graph.edge_ids {
            for &face_id in &self.graph.face_ids {
                topo.pcurves_mut().remove(edge_id, face_id);
            }
        }
        for (edge_id, face_id, pcurve) in &self.graph.pcurves {
            topo.pcurves_mut().set(*edge_id, *face_id, pcurve.clone());
        }
        Ok(())
    }
    /// Clone source faces that the result shell still references before the
    /// source graph is restored. Trimming propagates boundary splits through
    /// every wire using the split edge, so a result that carries an
    /// untrimmed source face must retain that face's current wire while the
    /// original face and wire are put back for the caller.
    fn clone_result_source_faces(
        &self,
        topo: &mut Topology,
        result_solid: SolidId,
    ) -> Result<(), BlendError> {
        if result_solid == self.graph.solid_id {
            return Ok(());
        }

        let shell_id = topo.solid(result_solid)?.outer_shell();
        let result_faces = topo.shell(shell_id)?.faces().to_vec();
        let source_faces: HashSet<FaceId> = self.graph.face_ids.iter().copied().collect();
        let mut replacements = std::collections::HashMap::new();

        for face_id in result_faces {
            if !source_faces.contains(&face_id) || replacements.contains_key(&face_id) {
                continue;
            }

            let source_face = topo.face(face_id)?.clone();
            let source_wires: Vec<WireId> = std::iter::once(source_face.outer_wire())
                .chain(source_face.inner_wires().iter().copied())
                .collect();
            let wire_ids: Vec<WireId> = source_wires
                .iter()
                .map(|&wire_id| topo.wire(wire_id).cloned())
                .collect::<Result<Vec<Wire>, _>>()?
                .into_iter()
                .map(|wire| topo.add_wire(wire))
                .collect();
            let (&outer_wire, inner_wires) =
                wire_ids
                    .split_first()
                    .ok_or(brepkit_topology::TopologyError::Empty {
                        entity: "face wires",
                    })?;

            let mut replacement = Face::new(
                outer_wire,
                inner_wires.to_vec(),
                source_face.surface().clone(),
            );
            replacement.set_reversed(source_face.is_reversed());
            let replacement_id = topo.add_face(replacement);

            let pcurves: Vec<(EdgeId, brepkit_topology::pcurve::PCurve)> = topo
                .pcurves()
                .pcurves_for_face(face_id)
                .into_iter()
                .map(|(edge_id, pcurve)| (edge_id, pcurve.clone()))
                .collect();
            for (edge_id, pcurve) in pcurves {
                topo.pcurves_mut().set(edge_id, replacement_id, pcurve);
            }
            replacements.insert(face_id, replacement_id);
        }

        let shell = topo.shell_mut(shell_id)?;
        for face_id in shell.faces_mut() {
            if let Some(&replacement_id) = replacements.get(face_id) {
                *face_id = replacement_id;
            }
        }
        Ok(())
    }
}

/// Geometry of a full-revolution rim fillet (a closed circular edge between a
/// bounded disc cap and an axisymmetric wall), recovered from a stripe whose
/// blend surface is a torus.
struct ClosedRimInfo {
    /// The bounded disc cap face (a `Plane`).
    plane_face: FaceId,
    /// The axisymmetric wall face (`Cylinder` or `Cone`).
    wall_face: FaceId,
    /// The original closed rim edge on the wall, to be replaced by the
    /// wall-contact circle.
    rim_edge: EdgeId,
    /// Contact circle on the plate (radius `r_c − r`), in the plane.
    plate_circle: Circle3D,
    /// Contact circle on the wall (radius `r_c` for a cylinder), one fillet
    /// radius along the axis from the plate.
    wall_circle: Circle3D,
}

/// Project a point onto the infinite axis line through `origin` with unit
/// direction `axis`, returning the foot of the perpendicular.
fn project_onto_axis(p: Point3, origin: Point3, axis: Vec3) -> Point3 {
    let d = p - origin;
    origin + axis * axis.dot(d)
}

/// Radial distance from a point to the axis line.
fn radial_distance(p: Point3, origin: Point3, axis: Vec3) -> f64 {
    let d = p - origin;
    (d - axis * axis.dot(d)).length()
}

/// Detect a full-revolution rim-fillet stripe and recover its annular geometry.
///
/// Returns `Some` when the blend surface is a torus, the spine is a single
/// closed circular edge (start vertex == end vertex), and the two adjacent
/// faces are a plane (the disc cap) and a cylinder/cone (the wall). Returns
/// `None` for every other configuration (so the caller uses the normal trim
/// path).
///
/// # Errors
///
/// Returns [`BlendError`] if topology lookups or circle construction fail.
fn closed_rim_info(topo: &Topology, stripe: &Stripe) -> Result<Option<ClosedRimInfo>, BlendError> {
    if !matches!(stripe.surface, FaceSurface::Torus(_)) {
        return Ok(None);
    }

    // Spine must be a single closed circular edge.
    let edges = stripe.spine.edges();
    if edges.len() != 1 {
        return Ok(None);
    }
    let rim_edge = edges[0];
    {
        let e = topo.edge(rim_edge)?;
        if e.start() != e.end() {
            return Ok(None);
        }
        if !matches!(e.curve(), EdgeCurve::Circle(_)) {
            return Ok(None);
        }
    }

    // One side is the plane (cap), the other the cylinder/cone wall.
    let s1 = topo.face(stripe.face1)?.surface().clone();
    let s2 = topo.face(stripe.face2)?.surface().clone();
    let (plane_face, wall_face) = match (&s1, &s2) {
        (FaceSurface::Plane { .. }, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) => {
            (stripe.face1, stripe.face2)
        }
        (FaceSurface::Cylinder(_) | FaceSurface::Cone(_), FaceSurface::Plane { .. }) => {
            (stripe.face2, stripe.face1)
        }
        _ => return Ok(None),
    };

    // The annular rebuild replaces the cap's whole outer wire with the
    // plate-contact circle, so it only applies when the cap is a bare disc
    // whose sole boundary is this rim (no inner wires). A more complex cap
    // falls back to the normal trim path.
    {
        let cap = topo.face(plane_face)?;
        if !cap.inner_wires().is_empty() {
            return Ok(None);
        }
        let cap_wire = topo.wire(cap.outer_wire())?;
        let edges = cap_wire.edges();
        if edges.len() != 1 || edges[0].edge() != rim_edge {
            return Ok(None);
        }
    }

    // The plane-side contact curve is the one whose face is the plane.
    let (plate_contact, wall_contact) = if plane_face == stripe.face1 {
        (&stripe.contact1, &stripe.contact2)
    } else {
        (&stripe.contact2, &stripe.contact1)
    };

    // Recover the wall axis line from the wall surface.
    let wall_surf = topo.face(wall_face)?.surface().clone();
    let (axis, axis_origin) = match &wall_surf {
        FaceSurface::Cylinder(c) => (c.axis(), c.origin()),
        FaceSurface::Cone(c) => (c.axis(), c.apex()),
        _ => return Ok(None),
    };

    // Each contact is a full circle perpendicular to the axis; recover its
    // centre (foot on the axis line) and radius (radial distance) from one
    // sampled point.
    let (pt0, _) = plate_contact.domain();
    let plate_pt = plate_contact.evaluate(pt0);
    let plate_center = project_onto_axis(plate_pt, axis_origin, axis);
    let plate_radius = radial_distance(plate_pt, axis_origin, axis);

    let (wt0, _) = wall_contact.domain();
    let wall_pt = wall_contact.evaluate(wt0);
    let wall_center = project_onto_axis(wall_pt, axis_origin, axis);
    let wall_radius = radial_distance(wall_pt, axis_origin, axis);

    let plate_circle = Circle3D::new(plate_center, axis, plate_radius)?;
    let wall_circle = Circle3D::new(wall_center, axis, wall_radius)?;

    Ok(Some(ClosedRimInfo {
        plane_face,
        wall_face,
        rim_edge,
        plate_circle,
        wall_circle,
    }))
}

/// Assemble a full-revolution rim fillet: rebuild the disc cap bounded by the
/// plate-contact circle, shorten the wall to the wall-contact circle, and emit
/// the toroidal band between them. The cap and wall edges are shared with the
/// band so the result is watertight.
///
/// Updates `face_replacements` for the cap and wall (so a later stripe sees the
/// shortened wall). Returns the new toroidal band face.
///
/// # Errors
///
/// Returns [`BlendError`] if topology lookups or wire/face construction fail.
fn assemble_closed_rim(
    topo: &mut Topology,
    stripe: &Stripe,
    rim: &ClosedRimInfo,
    contour_id: usize,
    registry: &mut BoundaryRegistry,
    face_replacements: &mut std::collections::HashMap<FaceId, FaceId>,
) -> Result<FaceId, BlendError> {
    const TOL: f64 = 1e-7;

    // Snapshot the cap and wall (resolving any prior replacement) before
    // mutating the arena.
    let plane_surf = topo.face(rim.plane_face)?.surface().clone();
    let plane_reversed = topo.face(rim.plane_face)?.is_reversed();
    let current_wall = face_replacements
        .get(&rim.wall_face)
        .copied()
        .unwrap_or(rim.wall_face);
    let wall_surf = topo.face(current_wall)?.surface().clone();
    let wall_reversed = topo.face(current_wall)?.is_reversed();
    let wall_outer_wire = topo.face(current_wall)?.outer_wire();
    let wall_inner = topo.face(current_wall)?.inner_wires().to_vec();
    let wall_oriented: Vec<OrientedEdge> = topo.wire(wall_outer_wire)?.edges().to_vec();
    let cap_orig_wire = topo.face(
        face_replacements
            .get(&rim.plane_face)
            .copied()
            .unwrap_or(rim.plane_face),
    )?;
    let cap_orig_wire_id = cap_orig_wire.outer_wire();
    let cap_forward = topo
        .wire(cap_orig_wire_id)?
        .edges()
        .iter()
        .find(|oe| oe.edge() == rim.rim_edge)
        .is_some_and(OrientedEdge::is_forward);

    // Vertices for the two closed contact circles (start == end → degenerate).
    let plate_v = topo.add_vertex(Vertex::new(rim.plate_circle.evaluate(0.0), TOL));
    let wall_v = topo.add_vertex(Vertex::new(rim.wall_circle.evaluate(0.0), TOL));

    // Shared contact-circle edges are planned and materialized through the
    // same registry used by open contours.
    let torus = match &stripe.surface {
        FaceSurface::Torus(t) => t.clone(),
        _ => {
            return Err(BlendError::TrimmingFailure {
                face: rim.wall_face,
            });
        }
    };
    let band_reversed = torus_band_needs_reversal(&torus, rim);
    let wall_forward = wall_oriented
        .iter()
        .find(|oe| oe.edge() == rim.rim_edge)
        .is_some_and(OrientedEdge::is_forward);
    let plane_side = usize::from(stripe.face2 == rim.plane_face);
    let wall_side = usize::from(stripe.face2 == rim.wall_face);
    let cap_sense = if matches!(&wall_surf, FaceSurface::Cylinder(_)) {
        cap_forward
    } else {
        !cap_forward
    };
    let plate_sense = (cap_forward == plane_reversed) != band_reversed;
    let wall_sense = (wall_forward == wall_reversed) != band_reversed;
    let plate_handle = registry.register(
        BoundaryKey::contact(contour_id, 0, plane_side as u8),
        PlannedVertex::new(plate_v),
        PlannedVertex::new(plate_v),
        EdgeCurve::Circle(rim.plate_circle.clone()),
        (0.0, std::f64::consts::TAU),
        [
            // Result-face orientation propagation determines the final cap
            // use; cylinders retain the source sense while cones reverse it.
            BoundaryOwner::planned("closed-rim support cap", cap_sense),
            BoundaryOwner::planned("closed-rim torus band", plate_sense),
        ],
    )?;
    let wall_handle = registry.register(
        BoundaryKey::contact(contour_id, 0, wall_side as u8),
        PlannedVertex::new(wall_v),
        PlannedVertex::new(wall_v),
        EdgeCurve::Circle(rim.wall_circle.clone()),
        (0.0, std::f64::consts::TAU),
        [
            BoundaryOwner::planned("closed-rim support wall", wall_forward),
            BoundaryOwner::planned("closed-rim torus band", wall_sense),
        ],
    )?;
    let seam_edge = topo.add_edge(Edge::new(plate_v, wall_v, EdgeCurve::Line));
    let cap_boundary = registry.oriented_edge(topo, plate_handle, 0)?;
    let cap_wire = Wire::new(vec![cap_boundary], true)?;
    let cap_wire_id = topo.add_wire(cap_wire);
    let mut cap_face = Face::new(cap_wire_id, Vec::new(), plane_surf);
    cap_face.set_reversed(plane_reversed);
    let cap_face_id = topo.add_face(cap_face);
    registry.set_owner_face(plate_handle, 0, cap_face_id)?;
    face_replacements.insert(rim.plane_face, cap_face_id);

    // --- Shorten the wall to the wall-contact circle. ---
    // The wall's outer wire references the rim circle plus (for the cylinder /
    // cone primitive) a degenerate seam line whose lower endpoint is the rim
    // vertex. Replace the rim circle with the wall-contact circle, and rebuild
    // any seam edge touching the old rim vertex so its lower endpoint becomes
    // the new wall-circle vertex (otherwise the wire no longer closes — the
    // seam would still start at the old rim height).
    let old_rim_vertex = topo.edge(rim.rim_edge)?.start();
    // A seam edge may appear twice in the wall wire (fwd + rev); rebuild each
    // distinct edge once so both references share the new edge (otherwise the
    // two copies each become a free edge).
    let mut rebuilt: std::collections::HashMap<EdgeId, EdgeId> = std::collections::HashMap::new();
    let mut new_wall_edges: Vec<OrientedEdge> = Vec::with_capacity(wall_oriented.len());
    let mut replaced = false;
    for oe in &wall_oriented {
        if oe.edge() == rim.rim_edge {
            let wall_boundary = registry.oriented_edge(topo, wall_handle, 0)?;
            new_wall_edges.push(wall_boundary);
            replaced = true;
            continue;
        }
        let e = topo.edge(oe.edge())?;
        let touches_rim = e.start() == old_rim_vertex || e.end() == old_rim_vertex;
        if touches_rim {
            let new_eid = if let Some(&id) = rebuilt.get(&oe.edge()) {
                id
            } else {
                // Rebuild this edge with `wall_v` substituted for the old rim vertex.
                let curve = e.curve().clone();
                let new_start = if e.start() == old_rim_vertex {
                    wall_v
                } else {
                    e.start()
                };
                let new_end = if e.end() == old_rim_vertex {
                    wall_v
                } else {
                    e.end()
                };
                let id = topo.add_edge(Edge::new(new_start, new_end, curve));
                rebuilt.insert(oe.edge(), id);
                id
            };
            new_wall_edges.push(OrientedEdge::new(new_eid, oe.is_forward()));
        } else {
            new_wall_edges.push(*oe);
        }
    }
    if !replaced {
        return Err(BlendError::TrimmingFailure {
            face: rim.wall_face,
        });
    }
    let new_wall_wire = Wire::new(new_wall_edges, true)?;
    let new_wall_wire_id = topo.add_wire(new_wall_wire);
    let mut new_wall_face = Face::new(new_wall_wire_id, wall_inner, wall_surf);
    new_wall_face.set_reversed(wall_reversed);
    let new_wall_face_id = topo.add_face(new_wall_face);
    registry.rebind_owner_face(current_wall, new_wall_face_id);
    registry.set_owner_face(wall_handle, 0, new_wall_face_id)?;
    face_replacements.insert(rim.wall_face, new_wall_face_id);

    // --- Toroidal band between the two contact circles. ---
    // Degenerate-seam wire (plate circle, seam up, wall circle reversed, seam
    // down). The seam runs plate_v → wall_v, so this fixed order always closes
    // (plate_v → plate_v → wall_v → wall_v → plate_v). The shared circle edges
    // are used opposite to the standard-wound cap and wall, keeping the shell
    // manifold.
    let plate_boundary = registry.oriented_edge(topo, plate_handle, 1)?;
    let wall_boundary = registry.oriented_edge(topo, wall_handle, 1)?;
    let band_wire = Wire::new(
        vec![
            plate_boundary,
            OrientedEdge::new(seam_edge, true),
            wall_boundary,
            OrientedEdge::new(seam_edge, false),
        ],
        true,
    )?;
    let band_wire_id = topo.add_wire(band_wire);
    let mut band_face = Face::new(band_wire_id, Vec::new(), stripe.surface.clone());
    if band_reversed {
        band_face.set_reversed(true);
    }
    let band_face_id = topo.add_face(band_face);
    registry.set_owner_face(plate_handle, 1, band_face_id)?;
    registry.set_owner_face(wall_handle, 1, band_face_id)?;
    registry.install_pcurves(topo, plate_handle)?;
    registry.install_pcurves(topo, wall_handle)?;

    Ok(band_face_id)
}

fn succeeded_candidates(
    plan: Option<&FilletPlan>,
    stripe_results: &[StripeResult],
    all_edges: &[(EdgeId, usize)],
) -> Vec<EdgeId> {
    let mut result = Vec::new();
    if let Some(fillet_plan) = plan {
        for contour in &fillet_plan.contours {
            if stripe_results
                .iter()
                .any(|stripe| stripe.stripe.spine_edges() == contour.spine.edges())
            {
                result.extend(contour.edges.iter().copied());
            }
        }
    } else {
        result.extend(
            stripe_results
                .iter()
                .flat_map(|stripe| stripe.stripe.spine_edges().iter().copied()),
        );
        result.retain(|edge| all_edges.iter().any(|(requested, _)| requested == edge));
    }
    result.sort_unstable_by_key(|edge| edge.index());
    result.dedup_by_key(|edge| edge.index());
    result
}
/// Check the geometric constraints retained by a computed stripe.
///
/// The walking solver already checks its Newton residual while marching. This
/// second check covers analytic stripes too and guards the data consumed by
/// topology assembly: each section must remain on both source surfaces, keep
/// its requested radius, and have a section radius aligned with each source
/// normal.
fn check_stripe_residuals(topo: &Topology, stripe: &Stripe) -> Result<(), BlendError> {
    const CONTACT_TOL: f64 = 1e-5;
    const RADIUS_TOL: f64 = 1e-5;
    const TANGENT_TOL: f64 = 2e-4;
    let surface1 = topo.face(stripe.face1)?.surface().clone();
    let surface2 = topo.face(stripe.face2)?.surface().clone();
    let mut adapter1 = None;
    let mut adapter2 = None;
    let ps1 = surface_ref_or_adapter(&surface1, &mut adapter1);
    let ps2 = surface_ref_or_adapter(&surface2, &mut adapter2);
    let mut max_contact: f64 = 0.0;
    let mut max_radius: f64 = 0.0;
    let mut max_tangent: f64 = 0.0;
    for section in &stripe.sections {
        let uv1 = ps1.project_point(section.p1);
        let uv2 = ps2.project_point(section.p2);
        max_contact = max_contact
            .max((ps1.evaluate(uv1.0, uv1.1) - section.p1).length())
            .max((ps2.evaluate(uv2.0, uv2.1) - section.p2).length());
        max_radius = max_radius
            .max(((section.center - section.p1).length() - section.radius).abs())
            .max(((section.center - section.p2).length() - section.radius).abs());
        let radial1 = (section.p1 - section.center).normalize();
        let radial2 = (section.p2 - section.center).normalize();
        if let (Ok(radial1), Ok(radial2)) = (radial1, radial2) {
            max_tangent = max_tangent
                .max((1.0 - ps1.normal(uv1.0, uv1.1).dot(radial1).abs()).abs())
                .max((1.0 - ps2.normal(uv2.0, uv2.1).dot(radial2).abs()).abs());
        }
    }
    if max_contact > CONTACT_TOL || max_radius > RADIUS_TOL || max_tangent > TANGENT_TOL {
        return Err(BlendError::PlanningFailure {
            reason: format!(
                "stripe residuals exceed tolerance: contact={max_contact:.3e}, radius={max_radius:.3e}, tangent={max_tangent:.3e}"
            ),
        });
    }
    Ok(())
}

/// Decide whether a rim-fillet torus band must carry `reversed` so its outward
/// normal points away from the solid.
///
/// The band's mid-arc geometric normal points radially out from the tube; we
/// need it to also point to the *empty* side along the axis. The empty side is
/// opposite the wall material: for a non-reversed cylinder/cone wall the
/// material is on the axis-interior side, and the band sits one fillet radius
/// from the plate toward the material — so the band's outward axial direction is
/// the one pointing from the wall-contact circle back toward the plate.
fn torus_band_needs_reversal(
    torus: &brepkit_math::surfaces::ToroidalSurface,
    rim: &ClosedRimInfo,
) -> bool {
    // The torus geometric normal at the mid-arc point (halfway between the two
    // contacts) should point away from the segment plate→wall along the axis.
    // The "away from material" axial direction is plate_center → (plate_center −
    // wall_center) i.e. from the wall contact toward the plate.
    let axis = torus.z_axis();
    let to_plate = rim.plate_circle.center() - rim.wall_circle.center();
    let outward_axial = axis * axis.dot(to_plate); // component along the axis toward the plate
    // Mid-arc point and its geometric normal.
    let v_plate = torus.project_point(rim.plate_circle.evaluate(0.0)).1;
    let v_wall = torus.project_point(rim.wall_circle.evaluate(0.0)).1;
    // Shortest signed mid-angle between the two contact v-parameters (periodic):
    // reduce the raw difference into (−π, π].
    let dv = (v_wall - v_plate + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI;
    let v_mid = v_plate + dv * 0.5;
    let n = torus.normal(0.0, v_mid);
    // If the geometric normal's axial part opposes the outward axial direction,
    // the band must be reversed.
    n.dot(outward_axial) < 0.0
}

/// Compute a stripe for a single edge using the adjacency index.
///
/// # Errors
///
/// Returns [`BlendError`] if the edge is non-manifold, if topology lookups
/// fail, or if neither the analytic nor walking path can produce a result.
#[allow(clippy::too_many_lines)]
fn compute_stripe_for_edge(
    topo: &Topology,
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    edge_id: EdgeId,
    law: &RadiusLaw,
) -> Result<StripeResult, BlendError> {
    let adj_faces = adjacency.faces_for_edge(edge_id);
    if adj_faces.len() != 2 {
        log::warn!(
            "edge {edge_id:?} has {} adjacent faces (expected 2) — cannot fillet non-manifold or boundary edges",
            adj_faces.len()
        );
        return Err(BlendError::StartSolutionFailure {
            edge: edge_id,
            t: 0.0,
        });
    }
    let spine = Spine::from_single_edge(topo, edge_id)?;
    compute_stripe_for_spine(topo, adj_faces[0], adj_faces[1], spine, law)
}

fn radius_law_from_plan(plan: &crate::fillet_plan::RadiusLawPlan) -> RadiusLaw {
    match plan {
        crate::fillet_plan::RadiusLawPlan::Constant(radius) => RadiusLaw::Constant(*radius),
        crate::fillet_plan::RadiusLawPlan::Linear { start, end } => RadiusLaw::Linear {
            start: *start,
            end: *end,
        },
        crate::fillet_plan::RadiusLawPlan::SCurve { start, end } => RadiusLaw::SCurve {
            start: *start,
            end: *end,
        },
        crate::fillet_plan::RadiusLawPlan::Sampled { start, end } => RadiusLaw::Linear {
            start: *start,
            end: *end,
        },
    }
}

fn compute_stripe_for_contour(
    topo: &Topology,
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    contour: &crate::fillet_plan::FilletContour,
) -> Result<StripeResult, BlendError> {
    let edge_id = contour.edges[0];
    if adjacency.faces_for_edge(edge_id).len() != 2 {
        return Err(BlendError::StartSolutionFailure {
            edge: edge_id,
            t: 0.0,
        });
    }
    let law = radius_law_from_plan(&contour.radius_law);
    compute_stripe_for_spine(
        topo,
        contour.side1,
        contour.side2,
        contour.spine.clone(),
        &law,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_stripe_for_spine(
    topo: &Topology,
    face1: FaceId,
    face2: FaceId,
    spine: Spine,
    law: &RadiusLaw,
) -> Result<StripeResult, BlendError> {
    // Snapshot surface data, respecting face orientation.
    let face1_data = topo.face(face1)?;
    let surf1 = face1_data.surface().clone();
    let face1_reversed = face1_data.is_reversed();
    let face2_data = topo.face(face2)?;
    let surf2 = face2_data.surface().clone();
    let face2_reversed = face2_data.is_reversed();

    // Get radius at the spine midpoint for the analytic path.
    let radius = law.evaluate(0.5);

    // Try analytic fast path (only for constant radius).
    // The analytic fillet expects INWARD-pointing normals (toward material).
    // Compute inward normals from the surface normals and face reversal:
    // - Not reversed: outward = surface_normal → inward = -surface_normal
    // - Reversed: outward = -surface_normal → inward = surface_normal
    if matches!(law, RadiusLaw::Constant(_)) {
        let flipped1 = orient_plane_surface(&surf1);
        let flipped2 = orient_plane_surface(&surf2);
        let inward_surf1 = if face1_reversed { &surf1 } else { &flipped1 };
        let inward_surf2 = if face2_reversed { &surf2 } else { &flipped2 };
        if let Some(result) = analytic::try_analytic_fillet(
            inward_surf1,
            inward_surf2,
            &spine,
            topo,
            radius,
            face1,
            face2,
        )? {
            check_stripe_residuals(topo, &result.stripe)?;
            return Ok(result);
        }
    }

    log::debug!(
        target: "brepkit_approx",
        "fillet: analytic fast-path unavailable for {}+{} ({} radius) — using Newton-Raphson walker (approximate NURBS blend surface)",
        surf1.type_tag(),
        surf2.type_tag(),
        if matches!(law, RadiusLaw::Constant(_)) { "constant" } else { "variable" }
    );

    // Build ParametricSurface references via PlaneAdapter for planes.
    // When a face is reversed, the outward normal is flipped. For PlaneAdapter,
    // we negate the normal. For analytic/NURBS surfaces the ParametricSurface
    // impl already returns the geometric normal; the walker uses the sign
    // convention from the face orientation.
    let oriented_surf1 = if face1_reversed {
        orient_plane_surface(&surf1)
    } else {
        surf1
    };
    let oriented_surf2 = if face2_reversed {
        orient_plane_surface(&surf2)
    } else {
        surf2
    };
    let mut adapter1 = None;
    let mut adapter2 = None;

    let ps1 = surface_ref_or_adapter(&oriented_surf1, &mut adapter1);
    let ps2 = surface_ref_or_adapter(&oriented_surf2, &mut adapter2);

    let config = WalkerConfig::default();

    let walk_result = if let RadiusLaw::Constant(r) = law {
        let blend = ConstRadBlend { radius: *r };
        let walker = Walker::new(&blend, ps1, ps2, &spine, topo, config);
        let start = walker.find_start(0.0)?;
        walker.walk(start, 0.0, spine.length())?
    } else {
        let evol = EvolRadBlend {
            law: mirror_law(law),
        };
        let walker = Walker::new(&evol, ps1, ps2, &spine, topo, config);
        let start = walker.find_start(0.0)?;
        walker.walk(start, 0.0, spine.length())?
    };

    let blend_surface = approximate_blend_surface(&walk_result.sections)?;
    let blend_face_surface = brepkit_topology::face::FaceSurface::Nurbs(blend_surface);

    let contact1 = sections_to_contact_curve(&walk_result.sections, |s| s.p1)?;
    let contact2 = sections_to_contact_curve(&walk_result.sections, |s| s.p2)?;

    let pcurve1 = build_pcurve_from_contact(ps1, &contact1, face_u_period(&oriented_surf1))?;
    let pcurve2 = build_pcurve_from_contact(ps2, &contact2, face_u_period(&oriented_surf2))?;

    let stripe = Stripe {
        spine,
        surface: blend_face_surface,
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: walk_result.sections,
    };

    check_stripe_residuals(topo, &stripe)?;
    Ok(StripeResult {
        stripe,
        new_edges: Vec::new(),
    })
}

/// A single cross-section of a rolling-ball blend: the two surface contact
/// points, the rational-quadratic arc apex (middle control point), and its
/// weight `cos(half_angle)`.
#[derive(Debug, Clone, Copy)]
pub struct BlendCrossSection {
    /// Contact point on the first surface (`u = 0` end of the arc).
    pub contact1: brepkit_math::vec::Point3,
    /// Arc apex / middle control point (tangent intersection).
    pub apex: brepkit_math::vec::Point3,
    /// Contact point on the second surface (`u = 1` end of the arc).
    pub contact2: brepkit_math::vec::Point3,
    /// Rational-quadratic weight of the apex (`cos(half_angle)`).
    pub weight: f64,
}

/// Compute the true rolling-ball blend cross-sections for a constant-radius
/// fillet of `edge_id`, at the requested spine `fractions` (each in `[0, 1]`).
///
/// Unlike a tangent-plane offset (`contact = p + dir·r`), this solves the
/// actual ball-tangent-to-both-surfaces constraint via the walking engine, so
/// the contacts land *on* curved neighbours (cylinders, NURBS blend faces).
/// Newton continuation seeds each station from the previous one for robustness.
///
/// `surf1`/`surf2` are the neighbour surfaces with their face `reversed` flags
/// (so plane normals point outward consistently with the walker convention).
///
/// # Errors
///
/// Returns [`BlendError`] if the spine cannot be built or Newton fails to
/// converge at a requested station.
#[allow(clippy::too_many_arguments)]
pub fn blend_cross_sections(
    topo: &Topology,
    edge_id: EdgeId,
    surf1: &brepkit_topology::face::FaceSurface,
    surf1_reversed: bool,
    surf2: &brepkit_topology::face::FaceSurface,
    surf2_reversed: bool,
    radius: f64,
    fractions: &[f64],
) -> Result<Vec<BlendCrossSection>, BlendError> {
    use brepkit_math::vec::Point3;

    let spine = Spine::from_single_edge(topo, edge_id)?;
    let len = spine.length();

    let mut adapter1 = None;
    let mut adapter2 = None;
    let base1 = surface_ref_or_adapter(surf1, &mut adapter1);
    let base2 = surface_ref_or_adapter(surf2, &mut adapter2);
    // The walker places the ball centre on the `+normal` side of each surface,
    // so feed it INWARD (toward-material) normals or it solves the external
    // common-tangent branch (fillet outside the solid). The face's outward
    // normal equals the surface normal when the face is not reversed, so flip
    // then; keep it when the face is reversed.
    let flip1 = FlippedNormalSurface::new(base1);
    let flip2 = FlippedNormalSurface::new(base2);
    let ps1: &dyn brepkit_math::traits::ParametricSurface =
        if surf1_reversed { base1 } else { &flip1 };
    let ps2: &dyn brepkit_math::traits::ParametricSurface =
        if surf2_reversed { base2 } else { &flip2 };

    let blend = ConstRadBlend { radius };
    let walker = Walker::new(&blend, ps1, ps2, &spine, topo, WalkerConfig::default());

    let mut out = Vec::with_capacity(fractions.len());
    let mut prev: Option<crate::blend_func::BlendParams> = None;
    for &f in fractions {
        let s = f.clamp(0.0, 1.0) * len;
        let (params, sec) =
            walker
                .solve_section(s, prev)
                .ok_or(BlendError::StartSolutionFailure {
                    edge: edge_id,
                    t: f,
                })?;
        prev = Some(params);

        let half_angle = sec.half_angle();
        let w = half_angle.cos();
        let midpoint = Point3::new(
            (sec.p1.x() + sec.p2.x()) * 0.5,
            (sec.p1.y() + sec.p2.y()) * 0.5,
            (sec.p1.z() + sec.p2.z()) * 0.5,
        );
        // Apex at the tangent intersection (r/cos θ from the centre), matching
        // `approximate_blend_surface`. Falls back to the chord midpoint when the
        // arc approaches a half-turn (cos θ → 0).
        let apex = if w.abs() > 1e-15 {
            let scale = 1.0 / (w * w);
            Point3::new(
                sec.center.x() + (midpoint.x() - sec.center.x()) * scale,
                sec.center.y() + (midpoint.y() - sec.center.y()) * scale,
                sec.center.z() + (midpoint.z() - sec.center.z()) * scale,
            )
        } else {
            midpoint
        };

        out.push(BlendCrossSection {
            contact1: sec.p1,
            apex,
            contact2: sec.p2,
            weight: w,
        });
    }
    Ok(out)
}

/// Flip the normal of a `Plane` surface to account for face reversal.
///
/// For non-plane surfaces, returns a clone unchanged — the walker already
/// accounts for orientation through the `ParametricSurface` trait.
fn orient_plane_surface(
    surface: &brepkit_topology::face::FaceSurface,
) -> brepkit_topology::face::FaceSurface {
    match surface {
        brepkit_topology::face::FaceSurface::Plane { normal, d } => {
            brepkit_topology::face::FaceSurface::Plane {
                normal: -*normal,
                d: -*d,
            }
        }
        other => other.clone(),
    }
}

/// Mirror a `RadiusLaw` into a new instance with the same behavior.
///
/// This is needed because `RadiusLaw::Custom` contains a `Box<dyn Fn>`
/// which is not `Clone`. For non-custom laws, we reconstruct the same
/// variant. For custom laws, we evaluate at a fixed set of points and
/// create a linear interpolation.
fn mirror_law(law: &RadiusLaw) -> RadiusLaw {
    match law {
        RadiusLaw::Constant(r) => RadiusLaw::Constant(*r),
        RadiusLaw::Linear { start, end } => RadiusLaw::Linear {
            start: *start,
            end: *end,
        },
        RadiusLaw::SCurve { start, end } => RadiusLaw::SCurve {
            start: *start,
            end: *end,
        },
        RadiusLaw::Custom(_) => {
            // Sample the custom law at endpoints and build a linear
            // approximation. This is a v1 simplification; a proper
            // implementation would share the closure via Arc.
            let r0 = law.evaluate(0.0);
            let r1 = law.evaluate(1.0);
            RadiusLaw::Linear { start: r0, end: r1 }
        }
    }
}

/// Build a degree-1 NURBS curve from section contact points.
fn sections_to_contact_curve(
    sections: &[crate::section::CircSection],
    pick: impl Fn(&crate::section::CircSection) -> brepkit_math::vec::Point3,
) -> Result<brepkit_math::nurbs::curve::NurbsCurve, BlendError> {
    let pts: Vec<brepkit_math::vec::Point3> = sections.iter().map(&pick).collect();
    if pts.len() < 2 {
        return Err(BlendError::Math(brepkit_math::MathError::EmptyInput));
    }
    let n = pts.len();
    let degree = 1.min(n - 1);
    let mut knots = vec![0.0; degree + 1];
    if n > 2 {
        for i in 1..n - 1 {
            #[allow(clippy::cast_precision_loss)]
            knots.push(i as f64 / (n - 1) as f64);
        }
    }
    knots.extend(vec![1.0; degree + 1]);
    let weights = vec![1.0; n];
    let curve = brepkit_math::nurbs::curve::NurbsCurve::new(degree, knots, pts, weights)?;
    Ok(curve)
}

fn face_u_period(surface: &FaceSurface) -> Option<f64> {
    match surface {
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => Some(std::f64::consts::TAU),
        FaceSurface::Nurbs(surface) if surface.is_periodic_u() => {
            let (u0, u1) = surface.domain_u();
            (u1 > u0).then_some(u1 - u0)
        }
        _ => None,
    }
}

/// Project a NURBS contact curve into a plane's affine UV frame exactly.
///
/// Projecting only the endpoints loses every closed contact and can also use
/// the analytic helper's inward-normal frame instead of the support face's
/// frame. An affine projection of the homogeneous control polygon preserves
/// the degree, knots, weights, and parameter domain.
fn build_planar_pcurve_from_contact(
    surface: &dyn brepkit_math::traits::ParametricSurface,
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
) -> Result<(brepkit_math::curves2d::Curve2D, f64, f64), BlendError> {
    let control_points = contact
        .control_points()
        .iter()
        .map(|&point| {
            let (u, v) = surface.project_point(point);
            brepkit_math::vec::Point2::new(u, v)
        })
        .collect();
    let curve = brepkit_math::curves2d::NurbsCurve2D::new(
        contact.degree(),
        contact.knots().to_vec(),
        control_points,
        contact.weights().to_vec(),
    )?;
    let (start, end) = curve.domain();
    Ok((brepkit_math::curves2d::Curve2D::Nurbs(curve), start, end))
}

/// Build a contact pcurve in a support surface's native UV frame.
///
/// Open contacts use their endpoint line. Closed periodic contacts are
/// sampled through one full carrier domain and unwrapped across the U seam;
/// projecting equal endpoints alone would collapse them to a zero vector.
fn build_pcurve_from_contact(
    surf: &dyn brepkit_math::traits::ParametricSurface,
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
    u_period: Option<f64>,
) -> Result<brepkit_math::curves2d::Curve2D, BlendError> {
    let (t0, t1) = contact.domain();
    let p_start = contact.evaluate(t0);
    let p_end = contact.evaluate(t1);
    if (p_start - p_end).length() <= 1e-7 {
        const SAMPLES: usize = 64;
        let mut points = Vec::with_capacity(SAMPLES + 1);
        let mut previous_u: Option<f64> = None;
        for index in 0..=SAMPLES {
            let fraction = index as f64 / SAMPLES as f64;
            let point = contact.evaluate(t0 + (t1 - t0) * fraction);
            let (mut u, v) = surf.project_point(point);
            if let (Some(period), Some(reference)) = (u_period, previous_u) {
                u -= period * ((u - reference) / period).round();
            }
            previous_u = Some(u);
            points.push(brepkit_math::vec::Point2::new(u, v));
        }
        let mut knots = Vec::with_capacity(SAMPLES + 3);
        knots.extend([0.0, 0.0]);
        knots.extend((1..SAMPLES).map(|index| index as f64 / SAMPLES as f64));
        knots.extend([1.0, 1.0]);
        let weights = vec![1.0; points.len()];
        return Ok(brepkit_math::curves2d::Curve2D::Nurbs(
            brepkit_math::curves2d::NurbsCurve2D::new(1, knots, points, weights)?,
        ));
    }

    let (u0, v0) = surf.project_point(p_start);
    let (mut u1, v1) = surf.project_point(p_end);
    if let Some(period) = u_period {
        u1 -= period * ((u1 - u0) / period).round();
    }
    let origin = brepkit_math::vec::Point2::new(u0, v0);
    let dir = brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0);
    Ok(brepkit_math::curves2d::Curve2D::Line(
        brepkit_math::curves2d::Line2D::new(origin, dir)?,
    ))
}

/// Rebuild faces whose entire outer wire is consumed by fillet spine edges.
///
/// For a closed rim, the face's post-fillet boundary is the chained loop of
/// the stripes' contact curves on that face. Returns the loop edge chosen
/// for each `(face, stripe index)` so the blend walls share those edges.
/// Faces that fail any structural requirement (an outer-wire edge with no
/// stripe, or a junction gap wider than weld distance) are left for the
/// per-stripe trim path.
#[allow(clippy::type_complexity, clippy::too_many_lines)]
/// A line edge bridging a fillet stripe's abrupt end: from the original
/// outline vertex to the stripe's terminal contact point on one face.
struct NotchRecord {
    stripe: usize,
    edge: EdgeId,
    outline_vid: brepkit_topology::vertex::VertexId,
    contact_vid: brepkit_topology::vertex::VertexId,
    contact_pt: Point3,
}

/// The vertex shared by two stripes' spine edges, if any.
fn shared_spine_vertex(
    topo: &Topology,
    a: &Stripe,
    b: &Stripe,
) -> Option<brepkit_topology::vertex::VertexId> {
    let verts = |st: &Stripe| -> Vec<brepkit_topology::vertex::VertexId> {
        let mut v = Vec::new();
        for &eid in st.spine.edges() {
            if let Ok(e) = topo.edge(eid) {
                v.push(e.start());
                v.push(e.end());
            }
        }
        v
    };
    let va = verts(a);
    verts(b).into_iter().find(|v| va.contains(v))
}

#[allow(
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::match_wildcard_for_single_variants,
    clippy::single_match_else,
    clippy::collapsible_if
)]
fn rebuild_closed_rim_loop_faces(
    topo: &mut Topology,
    regular_results: &[&StripeResult],
    face_replacements: &mut std::collections::HashMap<FaceId, FaceId>,
) -> Result<
    (
        std::collections::HashMap<(FaceId, usize), EdgeId>,
        Vec<NotchRecord>,
    ),
    BlendError,
> {
    use std::collections::HashMap;

    const WELD: f64 = 1e-6;
    let mut out: HashMap<(FaceId, usize), EdgeId> = HashMap::new();
    let mut notches: Vec<NotchRecord> = Vec::new();

    // Spine edge -> stripe index.
    let mut spine_owner: HashMap<EdgeId, usize> = HashMap::new();
    for (si, sr) in regular_results.iter().enumerate() {
        for &eid in sr.stripe.spine_edges() {
            spine_owner.insert(eid, si);
        }
    }

    // Candidate faces: those adjacent to any stripe.
    let mut candidates: Vec<FaceId> = Vec::new();
    for sr in regular_results {
        for f in [sr.stripe.face1, sr.stripe.face2] {
            if !candidates.contains(&f) {
                candidates.push(f);
            }
        }
    }

    'faces: for face_id in candidates {
        let face = topo.face(face_id)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let inner_wires = face.inner_wires().to_vec();
        let wire = topo.wire(face.outer_wire())?;
        let oriented: Vec<OrientedEdge> = wire.edges().to_vec();
        if oriented.len() < 2 {
            continue;
        }

        // Owner per outer-wire edge: Some(stripe) for spine edges, None for
        // edges the fillet does not touch (kept verbatim, preserving their
        // shared-edge identity with neighbouring faces).
        let owners: Vec<Option<usize>> = oriented
            .iter()
            .map(|oe| spine_owner.get(&oe.edge()).copied())
            .collect();
        if !owners.iter().any(Option::is_some) {
            continue 'faces;
        }

        // Rotate so the walk starts at a stripe-run boundary.
        let n = oriented.len();
        let start = (0..n)
            .find(|&i| owners[i] != owners[(i + n - 1) % n])
            .unwrap_or(0);

        // Group consecutive edges into runs (same stripe, or untouched), in
        // wire order.
        let mut runs: Vec<Option<usize>> = Vec::new();
        for k in 0..n {
            let si = owners[(start + k) % n];
            if runs.last() != Some(&si) {
                runs.push(si);
            }
        }
        if runs.len() >= 2 && runs.first() == runs.last() {
            runs.pop();
        }

        // Collect pieces per run: a contact curve for a stripe run
        // (oriented to follow the wire traversal), or the original oriented
        // edges kept verbatim for an untouched run.
        enum RunPiece {
            Contact {
                stripe: usize,
                forward: bool,
                from: Point3,
                to: Point3,
                curve: brepkit_math::nurbs::curve::NurbsCurve,
                from_vid: Option<brepkit_topology::vertex::VertexId>,
                to_vid: Option<brepkit_topology::vertex::VertexId>,
            },
            Original {
                edges: Vec<OrientedEdge>,
                to_vid: brepkit_topology::vertex::VertexId,
                from_vid: brepkit_topology::vertex::VertexId,
                from: Point3,
                to: Point3,
            },
        }
        let mut pieces: Vec<RunPiece> = Vec::with_capacity(runs.len());
        let mut cursor = start;
        for &owner in &runs {
            let run_len = (0..n)
                .take_while(|&k| owners[(cursor + k) % n] == owner)
                .count();
            let run_edges: Vec<OrientedEdge> =
                (0..run_len).map(|k| oriented[(cursor + k) % n]).collect();
            let first_oe = run_edges[0];
            cursor += run_len;

            let first_edge = topo.edge(first_oe.edge())?;
            let from_vid = if first_oe.is_forward() {
                first_edge.start()
            } else {
                first_edge.end()
            };
            let run_start = topo.vertex(from_vid)?.point();

            match owner {
                Some(si) => {
                    let stripe = &regular_results[si].stripe;
                    let contact = if stripe.face1 == face_id {
                        stripe.contact1.clone()
                    } else {
                        stripe.contact2.clone()
                    };
                    let (c0, c1) = {
                        let (d0, d1) = contact.domain();
                        (contact.evaluate(d0), contact.evaluate(d1))
                    };
                    let forward = (c0 - run_start).length() <= (c1 - run_start).length();
                    let (from, to) = if forward { (c0, c1) } else { (c1, c0) };
                    pieces.push(RunPiece::Contact {
                        stripe: si,
                        forward,
                        from,
                        to,
                        curve: contact,
                        from_vid: Option::None,
                        to_vid: Option::None,
                    });
                }
                Option::None => {
                    let last_oe = run_edges[run_edges.len() - 1];
                    let last_edge = topo.edge(last_oe.edge())?;
                    let to_vid = if last_oe.is_forward() {
                        last_edge.end()
                    } else {
                        last_edge.start()
                    };
                    let to = topo.vertex(to_vid)?.point();
                    pieces.push(RunPiece::Original {
                        edges: run_edges,
                        to_vid,
                        from_vid,
                        from: run_start,
                        to,
                    });
                }
            }
        }

        if std::env::var("BK_PIECES").is_ok() {
            for (k, piece) in pieces.iter().enumerate() {
                match piece {
                    RunPiece::Contact {
                        stripe, from, to, ..
                    } => log::warn!(
                        "PIECES face={face_id:?} [{k}] Contact s{stripe} ({from:?})->({to:?})"
                    ),
                    RunPiece::Original {
                        edges, from, to, ..
                    } => log::warn!(
                        "PIECES face={face_id:?} [{k}] Original n={} ({from:?})->({to:?})",
                        edges.len()
                    ),
                }
            }
        }

        let m = pieces.len();
        let piece_end = |p: &RunPiece| match p {
            RunPiece::Contact { to, .. } | RunPiece::Original { to, .. } => *to,
        };
        let piece_start = |p: &RunPiece| match p {
            RunPiece::Contact { from, .. } | RunPiece::Original { from, .. } => *from,
        };

        // Contact-to-contact junctions must weld (a failed corner leaves a
        // gap; those faces keep the trim path). Junctions INVOLVING an
        // original run are bridged with a line notch edge — the fillet band
        // ends abruptly there and the notch is the end cap's floor edge.
        // Contact-to-contact gaps (a corner whose vertex patch failed or a
        // mixed-radius junction) are bridged with a chord edge below —
        // the corner region's floor. Only unreasonably large gaps bail.
        let max_bridge = 4.0
            * regular_results
                .iter()
                .flat_map(|sr| sr.stripe.sections.iter().map(|s| s.radius))
                .fold(0.0_f64, f64::max);
        for k in 0..m {
            let a = &pieces[k];
            let b = &pieces[(k + 1) % m];
            let both_contact =
                matches!(a, RunPiece::Contact { .. }) && matches!(b, RunPiece::Contact { .. });
            let gap = (piece_end(a) - piece_start(b)).length();
            if both_contact && gap > max_bridge {
                continue 'faces;
            }
        }

        // Where a contact endpoint lands ON a neighbouring original LINE
        // edge (a full-edge fillet whose contact ends on the perpendicular
        // boundary), split that edge there — the classic trim behaviour,
        // propagated into neighbour wires — instead of overlaying a notch
        // line on top of the boundary.
        let seg_dist = |a: Point3, b: Point3, q: Point3| -> f64 {
            let ab = b - a;
            let len2 = ab.dot(ab);
            if len2 < 1e-18 {
                return (q - a).length();
            }
            let t = (ab.dot(q - a) / len2).clamp(0.0, 1.0);
            (q - (a + ab * t)).length()
        };
        for k in 0..m {
            let next = (k + 1) % m;
            // contact (k) -> original (next): contact END may lie on the
            // original run's FIRST edge.
            let (cp, is_end_side) = match (&pieces[k], &pieces[next]) {
                (RunPiece::Contact { to, .. }, RunPiece::Original { .. }) => (*to, true),
                (RunPiece::Original { .. }, RunPiece::Contact { from, .. }) => (*from, false),
                _ => continue,
            };
            let (orig_idx, edge_pos) = if is_end_side {
                (next, 0usize)
            } else {
                (k, usize::MAX)
            };
            let RunPiece::Original { edges, .. } = &pieces[orig_idx] else {
                continue;
            };
            let epos = if edge_pos == usize::MAX {
                edges.len() - 1
            } else {
                0
            };
            let oe = edges[epos];
            let edge = topo.edge(oe.edge())?;
            let (pa, pb) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            // Endpoint coincidence is the weld case, not a split.
            if (cp - pa).length() <= WELD || (cp - pb).length() <= WELD {
                continue;
            }
            enum SplitPlan {
                Line,
                Curve(
                    brepkit_math::nurbs::curve::NurbsCurve,
                    brepkit_math::nurbs::curve::NurbsCurve,
                ),
            }
            let plan = match edge.curve() {
                EdgeCurve::Line => {
                    if seg_dist(pa, pb, cp) > WELD {
                        continue;
                    }
                    SplitPlan::Line
                }
                EdgeCurve::NurbsCurve(nc) => {
                    let Ok(proj) =
                        brepkit_math::nurbs::projection::project_point_to_curve(nc, cp, 1e-9)
                    else {
                        continue;
                    };
                    if (proj.point - cp).length() > WELD {
                        continue;
                    }
                    let Ok((left, right)) =
                        brepkit_math::nurbs::knot_ops::curve_split(nc, proj.parameter)
                    else {
                        continue;
                    };
                    SplitPlan::Curve(left, right)
                }
                _ => continue,
            };
            let v_split = topo.add_vertex(Vertex::new(cp, 1e-7));
            let (pre, post) = match plan {
                SplitPlan::Line => trimmer::split_edge_at(topo, &oe, v_split)?,
                SplitPlan::Curve(left, right) => {
                    trimmer::split_edge_at_with_curves(topo, &oe, v_split, left, right)?
                }
            };
            // The kept sub-piece: the part AWAY from the contact junction.
            match (&mut pieces[orig_idx], is_end_side) {
                (
                    RunPiece::Original {
                        edges,
                        from_vid,
                        from,
                        ..
                    },
                    true,
                ) => {
                    edges[0] = post;
                    *from_vid = v_split;
                    *from = cp;
                }
                (
                    RunPiece::Original {
                        edges, to_vid, to, ..
                    },
                    false,
                ) => {
                    let last = edges.len() - 1;
                    edges[last] = pre;
                    *to_vid = v_split;
                    *to = cp;
                }
                _ => {}
            }
            let contact_idx = if is_end_side { k } else { next };
            match &mut pieces[contact_idx] {
                RunPiece::Contact { to_vid, .. } if is_end_side => {
                    *to_vid = Some(v_split);
                }
                RunPiece::Contact { from_vid, .. } if !is_end_side => {
                    *from_vid = Some(v_split);
                }
                _ => {}
            }
        }

        // Start vertex per piece: original runs reuse their existing outline
        // vertex; contact runs mint one.
        let mut junction_vids: Vec<brepkit_topology::vertex::VertexId> = Vec::with_capacity(m);
        for piece in &pieces {
            let vid = match piece {
                RunPiece::Original { from_vid, .. } => *from_vid,
                RunPiece::Contact {
                    from_vid: Some(v), ..
                } => *v,
                RunPiece::Contact { from, .. } => topo.add_vertex(Vertex::new(*from, 1e-7)),
            };
            junction_vids.push(vid);
        }

        let mut loop_edges: Vec<OrientedEdge> = Vec::with_capacity(m * 2);
        let mut notch_count = 0usize;
        for k in 0..m {
            let next = (k + 1) % m;
            let next_start = piece_start(&pieces[next]);
            match &pieces[k] {
                RunPiece::Contact {
                    stripe,
                    forward,
                    curve,
                    to,
                    to_vid,
                    ..
                } => {
                    let v_from = junction_vids[k];
                    let welds = (*to - next_start).length() <= WELD;
                    let v_to = if let Some(v) = to_vid {
                        *v
                    } else if welds {
                        junction_vids[next]
                    } else {
                        topo.add_vertex(Vertex::new(*to, 1e-7))
                    };
                    let curve_e = EdgeCurve::NurbsCurve(curve.clone());
                    let eid = if *forward {
                        topo.add_edge(Edge::new(v_from, v_to, curve_e))
                    } else {
                        topo.add_edge(Edge::new(v_to, v_from, curve_e))
                    };
                    loop_edges.push(OrientedEdge::new(eid, *forward));
                    out.insert((face_id, *stripe), eid);
                    if !welds {
                        // Contact-to-contact junction: the true boundary at
                        // an equal-radius corner is the OFFSET CORNER ARC
                        // centred on the shared spine vertex — the sphere
                        // corner patch's own bottom rim, so the weld pass
                        // pairs them. Mixed radii fall back to a chord.
                        let mut bridge_curve = EdgeCurve::Line;
                        if let RunPiece::Contact {
                            stripe: nsi,
                            from: nfrom,
                            ..
                        } = &pieces[next]
                        {
                            if let Some(cv) = shared_spine_vertex(
                                topo,
                                &regular_results[*stripe].stripe,
                                &regular_results[*nsi].stripe,
                            ) {
                                let c = topo.vertex(cv)?.point();
                                let r1 = (*to - c).length();
                                let r2 = (*nfrom - c).length();
                                if (r1 - r2).abs() <= 1e-6
                                    && let Ok(nrm) = (*to - c).cross(*nfrom - c).normalize()
                                    && let Ok(circle) = Circle3D::new(c, nrm, r1)
                                {
                                    bridge_curve = EdgeCurve::Circle(circle);
                                }
                            }
                        }
                        let notch =
                            topo.add_edge(Edge::new(v_to, junction_vids[next], bridge_curve));
                        loop_edges.push(OrientedEdge::new(notch, true));
                        notch_count += 1;
                        if matches!(pieces[next], RunPiece::Original { .. }) {
                            notches.push(NotchRecord {
                                stripe: *stripe,
                                edge: notch,
                                outline_vid: junction_vids[next],
                                contact_vid: v_to,
                                contact_pt: *to,
                            });
                        }
                    }
                }
                RunPiece::Original { edges, to_vid, .. } => {
                    loop_edges.extend(edges.iter().copied());
                    if (piece_end(&pieces[k]) - next_start).length() > WELD {
                        let notch =
                            topo.add_edge(Edge::new(*to_vid, junction_vids[next], EdgeCurve::Line));
                        loop_edges.push(OrientedEdge::new(notch, true));
                        notch_count += 1;
                        if let RunPiece::Contact { stripe: nsi, .. } = &pieces[next] {
                            notches.push(NotchRecord {
                                stripe: *nsi,
                                edge: notch,
                                outline_vid: *to_vid,
                                contact_vid: junction_vids[next],
                                contact_pt: next_start,
                            });
                        }
                    }
                }
            }
        }

        let new_wire = Wire::new(loop_edges, true)?;
        let new_wire_id = topo.add_wire(new_wire);
        let mut new_face = Face::new(new_wire_id, inner_wires, surface);
        new_face.set_reversed(reversed);
        let new_face_id = topo.add_face(new_face);
        face_replacements.insert(face_id, new_face_id);
        log::debug!(
            "mixed-loop rebuild: face {face_id:?} -> {new_face_id:?} pieces={m} notches={notch_count}"
        );
    }

    Ok((out, notches))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_topology::adjacency::AdjacencyIndex;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    #[test]
    fn fillet_builder_empty_edges_error() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let builder = FilletBuilder::new(&mut topo, solid);
        let result = builder.build();
        assert!(result.is_err(), "empty edge set should produce an error");
    }

    #[test]
    fn fillet_builder_plane_plane_box_edge() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let adjacency = AdjacencyIndex::build(&topo, solid).unwrap();
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let faces = topo.shell(shell_id).unwrap().faces().to_vec();

        let mut target_edge = None;
        'outer: for &fid in &faces {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                let adj = adjacency.faces_for_edge(oe.edge());
                if adj.len() == 2 {
                    target_edge = Some(oe.edge());
                    break 'outer;
                }
            }
        }
        let target_edge = target_edge.expect("cube should have manifold edges");

        let original_face_count = faces.len();
        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&[target_edge], 0.1);
        let result = builder.build().expect("fillet build should succeed");

        let result_solid = topo.solid(result.solid).unwrap();
        let result_shell = topo.shell(result_solid.outer_shell()).unwrap();

        // More faces than the original (6 original + 1 blend, minus possibly trimmed).
        assert!(
            result_shell.faces().len() > original_face_count,
            "expected more faces after fillet: got {}, original {}",
            result_shell.faces().len(),
            original_face_count,
        );

        assert!(result.succeeded.contains(&target_edge));
        assert!(result.failed.is_empty());
        assert!(!result.is_partial);
        assert_shell_edges_are_closed(&topo, result.solid);

        let mut found_cylinder = false;
        for &fid in result_shell.faces() {
            let face = topo.face(fid).unwrap();
            if matches!(face.surface(), FaceSurface::Cylinder(_)) {
                found_cylinder = true;
            }
        }
        assert!(
            found_cylinder,
            "fillet should produce a cylindrical blend surface"
        );
    }

    #[test]
    fn fillet_builder_uses_one_stripe_for_split_g1_contour() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let adjacency = AdjacencyIndex::build(&topo, solid).unwrap();
        let shell = topo.solid(solid).unwrap().outer_shell();
        let face_ids = topo.shell(shell).unwrap().faces().to_vec();
        let target = face_ids
            .iter()
            .flat_map(|&face_id| {
                topo.wire(topo.face(face_id).unwrap().outer_wire())
                    .unwrap()
                    .edges()
            })
            .find(|oriented| adjacency.faces_for_edge(oriented.edge()).len() == 2)
            .map(OrientedEdge::edge)
            .unwrap();
        let source_edge = topo.edge(target).unwrap().clone();
        let start = source_edge.start();
        let end = source_edge.end();
        let start_point = topo.vertex(start).unwrap().point();
        let end_point = topo.vertex(end).unwrap().point();
        let midpoint = topo.add_vertex(Vertex::new(
            start_point + (end_point - start_point) * 0.5,
            1e-7,
        ));
        let first = topo.add_edge(Edge::new(start, midpoint, EdgeCurve::Line));
        let second = topo.add_edge(Edge::new(midpoint, end, EdgeCurve::Line));
        for &face_id in adjacency.faces_for_edge(target) {
            let wire_id = topo.face(face_id).unwrap().outer_wire();
            let old = topo.wire(wire_id).unwrap().edges().to_vec();
            let mut replacement = Vec::with_capacity(old.len() + 1);
            for oriented in old {
                if oriented.edge() == target {
                    if oriented.is_forward() {
                        replacement.push(OrientedEdge::new(first, true));
                        replacement.push(OrientedEdge::new(second, true));
                    } else {
                        replacement.push(OrientedEdge::new(second, false));
                        replacement.push(OrientedEdge::new(first, false));
                    }
                } else {
                    replacement.push(oriented);
                }
            }
            *topo.wire_mut(wire_id).unwrap() = Wire::new(replacement, true).unwrap();
        }

        let plan = FilletPlan::build(
            &topo,
            solid,
            &[(vec![second, first], RadiusLaw::Constant(0.1))],
        )
        .unwrap();
        assert_eq!(plan.contours.len(), 1);
        assert_eq!(plan.contours[0].edges, vec![first, second]);
        let midpoint_junction = plan
            .junctions
            .iter()
            .find(|junction| junction.vertex == midpoint)
            .expect("split contour midpoint must be planned");
        assert_eq!(
            midpoint_junction.classification,
            crate::fillet_plan::CornerClassification::G1Continuation
        );

        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&[second, first], 0.1);
        let result = builder.build().expect("split contour should build");
        assert_eq!(result.succeeded, vec![first, second]);
        assert!(result.failed.is_empty());
        assert!(!result.is_partial);
        assert_shell_edges_are_closed(&topo, result.solid);
    }
    #[test]
    fn fillet_builder_records_failed_edges() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let v0 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            brepkit_math::vec::Point3::new(10.0, 10.0, 10.0),
            1e-7,
        ));
        let v1 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            brepkit_math::vec::Point3::new(11.0, 10.0, 10.0),
            1e-7,
        ));
        let fake_edge = topo.add_edge(brepkit_topology::edge::Edge::new(
            v0,
            v1,
            brepkit_topology::edge::EdgeCurve::Line,
        ));

        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&[fake_edge], 0.2);
        let result = builder.build().expect("build should succeed (partial)");

        assert!(result.failed.len() == 1);
        assert_eq!(result.failed[0].0, fake_edge);
        // With no successes, the original solid is returned.
        assert_eq!(result.solid, solid);
    }
    #[test]
    fn successful_stripe_is_failed_when_postassembly_gate_is_forced() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let adjacency = AdjacencyIndex::build(&topo, solid).unwrap();
        let shell = topo.solid(solid).unwrap().outer_shell();
        let edge = topo
            .shell(shell)
            .unwrap()
            .faces()
            .iter()
            .flat_map(|&face| {
                topo.wire(topo.face(face).unwrap().outer_wire())
                    .unwrap()
                    .edges()
            })
            .find(|oriented| adjacency.faces_for_edge(oriented.edge()).len() == 2)
            .unwrap()
            .edge();
        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&[edge], 0.1);
        let result = builder
            .build_with_forced_postassembly_failure()
            .expect("forced gate should be a recoverable per-edge failure");
        assert!(result.succeeded.is_empty());
        assert_eq!(result.failed.len(), 1);
        assert_eq!(result.failed[0].0, edge);
        assert!(
            result.failed[0]
                .1
                .to_string()
                .contains("assembly incidence gate")
        );
        assert!(result.is_partial);
    }
    #[test]
    fn second_pass_adjacent_to_nurbs_blend_reuses_registry_edges() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let adjacency = AdjacencyIndex::build(&topo, solid).unwrap();
        let shell = topo.solid(solid).unwrap().outer_shell();
        let target = topo
            .shell(shell)
            .unwrap()
            .faces()
            .iter()
            .flat_map(|&face| {
                topo.wire(topo.face(face).unwrap().outer_wire())
                    .unwrap()
                    .edges()
            })
            .find(|oe| adjacency.faces_for_edge(oe.edge()).len() == 2)
            .unwrap()
            .edge();
        let mut first = FilletBuilder::new(&mut topo, solid);
        first.add_edges_with_law(
            &[target],
            RadiusLaw::Linear {
                start: 0.08,
                end: 0.12,
            },
        );
        let first_result = first.build().unwrap();
        assert!(
            !first_result.is_partial,
            "first pass failed: {}",
            first_result
                .failed
                .iter()
                .map(|(_, error)| error.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let result_shell = topo.solid(first_result.solid).unwrap().outer_shell();
        let nurbs_edge = topo
            .shell(result_shell)
            .unwrap()
            .faces()
            .iter()
            .filter(|&&face| matches!(topo.face(face).unwrap().surface(), FaceSurface::Nurbs(_)))
            .flat_map(|&face| {
                topo.wire(topo.face(face).unwrap().outer_wire())
                    .unwrap()
                    .edges()
            })
            .find(|oe| oe.edge() != target)
            .unwrap()
            .edge();
        let mut second = FilletBuilder::new(&mut topo, first_result.solid);
        second.add_edges(&[nurbs_edge], 0.03);
        let second_result = second.build().unwrap();
        assert_eq!(second_result.succeeded, vec![nurbs_edge]);
        assert!(second_result.failed.is_empty());
        assert!(!second_result.is_partial);
    }
    fn assert_shell_edges_are_closed(topo: &Topology, solid: SolidId) {
        let shell = topo.solid(solid).unwrap().outer_shell();
        let mut uses = std::collections::HashMap::<usize, usize>::new();
        for &face_id in topo.shell(shell).unwrap().faces() {
            let face = topo.face(face_id).unwrap();
            let wires =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
            for wire_id in wires {
                for oriented in topo.wire(wire_id).unwrap().edges() {
                    *uses.entry(oriented.edge().index()).or_default() += 1;
                }
            }
        }
        let invalid: Vec<_> = uses.into_iter().filter(|(_, count)| *count != 2).collect();
        assert!(
            invalid.is_empty(),
            "result shell has free/non-manifold edges: {invalid:?}"
        );
    }

    fn cube_edges_at_vertex(topo: &Topology, solid: SolidId, vertex: VertexId) -> Vec<EdgeId> {
        let shell = topo.solid(solid).unwrap().outer_shell();
        let mut edges = Vec::new();
        for &face_id in topo.shell(shell).unwrap().faces() {
            let face = topo.face(face_id).unwrap();
            let wires =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
            for wire_id in wires {
                for oriented in topo.wire(wire_id).unwrap().edges() {
                    let edge_id = oriented.edge();
                    let edge = topo.edge(edge_id).unwrap();
                    if (edge.start() == vertex || edge.end() == vertex) && !edges.contains(&edge_id)
                    {
                        edges.push(edge_id);
                    }
                }
            }
        }
        edges.sort_unstable_by_key(|edge| edge.index());
        edges
    }

    #[test]
    fn fillet_builder_two_non_g1_edges_use_one_junction_solution() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let shell = topo.solid(solid).unwrap().outer_shell();
        let first_face = topo.shell(shell).unwrap().faces()[0];
        let first_wire = topo
            .wire(topo.face(first_face).unwrap().outer_wire())
            .unwrap();
        let target = first_wire.edges()[0].edge();
        let vertex = topo.edge(target).unwrap().start();
        let selected = cube_edges_at_vertex(&topo, solid, vertex);
        assert_eq!(selected.len(), 3);

        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&selected[..2], 0.1);
        let result = builder.build().expect("two-edge fillet should build");
        assert_eq!(result.succeeded.len(), 2);
        assert!(result.failed.is_empty());
        assert!(!result.is_partial);
        assert_shell_edges_are_closed(&topo, result.solid);
    }

    #[test]
    fn fillet_builder_trihedral_junction_reuses_registry_cross_edges() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let shell = topo.solid(solid).unwrap().outer_shell();
        let first_face = topo.shell(shell).unwrap().faces()[0];
        let first_wire = topo
            .wire(topo.face(first_face).unwrap().outer_wire())
            .unwrap();
        let target = first_wire.edges()[0].edge();
        let vertex = topo.edge(target).unwrap().start();
        let selected = cube_edges_at_vertex(&topo, solid, vertex);
        assert_eq!(selected.len(), 3);

        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&selected, 0.1);
        let result = builder.build().expect("three-edge fillet should build");
        assert_eq!(result.succeeded.len(), 3);
        assert!(result.failed.is_empty());
        assert!(!result.is_partial);
        let result_shell = topo.solid(result.solid).unwrap().outer_shell();
        assert!(
            topo.shell(result_shell)
                .unwrap()
                .faces()
                .iter()
                .any(|&face| {
                    matches!(topo.face(face).unwrap().surface(), FaceSurface::Nurbs(_))
                }),
            "three-stripe junction must publish a spherical NURBS patch"
        );
        assert_shell_edges_are_closed(&topo, result.solid);
    }

    #[test]
    fn fillet_builder_all_box_edges_use_ordered_fans() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let shell = topo.solid(solid).unwrap().outer_shell();
        let mut edges = Vec::new();
        for &face_id in topo.shell(shell).unwrap().faces() {
            let face = topo.face(face_id).unwrap();
            let wires =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
            for wire_id in wires {
                for oriented in topo.wire(wire_id).unwrap().edges() {
                    if !edges.contains(&oriented.edge()) {
                        edges.push(oriented.edge());
                    }
                }
            }
        }
        edges.sort_unstable_by_key(|edge| edge.index());
        assert_eq!(edges.len(), 12);

        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&edges, 0.1);
        let result = builder.build().expect("all-edge fillet should build");
        assert_eq!(result.succeeded.len(), edges.len());
        assert!(result.failed.is_empty());
        assert!(!result.is_partial);
        assert_shell_edges_are_closed(&topo, result.solid);
    }
}
