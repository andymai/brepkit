//! Fillet builder: orchestrates the full fillet pipeline.
//!
//! Spine construction, analytic/walking stripe computation, face trimming,
//! and solid assembly. Supports constant and variable radius fillets on
//! planar face pairs (v1).

use brepkit_math::madd::Madd;
use std::collections::HashSet;

use brepkit_math::aabb::Aabb3;
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
use crate::builder_utils::{
    FlippedNormalSurface, SolidSnapshot, sample_nurbs_endpoints, surface_ref_or_adapter,
};

use crate::corner;
use crate::fillet_plan::FilletPlan;
use crate::radius_law::RadiusLaw;
use crate::spine::Spine;
use crate::stripe::{Stripe, StripeResult};
use crate::trimmer;
use crate::walker::{Walker, WalkerConfig, approximate_blend_surface, interpolate_blend_surface};
use crate::{BlendError, BlendResult};

mod concave_first;
mod one_sphere;
mod wall_run_up;

type BlendCrossEdge = (EdgeId, VertexId, VertexId);
type BlendCrossPair = (Option<BlendCrossEdge>, Option<BlendCrossEdge>);
type BlendCrossHandlePair = (
    Option<crate::boundary_registry::BoundaryHandle>,
    Option<crate::boundary_registry::BoundaryHandle>,
);
/// The edges in the order given, each once: an edge picked twice in one set
/// (two picks landing on the same edge) asks for the same round, so it is
/// the same request. Across sets an edge may still be asked for twice, with
/// two laws, which is refused.
fn distinct_edges(edges: &[EdgeId]) -> Vec<EdgeId> {
    let mut seen = HashSet::new();
    edges
        .iter()
        .copied()
        .filter(|edge| seen.insert(edge.index()))
        .collect()
}

/// Builder for fillet (rounding) operations on solid edges.
///
/// Collects edge sets with their radius laws, then computes and assembles
/// the filleted solid in a single `build()` call.
///
/// The builder does not check its result. It checks how its faces share
/// edges, not where the faces lie, so a result can have a face whose wire
/// crosses itself, a hole across its face's outline, or two plane faces over
/// one area, with no edge in [`BlendResult::failed`]. Those checks live in
/// a crate above this one. Use `brepkit_operations::blend_ops::fillet_v2`
/// or `fillet_v2_variable`, which refuse such a result and make a fillet
/// across flat seams again with the seams merged.
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
            .push((distinct_edges(edges), RadiusLaw::Constant(radius)));
        self
    }

    /// Add edges with variable radius law.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn add_edges_with_law(&mut self, edges: &[EdgeId], law: RadiusLaw) -> &mut Self {
        self.edge_sets.push((distinct_edges(edges), law));
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
    /// The result is not checked for faults in where its faces lie (see
    /// [`FilletBuilder`]).
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
        if let Some(stages) = concave_first::stages(topo, solid, &edge_sets)? {
            // A staged attempt that falls back leaves nothing in the arena.
            let staged = topo.transaction(|topo| {
                concave_first::build(topo, solid, stages)
                    .map_err(Some)?
                    .ok_or(None)
            });
            match staged {
                Ok(result) => return Ok(result),
                Err(Some(error)) => return Err(error),
                Err(None) => {}
            }
        }
        let snapshot = SolidSnapshot::capture(topo, solid)?;
        let picked: Vec<EdgeId> = edge_sets
            .iter()
            .flat_map(|(edges, _)| edges.iter().copied())
            .collect();
        let shared: Vec<(Vec<EdgeId>, std::sync::Arc<RadiusLaw>)> = edge_sets
            .into_iter()
            .map(|(edges, law)| (edges, std::sync::Arc::new(law)))
            .collect();
        let first = build_after_joins(topo, solid, &snapshot, &shared, |topo| {
            one_sphere::join_sphere_faces_the_rim_crosses(topo, solid, &picked).map(Some)
        })
        .and_then(|built| {
            built.ok_or_else(|| BlendError::PlanningFailure {
                reason: "the fillet was not built".into(),
            })
        });
        if first.as_ref().is_ok_and(|result| result.failed.is_empty()) {
            return first;
        }
        // A round of a rim on a sphere's narrow band whose contact runs past
        // the band's far loop in part is made again with the band and the
        // face beyond joined. It is kept only where it rounds every edge.
        let joined = topo.transaction(|topo| {
            let built = build_after_joins(topo, solid, &snapshot, &shared, |topo| {
                let bands = one_sphere::join_sphere_bands_at_picked_rims(topo, solid, &picked)?;
                if bands.is_empty() {
                    return Ok(None);
                }
                let mut gone = one_sphere::join_sphere_faces_the_rim_crosses(topo, solid, &picked)?;
                gone.extend(bands);
                Ok(Some(gone))
            });
            match built {
                Ok(Some(result)) if result.failed.is_empty() => Ok(result),
                _ => Err(()),
            }
        });
        joined.or(first)
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
        let scope = trimmer::SplitScope::of_solid(self.topo, self.solid)?;
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

        // Refuse an inadmissible equal-radius two-edge miter request on its
        // real source-edge lengths before any mutation below, since stripe
        // computation is the first thing that allocates. A refusal must not
        // fall through to the horn-torus path (see
        // `corner::check_equal_two_edge_admissibility`).
        if let Some(fillet_plan) = plan.as_ref() {
            // The *requested* selection, not the plan's post-drop contour
            // list: the plan drops tangent-continuous selected edges, and a
            // seam's continuation may be exactly such an edge.
            let requested: HashSet<EdgeId> = self
                .edge_sets
                .iter()
                .flat_map(|(edges, _)| edges.iter().copied())
                .collect();
            corner::check_equal_two_edge_admissibility(self.topo, fillet_plan, &requested)?;
        }

        // Keep actual RadiusLaw values only for the recoverable invalid-edge
        // fallback. Valid geometry always consumes the immutable plan's law.
        let mut all_edges: Vec<(EdgeId, usize)> = Vec::new();
        // The edges in the order they were picked: where two rounds cross,
        // the first picked is made as it would be alone.
        let pick_order: Vec<EdgeId> = self
            .edge_sets
            .iter()
            .flat_map(|(edges, _)| edges.iter().copied())
            .collect();
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
        let source = topo.solid(self.solid)?;
        // Only the outer shell's edges are blended (the adjacency reads no
        // other), so the cavities carry over as they are.
        let (shell_id, inner_shells) = (source.outer_shell(), source.inner_shells().to_vec());
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
        // The cavities carry over untouched, so the material a stripe takes
        // off must stay clear of them: a fillet reaching one would leave the
        // new outer shell running through it.
        if !inner_shells.is_empty() {
            let cavities = inner_shells
                .iter()
                .map(|&shell| shell_bounds(topo, shell))
                .collect::<Result<Vec<_>, _>>()?;
            for sr in &stripe_results {
                let removed = removed_material_bounds(topo, &sr.stripe)?;
                if cavities.iter().any(|cavity| cavity.intersects(removed)) {
                    for edge_id in succeeded_candidates(plan.as_ref(), &stripe_results, &all_edges)
                    {
                        failed.push((
                            edge_id,
                            BlendError::PlanningFailure {
                                reason: "the fillet reaches a cavity".to_owned(),
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
            }
        }
        let setback_edges = if let Some(plan) = plan.as_ref() {
            // `original_faces` is this build's own solid's shell, captured
            // before any mutation: the correct scope for "incident to this
            // corner" even when `topo`'s arena also holds an unrelated,
            // previously-built result solid from an earlier command against
            // the same source (see `incident_face_count`'s doc comment).
            corner::set_back_convex_trihedral_stripes(
                topo,
                plan,
                &mut stripe_results,
                &original_faces,
            )?
        } else {
            HashSet::new()
        };
        pivot_crossing_rounds(topo, &mut stripe_results, &pick_order)?;
        let tilted_caps = if let Some(plan) = plan.as_ref() {
            run_stripes_to_tilted_caps(topo, plan, &mut stripe_results)?
        } else {
            TiltedCaps::new()
        };
        // Running a round on to a tilted cap carries its contacts past where
        // the cavity check above read them.
        if !tilted_caps.is_empty() && !inner_shells.is_empty() {
            let cavities = inner_shells
                .iter()
                .map(|&shell| shell_bounds(topo, shell))
                .collect::<Result<Vec<_>, _>>()?;
            for sr in &stripe_results {
                let removed = removed_material_bounds(topo, &sr.stripe)?;
                if cavities.iter().any(|cavity| cavity.intersects(removed)) {
                    for edge_id in succeeded_candidates(plan.as_ref(), &stripe_results, &all_edges)
                    {
                        failed.push((
                            edge_id,
                            BlendError::PlanningFailure {
                                reason: "the fillet reaches a cavity".to_owned(),
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
            }
        }
        // through the trim + corner + blend-face path below.
        let mut boundary_registry = BoundaryRegistry::new();
        let mut blend_face_ids: Vec<FaceId> = Vec::new();
        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();
        let mut regular_results: Vec<&StripeResult> = Vec::new();

        let recognized = stripe_results
            .iter()
            .map(|sr| closed_rim_info(topo, &sr.stripe))
            .collect::<Result<Vec<_>, BlendError>>()?;
        // Each rim's wall contact is checked against the wall alone, so two
        // rims of one wall must also leave wall between their contacts, or
        // their bands overlap, whichever path below builds them.
        let rims: Vec<&ClosedRimInfo> = recognized.iter().flatten().collect();
        let crossing = rims.iter().enumerate().any(|(i, a)| {
            rims[i + 1..]
                .iter()
                .any(|b| a.wall_face == b.wall_face && !contacts_leave_wall_between(a, b))
        });
        if crossing {
            for edge_id in succeeded_candidates(plan.as_ref(), &stripe_results, &all_edges) {
                failed.push((
                    edge_id,
                    BlendError::PlanningFailure {
                        reason: "two rim fillets' wall contacts meet on their shared wall"
                            .to_owned(),
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

        // The closed-rim rebuild owns its cap: a cap another stripe also runs
        // along (a hole's rim filleted in the same call) takes the normal path.
        let mut closed_rims: Vec<Option<ClosedRimInfo>> = recognized
            .into_iter()
            .enumerate()
            .map(|(stripe_index, rim)| {
                rim.filter(|rim| {
                    !stripe_results.iter().enumerate().any(|(other, o)| {
                        other != stripe_index
                            && (o.stripe.face1 == rim.plane_face
                                || o.stripe.face2 == rim.plane_face)
                    })
                })
            })
            .collect();
        // Nor may it own a wall that a stripe on the normal path trims too
        // (the wall's other rim, its cap holed by a B-spline outline): that
        // trim starts from the wall as it came, so it would drop the rebuild's
        // contact circle and keep the rim it replaced. Such a rim takes the
        // normal path as well, which trims both contacts on the one wall.
        loop {
            let normal_path_faces: HashSet<FaceId> = stripe_results
                .iter()
                .zip(&closed_rims)
                .filter(|(_, rim)| rim.is_none())
                .flat_map(|(sr, _)| [sr.stripe.face1, sr.stripe.face2])
                .collect();
            let mut changed = false;
            for rim in &mut closed_rims {
                if rim
                    .as_ref()
                    .is_some_and(|rim| normal_path_faces.contains(&rim.wall_face))
                {
                    *rim = None;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut closed_rim_supports: HashSet<FaceId> = HashSet::new();
        // Walls a closed rim's round took whole (a wall exactly a radius
        // high): left out of the result.
        let mut vanished_walls: HashSet<FaceId> = HashSet::new();
        for (stripe_index, (sr, closed_rim)) in stripe_results.iter().zip(closed_rims).enumerate() {
            if let Some(rim) = closed_rim {
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
                    Ok(band) => {
                        blend_face_ids.push(band);
                        if rim.far_rim.is_some() {
                            vanished_walls.insert(rim.wall_face);
                        }
                        closed_rim_supports.extend(
                            [rim.plane_face, rim.wall_face]
                                .iter()
                                .filter_map(|support| face_replacements.get(support)),
                        );
                    }
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
            rebuild_closed_rim_loop_faces(topo, &scope, &regular_results, &mut face_replacements)?
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
        // Every piece of each side's contact, along the contact, where the
        // trim cut it in pieces (on its support's own seam); empty otherwise,
        // the side's one handle above being the whole contact.
        let mut stripe_contact_chains: Vec<[Vec<crate::boundary_registry::BoundaryHandle>; 2]> =
            regular_results
                .iter()
                .map(|_| [Vec::new(), Vec::new()])
                .collect();
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
                // A face the stripe does not run along lies beyond a plane
                // face the round swallowed (`swallow_narrow_face`): its
                // contact replaces the loop it shares with that face, and
                // the face goes.
                let spine_faces = stripe
                    .spine_edges()
                    .first()
                    .map(|&edge| adjacency.faces_for_edge(edge).to_vec())
                    .unwrap_or_default();
                if !spine_faces.contains(&face_id) {
                    let swallowed: Vec<FaceId> = spine_faces
                        .iter()
                        .copied()
                        .filter(|&f| f != stripe.face1 && f != stripe.face2)
                        .collect();
                    let [plane] = swallowed[..] else {
                        return Err(BlendError::TrimmingFailure { face: face_id });
                    };
                    let face_edges = |face: FaceId| -> Result<HashSet<EdgeId>, BlendError> {
                        let f = topo.face(face)?;
                        let mut edges = HashSet::new();
                        for wire in
                            std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied())
                        {
                            edges.extend(topo.wire(wire)?.edges().iter().map(OrientedEdge::edge));
                        }
                        Ok(edges)
                    };
                    let plane_edges = face_edges(plane)?;
                    let f = topo.face(face_id)?;
                    let mut far = Vec::new();
                    for wire in
                        std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied())
                    {
                        for oriented in topo.wire(wire)?.edges() {
                            if plane_edges.contains(&oriented.edge())
                                && !far.contains(&oriented.edge())
                            {
                                far.push(oriented.edge());
                            }
                        }
                    }
                    restriction.source_edges = far;
                    vanished_walls.insert(plane);
                }
                let source_surface = topo.face(face_id)?.surface().clone();
                let mut adapter = None;
                let support = surface_ref_or_adapter(&source_surface, &mut adapter);
                let pcurve = if source_surface.is_planar() {
                    let (pcurve, start, end) = build_planar_pcurve_from_contact(support, &contact)?;
                    brepkit_topology::pcurve::PCurve::new(pcurve, start, end)
                } else {
                    // The trimmer's weld distance: ends this close make one
                    // closed contact.
                    const CLOSED: f64 = 1e-7;
                    let [start, end] = restriction.contact_3d[..] else {
                        return Err(BlendError::TrimmingFailure { face: face_id });
                    };
                    let end = if (end - start).length() <= CLOSED {
                        start
                    } else {
                        end
                    };
                    // The trim cuts a contact that crosses the face's seam
                    // there.
                    let fitted = crate::builder_utils::pcurve_along(
                        &source_surface,
                        &EdgeCurve::NurbsCurve(contact.clone()),
                        (start, end),
                        true,
                    );
                    // A walked contact standing off a file's B-spline wall
                    // farther than the fit allows takes the line between
                    // its ends' parameters, as it did before contacts were
                    // fitted along their edges.
                    if let Some(pcurve) = fitted {
                        pcurve
                    } else {
                        let analytic = if face_id == stripe.face1 {
                            &stripe.pcurve1
                        } else {
                            &stripe.pcurve2
                        };
                        fallback_contact_pcurve(support, &contact, &source_surface, analytic)?
                    }
                };
                restriction.curve = Some(EdgeCurve::NurbsCurve(contact));
                restriction.pcurve = Some(pcurve);
                restrictions_by_face
                    .entry(face_id)
                    .or_default()
                    .push((si, restriction));
            }
        }
        let single_planar_stripe = all_edges.len() == 1
            && regular_results.len() == 1
            && laws.len() == 1
            && matches!(laws[0], RadiusLaw::Constant(_))
            && [
                regular_results[0].stripe.face1,
                regular_results[0].stripe.face2,
            ]
            .into_iter()
            .all(|face_id| {
                topo.face(face_id)
                    .is_ok_and(|face| face.surface().is_planar())
            });
        // A terminal vertex whose third face is a planar cap that both
        // contacts end on takes the notch: the spoke splits there on every
        // face and the cap adopts the cross-section arc. Any other terminal
        // keeps the runout closure, which expects the whole spoke. Without a
        // plan every source vertex is treated as an unsplit terminal.
        let mut unsplit_vertices: HashSet<VertexId> = HashSet::new();
        let mut notchable_vertices: HashSet<VertexId> = HashSet::new();
        if let Some(fillet_plan) = plan.as_ref() {
            for junction in &fillet_plan.junctions {
                if !matches!(
                    junction.classification,
                    crate::fillet_plan::CornerClassification::Terminal
                ) {
                    continue;
                }
                let contour = junction
                    .incident_contours
                    .first()
                    .and_then(|&contour| fillet_plan.contours.get(contour));
                let stripe = contour.and_then(|contour| {
                    regular_results
                        .iter()
                        .map(|result| &result.stripe)
                        .find(|stripe| stripe.spine_edges() == contour.spine.edges())
                });
                let notchable = match (contour, stripe) {
                    (Some(contour), Some(stripe)) => {
                        terminal_cap_is_notchable(topo, junction, contour, stripe, &tilted_caps)?
                    }
                    _ => false,
                };
                log::debug!(
                    "terminal {:?}: notchable={notchable} contours={:?} fan={:?}",
                    junction.vertex,
                    junction.incident_contours,
                    junction.face_fan
                );
                if notchable {
                    notchable_vertices.insert(junction.vertex);
                } else {
                    unsplit_vertices.insert(junction.vertex);
                }
            }
        } else {
            unsplit_vertices = all_edges
                .iter()
                .filter_map(|(edge_id, _)| topo.edge(*edge_id).ok())
                .flat_map(|edge| [edge.start(), edge.end()])
                .collect();
        }
        // Straight edges a trim splits whatever its policy, by their two
        // ends (see the policy's `spokes`): every straight sharp edge left
        // unselected at a junction of two contours, and the smooth edge two
        // stripes run on round on one section. The contacts of the rounds on
        // a sharp edge's two faces end on it: at a convex corner it is a
        // mitre's retained edge, and at a reflex one the rounds pivot on it.
        // Whole, it would take each face's wire on along it to the corner and
        // back.
        let mut spokes: Vec<(brepkit_math::vec::Point3, brepkit_math::vec::Point3)> = Vec::new();
        if let Some(fillet_plan) = plan.as_ref() {
            for junction in &fillet_plan.junctions {
                if junction.incident_contours.len() != 2 {
                    continue;
                }
                for &edge_id in &junction.unselected_sharp_edges {
                    let edge = topo.edge(edge_id)?;
                    let spoke = (
                        topo.vertex(edge.start())?.point(),
                        topo.vertex(edge.end())?.point(),
                    );
                    if trimmer::is_straight(edge.curve(), spoke.0, spoke.1)
                        && !spokes.contains(&spoke)
                    {
                        spokes.push(spoke);
                    }
                }
            }
            let stripes: Vec<&Stripe> = regular_results
                .iter()
                .map(|result| &result.stripe)
                .collect();
            for spoke in corner::tangent_junction_spokes(topo, fillet_plan, &stripes)? {
                if !spokes.contains(&spoke) {
                    spokes.push(spoke);
                }
            }
        }
        // A one-edge fillet splits every edge so its end caps take the notch,
        // unless an end's cap cannot: that end keeps its runout closure,
        // which expects the whole spoke, as it does in a fillet of several.
        let split_policy = if single_planar_stripe && unsplit_vertices.is_empty() {
            trimmer::BoundarySplitPolicy::All
        } else {
            trimmer::BoundarySplitPolicy::Selective {
                unsplit: &unsplit_vertices,
                notchable: &notchable_vertices,
                spokes: &spokes,
            }
        };
        let mut restriction_faces: Vec<FaceId> = restrictions_by_face.keys().copied().collect();
        restriction_faces.sort_unstable_by_key(|face| face.index());
        for face_id in &restriction_faces {
            if let Some(restrictions) = restrictions_by_face.get(face_id) {
                check_closed_contacts_do_not_cross(*face_id, restrictions)?;
                check_contacts_keep_their_edges(topo, *face_id, restrictions)?;
                check_apart_contacts_do_not_cross(topo, *face_id, restrictions)?;
            }
        }
        for face_id in restriction_faces {
            let restrictions = restrictions_by_face
                .remove(&face_id)
                .ok_or(BlendError::TrimmingFailure { face: face_id })?;
            let restriction_plan: Vec<trimmer::ParametricRestriction> = restrictions
                .iter()
                .map(|(_, restriction)| restriction.clone())
                .collect();
            let batch = trimmer::trim_parametric_face_batch(
                topo,
                &scope,
                face_id,
                &restriction_plan,
                split_policy,
            )?;
            face_replacements.insert(face_id, batch.trimmed_face);
            for &(source, replacement) in &batch.incident_replacements {
                face_replacements.entry(source).or_insert(replacement);
            }
            for (index, &(si, _)) in restrictions.iter().enumerate() {
                let Some(pieces) = batch.contact_pieces.get(index) else {
                    return Err(BlendError::TrimmingFailure { face: face_id });
                };
                let mut chain = Vec::with_capacity(pieces.len());
                for (piece, &contact_edge) in pieces.iter().enumerate() {
                    let stripe = &regular_results[si].stripe;
                    let support_forward =
                        std::iter::once(topo.face(batch.trimmed_face)?.outer_wire())
                            .chain(topo.face(batch.trimmed_face)?.inner_wires().iter().copied())
                            .find_map(|wire_id| {
                                topo.wire(wire_id)
                                    .ok()?
                                    .edges()
                                    .iter()
                                    .find_map(|oriented| {
                                        (oriented.edge() == contact_edge)
                                            .then_some(oriented.is_forward())
                                    })
                            })
                            .ok_or(BlendError::TrimmingFailure { face: face_id })?;
                    let contour_id =
                        plan.as_ref()
                            .and_then(|fillet_plan| {
                                fillet_plan.contours.iter().position(|contour| {
                                    contour.spine.edges() == stripe.spine_edges()
                                })
                            })
                            .unwrap_or(si);
                    let side = if stripe.face1 == face_id { 0 } else { 1 };
                    let edge = topo.edge(contact_edge)?;
                    let start = edge.start();
                    let end = edge.end();
                    let curve = if pieces.len() == 1 {
                        restrictions[index]
                            .1
                            .curve
                            .clone()
                            .unwrap_or(EdgeCurve::Line)
                    } else {
                        edge.curve().clone()
                    };
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
                    // A contact in pieces keys the vertices between them
                    // apart from its two ends.
                    let key = |at: usize| -> u64 {
                        let slot = match at {
                            0 => 0,
                            at if at == pieces.len() => 1,
                            at => 1 + at as u64,
                        };
                        (contour_id as u64) << 32 | (side as u64) << 8 | slot
                    };
                    let start_vertex = if periodic_support {
                        PlannedVertex::periodic(start, key(piece))
                    } else {
                        PlannedVertex::new(start)
                    };
                    let end_vertex = if periodic_support {
                        PlannedVertex::periodic(end, key(piece + 1))
                    } else {
                        PlannedVertex::new(end)
                    };
                    let handle = boundary_registry.register(
                        BoundaryKey::contact(contour_id, piece, side),
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
                    chain.push(handle);
                    if piece > 0 {
                        continue;
                    }
                    if side == 0 {
                        stripe_contact_handles[si].0 = Some(handle);
                        stripe_contact_edges[si].0 = Some(contact_edge);
                    } else {
                        stripe_contact_handles[si].1 = Some(handle);
                        stripe_contact_edges[si].1 = Some(contact_edge);
                    }
                }
                if chain.len() > 1 {
                    let side = usize::from(regular_results[si].stripe.face1 != face_id);
                    stripe_contact_chains[si][side] = chain;
                }
            }
        }

        let mut blend_cross_edges: Vec<BlendCrossEdge> = Vec::new();
        let mut blend_cross_by_stripe: Vec<BlendCrossPair> =
            vec![(None, None); regular_results.len()];
        let mut blend_cross_handles: Vec<BlendCrossHandlePair> =
            vec![(None, None); regular_results.len()];
        // A sharp-mitered two-edge corner installs one exact crease edge
        // shared by both stripe faces in place of what the ordinary
        // cross-section placeholder arc would otherwise be. Two
        // independently registered placeholder arcs (one per stripe, at two
        // different vertex identities) can never be merged into one true
        // shared crease afterwards without orphaning a registry entry, so a
        // miter end's cross-section boundary is not registered at all here:
        // that stripe face is built with an intentionally open wire at this
        // one end (`Wire::new` does not itself validate closure), and
        // `corner.rs`'s miter builder closes it by inserting the crease edge
        // once both stripe faces and their shared p00/vtop vertices exist.
        //
        // Terminal ends are deliberately NOT special-cased: the
        // notch/`Selective` machinery owns every terminal cap
        // (`terminal_cap_is_notchable` + `notch_face_corner_with_arc`), and
        // for a mitered stripe's far terminal it already produces the same
        // "cap adopts the registered cross-section arc" outcome.
        let miter_vertices: HashSet<VertexId> = plan
            .as_ref()
            .map(|fillet_plan| {
                corner::find_equal_two_edge_miter_junctions(topo, fillet_plan)
                    .into_iter()
                    .map(|m| m.vertex)
                    .collect()
            })
            .unwrap_or_default();
        // The miter corner builder needs each stripe's own generated face id
        // directly: when a stripe has a miter junction at *both* ends,
        // `blend_cross_handles` (which the ordinary junction/runout stage
        // otherwise uses to recover a stripe's face via its cross-section
        // boundary's owner) carries `(None, None)` and cannot serve this
        // lookup.
        let mut stripe_faces: Vec<Option<FaceId>> = vec![None; regular_results.len()];
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
                let chains = [
                    if stripe_contact_chains[si][0].is_empty() {
                        vec![contact1]
                    } else {
                        stripe_contact_chains[si][0].clone()
                    },
                    if stripe_contact_chains[si][1].is_empty() {
                        vec![contact2]
                    } else {
                        stripe_contact_chains[si][1].clone()
                    },
                ];
                for (side, chain) in chains.iter().enumerate() {
                    if let Some(forward) = contact_forward_for_spine(
                        topo,
                        &boundary_registry,
                        chain,
                        support_faces[side],
                        terminals,
                        side == 0,
                    )? {
                        for &handle in chain {
                            boundary_registry.set_owner_forward(handle, 1, forward)?;
                        }
                    }
                }
                // The band runs each contact's chain forward or backward
                // whole: its ends are the chain's.
                let contact_vertices = |chain: &[crate::boundary_registry::BoundaryHandle]| -> Result<
                    (VertexId, VertexId),
                    BlendError,
                > {
                    let (start, end) = contact_chain_ends(&boundary_registry, chain)?;
                    let forward = chain
                        .first()
                        .and_then(|&handle| boundary_registry.entry(handle))
                        .is_some_and(|entry| entry.owners[1].forward);
                    Ok(if forward { (start, end) } else { (end, start) })
                };
                let (p1_start, p1_end) = contact_vertices(&chains[0])?;
                let (p2_start, p2_end) = contact_vertices(&chains[1])?;
                let mut section_curve = |section: Option<&crate::section::CircSection>,
                                         cap: Option<&RunOnCap>,
                                         start: VertexId,
                                         end: VertexId|
                 -> Result<
                    Option<(crate::boundary_registry::BoundaryHandle, usize)>,
                    BlendError,
                > {
                    if start == end {
                        return Ok(None);
                    }
                    let start_point = topo.vertex(start)?.point();
                    let end_point = topo.vertex(end)?.point();
                    let curve = match cap {
                        // A chord would leave the band's end off the cap and
                        // off the band.
                        Some(&cap) => band_meets_run_on_cap(
                            topo,
                            &stripe.surface,
                            cap,
                            start_point,
                            end_point,
                        )?
                        .ok_or_else(|| BlendError::PlanningFailure {
                            reason: match cap {
                                RunOnCap::Plane(..) => {
                                    "the round's end on its tilted cap cannot be traced".to_owned()
                                }
                                RunOnCap::Face(face) => format!(
                                    "the round's end on face {} cannot be traced",
                                    face.index()
                                ),
                            },
                        })?,
                        None => section
                            .and_then(|section| {
                                crate::builder_utils::cross_section_curve(
                                    section,
                                    start_point,
                                    end_point,
                                )
                            })
                            .unwrap_or(EdgeCurve::Line),
                    };
                    // Two stripes meeting tangentially across a seam between
                    // faces on one surface end at the same section. The
                    // second band becomes the second owner of the first
                    // band's cross-section edge instead of minting a twin
                    // arc that would leave both with a single face use.
                    let midpoint =
                        |curve: &EdgeCurve,
                         a: brepkit_math::vec::Point3,
                         b: brepkit_math::vec::Point3| {
                            let (t0, t1) = curve.domain_with_endpoints(a, b);
                            curve.evaluate_with_endpoints(0.5 * (t0 + t1), a, b)
                        };
                    let own_midpoint = midpoint(&curve, start_point, end_point);
                    let shared = blend_cross_handles
                        .iter()
                        .flat_map(|(end_handle, start_handle)| [*end_handle, *start_handle])
                        .flatten()
                        .find(|&handle| {
                            boundary_registry.entry(handle).is_some_and(|entry| {
                                let same_vertices = (entry.start.vertex == start
                                    && entry.end.vertex == end)
                                    || (entry.start.vertex == end && entry.end.vertex == start);
                                if !same_vertices
                                    || entry.owners[1].face.is_some()
                                    || entry.edge_id().is_none()
                                {
                                    return false;
                                }
                                let (Ok(a), Ok(b)) = (
                                    topo.vertex(entry.start.vertex),
                                    topo.vertex(entry.end.vertex),
                                ) else {
                                    return false;
                                };
                                (midpoint(&entry.curve, a.point(), b.point()) - own_midpoint)
                                    .length()
                                    <= 1e-6
                            })
                        });
                    if let Some(handle) = shared {
                        let forward = boundary_registry
                            .entry(handle)
                            .is_some_and(|entry| entry.start.vertex == start);
                        boundary_registry.set_owner_forward(handle, 1, forward)?;
                        log::debug!(
                            "cross-section {start:?}->{end:?} of contour {contour_id} adopts handle {handle}"
                        );
                        return Ok(Some((handle, 1)));
                    }
                    log::debug!(
                        "cross-section {start:?}->{end:?} of contour {contour_id} registered fresh ({:.7},{:.7},{:.7})->({:.7},{:.7},{:.7})",
                        start_point.x(),
                        start_point.y(),
                        start_point.z(),
                        end_point.x(),
                        end_point.y(),
                        end_point.z()
                    );
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
                    Ok(Some((handle, 0)))
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
                        &chains[0],
                        &chains[1],
                    )?;
                    for &handle in chains.iter().flatten() {
                        boundary_registry.set_owner_face(handle, 1, info.face)?;
                    }
                    blend_cross_handles[si] = (None, None);
                    stripe_faces[si] = Some(info.face);
                    info
                } else {
                    let (spine_start_v, spine_end_v) =
                        terminals.ok_or_else(|| BlendError::PlanningFailure {
                            reason: "open stripe spine missing terminals".to_owned(),
                        })?;
                    let cross_start = if miter_vertices.contains(&spine_start_v) {
                        None
                    } else {
                        section_curve(
                            stripe.sections.first(),
                            tilted_caps.get(&(stripe.spine_edges()[0], 0)),
                            p2_end,
                            p1_start,
                        )?
                    };
                    let cross_end = if miter_vertices.contains(&spine_end_v) {
                        None
                    } else {
                        section_curve(
                            stripe.sections.last(),
                            tilted_caps.get(&(stripe.spine_edges()[0], 1)),
                            p1_end,
                            p2_start,
                        )?
                    };
                    // Each chain in the order the band runs it.
                    let band_order = |chain: &[crate::boundary_registry::BoundaryHandle]| {
                        let forward = chain
                            .first()
                            .and_then(|&handle| boundary_registry.entry(handle))
                            .is_some_and(|entry| entry.owners[1].forward);
                        let mut ordered: Vec<_> = chain.iter().map(|&handle| (handle, 1)).collect();
                        if !forward {
                            ordered.reverse();
                        }
                        ordered
                    };
                    let (band1, band2) = (band_order(&chains[0]), band_order(&chains[1]));
                    let info = crate::builder_utils::create_blend_face_from_registry(
                        topo,
                        stripe,
                        &mut boundary_registry,
                        &band1,
                        &band2,
                        cross_end,
                        cross_start,
                    )?;
                    for (handle, owner) in [cross_end, cross_start].into_iter().flatten() {
                        boundary_registry.set_owner_face(handle, owner, info.face)?;
                    }
                    for &handle in chains.iter().flatten() {
                        boundary_registry.set_owner_face(handle, 1, info.face)?;
                    }
                    blend_cross_handles[si] = (
                        cross_end.map(|(handle, _)| handle),
                        cross_start.map(|(handle, _)| handle),
                    );
                    stripe_faces[si] = Some(info.face);
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
            if stripe
                .spine_edges()
                .iter()
                .any(|edge| setback_edges.contains(edge))
                && let Some((side, contact_edge)) =
                    [stripe_contact_edges[si].0, stripe_contact_edges[si].1]
                        .into_iter()
                        .enumerate()
                        .find_map(|(side, edge)| edge.map(|edge| (side, edge)))
            {
                align_face_boundary_against_neighbor(
                    topo,
                    info.face,
                    support_faces[side],
                    contact_edge,
                )?;
                boundary_registry.refresh_owner_orientations(topo)?;
            }
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

        let mut notched_caps: HashSet<FaceId> = HashSet::new();
        // Terminal ends notch the untouched cap BEFORE the corner solver runs:
        // once every wall has dropped the spoke pieces below its contact, the
        // runout would otherwise chain the corner arcs and the walls' kept rim
        // pieces into one loop and rebuild the whole cap as a second patch.
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
            log::debug!("notch attempt: arc {:?} {:?}->{:?}", arc.0, arc.1, arc.2);
            for (orig, fid) in candidates {
                if let Some(nf) = crate::builder_utils::notch_face_corner_with_arc(topo, fid, *arc)?
                {
                    face_replacements.insert(orig, nf);
                    notched_caps.insert(nf);
                    log::debug!("  notched cap {orig:?} -> {nf:?}");
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

        // The ordered junction solver needs the final copy-on-write support
        // faces so residual terminal edges can be registered as runout
        // boundaries instead of being closed by positional repair.
        let support_faces_of = |face_replacements: &std::collections::HashMap<FaceId, FaceId>| {
            stripes
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
                .collect::<Vec<(FaceId, FaceId)>>()
        };
        let mut stripe_support_faces = support_faces_of(&face_replacements);
        if let Some(fillet_plan) = plan.as_ref() {
            let current = |face: FaceId| face_replacements.get(&face).copied().unwrap_or(face);
            let (crossed, stub_faces) = corner::apply_stub_crossings(
                topo,
                &stripes,
                &fillet_plan.junctions,
                &contour_to_stripe,
                &stripe_support_faces,
                &stripe_faces,
                &current,
                &mut boundary_registry,
            )?;
            face_replacements.extend(stub_faces);
            stripe_support_faces = support_faces_of(&face_replacements);
            let met = corner::apply_junction_crossings(
                topo,
                &stripes,
                &fillet_plan.junctions,
                &contour_to_stripe,
                &stripe_support_faces,
                &stripe_faces,
                &mut boundary_registry,
            )?;
            let junctions: Vec<crate::fillet_plan::VertexJunction> = fillet_plan
                .junctions
                .iter()
                .filter(|junction| {
                    !crossed.contains(&junction.vertex) && !met.contains(&junction.vertex)
                })
                .cloned()
                .collect();
            let corner_results = corner::compute_ordered_corners(
                topo,
                &stripes,
                &junctions,
                &contour_to_stripe,
                &blend_cross_handles,
                &stripe_support_faces,
                &stripe_faces,
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
            log::debug!("notch attempt: arc {:?} {:?}->{:?}", arc.0, arc.1, arc.2);
            for (orig, fid) in candidates {
                if let Some(nf) = crate::builder_utils::notch_face_corner_with_arc(topo, fid, *arc)?
                {
                    face_replacements.insert(orig, nf);
                    notched_caps.insert(nf);
                    log::debug!("  notched cap {orig:?} -> {nf:?}");
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
                let chain = if stripe_contact_chains[si][side].is_empty() {
                    vec![handle]
                } else {
                    stripe_contact_chains[si][side].clone()
                };
                for handle in chain {
                    boundary_registry.set_owner_face(handle, 0, support_face)?;
                    let _ = boundary_registry.oriented_edge(topo, handle, 0)?;
                }
            }
        }
        boundary_registry.preassembly_audit()?;

        let mut result_faces: Vec<FaceId> = Vec::new();

        for &fid in &original_faces {
            if !touched_faces.contains(&fid) && !vanished_walls.contains(&fid) {
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
            if vanished_walls.contains(&fid) {
                continue;
            }
            let replacement = face_replacements.get(&fid).copied();
            result_faces.push(replacement.unwrap_or(fid));
        }

        result_faces.extend(&blend_face_ids);
        result_faces.extend(&corner_face_ids);

        // A face whose *entire* boundary has collapsed to a single point
        // encloses no surface and cannot close a shell: it can only make the
        // assembly non-manifold. The exact-limit miter corner produces one,
        // because at `r = S` the shared face's retained remnant is consumed
        // down to the top-contact meeting vertex where the two stripes'
        // creases and cross-section arcs already meet
        // (`corner::close_collapsed_miter_corner` restricts the two collapsed
        // contacts onto that vertex), so its two edges become zero-length and
        // the face contributes no boundary anywhere. The test is extent, not
        // vertex count: every edge of the wire must be a point (its endpoints
        // coincide), and a wire with fewer than two edges is left alone,
        // because a disc bounded by one closed circle also has a single
        // vertex but a real boundary. A two-vertex face whose edges are
        // anti-parallel (a zero-area planar remnant) fails the extent test
        // and keeps closing the shell. A closed edge (a contact round a whole
        // rim) starts and ends at one vertex too, but runs round its curve:
        // its middle stands off that vertex.
        result_faces.retain(|&face_id| {
            let Ok(face) = topo.face(face_id) else {
                return true;
            };
            let Ok(wire) = topo.wire(face.outer_wire()) else {
                return true;
            };
            let edges = wire.edges();
            if edges.len() < 2 {
                return true;
            }
            edges.iter().any(|oriented| {
                let Ok(edge) = topo.edge(oriented.edge()) else {
                    return true;
                };
                let (Ok(start), Ok(end)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
                else {
                    return true;
                };
                let (start, end) = (start.point(), end.point());
                let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
                let middle =
                    edge.curve()
                        .evaluate_with_endpoints(f64::midpoint(t0, t1), start, end);
                (start - end).length() > 1e-7 || (middle - start).length() > 1e-7
            })
        });

        // Faces carried over from the input solid keep their (correct)
        // orientation: they seed the sense propagation and calibrate the
        // boundary-walk convention. Only faces built by THIS pass — walls,
        // corner patches, bands, fills, and rebuilt originals — are
        // eligible for repair; a previous fillet's NURBS wall arriving as
        // an input face must never be re-judged.
        let original_set: std::collections::HashSet<FaceId> =
            original_faces.iter().copied().collect();
        // A notched cap keeps its source's surface, reversal flag and traversal
        // with one corner path swapped for the blend's arc: it anchors the
        // convention like the untouched face it replaced. Without it a prism
        // whose every face was rebuilt has no seed, and the pairing walk
        // inverts the two mirrored corner bands it reaches through a
        // clockwise-wound neighbour. A closed rim's rebuilt cap and wall keep
        // their sources' surfaces and reversal flags, and take their new
        // circles' senses from the rim's against the axis, so they anchor it
        // too: a frustum with both rims filleted has no other face to.
        let seeds: Vec<FaceId> = result_faces
            .iter()
            .filter(|f| {
                original_set.contains(f)
                    || notched_caps.contains(f)
                    || closed_rim_supports.contains(f)
            })
            .copied()
            .collect();
        let seed_set: std::collections::HashSet<FaceId> = seeds.iter().copied().collect();
        let new_faces: Vec<FaceId> = result_faces
            .iter()
            .filter(|f| !seed_set.contains(f))
            .copied()
            .collect();
        // Registry-backed boundaries are the sole closure mechanism for the
        // authoritative fillet path. Positional welding and residual-loop
        // filling are intentionally not available here.
        for edge in drop_point_edges(topo, &result_faces)? {
            if let Some(handle) = boundary_registry.handle_for_edge(edge) {
                boundary_registry.retire(handle);
            }
        }
        brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
        brepkit_topology::orientation::normalize_face_normals(topo, &new_faces, &seeds)?;
        brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
        boundary_registry.refresh_owner_orientations(topo)?;
        // Without a seed the propagation anchors on an arbitrary face, so the
        // post-assembly shell may come out globally inverted (every face's
        // effective normal pointing inward) while still passing the
        // incidence and closure gates. Detect that and flip the whole shell,
        // then re-run the orientation passes so consistency is restored.
        // Seeds carry the input's orientation, which the propagation never
        // flips, so nothing is asked to overrule them.
        if seeds.is_empty() && shell_faces_inward(topo, &result_faces, &original_faces)? {
            log::debug!("orientation flip: the rebuilt shell faces inward");
            for &face_id in &result_faces {
                let face = topo.face_mut(face_id)?;
                face.set_reversed(!face.is_reversed());
            }
            brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
            brepkit_topology::orientation::normalize_face_normals(topo, &new_faces, &seeds)?;
            brepkit_topology::orientation::propagate_orientation(topo, &result_faces, &seeds)?;
            boundary_registry.refresh_owner_orientations(topo)?;
        }
        let mut sources: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();
        for &original in &original_faces {
            let mut current = original;
            for _ in 0..=face_replacements.len() {
                match face_replacements.get(&current) {
                    Some(&next) if next != current => current = next,
                    _ => break,
                }
            }
            if current != original {
                sources.insert(current, original);
            }
        }
        if align_seamed_faces_with_tangent_neighbours(topo, &new_faces, &result_faces, &sources)? {
            boundary_registry.refresh_owner_orientations(topo)?;
        }
        let postassembly = if force_postassembly_failure {
            Err("forced postassembly incidence gate failure".to_owned())
        } else {
            boundary_registry
                .postassembly_audit_with_input(
                    topo,
                    &result_faces,
                    &input_edge_uses(topo, &original_faces),
                )
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
        let input_edges = input_edge_set(topo, &original_faces);
        let result_faces = drop_contact_slivers(topo, result_faces, &input_edges)?;
        fold_spoke_spurs(topo, &result_faces, &input_edges)?;
        let result_faces = split_pinched_faces(topo, result_faces, &input_edges)?;
        let result_faces = drop_contact_slivers(topo, result_faces, &input_edges)?;
        let result_faces = extend_caps_over_runouts(topo, result_faces, &input_edges)?;
        let result_faces =
            wall_run_up::run_walls_up_over_runouts(topo, result_faces, &input_edges)?;
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
        let cavities = crate::builder_utils::copy_shells(topo, &inner_shells)?;
        let new_solid = Solid::new(new_shell_id, cavities);
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

/// Drop from `faces`' wires every edge that runs from a vertex back to itself
/// with no length. A round whose ball pivots on one face while it rolls round
/// the other (its radius that of a round it meets, so its centre stays on
/// that round's axis) has a contact that shrinks to a point: the band and the
/// face meet at a vertex alone. The edge the contact leaves has no sense, so
/// two faces cannot use it in opposite senses, and the orientation passes
/// read it as a fold. An edge that is all of some face's wire stays (a pin's
/// end rounded at the pin's own radius leaves its end face a point, which
/// shares that edge with the band). Returns the edges dropped.
fn drop_point_edges(topo: &mut Topology, faces: &[FaceId]) -> Result<Vec<EdgeId>, BlendError> {
    const SAMPLES: usize = 8;
    let tol = brepkit_math::tolerance::Tolerance::new().linear;
    let mut wires_of: Vec<(FaceId, WireId)> = Vec::new();
    for &face_id in faces {
        let face = topo.face(face_id)?;
        wires_of.extend(
            std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .map(|wire| (face_id, wire)),
        );
    }
    let mut points: HashSet<EdgeId> = HashSet::new();
    let mut kept_whole: HashSet<EdgeId> = HashSet::new();
    for &(_, wire_id) in &wires_of {
        let wire = topo.wire(wire_id)?;
        let mut in_wire = Vec::new();
        for &oriented in wire.edges() {
            let edge = topo.edge(oriented.edge())?;
            let point = (edge.start() == edge.end())
                .then(|| topo.vertex(edge.start()).map(Vertex::point))
                .transpose()?
                .is_some_and(|p| {
                    let (t0, t1) = edge.curve().domain_with_endpoints(p, p);
                    (0..=SAMPLES).all(|k| {
                        #[allow(clippy::cast_precision_loss)]
                        let t = t0 + (t1 - t0) * k as f64 / SAMPLES as f64;
                        (edge.curve().evaluate_with_endpoints(t, p, p) - p).length() <= tol
                    })
                });
            if point {
                in_wire.push(oriented.edge());
            }
        }
        if in_wire.len() == wire.edges().len() {
            kept_whole.extend(in_wire.iter().copied());
        }
        points.extend(in_wire);
    }
    let mut dropped: Vec<EdgeId> = points.difference(&kept_whole).copied().collect();
    dropped.sort_unstable_by_key(|edge| edge.index());
    if dropped.is_empty() {
        return Ok(dropped);
    }
    for (face_id, wire_id) in wires_of {
        let wire = topo.wire(wire_id)?;
        let kept: Vec<_> = wire
            .edges()
            .iter()
            .copied()
            .filter(|oriented| !dropped.contains(&oriented.edge()))
            .collect();
        if kept.len() == wire.edges().len() {
            continue;
        }
        let closed = wire.is_closed();
        let new_wire = topo.add_wire(Wire::new(kept, closed)?);
        let face = topo.face_mut(face_id)?;
        if face.outer_wire() == wire_id {
            face.set_outer_wire(new_wire);
        }
        for inner in face.inner_wires_mut() {
            if *inner == wire_id {
                *inner = new_wire;
            }
        }
    }
    Ok(dropped)
}

/// A planar face's outward normal and offset, its reversal applied.
/// Whether a shell rebuilt with no face of the input left came out inside out.
/// A rebuilt plane lying in a plane of the input keeps that plane's outward side,
/// which says so exactly. Only with no such plane is it left to the signed volume,
/// which reads curved faces over their whole parameter domain and about the
/// origin, so a part far from the origin can read the wrong way.
fn shell_faces_inward(
    topo: &Topology,
    faces: &[FaceId],
    input: &[FaceId],
) -> Result<bool, BlendError> {
    const SAME_NORMAL: f64 = 1e-6;
    const SAME_OFFSET: f64 = 1e-6;
    let mut planes = Vec::with_capacity(input.len());
    for &face in input {
        planes.extend(outward_plane(topo, face)?);
    }
    let (mut kept, mut turned) = (0_usize, 0_usize);
    for &face in faces {
        let Some((normal, d)) = outward_plane(topo, face)? else {
            continue;
        };
        for &(input_normal, input_d) in &planes {
            let along = normal.dot(input_normal);
            if (along - 1.0).abs() <= SAME_NORMAL && (d - input_d).abs() <= SAME_OFFSET {
                kept += 1;
                break;
            }
            if (along + 1.0).abs() <= SAME_NORMAL && (d + input_d).abs() <= SAME_OFFSET {
                turned += 1;
                break;
            }
        }
    }
    if kept + turned > 0 {
        return Ok(turned > kept);
    }
    Ok(crate::signed_volume::signed_shell_volume(topo, faces)? < 0.0)
}

fn outward_plane(topo: &Topology, face_id: FaceId) -> Result<Option<(Vec3, f64)>, BlendError> {
    let face = topo.face(face_id)?;
    let FaceSurface::Plane { d, .. } = face.surface() else {
        return Ok(None);
    };
    let sign = if face.is_reversed() { -1.0 } else { 1.0 };
    Ok(face.effective_plane_normal().map(|n| (n, d * sign)))
}

/// A face's outward normal at a point on its surface (within `tol`), its
/// reversal applied; `None` where the point is off the surface.
fn outward_normal_at(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
    tol: f64,
) -> Result<Option<Vec3>, BlendError> {
    let face = topo.face(face_id)?;
    if let Some((normal, d)) = outward_plane(topo, face_id)? {
        let off = normal.dot(Vec3::new(point.x(), point.y(), point.z())) - d;
        return Ok((off.abs() <= tol).then_some(normal));
    }
    let surface = face.surface();
    let Some((u, v)) = surface.project_point(point) else {
        return Ok(None);
    };
    let Some(on) = surface.evaluate(u, v) else {
        return Ok(None);
    };
    if (on - point).length() > tol {
        return Ok(None);
    }
    let normal = surface.normal(u, v);
    Ok(Some(if face.is_reversed() { -normal } else { normal }))
}

/// Fold out a spur where a face's wire runs along an edge and back along a
/// longer edge that overlaps it.
///
/// A terminal whose cap cannot be notched keeps the edge leaving the spine's
/// end (the spoke) whole, and the trim bridges the contact's end to that
/// vertex with a straight runout edge. When the contact ends on the spoke
/// itself (a concave fillet ending on a plate's face, with another face of
/// the fan beside it), the runout edge lies along the spoke: the support
/// face's wire then runs out to the spine's end along the runout edge and
/// back along the spoke. That is closed by edge count, but the face's mesh
/// skips the doubled-back stretch while the spoke's other face meshes the
/// spoke whole, which leaves the shell's mesh open along it.
///
/// The spoke is split where the runout edge ends, and the runout edge takes
/// the piece it overlaps: the spur's face keeps only the far piece, and each
/// other face along the spoke takes the runout edge and the far piece. Every
/// edge keeps two uses. This runs after the registry audit, which has
/// already checked each boundary's owners.
///
/// The same spur arises where a round's contact runs exactly along the
/// support face's own straight boundary edges (a round whose width lands on
/// a step's edge): the face's wire runs along those edges and back along the
/// contact. Folding one edge at a time, the contact takes the input's edges,
/// and the round meets each face beyond them along its own edge.
///
/// The spoke may be curved: a round ending on a floor that flat seams cut
/// in pieces at the spine's end keeps the arc where the floor meets a bore
/// whole, and the runout edge runs back along it to the contact. The arc is
/// split there the same way.
fn fold_spoke_spurs(
    topo: &mut Topology,
    faces: &[FaceId],
    input_edges: &HashSet<EdgeId>,
) -> Result<(), BlendError> {
    const ON_LINE: f64 = 1e-6;
    // The angle (radians) a curved face beyond the spoke may turn from the
    // patch and still be carried on by it.
    const CURVED_CARRIES_ON: f64 = 0.05;
    let point = |topo: &Topology, v: VertexId| topo.vertex(v).map(Vertex::point);
    for &face_id in faces {
        let face = topo.face(face_id)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wire_id in wire_ids {
            // Fold one spur at a time: each fold rewrites the wire.
            'folds: loop {
                let edges = topo.wire(wire_id)?.edges().to_vec();
                let n = edges.len();
                if n < 3 {
                    break;
                }
                for i in 0..n {
                    let (first, second) = (edges[i], edges[(i + 1) % n]);
                    if first.edge() == second.edge() {
                        continue;
                    }
                    let (e1, e2) = (topo.edge(first.edge())?, topo.edge(second.edge())?);
                    let (p, v) = (first.oriented_start(e1), first.oriented_end(e1));
                    let (v2, q) = (second.oriented_start(e2), second.oriented_end(e2));
                    if v != v2 || p == q {
                        continue;
                    }
                    let straight =
                        straight_edge(topo, e1, ON_LINE)? && straight_edge(topo, e2, ON_LINE)?;
                    let (pp, vp, qp) = (point(topo, p)?, point(topo, v)?, point(topo, q)?);
                    // Which of the two is the short one doubling back over
                    // the other: its far end lies inside the long one.
                    let inside = |far: Point3, along: Point3| {
                        let span = along - vp;
                        let length2 = span.dot(span);
                        if length2 <= 1e-18 {
                            return false;
                        }
                        let t = (far - vp).dot(span) / length2;
                        t > 1e-9 && t < 1.0 - 1e-9 && (vp + span * t - far).length() <= ON_LINE
                    };
                    let (short, long, short_first) = if straight {
                        if inside(pp, qp) {
                            (first, second, true)
                        } else if inside(qp, pp) {
                            (second, first, false)
                        } else {
                            continue;
                        }
                    } else if curved_spur(topo, first.edge(), second.edge(), v, ON_LINE)? {
                        (first, second, true)
                    } else if curved_spur(topo, second.edge(), first.edge(), v, ON_LINE)? {
                        (second, first, false)
                    } else {
                        continue;
                    };
                    // The long edge must be shared with exactly one other
                    // face, and only once here; the short edge too.
                    let mut others = Vec::new();
                    let mut short_faces = Vec::new();
                    let mut here = 0;
                    for &other in faces {
                        let f = topo.face(other)?;
                        for w in
                            std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied())
                        {
                            for oe in topo.wire(w)?.edges() {
                                if oe.edge() == long.edge() {
                                    if w == wire_id {
                                        here += 1;
                                    } else {
                                        others.push((w, other));
                                    }
                                }
                                if oe.edge() == short.edge() && w != wire_id {
                                    short_faces.push(other);
                                }
                            }
                        }
                    }
                    let ([(other_wire, beyond_long)], [beyond_short]) =
                        (others.as_slice(), short_faces.as_slice())
                    else {
                        continue;
                    };
                    if here != 1 {
                        continue;
                    }
                    // Only where the patch beyond the short edge carries on
                    // the face beyond the long one, in its plane (or, on a
                    // curved face, its tangent plane along the short edge)
                    // and facing its way (the patch adds material there). A
                    // patch standing in its own plane, or facing the other
                    // way, covers material the fillet removed: folding it in
                    // would only close a wrong shell.
                    //
                    // Or where the short edge is one of the input's own and
                    // the long one an edge the fillet made: a contact that
                    // runs exactly along the face's own boundary (a round
                    // whose width lands on a step's edge). The face beyond
                    // the short edge is the input's, and the round meets it
                    // along that edge.
                    let own_boundary =
                        input_edges.contains(&short.edge()) && !input_edges.contains(&long.edge());
                    if !own_boundary
                        && !patch_carries_on(
                            topo,
                            *beyond_short,
                            *beyond_long,
                            short.edge(),
                            ON_LINE,
                            CURVED_CARRIES_ON,
                        )?
                    {
                        continue;
                    }
                    let other_wire = *other_wire;
                    // The short edge's far end, where the long edge splits.
                    let split = if short_first { p } else { q };
                    let long_edge = topo.edge(long.edge())?.clone();
                    let short_edge = topo.edge(short.edge())?.clone();
                    let far = if long_edge.start() == v {
                        long_edge.end()
                    } else {
                        long_edge.start()
                    };
                    log::debug!(
                        "spoke spur on face {face_id:?}: {:?} over {:?}, split at {:?}",
                        short.edge(),
                        long.edge(),
                        point(topo, split)?
                    );
                    // The far piece, stored the way the long edge is: from
                    // the split to the far end, or (where the long edge
                    // starts at the far end) back from it.
                    let rest_from_split = long_edge.end() == far;
                    let rest = if straight {
                        topo.add_edge(Edge::new(split, far, EdgeCurve::Line))
                    } else {
                        let ends = (
                            point(topo, long_edge.start())?,
                            point(topo, long_edge.end())?,
                        );
                        let Some(piece) =
                            long_piece(&long_edge, ends, point(topo, split)?, rest_from_split)
                        else {
                            continue;
                        };
                        topo.add_edge(if rest_from_split {
                            Edge::new(split, far, piece)
                        } else {
                            Edge::new(far, split, piece)
                        })
                    };
                    let rest_from_split = straight || rest_from_split;
                    // In this wire the pair becomes the far piece alone, run
                    // the way the long edge was.
                    let rest_here = OrientedEdge::new(
                        rest,
                        (long.oriented_start(&long_edge) == v) == rest_from_split,
                    );
                    let mut folded = Vec::with_capacity(n - 1);
                    for (k, &oe) in edges.iter().enumerate() {
                        if k == i {
                            folded.push(rest_here);
                        } else if k != (i + 1) % n {
                            folded.push(oe);
                        }
                    }
                    *topo.wire_mut(wire_id)? = Wire::new(folded, true)?;
                    // The other face runs the long edge one way: from the
                    // spine's end, the short edge (against its use here)
                    // then the far piece; toward it, the reverse.
                    let other_edges = topo.wire(other_wire)?.edges().to_vec();
                    let closed = topo.wire(other_wire)?.is_closed();
                    let mut rewritten = Vec::with_capacity(other_edges.len() + 1);
                    for oe in other_edges {
                        if oe.edge() != long.edge() {
                            rewritten.push(oe);
                            continue;
                        }
                        let from_v = oe.oriented_start(&long_edge) == v;
                        let short_from_v = short.oriented_start(&short_edge) == v;
                        let short_away =
                            OrientedEdge::new(short.edge(), short.is_forward() == short_from_v);
                        if from_v {
                            rewritten.push(short_away);
                            rewritten.push(OrientedEdge::new(rest, rest_from_split));
                        } else {
                            rewritten.push(OrientedEdge::new(rest, !rest_from_split));
                            rewritten
                                .push(OrientedEdge::new(short.edge(), !short_away.is_forward()));
                        }
                    }
                    *topo.wire_mut(other_wire)? = Wire::new(rewritten, closed)?;
                    continue 'folds;
                }
                break;
            }
        }
    }
    Ok(())
}

/// Whether `short`, leaving `v`, runs back along `long`, which also leaves
/// `v`: each of its points lies within `tol` of `long`, and its far end
/// lies inside `long`, clear of both of its ends. For a pair where either
/// edge is curved (a cap's arc kept whole past a round's contact, and the
/// runout edge along it).
fn curved_spur(
    topo: &Topology,
    short: EdgeId,
    long: EdgeId,
    v: VertexId,
    tol: f64,
) -> Result<bool, BlendError> {
    const SAMPLES: u32 = 8;
    // The cosine of the most angle between the two edges' directions at
    // `v`: a runout edge fitted along an arc leaves it within a degree.
    const SAME_WAY: f64 = 0.999;
    let (s, l) = (topo.edge(short)?, topo.edge(long)?);
    if s.start() == s.end() || l.start() == l.end() {
        return Ok(false);
    }
    let far = if s.start() == v { s.end() } else { s.start() };
    let far_point = topo.vertex(far)?.point();
    for end in [l.start(), l.end()] {
        if (topo.vertex(end)?.point() - far_point).length() <= tol {
            return Ok(false);
        }
    }
    // Both leave `v` the same way, which most pairs of a face's edges do
    // not: read first, before the samples.
    let away = |edge: &Edge| -> Result<Option<Vec3>, BlendError> {
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        let tangent = if edge.start() == v {
            edge.curve().tangent_with_endpoints(t0, a, b)
        } else {
            -edge.curve().tangent_with_endpoints(t1, a, b)
        };
        Ok(tangent.normalize().ok())
    };
    let (Some(ts), Some(tl)) = (away(s)?, away(l)?) else {
        return Ok(false);
    };
    if ts.dot(tl) < SAME_WAY {
        return Ok(false);
    }
    if edge_nearest(topo, long, far_point, None)?.0 > tol {
        return Ok(false);
    }
    let (a, b) = (
        topo.vertex(s.start())?.point(),
        topo.vertex(s.end())?.point(),
    );
    let (t0, t1) = s.curve().domain_with_endpoints(a, b);
    for k in 1..SAMPLES {
        let t = (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0);
        let at = s.curve().evaluate_with_endpoints(t, a, b);
        if edge_nearest(topo, long, at, None)?.0 > tol {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The curve of `edge`'s piece between `split` (a point on it) and its end
/// vertex (`toward_end`) or its start vertex, run from the piece's first
/// vertex to its last: from `split` toward the end, or from the start to
/// `split`. `ends` are the points of `edge`'s start and end vertices. An
/// analytic curve is its own piece, its span given by the piece's ends. A
/// B-spline is cut to the piece's span of the edge's own span, which runs
/// against the curve where the edge is stored so. `None` where `split` does
/// not project on to it.
fn long_piece(
    edge: &Edge,
    (start, end): (Point3, Point3),
    split: Point3,
    toward_end: bool,
) -> Option<EdgeCurve> {
    let EdgeCurve::NurbsCurve(curve) = edge.curve() else {
        return Some(edge.curve().clone());
    };
    let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
    let found = brepkit_math::nurbs::projection::project_point_to_curve(curve, split, 1e-9).ok()?;
    let at = found.parameter.clamp(t0.min(t1), t0.max(t1));
    let (from, to) = if toward_end { (at, t1) } else { (t0, at) };
    trimmer::edge_curve_for_span(edge.curve(), from, to).ok()
}

/// Split a face pinched along a contact that runs over one of the face's
/// own straight edges in its middle.
///
/// A round whose width lands exactly on a step (a slab between two uprights
/// exactly a radius high, its bottom edge rounded) runs its contact along
/// the step's edge, which lies wholly inside the contact. The side face's
/// wire then runs along the step's edge and, further on, back along the
/// contact over it: two regions joined by a strip of no width. That is
/// closed by edge count, but the face's mesh skips the strip while the
/// face beyond the step's edge meshes it, so the shell's mesh is open.
///
/// The contact is split at the step edge's ends and the round takes the
/// step's edge in place of the middle piece, so it meets the face beyond
/// along it; the side face's wire, then running the step's edge both
/// ways, is cut there into its two regions, each a face of its own. This
/// is the spur `fold_spoke_spurs` folds, with the short edge inside the
/// long one rather than at its end.
fn split_pinched_faces(
    topo: &mut Topology,
    mut faces: Vec<FaceId>,
    input_edges: &HashSet<EdgeId>,
) -> Result<Vec<FaceId>, BlendError> {
    const ON_LINE: f64 = 1e-6;
    let point = |topo: &Topology, v: VertexId| topo.vertex(v).map(Vertex::point);
    let mut k = 0;
    'faces: while k < faces.len() {
        let face_id = faces[k];
        let face = topo.face(face_id)?;
        if !face.inner_wires().is_empty() {
            k += 1;
            continue;
        }
        let wire_id = face.outer_wire();
        let edges = topo.wire(wire_id)?.edges().to_vec();
        let n = edges.len();
        for (i, &long) in edges.iter().enumerate() {
            if input_edges.contains(&long.edge())
                || edges.iter().filter(|oe| oe.edge() == long.edge()).count() != 1
            {
                continue;
            }
            let long_edge = topo.edge(long.edge())?.clone();
            if !straight_edge(topo, &long_edge, ON_LINE)? {
                continue;
            }
            let (from, to) = (
                long.oriented_start(&long_edge),
                long.oriented_end(&long_edge),
            );
            let (fp, tp) = (point(topo, from)?, point(topo, to)?);
            let span = tp - fp;
            let length2 = span.dot(span);
            if length2 <= 1e-18 {
                continue;
            }
            // Where a point lies along the contact, strictly inside it.
            let along = |p: Point3| {
                let t = (p - fp).dot(span) / length2;
                (t > 1e-9 && t < 1.0 - 1e-9 && (fp + span * t - p).length() <= ON_LINE).then_some(t)
            };
            for (j, &short) in edges.iter().enumerate() {
                if j == i
                    || !input_edges.contains(&short.edge())
                    || edges.iter().filter(|oe| oe.edge() == short.edge()).count() != 1
                {
                    continue;
                }
                let short_edge = topo.edge(short.edge())?.clone();
                if !straight_edge(topo, &short_edge, ON_LINE)? {
                    continue;
                }
                // Run here against the contact: from its far end back.
                let (b, a) = (
                    short.oriented_start(&short_edge),
                    short.oriented_end(&short_edge),
                );
                let (Some(ta), Some(tb)) = (along(point(topo, a)?), along(point(topo, b)?)) else {
                    continue;
                };
                if ta >= tb {
                    continue;
                }
                // The contact is shared with one other face, the short edge
                // with one other face too.
                let mut long_others = Vec::new();
                let mut short_others = 0;
                for &other in &faces {
                    if other == face_id {
                        continue;
                    }
                    let f = topo.face(other)?;
                    for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied())
                    {
                        for oe in topo.wire(w)?.edges() {
                            if oe.edge() == long.edge() {
                                long_others.push(w);
                            }
                            if oe.edge() == short.edge() {
                                short_others += 1;
                            }
                        }
                    }
                }
                let [other_wire] = long_others.as_slice() else {
                    continue;
                };
                if short_others != 1 {
                    continue;
                }
                let other_wire = *other_wire;
                // This wire with the contact cut at a and b, the short edge
                // run a to b in place of the middle, then cut where it runs
                // the short edge both ways.
                let first = topo.add_edge(Edge::new(from, a, EdgeCurve::Line));
                let last = topo.add_edge(Edge::new(b, to, EdgeCurve::Line));
                let short_ab = OrientedEdge::new(short.edge(), !short.is_forward());
                let mut cycle = Vec::with_capacity(n + 2);
                for (m, &oe) in edges.iter().enumerate() {
                    if m == i {
                        cycle.push(OrientedEdge::new(first, true));
                        cycle.push(short_ab);
                        cycle.push(OrientedEdge::new(last, true));
                    } else {
                        cycle.push(oe);
                    }
                }
                let positions: Vec<usize> = cycle
                    .iter()
                    .enumerate()
                    .filter(|(_, oe)| oe.edge() == short.edge())
                    .map(|(m, _)| m)
                    .collect();
                let [p0, p1] = positions.as_slice() else {
                    continue;
                };
                let len = cycle.len();
                let take = |start: usize, end: usize| -> Vec<OrientedEdge> {
                    let mut out = Vec::new();
                    let mut m = (start + 1) % len;
                    while m != end {
                        out.push(cycle[m]);
                        m = (m + 1) % len;
                    }
                    out
                };
                let (one, two) = (take(*p0, *p1), take(*p1, *p0));
                if one.len() < 3 || two.len() < 3 {
                    continue;
                }
                log::debug!(
                    "pinched face {face_id:?}: {:?} inside {:?}, split in two",
                    short.edge(),
                    long.edge()
                );
                // The other face runs the contact one way or the other.
                let other_edges = topo.wire(other_wire)?.edges().to_vec();
                let closed = topo.wire(other_wire)?.is_closed();
                let mut rewritten = Vec::with_capacity(other_edges.len() + 2);
                for oe in other_edges {
                    if oe.edge() != long.edge() {
                        rewritten.push(oe);
                    } else if oe.oriented_start(&long_edge) == from {
                        rewritten.push(OrientedEdge::new(first, true));
                        rewritten.push(short_ab);
                        rewritten.push(OrientedEdge::new(last, true));
                    } else {
                        rewritten.push(OrientedEdge::new(last, false));
                        rewritten.push(OrientedEdge::new(short.edge(), short.is_forward()));
                        rewritten.push(OrientedEdge::new(first, false));
                    }
                }
                *topo.wire_mut(other_wire)? = Wire::new(rewritten, closed)?;
                *topo.wire_mut(wire_id)? = Wire::new(one, true)?;
                let second_wire = topo.add_wire(Wire::new(two, true)?);
                let face = topo.face(face_id)?;
                let surface = face.surface().clone();
                let twin = if face.is_reversed() {
                    Face::new_reversed(second_wire, Vec::new(), surface)
                } else {
                    Face::new(second_wire, Vec::new(), surface)
                };
                let twin_id = topo.add_face(twin);
                faces.push(twin_id);
                continue 'faces;
            }
        }
        k += 1;
    }
    Ok(faces)
}

/// How many uses each edge of the input's faces has there.
fn input_edge_uses(
    topo: &Topology,
    original_faces: &[FaceId],
) -> std::collections::HashMap<EdgeId, usize> {
    let mut uses = std::collections::HashMap::new();
    for &face_id in original_faces {
        let Ok(face) = topo.face(face_id) else {
            continue;
        };
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            if let Ok(wire) = topo.wire(wire_id) {
                for oriented in wire.edges() {
                    *uses.entry(oriented.edge()).or_insert(0) += 1;
                }
            }
        }
    }
    uses
}

/// The edges of the input's faces after the trim: their own edges and the
/// pieces the trim split them into, which lie along the input's boundary
/// just the same.
fn input_edge_set(topo: &Topology, original_faces: &[FaceId]) -> HashSet<EdgeId> {
    let mut edges = HashSet::new();
    for &face_id in original_faces {
        let Ok(face) = topo.face(face_id) else {
            continue;
        };
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            if let Ok(wire) = topo.wire(wire_id) {
                edges.extend(wire.edges().iter().map(OrientedEdge::edge));
            }
        }
    }
    edges
}

/// Whether an edge is straight: a line, or a B-spline lying on the chord
/// between its ends within `tol` (a contact fitted along a straight ruling).
/// A B-spline is certified by its control points: with every weight
/// positive the curve lies in their hull, so it lies within `tol` of the
/// chord wherever they all do, however narrow a knot span is.
fn straight_edge(topo: &Topology, edge: &Edge, tol: f64) -> Result<bool, BlendError> {
    match edge.curve() {
        EdgeCurve::Line => Ok(true),
        EdgeCurve::NurbsCurve(curve) => {
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let chord = b - a;
            let length2 = chord.dot(chord);
            if length2 <= 1e-18 || !curve.weights().iter().all(|w| w.is_finite() && *w > 0.0) {
                return Ok(false);
            }
            Ok(curve.control_points().iter().all(|&p| {
                let along = ((p - a).dot(chord) / length2).clamp(0.0, 1.0);
                (a + chord * along - p).length() <= tol
            }))
        }
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => Ok(false),
    }
}

/// Drop the faces a round has left with no area: a wall exactly a radius
/// high, its top rim rounded, keeps only the strip between the round's
/// contact and the wall's own bottom edge, which the contact lies on. Such a
/// face's wire is the contact and that edge alone, joining the same two
/// vertices. The face goes, and the round takes the wall's edge in place of
/// its contact, so it meets the face beyond the wall along it.
///
/// A wall exactly twice the radius high, rounded along both rims (a full
/// round), keeps only the strip between the two rounds' contacts, which lie
/// on each other along its middle. The face goes the same way, and the one
/// round takes the other's contact.
fn drop_contact_slivers(
    topo: &mut Topology,
    faces: Vec<FaceId>,
    input_edges: &HashSet<EdgeId>,
) -> Result<Vec<FaceId>, BlendError> {
    // How far the contact may stand off the wall's edge, for each unit of
    // the edge's span; `edge_lies_on` spends up to a quarter of it on the
    // stretch between its samples. A walked band passes through its
    // sections, so its contact stands off only between them: a round of 3
    // along a quarter arc of 4.5 stands 1e-8 off it. The fallback band,
    // which takes the sections as its control points, stands about 3.2e-4
    // off the same arc.
    const ON_EDGE: f64 = 1e-4;
    let mut faces = faces;
    'slivers: loop {
        for (index, &face_id) in faces.iter().enumerate() {
            let face = topo.face(face_id)?;
            if !face.inner_wires().is_empty() {
                if drop_coincident_loops(topo, &faces, face_id, ON_EDGE)? {
                    faces.remove(index);
                    continue 'slivers;
                }
                continue;
            }
            let wire_id = face.outer_wire();
            let [first, second] = topo.wire(wire_id)?.edges() else {
                if drop_folded_sliver(topo, &faces, face_id, input_edges)? {
                    faces.remove(index);
                    continue 'slivers;
                }
                continue;
            };
            let (first, second) = (*first, *second);
            // Two contacts lying on each other (a full round, a wall twice
            // the radius high rounded along both rims) are kept as the first.
            let (own, contact) = match (
                input_edges.contains(&first.edge()),
                input_edges.contains(&second.edge()),
            ) {
                (_, false) => (first, second),
                (false, true) => (second, first),
                (true, true) => continue,
            };
            let (own_edge, contact_edge) = (
                topo.edge(own.edge())?.clone(),
                topo.edge(contact.edge())?.clone(),
            );
            let ends = [contact_edge.start(), contact_edge.end()];
            // Two closed contacts round one vertex (a full round of a closed
            // rim's wall) span their length; open edges their chord.
            let loops = own_edge.start() == own_edge.end() && ends[0] == ends[1];
            if (own_edge.start() == own_edge.end() && !loops)
                || ([own_edge.start(), own_edge.end()] != ends
                    && [own_edge.end(), own_edge.start()] != ends)
            {
                continue;
            }
            let span = if loops {
                loop_length(topo, &own_edge)?
            } else {
                (topo.vertex(own_edge.start())?.point() - topo.vertex(own_edge.end())?.point())
                    .length()
            };
            if !edge_lies_on(topo, &contact_edge, &own_edge, ON_EDGE * span.max(1e-2))? {
                continue;
            }
            // Each edge has one other use, in a face other than this one.
            let mut contact_uses = Vec::new();
            let mut own_uses = 0;
            for &other in &faces {
                if other == face_id {
                    continue;
                }
                let f = topo.face(other)?;
                for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
                    for oe in topo.wire(w)?.edges() {
                        if oe.edge() == contact.edge() {
                            contact_uses.push(w);
                        }
                        own_uses += usize::from(oe.edge() == own.edge());
                    }
                }
            }
            let ([round_wire], 1) = (contact_uses.as_slice(), own_uses) else {
                continue;
            };
            let round_wire = *round_wire;
            log::debug!(
                "contact sliver {face_id:?}: {:?} lies on {:?}",
                contact.edge(),
                own.edge()
            );
            // The round runs the wall's edge the way it ran the contact.
            let closed = topo.wire(round_wire)?.is_closed();
            let rewritten: Vec<OrientedEdge> = topo
                .wire(round_wire)?
                .edges()
                .iter()
                .map(|&oe| {
                    if oe.edge() == contact.edge() {
                        if loops {
                            // The sliver runs along one loop and back along
                            // the other: their curves run the same way where
                            // its wire takes them in opposite senses.
                            let same = own.is_forward() != contact.is_forward();
                            return OrientedEdge::new(own.edge(), oe.is_forward() == same);
                        }
                        let from = oe.oriented_start(&contact_edge);
                        OrientedEdge::new(own.edge(), own_edge.start() == from)
                    } else {
                        oe
                    }
                })
                .collect();
            *topo.wire_mut(round_wire)? = Wire::new(rewritten, closed)?;
            faces.remove(index);
            continue 'slivers;
        }
        return Ok(faces);
    }
}

/// A closed round's last section is its first, walked or pivoted apart
/// only by the solve (by up to about 1e-7): made one, the band through
/// them closes on itself and reads as periodic, which its mesher and trim
/// rely on. Sections further apart than a millionth of the radius are left.
fn close_sections(sections: &mut [crate::section::CircSection]) {
    let (Some(first), Some(last)) = (sections.first(), sections.last()) else {
        return;
    };
    let tol = 1e-6 * first.radius.max(1.0);
    if sections.len() < 3
        || (first.center - last.center).length() > tol
        || (first.p1 - last.p1).length() > tol
        || (first.p2 - last.p2).length() > tol
    {
        return;
    }
    let mut closing = first.clone();
    closing.t = last.t;
    if let Some(last) = sections.last_mut() {
        *last = closing;
    }
}

/// A face bounded by two closed contacts lying on each other, one its
/// outline and the other its hole: two rounds of a plane face's two rims
/// meeting along it (the second pivoting on the first's contact) leave
/// the face a ring with no area. The face goes, and the hole's round takes
/// the outline's contact in place of its own, cut at its own contact's
/// vertex where the two loops start apart, the outline's round taking the
/// two pieces too. `true` where the face was dropped so.
fn drop_coincident_loops(
    topo: &mut Topology,
    faces: &[FaceId],
    face_id: FaceId,
    on_edge: f64,
) -> Result<bool, BlendError> {
    let face = topo.face(face_id)?;
    let ([hole], outer) = (face.inner_wires(), face.outer_wire()) else {
        return Ok(false);
    };
    let (Some(&[kept]), Some(&[dropped])) = (
        topo.wire(outer).ok().map(Wire::edges),
        topo.wire(*hole).ok().map(Wire::edges),
    ) else {
        return Ok(false);
    };
    let (kept_edge, dropped_edge) = (
        topo.edge(kept.edge())?.clone(),
        topo.edge(dropped.edge())?.clone(),
    );
    if kept_edge.start() != kept_edge.end() || dropped_edge.start() != dropped_edge.end() {
        return Ok(false);
    }
    let span = loop_length(topo, &kept_edge)?;
    if !edge_lies_on(topo, &dropped_edge, &kept_edge, on_edge * span.max(1e-2))? {
        return Ok(false);
    }
    // Each loop's one use outside this face: its wire, whether that runs
    // the edge forward, and whether its face is reversed.
    let other_use = |edge: EdgeId| -> Result<Option<(WireId, bool, bool)>, BlendError> {
        let mut uses = Vec::new();
        for &other in faces {
            if other == face_id {
                continue;
            }
            let f = topo.face(other)?;
            for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
                for oe in topo.wire(w)?.edges() {
                    if oe.edge() == edge {
                        uses.push((w, oe.is_forward(), f.is_reversed()));
                    }
                }
            }
        }
        Ok(if let [only] = uses[..] {
            Some(only)
        } else {
            None
        })
    };
    let (Some((kept_wire, kept_forward, kept_reversed)), Some((dropped_wire, _, dropped_reversed))) =
        (other_use(kept.edge())?, other_use(dropped.edge())?)
    else {
        return Ok(false);
    };
    if kept_wire == dropped_wire {
        return Ok(false);
    }
    // The dropped contact's round runs the outline the way it ran its own
    // contact, read where the two lie on each other; the outline's round
    // must run it the other way, its face's sense taken into account.
    let Some(along) = runs_along(
        topo,
        &dropped_edge,
        dropped_wire,
        dropped.edge(),
        &kept_edge,
    )?
    else {
        return Ok(false);
    };
    if (kept_forward != kept_reversed) == (along != dropped_reversed) {
        return Ok(false);
    }
    // Loops starting at one point, as separate vertices (a walked contact
    // closed a hair from where the other starts), start together: the
    // hole's round takes the outline whole, and its edges ending at the
    // hole's vertex (its seam) end at the outline's.
    let (kept_start, dropped_start) = (kept_edge.start(), dropped_edge.start());
    let start_gap =
        (topo.vertex(kept_start)?.point() - topo.vertex(dropped_start)?.point()).length();
    let pieces: Vec<OrientedEdge> = if kept_start == dropped_start
        || start_gap <= on_edge * span.max(1e-2)
    {
        if kept_start != dropped_start {
            let mut moved = HashSet::new();
            for &other in faces {
                let f = topo.face(other)?;
                let wires: Vec<WireId> = std::iter::once(f.outer_wire())
                    .chain(f.inner_wires().iter().copied())
                    .collect();
                for w in wires {
                    let edges: Vec<EdgeId> = topo
                        .wire(w)?
                        .edges()
                        .iter()
                        .map(OrientedEdge::edge)
                        .collect();
                    for e in edges {
                        if e == dropped.edge() || !moved.insert(e) {
                            continue;
                        }
                        let edge = topo.edge_mut(e)?;
                        if edge.start() == dropped_start {
                            edge.set_start(kept_start);
                        }
                        if edge.end() == dropped_start {
                            edge.set_end(kept_start);
                        }
                    }
                }
            }
        }
        vec![OrientedEdge::new(kept.edge(), along)]
    } else {
        // Cut the outline at the hole's vertex, which must lie on it.
        let EdgeCurve::NurbsCurve(curve) = kept_edge.curve() else {
            return Ok(false);
        };
        let start = topo.vertex(kept_edge.start())?.point();
        let (t0, t1) = curve.domain();
        if (curve.evaluate(t0) - start).length() > on_edge * span.max(1e-2)
            || (curve.evaluate(t1) - start).length() > on_edge * span.max(1e-2)
        {
            return Ok(false);
        }
        let cut_vertex = dropped_edge.start();
        let at = topo.vertex(cut_vertex)?.point();
        let Ok(found) = brepkit_math::nurbs::projection::project_point_to_curve(curve, at, 1e-9)
        else {
            return Ok(false);
        };
        let margin = 1e-6 * (t1 - t0);
        if found.distance > on_edge * span.max(1e-2)
            || found.parameter <= t0 + margin
            || found.parameter >= t1 - margin
        {
            return Ok(false);
        }
        let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(curve, found.parameter)?;
        let first = topo.add_edge(Edge::new(
            kept_edge.start(),
            cut_vertex,
            EdgeCurve::NurbsCurve(left),
        ));
        let second = topo.add_edge(Edge::new(
            cut_vertex,
            kept_edge.start(),
            EdgeCurve::NurbsCurve(right),
        ));
        // The outline's round runs it as it did, from its own vertex.
        let back = if kept_forward {
            vec![
                OrientedEdge::new(first, true),
                OrientedEdge::new(second, true),
            ]
        } else {
            vec![
                OrientedEdge::new(second, false),
                OrientedEdge::new(first, false),
            ]
        };
        rewrite_wire(topo, kept_wire, |oe| {
            if oe.edge() == kept.edge() {
                back.clone()
            } else {
                vec![oe]
            }
        })?;
        // The hole's round runs `along`, from the cut.
        if along {
            vec![
                OrientedEdge::new(second, true),
                OrientedEdge::new(first, true),
            ]
        } else {
            vec![
                OrientedEdge::new(first, false),
                OrientedEdge::new(second, false),
            ]
        }
    };
    log::debug!(
        "coincident loops {face_id:?}: {:?} lies on {:?}",
        dropped.edge(),
        kept.edge()
    );
    rewrite_wire(topo, dropped_wire, |oe| {
        if oe.edge() == dropped.edge() {
            pieces.clone()
        } else {
            vec![oe]
        }
    })?;
    Ok(true)
}

/// Whether `wire`'s use of `edge` (`edge_data`) runs the way `on`'s curve
/// does, read at the edge's middle; `None` where the two are square there.
fn runs_along(
    topo: &Topology,
    edge_data: &Edge,
    wire: WireId,
    edge: EdgeId,
    on: &Edge,
) -> Result<Option<bool>, BlendError> {
    let Some(forward) = topo
        .wire(wire)?
        .edges()
        .iter()
        .find(|oe| oe.edge() == edge)
        .map(OrientedEdge::is_forward)
    else {
        return Ok(None);
    };
    let (a, b) = (
        topo.vertex(edge_data.start())?.point(),
        topo.vertex(edge_data.end())?.point(),
    );
    let (t0, t1) = edge_data.curve().domain_with_endpoints(a, b);
    let at = |t: f64| edge_data.curve().evaluate_with_endpoints(t, a, b);
    let (middle, h) = (f64::midpoint(t0, t1), 1e-4 * (t1 - t0));
    let tangent = at(middle + h) - at(middle - h);
    let point = at(middle);
    let (p, q) = (
        topo.vertex(on.start())?.point(),
        topo.vertex(on.end())?.point(),
    );
    let EdgeCurve::NurbsCurve(curve) = on.curve() else {
        return Ok(None);
    };
    let Ok(found) = brepkit_math::nurbs::projection::project_point_to_curve(curve, point, 1e-9)
    else {
        return Ok(None);
    };
    let (s0, s1) = on.curve().domain_with_endpoints(p, q);
    let k = 1e-4 * (s1 - s0);
    let along = curve.evaluate((found.parameter + k).min(s1))
        - curve.evaluate((found.parameter - k).max(s0));
    let dot = tangent.dot(along);
    if dot.abs() <= 0.5 * tangent.length() * along.length() {
        return Ok(None);
    }
    Ok(Some((dot > 0.0) == forward))
}

/// A face whose wire only runs out and back along itself, enclosing
/// nothing: where two crossing rounds meet along a contact on a wall that
/// also ends on split edges, the wall is left running up an end edge, back
/// down its pieces, and out and back along the two contacts. Two edges
/// running between the same vertices both ways, lying on each other, are
/// paired, the one kept (an input edge where there is one) taking the
/// other's place in its other face; every edge left must have a chain of
/// pieces lying on it running back from its end to its start, which take
/// its place in its other face; there must be at least one pair. Every edge
/// involved has exactly one use outside this face. `true` where the
/// face was dropped so.
#[allow(clippy::too_many_lines)]
fn drop_folded_sliver(
    topo: &mut Topology,
    faces: &[FaceId],
    face_id: FaceId,
    input_edges: &HashSet<EdgeId>,
) -> Result<bool, BlendError> {
    // As in `drop_contact_slivers`.
    const ON_EDGE: f64 = 1e-4;
    let face = topo.face(face_id)?;
    if !face.inner_wires().is_empty() {
        return Ok(false);
    }
    let mut ring: Vec<OrientedEdge> = topo.wire(face.outer_wire())?.edges().to_vec();
    if ring.len() < 3 {
        return Ok(false);
    }
    // The one wire outside this face using each edge.
    let other_use = |topo: &Topology, edge: EdgeId| -> Result<Option<WireId>, BlendError> {
        let mut uses = Vec::new();
        for &other in faces {
            if other == face_id {
                continue;
            }
            let f = topo.face(other)?;
            for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
                for oe in topo.wire(w)?.edges() {
                    if oe.edge() == edge {
                        uses.push(w);
                    }
                }
            }
        }
        Ok(if let [only] = uses.as_slice() {
            Some(*only)
        } else {
            None
        })
    };
    let ends = |topo: &Topology, oe: OrientedEdge| -> Result<(VertexId, VertexId), BlendError> {
        let edge = topo.edge(oe.edge())?;
        Ok(if oe.is_forward() {
            (edge.start(), edge.end())
        } else {
            (edge.end(), edge.start())
        })
    };
    let span = |topo: &Topology, edge: &Edge| -> Result<f64, BlendError> {
        Ok((topo.vertex(edge.start())?.point() - topo.vertex(edge.end())?.point()).length())
    };
    // (kept, dropped): `dropped` runs back along `kept`, between the same
    // two vertices, anywhere in the wire.
    let mut pairs: Vec<(OrientedEdge, OrientedEdge)> = Vec::new();
    'pair: loop {
        for i in 0..ring.len() {
            for j in i + 1..ring.len() {
                let (a, b) = (ring[i], ring[j]);
                if a.edge() == b.edge() {
                    continue;
                }
                let ((sa, ea), (sb, eb)) = (ends(topo, a)?, ends(topo, b)?);
                if sb != ea || eb != sa || sa == ea {
                    continue;
                }
                let (kept, dropped) = match (
                    input_edges.contains(&a.edge()),
                    input_edges.contains(&b.edge()),
                ) {
                    (_, false) => (a, b),
                    (false, true) => (b, a),
                    (true, true) => continue,
                };
                let (kept_edge, dropped_edge) = (
                    topo.edge(kept.edge())?.clone(),
                    topo.edge(dropped.edge())?.clone(),
                );
                let tol = ON_EDGE * span(topo, &kept_edge)?.max(1e-2);
                if !edge_lies_on(topo, &dropped_edge, &kept_edge, tol)? {
                    continue;
                }
                pairs.push((kept, dropped));
                ring.remove(j);
                ring.remove(i);
                continue 'pair;
            }
        }
        break;
    }
    // A strip between two contacts lying on each other: without such a
    // pair, a face of an edge and pieces near it may be a thin face of the
    // part's own.
    if pairs.is_empty() {
        return Ok(false);
    }
    // Each edge left and the chain of pieces lying on it, running from its
    // end back to its start.
    let mut chains: Vec<(OrientedEdge, Vec<OrientedEdge>)> = Vec::new();
    'chain: while !ring.is_empty() {
        for w in 0..ring.len() {
            let whole = ring[w];
            let (start, end) = ends(topo, whole)?;
            if start == end {
                continue;
            }
            let whole_edge = topo.edge(whole.edge())?.clone();
            let tol = ON_EDGE * span(topo, &whole_edge)?.max(1e-2);
            let mut used = vec![w];
            let mut at = end;
            while at != start {
                let mut next = None;
                for (k, &piece) in ring.iter().enumerate() {
                    if used.contains(&k) || piece.edge() == whole.edge() {
                        continue;
                    }
                    let (from, _) = ends(topo, piece)?;
                    if from != at {
                        continue;
                    }
                    let piece_edge = topo.edge(piece.edge())?.clone();
                    if stays_within(topo, &piece_edge, &whole_edge, tol)? {
                        next = Some(k);
                        break;
                    }
                }
                let Some(k) = next else {
                    break;
                };
                used.push(k);
                at = ends(topo, ring[k])?.1;
            }
            if at != start || used.len() < 2 {
                continue;
            }
            let pieces: Vec<OrientedEdge> = used[1..].iter().map(|&k| ring[k]).collect();
            chains.push((whole, pieces));
            used.sort_unstable();
            for &k in used.iter().rev() {
                ring.remove(k);
            }
            continue 'chain;
        }
        return Ok(false);
    }
    // Each edge used once outside this face.
    let mut rewrites: Vec<(WireId, EdgeId, Vec<OrientedEdge>)> = Vec::new();
    for &(kept, dropped) in &pairs {
        let (Some(_), Some(wire)) = (
            other_use(topo, kept.edge())?,
            other_use(topo, dropped.edge())?,
        ) else {
            return Ok(false);
        };
        // The other face ran the dropped edge one way; it runs the kept
        // edge the same way between the same vertices.
        let kept_edge = topo.edge(kept.edge())?.clone();
        let dropped_edge = topo.edge(dropped.edge())?.clone();
        let Some(oe) = topo
            .wire(wire)?
            .edges()
            .iter()
            .find(|oe| oe.edge() == dropped.edge())
            .copied()
        else {
            return Ok(false);
        };
        let from = oe.oriented_start(&dropped_edge);
        rewrites.push((
            wire,
            dropped.edge(),
            vec![OrientedEdge::new(kept.edge(), kept_edge.start() == from)],
        ));
    }
    for (whole, pieces) in &chains {
        let Some(wire) = other_use(topo, whole.edge())? else {
            return Ok(false);
        };
        for piece in pieces {
            if other_use(topo, piece.edge())?.is_none() {
                return Ok(false);
            }
        }
        let whole_edge = topo.edge(whole.edge())?.clone();
        let Some(oe) = topo
            .wire(wire)?
            .edges()
            .iter()
            .find(|oe| oe.edge() == whole.edge())
            .copied()
        else {
            return Ok(false);
        };
        // The pieces run from the whole edge's end back to its start in
        // this face; the other face takes them the way it ran the edge.
        let (_, whole_end) = ends(topo, *whole)?;
        let replacement: Vec<OrientedEdge> = if oe.oriented_start(&whole_edge) == whole_end {
            pieces.clone()
        } else {
            pieces
                .iter()
                .rev()
                .map(|piece| OrientedEdge::new(piece.edge(), !piece.is_forward()))
                .collect()
        };
        rewrites.push((wire, whole.edge(), replacement));
    }
    if rewrites.is_empty() {
        return Ok(false);
    }
    log::debug!(
        "folded sliver {face_id:?}: {} pairs run back on each other, {} edges replaced by their pieces",
        pairs.len(),
        chains.len()
    );
    for (wire, edge, replacement) in rewrites {
        let closed = topo.wire(wire)?.is_closed();
        let rewritten: Vec<OrientedEdge> = topo
            .wire(wire)?
            .edges()
            .iter()
            .flat_map(|&oe| {
                if oe.edge() == edge {
                    replacement.clone()
                } else {
                    vec![oe]
                }
            })
            .collect();
        *topo.wire_mut(wire)? = Wire::new(rewritten, closed)?;
    }
    Ok(true)
}

/// A closed edge's length, along its curve.
fn loop_length(topo: &Topology, edge: &Edge) -> Result<f64, BlendError> {
    const SAMPLES: u32 = 256;
    let at = topo.vertex(edge.start())?.point();
    let curve = edge.curve();
    let (t0, t1) = curve.domain_with_endpoints(at, at);
    let point = |k: u32| {
        curve.evaluate_with_endpoints(
            (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0),
            at,
            at,
        )
    };
    Ok((0..SAMPLES)
        .map(|k| (point(k + 1) - point(k)).length())
        .sum())
}

/// Ends walked rounds on the curved caps they ran into at a kink
/// ([`extend_cap_over_runout`]), each runout patch in turn. Where nothing
/// else is left at the corner, the cap's corner edge and its run on are
/// then one edge.
fn extend_caps_over_runouts(
    topo: &mut Topology,
    faces: Vec<FaceId>,
    input_edges: &HashSet<EdgeId>,
) -> Result<Vec<FaceId>, BlendError> {
    let mut faces = faces;
    let mut runs = Vec::new();
    'runouts: loop {
        for (index, &face_id) in faces.iter().enumerate() {
            if let Some(run) = extend_cap_over_runout(topo, &faces, face_id, input_edges)? {
                runs.push(run);
                faces.remove(index);
                continue 'runouts;
            }
        }
        break;
    }
    for run in runs {
        join_run_on(topo, &faces, run)?;
    }
    Ok(faces)
}

/// A cap's corner edge run on past its corner: the corner, the edge, the
/// run on, and the two wires using both.
struct RunOn {
    corner: VertexId,
    edge: EdgeId,
    run: EdgeId,
    wires: [WireId; 2],
}

/// Rewrites a wire edge by edge, each edge taking the edges `swap` gives.
fn rewrite_wire(
    topo: &mut Topology,
    wire: WireId,
    swap: impl Fn(OrientedEdge) -> Vec<OrientedEdge>,
) -> Result<(), BlendError> {
    let closed = topo.wire(wire)?.is_closed();
    let edges: Vec<OrientedEdge> = topo
        .wire(wire)?
        .edges()
        .iter()
        .flat_map(|&oe| swap(oe))
        .collect();
    *topo.wire_mut(wire)? = Wire::new(edges, closed)?;
    Ok(())
}

/// Joins a cap's corner edge and its run on into one edge where they are
/// the only edges left at the corner and are the same kind of curve, run
/// the same way.
fn join_run_on(topo: &mut Topology, faces: &[FaceId], run_on: RunOn) -> Result<(), BlendError> {
    let RunOn {
        corner,
        edge,
        run,
        wires,
    } = run_on;
    for &face_id in faces {
        let face = topo.face(face_id)?;
        for w in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(w)?.edges() {
                if oe.edge() == edge || oe.edge() == run {
                    if !wires.contains(&w) {
                        return Ok(());
                    }
                    continue;
                }
                let e = topo.edge(oe.edge())?;
                if e.start() == corner || e.end() == corner {
                    return Ok(());
                }
            }
        }
    }
    let (edge_data, run_data) = (topo.edge(edge)?.clone(), topo.edge(run)?.clone());
    let joined = match (edge_data.curve(), run_data.curve()) {
        (EdgeCurve::Line, EdgeCurve::Line) => EdgeCurve::Line,
        (EdgeCurve::Circle(a), EdgeCurve::Circle(_)) => EdgeCurve::Circle(a.clone()),
        _ => return Ok(()),
    };
    let (start, end) = if run_data.end() == corner && edge_data.start() == corner {
        (run_data.start(), edge_data.end())
    } else if edge_data.end() == corner && run_data.start() == corner {
        (edge_data.start(), run_data.end())
    } else {
        return Ok(());
    };
    let joined = topo.add_edge(Edge::new(start, end, joined));
    for wire in wires {
        // The joined edge runs the way the corner edge did.
        rewrite_wire(topo, wire, |oe| {
            if oe.edge() == run {
                vec![]
            } else if oe.edge() == edge {
                vec![OrientedEdge::new(joined, oe.is_forward())]
            } else {
                vec![oe]
            }
        })?;
    }
    Ok(())
}

/// A walked round whose edge ends where its wall meets a cap at a kink (a
/// plate's spline side running into its rounded end at an angle) ends in
/// the plane square to its spine, closed by a three-edge runout patch: the
/// band's end section `B` from its contact end `P` to `A`, the piece `V` of
/// the cap's corner edge from `A` to the corner `C`, and a line `L` on the
/// contact's face from `P` to `C`. Where `P` lies inside the cap's material
/// (the cap's surface turned in past the kink), the round has cut into the
/// cap: the band and its contact are cut back where the contact crosses the
/// cap's surface, at `X`, which lies on the carrier of the cap's corner
/// edge `E` on the contact's face (a circle or a line); the band ends on its
/// intersection with the cap from `X` to `A`, the cap takes that curve and
/// `E` run on from `C` to `X` in place of `V`, the contact's face takes
/// that run in place of `L`, and the patch goes. The run on where the face
/// was such a patch and is dropped.
#[allow(clippy::too_many_lines)]
fn extend_cap_over_runout(
    topo: &mut Topology,
    faces: &[FaceId],
    face_id: FaceId,
    input_edges: &HashSet<EdgeId>,
) -> Result<Option<RunOn>, BlendError> {
    let face = topo.face(face_id)?;
    if !face.inner_wires().is_empty() || !matches!(face.surface(), FaceSurface::Plane { .. }) {
        return Ok(None);
    }
    let ring = topo.wire(face.outer_wire())?.edges().to_vec();
    if ring.len() != 3 || ring.iter().any(|oe| input_edges.contains(&oe.edge())) {
        return Ok(None);
    }
    // The one other face using each edge, with its wire.
    let other_use = |edge: EdgeId| -> Result<Option<(FaceId, WireId)>, BlendError> {
        let mut uses = Vec::new();
        for &other in faces {
            if other == face_id {
                continue;
            }
            let f = topo.face(other)?;
            for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
                for oe in topo.wire(w)?.edges() {
                    if oe.edge() == edge {
                        uses.push((other, w));
                    }
                }
            }
        }
        Ok(if let [one] = uses.as_slice() {
            Some(*one)
        } else {
            None
        })
    };
    let mut users = Vec::with_capacity(3);
    for oe in &ring {
        let Some(user) = other_use(oe.edge())? else {
            return Ok(None);
        };
        users.push(user);
    }
    let ends = |e: &Edge| [e.start(), e.end()];
    let other_end = |e: &Edge, v: VertexId| if e.start() == v { e.end() } else { e.start() };
    // Each edge with a B-spline face beyond may be the band's end.
    for b in 0..3 {
        if !topo
            .face(users[b].0)
            .is_ok_and(|f| matches!(f.surface(), FaceSurface::Nurbs(_)))
        {
            continue;
        }
        let (band_face, band_wire) = users[b];
        let b_id = ring[b].edge();
        let b_edge = topo.edge(b_id)?.clone();
        let band = topo.face(band_face)?.surface().clone();
        let (k1, k2) = ((b + 1) % 3, (b + 2) % 3);
        for (kl, kv) in [(k1, k2), (k2, k1)] {
            let (l_id, v_id) = (ring[kl].edge(), ring[kv].edge());
            let (l_edge, v_edge) = (topo.edge(l_id)?.clone(), topo.edge(v_id)?.clone());
            let Some(c) = ends(&l_edge)
                .into_iter()
                .find(|v| ends(&v_edge).contains(v))
            else {
                continue;
            };
            let (p, a) = (other_end(&l_edge, c), other_end(&v_edge, c));
            if p == a || ends(&b_edge).contains(&c) || !ends(&b_edge).contains(&p) {
                continue;
            }
            if let Some(run) = runout_onto_cap(
                topo,
                faces,
                face_id,
                RunoutPatch {
                    band: (band_wire, &band, b_id, &b_edge),
                    line: (users[kl].1, l_id, &l_edge),
                    piece: (users[kv], v_id, &v_edge),
                    corner: c,
                    contact_end: p,
                    cap_end: a,
                },
            )? {
                return Ok(Some(run));
            }
        }
    }
    Ok(None)
}

/// A runout patch's three edges: the band's end section (the band's wire,
/// surface, the edge), the line on the contact's face (that face's wire, the
/// edge), and the piece of the cap's corner edge (the face and wire beyond
/// it, the edge), with the corner `C`, the contact's end `P` and the piece's
/// end `A`.
struct RunoutPatch<'a> {
    band: (WireId, &'a FaceSurface, EdgeId, &'a Edge),
    line: (WireId, EdgeId, &'a Edge),
    piece: ((FaceId, WireId), EdgeId, &'a Edge),
    corner: VertexId,
    contact_end: VertexId,
    cap_end: VertexId,
}

/// A runout patch's cap (its face and wire), with the whole corner edge and
/// the wall's wire where a wall runs up that edge and back down the piece.
type RunoutCap = ((FaceId, WireId), Option<(EdgeId, WireId)>);

/// The cap a runout patch's corner piece lies on: the face beyond the piece
/// where its wire takes the cap's corner edge on the contact's face too, or,
/// where the face beyond is a wall running up the whole corner edge and
/// back down the piece (a spur), the face beyond that edge. The cap's face
/// and wire, with the spur's whole edge and wall wire.
fn runout_cap(
    topo: &Topology,
    faces: &[FaceId],
    beyond: (FaceId, WireId),
    piece: (EdgeId, &Edge),
    corner: VertexId,
    f_wire: WireId,
) -> Result<Option<RunoutCap>, BlendError> {
    const ON_LINE: f64 = 1e-6;
    let at_corner = |e: &Edge| e.start() == corner || e.end() == corner;
    let f_edges = topo.wire(f_wire)?.edges().to_vec();
    let beyond_edges = topo.wire(beyond.1)?.edges().to_vec();
    let shares_corner = beyond_edges.iter().any(|oe| {
        oe.edge() != piece.0
            && f_edges.iter().any(|f| f.edge() == oe.edge())
            && topo.edge(oe.edge()).is_ok_and(at_corner)
    });
    if shares_corner {
        return Ok(Some((beyond, None)));
    }
    if !straight_edge(topo, piece.1, ON_LINE)? {
        return Ok(None);
    }
    let a = if piece.1.start() == corner {
        piece.1.end()
    } else {
        piece.1.start()
    };
    let ap = topo.vertex(a)?.point();
    let cp = topo.vertex(corner)?.point();
    for oe in &beyond_edges {
        if oe.edge() == piece.0 {
            continue;
        }
        let whole = topo.edge(oe.edge())?;
        if !at_corner(whole) || !straight_edge(topo, whole, ON_LINE)? {
            continue;
        }
        let far = if whole.start() == corner {
            whole.end()
        } else {
            whole.start()
        };
        let span = topo.vertex(far)?.point() - cp;
        let t = (ap - cp).dot(span) / span.dot(span).max(1e-300);
        if !(1e-9..1.0 - 1e-9).contains(&t) || (cp + span * t - ap).length() > ON_LINE {
            continue;
        }
        let mut uses = Vec::new();
        for &other in faces {
            let f = topo.face(other)?;
            for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
                if w == beyond.1 {
                    continue;
                }
                if topo.wire(w)?.edges().iter().any(|e| e.edge() == oe.edge()) {
                    uses.push((other, w));
                }
            }
        }
        if let [cap] = uses.as_slice() {
            return Ok(Some((*cap, Some((oe.edge(), beyond.1)))));
        }
    }
    Ok(None)
}

/// Extends the cap over a runout patch where its contact runs on into the
/// cap ([`extend_cap_over_runout`]).
#[allow(clippy::too_many_lines)]
fn runout_onto_cap(
    topo: &mut Topology,
    faces: &[FaceId],
    face_id: FaceId,
    patch: RunoutPatch<'_>,
) -> Result<Option<RunOn>, BlendError> {
    const SAMPLES: usize = 200;
    let ends = |e: &Edge| [e.start(), e.end()];
    let other_end = |e: &Edge, v: VertexId| if e.start() == v { e.end() } else { e.start() };
    let RunoutPatch {
        band: (band_wire, band, b_id, b_edge),
        line: (f_wire, l_id, l_edge),
        piece: (beyond, v_id, v_edge),
        corner: c,
        contact_end: p,
        cap_end: a,
    } = patch;
    let b_edge = b_edge.clone();
    let l_edge = l_edge.clone();
    let v_edge = v_edge.clone();
    let Some(((k_face, k_wire), spur)) =
        runout_cap(topo, faces, beyond, (v_id, &v_edge), c, f_wire)?
    else {
        return Ok(None);
    };
    let band_edges = topo.wire(band_wire)?.edges().to_vec();
    let f_edges = topo.wire(f_wire)?.edges().to_vec();
    let k_edges = topo.wire(k_wire)?.edges().to_vec();
    // The band's contact on the contact's face, ending at `P`.
    let contact = band_edges.iter().map(OrientedEdge::edge).find(|&e| {
        e != b_id
            && f_edges.iter().any(|oe| oe.edge() == e)
            && topo.edge(e).is_ok_and(|edge| ends(edge).contains(&p))
    });
    // The cap's corner edge on the contact's face, from `C`.
    let corner = f_edges.iter().map(OrientedEdge::edge).find(|&e| {
        e != l_id
            && k_edges.iter().any(|oe| oe.edge() == e)
            && topo.edge(e).is_ok_and(|edge| ends(edge).contains(&c))
    });
    let (Some(contact), Some(corner)) = (contact, corner) else {
        return Ok(None);
    };
    let contact_edge = topo.edge(contact)?.clone();
    let corner_edge = topo.edge(corner)?.clone();
    let q = other_end(&contact_edge, p);
    let [pp, qp, cp, ap] = [p, q, c, a].map(|v| topo.vertex(v).map(Vertex::point));
    let (pp, qp, cp, ap) = (pp?, qp?, cp?, ap?);
    let tol = 1e-6 * (pp - ap).length().max(1.0);
    let offset = |point: Point3| cap_offset(topo, RunOnCap::Face(k_face), point, tol);
    let (Some(at_p), Some(at_q)) = (offset(pp), offset(qp)) else {
        return Ok(None);
    };
    log::debug!("runout {face_id:?}: P offset {at_p}, far end {at_q}");
    if at_p >= -tol || at_q <= tol {
        return Ok(None);
    }
    // Where the contact crosses the cap's surface, back from `P`.
    let (sp, ep) = (
        topo.vertex(contact_edge.start())?.point(),
        topo.vertex(contact_edge.end())?.point(),
    );
    let curve = contact_edge.curve();
    let (t0, t1) = curve.domain_with_endpoints(sp, ep);
    let (tp, tq) = if contact_edge.start() == p {
        (t0, t1)
    } else {
        (t1, t0)
    };
    #[allow(clippy::cast_precision_loss)]
    let at_sample = |k: usize| tp + (tq - tp) * (k as f64) / (SAMPLES as f64);
    let mut bracket = None;
    for k in 1..=SAMPLES {
        match offset(curve.evaluate_with_endpoints(at_sample(k), sp, ep)) {
            Some(off) if off >= 0.0 => {
                bracket = Some((at_sample(k - 1), at_sample(k)));
                break;
            }
            Some(_) => {}
            None => return Ok(None),
        }
    }
    let Some((mut inside, mut outside)) = bracket else {
        return Ok(None);
    };
    for _ in 0..100 {
        let middle = f64::midpoint(inside, outside);
        match offset(curve.evaluate_with_endpoints(middle, sp, ep)) {
            Some(off) if off < 0.0 => inside = middle,
            Some(_) => outside = middle,
            None => return Ok(None),
        }
    }
    let tx = f64::midpoint(inside, outside);
    let x = curve.evaluate_with_endpoints(tx, sp, ep);
    // `E` run on from `C` to `X`: `X` on its carrier, past `C`.
    let on_carrier = 1e-5 * (pp - ap).length().max(1.0);
    let other = topo.vertex(other_end(&corner_edge, c))?.point();
    let run_on = match corner_edge.curve() {
        EdgeCurve::Line => {
            let along = (cp - other).normalize()?;
            let reach = (x - cp).dot(along);
            if reach <= tol || (x - cp - along * reach).length() > on_carrier {
                return Ok(None);
            }
            EdgeCurve::Line
        }
        EdgeCurve::Circle(circle) => {
            let off_plane = (x - circle.center()).dot(circle.normal()).abs();
            let off_circle = ((x - circle.center()).length() - circle.radius()).abs();
            if off_plane > on_carrier || off_circle > on_carrier {
                return Ok(None);
            }
            EdgeCurve::Circle(circle.clone())
        }
        _ => return Ok(None),
    };
    // The run must leave `C` away from `E` and take the short way round;
    // a circle runs its own way, so the edge runs from whichever end
    // makes it so.
    let probe = Edge::new(c, c, run_on.clone());
    let mut run_from_x = None;
    for from_x in [true, false] {
        let (s0, s1) = if from_x { (x, cp) } else { (cp, x) };
        let (r0, r1) = probe.curve().domain_with_endpoints(s0, s1);
        let middle = probe
            .curve()
            .evaluate_with_endpoints(f64::midpoint(r0, r1), s0, s1);
        if (middle - x).length() < (cp - x).length()
            && !point_on_edge(topo, corner, middle, on_carrier)?
        {
            run_from_x = Some(from_x);
            break;
        }
    }
    let Some(run_from_x) = run_from_x else {
        return Ok(None);
    };
    let Some(meet) = band_meets_cap(band, x, ap, offset) else {
        return Ok(None);
    };
    let piece = match curve {
        EdgeCurve::NurbsCurve(nurbs) => {
            let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(nurbs, tx)?;
            EdgeCurve::NurbsCurve(if tq > tx { right } else { left })
        }
        other => other.clone(),
    };
    log::debug!(
        "runout {face_id:?}: the round is cut back to its cap {k_face:?} at {x:?}, the cap run on over the notch"
    );
    // A wall running up the cap's whole corner edge and back down the
    // piece: the edge splits at `A`, the wall keeps the piece below, and
    // the cap takes the piece below and the piece above.
    if let Some((whole, wall_wire)) = spur {
        let whole_edge = topo.edge(whole)?.clone();
        let far = other_end(&whole_edge, c);
        let wall = topo.wire(wall_wire)?.edges().to_vec();
        let n = wall.len();
        let Some(at) = (0..n).find(|&k| {
            let pair = [wall[k].edge(), wall[(k + 1) % n].edge()];
            pair == [whole, v_id] || pair == [v_id, whole]
        }) else {
            return Ok(None);
        };
        let rest = topo.add_edge(Edge::new(a, far, EdgeCurve::Line));
        let first = wall[at];
        let first_edge = topo.edge(first.edge())?.clone();
        let from = first.oriented_start(&first_edge);
        let mut folded: Vec<OrientedEdge> = Vec::with_capacity(n - 1);
        for (k, &oe) in wall.iter().enumerate() {
            if k == at {
                folded.push(OrientedEdge::new(rest, from == a));
            } else if k != (at + 1) % n {
                folded.push(oe);
            }
        }
        let closed = topo.wire(wall_wire)?.is_closed();
        *topo.wire_mut(wall_wire)? = Wire::new(folded, closed)?;
        rewrite_wire(topo, k_wire, |oe| {
            if oe.edge() != whole {
                vec![oe]
            } else if oe.oriented_start(&whole_edge) == c {
                vec![
                    OrientedEdge::new(v_id, v_edge.start() == c),
                    OrientedEdge::new(rest, true),
                ]
            } else {
                vec![
                    OrientedEdge::new(rest, false),
                    OrientedEdge::new(v_id, v_edge.start() == a),
                ]
            }
        })?;
    }
    let vertex_tol = topo.vertex(p)?.tolerance();
    let vx = topo.add_vertex(Vertex::new(x, vertex_tol));
    let new_contact = topo.add_edge(if contact_edge.start() == p {
        Edge::new(vx, q, piece)
    } else {
        Edge::new(q, vx, piece)
    });
    let meet_id = topo.add_edge(Edge::new(vx, a, meet));
    let run_id = topo.add_edge(if run_from_x {
        Edge::new(vx, c, run_on)
    } else {
        Edge::new(c, vx, run_on)
    });
    // `from` -> the start the replacement takes.
    let moved = |from: VertexId| if from == p { vx } else { from };
    let swap_contact = |oe: OrientedEdge| {
        (oe.edge() == contact).then(|| OrientedEdge::new(new_contact, oe.is_forward()))
    };
    rewrite_wire(topo, band_wire, |oe| {
        if let Some(swapped) = swap_contact(oe) {
            vec![swapped]
        } else if oe.edge() == b_id {
            let from = moved(oe.oriented_start(&b_edge));
            vec![OrientedEdge::new(meet_id, from == vx)]
        } else {
            vec![oe]
        }
    })?;
    rewrite_wire(topo, f_wire, |oe| {
        if let Some(swapped) = swap_contact(oe) {
            vec![swapped]
        } else if oe.edge() == l_id {
            let from = moved(oe.oriented_start(&l_edge));
            vec![OrientedEdge::new(run_id, (from == vx) == run_from_x)]
        } else {
            vec![oe]
        }
    })?;
    rewrite_wire(topo, k_wire, |oe| {
        if oe.edge() != v_id {
            vec![oe]
        } else if oe.oriented_start(&v_edge) == a {
            // `A` -> `X` -> `C`.
            vec![
                OrientedEdge::new(meet_id, false),
                OrientedEdge::new(run_id, run_from_x),
            ]
        } else {
            vec![
                OrientedEdge::new(run_id, !run_from_x),
                OrientedEdge::new(meet_id, true),
            ]
        }
    })?;
    Ok(Some(RunOn {
        corner: c,
        edge: corner,
        run: run_id,
        wires: [f_wire, k_wire],
    }))
}

/// Whether `edge` traces `on` within `tol`, certified rather than sampled.
/// Every point of `edge` is held within `tol` of `on` by a bound on
/// `edge`'s speed along its parameter: between two samples, no point
/// strays farther than that speed allows. Where `on` is a line or an arc
/// of a circle short of a whole turn, `edge` runs between `on`'s ends, so
/// the feet of its points on `on` sweep all of `on`, and every point of
/// `on` then lies within `tol` of `edge` too. An ellipse or a B-spline `on`
/// (the other round's contact, in a full round) has no such sweep to stand
/// on, so it is held to `edge` the same way in turn.
fn edge_lies_on(topo: &Topology, edge: &Edge, on: &Edge, tol: f64) -> Result<bool, BlendError> {
    let (p, q) = (
        topo.vertex(on.start())?.point(),
        topo.vertex(on.end())?.point(),
    );
    match on.curve() {
        EdgeCurve::Line => {
            if (q - p).length_squared() <= 1e-18 {
                return Ok(false);
            }
        }
        // Points within `tol` of the arc must keep off the half-plane across
        // its gap, where the nearer end (and so the foot) jumps.
        EdgeCurve::Circle(circle) => {
            let (s0, s1) = on.curve().domain_with_endpoints(p, q);
            let gap = std::f64::consts::TAU - (s1 - s0);
            if circle.radius() * (0.5 * gap).min(std::f64::consts::FRAC_PI_2).sin() <= tol {
                return Ok(false);
            }
        }
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
            return Ok(stays_within(topo, edge, on, tol)? && stays_within(topo, on, edge, tol)?);
        }
    }
    stays_within(topo, edge, on, tol)
}

/// Whether every point of `edge` lies within `tol` of `on`, between `on`'s
/// ends: `edge` sampled closely enough for its speed bound to cover the
/// gaps, each sample's distance to `on` read as no less than the true one.
fn stays_within(topo: &Topology, edge: &Edge, on: &Edge, tol: f64) -> Result<bool, BlendError> {
    let (a, b) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let (p, q) = (
        topo.vertex(on.start())?.point(),
        topo.vertex(on.end())?.point(),
    );
    let (s0, s1) = on.curve().domain_with_endpoints(p, q);
    let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
    let Some(speed) = speed_bound(edge.curve(), a, b) else {
        return Ok(false);
    };
    // Samples close enough that every point lies within a quarter of `tol`
    // of one.
    let count = (2.0 * speed * (t1 - t0).abs() / tol).ceil().max(1.0);
    if !count.is_finite() || count > 1e6 {
        return Ok(false);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let count = count as u32;
    let step = (t1 - t0) / f64::from(count);
    // Where a point's foot on the whole curve falls outside `on`, its
    // distance to the nearer end stands in, which is never less than its
    // distance to `on`.
    let nearer_end = |at: Point3| (p - at).length().min((q - at).length());
    // The bound only grows with `worst`, so once one sample puts it past
    // `tol` the answer is settled.
    let mut worst: f64 = 0.0;
    for k in 0..=count {
        let at = edge
            .curve()
            .evaluate_with_endpoints(f64::from(k).madd(step, t0), a, b);
        let off = match on.curve() {
            EdgeCurve::Line => {
                let chord = q - p;
                if chord.length_squared() <= 1e-18 {
                    (p - at).length()
                } else {
                    let along = ((at - p).dot(chord) / chord.dot(chord)).clamp(0.0, 1.0);
                    (p + chord * along - at).length()
                }
            }
            EdgeCurve::Circle(circle) => {
                let angle = circle.project(at);
                if (angle - s0).rem_euclid(std::f64::consts::TAU) <= s1 - s0 {
                    (circle.evaluate(angle) - at).length()
                } else {
                    nearer_end(at)
                }
            }
            EdgeCurve::Ellipse(ellipse) => {
                let angle = ellipse.project(at);
                if (angle - s0).rem_euclid(std::f64::consts::TAU) <= s1 - s0 {
                    (ellipse.evaluate(angle) - at).length()
                } else {
                    nearer_end(at)
                }
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                match brepkit_math::nurbs::projection::project_point_to_curve(nurbs, at, 1e-9) {
                    Ok(found) if (s0.min(s1)..=s0.max(s1)).contains(&found.parameter) => {
                        found.distance
                    }
                    Ok(_) => nearer_end(at),
                    Err(_) => return Ok(false),
                }
            }
        };
        worst = worst.max(off);
        if (0.5 * speed).madd(step.abs(), worst) > tol {
            log::debug!("edge stands {worst:e} off its neighbour at a sample, past {tol:e}");
            return Ok(false);
        }
    }
    let bound = (0.5 * speed).madd(step.abs(), worst);
    log::debug!("edge lies within {bound:e} of its neighbour ({worst:e} at its samples)");
    Ok(bound <= tol)
}

/// Whether a patch beyond a spur's short edge carries on the face beyond its
/// long edge: in its plane (or, on a curved face, its tangent plane along the
/// short edge, within `curved` radians) and facing its way.
fn patch_carries_on(
    topo: &Topology,
    patch: FaceId,
    beyond: FaceId,
    short: EdgeId,
    tol: f64,
    curved: f64,
) -> Result<bool, BlendError> {
    let Some((n2, _)) = outward_plane(topo, patch)? else {
        return Ok(false);
    };
    let edge = topo.edge(short)?;
    let (a, b) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    // A curved face beyond may meet the patch at a slight kink: a fitted
    // surface rarely runs on exactly level.
    let alike = if topo.face(beyond)?.surface().is_planar() {
        1.0 - 1e-9
    } else {
        curved.cos()
    };
    for at in [a, b, a + (b - a) * 0.5] {
        if !outward_normal_at(topo, beyond, at, tol)?.is_some_and(|n1| n1.dot(n2) >= alike) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A bound on `curve`'s speed along its parameter: a line's length (its
/// parameter runs from 0 to 1), a circle's radius, an ellipse's semi-major
/// axis. A B-spline needs every weight positive (`None` otherwise). Its
/// derivative is `sum_j N_{j,p-1} p [w_j (P_j - C) - w_{j-1} (P_{j-1} - C)]
/// / ((u_{j+p} - u_j) w)`, where the `N_{j,p-1}` sum to one, `w` is at least
/// the least weight and `C` lies in the control points' hull, so `|P - C|`
/// is at most their spread.
pub(crate) fn speed_bound(curve: &EdgeCurve, a: Point3, b: Point3) -> Option<f64> {
    match curve {
        EdgeCurve::Line => Some((b - a).length()),
        EdgeCurve::Circle(circle) => Some(circle.radius()),
        EdgeCurve::Ellipse(ellipse) => Some(ellipse.semi_major()),
        EdgeCurve::NurbsCurve(nurbs) => {
            let (points, weights, knots) = (nurbs.control_points(), nurbs.weights(), nurbs.knots());
            let degree = nurbs.degree();
            let first = *points.first()?;
            if degree == 0 || !weights.iter().all(|w| w.is_finite() && *w > 0.0) {
                return None;
            }
            let (least, most) = weights
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &w| {
                    (lo.min(w), hi.max(w))
                });
            let spread = 2.0
                * points
                    .iter()
                    .map(|&point| (point - first).length())
                    .fold(0.0, f64::max);
            #[allow(clippy::cast_precision_loss)]
            let degree_f = degree as f64;
            let mut bound: f64 = 0.0;
            for j in 1..points.len() {
                let span = knots.get(j + degree)? - knots.get(j)?;
                if span <= 0.0 {
                    continue;
                }
                let lift = most.madd(
                    (points[j] - points[j - 1]).length(),
                    (weights[j] - weights[j - 1]).abs() * spread,
                );
                bound = bound.max(degree_f * lift / (span * least));
            }
            bound.is_finite().then_some(bound)
        }
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

fn align_face_boundary_against_neighbor(
    topo: &mut Topology,
    face_id: FaceId,
    neighbor_id: FaceId,
    shared_edge: EdgeId,
) -> Result<(), BlendError> {
    let face_forward = face_edge_forward(topo, face_id, shared_edge)
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;
    let neighbor_forward = face_edge_forward(topo, neighbor_id, shared_edge)
        .ok_or(BlendError::TrimmingFailure { face: neighbor_id })?;
    let face_reversed = topo.face(face_id)?.is_reversed();
    let neighbor_reversed = topo.face(neighbor_id)?.is_reversed();
    if face_forward ^ face_reversed != neighbor_forward ^ neighbor_reversed {
        return Ok(());
    }

    let mut wires = vec![topo.face(face_id)?.outer_wire()];
    wires.extend_from_slice(topo.face(face_id)?.inner_wires());
    for wire_id in wires {
        let wire = topo.wire_mut(wire_id)?;
        let mut edges = wire.edges().to_vec();
        edges.reverse();
        for edge in &mut edges {
            *edge = OrientedEdge::new(edge.edge(), !edge.is_forward());
        }
        for (slot, edge) in wire.edges_mut().iter_mut().zip(edges) {
            *slot = edge;
        }
    }
    Ok(())
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

/// The first and last vertex of a contact's chain of registry boundaries,
/// along the contact.
fn contact_chain_ends(
    registry: &BoundaryRegistry,
    chain: &[crate::boundary_registry::BoundaryHandle],
) -> Result<(VertexId, VertexId), BlendError> {
    let entry = |handle: crate::boundary_registry::BoundaryHandle| {
        registry
            .entry(handle)
            .ok_or_else(|| BlendError::PlanningFailure {
                reason: format!("unknown contact boundary {handle}"),
            })
    };
    let (Some(&first), Some(&last)) = (chain.first(), chain.last()) else {
        return Err(BlendError::PlanningFailure {
            reason: "empty contact chain".to_owned(),
        });
    };
    Ok((entry(first)?.start.vertex, entry(last)?.end.vertex))
}

fn contact_forward_for_spine(
    topo: &Topology,
    registry: &BoundaryRegistry,
    chain: &[crate::boundary_registry::BoundaryHandle],
    support_face: FaceId,
    terminals: Option<(VertexId, VertexId)>,
    starts_at_spine_start: bool,
) -> Result<Option<bool>, BlendError> {
    let Some((spine_start, spine_end)) = terminals else {
        return Ok(None);
    };
    let (first, last) = contact_chain_ends(registry, chain)?;
    let direct_vertices = (
        contact_source_vertex(topo, support_face, first, (spine_start, spine_end)),
        contact_source_vertex(topo, support_face, last, (spine_start, spine_end)),
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

/// The fillet of `shared` built in place on `solid` after `join` edits its
/// shell, with `solid` then put back as `snapshot` holds it. `join` returns
/// the edges it took out of the shell, which lie between faces tangent along
/// them and leave the selection, or `None` to build nothing.
fn build_after_joins(
    topo: &mut Topology,
    solid: SolidId,
    snapshot: &SolidSnapshot,
    shared: &[(Vec<EdgeId>, std::sync::Arc<RadiusLaw>)],
    join: impl FnOnce(&mut Topology) -> Result<Option<HashSet<EdgeId>>, BlendError>,
) -> Result<Option<BlendResult>, BlendError> {
    let built = join(topo).and_then(|gone| {
        let Some(gone) = gone else {
            return Ok(None);
        };
        let edge_sets = shared
            .iter()
            .map(|(edges, law)| {
                let edges = edges
                    .iter()
                    .copied()
                    .filter(|e| !gone.contains(e))
                    .collect();
                (edges, read_law(law))
            })
            .collect();
        build_in_place(topo, solid, edge_sets).map(Some)
    });
    match built {
        Ok(Some(result)) => {
            let preserve = snapshot.clone_result_source_faces(topo, result.solid);
            let restore = snapshot.restore(topo);
            match (preserve, restore) {
                (Ok(()), Ok(())) => Ok(Some(result)),
                (Err(error), _) | (_, Err(error)) => Err(error),
            }
        }
        Ok(None) => {
            snapshot.restore(topo)?;
            Ok(None)
        }
        Err(error) => {
            snapshot.restore(topo)?;
            Err(error)
        }
    }
}

/// A law that reads as `law` does, for one more build.
fn read_law(law: &std::sync::Arc<RadiusLaw>) -> RadiusLaw {
    match **law {
        RadiusLaw::Constant(r) => RadiusLaw::Constant(r),
        RadiusLaw::Linear { start, end } => RadiusLaw::Linear { start, end },
        RadiusLaw::SCurve { start, end } => RadiusLaw::SCurve { start, end },
        RadiusLaw::Custom(_) => {
            let law = std::sync::Arc::clone(law);
            RadiusLaw::Custom(Box::new(move |t| law.evaluate(t)))
        }
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
    /// The rim is one of the cap's holes (a post standing on a plate), not
    /// its whole outer wire (a disc cap).
    rim_is_hole: bool,
    /// The wall's other rim, where the wall contact lands on it: a wall
    /// exactly a radius high. The wall goes, and the band takes this rim.
    far_rim: Option<EdgeId>,
}

/// The lowest height along `toward`, from `base`, of the wall boundary that
/// `assemble_closed_rim` keeps: every edge but the rim, and the far end of
/// each seam leaving the rim's vertex. A conic counts its whole curve's
/// lowest point and a NURBS edge its control hull's, so the value never sits
/// above the boundary's own; a NURBS edge whose weights are not all positive
/// has no hull bound and counts as reaching the base.
fn kept_wall_boundary_height(
    topo: &Topology,
    wall_face: FaceId,
    rim_edge: EdgeId,
    base: Point3,
    toward: Vec3,
) -> Result<f64, BlendError> {
    let height = |p: Point3| (p - base).dot(toward);
    let rim_vertex = topo.edge(rim_edge)?.start();
    let face = topo.face(wall_face)?;
    let mut lowest = f64::INFINITY;
    for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oriented in topo.wire(wire)?.edges() {
            if oriented.edge() == rim_edge {
                continue;
            }
            let edge = topo.edge(oriented.edge())?;
            let ends = [edge.start(), edge.end()];
            if ends.contains(&rim_vertex) {
                for v in ends.into_iter().filter(|&v| v != rim_vertex) {
                    lowest = lowest.min(height(topo.vertex(v)?.point()));
                }
                continue;
            }
            let edge_lowest = match edge.curve() {
                EdgeCurve::Line => {
                    height(topo.vertex(ends[0])?.point()).min(height(topo.vertex(ends[1])?.point()))
                }
                EdgeCurve::Circle(c) => {
                    let across = (1.0 - c.normal().dot(toward).powi(2)).max(0.0).sqrt();
                    height(c.center()) - c.radius() * across
                }
                EdgeCurve::Ellipse(el) => {
                    let u = el.semi_major() * el.u_axis().dot(toward);
                    let v = el.semi_minor() * el.v_axis().dot(toward);
                    height(el.center()) - u.hypot(v)
                }
                EdgeCurve::NurbsCurve(n) => {
                    if n.weights().iter().all(|&w| w.is_finite() && w > 0.0) {
                        n.control_points()
                            .iter()
                            .map(|&p| height(p))
                            .fold(f64::INFINITY, f64::min)
                    } else {
                        0.0
                    }
                }
            };
            lowest = lowest.min(edge_lowest);
        }
    }
    Ok(lowest)
}

/// Two rims of one wall: each wall contact moves from its rim toward the
/// other's, and the pair must keep that order with some wall left between.
fn contacts_leave_wall_between(a: &ClosedRimInfo, b: &ClosedRimInfo) -> bool {
    let span = b.plate_circle.center() - a.plate_circle.center();
    let Ok(along) = span.normalize() else {
        return false;
    };
    (b.wall_circle.center() - a.wall_circle.center()).dot(along) > 1e-7
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

    // The annular rebuild replaces the wire that is this rim alone with the
    // plate-contact circle: the cap's outer wire (a disc cap) or one of its
    // holes (a post on a plate). The cap's other wires are kept, which is
    // checked below once the contact radius is known.
    let is_rim_wire = |wire: brepkit_topology::wire::WireId| {
        topo.wire(wire)
            .is_ok_and(|w| w.edges().len() == 1 && w.edges()[0].edge() == rim_edge)
    };
    let (rim_is_hole, kept_wires) = {
        let cap = topo.face(plane_face)?;
        if is_rim_wire(cap.outer_wire()) {
            (false, cap.inner_wires().to_vec())
        } else if cap.inner_wires().iter().any(|&w| is_rim_wire(w)) {
            let kept: Vec<_> = std::iter::once(cap.outer_wire())
                .chain(cap.inner_wires().iter().copied())
                .filter(|&w| !is_rim_wire(w))
                .collect();
            (true, kept)
        } else {
            return Ok(None);
        }
    };

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

    // The wall contact circle must land on the wall, short of every point of
    // the boundary the rebuild keeps: a fillet wider than a plate is thick, or
    // a post whose sloped top dips below the contact, would carry it past.
    // Landing on the wall's other rim (a wall exactly a radius high) leaves
    // no wall: the band takes that rim.
    let rise = wall_center - plate_center;
    let mut far_rim = None;
    if let Ok(toward) = rise.normalize() {
        let height = kept_wall_boundary_height(topo, wall_face, rim_edge, plate_center, toward)?;
        if rise.length() >= height - 1e-7 {
            far_rim = landed_far_rim(topo, wall_face, rim_edge, axis, axis_origin, wall_center)?;
            if far_rim.is_none() {
                return Err(BlendError::PlanningFailure {
                    reason: format!(
                        "the fillet's wall contact would land past the wall bounding rim {rim_edge:?}"
                    ),
                });
            }
        }
    }

    if rim_is_hole {
        // Every kept wire must stay outside the plate contact, or the band
        // would run over it.
        let clearance = |a: Point3, b: Point3| {
            let ab = b - a;
            let len2 = ab.length_squared();
            let t = if len2 <= 1e-24 {
                0.0
            } else {
                ((plate_center - a).dot(ab) / len2).clamp(0.0, 1.0)
            };
            (a + ab * t - plate_center).length()
        };
        for &wire in &kept_wires {
            for oe in topo.wire(wire)?.edges() {
                let e = topo.edge(oe.edge())?;
                let near = match e.curve() {
                    EdgeCurve::Line => clearance(
                        topo.vertex(e.start())?.point(),
                        topo.vertex(e.end())?.point(),
                    ),
                    EdgeCurve::Circle(c) => {
                        ((c.center() - plate_center).length() - c.radius()).abs()
                    }
                    // Every point of an ellipse lies between its semi-axes
                    // from its centre.
                    EdgeCurve::Ellipse(el) => {
                        let d = (el.center() - plate_center).length();
                        (d - el.semi_major()).max(el.semi_minor() - d)
                    }
                    EdgeCurve::NurbsCurve(_) => return Ok(None),
                };
                if near <= plate_radius + 1e-7 {
                    return Ok(None);
                }
            }
        }
    }

    // A hole reaching the plate contact would meet the blend; the normal trim
    // path handles that.
    let cap_holes: &[brepkit_topology::wire::WireId] = if rim_is_hole { &[] } else { &kept_wires };
    for &hole in cap_holes {
        for oe in topo.wire(hole)?.edges() {
            let e = topo.edge(oe.edge())?;
            let reach = match e.curve() {
                EdgeCurve::Circle(c) => radial_distance(c.center(), axis_origin, axis) + c.radius(),
                EdgeCurve::Ellipse(el) => {
                    radial_distance(el.center(), axis_origin, axis) + el.semi_major()
                }
                EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => {
                    let (a, b) = (
                        topo.vertex(e.start())?.point(),
                        topo.vertex(e.end())?.point(),
                    );
                    radial_distance(a, axis_origin, axis).max(radial_distance(b, axis_origin, axis))
                }
            };
            if matches!(e.curve(), EdgeCurve::NurbsCurve(_)) || reach >= plate_radius - 1e-7 {
                return Ok(None);
            }
        }
    }

    // Both contact circles start where the rim's vertex is, so their
    // vertices sit on the wall's seam; where the wall goes, where its other
    // rim's vertex is, which the band's seam then runs to.
    let anchor = far_rim.unwrap_or(rim_edge);
    let rim_point = topo.vertex(topo.edge(anchor)?.start())?.point();
    let reference = (rim_point - project_onto_axis(rim_point, axis_origin, axis)).normalize()?;
    let plate_circle = Circle3D::new_with_ref(plate_center, axis, plate_radius, reference)?;
    let wall_circle = match far_rim.map(|edge| topo.edge(edge).map(|e| e.curve().clone())) {
        Some(curve) => match curve? {
            EdgeCurve::Circle(circle) => circle,
            _ => return Ok(None),
        },
        None => Circle3D::new_with_ref(wall_center, axis, wall_radius, reference)?,
    };

    Ok(Some(ClosedRimInfo {
        plane_face,
        wall_face,
        rim_edge,
        plate_circle,
        wall_circle,
        rim_is_hole,
        far_rim,
    }))
}

/// The wall's other rim, where a closed rim's wall contact lands exactly on
/// it: one closed circle about the wall's axis through the contact's centre,
/// of the contact's radius, the wall's wire otherwise only that rim, `rim`
/// and the seam edges between their vertices, and no holes. `None` for any
/// other wall.
fn landed_far_rim(
    topo: &Topology,
    wall_face: FaceId,
    rim: EdgeId,
    axis: Vec3,
    axis_origin: Point3,
    contact_center: Point3,
) -> Result<Option<EdgeId>, BlendError> {
    // A file's digits: the wall's height and the round's radius each read
    // to about 1e-7 of the part.
    const ON: f64 = 1e-6;
    let face = topo.face(wall_face)?;
    if !face.inner_wires().is_empty() {
        return Ok(None);
    }
    let FaceSurface::Cylinder(cylinder) = face.surface() else {
        return Ok(None);
    };
    let tol = ON * cylinder.radius().max(1.0);
    let rim_vertex = topo.edge(rim)?.start();
    let mut far = None;
    for oriented in topo.wire(face.outer_wire())?.edges() {
        let id = oriented.edge();
        if id == rim {
            continue;
        }
        let edge = topo.edge(id)?;
        if let EdgeCurve::Circle(circle) = edge.curve()
            && edge.start() == edge.end()
        {
            let on_axis = radial_distance(circle.center(), axis_origin, axis);
            if far.is_some_and(|f| f != id)
                || on_axis > tol
                || (circle.center() - contact_center).length() > tol
                || (circle.radius() - cylinder.radius()).abs() > tol
                || circle.normal().cross(axis).length() > ON
            {
                return Ok(None);
            }
            far = Some(id);
            continue;
        }
        // A seam: a line from the rim's vertex up the wall.
        if !matches!(edge.curve(), EdgeCurve::Line)
            || ![edge.start(), edge.end()].contains(&rim_vertex)
        {
            return Ok(None);
        }
    }
    Ok(far)
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
    let cap_outer = cap_orig_wire.outer_wire();
    let cap_holes = cap_orig_wire.inner_wires().to_vec();
    let rim_wire = if rim.rim_is_hole {
        cap_holes
            .iter()
            .copied()
            .find(|&w| {
                topo.wire(w)
                    .is_ok_and(|w| w.edges().iter().any(|oe| oe.edge() == rim.rim_edge))
            })
            .ok_or(BlendError::TrimmingFailure {
                face: rim.plane_face,
            })?
    } else {
        cap_outer
    };
    let cap_forward = topo
        .wire(rim_wire)?
        .edges()
        .iter()
        .find(|oe| oe.edge() == rim.rim_edge)
        .is_some_and(OrientedEdge::is_forward);

    // Vertices for the two closed contact circles (start == end → degenerate);
    // where the wall goes, the band's wall side is the wall's other rim, at
    // its own vertex.
    let plate_v = topo.add_vertex(Vertex::new(rim.plate_circle.evaluate(0.0), TOL));
    let wall_v = match rim.far_rim {
        Some(far) => topo.edge(far)?.start(),
        None => topo.add_vertex(Vertex::new(rim.wall_circle.evaluate(0.0), TOL)),
    };

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
    let band_reversed = torus_band_needs_reversal(&torus, rim, (&wall_surf, wall_reversed));
    let wall_forward = wall_oriented
        .iter()
        .find(|oe| oe.edge() == rim.rim_edge)
        .is_some_and(OrientedEdge::is_forward);
    let plane_side = usize::from(stripe.face2 == rim.plane_face);
    let wall_side = usize::from(stripe.face2 == rim.wall_face);
    // Both contact circles turn about the wall's axis; the cap and the wall
    // keep their traversal direction whichever way the rim circle's own
    // normal runs, and the band runs each circle against its neighbour.
    let rim_turns_with_axis = match topo.edge(rim.rim_edge)?.curve() {
        EdgeCurve::Circle(c) => c.normal().dot(rim.plate_circle.normal()) > 0.0,
        _ => true,
    };
    let cap_sense = cap_forward == rim_turns_with_axis;
    let wall_support_sense = wall_forward == rim_turns_with_axis;
    let plate_sense = (cap_sense == plane_reversed) != band_reversed;
    let wall_sense = (wall_support_sense == wall_reversed) != band_reversed;
    let plate_handle = registry.register(
        BoundaryKey::contact(contour_id, 0, plane_side as u8),
        PlannedVertex::new(plate_v),
        PlannedVertex::new(plate_v),
        EdgeCurve::Circle(rim.plate_circle.clone()),
        (0.0, std::f64::consts::TAU),
        [
            BoundaryOwner::planned("closed-rim support cap", cap_sense),
            BoundaryOwner::planned("closed-rim torus band", plate_sense),
        ],
    )?;
    if let Some(far) = rim.far_rim {
        return assemble_closed_rim_without_wall(
            topo,
            stripe,
            rim,
            far,
            (plate_v, wall_v, plate_handle),
            (band_reversed, wall_reversed),
            &wall_oriented,
            registry,
            face_replacements,
            (plane_surf, plane_reversed, cap_outer, cap_holes, rim_wire),
        );
    }
    let wall_handle = registry.register(
        BoundaryKey::contact(contour_id, 0, wall_side as u8),
        PlannedVertex::new(wall_v),
        PlannedVertex::new(wall_v),
        EdgeCurve::Circle(rim.wall_circle.clone()),
        (0.0, std::f64::consts::TAU),
        [
            BoundaryOwner::planned("closed-rim support wall", wall_support_sense),
            BoundaryOwner::planned("closed-rim torus band", wall_sense),
        ],
    )?;
    let seam_curve = crate::builder_utils::seam_curve(
        &stripe.surface,
        &stripe.sections,
        topo.vertex(plate_v)?.point(),
        topo.vertex(wall_v)?.point(),
    );
    let seam_edge = topo.add_edge(Edge::new(plate_v, wall_v, seam_curve));
    let cap_boundary = registry.oriented_edge(topo, plate_handle, 0)?;
    let cap_wire = Wire::new(vec![cap_boundary], true)?;
    let cap_wire_id = topo.add_wire(cap_wire);
    let (cap_outer, cap_holes) = if rim.rim_is_hole {
        let holes = cap_holes
            .iter()
            .map(|&w| if w == rim_wire { cap_wire_id } else { w })
            .collect();
        (cap_outer, holes)
    } else {
        (cap_wire_id, cap_holes)
    };
    let mut cap_face = Face::new(cap_outer, cap_holes, plane_surf);
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

/// [`assemble_closed_rim`] where the wall contact lands on the wall's other
/// rim: the cap is rebuilt as there, the wall goes (the caller leaves it out
/// of the result), and the band runs from the plate contact to that rim,
/// using it as the wall did.
#[allow(clippy::too_many_arguments)]
fn assemble_closed_rim_without_wall(
    topo: &mut Topology,
    stripe: &Stripe,
    rim: &ClosedRimInfo,
    far: EdgeId,
    (plate_v, wall_v, plate_handle): (VertexId, VertexId, crate::boundary_registry::BoundaryHandle),
    (band_reversed, wall_reversed): (bool, bool),
    wall_oriented: &[OrientedEdge],
    registry: &mut BoundaryRegistry,
    face_replacements: &mut std::collections::HashMap<FaceId, FaceId>,
    (plane_surf, plane_reversed, cap_outer, cap_holes, rim_wire): (
        FaceSurface,
        bool,
        WireId,
        Vec<WireId>,
        WireId,
    ),
) -> Result<FaceId, BlendError> {
    // The band takes the rim as the wall did, its sense read against its own
    // surface: a wire runs about its surface's normal, which a reversed face
    // turns.
    let far_forward = wall_oriented
        .iter()
        .find(|oe| oe.edge() == far)
        .map(|oe| oe.is_forward() != (wall_reversed != band_reversed))
        .ok_or(BlendError::TrimmingFailure {
            face: rim.wall_face,
        })?;
    let seam_curve = crate::builder_utils::seam_curve(
        &stripe.surface,
        &stripe.sections,
        topo.vertex(plate_v)?.point(),
        topo.vertex(wall_v)?.point(),
    );
    let seam_edge = topo.add_edge(Edge::new(plate_v, wall_v, seam_curve));
    let cap_boundary = registry.oriented_edge(topo, plate_handle, 0)?;
    let cap_wire_id = topo.add_wire(Wire::new(vec![cap_boundary], true)?);
    let (cap_outer, cap_holes) = if rim.rim_is_hole {
        let holes = cap_holes
            .iter()
            .map(|&w| if w == rim_wire { cap_wire_id } else { w })
            .collect();
        (cap_outer, holes)
    } else {
        (cap_wire_id, cap_holes)
    };
    let mut cap_face = Face::new(cap_outer, cap_holes, plane_surf);
    cap_face.set_reversed(plane_reversed);
    let cap_face_id = topo.add_face(cap_face);
    registry.set_owner_face(plate_handle, 0, cap_face_id)?;
    face_replacements.insert(rim.plane_face, cap_face_id);

    let plate_boundary = registry.oriented_edge(topo, plate_handle, 1)?;
    let band_wire = Wire::new(
        vec![
            plate_boundary,
            OrientedEdge::new(seam_edge, true),
            OrientedEdge::new(far, far_forward),
            OrientedEdge::new(seam_edge, false),
        ],
        true,
    )?;
    let band_wire_id = topo.add_wire(band_wire);
    let mut band_face = Face::new(band_wire_id, Vec::new(), stripe.surface.clone());
    band_face.set_reversed(band_reversed);
    let band_face_id = topo.add_face(band_face);
    registry.set_owner_face(plate_handle, 1, band_face_id)?;
    registry.install_pcurves(topo, plate_handle)?;
    Ok(band_face_id)
}

/// Whether a stripe's spine ends (rather than starts) at `vertex`.
fn spine_ends_at(topo: &Topology, stripe: &Stripe, vertex: Point3) -> Result<bool, BlendError> {
    let length = stripe.spine.length();
    Ok((stripe.spine.evaluate(topo, length)? - vertex).length()
        < (stripe.spine.evaluate(topo, 0.0)? - vertex).length())
}

/// A stripe's cross-section at its spine's end at `vertex`.
fn end_section<'s>(
    topo: &Topology,
    stripe: &'s Stripe,
    vertex: Point3,
) -> Result<Option<&'s crate::section::CircSection>, BlendError> {
    Ok(if spine_ends_at(topo, stripe, vertex)? {
        stripe.sections.last()
    } else {
        stripe.sections.first()
    })
}

/// A contact curve's end nearer `point`, and whether it is the curve's end
/// (rather than its start).
fn contact_end_at(
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
    point: Point3,
) -> (Point3, bool) {
    let (t0, t1) = contact.domain();
    let (start, end) = (contact.evaluate(t0), contact.evaluate(t1));
    if (end - point).length() < (start - point).length() {
        (end, true)
    } else {
        (start, false)
    }
}

/// The edges of `cap`'s wires that end at `vertex`.
fn cap_corner_edges(
    topo: &Topology,
    cap: FaceId,
    vertex: VertexId,
) -> Result<Vec<EdgeId>, BlendError> {
    let face = topo.face(cap)?;
    let mut edges = Vec::new();
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oriented in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oriented.edge())?;
            if (edge.start() == vertex || edge.end() == vertex) && !edges.contains(&oriented.edge())
            {
                edges.push(oriented.edge());
            }
        }
    }
    Ok(edges)
}

/// Whether `point` lies on `edge` (between its ends) within `tol`.
fn point_on_edge(
    topo: &Topology,
    edge_id: EdgeId,
    point: Point3,
    tol: f64,
) -> Result<bool, BlendError> {
    Ok(edge_distance(topo, edge_id, point, None)? <= tol)
}

/// `point`'s distance from an edge, or from its trace on `onto` (each of
/// its points projected on to that surface): with no `onto`, as
/// [`EdgeReads`] reads it; else its nearest sample, refined by golden
/// section between the sample's neighbours.
fn edge_distance(
    topo: &Topology,
    edge_id: EdgeId,
    point: Point3,
    onto: Option<&FaceSurface>,
) -> Result<f64, BlendError> {
    edge_nearest(topo, edge_id, point, onto).map(|(distance, _)| distance)
}

/// `point`'s distance from an edge, or from its trace on `onto`, as
/// [`edge_distance`], with the point of the edge (or its trace) nearest it.
fn edge_nearest(
    topo: &Topology,
    edge_id: EdgeId,
    point: Point3,
    onto: Option<&FaceSurface>,
) -> Result<(f64, Point3), BlendError> {
    let Some(onto) = onto else {
        return Ok(EdgeReads::new(topo, edge_id)?.nearest(point));
    };
    let edge = topo.edge(edge_id)?;
    let (a, b) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let curve = edge.curve();
    let at = |t: f64| {
        let on = curve.evaluate_with_endpoints(t, a, b);
        let on = match onto {
            FaceSurface::Plane { normal, d } => {
                on - *normal
                    * ((normal.dot(Vec3::new(on.x(), on.y(), on.z())) - d)
                        / normal.length_squared())
            }
            surface => surface
                .project_point(on)
                .and_then(|(u, v)| surface.evaluate(u, v))
                .unwrap_or(on),
        };
        ((on - point).length(), on)
    };
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    let (lo, hi) = nearest_bracket(t0, t1, |t| at(t).0);
    Ok(at(golden_least(lo, hi, |t| at(t).0)))
}

/// Points along a curve's domain `t0` to `t1` at which a nearest point is
/// first looked for.
const NEAREST_SAMPLES: usize = 64;

/// The parameters of the even samples either side of the sample of `t0`
/// to `t1` at which `dist` is least.
fn nearest_bracket(t0: f64, t1: f64, dist: impl Fn(f64) -> f64) -> (f64, f64) {
    #[allow(clippy::cast_precision_loss)]
    let param = |k: usize| t0 + (t1 - t0) * (k as f64) / (NEAREST_SAMPLES as f64);
    let nearest = (0..=NEAREST_SAMPLES)
        .map(|k| (k, dist(param(k))))
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .map_or(0, |(k, _)| k);
    (
        param(nearest.saturating_sub(1)),
        param((nearest + 1).min(NEAREST_SAMPLES)),
    )
}

/// Where `dist` is least between `lo` and `hi`, by golden section.
fn golden_least(mut lo: f64, mut hi: f64, dist: impl Fn(f64) -> f64) -> f64 {
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..80 {
        let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if dist(x1) <= dist(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    f64::midpoint(lo, hi)
}

/// One edge, read for the distances of many points from it: a line or an
/// arc in closed form; any other curve from the nearest of its even samples,
/// taken once. A B-spline is refined from that sample by Newton steps on the
/// squared distance, each step halved until it comes nearer, an ellipse by
/// golden section, each between the samples either side.
struct EdgeReads<'a> {
    curve: &'a EdgeCurve,
    a: Point3,
    b: Point3,
    samples: Vec<(f64, Point3)>,
}

impl<'a> EdgeReads<'a> {
    fn new(topo: &'a Topology, edge_id: EdgeId) -> Result<Self, BlendError> {
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let curve = edge.curve();
        let samples = match curve {
            EdgeCurve::Line | EdgeCurve::Circle(_) => Vec::new(),
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                let (t0, t1) = curve.domain_with_endpoints(a, b);
                (0..=NEAREST_SAMPLES)
                    .map(|k| {
                        #[allow(clippy::cast_precision_loss)]
                        let t = t0 + (t1 - t0) * (k as f64) / (NEAREST_SAMPLES as f64);
                        (t, curve.evaluate_with_endpoints(t, a, b))
                    })
                    .collect()
            }
        };
        Ok(Self {
            curve,
            a,
            b,
            samples,
        })
    }

    /// `point`'s distance from the edge, and the edge's point nearest it.
    fn nearest(&self, point: Point3) -> (f64, Point3) {
        if let Some(on) = nearest_on_line_or_arc(self.curve, self.a, self.b, point) {
            return ((on - point).length(), on);
        }
        let Some((k, &(t, at))) = self.samples.iter().enumerate().min_by(|x, y| {
            (x.1.1 - point)
                .length_squared()
                .total_cmp(&(y.1.1 - point).length_squared())
        }) else {
            return ((self.a - point).length(), self.a);
        };
        let last = self.samples.len() - 1;
        let (lo, hi) = (
            self.samples[k.saturating_sub(1)].0,
            self.samples[(k + 1).min(last)].0,
        );
        let on = match self.curve {
            EdgeCurve::NurbsCurve(nurbs) => {
                newton_nearest(nurbs, point, (t, at), (lo.min(hi), lo.max(hi)))
            }
            curve => {
                let dist =
                    |t: f64| (curve.evaluate_with_endpoints(t, self.a, self.b) - point).length();
                curve.evaluate_with_endpoints(golden_least(lo, hi, dist), self.a, self.b)
            }
        };
        ((on - point).length(), on)
    }
}

/// The point of `curve` between `lo` and `hi` nearest `point`, from its point
/// `at` at `t`: Gauss-Newton steps on the squared distance, each halved until
/// it comes nearer, so each point taken is nearer than the last. The steps
/// are kept where they stop at a point the distance curves up from, or at
/// an end of the span they step out past. Where they stop anywhere else (a
/// peak of the distance, where a step is zero), at a point of no speed, or
/// without settling, the span is searched instead ([`span_nearest`]).
fn newton_nearest(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    point: Point3,
    (mut t, mut at): (f64, Point3),
    (lo, hi): (f64, f64),
) -> Point3 {
    const MOST_STEPS: usize = 64;
    const MOST_HALVINGS: usize = 8;
    let mut gap = (at - point).length_squared();
    for _ in 0..MOST_STEPS {
        let derivatives = curve.derivatives(t, 2);
        let (Some(&speed), Some(&bend)) = (derivatives.get(1), derivatives.get(2)) else {
            break;
        };
        let speed_sq = speed.dot(speed);
        if speed_sq <= 0.0 {
            break;
        }
        let off = at - point;
        let mut step = -speed.dot(off) / speed_sq;
        let mut nearer = false;
        let mut settled = false;
        for _ in 0..MOST_HALVINGS {
            let next = (t + step).clamp(lo, hi);
            if (next - t).abs() <= f64::EPSILON * t.abs().max(hi - lo) {
                // Half the second derivative of the squared distance.
                let curving = speed_sq + bend.dot(off);
                settled = t + step < lo || t + step > hi || curving > 0.0;
                break;
            }
            let on = curve.evaluate(next);
            let next_gap = (on - point).length_squared();
            if next_gap < gap {
                (t, at, gap) = (next, on, next_gap);
                nearer = true;
                break;
            }
            step *= 0.5;
        }
        if settled {
            return at;
        }
        if !nearer {
            break;
        }
    }
    span_nearest(curve, point, (lo, hi), (gap, at))
}

/// The point of `curve` between `lo` and `hi` nearest `point`, by search: the
/// nearest of 17 even points of the span, refined by golden section between
/// its neighbours, or `best` (a squared distance and its point) where that
/// is nearer. Never farther than `best` or the span's ends.
fn span_nearest(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    point: Point3,
    (lo, hi): (f64, f64),
    best: (f64, Point3),
) -> Point3 {
    const POINTS: usize = 16;
    let gap = |t: f64| (curve.evaluate(t) - point).length_squared();
    #[allow(clippy::cast_precision_loss)]
    let param = |k: usize| lo + (hi - lo) * (k as f64) / (POINTS as f64);
    let (k, sampled) = (0..=POINTS)
        .map(|k| (k, gap(param(k))))
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .unwrap_or((0, f64::INFINITY));
    let refined = golden_least(param(k.saturating_sub(1)), param((k + 1).min(POINTS)), gap);
    [(gap(refined), refined), (sampled, param(k))]
        .into_iter()
        .map(|(g, t)| (g, curve.evaluate(t)))
        .chain(std::iter::once(best))
        .min_by(|x, y| x.0.total_cmp(&y.0))
        .map_or(best.1, |(_, on)| on)
}

/// The point of a line or a circular arc from `a` to `b` nearest `point`:
/// on the segment, its foot clamped to the ends; on the arc, the point at
/// `point`'s angle about the axis where the arc spans it, else the end
/// nearer in angle (the distance grows with the turn from that angle).
/// `None` for any other curve.
fn nearest_on_line_or_arc(
    curve: &EdgeCurve,
    a: Point3,
    b: Point3,
    point: Point3,
) -> Option<Point3> {
    match curve {
        EdgeCurve::Line => {
            let along = b - a;
            let length_sq = along.dot(along);
            let s = if length_sq > 0.0 {
                ((point - a).dot(along) / length_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            Some(a + along * s)
        }
        EdgeCurve::Circle(circle) => {
            let (t0, t1) = curve.domain_with_endpoints(a, b);
            let (lo, hi) = (t0.min(t1), t0.max(t1));
            let t = lo + (circle.project(point) - lo).rem_euclid(std::f64::consts::TAU);
            let on = |t: f64| curve.evaluate_with_endpoints(t, a, b);
            Some(if t <= hi {
                on(t)
            } else if (point - on(lo)).length() <= (point - on(hi)).length() {
                on(lo)
            } else {
                on(hi)
            })
        }
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => None,
    }
}

/// Where the line `origin + run * s` passes through an edge: the edge's
/// point nearest the line (its nearest sample, refined by golden section
/// between the sample's neighbours), and the line's `s` there; `None` where
/// the line passes farther than `tol` from the edge.
fn line_meets_edge(
    topo: &Topology,
    edge_id: EdgeId,
    origin: Point3,
    run: Vec3,
    tol: f64,
) -> Result<Option<f64>, BlendError> {
    const SAMPLES: usize = 64;
    let len2 = run.length_squared();
    if len2 <= 1e-24 {
        return Ok(None);
    }
    let edge = topo.edge(edge_id)?;
    let (a, b) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let curve = edge.curve();
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    let off = |t: f64| {
        let d = curve.evaluate_with_endpoints(t, a, b) - origin;
        (d - run * (d.dot(run) / len2)).length()
    };
    #[allow(clippy::cast_precision_loss)]
    let param = |k: usize| t0 + (t1 - t0) * (k as f64) / (SAMPLES as f64);
    let nearest = (0..=SAMPLES)
        .min_by(|&i, &j| off(param(i)).total_cmp(&off(param(j))))
        .unwrap_or(0);
    let (mut lo, mut hi) = (
        param(nearest.saturating_sub(1)),
        param((nearest + 1).min(SAMPLES)),
    );
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..80 {
        let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if off(x1) <= off(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    let t = f64::midpoint(lo, hi);
    if off(t) > tol {
        return Ok(None);
    }
    Ok(Some(
        (curve.evaluate_with_endpoints(t, a, b) - origin).dot(run) / len2,
    ))
}

/// Where the line `origin + run * s` meets `surface`, by Newton steps from
/// `s0` along the surface's normal at its nearest point; `None` where they
/// do not converge to a point on the surface within `tol`.
fn line_meets_surface(
    surface: &FaceSurface,
    origin: Point3,
    run: Vec3,
    s0: f64,
    tol: f64,
) -> Option<f64> {
    let mut s = s0;
    for _ in 0..60 {
        let p = origin + run * s;
        let (u, v) = surface.project_point(p)?;
        let q = surface.evaluate(u, v)?;
        let n = surface.normal(u, v);
        let slope = run.dot(n);
        if slope.abs() <= 1e-12 * run.length() {
            return None;
        }
        let step = (p - q).dot(n) / slope;
        s -= step;
        if (step * run).length() <= 1e-3 * tol {
            let p = origin + run * s;
            let (u, v) = surface.project_point(p)?;
            return ((surface.evaluate(u, v)? - p).length() <= tol).then_some(s);
        }
    }
    None
}

/// The curve where a cylinder band meets a curved cap between the contact
/// ends `a` and `b`, and a cubic through points along it.
///
/// The curve is traced in the band's development, `w` the arc round the
/// axis from `a` and `s` the height along it, by steps of equal length
/// along the curve, each pulled back on to the cap along the gradient of
/// its offset from the cap. Cutting each ruling of the band by the cap
/// instead cannot follow a curve that runs nearly along the rulings (a cap
/// meeting the support face almost tangentially at the corner, where the
/// band's end runs to the corner along the rulings, and a ruling there can
/// cross the cap beyond the face). The steps are halved until the cubic
/// lies on both the band and the cap between its points. `None` where the
/// trace leaves the cap, or the cubic stays off them.
fn cylinder_meets_cap(
    cylinder: &brepkit_math::surfaces::CylindricalSurface,
    cap: &FaceSurface,
    a: Point3,
    b: Point3,
) -> Option<EdgeCurve> {
    const FIRST: usize = 32;
    const MOST: usize = 2048;
    let axis = cylinder.axis();
    let origin = cylinder.origin();
    let radius = cylinder.radius();
    let tol = 1e-7 * radius.max(1.0);
    let radial = |p: Point3| {
        let d = p - origin;
        d - axis * d.dot(axis)
    };
    let (ra, rb) = (radial(a).normalize().ok()?, radial(b).normalize().ok()?);
    let across = axis.cross(ra);
    let angle = rb.dot(across).atan2(rb.dot(ra));
    let (sa, sb) = ((a - origin).dot(axis), (b - origin).dot(axis));
    let target = (radius * angle, sb);
    let at = |w: f64, s: f64| {
        let theta = w / radius;
        origin + (ra * theta.cos() + across * theta.sin()) * radius + axis * s
    };
    // The point's offset from the cap along the cap's normal, and that
    // offset's gradient in (w, s).
    let offset = |w: f64, s: f64| -> Option<(f64, f64, f64)> {
        let p = at(w, s);
        let (u, v) = cap.project_point(p)?;
        let q = cap.evaluate(u, v)?;
        let n = cap.normal(u, v);
        let theta = w / radius;
        let round = across * theta.cos() - ra * theta.sin();
        Some(((p - q).dot(n), n.dot(round), n.dot(axis)))
    };
    // Pulls (w, s) back on to the cap along the offset's gradient.
    let settle = |mut w: f64, mut s: f64| -> Option<(f64, f64)> {
        for _ in 0..30 {
            let (f, gw, gs) = offset(w, s)?;
            if f.abs() <= 1e-3 * tol {
                return Some((w, s));
            }
            let g2 = gw.madd(gw, gs * gs);
            if g2 <= 1e-24 {
                return None;
            }
            w -= f * gw / g2;
            s -= f * gs / g2;
        }
        None
    };
    // How far a point lies off the band or the cap, whichever is farther.
    let off = |p: Point3| -> Option<f64> {
        let (u, v) = cap.project_point(p)?;
        let on_cap = (cap.evaluate(u, v)? - p).length();
        Some(on_cap.max((radial(p).length() - radius).abs()))
    };
    let span = target.0.hypot(target.1 - sa);
    if span <= tol {
        return None;
    }
    let mut steps = FIRST;
    loop {
        #[allow(clippy::cast_precision_loss)]
        let h = span / steps as f64;
        let mut points = vec![a];
        let (mut w, mut s) = (0.0, sa);
        let mut heading: Option<(f64, f64)> = None;
        loop {
            if (target.0 - w).hypot(target.1 - s) <= 1.5 * h {
                points.push(b);
                break;
            }
            if points.len() > 8 * steps {
                return None;
            }
            let (_, gw, gs) = offset(w, s)?;
            let g = gw.hypot(gs);
            if g <= 1e-12 {
                return None;
            }
            let (mut tw, mut ts) = (-gs / g, gw / g);
            let backwards = heading.map_or_else(
                || tw.madd(target.0 - w, ts * (target.1 - s)) < 0.0,
                |(hw, hs)| tw.madd(hw, ts * hs) < 0.0,
            );
            if backwards {
                (tw, ts) = (-tw, -ts);
            }
            heading = Some((tw, ts));
            (w, s) = settle(tw.madd(h, w), ts.madd(h, s))?;
            points.push(at(w, s));
        }
        let curve = brepkit_math::nurbs::fitting::interpolate(&points, 3).ok()?;
        // The cubic takes its points at their chord-length parameters: check
        // it at the quarters between each two.
        let mut lengths = Vec::with_capacity(points.len());
        let mut total = 0.0;
        lengths.push(0.0);
        for pair in points.windows(2) {
            total += (pair[1] - pair[0]).length();
            lengths.push(total);
        }
        if total <= tol {
            return None;
        }
        let (t0, t1) = curve.domain();
        let mut on = true;
        'check: for pair in lengths.windows(2) {
            for quarter in [0.25, 0.5, 0.75] {
                let along = (pair[1] - pair[0]).madd(quarter, pair[0]) / total;
                if off(curve.evaluate((t1 - t0).madd(along, t0)))? > tol {
                    on = false;
                    break 'check;
                }
            }
        }
        if on {
            return Some(EdgeCurve::NurbsCurve(curve));
        }
        if steps >= MOST {
            return None;
        }
        steps *= 2;
    }
}

/// The cap a stripe's end was run on to: a tilted plane (outward normal and
/// offset), or a curved face.
#[derive(Debug, Clone, Copy)]
enum RunOnCap {
    Plane(Vec3, f64),
    Face(FaceId),
}

/// The caps stripe ends were run on to, keyed by the stripe's first spine
/// edge and its end (0 at the spine's start, 1 at its end).
type TiltedCaps = std::collections::HashMap<(EdgeId, usize), RunOnCap>;

/// Two rounds of parallel straight edges of one plane face, together wider
/// than the face (a slab's end rounded at both corners past half its
/// width), cross on it: each one's contact on the face lies past the
/// other's. The round picked first (`pick_order`) is kept; the second rests
/// on its other face and pivots on the first's contact on the shared face,
/// so the face goes and the two bands meet there along the first's contact.
/// Both rounds then have that contact on the face, which is trimmed to the
/// strip between them, with no area, and dropped as a full round's is.
/// Walked rounds crossing on a face (a curved plate rounded on both rims,
/// thinner than twice the radius) pivot the same way
/// ([`pivot_walked_round`]).
fn pivot_crossing_rounds(
    topo: &Topology,
    stripe_results: &mut [StripeResult],
    pick_order: &[EdgeId],
) -> Result<(), BlendError> {
    const PARALLEL: f64 = 1e-12;
    let picked = |stripe: &Stripe| {
        stripe
            .spine_edges()
            .iter()
            .filter_map(|edge| pick_order.iter().position(|picked| picked == edge))
            .min()
            .unwrap_or(usize::MAX)
    };
    let mut moved = vec![false; stripe_results.len()];
    for first in 0..stripe_results.len() {
        for second in first + 1..stripe_results.len() {
            if moved[first] || moved[second] {
                continue;
            }
            // `i` is the round picked first, made as it would be alone.
            let (i, j) =
                if picked(&stripe_results[second].stripe) < picked(&stripe_results[first].stripe) {
                    (second, first)
                } else {
                    (first, second)
                };
            let (a, b) = (&stripe_results[i].stripe, &stripe_results[j].stripe);
            if let (FaceSurface::Nurbs(_), FaceSurface::Nurbs(_)) = (&a.surface, &b.surface) {
                if let Some(pivoted) = pivot_walked_round(topo, a, b)? {
                    log::debug!(
                        "crossing walked rounds: stripe {j} pivots on stripe {i}'s contact"
                    );
                    stripe_results[j].stripe = pivoted;
                    moved[i] = true;
                    moved[j] = true;
                }
                continue;
            }
            let (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) = (&a.surface, &b.surface)
            else {
                continue;
            };
            if a.sections.len() != 2 || b.sections.len() != 2 {
                continue;
            }
            let axis = ca.axis();
            let r = ca.radius();
            if axis.dot(cb.axis()).abs() < 1.0 - PARALLEL || (r - cb.radius()).abs() > 1e-9 * r {
                continue;
            }
            let side = |stripe: &Stripe, face: FaceId| {
                if stripe.face1 == face {
                    Some(1_u8)
                } else if stripe.face2 == face {
                    Some(2)
                } else {
                    None
                }
            };
            let Some((face, sa, sb)) = [a.face1, a.face2]
                .into_iter()
                .find_map(|face| Some((face, side(a, face)?, side(b, face)?)))
            else {
                continue;
            };
            let Some((normal, _)) = outward_plane(topo, face)? else {
                continue;
            };
            let other = if sb == 1 { b.face2 } else { b.face1 };
            let Some((other_normal, _)) = outward_plane(topo, other)? else {
                continue;
            };
            let on = |section: &crate::section::CircSection, side: u8| {
                if side == 1 { section.p1 } else { section.p2 }
            };
            let across = normal.cross(axis);
            let (pa, pb) = (on(&a.sections[0], sa), on(&b.sections[0], sb));
            let (ea, eb) = (a.spine.evaluate(topo, 0.0)?, b.spine.evaluate(topo, 0.0)?);
            // Each contact lies past the other, toward the other's edge.
            if across.dot(pa - pb) * across.dot(eb - ea) <= 0.0 {
                continue;
            }
            // The second round's ball, kept a radius off its other face,
            // moves along it until it passes through the first's contact.
            let along = other_normal.cross(axis).normalize()?;
            let line = on(&a.sections[0], sa);
            let mut sections = b.sections.clone();
            let mut ok = true;
            for section in &mut sections {
                let c = section.center;
                let pivot = line + axis * (c - line).dot(axis);
                let off = c - pivot;
                let half = along.dot(off);
                let disc = half * half - (off.length_squared() - r * r);
                if disc < 0.0 {
                    ok = false;
                    break;
                }
                let root = disc.sqrt();
                let lambda = if (-half + root).abs() <= (-half - root).abs() {
                    -half + root
                } else {
                    -half - root
                };
                let centre = c + along * lambda;
                let touch = centre + other_normal * r;
                if sb == 1 {
                    section.p1 = pivot;
                    section.p2 = touch;
                } else {
                    section.p2 = pivot;
                    section.p1 = touch;
                }
                section.center = centre;
            }
            if !ok {
                continue;
            }
            let line_of = |from: Point3, to: Point3| {
                brepkit_math::nurbs::curve::NurbsCurve::new(
                    1,
                    vec![0.0, 0.0, 1.0, 1.0],
                    vec![from, to],
                    vec![1.0, 1.0],
                )
            };
            let contact1 = line_of(sections[0].p1, sections[1].p1)?;
            let contact2 = line_of(sections[0].p2, sections[1].p2)?;
            let cylinder =
                brepkit_math::surfaces::CylindricalSurface::new(sections[0].center, cb.axis(), r)?;
            log::debug!(
                "crossing rounds on face {face:?}: stripe {j} pivots on stripe {i}'s contact"
            );
            let b = &mut stripe_results[j].stripe;
            b.surface = FaceSurface::Cylinder(cylinder);
            b.contact1 = contact1;
            b.contact2 = contact2;
            b.sections = sections;
            moved[i] = true;
            moved[j] = true;
        }
    }
    Ok(())
}

/// [`pivot_crossing_rounds`] for two walked rounds (fitted bands) on one
/// face: `second`, crossing `first` on it, rebuilt with its ball resting on
/// its other face and pivoting on `first`'s contact there. `None` where they
/// do not cross, or the pivot cannot be found.
fn pivot_walked_round(
    topo: &Topology,
    first: &Stripe,
    second: &Stripe,
) -> Result<Option<Stripe>, BlendError> {
    const SAMPLES: usize = 512;
    let Some(radius) = second.sections.first().map(|section| section.radius) else {
        return Ok(None);
    };
    if first
        .sections
        .iter()
        .chain(&second.sections)
        .any(|section| (section.radius - radius).abs() > 1e-9 * radius)
    {
        return Ok(None);
    }
    let side = |stripe: &Stripe, face: FaceId| {
        if stripe.face1 == face {
            Some(0_usize)
        } else if stripe.face2 == face {
            Some(1)
        } else {
            None
        }
    };
    let Some((face, sa, sb)) = [first.face1, first.face2]
        .into_iter()
        .find_map(|face| Some((face, side(first, face)?, side(second, face)?)))
    else {
        return Ok(None);
    };
    let on_first = if sa == 0 {
        &first.contact1
    } else {
        &first.contact2
    };
    let on_second = if sb == 0 {
        &second.contact1
    } else {
        &second.contact2
    };
    // Crossing: each round's edge nearer the other's contact on the face
    // than its own.
    let distance = |p: Point3, curve: &brepkit_math::nurbs::curve::NurbsCurve| {
        brepkit_math::nurbs::projection::project_point_to_curve(curve, p, 1e-9)
            .map_or(f64::INFINITY, |found| found.distance)
    };
    let middle = |stripe: &Stripe| stripe.spine.evaluate(topo, 0.5 * stripe.spine.length());
    let (ma, mb) = (middle(first)?, middle(second)?);
    // Contacts that only meet (a full round, a wall exactly twice the
    // radius high, its walked contacts a little apart) are left to the
    // full-round check.
    let clear = 1e-3 * radius;
    if distance(ma, on_second) + clear >= distance(ma, on_first)
        || distance(mb, on_first) + clear >= distance(mb, on_second)
    {
        return Ok(None);
    }
    let other = if sb == 0 { second.face2 } else { second.face1 };
    let wall_surface = topo.face(other)?.surface().clone();
    let face_surface = topo.face(face)?.surface().clone();
    let mut adapter = None;
    let wall = surface_ref_or_adapter(&wall_surface, &mut adapter);
    let mut face_adapter = None;
    let face_surface = surface_ref_or_adapter(&face_surface, &mut face_adapter);
    // The first's contact, carried on through the sections its ball rolls
    // on past the spine's ends (where its end is run on to a cap, its
    // contact is the side of the band rebuilt through them).
    let carried: Vec<crate::section::CircSection> = first.beyond[0]
        .iter()
        .rev()
        .chain(&first.sections)
        .chain(&first.beyond[1])
        .cloned()
        .collect();
    let extended = if first.beyond.iter().all(Vec::is_empty) {
        None
    } else {
        interpolate_blend_surface(&carried)
            .ok()
            .map(|band| {
                band_side(
                    &band,
                    if sa == 0 {
                        0
                    } else {
                        band.control_points().len() - 1
                    },
                )
            })
            .transpose()?
    };
    let rim = extended.as_ref().unwrap_or(on_first);
    let (t0, t1) = rim.domain();
    let pieces: Vec<((), f64, f64)> = (0..SAMPLES)
        .map(|k| {
            let f = |k: usize| t0 + (t1 - t0) * k as f64 / SAMPLES as f64;
            ((), f(k), f(k + 1))
        })
        .collect();
    let at = |(): &(), t: f64| -> Result<Point3, BlendError> { Ok(rim.evaluate(t)) };
    // The ball pivoting stays on the walk's side of the first's contact:
    // within a radius of the walk's centre, as it rests on the same face
    // and touches a line on the wall the walk touches (the far root lies
    // past the contact, outside the part).
    let near = |new: &crate::section::CircSection, old: &crate::section::CircSection| {
        (new.center - old.center).length() < radius
    };
    // The sections, then those rolled on past one end, pivoted in turn;
    // the rolled-on ones as far as they still pivot.
    let core = second.sections.len();
    let pivot_on = |run: Vec<crate::section::CircSection>| {
        let pivoted = pivot_sections(
            &pieces,
            &at,
            wall,
            face_surface,
            sb,
            radius,
            &run,
            true,
            core,
        )?;
        Ok::<_, BlendError>(pivoted.and_then(|pivoted| {
            let kept = pivoted
                .iter()
                .zip(&run)
                .take_while(|(new, old)| near(new, old))
                .count();
            (kept >= core).then(|| pivoted[..kept].to_vec())
        }))
    };
    let Some(mut ahead) = pivot_on(
        second
            .sections
            .iter()
            .chain(&second.beyond[1])
            .cloned()
            .collect(),
    )?
    else {
        return Ok(None);
    };
    let Some(mut behind) = pivot_on(
        second
            .sections
            .iter()
            .rev()
            .chain(&second.beyond[0])
            .cloned()
            .collect(),
    )?
    else {
        return Ok(None);
    };
    let beyond = [behind.split_off(core), ahead.split_off(core)];
    let mut sections = ahead;
    if second.spine.is_closed() {
        close_sections(&mut sections);
    }
    let Ok(surface) = interpolate_blend_surface(&sections) else {
        return Ok(None);
    };
    let contact1 = band_side(&surface, 0)?;
    let contact2 = band_side(&surface, surface.control_points().len() - 1)?;
    let surfaces = [
        topo.face(second.face1)?.surface().clone(),
        topo.face(second.face2)?.surface().clone(),
    ];
    let (mut adapter1, mut adapter2) = (None, None);
    let pcurve1 = build_pcurve_from_contact(
        surface_ref_or_adapter(&surfaces[0], &mut adapter1),
        &contact1,
        face_u_period(&surfaces[0]),
    )?;
    let pcurve2 = build_pcurve_from_contact(
        surface_ref_or_adapter(&surfaces[1], &mut adapter2),
        &contact2,
        face_u_period(&surfaces[1]),
    )?;
    let mut pivoted = second.clone();
    pivoted.surface = FaceSurface::Nurbs(surface);
    pivoted.contact1 = contact1;
    pivoted.contact2 = contact2;
    pivoted.pcurve1 = pcurve1;
    pivoted.pcurve2 = pcurve2;
    pivoted.sections = sections;
    pivoted.beyond = beyond;
    Ok(Some(pivoted))
}

/// Runs a straight round's ends on to tilted end caps.
///
/// A round of a straight edge between two planes is a cylinder whose
/// sections stand square to the edge. Where the edge ends on a single planar
/// cap square to it, the end section lies in the cap and the cap takes the
/// notch. Where the cap is tilted, the end section stands off it, the
/// contacts end short of the cap's corner edges, and the end was closed by a
/// runout patch in the section plane instead. Each contact (a straight line
/// on its plane) is run on, or back, to where it meets the cap's plane,
/// which lies on that support face's corner edge with the cap; the band's
/// end is then the cylinder's section by the cap's plane, an ellipse, and
/// the cap takes the notch as a square cap does.
///
/// A curved cap (a round ending on an arc of the part's outline, or on a
/// fitted face) is taken the same way: each contact runs to where it meets
/// the cap's surface, on the cap's corner edge, and the band ends on its
/// intersection with that surface, each of the band's rulings cut by it.
fn run_stripes_to_tilted_caps(
    topo: &Topology,
    plan: &crate::fillet_plan::FilletPlan,
    stripe_results: &mut [StripeResult],
) -> Result<TiltedCaps, BlendError> {
    // Below this the cap is square to the band's axis.
    const SQUARE: f64 = 1e-9;
    // A cap this oblique to the axis is left to the runout.
    const OBLIQUE: f64 = 0.05;
    carry_walked_rounds(topo, plan, stripe_results)?;
    let mut caps = TiltedCaps::new();
    // A terminal whose contact would run back past nearly all of it, or all
    // of it and on, is tried again once the others are done: the far end may
    // run on outward at the contour's other terminal (a cap leaning the same
    // way at both ends), and the contact read from there is long enough.
    let mut order: Vec<usize> = (0..plan.junctions.len()).collect();
    let mut retried = vec![false; plan.junctions.len()];
    let mut next = 0;
    while let Some(&slot) = order.get(next) {
        next += 1;
        let junction = &plan.junctions[slot];
        if !matches!(
            junction.classification,
            crate::fillet_plan::CornerClassification::Terminal
        ) {
            continue;
        }
        let [contour_index] = junction.incident_contours.as_slice() else {
            continue;
        };
        let Some(contour) = plan.contours.get(*contour_index) else {
            continue;
        };
        let Some(index) = stripe_results
            .iter()
            .position(|result| result.stripe.spine_edges() == contour.spine.edges())
        else {
            continue;
        };
        let result = &mut stripe_results[index];
        let stripe = &result.stripe;
        let cylinder = match &stripe.surface {
            FaceSurface::Cylinder(cylinder) => cylinder,
            FaceSurface::Torus(torus) => {
                let torus = torus.clone();
                if let Some(run) = torus_run_on(topo, junction, contour, stripe, &torus)? {
                    let edge = stripe.spine_edges()[0];
                    let result_stripe = &mut result.stripe;
                    result_stripe.contact1 = run.contact1;
                    result_stripe.contact2 = run.contact2;
                    let section = &mut result_stripe.sections[run.end];
                    section.p1 = run.p1;
                    section.p2 = run.p2;
                    section.center = run.centre;
                    log::debug!(
                        "terminal {:?}: torus round runs on to its tilted cap",
                        junction.vertex
                    );
                    caps.insert((edge, run.end), run.cap);
                }
                continue;
            }
            FaceSurface::Nurbs(_) => {
                if let Some(run) = walked_run_back(topo, junction, contour, stripe)? {
                    let edge = stripe.spine_edges()[0];
                    let result_stripe = &mut result.stripe;
                    result_stripe.contact1 = run.contact1;
                    result_stripe.contact2 = run.contact2;
                    // The sections stay as walked (or carried on): a
                    // section's contact points moved to the contacts' new
                    // ends would no longer lie on its arc, and a band rebuilt
                    // through them at its other end would fold. The contacts'
                    // own ends are what the trim and the notch read.
                    log::debug!(
                        "terminal {:?}: walked round cut back to its tilted cap",
                        junction.vertex
                    );
                    caps.insert((edge, run.end), run.cap);
                }
                continue;
            }
            _ => continue,
        };
        if stripe.sections.len() != 2 {
            continue;
        }
        let fan: Vec<FaceId> = junction
            .face_fan
            .iter()
            .copied()
            .filter(|&face| face != contour.side1 && face != contour.side2)
            .collect();
        let [cap] = fan.as_slice() else {
            continue;
        };
        let plane = outward_plane(topo, *cap)?;
        let tilt = plane.map_or(0.0, |(normal, _)| normal.dot(cylinder.axis()).abs());
        if plane.is_some() && (tilt >= 1.0 - SQUARE || tilt <= OBLIQUE) {
            continue;
        }
        let cap_surface = topo.face(*cap)?.surface().clone();
        let vertex = topo.vertex(junction.vertex)?.point();
        let end = usize::from(spine_ends_at(topo, stripe, vertex)?);
        let section = stripe.sections[end].clone();
        // The cap's two edges at the corner: straight on a plane, any curve
        // on a curved cap.
        let corner_edges = cap_corner_edges(topo, *cap, junction.vertex)?;
        if corner_edges.len() != 2
            || (plane.is_some()
                && corner_edges.iter().any(|&edge| {
                    !topo
                        .edge(edge)
                        .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line))
                }))
        {
            continue;
        }
        let scale = (stripe.sections[0].p1 - stripe.sections[0].p2).length();
        let tol = 1e-6 * scale.max(1.0);
        let on_corner = |point: Point3| -> Result<bool, BlendError> {
            for &edge in &corner_edges {
                if point_on_edge(topo, edge, point, tol)? {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        // Each contact run to the cap's plane, kept running from the same
        // end as before; the stripe is left alone unless both reach the
        // cap's corner edges.
        let mut contacts = Vec::with_capacity(2);
        let mut short = false;
        for (contact, at) in [
            (&stripe.contact1, section.p1),
            (&stripe.contact2, section.p2),
        ] {
            let (t0, t1) = contact.domain();
            let (a, b) = (contact.evaluate(t0), contact.evaluate(t1));
            let straight = contact
                .control_points()
                .iter()
                .all(|&p| (p - a).cross(b - a).length() <= 1e-9 * (b - a).length_squared());
            // The end at this end of the spine and the far one.
            let (near, at_end) = contact_end_at(contact, at);
            let far = if at_end { a } else { b };
            let run = near - far;
            if !straight {
                break;
            }
            let fraction = match plane {
                Some((normal, d)) => {
                    let along = normal.dot(run);
                    if along.abs() <= 1e-12 {
                        break;
                    }
                    (d - normal.dot(Vec3::new(far.x(), far.y(), far.z()))) / along
                }
                None => match line_meets_surface(&cap_surface, far, run, 1.0, tol) {
                    Some(fraction) => fraction,
                    None => break,
                },
            };
            let mut reached = far + run * fraction;
            if plane.is_none() && !on_corner(reached)? {
                // A curved cap may cross the contact's line more than once
                // (an extruded profile curling back across it beyond the
                // face): the crossing the run-on wants is the one on the
                // cap's corner edge with the contact's face, the nearest to
                // the end section.
                let mut best: Option<f64> = None;
                for &edge in &corner_edges {
                    if let Some(s) = line_meets_edge(topo, edge, far, run, tol)?
                        && s > 0.05
                        && best.is_none_or(|b| (s - 1.0).abs() < (b - 1.0).abs())
                    {
                        best = Some(s);
                    }
                }
                if let Some(s) = best {
                    reached = far + run * s;
                }
            }
            if fraction <= 0.05 && !retried[slot] && on_corner(reached)? {
                short = true;
            }
            if fraction <= 0.05 || !on_corner(reached)? {
                break;
            }
            let points = if at_end {
                vec![far, reached]
            } else {
                vec![reached, far]
            };
            contacts.push((
                brepkit_math::nurbs::curve::NurbsCurve::new(
                    1,
                    vec![t0, t0, t1, t1],
                    points,
                    vec![1.0, 1.0],
                )?,
                reached,
            ));
        }
        let Ok([(contact1, p1), (contact2, p2)]) = <[_; 2]>::try_from(contacts) else {
            if short {
                retried[slot] = true;
                order.push(slot);
            }
            continue;
        };
        // The section's centre: where the axis meets the cap's plane (the
        // ellipse's centre), or, on a curved cap, the axis's point level with
        // the two contact ends' middle.
        let axis = cylinder.axis();
        let origin = cylinder.origin();
        let to = |p: Point3| Vec3::new(p.x(), p.y(), p.z());
        let centre = match plane {
            Some((normal, d)) => origin + axis * ((d - normal.dot(to(origin))) / normal.dot(axis)),
            None => origin + axis * (p1 - origin + (p2 - origin)).dot(axis) * 0.5,
        };
        let run_on = match plane {
            Some((normal, d)) => RunOnCap::Plane(normal, d),
            None => RunOnCap::Face(*cap),
        };
        let edge = stripe.spine_edges()[0];
        let result_stripe = &mut result.stripe;
        result_stripe.contact1 = contact1;
        result_stripe.contact2 = contact2;
        let section = &mut result_stripe.sections[end];
        section.p1 = p1;
        section.p2 = p2;
        section.center = centre;
        log::debug!(
            "terminal {:?}: round runs on to its cap {:?} ({run_on:?})",
            junction.vertex,
            cap
        );
        caps.insert((edge, end), run_on);
    }
    Ok(caps)
}

/// How far `point` stands outside a run-on cap, along the cap's outward
/// normal: the plane's offset, or on a curved cap the point's offset from
/// its nearest point on the cap's surface (within `tol` of it). `None` where
/// the point does not project on to the cap.
fn cap_offset(topo: &Topology, cap: RunOnCap, point: Point3, tol: f64) -> Option<f64> {
    match cap {
        RunOnCap::Plane(normal, d) => {
            Some(normal.dot(Vec3::new(point.x(), point.y(), point.z())) - d)
        }
        RunOnCap::Face(face) => {
            let surface = topo.face(face).ok()?.surface();
            let (u, v) = surface.project_point(point)?;
            let on = surface.evaluate(u, v)?;
            let normal = outward_normal_at(topo, face, on, tol).ok()??;
            Some((point - on).dot(normal))
        }
    }
}

/// A walked round's single cap at a terminal: the cap, its two corner edges,
/// the run-on cap, which end of the spine meets it and its end section.
struct WalkedEnd {
    corner_edges: Vec<EdgeId>,
    run_on: RunOnCap,
    end: usize,
    section: crate::section::CircSection,
    tol: f64,
    /// The sign of the cap's offset past it, away from the round: `1.0` for
    /// an end face the round runs out through, `-1.0` for a floor it stands
    /// on (the walls meeting the cap in concave edges), whose outward side
    /// is the round's own.
    past: f64,
    /// Where the round swallowed a narrow side face: which contact (0 or 1)
    /// is pivoted on the face's far edge, and that edge's corner on the cap,
    /// where that contact ends.
    pivot: Option<(usize, Point3)>,
}

impl WalkedEnd {
    /// How far `point` lies past the cap, away from the round.
    fn past_cap(&self, topo: &Topology, point: Point3) -> Option<f64> {
        cap_offset(topo, self.run_on, point, self.tol).map(|offset| self.past * offset)
    }
}

/// The cap a walked round's end meets at `junction`, where there is exactly
/// one, with its two corner edges.
fn walked_end(
    topo: &Topology,
    junction: &crate::fillet_plan::VertexJunction,
    contour: &crate::fillet_plan::FilletContour,
    stripe: &Stripe,
) -> Result<Option<WalkedEnd>, BlendError> {
    let fan: Vec<FaceId> = junction
        .face_fan
        .iter()
        .copied()
        .filter(|&face| face != contour.side1 && face != contour.side2)
        .collect();
    let [cap] = fan.as_slice() else {
        return Ok(None);
    };
    let mut corner_edges = cap_corner_edges(topo, *cap, junction.vertex)?;
    if corner_edges.len() != 2 {
        return Ok(None);
    }
    let mut pivot = None;
    // A round that swallowed a narrow side face (`swallow_narrow_face`)
    // rests its pivoted contact on the face beyond, so that contact ends on
    // the cap's edges at the swallowed face's far corner: the other end of
    // the cap edge the swallowed face shares with the cap.
    for side in [contour.side1, contour.side2] {
        if side == stripe.face1 || side == stripe.face2 {
            continue;
        }
        let Some(&shared) = corner_edges
            .iter()
            .find(|&&edge| topo.faces_of_edge(edge).contains(&side))
        else {
            continue;
        };
        let edge = topo.edge(shared)?;
        let far = if edge.start() == junction.vertex {
            edge.end()
        } else {
            edge.start()
        };
        for extra in cap_corner_edges(topo, *cap, far)? {
            if !corner_edges.contains(&extra) {
                corner_edges.push(extra);
            }
        }
        let pivoted = usize::from(stripe.face1 == contour.side1 || stripe.face1 == contour.side2);
        pivot = Some((pivoted, topo.vertex(far)?.point()));
    }
    let run_on = outward_plane(topo, *cap)?.map_or(RunOnCap::Face(*cap), |(normal, d)| {
        RunOnCap::Plane(normal, d)
    });
    let vertex = topo.vertex(junction.vertex)?.point();
    let end = usize::from(spine_ends_at(topo, stripe, vertex)?);
    let Some(section) = end_section(topo, stripe, vertex)? else {
        return Ok(None);
    };
    let tol = 1e-6 * (section.p1 - section.p2).length().max(1.0);
    // The round lies on the cap's inner side: read where the ball's centre
    // moves from the end section to the next one in (the far end of a
    // spine curving round may lie anywhere about the cap's plane).
    let count = stripe.sections.len();
    if count < 2 {
        return Ok(None);
    }
    let next = &stripe.sections[if end == 0 { 1 } else { count - 2 }];
    let (Some(at_end), Some(inward)) = (
        cap_offset(topo, run_on, section.center, tol),
        cap_offset(topo, run_on, next.center, tol),
    ) else {
        return Ok(None);
    };
    let towards_round = inward - at_end;
    if towards_round.abs() <= tol * 1e-3 {
        return Ok(None);
    }
    Ok(Some(WalkedEnd {
        corner_edges,
        run_on,
        end,
        section: section.clone(),
        tol,
        past: -towards_round.signum(),
        pivot,
    }))
}

/// A contact's `(near, far, near is its end)` parameters, `near` being its
/// end at `at`.
fn contact_ends(contact: &brepkit_math::nurbs::curve::NurbsCurve, at: Point3) -> (f64, f64, bool) {
    let (t0, t1) = contact.domain();
    let (_, at_end) = contact_end_at(contact, at);
    if at_end {
        (t1, t0, true)
    } else {
        (t0, t1, false)
    }
}

/// The sections a walked round is carried on through past its end at
/// `junction`, outwards, where a contact stops short of the end's single
/// tilted or curved cap: the sections the ball rolls through past the
/// spine's end ([`Stripe::beyond`]), until both contacts are past the cap,
/// and a couple more. `None` where no contact falls short, or the ball's
/// sections never take it past.
fn walked_carry(
    topo: &Topology,
    junction: &crate::fillet_plan::VertexJunction,
    contour: &crate::fillet_plan::FilletContour,
    stripe: &Stripe,
) -> Result<Option<(usize, Vec<crate::section::CircSection>)>, BlendError> {
    // Sections carried past the first one with both contacts past the cap.
    const MARGIN: usize = 2;
    let FaceSurface::Nurbs(_) = &stripe.surface else {
        return Ok(None);
    };
    let Some(walked) = walked_end(topo, junction, contour, stripe)? else {
        return Ok(None);
    };
    let tol = walked.tol;
    let offset = |point: Point3| walked.past_cap(topo, point);
    let past = &stripe.beyond[walked.end];
    let mut needed = None;
    for (side, contact, at) in [
        (0, &stripe.contact1, walked.section.p1),
        (1, &stripe.contact2, walked.section.p2),
    ] {
        let (near, _, _) = contact_ends(contact, at);
        let Some(beyond) = offset(contact.evaluate(near)) else {
            return Ok(None);
        };
        if beyond >= -tol {
            continue;
        }
        let reached = past.iter().position(|section| {
            let point = if side == 0 { section.p1 } else { section.p2 };
            offset(point).is_some_and(|off| off > tol)
        });
        let Some(index) = reached else {
            return Ok(None);
        };
        needed = Some(needed.map_or(index + 1, |most: usize| most.max(index + 1)));
    }
    Ok(needed.map(|needed| {
        (
            walked.end,
            past[..(needed + MARGIN).min(past.len())].to_vec(),
        )
    }))
}

/// Carries walked rounds on past their ends where a contact stops short of
/// a tilted or curved cap ([`walked_carry`]), every end before any is cut
/// back, so a band rebuilt for one end is never one already cut at its
/// other. The band is rebuilt through its sections and the carried ones,
/// passing through each ([`interpolate_blend_surface`]): the end section,
/// now inside the band, keeps its contact points on it, the one on the
/// cap's corner edge among them. Its contacts are the rebuilt band's sides.
fn carry_walked_rounds(
    topo: &Topology,
    plan: &crate::fillet_plan::FilletPlan,
    stripe_results: &mut [StripeResult],
) -> Result<(), BlendError> {
    let mut carries: Vec<(
        usize,
        usize,
        Vec<crate::section::CircSection>,
        &crate::fillet_plan::VertexJunction,
        &crate::fillet_plan::FilletContour,
    )> = Vec::new();
    for junction in &plan.junctions {
        if !matches!(
            junction.classification,
            crate::fillet_plan::CornerClassification::Terminal
        ) {
            continue;
        }
        let [contour_index] = junction.incident_contours.as_slice() else {
            continue;
        };
        let Some(contour) = plan.contours.get(*contour_index) else {
            continue;
        };
        let Some(index) = stripe_results
            .iter()
            .position(|result| result.stripe.spine_edges() == contour.spine.edges())
        else {
            continue;
        };
        if let Some((end, sections)) =
            walked_carry(topo, junction, contour, &stripe_results[index].stripe)?
        {
            carries.push((index, end, sections, junction, contour));
        }
    }
    for index in 0..stripe_results.len() {
        let mut ends: [Option<&Vec<crate::section::CircSection>>; 2] = [None, None];
        for (at, end, sections, _, _) in &carries {
            if *at == index {
                ends[*end] = Some(sections);
            }
        }
        if ends.iter().all(Option::is_none) {
            continue;
        }
        let stripe = &stripe_results[index].stripe;
        let mut sections = Vec::with_capacity(stripe.sections.len() + 16);
        if let Some(carried) = ends[0] {
            sections.extend(carried.iter().rev().cloned());
        }
        sections.extend(stripe.sections.iter().cloned());
        if let Some(carried) = ends[1] {
            sections.extend(carried.iter().cloned());
        }
        let Ok(surface) = interpolate_blend_surface(&sections) else {
            continue;
        };
        let mut carried = stripe.clone();
        carried.contact1 = band_side(&surface, 0)?;
        carried.contact2 = band_side(&surface, surface.control_points().len() - 1)?;
        carried.sections = sections;
        carried.surface = FaceSurface::Nurbs(surface);
        // Kept only where every end carried on then cuts back to its cap: a
        // contact left running past its face could not be trimmed.
        let mut cuts = true;
        for (at, _, _, junction, contour) in &carries {
            if *at == index && walked_run_back(topo, junction, contour, &carried)?.is_none() {
                cuts = false;
            }
        }
        if !cuts {
            continue;
        }
        log::debug!(
            "walked round carried on past {} end(s)",
            ends.iter().filter(|end| end.is_some()).count()
        );
        stripe_results[index].stripe = carried;
    }
    Ok(())
}

/// Cuts a walked round's end back to a tilted or curved cap.
///
/// A walked round (a fitted band between curved faces) ends in the plane
/// square to its spine. Where its edge ends on a single cap turned to that
/// plane, a contact runs on past the cap (the corner was closed by a runout
/// patch, or the trim found the contact's end on no edge it could split);
/// one stopping short has been carried on past it ([`carry_walked_rounds`]).
/// Each contact running past the cap is cut where it crosses it, which lies
/// on the cap's corner edge with that contact's face; a contact ending on
/// the cap keeps its end. The band then ends on its intersection with the
/// cap, between those two points, which the cap takes as its notch.
fn walked_run_back(
    topo: &Topology,
    junction: &crate::fillet_plan::VertexJunction,
    contour: &crate::fillet_plan::FilletContour,
    stripe: &Stripe,
) -> Result<Option<TorusRunOn>, BlendError> {
    const SAMPLES: usize = 400;
    let Some(walked) = walked_end(topo, junction, contour, stripe)? else {
        return Ok(None);
    };
    let offset = |point: Point3| walked.past_cap(topo, point);
    let WalkedEnd {
        ref corner_edges,
        run_on,
        end,
        ref section,
        tol,
        pivot,
        ..
    } = walked;
    // A contact on a side face ends on the cap's corner edge with that face,
    // within as far as the edge stands off the face (a B-spline skin's edge
    // stands off it, a file's tolerance) and the walk's own tolerance. A
    // round that swallowed a narrow face has its sections pivoted and its
    // contact on the wall refitted through them, within that fit too. A
    // contact missing its corner edge by more is left to the runout.
    let sides = [contour.side1, contour.side2]
        .map(|side| topo.face(side).map(|f| f.surface().clone()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mut slack: f64 = 0.0;
    for &edge_id in corner_edges {
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let off = sides
            .iter()
            .map(|surface| edge_off_surface(edge.curve(), a, b, surface))
            .fold(f64::INFINITY, f64::min);
        if off.is_finite() {
            slack = slack.max(off);
        }
    }
    let corner_tol = if pivot.is_some() {
        tol.max(1e-5 + slack)
    } else {
        tol.max(tol + slack)
    };
    let mut contacts = Vec::with_capacity(2);
    let mut cut = false;
    for (index, (contact, at)) in [
        (&stripe.contact1, section.p1),
        (&stripe.contact2, section.p2),
    ]
    .into_iter()
    .enumerate()
    {
        let (near, far, at_end) = contact_ends(contact, at);
        // The contact pivoted on a swallowed face's far edge runs along that
        // edge, so it ends where the edge does, at its corner on the cap.
        if let Some((_, corner)) = pivot.filter(|&(pivoted, _)| pivoted == index) {
            let Some(piece) =
                pivoted_contact_to_corner(contact, near, far, at_end, corner, section.radius)?
            else {
                log::debug!("walked_run_back: pivoted contact misses its corner");
                return Ok(None);
            };
            cut |= (contact.evaluate(near) - corner).length() > tol;
            contacts.push((piece, corner));
            continue;
        }
        let Some(beyond) = offset(contact.evaluate(near)) else {
            return Ok(None);
        };
        if beyond < -tol {
            return Ok(None);
        }
        let at_near = beyond <= tol;
        let crossing = if at_near {
            near
        } else {
            // The first crossing back from the end: a scan, then bisection
            // between the last sample past the cap and the first inside it.
            #[allow(clippy::cast_precision_loss)]
            let at_sample = |k: usize| near + (far - near) * (k as f64) / (SAMPLES as f64);
            let mut bracket = None;
            for k in 1..=SAMPLES {
                match offset(contact.evaluate(at_sample(k))) {
                    Some(off) if off <= 0.0 => {
                        bracket = Some((at_sample(k - 1), at_sample(k)));
                        break;
                    }
                    Some(_) => {}
                    None => return Ok(None),
                }
            }
            let Some((mut outside, mut inside)) = bracket else {
                return Ok(None);
            };
            for _ in 0..100 {
                let middle = f64::midpoint(outside, inside);
                match offset(contact.evaluate(middle)) {
                    Some(off) if off > 0.0 => outside = middle,
                    Some(_) => inside = middle,
                    None => return Ok(None),
                }
            }
            cut = true;
            f64::midpoint(outside, inside)
        };
        let reached = contact.evaluate(crossing);
        // The contact ends on the corner edge it reaches: within the
        // tolerance it may miss the edge by, it is ended on the edge itself,
        // where the trim splits the edge.
        let mut corner = None;
        for &edge in corner_edges {
            let (distance, on) = edge_nearest(topo, edge, reached, None)?;
            if distance <= corner_tol && corner.is_none_or(|(best, _)| distance < best) {
                corner = Some((distance, on));
            }
        }
        let Some((_, on)) = corner else {
            return Ok(None);
        };
        let piece = if at_near {
            contact.clone()
        } else {
            let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(contact, crossing)?;
            if at_end { left } else { right }
        };
        contacts.push((pin_contact_end(&piece, at_end, on)?, on));
    }
    if !cut {
        return Ok(None);
    }
    let Ok([(contact1, p1), (contact2, p2)]) = <[_; 2]>::try_from(contacts) else {
        return Ok(None);
    };
    Ok(Some(TorusRunOn {
        end,
        contact1,
        p1,
        contact2,
        p2,
        centre: section.center,
        cap: run_on,
    }))
}

/// A contact pivoted on a swallowed face's far edge, cut where it passes the
/// edge's `corner` on the cap and ended exactly there: `near` is its end at
/// the cap and `far` its other end (`at_end` when `near` is the curve's own
/// end). `None` where it passes farther from the corner than its fit could
/// stray (a thousandth of the round's radius).
fn pivoted_contact_to_corner(
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
    near: f64,
    far: f64,
    at_end: bool,
    corner: Point3,
    radius: f64,
) -> Result<Option<brepkit_math::nurbs::curve::NurbsCurve>, BlendError> {
    let reach = 1e-3 * radius.abs().max(1e-9);
    // The corner may lie past the contact's end (the contact carried on
    // stopped short of it by its fit) or inside it.
    let found = brepkit_math::nurbs::projection::project_point_to_curve(contact, corner, 1e-12)?;
    if found.distance > reach {
        return Ok(None);
    }
    let (low, high) = (near.min(far), near.max(far));
    let inside =
        found.parameter > low + 1e-9 * (high - low) && found.parameter < high - 1e-9 * (high - low);
    let piece = if inside {
        let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(contact, found.parameter)?;
        if at_end { left } else { right }
    } else {
        contact.clone()
    };
    Ok(Some(pin_contact_end(&piece, at_end, corner)?))
}

/// `contact` with its end at the cap (its last end with `at_end`, else its
/// first) moved on to `point`, the end being a control point of the curve.
fn pin_contact_end(
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
    at_end: bool,
    point: Point3,
) -> Result<brepkit_math::nurbs::curve::NurbsCurve, BlendError> {
    let mut points = contact.control_points().to_vec();
    let slot = if at_end { points.len() - 1 } else { 0 };
    points[slot] = point;
    Ok(brepkit_math::nurbs::curve::NurbsCurve::new(
        contact.degree(),
        contact.knots().to_vec(),
        points,
        contact.weights().to_vec(),
    )?)
}

/// The curve where a walked band meets its cap between the contact ends `a`
/// and `b`: across the band from `a` to `b`, each row of the band (fixed
/// `u`) followed along the spine (`v`, by secant steps from the straight
/// guess between the ends) to where `offset`, the point's offset outside the
/// cap, is zero, and a cubic through those points. `None` where a row
/// misses the cap.
fn band_meets_cap(
    band: &FaceSurface,
    a: Point3,
    b: Point3,
    offset: impl Fn(Point3) -> Option<f64>,
) -> Option<EdgeCurve> {
    const SAMPLES: usize = 32;
    let (ua, va) = band.project_point(a)?;
    let (ub, vb) = band.project_point(b)?;
    let tol = 1e-9 * (a - b).length().max(1.0);
    let row = |u: f64, guess: f64| -> Option<Point3> {
        let at = |v: f64| {
            band.evaluate(u, v)
                .and_then(|p| offset(p).map(|off| (p, off)))
        };
        let (mut v0, mut v1) = (guess, guess + 1e-4);
        let (_, mut f0) = at(v0)?;
        let (mut p1, mut f1) = at(v1)?;
        for _ in 0..60 {
            if f1.abs() <= tol {
                return Some(p1);
            }
            if (f1 - f0).abs() <= f64::EPSILON * f1.abs() {
                return None;
            }
            let v2 = v1 - f1 * (v1 - v0) / (f1 - f0);
            (v0, f0) = (v1, f1);
            v1 = v2;
            (p1, f1) = at(v1)?;
        }
        None
    };
    let mut points = Vec::with_capacity(SAMPLES + 1);
    points.push(a);
    for k in 1..SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let f = k as f64 / SAMPLES as f64;
        points.push(row(ua + (ub - ua) * f, va + (vb - va) * f)?);
    }
    points.push(b);
    brepkit_math::nurbs::fitting::interpolate(&points, 3)
        .ok()
        .map(EdgeCurve::NurbsCurve)
}

/// A torus round's end run on to a tilted or curved cap.
struct TorusRunOn {
    /// Which end of the stripe (0 at the spine's start, 1 at its end).
    end: usize,
    contact1: brepkit_math::nurbs::curve::NurbsCurve,
    p1: Point3,
    contact2: brepkit_math::nurbs::curve::NurbsCurve,
    p2: Point3,
    centre: Point3,
    cap: RunOnCap,
}

/// Where the parallel of `torus` at tube angle `v` meets the plane
/// `normal . p = d`: the angle round the axis nearest `near`, or `None`
/// where it misses the plane.
fn torus_parallel_meets_plane(
    torus: &brepkit_math::surfaces::ToroidalSurface,
    normal: Vec3,
    d: f64,
    v: f64,
    near: f64,
) -> Option<f64> {
    let rho = torus.minor_radius().madd(v.cos(), torus.major_radius());
    let centre = torus.center();
    let a = rho * normal.dot(torus.x_axis());
    let b = rho * normal.dot(torus.y_axis());
    let k = d
        - normal.dot(Vec3::new(centre.x(), centre.y(), centre.z()))
        - torus.minor_radius() * v.sin() * normal.dot(torus.z_axis());
    let h = a.hypot(b);
    if h <= 1e-12 * rho.abs().max(1.0) || (k / h).abs() > 1.0 {
        return None;
    }
    let (phase, spread) = (b.atan2(a), (k / h).acos());
    let tau = std::f64::consts::TAU;
    [phase + spread, phase - spread]
        .into_iter()
        .map(|u| u + tau * ((near - u) / tau).round())
        .min_by(|p, q| (p - near).abs().total_cmp(&(q - near).abs()))
}

/// An angle wrapped into `(-pi, pi]`.
fn wrapped(angle: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    angle - tau * (angle / tau).round()
}

/// Where the parallel of `torus` at tube angle `v` meets a curved cap's
/// surface: Newton round the axis from `near`, each step the point's offset
/// along the surface normal over the parallel's slope across it. `None`
/// where it runs along the surface or misses it.
fn torus_parallel_meets_surface(
    torus: &brepkit_math::surfaces::ToroidalSurface,
    surface: &FaceSurface,
    v: f64,
    near: f64,
    tol: f64,
) -> Option<f64> {
    let rho = torus.minor_radius().madd(v.cos(), torus.major_radius());
    if rho <= 0.0 {
        return None;
    }
    let mut u = near;
    for _ in 0..60 {
        let p = torus.evaluate(u, v);
        let (su, sv) = surface.project_point(p)?;
        let q = surface.evaluate(su, sv)?;
        let n = surface.normal(su, sv).normalize().ok()?;
        let along = (torus.y_axis() * u.cos() - torus.x_axis() * u.sin()) * rho;
        let slope = along.dot(n);
        if slope.abs() <= 1e-12 * rho {
            return None;
        }
        let step = (p - q).dot(n) / slope;
        u -= step;
        if (step * rho).abs() <= 1e-3 * tol {
            let p = torus.evaluate(u, v);
            let (su, sv) = surface.project_point(p)?;
            return ((surface.evaluate(su, sv)? - p).length() <= tol).then_some(u);
        }
    }
    None
}

/// Where the parallel of `torus` at tube angle `v` passes closest to
/// `edge`'s trace on `support` (the face the parallel lies on) within a
/// twentieth of a radian of `near` round the axis: the angle and the gap,
/// by a scan and golden-section refinement. `None` where the nearest pass
/// is at the window's end (the parallel does not cross the edge there).
fn parallel_meets_edge(
    topo: &Topology,
    torus: &brepkit_math::surfaces::ToroidalSurface,
    v: f64,
    (edge, support): (EdgeId, &FaceSurface),
    near: f64,
) -> Result<Option<(f64, f64)>, BlendError> {
    const WINDOW: f64 = 0.05;
    const SAMPLES: usize = 40;
    let gap = |u: f64| edge_distance(topo, edge, torus.evaluate(u, v), Some(support));
    #[allow(clippy::cast_precision_loss)]
    let at = |k: usize| near - WINDOW + 2.0 * WINDOW * (k as f64) / (SAMPLES as f64);
    let mut best = (0, f64::MAX);
    for k in 0..=SAMPLES {
        let g = gap(at(k))?;
        if g < best.1 {
            best = (k, g);
        }
    }
    if best.0 == 0 || best.0 == SAMPLES {
        return Ok(None);
    }
    let (mut lo, mut hi) = (at(best.0 - 1), at(best.0 + 1));
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..80 {
        let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if gap(x1)? <= gap(x2)? {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    let u = f64::midpoint(lo, hi);
    Ok(Some((u, gap(u)?)))
}

/// Runs a torus round's end on to a tilted planar cap, or a curved one.
///
/// A round of an arc between a plane and a coaxial wall (or of a circle on
/// a revolved part) is a torus whose sections are its tube circles, square
/// to the arc. Where the arc ends on a planar cap square to it the end
/// section lies in the cap; where the cap is tilted the section stands off
/// it, and the end was closed by a runout patch. Each contact, a parallel
/// of the torus, is run on (or back) round the axis to where it meets the
/// cap's plane, which lies on that support face's corner edge with the
/// cap, and the band ends on its section by the cap's plane: each parallel
/// between the two contacts cut by it. A curved cap (a fitted face the arc
/// ends on) is taken the same way: each contact runs to where its parallel
/// meets the cap's surface, on the cap's corner edges, and the band ends on
/// its intersection with that surface.
fn torus_run_on(
    topo: &Topology,
    junction: &crate::fillet_plan::VertexJunction,
    contour: &crate::fillet_plan::FilletContour,
    stripe: &Stripe,
    torus: &brepkit_math::surfaces::ToroidalSurface,
) -> Result<Option<TorusRunOn>, BlendError> {
    // Below this the cap is square to the round's end.
    const SQUARE: f64 = 1e-9;
    // A cap this oblique to the end is left to the runout.
    const OBLIQUE: f64 = 0.05;
    const SAMPLES: usize = 16;
    if stripe.sections.len() != 2 {
        return Ok(None);
    }
    let fan: Vec<FaceId> = junction
        .face_fan
        .iter()
        .copied()
        .filter(|&face| face != contour.side1 && face != contour.side2)
        .collect();
    let [cap] = fan.as_slice() else {
        return Ok(None);
    };
    let plane = outward_plane(topo, *cap)?;
    let cap_surface = topo.face(*cap)?.surface();
    // The cap's two edges at the corner: straight on a plane, any curve on
    // a curved cap.
    let corner_edges = cap_corner_edges(topo, *cap, junction.vertex)?;
    if corner_edges.len() != 2
        || (plane.is_some()
            && corner_edges.iter().any(|&edge| {
                !topo
                    .edge(edge)
                    .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line))
            }))
    {
        return Ok(None);
    }
    let vertex = topo.vertex(junction.vertex)?.point();
    let end = usize::from(spine_ends_at(topo, stripe, vertex)?);
    let section = &stripe.sections[end];
    if let Some((normal, _)) = plane {
        let (u_end, _) = torus.project_point(section.center);
        let tangent = torus.y_axis() * u_end.cos() - torus.x_axis() * u_end.sin();
        let tilt = normal.dot(tangent).abs();
        if tilt >= 1.0 - SQUARE || tilt <= OBLIQUE {
            return Ok(None);
        }
    }
    let tol = torus_cap_tol(torus, section.p1, section.p2);
    let mut contacts = Vec::with_capacity(2);
    for (contact, at, support) in [
        (&stripe.contact1, section.p1, stripe.face1),
        (&stripe.contact2, section.p2, stripe.face2),
    ] {
        let support = topo.face(support)?.surface();
        let (t0, t1) = contact.domain();
        let (_, at_end) = contact_end_at(contact, at);
        let (from, to) = if at_end { (t0, t1) } else { (t1, t0) };
        let far = contact.evaluate(from);
        let (u_far, v) = torus.project_point(far);
        // The contact is the parallel at `v`: its sweep round the axis from
        // its far end to its near one, read over its samples.
        let mut sweep = 0.0;
        let mut previous = u_far;
        let mut parallel = true;
        for k in 1..=SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let t = from + (to - from) * (k as f64) / (SAMPLES as f64);
            let point = contact.evaluate(t);
            let (u, v_k) = torus.project_point(point);
            if wrapped(v_k - v).abs() > 1e-9 || (torus.evaluate(u, v) - point).length() > tol {
                parallel = false;
                break;
            }
            sweep += wrapped(u - previous);
            previous = u;
        }
        if !parallel {
            break;
        }
        let u_new = match plane {
            Some((normal, d)) => torus_parallel_meets_plane(torus, normal, d, v, u_far + sweep),
            None => torus_parallel_meets_surface(torus, cap_surface, v, u_far + sweep, tol),
        };
        let Some(u_new) = u_new else {
            break;
        };
        // A fitted cap need not hold its corner edges to more than a file's
        // tolerance, nor a fitted edge its support face (one stood 8e-5 off
        // its plane on a 500 mm part), so the contact lands where its
        // parallel crosses a corner edge's trace on the contact's support
        // face, near where it meets the cap's surface; the trim then finds
        // it on that edge.
        let mut landing = None;
        for &edge in &corner_edges {
            if let Some((u, gap)) = parallel_meets_edge(topo, torus, v, (edge, support), u_new)?
                && landing.is_none_or(|(_, best)| gap < best)
            {
                landing = Some((u, gap));
            }
        }
        let u_new = match landing {
            Some((u, gap)) if gap <= tol => u,
            _ => break,
        };
        let new_sweep = u_new - u_far;
        if new_sweep * sweep <= 0.0 || new_sweep.abs() <= 0.05 * sweep.abs() {
            break;
        }
        let reached = torus.evaluate(u_new, v);
        let rho = torus.minor_radius().madd(v.cos(), torus.major_radius());
        if rho <= 0.0 {
            break;
        }
        let circle = Circle3D::with_axes(
            torus.center() + torus.z_axis() * (torus.minor_radius() * v.sin()),
            torus.z_axis(),
            rho,
            torus.x_axis(),
            torus.y_axis(),
        )?;
        let curve = if at_end {
            crate::analytic::circle_arc_to_nurbs(&circle, u_far, u_new)?
        } else {
            crate::analytic::circle_arc_to_nurbs(&circle, u_new, u_far)?
        };
        contacts.push((curve, reached, u_new));
    }
    let Ok([(contact1, p1, u1), (contact2, p2, u2)]) = <[_; 2]>::try_from(contacts) else {
        return Ok(None);
    };
    // The section's centre: the tube's centre midway round between the two
    // contact ends.
    let u_mid = u1 + 0.5 * wrapped(u2 - u1);
    let centre = torus.center()
        + (torus.x_axis() * u_mid.cos() + torus.y_axis() * u_mid.sin()) * torus.major_radius();
    Ok(Some(TorusRunOn {
        end,
        contact1,
        p1,
        contact2,
        p2,
        centre,
        cap: plane.map_or(RunOnCap::Face(*cap), |(normal, d)| {
            RunOnCap::Plane(normal, d)
        }),
    }))
}

/// The curve where a torus band meets its cap between the contact ends `a`
/// and `b`: each parallel of the band, the shorter way round its tube from
/// `a` to `b`, cut by the cap (`hit` takes a parallel's tube angle and a
/// guess round the axis to where it meets the cap), and a cubic through
/// those points. `None` where a parallel misses the cap.
fn torus_meets_cap(
    torus: &brepkit_math::surfaces::ToroidalSurface,
    a: Point3,
    b: Point3,
    hit: impl Fn(f64, f64) -> Option<f64>,
) -> Option<EdgeCurve> {
    const SAMPLES: usize = 32;
    let (ua, va) = torus.project_point(a);
    let (ub, vb) = torus.project_point(b);
    let (du, dv) = (wrapped(ub - ua), wrapped(vb - va));
    let mut points = Vec::with_capacity(SAMPLES + 1);
    points.push(a);
    for k in 1..SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let f = k as f64 / SAMPLES as f64;
        let v = va + dv * f;
        let u = hit(v, ua + du * f)?;
        points.push(torus.evaluate(u, v));
    }
    points.push(b);
    brepkit_math::nurbs::fitting::interpolate(&points, 3)
        .ok()
        .map(EdgeCurve::NurbsCurve)
}

/// The tolerance a torus round's run-on holds its contacts to, which the
/// trace of its end across a curved cap takes too: a millionth of the tube's
/// chord between the contacts' parallels through `a` and `b`, and a
/// millionth at least.
fn torus_cap_tol(torus: &brepkit_math::surfaces::ToroidalSurface, a: Point3, b: Point3) -> f64 {
    let (_, va) = torus.project_point(a);
    let (_, vb) = torus.project_point(b);
    let chord = 2.0 * torus.minor_radius() * (0.5 * wrapped(vb - va)).sin().abs();
    1e-6 * chord.max(1.0)
}

/// The curve where a band run on to `cap` meets it between the contact ends
/// `a` and `b`. `None` where the trace fails, and for a band of a kind no
/// run-on makes (one other than a cylinder, a torus or a walked band).
fn band_meets_run_on_cap(
    topo: &Topology,
    band: &FaceSurface,
    cap: RunOnCap,
    a: Point3,
    b: Point3,
) -> Result<Option<EdgeCurve>, BlendError> {
    Ok(match (cap, band) {
        (RunOnCap::Plane(normal, d), FaceSurface::Cylinder(cylinder)) => {
            crate::builder_utils::cylinder_plane_arc(cylinder, normal, d, a, b)
        }
        (RunOnCap::Face(cap), FaceSurface::Cylinder(cylinder)) => {
            cylinder_meets_cap(cylinder, topo.face(cap)?.surface(), a, b)
        }
        (RunOnCap::Plane(normal, d), FaceSurface::Torus(torus)) => {
            torus_meets_cap(torus, a, b, |v, near| {
                torus_parallel_meets_plane(torus, normal, d, v, near)
            })
        }
        (RunOnCap::Face(cap), FaceSurface::Torus(torus)) => {
            let cap = topo.face(cap)?.surface();
            let tol = torus_cap_tol(torus, a, b);
            torus_meets_cap(torus, a, b, |v, near| {
                torus_parallel_meets_surface(torus, cap, v, near, tol)
            })
        }
        (_, FaceSurface::Nurbs(_)) => {
            let scale = (a - b).length().max(1.0);
            band_meets_cap(band, a, b, |p| cap_offset(topo, cap, p, 1e-6 * scale))
        }
        (_, FaceSurface::Plane { .. } | FaceSurface::Cone(_) | FaceSurface::Sphere(_)) => None,
    })
}

/// Whether a stripe's terminal vertex sits on a planar cap whose two edges
/// at the vertex the stripe's contacts end on (lines, a spline profile's
/// rim, or a vault's end arc), or on a curved cap the stripe was run on to: the end-cap notch can replace that
/// corner path with the stripe's end curve. A notch that would cross another
/// boundary of its cap is an error.
fn terminal_cap_is_notchable(
    topo: &Topology,
    junction: &crate::fillet_plan::VertexJunction,
    contour: &crate::fillet_plan::FilletContour,
    stripe: &Stripe,
    run_on: &TiltedCaps,
) -> Result<bool, BlendError> {
    let caps: Vec<FaceId> = junction
        .face_fan
        .iter()
        .copied()
        .filter(|&face| face != contour.side1 && face != contour.side2)
        .collect();
    let [cap] = caps.as_slice() else {
        return Ok(false);
    };
    let Ok(face) = topo.face(*cap) else {
        return Ok(false);
    };
    let Ok(vertex) = topo.vertex(junction.vertex) else {
        return Ok(false);
    };
    let vertex = vertex.point();
    if !face.surface().is_planar() {
        // A curved cap takes the notch only where the stripe was run on to
        // it: its contacts then end on the cap's corner edges, checked there.
        let end = usize::from(spine_ends_at(topo, stripe, vertex)?);
        if !matches!(
            run_on.get(&(stripe.spine_edges()[0], end)),
            Some(RunOnCap::Face(face)) if face == cap
        ) {
            return Ok(false);
        }
        // Seen along the band's axis at its end, the notch lies in the
        // triangle between its ends and the corner, as on a planar cap seen
        // square to it. A torus round's axis there is its parallels'
        // tangent, a walked band's its end section's normal.
        let Some(section) = end_section(topo, stripe, vertex)? else {
            return Ok(false);
        };
        let view = match &stripe.surface {
            FaceSurface::Cylinder(cylinder) => cylinder.axis(),
            FaceSurface::Torus(torus) => {
                let (u, _) = torus.project_point(section.center);
                torus.y_axis() * u.cos() - torus.x_axis() * u.sin()
            }
            FaceSurface::Nurbs(_) => {
                let Ok(normal) = (section.p1 - section.center)
                    .cross(section.p2 - section.center)
                    .normalize()
                else {
                    return Ok(false);
                };
                normal
            }
            FaceSurface::Plane { .. } | FaceSurface::Cone(_) | FaceSurface::Sphere(_) => {
                return Ok(false);
            }
        };
        let near_ends = [
            (&stripe.contact1, section.p1),
            (&stripe.contact2, section.p2),
        ]
        .map(|(contact, at)| contact_end_at(contact, at).0);
        let end_curve = band_meets_run_on_cap(
            topo,
            &stripe.surface,
            RunOnCap::Face(*cap),
            near_ends[0],
            near_ends[1],
        )?;
        let end_hull = match &end_curve {
            Some(EdgeCurve::NurbsCurve(curve))
                if curve.weights().iter().all(|w| w.is_finite() && *w > 0.0) =>
            {
                Some(curve.control_points())
            }
            _ => None,
        };
        return notch_stays_clear(topo, face, junction.vertex, view, near_ends, end_hull);
    }
    // The corner sits on the cap's outer wire (a convex fillet) or on one of
    // its holes (a concave one ending on a cup's rim); the notch replaces it
    // on whichever wire holds it.
    let mut corner_edges: Vec<(Point3, Point3, &brepkit_topology::edge::EdgeCurve)> = Vec::new();
    let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for oriented in wires
        .filter_map(|wire| topo.wire(wire).ok())
        .flat_map(|wire| wire.edges().iter())
    {
        let Ok(edge) = topo.edge(oriented.edge()) else {
            return Ok(false);
        };
        if edge.start() != junction.vertex && edge.end() != junction.vertex {
            continue;
        }
        // The trim splits a corner edge of any kind where the contact ends
        // on it: a line, a B-spline (a spline wall's rim), or an arc (the
        // end of a vault).
        let (Ok(a), Ok(b)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            return Ok(false);
        };
        corner_edges.push((a.point(), b.point(), edge.curve()));
    }
    if corner_edges.len() != 2 {
        return Ok(false);
    }
    // A file's corner edge can stand off the side face it bounds (a B-spline
    // skin's edges 3e-3 off it), and the contact on that face ends off the
    // edge by as much: the ends are read on the edges within that too.
    let sides = [contour.side1, contour.side2]
        .map(|side| topo.face(side).map(|f| f.surface().clone()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let slack = corner_edges
        .iter()
        .map(|&(a, b, curve)| {
            sides
                .iter()
                .map(|surface| edge_off_surface(curve, a, b, surface))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0_f64, f64::max);
    let reach = 1e-5 + if slack.is_finite() { slack } else { 0.0 };
    // Both contacts must end on those corner edges, or the split lands on
    // the cap's edge while the notch can never match its corner path.
    let on_segment = |point: Point3| {
        corner_edges.iter().any(|&(a, b, curve)| {
            if let brepkit_topology::edge::EdgeCurve::Circle(circle) = curve {
                let t = circle.project(point);
                if (circle.evaluate(t) - point).length() > reach {
                    return false;
                }
                let (t0, t1) = curve.domain_with_endpoints(a, b);
                let (low, high) = (t0.min(t1) - 1e-9, t0.max(t1) + 1e-9);
                let tau = std::f64::consts::TAU;
                return [t - tau, t, t + tau]
                    .iter()
                    .any(|&t| (low..=high).contains(&t));
            }
            if let brepkit_topology::edge::EdgeCurve::NurbsCurve(nurbs) = curve {
                let (t0, t1) = curve.domain_with_endpoints(a, b);
                let (low, high) = (t0.min(t1) - 1e-9, t0.max(t1) + 1e-9);
                return brepkit_math::nurbs::projection::project_point_to_curve(nurbs, point, 1e-9)
                    .is_ok_and(|found| {
                        found.distance <= reach && (low..=high).contains(&found.parameter)
                    });
            }
            let turn = match curve {
                brepkit_topology::edge::EdgeCurve::Circle(circle) => Some(circle.project(point)),
                brepkit_topology::edge::EdgeCurve::Ellipse(ellipse) => Some(ellipse.project(point)),
                brepkit_topology::edge::EdgeCurve::Line
                | brepkit_topology::edge::EdgeCurve::NurbsCurve(_) => None,
            };
            if let Some(turn) = turn {
                // Within the arc's own span, from its start.
                const TAU: f64 = std::f64::consts::TAU;
                let (t0, t1) = curve.domain_with_endpoints(a, b);
                let along = (turn - t0).rem_euclid(TAU);
                return (along <= t1 - t0 + 1e-9 || along >= TAU - 1e-9)
                    && (curve.evaluate_with_endpoints(turn, a, b) - point).length() <= reach;
            }
            let ab = b - a;
            let len2 = ab.length_squared();
            if len2 <= 1e-18 {
                return false;
            }
            let t = ((point - a).dot(ab) / len2).clamp(0.0, 1.0);
            (a + ab * t - point).length() <= reach
        })
    };
    // Each contact's end at this end of the spine: the one at the end
    // section's contact point (the end nearer the corner need not be, where
    // the round is wider than the edge is long).
    let Some(section) = end_section(topo, stripe, vertex)? else {
        return Ok(false);
    };
    let near_ends = [
        (&stripe.contact1, section.p1),
        (&stripe.contact2, section.p2),
    ]
    .map(|(contact, at)| contact_end_at(contact, at).0);
    if !near_ends.iter().all(|&point| on_segment(point)) {
        return Ok(false);
    }
    let FaceSurface::Plane { normal, .. } = face.surface() else {
        return Ok(false);
    };
    notch_stays_clear(topo, face, junction.vertex, *normal, near_ends, None)
}

/// How far the edge from `a` to `b` along `curve` stands off `surface`, read
/// at its ends and at points along it (infinite where the surface cannot
/// be read).
fn edge_off_surface(
    curve: &brepkit_topology::edge::EdgeCurve,
    a: Point3,
    b: Point3,
    surface: &FaceSurface,
) -> f64 {
    const SAMPLES: u32 = 8;
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    let mut farthest: f64 = 0.0;
    for k in 0..=SAMPLES {
        let t = (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0);
        let p = curve.evaluate_with_endpoints(t, a, b);
        let off = match surface {
            FaceSurface::Plane { normal, d } => {
                (normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d).abs()
            }
            _ => match surface
                .project_point(p)
                .and_then(|(u, v)| surface.evaluate(u, v))
            {
                Some(q) => (q - p).length(),
                None => return f64::INFINITY,
            },
        };
        farthest = farthest.max(off);
    }
    farthest
}

/// Whether a cap's notch at `corner`, ending at `near_ends`, stays clear of
/// the cap's other boundaries, seen along `view` (the normal of a planar cap,
/// or the axis at its end of a band run on to a curved one). The notch lies
/// in the triangle between its ends and the corner, so no other boundary of
/// the cap may reach into that triangle (a fillet wider than a cup's wall
/// would cross its mouth): that is an error. `Ok(false)` where the view
/// cannot be read.
///
/// On a curved cap, the notch lies in the triangle only while the cap does
/// not turn parallel to the axis over it, and the checks below take that
/// as given. `end_hull` is the control hull of the notch's end curve
/// across the band. An edge wholly beyond that hull along the axis cannot
/// meet the end curve, and on a valid face no edge crosses the corner's own
/// edges. So a wire whose every edge is beyond the hull or clear of the
/// triangle lies wholly inside the notch or wholly outside it. The corner's
/// own wire runs on outside the notch past the near ends, and any other
/// wire is outside once one of its vertices is seen outside the triangle.
/// Its edges beyond the hull are then passed over, however they lie when
/// seen along the axis.
fn notch_stays_clear(
    topo: &Topology,
    face: &Face,
    corner: VertexId,
    view: Vec3,
    near_ends: [Point3; 2],
    end_hull: Option<&[Point3]>,
) -> Result<bool, BlendError> {
    let Ok(vertex) = topo.vertex(corner) else {
        return Ok(false);
    };
    let vertex = vertex.point();
    let Ok(frame) = brepkit_math::frame::Frame3::from_normal(vertex, view) else {
        return Ok(false);
    };
    let flat = |p: Point3| {
        let d = p - vertex;
        (d.dot(frame.x), d.dot(frame.y))
    };
    let triangle = [flat(near_ends[0]), (0.0, 0.0), flat(near_ends[1])];
    // The notch lies between the cap's two corner edges and the round's end.
    // A straight corner edge is a side of the triangle. An arc that bows
    // away from the triangle takes the notch past that side, as far as the
    // arc's tangents at its two ends: the triangle of the corner, the
    // contact's end and the point where those tangents meet holds that part.
    // A B-spline's piece lies in the hull of its control points.
    let mut regions: Vec<Vec<(f64, f64)>> = vec![triangle.to_vec()];
    let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for oriented in wires
        .filter_map(|wire| topo.wire(wire).ok())
        .flat_map(|wire| wire.edges().iter())
    {
        let Ok(edge) = topo.edge(oriented.edge()) else {
            return Ok(false);
        };
        if edge.start() != corner && edge.end() != corner {
            continue;
        }
        let (Ok(a), Ok(b)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            return Ok(false);
        };
        let (a, b) = (a.point(), b.point());
        let across = (frame.x, frame.y);
        if let Some(bow) = arc_bow(edge, a, b, edge.start() == corner, near_ends, &flat, across) {
            regions.push(bow?.to_vec());
        }
        regions.extend(spline_bow(
            edge,
            a,
            b,
            edge.start() == corner,
            near_ends,
            &flat,
        )?);
    }
    let slab = end_hull.filter(|hull| !hull.is_empty()).map(|hull| {
        let along = |p: Point3| (p - vertex).dot(frame.z);
        let (lo, hi) = hull
            .iter()
            .map(|&p| along(p))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
                (lo.min(t), hi.max(t))
            });
        // The end edge is rebuilt later from the trimmer's vertices, which
        // may sit 1e-6 off these ends; on a steep cap that moves the curve
        // along the axis by that times the slope.
        let size = hull
            .iter()
            .map(|&p| (p - vertex).length())
            .fold(0.0_f64, f64::max);
        let slack = 1e-4 * size.max(1.0);
        (frame.z, vertex, lo - slack, hi + slack)
    });
    // Outside the notch: outside each region that holds a part of it.
    let outside_regions = |p: (f64, f64)| {
        regions
            .iter()
            .all(|region| region_side(region, p) == RegionSide::Outside)
    };
    let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for wire in wires.filter_map(|wire| topo.wire(wire).ok()) {
        let mut edges = Vec::with_capacity(wire.edges().len());
        for oriented in wire.edges() {
            let Ok(edge) = topo.edge(oriented.edge()) else {
                return Ok(false);
            };
            let (Ok(a), Ok(b)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                return Ok(false);
            };
            edges.push((edge, a.point(), b.point()));
        }
        let outside_notch = slab.is_some()
            && edges.iter().any(|&(edge, a, b)| {
                edge.start() == corner
                    || edge.end() == corner
                    || outside_regions(flat(a))
                    || outside_regions(flat(b))
            });
        for (edge, a, b) in edges {
            if edge.start() == corner || edge.end() == corner {
                continue;
            }
            if outside_notch && slab.is_some_and(|slab| cap_edge_beyond_slab(edge, a, b, slab)) {
                continue;
            }
            if regions
                .iter()
                .any(|region| cap_edge_reaches_region(edge, a, b, &flat, region))
            {
                return Err(BlendError::PlanningFailure {
                    reason: format!(
                        "the fillet's end at {corner:?} would cross another boundary of its cap"
                    ),
                });
            }
        }
    }
    Ok(true)
}

/// For a cap's corner edge that is an arc bowing away from the notch
/// triangle: the triangle of the corner, the contact's end on the arc, and
/// the point where the arc's tangents at those two meet, which holds the arc
/// between them. `None` for a line or a B-spline, an arc that bows into the
/// notch triangle, and an arc too flat to leave it; an error for an arc whose
/// piece turns more than a quarter turn, too far for its tangents to bound
/// it.
fn arc_bow(
    edge: &brepkit_topology::edge::Edge,
    a: Point3,
    b: Point3,
    corner_at_start: bool,
    near_ends: [Point3; 2],
    flat: &impl Fn(Point3) -> (f64, f64),
    (across_x, across_y): (Vec3, Vec3),
) -> Option<Result<[(f64, f64); 3], BlendError>> {
    use brepkit_topology::edge::EdgeCurve;
    const TAU: f64 = std::f64::consts::TAU;
    let curve = edge.curve();
    let turn_of = |point: Point3| match curve {
        EdgeCurve::Circle(circle) => Some(circle.project(point)),
        EdgeCurve::Ellipse(ellipse) => Some(ellipse.project(point)),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
    };
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    // The contact's end on this arc: the nearer of the two.
    let (end, other, turn) = [(near_ends[0], near_ends[1]), (near_ends[1], near_ends[0])]
        .into_iter()
        .filter_map(|(end, other)| {
            // Within the arc's span, or just past the nearer of its ends.
            let turn = t0 + (turn_of(end)? - t0).rem_euclid(TAU);
            let turn = if turn > t1 && t0 - (turn - TAU) < turn - t1 {
                turn - TAU
            } else {
                turn
            };
            let off = (curve.evaluate_with_endpoints(turn, a, b) - end).length();
            Some((off, end, other, turn))
        })
        .min_by(|p, q| p.0.total_cmp(&q.0))
        .map(|(_, end, other, turn)| (end, other, turn))?;
    let at_corner = if corner_at_start { t0 } else { t1 };
    let corner = if corner_at_start { a } else { b };
    let cross = |o: (f64, f64), p: (f64, f64), q: (f64, f64)| {
        (p.0 - o.0) * (q.1 - o.1) - (p.1 - o.1) * (q.0 - o.0)
    };
    let (c, e) = (flat(corner), flat(end));
    // Each tangent in the view, with its own length before the view
    // shortened it: a tangent that points along the view has no direction
    // there, and turns from nothing.
    let heading = |turn: f64| {
        let tangent = curve.tangent_with_endpoints(turn, a, b);
        (
            (tangent.dot(across_x), tangent.dot(across_y)),
            tangent.length(),
        )
    };
    let chord = (e.0 - c.0, e.1 - c.1);
    let turns = |(p, p_len): ((f64, f64), f64), (q, q_len): ((f64, f64), f64)| {
        let sine = p.0 * q.1 - p.1 * q.0;
        (sine.abs() > 1e-9 * p_len * q_len).then_some(sine)
    };
    let dc = heading(at_corner);
    // An arc seen edge on (a curved cap's rim seen along the round's axis)
    // is a straight line, however long its piece: it does not leave the
    // triangle. Its tangents a quarter turn apart are then parallel.
    let edge_on = turns(dc, heading(at_corner + std::f64::consts::FRAC_PI_2)).is_none();
    if chord.0.hypot(chord.1) <= 0.0 || edge_on {
        return None;
    }
    // Past a quarter turn the tangents meet far from the arc, or behind it.
    if (turn - at_corner).abs() > std::f64::consts::FRAC_PI_2 {
        return Some(Err(BlendError::PlanningFailure {
            reason: "the fillet's end takes too long a piece of its cap's arc edge".into(),
        }));
    }
    // A piece too short to turn leaves the triangle by a rounding at most.
    let de = heading(turn);
    let sine = turns(dc, de)?;
    let (dc, de) = (dc.0, de.0);
    let along = (chord.0 * de.1 - chord.1 * de.0) / sine;
    let apex = (c.0 + dc.0 * along, c.1 + dc.1 * along);
    let bows_out = cross(c, e, apex) * cross(c, e, flat(other)) < 0.0;
    bows_out.then_some(Ok([c, apex, e]))
}

/// For a cap's corner edge that is a B-spline: the convex hull of the
/// corner, the contact's end on the edge, and the control points of the
/// edge's piece between the two. While every weight is positive that hull
/// holds the piece, and so the part of the notch between the piece and its
/// chord, whichever way the piece bows. Where the piece cannot be cut out
/// of the curve, or is too narrow to cut, the hull of the whole curve's
/// control points stands in. `None` for any other curve and for a piece
/// seen edge on; an error for a weight that the hull does not bound.
fn spline_bow(
    edge: &brepkit_topology::edge::Edge,
    a: Point3,
    b: Point3,
    corner_at_start: bool,
    near_ends: [Point3; 2],
    flat: &impl Fn(Point3) -> (f64, f64),
) -> Result<Option<Vec<(f64, f64)>>, BlendError> {
    use brepkit_math::nurbs::{curve_split, project_point_to_curve};
    use brepkit_topology::edge::EdgeCurve;
    // The weld scale: a vertex sits within this of its edge's curve.
    const ON_CURVE: f64 = 1e-5;
    let EdgeCurve::NurbsCurve(curve) = edge.curve() else {
        return Ok(None);
    };
    if !curve.weights().iter().all(|w| w.is_finite() && *w > 0.0) {
        return Err(BlendError::PlanningFailure {
            reason: "the fillet's end runs along a cap edge whose weights do not bound it".into(),
        });
    }
    let corner = if corner_at_start { a } else { b };
    let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
    let at_corner = if corner_at_start { t0 } else { t1 };
    let (low, high) = (t0.min(t1), t0.max(t1));
    // The contact's end on this edge: the nearer of the two to the edge's
    // own span. The other can lie nearer the rest of the curve past it.
    let found = near_ends
        .into_iter()
        .filter_map(|end| {
            let found = project_point_to_curve(curve, end, 1e-9).ok()?;
            let at = found.parameter.clamp(low, high);
            Some(((curve.evaluate(at) - end).length(), end, at))
        })
        .min_by(|p, q| p.0.total_cmp(&q.0));
    let Some((_, end, at_end)) = found else {
        return Ok(None);
    };
    let (d0, d1) = brepkit_math::traits::ParametricCurve::domain(curve);
    let narrow = 1e-12 * (d1 - d0).abs().max(1.0);
    let piece = || -> Option<Vec<Point3>> {
        if (curve.evaluate(at_corner) - corner).length() > ON_CURVE {
            return None;
        }
        // A piece too narrow to split can still bow far, as fast as the
        // curve runs there: the whole curve's hull stands in.
        let (from, to) = (at_corner.min(at_end), at_corner.max(at_end));
        if to - from <= narrow {
            return None;
        }
        let mut piece = curve.clone();
        if to < d1 - narrow {
            piece = curve_split(&piece, to).ok()?.0;
        }
        if from > d0 + narrow {
            piece = curve_split(&piece, from).ok()?.1;
        }
        Some(piece.control_points().to_vec())
    };
    let controls = piece().unwrap_or_else(|| curve.control_points().to_vec());
    let points: Vec<(f64, f64)> = [corner, end]
        .into_iter()
        .chain(controls)
        .map(flat)
        .collect();
    let hull = convex_hull_2d(&points);
    Ok((hull.len() >= 3).then_some(hull))
}

/// Whether a cap edge lies wholly to one side of the slab `lo..=hi` along
/// `axis` from `origin`, certified: a line by its ends, a circle or an
/// ellipse by samples clear of it by their sag, a B-spline by its control
/// hull (only while every weight is positive).
fn cap_edge_beyond_slab(
    edge: &brepkit_topology::edge::Edge,
    a: Point3,
    b: Point3,
    (axis, origin, lo, hi): (Vec3, Point3, f64, f64),
) -> bool {
    use brepkit_topology::edge::EdgeCurve;
    const SAMPLES: u32 = 64;
    let along = |p: Point3| (p - origin).dot(axis);
    let beyond = |values: &mut dyn Iterator<Item = f64>, margin: f64| {
        let (mut below, mut above) = (true, true);
        for t in values {
            below &= t < lo - margin;
            above &= t > hi + margin;
        }
        below || above
    };
    let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
    let step = (t1 - t0).abs() / f64::from(SAMPLES);
    let mut samples = (0..=SAMPLES).map(|k| {
        let t = t0 + (t1 - t0) * f64::from(k) / f64::from(SAMPLES);
        along(edge.curve().evaluate_with_endpoints(t, a, b))
    });
    match edge.curve() {
        EdgeCurve::Line => beyond(&mut [along(a), along(b)].into_iter(), 0.0),
        EdgeCurve::Circle(circle) => {
            beyond(&mut samples, circle.radius() * step * step / 8.0 * 1.01)
        }
        EdgeCurve::Ellipse(ellipse) => beyond(&mut samples, ellipse_sag(ellipse, step) * 1.01),
        EdgeCurve::NurbsCurve(curve) => {
            curve.weights().iter().all(|w| w.is_finite() && *w > 0.0)
                && beyond(&mut curve.control_points().iter().map(|&p| along(p)), 0.0)
        }
    }
}

/// How far an ellipse strays from its chords over `step` of its angle. The
/// curve's second derivative in its angle is at most its larger semi-axis,
/// and `Ellipse3D::with_axes` takes the two in either order.
fn ellipse_sag(ellipse: &brepkit_math::curves::Ellipse3D, step: f64) -> f64 {
    let big = ellipse.semi_major().max(ellipse.semi_minor());
    let small = ellipse.semi_major().min(ellipse.semi_minor());
    big.powi(3) / (small * small) * step * step / 8.0
}

/// Whether a cap edge may pass inside a convex region of the notch,
/// certified: a line exactly, a circle or ellipse by a polyline held to the
/// region widened by its chords' sag, and a NURBS curve by the hull of its
/// control points, which holds the whole curve however narrow a knot span
/// is.
fn cap_edge_reaches_region(
    edge: &brepkit_topology::edge::Edge,
    a: Point3,
    b: Point3,
    flat: &impl Fn(Point3) -> (f64, f64),
    region: &[(f64, f64)],
) -> bool {
    use brepkit_topology::edge::EdgeCurve;
    const SAMPLES: u32 = 64;
    let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
    let polyline = || -> Vec<(f64, f64)> {
        (0..=SAMPLES)
            .map(|k| {
                let t = t0 + (t1 - t0) * f64::from(k) / f64::from(SAMPLES);
                flat(edge.curve().evaluate_with_endpoints(t, a, b))
            })
            .collect()
    };
    let step = (t1 - t0).abs() / f64::from(SAMPLES);
    match edge.curve() {
        // A line whose ends lie on two sides of the region crosses neither.
        EdgeCurve::Line => hull_overlaps_region(&[flat(a), flat(b)], region),
        EdgeCurve::Circle(circle) => {
            let sag = circle.radius() * step * step / 8.0;
            path_within_of_region(&polyline(), region, sag * 1.01)
        }
        EdgeCurve::Ellipse(ellipse) => {
            path_within_of_region(&polyline(), region, ellipse_sag(ellipse, step) * 1.01)
        }
        // The hull bounds a rational curve only while every weight is
        // positive; any other curve cannot be certified clear.
        EdgeCurve::NurbsCurve(curve) => {
            if !curve.weights().iter().all(|w| w.is_finite() && *w > 0.0) {
                return true;
            }
            let hull: Vec<(f64, f64)> = curve.control_points().iter().map(|&p| flat(p)).collect();
            hull_overlaps_region(&hull, region)
                && nurbs_pieces_reach_region(curve, t0, t1, flat, region)
        }
    }
}

/// Whether the edge's stretch `t0..t1` of a B-spline with positive weights
/// reaches into the region, its pieces certified by their own Bezier
/// hulls. The whole curve's hull is loose on a long curve of many spans (a
/// cap's outline bowing away from a corner its chord passes close to), so
/// each span's piece is tested, and halved while its hull overlaps, down
/// to a limit at which an overlap counts as reaching.
fn nurbs_pieces_reach_region(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    t0: f64,
    t1: f64,
    flat: &impl Fn(Point3) -> (f64, f64),
    region: &[(f64, f64)],
) -> bool {
    const DEPTH: u32 = 16;
    let (d0, d1) = curve.domain();
    let (lo, hi) = (t0.min(t1).max(d0), t0.max(t1).min(d1));
    if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
        return true;
    }
    let mut cuts: Vec<f64> = std::iter::once(lo)
        .chain(curve.knots().iter().copied().filter(|&k| k > lo && k < hi))
        .chain(std::iter::once(hi))
        .collect();
    cuts.dedup();
    let mut pending: Vec<(f64, f64, u32)> = cuts.windows(2).map(|w| (w[0], w[1], 0)).collect();
    while let Some((a, b, depth)) = pending.pop() {
        let Ok(points) = brepkit_math::nurbs::decompose::curve_bezier_between(curve, a, b) else {
            return true;
        };
        let mut hull = Vec::with_capacity(points.len());
        for [x, y, z, w] in points {
            if !(w.is_finite() && w > 0.0) {
                return true;
            }
            hull.push(flat(Point3::new(x / w, y / w, z / w)));
        }
        if !hull_overlaps_region(&hull, region) {
            continue;
        }
        if depth >= DEPTH {
            return true;
        }
        let mid = 0.5 * (a + b);
        pending.push((a, mid, depth + 1));
        pending.push((mid, b, depth + 1));
    }
    false
}

/// Whether a polyline comes within `margin` of a convex region, its inside
/// included.
fn path_within_of_region(path: &[(f64, f64)], region: &[(f64, f64)], margin: f64) -> bool {
    let to_segment = |p: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len2 = dx.madd(dx, dy * dy);
        let t = if len2 <= 0.0 {
            0.0
        } else {
            ((p.0 - a.0).madd(dx, (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
        };
        (p.0 - t.madd(dx, a.0)).hypot(p.1 - t.madd(dy, a.1))
    };
    let sides = || (0..region.len()).map(|i| (region[i], region[(i + 1) % region.len()]));
    path_enters_region(path, region)
        || path.windows(2).any(|pair| {
            sides().any(|(a, b)| {
                to_segment(pair[0], a, b) <= margin
                    || to_segment(pair[1], a, b) <= margin
                    || to_segment(a, pair[0], pair[1]) <= margin
                    || to_segment(b, pair[0], pair[1]) <= margin
            })
        })
}

/// The convex hull of `points`, counter-clockwise, without points on its
/// sides: two points for points along a line, none for one point.
fn convex_hull_2d(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.total_cmp(&q.1)));
    sorted.dedup();
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut hull: Vec<(f64, f64)> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let ordered: Box<dyn Iterator<Item = &(f64, f64)>> = if pass == 0 {
            Box::new(sorted.iter())
        } else {
            Box::new(sorted.iter().rev())
        };
        for &p in ordered {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// Whether the convex hull of `points` overlaps a convex region by more than
/// a rounding of the region's size (touching alone does not count). The
/// rounding does not grow with the hull: a long edge through a small
/// notch still crosses it.
fn hull_overlaps_region(points: &[(f64, f64)], region: &[(f64, f64)]) -> bool {
    let hull = convex_hull_2d(points);
    if hull.is_empty() {
        return false;
    }
    let scale = region
        .iter()
        .map(|&(x, y)| x.abs().max(y.abs()))
        .fold(0.0_f64, f64::max);
    let eps = 1e-9 * scale.max(1.0);
    let polygons: [&[(f64, f64)]; 2] = [&hull, region];
    for polygon in polygons {
        let n = polygon.len();
        for i in 0..n {
            let (a, b) = (polygon[i], polygon[(i + 1) % n]);
            let axis = (b.1 - a.1, a.0 - b.0);
            let len = axis.0.hypot(axis.1);
            if len <= eps {
                continue;
            }
            let project = |p: &(f64, f64)| (p.0 * axis.0 + p.1 * axis.1) / len;
            let (h0, h1) = hull
                .iter()
                .map(project)
                .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
            let (t0, t1) = region
                .iter()
                .map(project)
                .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
            if h1 <= t0 + eps || t1 <= h0 + eps {
                return false;
            }
        }
    }
    true
}

/// Where a point lies against a convex region, to a rounding of its size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RegionSide {
    Inside,
    OnBoundary,
    Outside,
}

fn region_side(region: &[(f64, f64)], p: (f64, f64)) -> RegionSide {
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let n = region.len();
    let scale = region
        .iter()
        .map(|&(x, y)| x.abs().max(y.abs()))
        .fold(0.0_f64, f64::max);
    let eps = 1e-9 * scale * scale;
    let area: f64 = (0..n)
        .map(|i| cross((0.0, 0.0), region[i], region[(i + 1) % n]))
        .sum();
    let sides = (0..n).map(|i| cross(region[i], region[(i + 1) % n], p) * area.signum());
    let mut side = RegionSide::Inside;
    for s in sides {
        if s < -eps {
            return RegionSide::Outside;
        }
        if s <= eps {
            side = RegionSide::OnBoundary;
        }
    }
    side
}

/// Whether a polyline in the plane passes inside a convex region: a point
/// strictly inside it, or a segment crossing one of its sides.
fn path_enters_region(path: &[(f64, f64)], region: &[(f64, f64)]) -> bool {
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let n = region.len();
    let scale = region
        .iter()
        .map(|&(x, y)| x.abs().max(y.abs()))
        .fold(0.0_f64, f64::max);
    let eps = 1e-9 * scale * scale;
    let opposite = |x: f64, y: f64| (x > eps && y < -eps) || (x < -eps && y > eps);
    let crosses = |p: (f64, f64), q: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        opposite(cross(a, b, p), cross(a, b, q)) && opposite(cross(p, q, a), cross(p, q, b))
    };
    path.iter()
        .any(|&p| region_side(region, p) == RegionSide::Inside)
        || path
            .windows(2)
            .any(|pair| (0..n).any(|i| crosses(pair[0], pair[1], region[i], region[(i + 1) % n])))
}

/// A face's normal at `p`, outward as the face is used.
fn effective_normal(
    topo: &Topology,
    face_id: FaceId,
    p: Point3,
) -> Result<Option<Vec3>, BlendError> {
    let face = topo.face(face_id)?;
    let normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        surface => {
            let Some((u, v)) = surface.project_point(p) else {
                return Ok(None);
            };
            surface.normal(u, v)
        }
    };
    Ok(Some(if face.is_reversed() { -normal } else { normal }))
}

/// Sets each built face whose wire runs along a seam edge both ways (a
/// closed band) facing as the neighbours it meets tangentially do.
///
/// The normal pass reads a face's winding from its boundary and skips any
/// face that uses an edge twice, so a band round a closed spine keeps
/// whichever normal its sections gave it, and may face into the solid. A
/// blend meets its supports tangentially along its contacts, so its outward
/// normal must agree with theirs there: it is turned (wire reversed, edge
/// senses and reversal flag toggled, so shared edges keep their senses) when
/// the tangent neighbours disagree with it. Only faces already settled vote,
/// so a seamed wall and the band beside it are not judged by each other
/// first; they settle in rounds outward from the other faces. A seamed
/// face trimmed from an input face (`sources` maps it to that face) is set
/// facing as its source did, before the rounds: it lies on the source's
/// surface, while its neighbours may meet it at a knife edge, facing the
/// other way rightly.
fn align_seamed_faces_with_tangent_neighbours(
    topo: &mut Topology,
    new_faces: &[FaceId],
    result_faces: &[FaceId],
    sources: &std::collections::HashMap<FaceId, FaceId>,
) -> Result<bool, BlendError> {
    const TANGENT_COS: f64 = 0.985;
    let mut pending: Vec<FaceId> = Vec::new();
    let mut turned_any = false;
    for &face_id in new_faces {
        let face = topo.face(face_id)?;
        if matches!(face.surface(), FaceSurface::Plane { .. }) || !face.inner_wires().is_empty() {
            continue;
        }
        let edges = topo.wire(face.outer_wire())?.edges().to_vec();
        let mut seen = HashSet::new();
        if edges.iter().all(|oe| seen.insert(oe.edge())) {
            continue;
        }
        let Some(&source) = sources.get(&face_id) else {
            pending.push(face_id);
            continue;
        };
        // A trimmed input face lies on its source's surface, so it faces as
        // its source did; its neighbours cannot say (a knife edge, two
        // sheets meeting at no angle, has them facing opposite ways).
        let uses = |e: EdgeId| edges.iter().filter(|oe| oe.edge() == e).count();
        for oe in edges.iter().filter(|oe| uses(oe.edge()) == 1) {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            let p = edge
                .curve()
                .evaluate_with_endpoints(f64::midpoint(t0, t1), a, b);
            let (Some(mine), Some(theirs)) = (
                effective_normal(topo, face_id, p)?,
                effective_normal(topo, source, p)?,
            ) else {
                continue;
            };
            if mine.dot(theirs) < 0.0 {
                log::debug!("trimmed face {face_id:?} faced away from its source: turned");
                turn_face(topo, face_id)?;
                turned_any = true;
            }
            break;
        }
    }
    let mut edge_faces: std::collections::HashMap<EdgeId, Vec<FaceId>> =
        std::collections::HashMap::new();
    for &face_id in result_faces {
        let face = topo.face(face_id)?;
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for w in wires {
            for oe in topo.wire(w)?.edges() {
                let faces = edge_faces.entry(oe.edge()).or_default();
                if !faces.contains(&face_id) {
                    faces.push(face_id);
                }
            }
        }
    }
    while !pending.is_empty() {
        let mut settled = Vec::new();
        for &face_id in &pending {
            let edges = topo
                .wire(topo.face(face_id)?.outer_wire())?
                .edges()
                .to_vec();
            let uses = |e: EdgeId| edges.iter().filter(|oe| oe.edge() == e).count();
            let mut votes = (0_u32, 0_u32);
            for oe in edges.iter().filter(|oe| uses(oe.edge()) == 1) {
                let edge = topo.edge(oe.edge())?;
                let (a, b) = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                let p = edge
                    .curve()
                    .evaluate_with_endpoints(f64::midpoint(t0, t1), a, b);
                let Some(mine) = effective_normal(topo, face_id, p)? else {
                    continue;
                };
                for &other in edge_faces.get(&oe.edge()).map_or(&[][..], Vec::as_slice) {
                    if other == face_id || pending.contains(&other) {
                        continue;
                    }
                    let Some(theirs) = effective_normal(topo, other, p)? else {
                        continue;
                    };
                    let cos = mine.dot(theirs);
                    if cos >= TANGENT_COS {
                        votes.0 += 1;
                    } else if cos <= -TANGENT_COS {
                        votes.1 += 1;
                    }
                }
            }
            if votes == (0, 0) {
                continue;
            }
            settled.push(face_id);
            if votes.1 <= votes.0 {
                continue;
            }
            log::debug!("seamed face {face_id:?} faced away from its tangent neighbours: turned");
            turn_face(topo, face_id)?;
            turned_any = true;
        }
        if settled.is_empty() {
            break;
        }
        pending.retain(|f| !settled.contains(f));
    }
    Ok(turned_any)
}

/// Turns a face over: its wire reversed, edge senses and reversal flag
/// toggled, so the edges it shares keep their senses.
fn turn_face(topo: &mut Topology, face_id: FaceId) -> Result<(), BlendError> {
    let wire_id = topo.face(face_id)?.outer_wire();
    let wire = topo.wire_mut(wire_id)?;
    let turned: Vec<OrientedEdge> = wire
        .edges()
        .iter()
        .rev()
        .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
        .collect();
    for (slot, oe) in wire.edges_mut().iter_mut().zip(turned) {
        *slot = oe;
    }
    let face = topo.face_mut(face_id)?;
    face.set_reversed(!face.is_reversed());
    Ok(())
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

/// Two corners of the cube of half-width `half` about `center`.
fn cube_about(center: Point3, half: f64) -> [Point3; 2] {
    let d = Vec3::new(half, half, half);
    [center - d, center + d]
}

/// Points whose box holds a NURBS curve: its control points, whose hull
/// holds it while every weight is positive, or an unbounded box.
fn nurbs_curve_hull(curve: &brepkit_math::nurbs::curve::NurbsCurve) -> Vec<Point3> {
    if curve.weights().iter().all(|&w| w > 0.0) {
        curve.control_points().to_vec()
    } else {
        cube_about(Point3::new(0.0, 0.0, 0.0), f64::INFINITY).to_vec()
    }
}

/// Two corners of the box of the whole conic `center + a cos t u + b sin t
/// v`, which reaches `hypot(a u_i, b v_i)` either side along each axis.
fn conic_box(center: Point3, (a, u): (f64, Vec3), (b, v): (f64, Vec3)) -> [Point3; 2] {
    let d = Vec3::new(
        (a * u.x()).hypot(b * v.x()),
        (a * u.y()).hypot(b * v.y()),
        (a * u.z()).hypot(b * v.z()),
    );
    [center - d, center + d]
}

/// Points whose box holds an edge: a line's ends, the box of the whole
/// circle or ellipse it runs along, a NURBS curve's hull.
fn edge_hull_points(topo: &Topology, edge_id: EdgeId) -> Result<Vec<Point3>, BlendError> {
    let edge = topo.edge(edge_id)?;
    Ok(match edge.curve() {
        EdgeCurve::Line => vec![
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        ],
        EdgeCurve::Circle(c) => conic_box(
            c.center(),
            (c.radius(), c.u_axis()),
            (c.radius(), c.v_axis()),
        )
        .to_vec(),
        EdgeCurve::Ellipse(e) => conic_box(
            e.center(),
            (e.semi_major(), e.u_axis()),
            (e.semi_minor(), e.v_axis()),
        )
        .to_vec(),
        EdgeCurve::NurbsCurve(n) => nurbs_curve_hull(n),
    })
}

/// A box holding every face of `shell`. A plane, cylinder or cone face lies
/// within the hull of its boundary (each of its points is on a straight
/// segment between two boundary points), a sphere or torus face within its
/// whole surface's extent, a NURBS face within its control points.
fn shell_bounds(topo: &Topology, shell: ShellId) -> Result<Aabb3, BlendError> {
    let mut points = Vec::new();
    for &face_id in topo.shell(shell)?.faces() {
        let face = topo.face(face_id)?;
        match face.surface() {
            FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {}
            FaceSurface::Sphere(s) => points.extend(cube_about(s.center(), s.radius())),
            FaceSurface::Torus(t) => {
                points.extend(cube_about(t.center(), t.major_radius() + t.minor_radius()));
            }
            FaceSurface::Nurbs(n) => {
                if n.weights().iter().flatten().all(|&w| w > 0.0) {
                    points.extend(n.control_points().iter().flatten().copied());
                } else {
                    points.extend(cube_about(Point3::new(0.0, 0.0, 0.0), f64::INFINITY));
                }
            }
        }
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wire)?.edges() {
                points.extend(edge_hull_points(topo, oe.edge())?);
            }
        }
    }
    Ok(Aabb3::from_points(points))
}

/// Distance from `p` to the stretch of curve an edge spans: exact for a
/// line, the curve's own projection for a circle, ellipse or NURBS curve,
/// or the nearer end where that falls outside the edge's span.
fn distance_to_edge(topo: &Topology, edge_id: EdgeId, p: Point3) -> Result<f64, BlendError> {
    let edge = topo.edge(edge_id)?;
    let (sp, ep) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let curve = edge.curve();
    let ends = (p - sp).length().min((p - ep).length());
    let (t0, t1) = curve.domain_with_endpoints(sp, ep);
    let on_arc = |t: f64| {
        let t = t0 + (t - t0).rem_euclid(std::f64::consts::TAU);
        if t <= t1 {
            (p - curve.evaluate_with_endpoints(t, sp, ep)).length()
        } else {
            ends
        }
    };
    Ok(match curve {
        EdgeCurve::Line => {
            let d = ep - sp;
            let len2 = d.dot(d);
            let s = if len2 > 0.0 {
                ((p - sp).dot(d) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (p - (sp + d * s)).length()
        }
        EdgeCurve::Circle(c) => on_arc(c.project(p)),
        EdgeCurve::Ellipse(e) => on_arc(e.project(p)),
        // A foot on the stored curve beyond the edge's span would read a
        // stretch the edge does not run along.
        EdgeCurve::NurbsCurve(n) => {
            let (lo, hi) = (t0.min(t1), t0.max(t1));
            brepkit_math::nurbs::projection::project_point_to_curve(n, p, 1e-9)
                .ok()
                .filter(|found| (lo..=hi).contains(&found.parameter))
                .map_or(ends, |found| found.distance.min(ends))
        }
    })
}

/// A box holding the material a stripe takes off. Across the stripe that
/// material lies between the spine, the two neighbours' curves out to the
/// contacts and the blend arc, each turning less than half a circle and so
/// lying within its chord's length of its ends: the box of the spine and
/// contact curves, grown by twice the farthest contact's distance from the
/// spine (a chord across a curved spine runs a little longer than that
/// distance). The distance is read from points on the contact curves to
/// the spine edges' own curves, as a rational curve's control points can
/// stand well off it.
fn removed_material_bounds(topo: &Topology, stripe: &Stripe) -> Result<Aabb3, BlendError> {
    const CONTACT_SAMPLES: u32 = 64;
    let mut points = Vec::new();
    for &edge_id in stripe.spine.edges() {
        points.extend(edge_hull_points(topo, edge_id)?);
    }
    let mut reach = 0.0_f64;
    for contact in [&stripe.contact1, &stripe.contact2] {
        points.extend(nurbs_curve_hull(contact));
        let (t0, t1) = contact.domain();
        for k in 0..=CONTACT_SAMPLES {
            let c = contact.evaluate((t1 - t0).madd(f64::from(k) / f64::from(CONTACT_SAMPLES), t0));
            let mut near = f64::INFINITY;
            for &edge_id in stripe.spine.edges() {
                near = near.min(distance_to_edge(topo, edge_id, c)?);
            }
            reach = reach.max(near);
        }
    }
    Ok(Aabb3::from_points(points).expanded(2.0 * reach))
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
    // A side whose face the spine does not run along is a face beyond a
    // narrow face the round swallowed: the ball pivots on its rim there
    // (`pivot_on_far_loop`), and is not tangent to it. Its contact is the
    // rim itself, carried straight on past the rim's ends where the round
    // runs on to a tilted cap: off the face's surface there, until the cap
    // cuts it back to the rim's end (`walked_run_back`).
    let along = |face_id: FaceId| -> Result<bool, BlendError> {
        let Some(&edge) = stripe.spine_edges().first() else {
            return Ok(true);
        };
        let face = topo.face(face_id)?;
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            if topo.wire(wire)?.edges().iter().any(|oe| oe.edge() == edge) {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let (tangent1, tangent2) = (along(stripe.face1)?, along(stripe.face2)?);
    let mut max_contact: f64 = 0.0;
    let mut max_radius: f64 = 0.0;
    let mut max_tangent: f64 = 0.0;
    // With the ball pivoted on a swallowed face's rim, a round ending on a
    // tilted cap can end with its wall contact past the cap, beyond the edge
    // of the wall's spline surface, where it cannot be read: the cap cuts
    // the contact back inside it (`walked_run_back`).
    let pivoted = !(tangent1 && tangent2);
    let past_spline = |surface: &FaceSurface, (u, v): (f64, f64)| -> bool {
        let FaceSurface::Nurbs(spline) = surface else {
            return false;
        };
        let on = |x: f64, (low, high): (f64, f64)| {
            let slack = 1e-12 * (high - low).abs().max(1.0);
            (x - low).abs() <= slack || (x - high).abs() <= slack
        };
        on(u, spline.domain_u()) || on(v, spline.domain_v())
    };
    let last = stripe.sections.len().saturating_sub(1);
    for (index, section) in stripe.sections.iter().enumerate() {
        let uv1 = ps1.project_point(section.p1);
        let uv2 = ps2.project_point(section.p2);
        let at_end = pivoted && (index == 0 || index == last);
        if tangent1 && !(at_end && past_spline(&surface1, uv1)) {
            max_contact = max_contact.max((ps1.evaluate(uv1.0, uv1.1) - section.p1).length());
        }
        if tangent2 && !(at_end && past_spline(&surface2, uv2)) {
            max_contact = max_contact.max((ps2.evaluate(uv2.0, uv2.1) - section.p2).length());
        }
        max_radius = max_radius
            .max(((section.center - section.p1).length() - section.radius).abs())
            .max(((section.center - section.p2).length() - section.radius).abs());
        let radial1 = (section.p1 - section.center).normalize();
        let radial2 = (section.p2 - section.center).normalize();
        // A contact standing still between two sections is a cut crossing's
        // point (`cut_contact_crossing`): the arcs there pass through it
        // without touching the face tangentially.
        let still = |side: fn(&crate::section::CircSection) -> Point3| {
            [index.checked_sub(1), Some(index + 1)]
                .into_iter()
                .flatten()
                .filter_map(|k| stripe.sections.get(k))
                .any(|other| (side(other) - side(section)).length() <= 1e-12)
        };
        let (still1, still2) = (still(|s| s.p1), still(|s| s.p2));
        if let (Ok(radial1), Ok(radial2)) = (radial1, radial2) {
            if tangent1 && !still1 {
                max_tangent =
                    max_tangent.max((1.0 - ps1.normal(uv1.0, uv1.1).dot(radial1).abs()).abs());
            }
            if tangent2 && !still2 {
                max_tangent =
                    max_tangent.max((1.0 - ps2.normal(uv2.0, uv2.1).dot(radial2).abs()).abs());
            }
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
/// The band is tangent to the wall along the wall-contact circle, so there its
/// outward normal is the wall's: the band is reversed when the torus's own
/// normal (out of the tube) opposes it. That holds for a convex rim, whose
/// tube lies in the material (a rod's top, a bore's mouth), and a concave one,
/// whose tube lies in the air (a post's foot, a round pocket's floor). Without
/// a wall normal, the band is taken as convex: its axial direction at mid-arc
/// points from the wall contact back toward the plate.
fn torus_band_needs_reversal(
    torus: &brepkit_math::surfaces::ToroidalSurface,
    rim: &ClosedRimInfo,
    wall: (&FaceSurface, bool),
) -> bool {
    let (wall_surface, wall_reversed) = wall;
    let contact = rim.wall_circle.evaluate(0.0);
    if let Some((u, v)) = wall_surface.project_point(contact) {
        let wall_out = wall_surface.normal(u, v);
        let wall_out = if wall_reversed { -wall_out } else { wall_out };
        let (tu, tv) = torus.project_point(contact);
        let along = torus.normal(tu, tv).dot(wall_out);
        if along.abs() > 0.5 {
            return along < 0.0;
        }
    }
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
    let mut result = compute_stripe_for_spine(
        topo,
        Some(adjacency),
        adj_faces[0],
        adj_faces[1],
        spine,
        law,
    )?;
    run_on_to_same_surface(topo, adjacency, &mut result.stripe)?;
    Ok(result)
}

/// A closed round's contact on a curved face that reaches the face's far
/// loop, or runs past it, where the one face beyond that loop lies on the
/// same surface (a ball cut below its equator, whose two faces meet there):
/// the contact lies on the face beyond. The band then meets that face in
/// place of the narrow face, which goes, as a face [`swallow_narrow_face`]
/// finds does. The band and its contacts stay as they are. The contact
/// moves only where it is proved past the far loop's plane
/// ([`contact_against_far_plane`]); one that reaches the far loop and is
/// not proved past it is refused: no trim of either face alone holds it.
fn run_on_to_same_surface(
    topo: &Topology,
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    stripe: &mut Stripe,
) -> Result<(), BlendError> {
    const SAMPLES: u32 = 64;
    if !stripe.spine.is_closed() {
        return Ok(());
    }
    let spine_edges: HashSet<EdgeId> = stripe.spine.edges().iter().copied().collect();
    let length = stripe.spine.length();
    let mut rim = Vec::with_capacity(SAMPLES as usize + 1);
    for k in 0..=SAMPLES {
        rim.push(
            stripe
                .spine
                .evaluate(topo, length * f64::from(k) / f64::from(SAMPLES))?,
        );
    }
    let size = {
        let bbox = Aabb3::from_points(rim.iter().copied());
        (bbox.max - bbox.min).length()
    };
    let on_tol = 1e-7 * (1.0 + size);
    let same_tol = 1e-6 * (1.0 + size);
    for side in 0..2 {
        let (face_id, other) = if side == 0 {
            (stripe.face1, stripe.face2)
        } else {
            (stripe.face2, stripe.face1)
        };
        let face = topo.face(face_id)?;
        let surface = face.surface();
        let Some(period) = face_u_period(surface) else {
            continue;
        };
        // The test below reads `v` as it comes, which a surface closed in
        // `v` (a torus) wraps between the rim and the far loop.
        if crate::builder_utils::face_v_period(surface).is_some() {
            continue;
        }
        let loops: Vec<Vec<OrientedEdge>> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .map(|wire| topo.wire(wire).map(|w| w.edges().to_vec()))
            .collect::<Result<_, _>>()?;
        let is_rim =
            |edges: &[OrientedEdge]| edges.iter().all(|oe| spine_edges.contains(&oe.edge()));
        let far = match &loops[..] {
            [a, b] if is_rim(a) => b,
            [a, b] if is_rim(b) => a,
            _ => continue,
        };
        if far.iter().any(|oe| spine_edges.contains(&oe.edge())) {
            continue;
        }
        let mut beyond = None;
        for oe in far {
            let next = match adjacency.faces_for_edge(oe.edge()) {
                [a, b] if *a == face_id => *b,
                [a, b] if *b == face_id => *a,
                _ => {
                    beyond = None;
                    break;
                }
            };
            if next == other || next == face_id || beyond.is_some_and(|b| b != next) {
                beyond = None;
                break;
            }
            beyond = Some(next);
        }
        let Some(beyond) = beyond else {
            continue;
        };
        let contact_curve = if side == 0 {
            stripe.contact1.clone()
        } else {
            stripe.contact2.clone()
        };
        let (t0, t1) = contact_curve.domain();
        let contact: Vec<Point3> = (0..=SAMPLES)
            .map(|k| contact_curve.evaluate((t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0)))
            .collect();
        // The face beyond lies on the same surface, facing the same way, at
        // the rim and at the contact.
        let beyond_face = topo.face(beyond)?;
        let outward = |reversed: bool, n: Vec3| if reversed { -n } else { n };
        let same = rim.iter().chain(&contact).all(|&q| {
            let (Some((u, v)), Some((bu, bv))) = (
                surface.project_point(q),
                beyond_face.surface().project_point(q),
            ) else {
                return false;
            };
            let Some(foot) = beyond_face.surface().evaluate(bu, bv) else {
                return false;
            };
            let n = outward(face.is_reversed(), surface.normal(u, v));
            let nb = outward(
                beyond_face.is_reversed(),
                beyond_face.surface().normal(bu, bv),
            );
            (foot - q).length() <= same_tol && n.dot(nb) > 1.0 - 1e-6
        });
        if !same {
            continue;
        }
        // The rim and the far loop in the surface's parameters, each a run
        // once round in u.
        let mut far_points = Vec::new();
        for oe in far {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (s0, s1) = edge.curve().domain_with_endpoints(a, b);
            for k in 0..=SAMPLES {
                let f = f64::from(k) / f64::from(SAMPLES);
                let f = if oe.is_forward() { f } else { 1.0 - f };
                far_points.push(
                    edge.curve()
                        .evaluate_with_endpoints((s1 - s0).madd(f, s0), a, b),
                );
            }
        }
        // The contact moves to the face beyond only where that is proved:
        // short of the far loop's plane it stays, and where neither is
        // proved, a contact any station of which reaches the far loop is
        // refused.
        match contact_against_far_plane(&rim, &far_points, &contact_curve, on_tol) {
            FarSide::Beyond => {
                log::debug!("closed round: contact on {face_id:?} runs on to {beyond:?}");
                if side == 0 {
                    stripe.face1 = beyond;
                } else {
                    stripe.face2 = beyond;
                }
                continue;
            }
            FarSide::Short => continue,
            FarSide::Unsure => {}
        }
        let (Some(rim_run), Some(far_run)) = (
            RunRound::new(surface, &rim, period),
            RunRound::new(surface, &far_points, period),
        ) else {
            continue;
        };
        let mut reached = false;
        for &q in &contact {
            let Some((u, v)) = surface.project_point(q) else {
                return Ok(());
            };
            let (v_far, v_rim) = (far_run.v_at(u), rim_run.v_at(u));
            let on = surface
                .evaluate(u, v_far)
                .is_some_and(|p| (p - q).length() <= on_tol);
            reached |= on || (v - v_far) * (v_far - v_rim) > 0.0;
        }
        if !reached {
            continue;
        }
        return Err(BlendError::PlanningFailure {
            reason: format!(
                "the fillet's contact on face {} reaches its far loop, and it is not proved past it",
                side + 1
            ),
        });
    }
    Ok(())
}

/// Where a closed contact lies against the plane of its face's far loop.
#[derive(Debug, PartialEq, Eq)]
enum FarSide {
    /// All of it past the plane, or on it, away from the rim.
    Beyond,
    /// All of it short of the plane, on the rim's side.
    Short,
    /// Neither proved: the far loop is not plane, the rim does not lie on
    /// one side of it, or the contact's control points lie on both sides.
    Unsure,
}

/// Where `contact` lies against the plane the far loop (`far`, points of
/// it in order round) lies in, read against the side the rim (`rim`) lies
/// on, within `tol`. A rational curve whose weights are all positive lies
/// in the hull of its control points, so where they all lie past the
/// plane, or on it, so does the whole curve.
fn contact_against_far_plane(
    rim: &[Point3],
    far: &[Point3],
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
    tol: f64,
) -> FarSide {
    let Some(&first) = far.first() else {
        return FarSide::Unsure;
    };
    // Newell's normal of the far loop.
    let mut normal = Vec3::new(0.0, 0.0, 0.0);
    for (k, &p) in far.iter().enumerate() {
        let q = far[(k + 1) % far.len()];
        normal += (p - first).cross(q - first);
    }
    let Ok(normal) = normal.normalize() else {
        return FarSide::Unsure;
    };
    let height = |p: Point3| normal.dot(p - first);
    if far.iter().any(|&p| height(p).abs() > tol) {
        return FarSide::Unsure;
    }
    let rim_side = if rim.iter().all(|&p| height(p) < -tol) {
        -1.0
    } else if rim.iter().all(|&p| height(p) > tol) {
        1.0
    } else {
        return FarSide::Unsure;
    };
    if contact.weights().iter().any(|&w| w <= 0.0) {
        return FarSide::Unsure;
    }
    let past: Vec<f64> = contact
        .control_points()
        .iter()
        .map(|&p| -rim_side * height(p))
        .collect();
    if past.iter().all(|&h| h >= -tol) {
        FarSide::Beyond
    } else if past.iter().all(|&h| h < -tol) {
        FarSide::Short
    } else {
        FarSide::Unsure
    }
}

/// A loop on a surface that closes round in u, read in the surface's
/// parameters as `v` against `u`: it must run once round, always the same
/// way.
struct RunRound {
    /// `(u, v)` in order of increasing `u`, over one period from the first.
    points: Vec<(f64, f64)>,
    period: f64,
}

impl RunRound {
    fn new(surface: &FaceSurface, points: &[Point3], period: f64) -> Option<Self> {
        let mut run: Vec<(f64, f64)> = Vec::with_capacity(points.len());
        for &p in points {
            let (mut u, v) = surface.project_point(p)?;
            if let Some(&(last, _)) = run.last() {
                u -= period * ((u - last) / period).round();
            }
            run.push((u, v));
        }
        let (&(first, _), &(last, _)) = (run.first()?, run.last()?);
        if ((last - first).abs() - period).abs() > 1e-6 * period {
            return None;
        }
        if last < first {
            run.reverse();
        }
        if run.windows(2).any(|w| w[1].0 < w[0].0) {
            return None;
        }
        Some(Self {
            points: run,
            period,
        })
    }

    fn v_at(&self, u: f64) -> f64 {
        let base = self.points[0].0;
        let u = base + (u - base).rem_euclid(self.period);
        let at = self.points.partition_point(|&(pu, _)| pu < u);
        match (
            at.checked_sub(1).and_then(|i| self.points.get(i)),
            self.points.get(at),
        ) {
            (Some(&(ua, va)), Some(&(ub, vb))) if ub > ua => va + (vb - va) * (u - ua) / (ub - ua),
            (_, Some(&(_, v))) | (Some(&(_, v)), None) => v,
            (None, None) => 0.0,
        }
    }
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
    let mut result = compute_stripe_for_spine(
        topo,
        Some(adjacency),
        contour.side1,
        contour.side2,
        contour.spine.clone(),
        &law,
    )?;
    run_on_to_same_surface(topo, adjacency, &mut result.stripe)?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn compute_stripe_for_spine(
    topo: &Topology,
    adjacency: Option<&brepkit_topology::adjacency::AdjacencyIndex>,
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
            // An exact band that misses its faces' tolerance is walked
            // instead, as a pair no fast path takes is.
            let fits = match check_stripe_residuals(topo, &result.stripe) {
                Ok(()) => true,
                Err(error) => {
                    log::debug!("fillet: the exact band misses its faces ({error}); walking it");
                    false
                }
            };
            // A closed round whose contact runs off its face (a tube's rim
            // rounded wider than the wall is thick) is not this exact band:
            // its trim built a valid solid with material added, not taken
            // off. The walk below finds such a round's ball pivoting on the
            // face's far rim, or refuses it.
            // An open one is too where the face it runs off is a narrow
            // band between the rim and a far rim (a ledge cut through by
            // the plane the round ends on).
            // (An open one is kept with its contacts on its faces' edges, as
            // at a round exactly the face's width: only a contact run off
            // the face sends it to the walk.)
            let on_faces = check_contacts_on_faces(topo, face1, face2, &result.stripe.sections)
                .is_ok()
                && (!spine.is_closed()
                    || contacts_inside_plane_faces(topo, face1, face2, &result.stripe.sections)?);
            let open_band = || -> Result<bool, BlendError> {
                let Some(adjacency) = adjacency else {
                    return Ok(false);
                };
                Ok(!swallow_narrow_face(
                    topo,
                    adjacency,
                    face1,
                    face2,
                    &spine,
                    &result.stripe.sections,
                )?
                .is_empty())
            };
            if fits && (on_faces || (!spine.is_closed() && !open_band()?)) {
                return Ok(result);
            }
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
    // The walker rolls its ball on the side each normal points to, so a plane
    // faces the ball: into the material at a convex edge, where the other
    // face reaches into that side of its plane, and out of it at a concave
    // one.
    // A curved surface is turned the same way, through a view that negates
    // its normal: left as it was, a B-spline wall's outward normal put the
    // ball outside the wall at a convex rim. Its normal is read at the
    // edge's middle, where `into_face_from_edge` reads the other face.
    let middle = {
        let e = topo.edge(spine.edges()[0])?;
        let (ps, pe) = (
            topo.vertex(e.start())?.point(),
            topo.vertex(e.end())?.point(),
        );
        let (t0, t1) = e.curve().domain_with_endpoints(ps, pe);
        e.curve()
            .evaluate_with_endpoints(f64::midpoint(t0, t1), ps, pe)
    };
    let flips_toward_ball = |surf: &brepkit_topology::face::FaceSurface,
                             reversed: bool,
                             other: FaceId|
     -> Result<bool, BlendError> {
        let normal = if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = surf {
            *normal
        } else {
            let (u, v) = surf.project_point(middle).unwrap_or((0.0, 0.0));
            surf.normal(u, v)
        };
        let outward = if reversed { -normal } else { normal };
        let convex = match crate::builder_utils::into_face_from_edge(topo, other, spine.edges()[0])?
        {
            Some(into_other) => into_other.dot(-outward) > 0.0,
            None => analytic::material_side_witness(topo, other, -outward, middle)? > 1e-9,
        };
        Ok(reversed != convex)
    };
    let flip1 = flips_toward_ball(&surf1, face1_reversed, face2)?;
    let flip2 = flips_toward_ball(&surf2, face2_reversed, face1)?;
    let is_plane = |surf: &brepkit_topology::face::FaceSurface| {
        matches!(surf, brepkit_topology::face::FaceSurface::Plane { .. })
    };
    let oriented_surf1 = if flip1 && is_plane(&surf1) {
        orient_plane_surface(&surf1)
    } else {
        surf1
    };
    let oriented_surf2 = if flip2 && is_plane(&surf2) {
        orient_plane_surface(&surf2)
    } else {
        surf2
    };
    let mut adapter1 = None;
    let mut adapter2 = None;

    let base1 = surface_ref_or_adapter(&oriented_surf1, &mut adapter1);
    let base2 = surface_ref_or_adapter(&oriented_surf2, &mut adapter2);
    let flipped1 = FlippedNormalSurface::new(base1);
    let flipped2 = FlippedNormalSurface::new(base2);
    let ps1: &dyn brepkit_math::traits::ParametricSurface = if flip1 && !is_plane(&oriented_surf1) {
        &flipped1
    } else {
        base1
    };
    let ps2: &dyn brepkit_math::traits::ParametricSurface = if flip2 && !is_plane(&oriented_surf2) {
        &flipped2
    } else {
        base2
    };

    let config = WalkerConfig::default();

    let closed = spine.is_closed();
    let plane_normal = |surf: &FaceSurface| match surf {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        _ => None,
    };
    let planes = [plane_normal(&oriented_surf1), plane_normal(&oriented_surf2)];
    // Where a round's contact on a plane turns back on itself past a tip a
    // little tighter than the round, the crossing it makes is cut out, and
    // the band is built in three pieces about the cut.
    let mut crossing_cut: Option<[usize; 2]> = None;
    let (walk_result, mut beyond) = if let RadiusLaw::Constant(r) = law {
        let blend = ConstRadBlend { radius: *r };
        let walker = Walker::new(&blend, ps1, ps2, &spine, topo, config);
        let start = walker.find_start(0.0)?;
        // A closed round whose stations square to the spine fold over the
        // curve of its ball's centres is walked along that curve instead.
        let mut result = match walker.walk(start, 0.0, spine.length()) {
            Ok(result) => result,
            Err(error @ BlendError::WalkingFailure { .. }) if closed => {
                walker.walk_by_centres(start).map_err(|_| error)?
            }
            Err(error) => return Err(error),
        };
        if !closed
            && check_contacts_run_on(&result.sections).is_err()
            && let Some(cut) =
                cut_contact_crossing(&walker, spine.length(), &result.sections, planes)
        {
            result.sections = cut.sections;
            crossing_cut = Some(cut.bounds);
        }
        let beyond = walk_beyond(&walker, start, &result, closed);
        (result, beyond)
    } else {
        let evol = EvolRadBlend {
            law: mirror_law(law),
        };
        let walker = Walker::new(&evol, ps1, ps2, &spine, topo, config);
        let start = walker.find_start(0.0)?;
        let result = walker.walk(start, 0.0, spine.length())?;
        let beyond = walk_beyond(&walker, start, &result, closed);
        (result, beyond)
    };

    let mut walk_result = walk_result;
    // A closed round wider than a narrow face (a thin shell's rim or a
    // counterbore's ledge rounded past its width, a thin disc's edge past
    // its thickness) runs its contact off the face, past its far rim: the
    // ball, resting on the other face, pivots on that rim, the band meets
    // the face beyond there in place of the face, and the face goes.
    // An open round does so where the face is a band between the rim and a
    // far rim, its ends in the round's end sections (see
    // [`open_band_far_run`]); with no such face, its contacts are left as
    // walked.
    let off_faces = check_contacts_on_faces(topo, face1, face2, &walk_result.sections).err();
    let candidates = match (&off_faces, adjacency) {
        (Some(_), Some(adjacency)) => {
            swallow_narrow_face(topo, adjacency, face1, face2, &spine, &walk_result.sections)?
        }
        _ => Vec::new(),
    };
    let swallowed = if let Some(error) = off_faces
        && (spine.is_closed() || !candidates.is_empty())
    {
        // Each face the contacts run off is tried in turn: the first whose
        // pivot keeps both contacts on their faces.
        let mut pivoted = None;
        for found in candidates {
            let Some(sections) = pivot_on_far_loop(
                topo,
                &found,
                ps1,
                ps2,
                law,
                &walk_result.sections,
                walk_result.sections.len(),
            )?
            else {
                continue;
            };
            let (on1, on2) = if found.side == 0 {
                (found.beyond, face2)
            } else {
                (face1, found.beyond)
            };
            // An open round's end sections may pivot past the far run's
            // ends, on to its extension, where an end face leans across
            // them: the end face cuts them back (`walked_run_back`). The
            // sections between must keep both contacts on their faces.
            let inner = if spine.is_closed() || sections.len() < 3 {
                &sections[..]
            } else {
                &sections[1..sections.len() - 1]
            };
            if check_contacts_on_faces(topo, on1, on2, inner).is_ok() {
                pivoted = Some((found, sections));
                break;
            }
        }
        // An open round none of whose candidates pivots is left as walked,
        // as one with no candidate is.
        match pivoted {
            Some((found, sections)) => {
                walk_result.sections = sections;
                // The sections the ball rolls through past the spine's ends,
                // which carry the round on to a tilted cap, pivot on the far
                // run too.
                for past in &mut beyond {
                    *past = pivot_on_far_loop(topo, &found, ps1, ps2, law, past, 0)?
                        .unwrap_or_default();
                }
                Some(found)
            }
            None if spine.is_closed() => return Err(error),
            None => None,
        }
    } else {
        None
    };
    let (face1, face2) = match &swallowed {
        Some(found) if found.side == 0 => (found.beyond, face2),
        Some(found) => (face1, found.beyond),
        None => (face1, face2),
    };
    let beyond_surface = swallowed
        .as_ref()
        .map(|found| topo.face(found.beyond).map(|f| f.surface().clone()))
        .transpose()?;
    let mut beyond_adapter = None;
    let beyond_ps = beyond_surface
        .as_ref()
        .map(|surface| surface_ref_or_adapter(surface, &mut beyond_adapter));
    let (ps1, period1) = match (&swallowed, beyond_ps, &beyond_surface) {
        (Some(found), Some(ps), Some(surface)) if found.side == 0 => (ps, face_u_period(surface)),
        _ => (ps1, face_u_period(&oriented_surf1)),
    };
    let (ps2, period2) = match (&swallowed, beyond_ps, &beyond_surface) {
        (Some(found), Some(ps), Some(surface)) if found.side == 1 => (ps, face_u_period(surface)),
        _ => (ps2, face_u_period(&oriented_surf2)),
    };
    // A cut contact turns at its crossing point, where the band's pieces
    // meet: each piece runs on by itself.
    match crossing_cut.filter(|_| swallowed.is_none()) {
        Some([first, last]) => {
            let all = &walk_result.sections;
            for piece in [&all[..=first], &all[first..=last], &all[last..]] {
                check_contacts_run_on(piece)?;
            }
        }
        None => check_contacts_run_on(&walk_result.sections)?,
    }
    if closed {
        close_sections(&mut walk_result.sections);
    }
    // The band passes through its walked sections, so its boundary rows pass
    // through the walked contacts. Where the interpolation is refused, the
    // band takes the sections as its control points.
    // A round with its contact's crossing cut out is three bands end to end:
    // up to the cut, across it, and on from it, joined at the cut's bounds.
    let blend_surface = match crossing_cut.filter(|_| swallowed.is_none()) {
        Some([first, last]) => {
            let all = &walk_result.sections;
            let pieces = [&all[..=first], &all[first..=last], &all[last..]];
            let mut bands = Vec::with_capacity(3);
            let mut lengths = Vec::with_capacity(3);
            for piece in pieces {
                bands.push(interpolate_blend_surface(piece)?);
                lengths.push(
                    piece
                        .windows(2)
                        .map(|w| {
                            (w[1].p1 - w[0].p1)
                                .length()
                                .max((w[1].p2 - w[0].p2).length())
                        })
                        .sum::<f64>(),
                );
            }
            join_bands(&bands, &lengths)?
        }
        None => interpolate_blend_surface(&walk_result.sections)
            .or_else(|_| approximate_blend_surface(&walk_result.sections))?,
    };
    // The band's sides are its own boundary rows: between two sections, a
    // polyline through the contacts leaves the band.
    let contact1 = band_side(&blend_surface, 0)?;
    let contact2 = band_side(&blend_surface, blend_surface.control_points().len() - 1)?;
    let blend_face_surface = brepkit_topology::face::FaceSurface::Nurbs(blend_surface);

    let pcurve1 = build_pcurve_from_contact(ps1, &contact1, period1)?;
    let pcurve2 = build_pcurve_from_contact(ps2, &contact2, period2)?;

    let stripe = Stripe {
        spine,
        surface: blend_face_surface,
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        beyond,
        sections: walk_result.sections,
    };

    check_stripe_residuals(topo, &stripe)?;
    Ok(StripeResult {
        stripe,
        new_edges: Vec::new(),
    })
}

/// A narrow face a round swallows (see
/// [`swallow_narrow_face`]).
struct SwallowedFace {
    /// Which side of the stripe the face was (0: `face1`).
    side: usize,
    /// The face beyond it, which the band meets in its place.
    beyond: FaceId,
    /// The plane face's far loop, which it shares with `beyond`.
    far: Vec<EdgeId>,
}

/// The faces a round's contact runs off past their far loop all
/// round, each the face a narrow annulus between the rim
/// rounded and one other loop (a thin shell's rim, a counterbore's ledge),
/// or a wall lower than the round running up its seam from the rim to its
/// far rim (a thin disc's edge), or, for an open round, a narrow plane band
/// between the rim and a far rim whose ends lie in the round's end sections
/// (a ledge cut through by the plane the round ends on), its far loop all
/// shared with one face beyond. Each section's arc, from the contact on the other face towards
/// the narrow face's, then crosses the face beyond's surface before it
/// reaches the narrow face's: it is cut there.
fn swallow_narrow_face(
    topo: &Topology,
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    face1: FaceId,
    face2: FaceId,
    spine: &Spine,
    sections: &[crate::section::CircSection],
) -> Result<Vec<SwallowedFace>, BlendError> {
    let mut found = Vec::new();
    let spine_edges: HashSet<EdgeId> = spine.edges().iter().copied().collect();
    'sides: for (side, (plane, wall)) in [(face1, face2), (face2, face1)].into_iter().enumerate() {
        let face = topo.face(plane)?;
        let loops: Vec<Vec<EdgeId>> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .map(|wire| {
                topo.wire(wire)
                    .map(|w| w.edges().iter().map(OrientedEdge::edge).collect())
            })
            .collect::<Result<_, _>>()?;
        let rim_loop = |edges: &[EdgeId]| edges.iter().all(|e| spine_edges.contains(e));
        let far: Vec<EdgeId> = if !spine.is_closed() {
            // An open round of a narrow band's edge (a ledge between two
            // coaxial walls, cut through by a plane the round ends on): the
            // band's one loop runs along the rim, out along an end edge in
            // the round's end section, back along its far rim and in along
            // the other end edge.
            let [wire] = &loops[..] else {
                continue;
            };
            match open_band_far_run(topo, spine, wire)? {
                Some(far) => far,
                None => continue,
            }
        } else if face.surface().is_planar() {
            let [first, second] = &loops[..] else {
                continue;
            };
            match (rim_loop(first), rim_loop(second)) {
                (true, false) => second.clone(),
                (false, true) => first.clone(),
                _ => continue,
            }
        } else {
            // A wall lower than the round (a thin disc's rim rounded past
            // its thickness): one wire running round the rim, up its seam,
            // round its far rim and back down the seam.
            let [wire] = &loops[..] else {
                continue;
            };
            let count = |e: EdgeId| wire.iter().filter(|&&x| x == e).count();
            let (rim, far): (Vec<EdgeId>, Vec<EdgeId>) = wire
                .iter()
                .copied()
                .filter(|&e| count(e) == 1)
                .partition(|e| spine_edges.contains(e));
            if rim.is_empty() || far.is_empty() || wire.len() == rim.len() + far.len() {
                continue;
            }
            far
        };
        let far = &far;
        let mut beyond = None;
        let mut shared = true;
        for &edge in far {
            let faces = adjacency.faces_for_edge(edge);
            let other = match faces {
                [a, b] if *a == plane => *b,
                [a, b] if *b == plane => *a,
                _ => {
                    shared = false;
                    break;
                }
            };
            if other == wall || beyond.is_some_and(|b| b != other) {
                shared = false;
                break;
            }
            beyond = Some(other);
        }
        let (true, Some(beyond)) = (shared, beyond) else {
            continue;
        };
        let beyond_face = topo.face(beyond)?;
        let surface = beyond_face.surface();
        let outward = if beyond_face.is_reversed() { -1.0 } else { 1.0 };
        // Signed distance off the face beyond's surface, outward of the
        // solid, and the surface's (u, v) there.
        let off = |q: Point3| -> Option<(f64, (f64, f64))> {
            if let FaceSurface::Plane { normal, d } = surface {
                let value = (normal.dot(Vec3::new(q.x(), q.y(), q.z())) - d) / normal.length();
                return Some((outward * value, (0.0, 0.0)));
            }
            let (u, v) = surface.project_point(q)?;
            let foot = surface.evaluate(u, v)?;
            let n = surface.normal(u, v);
            Some((outward * (q - foot).dot(n), (u, v)))
        };
        // Each section's arc must cross the face beyond, from inside the
        // solid out, for the face to be swallowed.
        for section in sections {
            let (on_plane, on_wall) = if side == 0 {
                (section.p1, section.p2)
            } else {
                (section.p2, section.p1)
            };
            let (a, b) = (on_wall - section.center, on_plane - section.center);
            let angle = a
                .normalize()
                .ok()
                .zip(b.normalize().ok())
                .map(|(a, b)| a.dot(b).clamp(-1.0, 1.0).acos());
            let Some(angle) = angle.filter(|&angle| angle > 1e-9) else {
                continue 'sides;
            };
            // The arc from the wall's contact to the plane's, by slerp.
            let at = |f: f64| {
                let (sa, sb) = (((1.0 - f) * angle).sin(), (f * angle).sin());
                section.center + (a * sa + b * sb) * (1.0 / angle.sin())
            };
            // From either side: past a ledge's far rim the face beyond (a
            // wider bore's wall) stands over the wall's side of its surface.
            let Some((start, _)) = off(at(0.0)) else {
                continue 'sides;
            };
            if start == 0.0 {
                continue 'sides;
            }
            let inside = start < 0.0;
            let samples = 32_u32;
            let mut low = 0.0;
            let mut high = None;
            for k in 1..=samples {
                let f = f64::from(k) / f64::from(samples);
                let Some((value, _)) = off(at(f)) else {
                    continue 'sides;
                };
                if (value < 0.0) != inside {
                    high = Some(f);
                    break;
                }
                low = f;
            }
            let Some(mut high) = high else {
                continue 'sides;
            };
            for _ in 0..60 {
                let mid = 0.5 * (low + high);
                match off(at(mid)) {
                    Some((value, _)) if (value < 0.0) == inside => low = mid,
                    Some(_) => high = mid,
                    None => continue 'sides,
                }
            }
            if off(at(0.5 * (low + high))).is_none() {
                continue 'sides;
            }
        }
        found.push(SwallowedFace {
            side,
            beyond,
            far: far.clone(),
        });
    }
    Ok(found)
}

/// The far run of a narrow band an open round's spine runs along: `wire`
/// (the band's one loop) is the spine's edges, one straight edge out from
/// each of the spine's ends (in that end's section plane, or leaning across
/// it where the end face is tilted to the round), and the rest, its far
/// rim. `None` when the loop is not that.
fn open_band_far_run(
    topo: &Topology,
    spine: &Spine,
    wire: &[EdgeId],
) -> Result<Option<Vec<EdgeId>>, BlendError> {
    let spine_edges: HashSet<EdgeId> = spine.edges().iter().copied().collect();
    if wire.iter().filter(|e| spine_edges.contains(e)).count() != spine_edges.len() {
        return Ok(None);
    }
    let length = spine.length();
    let ends = [spine.evaluate(topo, 0.0)?, spine.evaluate(topo, length)?];
    let size = length.max(1e-9);
    let at_end = |p: Point3| ends.iter().position(|q| (p - *q).length() < 1e-6 * size);
    let mut far = Vec::new();
    let mut end_edges = [false; 2];
    for &edge_id in wire {
        if spine_edges.contains(&edge_id) {
            continue;
        }
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let end = match (at_end(a), at_end(b)) {
            (Some(end), None) | (None, Some(end)) => end,
            (None, None) => {
                far.push(edge_id);
                continue;
            }
            (Some(_), Some(_)) => return Ok(None),
        };
        // The end edge is straight: in the end's section plane the band's
        // end section closes the end face where it was; leaning across it
        // (an end face tilted to the round), the end face cuts the band
        // back (`walked_run_back`).
        if !matches!(edge.curve(), EdgeCurve::Line) || end_edges[end] {
            return Ok(None);
        }
        end_edges[end] = true;
    }
    Ok((end_edges == [true, true] && !far.is_empty()).then_some(far))
}

/// The round's sections where it swallows a narrow face: the ball
/// rests on the wall and pivots on the face's far rim (`found.far`), each
/// in its section's plane, so the band runs from its contact on the wall to
/// that rim, where it meets the face beyond. `sections` are the walk on the
/// plane, whose contacts on it lie past the rim.
fn pivot_on_far_loop(
    topo: &Topology,
    found: &SwallowedFace,
    ps1: &dyn brepkit_math::traits::ParametricSurface,
    ps2: &dyn brepkit_math::traits::ParametricSurface,
    law: &RadiusLaw,
    sections: &[crate::section::CircSection],
    required: usize,
) -> Result<Option<Vec<crate::section::CircSection>>, BlendError> {
    const SAMPLES: usize = 128;
    let RadiusLaw::Constant(radius) = law else {
        return Ok(None);
    };
    let radius = *radius;
    // The far rim as a closed polyline of curve samples, each segment
    // refined on its curve where a section's plane crosses it.
    let mut rim: Vec<(EdgeId, f64, f64)> = Vec::new();
    for &edge_id in &found.far {
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        for k in 0..SAMPLES {
            let f = |k: usize| t0 + (t1 - t0) * k as f64 / SAMPLES as f64;
            rim.push((edge_id, f(k), f(k + 1)));
        }
    }
    let at = |edge_id: EdgeId, t: f64| -> Result<Point3, BlendError> {
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        Ok(edge.curve().evaluate_with_endpoints(t, a, b))
    };
    let wall = if found.side == 0 { ps2 } else { ps1 };
    let beyond_surface = topo.face(found.beyond)?.surface().clone();
    let mut beyond_adapter = None;
    let beyond_surface = surface_ref_or_adapter(&beyond_surface, &mut beyond_adapter);
    // An open round's far run ends on the end faces, which may lean across
    // the round's end sections: the last section's plane can pass beyond
    // the run's end. The run is carried on straight past each free end, a
    // few radii along its tangent there, so every section finds its pivot;
    // the band is cut back to the end face afterwards.
    let mut ends: std::collections::HashMap<VertexId, usize> = std::collections::HashMap::new();
    for &edge_id in &found.far {
        let edge = topo.edge(edge_id)?;
        *ends.entry(edge.start()).or_default() += 1;
        *ends.entry(edge.end()).or_default() += 1;
    }
    let mut extend: Vec<(EdgeId, f64, f64)> = Vec::new();
    {
        for &edge_id in &found.far {
            let edge = topo.edge(edge_id)?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            let span = (t1 - t0).abs().max(1e-12);
            let length = (b - a).length().max(1e-12);
            // Eight radii on, in the curve's parameter.
            let reach = 8.0 * radius * span / length;
            if ends.get(&edge.start()) == Some(&1) {
                extend.push((edge_id, t0 - reach.copysign(t1 - t0), t0));
            }
            if ends.get(&edge.end()) == Some(&1) {
                extend.push((edge_id, t1, t1 + reach.copysign(t1 - t0)));
            }
        }
    }
    let at = |&(edge_id, _, _): &(EdgeId, f64, f64), t: f64| -> Result<Point3, BlendError> {
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        let (lo, hi) = (t0.min(t1), t0.max(t1));
        if t >= lo && t <= hi {
            return at(edge_id, t);
        }
        // Past an end: straight on along the curve's tangent there.
        let end = if t < lo { lo } else { hi };
        let step = 1e-6 * (hi - lo).max(1e-12);
        let inner = if t < lo { end + step } else { end - step };
        let (p, q) = (at(edge_id, end)?, at(edge_id, inner)?);
        Ok(p + (p - q) * ((t - end).abs() / step))
    };
    let pieces: Vec<((EdgeId, f64, f64), f64, f64)> = rim
        .iter()
        .chain(extend.iter())
        .map(|&piece| (piece, piece.1, piece.2))
        .collect();
    pivot_sections(
        &pieces,
        &at,
        wall,
        beyond_surface,
        found.side,
        radius,
        sections,
        false,
        required,
    )
}

/// The sections of a round whose ball rests on `wall` (the side other than
/// `side_index`) and pivots on a rim, each in its section's plane: the
/// rim's crossing of that plane nearest its contact on side `side_index`,
/// which that contact moves to (its `(u, v)` on `pivot_surface`). With
/// `follow`, each section after the first takes the crossing nearest the
/// last one's pivot and starts its ball from the last one's contact on the
/// wall, as the ball rolls on (a plane cutting a curved rim more than once,
/// a curved wall the walk's own contact would lead it off). The rim is `pieces`, each a
/// key evaluated by `at` over its parameter range. Where a plane misses the
/// rim or the ball cannot be placed against it, the sections pivoted before
/// it, if they are at least `required`, else `None`.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn pivot_sections<K>(
    pieces: &[(K, f64, f64)],
    at: &dyn Fn(&K, f64) -> Result<Point3, BlendError>,
    wall: &dyn brepkit_math::traits::ParametricSurface,
    pivot_surface: &dyn brepkit_math::traits::ParametricSurface,
    side_index: usize,
    radius: f64,
    sections: &[crate::section::CircSection],
    follow: bool,
    required: usize,
) -> Result<Option<Vec<crate::section::CircSection>>, BlendError> {
    let tol = 1e-9 * radius.max(1.0);
    let mut pivoted: Vec<crate::section::CircSection> = Vec::with_capacity(sections.len());
    for section in sections {
        let (on_plane, on_wall, wall_uv) = if side_index == 0 {
            (section.p1, section.p2, section.uv2)
        } else {
            (section.p2, section.p1, section.uv1)
        };
        let Ok(normal) = (on_plane - section.center)
            .cross(on_wall - section.center)
            .normalize()
        else {
            return Ok((pivoted.len() >= required).then_some(pivoted));
        };
        let height = |p: Point3| (p - section.center).dot(normal);
        // The rim's crossing of the section's plane nearest the plane's
        // contact.
        let mut pivot: Option<Point3> = None;
        for (key, a, b) in pieces {
            let (a, b) = (*a, *b);
            let (pa, pb) = (at(key, a)?, at(key, b)?);
            let (ha, hb) = (height(pa), height(pb));
            // A crossing at a piece's end (a section in the plane of the
            // rim's seam) reads a hair off zero on both sides of it.
            if ha * hb > 0.0 && ha.abs().min(hb.abs()) > tol {
                continue;
            }
            let (mut lo, mut hi, mut h_lo) = (a, b, ha);
            if ha * hb > 0.0 {
                (lo, hi) = if ha.abs() <= hb.abs() { (a, a) } else { (b, b) };
            }
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                let h = height(at(key, mid)?);
                if h * h_lo > 0.0 {
                    lo = mid;
                    h_lo = h;
                } else {
                    hi = mid;
                }
            }
            let q = at(key, 0.5 * (lo + hi))?;
            // Following, from the last section's pivot, so a plane cutting
            // the rim more than once takes the crossing the rim runs on to.
            let from = pivoted.last().filter(|_| follow).map_or(on_plane, |last| {
                if side_index == 0 { last.p1 } else { last.p2 }
            });
            if pivot.is_none_or(|p| (q - from).length() < (p - from).length()) {
                pivot = Some(q);
            }
        }
        let Some(pivot) = pivot else {
            return Ok((pivoted.len() >= required).then_some(pivoted));
        };
        // The ball on the wall: its centre a radius off the wall, toward
        // the ball, in the section's plane, a radius from the pivot.
        let side = (section.center - on_wall).dot(wall.normal(wall_uv.0, wall_uv.1));
        let sign = if side < 0.0 { -1.0 } else { 1.0 };
        // A wall closed round on itself (a B-spline rim's wall) is read
        // across its seam.
        let centre = |u: f64, v: f64| {
            let (u, v) = crate::walker::wrap_params(wall, u, v);
            wall.evaluate(u, v) + wall.normal(u, v) * (sign * radius)
        };
        let residual = |u: f64, v: f64| {
            let c = centre(u, v);
            (height(c), (c - pivot).length() - radius)
        };
        // Following, from the last section's contact on the wall, as the
        // ball rolls on (from the walk's own contact it can land on
        // another part of a curved wall). The first from the walk's
        // `(u, v)` on the wall, or its contact projected where the walk read
        // the face in a frame of its own (a reversed plane).
        let start = if (wall.evaluate(wall_uv.0, wall_uv.1) - on_wall).length() <= tol {
            wall_uv
        } else {
            wall.project_point(on_wall)
        };
        let (mut u, mut v) = pivoted.last().filter(|_| follow).map_or(start, |last| {
            if side_index == 0 { last.uv2 } else { last.uv1 }
        });
        let mut solved = false;
        for _ in 0..40 {
            let (f1, f2) = residual(u, v);
            if f1.abs() <= tol && f2.abs() <= tol {
                solved = true;
                break;
            }
            let h = 1e-7 * (1.0 + u.abs().max(v.abs()));
            let (a1, a2) = residual(u + h, v);
            let (b1, b2) = residual(u, v + h);
            let (j11, j21, j12, j22) = ((a1 - f1) / h, (a2 - f2) / h, (b1 - f1) / h, (b2 - f2) / h);
            let det = j11 * j22 - j12 * j21;
            if det.abs() < 1e-30 {
                break;
            }
            u -= (j22 * f1 - j12 * f2) / det;
            v -= (j11 * f2 - j21 * f1) / det;
        }
        if !solved {
            return Ok((pivoted.len() >= required).then_some(pivoted));
        }
        let pivot_uv = pivot_surface.project_point(pivot);
        let (u, v) = crate::walker::wrap_params(wall, u, v);
        let mut next = section.clone();
        next.center = centre(u, v);
        if side_index == 0 {
            next.p1 = pivot;
            next.uv1 = pivot_uv;
            next.p2 = wall.evaluate(u, v);
            next.uv2 = (u, v);
        } else {
            next.p1 = wall.evaluate(u, v);
            next.uv1 = (u, v);
            next.p2 = pivot;
            next.uv2 = pivot_uv;
        }
        pivoted.push(next);
    }
    Ok(Some(pivoted))
}

/// The ball rolled on past both ends of an open spine, out to four radii in
/// steps of a quarter radius (see [`Stripe::beyond`]).
fn walk_beyond<F: crate::blend_func::BlendFunction>(
    walker: &Walker<'_, F>,
    start: crate::blend_func::BlendParams,
    result: &crate::walker::WalkResult,
    closed: bool,
) -> [Vec<crate::section::CircSection>; 2] {
    if closed {
        return Default::default();
    }
    let past = |at_end: bool, from, section: Option<&crate::section::CircSection>| {
        section.map_or_else(Vec::new, |section| {
            walker.walk_past_end(at_end, from, 0.25 * section.radius, 4.0 * section.radius)
        })
    };
    [
        past(false, start, result.sections.first()),
        past(true, result.end_params, result.sections.last()),
    ]
}

/// Refuses a walk whose contact on either face turns back on itself: a
/// ball wider than a convex bend of its face (a round past the radius of
/// an outline's tip) still solves at every station, but its contact on the
/// face the bend is in runs to a cusp and back, the swallowtail of the
/// face's outline set in by the radius. The band through such sections
/// crosses itself, and the trimmed face's loop with it, though every edge
/// still has two faces.
fn check_contacts_run_on(sections: &[crate::section::CircSection]) -> Result<(), BlendError> {
    for (side, contact) in [
        sections.iter().map(|s| s.p1).collect::<Vec<_>>(),
        sections.iter().map(|s| s.p2).collect::<Vec<_>>(),
    ]
    .iter()
    .enumerate()
    {
        let steps: Vec<Vec3> = contact
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|d| d.length() > CONTACT_STEP_FLOOR)
            .collect();
        if let Some(i) = steps.windows(2).position(|w| w[0].dot(w[1]) < 0.0) {
            return Err(BlendError::PlanningFailure {
                reason: format!(
                    "the fillet's contact on face {} turns back on itself near section {i}: the radius is larger than the face's bend there",
                    side + 1
                ),
            });
        }
    }
    Ok(())
}

/// Contact steps shorter than this are read as standing still, not as a
/// direction (a ball pinned where a face's contact barely moves).
const CONTACT_STEP_FLOOR: f64 = 1e-9;

/// A walked round's sections with the crossing of its contact on a plane
/// face cut out, and the places of the two sections that bound the cut.
struct CrossingCut {
    sections: Vec<crate::section::CircSection>,
    /// The section where the contact reaches its crossing, and the one
    /// where it leaves it.
    bounds: [usize; 2],
}

/// Cut out the stretch where a round's contact on a plane face crosses
/// itself: an outline's tip whose bend is a little tighter than the round,
/// where the ball cannot roll round the tip.
///
/// Past the bend the contact on the plane runs back, forms a small loop and
/// crosses where it was before. The sections through that crossing bound
/// the cut: both are arcs of one ball, its centre where the ball's path
/// crosses itself, from the crossing point down to the wall. Between them
/// the contact on the plane stands still at the crossing point and the
/// contact on the wall runs on as walked; each section there is the arc
/// from the crossing point to the wall contact, tangent to the wall there.
/// At the two bounds those arcs are the ball's own.
///
/// `None` unless one side's contact crosses itself once, that side is a
/// plane, and every cut section solves.
fn cut_contact_crossing<F: crate::blend_func::BlendFunction>(
    walker: &Walker<'_, F>,
    spine_length: f64,
    sections: &[crate::section::CircSection],
    planes: [Option<Vec3>; 2],
) -> Option<CrossingCut> {
    use crate::blend_func::BlendParams;
    use crate::section::CircSection;

    const CUT_STEPS: usize = 8;
    let contact = |sec: &CircSection, side: usize| if side == 0 { sec.p1 } else { sec.p2 };
    let params = |sec: &CircSection| BlendParams {
        u1: sec.uv1.0,
        v1: sec.uv1.1,
        u2: sec.uv2.0,
        v2: sec.uv2.1,
    };
    for (side, normal) in planes.iter().enumerate() {
        let Some(normal) = *normal else {
            continue;
        };
        let e1 = normal.cross(Vec3::new(1.0, 0.0, 0.0));
        let e1 = if e1.length() < 0.1 {
            normal.cross(Vec3::new(0.0, 1.0, 0.0))
        } else {
            e1
        };
        let Ok(e1) = e1.normalize() else {
            continue;
        };
        let e2 = normal.cross(e1);
        let flat = |p: Point3| {
            (
                Vec3::new(p.x(), p.y(), p.z()).dot(e1),
                Vec3::new(p.x(), p.y(), p.z()).dot(e2),
            )
        };
        let pts: Vec<(f64, f64)> = sections.iter().map(|s| flat(contact(s, side))).collect();
        // The first two segments of the contact that cross, not neighbours.
        let cross = |a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)| {
            let r = (b.0 - a.0, b.1 - a.1);
            let s = (d.0 - c.0, d.1 - c.1);
            let den = r.0 * s.1 - r.1 * s.0;
            if den.abs() < 1e-300 {
                return None;
            }
            let q = (c.0 - a.0, c.1 - a.1);
            let ta = (q.0 * s.1 - q.1 * s.0) / den;
            let tb = (q.0 * r.1 - q.1 * r.0) / den;
            ((0.0..=1.0).contains(&ta) && (0.0..=1.0).contains(&tb)).then_some((ta, tb))
        };
        let mut found = None;
        'outer: for i in 0..pts.len().saturating_sub(1) {
            for j in i + 2..pts.len().saturating_sub(1) {
                if let Some(at) = cross(pts[i], pts[i + 1], pts[j], pts[j + 1]) {
                    found = Some((i, j, at));
                    break 'outer;
                }
            }
        }
        let Some((i, j, (fa, fb))) = found else {
            continue;
        };
        // Pin the two sections whose contacts on the plane meet.
        let length = spine_length;
        let s_of =
            |k: usize, f: f64| (sections[k].t + (sections[k + 1].t - sections[k].t) * f) * length;
        let (mut sa, mut sb) = (s_of(i, fa), s_of(j, fb));
        let (mut ga, mut gb) = (params(&sections[i]), params(&sections[j]));
        let mut solved = None;
        let h = 1e-7 * length.max(1e-9);
        for _ in 0..30 {
            let (pa, a) = walker.solve_section(sa, Some(ga))?;
            let (pb, b) = walker.solve_section(sb, Some(gb))?;
            ga = pa;
            gb = pb;
            let (ca, cb) = (flat(contact(&a, side)), flat(contact(&b, side)));
            let f = (ca.0 - cb.0, ca.1 - cb.1);
            // The walker solves each section to its own tolerance, which
            // bounds how closely two of them meet.
            if f.0.hypot(f.1) <= 1e-7 * sections[i].radius.max(1e-6) {
                solved = Some((a, b));
                break;
            }
            let (_, a2) = walker.solve_section(sa + h, Some(pa))?;
            let (_, b2) = walker.solve_section(sb + h, Some(pb))?;
            let (da, db) = (flat(contact(&a2, side)), flat(contact(&b2, side)));
            let ja = ((da.0 - ca.0) / h, (da.1 - ca.1) / h);
            let jb = ((cb.0 - db.0) / h, (cb.1 - db.1) / h);
            let det = ja.0 * jb.1 - jb.0 * ja.1;
            if det.abs() < 1e-300 {
                return None;
            }
            let dsa = (-f.0 * jb.1 + jb.0 * f.1) / det;
            let dsb = (-ja.0 * f.1 + ja.1 * f.0) / det;
            sa += dsa;
            sb += dsb;
            if !(sa.is_finite() && sb.is_finite()) || sa <= 0.0 || sb >= length || sa >= sb {
                return None;
            }
        }
        let (mut a, mut b) = solved?;
        let crossing = {
            let (p, q) = (contact(&a, side), contact(&b, side));
            Point3::new(
                f64::midpoint(p.x(), q.x()),
                f64::midpoint(p.y(), q.y()),
                f64::midpoint(p.z(), q.z()),
            )
        };
        for sec in [&mut a, &mut b] {
            if side == 0 {
                sec.p1 = crossing;
            } else {
                sec.p2 = crossing;
            }
        }
        // Between the bounds: from the crossing point to the walked wall
        // contact, tangent to the wall there.
        let mut cut = Vec::with_capacity(CUT_STEPS + 1);
        let mut guess = ga;
        for k in 1..CUT_STEPS {
            #[allow(clippy::cast_precision_loss)]
            let s = sa + (sb - sa) * (k as f64 / CUT_STEPS as f64);
            let (p, sec) = walker.solve_section(s, Some(guess))?;
            guess = p;
            let wall = contact(&sec, 1 - side);
            let inward = (sec.center - wall).normalize().ok()?;
            let chord = crossing - wall;
            let along = chord.dot(inward);
            if along <= 1e-9 * sec.radius {
                return None;
            }
            let radius = chord.dot(chord) / (2.0 * along);
            if !(radius.is_finite() && radius < 4.0 * sec.radius) {
                return None;
            }
            let mut mid = sec.clone();
            mid.center = wall + inward * radius;
            mid.radius = radius;
            if side == 0 {
                mid.p1 = crossing;
                mid.uv1 = a.uv1;
            } else {
                mid.p2 = crossing;
                mid.uv2 = a.uv2;
            }
            cut.push(mid);
        }
        let mut out: Vec<CircSection> = sections[..=i].to_vec();
        let first = out.len();
        out.push(a);
        out.extend(cut);
        let last = out.len();
        out.push(b);
        out.extend_from_slice(&sections[j + 1..]);
        // Each piece must hold enough sections for a cubic of its own.
        if first < 3 || out.len() - last < 4 {
            return None;
        }
        return Some(CrossingCut {
            sections: out,
            bounds: [first, last],
        });
    }
    None
}

/// Join bands built piece by piece along a round end to end, each piece's
/// last section the next one's first: a band only C0 at the joins, each
/// piece taking a share of the parameter range as long as its `lengths`.
fn join_bands(
    pieces: &[brepkit_math::nurbs::surface::NurbsSurface],
    lengths: &[f64],
) -> Result<brepkit_math::nurbs::surface::NurbsSurface, BlendError> {
    let fail = || BlendError::Math(brepkit_math::MathError::EmptyInput);
    let first = pieces.first().ok_or_else(fail)?;
    let degree = first.degree_v();
    let rows = first.control_points().len();
    let total: f64 = lengths.iter().sum();
    if pieces.len() != lengths.len() || total.is_nan() || total <= 0.0 {
        return Err(fail());
    }
    let mut knots = vec![0.0; degree + 1];
    let mut points: Vec<Vec<Point3>> = vec![Vec::new(); rows];
    let mut weights: Vec<Vec<f64>> = vec![Vec::new(); rows];
    let mut start = 0.0;
    for (k, (piece, &length)) in pieces.iter().zip(lengths).enumerate() {
        if piece.degree_v() != degree
            || piece.control_points().len() != rows
            || piece.knots_u() != first.knots_u()
        {
            return Err(fail());
        }
        let share = length / total;
        let pk = piece.knots_v();
        for &knot in &pk[degree + 1..pk.len() - degree - 1] {
            knots.push(start + share * knot);
        }
        start += share;
        let end = if k + 1 == pieces.len() { 1.0 } else { start };
        knots.extend(std::iter::repeat_n(
            end,
            if k + 1 == pieces.len() {
                degree + 1
            } else {
                degree
            },
        ));
        let skip = usize::from(k > 0);
        for row in 0..rows {
            points[row].extend_from_slice(&piece.control_points()[row][skip..]);
            weights[row].extend_from_slice(&piece.weights()[row][skip..]);
        }
    }
    Ok(brepkit_math::nurbs::surface::NurbsSurface::new(
        first.degree_u(),
        degree,
        first.knots_u().to_vec(),
        knots,
        points,
        weights,
    )?)
}

/// Whether every contact a band leaves on a plane face lies inside that
/// face, read against its exact outline in its plane (a box holds a tilted
/// ring's whole circle a little past its rim). `true` where an outline
/// cannot be read.
fn contacts_inside_plane_faces(
    topo: &Topology,
    face1: FaceId,
    face2: FaceId,
    sections: &[crate::section::CircSection],
) -> Result<bool, BlendError> {
    for (side, face_id) in [face1, face2].into_iter().enumerate() {
        let FaceSurface::Plane { normal, d } = *topo.face(face_id)?.surface() else {
            continue;
        };
        let Ok(n) = normal.normalize() else {
            continue;
        };
        let origin = Point3::new(0.0, 0.0, 0.0) + n * (d / normal.length());
        let seed = if n.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let Ok(x) = n.cross(seed).normalize() else {
            continue;
        };
        let y = n.cross(x);
        let Some(region) =
            brepkit_topology::planar::face_region_2d(topo, face_id, origin, x, y, 1e-7)?
        else {
            continue;
        };
        for section in sections {
            let p = if side == 0 { section.p1 } else { section.p2 };
            let q = brepkit_math::vec::Point2::new((p - origin).dot(x), (p - origin).dot(y));
            if !region.contains_clear(q, 0.0) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Refuses a walked band round a closed spine whose contacts leave their
/// faces: a ball too big for a rim (a bore's mouth wider than its plate is
/// thick, a wall shorter than the round) still solves on the faces' whole
/// surfaces, past the faces themselves. Each contact must lie within the box
/// round its face's own edges.
fn check_contacts_on_faces(
    topo: &Topology,
    face1: FaceId,
    face2: FaceId,
    sections: &[crate::section::CircSection],
) -> Result<(), BlendError> {
    let extent = |face: FaceId| -> Result<brepkit_math::aabb::Aabb3, BlendError> {
        let f = topo.face(face)?;
        let mut points = Vec::new();
        for wire in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oriented in topo.wire(wire)?.edges() {
                let edge = topo.edge(oriented.edge())?;
                let (a, b) = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                points.extend([a, b]);
                match edge.curve() {
                    // A B-spline lies within its hull: samples fall short of
                    // its bulges, where a wall's contact lies.
                    EdgeCurve::NurbsCurve(curve) => {
                        points.extend(nurbs_curve_hull(curve));
                    }
                    EdgeCurve::Line => {}
                    curve @ (EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) => {
                        let (t0, t1) = curve.domain_with_endpoints(a, b);
                        for k in 0..=CONTACT_EXTENT_SAMPLES {
                            let t =
                                t0 + (t1 - t0) * f64::from(k) / f64::from(CONTACT_EXTENT_SAMPLES);
                            points.push(curve.evaluate_with_endpoints(t, a, b));
                        }
                    }
                }
            }
        }
        // A ruled face (plane, cylinder, cone) lies within its boundary's
        // box. A sphere, torus or B-spline face can bulge past it (a cap
        // bounded only by its equator): add the surface's own bound, its
        // ball or its control net, so no contact on the face is refused.
        match f.surface() {
            FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {}
            FaceSurface::Sphere(sphere) => {
                points.extend(cube_about(sphere.center(), sphere.radius()));
            }
            FaceSurface::Torus(torus) => {
                points.extend(cube_about(
                    torus.center(),
                    torus.major_radius() + torus.minor_radius(),
                ));
            }
            FaceSurface::Nurbs(surface) => {
                points.extend(surface.control_points().iter().flatten().copied());
            }
        }
        let bbox = brepkit_math::aabb::Aabb3::from_points(points);
        let size = (bbox.max - bbox.min).length();
        // A conic sampled this finely falls short of its extent by under
        // 1e-4 of its size.
        Ok(bbox.expanded(1e-4 * size.max(1.0)))
    };
    let (box1, box2) = (extent(face1)?, extent(face2)?);
    // A B-spline face's control net bounds its whole surface, which can run far
    // past its boundary. In `(u, v)` its boundary bounds it, across each
    // direction the surface does not close in.
    let trim = |face: FaceId| -> Result<Option<UvWindow>, BlendError> {
        let f = topo.face(face)?;
        let FaceSurface::Nurbs(surface) = f.surface() else {
            return Ok(None);
        };
        let mut window = UvWindow::empty(surface);
        for oriented in topo.wire(f.outer_wire())?.edges() {
            let edge = topo.edge(oriented.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            for k in 0..=CONTACT_EXTENT_SAMPLES {
                let t = t0 + (t1 - t0) * f64::from(k) / f64::from(CONTACT_EXTENT_SAMPLES);
                let p = edge.curve().evaluate_with_endpoints(t, a, b);
                window.take(brepkit_math::traits::ParametricSurface::project_point(
                    surface, p,
                ));
            }
        }
        Ok(Some(window))
    };
    let (trim1, trim2) = (trim(face1)?, trim(face2)?);
    let on_trim =
        |window: &Option<UvWindow>, face: FaceId, p: Point3| -> Result<bool, BlendError> {
            let Some(window) = window else {
                return Ok(true);
            };
            let FaceSurface::Nurbs(surface) = topo.face(face)?.surface() else {
                return Ok(true);
            };
            Ok(
                window.holds(brepkit_math::traits::ParametricSurface::project_point(
                    surface, p,
                )),
            )
        };
    let mut past_trim = None;
    for section in sections {
        if !on_trim(&trim1, face1, section.p1)? || !on_trim(&trim2, face2, section.p2)? {
            past_trim = Some(section);
            break;
        }
    }
    if let Some(section) = past_trim.or_else(|| {
        sections
            .iter()
            .find(|s| !box1.contains_point(s.p1) || !box2.contains_point(s.p2))
    }) {
        return Err(BlendError::PlanningFailure {
            reason: format!(
                "the fillet's contact at spine t = {:.3} would land past its face",
                section.t
            ),
        });
    }
    Ok(())
}

/// Whether the segments `a0 a1` and `b0 b1` come within `gap` of each other.
fn segments_within(a0: Point3, a1: Point3, b0: Point3, b1: Point3, gap: f64) -> bool {
    let (d1, d2, r) = (a1 - a0, b1 - b0, a0 - b0);
    let (aa, ee, ff) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    let (s, t) = if aa <= f64::EPSILON && ee <= f64::EPSILON {
        (0.0, 0.0)
    } else if aa <= f64::EPSILON {
        (0.0, (ff / ee).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if ee <= f64::EPSILON {
            ((-c / aa).clamp(0.0, 1.0), 0.0)
        } else {
            let bb = d1.dot(d2);
            let denom = aa * ee - bb * bb;
            let s = if denom > f64::EPSILON {
                ((bb * ff - c * ee) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t = (bb * s + ff) / ee;
            if t < 0.0 {
                ((-c / aa).clamp(0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (((bb - c) / aa).clamp(0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };
    ((a0 + d1 * s) - (b0 + d2 * t)).length() <= gap
}

/// The `(u, v)` box a B-spline face's boundary reaches, along each direction
/// its surface does not close in.
struct UvWindow {
    u: Option<(f64, f64)>,
    v: Option<(f64, f64)>,
    margin: (f64, f64),
}

impl UvWindow {
    fn empty(surface: &brepkit_math::nurbs::surface::NurbsSurface) -> Self {
        use brepkit_math::traits::ParametricSurface;
        let (u0, u1) = surface.domain_u();
        let (v0, v1) = surface.domain_v();
        let empty = (f64::INFINITY, f64::NEG_INFINITY);
        Self {
            u: surface.closed_u().is_none().then_some(empty),
            v: surface.closed_v().is_none().then_some(empty),
            margin: (1e-3 * (u1 - u0), 1e-3 * (v1 - v0)),
        }
    }

    fn take(&mut self, (u, v): (f64, f64)) {
        if let Some((lo, hi)) = &mut self.u {
            (*lo, *hi) = (lo.min(u), hi.max(u));
        }
        if let Some((lo, hi)) = &mut self.v {
            (*lo, *hi) = (lo.min(v), hi.max(v));
        }
    }

    fn holds(&self, (u, v): (f64, f64)) -> bool {
        let within = |x: f64, range: Option<(f64, f64)>, margin: f64| {
            range.is_none_or(|(lo, hi)| x >= lo - margin && x <= hi + margin)
        };
        within(u, self.u, self.margin.0) && within(v, self.v, self.margin.1)
    }
}

/// Refuses two closed contacts on one face that cross: both rims of a wall
/// rounded by radii that add up to more than the wall's height. Each closed
/// contact must lie nearer the rim it rounds than the other contact does;
/// otherwise the two bands overlap, and the wall between them is trimmed
/// to nothing yet still kept, as a solid whose faces cross.
fn check_closed_contacts_do_not_cross(
    face_id: FaceId,
    restrictions: &[(usize, trimmer::ParametricRestriction)],
) -> Result<(), BlendError> {
    const SAMPLES: u32 = 256;
    const GAP: f64 = 1e-4;
    const FULL_ROUND: f64 = 1e-5;
    // A closed contact is a loop: its ends meet, within a small fraction of
    // its own length (a fillet near half a box's size leaves contacts a few
    // microns long whose ends lie a few microns apart).
    let closed: Vec<(Point3, &brepkit_math::nurbs::curve::NurbsCurve)> = restrictions
        .iter()
        .filter_map(|(_, r)| {
            let (first, last) = (r.contact_3d.first()?, r.contact_3d.last()?);
            let trimmer::TrimKeep::AwayFrom(rim) = r.keep else {
                return None;
            };
            let Some(EdgeCurve::NurbsCurve(curve)) = r.curve.as_ref() else {
                return None;
            };
            if (*first - *last).length() > GAP {
                return None;
            }
            let (t0, t1) = curve.domain();
            let t = |k: u32| t0 + (t1 - t0) * f64::from(k) / f64::from(SAMPLES);
            let length: f64 = (0..SAMPLES)
                .map(|k| (curve.evaluate(t(k + 1)) - curve.evaluate(t(k))).length())
                .sum();
            ((*first - *last).length() <= GAP.min(1e-3 * length)).then_some((rim, curve))
        })
        .collect();
    let distance = |p: Point3, curve: &brepkit_math::nurbs::curve::NurbsCurve| {
        brepkit_math::nurbs::projection::project_point_to_curve(curve, p, 1e-9)
            .map_or(f64::INFINITY, |found| found.distance)
    };
    // Anywhere round the loops, not just where each rim's seam reads them: a
    // wall can be tall at its seam and thin elsewhere.
    let sampled: Vec<Vec<Point3>> = closed
        .iter()
        .map(|&(_, curve)| {
            let (t0, t1) = curve.domain();
            (0..=SAMPLES)
                .map(|k| curve.evaluate(t0 + (t1 - t0) * f64::from(k) / f64::from(SAMPLES)))
                .collect()
        })
        .collect();
    // Two contacts lying on each other all round, within a small fraction of
    // their length, are a full round: a wall exactly twice the radius high,
    // rounded along both rims, kept as the strip between them, which has
    // no area (`drop_contact_slivers` takes it out, one round taking the
    // other's contact). Walked contacts through the same sections differ
    // only by their fits between them.
    let length = |points: &[Point3]| {
        points
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).length())
            .sum::<f64>()
    };
    let full_round = |i: usize, j: usize| {
        let (a, b) = (&sampled[i], &sampled[j]);
        let tol = FULL_ROUND * length(a).min(length(b));
        a.iter().all(|&p| distance(p, closed[j].1) <= tol)
            && b.iter().all(|&p| distance(p, closed[i].1) <= tol)
    };
    for (i, a) in sampled.iter().enumerate() {
        for (j, b) in sampled.iter().enumerate().skip(i + 1) {
            if full_round(i, j) {
                continue;
            }
            let meet = a.windows(2).any(|p| {
                b.windows(2)
                    .any(|q| segments_within(p[0], p[1], q[0], q[1], GAP))
            });
            if meet {
                return Err(BlendError::PlanningFailure {
                    reason: format!(
                        "two fillets on face {} cross: their radii are too large for the face between them",
                        face_id.index()
                    ),
                });
            }
        }
    }
    for (i, &(rim, own)) in closed.iter().enumerate() {
        let near = distance(rim, own);
        for (j, &(_, other)) in closed.iter().enumerate() {
            if i != j && !full_round(i.min(j), i.max(j)) && distance(rim, other) <= near {
                return Err(BlendError::PlanningFailure {
                    reason: format!(
                        "two fillets on face {} cross: their radii are too large for the face between them",
                        face_id.index()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// How much nearer another contact must lie to an edge's middle than the
/// edge's own contact for the two to read as passed each other.
const SWAP_MARGIN: f64 = 0.9;

/// How nearly a contact must run along an edge, as the cosine of the angle
/// between them, to be read as passing it.
const SWAP_PARALLEL: f64 = 0.9;

/// Refuses two contacts on one face that have passed each other: two rounds
/// along the two edges of a strip narrower than their setbacks together
/// (a ridge between two walls). The contacts are then swapped, each lying
/// nearer the other round's edge than its own, and the strip between them
/// would be trimmed to a loop that crosses itself. Two contacts that both
/// run alongside the middles of both edges are refused when each lies nearer
/// the other's edge than its own.
fn check_contacts_keep_their_edges(
    topo: &Topology,
    face_id: FaceId,
    restrictions: &[(usize, trimmer::ParametricRestriction)],
) -> Result<(), BlendError> {
    // The distance from `p` to a polyline, whether its nearest point lies
    // alongside `p` (not at one of the polyline's ends), and the cosine of
    // the angle between the polyline there and `tangent`.
    let distance = |p: Point3, tangent: Vec3, points: &[Point3]| {
        let last = points.len().saturating_sub(2);
        points
            .windows(2)
            .enumerate()
            .map(|(k, w)| {
                let (a, b) = (w[0], w[1]);
                let ab = b - a;
                let len2 = ab.dot(ab);
                let t = if len2 <= 1e-24 {
                    0.0
                } else {
                    ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
                };
                let alongside = !(k == 0 && t <= 0.0 || k == last && t >= 1.0);
                let cos = ab.normalize().map_or(0.0, |d| d.dot(tangent).abs());
                ((a + ab * t - p).length(), alongside, cos)
            })
            .fold((f64::INFINITY, false, 0.0), |best, d| {
                if d.0 < best.0 { d } else { best }
            })
    };
    // Each contact as points along its exact curve where it has one: a
    // walked contact's sampled points may be its two ends alone, whose chord
    // stands far off the contact on a curved face.
    let contacts: Vec<Vec<Point3>> = restrictions
        .iter()
        .map(|(_, restriction)| contact_points(restriction))
        .collect();
    let mut middles = Vec::with_capacity(restrictions.len());
    for (_, restriction) in restrictions {
        let edges = &restriction.source_edges;
        let middle = if edges.is_empty() || restriction.contact_3d.len() < 2 {
            None
        } else {
            let edge = topo.edge(edges[edges.len() / 2])?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            let t = f64::midpoint(t0, t1);
            edge.curve()
                .tangent_with_endpoints(t, a, b)
                .normalize()
                .ok()
                .map(|tangent| (edge.curve().evaluate_with_endpoints(t, a, b), tangent))
        };
        middles.push(middle);
    }
    // Contacts that passed each other are each nearer the other's edge than
    // their own, both running along it there. A contact merely nearer
    // another's edge is no crossing: one running past a short edge in an
    // acute corner, or across an edge's middle (four rounds meeting in a
    // square face's middle); nor are two contacts that nearly coincide (two
    // stripes' sections ending on one line), which `SWAP_MARGIN` leaves.
    let swapped = |i: usize, j: usize| {
        middles[i].is_some_and(|(middle, tangent)| {
            let (own, own_alongside, own_cos) = distance(middle, tangent, &contacts[i]);
            let (other, other_alongside, other_cos) = distance(middle, tangent, &contacts[j]);
            own_alongside
                && other_alongside
                && own_cos >= SWAP_PARALLEL
                && other_cos >= SWAP_PARALLEL
                && other < SWAP_MARGIN * own
        })
    };
    for i in 0..restrictions.len() {
        for j in i + 1..restrictions.len() {
            if swapped(i, j) && swapped(j, i) {
                return Err(BlendError::PlanningFailure {
                    reason: format!(
                        "two fillets on face {} cross: their radii are too large for the face between them",
                        face_id.index()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Refuses two contacts on a plane face that cross although their edges
/// share no vertex, when a round between them runs on smoothly from one of
/// them. An outline that runs on round a short edge tangent to its
/// neighbour and then turns a convex corner is rounded as one corner of two
/// rounds, but a radius past the short edge's length puts the corner's
/// crossing beyond the short edge, on the contact before it. The trim, which
/// ends each contact on its own neighbour's, then leaves the face a loop
/// that crosses itself, a lobe of it walked backwards, which the shell still
/// closes over (and which a reference kernel reads inside out).
fn check_apart_contacts_do_not_cross(
    topo: &Topology,
    face_id: FaceId,
    restrictions: &[(usize, trimmer::ParametricRestriction)],
) -> Result<(), BlendError> {
    let face = topo.face(face_id)?;
    let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() else {
        return Ok(());
    };
    let origin = Point3::new(0.0, 0.0, 0.0);
    let frame = brepkit_math::frame::Frame3::from_normal(origin, *normal)
        .map_err(|_| BlendError::TrimmingFailure { face: face_id })?;
    let in_plane = |p: Point3| ((p - origin).dot(frame.x), (p - origin).dot(frame.y));
    // Each round's source edges' ends, with the edge's direction there.
    let mut ends: Vec<Vec<(VertexId, Vec3)>> = Vec::with_capacity(restrictions.len());
    for (_, restriction) in restrictions {
        let mut round = Vec::new();
        for &edge_id in &restriction.source_edges {
            let edge = topo.edge(edge_id)?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            for (vertex, t) in [(edge.start(), t0), (edge.end(), t1)] {
                if let Ok(direction) = edge.curve().tangent_with_endpoints(t, a, b).normalize() {
                    round.push((vertex, direction));
                }
            }
        }
        ends.push(round);
    }
    let meet = |x: usize, y: usize| {
        ends[x]
            .iter()
            .flat_map(|&(v, a)| {
                ends[y]
                    .iter()
                    .filter(move |&&(w, _)| w == v)
                    .map(move |&(_, b)| a.dot(b).abs())
            })
            .fold(None, |best: Option<f64>, cos| {
                Some(best.map_or(cos, |b| b.max(cos)))
            })
    };
    let tangent = |cos: Option<f64>| cos.is_some_and(|cos| cos >= APART_TANGENT_COS);
    let flat: Vec<Vec<(f64, f64)>> = restrictions
        .iter()
        .map(|(_, restriction)| {
            contact_points(restriction)
                .into_iter()
                .map(in_plane)
                .collect()
        })
        .collect();
    let orient = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| {
        (q.0 - p.0).madd(r.1 - p.1, -((q.1 - p.1) * (r.0 - p.0)))
    };
    let cross = |a: &[(f64, f64)], b: &[(f64, f64)]| {
        a.windows(2).any(|s| {
            b.windows(2).any(|t| {
                orient(s[0], s[1], t[0]) * orient(s[0], s[1], t[1]) < 0.0
                    && orient(t[0], t[1], s[0]) * orient(t[0], t[1], s[1]) < 0.0
            })
        })
    };
    for i in 0..restrictions.len() {
        for j in i + 1..restrictions.len() {
            if meet(i, j).is_some() {
                continue;
            }
            let runs_on = (0..restrictions.len()).any(|k| {
                let (ik, kj) = (meet(i, k), meet(k, j));
                k != i && k != j && ik.is_some() && kj.is_some() && (tangent(ik) || tangent(kj))
            });
            if runs_on && cross(&flat[i], &flat[j]) {
                return Err(BlendError::PlanningFailure {
                    reason: format!(
                        "two fillets on face {} cross past the short edge between them: the radius is larger than that edge allows",
                        face_id.index()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// How nearly two edges must run on from each other at their shared vertex,
/// as the cosine of the angle between them, to read as one smooth outline.
const APART_TANGENT_COS: f64 = 0.999_95;

/// Samples along a contact's exact curve when reading where it runs.
const CONTACT_CURVE_SAMPLES: u32 = 64;

/// A contact's points in traversal order: along its exact curve, between its
/// first and last sampled points, when it has one; else its sampled points.
fn contact_points(restriction: &trimmer::ParametricRestriction) -> Vec<Point3> {
    let points = &restriction.contact_3d;
    let (Some(curve), Some(&first), Some(&last)) =
        (restriction.curve.as_ref(), points.first(), points.last())
    else {
        return points.clone();
    };
    let (t0, t1) = curve.domain_with_endpoints(first, last);
    (0..=CONTACT_CURVE_SAMPLES)
        .map(|k| {
            let t = (t1 - t0).madd(f64::from(k) / f64::from(CONTACT_CURVE_SAMPLES), t0);
            curve.evaluate_with_endpoints(t, first, last)
        })
        .collect()
}

/// Samples per boundary edge when bounding a face's extent.
const CONTACT_EXTENT_SAMPLES: u32 = 256;

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

/// The side of a blend band at its `row`-th row of control points, along `v`:
/// the band's own boundary curve there.
pub(crate) fn band_side(
    band: &brepkit_math::nurbs::surface::NurbsSurface,
    row: usize,
) -> Result<brepkit_math::nurbs::curve::NurbsCurve, BlendError> {
    let points = band
        .control_points()
        .get(row)
        .ok_or(BlendError::Math(brepkit_math::MathError::EmptyInput))?;
    let weights = band
        .weights()
        .get(row)
        .ok_or(BlendError::Math(brepkit_math::MathError::EmptyInput))?;
    Ok(brepkit_math::nurbs::curve::NurbsCurve::new(
        band.degree_v(),
        band.knots_v().to_vec(),
        points.clone(),
        weights.clone(),
    )?)
}

pub(crate) fn face_u_period(surface: &FaceSurface) -> Option<f64> {
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
pub(crate) fn build_planar_pcurve_from_contact(
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

/// A contact's pcurve on a curved face where `pcurve_along` finds none:
/// `build_pcurve_from_contact`'s, or where its ends project to one point,
/// the stripe's own analytic pcurve.
fn fallback_contact_pcurve(
    support: &dyn brepkit_math::traits::ParametricSurface,
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
    surface: &FaceSurface,
    analytic: &brepkit_math::curves2d::Curve2D,
) -> Result<brepkit_topology::pcurve::PCurve, BlendError> {
    use brepkit_math::curves2d::Curve2D;
    use brepkit_topology::pcurve::PCurve;
    match build_pcurve_from_contact(support, contact, face_u_period(surface)) {
        Ok(pcurve) => Ok(PCurve::new(pcurve, 0.0, 1.0)),
        Err(BlendError::Math(brepkit_math::MathError::ZeroVector)) => match analytic {
            Curve2D::Circle(_) | Curve2D::Ellipse(_) => {
                Ok(PCurve::new(analytic.clone(), 0.0, std::f64::consts::TAU))
            }
            Curve2D::Nurbs(curve) => {
                let (start, end) = curve.domain();
                Ok(PCurve::new(analytic.clone(), start, end))
            }
            Curve2D::Line(_) => Err(BlendError::Math(brepkit_math::MathError::ZeroVector)),
        },
        Err(error) => Err(error),
    }
}

/// Build a contact pcurve in a support surface's native UV frame.
///
/// Open contacts use their endpoint line. Closed periodic contacts are
/// sampled through one full carrier domain and unwrapped across the U seam;
/// projecting equal endpoints alone would collapse them to a zero vector.
pub(crate) fn build_pcurve_from_contact(
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
    scope: &trimmer::SplitScope,
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
                SplitPlan::Line => trimmer::split_edge_at(topo, scope, &oe, v_split)?,
                SplitPlan::Curve(left, right) => {
                    trimmer::split_edge_at_with_curves(topo, scope, &oe, v_split, left, right)?
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
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use brepkit_topology::adjacency::AdjacencyIndex;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// The notch triangle at a corner filleted r 6: the arc's ends on the two
    /// corner edges and the corner itself.
    const NOTCH: [(f64, f64); 3] = [(6.0, 0.0), (0.0, 0.0), (0.0, 6.0)];

    /// The nearest point of a line or an arc, in closed form, is the nearest
    /// of 200001 even samples to within the samples' spacing: on a segment,
    /// on a quarter arc tilted out of its plane's frame, run either way, and
    /// on a whole circle, from points beside, past the ends and off the
    /// plane.
    #[test]
    fn the_nearest_point_of_a_line_or_arc_is_the_nearest_sample() {
        use brepkit_math::curves::Circle3D;
        const SAMPLES: u32 = 200_000;
        let p = Point3::new;
        let tilted = Circle3D::new(p(1.0, -2.0, 0.5), Vec3::new(0.2, 0.3, 1.0), 3.0).unwrap();
        let (t0, t1) = (0.4, 0.4 + std::f64::consts::FRAC_PI_2);
        let (s, e) = (tilted.evaluate(t0), tilted.evaluate(t1));
        let whole = tilted.evaluate(1.0);
        let cases = [
            (EdgeCurve::Line, p(0.0, 0.0, 0.0), p(10.0, 2.0, -1.0)),
            (EdgeCurve::Circle(tilted.clone()), s, e),
            (EdgeCurve::Circle(tilted.clone()), e, s),
            (EdgeCurve::Circle(tilted), whole, whole),
        ];
        let points = [
            p(4.0, 3.0, 0.0),
            p(-3.0, -1.0, 2.0),
            p(12.0, 5.0, -4.0),
            p(1.0, -2.0, 6.0),
            p(3.0, 1.0, 0.7),
            p(-2.5, -4.0, 0.0),
        ];
        for (curve, a, b) in &cases {
            let (t0, t1) = curve.domain_with_endpoints(*a, *b);
            let at = |k: u32| {
                curve.evaluate_with_endpoints(
                    t0 + (t1 - t0) * f64::from(k) / f64::from(SAMPLES),
                    *a,
                    *b,
                )
            };
            let spacing = (0..SAMPLES)
                .map(|k| (at(k + 1) - at(k)).length())
                .fold(0.0, f64::max);
            for &point in &points {
                let on = nearest_on_line_or_arc(curve, *a, *b, point).unwrap();
                let sampled = (0..=SAMPLES)
                    .map(|k| (at(k) - point).length())
                    .fold(f64::INFINITY, f64::min);
                let found = (on - point).length();
                assert!(
                    found <= sampled + 1e-12 && sampled - found <= spacing,
                    "{curve:?} from {point:?}: {found}, sampled {sampled}"
                );
            }
        }
    }

    /// Checks that an edge on `curve` (on the domain 0 to 1), run either way,
    /// reads each of `points` at the distance of the nearest of 200001 even
    /// samples, to within their spacing, and each of `on_it` as on it.
    fn assert_spline_reads_as_a_dense_scan(
        curve: &brepkit_math::nurbs::curve::NurbsCurve,
        points: &[Point3],
        on_it: &[Point3],
    ) {
        const SAMPLES: u32 = 200_000;
        let (start, end) = (curve.evaluate(0.0), curve.evaluate(1.0));
        let at = |k: u32| curve.evaluate(f64::from(k) / f64::from(SAMPLES));
        let spacing = (0..SAMPLES)
            .map(|k| (at(k + 1) - at(k)).length())
            .fold(0.0, f64::max);
        for reversed in [false, true] {
            let mut topo = Topology::new();
            let v0 = topo.add_vertex(Vertex::new(start, 1e-7));
            let v1 = topo.add_vertex(Vertex::new(end, 1e-7));
            let (from, to) = if reversed { (v1, v0) } else { (v0, v1) };
            let edge = topo.add_edge(Edge::new(from, to, EdgeCurve::NurbsCurve(curve.clone())));
            let reads = EdgeReads::new(&topo, edge).unwrap();
            for &point in points.iter().chain(on_it) {
                let (found, on) = reads.nearest(point);
                assert!(((on - point).length() - found).abs() < 1e-15);
                let sampled = (0..=SAMPLES)
                    .map(|k| (at(k) - point).length())
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    found <= sampled + 1e-12 && sampled - found <= spacing,
                    "reversed {reversed}, from {point:?}: {found}, sampled {sampled}"
                );
            }
            for &point in on_it {
                let found = reads.nearest(point).0;
                assert!(
                    found < 1e-12,
                    "reversed {reversed}, {point:?} reads {found} off"
                );
            }
        }
    }

    /// A B-spline edge's nearest point, from its nearest sample by Newton
    /// steps, is the nearest of the dense samples; a point on the edge reads
    /// as on it.
    #[test]
    fn the_nearest_point_of_a_spline_edge_is_the_nearest_sample() {
        let p = Point3::new;
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            vec![
                p(0.0, 0.0, 0.0),
                p(2.0, 3.0, 0.5),
                p(4.0, -2.0, 1.0),
                p(6.0, 4.0, -0.5),
                p(8.0, 0.0, 0.0),
            ],
            vec![1.0, 0.8, 1.2, 0.9, 1.0],
        )
        .unwrap();
        let points = [
            p(3.0, 2.0, 0.0),
            p(-1.0, -1.0, 1.0),
            p(9.0, 1.0, -1.0),
            p(5.0, 0.5, 3.0),
        ];
        assert_spline_reads_as_a_dense_scan(&curve, &points, &[curve.evaluate(0.37)]);
    }

    /// The parabola `(u - 256 u^2, 256 u^2)`, `u = t - 1/2`: its point at
    /// `u = 1/256` is nearest the sample at `u = 0`, where the distance to
    /// it is a maximum along the curve and a Newton step is zero. The read
    /// still finds the point on the curve.
    #[test]
    fn a_read_from_a_sample_where_the_distance_peaks_finds_the_curve() {
        let p = Point3::new;
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                p(-64.5, 64.0, 0.0),
                p(64.0, -64.0, 0.0),
                p(-63.5, 64.0, 0.0),
            ],
            vec![1.0; 3],
        )
        .unwrap();
        assert!((curve.evaluate(0.5) - p(0.0, 0.0, 0.0)).length() < 1e-12);
        let on_it = curve.evaluate(0.503_906_25);
        assert!((on_it - p(0.0, 1.0 / 256.0, 0.0)).length() < 1e-12);
        assert_spline_reads_as_a_dense_scan(&curve, &[p(0.0, 0.002, 0.0)], &[on_it]);
    }

    /// A cubic whose first two poles coincide leaves its start at no speed.
    /// The nearest sample to a point just off the curve near that end is the
    /// end itself, where no Newton step can be taken; the read still finds
    /// the nearest point a little way in.
    #[test]
    fn a_read_from_a_spline_end_at_no_speed_finds_the_nearest_point() {
        let p = Point3::new;
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                p(0.0, 0.0, 0.0),
                p(0.0, 0.0, 0.0),
                p(1.0, 0.0, 0.0),
                p(2.0, 1.0, 0.0),
            ],
            vec![1.0; 4],
        )
        .unwrap();
        let points = [p(2e-4, 1e-4, 0.0), p(5e-5, -3e-5, 1e-5)];
        assert_spline_reads_as_a_dense_scan(&curve, &points, &[curve.evaluate(0.004)]);
    }

    /// The piece of a B-spline edge a spur leaves runs between the split and
    /// the edge's own vertex: where the edge runs against its curve, and
    /// where it takes only a span of its curve.
    #[test]
    fn a_spline_spokes_piece_keeps_to_the_edge() {
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(5.0, 5.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        let at = |t: f64| curve.evaluate(t);
        let mut topo = Topology::new();
        let mut piece_ends = |from: f64, to: f64, split: f64, toward_end: bool| {
            let (a, b) = (at(from), at(to));
            let edge = Edge::new(
                topo.add_vertex(Vertex::new(a, 1e-7)),
                topo.add_vertex(Vertex::new(b, 1e-7)),
                EdgeCurve::NurbsCurve(curve.clone()),
            );
            let Some(EdgeCurve::NurbsCurve(piece)) =
                long_piece(&edge, (a, b), at(split), toward_end)
            else {
                panic!("no piece");
            };
            let (lo, hi) = piece.domain();
            (piece.evaluate(lo), piece.evaluate(hi))
        };
        let near = |p: Point3, q: Point3| (p - q).length() < 1e-9;
        // Stored from C(1) to C(0): toward its end runs to C(0).
        let (first, last) = piece_ends(1.0, 0.0, 0.5, true);
        assert!(
            near(first, at(0.5)) && near(last, at(0.0)),
            "{first:?} {last:?}"
        );
        let (first, last) = piece_ends(1.0, 0.0, 0.5, false);
        assert!(
            near(first, at(1.0)) && near(last, at(0.5)),
            "{first:?} {last:?}"
        );
        // A span from C(0.2) to C(0.8): toward its end stops at C(0.8).
        let (first, last) = piece_ends(0.2, 0.8, 0.5, true);
        assert!(
            near(first, at(0.5)) && near(last, at(0.8)),
            "{first:?} {last:?}"
        );
    }

    /// An arc from the corner at angle 0 to a contact's end at `turn`, on a
    /// unit circle in the plane z = 0, seen along z.
    fn bow_of_unit_arc(turn: f64, other: Point3) -> Option<Result<[(f64, f64); 3], BlendError>> {
        bow_of_arc(1.0, 3.0, turn, other)
    }

    /// The same on a circle of radius `radius` whose edge ends at `span`.
    fn bow_of_arc(
        radius: f64,
        span: f64,
        turn: f64,
        other: Point3,
    ) -> Option<Result<[(f64, f64); 3], BlendError>> {
        let mut topo = Topology::new();
        let at = |angle: f64| Point3::new(radius * angle.cos(), radius * angle.sin(), 0.0);
        let (corner, far) = (at(0.0), at(span));
        let a = topo.add_vertex(brepkit_topology::vertex::Vertex::new(corner, 1e-7));
        let b = topo.add_vertex(brepkit_topology::vertex::Vertex::new(far, 1e-7));
        let circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            radius,
        )
        .unwrap();
        let edge = brepkit_topology::edge::Edge::new(a, b, EdgeCurve::Circle(circle));
        arc_bow(
            &edge,
            corner,
            far,
            true,
            [at(turn), other],
            &|p: Point3| (p.x(), p.y()),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        )
    }

    #[test]
    fn an_arc_bowing_away_from_the_notch_is_held_by_its_tangents() {
        // The other contact end is inside the circle: the arc bows away.
        let inside = Point3::new(0.5, 0.2, 0.0);
        let [c, apex, e] = bow_of_unit_arc(1.0, inside).unwrap().unwrap();
        assert!((c.0 - 1.0).abs() < 1e-12 && c.1.abs() < 1e-12);
        assert!((e.0 - 1.0_f64.cos()).abs() < 1e-12 && (e.1 - 1.0_f64.sin()).abs() < 1e-12);
        // The tangents meet on the bisector, at 1 / cos(half the turn).
        let out = 1.0 / 0.5_f64.cos();
        assert!((apex.0 - out * 0.5_f64.cos()).abs() < 1e-9);
        assert!((apex.1 - out * 0.5_f64.sin()).abs() < 1e-9);
        // The other end outside the circle: the arc bows into the notch.
        assert!(bow_of_unit_arc(1.0, Point3::new(1.5, 0.9, 0.0)).is_none());
    }

    #[test]
    fn an_arc_piece_past_a_quarter_turn_is_refused() {
        let inside = Point3::new(0.5, 0.2, 0.0);
        // At a half turn the two tangents are parallel.
        for turn in [1.6, std::f64::consts::PI - 1e-12, 2.9] {
            assert!(
                matches!(bow_of_unit_arc(turn, inside), Some(Err(_))),
                "turn {turn}"
            );
        }
        // Almost a full turn: the tangents at the piece's ends and at its
        // middle are all parallel.
        let tau = std::f64::consts::TAU;
        let almost = bow_of_arc(
            1000.0,
            tau - 2e-10,
            tau - 4e-10,
            Point3::new(500.0, 200.0, 0.0),
        );
        assert!(matches!(almost, Some(Err(_))));
    }

    #[test]
    fn an_arc_seen_edge_on_is_a_straight_side() {
        // A unit circle in the plane y = 0, seen along z: a line along x,
        // whatever the turn of its piece, and wherever the solid is.
        let poses = [
            brepkit_math::mat::Mat4::identity(),
            brepkit_math::mat::Mat4::translation(310.0, -720.0, 150.0)
                * brepkit_math::mat::Mat4::rotation_z(0.7)
                * brepkit_math::mat::Mat4::rotation_x(-1.1),
        ];
        for pose in poses {
            let turned = |v: Vec3| {
                let to = pose.mul_point(Point3::new(v.x(), v.y(), v.z()));
                let from = pose.mul_point(Point3::new(0.0, 0.0, 0.0));
                to - from
            };
            let mut topo = Topology::new();
            let at = |angle: f64| pose.mul_point(Point3::new(angle.cos(), 0.0, angle.sin()));
            let (corner, far) = (at(0.0), at(3.0));
            let a = topo.add_vertex(brepkit_topology::vertex::Vertex::new(corner, 1e-7));
            let b = topo.add_vertex(brepkit_topology::vertex::Vertex::new(far, 1e-7));
            let circle = brepkit_math::curves::Circle3D::new(
                pose.mul_point(Point3::new(0.0, 0.0, 0.0)),
                turned(Vec3::new(0.0, -1.0, 0.0)),
                1.0,
            )
            .unwrap();
            let edge = brepkit_topology::edge::Edge::new(a, b, EdgeCurve::Circle(circle));
            let other = pose.mul_point(Point3::new(0.5, 0.4, 0.0));
            let (x, y) = (
                turned(Vec3::new(1.0, 0.0, 0.0)),
                turned(Vec3::new(0.0, 1.0, 0.0)),
            );
            let flat = |p: Point3| {
                let d = p - corner;
                (d.dot(x), d.dot(y))
            };
            for turn in [0.3, 1.6, 2.9] {
                let bow = arc_bow(&edge, corner, far, true, [at(turn), other], &flat, (x, y));
                assert!(bow.is_none(), "turn {turn}");
            }
        }
    }

    /// A cubic B-spline in the plane z = 0 from the corner at the origin to
    /// (6, 0), bowing to +y over its first half and to -y over its second.
    fn wave(weights: [f64; 5]) -> brepkit_math::nurbs::NurbsCurve {
        brepkit_math::nurbs::NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 3.0, 0.0),
                Point3::new(3.0, 3.0, 0.0),
                Point3::new(4.0, -3.0, 0.0),
                Point3::new(6.0, 0.0, 0.0),
            ],
            weights.to_vec(),
        )
        .unwrap()
    }

    /// The region `spline_bow` gives for `curve` as an edge from the corner
    /// (or, with `backward`, to it), the contact's end at `at_end` on it and
    /// the other inside the notch, seen along z.
    fn bow_of_spline(
        curve: &brepkit_math::nurbs::NurbsCurve,
        at_end: f64,
        backward: bool,
    ) -> Result<Option<Vec<(f64, f64)>>, BlendError> {
        let near_ends = [curve.evaluate(at_end), Point3::new(0.5, -3.0, 0.0)];
        bow_of_spline_span(curve, (0.0, 1.0), backward, near_ends)
    }

    /// The same for the edge along `curve` over `span`, the corner at its
    /// first end, and the contacts' ends at `near_ends`.
    fn bow_of_spline_span(
        curve: &brepkit_math::nurbs::NurbsCurve,
        span: (f64, f64),
        backward: bool,
        near_ends: [Point3; 2],
    ) -> Result<Option<Vec<(f64, f64)>>, BlendError> {
        let mut topo = Topology::new();
        let (corner, far) = (curve.evaluate(span.0), curve.evaluate(span.1));
        let (start, end) = if backward {
            (far, corner)
        } else {
            (corner, far)
        };
        let a = topo.add_vertex(brepkit_topology::vertex::Vertex::new(start, 1e-7));
        let b = topo.add_vertex(brepkit_topology::vertex::Vertex::new(end, 1e-7));
        let edge = brepkit_topology::edge::Edge::new(a, b, EdgeCurve::NurbsCurve(curve.clone()));
        spline_bow(&edge, start, end, !backward, near_ends, &|p: Point3| {
            (p.x(), p.y())
        })
    }

    /// Whether `p` lies in `region`, its boundary included.
    fn in_region(region: &[(f64, f64)], p: (f64, f64)) -> bool {
        region_side(region, p) != RegionSide::Outside
    }

    #[test]
    fn a_spline_corner_edges_piece_is_held_by_its_control_hull() {
        let curve = wave([1.0, 1.0, 2.0, 1.0, 1.0]);
        for backward in [false, true] {
            let bow = bow_of_spline(&curve, 0.95, backward).unwrap().unwrap();
            // The piece up to the contact's end bows both ways off its
            // chord, and the region holds it all.
            for k in 0..=50 {
                let p = curve.evaluate(0.95 * f64::from(k) / 50.0);
                assert!(in_region(&bow, (p.x(), p.y())), "{p:?}");
            }
            // Past the contact's end, the edge runs on clear of it.
            let past = curve.evaluate(1.0);
            assert!(!in_region(&bow, (past.x(), past.y())), "{past:?}");
        }
    }

    #[test]
    fn a_contact_end_is_read_on_a_trimmed_splines_own_span() {
        // The edge runs along the curve from 0.3 to its end, with the
        // corner at 0.3. The other contact's end lies on the curve before
        // the corner, and this one a rounding off the edge.
        let curve = wave([1.0, 1.0, 2.0, 1.0, 1.0]);
        let on_edge = curve.evaluate(0.8) + Vec3::new(0.0, 0.0, 1e-6);
        let near_ends = [curve.evaluate(0.1), on_edge];
        let bow = bow_of_spline_span(&curve, (0.3, 1.0), false, near_ends)
            .unwrap()
            .unwrap();
        for k in 0..=50 {
            let p = curve.evaluate(0.3 + 0.5 * f64::from(k) / 50.0);
            assert!(in_region(&bow, (p.x(), p.y())), "{p:?}");
        }
    }

    #[test]
    fn a_spline_piece_too_narrow_to_cut_takes_the_whole_curves_hull() {
        let curve = wave([1.0, 1.0, 2.0, 1.0, 1.0]);
        let bow = bow_of_spline(&curve, 0.0, false).unwrap().unwrap();
        for k in 0..=50 {
            let p = curve.evaluate(f64::from(k) / 50.0);
            assert!(in_region(&bow, (p.x(), p.y())), "{p:?}");
        }
    }

    #[test]
    fn a_spline_corner_edge_seen_edge_on_adds_nothing() {
        let line = brepkit_math::nurbs::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 1.0),
                Point3::new(6.0, 0.0, -1.0),
            ],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        assert!(bow_of_spline(&line, 0.5, false).unwrap().is_none());
    }

    #[test]
    fn a_line_from_side_to_side_of_a_region_reaches_it() {
        let mut topo = Topology::new();
        let flat = |p: Point3| (p.x(), p.y());
        let mut line = |from: (f64, f64), to: (f64, f64)| {
            let (a, b) = (
                Point3::new(from.0, from.1, 0.0),
                Point3::new(to.0, to.1, 0.0),
            );
            let start = topo.add_vertex(Vertex::new(a, 1e-7));
            let end = topo.add_vertex(Vertex::new(b, 1e-7));
            (Edge::new(start, end, EdgeCurve::Line), a, b)
        };
        // Its ends on two sides of the notch triangle, and on two sides of
        // a hexagon: it runs through each inside.
        let hexagon: Vec<(f64, f64)> = (0..6)
            .map(|k| {
                let angle = std::f64::consts::FRAC_PI_3 * f64::from(k);
                (angle.cos(), angle.sin())
            })
            .collect();
        for (region, from, to) in [
            (NOTCH.to_vec(), (3.0, 0.0), (0.0, 3.0)),
            (hexagon.clone(), (1.0, 0.0), (-1.0, 0.0)),
            (
                hexagon.clone(),
                (0.75, 0.25 * 3.0_f64.sqrt()),
                (-0.5, -0.5 * 3.0_f64.sqrt()),
            ),
        ] {
            let (edge, a, b) = line(from, to);
            assert!(cap_edge_reaches_region(&edge, a, b, &flat, &region));
        }
        // A long line through the triangle near its side reaches it, and one
        // as long just past the side does not.
        for (y, reaches) in [(5e-4, true), (-5e-4, false)] {
            let (edge, a, b) = line((-1e6, y), (1e6, y));
            assert_eq!(
                cap_edge_reaches_region(&edge, a, b, &flat, &NOTCH),
                reaches,
                "{y}"
            );
        }
        // Along a side, or touching a corner, it does not.
        for (region, from, to) in [
            (NOTCH.to_vec(), (1.0, 0.0), (5.0, 0.0)),
            (NOTCH.to_vec(), (6.0, 0.0), (9.0, 1.0)),
            (hexagon, (1.0, 0.0), (0.5, 0.5 * 3.0_f64.sqrt())),
        ] {
            let (edge, a, b) = line(from, to);
            assert!(!cap_edge_reaches_region(&edge, a, b, &flat, &region));
        }
    }

    #[test]
    fn a_spline_corner_edge_with_a_weight_off_its_hull_is_refused() {
        for weight in [0.0, -1.0, f64::NAN] {
            let curve = wave([1.0, 1.0, weight, 1.0, 1.0]);
            assert!(bow_of_spline(&curve, 0.5, false).is_err(), "{weight}");
        }
    }

    #[test]
    fn a_spike_inside_a_narrow_knot_span_reaches_the_notch_triangle() {
        // A degree-1 boundary whose excursion to (1, 1) lies in knot spans
        // 0.001 wide, between two of 64 evenly spaced samples.
        let points = [
            (12.0, 2.0),
            (4.0, 4.0),
            (1.0, 1.0),
            (4.0, 4.01),
            (2.0, 12.0),
        ];
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.5, 0.501, 0.502, 1.0, 1.0],
            points
                .iter()
                .map(|&(x, y)| Point3::new(x, y, 0.0))
                .collect(),
            vec![1.0; 5],
        )
        .unwrap();
        let samples: Vec<(f64, f64)> = (0..=64)
            .map(|k| {
                let p = curve.evaluate(f64::from(k) / 64.0);
                (p.x(), p.y())
            })
            .collect();
        assert!(
            !path_enters_region(&samples, &NOTCH),
            "the samples step over the spike"
        );
        assert!(hull_overlaps_region(&points, &NOTCH));
    }

    /// A degree-1 edge from the origin to `(10, 0, 0)` through `points`, its
    /// interior knots `knots`.
    fn polyline_edge(topo: &mut Topology, points: &[Point3], knots: &[f64]) -> Edge {
        let mut all = vec![0.0, 0.0];
        all.extend_from_slice(knots);
        all.extend_from_slice(&[1.0, 1.0]);
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            all,
            points.to_vec(),
            vec![1.0; points.len()],
        )
        .unwrap();
        let a = topo.add_vertex(Vertex::new(points[0], 1e-7));
        let b = topo.add_vertex(Vertex::new(points[points.len() - 1], 1e-7));
        Edge::new(a, b, EdgeCurve::NurbsCurve(curve))
    }

    #[test]
    fn a_spike_inside_a_narrow_knot_span_is_not_straight() {
        // On its chord at every sixteenth of its domain, with an excursion 3
        // off it in knot spans 0.0005 wide between two of them.
        let mut topo = Topology::new();
        let edge = polyline_edge(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(5.01, 0.0, 0.0),
                Point3::new(5.015, 3.0, 0.0),
                Point3::new(5.02, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
            ],
            &[0.501, 0.5015, 0.502],
        );
        assert!(!straight_edge(&topo, &edge, 1e-6).unwrap());
        let straight = polyline_edge(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(5.01, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
            ],
            &[0.501],
        );
        assert!(straight_edge(&topo, &straight, 1e-6).unwrap());
    }

    /// A cubic B-spline edge of two spans from `(8, -4)` to `(-4, 8)`, its
    /// chord across the notch triangle, bowing through `bow`.
    fn bowed_edge(topo: &mut Topology, bow: [(f64, f64); 3]) -> (Edge, Point3, Point3) {
        let (a, b) = (Point3::new(8.0, -4.0, 0.0), Point3::new(-4.0, 8.0, 0.0));
        let mut points = vec![a];
        points.extend(bow.iter().map(|&(x, y)| Point3::new(x, y, 0.0)));
        points.push(b);
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            points,
            vec![1.0; 5],
        )
        .unwrap();
        let va = topo.add_vertex(Vertex::new(a, 1e-7));
        let vb = topo.add_vertex(Vertex::new(b, 1e-7));
        (Edge::new(va, vb, EdgeCurve::NurbsCurve(curve)), a, b)
    }

    #[test]
    fn a_curve_bowing_away_is_clear_though_its_whole_hull_is_not() {
        // A cap's long outline whose chord passes the corner: the hull of
        // all its control points holds the notch, its spans' hulls do not.
        let mut topo = Topology::new();
        let flat = |p: Point3| (p.x(), p.y());
        let bow = [(12.0, 4.0), (12.0, 12.0), (4.0, 12.0)];
        let (edge, a, b) = bowed_edge(&mut topo, bow);
        let mut hull = vec![(8.0, -4.0), (-4.0, 8.0)];
        hull.extend(bow);
        assert!(hull_overlaps_region(&hull, &NOTCH));
        assert!(!cap_edge_reaches_region(&edge, a, b, &flat, &NOTCH));
        // Bowed into the notch, it reaches it.
        let (edge, a, b) = bowed_edge(&mut topo, [(2.0, -1.0), (1.0, 1.0), (-1.0, 2.0)]);
        assert!(cap_edge_reaches_region(&edge, a, b, &flat, &NOTCH));
    }

    #[test]
    fn a_curve_with_a_negative_weight_is_never_certified_clear() {
        // Its control hull lies outside the triangle, yet the curve passes
        // through (1, 1).
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(4.0, 4.0, 0.0),
                Point3::new(7.0, 8.0, 0.0),
                Point3::new(4.0, 5.0, 0.0),
            ],
            vec![1.0, -0.5, 1.0],
        )
        .unwrap();
        let middle = curve.evaluate(0.5);
        assert!((middle.x() - 1.0).abs() < 1e-9 && (middle.y() - 1.0).abs() < 1e-9);
        assert!(!hull_overlaps_region(
            &[(4.0, 4.0), (7.0, 8.0), (4.0, 5.0)],
            &NOTCH
        ));

        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(4.0, 4.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(4.0, 5.0, 0.0), 1e-7));
        let edge_id = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve)));
        let edge = topo.edge(edge_id).unwrap();
        let flat = |p: Point3| (p.x(), p.y());
        assert!(cap_edge_reaches_region(
            edge,
            Point3::new(4.0, 4.0, 0.0),
            Point3::new(4.0, 5.0, 0.0),
            &flat,
            &NOTCH,
        ));
    }

    #[test]
    fn a_hull_touching_the_notch_chord_does_not_reach_it() {
        let touching = [(3.0, 3.0), (10.0, 3.0), (10.0, 10.0), (3.0, 10.0)];
        assert!(!hull_overlaps_region(&touching, &NOTCH));
        let clear = [(20.0, 20.0), (30.0, 20.0), (25.0, 30.0)];
        assert!(!hull_overlaps_region(&clear, &NOTCH));
        let crossing = [(2.0, 2.0), (10.0, 2.0), (10.0, 10.0), (2.0, 10.0)];
        assert!(hull_overlaps_region(&crossing, &NOTCH));
    }

    #[test]
    fn a_circle_within_its_chords_sag_of_the_notch_reaches_it() {
        // A polyline that stays just outside the triangle reaches it once
        // widened by the sag of the arc it stands for.
        let outside = [(4.0, 2.0005), (2.0005, 4.0)];
        assert!(!path_within_of_region(&outside, &NOTCH, 1e-4));
        assert!(path_within_of_region(&outside, &NOTCH, 1e-3));
    }

    /// A NURBS edge running along the first half of a 10 long line, forward
    /// and reversed: a point over the unused half reads its distance to the
    /// edge's own end, not to the curve beyond it.
    #[test]
    fn a_trimmed_nurbs_edge_reads_distance_within_its_span() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let line = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            vec![1.0, 1.0],
        )
        .unwrap();
        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(5.0, 0.0, 0.0), 1e-7));
        let p = Point3::new(8.0, 1.0, 0.0);
        let want = (p - Point3::new(5.0, 0.0, 0.0)).length();
        for (start, end) in [(a, b), (b, a)] {
            let edge = topo.add_edge(Edge::new(start, end, EdgeCurve::NurbsCurve(line.clone())));
            let d = distance_to_edge(&topo, edge, p).unwrap();
            assert!((d - want).abs() < 1e-9, "{d} vs {want}");
        }
    }

    #[test]
    fn fillet_builder_empty_edges_error() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let builder = FilletBuilder::new(&mut topo, solid);
        let result = builder.build();
        assert!(result.is_err(), "empty edge set should produce an error");
    }

    /// The plate z = 0 from -20 to 20, with the foot of a boss over
    /// [-5, 5]² as its hole, and the boss's wall x = 5 up to 10. The plate
    /// reaches farther behind the wall than in front of it, but the round
    /// along the foot rolls on the exposed plate in front: its contacts lie
    /// at x = 5 + r on the plate and z = r on the wall.
    #[test]
    fn a_variable_round_at_a_boss_foot_rolls_on_the_exposed_plate() {
        let mut topo = Topology::new();
        let mut vertex =
            |x: f64, y: f64, z: f64| topo.add_vertex(Vertex::new(Point3::new(x, y, z), 1e-7));
        let plate_corners = [(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)]
            .map(|(x, y)| vertex(x, y, 0.0));
        let foot =
            [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)].map(|(x, y)| vertex(x, y, 0.0));
        let (top_front, top_back) = (vertex(5.0, 5.0, 10.0), vertex(5.0, -5.0, 10.0));
        let mut loop_of = |vs: &[VertexId]| {
            (0..vs.len())
                .map(|i| {
                    let e =
                        topo.add_edge(Edge::new(vs[i], vs[(i + 1) % vs.len()], EdgeCurve::Line));
                    OrientedEdge::new(e, true)
                })
                .collect::<Vec<_>>()
        };
        let plate_outer = loop_of(&plate_corners);
        let foot_loop = loop_of(&foot);
        let foot_edge = foot_loop[1].edge();
        let wall_rest = loop_of(&[foot[2], top_front, top_back, foot[1]]);
        // The hole runs clockwise about +z, so the foot backward; the wall
        // runs it forward, up the wall on its left.
        let hole: Vec<_> = foot_loop
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), false))
            .collect();
        let wall: Vec<_> = std::iter::once(OrientedEdge::new(foot_edge, true))
            .chain(wall_rest[..3].iter().copied())
            .collect();
        let outer = topo.add_wire(Wire::new(plate_outer, true).unwrap());
        let hole = topo.add_wire(Wire::new(hole, true).unwrap());
        let wall = topo.add_wire(Wire::new(wall, true).unwrap());
        let plate = topo.add_face(Face::new(
            outer,
            vec![hole],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let wall = topo.add_face(Face::new(
            wall,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(1.0, 0.0, 0.0),
                d: 5.0,
            },
        ));
        let law = RadiusLaw::Linear {
            start: 1.0,
            end: 2.0,
        };
        for (face1, face2) in [(wall, plate), (plate, wall)] {
            let spine = Spine::from_single_edge(&topo, foot_edge).unwrap();
            let result = compute_stripe_for_spine(&topo, None, face1, face2, spine, &law).unwrap();
            for contact in [&result.stripe.contact1, &result.stripe.contact2] {
                let (t0, t1) = contact.domain();
                for k in 0..=8 {
                    let p = contact.evaluate((t1 - t0).madd(f64::from(k) / 8.0, t0));
                    let on_plate = p.z().abs() < 1e-6 && (5.9..7.1).contains(&p.x());
                    let on_wall = (p.x() - 5.0).abs() < 1e-6 && (0.9..2.1).contains(&p.z());
                    assert!(on_plate || on_wall, "a contact at {p:?}");
                }
            }
        }
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

    /// An edge the solid's arena does not hold, picked with one it does: the
    /// one it holds is rounded, and the other fails alone.
    #[test]
    fn an_edge_outside_the_arena_fails_alone() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let adjacency = AdjacencyIndex::build(&topo, solid).unwrap();
        let edge = brepkit_topology::explorer::solid_edges(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&edge| adjacency.faces_for_edge(edge).len() == 2)
            .expect("a manifold edge");
        let mut other = Topology::new();
        while other.num_edges() <= topo.num_edges() {
            let _ = make_unit_cube_manifold(&mut other);
        }
        let stale = other.edge_id_from_index(other.num_edges() - 1).unwrap();
        assert!(topo.edge(stale).is_err());
        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&[edge, stale], 0.1);
        let result = builder.build().expect("fillet build should succeed");
        assert_eq!(result.succeeded, vec![edge]);
        assert!(
            result.failed.iter().any(|(failed, _)| *failed == stale),
            "{:?}",
            result.failed
        );
        assert!(result.failed.iter().all(|(failed, _)| *failed == stale));
        assert_shell_edges_are_closed(&topo, result.solid);
    }

    /// An edge picked twice in one set is the same request: it is rounded
    /// once, as if picked once.
    #[test]
    fn an_edge_picked_twice_is_rounded_once() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let adjacency = AdjacencyIndex::build(&topo, solid).unwrap();
        let edge = brepkit_topology::explorer::solid_edges(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&edge| adjacency.faces_for_edge(edge).len() == 2)
            .expect("a manifold edge");
        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&[edge, edge], 0.1);
        let result = builder.build().expect("fillet build should succeed");
        assert!(result.failed.is_empty(), "{:?}", result.failed);
        assert_eq!(result.succeeded, vec![edge]);
        assert_shell_edges_are_closed(&topo, result.solid);
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
                    matches!(
                        topo.face(face).unwrap().surface(),
                        FaceSurface::Sphere(sphere) if (sphere.radius() - 0.1).abs() < 1e-12
                    )
                }),
            "three-stripe junction must publish a patch of the ball's sphere"
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

    /// Two arcs of one circle join the same two vertices: each lies on
    /// itself, and neither lies on the other. A rational B-spline traced
    /// along the shorter arc lies on it and not on the longer. An elliptical
    /// wall edge is refused.
    #[test]
    fn an_arc_does_not_lie_on_the_rest_of_its_curve() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            normal,
            2.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let (from, to) = (0.3_f64, 2.0_f64);
        let half = 0.5 * (to - from);
        let middle = from + half;
        let traced = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                circle.evaluate(from),
                Point3::new(
                    2.0 * middle.cos() / half.cos(),
                    2.0 * middle.sin() / half.cos(),
                    0.0,
                ),
                circle.evaluate(to),
            ],
            vec![1.0, half.cos(), 1.0],
        )
        .unwrap();
        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(circle.evaluate(from), 1e-7));
        let b = topo.add_vertex(Vertex::new(circle.evaluate(to), 1e-7));
        let short = Edge::new(a, b, EdgeCurve::Circle(circle.clone()));
        let long = Edge::new(b, a, EdgeCurve::Circle(circle));
        let spline = Edge::new(a, b, EdgeCurve::NurbsCurve(traced));
        let tol = 1e-4;
        assert!(edge_lies_on(&topo, &short, &short, tol).unwrap());
        assert!(edge_lies_on(&topo, &long, &long, tol).unwrap());
        assert!(!edge_lies_on(&topo, &short, &long, tol).unwrap());
        assert!(!edge_lies_on(&topo, &long, &short, tol).unwrap());
        assert!(edge_lies_on(&topo, &spline, &short, tol).unwrap());
        assert!(!edge_lies_on(&topo, &spline, &long, tol).unwrap());

        let ellipse =
            brepkit_math::curves::Ellipse3D::new(Point3::new(0.0, 0.0, 0.0), normal, 3.0, 2.0)
                .unwrap();
        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(ellipse.evaluate(from), 1e-7));
        let b = topo.add_vertex(Vertex::new(ellipse.evaluate(to), 1e-7));
        let short = Edge::new(a, b, EdgeCurve::Ellipse(ellipse.clone()));
        let long = Edge::new(b, a, EdgeCurve::Ellipse(ellipse));
        // An ellipse is held both ways: an arc lies on itself, and not on
        // the rest of its curve.
        assert!(edge_lies_on(&topo, &short, &short, tol).unwrap());
        assert!(edge_lies_on(&topo, &long, &long, tol).unwrap());
        assert!(!edge_lies_on(&topo, &short, &long, tol).unwrap());
        assert!(!edge_lies_on(&topo, &long, &short, tol).unwrap());
    }

    /// A degree-1 B-spline from `(0, 0)` to `(1, 0)` with knots at 0.52,
    /// 0.53 and 0.54, its middle control point `bump` off the x axis.
    fn bumped_polyline(bump: f64) -> brepkit_math::nurbs::curve::NurbsCurve {
        let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
        brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.52, 0.53, 0.54, 1.0, 1.0],
            vec![
                at(0.0, 0.0),
                at(0.52, 0.0),
                at(0.53, bump),
                at(0.54, 0.0),
                at(1.0, 0.0),
            ],
            vec![1.0; 5],
        )
        .unwrap()
    }

    /// A B-spline contact on its line at every sixteenth of its domain,
    /// with a bump 0.75 high between 0.52 and 0.54: it does not lie on the
    /// line. The same contact without the bump does.
    #[test]
    fn a_contact_bulging_between_samples_does_not_lie_on_its_line() {
        let mut topo = Topology::new();
        let va = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let vb = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let line = Edge::new(va, vb, EdgeCurve::Line);
        let flat = Edge::new(va, vb, EdgeCurve::NurbsCurve(bumped_polyline(0.0)));
        let bulging = Edge::new(va, vb, EdgeCurve::NurbsCurve(bumped_polyline(0.75)));
        let tol = 1e-3;
        assert!(edge_lies_on(&topo, &flat, &line, tol).unwrap());
        assert!(!edge_lies_on(&topo, &bulging, &line, tol).unwrap());
    }

    /// A B-spline wall edge with a notch 0.75 deep between 0.52 and 0.54 of
    /// its domain, and a straight contact between its ends: the edge lies
    /// under the contact at every sixteenth, but the contact does not lie
    /// on it.
    #[test]
    fn a_contact_does_not_lie_on_a_notched_spline() {
        let mut topo = Topology::new();
        let va = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let vb = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let notched = Edge::new(va, vb, EdgeCurve::NurbsCurve(bumped_polyline(-0.75)));
        let line = Edge::new(va, vb, EdgeCurve::Line);
        assert!(!edge_lies_on(&topo, &line, &notched, 1e-3).unwrap());
    }

    /// A closed B-spline square, an edge trimmed to one side of it (stored
    /// either way along the curve), and a contact along the other three
    /// sides between the same two vertices: the contact does not lie on the
    /// side. A B-spline wall edge is refused, so neither does the side
    /// itself.
    #[test]
    fn a_contact_does_not_lie_on_the_rest_of_a_closed_spline() {
        let corner = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let [a, b, c, d] = [
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(1.0, 1.0),
            corner(0.0, 1.0),
        ];
        let square = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0],
            vec![a, b, c, d, a],
            vec![1.0; 5],
        )
        .unwrap();
        let around = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
            vec![c, d, a, b],
            vec![1.0; 4],
        )
        .unwrap();
        let mut topo = Topology::new();
        let vb = topo.add_vertex(Vertex::new(b, 1e-7));
        let vc = topo.add_vertex(Vertex::new(c, 1e-7));
        let side = Edge::new(vb, vc, EdgeCurve::NurbsCurve(square.clone()));
        let side_back = Edge::new(vc, vb, EdgeCurve::NurbsCurve(square));
        let contact = Edge::new(vc, vb, EdgeCurve::NurbsCurve(around));
        let tol = 1e-6;
        for on in [&side, &side_back] {
            for edge in [&contact, &side, &side_back] {
                assert!(!edge_lies_on(&topo, edge, on, tol).unwrap());
            }
        }
    }

    /// A round r = 1 along x ends at the corner (0, 0, 0) of a curved cap,
    /// its contacts ending 1 from the corner on the cap's edges along -y and
    /// -z. A steep cap carries its end curve to 3 along the axis between
    /// them. The check reads only the cap's wires, so a plane stands in for
    /// the cap's surface. `hole` gives the corners of the cap's one hole.
    fn notch_clear_of_hole(hole: [(f64, f64, f64); 3]) -> Result<bool, BlendError> {
        let mut topo = Topology::new();
        let mut vertex =
            |(x, y, z): (f64, f64, f64)| topo.add_vertex(Vertex::new(Point3::new(x, y, z), 1e-7));
        let corner = vertex((0.0, 0.0, 0.0));
        let outer = [corner, vertex((0.0, -2.0, 0.0)), vertex((0.0, 0.0, -2.0))];
        let hole = hole.map(vertex);
        let mut loop_of = |vs: &[VertexId]| {
            (0..vs.len())
                .map(|i| {
                    let e =
                        topo.add_edge(Edge::new(vs[i], vs[(i + 1) % vs.len()], EdgeCurve::Line));
                    OrientedEdge::new(e, true)
                })
                .collect::<Vec<_>>()
        };
        let (outer, hole) = (loop_of(&outer), loop_of(&hole));
        let outer = topo.add_wire(Wire::new(outer, true).unwrap());
        let hole = topo.add_wire(Wire::new(hole, true).unwrap());
        let face = Face::new(
            outer,
            vec![hole],
            FaceSurface::Plane {
                normal: Vec3::new(1.0, 0.0, 0.0),
                d: 0.0,
            },
        );
        let near_ends = [Point3::new(0.0, -1.0, 0.0), Point3::new(0.0, 0.0, -1.0)];
        let end_hull = [
            near_ends[0],
            Point3::new(3.0, -0.6, -0.3),
            Point3::new(3.0, -0.3, -0.6),
            near_ends[1],
        ];
        notch_stays_clear(
            &topo,
            &face,
            corner,
            Vec3::new(1.0, 0.0, 0.0),
            near_ends,
            Some(&end_hull),
        )
    }

    /// The radius and the notch's size, the ends' span widened by both, put
    /// every hole below past 1 + sqrt(2) along the axis.
    fn past_radius_and_size(hole: [(f64, f64, f64); 3]) -> bool {
        hole.iter().all(|&(x, _, _)| x > 1.0 + 2.0_f64.sqrt())
    }

    #[test]
    fn a_hole_within_the_end_curves_span_crossing_the_triangle_is_refused() {
        // One corner seen in the triangle, two outside it.
        let hole = [(2.6, -0.3, -0.3), (2.8, -1.6, -0.3), (2.9, -0.3, -1.6)];
        assert!(past_radius_and_size(hole));
        assert!(matches!(
            notch_clear_of_hole(hole),
            Err(BlendError::PlanningFailure { .. })
        ));
    }

    #[test]
    fn a_hole_past_the_end_curve_wholly_in_the_triangle_is_refused() {
        let hole = [(5.0, -0.2, -0.2), (5.5, -0.6, -0.2), (6.0, -0.2, -0.6)];
        assert!(past_radius_and_size(hole));
        assert!(matches!(
            notch_clear_of_hole(hole),
            Err(BlendError::PlanningFailure { .. })
        ));
    }

    /// An ellipse given its longer semi-axis second, along the slab's axis,
    /// closed where its samples fall half a step either side of its low
    /// point: every sample lies above the slab, and the curve dips into it
    /// between two of them.
    #[test]
    fn an_ellipse_with_its_axes_given_short_first_is_not_beyond_a_slab_it_dips_into() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let ellipse = brepkit_math::curves::Ellipse3D::with_axes(
            Point3::new(9.994, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            1.0,
            10.0,
            Vec3::new(0.0, 1.0, 0.0),
            axis,
        )
        .unwrap();
        let at = ellipse.evaluate(std::f64::consts::PI * (1.0 + 1.0 / 64.0));
        let mut topo = Topology::new();
        let vertex = topo.add_vertex(Vertex::new(at, 1e-7));
        let low = ellipse.evaluate(1.5 * std::f64::consts::PI).x();
        assert!(low < -0.002);
        let edge = Edge::new(vertex, vertex, EdgeCurve::Ellipse(ellipse));
        assert!(!cap_edge_beyond_slab(
            &edge,
            at,
            at,
            (axis, Point3::new(0.0, 0.0, 0.0), -0.002, 0.002),
        ));
    }

    #[test]
    fn a_hole_past_the_end_curve_reaching_outside_the_triangle_is_passed_over() {
        let hole = [(5.0, -0.2, -0.2), (5.5, -1.6, -0.2), (6.0, -0.2, -1.6)];
        assert!(notch_clear_of_hole(hole).unwrap());
    }

    /// `count` points round the circle about `z` of radius `radius` at
    /// height `z`, the first again at the end.
    fn ring(radius: f64, z: f64, count: u32) -> Vec<Point3> {
        (0..=count)
            .map(|k| {
                let a = std::f64::consts::TAU * f64::from(k) / f64::from(count);
                Point3::new(radius * a.cos(), radius * a.sin(), z)
            })
            .collect()
    }

    /// The polyline through `points` as a curve of degree 1, `knots` its
    /// inner knots.
    fn polyline(points: Vec<Point3>, inner: &[f64]) -> brepkit_math::nurbs::curve::NurbsCurve {
        let mut knots = vec![0.0, 0.0];
        knots.extend_from_slice(inner);
        knots.extend([1.0, 1.0]);
        let weights = vec![1.0; points.len()];
        brepkit_math::nurbs::curve::NurbsCurve::new(1, knots, points, weights).unwrap()
    }

    fn even_knots(count: usize) -> Vec<f64> {
        (1..count - 1)
            .map(|k| k as f64 / (count - 1) as f64)
            .collect()
    }

    /// A unit ball cut at `z = -0.2`: its equator, 32 chords at `z = 0`, is
    /// the far loop of the band the rim at `z = -0.2` lies on. A contact all
    /// past the equator's plane, or on it, moves to the face beyond; one all
    /// short of it stays. A contact that dips below the plane between the
    /// 65 stations at which it was read has every station past it, which
    /// moved it, though part of it lies on the band: it is not proved past.
    #[test]
    fn a_contact_is_past_its_far_loop_only_where_its_control_points_prove_it() {
        let tol = 1e-7;
        let far: Vec<Point3> = ring(1.0, 0.0, 32);
        let rim: Vec<Point3> = ring(0.96_f64.sqrt(), -0.2, 64);
        let circle = |z: f64| {
            let points = ring((1.0 - z * z).sqrt(), z, 64);
            let count = points.len();
            polyline(points, &even_knots(count))
        };
        assert_eq!(
            contact_against_far_plane(&rim, &far, &circle(0.067), tol),
            FarSide::Beyond
        );
        assert_eq!(
            contact_against_far_plane(&rim, &far, &circle(0.0), tol),
            FarSide::Beyond
        );
        assert_eq!(
            contact_against_far_plane(&rim, &far, &circle(-0.1), tol),
            FarSide::Short
        );

        // A dip to z = -0.05 from z = 0.05, narrow, between stations 32 and
        // 33 of 64.
        let at = 0.5 + 1.0 / 256.0;
        let inner = [at - 1.0 / 1024.0, at, at + 1.0 / 1024.0];
        let points = vec![
            Point3::new(1.0, 0.0, 0.05),
            Point3::new(0.0, 1.0, 0.05),
            Point3::new(-1.0, 0.0, 0.05),
            Point3::new(-1.0, -0.01, -0.05),
            Point3::new(-1.0, -0.02, 0.05),
            Point3::new(1.0, 0.0, 0.05),
        ];
        let dip = polyline(points, &[0.25, inner[0], inner[1], inner[2]]);
        let stations_past = (0..=64).all(|k| dip.evaluate(f64::from(k) / 64.0).z() > 0.0);
        assert!(stations_past, "every station lies past the plane");
        assert!(dip.evaluate(at).z() < 0.0, "the dip lies short of it");
        assert_eq!(
            contact_against_far_plane(&rim, &far, &dip, tol),
            FarSide::Unsure
        );

        // A far loop off a plane proves nothing.
        let mut bent = far;
        bent[5] = Point3::new(bent[5].x(), bent[5].y(), 0.01);
        assert_eq!(
            contact_against_far_plane(&rim, &bent, &circle(0.067), tol),
            FarSide::Unsure
        );
    }
}
