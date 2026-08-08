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
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::analytic;
use crate::blend_func::{ConstRadBlend, EvolRadBlend};
use crate::builder_utils::{FlippedNormalSurface, sample_nurbs_endpoints, surface_ref_or_adapter};
use crate::corner;
use crate::radius_law::RadiusLaw;
use crate::spine::Spine;
use crate::stripe::{Stripe, StripeResult};
use crate::trimmer;
use crate::walker::{Walker, WalkerConfig, approximate_blend_surface};
use crate::{BlendError, BlendResult};

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
    /// 2. For each target edge, find the two adjacent faces.
    /// 3. Build single-edge spines (no G1 chain propagation in v1).
    /// 4. Compute stripes via analytic fast path or walking engine.
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
        // Expand edge sets: keep actual RadiusLaw references via indices.
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

        // Track which faces are touched (adjacent to a fillet edge).
        let mut touched_faces: HashSet<FaceId> = HashSet::new();

        let mut succeeded: Vec<EdgeId> = Vec::new();
        let mut failed: Vec<(EdgeId, BlendError)> = Vec::new();
        let mut stripe_results: Vec<StripeResult> = Vec::new();

        for &(edge_id, law_idx) in &all_edges {
            let result = compute_stripe_for_edge(topo, &adjacency, edge_id, &laws[law_idx]);
            match result {
                Ok(sr) => {
                    touched_faces.insert(sr.stripe.face1);
                    touched_faces.insert(sr.stripe.face2);
                    stripe_results.push(sr);
                    succeeded.push(edge_id);
                }
                Err(e) => {
                    failed.push((edge_id, e));
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
        let mut blend_face_ids: Vec<FaceId> = Vec::new();
        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();
        let mut regular_results: Vec<&StripeResult> = Vec::new();
        for sr in &stripe_results {
            if let Some(rim) = closed_rim_info(topo, &sr.stripe)? {
                match assemble_closed_rim(topo, &sr.stripe, &rim, &mut face_replacements) {
                    Ok(band) => blend_face_ids.push(band),
                    Err(e) => {
                        log::warn!("closed-rim assembly failed: {e}, falling back to trim path");
                        regular_results.push(sr);
                    }
                }
            } else {
                regular_results.push(sr);
            }
        }

        // Closed-rim faces: when every edge of a face's outer wire is some
        // stripe's spine edge, the face's boundary is consumed whole by the
        // fillet and its trimmed form is the CHAINED CONTACT LOOP. The
        // per-stripe trims cannot build that (each cuts with an infinite
        // contact line, and on a non-convex outline stripe k's line
        // amputates territory stripe j needs). Rebuild such faces directly
        // and record the loop edges so the blend walls share them.
        let (rim_contact_edges, rim_notches) =
            rebuild_closed_rim_loop_faces(topo, &regular_results, &mut face_replacements)?;

        let stripes: Vec<Stripe> = regular_results.iter().map(|sr| sr.stripe.clone()).collect();
        let corner_results = match corner::compute_corners(topo, &stripes, self.solid) {
            Ok(results) => results,
            Err(e) => {
                log::warn!("corner computation failed: {e}, proceeding without corner patches");
                Vec::new()
            }
        };

        let mut corner_face_ids: Vec<FaceId> = Vec::new();
        for cr in &corner_results {
            corner_face_ids.push(cr.face_id);
        }

        let mut stripe_contact_edges: Vec<(
            Option<brepkit_topology::edge::EdgeId>,
            Option<brepkit_topology::edge::EdgeId>,
        )> = Vec::new();
        for (si, sr) in regular_results.iter().enumerate() {
            let stripe = &sr.stripe;
            stripe_contact_edges.push((
                rim_contact_edges.get(&(stripe.face1, si)).copied(),
                rim_contact_edges.get(&(stripe.face2, si)).copied(),
            ));
            let (rim1, rim2) = (
                rim_contact_edges.contains_key(&(stripe.face1, si)),
                rim_contact_edges.contains_key(&(stripe.face2, si)),
            );

            let contact1_pts = sample_nurbs_endpoints(&stripe.contact1);
            let contact2_pts = sample_nurbs_endpoints(&stripe.contact2);

            // Keep the side of the contact line AWAY from the spine edge: the
            // strip between the contact line and the old edge is what the
            // blend face replaces. The side is resolved inside the trimmer,
            // whose Left/Right frame follows each face's wire traversal and
            // cannot be predicted here; a ball-centre plane-side test flips
            // for concave edges even though the in-plane keep side does not.
            let spine_pt = stripe.spine.evaluate(topo, 0.0)?;
            let keep = trimmer::TrimKeep::AwayFrom(spine_pt);

            // Trim face 1 — use current replacement if face was already trimmed.
            let current_face1 = face_replacements
                .get(&stripe.face1)
                .copied()
                .unwrap_or(stripe.face1);
            let trim1 = if rim1 {
                Ok(trimmer::TrimResult {
                    trimmed_face: current_face1,
                    new_edges: Vec::new(),
                    new_vertices: Vec::new(),
                    contact_edge: None,
                })
            } else {
                trimmer::trim_face_general(topo, current_face1, &contact1_pts, keep)
            };

            match trim1 {
                Ok(tr) if tr.trimmed_face != current_face1 => {
                    if let Some(slot) = stripe_contact_edges.last_mut() {
                        slot.0 = tr.contact_edge;
                    }
                    face_replacements.insert(stripe.face1, tr.trimmed_face);
                }
                Ok(_) => {} // untrimmed (non-planar), keep original
                Err(e) => {
                    log::warn!("trimming failed on face {:?}: {e}", stripe.face1);
                    // Trimming is best-effort in v1. Non-planar faces and complex
                    // geometries may fail to trim. We continue with the original face.
                }
            }

            let current_face2 = face_replacements
                .get(&stripe.face2)
                .copied()
                .unwrap_or(stripe.face2);
            let trim2 = if rim2 {
                Ok(trimmer::TrimResult {
                    trimmed_face: current_face2,
                    new_edges: Vec::new(),
                    new_vertices: Vec::new(),
                    contact_edge: None,
                })
            } else {
                trimmer::trim_face_general(topo, current_face2, &contact2_pts, keep)
            };

            match trim2 {
                Ok(tr) if tr.trimmed_face != current_face2 => {
                    if let Some(slot) = stripe_contact_edges.last_mut() {
                        slot.1 = tr.contact_edge;
                    }
                    face_replacements.insert(stripe.face2, tr.trimmed_face);
                }
                Ok(_) => {}
                Err(e) => {
                    log::warn!("trimming failed on face {:?}: {e}", stripe.face2);
                }
            }
        }

        let mut blend_cross_edges: Vec<(
            brepkit_topology::edge::EdgeId,
            brepkit_topology::vertex::VertexId,
            brepkit_topology::vertex::VertexId,
        )> = Vec::new();
        for (si, sr) in regular_results.iter().enumerate() {
            let stripe = &sr.stripe;

            // Reuse the trimmed neighbours' contact edges so the blend flank
            // shares one edge entity per contact instead of minting a
            // duplicate that leaves both faces' copies use-1.
            let (c1, c2) = stripe_contact_edges
                .get(si)
                .copied()
                .unwrap_or((None, None));
            let info = crate::builder_utils::create_blend_face_with_contacts(topo, stripe, c1, c2)?;
            blend_face_ids.push(info.face);
            blend_cross_edges.push(info.cross_end);
            blend_cross_edges.push(info.cross_start);
        }

        // Faces using each vertex, over the ORIGINAL shell: a stripe end
        // whose outline vertex belongs to a third face (a perpendicular
        // end face) is closed by the notch-arc path, not by a cap.
        let mut vertex_face_count: std::collections::HashMap<
            brepkit_topology::vertex::VertexId,
            usize,
        > = std::collections::HashMap::new();
        for &fid in &original_faces {
            let face = topo.face(fid)?;
            let mut wires = vec![face.outer_wire()];
            wires.extend_from_slice(face.inner_wires());
            let mut seen: HashSet<brepkit_topology::vertex::VertexId> = HashSet::new();
            for wid in wires {
                for oe in topo.wire(wid)?.edges() {
                    let e = topo.edge(oe.edge())?;
                    seen.insert(e.start());
                    seen.insert(e.end());
                }
            }
            for v in seen {
                *vertex_face_count.entry(v).or_insert(0) += 1;
            }
        }

        // Cap each abrupt stripe end that produced notch edges on BOTH
        // adjacent faces: the cap is the cross-section wedge [notch on one
        // face, terminal section arc, notch on the other face] sharing the
        // original outline vertex. The arc is minted from the terminal
        // section; the weld pass then unifies it with the blend wall's own
        // cross edge (identical geometry, both use-1).
        for (si, sr) in regular_results.iter().enumerate() {
            let stripe = &sr.stripe;
            let ends: [Option<&crate::section::CircSection>; 2] =
                [stripe.sections.first(), stripe.sections.last()];
            for sec in ends.into_iter().flatten() {
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
                if vertex_face_count
                    .get(&pair[0].outline_vid)
                    .copied()
                    .unwrap_or(0)
                    > 2
                {
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
                let Ok(circle_n) = (sec.p1 - sec.center).cross(sec.p2 - sec.center).normalize()
                else {
                    continue;
                };
                let Ok(circle) = Circle3D::new(sec.center, circle_n, sec.radius) else {
                    continue;
                };
                let arc = topo.add_edge(Edge::new(
                    na.contact_vid,
                    nb.contact_vid,
                    EdgeCurve::Circle(circle),
                ));
                let fwd_of = |topo: &Topology,
                              eid: EdgeId,
                              from: brepkit_topology::vertex::VertexId|
                 -> Result<bool, BlendError> {
                    Ok(topo.edge(eid)?.start() == from)
                };
                let oe1 = OrientedEdge::new(na.edge, fwd_of(topo, na.edge, na.outline_vid)?);
                let oe2 = OrientedEdge::new(arc, true);
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
                blend_face_ids.push(cap);
                log::debug!("stripe {si} end cap built at {outline_pt:?}");
            }
        }

        // Notch the fillet's end cross-section arcs out of the faces that
        // still cover the scooped corner (the untouched end caps): replace
        // each cap's two-edge corner path with the blend's own cross edge so
        // both sides share one edge entity.
        for arc in &blend_cross_edges {
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

        let mut result_faces: Vec<FaceId> = Vec::new();

        for &fid in &original_faces {
            if !touched_faces.contains(&fid) {
                // An untouched face may still have been rebuilt by the
                // end-cap notch pass.
                result_faces.push(face_replacements.get(&fid).copied().unwrap_or(fid));
            }
        }

        for &fid in &touched_faces {
            let replacement = face_replacements.get(&fid).copied();
            result_faces.push(replacement.unwrap_or(fid));
        }

        result_faces.extend(&blend_face_ids);
        result_faces.extend(&corner_face_ids);

        // Adjacent stripes whose terminal sections coincide (a tangent
        // junction's shared arc, or a tangency point where the section
        // collapses) each mint their own copy of the shared cross edge —
        // two use-1 edges with identical geometry. Weld those: two open
        // rims tracing the same curve can only be a stitching failure.
        crate::builder_utils::weld_coincident_free_edges(topo, &result_faces)?;
        crate::builder_utils::close_residual_free_loops(topo, &mut result_faces)?;

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

    // Vertices for the two closed contact circles (start == end → degenerate).
    let plate_v = topo.add_vertex(Vertex::new(rim.plate_circle.evaluate(0.0), TOL));
    let wall_v = topo.add_vertex(Vertex::new(rim.wall_circle.evaluate(0.0), TOL));

    // Shared contact-circle edges.
    let plate_edge = topo.add_edge(Edge::new(
        plate_v,
        plate_v,
        EdgeCurve::Circle(rim.plate_circle.clone()),
    ));
    let wall_edge = topo.add_edge(Edge::new(
        wall_v,
        wall_v,
        EdgeCurve::Circle(rim.wall_circle.clone()),
    ));
    // Seam connecting the two circles (degenerate-seam band, as the primitive
    // cylinder lateral uses).
    let seam_edge = topo.add_edge(Edge::new(plate_v, wall_v, EdgeCurve::Line));

    // --- Rebuild the disc cap bounded by the plate-contact circle. ---
    // The cap originally borders the rim via a single closed-circle wire; the
    // new cap reuses the plate-contact circle with the same orientation the cap
    // had on the original rim edge.
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
    let cap_wire = Wire::new(vec![OrientedEdge::new(plate_edge, cap_forward)], true)?;
    let cap_wire_id = topo.add_wire(cap_wire);
    let mut cap_face = Face::new(cap_wire_id, Vec::new(), plane_surf);
    cap_face.set_reversed(plane_reversed);
    let cap_face_id = topo.add_face(cap_face);
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
    let mut wall_forward = true;
    for oe in &wall_oriented {
        if oe.edge() == rim.rim_edge {
            new_wall_edges.push(OrientedEdge::new(wall_edge, oe.is_forward()));
            wall_forward = oe.is_forward();
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
    face_replacements.insert(rim.wall_face, new_wall_face_id);

    // --- Toroidal band between the two contact circles. ---
    // Degenerate-seam wire (plate circle, seam up, wall circle reversed, seam
    // down). The seam runs plate_v → wall_v, so this fixed order always closes
    // (plate_v → plate_v → wall_v → wall_v → plate_v). The shared circle edges
    // are used opposite to the standard-wound cap and wall, keeping the shell
    // manifold.
    let torus = match &stripe.surface {
        FaceSurface::Torus(t) => t.clone(),
        _ => {
            return Err(BlendError::TrimmingFailure {
                face: rim.wall_face,
            });
        }
    };
    // Orient the band so its outward normal points away from the solid. The
    // solid tessellator orients a torus band's triangles from the surface's
    // intrinsic (u, v) frame, then applies the face `reversed` flag; pick the
    // flag that makes the geometric normal at the band's mid-arc point outward.
    // Outward at a rim fillet points away from the cylinder axis (positive
    // radial) and away from the material along the axis; the torus geometric
    // normal at the mid-arc already has the correct radial sign, so we compare
    // its axial component against the material side.
    //
    // The band must traverse each shared contact circle in the EFFECTIVE
    // sense (is_forward XOR is_reversed) OPPOSITE its other user: the cap
    // holds `plate_edge` at `cap_forward` under `plane_reversed`, the wall
    // holds `wall_edge` at `wall_forward` under `wall_reversed`. Both
    // circles are degenerate (start == end vertex), so the chain closes
    // for any sense choice and the two senses are picked independently. A
    // fixed wire order cannot serve both rims of a cylinder — their caps
    // traverse the shared circles in opposite directions.
    let band_reversed = torus_band_needs_reversal(&torus, rim);
    let plate_sense = (cap_forward == plane_reversed) != band_reversed;
    let wall_sense = (wall_forward == wall_reversed) != band_reversed;
    let band_wire = Wire::new(
        vec![
            OrientedEdge::new(plate_edge, plate_sense),
            OrientedEdge::new(seam_edge, true),
            OrientedEdge::new(wall_edge, wall_sense),
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

    Ok(band_face_id)
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
        // Non-manifold (3+ faces) or boundary (0-1 faces) edge cannot be filleted.
        log::warn!(
            "edge {edge_id:?} has {} adjacent faces (expected 2) — cannot fillet non-manifold or boundary edges",
            adj_faces.len()
        );
        return Err(BlendError::StartSolutionFailure {
            edge: edge_id,
            t: 0.0,
        });
    }
    let face1 = adj_faces[0];
    let face2 = adj_faces[1];

    // Snapshot surface data, respecting face orientation.
    let face1_data = topo.face(face1)?;
    let surf1 = face1_data.surface().clone();
    let face1_reversed = face1_data.is_reversed();
    let face2_data = topo.face(face2)?;
    let surf2 = face2_data.surface().clone();
    let face2_reversed = face2_data.is_reversed();

    let spine = Spine::from_single_edge(topo, edge_id)?;

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

    let pcurve1 = build_pcurve_from_contact(ps1, &contact1)?;
    let pcurve2 = build_pcurve_from_contact(ps2, &contact2)?;

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

/// Build a PCurve (2D UV line) by projecting 3D contact endpoints onto a surface.
fn build_pcurve_from_contact(
    surf: &dyn brepkit_math::traits::ParametricSurface,
    contact: &brepkit_math::nurbs::curve::NurbsCurve,
) -> Result<brepkit_math::curves2d::Curve2D, BlendError> {
    let (t0, t1) = contact.domain();
    let p_start = contact.evaluate(t0);
    let p_end = contact.evaluate(t1);

    let (u0, v0) = surf.project_point(p_start);
    let (u1, v1) = surf.project_point(p_end);

    let origin = brepkit_math::vec::Point2::new(u0, v0);
    let dir = brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0);

    let line = brepkit_math::curves2d::Line2D::new(origin, dir)?;
    Ok(brepkit_math::curves2d::Curve2D::Line(line))
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
        for &eid in sr.stripe.spine.edges() {
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
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

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
}
