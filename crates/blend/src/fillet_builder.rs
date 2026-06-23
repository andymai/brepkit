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
use crate::builder_utils::{
    FlippedNormalSurface, create_blend_face, sample_nurbs_endpoints, surface_ref_or_adapter,
};
use crate::corner;
use crate::radius_law::RadiusLaw;
use crate::spine::Spine;
use crate::stripe::{Stripe, StripeResult};
use crate::trimmer::{self, TrimSide};
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

        for sr in &regular_results {
            let stripe = &sr.stripe;

            let contact1_pts = sample_nurbs_endpoints(&stripe.contact1);
            let contact2_pts = sample_nurbs_endpoints(&stripe.contact2);

            // Compute which side to keep: the side AWAY from the blend ball center.
            // Use the first section's ball center to determine direction relative
            // to each face normal. If center is on the normal side, keep Right
            // (away from center); otherwise keep Left.
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

            // Trim face 1 — use current replacement if face was already trimmed.
            let current_face1 = face_replacements
                .get(&stripe.face1)
                .copied()
                .unwrap_or(stripe.face1);
            let trim1 = trimmer::trim_face_general(topo, current_face1, &contact1_pts, keep_side1);

            match trim1 {
                Ok(tr) if tr.trimmed_face != current_face1 => {
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
            let trim2 = trimmer::trim_face_general(topo, current_face2, &contact2_pts, keep_side2);

            match trim2 {
                Ok(tr) if tr.trimmed_face != current_face2 => {
                    face_replacements.insert(stripe.face2, tr.trimmed_face);
                }
                Ok(_) => {}
                Err(e) => {
                    log::warn!("trimming failed on face {:?}: {e}", stripe.face2);
                }
            }
        }

        for sr in &regular_results {
            let stripe = &sr.stripe;

            // For v1, we create a minimal wire from the contact curve endpoints.
            let blend_face_id = create_blend_face(topo, stripe)?;
            blend_face_ids.push(blend_face_id);
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

        result_faces.extend(&blend_face_ids);
        result_faces.extend(&corner_face_ids);

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
    for oe in &wall_oriented {
        if oe.edge() == rim.rim_edge {
            new_wall_edges.push(OrientedEdge::new(wall_edge, oe.is_forward()));
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
    let band_wire = Wire::new(
        vec![
            OrientedEdge::new(plate_edge, true),
            OrientedEdge::new(seam_edge, true),
            OrientedEdge::new(wall_edge, false),
            OrientedEdge::new(seam_edge, false),
        ],
        true,
    )?;
    let band_wire_id = topo.add_wire(band_wire);
    let mut band_face = Face::new(band_wire_id, Vec::new(), stripe.surface.clone());
    // Orient the band so its outward normal points away from the solid. The
    // solid tessellator orients a torus band's triangles from the surface's
    // intrinsic (u, v) frame, then applies the face `reversed` flag; pick the
    // flag that makes the geometric normal at the band's mid-arc point outward.
    // Outward at a rim fillet points away from the cylinder axis (positive
    // radial) and away from the material along the axis; the torus geometric
    // normal at the mid-arc already has the correct radial sign, so we compare
    // its axial component against the material side.
    if torus_band_needs_reversal(&torus, rim) {
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
