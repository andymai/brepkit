//! Fillet builder: orchestrates the full fillet pipeline.
//!
//! Spine construction, analytic/walking stripe computation, face trimming,
//! and solid assembly. Supports constant and variable radius fillets on
//! planar face pairs (v1).

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

use crate::analytic;
use crate::blend_func::{ConstRadBlend, EvolRadBlend};
use crate::builder_utils::{create_blend_face, sample_nurbs_endpoints, surface_ref_or_adapter};
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
        // ── Validate input ──────────────────────────────────────────────
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

        // ── Build adjacency ─────────────────────────────────────────────
        let adjacency = topo.build_adjacency(self.solid)?;

        // Collect all original face IDs.
        let shell_id = topo.solid(self.solid)?.outer_shell();
        let original_faces: Vec<FaceId> = topo.shell(shell_id)?.faces().to_vec();

        // Track which faces are touched (adjacent to a fillet edge).
        let mut touched_faces: HashSet<FaceId> = HashSet::new();

        // ── Phase 1: Compute stripes ────────────────────────────────────
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

        // If no stripes succeeded, return the original solid with all failures.
        if stripe_results.is_empty() {
            return Ok(BlendResult {
                solid: self.solid,
                succeeded: Vec::new(),
                failed,
                is_partial: false,
            });
        }

        // ── Phase 2: Compute corner patches ────────────────────────────
        let stripes: Vec<Stripe> = stripe_results.iter().map(|sr| sr.stripe.clone()).collect();
        let corner_results =
            corner::compute_corners(topo, &stripes, self.solid).unwrap_or_default();

        // Add corner faces to the result and mark their adjacent faces as touched.
        let mut corner_face_ids: Vec<FaceId> = Vec::new();
        for cr in &corner_results {
            corner_face_ids.push(cr.face_id);
        }

        // ── Phase 3: Trim faces ─────────────────────────────────────────
        // Map from original face ID to its latest trimmed replacement.
        let mut face_replacements: std::collections::HashMap<FaceId, FaceId> =
            std::collections::HashMap::new();

        for sr in &stripe_results {
            let stripe = &sr.stripe;

            // Collect contact points for trimming.
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

            // Trim face 2.
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

        // ── Phase 3: Create blend faces ─────────────────────────────────
        let mut blend_face_ids: Vec<FaceId> = Vec::new();

        for sr in &stripe_results {
            let stripe = &sr.stripe;

            // Create a face for the blend surface.
            // For v1, we create a minimal wire from the contact curve endpoints.
            let blend_face_id = create_blend_face(topo, stripe)?;
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

        // Add corner patch faces.
        result_faces.extend(&corner_face_ids);

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
    // Find the two adjacent faces.
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

    // Build a single-edge spine.
    let spine = Spine::from_single_edge(topo, edge_id)?;

    // Get radius at the spine midpoint for the analytic path.
    let radius = law.evaluate(0.5);

    // Try analytic fast path (only for constant radius).
    if matches!(law, RadiusLaw::Constant(_)) {
        if let Some(result) =
            analytic::try_analytic_fillet(&surf1, &surf2, &spine, topo, radius, face1, face2)?
        {
            return Ok(result);
        }
    }

    // ── Walking fallback for non-analytic surface pairs ─────────────
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

    // Choose blend function based on law type.
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

    // Build NURBS surface from the walked sections.
    let blend_surface = approximate_blend_surface(&walk_result.sections)?;
    let blend_face_surface = brepkit_topology::face::FaceSurface::Nurbs(blend_surface);

    // Build contact curves from the sections.
    let contact1 = sections_to_contact_curve(&walk_result.sections, |s| s.p1)?;
    let contact2 = sections_to_contact_curve(&walk_result.sections, |s| s.p2)?;

    // PCurves: project contact 3D endpoints onto each face surface.
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

// ===========================================================================
// Tests
// ===========================================================================

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
        // No edges added — should error.
        let result = builder.build();
        assert!(result.is_err(), "empty edge set should produce an error");
    }

    #[test]
    fn fillet_builder_plane_plane_box_edge() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        // Find a manifold edge (any of the 12 edges should work).
        let adjacency = AdjacencyIndex::build(&topo, solid).unwrap();
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let faces = topo.shell(shell_id).unwrap().faces().to_vec();

        // Find the first edge shared by two faces.
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

        // Build fillet.
        let original_face_count = faces.len();
        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&[target_edge], 0.1);
        let result = builder.build().expect("fillet build should succeed");

        // The result solid should exist.
        let result_solid = topo.solid(result.solid).unwrap();
        let result_shell = topo.shell(result_solid.outer_shell()).unwrap();

        // More faces than the original (6 original + 1 blend, minus possibly trimmed).
        assert!(
            result_shell.faces().len() > original_face_count,
            "expected more faces after fillet: got {}, original {}",
            result_shell.faces().len(),
            original_face_count,
        );

        // Edge should be in the succeeded list.
        assert!(result.succeeded.contains(&target_edge));
        assert!(result.failed.is_empty());
        assert!(!result.is_partial);

        // The blend surface should be a cylinder (plane-plane fillet).
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

        // Create a fake edge that is not part of the solid (will fail adjacency).
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

        // The fake edge should be in failed.
        assert!(result.failed.len() == 1);
        assert_eq!(result.failed[0].0, fake_edge);
        // With no successes, the original solid is returned.
        assert_eq!(result.solid, solid);
    }
}
