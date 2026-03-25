//! Builder — splits faces and classifies sub-faces for boolean assembly.
//!
//! Takes the PaveFiller's output ([`GfaArena`] with pave blocks, face info,
//! and intersection curves) and produces classified sub-faces ready for
//! boolean operation selection.
//!
//! # Flow
//!
//! 1. **`fill_images`** — map original edges to their split images
//! 2. **`fill_images_faces`** — build sub-faces from face info
//! 3. **`same_domain`** — detect coplanar face pairs
//! 4. **`classify_sub_faces`** — classify each sub-face as IN/OUT
//!
//! [`GfaArena`]: crate::ds::GfaArena

pub mod assemble;
pub mod builder_solid;
pub mod classify_2d;
pub mod face_class;
pub mod face_splitter;
pub mod fill_images;
pub mod fill_images_faces;
pub mod pcurve_compute;
pub mod plane_frame;
pub mod same_domain;
pub mod split_types;
pub mod wire_builder;

pub use face_class::FaceClass;

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::bop::{self, BooleanOp};
use crate::classifier;
use crate::ds::{GfaArena, Rank};
use crate::error::AlgoError;

/// A sub-face produced by the Builder after splitting.
#[derive(Debug, Clone)]
pub struct SubFace {
    /// The face entity in topology (same as parent if no split occurred).
    pub face_id: FaceId,
    /// Classification relative to the opposing solid.
    pub classification: FaceClass,
    /// Which boolean argument this face came from.
    pub rank: Rank,
    /// Pre-computed interior sample point for classification.
    /// When `Some`, the classifier uses this instead of sampling from face geometry.
    /// Set by the face splitter for split sub-faces.
    pub interior_point: Option<Point3>,
}

/// Builder — orchestrates face splitting and classification.
///
/// Owns both the `Topology` and `GfaArena`, mutating them as needed.
/// After `perform()`, call `build_result()` to extract the results.
pub struct Builder {
    /// The topology containing both solids (owned, mutable).
    topo: Topology,
    /// GFA transient state from the PaveFiller (owned).
    arena: GfaArena,
    /// First boolean argument.
    solid_a: SolidId,
    /// Second boolean argument.
    solid_b: SolidId,
    /// Geometric tolerance.
    tol: Tolerance,
    /// Sub-faces produced by splitting.
    sub_faces: Vec<SubFace>,
    /// Map from face ID to its argument rank.
    face_ranks: HashMap<FaceId, Rank>,
    /// Same-domain face pairs detected by `same_domain`.
    sd_pairs: Vec<same_domain::SameDomainPair>,
}

impl Builder {
    /// Create a Builder with custom tolerance.
    #[must_use]
    pub fn with_tolerance(
        topo: Topology,
        arena: GfaArena,
        solid_a: SolidId,
        solid_b: SolidId,
        tol: Tolerance,
    ) -> Self {
        Self {
            topo,
            arena,
            solid_a,
            solid_b,
            tol,
            sub_faces: Vec::new(),
            face_ranks: HashMap::new(),
            sd_pairs: Vec::new(),
        }
    }

    /// Run the Builder pipeline: fill images, split faces, classify.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if topology lookups or classification fails.
    pub fn perform(&mut self) -> Result<(), AlgoError> {
        self.build_face_ranks()?;
        self.fill_images();
        self.classify_sub_faces()?;
        Ok(())
    }

    /// Select faces for the given boolean operation and assemble them
    /// into a solid.
    ///
    /// Consumes the Builder, returning the (potentially modified) topology
    /// and the result solid ID.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if face selection produces no faces or
    /// assembly fails.
    pub fn build_result(mut self, op: BooleanOp) -> Result<(Topology, SolidId), AlgoError> {
        let selected = bop::select_faces(&self.sub_faces, op, &self.sd_pairs);
        let solid_id = assemble::assemble_solid(&mut self.topo, &selected)?;
        Ok((self.topo, solid_id))
    }

    /// Get the sub-faces, SD pairs, and topology for testing.
    #[cfg(test)]
    pub(crate) fn debug_info(&self) -> (&[SubFace], &[same_domain::SameDomainPair], &Topology) {
        (&self.sub_faces, &self.sd_pairs, &self.topo)
    }

    /// Build the face-to-rank mapping from both solids.
    fn build_face_ranks(&mut self) -> Result<(), AlgoError> {
        let faces_a = brepkit_topology::explorer::solid_faces(&self.topo, self.solid_a)?;
        for fid in faces_a {
            self.face_ranks.insert(fid, Rank::A);
        }

        let faces_b = brepkit_topology::explorer::solid_faces(&self.topo, self.solid_b)?;
        for fid in faces_b {
            self.face_ranks.insert(fid, Rank::B);
        }

        Ok(())
    }

    /// Phase 1: map edges to split images and build sub-faces.
    fn fill_images(&mut self) {
        // Step 1: edge images
        let edge_images = fill_images::fill_edge_images(&self.arena);
        log::debug!(
            "Builder: {} original edges mapped to split images",
            edge_images.len()
        );

        // Step 2: face images (sub-faces)
        self.sub_faces = fill_images_faces::fill_images_faces(
            &mut self.topo,
            &self.arena,
            &edge_images,
            &self.face_ranks,
            self.tol,
        );
        log::debug!("Builder: {} sub-faces created", self.sub_faces.len());

        // Step 3: same-domain detection (records pairs, does NOT set FaceClass)
        self.sd_pairs = same_domain::detect_same_domain(
            &self.topo,
            &self.arena,
            &self.sub_faces,
            &self.face_ranks,
            self.tol,
        );

        // Note: SD representative replacement (replacing B's face_id with
        // A's representative) is deferred to a follow-up. While it produces
        // correct 2-shell topology for coplanar cuts (d1a2), the AABB
        // containment check is insufficiently precise for near-tangent
        // geometries, causing flaky test failures. A stricter SD detection
        // (edge-set matching) is needed first.
    }

    /// Phase 2: classify each sub-face as inside/outside the opposing solid.
    #[allow(clippy::too_many_lines)]
    fn classify_sub_faces(&mut self) -> Result<(), AlgoError> {
        // SD faces are excluded from non-SD BOP selection, so their
        // classification doesn't affect the result. But the ray-cast
        // classifier is non-deterministic at coplanar boundaries,
        // which can produce non-manifold results for near-tangent
        // geometries. Mark SD faces deterministically to skip ray-cast.
        let sd_indices: std::collections::HashSet<usize> = self
            .sd_pairs
            .iter()
            .flat_map(|p| [p.idx_a, p.idx_b])
            .collect();

        for (idx, sf) in self.sub_faces.iter_mut().enumerate() {
            if sd_indices.contains(&idx) {
                sf.classification = FaceClass::On;
                continue;
            }

            // Determine the opposing solid
            let opposing_solid = match sf.rank {
                Rank::A => self.solid_b,
                Rank::B => self.solid_a,
            };

            // Use pre-computed interior point if available (from face splitter),
            // otherwise sample from face geometry.
            let sample = if let Some(pt) = sf.interior_point {
                Ok(pt)
            } else {
                sample_face_interior(&self.topo, sf.face_id, self.tol)
            };

            match sample {
                Ok(point) => {
                    sf.classification =
                        classifier::classify_point(&self.topo, opposing_solid, point)?;
                }
                Err(e) => {
                    return Err(AlgoError::ClassificationFailed(format!(
                        "could not sample interior of face {:?}: {e}",
                        sf.face_id
                    )));
                }
            }
        }

        let unknown_count = self
            .sub_faces
            .iter()
            .filter(|sf| sf.classification == FaceClass::Unknown)
            .count();
        let total = self.sub_faces.len();
        log::debug!(
            "Builder: {}/{total} sub-faces classified",
            total - unknown_count
        );

        if unknown_count > 0 {
            return Err(AlgoError::ClassificationFailed(format!(
                "{unknown_count} sub-faces could not be classified"
            )));
        }

        Ok(())
    }
}

/// Sample a point in the interior of a face.
///
/// Uses the midpoint of the first boundary edge, then offsets slightly
/// inward along (edge_tangent x face_normal) to get a point that is
/// reliably inside the face — unlike a vertex centroid, which can fall
/// outside non-convex faces.
fn sample_face_interior(
    topo: &Topology,
    face_id: FaceId,
    _tol: Tolerance,
) -> Result<Point3, AlgoError> {
    use brepkit_math::vec::Vec3;

    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let edges = wire.edges();

    if edges.is_empty() {
        return Err(AlgoError::FaceSplitFailed(format!(
            "face {face_id:?} has empty outer wire"
        )));
    }

    // Take the first boundary edge and evaluate at its midpoint
    let first_oe = &edges[0];
    let edge = topo.edge(first_oe.edge())?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();
    let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
    let t_mid = 0.5_f64.mul_add(t1 - t0, t0);
    let mid_pt = edge
        .curve()
        .evaluate_with_endpoints(t_mid, start_pos, end_pos);

    // Get the edge tangent and face normal at the midpoint
    let tangent = edge
        .curve()
        .tangent_with_endpoints(t_mid, start_pos, end_pos);
    let surface = face.surface();

    // Use the surface normal at the midpoint (project first to get UV)
    let face_normal = if let Some((u, v)) = surface.project_point(mid_pt) {
        surface.normal(u, v)
    } else {
        // Plane: normal is constant
        match surface {
            brepkit_topology::face::FaceSurface::Plane { normal, .. } => *normal,
            _ => Vec3::new(0.0, 0.0, 1.0),
        }
    };

    // Inward direction: tangent x face_normal points into the face interior
    // (assuming CCW winding when viewed from the face normal direction)
    let inward = tangent.cross(face_normal);
    let inward_len = inward.length();

    let offset = if inward_len > 1e-12 {
        inward * (1e-6 / inward_len)
    } else {
        // Degenerate — use a tiny offset along the face normal instead
        face_normal * 1e-6
    };

    let interior_pt = mid_pt + offset;

    // Project back onto the surface to ensure the point is on-surface
    if let Some((u, v)) = surface.project_point(interior_pt) {
        if let Some(on_surface) = surface.evaluate(u, v) {
            return Ok(on_surface);
        }
    }

    // Fallback: use the midpoint itself (it's on the boundary, not ideal
    // but better than a centroid that may be outside the face)
    Ok(mid_pt)
}
