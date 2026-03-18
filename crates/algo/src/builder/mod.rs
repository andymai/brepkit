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
pub mod face_class;
pub mod fill_images;
pub mod fill_images_faces;
pub mod same_domain;

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
    /// The original face this sub-face was split from.
    pub parent_face: FaceId,
    /// The face entity in topology (same as parent if no split occurred).
    pub face_id: FaceId,
    /// Classification relative to the opposing solid.
    pub classification: FaceClass,
    /// Which boolean argument this face came from.
    pub rank: Rank,
}

/// Builder — orchestrates face splitting and classification.
///
/// Owns both the `Topology` and `GfaArena`, mutating them as needed.
/// After `perform()`, call `build_result()` or `into_parts()` to
/// extract the results.
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
}

impl Builder {
    /// Create a new Builder from PaveFiller output.
    #[must_use]
    pub fn new(topo: Topology, arena: GfaArena, solid_a: SolidId, solid_b: SolidId) -> Self {
        Self {
            topo,
            arena,
            solid_a,
            solid_b,
            tol: Tolerance::default(),
            sub_faces: Vec::new(),
            face_ranks: HashMap::new(),
        }
    }

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

    /// Returns the classified sub-faces.
    #[must_use]
    pub fn sub_faces(&self) -> &[SubFace] {
        &self.sub_faces
    }

    /// Consume the Builder, returning the topology, arena, and sub-faces.
    #[must_use]
    pub fn into_parts(self) -> (Topology, GfaArena, Vec<SubFace>) {
        (self.topo, self.arena, self.sub_faces)
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
        let selected = bop::select_faces(&self.sub_faces, op);
        let solid_id = assemble::assemble_solid(&mut self.topo, &selected)?;
        Ok((self.topo, solid_id))
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
        self.sub_faces =
            fill_images_faces::fill_images_faces(&self.arena, &edge_images, &self.face_ranks);
        log::debug!("Builder: {} sub-faces created", self.sub_faces.len());

        // Step 3: same-domain detection
        same_domain::detect_same_domain(&self.topo, &self.arena, &self.sub_faces, &self.face_ranks);
    }

    /// Phase 2: classify each sub-face as inside/outside the opposing solid.
    #[allow(clippy::too_many_lines)]
    fn classify_sub_faces(&mut self) -> Result<(), AlgoError> {
        for sf in &mut self.sub_faces {
            // Skip already-classified faces (e.g. same-domain)
            if sf.classification != FaceClass::Unknown {
                continue;
            }

            // Determine the opposing solid
            let opposing_solid = match sf.rank {
                Rank::A => self.solid_b,
                Rank::B => self.solid_a,
            };

            // Sample a point inside the sub-face for classification
            let sample = sample_face_interior(&self.topo, sf.face_id, self.tol);

            match sample {
                Ok(point) => {
                    sf.classification =
                        classifier::classify_point(&self.topo, opposing_solid, point)?;
                }
                Err(e) => {
                    log::warn!(
                        "Builder: could not sample interior of face {:?}: {e}",
                        sf.face_id
                    );
                    // Leave as Unknown — the BOP will skip it
                }
            }
        }

        let classified = self
            .sub_faces
            .iter()
            .filter(|sf| sf.classification != FaceClass::Unknown)
            .count();
        log::debug!(
            "Builder: {classified}/{} sub-faces classified",
            self.sub_faces.len()
        );

        Ok(())
    }
}

/// Sample a point in the interior of a face.
///
/// Uses the face's surface parameterization: evaluates at the midpoint
/// of the face's outer wire's first edge, then offsets slightly toward
/// the face interior.
fn sample_face_interior(
    topo: &Topology,
    face_id: FaceId,
    _tol: Tolerance,
) -> Result<Point3, AlgoError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let edges = wire.edges();

    if edges.is_empty() {
        return Err(AlgoError::FaceSplitFailed(format!(
            "face {face_id:?} has empty outer wire"
        )));
    }

    // Compute centroid of wire vertices as a robust interior sample
    let mut sum = Point3::new(0.0, 0.0, 0.0);
    let mut count = 0u32;

    for oe in edges {
        let v = topo.vertex(topo.edge(oe.edge())?.start())?;
        let p = v.point();
        sum = Point3::new(sum.x() + p.x(), sum.y() + p.y(), sum.z() + p.z());
        count += 1;
    }

    if count == 0 {
        return Err(AlgoError::FaceSplitFailed(format!(
            "face {face_id:?} has no vertices"
        )));
    }

    let centroid = Point3::new(
        sum.x() / f64::from(count),
        sum.y() / f64::from(count),
        sum.z() / f64::from(count),
    );

    // Project the centroid onto the surface to get an on-surface point
    let surface = face.surface();
    let Some((u, v)) = surface.project_point(centroid) else {
        // Plane faces don't support project_point — use centroid directly
        return Ok(centroid);
    };
    let Some(on_surface) = surface.evaluate(u, v) else {
        return Ok(centroid);
    };

    Ok(on_surface)
}
