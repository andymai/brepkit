//! `BooleanState` — centralized provenance tracking for boolean operations.
//!
//! Tracks face provenance (images/origins), solid membership (in_parts),
//! and same-domain face pairs through all boolean pipeline stages.
//! Modeled after the reference implementation's `myImages`/`myOrigins`/
//! `myInParts`/`myShapesSD` maps.

use std::collections::HashMap;

use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// Centralized state tracking face provenance through a boolean operation.
///
/// Populated during classification and assembly, consumed during shell
/// building. Each map mirrors a reference implementation concept:
///
/// - `images`: input face → output face(s) it was split into
/// - `origins`: output face → input face(s) it came from
/// - `in_parts`: solid → indices of fragments classified as IN that solid
/// - `same_domain`: face → its coplanar counterpart on the other operand
#[allow(dead_code)] // API methods used progressively as more pipeline stages consume state
pub(super) struct BooleanState {
    /// Input face → output faces produced from it.
    images: HashMap<FaceId, Vec<FaceId>>,
    /// Output face → input faces it originated from.
    origins: HashMap<FaceId, Vec<FaceId>>,
    /// Solid → fragment indices classified as being IN that solid.
    in_parts: HashMap<SolidId, Vec<usize>>,
    /// Face → same-domain (coplanar) counterpart on the other operand.
    same_domain: HashMap<FaceId, FaceId>,
}

#[allow(dead_code)]
impl BooleanState {
    /// Creates an empty state.
    pub(super) fn new() -> Self {
        Self {
            images: HashMap::new(),
            origins: HashMap::new(),
            in_parts: HashMap::new(),
            same_domain: HashMap::new(),
        }
    }

    // ── Images (input → output splits) ───────────────────────────────

    /// Record that `source` produced `output` during assembly.
    pub(super) fn add_image(&mut self, source: FaceId, output: FaceId) {
        self.images.entry(source).or_default().push(output);
        self.origins.entry(output).or_default().push(source);
    }

    /// Returns the output faces produced from `source`, if any.
    pub(super) fn images_of(&self, source: FaceId) -> Option<&[FaceId]> {
        self.images.get(&source).map(Vec::as_slice)
    }

    /// Returns the input faces that produced `output`, if any.
    pub(super) fn origins_of(&self, output: FaceId) -> Option<&[FaceId]> {
        self.origins.get(&output).map(Vec::as_slice)
    }

    // ── In-parts (solid → fragment indices) ──────────────────────────

    /// Record that fragment at `index` is classified IN `solid`.
    pub(super) fn add_in_part(&mut self, solid: SolidId, index: usize) {
        self.in_parts.entry(solid).or_default().push(index);
    }

    /// Returns fragment indices classified as IN `solid`.
    pub(super) fn fragments_in_solid(&self, solid: SolidId) -> Option<&[usize]> {
        self.in_parts.get(&solid).map(Vec::as_slice)
    }

    // ── Same-domain (coplanar pairs) ─────────────────────────────────

    /// Record that `face_a` and `face_b` are coplanar counterparts.
    pub(super) fn set_same_domain(&mut self, face_a: FaceId, face_b: FaceId) {
        self.same_domain.insert(face_a, face_b);
        self.same_domain.insert(face_b, face_a);
    }

    /// Returns the same-domain counterpart of `face`, if any.
    pub(super) fn same_domain_of(&self, face: FaceId) -> Option<FaceId> {
        self.same_domain.get(&face).copied()
    }
}
