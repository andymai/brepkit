//! Stripe: a fillet band connecting two adjacent faces.

use brepkit_math::curves2d::Curve2D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::section::CircSection;
use crate::spine::Spine;

/// A fillet band (one per edge or edge chain) connecting two faces.
#[derive(Debug, Clone)]
pub struct Stripe {
    /// The spine (guideline) for this stripe.
    pub spine: Spine,
    /// The blend surface (NURBS or analytic).
    pub surface: FaceSurface,
    /// Contact PCurve on face 1 (UV-space).
    pub pcurve1: Curve2D,
    /// Contact PCurve on face 2 (UV-space).
    pub pcurve2: Curve2D,
    /// 3D contact curve on face 1.
    pub contact1: NurbsCurve,
    /// 3D contact curve on face 2.
    pub contact2: NurbsCurve,
    /// Face on side 1 of the blend.
    pub face1: FaceId,
    /// Face on side 2 of the blend.
    pub face2: FaceId,
    /// Cross-sections computed during walking.
    pub sections: Vec<CircSection>,
}

/// Result from computing a single stripe (before topology reconstruction).
pub struct StripeResult {
    /// The blend stripe data.
    pub stripe: Stripe,
    /// New edges created for the blend surface boundaries.
    pub new_edges: Vec<EdgeId>,
}
