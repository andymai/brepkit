//! Transient data structures for the face-splitting pipeline.
//!
//! These types carry edge and face data through the splitting stages:
//! pcurve computation, wire building, and sub-face construction.

#![allow(dead_code)] // Used by later pipeline stages.

use brepkit_math::curves2d::Curve2D;
use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::plane_frame::PlaneFrame;
use crate::ds::Rank;

// ---------------------------------------------------------------------------
// Edge-level types
// ---------------------------------------------------------------------------

/// A 2D-oriented edge on a face's parameter space.
///
/// Each edge carries both 3D geometry (for assembly) and a 2D pcurve
/// (for wire construction and classification in parameter space).
#[derive(Debug, Clone)]
pub struct OrientedPCurveEdge {
    /// 3D edge curve (Line, Circle, Ellipse, NurbsCurve).
    pub curve_3d: EdgeCurve,
    /// 2D curve in this face's (u,v) parameter space.
    pub pcurve: Curve2D,
    /// Start point in (u,v) space.
    pub start_uv: Point2,
    /// End point in (u,v) space.
    pub end_uv: Point2,
    /// Start point in 3D.
    pub start_3d: Point3,
    /// End point in 3D.
    pub end_3d: Point3,
    /// Whether this edge is traversed in its natural direction.
    pub forward: bool,
}

/// An intersection curve between two faces, with pcurves on each.
///
/// Produced by face-face intersection. Consumed by the face splitter.
#[derive(Debug, Clone)]
pub struct SectionEdge {
    /// 3D intersection curve.
    pub curve_3d: EdgeCurve,
    /// pcurve on face A's surface.
    pub pcurve_a: Curve2D,
    /// pcurve on face B's surface.
    pub pcurve_b: Curve2D,
    /// 3D start point (trimmed to face boundaries).
    pub start: Point3,
    /// 3D end point (trimmed to face boundaries).
    pub end: Point3,
    /// Optional pre-computed UV endpoints on face A (avoids re-projection).
    /// When `Some`, `split_face_2d` uses these instead of projecting `start`/`end`.
    pub start_uv_a: Option<Point2>,
    /// Optional pre-computed UV endpoint on face A.
    pub end_uv_a: Option<Point2>,
    /// Optional pre-computed UV endpoints on face B.
    pub start_uv_b: Option<Point2>,
    /// Optional pre-computed UV endpoint on face B.
    pub end_uv_b: Option<Point2>,
    /// When set, this section edge only applies to this face during splitting.
    /// `None` means the edge applies to both faces in the pair (normal case).
    /// `Some(id)` means only distribute to that face (coplanar case -- each
    /// face gets boundary edges clipped to the other's interior).
    pub target_face: Option<FaceId>,
}

// ---------------------------------------------------------------------------
// Face-level types
// ---------------------------------------------------------------------------

/// All edges incident to a face, ready for 2D wire construction.
///
/// Boundary edges come from the face's original wire(s). Section edges
/// come from face-face intersections. Both must be expressed as pcurves
/// in the face's parameter space before feeding to the wire builder.
#[derive(Debug, Clone, Default)]
pub struct FaceEdgeSet {
    /// Original boundary edges (possibly split at intersection vertices).
    pub boundary: Vec<OrientedPCurveEdge>,
    /// New edges from face-face intersections.
    pub section: Vec<OrientedPCurveEdge>,
}

/// A sub-face produced by the wire builder after face splitting.
///
/// Retains the parent face's surface geometry (never tessellated).
/// The wire loops are expressed in both 2D (for classification) and
/// 3D (for assembly).
#[derive(Debug, Clone)]
pub struct SplitSubFace {
    /// Surface from the parent face (preserved, never tessellated).
    pub surface: FaceSurface,
    /// Outer wire boundary in 2D + 3D.
    pub outer_wire: Vec<OrientedPCurveEdge>,
    /// Inner wire boundaries (holes).
    pub inner_wires: Vec<Vec<OrientedPCurveEdge>>,
    /// Whether the face normal is reversed relative to the surface.
    pub reversed: bool,
    /// The original face this sub-face was split from.
    pub parent: FaceId,
    /// Which solid this face came from.
    pub rank: Rank,
}

// ---------------------------------------------------------------------------
// Surface info cache
// ---------------------------------------------------------------------------

/// Cached surface info per face for consistent UV operations across stages.
///
/// For plane faces, stores a [`PlaneFrame`] for 3D<->UV projection.
/// For analytic faces (cylinder, cone, sphere, torus), stores periodicity
/// flags -- UV projection uses the surface's native parameterization.
#[derive(Debug, Clone)]
pub enum SurfaceInfo {
    /// Plane face with a cached reference frame.
    Plane(PlaneFrame),
    /// Parametric surface with native UV. Periodicity flags indicate whether
    /// the u or v parameter wraps (e.g. cylinder u in [0, 2pi)).
    Parametric {
        /// Whether the u parameter is periodic.
        u_periodic: bool,
        /// Whether the v parameter is periodic.
        v_periodic: bool,
    },
}

impl SurfaceInfo {
    /// Returns the `PlaneFrame` if this is a plane face, `None` otherwise.
    #[must_use]
    pub fn as_plane_frame(&self) -> Option<&PlaneFrame> {
        match self {
            Self::Plane(f) => Some(f),
            Self::Parametric { .. } => None,
        }
    }

    /// Returns `(u_periodic, v_periodic)`.
    #[must_use]
    pub fn periodicity(&self) -> (bool, bool) {
        match self {
            Self::Plane(_) => (false, false),
            Self::Parametric {
                u_periodic,
                v_periodic,
            } => (*u_periodic, *v_periodic),
        }
    }
}
