//! Transient data structures for the boolean_v2 pipeline.
//!
//! [`BooleanPipeline`] holds all intermediate state during a boolean operation.
//! It is NOT persisted on [`Topology`] — only the final faces/edges/wires
//! are committed at assembly time.

#![allow(dead_code)] // Used by later boolean_v2 pipeline stages.

use std::collections::HashMap;

use brepkit_math::curves2d::Curve2D;
use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::plane_frame::PlaneFrame;
use super::types::Source;

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
/// Produced by Stage 1 (intersect). Consumed by Stage 2 (split edges)
/// and Stage 3 (split faces).
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
pub struct SubFace {
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
    pub source: Source,
}

// ---------------------------------------------------------------------------
// Surface info cache
// ---------------------------------------------------------------------------

/// Cached surface info per face for consistent UV operations across stages.
///
/// For plane faces, stores a [`PlaneFrame`] for 3D↔UV projection.
/// For analytic faces (cylinder, cone, sphere, torus), stores periodicity
/// flags — UV projection uses the surface's native parameterization.
#[derive(Debug, Clone)]
pub enum SurfaceInfo {
    /// Plane face with a cached reference frame.
    Plane(PlaneFrame),
    /// Parametric surface with native UV. Periodicity flags indicate whether
    /// the u or v parameter wraps (e.g. cylinder u ∈ [0, 2π)).
    Parametric {
        /// Whether the u parameter is periodic.
        u_periodic: bool,
        /// Whether the v parameter is periodic.
        v_periodic: bool,
    },
}

impl SurfaceInfo {
    /// Returns the `PlaneFrame` if this is a plane face, `None` otherwise.
    pub fn as_plane_frame(&self) -> Option<&PlaneFrame> {
        match self {
            Self::Plane(f) => Some(f),
            Self::Parametric { .. } => None,
        }
    }

    /// Returns `(u_periodic, v_periodic)`.
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

// ---------------------------------------------------------------------------
// Pipeline state
// ---------------------------------------------------------------------------

/// Transient state for a `boolean_v2` operation.
///
/// Created at the start of a boolean, populated stage-by-stage, consumed
/// during assembly, then dropped. Never stored on [`Topology`].
#[derive(Debug, Default)]
pub struct BooleanPipeline {
    /// Face pair → intersection curves with pcurves on each face.
    pub intersections: HashMap<(FaceId, FaceId), Vec<SectionEdge>>,
    /// Face → collected edge set (boundary + section edges).
    pub face_edges: HashMap<FaceId, FaceEdgeSet>,
    /// All sub-faces after wire-builder splitting + classification.
    pub sub_faces: Vec<SubFace>,
    /// Cached `PlaneFrame` per plane face (consistent UV origin across all stages).
    ///
    /// Retained for backward compatibility with plane-only code paths.
    /// New code should prefer `surface_info` which covers all surface types.
    pub plane_frames: HashMap<FaceId, PlaneFrame>,
    /// Cached surface info per face (plane frame or parametric periodicity).
    pub surface_info: HashMap<FaceId, SurfaceInfo>,
    /// Solid A handle.
    pub solid_a: Option<SolidId>,
    /// Solid B handle.
    pub solid_b: Option<SolidId>,
}
