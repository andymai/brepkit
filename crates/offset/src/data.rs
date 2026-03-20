//! Central data structures shared across all offset pipeline phases.

use std::collections::BTreeMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::WireId;

/// Classification of an edge based on the dihedral angle between its
/// two adjacent faces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeClass {
    /// The two faces are tangent-continuous across this edge.
    Tangent,
    /// The edge is convex (outside corner) with the given dihedral angle in
    /// radians.
    Convex {
        /// Dihedral angle in radians (0, pi).
        angle: f64,
    },
    /// The edge is concave (inside corner) with the given dihedral angle in
    /// radians.
    Concave {
        /// Dihedral angle in radians (0, pi).
        angle: f64,
    },
}

/// Classification of a vertex based on its surrounding edge classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexClass {
    /// All incident edges are convex or tangent.
    Convex,
    /// All incident edges are concave or tangent.
    Concave,
    /// The vertex has both convex and concave incident edges.
    Mixed,
}

/// Tracking status for a single offset face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetStatus {
    /// The face was successfully offset.
    Done,
    /// The face was excluded from offsetting (e.g. thick-solid open faces).
    Excluded,
    /// The face offset failed and was skipped.
    Failed,
}

/// An offset face: the original face, its offset surface, and status.
#[derive(Debug, Clone)]
pub struct OffsetFace {
    /// The original face that was offset.
    pub original: FaceId,
    /// The offset surface geometry.
    pub surface: FaceSurface,
    /// The signed offset distance applied.
    pub distance: f64,
    /// Current status of this offset face.
    pub status: OffsetStatus,
}

/// The intersection curve between two adjacent offset faces, replacing
/// the original shared edge.
#[derive(Debug, Clone)]
pub struct FaceIntersection {
    /// The original edge shared by the two faces.
    pub original_edge: EdgeId,
    /// First adjacent face.
    pub face_a: FaceId,
    /// Second adjacent face.
    pub face_b: FaceId,
    /// Sampled points along the intersection curve.
    pub curve_points: Vec<Point3>,
    /// New edges created from this intersection.
    pub new_edges: Vec<EdgeId>,
}

/// A split point on an edge, recording the parameter value and the vertex
/// created at that location.
#[derive(Debug, Clone)]
pub struct SplitPoint {
    /// Parameter value on the original edge curve.
    pub parameter: f64,
    /// The vertex inserted at this split.
    pub vertex: VertexId,
}

/// Record of how an original edge was split into sub-edges.
#[derive(Debug, Clone)]
pub struct EdgeSplitRecord {
    /// The original edge before splitting.
    pub original: EdgeId,
    /// Ordered split points along the edge.
    pub splits: Vec<SplitPoint>,
    /// The new edges produced after splitting.
    pub new_edges: Vec<EdgeId>,
}

/// Strategy for joining adjacent offset faces at convex edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JointType {
    /// Extend faces until they intersect (sharp corners).
    #[default]
    Intersection,
    /// Insert a rolling-ball arc fillet between faces.
    Arc,
}

/// Configuration options for solid offset.
#[derive(Debug, Clone)]
pub struct OffsetOptions {
    /// How to join offset faces at convex edges.
    pub joint: JointType,
    /// Geometric tolerance for intersection and fitting.
    pub tolerance: Tolerance,
    /// Whether to detect and remove global self-intersections.
    pub remove_self_intersections: bool,
}

impl Default for OffsetOptions {
    fn default() -> Self {
        Self {
            joint: JointType::default(),
            tolerance: Tolerance::default(),
            remove_self_intersections: true,
        }
    }
}

/// Accumulated data from all phases of the offset pipeline.
///
/// Each phase reads from earlier fields and writes its own outputs.
#[derive(Debug, Clone)]
pub struct OffsetData {
    // --- Configuration ---
    /// The signed offset distance.
    pub distance: f64,
    /// Pipeline options.
    pub options: OffsetOptions,
    /// Faces excluded from offsetting (kept as-is in thick solid).
    pub excluded_faces: Vec<FaceId>,

    // --- Phase 1: analysis ---
    /// Edge convexity classification. Keys are edge indices from
    /// `edge_to_face_map`.
    pub edge_class: BTreeMap<usize, EdgeClass>,
    /// Vertex classification derived from incident edge classes.
    pub vertex_class: BTreeMap<VertexId, VertexClass>,

    // --- Phase 2: offset surfaces ---
    /// Offset face for each original face.
    pub offset_faces: BTreeMap<FaceId, OffsetFace>,

    // --- Phase 3 & 4: intersections ---
    /// Intersection curves between adjacent offset faces.
    pub intersections: Vec<FaceIntersection>,

    // --- Phase 5: edge splitting ---
    /// Records of how original edges were split at intersection points.
    pub edge_splits: BTreeMap<usize, EdgeSplitRecord>,

    // --- Phase 6: arc joints ---
    /// Faces created as rolling-ball arc joints at convex edges.
    pub joint_faces: Vec<FaceId>,

    // --- Phase 7: loops ---
    /// Wire loops built for each offset face from trimmed intersection
    /// curves.
    pub face_wires: BTreeMap<FaceId, Vec<WireId>>,
}

impl OffsetData {
    /// Create a new, empty `OffsetData` with the given configuration.
    #[must_use]
    pub fn new(distance: f64, options: OffsetOptions, excluded_faces: Vec<FaceId>) -> Self {
        Self {
            distance,
            options,
            excluded_faces,
            edge_class: BTreeMap::new(),
            vertex_class: BTreeMap::new(),
            offset_faces: BTreeMap::new(),
            intersections: Vec::new(),
            edge_splits: BTreeMap::new(),
            joint_faces: Vec::new(),
            face_wires: BTreeMap::new(),
        }
    }
}
