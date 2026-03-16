//! Shared type definitions, constants, and the selection truth table for the
//! boolean pipeline.

use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of samples used when discretizing closed curves (circles, ellipses)
/// in the analytic boolean path. All code paths must use this constant so that
/// band fragments, cap face polygons, and holed-face inner wires share the
/// same vertices and edges at their boundaries.
pub(super) const CLOSED_CURVE_SAMPLES: usize = 32;

/// Minimum fragment count for parallel classification via rayon.
/// Below this threshold, sequential iteration is faster due to rayon's
/// thread-pool synchronization overhead (~5-20us).
#[cfg(not(target_arch = "wasm32"))]
pub(super) const PARALLEL_THRESHOLD: usize = 64;

/// Default tessellation deflection for non-planar faces in boolean operations.
///
/// A larger value produces fewer triangles (faster but coarser approximation).
/// Since the boolean result decomposes curved faces into individual planar
/// triangles, keeping this coarse avoids face-count explosion in sequential
/// boolean operations.
pub(super) const DEFAULT_BOOLEAN_DEFLECTION: f64 = 0.1;

/// Number of angular segments used to approximate cylinder faces in the
/// classification face data. 16 segments = 16 quads per cylinder band,
/// sufficient for correct ray-crossing parity.
pub(super) const CLASSIFIER_CYL_SEGMENTS: usize = 16;

/// Per-solid face count above which the chord-based tessellated path is
/// too expensive. If EITHER solid exceeds this, fall back to mesh boolean.
///
/// This catches individual complex solids (e.g. shelled geometry with merged
/// faces from `unify_faces`) that would cause O(N²) intersection computation.
/// Matches the v2.6.0 behavior that prevented hangs on gridfinity bins.
pub(super) const MESH_BOOLEAN_PER_SOLID_THRESHOLD: usize = 100;

/// Combined face count (A + B) above which the chord-based tessellated
/// path is too expensive. Falls back to the mesh boolean (co-refinement
/// on triangle meshes, O(N log N)).
///
/// Checked BEFORE `collect_face_data` to avoid expensive NURBS
/// tessellation when the mesh boolean will be used anyway.
pub(super) const MESH_BOOLEAN_FACE_THRESHOLD: usize = 500;

/// Threshold: use CDT batch splitting for faces with this many or more chords.
///
/// Below this threshold, the iterative approach is fast enough and avoids the
/// CDT setup overhead. Above it, the iterative O(N*F) approach becomes a
/// bottleneck while CDT stays O(N log N).
pub(super) const CDT_CHORD_THRESHOLD: usize = 5;

/// Snap distance multiplier for CDT vertex matching.
///
/// Chord endpoints are computed by line-edge intersection, which accumulates
/// floating-point error on the order of ~10x `tol.linear`. Use 100x as the
/// snap threshold to reliably capture all on-chord/on-boundary vertices
/// without pulling in nearby-but-off-chord polygon vertices.
pub(super) const CDT_SNAP_FACTOR: f64 = 100.0;

/// Minimum face count for a valid solid.
///
/// A cylinder (2 caps + 1 barrel = 3 faces) is the minimal closed solid
/// produced by boolean operations between boxes and curved primitives.
pub(super) const MIN_SOLID_FACES: usize = 3;

// ---------------------------------------------------------------------------
// Public enums
// ---------------------------------------------------------------------------

/// The type of boolean operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Union of two solids.
    Fuse,
    /// Subtraction: first minus second.
    Cut,
    /// Intersection: common volume.
    Intersect,
}

/// A face specification for mixed-surface solid assembly.
///
/// Used by [`super::assemble_solid_mixed`] to build solids with faces of any
/// surface type -- not just planar.
#[derive(Clone)]
pub enum FaceSpec {
    /// A planar face defined by vertex positions and plane equation.
    Planar {
        /// Vertex positions (at least 3).
        vertices: Vec<Point3>,
        /// Outward-facing normal.
        normal: Vec3,
        /// Plane equation signed distance (n * p = d).
        d: f64,
        /// Inner wire vertex loops (holes in the face).
        inner_wires: Vec<Vec<Point3>>,
    },
    /// A face with a pre-built surface and vertex positions for the boundary wire.
    Surface {
        /// Vertex positions for the outer wire (at least 3).
        vertices: Vec<Point3>,
        /// The surface geometry.
        surface: FaceSurface,
        /// Whether the face's surface normal should be reversed.
        reversed: bool,
        /// Inner wire vertex loops (holes in the face).
        inner_wires: Vec<Vec<Point3>>,
    },
    /// A cylindrical face with circle edges on angular boundaries.
    ///
    /// Unlike `Surface`, this variant creates `EdgeCurve::Circle` for edges
    /// that span an angular range on the cylinder (constant-v boundaries),
    /// preserving curve geometry for correct tessellation and volume computation.
    CylindricalFace {
        /// Vertex positions for the outer wire (at least 3).
        vertices: Vec<Point3>,
        /// The cylindrical surface geometry.
        cylinder: CylindricalSurface,
        /// Whether the face's surface normal should be reversed.
        reversed: bool,
        /// Inner wire vertex loops (holes in the face).
        inner_wires: Vec<Vec<Point3>>,
    },
}

impl FaceSpec {
    /// Returns a reference to this face's inner wires.
    #[must_use]
    pub fn inner_wires(&self) -> &[Vec<Point3>] {
        match self {
            Self::Planar { inner_wires, .. }
            | Self::Surface { inner_wires, .. }
            | Self::CylindricalFace { inner_wires, .. } => inner_wires,
        }
    }

    /// Returns a mutable reference to this face's inner wires.
    pub fn inner_wires_mut(&mut self) -> &mut Vec<Vec<Point3>> {
        match self {
            Self::Planar { inner_wires, .. }
            | Self::Surface { inner_wires, .. }
            | Self::CylindricalFace { inner_wires, .. } => inner_wires,
        }
    }
}

// ---------------------------------------------------------------------------
// Public structs
// ---------------------------------------------------------------------------

/// Options for boolean operations.
#[derive(Debug, Clone, Copy)]
pub struct BooleanOptions {
    /// Tessellation deflection for non-planar faces.
    ///
    /// Lower values produce more triangles (more accurate but slower).
    /// Default: 0.1.
    pub deflection: f64,
    /// Tolerance for geometric comparisons.
    ///
    /// Controls vertex merging, point classification, and predicate
    /// thresholds throughout the boolean pipeline. Default: `Tolerance::new()`.
    pub tolerance: Tolerance,
    /// Merge co-surface face fragments after assembly.
    ///
    /// When `true`, the pipeline calls `unify_faces` to merge adjacent faces
    /// that share the same underlying surface (analogous to OCCT's same-domain
    /// analysis). This dramatically reduces face count -- e.g. sequential
    /// booleans on curved surfaces drop from 2871 to ~106 faces.
    ///
    /// Non-convex merged faces are handled correctly by the
    /// `polygon_clip_intervals` fallback in the analytic chord splitter,
    /// so this is safe for intermediate results fed into further booleans.
    ///
    /// Default: `true`.
    pub unify_faces: bool,
    /// Run full shape healing on the boolean result via [`heal_solid`].
    ///
    /// Use for final results only -- healing can corrupt intermediates fed into
    /// further booleans (non-convex merged faces confuse chord splitting).
    ///
    /// Default: `false`.
    pub heal_after_boolean: bool,
}

impl Default for BooleanOptions {
    fn default() -> Self {
        Self {
            deflection: DEFAULT_BOOLEAN_DEFLECTION,
            tolerance: Tolerance::new(),
            unify_faces: true,
            heal_after_boolean: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal enums
// ---------------------------------------------------------------------------

/// Which operand a face fragment originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    A,
    B,
}

/// Classification of a face fragment relative to the opposite solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FaceClass {
    Outside,
    Inside,
    CoplanarSame,
    CoplanarOpposite,
}

/// Analytic classifier for simple convex solids.
///
/// Instead of ray-casting against hundreds of tessellated triangles, use
/// exact geometric predicates to classify points inside/outside a solid.
pub(super) enum AnalyticClassifier {
    /// Point-in-sphere: `|p - center| <= radius`.
    Sphere { center: Point3, radius: f64 },
    /// Point-in-cylinder: radial distance from axis <= radius AND axial
    /// position within [z_min, z_max].
    Cylinder {
        origin: Point3,
        axis: Vec3,
        radius: f64,
        z_min: f64,
        z_max: f64,
    },
    /// Point-in-cone-frustum: radial distance from axis <= interpolated radius
    /// AND axial position within [z_min, z_max]. Uses linear interpolation
    /// between r_min (at z_min) and r_max (at z_max) for the expected radius,
    /// which is robust regardless of `ConicalSurface` apex/axis orientation.
    Cone {
        origin: Point3,
        axis: Vec3,
        z_min: f64,
        z_max: f64,
        r_at_z_min: f64,
        r_at_z_max: f64,
    },
    /// Point-in-box: axis-aligned bounding box test.
    /// O(1) with just 6 comparisons -- the fastest classifier.
    Box { min: Point3, max: Point3 },
    /// Point-in-convex-polyhedron: half-plane test against each face.
    /// A point is inside iff `normal_i * p < d_i` for all face planes
    /// (outward-pointing normals, so `normal * p > d` means outside).
    /// O(F) where F is the number of faces -- fast for hex prisms (F=8).
    ConvexPolyhedron {
        /// Outward-pointing normals and signed distances: `normal * p > d` means outside.
        planes: Vec<(Vec3, f64)>,
    },
}

/// Result of classifying an intersection curve against a face boundary.
pub(super) enum CurveClassification {
    /// The curve crosses the face boundary -- contains entry/exit points.
    Crossings(Vec<Point3>),
    /// The entire curve lies inside the face (no boundary crossings).
    FullyContained,
    /// The entire curve lies outside the face.
    FullyOutside,
}

// ---------------------------------------------------------------------------
// Internal structs
// ---------------------------------------------------------------------------

/// Internal context carrying tolerance-derived thresholds through the boolean
/// pipeline. Computed once from `BooleanOptions` at the start of a boolean
/// operation to avoid repeated derivation and hardcoded epsilon values.
#[derive(Debug, Clone, Copy)]
pub(super) struct BooleanContext {
    /// Base tolerance (used when wiring ctx through the full pipeline).
    #[allow(dead_code)]
    pub(super) tol: Tolerance,
    /// Vertex merge distance: vertices closer than this are considered identical.
    pub(super) vertex_merge: f64,
    /// Point classification tolerance: distance threshold for on-surface tests.
    pub(super) classify_tol: f64,
    /// Degenerate polygon threshold: skip polygons with area below this.
    pub(super) degenerate_area: f64,
}

impl BooleanContext {
    pub(super) fn from_options(opts: &BooleanOptions) -> Self {
        let tol = opts.tolerance;
        Self {
            tol,
            // 1000x linear tolerance for vertex merging -- aggressive enough to
            // catch coincident vertices while preserving distinct features.
            vertex_merge: tol.linear * 1000.0,
            // Classification tolerance for point-in-solid tests.
            classify_tol: tol.linear * 100.0,
            // Degenerate area threshold (area < this -> skip polygon).
            degenerate_area: tol.linear * tol.linear,
        }
    }
}

/// An intersection segment between two faces.
#[derive(Debug)]
pub(super) struct IntersectionSegment {
    pub(super) face_a: FaceId,
    pub(super) face_b: FaceId,
    pub(super) p0: Point3,
    pub(super) p1: Point3,
}

/// A fragment of a face after splitting along intersection chords.
#[derive(Debug)]
pub(super) struct FaceFragment {
    pub(super) vertices: Vec<Point3>,
    pub(super) normal: Vec3,
    pub(super) d: f64,
    pub(super) source: Source,
}

/// Parameters for a single face in a face-pair intersection test.
pub(super) struct FacePairSide<'a> {
    pub(super) fid: FaceId,
    pub(super) verts: &'a [Point3],
    pub(super) normal: Vec3,
    pub(super) d: f64,
}

/// Snapshot of face data for analytic boolean processing.
pub(super) struct FaceSnapshot {
    pub(super) id: FaceId,
    pub(super) surface: FaceSurface,
    pub(super) vertices: Vec<Point3>,
    pub(super) normal: Vec3,
    pub(super) d: f64,
    /// Whether the original face was reversed (needed to preserve orientation
    /// when carrying unsplit faces through sequential booleans).
    pub(super) reversed: bool,
}

/// Analytic face fragment preserving the original surface type.
pub(super) struct AnalyticFragment {
    /// Polygon boundary in 3D (for classification and planar assembly fallback).
    pub(super) vertices: Vec<Point3>,
    /// The original surface type of the face.
    pub(super) surface: FaceSurface,
    /// Normal of the face (for planar) or of the polygon approximation.
    pub(super) normal: Vec3,
    /// Plane d coefficient (for planar faces).
    pub(super) d: f64,
    /// Which operand this fragment came from.
    pub(super) source: Source,
    /// Edge curve types for the boundary segments.
    /// `None` = straight line, `Some(curve)` = exact curve (circle, ellipse).
    pub(super) edge_curves: Vec<Option<EdgeCurve>>,
    /// Whether the source face was reversed (preserved for non-planar faces).
    pub(super) source_reversed: bool,
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Extracted face data: `(FaceId, vertices, normal, d)`.
pub(super) type FaceData = Vec<(FaceId, Vec<Point3>, Vec3, f64)>;

// ---------------------------------------------------------------------------
// Selection truth table
// ---------------------------------------------------------------------------

/// Determine whether a fragment should be kept and whether to flip its normal.
///
/// Returns `Some(false)` to keep as-is, `Some(true)` to keep and flip, or
/// `None` to discard.
#[allow(clippy::match_same_arms)] // arms are semantically distinct (truth table rows)
pub(super) const fn select_fragment(
    source: Source,
    class: FaceClass,
    op: BooleanOp,
) -> Option<bool> {
    match (source, class, op) {
        // From A, Outside B
        (Source::A, FaceClass::Outside, BooleanOp::Fuse | BooleanOp::Cut) => Some(false),
        (Source::A, FaceClass::Outside, BooleanOp::Intersect) => None,
        // From A, Inside B
        (Source::A, FaceClass::Inside, BooleanOp::Fuse | BooleanOp::Cut) => None,
        (Source::A, FaceClass::Inside, BooleanOp::Intersect) => Some(false),
        // From B, Outside A
        (Source::B, FaceClass::Outside, BooleanOp::Fuse) => Some(false),
        (Source::B, FaceClass::Outside, BooleanOp::Cut | BooleanOp::Intersect) => None,
        // From B, Inside A
        (Source::B, FaceClass::Inside, BooleanOp::Fuse) => None,
        (Source::B, FaceClass::Inside, BooleanOp::Cut) => Some(true), // flip
        (Source::B, FaceClass::Inside, BooleanOp::Intersect) => Some(false),
        // Coplanar same -- keep only from A to avoid duplicates.
        (Source::A, FaceClass::CoplanarSame, BooleanOp::Fuse | BooleanOp::Intersect) => Some(false),
        (_, FaceClass::CoplanarSame, _) => None,
        // Coplanar opposite -- for Cut, A's face facing opposite B should be kept
        // (it forms the "skin" at the cut boundary). In all other cases, discard.
        (Source::A, FaceClass::CoplanarOpposite, BooleanOp::Cut) => Some(false),
        (_, FaceClass::CoplanarOpposite, _) => None,
    }
}
