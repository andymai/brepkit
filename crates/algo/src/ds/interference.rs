//! Interference records from PaveFiller phases.

use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

use super::pave::PaveBlockId;

/// A single interference record between two shapes.
#[derive(Debug, Clone)]
pub enum Interference {
    /// Vertex-vertex coincidence.
    VV {
        /// First vertex.
        v1: VertexId,
        /// Second vertex.
        v2: VertexId,
    },
    /// Vertex lies on edge.
    VE {
        /// The vertex.
        vertex: VertexId,
        /// The edge it lies on.
        edge: EdgeId,
        /// Parameter on the edge curve.
        parameter: f64,
    },
    /// Edge-edge intersection or overlap.
    EE {
        /// First edge.
        e1: EdgeId,
        /// Second edge.
        e2: EdgeId,
        /// New vertex at crossing point, if created.
        new_vertex: Option<VertexId>,
        /// Common pave block if edges overlap.
        common_pave_block: Option<PaveBlockId>,
    },
    /// Vertex lies on face.
    VF {
        /// The vertex.
        vertex: VertexId,
        /// The face it lies on.
        face: FaceId,
        /// UV parameters on the face surface.
        uv: (f64, f64),
    },
    /// Edge crosses or lies on face.
    EF {
        /// The edge.
        edge: EdgeId,
        /// The face.
        face: FaceId,
        /// New vertex at crossing point, if created.
        new_vertex: Option<VertexId>,
        /// Parameter on edge at intersection.
        parameter: Option<f64>,
    },
    /// Face-face intersection.
    FF {
        /// First face.
        f1: FaceId,
        /// Second face.
        f2: FaceId,
        /// Index into `GfaArena.curves` for the intersection curve.
        curve_index: usize,
    },
}

/// Table of all interferences discovered during PaveFiller.
#[derive(Debug, Clone, Default)]
pub struct InterferenceTable {
    /// Vertex-vertex coincidences.
    pub vv: Vec<Interference>,
    /// Vertex-on-edge interferences.
    pub ve: Vec<Interference>,
    /// Edge-edge intersections.
    pub ee: Vec<Interference>,
    /// Vertex-on-face interferences.
    pub vf: Vec<Interference>,
    /// Edge-face intersections.
    pub ef: Vec<Interference>,
    /// Face-face intersections.
    pub ff: Vec<Interference>,
}
