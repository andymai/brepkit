//! Convert a tessellated solid into GPU-ready vertex/index/edge buffers.
//!
//! Tessellation runs in f64. Positions are uploaded as f32 offsets from the
//! model AABB center (RTC); the f64 center travels separately into the camera
//! matrix (see [`crate::camera`]).

use brepkit_math::vec::Point3;
use brepkit_operations::tessellate::{
    EdgeLines, sample_solid_edges, tessellate_solid_grouped_with_tolerance,
};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

use crate::error::RenderError;

/// A single mesh vertex as uploaded to the GPU.
///
/// `position` is relative to the model center (RTC). `face_id` is the owning
/// face's arena index **plus one** so the id target's `0` clear value is an
/// unambiguous "background" sentinel.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Center-relative position (f32).
    pub position: [f32; 3],
    /// Per-vertex normal (f32, world-space direction).
    pub normal: [f32; 3],
    /// `FaceId.index() + 1`; `0` is reserved for background.
    pub face_id: u32,
}

/// A single edge-line vertex (center-relative position only).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeVertex {
    /// Center-relative position (f32).
    pub position: [f32; 3],
}

/// CPU-side render geometry derived from a solid.
pub struct RenderMesh {
    /// Interleaved mesh vertices (position, normal, face id).
    pub vertices: Vec<Vertex>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u32>,
    /// Line-list vertices for the topological edges (pairs form segments).
    pub edge_vertices: Vec<EdgeVertex>,
    /// Model AABB center in f64 world space (the RTC origin).
    pub center: Point3,
}

impl RenderMesh {
    /// Tessellate `solid` and build center-relative GPU geometry.
    ///
    /// `deflection` is the linear chord tolerance passed to the tessellator.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Operations`] / [`RenderError::Topology`] if
    /// tessellation or topology traversal fails, or [`RenderError::MeshData`]
    /// if the tessellation violates a mesh invariant (index buffer not a whole
    /// number of triangles, an out-of-range index, or incomplete face offsets).
    pub fn build(topo: &Topology, solid: SolidId, deflection: f64) -> Result<Self, RenderError> {
        let angular_tol = brepkit_math::chord::DEFAULT_ANGULAR_TOL;
        let (mesh, face_offsets) =
            tessellate_solid_grouped_with_tolerance(topo, solid, deflection, angular_tol)?;
        let faces = solid_faces(topo, solid)?;

        let center = aabb_center(&mesh.positions);

        // Per-vertex base data in RTC. Normals are direction-only, so they are
        // unaffected by the center subtraction.
        let positions_rtc: Vec<[f32; 3]> = mesh
            .positions
            .iter()
            .map(|&p| {
                let d = p - center;
                #[allow(clippy::cast_possible_truncation)]
                [d.x() as f32, d.y() as f32, d.z() as f32]
            })
            .collect();
        let normals: Vec<[f32; 3]> = mesh
            .normals
            .iter()
            .map(|&n| {
                #[allow(clippy::cast_possible_truncation)]
                [n.x() as f32, n.y() as f32, n.z() as f32]
            })
            .collect();

        // A triangle mesh's index buffer must be a whole number of triangles.
        if mesh.indices.len() % 3 != 0 {
            return Err(RenderError::MeshData(format!(
                "index buffer length {} is not divisible by 3",
                mesh.indices.len()
            )));
        }
        let tri_count = mesh.indices.len() / 3;

        // The grouped tessellator returns one offset per face plus a final
        // sentinel, so `face_offsets[i]..face_offsets[i + 1]` is face i's range.
        // A short array would silently mis-attribute triangles, so require it.
        if face_offsets.len() != faces.len() + 1 {
            return Err(RenderError::MeshData(format!(
                "face_offsets has {} entries, expected {} (one per face + sentinel)",
                face_offsets.len(),
                faces.len() + 1
            )));
        }

        // A vertex shared between two faces would need two different face ids,
        // so we cannot reuse the welded index buffer directly. Expand to
        // non-indexed-per-face: one fresh vertex per triangle corner, tagged
        // with that triangle's owning face. (Edges still come from topology.)
        let mut vertices: Vec<Vertex> = Vec::with_capacity(tri_count * 3);
        let mut indices: Vec<u32> = Vec::with_capacity(tri_count * 3);

        // Walk faces in lockstep with the index buffer rather than searching
        // per triangle.
        let mut tri_face_ids = vec![0_u32; tri_count];
        for (i, face) in faces.iter().enumerate() {
            let start = face_offsets[i] as usize / 3;
            let end = face_offsets[i + 1] as usize / 3;
            #[allow(clippy::cast_possible_truncation)]
            let id = face.index() as u32 + 1;
            for slot in tri_face_ids
                .iter_mut()
                .take(end.min(tri_count))
                .skip(start.min(tri_count))
            {
                *slot = id;
            }
        }

        for t in 0..tri_count {
            let face_id = tri_face_ids[t];
            for k in 0..3 {
                let vi = mesh.indices[t * 3 + k] as usize;
                // Out-of-range indices mean a corrupt tessellation; fail rather
                // than substituting placeholder geometry / picking ids.
                let (Some(&position), Some(&normal)) = (positions_rtc.get(vi), normals.get(vi))
                else {
                    return Err(RenderError::MeshData(format!(
                        "triangle index {vi} is out of range ({} vertices)",
                        positions_rtc.len()
                    )));
                };
                #[allow(clippy::cast_possible_truncation)]
                let idx = vertices.len() as u32;
                vertices.push(Vertex {
                    position,
                    normal,
                    face_id,
                });
                indices.push(idx);
            }
        }

        let edges = sample_solid_edges(topo, solid, deflection)?;
        let edge_vertices = build_edge_lines(&edges, center);

        Ok(Self {
            vertices,
            indices,
            edge_vertices,
            center,
        })
    }
}

/// Build a line-list (segment-pair) vertex buffer from edge polylines.
///
/// Each polyline of `n` points becomes `n - 1` segments, i.e. `2 * (n - 1)`
/// vertices, so the GPU `LineList` topology draws every span.
fn build_edge_lines(edges: &EdgeLines, center: Point3) -> Vec<EdgeVertex> {
    let mut out = Vec::new();
    let n_edges = edges.offsets.len();
    for e in 0..n_edges {
        let start = edges.offsets[e];
        let end = edges
            .offsets
            .get(e + 1)
            .copied()
            .unwrap_or(edges.positions.len());
        let pts = &edges.positions[start..end];
        for w in pts.windows(2) {
            for &p in w {
                let d = p - center;
                #[allow(clippy::cast_possible_truncation)]
                out.push(EdgeVertex {
                    position: [d.x() as f32, d.y() as f32, d.z() as f32],
                });
            }
        }
    }
    out
}

/// Compute the center of the axis-aligned bounding box of a point set.
///
/// Returns the origin for an empty set so the RTC math stays well-defined.
fn aabb_center(positions: &[Point3]) -> Point3 {
    if positions.is_empty() {
        return Point3::new(0.0, 0.0, 0.0);
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in positions {
        let c = [p.x(), p.y(), p.z()];
        for i in 0..3 {
            if c[i] < min[i] {
                min[i] = c[i];
            }
            if c[i] > max[i] {
                max[i] = c[i];
            }
        }
    }
    Point3::new(
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    )
}
