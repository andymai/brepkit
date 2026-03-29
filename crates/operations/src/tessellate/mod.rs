//! Tessellation: convert B-Rep faces to triangle meshes.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::tuple_array_conversions,
    clippy::imprecise_flops,
    clippy::too_many_lines,
    clippy::option_if_let_else,
    clippy::bool_to_int_with_if,
    clippy::if_same_then_else,
    clippy::used_underscore_binding,
    clippy::map_unwrap_or
)]

mod edge_sampling;
mod face;
mod mesh_ops;
mod nonplanar;
mod nurbs;
mod planar;
mod solid;
#[cfg(test)]
mod tests;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

// Re-export all public items.
pub use face::tessellate_with_uvs;
pub use mesh_ops::{
    EdgeLines, boundary_edge_count, is_watertight, sample_solid_edges, sample_solid_edges_filtered,
    weld_boundary_vertices,
};
pub use solid::tessellate_solid;

/// Merge-grid cell size for tolerance-based vertex deduplication.
///
/// Vertices within this distance are quantized to the same grid cell and share
/// a single global vertex index. This catches near-identical vertices produced
/// when boolean operations create separate edge entities for the same curve.
const MERGE_GRID: f64 = 1e-7;

/// Quantize a 3D point to a spatial grid cell for tolerance-based deduplication.
///
/// Two points whose coordinates differ by less than `grid` will (usually) map to
/// the same `(i64, i64, i64)` key. Grid-boundary splits are possible but rare,
/// and the subsequent CDT / weld phases handle any remaining gaps.
pub(super) fn point_merge_key(pt: Point3, grid: f64) -> (i64, i64, i64) {
    #[allow(clippy::cast_possible_truncation)]
    (
        (pt.x() / grid).round() as i64,
        (pt.y() / grid).round() as i64,
        (pt.z() / grid).round() as i64,
    )
}

/// Compute the shorter arc range (<=pi) from an edge's start to end on a circle.
///
/// Returns `(t_start, t_end)` where the shorter arc goes from `t_start` to `t_end`.
/// When the shorter arc is CW, `t_end < t_start` so that linear interpolation
/// between them traces the correct (shorter) path via `circle.evaluate()`.
pub(super) fn shorter_arc_range(
    circle: &brepkit_math::curves::Circle3D,
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
) -> Result<(f64, f64), crate::OperationsError> {
    let sp = topo.vertex(edge.start())?.point();
    let ep = topo.vertex(edge.end())?.point();
    let ts = circle.project(sp);
    let te_raw = circle.project(ep);
    let fwd_span = (te_raw - ts).rem_euclid(std::f64::consts::TAU);
    if fwd_span <= std::f64::consts::PI {
        // CCW arc is the shorter path.
        Ok((ts, ts + fwd_span))
    } else {
        // CW arc is shorter: t_end < t_start so interpolation goes backward.
        let rev_span = std::f64::consts::TAU - fwd_span;
        Ok((ts, ts - rev_span))
    }
}

/// A triangle mesh produced by tessellation.
#[derive(Debug, Clone, Default)]
pub struct TriangleMesh {
    /// Vertex positions.
    pub positions: Vec<Point3>,
    /// Per-vertex normals.
    pub normals: Vec<Vec3>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

/// A triangle mesh with per-vertex UV coordinates.
#[derive(Debug, Clone, Default)]
pub struct TriangleMeshUV {
    /// The base mesh (positions, normals, indices).
    pub mesh: TriangleMesh,
    /// Per-vertex UV coordinates (same length as `mesh.positions`).
    pub uvs: Vec<[f64; 2]>,
}

/// Kind of special handling needed for analytic surface tessellation.
pub(super) enum AnalyticKind {
    /// Standard quad grid with no degenerate handling.
    General,
    /// Triangle fan at v extremes (sphere poles at v_min and v_max).
    SpherePole,
    /// Triangle fan at v_min (cone apex at v = 0).
    ConeApex,
    /// Triangle fan at v_max only (sphere north pole for a hemisphere face).
    VMaxPole,
}

/// Tessellate a face into a triangle mesh.
///
/// For planar faces, this performs fan triangulation from the first vertex,
/// which produces correct results for convex polygons.
///
/// For NURBS faces, the surface is sampled on a uniform (u, v) grid whose
/// density is derived from `deflection` — smaller values produce finer meshes.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    tessellate_with_uvs(topo, face, deflection).map(|uv| uv.mesh)
}

/// Check if a mesh is watertight (every edge shared by exactly 2 triangles).
///
/// Returns `true` if the mesh is a closed 2-manifold: every half-edge
/// `(a, b)` in the mesh has a corresponding reverse half-edge `(b, a)`.
///
/// This is useful for validating that `tessellate_solid` produces
/// gap-free meshes.
#[cfg(test)]
#[must_use]
pub(super) fn position_based_boundary_count(mesh: &TriangleMesh) -> usize {
    use std::collections::{HashMap, HashSet};

    /// 1um grid -- intentionally coarser than `MERGE_GRID` (1e-7) to catch
    /// gaps the production pipeline should have closed.
    const DIAGNOSTIC_GRID: f64 = 1e-6;

    // Build canonical vertex ID from snapped position.
    let mut pos_to_canonical: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut canonical_ids: Vec<u32> = Vec::with_capacity(mesh.positions.len());
    let mut next_id: u32 = 0;

    for pos in &mesh.positions {
        let key = point_merge_key(*pos, DIAGNOSTIC_GRID);
        let id = *pos_to_canonical.entry(key).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        canonical_ids.push(id);
    }

    // Build half-edge set using canonical IDs.
    let mut half_edges: HashSet<(u32, u32)> = HashSet::new();
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = canonical_ids[tri[0] as usize];
        let i1 = canonical_ids[tri[1] as usize];
        let i2 = canonical_ids[tri[2] as usize];
        half_edges.insert((i0, i1));
        half_edges.insert((i1, i2));
        half_edges.insert((i2, i0));
    }

    half_edges
        .iter()
        .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
        .count()
}
