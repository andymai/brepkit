//! Mesh validation and boundary operations.

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use super::TriangleMesh;
use super::edge_sampling::sample_edge;

/// Check if a mesh is watertight (every edge shared by exactly 2 triangles).
///
/// Returns `true` if the mesh is a closed 2-manifold: every half-edge
/// `(a, b)` in the mesh has a corresponding reverse half-edge `(b, a)`.
///
/// This is useful for validating that `tessellate_solid` produces
/// gap-free meshes.
#[must_use]
pub fn is_watertight(mesh: &TriangleMesh) -> bool {
    use std::collections::HashSet;

    let mut half_edges: HashSet<(u32, u32)> = HashSet::new();
    let tri_count = mesh.indices.len() / 3;

    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3];
        let i1 = mesh.indices[t * 3 + 1];
        let i2 = mesh.indices[t * 3 + 2];
        half_edges.insert((i0, i1));
        half_edges.insert((i1, i2));
        half_edges.insert((i2, i0));
    }

    half_edges
        .iter()
        .all(|&(a, b)| half_edges.contains(&(b, a)))
}

/// Count boundary (non-manifold) edges in a mesh.
///
/// A boundary edge is one where the half-edge `(a, b)` exists but `(b, a)`
/// does not. Returns the number of such edges. A watertight mesh has 0.
#[must_use]
pub fn boundary_edge_count(mesh: &TriangleMesh) -> usize {
    use std::collections::HashSet;

    let mut half_edges: HashSet<(u32, u32)> = HashSet::new();
    let tri_count = mesh.indices.len() / 3;

    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3];
        let i1 = mesh.indices[t * 3 + 1];
        let i2 = mesh.indices[t * 3 + 2];
        half_edges.insert((i0, i1));
        half_edges.insert((i1, i2));
        half_edges.insert((i2, i0));
    }

    half_edges
        .iter()
        .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
        .count()
}

/// Edge polyline data for wireframe visualization.
///
/// Contains flattened position data for all edges in a solid, plus offsets
/// to identify where each edge's polyline starts.
#[derive(Debug, Clone, Default)]
pub struct EdgeLines {
    /// Vertex positions for all edge polylines (concatenated).
    pub positions: Vec<Point3>,
    /// Start index (in vertex count, not float count) of each edge polyline.
    /// The i-th edge's points are `positions[offsets[i]..offsets[i+1]]`
    /// (or `..positions.len()` for the last edge).
    pub offsets: Vec<usize>,
}

/// Check whether two face surfaces represent the same geometric surface.
fn surfaces_equivalent(a: &FaceSurface, b: &FaceSurface) -> bool {
    let tol = brepkit_math::tolerance::Tolerance::new();
    let lin = tol.linear;
    let ang = tol.angular;

    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            let dot = na.dot(*nb);
            (dot.abs() - 1.0).abs() < ang && (da - db * dot.signum()).abs() < lin
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            (ca.radius() - cb.radius()).abs() < lin
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.origin() - ca.origin();
                    let cross = d.cross(ca.axis());
                    cross.dot(cross) < lin * lin
                }
        }
        (FaceSurface::Cone(ca), FaceSurface::Cone(cb)) => {
            (ca.half_angle() - cb.half_angle()).abs() < ang
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.apex() - ca.apex();
                    d.dot(d) < lin * lin
                }
        }
        (FaceSurface::Sphere(sa), FaceSurface::Sphere(sb)) => {
            (sa.radius() - sb.radius()).abs() < lin && {
                let d = sb.center() - sa.center();
                d.dot(d) < lin * lin
            }
        }
        (FaceSurface::Torus(ta), FaceSurface::Torus(tb)) => {
            (ta.major_radius() - tb.major_radius()).abs() < lin
                && (ta.minor_radius() - tb.minor_radius()).abs() < lin
                && ta.z_axis().dot(tb.z_axis()).abs() > 1.0 - ang
                && {
                    let d = tb.center() - ta.center();
                    d.dot(d) < lin * lin
                }
        }
        (FaceSurface::Nurbs(_), FaceSurface::Nurbs(_)) => false,
        _ => false,
    }
}

/// Sample all edges of a solid into polylines for wireframe rendering.
///
/// Each edge is sampled according to the given `deflection` tolerance.
/// Returns [`EdgeLines`] containing the polyline data for all unique edges.
///
/// # Errors
///
/// Returns an error if topology traversal or edge sampling fails.
pub fn sample_solid_edges(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<EdgeLines, crate::OperationsError> {
    sample_solid_edges_filtered(topo, solid, deflection, true)
}

/// Sample edges of a solid, optionally filtering out smooth (co-surface) edges.
///
/// When `filter_smooth` is `true`, edges shared by two faces on the same
/// underlying geometric surface are omitted. These edges arise from boolean
/// face-splitting and add wireframe clutter without representing visible creases.
///
/// # Errors
///
/// Returns an error if topology traversal or edge sampling fails.
pub fn sample_solid_edges_filtered(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    filter_smooth: bool,
) -> Result<EdgeLines, crate::OperationsError> {
    let edges = brepkit_topology::explorer::solid_edges(topo, solid)?;

    let edge_face_map = if filter_smooth {
        Some(brepkit_topology::explorer::edge_to_face_map(topo, solid)?)
    } else {
        None
    };

    let mut result = EdgeLines {
        positions: Vec::new(),
        offsets: Vec::with_capacity(edges.len()),
    };

    for edge_id in &edges {
        if let Some(ref efm) = edge_face_map {
            if let Some(faces) = efm.get(&edge_id.index()) {
                if faces.len() == 2 {
                    let fa = topo.face(faces[0])?;
                    let fb = topo.face(faces[1])?;
                    if surfaces_equivalent(fa.surface(), fb.surface()) {
                        continue;
                    }
                }
            }
        }

        result.offsets.push(result.positions.len());
        let edge = topo.edge(*edge_id)?;
        let points = sample_edge(topo, edge, deflection)?;
        result.positions.extend(points);
    }

    Ok(result)
}

/// Weld remaining boundary vertices by merging coincident positions.
///
/// Uses union-find over a spatial hash grid to merge boundary vertices that
/// are within `weld_tol` of each other. Rewrites triangle indices and removes
/// degenerate triangles (where merged indices create duplicate vertices).
pub(super) fn weld_boundary_vertices(mesh: &mut TriangleMesh, deflection: f64) {
    use std::collections::{HashMap, HashSet};

    let n_verts = mesh.positions.len();
    if n_verts == 0 || mesh.indices.is_empty() {
        return;
    }

    // Build half-edge set to find boundary edges.
    let mut half_edges: HashMap<(u32, u32), usize> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
        *half_edges.entry((i0, i1)).or_default() += 1;
        *half_edges.entry((i1, i2)).or_default() += 1;
        *half_edges.entry((i2, i0)).or_default() += 1;
    }

    // Boundary vertices: incident on half-edges without a matching reverse.
    let mut boundary_verts: HashSet<u32> = HashSet::new();
    for &(a, b) in half_edges.keys() {
        if !half_edges.contains_key(&(b, a)) {
            boundary_verts.insert(a);
            boundary_verts.insert(b);
        }
    }

    if boundary_verts.is_empty() {
        return;
    }

    #[allow(clippy::items_after_statements)]
    fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    #[allow(clippy::items_after_statements)]
    fn uf_union(parent: &mut [u32], a: u32, b: u32) {
        let ra = uf_find(parent, a);
        let rb = uf_find(parent, b);
        if ra != rb {
            parent[rb as usize] = ra;
        }
    }

    let mut parent: Vec<u32> = (0..n_verts as u32).collect();

    let weld_tol = deflection.max(1e-6) * 2.0;
    let inv_cell = 1.0 / weld_tol;

    #[allow(clippy::cast_possible_truncation)]
    let cell_key = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * inv_cell).floor() as i64,
            (p.y() * inv_cell).floor() as i64,
            (p.z() * inv_cell).floor() as i64,
        )
    };

    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for &vid in &boundary_verts {
        let p = mesh.positions[vid as usize];
        grid.entry(cell_key(p)).or_default().push(vid);
    }

    for &vid in &boundary_verts {
        let p = mesh.positions[vid as usize];
        let (cx, cy, cz) = cell_key(p);

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(cell) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &other in cell {
                            if other <= vid {
                                continue;
                            }
                            let q = mesh.positions[other as usize];
                            if (p - q).length() < weld_tol {
                                uf_union(&mut parent, vid, other);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut changed = false;
    for idx in &mut mesh.indices {
        let root = uf_find(&mut parent, *idx);
        if root != *idx {
            *idx = root;
            changed = true;
        }
    }

    if changed {
        let mut new_indices = Vec::with_capacity(mesh.indices.len());
        for tri in mesh.indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
            if i0 != i1 && i1 != i2 && i2 != i0 {
                new_indices.push(i0);
                new_indices.push(i1);
                new_indices.push(i2);
            }
        }
        mesh.indices = new_indices;
    }
}
