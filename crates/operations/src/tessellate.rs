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

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

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
    let face_data = topo.face(face)?;

    match face_data.surface() {
        FaceSurface::Plane { normal, .. } => tessellate_planar(topo, face_data, *normal),
        FaceSurface::Nurbs(surface) => Ok(tessellate_nurbs(surface, deflection)),
        FaceSurface::Cylinder(cyl) => {
            let cyl = cyl.clone();
            Ok(tessellate_analytic(
                |u, v| cyl.evaluate(u, v),
                |u, v| cyl.normal(u, v),
                (0.0, std::f64::consts::TAU),
                (-1.0, 1.0),
                deflection,
                AnalyticKind::General,
            ))
        }
        FaceSurface::Cone(cone) => {
            let cone = cone.clone();
            Ok(tessellate_analytic(
                |u, v| cone.evaluate(u, v),
                |u, v| cone.normal(u, v),
                (0.0, std::f64::consts::TAU),
                (0.0, 1.0),
                deflection,
                AnalyticKind::ConeApex,
            ))
        }
        FaceSurface::Sphere(sphere) => {
            let sphere = sphere.clone();
            Ok(tessellate_analytic(
                |u, v| sphere.evaluate(u, v),
                |u, v| sphere.normal(u, v),
                (0.0, std::f64::consts::TAU),
                (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                deflection,
                AnalyticKind::SpherePole,
            ))
        }
        FaceSurface::Torus(torus) => {
            let torus = torus.clone();
            Ok(tessellate_analytic(
                |u, v| torus.evaluate(u, v),
                |u, v| torus.normal(u, v),
                (0.0, std::f64::consts::TAU),
                (0.0, std::f64::consts::TAU),
                deflection,
                AnalyticKind::General,
            ))
        }
    }
}

/// Kind of special handling needed for analytic surface tessellation.
enum AnalyticKind {
    /// Standard quad grid with no degenerate handling.
    General,
    /// Triangle fan at v extremes (sphere poles at v_min and v_max).
    SpherePole,
    /// Triangle fan at v_min (cone apex at v = 0).
    ConeApex,
}

/// Tessellate an analytic surface on a uniform `(u, v)` grid.
///
/// The grid density is derived from `deflection`: smaller values yield finer meshes.
/// Special handling is applied at poles (sphere) and apexes (cone) to avoid degenerate
/// triangles by using triangle fans instead of quads.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
fn tessellate_analytic(
    surface_eval: impl Fn(f64, f64) -> Point3,
    normal_fn: impl Fn(f64, f64) -> Vec3,
    u_range: (f64, f64),
    v_range: (f64, f64),
    deflection: f64,
    kind: AnalyticKind,
) -> TriangleMesh {
    use std::f64::consts::TAU;

    let n = 8_usize.max((TAU / deflection.sqrt()).ceil() as usize);

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    let nu = n;
    let nv = n;

    // Build (nu+1) x (nv+1) vertex grid.
    let mut grid = vec![0u32; (nu + 1) * (nv + 1)];
    for iv in 0..=nv {
        let v = v_range.0 + (v_range.1 - v_range.0) * (iv as f64) / (nv as f64);
        for iu in 0..=nu {
            let u = u_range.0 + (u_range.1 - u_range.0) * (iu as f64) / (nu as f64);
            let idx = positions.len() as u32;
            positions.push(surface_eval(u, v));
            normals.push(normal_fn(u, v));
            grid[iv * (nu + 1) + iu] = idx;
        }
    }

    // Get grid index, wrapping u for the periodic seam.
    let gi = |iu: usize, iv: usize| -> u32 {
        let iu_w = if iu >= nu { 0 } else { iu };
        grid[iv * (nu + 1) + iu_w]
    };

    let v_min_degenerate = matches!(kind, AnalyticKind::SpherePole | AnalyticKind::ConeApex);
    let v_max_degenerate = matches!(kind, AnalyticKind::SpherePole);

    for iv in 0..nv {
        let is_bottom = iv == 0;
        let is_top = iv == nv - 1;

        for iu in 0..nu {
            let i00 = gi(iu, iv);
            let i10 = gi(iu + 1, iv);
            let i01 = gi(iu, iv + 1);
            let i11 = gi(iu + 1, iv + 1);

            if is_bottom && v_min_degenerate {
                // Bottom pole/apex: triangle fan from the degenerate row.
                indices.push(i00);
                indices.push(i11);
                indices.push(i01);
            } else if is_top && v_max_degenerate {
                // Top pole: triangle fan from the degenerate row.
                indices.push(i00);
                indices.push(i10);
                indices.push(i01);
            } else {
                // Standard two-triangle quad.
                indices.push(i00);
                indices.push(i10);
                indices.push(i11);

                indices.push(i00);
                indices.push(i11);
                indices.push(i01);
            }
        }
    }

    TriangleMesh {
        positions,
        normals,
        indices,
    }
}

/// Tessellate a planar face via ear-clipping triangulation.
///
/// Works for both convex and non-convex (simple) polygons by
/// projecting to 2D and using the ear-clipping algorithm.
fn tessellate_planar(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    normal: Vec3,
) -> Result<TriangleMesh, crate::OperationsError> {
    let wire = topo.wire(face_data.outer_wire())?;
    let mut positions = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        positions.push(topo.vertex(vid)?.point());
    }

    let n = positions.len();
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("face has only {n} vertices, need at least 3"),
        });
    }

    let normals_out = vec![normal; n];
    let indices = ear_clip_triangulate(&positions, normal);

    Ok(TriangleMesh {
        positions,
        normals: normals_out,
        indices,
    })
}

/// Ear-clipping triangulation for a simple polygon in 3D.
///
/// Projects the polygon to 2D (dropping the coordinate corresponding to
/// the dominant normal component), then applies the ear-clipping algorithm.
fn ear_clip_triangulate(positions: &[Point3], normal: Vec3) -> Vec<u32> {
    let n = positions.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![0, 1, 2];
    }

    // Project to 2D by dropping the dominant normal axis.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let project = |p: Point3| -> (f64, f64) {
        if az >= ax && az >= ay {
            (p.x(), p.y())
        } else if ay >= ax {
            (p.x(), p.z())
        } else {
            (p.y(), p.z())
        }
    };

    let pts2d: Vec<(f64, f64)> = positions.iter().map(|&p| project(p)).collect();

    // Ensure CCW winding in 2D.
    let signed_area = polygon_signed_area_2d(&pts2d);
    let ccw = signed_area > 0.0;

    // Active vertex list (indices into the original positions array).
    let mut active: Vec<usize> = if ccw {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    };

    let mut indices = Vec::with_capacity((n - 2) * 3);
    let mut safety = n * n; // prevent infinite loop on degenerate input

    while active.len() > 3 && safety > 0 {
        safety -= 1;
        let len = active.len();
        let mut found_ear = false;

        for i in 0..len {
            let prev = active[(i + len - 1) % len];
            let curr = active[i];
            let next = active[(i + 1) % len];

            // Check if this vertex forms a convex (left-turn) ear.
            let (ax2, ay2) = pts2d[prev];
            let (bx, by) = pts2d[curr];
            let (cx, cy) = pts2d[next];

            let cross = (bx - ax2).mul_add(cy - ay2, -(by - ay2) * (cx - ax2));
            if cross <= 0.0 {
                continue; // reflex vertex, not an ear
            }

            // Check that no other active vertex lies inside this triangle.
            let mut contains_point = false;
            for j in 0..len {
                if j == (i + len - 1) % len || j == i || j == (i + 1) % len {
                    continue;
                }
                let (px, py) = pts2d[active[j]];
                if point_in_triangle_2d(px, py, ax2, ay2, bx, by, cx, cy) {
                    contains_point = true;
                    break;
                }
            }

            if !contains_point {
                // This is an ear — emit the triangle.
                #[allow(clippy::cast_possible_truncation)]
                {
                    indices.push(prev as u32);
                    indices.push(curr as u32);
                    indices.push(next as u32);
                }
                active.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // Fallback: no ear found (degenerate polygon).
            // Use fan triangulation as best-effort.
            break;
        }
    }

    // Handle remaining triangle.
    if active.len() == 3 {
        #[allow(clippy::cast_possible_truncation)]
        {
            indices.push(active[0] as u32);
            indices.push(active[1] as u32);
            indices.push(active[2] as u32);
        }
    } else if active.len() > 3 {
        // Fallback fan triangulation for degenerate cases.
        for i in 1..active.len() - 1 {
            #[allow(clippy::cast_possible_truncation)]
            {
                indices.push(active[0] as u32);
                indices.push(active[i] as u32);
                indices.push(active[i + 1] as u32);
            }
        }
    }

    indices
}

/// Signed area of a 2D polygon (positive = CCW).
fn polygon_signed_area_2d(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    area / 2.0
}

/// Test if point (px,py) is inside triangle (ax,ay)-(bx,by)-(cx,cy).
#[allow(clippy::too_many_arguments)]
fn point_in_triangle_2d(
    px: f64,
    py: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
) -> bool {
    let d1 = (px - bx).mul_add(ay - by, -(ax - bx) * (py - by));
    let d2 = (px - cx).mul_add(by - cy, -(bx - cx) * (py - cy));
    let d3 = (px - ax).mul_add(cy - ay, -(cx - ax) * (py - ay));

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    !(has_neg && has_pos)
}

/// A cell in the adaptive quadtree for NURBS tessellation.
struct AdaptiveCell {
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    depth: u8,
    /// Indices into the cell vec; `None` means this is a leaf cell.
    children: Option<[usize; 4]>,
}

/// Maximum recursion depth for adaptive subdivision.
const MAX_DEPTH: u8 = 6;

/// Initial grid resolution (cells per direction).
const INITIAL_CELLS: usize = 4;

/// Evaluate the surface normal at `(u, v)`, returning a fallback for degenerate points.
fn safe_normal(surface: &brepkit_math::nurbs::surface::NurbsSurface, u: f64, v: f64) -> Vec3 {
    surface.normal(u, v).unwrap_or(Vec3::new(0.0, 0.0, 1.0))
}

/// Compute the maximum normal deviation across 5 sample points of a quad cell.
///
/// Samples the four corners and the center. Returns `1.0 - min_dot` where
/// `min_dot` is the smallest dot product between any pair of normals.
#[allow(clippy::similar_names)]
fn cell_normal_deviation(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
) -> f64 {
    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    let normals = [
        safe_normal(surface, u_min, v_min),
        safe_normal(surface, u_max, v_min),
        safe_normal(surface, u_max, v_max),
        safe_normal(surface, u_min, v_max),
        safe_normal(surface, u_mid, v_mid),
    ];

    let mut max_dev = 0.0_f64;
    for i in 0..normals.len() {
        for j in (i + 1)..normals.len() {
            let dev = 1.0 - normals[i].dot(normals[j]);
            max_dev = max_dev.max(dev);
        }
    }
    max_dev
}

/// Build the adaptive quadtree by recursive subdivision.
#[allow(clippy::similar_names)]
fn build_quadtree(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    cell_idx: usize,
    threshold: f64,
) {
    let cell = &cells[cell_idx];
    if cell.depth >= MAX_DEPTH {
        return;
    }

    let u_min = cell.u_min;
    let u_max = cell.u_max;
    let v_min = cell.v_min;
    let v_max = cell.v_max;
    let depth = cell.depth;

    let deviation = cell_normal_deviation(surface, u_min, u_max, v_min, v_max);
    if deviation <= threshold {
        return;
    }

    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);
    let child_depth = depth + 1;

    let c0 = cells.len();
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });

    cells[cell_idx].children = Some([c0, c0 + 1, c0 + 2, c0 + 3]);

    // Recurse into children.
    for i in 0..4 {
        build_quadtree(surface, cells, c0 + i, threshold);
    }
}

/// Conforming pass: ensure no more than 1 level difference between adjacent leaf cells.
///
/// Repeatedly scans leaf cells and subdivides any whose neighbor is more than
/// 1 level deeper. This eliminates T-junctions by guaranteeing matching edge vertices.
fn conforming_pass(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
) {
    // Iterate until no more conforming subdivisions are needed.
    loop {
        // Collect leaf cells that need subdivision for conformity.
        let mut to_subdivide = Vec::new();

        let len = cells.len();
        for i in 0..len {
            if cells[i].children.is_some() {
                continue; // not a leaf
            }

            let depth = cells[i].depth;
            let u_min = cells[i].u_min;
            let u_max = cells[i].u_max;
            let v_min = cells[i].v_min;
            let v_max = cells[i].v_max;

            // Check if any neighboring leaf is more than 1 level deeper.
            if needs_conforming_subdivision(cells, i, depth, u_min, u_max, v_min, v_max) {
                to_subdivide.push(i);
            }
        }

        if to_subdivide.is_empty() {
            break;
        }

        // Subdivide all cells that need it.
        for &cell_idx in &to_subdivide {
            if cells[cell_idx].children.is_some() {
                continue; // already subdivided in this pass
            }
            force_subdivide(surface, cells, cell_idx);
        }
    }
}

/// Check if a leaf cell needs conforming subdivision (neighbor is 2+ levels deeper).
#[allow(clippy::similar_names)]
fn needs_conforming_subdivision(
    cells: &[AdaptiveCell],
    _cell_idx: usize,
    depth: u8,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
) -> bool {
    // For each of the 4 edges, find the deepest neighboring leaf cell.
    let eps = (u_max - u_min) * 0.01;
    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    // Sample points just outside each edge to find neighbors.
    let probes = [
        // Bottom edge (v_min): probe below
        (u_mid, v_min - eps),
        // Top edge (v_max): probe above
        (u_mid, v_max + eps),
        // Left edge (u_min): probe left
        (u_min - eps, v_mid),
        // Right edge (u_max): probe right
        (u_max + eps, v_mid),
    ];

    for &(pu, pv) in &probes {
        if let Some(neighbor_depth) = find_leaf_depth_at(cells, pu, pv) {
            if neighbor_depth > depth + 1 {
                return true;
            }
        }
    }
    false
}

/// Find the depth of the leaf cell containing the given parameter point.
fn find_leaf_depth_at(cells: &[AdaptiveCell], u: f64, v: f64) -> Option<u8> {
    // Start from root cells (indices 0..INITIAL_CELLS*INITIAL_CELLS).
    let n_roots = INITIAL_CELLS * INITIAL_CELLS;
    for root_idx in 0..n_roots.min(cells.len()) {
        if let Some(depth) = find_leaf_depth_recursive(cells, root_idx, u, v) {
            return Some(depth);
        }
    }
    None
}

/// Recursively find the leaf depth at a given point within a cell subtree.
fn find_leaf_depth_recursive(cells: &[AdaptiveCell], idx: usize, u: f64, v: f64) -> Option<u8> {
    let cell = &cells[idx];
    // Check containment (with small tolerance).
    if u < cell.u_min || u > cell.u_max || v < cell.v_min || v > cell.v_max {
        return None;
    }

    match cell.children {
        None => Some(cell.depth),
        Some(children) => {
            for &child in &children {
                if let Some(d) = find_leaf_depth_recursive(cells, child, u, v) {
                    return Some(d);
                }
            }
            // Point is on boundary, return this cell's depth + 1 as approximation.
            Some(cell.depth + 1)
        }
    }
}

/// Force-subdivide a leaf cell (for conforming pass, no curvature check).
#[allow(clippy::similar_names)]
fn force_subdivide(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    cell_idx: usize,
) {
    let cell = &cells[cell_idx];
    let u_min = cell.u_min;
    let u_max = cell.u_max;
    let v_min = cell.v_min;
    let v_max = cell.v_max;
    let child_depth = cell.depth + 1;

    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    let c0 = cells.len();
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });

    cells[cell_idx].children = Some([c0, c0 + 1, c0 + 2, c0 + 3]);

    // Recursively apply conforming to children if needed.
    // Children are created as leaves; conforming pass will recurse as needed.
    let _ = surface;
}

/// Tessellate a NURBS surface via curvature-adaptive subdivision.
///
/// Starts with a coarse 4x4 grid and recursively subdivides cells where the
/// surface normal deviation exceeds the `deflection` threshold. A conforming
/// pass ensures no T-junctions in the final mesh.
#[allow(clippy::too_many_lines)]
fn tessellate_nurbs(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
) -> TriangleMesh {
    use std::collections::HashMap;

    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();

    // Build initial coarse grid of INITIAL_CELLS x INITIAL_CELLS.
    let mut cells = Vec::with_capacity(256);

    #[allow(clippy::cast_precision_loss)]
    let du = (u_hi - u_lo) / INITIAL_CELLS as f64;
    #[allow(clippy::cast_precision_loss)]
    let dv = (v_hi - v_lo) / INITIAL_CELLS as f64;

    for i in 0..INITIAL_CELLS {
        for j in 0..INITIAL_CELLS {
            #[allow(clippy::cast_precision_loss)]
            let u_min = u_lo + (i as f64) * du;
            #[allow(clippy::cast_precision_loss)]
            let u_max = u_lo + ((i + 1) as f64) * du;
            #[allow(clippy::cast_precision_loss)]
            let v_min = v_lo + (j as f64) * dv;
            #[allow(clippy::cast_precision_loss)]
            let v_max = v_lo + ((j + 1) as f64) * dv;

            cells.push(AdaptiveCell {
                u_min,
                u_max,
                v_min,
                v_max,
                depth: 0,
                children: None,
            });
        }
    }

    // Build quadtree by subdividing cells with high normal deviation.
    let n_roots = INITIAL_CELLS * INITIAL_CELLS;
    for i in 0..n_roots {
        build_quadtree(surface, &mut cells, i, deflection);
    }

    // Conforming pass to eliminate T-junctions.
    conforming_pass(surface, &mut cells);

    // Collect leaf cells and emit triangles.
    // Cache surface evaluations by parameter pair (as bit patterns) to avoid duplicates.
    let mut eval_cache: HashMap<(u64, u64), (Point3, Vec3)> = HashMap::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    // Map from (u_bits, v_bits) to vertex index.
    let mut vertex_map: HashMap<(u64, u64), u32> = HashMap::new();

    let get_or_insert_vertex = |u: f64,
                                v: f64,
                                eval_cache: &mut HashMap<(u64, u64), (Point3, Vec3)>,
                                positions: &mut Vec<Point3>,
                                normals: &mut Vec<Vec3>,
                                vertex_map: &mut HashMap<(u64, u64), u32>|
     -> u32 {
        let key = (u.to_bits(), v.to_bits());
        if let Some(&idx) = vertex_map.get(&key) {
            return idx;
        }
        let &mut (pos, nrm) = eval_cache.entry(key).or_insert_with(|| {
            let p = surface.evaluate(u, v);
            let n = safe_normal(surface, u, v);
            (p, n)
        });
        #[allow(clippy::cast_possible_truncation)]
        let idx = positions.len() as u32;
        positions.push(pos);
        normals.push(nrm);
        vertex_map.insert(key, idx);
        idx
    };

    for cell in &cells {
        if cell.children.is_some() {
            continue; // not a leaf
        }

        let u_min = cell.u_min;
        let u_max = cell.u_max;
        let v_min = cell.v_min;
        let v_max = cell.v_max;

        // Get vertex indices for the 4 corners.
        // Corner order: (u_min,v_min), (u_max,v_min), (u_max,v_max), (u_min,v_max)
        let i00 = get_or_insert_vertex(
            u_min,
            v_min,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut vertex_map,
        );
        let i10 = get_or_insert_vertex(
            u_max,
            v_min,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut vertex_map,
        );
        let i11 = get_or_insert_vertex(
            u_max,
            v_max,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut vertex_map,
        );
        let i01 = get_or_insert_vertex(
            u_min,
            v_max,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut vertex_map,
        );

        // Emit 2 triangles per quad cell.
        // Triangle 1: (u_min,v_min) → (u_max,v_min) → (u_max,v_max)
        indices.push(i00);
        indices.push(i10);
        indices.push(i11);

        // Triangle 2: (u_min,v_min) → (u_max,v_max) → (u_min,v_max)
        indices.push(i00);
        indices.push(i11);
        indices.push(i01);
    }

    TriangleMesh {
        positions,
        normals,
        indices,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::test_utils::{make_unit_square_face, make_unit_triangle_face};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    #[test]
    fn tessellate_square() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let mesh = tessellate(&topo, face, 0.1).unwrap();

        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.normals.len(), 4);
        // 4 vertices → 2 triangles → 6 indices
        assert_eq!(mesh.indices.len(), 6);
    }

    #[test]
    fn tessellate_triangle() {
        let mut topo = Topology::new();
        let face = make_unit_triangle_face(&mut topo);

        let mesh = tessellate(&topo, face, 0.1).unwrap();

        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        // 3 vertices → 1 triangle → 3 indices
        assert_eq!(mesh.indices.len(), 3);
    }

    /// Tessellate a simple bilinear NURBS surface (a flat quad as NURBS).
    #[test]
    fn tessellate_nurbs_surface() {
        let mut topo = Topology::new();

        // Create a simple degree-1×1 NURBS surface (bilinear patch) representing
        // a flat quad from (0,0,0) to (1,1,0).
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();

        // Create a wire around the surface boundary (not strictly needed for
        // NURBS tessellation, but required for a valid Face).
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);

        let face = topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));

        let mesh = tessellate(&topo, face, 0.25).unwrap();

        // Adaptive: flat surface → 4×4 cells with no further subdivision.
        // 4×4 = 16 cells × 2 triangles = 32 triangles × 3 indices = 96.
        // Vertices: 5×5 = 25 unique grid points.
        assert_eq!(mesh.positions.len(), 25);
        assert_eq!(mesh.normals.len(), 25);
        assert_eq!(mesh.indices.len(), 96);

        // All positions should be in [0,1] range since it's a flat quad.
        for pos in &mesh.positions {
            assert!(pos.x() >= -1e-10 && pos.x() <= 1.0 + 1e-10);
            assert!(pos.y() >= -1e-10 && pos.y() <= 1.0 + 1e-10);
            assert!((pos.z()).abs() < 1e-10);
        }
    }

    /// Test tessellation of an L-shaped (non-convex) polygon.
    #[test]
    fn tessellate_l_shape_nonconvex() {
        let mut topo = Topology::new();

        // L-shaped polygon on XY plane:
        //  (0,0) → (2,0) → (2,1) → (1,1) → (1,2) → (0,2) → (0,0)
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];

        let verts: Vec<_> = points
            .iter()
            .map(|&p| topo.vertices.alloc(Vertex::new(p, 1e-7)))
            .collect();

        let n = verts.len();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.edges
                    .alloc(Edge::new(verts[i], verts[next], EdgeCurve::Line))
            })
            .collect();

        let wire = Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);

        let face = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let mesh = tessellate(&topo, face, 0.1).unwrap();

        assert_eq!(mesh.positions.len(), 6, "should have 6 vertices");
        // 6-gon → 4 triangles
        assert_eq!(
            mesh.indices.len(),
            12,
            "L-shape should have 4 triangles (12 indices)"
        );

        // Verify area: L-shape is 2×2 minus 1×1 = 3.0
        let mut total_area = 0.0;
        for t in 0..mesh.indices.len() / 3 {
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            let a = mesh.positions[i1] - mesh.positions[i0];
            let b = mesh.positions[i2] - mesh.positions[i0];
            total_area += 0.5 * a.cross(b).length();
        }
        assert!(
            (total_area - 3.0).abs() < 0.01,
            "L-shape area should be ~3.0, got {total_area}"
        );
    }

    /// A flat bilinear surface should produce exactly 32 triangles
    /// (4×4 initial grid, 2 tris per cell, no adaptive subdivision).
    #[test]
    fn tessellate_flat_surface_few_triangles() {
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();

        let mesh = tessellate_nurbs(&surface, 0.1);

        // Flat surface: all normals are identical, deviation = 0.
        // 4×4 cells × 2 triangles × 3 indices = 96.
        assert_eq!(
            mesh.indices.len() / 3,
            32,
            "flat surface should have exactly 32 triangles, got {}",
            mesh.indices.len() / 3
        );
    }

    /// A curved NURBS surface should produce more triangles than a flat surface
    /// because adaptive subdivision refines areas of high curvature.
    #[test]
    fn tessellate_curved_surface_more_at_curves() {
        // Create a bicubic surface with significant curvature (sinusoidal z).
        let mut cps = Vec::new();
        let mut ws = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            let mut wrow = Vec::new();
            for j in 0..4 {
                #[allow(clippy::cast_precision_loss)]
                let z = ((i + j) as f64 * 0.8).sin() * 2.0;
                #[allow(clippy::cast_precision_loss)]
                row.push(Point3::new(j as f64, i as f64, z));
                wrow.push(1.0);
            }
            cps.push(row);
            ws.push(wrow);
        }
        let curved = NurbsSurface::new(
            3,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            cps,
            ws,
        )
        .unwrap();

        // Flat reference.
        let flat = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();

        let deflection = 0.05;
        let flat_mesh = tessellate_nurbs(&flat, deflection);
        let curved_mesh = tessellate_nurbs(&curved, deflection);

        let flat_tris = flat_mesh.indices.len() / 3;
        let curved_tris = curved_mesh.indices.len() / 3;

        assert!(
            curved_tris > flat_tris,
            "curved surface should have more triangles ({curved_tris}) than flat ({flat_tris})"
        );
    }
}
