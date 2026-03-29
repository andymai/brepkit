//! NURBS adaptive quadtree tessellation.

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;

use super::{TriangleMesh, TriangleMeshUV};

/// A cell in the adaptive quadtree for NURBS tessellation.
pub(super) struct AdaptiveCell {
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

/// Compute the v-parameter range for a surface by projecting boundary vertices.
///
/// `project_v` maps a 3D point to its v-parameter on the surface.
/// Falls back to (-1.0, 1.0) if the face has no usable vertices.
pub(super) fn compute_v_param_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    project_v: impl Fn(Point3) -> f64,
) -> (f64, f64) {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;

    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                for &vid in &[edge.start(), edge.end()] {
                    if let Ok(vertex) = topo.vertex(vid) {
                        let v = project_v(vertex.point());
                        v_min = v_min.min(v);
                        v_max = v_max.max(v);
                    }
                }
            }
        }
    }

    if v_min < v_max {
        (v_min, v_max)
    } else {
        (-1.0, 1.0) // fallback
    }
}

/// Compute the v-range (axial extent) for an analytic surface from its face
/// wire boundary vertices.
///
/// Projects all wire vertices onto the surface axis and returns (v_min, v_max).
/// Falls back to (-1.0, 1.0) if the face has no usable vertices.
pub(super) fn compute_axial_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    origin: Point3,
    axis: Vec3,
) -> (f64, f64) {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;

    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                for &vid in &[edge.start(), edge.end()] {
                    if let Ok(vertex) = topo.vertex(vid) {
                        let pt = vertex.point();
                        let to_pt = Vec3::new(
                            pt.x() - origin.x(),
                            pt.y() - origin.y(),
                            pt.z() - origin.z(),
                        );
                        let v = axis.dot(to_pt);
                        v_min = v_min.min(v);
                        v_max = v_max.max(v);
                    }
                }
            }
        }
    }

    if v_min < v_max {
        (v_min, v_max)
    } else {
        (-1.0, 1.0) // fallback
    }
}

/// Compute the angular (u) range for an analytic face from its wire boundary.
///
/// Projects boundary edge vertices -- and midpoints of curved edges -- onto
/// the surface and collects their u-parameters. If the face doesn't span
/// the full revolution, returns the tighter `[u_min, u_max]` range.
/// Returns `(0, 2*pi)` for full-circle faces or when fewer than 3 boundary
/// vertices exist.
pub(super) fn compute_angular_range<F>(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    project: F,
) -> (f64, f64)
where
    F: Fn(Point3) -> (f64, f64),
{
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::TAU;

    let mut angles: Vec<f64> = Vec::new();

    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                for &vid in &[edge.start(), edge.end()] {
                    if let Ok(vertex) = topo.vertex(vid) {
                        let (u, _v) = project(vertex.point());
                        angles.push(u);
                    }
                }

                // Sample edge midpoints to provide angular coverage
                // between vertices.
                if !edge.is_closed() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        match edge.curve() {
                            EdgeCurve::Circle(circle) => {
                                let ts = circle.project(sv.point());
                                let te = circle.project(ev.point());
                                let fwd = (te - ts).rem_euclid(TAU);
                                let mid_t = if fwd <= std::f64::consts::PI {
                                    ts + fwd * 0.5
                                } else {
                                    ts - (TAU - fwd) * 0.5
                                };
                                let mid = circle.evaluate(mid_t);
                                let (u, _) = project(mid);
                                angles.push(u);
                            }
                            EdgeCurve::Ellipse(ellipse) => {
                                let ts = ellipse.project(sv.point());
                                let te = ellipse.project(ev.point());
                                let fwd = (te - ts).rem_euclid(TAU);
                                let mid_t = if fwd <= std::f64::consts::PI {
                                    ts + fwd * 0.5
                                } else {
                                    ts - (TAU - fwd) * 0.5
                                };
                                let mid = ellipse.evaluate(mid_t);
                                let (u, _) = project(mid);
                                angles.push(u);
                            }
                            EdgeCurve::NurbsCurve(nurbs) => {
                                let (t0, t1) = nurbs.domain();
                                let mid = nurbs.evaluate(f64::midpoint(t0, t1));
                                let (u, _) = project(mid);
                                angles.push(u);
                            }
                            EdgeCurve::Line => {}
                        }
                    }
                }
            }
        }
    }

    if angles.len() < 3 {
        return (0.0, TAU);
    }

    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    angles.dedup_by(|a, b| (*a - *b).abs() < brepkit_math::tolerance::Tolerance::default().linear);

    if angles.len() < 3 {
        return (0.0, TAU);
    }

    let mut max_gap = 0.0_f64;
    let mut gap_end_idx = 0_usize;
    for i in 0..angles.len() {
        let j = (i + 1) % angles.len();
        let gap = if j > i {
            angles[j] - angles[i]
        } else {
            angles[j] + TAU - angles[i]
        };
        if gap > max_gap {
            max_gap = gap;
            gap_end_idx = j;
        }
    }

    let n_angles = angles.len() as f64;
    let even_gap = TAU / n_angles;
    let gap_threshold = (2.5 * even_gap).min(TAU / 3.0);
    if max_gap < gap_threshold {
        return (0.0, TAU);
    }

    let u_start = angles[gap_end_idx];
    let gap_start_idx = if gap_end_idx == 0 {
        angles.len() - 1
    } else {
        gap_end_idx - 1
    };
    let u_end = angles[gap_start_idx];

    if u_end > u_start {
        (u_start, u_end)
    } else {
        (u_start, u_end + TAU)
    }
}

/// Compute the latitude (v) range for a sphere face from its wire boundary.
#[must_use]
pub fn compute_sphere_v_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> (f64, f64) {
    use std::f64::consts::FRAC_PI_2;

    let mut wire_pts = Vec::new();
    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                if let Ok(vertex) = topo.vertex(edge.start()) {
                    wire_pts.push(vertex.point());
                }
            }
        }
    }

    if wire_pts.len() < 3 {
        return (-FRAC_PI_2, FRAC_PI_2);
    }

    let avg_v: f64 = wire_pts
        .iter()
        .map(|pt| sphere.project_point(*pt).1)
        .sum::<f64>()
        / wire_pts.len() as f64;

    let signed_area = projected_signed_area(&wire_pts);
    if signed_area > 0.0 {
        (avg_v, FRAC_PI_2)
    } else {
        (-FRAC_PI_2, avg_v)
    }
}

/// Signed area of a polygon projected onto the XY plane.
/// Positive = CCW winding from +Z, negative = CW.
#[must_use]
pub fn projected_signed_area(pts: &[Point3]) -> f64 {
    let n = pts.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].x() * pts[j].y() - pts[j].x() * pts[i].y();
    }
    area * 0.5
}

/// Determine the [`AnalyticKind`] for sphere tessellation based on v-range.
pub(super) fn sphere_analytic_kind(v_range: (f64, f64)) -> super::AnalyticKind {
    use super::AnalyticKind;
    use std::f64::consts::FRAC_PI_2;
    let eps = 1e-6;
    let has_south_pole = (v_range.0 + FRAC_PI_2).abs() < eps;
    let has_north_pole = (v_range.1 - FRAC_PI_2).abs() < eps;
    match (has_south_pole, has_north_pole) {
        (true, true) => AnalyticKind::SpherePole,
        (true, false) => AnalyticKind::ConeApex,
        (false, true) => AnalyticKind::VMaxPole,
        (false, false) => AnalyticKind::General,
    }
}

/// Evaluate the surface normal at `(u, v)`, returning a fallback for degenerate points.
fn safe_normal(surface: &brepkit_math::nurbs::surface::NurbsSurface, u: f64, v: f64) -> Vec3 {
    surface.normal(u, v).unwrap_or(Vec3::new(0.0, 0.0, 1.0))
}

/// Compute the refinement error for a quad cell using combined metrics.
#[allow(clippy::similar_names)]
fn cell_refinement_error(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
) -> f64 {
    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    let p00 = surface.evaluate(u_min, v_min);
    let p10 = surface.evaluate(u_max, v_min);
    let p11 = surface.evaluate(u_max, v_max);
    let p01 = surface.evaluate(u_min, v_max);
    let p_mid = surface.evaluate(u_mid, v_mid);

    let bilinear_mid = Point3::new(
        0.25 * (p00.x() + p10.x() + p11.x() + p01.x()),
        0.25 * (p00.y() + p10.y() + p11.y() + p01.y()),
        0.25 * (p00.z() + p10.z() + p11.z() + p01.z()),
    );
    let sag = (p_mid - bilinear_mid).length();

    let normals = [
        safe_normal(surface, u_min, v_min),
        safe_normal(surface, u_max, v_min),
        safe_normal(surface, u_max, v_max),
        safe_normal(surface, u_min, v_max),
        safe_normal(surface, u_mid, v_mid),
    ];

    let mut max_normal_dev = 0.0_f64;
    for i in 0..normals.len() {
        for j in (i + 1)..normals.len() {
            let dev = 1.0 - normals[i].dot(normals[j]);
            max_normal_dev = max_normal_dev.max(dev);
        }
    }

    let edge_mids = [
        surface.evaluate(u_mid, v_min),
        surface.evaluate(u_mid, v_max),
        surface.evaluate(u_min, v_mid),
        surface.evaluate(u_max, v_mid),
    ];

    let edge_linear_mids = [
        lerp_point(p00, p10),
        lerp_point(p01, p11),
        lerp_point(p00, p01),
        lerp_point(p10, p11),
    ];

    let mut max_edge_sag = 0.0_f64;
    for i in 0..4 {
        let edge_sag = (edge_mids[i] - edge_linear_mids[i]).length();
        max_edge_sag = max_edge_sag.max(edge_sag);
    }

    let diag = (p11 - p00).length().max((p10 - p01).length());
    let normal_sag = max_normal_dev * diag * 0.5;

    sag.max(max_edge_sag).max(normal_sag)
}

/// Linear interpolation (midpoint) of two points.
fn lerp_point(a: Point3, b: Point3) -> Point3 {
    Point3::new(
        0.5 * (a.x() + b.x()),
        0.5 * (a.y() + b.y()),
        0.5 * (a.z() + b.z()),
    )
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

    let error = cell_refinement_error(surface, u_min, u_max, v_min, v_max);
    if error <= threshold {
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

    for i in 0..4 {
        build_quadtree(surface, cells, c0 + i, threshold);
    }
}

/// Conforming pass: ensure no more than 1 level difference between adjacent leaf cells.
fn conforming_pass(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
) {
    for _pass in 0..MAX_DEPTH {
        let mut to_subdivide = Vec::new();

        let len = cells.len();
        for i in 0..len {
            if cells[i].children.is_some() {
                continue;
            }

            let depth = cells[i].depth;
            let u_min = cells[i].u_min;
            let u_max = cells[i].u_max;
            let v_min = cells[i].v_min;
            let v_max = cells[i].v_max;

            if needs_conforming_subdivision(cells, i, depth, u_min, u_max, v_min, v_max) {
                to_subdivide.push(i);
            }
        }

        if to_subdivide.is_empty() {
            break;
        }

        for &cell_idx in &to_subdivide {
            if cells[cell_idx].children.is_some() {
                continue;
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
    let eps = (u_max - u_min) * 0.01;
    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    let probes = [
        (u_mid, v_min - eps),
        (u_mid, v_max + eps),
        (u_min - eps, v_mid),
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
    if cell.depth >= MAX_DEPTH + 2 {
        return;
    }
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

    let _ = surface;
}

/// Tessellate a NURBS surface via curvature-adaptive subdivision.
#[allow(clippy::too_many_lines)]
pub(super) fn tessellate_nurbs(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
) -> TriangleMeshUV {
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();

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

    let n_roots = INITIAL_CELLS * INITIAL_CELLS;
    for i in 0..n_roots {
        build_quadtree(surface, &mut cells, i, deflection);
    }

    conforming_pass(surface, &mut cells);

    let leaf_count = cells.iter().filter(|c| c.children.is_none()).count();
    let mut eval_cache: HashMap<(u64, u64), (Point3, Vec3)> = HashMap::new();
    let mut positions = Vec::with_capacity(leaf_count * 4);
    let mut normals = Vec::with_capacity(leaf_count * 4);
    let mut uvs: Vec<[f64; 2]> = Vec::with_capacity(leaf_count * 4);
    let mut indices = Vec::with_capacity(leaf_count * 6);
    let mut vertex_map: HashMap<(u64, u64), u32> = HashMap::new();

    let get_or_insert_vertex = |u: f64,
                                v: f64,
                                eval_cache: &mut HashMap<(u64, u64), (Point3, Vec3)>,
                                positions: &mut Vec<Point3>,
                                normals: &mut Vec<Vec3>,
                                uvs: &mut Vec<[f64; 2]>,
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
        uvs.push([u, v]);
        vertex_map.insert(key, idx);
        idx
    };

    for cell in &cells {
        if cell.children.is_some() {
            continue;
        }

        let i00 = get_or_insert_vertex(
            cell.u_min,
            cell.v_min,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );
        let i10 = get_or_insert_vertex(
            cell.u_max,
            cell.v_min,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );
        let i11 = get_or_insert_vertex(
            cell.u_max,
            cell.v_max,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );
        let i01 = get_or_insert_vertex(
            cell.u_min,
            cell.v_max,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );

        indices.push(i00);
        indices.push(i10);
        indices.push(i11);

        indices.push(i00);
        indices.push(i11);
        indices.push(i01);
    }

    TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals,
            indices,
        },
        uvs,
    }
}
