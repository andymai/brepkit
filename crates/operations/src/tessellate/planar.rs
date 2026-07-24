//! Planar and simple analytic face tessellation.

use brepkit_math::det_hash::{DetHashMap, DetHashSet};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;

use super::edge_sampling::{
    measure_max_chord_deviation, sample_wire_positions, segments_for_chord_deviation_a,
};
use super::shorter_arc_range;
use super::{AnalyticKind, MERGE_GRID, TriangleMesh, TriangleMeshUV, point_merge_key};

/// Tessellate a cylindrical face using its actual boundary polygon (CDT-based).
///
/// Used for faces with non-rectangular boundaries (e.g., boolean sub-faces
/// bounded by intersection curves). Projects boundary to UV, CDT-triangulates,
/// then evaluates each vertex on the cylinder surface.
// TODO: Handle inner wires (holes) -- currently only tessellates the outer wire.
// The "outside" sub-face from `split_face_with_internal_loops` has holes that
// are ignored here. This is OK for now because the outside sub-face is discarded
// by classification in the Steinmetz case, but will need fixing for correct
// rendering of boolean results with internal loops.
pub(super) fn tessellate_analytic_with_boundary(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    _deflection: f64,
    _angular_tol: f64,
) -> Result<TriangleMeshUV, crate::OperationsError> {
    // NOTE: Do NOT handle is_reversed here -- `tessellate_with_uvs` applies
    // a common reversal pass for all face types after this function returns.
    let wire = topo.wire(face_data.outer_wire())?;

    let mut uv_pts: Vec<(f64, f64)> = Vec::new();
    let mut positions_3d: Vec<Point3> = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        let pos = topo.vertex(vid)?.point();
        let (mut u, v) = cyl.project_point(pos);

        // Unwrap u to be continuous with the previous sample.
        if let Some(&(prev_u, _)) = uv_pts.last() {
            let diff = u - prev_u;
            let shifts = (diff / std::f64::consts::TAU + 0.5).floor();
            u -= shifts * std::f64::consts::TAU;
        }

        uv_pts.push((u, v));
        positions_3d.push(pos);
    }

    if uv_pts.len() < 3 {
        return Ok(TriangleMeshUV::default());
    }

    // Ensure CCW winding in UV (positive signed area).
    // For non-reversed cylinder faces, CCW UV -> outward normal.
    // If the polygon is CW (negative area), reverse to get CCW.
    {
        let mut signed_area = 0.0;
        for i in 0..uv_pts.len() {
            let j = (i + 1) % uv_pts.len();
            signed_area += uv_pts[i].0 * uv_pts[j].1 - uv_pts[j].0 * uv_pts[i].1;
        }
        if signed_area < 0.0 {
            uv_pts.reverse();
            positions_3d.reverse();
        }
    }

    let uv_p2: Vec<brepkit_math::vec::Point2> = uv_pts
        .iter()
        .map(|&(u, v)| brepkit_math::vec::Point2::new(u, v))
        .collect();
    let u_min = uv_pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min) - 1.0;
    let u_max = uv_pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let v_min = uv_pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min) - 1.0;
    let v_max = uv_pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let mut cdt = brepkit_math::cdt::Cdt::with_capacity(
        (
            brepkit_math::vec::Point2::new(u_min, v_min),
            brepkit_math::vec::Point2::new(u_max, v_max),
        ),
        uv_p2.len() + 4,
    );

    let n_verts = uv_p2.len();
    let cdt_ids: Vec<usize> = match cdt.insert_points_hilbert(&uv_p2) {
        Ok(ids) => ids,
        Err(_) => return Ok(TriangleMeshUV::default()),
    };

    // Insert boundary constraints -- only record successfully inserted ones
    // so remove_exterior doesn't rely on non-existent barriers.
    let mut boundary_segs = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let j = (i + 1) % n_verts;
        if cdt.insert_constraint(cdt_ids[i], cdt_ids[j]).is_ok() {
            boundary_segs.push((cdt_ids[i], cdt_ids[j]));
        }
    }

    cdt.remove_exterior(&boundary_segs);

    let tris = cdt.triangles();
    let cdt_verts = cdt.vertices();

    let mut positions = Vec::with_capacity(cdt_verts.len());
    let mut normals_out = Vec::with_capacity(cdt_verts.len());
    let mut uvs = Vec::with_capacity(cdt_verts.len());
    for pt in cdt_verts {
        let u = pt.x();
        let v = pt.y();
        positions.push(cyl.evaluate(u, v));
        let nm = cyl.normal(u, 0.0);
        normals_out.push(nm);
        uvs.push([u, v]);
    }

    // Override boundary vertices with exact 3D positions from topology.
    for (i, &cdt_id) in cdt_ids.iter().enumerate() {
        if cdt_id < positions.len() {
            positions[cdt_id] = positions_3d[i];
        }
    }

    // Build index buffer (standard CCW winding -- reversal handled by caller).
    let mut indices = Vec::with_capacity(tris.len() * 3);
    #[allow(clippy::cast_possible_truncation)]
    for &(a, b, c) in &tris {
        indices.push(a as u32);
        indices.push(b as u32);
        indices.push(c as u32);
    }

    Ok(TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals: normals_out,
            indices,
        },
        uvs,
    })
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
pub(super) fn tessellate_analytic(
    surface_eval: impl Fn(f64, f64) -> Point3,
    normal_fn: impl Fn(f64, f64) -> Vec3,
    u_range: (f64, f64),
    v_range: (f64, f64),
    nu: usize,
    nv: usize,
    kind: AnalyticKind,
) -> TriangleMeshUV {
    let nu = nu.max(4);
    let nv = nv.max(1);

    let num_verts = (nu + 1) * (nv + 1);
    let num_indices = nu * nv * 6;
    let mut positions = Vec::with_capacity(num_verts);
    let mut normals = Vec::with_capacity(num_verts);
    let mut uvs = Vec::with_capacity(num_verts);
    let mut indices = Vec::with_capacity(num_indices);

    // Only wrap u when the range spans a full period (approx 2pi).
    // For partial arcs (e.g. quarter-cylinder), the last grid column is a
    // distinct point that must NOT wrap back to the first column.
    let u_periodic = (u_range.1 - u_range.0 - std::f64::consts::TAU).abs()
        < brepkit_math::tolerance::Tolerance::new().linear;

    let mut grid = vec![0u32; (nu + 1) * (nv + 1)];
    for iv in 0..=nv {
        let v = v_range.0 + (v_range.1 - v_range.0) * (iv as f64) / (nv as f64);
        for iu in 0..=nu {
            let u = u_range.0 + (u_range.1 - u_range.0) * (iu as f64) / (nu as f64);
            let idx = positions.len() as u32;
            positions.push(surface_eval(u, v));
            normals.push(normal_fn(u, v));
            uvs.push([u, v]);
            grid[iv * (nu + 1) + iu] = idx;
        }
    }

    // Get grid index, wrapping u only for periodic seams (full circles).
    let gi = |iu: usize, iv: usize| -> u32 {
        let iu_w = if u_periodic && iu >= nu { 0 } else { iu };
        grid[iv * (nu + 1) + iu_w]
    };

    let v_min_degenerate = matches!(kind, AnalyticKind::SpherePole | AnalyticKind::ConeApex);
    let v_max_degenerate = matches!(kind, AnalyticKind::SpherePole | AnalyticKind::VMaxPole);

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

    TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals,
            indices,
        },
        uvs,
    }
}

/// Tessellate a planar face using CDT (Constrained Delaunay Triangulation).
///
/// Works for both convex and non-convex (simple) polygons by
/// projecting to 2D and using CDT with fan-triangulation fallback for degenerate cases.
pub(super) fn tessellate_planar(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    normal: Vec3,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    let wire = topo.wire(face_data.outer_wire())?;
    let mut positions = Vec::new();
    let tol = 1e-10;

    // Sample a parametric curve into `positions`, skipping consecutive duplicates.
    // `t_for_index(i)` maps a sample index to a parameter value.
    // Iterates forward when `forward` is true, reversed otherwise.
    let sample_curve = |evaluate_fn: &dyn Fn(f64) -> Point3,
                        t_for_index: &dyn Fn(usize) -> f64,
                        n_samples: usize,
                        forward: bool,
                        positions: &mut Vec<Point3>| {
        let indices: Box<dyn Iterator<Item = usize>> = if forward {
            Box::new(0..n_samples)
        } else {
            Box::new((0..n_samples).rev())
        };
        for i in indices {
            #[allow(clippy::cast_precision_loss)]
            let t = t_for_index(i);
            let pt = evaluate_fn(t);
            if positions
                .last()
                .is_none_or(|p: &Point3| (*p - pt).length() > tol)
            {
                positions.push(pt);
            }
        }
    };

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        match edge.curve() {
            EdgeCurve::Circle(circle) => {
                let (t_start, t_end) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    shorter_arc_range(circle, topo, edge)?
                };
                let arc_range = (t_end - t_start).abs();
                let n_samples = segments_for_chord_deviation_a(
                    circle.radius(),
                    arc_range,
                    deflection,
                    angular_tol,
                    false,
                );
                #[allow(clippy::cast_precision_loss)]
                sample_curve(
                    &|t| circle.evaluate(t),
                    &|i| t_start + (t_end - t_start) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
                    &mut positions,
                );
            }
            EdgeCurve::Ellipse(ellipse) => {
                let (t_start, t_end) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    let ts = ellipse.project(sp);
                    let mut te = ellipse.project(ep);
                    if te <= ts {
                        te += std::f64::consts::TAU;
                    }
                    (ts, te)
                };
                let arc_range = t_end - t_start;
                // Largest radius of curvature (a^2/b) governs uniform-parameter
                // sampling density; matches the wall edge sampling so the cap
                // boundary stays watertight against the side faces.
                let max_curv_radius =
                    ellipse.semi_major() * ellipse.semi_major() / ellipse.semi_minor();
                let n_samples = segments_for_chord_deviation_a(
                    max_curv_radius,
                    arc_range,
                    deflection,
                    angular_tol,
                    true,
                );
                #[allow(clippy::cast_precision_loss)]
                sample_curve(
                    &|t| ellipse.evaluate(t),
                    &|i| t_start + (t_end - t_start) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
                    &mut positions,
                );
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                let (u0, u1) = nurbs.domain();
                let n_spans = nurbs
                    .control_points()
                    .len()
                    .saturating_sub(nurbs.degree())
                    .max(1);
                let coarse_n = (n_spans * 4).clamp(8, 128);
                let max_dev = measure_max_chord_deviation(nurbs, u0, u1, coarse_n);
                #[allow(clippy::cast_sign_loss)]
                let n_samples = if max_dev <= deflection {
                    coarse_n
                } else {
                    ((coarse_n as f64) * (max_dev / deflection).sqrt()).ceil() as usize
                }
                .clamp(8, 4096);
                let forward = oe.is_forward()
                    != super::edge_sampling::nurbs_runs_end_to_start(topo, edge, nurbs)?;
                #[allow(clippy::cast_precision_loss)]
                sample_curve(
                    &|t| nurbs.evaluate(t),
                    &|i| u0 + (u1 - u0) * (i as f64) / (n_samples as f64),
                    n_samples,
                    forward,
                    &mut positions,
                );
            }
            EdgeCurve::Line => {
                let vid = if oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                let pt = topo.vertex(vid)?.point();
                if positions
                    .last()
                    .is_none_or(|p: &Point3| (*p - pt).length() > tol)
                {
                    positions.push(pt);
                }
            }
        }
    }

    // Remove last point if it duplicates the first (closed wire).
    if positions.len() > 2
        && let (Some(first), Some(last)) = (positions.first(), positions.last())
        && (*last - *first).length() < tol
    {
        positions.pop();
    }

    let n = positions.len();
    if n < 3 {
        // Degenerate face (e.g. sliver from boolean) -- return empty mesh
        // rather than failing the entire solid tessellation.
        return Ok(TriangleMesh::default());
    }

    if face_data.inner_wires().is_empty() {
        let normals_out = vec![normal; n];
        let mut indices = cdt_triangulate_simple(&positions, normal);

        // Ensure triangle winding matches the face normal.
        // cdt_triangulate_simple forces CCW in 2D projection, which may
        // disagree with the face normal for faces whose normal opposes
        // the projection direction.
        if indices.len() >= 3 {
            let i0 = indices[0] as usize;
            let i1 = indices[1] as usize;
            let i2 = indices[2] as usize;
            let a = positions[i1] - positions[i0];
            let b = positions[i2] - positions[i0];
            let tri_normal = a.cross(b);
            if tri_normal.dot(normal) < 0.0 {
                for t in 0..indices.len() / 3 {
                    indices.swap(t * 3 + 1, t * 3 + 2);
                }
            }
        }

        Ok(TriangleMesh {
            positions,
            normals: normals_out,
            indices,
        })
    } else {
        tessellate_planar_with_holes(topo, face_data, &positions, normal, deflection, angular_tol)
    }
}

/// Project a 3D point to 2D by dropping the dominant normal axis.
pub(super) fn project_by_normal(p: Point3, normal: Vec3) -> brepkit_math::vec::Point2 {
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();
    if az >= ax && az >= ay {
        brepkit_math::vec::Point2::new(p.x(), p.y())
    } else if ay >= ax {
        brepkit_math::vec::Point2::new(p.x(), p.z())
    } else {
        brepkit_math::vec::Point2::new(p.y(), p.z())
    }
}

/// Compute an axis-aligned bounding box with margin for a set of 2D points.
pub(super) fn compute_cdt_bounds(
    pts2d: &[brepkit_math::vec::Point2],
) -> (brepkit_math::vec::Point2, brepkit_math::vec::Point2) {
    use brepkit_math::vec::Point2;

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for &p in pts2d {
        min_x = min_x.min(p.x());
        min_y = min_y.min(p.y());
        max_x = max_x.max(p.x());
        max_y = max_y.max(p.y());
    }
    let margin = ((max_x - min_x).max(max_y - min_y)) * 0.1 + 1e-6;
    (
        Point2::new(min_x - margin, min_y - margin),
        Point2::new(max_x + margin, max_y + margin),
    )
}

/// Tessellate a planar face with inner wires (holes) using CDT.
#[allow(clippy::too_many_lines)]
fn tessellate_planar_with_holes(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    outer_positions: &[Point3],
    normal: Vec3,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;

    let mut all_positions: Vec<Point3> = outer_positions.to_vec();
    let outer_count = all_positions.len();
    let mut inner_wire_ranges: Vec<(usize, usize)> = Vec::new();

    let tol = 1e-10;
    for &iw_id in face_data.inner_wires() {
        let iw = topo.wire(iw_id)?;
        let inner_pts = sample_wire_positions(topo, iw, tol, deflection, angular_tol)?;
        let start = all_positions.len();
        all_positions.extend_from_slice(&inner_pts);
        let end = all_positions.len();
        inner_wire_ranges.push((start, end));
    }

    let pts2d: Vec<Point2> = all_positions
        .iter()
        .map(|&p| project_by_normal(p, normal))
        .collect();
    let bounds = compute_cdt_bounds(&pts2d);

    let mut cdt = Cdt::with_capacity(bounds, pts2d.len());

    // Insert all points (Hilbert-ordered for O(1) amortized locate).
    let cdt_indices = cdt
        .insert_points_hilbert(&pts2d)
        .map_err(crate::OperationsError::Math)?;

    let mut all_constraints: Vec<(usize, usize)> = Vec::new();
    for i in 0..outer_count {
        let j = (i + 1) % outer_count;
        let ci = cdt_indices[i];
        let cj = cdt_indices[j];
        if ci != cj {
            cdt.insert_constraint(ci, cj)
                .map_err(crate::OperationsError::Math)?;
            all_constraints.push((ci, cj));
        }
    }

    for &(start, end) in &inner_wire_ranges {
        let count = end - start;
        for i in 0..count {
            let j = (i + 1) % count;
            let ci = cdt_indices[start + i];
            let cj = cdt_indices[start + j];
            if ci != cj {
                cdt.insert_constraint(ci, cj)
                    .map_err(crate::OperationsError::Math)?;
                all_constraints.push((ci, cj));
            }
        }
    }

    let outer_constraints: Vec<(usize, usize)> = (0..outer_count)
        .filter_map(|i| {
            let j = (i + 1) % outer_count;
            let ci = cdt_indices[i];
            let cj = cdt_indices[j];
            (ci != cj).then_some((ci, cj))
        })
        .collect();
    cdt.remove_exterior(&outer_constraints);

    // Constraint edges drive the flood-fill stop condition below.
    let constraint_set: DetHashSet<(usize, usize)> = all_constraints
        .iter()
        .flat_map(|&(a, b)| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            [(lo, hi), (hi, lo)]
        })
        .collect();

    for seed in hole_removal_seeds(&pts2d, &inner_wire_ranges) {
        let _removed = cdt.flood_remove_from_point(seed, &constraint_set);
    }

    let cdt_triangles = cdt.triangles();
    let cdt_verts = cdt.vertices();
    let num_tris = cdt_triangles.len();
    let mut positions_out = Vec::with_capacity(cdt_verts.len());
    let mut normals_out = Vec::with_capacity(cdt_verts.len());
    let mut indices_out = Vec::with_capacity(num_tris * 3);

    // Build O(1) reverse map: CDT vertex index -> original position index.
    let mut vi_to_orig: DetHashMap<usize, usize> = DetHashMap::default();
    for (orig_idx, &cdt_vi) in cdt_indices.iter().enumerate() {
        vi_to_orig.entry(cdt_vi).or_insert(orig_idx);
    }

    let mut cdt_to_mesh: DetHashMap<usize, u32> = DetHashMap::default();
    for &(v0, v1, v2) in &cdt_triangles {
        for &vi in &[v0, v1, v2] {
            if let std::collections::hash_map::Entry::Vacant(e) = cdt_to_mesh.entry(vi) {
                #[allow(clippy::cast_possible_truncation)]
                let mesh_idx = positions_out.len() as u32;
                if let Some(&orig_idx) = vi_to_orig.get(&vi) {
                    positions_out.push(all_positions[orig_idx]);
                } else {
                    // Steiner point inserted by CDT -- reconstruct 3D from 2D.
                    let p2d = cdt_verts[vi];
                    let p3d = unproject_point(p2d, normal, &all_positions[0]);
                    positions_out.push(p3d);
                }
                normals_out.push(normal);
                e.insert(mesh_idx);
            }
        }
    }

    for &(v0, v1, v2) in &cdt_triangles {
        let i0 = cdt_to_mesh[&v0];
        let i1 = cdt_to_mesh[&v1];
        let i2 = cdt_to_mesh[&v2];
        indices_out.push(i0);
        indices_out.push(i1);
        indices_out.push(i2);
    }

    // Ensure winding matches face normal.
    if indices_out.len() >= 3 {
        let i0 = indices_out[0] as usize;
        let i1 = indices_out[1] as usize;
        let i2 = indices_out[2] as usize;
        let a = positions_out[i1] - positions_out[i0];
        let b = positions_out[i2] - positions_out[i0];
        let tri_normal = a.cross(b);
        if tri_normal.dot(normal) < 0.0 {
            for t in 0..indices_out.len() / 3 {
                indices_out.swap(t * 3 + 1, t * 3 + 2);
            }
        }
    }

    Ok(TriangleMesh {
        positions: positions_out,
        normals: normals_out,
        indices: indices_out,
    })
}

/// Find a point guaranteed to be inside a simple polygon in 2D.
fn find_interior_seed(polygon: &[brepkit_math::vec::Point2]) -> brepkit_math::vec::Point2 {
    use brepkit_math::predicates::point_in_polygon;
    use brepkit_math::vec::Point2;

    let n = polygon.len();
    if n == 0 {
        return Point2::new(0.0, 0.0);
    }
    if n < 3 {
        return polygon[0];
    }

    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        let e_prev = Point2::new(prev.x() - curr.x(), prev.y() - curr.y());
        let e_next = Point2::new(next.x() - curr.x(), next.y() - curr.y());

        let len_prev = (e_prev.x() * e_prev.x() + e_prev.y() * e_prev.y()).sqrt();
        let len_next = (e_next.x() * e_next.x() + e_next.y() * e_next.y()).sqrt();
        if len_prev < 1e-30 || len_next < 1e-30 {
            continue;
        }

        let u_prev = Point2::new(e_prev.x() / len_prev, e_prev.y() / len_prev);
        let u_next = Point2::new(e_next.x() / len_next, e_next.y() / len_next);
        let bisector = Point2::new(u_prev.x() + u_next.x(), u_prev.y() + u_next.y());
        let bis_len = (bisector.x() * bisector.x() + bisector.y() * bisector.y()).sqrt();
        if bis_len < 1e-30 {
            continue;
        }

        let step = 1e-4 * len_prev.min(len_next);
        let candidate = Point2::new(
            curr.x() + step * bisector.x() / bis_len,
            curr.y() + step * bisector.y() / bis_len,
        );

        if point_in_polygon(candidate, polygon) {
            return candidate;
        }

        let candidate_flip = Point2::new(
            curr.x() - step * bisector.x() / bis_len,
            curr.y() - step * bisector.y() / bis_len,
        );

        if point_in_polygon(candidate_flip, polygon) {
            return candidate_flip;
        }
    }

    let mut cx = 0.0;
    let mut cy = 0.0;
    for p in polygon {
        cx += p.x();
        cy += p.y();
    }
    cx /= n as f64;
    cy /= n as f64;
    Point2::new(cx, cy)
}

/// Seeds for the cells that must be flood-removed from a CDT of a face with
/// inner wires, one per wire that actually bounds a hole.
///
/// Inner wires can nest: an O-shaped bin's top face carries the cavity opening,
/// the island band around the central hole, and that hole itself. Nesting
/// alternates material and void, so only odd-depth wires bound a hole — taking
/// every inner wire as a hole erases the islands. Stored winding cannot be used
/// to tell them apart; a boolean can emit a hole wound like its outer.
///
/// Each seed sits just inside its own wire (never at its centroid, which
/// concentric wires share) so the flood starts in that wire's own cell.
pub(super) fn hole_removal_seeds(
    pts2d: &[brepkit_math::vec::Point2],
    inner_wire_ranges: &[(usize, usize)],
) -> Vec<brepkit_math::vec::Point2> {
    use brepkit_math::predicates::point_in_polygon;
    use brepkit_math::vec::Point2;

    let polys: Vec<Vec<Point2>> = inner_wire_ranges
        .iter()
        .map(|&(start, end)| (start..end).map(|i| pts2d[i]).collect())
        .collect();
    let seeds: Vec<Option<Point2>> = polys
        .iter()
        .map(|p| (p.len() >= 3).then(|| find_interior_seed(p)))
        .collect();

    // One wire cannot nest, so skip the containment scan entirely — the common
    // case, and it keeps this off the hot path for ordinary single-hole faces.
    if polys.len() < 2 {
        return seeds.into_iter().flatten().collect();
    }

    // Bounds gate the winding-number tests: honeycomb faces carry dozens of
    // disjoint holes whose boxes never overlap, so the scan stays near-linear.
    let boxes: Vec<Option<(f64, f64, f64, f64)>> = polys
        .iter()
        .map(|p| {
            (p.len() >= 3).then(|| {
                p.iter().fold(
                    (f64::MAX, f64::MAX, f64::MIN, f64::MIN),
                    |(x0, y0, x1, y1), q| {
                        (x0.min(q.x()), y0.min(q.y()), x1.max(q.x()), y1.max(q.y()))
                    },
                )
            })
        })
        .collect();

    let mut out = Vec::new();
    for (i, seed) in seeds.iter().enumerate() {
        let Some(seed) = *seed else { continue };
        let mut depth = 1;
        for (j, poly) in polys.iter().enumerate() {
            if j == i {
                continue;
            }
            let Some((x0, y0, x1, y1)) = boxes[j] else {
                continue;
            };
            if seed.x() < x0 || seed.x() > x1 || seed.y() < y0 || seed.y() > y1 {
                continue;
            }
            if point_in_polygon(seed, poly) {
                depth += 1;
            }
        }
        if depth % 2 == 1 {
            out.push(seed);
        }
    }
    out
}

/// Reconstruct a 3D point from a 2D projection, using the face plane.
pub(super) fn unproject_point(
    p2d: brepkit_math::vec::Point2,
    normal: Vec3,
    reference: &Point3,
) -> Point3 {
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let d = normal.x() * reference.x() + normal.y() * reference.y() + normal.z() * reference.z();
    if az >= ax && az >= ay {
        let z = (d - normal.x() * p2d.x() - normal.y() * p2d.y()) / normal.z();
        Point3::new(p2d.x(), p2d.y(), z)
    } else if ay >= ax {
        let y = (d - normal.x() * p2d.x() - normal.z() * p2d.y()) / normal.y();
        Point3::new(p2d.x(), y, p2d.y())
    } else {
        let x = (d - normal.y() * p2d.x() - normal.z() * p2d.y()) / normal.x();
        Point3::new(x, p2d.x(), p2d.y())
    }
}

/// Triangulate a simple polygon (no holes) in 3D using CDT.
pub(super) fn cdt_triangulate_simple(positions: &[Point3], normal: Vec3) -> Vec<u32> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;

    let n = positions.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![0, 1, 2];
    }

    let pts2d: Vec<Point2> = positions
        .iter()
        .map(|&p| project_by_normal(p, normal))
        .collect();

    let bounds = compute_cdt_bounds(&pts2d);
    let mut cdt = Cdt::with_capacity(bounds, n);

    let cdt_indices = match cdt.insert_points_hilbert(&pts2d) {
        Ok(indices) => indices,
        Err(_) => return fan_triangulate(n),
    };

    let mut constraints = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let ci = cdt_indices[i];
        let cj = cdt_indices[j];
        if ci != cj {
            if cdt.insert_constraint(ci, cj).is_err() {
                return fan_triangulate(n);
            }
            constraints.push((ci, cj));
        }
    }

    cdt.remove_exterior(&constraints);

    let cdt_triangles = cdt.triangles();

    let mut cdt_to_input: DetHashMap<usize, usize> = DetHashMap::default();
    for (input_idx, &cdt_idx) in cdt_indices.iter().enumerate() {
        cdt_to_input.entry(cdt_idx).or_insert(input_idx);
    }

    let mut indices = Vec::with_capacity(cdt_triangles.len() * 3);
    let mut mapped = 0usize;
    for &(v0, v1, v2) in &cdt_triangles {
        if let (Some(&i0), Some(&i1), Some(&i2)) = (
            cdt_to_input.get(&v0),
            cdt_to_input.get(&v1),
            cdt_to_input.get(&v2),
        ) {
            mapped += 1;
            #[allow(clippy::cast_possible_truncation)]
            {
                indices.push(i0 as u32);
                indices.push(i1 as u32);
                indices.push(i2 as u32);
            }
        }
    }

    // A boundary constraint can only cross another constraint when the polygon
    // self-intersects (booleans occasionally emit a planar face whose outer
    // wire pinches through zero width — two boundary arcs that overlap by a few
    // hundred microns). CDT recovers crossing constraints by inserting a Steiner
    // vertex at the crossing; triangles touching it have no input-vertex mapping
    // and would be dropped here, leaving a hole. The Steiner vertex also splits
    // the shared boundary edges, which cracks against the neighbouring faces.
    // Fall back to a fan, which uses only the original boundary vertices and is
    // manifold by construction (each boundary edge used once, each diagonal
    // twice) regardless of the self-overlap.
    if mapped < cdt_triangles.len() || indices.is_empty() {
        return fan_triangulate(n);
    }

    indices
}

/// Fan triangulation as a last-resort fallback.
pub(super) fn fan_triangulate(n: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity((n - 2) * 3);
    for i in 1..n - 1 {
        #[allow(clippy::cast_possible_truncation)]
        {
            indices.push(0_u32);
            indices.push(i as u32);
            indices.push((i + 1) as u32);
        }
    }
    indices
}

/// Collect global vertex IDs from a wire, deduplicating consecutive vertices.
pub(super) fn collect_wire_global_vertices(
    wire: &brepkit_topology::wire::Wire,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    positions: &[Point3],
    tol: f64,
) -> (Vec<Point3>, Vec<Option<u32>>) {
    let mut out_positions: Vec<Point3> = Vec::new();
    let mut out_global_ids: Vec<Option<u32>> = Vec::new();

    for oe in wire.edges() {
        let edge_idx = oe.edge().index();
        if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
            let is_fwd = oe.is_forward();
            let len = global_ids.len();
            for j in 0..len {
                let gid = if is_fwd {
                    global_ids[j]
                } else {
                    global_ids[len - 1 - j]
                };
                if j == 0 && !out_global_ids.is_empty() {
                    let last_gid = out_global_ids.last().and_then(|g| *g).unwrap_or(u32::MAX);
                    if last_gid == gid {
                        continue;
                    }
                    if (last_gid as usize) < positions.len()
                        && (gid as usize) < positions.len()
                        && (positions[last_gid as usize] - positions[gid as usize]).length() < tol
                    {
                        continue;
                    }
                }
                out_positions.push(positions[gid as usize]);
                out_global_ids.push(Some(gid));
            }
        }
    }

    (out_positions, out_global_ids)
}

/// Remove the last element from parallel position/ID vectors if it duplicates
/// the first (closed wire loop-back).
pub(super) fn remove_closing_duplicate_global(
    positions: &mut Vec<Point3>,
    global_ids: &mut Vec<Option<u32>>,
    all_positions: &[Point3],
    tol: f64,
) {
    if global_ids.len() > 2
        && let (Some(&Some(first)), Some(&Some(last))) = (global_ids.first(), global_ids.last())
        && (first == last
            || ((first as usize) < all_positions.len()
                && (last as usize) < all_positions.len()
                && (all_positions[first as usize] - all_positions[last as usize]).length() < tol))
    {
        positions.pop();
        global_ids.pop();
    }
}

/// Remove the last element from a global ID list if it duplicates the first.
pub(super) fn remove_closing_duplicate_ids(ids: &mut Vec<u32>, positions: &[Point3], tol: f64) {
    if ids.len() > 2
        && let (Some(&first), Some(&last)) = (ids.first(), ids.last())
        && (first == last
            || ((first as usize) < positions.len()
                && (last as usize) < positions.len()
                && (positions[first as usize] - positions[last as usize]).length() < tol))
    {
        ids.pop();
    }
}

/// CDT tessellation for a planar face with inner wires, writing into a shared mesh.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(super) fn tessellate_planar_shared_with_holes(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    boundary_global_ids: &[u32],
    outer_positions: &[Point3],
    normal: Vec3,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
) -> Result<(), crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;

    let mut all_positions: Vec<Point3> = outer_positions.to_vec();
    let mut all_global_ids: Vec<Option<u32>> =
        boundary_global_ids.iter().map(|&g| Some(g)).collect();
    let outer_count = all_positions.len();
    let mut inner_wire_ranges: Vec<(usize, usize)> = Vec::new();

    let tol = 1e-10;
    for &iw_id in face_data.inner_wires() {
        let iw = topo.wire(iw_id)?;
        let start = all_positions.len();
        let (inner_pos, inner_gids) =
            collect_wire_global_vertices(iw, edge_global_indices, &merged.positions, tol);
        let mut inner_flat_ids: Vec<u32> = Vec::with_capacity(inner_gids.len());
        for (pos, gid_opt) in inner_pos.into_iter().zip(inner_gids) {
            if let Some(gid) = gid_opt {
                inner_flat_ids.push(gid);
                all_positions.push(pos);
                all_global_ids.push(Some(gid));
            } else {
                let key = point_merge_key(pos, MERGE_GRID);
                let gid = *point_to_global.entry(key).or_insert_with(|| {
                    #[allow(clippy::cast_possible_truncation)]
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(pos);
                    merged.normals.push(normal);
                    idx
                });
                inner_flat_ids.push(gid);
                all_positions.push(pos);
                all_global_ids.push(Some(gid));
            }
        }
        if inner_flat_ids.len() > 2 {
            remove_closing_duplicate_ids(&mut inner_flat_ids, &merged.positions, tol);
            let expected_end = start + inner_flat_ids.len();
            all_positions.truncate(expected_end);
            all_global_ids.truncate(expected_end);
        }
        let end = all_positions.len();
        inner_wire_ranges.push((start, end));
    }

    let pts2d: Vec<Point2> = all_positions
        .iter()
        .map(|&p| project_by_normal(p, normal))
        .collect();
    let bounds = compute_cdt_bounds(&pts2d);

    let mut cdt = Cdt::with_capacity(bounds, pts2d.len());
    let cdt_indices = cdt
        .insert_points_hilbert(&pts2d)
        .map_err(crate::OperationsError::Math)?;

    let mut all_constraints: Vec<(usize, usize)> = Vec::new();
    for i in 0..outer_count {
        let j = (i + 1) % outer_count;
        let ci = cdt_indices[i];
        let cj = cdt_indices[j];
        if ci != cj {
            cdt.insert_constraint(ci, cj)
                .map_err(crate::OperationsError::Math)?;
            all_constraints.push((ci, cj));
        }
    }

    for &(start, end) in &inner_wire_ranges {
        let count = end - start;
        for i in 0..count {
            let j = (i + 1) % count;
            let ci = cdt_indices[start + i];
            let cj = cdt_indices[start + j];
            if ci != cj {
                cdt.insert_constraint(ci, cj)
                    .map_err(crate::OperationsError::Math)?;
                all_constraints.push((ci, cj));
            }
        }
    }

    let outer_constraints: Vec<(usize, usize)> = (0..outer_count)
        .filter_map(|i| {
            let j = (i + 1) % outer_count;
            let ci = cdt_indices[i];
            let cj = cdt_indices[j];
            (ci != cj).then_some((ci, cj))
        })
        .collect();
    cdt.remove_exterior(&outer_constraints);

    let constraint_set: DetHashSet<(usize, usize)> = all_constraints
        .iter()
        .flat_map(|&(a, b)| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            [(lo, hi), (hi, lo)]
        })
        .collect();

    for seed in hole_removal_seeds(&pts2d, &inner_wire_ranges) {
        let _removed = cdt.flood_remove_from_point(seed, &constraint_set);
    }

    let cdt_triangles = cdt.triangles();

    let mut cdt_to_global: DetHashMap<usize, u32> = DetHashMap::default();
    for (local_idx, &cdt_idx) in cdt_indices.iter().enumerate() {
        if let Some(gid) = all_global_ids[local_idx] {
            cdt_to_global.insert(cdt_idx, gid);
        }
    }

    for &(v0, v1, v2) in &cdt_triangles {
        for &vi in &[v0, v1, v2] {
            if let std::collections::hash_map::Entry::Vacant(e) = cdt_to_global.entry(vi) {
                let p2d = cdt.vertices()[vi];
                let p3d = unproject_point(p2d, normal, &all_positions[0]);
                #[allow(clippy::cast_possible_truncation)]
                let gid = merged.positions.len() as u32;
                merged.positions.push(p3d);
                merged.normals.push(normal);
                e.insert(gid);
            }
        }
    }

    let needs_flip = if let Some(&(v0, v1, v2)) = cdt_triangles.first() {
        let g0 = cdt_to_global[&v0] as usize;
        let g1 = cdt_to_global[&v1] as usize;
        let g2 = cdt_to_global[&v2] as usize;
        let a = merged.positions[g1] - merged.positions[g0];
        let b = merged.positions[g2] - merged.positions[g0];
        a.cross(b).dot(normal) < 0.0
    } else {
        false
    };

    for &(v0, v1, v2) in &cdt_triangles {
        let g0 = cdt_to_global[&v0];
        let g1 = cdt_to_global[&v1];
        let g2 = cdt_to_global[&v2];
        if needs_flip {
            merged.indices.push(g0);
            merged.indices.push(g2);
            merged.indices.push(g1);
        } else {
            merged.indices.push(g0);
            merged.indices.push(g1);
            merged.indices.push(g2);
        }
    }

    Ok(())
}

/// Pure CDT computation for parallel execution across faces.
#[allow(clippy::too_many_lines)]
pub(super) fn run_planar_cdt(
    pts2d: &[brepkit_math::vec::Point2],
    outer_count: usize,
    inner_wire_ranges: &[(usize, usize)],
) -> Result<Vec<(usize, usize, usize)>, crate::OperationsError> {
    use brepkit_math::cdt::Cdt;

    let bounds = compute_cdt_bounds(pts2d);

    let mut cdt = Cdt::with_capacity(bounds, pts2d.len());
    let cdt_indices = cdt
        .insert_points_hilbert(pts2d)
        .map_err(crate::OperationsError::Math)?;

    let mut all_constraints: Vec<(usize, usize)> = Vec::new();
    for i in 0..outer_count {
        let j = (i + 1) % outer_count;
        let ci = cdt_indices[i];
        let cj = cdt_indices[j];
        if ci != cj {
            cdt.insert_constraint(ci, cj)
                .map_err(crate::OperationsError::Math)?;
            all_constraints.push((ci, cj));
        }
    }

    for &(start, end) in inner_wire_ranges {
        let count = end - start;
        for i in 0..count {
            let j = (i + 1) % count;
            let ci = cdt_indices[start + i];
            let cj = cdt_indices[start + j];
            if ci != cj {
                cdt.insert_constraint(ci, cj)
                    .map_err(crate::OperationsError::Math)?;
                all_constraints.push((ci, cj));
            }
        }
    }

    let outer_constraints: Vec<(usize, usize)> = (0..outer_count)
        .filter_map(|i| {
            let j = (i + 1) % outer_count;
            let ci = cdt_indices[i];
            let cj = cdt_indices[j];
            (ci != cj).then_some((ci, cj))
        })
        .collect();
    cdt.remove_exterior(&outer_constraints);

    let constraint_set: DetHashSet<(usize, usize)> = all_constraints
        .iter()
        .flat_map(|&(a, b)| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            [(lo, hi), (hi, lo)]
        })
        .collect();

    for seed in hole_removal_seeds(pts2d, inner_wire_ranges) {
        let _removed = cdt.flood_remove_from_point(seed, &constraint_set);
    }

    let cdt_triangles = cdt.triangles();

    let mut cdt_to_input: DetHashMap<usize, usize> = DetHashMap::default();
    for (input_idx, &cdt_idx) in cdt_indices.iter().enumerate() {
        cdt_to_input.entry(cdt_idx).or_insert(input_idx);
    }

    let mut result = Vec::with_capacity(cdt_triangles.len());
    for &(v0, v1, v2) in &cdt_triangles {
        if let (Some(&i0), Some(&i1), Some(&i2)) = (
            cdt_to_input.get(&v0),
            cdt_to_input.get(&v1),
            cdt_to_input.get(&v2),
        ) {
            result.push((i0, i1, i2));
        }
    }

    Ok(result)
}
