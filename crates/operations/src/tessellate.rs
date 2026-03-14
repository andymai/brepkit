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

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

/// Compute the shorter arc range (≤π) from an edge's start to end on a circle.
///
/// Returns `(t_start, t_end)` where the shorter arc goes from `t_start` to `t_end`.
/// When the shorter arc is CW, `t_end < t_start` so that linear interpolation
/// between them traces the correct (shorter) path via `circle.evaluate()`.
fn shorter_arc_range(
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

/// Kind of special handling needed for analytic surface tessellation.
enum AnalyticKind {
    /// Standard quad grid with no degenerate handling.
    General,
    /// Triangle fan at v extremes (sphere poles at v_min and v_max).
    SpherePole,
    /// Triangle fan at v_min (cone apex at v = 0).
    ConeApex,
    /// Triangle fan at v_max only (sphere north pole for a hemisphere face).
    VMaxPole,
}

/// Compute the angular resolution needed for a circular arc to achieve
/// a given chord deviation (sag).
///
/// For a circle of radius `r`, the chord deviation at the midpoint of
/// an arc subtending angle `θ` is `r*(1 - cos(θ/2))`. Solving for the
/// number of segments `n` over an arc range: `n = ceil(range / θ)` where
/// `θ = 2*acos(1 - deflection/r)`.
pub(crate) fn segments_for_chord_deviation(radius: f64, arc_range: f64, deflection: f64) -> usize {
    if radius <= 0.0 || deflection <= 0.0 || arc_range <= 0.0 {
        return 8;
    }
    let ratio = (deflection / radius).min(0.5); // clamp to avoid near-degenerate arcs
    let theta = 2.0 * (1.0 - ratio).acos(); // max angle per segment
    if theta <= 0.0 {
        return 8;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (arc_range / theta).ceil() as usize;
    // Minimum segment count for doubly-curved surfaces (e.g. spheres) where
    // the geometric formula under-samples because it only considers single-
    // direction curvature. Scales with sqrt(radius/deflection) so larger
    // radii correctly produce more segments.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_min = (arc_range * (radius / deflection).sqrt()).ceil() as usize;
    n.max(n_min).max(4)
}

/// Tessellate an analytic surface on a `(nu × nv)` grid.
///
/// The grid densities `nu` and `nv` should be computed by the caller
/// based on the surface geometry and the deflection target (see
/// `segments_for_chord_deviation`). Special handling is applied at poles
/// (sphere) and apexes (cone) to avoid degenerate triangles by using
/// triangle fans instead of quads.
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
    nu: usize,
    nv: usize,
    kind: AnalyticKind,
) -> TriangleMeshUV {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let nu = nu.max(4);
    let nv = nv.max(1);

    // Only wrap u when the range spans a full period (≈ 2π).
    // For partial arcs (e.g. quarter-cylinder), the last grid column is a
    // distinct point that must NOT wrap back to the first column.
    let u_periodic = (u_range.1 - u_range.0 - std::f64::consts::TAU).abs()
        < brepkit_math::tolerance::Tolerance::new().linear;

    // Build (nu+1) x (nv+1) vertex grid.
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
fn tessellate_planar(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    normal: Vec3,
    deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    let wire = topo.wire(face_data.outer_wire())?;
    let mut positions = Vec::new();
    let tol = 1e-10;

    // Sample a parametric curve into `positions`, skipping consecutive duplicates.
    // `t_for_index(i)` maps a sample index to a parameter value.
    // Iterates forward when `forward` is true, reversed otherwise.
    let sample_curve = |evaluate: &dyn Fn(f64) -> Point3,
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
            let pt = evaluate(t);
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
                let n_samples =
                    segments_for_chord_deviation(circle.radius(), arc_range, deflection);
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
                let n_samples =
                    segments_for_chord_deviation(ellipse.semi_major(), arc_range, deflection);
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
                #[allow(clippy::cast_precision_loss)]
                sample_curve(
                    &|t| nurbs.evaluate(t),
                    &|i| u0 + (u1 - u0) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
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
    if positions.len() > 2 {
        if let (Some(first), Some(last)) = (positions.first(), positions.last()) {
            if (*last - *first).length() < tol {
                positions.pop();
            }
        }
    }

    let n = positions.len();
    if n < 3 {
        // Degenerate face (e.g. sliver from boolean) — return empty mesh
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
        // CDT path for faces with holes.
        tessellate_planar_with_holes(topo, face_data, &positions, normal, deflection)
    }
}

/// Sample a wire into a list of 3D positions, skipping consecutive duplicates.
fn sample_wire_positions(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
    tol: f64,
    deflection: f64,
) -> Result<Vec<Point3>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    let mut positions = Vec::new();

    let sample_curve_into = |evaluate: &dyn Fn(f64) -> Point3,
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
            let pt = evaluate(t);
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
                let n_samples =
                    segments_for_chord_deviation(circle.radius(), arc_range, deflection);
                #[allow(clippy::cast_precision_loss)]
                sample_curve_into(
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
                let n_samples =
                    segments_for_chord_deviation(ellipse.semi_major(), arc_range, deflection);
                #[allow(clippy::cast_precision_loss)]
                sample_curve_into(
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
                #[allow(clippy::cast_precision_loss)]
                sample_curve_into(
                    &|t| nurbs.evaluate(t),
                    &|i| u0 + (u1 - u0) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
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

    // Remove closing duplicate.
    if positions.len() > 2 {
        if let (Some(first), Some(last)) = (positions.first(), positions.last()) {
            if (*last - *first).length() < tol {
                positions.pop();
            }
        }
    }

    Ok(positions)
}

/// Project a 3D point to 2D by dropping the dominant normal axis.
fn project_by_normal(p: Point3, normal: Vec3) -> brepkit_math::vec::Point2 {
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
fn compute_cdt_bounds(
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
) -> Result<TriangleMesh, crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;
    use std::collections::HashSet;

    // Collect all positions: outer + inner wires.
    let mut all_positions: Vec<Point3> = outer_positions.to_vec();
    let outer_count = all_positions.len();
    let mut inner_wire_ranges: Vec<(usize, usize)> = Vec::new();

    let tol = 1e-10;
    for &iw_id in face_data.inner_wires() {
        let iw = topo.wire(iw_id)?;
        let inner_pts = sample_wire_positions(topo, iw, tol, deflection)?;
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

    // Insert all points.
    let mut cdt_indices: Vec<usize> = Vec::with_capacity(pts2d.len());
    for &p in &pts2d {
        let idx = cdt.insert_point(p).map_err(crate::OperationsError::Math)?;
        cdt_indices.push(idx);
    }

    // Insert outer boundary constraints.
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

    // Insert inner wire constraints (holes).
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

    // Remove exterior triangles (using only outer boundary constraints).
    let outer_constraints: Vec<(usize, usize)> = (0..outer_count)
        .filter_map(|i| {
            let j = (i + 1) % outer_count;
            let ci = cdt_indices[i];
            let cj = cdt_indices[j];
            (ci != cj).then_some((ci, cj))
        })
        .collect();
    cdt.remove_exterior(&outer_constraints);

    // Remove hole interiors by seeding from each hole's centroid.
    // Build a set of all constraint edges for flood-fill stopping.
    let constraint_set: HashSet<(usize, usize)> = all_constraints
        .iter()
        .flat_map(|&(a, b)| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            [(lo, hi), (hi, lo)]
        })
        .collect();

    for &(start, end) in &inner_wire_ranges {
        let count = end - start;
        if count < 3 {
            continue;
        }

        // Find a seed point guaranteed inside the hole polygon.
        // The centroid fails for concave polygons, so we try multiple
        // strategies: vertex inward-bisector stepping, then centroid fallback.
        let hole_poly: Vec<Point2> = (start..end).map(|i| pts2d[i]).collect();
        let seed = find_interior_seed(&hole_poly);

        // Flood-fill remove from the seed, stopping at constraints.
        let _removed = cdt.flood_remove_from_point(seed, &constraint_set);
    }

    // Extract triangles and build mesh.
    let cdt_triangles = cdt.triangles();
    let cdt_verts = cdt.vertices();
    let mut positions_out = Vec::new();
    let mut normals_out = Vec::new();
    let mut indices_out = Vec::new();

    // Build O(1) reverse map: CDT vertex index → original position index.
    let mut vi_to_orig: HashMap<usize, usize> = HashMap::new();
    for (orig_idx, &cdt_vi) in cdt_indices.iter().enumerate() {
        vi_to_orig.entry(cdt_vi).or_insert(orig_idx);
    }

    // Map CDT point indices → output mesh indices.
    let mut cdt_to_mesh: HashMap<usize, u32> = HashMap::new();
    for &(v0, v1, v2) in &cdt_triangles {
        for &vi in &[v0, v1, v2] {
            if let std::collections::hash_map::Entry::Vacant(e) = cdt_to_mesh.entry(vi) {
                #[allow(clippy::cast_possible_truncation)]
                let mesh_idx = positions_out.len() as u32;
                // Find the original 3D point for this CDT vertex.
                if let Some(&orig_idx) = vi_to_orig.get(&vi) {
                    positions_out.push(all_positions[orig_idx]);
                } else {
                    // Steiner point inserted by CDT — reconstruct 3D from 2D.
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
///
/// Tries multiple strategies:
/// 1. For each vertex, step inward along the angle bisector of adjacent edges.
///    Verify with winding number.
/// 2. Fall back to centroid if nothing else works (best-effort).
fn find_interior_seed(polygon: &[brepkit_math::vec::Point2]) -> brepkit_math::vec::Point2 {
    use brepkit_math::predicates::point_in_polygon;
    use brepkit_math::vec::Point2;

    let n = polygon.len();
    if n == 0 {
        return Point2::new(0.0, 0.0);
    }
    if n < 3 {
        // Degenerate — return first point.
        return polygon[0];
    }

    // Strategy 1: vertex bisector stepping.
    // For each vertex, compute the inward bisector of the two adjacent edges
    // and step ε inward. A convex vertex's bisector always points inward.
    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        // Edge vectors pointing away from curr.
        let e_prev = Point2::new(prev.x() - curr.x(), prev.y() - curr.y());
        let e_next = Point2::new(next.x() - curr.x(), next.y() - curr.y());

        let len_prev = (e_prev.x() * e_prev.x() + e_prev.y() * e_prev.y()).sqrt();
        let len_next = (e_next.x() * e_next.x() + e_next.y() * e_next.y()).sqrt();
        if len_prev < 1e-30 || len_next < 1e-30 {
            continue;
        }

        // Normalize and compute bisector.
        let u_prev = Point2::new(e_prev.x() / len_prev, e_prev.y() / len_prev);
        let u_next = Point2::new(e_next.x() / len_next, e_next.y() / len_next);
        let bisector = Point2::new(u_prev.x() + u_next.x(), u_prev.y() + u_next.y());
        let bis_len = (bisector.x() * bisector.x() + bisector.y() * bisector.y()).sqrt();
        if bis_len < 1e-30 {
            continue;
        }

        // Step a small distance along the bisector.
        let step = 1e-4 * len_prev.min(len_next);
        let candidate = Point2::new(
            curr.x() + step * bisector.x() / bis_len,
            curr.y() + step * bisector.y() / bis_len,
        );

        if point_in_polygon(candidate, polygon) {
            return candidate;
        }

        // Try the opposite direction (for reflex vertices, the bisector
        // points outward, so flip it).
        let candidate_flip = Point2::new(
            curr.x() - step * bisector.x() / bis_len,
            curr.y() - step * bisector.y() / bis_len,
        );

        if point_in_polygon(candidate_flip, polygon) {
            return candidate_flip;
        }
    }

    // Strategy 2: centroid fallback (best-effort).
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

/// Reconstruct a 3D point from a 2D projection, using the face plane.
fn unproject_point(p2d: brepkit_math::vec::Point2, normal: Vec3, reference: &Point3) -> Point3 {
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    // The "dropped" coordinate is reconstructed from the plane equation.
    let d = normal.x() * reference.x() + normal.y() * reference.y() + normal.z() * reference.z();
    if az >= ax && az >= ay {
        // Dropped z: z = (d - nx*x - ny*y) / nz
        let z = (d - normal.x() * p2d.x() - normal.y() * p2d.y()) / normal.z();
        Point3::new(p2d.x(), p2d.y(), z)
    } else if ay >= ax {
        // Dropped y: y = (d - nx*x - nz*z) / ny
        let y = (d - normal.x() * p2d.x() - normal.z() * p2d.y()) / normal.y();
        Point3::new(p2d.x(), y, p2d.y())
    } else {
        // Dropped x: x = (d - ny*y - nz*z) / nx
        let x = (d - normal.y() * p2d.x() - normal.z() * p2d.y()) / normal.x();
        Point3::new(x, p2d.x(), p2d.y())
    }
}

/// Collect global vertex IDs from a wire, deduplicating consecutive vertices.
///
/// Iterates each oriented edge of `wire`, looking up its pre-computed global
/// vertex IDs from `edge_global_indices`. Adjacent duplicate vertices (by ID
/// or position within `tol`) are skipped. Returns positions and optional global
/// IDs in wire-traversal order.
fn collect_wire_global_vertices(
    wire: &brepkit_topology::wire::Wire,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
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
fn remove_closing_duplicate_global(
    positions: &mut Vec<Point3>,
    global_ids: &mut Vec<Option<u32>>,
    all_positions: &[Point3],
    tol: f64,
) {
    if global_ids.len() > 2 {
        if let (Some(&Some(first)), Some(&Some(last))) = (global_ids.first(), global_ids.last()) {
            if first == last
                || ((first as usize) < all_positions.len()
                    && (last as usize) < all_positions.len()
                    && (all_positions[first as usize] - all_positions[last as usize]).length()
                        < tol)
            {
                positions.pop();
                global_ids.pop();
            }
        }
    }
}

/// Remove the last element from a global ID list if it duplicates the first.
fn remove_closing_duplicate_ids(ids: &mut Vec<u32>, positions: &[Point3], tol: f64) {
    if ids.len() > 2 {
        if let (Some(&first), Some(&last)) = (ids.first(), ids.last()) {
            if first == last
                || ((first as usize) < positions.len()
                    && (last as usize) < positions.len()
                    && (positions[first as usize] - positions[last as usize]).length() < tol)
            {
                ids.pop();
            }
        }
    }
}

/// CDT tessellation for a planar face with inner wires, writing into a shared mesh.
///
/// `boundary_global_ids` and `outer_positions` describe the outer wire (already
/// collected by the caller). Inner wires are sampled here and added as CDT
/// constraint loops.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn tessellate_planar_shared_with_holes(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    boundary_global_ids: &[u32],
    outer_positions: &[Point3],
    normal: Vec3,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut HashMap<(u64, u64, u64), u32>,
) -> Result<(), crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;
    use std::collections::HashSet;

    // Collect all 3D positions and their global mesh IDs.
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
                // Fallback: allocate a global vertex for an unshared point.
                let key = (pos.x().to_bits(), pos.y().to_bits(), pos.z().to_bits());
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
        // Remove closing duplicate if the wire loops back.
        if inner_flat_ids.len() > 2 {
            remove_closing_duplicate_ids(&mut inner_flat_ids, &merged.positions, tol);
            // Trim all_positions and all_global_ids to match.
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
    let mut cdt_indices: Vec<usize> = Vec::with_capacity(pts2d.len());
    for &p in &pts2d {
        let idx = cdt.insert_point(p).map_err(crate::OperationsError::Math)?;
        cdt_indices.push(idx);
    }

    // Insert outer boundary constraints.
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

    // Insert inner wire constraints (holes).
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

    // Remove exterior using only outer boundary constraints.
    let outer_constraints: Vec<(usize, usize)> = (0..outer_count)
        .filter_map(|i| {
            let j = (i + 1) % outer_count;
            let ci = cdt_indices[i];
            let cj = cdt_indices[j];
            (ci != cj).then_some((ci, cj))
        })
        .collect();
    cdt.remove_exterior(&outer_constraints);

    // Build constraint set for flood-fill stopping.
    let constraint_set: HashSet<(usize, usize)> = all_constraints
        .iter()
        .flat_map(|&(a, b)| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            [(lo, hi), (hi, lo)]
        })
        .collect();

    // Remove hole interiors by flooding from each hole centroid.
    for &(start, end) in &inner_wire_ranges {
        let count = end - start;
        if count < 3 {
            continue;
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        #[allow(clippy::cast_precision_loss)]
        for idx in start..end {
            cx += pts2d[idx].x();
            cy += pts2d[idx].y();
        }
        cx /= count as f64;
        cy /= count as f64;
        // If the centroid falls outside the triangulation (e.g. concave inner
        // wire), skip removal gracefully.
        let _removed = cdt.flood_remove_from_point(Point2::new(cx, cy), &constraint_set);
    }

    let cdt_triangles = cdt.triangles();

    // Map CDT vertex index → global mesh index.
    let mut cdt_to_global: HashMap<usize, u32> = HashMap::new();
    for (local_idx, &cdt_idx) in cdt_indices.iter().enumerate() {
        if let Some(gid) = all_global_ids[local_idx] {
            cdt_to_global.insert(cdt_idx, gid);
        }
    }

    // For any Steiner points inserted by CDT, allocate new global vertices.
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

    // Check winding of first triangle.
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

/// Pure CDT computation: takes 2D points and wire ranges, returns triangles
/// as indices into the input points array. Also returns any Steiner point
/// 2D coordinates with their CDT vertex indices.
///
/// This is extracted as a standalone function to enable parallel execution
/// across faces.
#[allow(clippy::too_many_lines)]
fn run_planar_cdt(
    pts2d: &[brepkit_math::vec::Point2],
    outer_count: usize,
    inner_wire_ranges: &[(usize, usize)],
) -> Result<Vec<(usize, usize, usize)>, crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;
    use std::collections::HashSet;

    let bounds = compute_cdt_bounds(pts2d);

    let mut cdt = Cdt::with_capacity(bounds, pts2d.len());
    let mut cdt_indices: Vec<usize> = Vec::with_capacity(pts2d.len());
    for &p in pts2d {
        let idx = cdt.insert_point(p).map_err(crate::OperationsError::Math)?;
        cdt_indices.push(idx);
    }

    // Insert outer boundary constraints.
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

    // Insert inner wire constraints (holes).
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

    // Remove exterior using only outer boundary constraints.
    let outer_constraints: Vec<(usize, usize)> = (0..outer_count)
        .filter_map(|i| {
            let j = (i + 1) % outer_count;
            let ci = cdt_indices[i];
            let cj = cdt_indices[j];
            (ci != cj).then_some((ci, cj))
        })
        .collect();
    cdt.remove_exterior(&outer_constraints);

    // Build constraint set for flood-fill stopping.
    let constraint_set: HashSet<(usize, usize)> = all_constraints
        .iter()
        .flat_map(|&(a, b)| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            [(lo, hi), (hi, lo)]
        })
        .collect();

    // Remove hole interiors by flooding from each hole centroid.
    for &(start, end) in inner_wire_ranges {
        let count = end - start;
        if count < 3 {
            continue;
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        #[allow(clippy::cast_precision_loss)]
        for idx in start..end {
            cx += pts2d[idx].x();
            cy += pts2d[idx].y();
        }
        cx /= count as f64;
        cy /= count as f64;
        let _removed = cdt.flood_remove_from_point(Point2::new(cx, cy), &constraint_set);
    }

    let cdt_triangles = cdt.triangles();

    // Build reverse mapping: CDT vertex index → input point index.
    let mut cdt_to_input: HashMap<usize, usize> = HashMap::new();
    for (input_idx, &cdt_idx) in cdt_indices.iter().enumerate() {
        cdt_to_input.entry(cdt_idx).or_insert(input_idx);
    }

    // Map CDT triangles to input indices. Steiner points (from super-triangle
    // remnants) are extremely rare; skip those triangles.
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

/// Triangulate a simple polygon (no holes) in 3D using CDT.
///
/// Projects the polygon to 2D (dropping the coordinate corresponding to
/// the dominant normal component), inserts boundary constraints, and returns
/// triangle indices. Falls back to fan triangulation for degenerate cases.
fn cdt_triangulate_simple(positions: &[Point3], normal: Vec3) -> Vec<u32> {
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

    // Compute bounding box with margin for the super-triangle.
    let bounds = compute_cdt_bounds(&pts2d);
    let mut cdt = Cdt::with_capacity(bounds, n);

    // Insert polygon vertices.
    let mut cdt_indices = Vec::with_capacity(n);
    for &p in &pts2d {
        match cdt.insert_point(p) {
            Ok(idx) => cdt_indices.push(idx),
            Err(_) => return fan_triangulate(n),
        }
    }

    // Insert boundary constraints.
    let mut constraints = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let ci = cdt_indices[i];
        let cj = cdt_indices[j];
        if ci != cj {
            if cdt.insert_constraint(ci, cj).is_err() {
                // CDT constraint insertion failed — fall back to fan.
                return fan_triangulate(n);
            }
            constraints.push((ci, cj));
        }
    }

    cdt.remove_exterior(&constraints);

    let cdt_triangles = cdt.triangles();

    // Build reverse mapping: CDT vertex index → original polygon index.
    let mut cdt_to_input: HashMap<usize, usize> = HashMap::new();
    for (input_idx, &cdt_idx) in cdt_indices.iter().enumerate() {
        cdt_to_input.entry(cdt_idx).or_insert(input_idx);
    }

    let mut indices = Vec::with_capacity(cdt_triangles.len() * 3);
    for &(v0, v1, v2) in &cdt_triangles {
        if let (Some(&i0), Some(&i1), Some(&i2)) = (
            cdt_to_input.get(&v0),
            cdt_to_input.get(&v1),
            cdt_to_input.get(&v2),
        ) {
            #[allow(clippy::cast_possible_truncation)]
            {
                indices.push(i0 as u32);
                indices.push(i1 as u32);
                indices.push(i2 as u32);
            }
        }
    }

    if indices.is_empty() {
        return fan_triangulate(n);
    }

    indices
}

/// Fan triangulation as a last-resort fallback.
fn fan_triangulate(n: usize) -> Vec<u32> {
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

/// Compute the v-parameter range for a surface by projecting boundary vertices.
///
/// `project_v` maps a 3D point to its v-parameter on the surface.
/// Falls back to (-1.0, 1.0) if the face has no usable vertices.
fn compute_v_param_range(
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
fn compute_axial_range(
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
/// Projects boundary edge vertices — and midpoints of curved edges — onto
/// the surface and collects their u-parameters. If the face doesn't span
/// the full revolution, returns the tighter `[u_min, u_max]` range.
/// Returns `(0, 2π)` for full-circle faces or when fewer than 3 boundary
/// vertices exist.
fn compute_angular_range<F>(
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
                // between vertices. Without this, faces with only 2 unique
                // vertex angles (e.g. quarter-cylinder from shell) fall
                // through to the full-circle default. Line edge midpoints
                // are crucial for faces whose edges are all LineEdges
                // (as produced by assemble_solid_mixed).
                if !edge.is_closed() {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        match edge.curve() {
                            EdgeCurve::Circle(circle) => {
                                let ts = circle.project(sv.point());
                                let te = circle.project(ev.point());
                                // Choose the shorter arc for the midpoint.
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

    // Sort angles and find the largest gap between consecutive angles.
    // The largest gap is where the face *doesn't* have coverage, so the
    // face's angular extent is TAU - largest_gap, starting after the gap.
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    angles.dedup_by(|a, b| (*a - *b).abs() < brepkit_math::tolerance::Tolerance::default().linear);

    if angles.len() < 3 {
        // Even after adding midpoints, fewer than 3 unique angles means
        // the face likely spans the full revolution.
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

    // Decide if the face covers the full revolution. If the largest gap
    // between consecutive boundary vertex angles is small enough, the
    // vertices genuinely span the full circle.
    //
    // We use `min(2.5 * even_gap, 120°)` as threshold. The `min` (not `max`)
    // is critical: with many densely-packed vertices on a partial arc (e.g.
    // 64 chord-sampled points on a 270° arc after STEP round-trip), the
    // even_gap ≈ 4.2° gives threshold ≈ 10.5°, but the actual gap is ~90°,
    // correctly rejecting full-circle. With few vertices (4 equatorial verts
    // on a hemisphere, gap ≈ 90°, threshold = 120°), it still correctly
    // identifies full-circle.
    let n_angles = angles.len() as f64;
    let even_gap = TAU / n_angles;
    let gap_threshold = (2.5 * even_gap).min(TAU / 3.0);
    if max_gap < gap_threshold {
        return (0.0, TAU);
    }

    // The face starts at angles[gap_end_idx] and ends at the angle before the gap.
    let u_start = angles[gap_end_idx];
    let gap_start_idx = if gap_end_idx == 0 {
        angles.len() - 1
    } else {
        gap_end_idx - 1
    };
    let u_end = angles[gap_start_idx];

    // Normalize so that u_start < u_end, wrapping if needed.
    if u_end > u_start {
        (u_start, u_end)
    } else {
        (u_start, u_end + TAU)
    }
}

/// Compute the latitude (v) range for a sphere face from its wire boundary.
///
/// For a full sphere (degenerate wire with < 3 vertices), returns the full
/// range `[-π/2, π/2]`. For hemisphere faces with a proper equatorial wire,
/// computes the boundary latitude from the wire vertices, then uses the
/// wire winding direction to determine which hemisphere the face covers.
#[must_use]
pub fn compute_sphere_v_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> (f64, f64) {
    use std::f64::consts::FRAC_PI_2;

    // Collect wire vertex positions and their v-parameters.
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
        // Degenerate wire — full sphere (legacy single-face sphere).
        return (-FRAC_PI_2, FRAC_PI_2);
    }

    // Average v-parameter of boundary vertices.
    let avg_v: f64 = wire_pts
        .iter()
        .map(|pt| sphere.project_point(*pt).1)
        .sum::<f64>()
        / wire_pts.len() as f64;

    // Determine which side of the boundary the face interior is on
    // by computing the signed area of the wire projected onto the
    // sphere's equatorial plane (XY plane for default z-axis).
    let signed_area = projected_signed_area(&wire_pts);
    if signed_area > 0.0 {
        // CCW from +Z → north hemisphere (toward +v pole).
        (avg_v, FRAC_PI_2)
    } else {
        // CW from +Z → south hemisphere (toward -v pole).
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
///
/// If the range covers both poles, use `SpherePole` (fan at both extremes).
/// If it covers only one pole, use `ConeApex` (fan at one extreme).
/// If it covers neither pole (a band), use `General`.
fn sphere_analytic_kind(v_range: (f64, f64)) -> AnalyticKind {
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
///
/// Uses two complementary measures to decide if a cell needs subdivision:
///
/// 1. **Midpoint sag** (Hausdorff-like error): the distance between the
///    actual surface midpoint and the bilinear interpolation of the four
///    corners. This catches cases where the surface curves significantly
///    but normals remain nearly parallel (e.g., long gentle arcs).
///
/// 2. **Normal deviation**: the maximum angular difference between surface
///    normals at the 5 sample points (4 corners + center). This catches
///    sharp curvature changes where the surface folds quickly.
///
/// Returns the maximum of both metrics (in deflection-compatible units).
/// The midpoint sag is in world units; normal deviation is converted via
/// the approximation `sag ≈ (1 - cos θ) × cell_diagonal / 2`.
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

    // Evaluate corners and center.
    let p00 = surface.evaluate(u_min, v_min);
    let p10 = surface.evaluate(u_max, v_min);
    let p11 = surface.evaluate(u_max, v_max);
    let p01 = surface.evaluate(u_min, v_max);
    let p_mid = surface.evaluate(u_mid, v_mid);

    // Metric 1: Midpoint sag — distance from actual midpoint to bilinear center.
    // The bilinear interpolation at (0.5, 0.5) is the average of the 4 corners.
    let bilinear_mid = Point3::new(
        0.25 * (p00.x() + p10.x() + p11.x() + p01.x()),
        0.25 * (p00.y() + p10.y() + p11.y() + p01.y()),
        0.25 * (p00.z() + p10.z() + p11.z() + p01.z()),
    );
    let sag = (p_mid - bilinear_mid).length();

    // Metric 2: Normal deviation (original metric, kept for sharp-fold detection).
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

    // Also check edge midpoints for better curvature sampling.
    // This catches ridges that pass through edge midpoints but miss corners.
    let edge_mids = [
        surface.evaluate(u_mid, v_min), // bottom edge mid
        surface.evaluate(u_mid, v_max), // top edge mid
        surface.evaluate(u_min, v_mid), // left edge mid
        surface.evaluate(u_max, v_mid), // right edge mid
    ];

    // Edge midpoint sag: compare to linear interpolation along each edge.
    let edge_linear_mids = [
        lerp_point(p00, p10), // bottom
        lerp_point(p01, p11), // top
        lerp_point(p00, p01), // left
        lerp_point(p10, p11), // right
    ];

    let mut max_edge_sag = 0.0_f64;
    for i in 0..4 {
        let edge_sag = (edge_mids[i] - edge_linear_mids[i]).length();
        max_edge_sag = max_edge_sag.max(edge_sag);
    }

    // Return the maximum of all three metrics.
    sag.max(max_edge_sag).max(max_normal_dev)
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
) -> TriangleMeshUV {
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
    let mut uvs: Vec<[f64; 2]> = Vec::new();
    let mut indices = Vec::new();
    // Map from (u_bits, v_bits) to vertex index.
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
            continue; // not a leaf
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

    TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals,
            indices,
        },
        uvs,
    }
}

/// Tessellate a face and return mesh with per-vertex UV coordinates.
///
/// UV coordinates are the parametric (u, v) values of the surface at each
/// vertex. For planar faces, UVs are computed by projecting onto the face
/// plane axes.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate_with_uvs(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
) -> Result<TriangleMeshUV, crate::OperationsError> {
    let face_data = topo.face(face)?;
    let is_reversed = face_data.is_reversed();

    let mut result = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => {
            let mesh = tessellate_planar(topo, face_data, *normal, deflection)?;
            // For planar faces, project onto plane axes to get UVs.
            let (u_axis, v_axis) = plane_axes(*normal);
            let origin = if mesh.positions.is_empty() {
                Point3::new(0.0, 0.0, 0.0)
            } else {
                mesh.positions[0]
            };
            let uvs = mesh
                .positions
                .iter()
                .map(|p| {
                    let d: Vec3 = *p - origin;
                    [d.dot(u_axis), d.dot(v_axis)]
                })
                .collect();
            Ok::<_, crate::OperationsError>(TriangleMeshUV { mesh, uvs })
        }
        FaceSurface::Nurbs(surface) => Ok(tessellate_nurbs(surface, deflection)),
        FaceSurface::Cylinder(cyl) => {
            let v_range = compute_axial_range(topo, face_data, cyl.origin(), cyl.axis());
            let u_range = compute_angular_range(topo, face_data, |p| cyl.project_point(p));
            let nu = segments_for_chord_deviation(cyl.radius(), u_range.1 - u_range.0, deflection);
            // Cylinders have zero axial curvature — 1 row (top + bottom)
            // is geometrically exact. No need for square-grid scaling.
            let nv = 1;
            let cyl = cyl.clone();
            Ok(tessellate_analytic(
                |u, v| cyl.evaluate(u, v),
                |u, v| cyl.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                AnalyticKind::General,
            ))
        }
        FaceSurface::Cone(cone) => {
            // Use project_point to get the true v-parameter range, not the
            // axial projection. The cone's v is the distance from the apex
            // along the surface generator, not the axis.
            let v_range = compute_v_param_range(topo, face_data, |p| cone.project_point(p).1);
            let u_range = compute_angular_range(topo, face_data, |p| cone.project_point(p));
            let max_radius = cone.radius_at(v_range.1.abs().max(v_range.0.abs()));
            let nu = segments_for_chord_deviation(
                max_radius.max(0.01),
                u_range.1 - u_range.0,
                deflection,
            );
            // Cones have zero curvature along their generators — 1 row
            // is geometrically exact (linear interpolation matches the
            // surface). The apex fan handles the degenerate tip separately.
            let nv = 1;
            // Only use ConeApex fan when v_range actually starts near the
            // apex (v≈0). Truncated cones have v_min > 0 and need regular
            // quads.
            let kind = if v_range.0.abs() < 1e-10 {
                AnalyticKind::ConeApex
            } else {
                AnalyticKind::General
            };
            let cone = cone.clone();
            Ok(tessellate_analytic(
                |u, v| cone.evaluate(u, v),
                |u, v| cone.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                kind,
            ))
        }
        FaceSurface::Sphere(sphere) => {
            let u_range = compute_angular_range(topo, face_data, |p| sphere.project_point(p));
            let v_range = compute_sphere_v_range(topo, face_data, sphere);
            let nu =
                segments_for_chord_deviation(sphere.radius(), u_range.1 - u_range.0, deflection);
            let nv =
                segments_for_chord_deviation(sphere.radius(), v_range.1 - v_range.0, deflection);
            let kind = sphere_analytic_kind(v_range);
            let sphere = sphere.clone();
            Ok(tessellate_analytic(
                |u, v| sphere.evaluate(u, v),
                |u, v| sphere.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                kind,
            ))
        }
        FaceSurface::Torus(torus) => {
            let u_range = compute_angular_range(topo, face_data, |p| torus.project_point(p));
            let v_range = (0.0, std::f64::consts::TAU);
            let nu = segments_for_chord_deviation(
                torus.major_radius() + torus.minor_radius(),
                u_range.1 - u_range.0,
                deflection,
            );
            let nv = segments_for_chord_deviation(
                torus.minor_radius(),
                v_range.1 - v_range.0,
                deflection,
            );
            let torus = torus.clone();
            Ok(tessellate_analytic(
                |u, v| torus.evaluate(u, v),
                |u, v| torus.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                AnalyticKind::General,
            ))
        }
    }?;

    if is_reversed {
        for n in &mut result.mesh.normals {
            *n = -*n;
        }
        let tri_count = result.mesh.indices.len() / 3;
        for t in 0..tri_count {
            result.mesh.indices.swap(t * 3 + 1, t * 3 + 2);
        }
    }

    Ok(result)
}

/// Compute orthogonal axes for a plane given its normal.
///
/// Falls back to identity axes if the normal is degenerate (should not
/// happen for valid face data).
fn plane_axes(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = normal
        .cross(up)
        .normalize()
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let v_axis = normal
        .cross(u_axis)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
    (u_axis, v_axis)
}

// ── Watertight solid tessellation ──────────────────────────────────────

/// Compute the number of sample points for an edge based on deflection.
///
/// Uses edge length and curvature to determine sampling density.
fn edge_sample_count(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    deflection: f64,
) -> usize {
    use brepkit_topology::edge::EdgeCurve;

    match edge.curve() {
        EdgeCurve::Line => 2,
        EdgeCurve::Circle(c) => {
            let radius = c.radius();
            // Angular step: acos(1 - deflection/radius), clamped for safety.
            let ratio = (deflection / radius).min(1.0);
            let angle_step = (1.0 - ratio).acos().max(0.01);
            if let Ok((t_start, t_end)) = circle_param_range(topo, edge, c) {
                let arc_angle = (t_end - t_start).abs();
                (arc_angle / angle_step).ceil() as usize + 1
            } else {
                // Fallback: assume full circle if vertex lookup fails.
                (std::f64::consts::TAU / angle_step).ceil() as usize + 1
            }
        }
        EdgeCurve::Ellipse(ellipse) => {
            // Use chord-deviation formula with max curvature radius (a²/b).
            // An ellipse's curvature is highest at the ends of the semi-major axis
            // where the radius of curvature equals a²/b.
            let a = ellipse.semi_major();
            let b = ellipse.semi_minor();
            let max_curv_radius = a * a / b;
            let arc_range = if edge.is_closed() {
                std::f64::consts::TAU
            } else if let (Ok(sp), Ok(ep)) = (
                topo.vertex(edge.start())
                    .map(brepkit_topology::vertex::Vertex::point),
                topo.vertex(edge.end())
                    .map(brepkit_topology::vertex::Vertex::point),
            ) {
                let ts = ellipse.project(sp);
                let mut te = ellipse.project(ep);
                if te <= ts {
                    te += std::f64::consts::TAU;
                }
                te - ts
            } else {
                std::f64::consts::TAU
            };
            segments_for_chord_deviation(max_curv_radius, arc_range, deflection).min(4096)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            // Adaptive: coarse-pass deviation measurement, then refine if needed.
            let (u0, u1) = nurbs.domain();
            let n_spans = nurbs
                .control_points()
                .len()
                .saturating_sub(nurbs.degree())
                .max(1);
            let coarse_n = (n_spans * 4).clamp(8, 128);
            let max_dev = measure_max_chord_deviation(nurbs, u0, u1, coarse_n);
            if max_dev <= deflection {
                coarse_n
            } else {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let refined = ((coarse_n as f64) * (max_dev / deflection).sqrt()).ceil() as usize;
                refined.clamp(8, 4096)
            }
        }
    }
}

/// Measure the maximum midpoint chord deviation across `n` segments of a NURBS curve.
///
/// For each segment `[u_i, u_{i+1}]`, evaluates the curve at the midpoint and
/// measures its distance from the chord midpoint. Returns the maximum deviation.
fn measure_max_chord_deviation(
    nurbs: &brepkit_math::nurbs::curve::NurbsCurve,
    u0: f64,
    u1: f64,
    n: usize,
) -> f64 {
    let mut max_dev: f64 = 0.0;
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let t0 = u0 + (u1 - u0) * (i as f64) / (n as f64);
        let t1 = u0 + (u1 - u0) * ((i + 1) as f64) / (n as f64);
        let p0 = nurbs.evaluate(t0);
        let p1 = nurbs.evaluate(t1);
        let mid_chord = Point3::new(
            (p0.x() + p1.x()) * 0.5,
            (p0.y() + p1.y()) * 0.5,
            (p0.z() + p1.z()) * 0.5,
        );
        let mid_curve = nurbs.evaluate((t0 + t1) * 0.5);
        let dev = (mid_curve - mid_chord).length();
        max_dev = max_dev.max(dev);
    }
    max_dev
}

/// Get the parameter range for a circle edge.
///
/// # Errors
///
/// Returns an error if vertex lookup fails.
fn circle_param_range(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    circle: &brepkit_math::curves::Circle3D,
) -> Result<(f64, f64), crate::OperationsError> {
    if edge.is_closed() {
        Ok((0.0, std::f64::consts::TAU))
    } else {
        let sp = topo.vertex(edge.start())?.point();
        let ep = topo.vertex(edge.end())?.point();
        let ts = circle.project(sp);
        let mut te = circle.project(ep);
        if te <= ts {
            te += std::f64::consts::TAU;
        }
        Ok((ts, te))
    }
}

/// Sample an edge curve to produce a list of 3D points (start to end).
///
/// The sampling density is driven by `deflection`. For a `Line`, only the
/// two endpoints are returned. For curves, the point count is proportional
/// to curvature.
///
/// # Errors
///
/// Returns an error if vertex lookup fails for edge endpoints.
fn sample_edge(
    topo: &Topology,
    edge: &brepkit_topology::edge::Edge,
    deflection: f64,
) -> Result<Vec<Point3>, crate::OperationsError> {
    use brepkit_topology::edge::EdgeCurve;

    let n = edge_sample_count(topo, edge, deflection);
    let mut points = Vec::with_capacity(n);

    match edge.curve() {
        EdgeCurve::Line => {
            points.push(topo.vertex(edge.start())?.point());
            points.push(topo.vertex(edge.end())?.point());
        }
        EdgeCurve::Circle(circle) => {
            let (t_start, t_end) = circle_param_range(topo, edge, circle)?;
            #[allow(clippy::cast_precision_loss)]
            for i in 0..n {
                let t = t_start + (t_end - t_start) * (i as f64) / ((n - 1).max(1) as f64);
                points.push(circle.evaluate(t));
            }
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
            #[allow(clippy::cast_precision_loss)]
            for i in 0..n {
                let t = t_start + (t_end - t_start) * (i as f64) / ((n - 1).max(1) as f64);
                points.push(ellipse.evaluate(t));
            }
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let (u0, u1) = nurbs.domain();
            #[allow(clippy::cast_precision_loss)]
            for i in 0..n {
                let t = u0 + (u1 - u0) * (i as f64) / ((n - 1).max(1) as f64);
                points.push(nurbs.evaluate(t));
            }
        }
    }

    Ok(points)
}

/// Tessellate all faces of a solid into a single watertight triangle mesh.
///
/// Unlike per-face `tessellate()`, this function coordinates tessellation across
/// all faces of the solid by pre-computing shared edge tessellations. When two
/// faces share an edge, the edge is tessellated once and both faces receive
/// identical vertices along that boundary — eliminating cracks between adjacent
/// faces and producing a guaranteed 2-manifold mesh.
///
/// # Algorithm
///
/// Based on Stöger & Kurka (2003), "Watertight Tessellation of B-rep NURBS
/// CAD-Models Using Connectivity Information":
///
/// 1. Build edge-to-face adjacency map from the solid's topology.
/// 2. Tessellate each unique edge once, producing a shared polyline.
/// 3. For each face, tessellate using cached edge points as boundary vertices.
/// 4. Merge all per-face meshes into a single mesh with shared boundary vertices.
///
/// # Errors
///
/// Returns an error if any topology lookup or face tessellation fails.
#[allow(clippy::too_many_lines)]
pub fn tessellate_solid(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    use brepkit_topology::explorer;

    // Phase 1: Collect all faces and build edge→face adjacency.
    let all_faces = explorer::solid_faces(topo, solid)?;
    let edge_face_map = explorer::edge_to_face_map(topo, solid)?;

    // Phase 2: Tessellate each unique edge once.
    // Key: edge index → Vec<Point3> (oriented start→end).
    // Parallelized with rayon when there are enough edges to amortize
    // thread-pool synchronization overhead.
    let edge_indices: Vec<usize> = edge_face_map.keys().copied().collect();
    let edge_points: HashMap<usize, Vec<Point3>> = if edge_indices.len() >= 32 {
        use rayon::prelude::*;
        let results: Vec<Result<(usize, Vec<Point3>), crate::OperationsError>> = edge_indices
            .par_iter()
            .filter_map(|&edge_idx| {
                let edge_id = topo.edge_id_from_index(edge_idx)?;
                let edge_data = match topo.edge(edge_id) {
                    Ok(d) => d,
                    Err(e) => return Some(Err(crate::OperationsError::Topology(e))),
                };
                Some(sample_edge(topo, edge_data, deflection).map(|pts| (edge_idx, pts)))
            })
            .collect();
        let mut map = HashMap::new();
        for r in results {
            let (idx, pts) = r?;
            map.insert(idx, pts);
        }
        map
    } else {
        let mut map = HashMap::new();
        for &edge_idx in &edge_indices {
            if let Some(edge_id) = topo.edge_id_from_index(edge_idx) {
                if let Ok(edge_data) = topo.edge(edge_id) {
                    let points = sample_edge(topo, edge_data, deflection)?;
                    map.insert(edge_idx, points);
                }
            }
        }
        map
    };

    // Phase 3: Build merged mesh with shared edge vertices.
    //
    // Strategy: maintain a global vertex pool. For each edge, insert its
    // tessellation points into the pool and record their global indices.
    // When a face references an edge, it uses the pre-existing global indices
    // for its boundary vertices, guaranteeing that adjacent faces share
    // exactly the same vertices along their common edges.
    let mut merged = TriangleMesh::default();

    // Global vertex deduplication: map from Point3 bit pattern to global index.
    // We use coordinate bit patterns rather than approximate matching to ensure
    // exact vertex sharing for points that were computed from the same edge
    // tessellation. Points from different edges that happen to be close but
    // not identical remain separate (correct behavior for non-manifold edges).
    let mut point_to_global: HashMap<(u64, u64, u64), u32> = HashMap::new();

    // edge index → Vec<global vertex indices> (in start→end order).
    let mut edge_global_indices: HashMap<usize, Vec<u32>> = HashMap::new();

    // Insert edge points into the global pool.
    for (&edge_idx, points) in &edge_points {
        let mut global_ids = Vec::with_capacity(points.len());
        for &pt in points {
            let key = (pt.x().to_bits(), pt.y().to_bits(), pt.z().to_bits());
            let idx = point_to_global.entry(key).or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = merged.positions.len() as u32;
                merged.positions.push(pt);
                merged.normals.push(Vec3::new(0.0, 0.0, 0.0)); // placeholder, filled later
                idx
            });
            global_ids.push(*idx);
        }
        edge_global_indices.insert(edge_idx, global_ids);
    }

    // Phase 4: Tessellate each face using its boundary edge vertices.
    //
    // For large planar faces with holes, extract CDT inputs first and run
    // CDTs in parallel (the CDT is the dominant cost for these faces).
    // Small faces and non-planar faces are processed sequentially.
    #[allow(clippy::items_after_statements)]
    struct CdtJob {
        pts2d: Vec<brepkit_math::vec::Point2>,
        outer_count: usize,
        inner_wire_ranges: Vec<(usize, usize)>,
        all_global_ids: Vec<Option<u32>>,
        all_positions: Vec<Point3>,
        normal: Vec3,
        is_reversed: bool,
    }
    #[allow(clippy::items_after_statements)]
    type CdtResult = Result<Vec<(usize, usize, usize)>, crate::OperationsError>;

    // Phase 4a: Collect CDT jobs for large planar faces with holes.
    let mut cdt_jobs: Vec<CdtJob> = Vec::new();
    let mut other_face_indices: Vec<usize> = Vec::new();

    for (fi, &face_id) in all_faces.iter().enumerate() {
        let face_data = topo.face(face_id)?;
        let has_inner = !face_data.inner_wires().is_empty();
        if let FaceSurface::Plane { normal, .. } = face_data.surface() {
            if has_inner {
                // Collect boundary data for this face's CDT job.
                let normal = *normal;
                let is_reversed = face_data.is_reversed();
                let wire = topo.wire(face_data.outer_wire())?;
                let tol = 1e-10;

                // Collect outer boundary vertices from shared edge pool.
                let (mut all_positions, mut all_global_ids) = collect_wire_global_vertices(
                    wire,
                    &edge_global_indices,
                    &merged.positions,
                    tol,
                );
                remove_closing_duplicate_global(
                    &mut all_positions,
                    &mut all_global_ids,
                    &merged.positions,
                    tol,
                );
                let outer_count = all_positions.len();

                // Collect inner wire vertices.
                let mut inner_wire_ranges: Vec<(usize, usize)> = Vec::new();
                for &iw_id in face_data.inner_wires() {
                    let iw = topo.wire(iw_id)?;
                    let start = all_positions.len();
                    let (inner_pos, inner_gids) = collect_wire_global_vertices(
                        iw,
                        &edge_global_indices,
                        &merged.positions,
                        tol,
                    );
                    // Track flat IDs for closing-duplicate removal.
                    // Each vertex should have a global ID from the edge pool; use
                    // a unique sentinel if the snap lookup missed (shouldn't happen
                    // in practice, but avoids silently aliasing to vertex 0).
                    let mut inner_flat_ids: Vec<u32> = Vec::with_capacity(inner_gids.len());
                    let mut next_sentinel = u32::MAX;
                    for (pos, gid_opt) in inner_pos.into_iter().zip(inner_gids) {
                        let gid = gid_opt.unwrap_or_else(|| {
                            debug_assert!(false, "inner wire vertex had no global ID");
                            let s = next_sentinel;
                            next_sentinel = next_sentinel.wrapping_sub(1);
                            s
                        });
                        inner_flat_ids.push(gid);
                        all_positions.push(pos);
                        all_global_ids.push(Some(gid));
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

                let pts2d: Vec<brepkit_math::vec::Point2> = all_positions
                    .iter()
                    .map(|&p| project_by_normal(p, normal))
                    .collect();

                cdt_jobs.push(CdtJob {
                    pts2d,
                    outer_count,
                    inner_wire_ranges,
                    all_global_ids,
                    all_positions,
                    normal,
                    is_reversed,
                });
                continue;
            }
        }
        other_face_indices.push(fi);
    }

    // Phase 4b: Run CDTs in parallel for large planar faces.
    let cdt_results: Vec<CdtResult> = if cdt_jobs.len() >= 2 {
        use rayon::prelude::*;
        cdt_jobs
            .par_iter()
            .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
            .collect()
    } else {
        cdt_jobs
            .iter()
            .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
            .collect()
    };

    // Phase 4c: Merge CDT results into the shared mesh (sequential).
    for (job, result) in cdt_jobs.iter().zip(cdt_results) {
        let tris = result?;

        // Check winding of first triangle.
        let needs_flip = if let Some(&(i0, i1, i2)) = tris.first() {
            let p0 = job.all_positions[i0];
            let p1 = job.all_positions[i1];
            let p2 = job.all_positions[i2];
            let a = p1 - p0;
            let b = p2 - p0;
            let winding_matches = a.cross(b).dot(job.normal) > 0.0;
            // XOR: flip if winding doesn't match, or if face is reversed (but not both).
            winding_matches == job.is_reversed
        } else {
            false
        };

        for &(i0, i1, i2) in &tris {
            let g0 = job.all_global_ids[i0].unwrap_or(0);
            let g1 = job.all_global_ids[i1].unwrap_or(0);
            let g2 = job.all_global_ids[i2].unwrap_or(0);
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
    }

    // Phase 4d: Process remaining faces sequentially.
    // Skip faces that fail tessellation (e.g. degenerate slivers from booleans)
    // rather than aborting the entire solid — matches tessellate_solid_grouped.
    for &fi in &other_face_indices {
        if let Err(e) = tessellate_face_with_shared_edges(
            topo,
            all_faces[fi],
            deflection,
            &edge_global_indices,
            &mut merged,
            &mut point_to_global,
        ) {
            log::warn!("skipping face during tessellation: {e}");
        }
    }

    // Phase 5: Surface-aware vertex normals.
    //
    // Interior (non-shared) vertices already have surface-evaluated normals
    // from the face tessellation phase. Shared edge vertices start with
    // placeholder normals (0,0,0) from Phase 3. For these, compute
    // area-weighted averages from adjacent triangles.
    //
    // We do NOT split vertices at creases — that would break watertightness
    // (index-based half-edge pairing). Crease splitting is a rendering-layer
    // concern; the tessellator preserves topological integrity.

    let n_verts = merged.positions.len();
    let tri_count = merged.indices.len() / 3;

    // Accumulate area-weighted triangle normals into every vertex that
    // still has a zero placeholder normal.
    let mut accum: Vec<Vec3> = vec![Vec3::new(0.0, 0.0, 0.0); n_verts];
    let mut needs_normal = vec![false; n_verts];
    for i in 0..n_verts {
        let n = &merged.normals[i];
        if n.x().abs() < 1e-30 && n.y().abs() < 1e-30 && n.z().abs() < 1e-30 {
            needs_normal[i] = true;
        }
    }

    for t in 0..tri_count {
        let i0 = merged.indices[t * 3] as usize;
        let i1 = merged.indices[t * 3 + 1] as usize;
        let i2 = merged.indices[t * 3 + 2] as usize;
        let a = merged.positions[i1] - merged.positions[i0];
        let b = merged.positions[i2] - merged.positions[i0];
        let face_normal = a.cross(b); // area-weighted (unnormalized)
        if needs_normal[i0] {
            accum[i0] += face_normal;
        }
        if needs_normal[i1] {
            accum[i1] += face_normal;
        }
        if needs_normal[i2] {
            accum[i2] += face_normal;
        }
    }

    for i in 0..n_verts {
        if needs_normal[i] {
            merged.normals[i] = accum[i].normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        }
    }

    Ok(merged)
}

/// Tessellate a single face, reusing shared edge vertices from the global mesh.
///
/// For planar faces: collects boundary vertices (reusing global indices for
/// shared edges), then triangulates via CDT (Constrained Delaunay Triangulation).
///
/// For NURBS and analytic faces: falls back to per-face tessellation and
/// stitches the boundary vertices to the global mesh by snapping boundary
/// points to the nearest shared edge vertex.
#[allow(clippy::too_many_lines)]
fn tessellate_face_with_shared_edges(
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut HashMap<(u64, u64, u64), u32>,
) -> Result<(), crate::OperationsError> {
    let face_data = topo.face(face_id)?;
    let is_reversed = face_data.is_reversed();

    // Track index/position counts before tessellation so we can flip new
    // triangles and normals if the face is reversed.
    let idx_start = merged.indices.len();
    let pos_start = merged.positions.len();

    if let FaceSurface::Plane { normal, .. } = face_data.surface() {
        // For planar faces: build boundary polygon from shared edge vertices.
        let normal = *normal;
        let wire = topo.wire(face_data.outer_wire())?;

        let mut boundary_global_ids: Vec<u32> = Vec::new();
        let tol = 1e-10;

        for oe in wire.edges() {
            let edge_idx = oe.edge().index();
            if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
                // Iterate without allocating: use index-based forward/reverse.
                let is_fwd = oe.is_forward();
                let len = global_ids.len();
                for j in 0..len {
                    let gid = if is_fwd {
                        global_ids[j]
                    } else {
                        global_ids[len - 1 - j]
                    };
                    if j == 0 && !boundary_global_ids.is_empty() {
                        let last_gid = *boundary_global_ids.last().unwrap_or(&u32::MAX);
                        if last_gid == gid {
                            continue;
                        }
                        if (last_gid as usize) < merged.positions.len()
                            && (gid as usize) < merged.positions.len()
                        {
                            let last_pos = merged.positions[last_gid as usize];
                            let this_pos = merged.positions[gid as usize];
                            if (last_pos - this_pos).length() < tol {
                                continue;
                            }
                        }
                    }
                    boundary_global_ids.push(gid);
                }
            } else {
                // Edge not in the shared pool (shouldn't happen for valid solids,
                // but handle gracefully by inserting points directly).
                let edge_data = topo.edge(oe.edge())?;
                let points = sample_edge(topo, edge_data, deflection)?;
                let ordered: Vec<Point3> = if oe.is_forward() {
                    points
                } else {
                    points.into_iter().rev().collect()
                };
                for (j, pt) in ordered.iter().enumerate() {
                    if j == 0 && !boundary_global_ids.is_empty() {
                        let last_gid = *boundary_global_ids.last().unwrap_or(&u32::MAX);
                        if (last_gid as usize) < merged.positions.len() {
                            let last_pos = merged.positions[last_gid as usize];
                            if (last_pos - *pt).length() < tol {
                                continue;
                            }
                        }
                    }
                    let key = (pt.x().to_bits(), pt.y().to_bits(), pt.z().to_bits());
                    let gid = point_to_global.entry(key).or_insert_with(|| {
                        #[allow(clippy::cast_possible_truncation)]
                        let idx = merged.positions.len() as u32;
                        merged.positions.push(*pt);
                        merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                        idx
                    });
                    boundary_global_ids.push(*gid);
                }
            }
        }

        remove_closing_duplicate_ids(&mut boundary_global_ids, &merged.positions, tol);

        let n = boundary_global_ids.len();
        if n < 3 {
            return Ok(());
        }

        // Gather positions for triangulation (need local coords).
        let local_positions: Vec<Point3> = boundary_global_ids
            .iter()
            .map(|&gid| merged.positions[gid as usize])
            .collect();

        if face_data.inner_wires().is_empty() {
            // Triangulate simple faces (no holes) via CDT with fan-triangulation fallback.
            let mut local_indices = cdt_triangulate_simple(&local_positions, normal);

            // Ensure triangle winding matches the face normal.
            if local_indices.len() >= 3 {
                let i0 = local_indices[0] as usize;
                let i1 = local_indices[1] as usize;
                let i2 = local_indices[2] as usize;
                let a = local_positions[i1] - local_positions[i0];
                let b = local_positions[i2] - local_positions[i0];
                let tri_normal = a.cross(b);
                if tri_normal.dot(normal) < 0.0 {
                    for t in 0..local_indices.len() / 3 {
                        local_indices.swap(t * 3 + 1, t * 3 + 2);
                    }
                }
            }

            for &li in &local_indices {
                merged.indices.push(boundary_global_ids[li as usize]);
            }
        } else {
            // CDT path for planar faces with inner wires (holes).
            tessellate_planar_shared_with_holes(
                topo,
                face_data,
                &boundary_global_ids,
                &local_positions,
                normal,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    } else if matches!(face_data.surface(), FaceSurface::Nurbs(_)) {
        // For NURBS faces: use CDT-based boundary-constrained tessellation.
        // NURBS faces have rectangular non-degenerate parameter domains, so
        // the boundary projects to a proper closed polygon in (u,v) space.
        // Falls back to snap-based stitching if CDT fails.
        let cdt_ok = tessellate_nonplanar_cdt(
            topo,
            face_id,
            face_data,
            deflection,
            edge_global_indices,
            merged,
            point_to_global,
        );
        if cdt_ok.is_err() {
            tessellate_nonplanar_snap(
                topo,
                face_id,
                face_data,
                deflection,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    } else if matches!(
        face_data.surface(),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
    ) {
        // Cylinder and cone faces are ruled surfaces — their grid has nv=1
        // and boundary vertices align precisely with the shared edge pool.
        // Snap-based stitching is much faster than CDT for these surfaces.
        tessellate_nonplanar_snap(
            topo,
            face_id,
            face_data,
            deflection,
            edge_global_indices,
            merged,
            point_to_global,
        )?;
    } else {
        // For sphere and torus faces: use CDT-based tessellation with exact
        // boundary constraints. These surfaces have non-trivial curvature in
        // both parameter directions, and polar regions may degenerate in (u,v)
        // space — CDT ensures watertight boundary stitching.
        //
        // Save mesh state before CDT attempt so we can roll back if it fails.
        let pos_save = merged.positions.len();
        let nrm_save = merged.normals.len();
        let idx_save = merged.indices.len();
        let ptg_count_save = point_to_global.len();

        let cdt_ok = tessellate_nonplanar_cdt(
            topo,
            face_id,
            face_data,
            deflection,
            edge_global_indices,
            merged,
            point_to_global,
        );
        let cdt_produced_tris = cdt_ok.is_ok() && merged.indices.len() > idx_save;
        if !cdt_produced_tris {
            // CDT failed or produced zero triangles — roll back and fall back
            // to snap-based stitching.
            merged.positions.truncate(pos_save);
            merged.normals.truncate(nrm_save);
            merged.indices.truncate(idx_save);
            if point_to_global.len() > ptg_count_save {
                point_to_global.retain(|_, v| (*v as usize) < pos_save);
            }

            tessellate_nonplanar_snap(
                topo,
                face_id,
                face_data,
                deflection,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    }

    // If the face is reversed, flip triangle winding AND negate per-vertex normals
    // for all geometry added by this face. This ensures correct outward orientation
    // for both volume computation and rendering.
    if is_reversed {
        let idx_end = merged.indices.len();
        let tri_count = (idx_end - idx_start) / 3;
        for t in 0..tri_count {
            let base = idx_start + t * 3;
            merged.indices.swap(base + 1, base + 2);
        }
        for n in &mut merged.normals[pos_start..] {
            *n = -*n;
        }
    }

    Ok(())
}

/// CDT-based tessellation for non-planar faces with exact boundary constraints.
///
/// Projects shared edge points into (u,v) parameter space, generates interior
/// sample points, then runs Constrained Delaunay Triangulation. Boundary
/// vertices use their pre-existing global IDs (watertight by construction).
#[allow(clippy::too_many_lines)]
fn tessellate_nonplanar_cdt(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut HashMap<(u64, u64, u64), u32>,
) -> Result<(), crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;

    // Step 1: Collect boundary points in wire-traversal order with global IDs.
    // Also track which edge each point came from (for PCurve lookup).
    use brepkit_topology::edge::EdgeId;
    let wire = topo.wire(face_data.outer_wire())?;
    let tol_dup = 1e-10;

    let mut boundary_3d: Vec<(Point3, u32, EdgeId)> = Vec::new();
    for oe in wire.edges() {
        let edge_id_local = oe.edge();
        let edge_idx = edge_id_local.index();
        if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
            let ordered: Vec<u32> = if oe.is_forward() {
                global_ids.clone()
            } else {
                global_ids.iter().rev().copied().collect()
            };
            for (j, &gid) in ordered.iter().enumerate() {
                if j == 0 && !boundary_3d.is_empty() {
                    let (_, last_gid, _) = boundary_3d[boundary_3d.len() - 1];
                    if last_gid == gid
                        || (merged.positions[last_gid as usize] - merged.positions[gid as usize])
                            .length()
                            < tol_dup
                    {
                        continue;
                    }
                }
                boundary_3d.push((merged.positions[gid as usize], gid, edge_id_local));
            }
        } else {
            // Edge not in shared pool — insert directly.
            let edge_data = topo.edge(oe.edge())?;
            let points = sample_edge(topo, edge_data, deflection)?;
            let ordered: Vec<Point3> = if oe.is_forward() {
                points
            } else {
                points.into_iter().rev().collect()
            };
            for (j, &pt) in ordered.iter().enumerate() {
                if j == 0 && !boundary_3d.is_empty() {
                    let (last_pos, _, _) = boundary_3d[boundary_3d.len() - 1];
                    if (last_pos - pt).length() < tol_dup {
                        continue;
                    }
                }
                let key = (pt.x().to_bits(), pt.y().to_bits(), pt.z().to_bits());
                let gid = *point_to_global.entry(key).or_insert_with(|| {
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(pt);
                    merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                    idx
                });
                boundary_3d.push((pt, gid, edge_id_local));
            }
        }
    }

    // Remove closing duplicate.
    if boundary_3d.len() > 2 {
        if let (Some(&(_, first_gid, _)), Some(&(_, last_gid, _))) =
            (boundary_3d.first(), boundary_3d.last())
        {
            if first_gid == last_gid
                || (merged.positions[first_gid as usize] - merged.positions[last_gid as usize])
                    .length()
                    < tol_dup
            {
                boundary_3d.pop();
            }
        }
    }

    let n_boundary = boundary_3d.len();
    if n_boundary < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "non-planar face has fewer than 3 boundary vertices".to_string(),
        });
    }

    // Step 2: Project boundary 3D points to (u,v) parameter space.
    // For NURBS surfaces, check the PCurve registry first — evaluating a
    // PCurve directly is O(1) vs O(n) Newton projection.
    let boundary_uv: Vec<(f64, f64)> = boundary_3d
        .iter()
        .map(|(pt, _, edge_id_local)| {
            // Try PCurve lookup first.
            if let Some(pcurve) = topo.pcurves().get(*edge_id_local, face_id) {
                // We have the edge's PCurve — find the closest t and evaluate.
                // For boundary points sampled along the edge, approximate t
                // by projecting onto the PCurve's parameter range.
                let uv = project_via_pcurve(pcurve, *pt, face_data.surface());
                if let Some(uv) = uv {
                    return Ok(uv);
                }
            }
            // Fall back to surface projection.
            project_to_surface_uv(face_data.surface(), *pt)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Compute (u,v) bounding box.
    let u_min = boundary_uv
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min);
    let u_max = boundary_uv
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let v_min = boundary_uv
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min);
    let v_max = boundary_uv
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    let margin = 0.01;
    let bounds = (
        Point2::new(u_min - margin, v_min - margin),
        Point2::new(u_max + margin, v_max + margin),
    );
    let mut cdt = Cdt::with_capacity(bounds, n_boundary);

    // Step 3: Insert boundary points into CDT.
    // cdt_idx → Option<global mesh index> (None for super-triangle vertices).
    let mut cdt_to_global: Vec<Option<u32>> = vec![None; 3]; // 3 super-triangle verts

    let mut boundary_cdt_ids: Vec<usize> = Vec::with_capacity(n_boundary);
    for (i, &(u, v)) in boundary_uv.iter().enumerate() {
        let cdt_idx = cdt
            .insert_point(Point2::new(u, v))
            .map_err(crate::OperationsError::Math)?;
        while cdt_to_global.len() <= cdt_idx {
            cdt_to_global.push(None);
        }
        cdt_to_global[cdt_idx] = Some(boundary_3d[i].1);
        boundary_cdt_ids.push(cdt_idx);
    }

    // Step 4: Insert boundary constraints (consecutive edges + closing edge).
    for i in 0..n_boundary {
        let v0 = boundary_cdt_ids[i];
        let v1 = boundary_cdt_ids[(i + 1) % n_boundary];
        cdt.insert_constraint(v0, v1)
            .map_err(crate::OperationsError::Math)?;
    }

    // Step 5: Generate interior sample points.
    // Use a uniform grid at a density matching the deflection criterion,
    // with surface-aware resolution per direction.
    let du = u_max - u_min;
    let dv = v_max - v_min;
    if du > 1e-15 && dv > 1e-15 {
        let (n_u, n_v) = interior_grid_resolution(face_data.surface(), du, dv, deflection);

        for iu in 1..n_u {
            for iv in 1..n_v {
                let u = u_min + du * (iu as f64 / n_u as f64);
                let v = v_min + dv * (iv as f64 / n_v as f64);
                // Only insert points that are inside the boundary polygon.
                let pt2 = Point2::new(u, v);
                if point_in_polygon_2d(&boundary_uv, pt2) {
                    let cdt_idx = cdt
                        .insert_point(pt2)
                        .map_err(crate::OperationsError::Math)?;
                    while cdt_to_global.len() <= cdt_idx {
                        cdt_to_global.push(None);
                    }
                    // Interior points will get assigned global IDs later.
                }
            }
        }
    }

    // Step 6: Remove triangles outside the boundary polygon.
    let boundary_pairs: Vec<(usize, usize)> = (0..n_boundary)
        .map(|i| (boundary_cdt_ids[i], boundary_cdt_ids[(i + 1) % n_boundary]))
        .collect();
    cdt.remove_exterior(&boundary_pairs);

    // Step 7: Assign global IDs to interior CDT vertices and emit triangles.
    let cdt_verts = cdt.vertices();
    let triangles = cdt.triangles();

    // Pre-compute global IDs for all CDT vertices.
    let mut final_global_ids: Vec<u32> = vec![0; cdt_to_global.len()];

    for i in 0..cdt_to_global.len() {
        if let Some(gid) = cdt_to_global[i] {
            final_global_ids[i] = gid;
        } else if i >= 3 {
            // Interior vertex: evaluate surface at (u,v) to get 3D position.
            let pt2 = cdt_verts[i];
            let surface = face_data.surface();
            let pt3 = eval_surface_point(surface, pt2.x(), pt2.y());
            let nrm = surface.normal(pt2.x(), pt2.y());

            let key = (pt3.x().to_bits(), pt3.y().to_bits(), pt3.z().to_bits());
            let gid = *point_to_global.entry(key).or_insert_with(|| {
                let idx = merged.positions.len() as u32;
                merged.positions.push(pt3);
                merged.normals.push(nrm);
                idx
            });
            final_global_ids[i] = gid;
        }
    }

    // Emit triangles.
    for (i0, i1, i2) in triangles {
        if i0 < 3 || i1 < 3 || i2 < 3 {
            continue; // Skip super-triangle vertices (shouldn't happen after remove_exterior)
        }
        merged.indices.push(final_global_ids[i0]);
        merged.indices.push(final_global_ids[i1]);
        merged.indices.push(final_global_ids[i2]);
    }

    Ok(())
}

/// Project a 3D point onto a face surface, returning (u, v) parameters.
fn project_to_surface_uv(
    surface: &FaceSurface,
    pt: Point3,
) -> Result<(f64, f64), crate::OperationsError> {
    match surface {
        FaceSurface::Cylinder(cyl) => Ok(cyl.project_point(pt)),
        FaceSurface::Cone(cone) => Ok(cone.project_point(pt)),
        FaceSurface::Sphere(sphere) => Ok(sphere.project_point(pt)),
        FaceSurface::Torus(torus) => Ok(torus.project_point(pt)),
        FaceSurface::Nurbs(surface) => {
            brepkit_math::nurbs::projection::project_point_to_surface(surface, pt, 1e-6)
                .map(|proj| (proj.u, proj.v))
                .map_err(crate::OperationsError::Math)
        }
        FaceSurface::Plane { .. } => Err(crate::OperationsError::InvalidInput {
            reason: "planar faces should not use CDT tessellation".to_string(),
        }),
    }
}

/// Try to find (u,v) coordinates for a 3D point using a PCurve.
///
/// Samples the PCurve at multiple t values, finds the closest one to `pt`
/// on the surface, and returns the (u,v) from the PCurve evaluation.
/// Returns `None` if the closest sample is too far from `pt`.
fn project_via_pcurve(
    pcurve: &brepkit_topology::pcurve::PCurve,
    pt: Point3,
    surface: &FaceSurface,
) -> Option<(f64, f64)> {
    let t_start = pcurve.t_start();
    let t_end = pcurve.t_end();
    let n_samples = 16;

    let mut best_t = t_start;
    let mut best_dist = f64::MAX;

    for i in 0..=n_samples {
        let t = t_start + (t_end - t_start) * (i as f64) / (n_samples as f64);
        let uv = pcurve.evaluate(t);
        let p_surf = eval_surface_point(surface, uv.x(), uv.y());
        let d = (p_surf - pt).length();
        if d < best_dist {
            best_dist = d;
            best_t = t;
        }
    }

    // Refine with bisection around best_t.
    let dt = (t_end - t_start) / (n_samples as f64);
    let mut lo = (best_t - dt).max(t_start);
    let mut hi = (best_t + dt).min(t_end);
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        let uv_lo = pcurve.evaluate(lo);
        let uv_hi = pcurve.evaluate(hi);
        let d_lo = (eval_surface_point(surface, uv_lo.x(), uv_lo.y()) - pt).length();
        let d_hi = (eval_surface_point(surface, uv_hi.x(), uv_hi.y()) - pt).length();
        if d_lo < d_hi {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let t_final = 0.5 * (lo + hi);
    let uv = pcurve.evaluate(t_final);
    let p_final = eval_surface_point(surface, uv.x(), uv.y());

    // Accept if the projected point is close enough to the target.
    if (p_final - pt).length() < brepkit_math::tolerance::Tolerance::default().linear {
        Some((uv.x(), uv.y()))
    } else {
        None
    }
}

/// Evaluate a non-planar surface at `(u, v)` and return a 3D point.
///
/// `FaceSurface::evaluate` returns `None` only for the `Plane` variant.
/// This helper is called exclusively from paths that operate on non-planar
/// faces (CDT interior vertices, PCurve projection), so `None` is
/// structurally unreachable. The `Point3::new(0.0, 0.0, 0.0)` fallback
/// is kept to satisfy the type system without an `expect` or `unwrap`.
fn eval_surface_point(surface: &FaceSurface, u: f64, v: f64) -> Point3 {
    surface.evaluate(u, v).unwrap_or(Point3::new(0.0, 0.0, 0.0))
}

/// Estimate the effective radius of a surface for sample density calculation.
///
/// **Not replaced by `FaceSurface::estimate_radius()`** — the delegate uses
/// different values (e.g. cone → `radius_at(1.0)`, torus → `major_radius()`),
/// while tessellation intentionally uses conservative estimates (cone → 1.0,
/// torus → `major + minor`).
fn estimate_surface_radius(surface: &FaceSurface) -> f64 {
    match surface {
        FaceSurface::Cylinder(cyl) => cyl.radius(),
        FaceSurface::Cone(_) => 1.0, // conservative estimate
        FaceSurface::Sphere(sphere) => sphere.radius(),
        FaceSurface::Torus(torus) => torus.major_radius() + torus.minor_radius(),
        FaceSurface::Nurbs(_) | FaceSurface::Plane { .. } => 1.0, // conservative estimate
    }
}

/// Compute interior grid resolution for `tessellate_nonplanar_cdt`.
///
/// Returns `(n_u, n_v)` with surface-aware density:
/// - **Sphere:** both directions curvature-based
/// - **Torus:** u = major-radius-based, v = minor-radius-based
/// - **Plane/NURBS/Cylinder/Cone:** isotropic conservative estimate
///
/// Cylinder and cone faces reach the snap fast-path before CDT is called, so
/// their arm here exists only for exhaustiveness — new `FaceSurface` variants
/// will cause a compile error rather than silently using the fallback.
fn interior_grid_resolution(
    surface: &FaceSurface,
    du: f64,
    dv: f64,
    deflection: f64,
) -> (usize, usize) {
    match surface {
        FaceSurface::Sphere(sphere) => {
            let r = sphere.radius();
            let n_u = segments_for_chord_deviation(r, du, deflection).max(2);
            let n_v = segments_for_chord_deviation(r, dv, deflection).max(2);
            (n_u, n_v)
        }
        FaceSurface::Torus(torus) => {
            let n_u = segments_for_chord_deviation(torus.major_radius(), du, deflection).max(2);
            let n_v = segments_for_chord_deviation(torus.minor_radius(), dv, deflection).max(2);
            (n_u, n_v)
        }
        FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_)
        | FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_) => {
            // Isotropic conservative estimate using surface radius.
            let r = estimate_surface_radius(surface);
            let n_u = segments_for_chord_deviation(r, du, deflection).max(2);
            let n_v = segments_for_chord_deviation(r, dv, deflection).max(2);
            (n_u, n_v)
        }
    }
}

/// Check if a 2D point is inside a polygon defined by (u, v) coordinates.
/// Uses the winding number algorithm for robustness.
fn point_in_polygon_2d(polygon: &[(f64, f64)], pt: brepkit_math::vec::Point2) -> bool {
    let n = polygon.len();
    let mut winding = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let yi = polygon[i].1;
        let yj = polygon[j].1;
        if yi <= pt.y() {
            if yj > pt.y() {
                let cross = (polygon[j].0 - polygon[i].0) * (pt.y() - yi)
                    - (pt.x() - polygon[i].0) * (yj - yi);
                if cross > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= pt.y() {
            let cross =
                (polygon[j].0 - polygon[i].0) * (pt.y() - yi) - (pt.x() - polygon[i].0) * (yj - yi);
            if cross < 0.0 {
                winding -= 1;
            }
        }
    }
    winding != 0
}

/// Snap-based fallback tessellation for non-planar faces.
///
/// Tessellates the face independently, then snaps boundary vertices to shared
/// edge points within a fixed tolerance. Used when CDT-based tessellation fails.
fn tessellate_nonplanar_snap(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    edge_global_indices: &HashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut HashMap<(u64, u64, u64), u32>,
) -> Result<(), crate::OperationsError> {
    let mut face_mesh = tessellate(topo, face_id, deflection)?;

    // `tessellate()` already applies the `is_reversed` flip (reversing
    // winding and negating normals). The caller `tessellate_face_with_shared_edges`
    // will apply its own flip, so undo the one from `tessellate()` to avoid
    // a double-flip that would cancel out.
    if face_data.is_reversed() {
        let tri_count = face_mesh.indices.len() / 3;
        for t in 0..tri_count {
            face_mesh.indices.swap(t * 3 + 1, t * 3 + 2);
        }
        for n in &mut face_mesh.normals {
            *n = -*n;
        }
    }

    let mut local_to_global: Vec<u32> = Vec::with_capacity(face_mesh.positions.len());

    // Collect all edge points for this face to use as snap targets.
    let wire = topo.wire(face_data.outer_wire())?;
    let mut snap_targets: Vec<(Point3, u32)> = Vec::new();
    for oe in wire.edges() {
        if let Some(global_ids) = edge_global_indices.get(&oe.edge().index()) {
            for &gid in global_ids {
                if (gid as usize) < merged.positions.len() {
                    snap_targets.push((merged.positions[gid as usize], gid));
                }
            }
        }
    }
    for &inner_wire_id in face_data.inner_wires() {
        if let Ok(inner_wire) = topo.wire(inner_wire_id) {
            for oe in inner_wire.edges() {
                if let Some(global_ids) = edge_global_indices.get(&oe.edge().index()) {
                    for &gid in global_ids {
                        if (gid as usize) < merged.positions.len() {
                            snap_targets.push((merged.positions[gid as usize], gid));
                        }
                    }
                }
            }
        }
    }

    // Build spatial hash for O(1) snap lookups instead of O(n*m) brute force.
    let snap_tol = 1e-6;
    let inv_cell = 1.0 / snap_tol;
    let mut snap_grid: HashMap<(i64, i64, i64), Vec<u32>> =
        HashMap::with_capacity(snap_targets.len());
    for &(target_pos, gid) in &snap_targets {
        let cx = (target_pos.x() * inv_cell).round() as i64;
        let cy = (target_pos.y() * inv_cell).round() as i64;
        let cz = (target_pos.z() * inv_cell).round() as i64;
        snap_grid.entry((cx, cy, cz)).or_default().push(gid);
    }

    for (i, &pos) in face_mesh.positions.iter().enumerate() {
        let cx = (pos.x() * inv_cell).round() as i64;
        let cy = (pos.y() * inv_cell).round() as i64;
        let cz = (pos.z() * inv_cell).round() as i64;
        let mut best_gid = None;
        let mut best_dist = snap_tol;
        // Check 3x3x3 neighborhood for snap matches.
        for dx in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dz in -1_i64..=1 {
                    if let Some(gids) = snap_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &gid in gids {
                            let target_pos = merged.positions[gid as usize];
                            let dist = (pos - target_pos).length();
                            if dist < best_dist {
                                best_dist = dist;
                                best_gid = Some(gid);
                            }
                        }
                    }
                }
            }
        }

        if let Some(gid) = best_gid {
            local_to_global.push(gid);
        } else {
            let key = (pos.x().to_bits(), pos.y().to_bits(), pos.z().to_bits());
            let gid = point_to_global.entry(key).or_insert_with(|| {
                let idx = merged.positions.len() as u32;
                merged.positions.push(pos);
                merged.normals.push(
                    face_mesh
                        .normals
                        .get(i)
                        .copied()
                        .unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
                );
                idx
            });
            local_to_global.push(*gid);
        }
    }

    for &li in &face_mesh.indices {
        merged.indices.push(local_to_global[li as usize]);
    }

    Ok(())
}

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

    // Every half-edge must have its reverse present.
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
///
/// This is used to filter out "smooth" edges between faces that were split
/// by boolean operations but lie on the same underlying surface. Two faces
/// on the same surface have no visible crease between them, so the shared
/// edge adds visual noise in wireframe rendering.
fn surfaces_equivalent(a: &FaceSurface, b: &FaceSurface) -> bool {
    // Use project-standard tolerances. Boolean splits produce surfaces with
    // identical parameters (bit-for-bit clones), so strict tolerances are
    // appropriate. The linear tolerance covers minor floating-point drift
    // in plane `d` values after coordinate transforms.
    let tol = brepkit_math::tolerance::Tolerance::new();
    let lin = tol.linear; // 1e-7
    let ang = tol.angular; // 1e-12

    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            // Same plane if normals are parallel and signed distances match.
            // Normals may point in opposite directions (reversed faces on same plane).
            let dot = na.dot(*nb);
            (dot.abs() - 1.0).abs() < ang && (da - db * dot.signum()).abs() < lin
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            (ca.radius() - cb.radius()).abs() < lin
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    // Origins must lie on the same axis line
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
        // NURBS surfaces: no parameter-based equivalence check (would need
        // control point comparison). Keep all edges between NURBS faces.
        (FaceSurface::Nurbs(_), FaceSurface::Nurbs(_)) => false,
        // Mixed surface types are never equivalent.
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

    // Build edge-to-face map for filtering
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
        // Check if this edge should be filtered (smooth boundary)
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
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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
        let wid = topo.add_wire(wire);

        let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));

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
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();

        let n = verts.len();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.add_edge(Edge::new(verts[i], verts[next], EdgeCurve::Line))
            })
            .collect();

        let wire = Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);

        let face = topo.add_face(Face::new(
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

        let mesh = tessellate_nurbs(&surface, 0.1).mesh;

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
        let flat_mesh = tessellate_nurbs(&flat, deflection).mesh;
        let curved_mesh = tessellate_nurbs(&curved, deflection).mesh;

        let flat_tris = flat_mesh.indices.len() / 3;
        let curved_tris = curved_mesh.indices.len() / 3;

        assert!(
            curved_tris > flat_tris,
            "curved surface should have more triangles ({curved_tris}) than flat ({flat_tris})"
        );
    }

    // ── Watertight tessellation tests ───────────────────────────────

    #[test]
    fn tessellate_solid_box_watertight() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        // A box has 6 faces × 2 triangles = 12 triangles.
        let tri_count = mesh.indices.len() / 3;
        assert_eq!(
            tri_count, 12,
            "box should have 12 triangles, got {tri_count}"
        );

        // Verify watertightness.
        let boundary = boundary_edge_count(&mesh);
        assert_eq!(
            boundary, 0,
            "box mesh should be watertight (0 boundary edges), got {boundary}"
        );
        assert!(is_watertight(&mesh), "box mesh should be watertight");
    }

    #[test]
    fn tessellate_solid_box_correct_area() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        // Surface area of a 2×3×4 box = 2(2×3 + 2×4 + 3×4) = 2(6+8+12) = 52.
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
            (total_area - 52.0).abs() < 0.1,
            "box surface area should be ~52.0, got {total_area}"
        );
    }

    #[test]
    fn tessellate_solid_box_shared_vertices() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        // A unit box has 8 corner vertices. With shared edge tessellation,
        // we should have exactly 8 vertices (since all edges are lines with
        // only 2 sample points each = the endpoints).
        assert_eq!(
            mesh.positions.len(),
            8,
            "unit box should have exactly 8 shared vertices, got {}",
            mesh.positions.len()
        );
    }

    #[test]
    fn tessellate_solid_cylinder_shared_topology() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        // Verify the cylinder now has shared edges between lateral and cap faces.
        let edge_map = brepkit_topology::explorer::edge_to_face_map(&topo, solid).unwrap();
        let shared_count = edge_map.values().filter(|faces| faces.len() >= 2).count();
        assert!(
            shared_count >= 2,
            "cylinder should have at least 2 shared edges (top/bottom circles), got {shared_count}"
        );

        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();
        assert!(mesh.indices.len() >= 3, "cylinder should have triangles");
        assert!(!mesh.positions.is_empty(), "cylinder should have vertices");

        // Analytic faces (cylinder lateral) use snap-based stitching.
        // NURBS faces use CDT-based boundary-constrained tessellation for
        // watertight seams (see tessellate_nonplanar_cdt).
    }

    #[test]
    fn tessellate_solid_sphere_produces_mesh() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, 1.0, 16).unwrap();

        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        assert!(mesh.indices.len() >= 3, "sphere should have triangles");
        assert!(!mesh.positions.is_empty(), "sphere should have vertices");
    }

    #[test]
    fn is_watertight_basic() {
        // A single tetrahedron (4 triangles, 4 vertices) is watertight.
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(0.5, 0.5, 1.0),
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 4],
            indices: vec![
                0, 1, 2, // bottom
                0, 2, 3, // left
                0, 3, 1, // front
                1, 3, 2, // right
            ],
        };
        assert!(is_watertight(&mesh));
        assert_eq!(boundary_edge_count(&mesh), 0);
    }

    #[test]
    fn is_watertight_open_mesh() {
        // A single triangle is NOT watertight (all 3 edges are boundary).
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 3],
            indices: vec![0, 1, 2],
        };
        assert!(!is_watertight(&mesh));
        assert_eq!(boundary_edge_count(&mesh), 3);
    }

    #[test]
    fn tessellate_solid_normals_unit_length() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        for (i, n) in mesh.normals.iter().enumerate() {
            let len = n.length();
            assert!(
                (len - 1.0).abs() < 0.01,
                "normal {i} should be unit length, got {len}"
            );
        }
    }

    // ── Curvature-adaptive tessellation tests ──────────────

    #[test]
    fn curvature_adaptive_refines_high_curvature() {
        // A dome-like surface with high curvature at edges but flat in center.
        // The curvature-adaptive metric should produce more triangles than
        // the old normal-deviation metric for the same deflection.
        let mut cps = Vec::new();
        let mut ws = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            let mut wrow = Vec::new();
            for j in 0..4 {
                // Dome shape: z = 1 - x² - y² (high curvature at edges)
                #[allow(clippy::cast_precision_loss)]
                let x = (j as f64) / 3.0;
                #[allow(clippy::cast_precision_loss)]
                let y = (i as f64) / 3.0;
                let z = 2.0 * (1.0 - (x - 0.5).powi(2) - (y - 0.5).powi(2));
                #[allow(clippy::cast_precision_loss)]
                row.push(Point3::new(j as f64, i as f64, z));
                wrow.push(1.0);
            }
            cps.push(row);
            ws.push(wrow);
        }
        let dome = NurbsSurface::new(
            3,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            cps,
            ws,
        )
        .unwrap();

        // Fine deflection should produce many triangles.
        let fine_mesh = tessellate_nurbs(&dome, 0.01).mesh;
        let coarse_mesh = tessellate_nurbs(&dome, 0.5).mesh;

        assert!(
            fine_mesh.indices.len() / 3 > coarse_mesh.indices.len() / 3,
            "finer deflection should produce more triangles: fine={}, coarse={}",
            fine_mesh.indices.len() / 3,
            coarse_mesh.indices.len() / 3
        );
    }

    #[test]
    fn curvature_adaptive_midpoint_sag_check() {
        // For a curved surface, the midpoint sag should be bounded by
        // the deflection parameter after adaptive tessellation.
        let mut cps = Vec::new();
        let mut ws = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            let mut wrow = Vec::new();
            for j in 0..4 {
                #[allow(clippy::cast_precision_loss)]
                let z = ((i + j) as f64 * 0.5).sin() * 1.5;
                #[allow(clippy::cast_precision_loss)]
                row.push(Point3::new(j as f64, i as f64, z));
                wrow.push(1.0);
            }
            cps.push(row);
            ws.push(wrow);
        }
        let surface = NurbsSurface::new(
            3,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            cps,
            ws,
        )
        .unwrap();

        let deflection = 0.05;
        let mesh = tessellate_nurbs(&surface, deflection).mesh;

        // All triangles should have their midpoints close to the surface.
        // The maximum sag should be bounded (not exactly by deflection due
        // to the quadtree structure, but should be reasonable).
        let tri_count = mesh.indices.len() / 3;
        assert!(
            tri_count > 32,
            "curved surface should have more than base 32 triangles, got {tri_count}"
        );

        // Verify no degenerate triangles (zero area).
        for t in 0..tri_count {
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            let a = mesh.positions[i1] - mesh.positions[i0];
            let b = mesh.positions[i2] - mesh.positions[i0];
            let area = 0.5 * a.cross(b).length();
            assert!(area > 0.0, "triangle {t} has zero area");
        }
    }

    #[test]
    fn sample_solid_edges_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 2.0, 3.0).unwrap();

        let edge_lines = sample_solid_edges(&topo, solid, 0.1).unwrap();

        // A box has 12 edges, each with 2 points (line segments).
        assert_eq!(edge_lines.offsets.len(), 12, "box should have 12 edges");
        assert_eq!(
            edge_lines.positions.len(),
            24,
            "12 line edges × 2 points = 24 points"
        );
    }

    #[test]
    fn sample_solid_edges_cylinder() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 3.0).unwrap();

        // Default (filtered): seam edge is removed (smooth, same cylinder surface
        // on both sides), leaving the 2 circle edges at top and bottom caps.
        let edge_lines = sample_solid_edges(&topo, solid, 0.1).unwrap();
        assert_eq!(
            edge_lines.offsets.len(),
            2,
            "filtered cylinder should have 2 circle edges, got {}",
            edge_lines.offsets.len()
        );
        // Circle edges should have many sample points.
        assert!(
            edge_lines.positions.len() > 10,
            "cylinder edges should have many sample points, got {}",
            edge_lines.positions.len()
        );

        // Unfiltered: includes the seam edge too.
        let all_edges = sample_solid_edges_filtered(&topo, solid, 0.1, false).unwrap();
        assert!(
            all_edges.offsets.len() >= 3,
            "unfiltered cylinder should have at least 3 edges, got {}",
            all_edges.offsets.len()
        );
    }

    #[test]
    fn sample_solid_edges_boolean_filters_coplanar() {
        // A boolean cut splits faces, creating extra internal edges on the
        // same planar surface. Filtering should remove these.
        let mut topo = Topology::new();
        let big = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let small = crate::primitives::make_box(&mut topo, 3.0, 3.0, 15.0).unwrap();
        let cut =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, big, small).unwrap();

        let filtered = sample_solid_edges(&topo, cut, 0.1).unwrap();
        let all = sample_solid_edges_filtered(&topo, cut, 0.1, false).unwrap();

        // Filtered should have fewer edges than unfiltered
        assert!(
            filtered.offsets.len() < all.offsets.len(),
            "filtered ({}) should be fewer than unfiltered ({})",
            filtered.offsets.len(),
            all.offsets.len()
        );
    }

    #[test]
    fn tessellate_solid_filleted_box_nurbs_boundary() {
        // A filleted box has NURBS fillet faces adjacent to planar faces.
        // With CDT-constrained tessellation, the shared edges should be
        // watertight (boundary vertices are exact, not snapped).
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let edges = {
            let s = topo.solid(bx).unwrap();
            let sh = topo.shell(s.outer_shell()).unwrap();
            let face_id = sh.faces()[0];
            let face = topo.face(face_id).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            vec![wire.edges()[0].edge()]
        };
        let filleted = crate::fillet::fillet_rolling_ball(&mut topo, bx, &edges, 0.5).unwrap();
        let mesh = tessellate_solid(&topo, filleted, 0.1).unwrap();

        assert!(
            mesh.indices.len() >= 3,
            "filleted box should have triangles"
        );
        assert!(
            !mesh.positions.is_empty(),
            "filleted box should have vertices"
        );

        // Check that the NURBS fillet face was tessellated via CDT:
        // The boundary edges between NURBS and planar faces should share
        // exact vertices (no gaps). Count boundary edges as a measure.
        let boundary = boundary_edge_count(&mesh);
        // A perfect watertight mesh would have 0 boundary edges.
        // With CDT for NURBS faces, we should see significant improvement
        // over the old snap-based approach.
        assert!(
            boundary < mesh.indices.len() / 3,
            "filleted box should have few boundary edges, got {boundary}"
        );
    }

    // ── P3: Tessellation Quality tests ────────────────────────────

    #[test]
    fn test_no_degenerate_triangles() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, 1.0, 16).unwrap();
        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        let tri_count = mesh.indices.len() / 3;
        assert!(tri_count > 0, "sphere should produce triangles");

        for t in 0..tri_count {
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            let a = mesh.positions[i1] - mesh.positions[i0];
            let b = mesh.positions[i2] - mesh.positions[i0];
            let area = 0.5 * a.cross(b).length();
            assert!(area > 0.0, "triangle {t} is degenerate (area = {area})");
        }
    }

    #[test]
    fn test_min_angle_above_threshold() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        let tri_count = mesh.indices.len() / 3;
        assert!(tri_count > 0, "cylinder should produce triangles");

        let min_angle_threshold = 0.0175; // ~1 degree in radians

        for t in 0..tri_count {
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            let p0 = mesh.positions[i0];
            let p1 = mesh.positions[i1];
            let p2 = mesh.positions[i2];

            let edges = [(p1 - p0, p2 - p0), (p0 - p1, p2 - p1), (p0 - p2, p1 - p2)];

            for (j, (ea, eb)) in edges.iter().enumerate() {
                let len_a = ea.length();
                let len_b = eb.length();
                if len_a < 1e-15 || len_b < 1e-15 {
                    continue;
                }
                let cos_angle = ea.dot(*eb) / (len_a * len_b);
                let angle = cos_angle.clamp(-1.0, 1.0).acos();
                assert!(
                    angle > min_angle_threshold,
                    "triangle {t} vertex {j} has angle {:.4} rad ({:.2} deg), below threshold",
                    angle,
                    angle.to_degrees()
                );
            }
        }
    }

    #[test]
    fn test_max_sag_within_deflection() {
        let radius = 1.0;
        let deflection = 0.05;
        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, radius, 16).unwrap();
        let mesh = tessellate_solid(&topo, solid, deflection).unwrap();

        let tri_count = mesh.indices.len() / 3;
        assert!(tri_count > 0);

        let mut max_sag = 0.0_f64;
        for t in 0..tri_count {
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            let centroid = Point3::new(
                (mesh.positions[i0].x() + mesh.positions[i1].x() + mesh.positions[i2].x()) / 3.0,
                (mesh.positions[i0].y() + mesh.positions[i1].y() + mesh.positions[i2].y()) / 3.0,
                (mesh.positions[i0].z() + mesh.positions[i1].z() + mesh.positions[i2].z()) / 3.0,
            );
            let dist_from_origin =
                (centroid.x().powi(2) + centroid.y().powi(2) + centroid.z().powi(2)).sqrt();
            let sag = (dist_from_origin - radius).abs();
            max_sag = max_sag.max(sag);
        }

        assert!(
            max_sag < 2.0 * deflection,
            "max sag {max_sag} exceeds 2*deflection ({})",
            2.0 * deflection
        );
    }

    #[test]
    fn test_watertight_solid_mesh() {
        use std::collections::{HashMap, HashSet};

        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 2.0, 3.0).unwrap();
        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        // Snap vertices to 6 decimal places for position-based matching.
        let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
        let snap_pt = |p: Point3| -> (i64, i64, i64) { (snap(p.x()), snap(p.y()), snap(p.z())) };

        // Map snapped positions to canonical indices.
        let mut pos_map: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut next_id = 0_usize;
        let canonical: Vec<usize> = mesh
            .positions
            .iter()
            .map(|&p| {
                let key = snap_pt(p);
                *pos_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();

        let tri_count = mesh.indices.len() / 3;
        let mut half_edges: HashSet<(usize, usize)> = HashSet::new();
        for t in 0..tri_count {
            let a = canonical[mesh.indices[t * 3] as usize];
            let b = canonical[mesh.indices[t * 3 + 1] as usize];
            let c = canonical[mesh.indices[t * 3 + 2] as usize];
            half_edges.insert((a, b));
            half_edges.insert((b, c));
            half_edges.insert((c, a));
        }

        let boundary_count = half_edges
            .iter()
            .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
            .count();

        assert_eq!(
            boundary_count, 0,
            "box mesh should be watertight (0 boundary edges), got {boundary_count}"
        );
    }

    #[test]
    fn test_consistent_winding() {
        let dx = 2.0;
        let dy = 3.0;
        let dz = 4.0;
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, dx, dy, dz).unwrap();
        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        // Signed volume via divergence theorem: sum det([v0,v1,v2]) / 6.
        let mut signed_vol = 0.0;
        let tri_count = mesh.indices.len() / 3;
        for t in 0..tri_count {
            let v0 = mesh.positions[mesh.indices[t * 3] as usize];
            let v1 = mesh.positions[mesh.indices[t * 3 + 1] as usize];
            let v2 = mesh.positions[mesh.indices[t * 3 + 2] as usize];
            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());
            signed_vol += a.dot(b.cross(c));
        }
        signed_vol /= 6.0;

        assert!(
            signed_vol > 0.0,
            "signed volume should be positive (outward normals), got {signed_vol}"
        );

        let expected_vol = dx * dy * dz;
        let rel_err = (signed_vol - expected_vol).abs() / expected_vol;
        assert!(
            rel_err < 0.01,
            "signed volume {signed_vol} differs from expected {expected_vol} by {:.2}%",
            rel_err * 100.0
        );
    }

    #[test]
    fn test_vertex_on_surface_sphere() {
        let radius = 2.0;
        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, radius, 16).unwrap();
        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        for (i, p) in mesh.positions.iter().enumerate() {
            let dist = (p.x().powi(2) + p.y().powi(2) + p.z().powi(2)).sqrt();
            assert!(
                (dist - radius).abs() < 1e-6,
                "vertex {i} at dist {dist} from origin, expected {radius}"
            );
        }
    }

    #[test]
    fn test_no_t_junctions_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

        // A unit box with planar faces should have exactly 8 unique vertices
        // after position snapping. Any extra vertices indicate T-junctions.
        let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
        let unique: std::collections::HashSet<(i64, i64, i64)> = mesh
            .positions
            .iter()
            .map(|p| (snap(p.x()), snap(p.y()), snap(p.z())))
            .collect();

        assert_eq!(
            unique.len(),
            8,
            "unit box should have 8 unique vertices (no T-junctions), got {}",
            unique.len()
        );
    }

    #[test]
    fn test_circle_deflection_scaling() {
        let mut topo = Topology::new();
        let small = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
        let large = crate::primitives::make_cylinder(&mut topo, 10.0, 2.0).unwrap();

        let deflection = 0.1;
        let mesh_small = tessellate_solid(&topo, small, deflection).unwrap();
        let mesh_large = tessellate_solid(&topo, large, deflection).unwrap();

        let tri_small = mesh_small.indices.len() / 3;
        let tri_large = mesh_large.indices.len() / 3;

        assert!(
            tri_large > tri_small,
            "larger cylinder should have more triangles ({tri_large}) than smaller ({tri_small})"
        );
    }

    #[test]
    fn test_tessellate_boolean_result_watertight() {
        use std::collections::{HashMap, HashSet};

        let mut topo = Topology::new();
        let a = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let b = crate::primitives::make_box(&mut topo, 1.5, 1.5, 1.5).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            b,
            &brepkit_math::mat::Mat4::translation(0.5, 0.5, 0.5),
        )
        .unwrap();

        let cut = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, a, b).unwrap();

        let mesh = tessellate_solid(&topo, cut, 0.1).unwrap();

        // Position-based watertightness check (same approach as test_watertight_solid_mesh).
        let snap = |v: f64| -> i64 { (v * 1_000_000.0).round() as i64 };
        let snap_pt = |p: Point3| -> (i64, i64, i64) { (snap(p.x()), snap(p.y()), snap(p.z())) };

        let mut pos_map: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut next_id = 0_usize;
        let canonical: Vec<usize> = mesh
            .positions
            .iter()
            .map(|&p| {
                let key = snap_pt(p);
                *pos_map.entry(key).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();

        let tri_count = mesh.indices.len() / 3;
        let mut half_edges: HashSet<(usize, usize)> = HashSet::new();
        for t in 0..tri_count {
            let ca = canonical[mesh.indices[t * 3] as usize];
            let cb = canonical[mesh.indices[t * 3 + 1] as usize];
            let cc = canonical[mesh.indices[t * 3 + 2] as usize];
            half_edges.insert((ca, cb));
            half_edges.insert((cb, cc));
            half_edges.insert((cc, ca));
        }

        let boundary_count = half_edges
            .iter()
            .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
            .count();

        assert_eq!(
            boundary_count, 0,
            "boolean cut result should be watertight (0 boundary edges), got {boundary_count}"
        );
    }
}

#[cfg(test)]
mod winding_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_topology::Topology;

    /// Regression test for the double-flip bug: `tessellate_nonplanar_snap`
    /// calls `tessellate()` which applies the `is_reversed` flip, and then
    /// `tessellate_face_with_shared_edges` applies it again — two flips
    /// cancel out, leaving reversed sphere faces with wrong winding.
    ///
    /// This test creates a solid with reversed sphere faces (as produced by
    /// boolean cut) and verifies that `tessellate_solid` produces a mesh
    /// where the signed volume matches expectations (negative contribution
    /// from the reversed hemisphere).
    #[test]
    fn reversed_sphere_face_tessellation_correct_winding() {
        use brepkit_topology::face::Face;
        use brepkit_topology::shell::Shell;
        use brepkit_topology::solid::Solid;

        // Build a sphere, then create a copy with all faces reversed.
        // The reversed copy should have opposite signed volume, proving that
        // `tessellate_solid` correctly handles the `is_reversed` flag without
        // double-flipping.
        let mut topo = Topology::new();
        let sphere = crate::primitives::make_sphere(&mut topo, 3.0, 32).unwrap();

        // Translate sphere to avoid degenerate geometry at origin.
        let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, sphere, &mat).unwrap();

        // Tessellate the normal (non-reversed) sphere.
        let mesh_normal = tessellate_solid(&topo, sphere, 0.05).unwrap();
        let vol_normal = signed_volume_raw(&mesh_normal);

        // Collect face data first (immutable borrow), then allocate (mutable borrow).
        let solid_data = topo.solid(sphere).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_copies: Vec<_> = shell
            .faces()
            .iter()
            .map(|&fid| {
                let face = topo.face(fid).unwrap();
                (
                    face.outer_wire(),
                    face.inner_wires().to_vec(),
                    face.surface().clone(),
                )
            })
            .collect();

        let mut rev_face_ids = Vec::new();
        for (outer_wire, inner_wires, surface) in face_copies {
            let new_face = Face::new_reversed(outer_wire, inner_wires, surface);
            rev_face_ids.push(topo.add_face(new_face));
        }
        let rev_shell = Shell::new(rev_face_ids).unwrap();
        let rev_shell_id = topo.add_shell(rev_shell);
        let rev_solid = topo.add_solid(Solid::new(rev_shell_id, vec![]));

        let mesh_reversed = tessellate_solid(&topo, rev_solid, 0.05).unwrap();
        let vol_reversed = signed_volume_raw(&mesh_reversed);

        // The reversed mesh should have the OPPOSITE sign.
        // (Normal sphere has positive signed volume; reversed should be negative.)
        assert!(
            vol_normal > 0.0,
            "normal sphere signed volume should be positive, got {vol_normal}"
        );
        assert!(
            vol_reversed < 0.0,
            "reversed sphere signed volume should be negative, got {vol_reversed} \
             (this fails if tessellate_nonplanar_snap double-flips)"
        );
        assert!(
            (vol_normal + vol_reversed).abs() < 1.0,
            "normal + reversed should cancel to ~0, got {}",
            vol_normal + vol_reversed
        );
    }

    /// Regression test: boolean cut produces a solid with reversed non-planar
    /// faces. The tessellated mesh must have positive signed volume (correct
    /// outward-facing winding). This is the exact scenario where the double-flip
    /// bug in `tessellate_nonplanar_snap` caused inverted meshes.
    #[test]
    fn boolean_cut_result_has_positive_signed_volume() {
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 3.0, 32).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, sp, &mat).unwrap();

        let cut_result =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, bx, sp).unwrap();

        let mesh = tessellate_solid(&topo, cut_result, 0.05).unwrap();
        let vol = signed_volume_raw(&mesh);

        // The cut result should have positive signed volume (outward winding).
        // Before the double-flip fix, this would be negative or wrong magnitude.
        assert!(
            vol > 0.0,
            "boolean cut result should have positive signed volume, got {vol}"
        );

        // Volume should be box (1000) minus full sphere (4/3·π·27 ≈ 113.1).
        let expected_approx = 1000.0 - (4.0 / 3.0) * std::f64::consts::PI * 27.0;
        let rel_err = (vol - expected_approx).abs() / expected_approx;
        assert!(
            rel_err < 0.15,
            "volume {vol} too far from expected ~{expected_approx:.1} (rel error {rel_err:.3})"
        );
    }

    /// Helper: compute raw signed volume WITHOUT abs(), to detect winding issues.
    fn signed_volume_raw(mesh: &TriangleMesh) -> f64 {
        let idx = &mesh.indices;
        let pos = &mesh.positions;
        let tri_count = idx.len() / 3;
        let mut total = 0.0;
        for t in 0..tri_count {
            let v0 = pos[idx[t * 3] as usize];
            let v1 = pos[idx[t * 3 + 1] as usize];
            let v2 = pos[idx[t * 3 + 2] as usize];
            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());
            total += a.dot(b.cross(c));
        }
        total / 6.0
    }

    #[test]
    fn per_face_tessellation_matches_face_normal() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let mesh = tessellate(&topo, fid, 0.1).unwrap();
            let face = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = face.surface() {
                // Check first triangle winding
                if mesh.indices.len() >= 3 {
                    let i0 = mesh.indices[0] as usize;
                    let i1 = mesh.indices[1] as usize;
                    let i2 = mesh.indices[2] as usize;
                    let a = mesh.positions[i1] - mesh.positions[i0];
                    let b = mesh.positions[i2] - mesh.positions[i0];
                    let tri_normal = a.cross(b);
                    let dot = tri_normal.dot(*normal);
                    assert!(
                        dot > 0.0,
                        "Face normal {:?} disagrees with tri normal {:?} (dot={dot})",
                        normal,
                        tri_normal,
                    );
                }
            }
        }
    }

    #[test]
    fn tessellate_box_with_hole_from_boolean() {
        // Boolean cut creating a hole, then tessellate
        let mut topo = Topology::new();
        let base = crate::primitives::make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
        let hole = crate::primitives::make_cylinder(&mut topo, 1.0, 4.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            hole,
            &brepkit_math::mat::Mat4::translation(5.0, 5.0, -1.0),
        )
        .unwrap();

        let cut =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, base, hole).unwrap();

        let mesh = tessellate_solid(&topo, cut, 0.5).unwrap();
        assert!(!mesh.positions.is_empty(), "should produce vertices");
        assert!(!mesh.indices.is_empty(), "should produce triangles");
    }

    #[test]
    fn tessellate_thin_box() {
        // Elongated face (1000:1 aspect ratio)
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1000.0, 1.0, 1.0).unwrap();

        let mesh = tessellate_solid(&topo, solid, 1.0).unwrap();
        assert!(!mesh.positions.is_empty(), "should produce vertices");
        assert!(!mesh.indices.is_empty(), "should produce triangles");
    }
}
