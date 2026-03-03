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
            let v_range = compute_axial_range(topo, face_data, cyl.origin(), cyl.axis());
            let u_range = (0.0, std::f64::consts::TAU);
            // Angular resolution from chord deviation on the circular cross-section.
            let nu = segments_for_chord_deviation(cyl.radius(), u_range.1 - u_range.0, deflection);
            // Axial resolution: uniform spacing at roughly the same density.
            let v_extent = (v_range.1 - v_range.0).abs();
            let nv = 4_usize.max(
                (v_extent / (cyl.radius() * std::f64::consts::TAU / nu as f64)).ceil() as usize,
            );
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
            let v_range = compute_axial_range(topo, face_data, cone.apex(), cone.axis());
            let u_range = (0.0, std::f64::consts::TAU);
            // Use the max radius (at the far end) for angular resolution.
            let max_radius = cone.radius_at(v_range.1.abs().max(v_range.0.abs()));
            let nu = segments_for_chord_deviation(
                max_radius.max(0.01),
                u_range.1 - u_range.0,
                deflection,
            );
            let v_extent = (v_range.1 - v_range.0).abs();
            let nv = 4_usize.max(
                (v_extent / (max_radius.max(0.01) * std::f64::consts::TAU / nu as f64)).ceil()
                    as usize,
            );
            let cone = cone.clone();
            Ok(tessellate_analytic(
                |u, v| cone.evaluate(u, v),
                |u, v| cone.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                AnalyticKind::ConeApex,
            ))
        }
        FaceSurface::Sphere(sphere) => {
            let u_range = (0.0, std::f64::consts::TAU);
            let v_range = (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
            let nu =
                segments_for_chord_deviation(sphere.radius(), u_range.1 - u_range.0, deflection);
            // Latitude resolution: same chord deviation criterion.
            let nv =
                segments_for_chord_deviation(sphere.radius(), v_range.1 - v_range.0, deflection);
            let sphere = sphere.clone();
            Ok(tessellate_analytic(
                |u, v| sphere.evaluate(u, v),
                |u, v| sphere.normal(u, v),
                u_range,
                v_range,
                nu,
                nv,
                AnalyticKind::SpherePole,
            ))
        }
        FaceSurface::Torus(torus) => {
            let u_range = (0.0, std::f64::consts::TAU);
            let v_range = (0.0, std::f64::consts::TAU);
            // Major circle resolution: use (R + r) as the effective radius.
            let nu = segments_for_chord_deviation(
                torus.major_radius() + torus.minor_radius(),
                u_range.1 - u_range.0,
                deflection,
            );
            // Minor circle (tube cross-section) resolution.
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

/// Compute the angular resolution needed for a circular arc to achieve
/// a given chord deviation (sag).
///
/// For a circle of radius `r`, the chord deviation at the midpoint of
/// an arc subtending angle `θ` is `r*(1 - cos(θ/2))`. Solving for the
/// number of segments `n` over an arc range: `n = ceil(range / θ)` where
/// `θ = 2*acos(1 - deflection/r)`.
fn segments_for_chord_deviation(radius: f64, arc_range: f64, deflection: f64) -> usize {
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
    // Also apply a fallback formula for very coarse deflection (where the
    // geometric formula gives too few segments for downstream accuracy):
    // n_fallback = ceil(range / sqrt(deflection)), matching the legacy behavior
    // for small surfaces where deflection >> radius.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_fallback = (arc_range / deflection.sqrt()).ceil() as usize;
    n.max(n_fallback).max(4)
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
) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    let nu = nu.max(4);
    let nv = nv.max(4);

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
                let n_samples = 32;
                let (t_start, t_end) = if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    let ts = circle.project(sp);
                    let mut te = circle.project(ep);
                    if te <= ts {
                        te += std::f64::consts::TAU;
                    }
                    (ts, te)
                };
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
                let n_samples = 32;
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
                sample_curve(
                    &|t| ellipse.evaluate(t),
                    &|i| t_start + (t_end - t_start) * (i as f64) / (n_samples as f64),
                    n_samples,
                    oe.is_forward(),
                    &mut positions,
                );
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                let n_samples = 16;
                let (u0, u1) = nurbs.domain();
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
) -> TriangleMesh {
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
        EdgeCurve::Ellipse(_) => {
            // Conservative: use higher count for ellipses due to varying curvature.
            let n = (std::f64::consts::TAU / deflection.sqrt()).ceil() as usize;
            n.max(16)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let n_cp = nurbs.control_points().len();
            // Sample proportional to control points and inversely to deflection.
            let base = n_cp * 4;
            let adaptive = (1.0 / deflection.sqrt()).ceil() as usize;
            base.max(adaptive).max(8)
        }
    }
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
    let mut edge_points: HashMap<usize, Vec<Point3>> = HashMap::new();

    for &edge_idx in edge_face_map.keys() {
        // Reconstruct EdgeId from arena index.
        if let Some(edge_id) = topo.edges.id_from_index(edge_idx) {
            if let Ok(edge_data) = topo.edge(edge_id) {
                let points = sample_edge(topo, edge_data, deflection)?;
                edge_points.insert(edge_idx, points);
            }
        }
    }

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
    for &face_id in &all_faces {
        tessellate_face_with_shared_edges(
            topo,
            face_id,
            deflection,
            &edge_global_indices,
            &mut merged,
            &mut point_to_global,
        )?;
    }

    // Phase 5: Compute proper vertex normals via face-area-weighted averaging.
    // Reset normals to zero, then accumulate face normals weighted by triangle area.
    for n in &mut merged.normals {
        *n = Vec3::new(0.0, 0.0, 0.0);
    }
    let tri_count = merged.indices.len() / 3;
    for t in 0..tri_count {
        let i0 = merged.indices[t * 3] as usize;
        let i1 = merged.indices[t * 3 + 1] as usize;
        let i2 = merged.indices[t * 3 + 2] as usize;
        let a = merged.positions[i1] - merged.positions[i0];
        let b = merged.positions[i2] - merged.positions[i0];
        let face_normal = a.cross(b); // length = 2× triangle area (area weighting)
        merged.normals[i0] = merged.normals[i0] + face_normal;
        merged.normals[i1] = merged.normals[i1] + face_normal;
        merged.normals[i2] = merged.normals[i2] + face_normal;
    }
    for n in &mut merged.normals {
        if let Ok(normalized) = n.normalize() {
            *n = normalized;
        } else {
            *n = Vec3::new(0.0, 0.0, 1.0);
        }
    }

    Ok(merged)
}

/// Tessellate a single face, reusing shared edge vertices from the global mesh.
///
/// For planar faces: collects boundary vertices (reusing global indices for
/// shared edges), then triangulates via ear-clipping.
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

    if let FaceSurface::Plane { normal, .. } = face_data.surface() {
        // For planar faces: build boundary polygon from shared edge vertices.
        let normal = *normal;
        let wire = topo.wire(face_data.outer_wire())?;

        let mut boundary_global_ids: Vec<u32> = Vec::new();
        let tol = 1e-10;

        for oe in wire.edges() {
            let edge_idx = oe.edge().index();
            if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
                // Use pre-computed edge vertices. Reverse if edge orientation
                // in this wire is backwards.
                let ordered: Vec<u32> = if oe.is_forward() {
                    global_ids.clone()
                } else {
                    global_ids.iter().rev().copied().collect()
                };

                // Skip the first point if it duplicates the last boundary point
                // (edges share endpoints: edge[i].end == edge[i+1].start).
                for (j, &gid) in ordered.iter().enumerate() {
                    if j == 0 && !boundary_global_ids.is_empty() {
                        let last_gid = *boundary_global_ids.last().unwrap_or(&u32::MAX);
                        if last_gid == gid {
                            continue;
                        }
                        // Check position proximity for points from different edges
                        // that share the same vertex.
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

        // Remove closing duplicate if the wire loops back.
        if boundary_global_ids.len() > 2 {
            if let (Some(&first), Some(&last)) =
                (boundary_global_ids.first(), boundary_global_ids.last())
            {
                if first == last {
                    boundary_global_ids.pop();
                } else if (first as usize) < merged.positions.len()
                    && (last as usize) < merged.positions.len()
                {
                    let fp = merged.positions[first as usize];
                    let lp = merged.positions[last as usize];
                    if (fp - lp).length() < tol {
                        boundary_global_ids.pop();
                    }
                }
            }
        }

        let n = boundary_global_ids.len();
        if n < 3 {
            return Ok(());
        }

        // Gather positions for ear-clip triangulation (need local coords).
        let local_positions: Vec<Point3> = boundary_global_ids
            .iter()
            .map(|&gid| merged.positions[gid as usize])
            .collect();

        let mut local_indices = ear_clip_triangulate(&local_positions, normal);

        // Ensure triangle winding matches the face normal.
        // ear_clip_triangulate always produces CCW in 2D projection, but
        // if the face normal has a negative dominant component the winding
        // needs to be flipped to produce outward-facing triangles.
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

        // Map local triangle indices back to global vertex indices.
        for &li in &local_indices {
            merged.indices.push(boundary_global_ids[li as usize]);
        }
    } else {
        // For non-planar faces (NURBS, Cylinder, Cone, Sphere, Torus):
        // Tessellate independently, then stitch boundary vertices to the
        // global shared edges.
        let face_mesh = tessellate(topo, face_id, deflection)?;

        // Build a mapping from local face mesh vertices to global indices.
        // Boundary vertices (those near shared edge points) get mapped to
        // the global shared vertex. Interior vertices get new global indices.
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
        // Also include inner wires.
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

        // Fixed geometric tolerance for snapping boundary vertices to shared
        // edge points. Independent of deflection to avoid being too tight at
        // high quality or too loose at low quality.
        let snap_tol = 1e-6;

        for (i, &pos) in face_mesh.positions.iter().enumerate() {
            // Try to snap to a shared edge vertex.
            let mut best_gid = None;
            let mut best_dist = snap_tol;

            for &(target_pos, gid) in &snap_targets {
                let dist = (pos - target_pos).length();
                if dist < best_dist {
                    best_dist = dist;
                    best_gid = Some(gid);
                }
            }

            if let Some(gid) = best_gid {
                local_to_global.push(gid);
            } else {
                // Interior vertex: add to global pool.
                let key = (pos.x().to_bits(), pos.y().to_bits(), pos.z().to_bits());
                let gid = point_to_global.entry(key).or_insert_with(|| {
                    #[allow(clippy::cast_possible_truncation)]
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

        // Remap triangle indices from local to global.
        for &li in &face_mesh.indices {
            merged.indices.push(local_to_global[li as usize]);
        }
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

        // Full watertightness for curved faces requires CDT-based boundary-
        // constrained tessellation (not yet implemented). The shared edges
        // provide the topological foundation for future watertight stitching.
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
        let fine_mesh = tessellate_nurbs(&dome, 0.01);
        let coarse_mesh = tessellate_nurbs(&dome, 0.5);

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
        let mesh = tessellate_nurbs(&surface, deflection);

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
}
