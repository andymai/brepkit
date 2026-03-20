//! Fragment-building functions for analytic surface boolean operations.
//!
//! These functions split analytic (cylinder, sphere, cone) faces into band or
//! cap fragments at intersection v-levels, preserving the analytic
//! `FaceSurface` on the output faces.

use std::collections::{HashMap, HashSet};

use brepkit_math::predicates::point_in_polygon;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use super::assembly::quantize_point;
use super::classify::polygon_centroid;
use super::precompute::{cone_v_extent, cylinder_v_extent, dedup_points_by_position};
use super::types::{
    AnalyticFragment, CLOSED_CURVE_SAMPLES, CurveClassification, FaceSnapshot, Source,
};

// ---------------------------------------------------------------------------
// V-range helpers (used only by fragment builders)
// ---------------------------------------------------------------------------

/// Merge overlapping v-ranges with padding, clamped to `[extent_min, extent_max]`.
///
/// Returns `None` if the merged zones cover more than 60% of the total extent
/// (band-splitting would not be worthwhile).
fn merge_vranges_with_padding(
    vranges: &[(f64, f64)],
    extent_min: f64,
    extent_max: f64,
    padding_fraction: f64,
) -> Option<Vec<(f64, f64)>> {
    let extent_height = extent_max - extent_min;
    let padding = extent_height * padding_fraction;
    let mut sorted: Vec<(f64, f64)> = vranges.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for &(lo, hi) in &sorted {
        let lo_padded = (lo - padding).max(extent_min);
        let hi_padded = (hi + padding).min(extent_max);
        if let Some(last) = merged.last_mut() {
            if lo_padded <= last.1 {
                last.1 = last.1.max(hi_padded);
                continue;
            }
        }
        merged.push((lo_padded, hi_padded));
    }

    let total_iz: f64 = merged.iter().map(|(lo, hi)| hi - lo).sum();
    if total_iz > extent_height * 0.6 {
        None
    } else {
        Some(merged)
    }
}

/// Build an ordered list of v-levels from extent bounds and merged intersection zones.
///
/// Deduplicates levels that are within 1e-10 of each other.
///
/// 1e-10 is a parametric-space dedup tolerance: v-parameters on analytic surfaces
/// (cylinder axis projection, sphere latitude) are in model-space units (meters),
/// so 1e-10 m = 0.1 nm — well below any meaningful geometric feature while still
/// catching floating-point duplicates from independent computations.
fn build_v_levels(extent_min: f64, extent_max: f64, merged: &[(f64, f64)]) -> Vec<f64> {
    let mut levels: Vec<f64> = vec![extent_min];
    for &(iz_lo, iz_hi) in merged {
        if iz_lo > extent_min + 1e-10 {
            levels.push(iz_lo);
        }
        if iz_hi < extent_max - 1e-10 {
            levels.push(iz_hi);
        }
    }
    levels.push(extent_max);
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    levels
}

// ---------------------------------------------------------------------------
// Fragment builders
// ---------------------------------------------------------------------------

/// Find where an intersection curve crosses the face boundary, returning
/// boundary crossing points.
///
/// Instead of returning all sample points inside the face (which creates many
/// chord segments), this finds the entry/exit points where the curve crosses
/// the polygon boundary. This gives 1-2 chord endpoints per crossing, keeping
/// the split count minimal.
///
/// When the curve lies entirely inside the face (zero crossings, all samples
/// inside), returns `FullyContained` so callers can create a splitting chord
/// through the curve.
pub(super) fn curve_boundary_crossings(
    curve: &brepkit_math::analytic_intersection::ExactIntersectionCurve,
    face_verts: &[Point3],
    face_normal: Vec3,
    _tol: Tolerance,
) -> CurveClassification {
    use brepkit_math::analytic_intersection::ExactIntersectionCurve;

    let n_samples = 64;
    let raw_points: Vec<Point3> = match curve {
        ExactIntersectionCurve::Circle(c) => {
            brepkit_geometry::sampling::sample_uniform(c, 0.0, std::f64::consts::TAU, n_samples + 1)
        }
        ExactIntersectionCurve::Ellipse(e) => {
            brepkit_geometry::sampling::sample_uniform(e, 0.0, std::f64::consts::TAU, n_samples + 1)
        }
        ExactIntersectionCurve::Points(pts) => pts.clone(),
    };

    if raw_points.len() < 2 {
        return CurveClassification::Crossings(raw_points);
    }

    // Pre-project face polygon to 2D once, then test all sample points
    // against the projected polygon. This avoids re-projecting the polygon
    // for every sample point (64 allocations -> 1 allocation).
    let ax = face_normal.x().abs();
    let ay = face_normal.y().abs();
    let az = face_normal.z().abs();
    let project_3d_to_2d = |p: Point3| -> Point2 {
        if az >= ax && az >= ay {
            Point2::new(p.x(), p.y())
        } else if ay >= ax {
            Point2::new(p.x(), p.z())
        } else {
            Point2::new(p.y(), p.z())
        }
    };
    let polygon_2d: Vec<Point2> = face_verts.iter().map(|p| project_3d_to_2d(*p)).collect();

    // Classify each sample as inside or outside the face polygon.
    let inside: Vec<bool> = raw_points
        .iter()
        .map(|pt| point_in_polygon(project_3d_to_2d(*pt), &polygon_2d))
        .collect();

    let all_inside = inside.iter().all(|&v| v);
    let none_inside = inside.iter().all(|&v| !v);

    if all_inside {
        return CurveClassification::FullyContained;
    }
    if none_inside {
        return CurveClassification::FullyOutside;
    }

    // Find boundary crossing points: transitions from inside->outside or
    // outside->inside. At each transition, interpolate the approximate crossing
    // point. Also include the first and last interior points as chord endpoints.
    let mut crossings = Vec::new();
    let mut in_run = false;

    for i in 0..raw_points.len() {
        if inside[i] && !in_run {
            // Entering the face: record the entry point (use midpoint of
            // crossing segment for better accuracy).
            if i > 0 && !inside[i - 1] {
                let mid = Point3::new(
                    (raw_points[i - 1].x() + raw_points[i].x()) * 0.5,
                    (raw_points[i - 1].y() + raw_points[i].y()) * 0.5,
                    (raw_points[i - 1].z() + raw_points[i].z()) * 0.5,
                );
                crossings.push(mid);
            } else {
                crossings.push(raw_points[i]);
            }
            in_run = true;
        } else if !inside[i] && in_run {
            // Exiting the face: record the exit point.
            if i > 0 {
                let mid = Point3::new(
                    (raw_points[i - 1].x() + raw_points[i].x()) * 0.5,
                    (raw_points[i - 1].y() + raw_points[i].y()) * 0.5,
                    (raw_points[i - 1].z() + raw_points[i].z()) * 0.5,
                );
                crossings.push(mid);
            }
            in_run = false;
        }
    }

    CurveClassification::Crossings(crossings)
}

/// Tessellate a non-planar face into triangle fragments for the analytic boolean.
///
/// Used for sphere faces where band decomposition isn't feasible. Each triangle
/// gets its own `AnalyticFragment` but retains the original surface type so that
/// output faces preserve the analytic geometry.
pub(super) fn tessellate_face_into_fragments(
    topo: &Topology,
    face_id: FaceId,
    source: Source,
    deflection: f64,
    fragments: &mut Vec<AnalyticFragment>,
) -> Result<(), crate::OperationsError> {
    let mesh = crate::tessellate::tessellate(topo, face_id, deflection)?;
    for tri in mesh.indices.chunks_exact(3) {
        let v0 = mesh.positions[tri[0] as usize];
        let v1 = mesh.positions[tri[1] as usize];
        let v2 = mesh.positions[tri[2] as usize];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let cross = edge1.cross(edge2);
        // Skip degenerate (zero-area) triangles.
        let Ok(normal) = cross.normalize() else {
            continue;
        };
        let d_val = crate::dot_normal_point(normal, v0);

        // Use planar surface for tessellated triangle fragments.
        // The original curved surface (e.g. Sphere) can't be used because
        // each triangle is a flat approximation -- re-tessellating with the
        // curved surface would project vertices back onto the sphere,
        // producing wrong geometry.
        let plane_surface = FaceSurface::Plane { normal, d: d_val };

        fragments.push(AnalyticFragment {
            vertices: vec![v0, v1, v2],
            surface: plane_surface,
            normal,
            d: d_val,
            source,
            edge_curves: vec![None; 3],
            source_reversed: false, // planar fragment, no reversal needed
            source_face_id: Some(face_id),
        });
    }
    Ok(())
}

/// Split a cylinder face at intersection v-ranges, producing:
/// - Cylinder-surface band fragments for regions *outside* the intersection
/// - Tessellated planar triangle fragments for the narrow intersection band
///
/// This prevents face-count explosion in sequential boolean operations:
/// instead of tessellating the *entire* barrel into ~500 triangles, only the
/// thin intersection region gets tessellated while the rest keeps
/// `FaceSurface::Cylinder`.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(super) fn split_cylinder_at_intersection(
    surface: &FaceSurface,
    face_verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    source_reversed: bool,
    vranges: &[(f64, f64)],
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    tol: Tolerance,
    fragments: &mut Vec<AnalyticFragment>,
) -> Result<(), crate::OperationsError> {
    let FaceSurface::Cylinder(cyl) = surface else {
        // Not a cylinder -- fall back to full tessellation.
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    // Compute the barrel's v extent from boundary vertices.
    let Some((barrel_vmin, barrel_vmax)) = cylinder_v_extent(cyl, face_verts) else {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
            source_face_id: Some(face_id),
        });
        return Ok(());
    };

    if vranges.is_empty() {
        // Degenerate or no ranges -- keep as-is.
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
            source_face_id: Some(face_id),
        });
        return Ok(());
    }

    // Merge overlapping v-ranges with padding. Fall back to full tessellation
    // if intersection zones cover >60% of the barrel height.
    let Some(merged) = merge_vranges_with_padding(vranges, barrel_vmin, barrel_vmax, 0.05) else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    // Build level list: regions outside merged ranges are cylinder bands,
    // regions inside merged ranges get tessellated.
    // Levels: barrel_vmin, [gap regions as cylinder], [intersection regions tessellated], barrel_vmax
    let n_samples: usize = CLOSED_CURVE_SAMPLES;

    // Collect face polygon vertices at barrel endpoints for exact vertex matching.
    // v_tol = 1e-6: axial snap tolerance for classifying face polygon vertices as
    // belonging to a barrel endpoint level. 1e-6 m = 1 micron — generous enough
    // to catch floating-point drift from coordinate transforms while still
    // distinguishing barrel endpoints from interior vertices on any practical
    // cylinder (minimum height ~0.01 mm).
    let v_tol = 1e-6;
    let mut verts_at_vmin: Vec<Point3> = Vec::new();
    let mut verts_at_vmax: Vec<Point3> = Vec::new();
    for &p in face_verts {
        let v = cyl.axis().dot(p - cyl.origin());
        if (v - barrel_vmin).abs() < v_tol {
            verts_at_vmin.push(p);
        } else if (v - barrel_vmax).abs() < v_tol {
            verts_at_vmax.push(p);
        }
    }
    dedup_points_by_position(&mut verts_at_vmin, tol);
    dedup_points_by_position(&mut verts_at_vmax, tol);

    #[allow(clippy::cast_precision_loss)]
    let sample_circle_at_v = |v: f64| -> Vec<Point3> {
        if (v - barrel_vmin).abs() < v_tol && verts_at_vmin.len() >= 3 {
            verts_at_vmin.clone()
        } else if (v - barrel_vmax).abs() < v_tol && verts_at_vmax.len() >= 3 {
            verts_at_vmax.clone()
        } else {
            (0..n_samples)
                .map(|i| {
                    let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                    cyl.evaluate(u, v)
                })
                .collect()
        }
    };

    // Helper: create a cylinder band fragment between two v-levels.
    let make_band = |v_bot: f64, v_top: f64, frags: &mut Vec<AnalyticFragment>| {
        // Degenerate band guard: skip bands thinner than 1e-10 m (0.1 nm).
        // Matches build_v_levels dedup tolerance.
        if (v_top - v_bot).abs() < 1e-10 {
            return;
        }
        let bot_pts = sample_circle_at_v(v_bot);
        let top_pts = sample_circle_at_v(v_top);

        let mut verts = Vec::with_capacity(bot_pts.len() + top_pts.len());
        verts.extend_from_slice(&bot_pts);
        verts.extend(top_pts.into_iter().rev());

        // Compute normal from a surface point (not centroid) to avoid
        // degenerate zero normal for full-circle bands.
        let surface_point = verts[0];
        let band_normal = (surface_point
            - cyl.origin()
            - cyl.axis() * cyl.axis().dot(surface_point - cyl.origin()))
        .normalize()
        .unwrap_or(normal);
        let centroid = polygon_centroid(&verts);
        let band_d = crate::dot_normal_point(band_normal, centroid);

        let n_verts = verts.len();
        frags.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: band_normal,
            d: band_d,
            source,
            edge_curves: vec![None; n_verts],
            source_reversed,
            source_face_id: Some(face_id),
        });
    };

    // Walk through the barrel, creating cylinder bands for all regions.
    // Both gap regions and intersection zones become cylinder bands,
    // preserving FaceSurface::Cylinder throughout. The classifier
    // determines inside/outside for each band independently.
    //
    let levels = build_v_levels(barrel_vmin, barrel_vmax, &merged);

    for w in levels.windows(2) {
        make_band(w[0], w[1], fragments);
    }

    Ok(())
}

/// Split a cone face into band fragments at intersection v-levels.
///
/// Analogous to `split_cylinder_at_intersection` but for conical geometry.
/// Each band preserves `FaceSurface::Cone`. Circles at each v-level have
/// different radii (`cone.radius_at(v)`).
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub(super) fn split_cone_at_intersection(
    surface: &FaceSurface,
    face_verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    source_reversed: bool,
    vranges: &[(f64, f64)],
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    tol: Tolerance,
    fragments: &mut Vec<AnalyticFragment>,
) -> Result<(), crate::OperationsError> {
    let FaceSurface::Cone(cone) = surface else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    // Compute the barrel's v extent from boundary vertices.
    let Some((barrel_vmin, barrel_vmax)) = cone_v_extent(cone, face_verts) else {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
            source_face_id: Some(face_id),
        });
        return Ok(());
    };

    if vranges.is_empty() {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
            source_face_id: Some(face_id),
        });
        return Ok(());
    }

    // Merge overlapping v-ranges with padding. Fall back to full tessellation
    // if intersection zones cover >60% of the barrel height.
    let Some(merged) = merge_vranges_with_padding(vranges, barrel_vmin, barrel_vmax, 0.05) else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    let n_samples: usize = CLOSED_CURVE_SAMPLES;

    // Collect face polygon vertices at barrel endpoints for exact vertex matching.
    let v_tol = 1e-6;
    let mut verts_at_vmin: Vec<Point3> = Vec::new();
    let mut verts_at_vmax: Vec<Point3> = Vec::new();
    for &p in face_verts {
        let (_, v) = cone.project_point(p);
        if (v - barrel_vmin).abs() < v_tol {
            verts_at_vmin.push(p);
        } else if (v - barrel_vmax).abs() < v_tol {
            verts_at_vmax.push(p);
        }
    }
    dedup_points_by_position(&mut verts_at_vmin, tol);
    dedup_points_by_position(&mut verts_at_vmax, tol);

    #[allow(clippy::cast_precision_loss)]
    let sample_circle_at_v = |v: f64| -> Vec<Point3> {
        if (v - barrel_vmin).abs() < v_tol && verts_at_vmin.len() >= 3 {
            verts_at_vmin.clone()
        } else if (v - barrel_vmax).abs() < v_tol && verts_at_vmax.len() >= 3 {
            verts_at_vmax.clone()
        } else {
            (0..n_samples)
                .map(|i| {
                    let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                    cone.evaluate(u, v)
                })
                .collect()
        }
    };

    // Helper: create a cone band fragment between two v-levels.
    let make_band = |v_bot: f64, v_top: f64, frags: &mut Vec<AnalyticFragment>| {
        if (v_top - v_bot).abs() < 1e-10 {
            return;
        }
        let bot_pts = sample_circle_at_v(v_bot);
        let top_pts = sample_circle_at_v(v_top);

        let mut verts = Vec::with_capacity(bot_pts.len() + top_pts.len());
        verts.extend_from_slice(&bot_pts);
        verts.extend(top_pts.into_iter().rev());

        // Compute normal from a surface point — radial direction on the cone.
        // Use the projected (u, v) so the normal matches the actual azimuthal
        // position of the surface point, not a fixed u=0 direction.
        let surface_point = verts[0];
        let (u0, v0) = cone.project_point(surface_point);
        let band_normal = cone.normal(u0, v0);
        let centroid = polygon_centroid(&verts);
        let band_d = crate::dot_normal_point(band_normal, centroid);

        let n_verts = verts.len();
        frags.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: band_normal,
            d: band_d,
            source,
            edge_curves: vec![None; n_verts],
            source_reversed,
            source_face_id: Some(face_id),
        });
    };

    let levels = build_v_levels(barrel_vmin, barrel_vmax, &merged);

    for w in levels.windows(2) {
        make_band(w[0], w[1], fragments);
    }

    Ok(())
}

/// Split a sphere face into spherical cap fragments at intersection v-levels.
///
/// Analogous to `split_cylinder_at_intersection` but for spherical geometry.
/// Each cap preserves `FaceSurface::Sphere`. At poles (v = +/-pi/2) the cap
/// degenerates to a single point surrounded by a circle.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub(super) fn split_sphere_at_intersection(
    surface: &FaceSurface,
    face_verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    source_reversed: bool,
    vranges: &[(f64, f64)],
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    fragments: &mut Vec<AnalyticFragment>,
) -> Result<(), crate::OperationsError> {
    let FaceSurface::Sphere(sph) = surface else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    // Compute the face's v-extent. For hemispheres the boundary wire only
    // covers one edge (equator); the other limit is a pole.
    let face_data = topo.face(face_id)?;
    let face_v_range = crate::tessellate::compute_sphere_v_range(topo, face_data, sph);
    let face_vmin = face_v_range.0;
    let face_vmax = face_v_range.1;

    // Degenerate v-extent guard: skip splitting if the face's latitude range is
    // thinner than 1e-10 rad (~0.006 arcsec). Matches build_v_levels dedup tolerance.
    if (face_vmax - face_vmin).abs() < 1e-10 || vranges.is_empty() {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
            source_face_id: Some(face_id),
        });
        return Ok(());
    }

    // Merge overlapping v-ranges with padding. Fall back to full tessellation
    // if intersection zones cover >60% of the face height.
    let Some(merged) = merge_vranges_with_padding(vranges, face_vmin, face_vmax, 0.05) else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    let n_samples: usize = CLOSED_CURVE_SAMPLES;
    // pole_eps = 1e-6 rad (~0.2 arcsec): latitude threshold for treating a
    // v-level as a sphere pole. At the pole the parallel circle degenerates to
    // a point, so the sampling switches from N points to 1. 1e-6 rad is tight
    // enough to avoid misclassifying near-pole latitudes on any practical sphere.
    let pole_eps = 1e-6;
    let is_south_pole = |v: f64| (v + std::f64::consts::FRAC_PI_2).abs() < pole_eps;
    let is_north_pole = |v: f64| (v - std::f64::consts::FRAC_PI_2).abs() < pole_eps;

    // Sample a circle of points at a given latitude on the sphere.
    #[allow(clippy::cast_precision_loss)]
    let sample_circle_at_v = |v: f64| -> Vec<Point3> {
        if is_south_pole(v) || is_north_pole(v) {
            // Pole: single point.
            vec![sph.evaluate(0.0, v)]
        } else {
            (0..n_samples)
                .map(|i| {
                    let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                    sph.evaluate(u, v)
                })
                .collect()
        }
    };

    // Create a spherical cap fragment between two v-levels.
    let make_cap = |v_bot: f64, v_top: f64, frags: &mut Vec<AnalyticFragment>| {
        // Degenerate cap guard: skip caps thinner than 1e-10 rad.
        // Matches build_v_levels dedup tolerance.
        if (v_top - v_bot).abs() < 1e-10 {
            return;
        }
        let bot_pts = sample_circle_at_v(v_bot);
        let top_pts = sample_circle_at_v(v_top);

        let mut verts = Vec::with_capacity(bot_pts.len() + top_pts.len());
        verts.extend_from_slice(&bot_pts);
        if top_pts.len() > 1 {
            verts.extend(top_pts.into_iter().rev());
        } else {
            // Pole point: append as-is (single vertex).
            verts.extend(top_pts);
        }

        // Compute outward normal from a surface point.
        let sample_pt = verts[0];
        let cap_normal = (sample_pt - sph.center()).normalize().unwrap_or(normal);
        let centroid = polygon_centroid(&verts);
        let cap_d = crate::dot_normal_point(cap_normal, centroid);

        let n_verts = verts.len();
        frags.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: cap_normal,
            d: cap_d,
            source,
            edge_curves: vec![None; n_verts],
            source_reversed,
            source_face_id: Some(face_id),
        });
    };

    let levels = build_v_levels(face_vmin, face_vmax, &merged);

    for w in levels.windows(2) {
        make_cap(w[0], w[1], fragments);
    }

    Ok(())
}

/// Compute v-parameter ranges where intersection chords cross analytic faces.
///
/// For each face that has intersection chords (and isn't in the analytic-analytic
/// set), projects chord endpoints onto the surface's v-parameter and records
/// the (vmin, vmax) interval.
#[allow(clippy::type_complexity)]
pub(super) fn collect_analytic_vranges(
    snaps: &[FaceSnapshot],
    face_intersections: &HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>>,
    analytic_analytic_faces: &HashSet<usize>,
    vranges_out: &mut HashMap<usize, Vec<(f64, f64)>>,
) {
    for (idx, snap) in snaps.iter().enumerate() {
        let Some(chords) = face_intersections.get(&idx) else {
            continue;
        };
        if analytic_analytic_faces.contains(&idx) {
            continue;
        }
        let v_of_point: Box<dyn Fn(Point3) -> f64> = match &snap.surface {
            FaceSurface::Cylinder(cyl) => {
                let axis = cyl.axis();
                let origin = cyl.origin();
                Box::new(move |p| axis.dot(p - origin))
            }
            FaceSurface::Sphere(sph) => {
                let sph = sph.clone();
                Box::new(move |p| sph.project_point(p).1)
            }
            FaceSurface::Cone(cone) => {
                let cone = cone.clone();
                Box::new(move |p| cone.project_point(p).1)
            }
            _ => continue,
        };
        let mut vmin = f64::MAX;
        let mut vmax = f64::MIN;
        for &(p0, p1, _) in chords {
            for p in &[p0, p1] {
                let v = v_of_point(*p);
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            }
        }
        if vmax > vmin {
            vranges_out.entry(idx).or_default().push((vmin, vmax));
        }
    }
}

/// Map a point to its v-parameter on a cylinder or cone surface.
///
/// For a cylinder, v is the signed axial distance from the origin along the axis.
/// For a cone, v is the generator-length parameter from `project_point`.
fn surface_v_param(surface: &FaceSurface, p: Point3) -> f64 {
    match surface {
        FaceSurface::Cylinder(cyl) => cyl.axis().dot(p - cyl.origin()),
        FaceSurface::Cone(cone) => cone.project_point(p).1,
        _ => 0.0,
    }
}

/// Create band fragments for a non-planar (analytic) face that has contained
/// curves. Splits the face into bands between the contained curves and the
/// face's natural boundary circles.
///
/// Supports cylinder and cone faces. Other surface types fall through without
/// creating bands (the face stays as one unsplit fragment via the caller's
/// else branch, but this path shouldn't be reached for unsupported types).
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub(super) fn create_band_fragments(
    surface: &FaceSurface,
    face_verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    source_reversed: bool,
    contained_curves: &[EdgeCurve],
    _topo: &Topology,
    tol: Tolerance,
    face_id: Option<FaceId>,
    fragments: &mut Vec<AnalyticFragment>,
) {
    // Compute surface-specific v-extent. Bail out for unsupported surfaces.
    let v_extent: Option<(f64, f64)> = match surface {
        FaceSurface::Cylinder(cyl) => cylinder_v_extent(cyl, face_verts),
        FaceSurface::Cone(cone) => cone_v_extent(cone, face_verts),
        _ => {
            // For unsupported analytic faces, fall back to unsplit fragment.
            fragments.push(AnalyticFragment {
                vertices: face_verts.to_vec(),
                surface: surface.clone(),
                normal,
                d,
                source,
                edge_curves: vec![None; face_verts.len()],
                source_reversed,
                source_face_id: face_id,
            });
            return;
        }
    };

    let n_samples: usize = CLOSED_CURVE_SAMPLES;

    // Pair each contained curve with its v-parameter.
    let mut cut_levels: Vec<(f64, &EdgeCurve)> = Vec::new();
    for ec in contained_curves {
        let center = match ec {
            EdgeCurve::Circle(c) => c.center(),
            EdgeCurve::Ellipse(e) => e.center(),
            _ => continue,
        };
        let v = surface_v_param(surface, center);
        cut_levels.push((v, ec));
    }

    if cut_levels.is_empty() {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
            source_face_id: face_id,
        });
        return;
    }

    // Compute the barrel's v extent from its boundary vertices.
    let Some((v_min, v_max)) = v_extent else {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
            source_face_id: face_id,
        });
        return;
    };

    // Sort cut levels by v-parameter.
    cut_levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build ordered level list: barrel_bottom, cut1, cut2, ..., barrel_top.
    // Each level is either a barrel endpoint (None) or a cut curve (Some).
    let mut levels: Vec<(f64, Option<&EdgeCurve>)> = vec![(v_min, None)];
    for &(cv, ec) in &cut_levels {
        if let Some(last) = levels.last() {
            // 1e-10: v-parameter dedup tolerance — same as build_v_levels.
            if (cv - last.0).abs() > 1e-10 {
                levels.push((cv, Some(ec)));
            }
        }
    }
    if let Some(last) = levels.last() {
        if (v_max - last.0).abs() > 1e-10 {
            levels.push((v_max, None));
        }
    }

    // Extract face polygon vertices at each barrel endpoint level.
    let v_tol = 1e-6;
    let mut verts_at_vmin: Vec<Point3> = Vec::new();
    let mut verts_at_vmax: Vec<Point3> = Vec::new();
    for &p in face_verts {
        let v = surface_v_param(surface, p);
        if (v - v_min).abs() < v_tol {
            verts_at_vmin.push(p);
        } else if (v - v_max).abs() < v_tol {
            verts_at_vmax.push(p);
        }
    }
    // Deduplicate points at each level (seam vertex duplicates circle[0]).
    dedup_points_by_position(&mut verts_at_vmin, tol);
    dedup_points_by_position(&mut verts_at_vmax, tol);

    // Sample a circle at a given v-level. For cut levels with an EdgeCurve,
    // use sample_edge_curve so the points match the holed-face inner wire.
    // For barrel endpoints, use the actual face polygon vertices so they
    // share vertices/edges with adjacent cap faces (exact float match).
    #[allow(clippy::cast_precision_loss)]
    let sample_level = |v: f64, curve: Option<&EdgeCurve>| -> Vec<Point3> {
        if let Some(ec) = curve {
            sample_edge_curve(ec, n_samples)
        } else if (v - v_min).abs() < v_tol && verts_at_vmin.len() >= 3 {
            verts_at_vmin.clone()
        } else if (v - v_max).abs() < v_tol && verts_at_vmax.len() >= 3 {
            verts_at_vmax.clone()
        } else {
            match surface {
                FaceSurface::Cylinder(cyl) => (0..n_samples)
                    .map(|i| {
                        let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                        cyl.evaluate(u, v)
                    })
                    .collect(),
                FaceSurface::Cone(cone) => (0..n_samples)
                    .map(|i| {
                        let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                        cone.evaluate(u, v)
                    })
                    .collect(),
                _ => vec![],
            }
        }
    };

    // Create a band fragment for each consecutive pair of levels.
    for w in levels.windows(2) {
        let (v_bot, ec_bot) = w[0];
        let (v_top, ec_top) = w[1];

        let bot_pts = sample_level(v_bot, ec_bot);
        let top_pts = sample_level(v_top, ec_top);

        // Build polygon: bottom circle forward, top circle reversed.
        let mut verts = Vec::with_capacity(2 * n_samples);
        verts.extend_from_slice(&bot_pts);
        verts.extend(top_pts.into_iter().rev());

        // Compute representative normal and d for the band.
        // Use a surface point (first vertex) for the outward normal, not
        // the polygon centroid — the centroid of a full-circle band falls
        // on the axis, making the radial direction degenerate.
        let surface_point = verts[0];
        let band_normal = match surface {
            FaceSurface::Cylinder(cyl) => (surface_point
                - cyl.origin()
                - cyl.axis() * cyl.axis().dot(surface_point - cyl.origin()))
            .normalize()
            .unwrap_or(normal),
            FaceSurface::Cone(cone) => {
                let (u0, v0) = cone.project_point(surface_point);
                cone.normal(u0, v0)
            }
            _ => normal,
        };
        let centroid = polygon_centroid(&verts);
        let band_d = crate::dot_normal_point(band_normal, centroid);

        let n_verts = verts.len();
        fragments.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: band_normal,
            d: band_d,
            source,
            edge_curves: vec![None; n_verts],
            source_reversed,
            source_face_id: face_id,
        });
    }
}

/// Build a proper cylinder barrel wire with Circle edges + seam line.
///
/// Cylinder barrel fragments are represented as polygons with layout:
///   `bot[0..n] ++ top_reversed[0..n]`  (2n vertices total)
/// where n = `CLOSED_CURVE_SAMPLES`. This function consolidates those 2n
/// line edges into 2 Circle edges + 1 seam Line (3 unique edges, 4 oriented),
/// matching the canonical B-Rep topology for a cylinder lateral face.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_cylinder_barrel_wire(
    topo: &mut Topology,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    verts: &[Point3],
    vertex_map: &mut HashMap<(i64, i64, i64), VertexId>,
    edge_map: &mut HashMap<(usize, usize), EdgeId>,
    resolution: f64,
    tol: Tolerance,
) -> Result<WireId, crate::OperationsError> {
    // The polygon layout is: bot[0..n/2] + top_reversed[0..n/2].
    // bot[0] is at u=0 on the bottom circle; verts[2n-1] = top[0] is at u=0 on the top circle.
    let bot_seam_pos = verts[0];
    let top_seam_pos = verts[verts.len() - 1];

    // Compute v-levels from vertex positions on the cylinder axis.
    let v_bot = cyl.axis().dot(bot_seam_pos - cyl.origin());
    let v_top = cyl.axis().dot(top_seam_pos - cyl.origin());

    // Create Circle3D at each level.
    let bot_center = cyl.origin() + cyl.axis() * v_bot;
    let top_center = cyl.origin() + cyl.axis() * v_top;
    let bot_circle = brepkit_math::curves::Circle3D::new(bot_center, cyl.axis(), cyl.radius())
        .map_err(crate::OperationsError::Math)?;
    let top_circle = brepkit_math::curves::Circle3D::new(top_center, cyl.axis(), cyl.radius())
        .map_err(crate::OperationsError::Math)?;

    // Create/lookup vertices at the seam points.
    let bot_vid = *vertex_map
        .entry(quantize_point(bot_seam_pos, resolution))
        .or_insert_with(|| topo.add_vertex(Vertex::new(bot_seam_pos, tol.linear)));
    let top_vid = *vertex_map
        .entry(quantize_point(top_seam_pos, resolution))
        .or_insert_with(|| topo.add_vertex(Vertex::new(top_seam_pos, tol.linear)));

    // Create/lookup closed Circle edges -- dedup key is (v, v) for closed edges.
    let bot_edge = *edge_map
        .entry((bot_vid.index(), bot_vid.index()))
        .or_insert_with(|| {
            topo.add_edge(Edge::new(bot_vid, bot_vid, EdgeCurve::Circle(bot_circle)))
        });
    let top_edge = *edge_map
        .entry((top_vid.index(), top_vid.index()))
        .or_insert_with(|| {
            topo.add_edge(Edge::new(top_vid, top_vid, EdgeCurve::Circle(top_circle)))
        });

    // Create/lookup seam line edge. Forward means bot->top in canonical order.
    let seam_fwd = bot_vid.index() <= top_vid.index();
    let seam_key = if seam_fwd {
        (bot_vid.index(), top_vid.index())
    } else {
        (top_vid.index(), bot_vid.index())
    };
    let seam_edge = *edge_map.entry(seam_key).or_insert_with(|| {
        let (start, end) = if seam_fwd {
            (bot_vid, top_vid)
        } else {
            (top_vid, bot_vid)
        };
        topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
    });

    // Wire: bot_circle(fwd) -> seam(fwd) -> top_circle(rev) -> seam(rev)
    // This matches the canonical cylinder lateral wire from make_cylinder.
    let wire = Wire::new(
        vec![
            OrientedEdge::new(bot_edge, true),
            OrientedEdge::new(seam_edge, seam_fwd),
            OrientedEdge::new(top_edge, false),
            OrientedEdge::new(seam_edge, !seam_fwd),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

/// Build a proper cone barrel wire with Circle edges + seam line.
///
/// Cone barrel fragments have the same polygon layout as cylinder barrels:
///   `bot[0..n] ++ top_reversed[0..n]`  (2n vertices total)
/// but circles at different v-levels have different radii.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_cone_barrel_wire(
    topo: &mut Topology,
    cone: &brepkit_math::surfaces::ConicalSurface,
    verts: &[Point3],
    vertex_map: &mut HashMap<(i64, i64, i64), VertexId>,
    edge_map: &mut HashMap<(usize, usize), EdgeId>,
    resolution: f64,
    tol: Tolerance,
) -> Result<WireId, crate::OperationsError> {
    // Same layout as cylinder: bot[0..n/2] + top_reversed[0..n/2].
    let bot_seam_pos = verts[0];
    let top_seam_pos = verts[verts.len() - 1];

    // Compute v-levels from vertex positions on the cone.
    let (_, v_bot) = cone.project_point(bot_seam_pos);
    let (_, v_top) = cone.project_point(top_seam_pos);

    // Create Circle3D at each level.
    let r_bot = cone.radius_at(v_bot);
    let r_top = cone.radius_at(v_top);
    let bot_center = cone.apex() + cone.axis() * (v_bot * cone.half_angle().sin());
    let top_center = cone.apex() + cone.axis() * (v_top * cone.half_angle().sin());
    let bot_circle = brepkit_math::curves::Circle3D::new(bot_center, cone.axis(), r_bot)
        .map_err(crate::OperationsError::Math)?;
    let top_circle = brepkit_math::curves::Circle3D::new(top_center, cone.axis(), r_top)
        .map_err(crate::OperationsError::Math)?;

    // Create/lookup vertices at the seam points.
    let bot_vid = *vertex_map
        .entry(quantize_point(bot_seam_pos, resolution))
        .or_insert_with(|| topo.add_vertex(Vertex::new(bot_seam_pos, tol.linear)));
    let top_vid = *vertex_map
        .entry(quantize_point(top_seam_pos, resolution))
        .or_insert_with(|| topo.add_vertex(Vertex::new(top_seam_pos, tol.linear)));

    // Create/lookup closed Circle edges.
    let bot_edge = *edge_map
        .entry((bot_vid.index(), bot_vid.index()))
        .or_insert_with(|| {
            topo.add_edge(Edge::new(bot_vid, bot_vid, EdgeCurve::Circle(bot_circle)))
        });
    let top_edge = *edge_map
        .entry((top_vid.index(), top_vid.index()))
        .or_insert_with(|| {
            topo.add_edge(Edge::new(top_vid, top_vid, EdgeCurve::Circle(top_circle)))
        });

    // Create/lookup seam line edge.
    let seam_fwd = bot_vid.index() <= top_vid.index();
    let seam_key = if seam_fwd {
        (bot_vid.index(), top_vid.index())
    } else {
        (top_vid.index(), bot_vid.index())
    };
    let seam_edge = *edge_map.entry(seam_key).or_insert_with(|| {
        let (start, end) = if seam_fwd {
            (bot_vid, top_vid)
        } else {
            (top_vid, bot_vid)
        };
        topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
    });

    // Wire: bot_circle(fwd) -> seam(fwd) -> top_circle(rev) -> seam(rev)
    let wire = Wire::new(
        vec![
            OrientedEdge::new(bot_edge, true),
            OrientedEdge::new(seam_edge, seam_fwd),
            OrientedEdge::new(top_edge, false),
            OrientedEdge::new(seam_edge, !seam_fwd),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

/// Sample an `EdgeCurve` into N points.
pub(super) fn sample_edge_curve(curve: &EdgeCurve, n: usize) -> Vec<Point3> {
    match curve {
        EdgeCurve::Circle(c) => (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                c.evaluate(t)
            })
            .collect(),
        EdgeCurve::Ellipse(e) => (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                e.evaluate(t)
            })
            .collect(),
        EdgeCurve::NurbsCurve(nc) => {
            let (u0, u1) = nc.domain();
            // For closed curves (start ~ end), use n as divisor to avoid
            // duplicating the first point at t=u_max.
            let start_pt = nc.evaluate(u0);
            let end_pt = nc.evaluate(u1);
            // 1e-6 m: closure detection threshold — if start and end points are
            // within 1 micron, treat the NURBS curve as closed to avoid
            // duplicating the first point at t=u_max.
            let is_closed = (start_pt - end_pt).length() < 1e-6;
            let divisor = if is_closed { n } else { n - 1 };
            (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = u0 + (u1 - u0) * (i as f64) / (divisor as f64);
                    nc.evaluate(t)
                })
                .collect()
        }
        EdgeCurve::Line => vec![],
    }
}
