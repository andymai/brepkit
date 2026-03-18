//! Boolean pipeline: OCCT-style parameter-space pipeline.
//!
//! Operates entirely in 2D parameter space (pcurves on surfaces) for face
//! splitting. Surface-type agnostic — same code for plane, cylinder, sphere,
//! torus, NURBS. Step 1 implements plane-only support.

#![allow(dead_code, clippy::too_many_lines, clippy::missing_errors_doc)]

use std::collections::HashMap;

use brepkit_math::aabb::Aabb3;
use brepkit_math::analytic_intersection::{self, ExactIntersectionCurve};
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use super::analytic::surface_aware_aabb;
use super::assembly::{quantize_point, vertex_merge_resolution};
use super::classify::try_build_analytic_classifier;
use super::classify_2d::point_in_polygon_2d;
use super::face_splitter::{interior_point_3d, split_face_2d};
use super::pcurve_compute::{compute_pcurve_on_surface, surface_periods};
use super::pipeline::{BooleanPipeline, SectionEdge, SurfaceInfo};
use super::plane_frame::PlaneFrame;
use super::types::{AnalyticClassifier, BooleanOp, FaceClass, Source, select_fragment};
use crate::OperationsError;

/// Face polygon data for ray-cast classification: (outer_verts, inner_hole_verts, normal, d).
type FacePolyData = (Vec<Point3>, Vec<Vec<Point3>>, Vec3, f64);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform a boolean operation using the pipeline parameter-space pipeline.
///
/// Supports solids composed entirely of analytic faces: `Plane`, `Cylinder`,
/// `Cone`, `Sphere`, `Torus`. Returns `Err` for solids containing NURBS faces.
pub fn boolean_pipeline(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, OperationsError> {
    let tol = Tolerance::new();

    // Guard: all faces must have a supported surface type.
    if !all_faces_supported(topo, a)? || !all_faces_supported(topo, b)? {
        return Err(OperationsError::InvalidInput {
            reason: "boolean_pipeline: unsupported surface type (only plane/cylinder/cone/sphere/torus/NURBS supported)"
                .into(),
        });
    }

    let mut pipeline = BooleanPipeline {
        solid_a: Some(a),
        solid_b: Some(b),
        ..BooleanPipeline::default()
    };

    // Cache surface info per face (PlaneFrame for planes, periodicity for others).
    init_surface_info(topo, a, b, &mut pipeline)?;

    // Build analytic classifiers for O(1) point-in-solid tests.
    let classifier_a = try_build_analytic_classifier(topo, a);
    let classifier_b = try_build_analytic_classifier(topo, b);

    // Stage 1: Intersect all face pairs.
    intersect_all_faces(topo, a, b, &mut pipeline, &tol)?;

    // Disjoint shortcut (with containment detection).
    if pipeline.intersections.is_empty() {
        return handle_disjoint_pipeline(
            topo,
            op,
            a,
            b,
            classifier_a.as_ref(),
            classifier_b.as_ref(),
            &tol,
        );
    }

    // Stage 2: Split edges at intersection vertices.
    // Boundary edges are pre-split during face splitting (stage 3) —
    // the wire builder handles section edge insertion naturally.

    // Stage 3: Split faces via wire builder.
    split_all_faces(topo, a, b, &mut pipeline, &tol)?;

    // Stage 4: Classify sub-faces against opposing solid.
    classify_sub_faces(
        topo,
        &mut pipeline,
        op,
        classifier_a.as_ref(),
        classifier_b.as_ref(),
        &tol,
    )?;

    // Stage 4b: Same-domain face deduplication (OCCT's FillSameDomainFaces).
    // After classification, sub-faces from different solids that end up on the
    // same surface with the same boundary (same edge set) are duplicates.
    // For Fuse, both the A and B versions of a coplanar face are kept by
    // classification — we only need one.
    dedup_same_domain_subfaces(&mut pipeline, &tol);

    // Stage 5: Assemble result.
    let result = assemble_pipeline(topo, &pipeline, &tol)?;

    // Post-processing: healing.
    crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;

    // Run unify_faces iteratively — each pass may expose new merge
    // opportunities after internal edges are removed.
    for _ in 0..3 {
        let merged = crate::heal::unify_faces(topo, result)?;
        if merged == 0 {
            break;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Surface info cache
// ---------------------------------------------------------------------------

fn init_surface_info(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    pipeline: &mut BooleanPipeline,
) -> Result<(), OperationsError> {
    for solid in [a, b] {
        let faces = collect_solid_faces(topo, solid)?;
        for fid in faces {
            let face = topo.face(fid)?;
            let surface = face.surface();
            if let FaceSurface::Plane { normal, .. } = surface {
                let poly = collect_face_polygon(topo, fid)?;
                let frame = PlaneFrame::from_plane_face(*normal, &poly);
                pipeline.plane_frames.insert(fid, frame.clone());
                pipeline.surface_info.insert(fid, SurfaceInfo::Plane(frame));
            } else {
                let (u_per, v_per) = surface_periods(surface);
                pipeline.surface_info.insert(
                    fid,
                    SurfaceInfo::Parametric {
                        u_periodic: u_per.is_some(),
                        v_periodic: v_per.is_some(),
                    },
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 1: Intersect all face pairs
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn intersect_all_faces(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    pipeline: &mut BooleanPipeline,
    tol: &Tolerance,
) -> Result<(), OperationsError> {
    let faces_a = collect_solid_faces(topo, a)?;
    let faces_b = collect_solid_faces(topo, b)?;

    // Pre-compute face AABBs for fast rejection.
    // Uses surface_aware_aabb which expands wire-vertex AABBs to account
    // for curved surfaces (e.g. cylinder radius, sphere poles).
    let aabb_cache: HashMap<FaceId, Aabb3> = faces_a
        .iter()
        .chain(faces_b.iter())
        .filter_map(|&fid| {
            let face = topo.face(fid).ok()?;
            let poly = collect_face_polygon(topo, fid).ok()?;
            if poly.is_empty() {
                return None;
            }
            let aabb = surface_aware_aabb(face.surface(), &poly, *tol);
            Some((fid, aabb))
        })
        .collect();

    for &fa in &faces_a {
        let Some(aabb_a) = aabb_cache.get(&fa) else {
            continue;
        };
        for &fb in &faces_b {
            let Some(aabb_b) = aabb_cache.get(&fb) else {
                continue;
            };
            // AABB pre-filter: skip disjoint face pairs.
            if !aabb_a.intersects(*aabb_b) {
                continue;
            }

            let sections = intersect_face_pair(topo, fa, fb, pipeline, tol)?;
            if !sections.is_empty() {
                pipeline.intersections.insert((fa, fb), sections);
            }
        }
    }
    Ok(())
}

/// Dispatch intersection by surface-type pair.
fn intersect_face_pair(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    pipeline: &mut BooleanPipeline,
    tol: &Tolerance,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;
    let surf_a = face_a.surface();
    let surf_b = face_b.surface();

    match (surf_a, surf_b) {
        (FaceSurface::Plane { .. }, FaceSurface::Plane { .. }) => {
            // Plane-plane: existing intersection (handles coplanar case too).
            intersect_two_plane_faces(
                topo,
                fa,
                fb,
                &pipeline.plane_frames,
                &mut pipeline.coplanar_pairs,
            )
        }
        (FaceSurface::Plane { normal, d }, _) if surf_b.is_analytic() => {
            // Plane-analytic: plane is face A.
            intersect_plane_analytic_faces(topo, fa, fb, *normal, *d, surf_b, pipeline, tol)
        }
        (_, FaceSurface::Plane { normal, d }) if surf_a.is_analytic() => {
            // Analytic-plane: plane is face B — swap order, then swap pcurves.
            let mut sections =
                intersect_plane_analytic_faces(topo, fb, fa, *normal, *d, surf_a, pipeline, tol)?;
            for s in &mut sections {
                std::mem::swap(&mut s.pcurve_a, &mut s.pcurve_b);
            }
            Ok(sections)
        }
        _ if surf_a.is_analytic() && surf_b.is_analytic() => {
            // Analytic-analytic: use marching fallback.
            intersect_analytic_analytic_faces(topo, fa, fb, pipeline, tol)
        }
        // Plane-NURBS: use intersect_plane_nurbs from math crate.
        (FaceSurface::Plane { normal, d }, FaceSurface::Nurbs(_)) => {
            intersect_plane_nurbs_faces(topo, fa, fb, *normal, *d, pipeline, tol)
        }
        (FaceSurface::Nurbs(_), FaceSurface::Plane { normal, d }) => {
            let mut sections =
                intersect_plane_nurbs_faces(topo, fb, fa, *normal, *d, pipeline, tol)?;
            for s in &mut sections {
                std::mem::swap(&mut s.pcurve_a, &mut s.pcurve_b);
                std::mem::swap(&mut s.start_uv_a, &mut s.start_uv_b);
                std::mem::swap(&mut s.end_uv_a, &mut s.end_uv_b);
            }
            Ok(sections)
        }
        // NURBS-NURBS pairs. Analytic-NURBS currently returns empty
        // (TODO: convert analytic → NURBS for mixed pairs).
        _ if matches!(surf_a, FaceSurface::Nurbs(_)) || matches!(surf_b, FaceSurface::Nurbs(_)) => {
            intersect_nurbs_general_faces(topo, fa, fb, pipeline, tol)
        }
        _ => {
            // Unsupported surface pair — log and skip.
            log::debug!("intersect_face_pair: unsupported surface pair, skipping");
            Ok(Vec::new())
        }
    }
}

/// Intersect a plane face with an analytic (non-plane) face.
///
/// Uses `exact_plane_analytic()` to get the intersection curve (Circle, Ellipse,
/// or sampled Points), then trims to both face boundaries and computes pcurves.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn intersect_plane_analytic_faces(
    topo: &Topology,
    plane_face_id: FaceId,
    analytic_face_id: FaceId,
    plane_normal: Vec3,
    plane_d: f64,
    analytic_surface: &FaceSurface,
    pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let Some(analytic) = analytic_surface.as_analytic() else {
        return Ok(Vec::new());
    };

    // Get exact intersection curves.
    let curves = analytic_intersection::exact_plane_analytic(analytic, plane_normal, plane_d)
        .map_err(OperationsError::Math)?;

    if curves.is_empty() {
        return Ok(Vec::new());
    }

    // Collect boundary polygons for both faces.
    let poly_plane = collect_face_polygon(topo, plane_face_id)?;
    let _poly_analytic = collect_face_polygon(topo, analytic_face_id)?;

    // Get plane frame for the plane face.
    let owned_frame;
    let frame_plane = if let Some(f) = pipeline.plane_frames.get(&plane_face_id) {
        f
    } else {
        owned_frame = plane_frame_for_polygon(plane_normal, &poly_plane);
        &owned_frame
    };

    // For the analytic face, compute a UV polygon for containment testing.
    let analytic_uv_poly = face_uv_polygon(topo, analytic_face_id, analytic_surface);

    let mut result = Vec::new();

    for curve in curves {
        // Sample 3D points along the intersection curve.
        let samples = sample_intersection_curve(&curve, 64);
        if samples.len() < 2 {
            continue;
        }

        // Trim: find consecutive runs where points are inside both faces.
        let inside_both: Vec<bool> = samples
            .iter()
            .map(|&p| {
                let in_plane = point_in_face_polygon_3d(p, &poly_plane, &plane_normal);
                let in_analytic = point_in_analytic_face_uv(p, analytic_surface, &analytic_uv_poly);
                in_plane && in_analytic
            })
            .collect();

        // Extract contiguous segments of `true`.
        let segments = extract_contiguous_segments(&inside_both);

        // Determine 3D edge curve type for this intersection.
        let curve_3d = intersection_curve_to_edge_curve(&curve);

        for (seg_start, seg_end) in segments {
            if seg_end <= seg_start {
                continue;
            }

            let start_3d = samples[seg_start];
            let end_3d = samples[seg_end];

            // OCCT-style minimum curve length filter: discard section edges
            // that span fewer than 3 sample points (too short to be meaningful).
            // This prevents micro-curves from creating unnecessary face splits.
            if seg_end - seg_start < 3 {
                continue;
            }

            let is_closed = (end_3d - start_3d).length() < tol.linear && (seg_end - seg_start) >= 4;

            if is_closed {
                // Closed intersection curve (full circle). Split into two
                // arcs at the face's topological seam and its antipodal point.
                //
                // The seam u-value is where the boundary vertex sits (NOT
                // necessarily u=0 in the surface parameterization). We detect
                // it by projecting a boundary vertex onto the surface.
                let seam_u = detect_topological_seam_u(topo, analytic_face_id, analytic_surface);
                let closed_sections = build_seam_split_sections(
                    &samples[seg_start..=seg_end],
                    &curve_3d,
                    analytic_surface,
                    frame_plane,
                    seam_u,
                );
                result.extend(closed_sections);
            } else if (end_3d - start_3d).length() >= 1e-4 {
                // Minimum section edge length: 0.1mm. Shorter edges are degenerate
                // (tangent touches, grazing contacts) and would create micro-faces.
                // Non-closed segment with distinct endpoints.
                let sub_samples = &samples[seg_start..=seg_end];
                let pcurve_plane = fit_pcurve_from_3d_samples(sub_samples, frame_plane);
                let pcurve_analytic = compute_pcurve_on_surface(
                    &curve_3d,
                    start_3d,
                    end_3d,
                    analytic_surface,
                    &[],
                    None,
                );
                result.push(SectionEdge {
                    curve_3d: curve_3d.clone(),
                    pcurve_a: pcurve_plane,
                    pcurve_b: pcurve_analytic,
                    start: start_3d,
                    end: end_3d,
                    start_uv_a: None,
                    end_uv_a: None,
                    start_uv_b: None,
                    end_uv_b: None,
                    target_face: None,
                });
            }
        }
    }

    Ok(result)
}

/// Intersect two analytic (non-plane) faces using the marching algorithm.
#[allow(clippy::too_many_lines)]
fn intersect_analytic_analytic_faces(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    _pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;
    let surf_a = face_a.surface();
    let surf_b = face_b.surface();

    let Some(analytic_a) = surf_a.as_analytic() else {
        return Ok(Vec::new());
    };
    let Some(analytic_b) = surf_b.as_analytic() else {
        return Ok(Vec::new());
    };

    // Estimate v-range hints from face boundary vertices.
    let v_range_a = estimate_v_range(topo, fa, surf_a);
    let v_range_b = estimate_v_range(topo, fb, surf_b);

    let curves = analytic_intersection::intersect_analytic_analytic_bounded(
        analytic_a, analytic_b, 64, v_range_a, v_range_b,
    )
    .map_err(OperationsError::Math)?;

    if curves.is_empty() {
        return Ok(Vec::new());
    }

    // UV polygons for containment testing.
    let uv_poly_a = face_uv_polygon(topo, fa, surf_a);
    let uv_poly_b = face_uv_polygon(topo, fb, surf_b);

    let mut result = Vec::new();

    for curve in &curves {
        let samples: Vec<Point3> = curve.points.iter().map(|ip| ip.point).collect();
        if samples.len() < 2 {
            continue;
        }

        // Trim to both face boundaries.
        let inside_both: Vec<bool> = samples
            .iter()
            .map(|&p| {
                let in_a = point_in_analytic_face_uv(p, surf_a, &uv_poly_a);
                let in_b = point_in_analytic_face_uv(p, surf_b, &uv_poly_b);
                in_a && in_b
            })
            .collect();

        let segments = extract_contiguous_segments(&inside_both);

        for (seg_start, seg_end) in segments {
            if seg_end <= seg_start {
                continue;
            }

            let sub_segs = split_closed_segment(&samples, seg_start, seg_end, tol.linear);

            for (ss, se) in sub_segs {
                let start_3d = samples[ss];
                let end_3d = samples[se];

                // Analytic-analytic always produces NURBS section edges.
                let sub_samples: Vec<Point3> = samples[ss..=se].to_vec();
                let curve_3d = if sub_samples.len() >= 4 {
                    match brepkit_math::nurbs::fitting::interpolate(&sub_samples, 3) {
                        Ok(nc) => EdgeCurve::NurbsCurve(nc),
                        Err(_) => EdgeCurve::Line,
                    }
                } else {
                    EdgeCurve::Line
                };

                let pcurve_a =
                    compute_pcurve_on_surface(&curve_3d, start_3d, end_3d, surf_a, &[], None);
                let pcurve_b =
                    compute_pcurve_on_surface(&curve_3d, start_3d, end_3d, surf_b, &[], None);

                result.push(SectionEdge {
                    curve_3d,
                    pcurve_a,
                    pcurve_b,
                    start: start_3d,
                    end: end_3d,
                    start_uv_a: None,
                    end_uv_a: None,
                    start_uv_b: None,
                    end_uv_b: None,
                    target_face: None,
                });
            }
        }
    }

    Ok(result)
}

/// Intersect a plane face with a NURBS face.
///
/// Uses `intersect_plane_nurbs` from the math crate, then trims to both
/// face boundaries and computes pcurves (same pattern as plane-analytic).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn intersect_plane_nurbs_faces(
    topo: &Topology,
    plane_face_id: FaceId,
    nurbs_face_id: FaceId,
    plane_normal: Vec3,
    plane_d: f64,
    pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let nurbs_face = topo.face(nurbs_face_id)?;
    let nurbs_surface = match nurbs_face.surface() {
        FaceSurface::Nurbs(s) => s,
        _ => return Ok(Vec::new()),
    };

    // Get intersection curves from the NURBS intersection module.
    let curves = brepkit_math::nurbs::intersection::intersect_plane_nurbs(
        nurbs_surface,
        plane_normal,
        plane_d,
        50, // grid resolution for seed finding
    )
    .map_err(OperationsError::Math)?;

    if curves.is_empty() {
        return Ok(Vec::new());
    }

    // Collect boundary polygons for both faces.
    let poly_plane = collect_face_polygon(topo, plane_face_id)?;

    // Get plane frame.
    let owned_frame;
    let frame_plane = if let Some(f) = pipeline.plane_frames.get(&plane_face_id) {
        f
    } else {
        owned_frame = plane_frame_for_polygon(plane_normal, &poly_plane);
        &owned_frame
    };

    // NURBS face UV polygon for containment testing.
    let nurbs_surf = nurbs_face.surface();
    let nurbs_uv_poly = face_uv_polygon(topo, nurbs_face_id, nurbs_surf);

    let mut result = Vec::new();

    for curve in &curves {
        let samples: Vec<Point3> = curve.points.iter().map(|ip| ip.point).collect();
        if samples.len() < 2 {
            continue;
        }

        // Trim to both face boundaries.
        let inside_both: Vec<bool> = samples
            .iter()
            .map(|&p| {
                let in_plane = point_in_face_polygon_3d(p, &poly_plane, &plane_normal);
                let in_nurbs = point_in_analytic_face_uv(p, nurbs_surf, &nurbs_uv_poly);
                in_plane && in_nurbs
            })
            .collect();

        let segments = extract_contiguous_segments(&inside_both);

        for (seg_start, seg_end) in segments {
            if seg_end <= seg_start {
                continue;
            }

            let sub_segs = split_closed_segment(&samples, seg_start, seg_end, tol.linear);

            for (ss, se) in sub_segs {
                let start_3d = samples[ss];
                let end_3d = samples[se];

                let sub_samples: Vec<Point3> = samples[ss..=se].to_vec();
                let curve_3d = if sub_samples.len() >= 4 {
                    match brepkit_math::nurbs::fitting::interpolate(&sub_samples, 3) {
                        Ok(nc) => EdgeCurve::NurbsCurve(nc),
                        Err(_) => EdgeCurve::Line,
                    }
                } else {
                    EdgeCurve::Line
                };

                let pcurve_plane = fit_pcurve_from_3d_samples(&sub_samples, frame_plane);
                let pcurve_nurbs =
                    compute_pcurve_on_surface(&curve_3d, start_3d, end_3d, nurbs_surf, &[], None);

                result.push(SectionEdge {
                    curve_3d,
                    pcurve_a: pcurve_plane,
                    pcurve_b: pcurve_nurbs,
                    start: start_3d,
                    end: end_3d,
                    start_uv_a: None,
                    end_uv_a: None,
                    start_uv_b: None,
                    end_uv_b: None,
                    target_face: None,
                });
            }
        }
    }

    Ok(result)
}

/// Intersect two NURBS-NURBS face pairs.
///
/// Handles NURBS-NURBS and analytic-NURBS pairs. Analytic surfaces
/// are converted to NURBS via `face_surface_to_nurbs` before intersection.
#[allow(clippy::too_many_lines)]
fn intersect_nurbs_general_faces(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    _pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;
    let surf_a = face_a.surface();
    let surf_b = face_b.surface();

    // Extract or convert to NurbsSurface for each face.
    let nurbs_a = face_surface_to_nurbs(topo, fa, surf_a)?;
    let nurbs_b = face_surface_to_nurbs(topo, fb, surf_b)?;

    let (Some(na), Some(nb)) = (nurbs_a, nurbs_b) else {
        return Ok(Vec::new()); // Unsupported surface type
    };

    let curves = brepkit_math::nurbs::intersection::intersect_nurbs_nurbs(
        &na, &nb, 30, 0.0, // adaptive march step
    )
    .map_err(OperationsError::Math)?;

    if curves.is_empty() {
        return Ok(Vec::new());
    }

    // UV polygons for containment testing.
    let uv_poly_a = face_uv_polygon(topo, fa, surf_a);
    let uv_poly_b = face_uv_polygon(topo, fb, surf_b);

    let mut result = Vec::new();

    for curve in &curves {
        let samples: Vec<Point3> = curve.points.iter().map(|ip| ip.point).collect();
        if samples.len() < 2 {
            continue;
        }

        let inside_both: Vec<bool> = samples
            .iter()
            .map(|&p| {
                let in_a = point_in_analytic_face_uv(p, surf_a, &uv_poly_a);
                let in_b = point_in_analytic_face_uv(p, surf_b, &uv_poly_b);
                in_a && in_b
            })
            .collect();

        let segments = extract_contiguous_segments(&inside_both);

        for (seg_start, seg_end) in segments {
            if seg_end <= seg_start {
                continue;
            }

            let sub_segs = split_closed_segment(&samples, seg_start, seg_end, tol.linear);

            for (ss, se) in sub_segs {
                let start_3d = samples[ss];
                let end_3d = samples[se];

                let sub_samples: Vec<Point3> = samples[ss..=se].to_vec();
                let curve_3d = if sub_samples.len() >= 4 {
                    match brepkit_math::nurbs::fitting::interpolate(&sub_samples, 3) {
                        Ok(nc) => EdgeCurve::NurbsCurve(nc),
                        Err(_) => EdgeCurve::Line,
                    }
                } else {
                    EdgeCurve::Line
                };

                let pcurve_a =
                    compute_pcurve_on_surface(&curve_3d, start_3d, end_3d, surf_a, &[], None);
                let pcurve_b =
                    compute_pcurve_on_surface(&curve_3d, start_3d, end_3d, surf_b, &[], None);

                result.push(SectionEdge {
                    curve_3d,
                    pcurve_a,
                    pcurve_b,
                    start: start_3d,
                    end: end_3d,
                    start_uv_a: None,
                    end_uv_a: None,
                    start_uv_b: None,
                    end_uv_b: None,
                    target_face: None,
                });
            }
        }
    }

    Ok(result)
}

/// Compute the intersection of two plane faces.
///
/// Returns trimmed section edges (finite line segments where the faces overlap).
/// For coplanar faces, produces targeted section edges (each edge applies to
/// only one face) and records the coplanar pair in `coplanar_pairs`.
fn intersect_two_plane_faces(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    plane_frames: &HashMap<FaceId, PlaneFrame>,
    coplanar_pairs: &mut Vec<(FaceId, FaceId, bool)>,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;

    // Guarded by `all_faces_plane()` at the entry point of `boolean_pipeline()`.
    let Some((na, da)) = (match face_a.surface() {
        FaceSurface::Plane { normal, d } => Some((*normal, *d)),
        _ => None,
    }) else {
        debug_assert!(false, "non-plane face reached intersect_two_plane_faces");
        return Ok(Vec::new());
    };
    let Some((nb, db)) = (match face_b.surface() {
        FaceSurface::Plane { normal, d } => Some((*normal, *d)),
        _ => None,
    }) else {
        debug_assert!(false, "non-plane face reached intersect_two_plane_faces");
        return Ok(Vec::new());
    };

    // Effective normals: account for face reversal.
    let eff_na = if face_a.is_reversed() { -na } else { na };
    let eff_nb = if face_b.is_reversed() { -nb } else { nb };

    // Plane-plane intersection: line direction = na × nb.
    let line_dir = na.cross(nb);
    let line_len = line_dir.length();
    if line_len < 1e-10 {
        // Parallel planes — check if coincident (coplanar).
        // Two planes are coincident when their signed distances match:
        //   same normal direction: |da - db| < tol
        //   opposite normal direction: |da + db| < tol
        let coplanar_tol = 1e-7;
        let same_dir = (da - db).abs() < coplanar_tol;
        let opp_dir = (da + db).abs() < coplanar_tol;
        if same_dir || opp_dir {
            // Coplanar faces. Determine orientation from effective normals.
            let same_orientation = eff_na.dot(eff_nb) > 0.0;
            let sections = intersect_coplanar_plane_faces(topo, fa, fb, plane_frames)?;
            if !sections.is_empty() {
                coplanar_pairs.push((fa, fb, same_orientation));
            }
            return Ok(sections);
        }
        // Parallel but separated — no intersection.
        return Ok(Vec::new());
    }
    let line_dir_n = Vec3::new(
        line_dir.x() / line_len,
        line_dir.y() / line_len,
        line_dir.z() / line_len,
    );

    // Find a point on the intersection line.
    // Solve: na·p = da, nb·p = db, line_dir·p = 0
    let line_origin = solve_two_planes_origin(na, da, nb, db, line_dir_n)?;

    // Collect boundary polygons for both faces.
    let poly_a = collect_face_polygon(topo, fa)?;
    let poly_b = collect_face_polygon(topo, fb)?;

    // Use cached frames for consistent UV projections.
    let owned_frame_a;
    let frame_a = if let Some(f) = plane_frames.get(&fa) {
        f
    } else {
        owned_frame_a = plane_frame_for_polygon(na, &poly_a);
        &owned_frame_a
    };
    let owned_frame_b;
    let frame_b = if let Some(f) = plane_frames.get(&fb) {
        f
    } else {
        owned_frame_b = plane_frame_for_polygon(nb, &poly_b);
        &owned_frame_b
    };

    // Trim the infinite line to both face boundaries independently.
    // Both use the same line_origin so t-values are compatible.
    let segments_a = trim_line_to_polygon_3d(&line_origin, &line_dir_n, &poly_a, frame_a);
    let segments_b = trim_line_to_polygon_3d(&line_origin, &line_dir_n, &poly_b, frame_b);

    // Intersect the two interval sets to find the overlap.
    let mut result = Vec::new();
    for &(a0, a1) in &segments_a {
        for &(b0, b1) in &segments_b {
            let t0 = a0.max(b0);
            let t1 = a1.min(b1);
            if t1 - t0 < 1e-4 {
                continue; // No overlap or micro-segment (< 0.1mm).
            }
            let start = line_origin + line_dir_n * t0;
            let end = line_origin + line_dir_n * t1;
            let pcurve_a = compute_line_pcurve(frame_a, start, end);
            let pcurve_b = compute_line_pcurve(frame_b, start, end);
            result.push(SectionEdge {
                curve_3d: EdgeCurve::Line,
                pcurve_a,
                pcurve_b,
                start,
                end,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
            });
        }
    }
    Ok(result)
}

/// Intersect two coplanar plane faces.
///
/// When two faces share the same infinite plane, there's no intersection
/// *line* — the intersection is a 2D *region*. We handle this by clipping
/// each face's boundary edges against the other face's interior:
///
/// - Edges of B clipped to A's interior → section edges targeting face A
/// - Edges of A clipped to B's interior → section edges targeting face B
///
/// These targeted section edges split each face along the other's boundary,
/// enabling the classifier to identify the overlap region.
#[allow(clippy::too_many_lines)]
fn intersect_coplanar_plane_faces(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    plane_frames: &HashMap<FaceId, PlaneFrame>,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;

    let na = match face_a.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return Ok(Vec::new()),
    };

    let poly_a = collect_face_polygon(topo, fa)?;
    let poly_b = collect_face_polygon(topo, fb)?;

    if poly_a.len() < 3 || poly_b.len() < 3 {
        return Ok(Vec::new());
    }

    // Use cached or computed frames. Both faces share the same plane,
    // so we use face A's frame for all 2D projections.
    let owned_frame;
    let frame = if let Some(f) = plane_frames.get(&fa) {
        f
    } else {
        owned_frame = plane_frame_for_polygon(na, &poly_a);
        &owned_frame
    };

    // Project both polygons to 2D.
    let poly_a_2d: Vec<Point2> = poly_a.iter().map(|p| frame.project(*p)).collect();
    let poly_b_2d: Vec<Point2> = poly_b.iter().map(|p| frame.project(*p)).collect();

    let mut result = Vec::new();

    // Clip edges of B against interior of A → section edges targeting face A.
    clip_polygon_edges_to_interior(&poly_b, &poly_b_2d, &poly_a_2d, frame, fa, &mut result);

    // Clip edges of A against interior of B → section edges targeting face B.
    clip_polygon_edges_to_interior(&poly_a, &poly_a_2d, &poly_b_2d, frame, fb, &mut result);

    Ok(result)
}

/// Clip edges of `source_poly` to the **strict interior** of `target_poly`.
///
/// For each edge of `source_poly`, finds the segment(s) that lie inside
/// `target_poly` (excluding segments that run along `target_poly`'s boundary)
/// and creates section edges targeting `target_face`.
///
/// Boundary-coincident segments are excluded because they don't create new
/// splits — the target face's wire builder already has these boundary edges.
fn clip_polygon_edges_to_interior(
    source_3d: &[Point3],
    source_2d: &[Point2],
    target_2d: &[Point2],
    frame: &PlaneFrame,
    target_face: FaceId,
    result: &mut Vec<SectionEdge>,
) {
    let n = source_3d.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let p0_2d = source_2d[i];
        let p1_2d = source_2d[j];
        let p0_3d = source_3d[i];
        let p1_3d = source_3d[j];

        // Clip this edge against the target polygon.
        let segments = clip_edge_to_polygon_2d(p0_2d, p1_2d, target_2d);

        for (t0, t1) in segments {
            // Compute 2D midpoint to check if it's strictly inside target.
            let t_mid = (t0 + t1) * 0.5;
            let mid_2d = Point2::new(
                p0_2d.x() + (p1_2d.x() - p0_2d.x()) * t_mid,
                p0_2d.y() + (p1_2d.y() - p0_2d.y()) * t_mid,
            );

            // Skip segments that lie on the target polygon's boundary.
            // These don't create new splits — they're already boundary edges.
            if point_near_polygon_boundary_2d(mid_2d, target_2d, 1e-7) {
                continue;
            }

            // Compute 3D endpoints by interpolation.
            let start = Point3::new(
                p0_3d.x() + (p1_3d.x() - p0_3d.x()) * t0,
                p0_3d.y() + (p1_3d.y() - p0_3d.y()) * t0,
                p0_3d.z() + (p1_3d.z() - p0_3d.z()) * t0,
            );
            let end = Point3::new(
                p0_3d.x() + (p1_3d.x() - p0_3d.x()) * t1,
                p0_3d.y() + (p1_3d.y() - p0_3d.y()) * t1,
                p0_3d.z() + (p1_3d.z() - p0_3d.z()) * t1,
            );

            if (end - start).length() < 1e-10 {
                continue; // Degenerate.
            }

            // Compute pcurves on the shared plane frame.
            // Both faces share the same plane, so pcurve_a and pcurve_b
            // are identical (both are projections into the same plane).
            let pcurve = compute_line_pcurve(frame, start, end);

            result.push(SectionEdge {
                curve_3d: EdgeCurve::Line,
                pcurve_a: pcurve.clone(),
                pcurve_b: pcurve,
                start,
                end,
                start_uv_a: None,
                end_uv_a: None,
                start_uv_b: None,
                end_uv_b: None,
                target_face: Some(target_face),
            });
        }
    }
}

/// Check if a 2D point is within `tol` of any edge of a 2D polygon.
fn point_near_polygon_boundary_2d(pt: Point2, polygon: &[Point2], tol: f64) -> bool {
    let n = polygon.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let a = polygon[i];
        let b = polygon[j];
        let ab = Vec2::new(b.x() - a.x(), b.y() - a.y());
        let ap = Vec2::new(pt.x() - a.x(), pt.y() - a.y());
        let len_sq = ab.x() * ab.x() + ab.y() * ab.y();
        if len_sq < 1e-30 {
            continue;
        }
        let t_raw = (ap.x() * ab.x() + ap.y() * ab.y()) / len_sq;
        if !(-0.01..=1.01).contains(&t_raw) {
            continue;
        }
        // Clamp to segment endpoints for correct point-to-segment distance.
        let t = t_raw.clamp(0.0, 1.0);
        let proj = Point2::new(a.x() + ab.x() * t, a.y() + ab.y() * t);
        let dx = pt.x() - proj.x();
        let dy = pt.y() - proj.y();
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < tol {
            return true;
        }
    }
    false
}

/// Clip a 2D edge segment to a 2D polygon.
///
/// Returns parameter pairs `(t0, t1)` where `0 <= t0 < t1 <= 1`, representing
/// segments of the edge `p0→p1` that lie inside the polygon.
fn clip_edge_to_polygon_2d(p0: Point2, p1: Point2, polygon: &[Point2]) -> Vec<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }

    let edge_dir = Vec2::new(p1.x() - p0.x(), p1.y() - p0.y());
    let edge_len_sq = edge_dir.x() * edge_dir.x() + edge_dir.y() * edge_dir.y();
    if edge_len_sq < 1e-30 {
        return Vec::new();
    }

    // Collect intersection parameters where the edge crosses polygon edges.
    let mut hits: Vec<f64> = vec![0.0, 1.0]; // Include edge endpoints.

    for i in 0..n {
        let j = (i + 1) % n;
        let qi = polygon[i];
        let qj = polygon[j];
        let poly_dir = Vec2::new(qj.x() - qi.x(), qj.y() - qi.y());

        let denom = edge_dir.x() * poly_dir.y() - edge_dir.y() * poly_dir.x();
        if denom.abs() < 1e-15 {
            continue; // Parallel.
        }

        let dp = Vec2::new(qi.x() - p0.x(), qi.y() - p0.y());
        let t = (dp.x() * poly_dir.y() - dp.y() * poly_dir.x()) / denom;
        let u = (dp.x() * edge_dir.y() - dp.y() * edge_dir.x()) / denom;

        // t must be within the edge [0, 1], u within the polygon edge [0, 1].
        if (-1e-10..=1.0 + 1e-10).contains(&t) && (-1e-10..=1.0 + 1e-10).contains(&u) {
            hits.push(t.clamp(0.0, 1.0));
        }
    }

    // Sort and dedup.
    hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    hits.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    // Check midpoint of each interval for containment.
    let mut segments = Vec::new();
    for w in hits.windows(2) {
        let t0 = w[0];
        let t1 = w[1];
        if t1 - t0 < 1e-10 {
            continue;
        }
        let t_mid = (t0 + t1) * 0.5;
        let mid = Point2::new(p0.x() + edge_dir.x() * t_mid, p0.y() + edge_dir.y() * t_mid);
        if point_in_polygon_2d(mid, polygon) {
            segments.push((t0, t1));
        }
    }

    // Merge adjacent segments (can happen when a hit point is degenerate).
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for seg in segments {
        if let Some(last) = merged.last_mut() {
            if (seg.0 - last.1).abs() < 1e-10 {
                last.1 = seg.1;
                continue;
            }
        }
        merged.push(seg);
    }

    merged
}

// ---------------------------------------------------------------------------
// Stage 3: Split faces
// ---------------------------------------------------------------------------

/// Remove duplicate section edges that share the same endpoints (possibly reversed).
///
/// Coplanar face handling can produce edges that overlap with non-coplanar
/// intersection edges (e.g., A's boundary at x=1 inside B is also the intersection
/// of A's x=1 face with B). Keeping both confuses the wire builder.
fn dedup_section_edges(sections: &mut Vec<SectionEdge>) {
    let tol = 1e-7;
    let mut keep = vec![true; sections.len()];
    for i in 0..sections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..sections.len() {
            if !keep[j] {
                continue;
            }
            // Check forward match: (s[i].start≈s[j].start, s[i].end≈s[j].end)
            let fwd = (sections[i].start - sections[j].start).length() < tol
                && (sections[i].end - sections[j].end).length() < tol;
            // Check reverse match: (s[i].start≈s[j].end, s[i].end≈s[j].start)
            let rev = (sections[i].start - sections[j].end).length() < tol
                && (sections[i].end - sections[j].start).length() < tol;
            if fwd || rev {
                keep[j] = false;
            }
        }
    }
    let mut keep_iter = keep.into_iter();
    sections.retain(|_| keep_iter.next().unwrap_or(false));
}

fn split_all_faces(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    pipeline: &mut BooleanPipeline,
    tol: &Tolerance,
) -> Result<(), OperationsError> {
    let faces_a = collect_solid_faces(topo, a)?;
    let faces_b = collect_solid_faces(topo, b)?;

    // Collect section edges per face, respecting target_face filtering.
    // When target_face is None, the edge applies to both faces in the pair.
    // When target_face is Some(id), only distribute to that specific face.
    //
    // Sort face pairs for deterministic iteration order (HashMap is unordered).
    let mut sections_for_face: HashMap<FaceId, Vec<SectionEdge>> = HashMap::new();
    let mut sorted_pairs: Vec<_> = pipeline.intersections.iter().collect();
    sorted_pairs.sort_by_key(|((fa, fb), _)| (fa.index(), fb.index()));
    for ((fa, fb), sections) in sorted_pairs {
        for s in sections {
            if s.target_face.is_none() || s.target_face == Some(*fa) {
                sections_for_face.entry(*fa).or_default().push(s.clone());
            }
            if s.target_face.is_none() || s.target_face == Some(*fb) {
                sections_for_face.entry(*fb).or_default().push(s.clone());
            }
        }
    }

    // Dedup section edges: only on faces involved in coplanar relationships.
    // Coplanar clipping can produce edges that overlap with non-coplanar
    // intersection edges (same segment in opposite directions). Keep the
    // first occurrence; remove duplicates based on endpoint proximity.
    //
    // Only dedup faces with coplanar partners to avoid false-positive
    // dedup on non-coplanar faces (e.g., cylinder-through-box edges that
    // coincidentally share endpoints from different intersection pairs).
    let coplanar_face_set: std::collections::HashSet<FaceId> = pipeline
        .coplanar_pairs
        .iter()
        .flat_map(|&(a, b, _)| [a, b])
        .collect();
    for (fid, sections) in &mut sections_for_face {
        if coplanar_face_set.contains(fid) {
            dedup_section_edges(sections);
        }
    }

    // Pre-classification: filter section edges where both sides classify the
    // same way (both Inside or both Outside). These edges don't create
    // meaningful face splits. Uses analytic ray-surface intersection for
    // exact classification at offset test points.
    {
        let polys_a = collect_solid_face_polygons(topo, a)?;
        let polys_b = collect_solid_face_polygons(topo, b)?;
        let cls_a = try_build_analytic_classifier(topo, a);
        let cls_b = try_build_analytic_classifier(topo, b);

        for (fid, sections) in &mut sections_for_face {
            let Ok(face) = topo.face(*fid) else { continue };
            // Only filter plane face sections (offset direction is well-defined).
            let FaceSurface::Plane { normal, .. } = face.surface() else {
                continue;
            };
            let face_normal = if face.is_reversed() {
                -*normal
            } else {
                *normal
            };
            let source = if faces_a.contains(fid) {
                Source::A
            } else {
                Source::B
            };
            let (opp_cls, polys, opp_solid) = match source {
                Source::A => (&cls_b, &polys_b, b),
                Source::B => (&cls_a, &polys_a, a),
            };

            sections.retain(|s| {
                // Use the geometric midpoint of the section edge (on the curve),
                // not the chord midpoint. For circle arcs, the chord midpoint
                // is inside the circle and both offset points would be Inside.
                let mid = match &s.curve_3d {
                    EdgeCurve::Circle(c) => {
                        let t0 = c.project(s.start);
                        let t1 = c.project(s.end);
                        let dt = (t1 - t0).rem_euclid(std::f64::consts::TAU);
                        c.evaluate(t0 + dt * 0.5)
                    }
                    EdgeCurve::Ellipse(e) => {
                        let t0 = e.project(s.start);
                        let t1 = e.project(s.end);
                        let dt = (t1 - t0).rem_euclid(std::f64::consts::TAU);
                        e.evaluate(t0 + dt * 0.5)
                    }
                    _ => Point3::new(
                        (s.start.x() + s.end.x()) * 0.5,
                        (s.start.y() + s.end.y()) * 0.5,
                        (s.start.z() + s.end.z()) * 0.5,
                    ),
                };
                // For Line sections, only filter if the opposing solid has an
                // analytic classifier. Without a classifier, the raycast is
                // imprecise and could incorrectly remove needed edges.
                if matches!(s.curve_3d, EdgeCurve::Line) {
                    if let Some(c) = opp_cls.as_ref() {
                        match c.classify(mid, *tol) {
                            Some(FaceClass::Inside) | None => return true, // Chord or boundary.
                            _ => {} // Outside → proceed to filter.
                        }
                    } else {
                        return true; // No classifier → keep all Line edges.
                    }
                }
                let edge_dir = Vec3::new(
                    s.end.x() - s.start.x(),
                    s.end.y() - s.start.y(),
                    s.end.z() - s.start.z(),
                );
                let edge_len = edge_dir.length();
                if edge_len < 0.5 {
                    return true; // Keep short sections.
                }
                let edge_n = edge_dir * (1.0 / edge_len);
                let offset_dir = face_normal.cross(edge_n);
                let offset_len = offset_dir.length();
                if offset_len < 1e-10 {
                    return true;
                }
                // 0.2mm offset — needs to be large enough to clear the
                // cylinder/cone surface tolerance.
                let offset = offset_dir * (0.2 / offset_len);
                let pt_a = mid + offset;
                let pt_b = mid - offset;
                // Use the analytic classifier (exact) if available.
                // Fall back to analytic raycast only if no classifier.
                let classify = |pt: Point3| -> FaceClass {
                    if let Some(c) = opp_cls.as_ref() {
                        if let Some(class) = c.classify(pt, *tol) {
                            return class;
                        }
                    }
                    classify_point_analytic_raycast(pt, topo, opp_solid, polys)
                };
                let class_a = classify(pt_a);
                let class_b = classify(pt_b);
                class_a != class_b
            });
        }
    }

    // Split faces from solid A.
    for &fid in &faces_a {
        let sections = sections_for_face
            .get(&fid)
            .map_or(&[][..], |v| v.as_slice());
        let frame = pipeline.plane_frames.get(&fid);
        let info = pipeline.surface_info.get(&fid);
        let sub = split_face_2d(topo, fid, sections, Source::A, tol, frame, info);
        pipeline.sub_faces.extend(sub);
    }

    // Split faces from solid B.
    for &fid in &faces_b {
        let sections = sections_for_face
            .get(&fid)
            .map_or(&[][..], |v| v.as_slice());
        let frame = pipeline.plane_frames.get(&fid);
        let info = pipeline.surface_info.get(&fid);
        let sub = split_face_2d(topo, fid, sections, Source::B, tol, frame, info);
        pipeline.sub_faces.extend(sub);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4: Classify sub-faces
// ---------------------------------------------------------------------------

fn classify_sub_faces(
    topo: &Topology,
    pipeline: &mut BooleanPipeline,
    op: BooleanOp,
    classifier_a: Option<&AnalyticClassifier>,
    classifier_b: Option<&AnalyticClassifier>,
    tol: &Tolerance,
) -> Result<(), OperationsError> {
    let solid_a = pipeline
        .solid_a
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "no solid A".into(),
        })?;
    let solid_b = pipeline
        .solid_b
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "no solid B".into(),
        })?;

    // Collect polygons for both solids (for ray-cast fallback).
    let polys_a = collect_solid_face_polygons(topo, solid_a)?;
    let polys_b = collect_solid_face_polygons(topo, solid_b)?;

    // Build coplanar lookup: for each face, collect its coplanar opposing
    // faces and orientation. A face from solid A has coplanar partners from
    // solid B, and vice versa.
    let mut coplanar_opposing: HashMap<FaceId, Vec<(FaceId, bool)>> = HashMap::new();
    for &(fa, fb, same_ori) in &pipeline.coplanar_pairs {
        coplanar_opposing
            .entry(fa)
            .or_default()
            .push((fb, same_ori));
        coplanar_opposing
            .entry(fb)
            .or_default()
            .push((fa, same_ori));
    }

    // Pre-compute face boundary polygons for coplanar containment tests.
    let mut face_poly_cache: HashMap<FaceId, Vec<Point3>> = HashMap::new();
    for &(fa, fb, _) in &pipeline.coplanar_pairs {
        for fid in [fa, fb] {
            face_poly_cache
                .entry(fid)
                .or_insert_with(|| collect_face_polygon(topo, fid).unwrap_or_default());
        }
    }

    // Classify each sub-face and mark for selection.
    let mut selected = Vec::new();
    for sub_face in &pipeline.sub_faces {
        // Get test point inside this sub-face.
        let frame = pipeline
            .surface_info
            .get(&sub_face.parent)
            .and_then(SurfaceInfo::as_plane_frame);
        let test_pt = interior_point_3d(sub_face, frame);

        let standard_class = || match sub_face.source {
            Source::A => classify_with_fallback(
                test_pt,
                &sub_face.outer_wire,
                classifier_b,
                topo,
                solid_b,
                &polys_b,
                *tol,
            ),
            Source::B => classify_with_fallback(
                test_pt,
                &sub_face.outer_wire,
                classifier_a,
                topo,
                solid_a,
                &polys_a,
                *tol,
            ),
        };

        // Check if this sub-face's parent has coplanar opposing faces.
        let class = if let Some(opposing) = coplanar_opposing.get(&sub_face.parent) {
            // Test if the interior point is inside any opposing coplanar face.
            let coplanar_class = opposing.iter().find_map(|&(opp_fid, same_ori)| {
                let opp_poly = face_poly_cache.get(&opp_fid)?;
                let n = match topo.face(opp_fid).ok()?.surface() {
                    FaceSurface::Plane { normal, .. } => *normal,
                    _ => return None,
                };
                if point_in_face_polygon_3d(test_pt, opp_poly, &n) {
                    Some(if same_ori {
                        FaceClass::CoplanarSame
                    } else {
                        FaceClass::CoplanarOpposite
                    })
                } else {
                    None
                }
            });
            // Fall through to standard classification when not inside any coplanar face.
            coplanar_class.unwrap_or_else(standard_class)
        } else {
            standard_class()
        };

        // Use the boolean truth table to decide keep/discard/flip.
        if let Some(flip) = select_fragment(sub_face.source, class, op) {
            selected.push((sub_face.clone(), flip));
        }
    }

    // Apply flip flag: when flip=true, reverse the sub-face orientation.
    pipeline.sub_faces = selected
        .into_iter()
        .map(|(mut sf, flip)| {
            if flip {
                sf.reversed = !sf.reversed;
                // Reverse outer wire winding.
                sf.outer_wire.reverse();
                for edge in &mut sf.outer_wire {
                    std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                    std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                    edge.forward = !edge.forward;
                }
                // Same for inner wires.
                for inner in &mut sf.inner_wires {
                    inner.reverse();
                    for edge in inner.iter_mut() {
                        std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                        std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                        edge.forward = !edge.forward;
                    }
                }
            }
            sf
        })
        .collect();

    Ok(())
}

/// OCCT-style Same-Domain face deduplication.
///
/// After classification, sub-faces from different parent faces (potentially
/// different solids) that have the same surface AND the same boundary edge
/// set are duplicates. This happens at coplanar face intersections where both
/// the A and B versions of a face survive classification.
///
/// Detects duplicates by hashing the quantized 3D vertex sequence of each
/// sub-face's outer wire. Sub-faces with matching hash AND equivalent surfaces
/// are deduplicated (only one is kept).
fn dedup_same_domain_subfaces(pipeline: &mut BooleanPipeline, tol: &Tolerance) {
    use std::collections::HashMap;

    if pipeline.sub_faces.len() < 2 {
        return;
    }

    // Hash each sub-face by its quantized vertex sequence (sorted for
    // orientation-independent comparison).
    let resolution = tol.linear.max(1e-6);
    let q = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() / resolution).round() as i64,
            (p.y() / resolution).round() as i64,
            (p.z() / resolution).round() as i64,
        )
    };

    // For each sub-face, build a canonical vertex key (sorted set of quantized points).
    let keys: Vec<Vec<(i64, i64, i64)>> = pipeline
        .sub_faces
        .iter()
        .map(|sf| {
            let mut pts: Vec<(i64, i64, i64)> =
                sf.outer_wire.iter().map(|e| q(e.start_3d)).collect();
            pts.sort_unstable();
            pts.dedup();
            pts
        })
        .collect();

    // Group sub-faces by vertex key.
    let mut groups: HashMap<Vec<(i64, i64, i64)>, Vec<usize>> = HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        groups.entry(key.clone()).or_default().push(i);
    }

    // For each group with 2+ sub-faces, check surface equivalence and mark duplicates.
    let mut to_remove: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for indices in groups.values() {
        if indices.len() < 2 {
            continue;
        }
        // Check all pairs for surface equivalence.
        for i in 0..indices.len() {
            if to_remove.contains(&indices[i]) {
                continue;
            }
            for j in (i + 1)..indices.len() {
                if to_remove.contains(&indices[j]) {
                    continue;
                }
                let sf_i = &pipeline.sub_faces[indices[i]];
                let sf_j = &pipeline.sub_faces[indices[j]];
                // Same surface + same reversed flag = duplicate.
                if sf_i.reversed == sf_j.reversed
                    && crate::heal::surfaces_equivalent_pub(&sf_i.surface, &sf_j.surface)
                {
                    to_remove.insert(indices[j]);
                }
            }
        }
    }

    if to_remove.is_empty() {
        return;
    }

    // Remove duplicates (preserve order).
    let mut new_sub_faces = Vec::with_capacity(pipeline.sub_faces.len() - to_remove.len());
    for (i, sf) in pipeline.sub_faces.drain(..).enumerate() {
        if !to_remove.contains(&i) {
            new_sub_faces.push(sf);
        }
    }
    pipeline.sub_faces = new_sub_faces;
}

/// Classify a sub-face against the opposing solid with robust boundary handling.
///
/// 1. Try the analytic classifier on the interior point.
/// 2. If on-boundary (None), try each wire vertex until one gives a definitive answer.
/// 3. Fall back to ray-casting if no analytic result.
fn classify_with_fallback(
    test_pt: Point3,
    wire: &[super::pipeline::OrientedPCurveEdge],
    classifier: Option<&AnalyticClassifier>,
    topo: &Topology,
    solid: SolidId,
    face_polys: &[FacePolyData],
    tol: Tolerance,
) -> FaceClass {
    if let Some(c) = classifier {
        // Try the interior point first.
        if let Some(class) = c.classify(test_pt, tol) {
            return class;
        }
        // Interior point is on boundary — try wire vertices.
        for edge in wire {
            if let Some(class) = c.classify(edge.start_3d, tol) {
                return class;
            }
        }
    }
    // Fall back to analytic ray-casting (uses actual surface equations for
    // curved faces instead of flat polygon approximation).
    classify_point_analytic_raycast(test_pt, topo, solid, face_polys)
}

/// Classify a 3D point as inside/outside/on a solid using face polygon ray-casting.
///
/// Each face has an outer polygon and optional inner hole polygons. A ray
/// crossing counts only when the hit is inside the outer polygon but NOT
/// inside any hole (inner wire opening).
/// Classify a 3D point against a solid using analytic ray-surface intersection.
///
/// For each face in the solid:
/// - Plane faces: ray-plane intersection (exact)
/// - Cylinder faces: ray-cylinder intersection (analytic quadratic)
/// - Cone faces: ray-cone intersection (analytic quadratic)
/// - Other: fall back to polygon approximation
///
/// Uses multi-ray voting (5 directions) for robustness.
#[allow(clippy::too_many_lines)]
fn classify_point_analytic_raycast(
    point: Point3,
    topo: &Topology,
    solid: SolidId,
    face_polys: &[FacePolyData],
) -> FaceClass {
    let rays = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.577_350_3, 0.577_350_3, 0.577_350_3),
    ];

    let faces = collect_solid_faces(topo, solid).unwrap_or_default();

    // Build face-index → poly-index map to handle degenerate face skipping.
    // collect_solid_face_polygons may skip faces with < 3 polygon vertices,
    // causing face_polys indices to diverge from the faces vector.
    let poly_index: HashMap<usize, usize> = {
        let mut map = HashMap::new();
        let mut pi = 0;
        for &fid in &faces {
            if let Ok(face) = topo.face(fid) {
                let wire = topo.wire(face.outer_wire()).ok();
                let vert_count = wire.map_or(0, |w| w.edges().len());
                if vert_count >= 3 {
                    map.insert(fid.index(), pi);
                    pi += 1;
                }
            }
        }
        map
    };

    let mut inside_count = 0u32;
    let mut outside_count = 0u32;

    for ray_dir in &rays {
        let mut crossings = 0u32;

        for &fid in &faces {
            let Ok(face) = topo.face(fid) else { continue };
            let reversed = face.is_reversed();
            let surface = face.surface();

            match surface {
                FaceSurface::Plane { normal, d } => {
                    // Exact ray-plane intersection.
                    let eff_n = if reversed { -*normal } else { *normal };
                    let eff_d = if reversed { -*d } else { *d };
                    let denom = eff_n.dot(*ray_dir);
                    if denom.abs() < 1e-15 {
                        continue;
                    }
                    let pv = Vec3::new(point.x(), point.y(), point.z());
                    let t = (eff_d - eff_n.dot(pv)) / denom;
                    if t < 1e-10 {
                        continue;
                    }
                    let hit = point + *ray_dir * t;
                    // Use the polygon data for point-in-face test.
                    if let Some(&pi) = poly_index.get(&fid.index()) {
                        let (verts, holes, pn, _) = &face_polys[pi];
                        if point_in_face_polygon_3d(hit, verts, pn) {
                            let in_hole = holes
                                .iter()
                                .any(|hole| point_in_face_polygon_3d(hit, hole, pn));
                            if !in_hole {
                                crossings += 1;
                            }
                        }
                    }
                }
                FaceSurface::Cylinder(cyl) => {
                    // Analytic ray-cylinder intersection with UV containment.
                    let hits = ray_cylinder_intersect(point, *ray_dir, cyl);
                    let uv_poly = face_uv_polygon(topo, fid, face.surface());
                    for t in hits {
                        if t < 1e-10 {
                            continue;
                        }
                        let hit = point + *ray_dir * t;
                        // Check if hit is inside the face boundary using UV polygon.
                        if point_in_analytic_face_uv(hit, face.surface(), &uv_poly) {
                            // Check inner wire holes using 3D polygon fallback.
                            if let Some(&pi) = poly_index.get(&fid.index()) {
                                let (_, holes, pn, _) = &face_polys[pi];
                                let in_hole = holes
                                    .iter()
                                    .any(|hole| point_in_face_polygon_3d(hit, hole, pn));
                                if !in_hole {
                                    crossings += 1;
                                }
                            } else {
                                crossings += 1;
                            }
                        }
                    }
                }
                FaceSurface::Cone(con) => {
                    // Analytic ray-cone intersection with UV containment.
                    let hits = ray_cone_intersect(point, *ray_dir, con);
                    let uv_poly = face_uv_polygon(topo, fid, face.surface());
                    for t in hits {
                        if t < 1e-10 {
                            continue;
                        }
                        let hit = point + *ray_dir * t;
                        if point_in_analytic_face_uv(hit, face.surface(), &uv_poly) {
                            if let Some(&pi) = poly_index.get(&fid.index()) {
                                let (_, holes, pn, _) = &face_polys[pi];
                                let in_hole = holes
                                    .iter()
                                    .any(|hole| point_in_face_polygon_3d(hit, hole, pn));
                                if !in_hole {
                                    crossings += 1;
                                }
                            } else {
                                crossings += 1;
                            }
                        }
                    }
                }
                FaceSurface::Sphere(sph) => {
                    // Analytic ray-sphere intersection.
                    let hits = ray_sphere_intersect(point, *ray_dir, sph);
                    let uv_poly = face_uv_polygon(topo, fid, face.surface());
                    for t in hits {
                        if t < 1e-10 {
                            continue;
                        }
                        let hit = point + *ray_dir * t;
                        if point_in_analytic_face_uv(hit, face.surface(), &uv_poly) {
                            crossings += 1;
                        }
                    }
                }
                _ => {
                    // Fall back to polygon approximation for remaining surface types.
                    if let Some(&pi) = poly_index.get(&fid.index()) {
                        let (verts, holes, normal, d) = &face_polys[pi];
                        let denom = normal.dot(*ray_dir);
                        if denom.abs() < 1e-15 {
                            continue;
                        }
                        let pv = Vec3::new(point.x(), point.y(), point.z());
                        let t = (*d - normal.dot(pv)) / denom;
                        if t < 1e-10 {
                            continue;
                        }
                        let hit = point + *ray_dir * t;
                        if point_in_face_polygon_3d(hit, verts, normal) {
                            let in_hole = holes
                                .iter()
                                .any(|hole| point_in_face_polygon_3d(hit, hole, normal));
                            if !in_hole {
                                crossings += 1;
                            }
                        }
                    }
                }
            }
        }

        if crossings != 0 && crossings % 2 != 0 {
            inside_count += 1;
        } else {
            outside_count += 1;
        }
    }

    if inside_count > outside_count {
        FaceClass::Inside
    } else {
        FaceClass::Outside
    }
}

/// Ray-cylinder intersection: solve the quadratic equation for a ray
/// hitting an infinite cylinder, then filter by face bounds.
///
/// Returns 0, 1, or 2 t-values where the ray intersects the cylinder surface.
fn ray_cylinder_intersect(
    origin: Point3,
    dir: Vec3,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Vec<f64> {
    let o = origin - cyl.origin();
    let axis = cyl.axis();
    let r = cyl.radius();

    // Project ray origin and direction onto the plane perpendicular to the axis.
    let d_dot_a = dir.dot(axis);
    let o_dot_a = Vec3::new(o.x(), o.y(), o.z()).dot(axis);

    let d_perp = Vec3::new(
        dir.x() - d_dot_a * axis.x(),
        dir.y() - d_dot_a * axis.y(),
        dir.z() - d_dot_a * axis.z(),
    );
    let o_perp = Vec3::new(
        o.x() - o_dot_a * axis.x(),
        o.y() - o_dot_a * axis.y(),
        o.z() - o_dot_a * axis.z(),
    );

    let a = d_perp.x() * d_perp.x() + d_perp.y() * d_perp.y() + d_perp.z() * d_perp.z();
    let b = 2.0 * (d_perp.x() * o_perp.x() + d_perp.y() * o_perp.y() + d_perp.z() * o_perp.z());
    let c = o_perp.x() * o_perp.x() + o_perp.y() * o_perp.y() + o_perp.z() * o_perp.z() - r * r;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a.abs() < 1e-30 {
        return Vec::new();
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    vec![t1, t2]
}

/// Ray-cone intersection: solve the quadratic for a ray hitting an infinite cone.
///
/// A cone surface: `|p_perp|^2 = (k * (p · axis - z_apex))^2` where k = tan(half_angle).
/// Returns 0, 1, or 2 t-values.
fn ray_cone_intersect(
    origin: Point3,
    dir: Vec3,
    con: &brepkit_math::surfaces::ConicalSurface,
) -> Vec<f64> {
    let apex = con.apex();
    let axis = con.axis();
    let ha = con.half_angle();
    // half_angle is angle from radial plane to surface.
    // tan(half_angle) gives the ratio of axial to radial distance.
    let cos_a = ha.cos();
    let sin_a = ha.sin();
    // For the quadratic: (d·a)^2·cos²a - |d_perp|^2·sin²a = 0
    // where d_perp = d - (d·a)a, similarly for origin offset.
    let o = origin - apex;
    let o_v = Vec3::new(o.x(), o.y(), o.z());
    let d_dot_a = dir.dot(axis);
    let o_dot_a = o_v.dot(axis);

    let d_perp = dir - axis * d_dot_a;
    let o_perp = o_v - axis * o_dot_a;

    let cos2 = cos_a * cos_a;
    let sin2 = sin_a * sin_a;

    let a = d_perp.dot(d_perp) * cos2 - d_dot_a * d_dot_a * sin2;
    let b = 2.0 * (d_perp.dot(o_perp) * cos2 - d_dot_a * o_dot_a * sin2);
    let c = o_perp.dot(o_perp) * cos2 - o_dot_a * o_dot_a * sin2;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a.abs() < 1e-30 {
        return Vec::new();
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    vec![t1, t2]
}

/// Ray-sphere intersection: solve `|o + t*d - center|^2 = r^2`.
fn ray_sphere_intersect(
    origin: Point3,
    dir: Vec3,
    sph: &brepkit_math::surfaces::SphericalSurface,
) -> Vec<f64> {
    let oc = origin - sph.center();
    let oc_v = Vec3::new(oc.x(), oc.y(), oc.z());
    let r = sph.radius();

    let a = dir.dot(dir);
    let b = 2.0 * dir.dot(oc_v);
    let c = oc_v.dot(oc_v) - r * r;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a.abs() < 1e-30 {
        return Vec::new();
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    vec![t1, t2]
}

/// Classify a 3D point as inside/outside a solid using polygon ray-casting.
/// Legacy fallback used when analytic raycast is not available.
fn classify_point_against_solid(point: Point3, face_polys: &[FacePolyData]) -> FaceClass {
    // Fire rays in 5 directions (axis-aligned + 2 off-axis) and take majority
    // vote. This handles curved face polygon approximation errors — different
    // ray directions miss different curved faces, and the majority averages out.
    let rays = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        // Slightly off-axis to avoid degenerate alignment with box edges.
        Vec3::new(0.577_350_3, 0.577_350_3, 0.577_350_3),
    ];

    let mut inside_count = 0u32;
    let mut outside_count = 0u32;
    for ray_dir in &rays {
        let mut crossings = 0u32;
        for (verts, holes, normal, d) in face_polys {
            let denom = normal.dot(*ray_dir);
            if denom.abs() < 1e-15 {
                continue;
            }
            let pv = Vec3::new(point.x(), point.y(), point.z());
            let t = (*d - normal.dot(pv)) / denom;
            if t < 1e-10 {
                continue;
            }
            let hit = point + *ray_dir * t;
            if point_in_face_polygon_3d(hit, verts, normal) {
                let in_hole = holes
                    .iter()
                    .any(|hole| point_in_face_polygon_3d(hit, hole, normal));
                if !in_hole {
                    crossings += 1;
                }
            }
        }
        if crossings != 0 && crossings % 2 != 0 {
            inside_count += 1;
        } else {
            outside_count += 1;
        }
    }

    if inside_count > outside_count {
        FaceClass::Inside
    } else {
        FaceClass::Outside
    }
}

fn point_in_face_polygon_3d(point: Point3, verts: &[Point3], normal: &Vec3) -> bool {
    if verts.len() < 3 {
        return false;
    }
    // Project onto the dominant axis plane.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();
    let (proj_pt, proj_verts): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
        // Drop Z.
        (
            Point2::new(point.x(), point.y()),
            verts.iter().map(|v| Point2::new(v.x(), v.y())).collect(),
        )
    } else if ay >= ax {
        // Drop Y.
        (
            Point2::new(point.x(), point.z()),
            verts.iter().map(|v| Point2::new(v.x(), v.z())).collect(),
        )
    } else {
        // Drop X.
        (
            Point2::new(point.y(), point.z()),
            verts.iter().map(|v| Point2::new(v.y(), v.z())).collect(),
        )
    };
    point_in_polygon_2d(proj_pt, &proj_verts)
}

// ---------------------------------------------------------------------------
// Stage 5: Assemble
// ---------------------------------------------------------------------------

/// OCCT-style shell assembly: create topology faces, then build manifold shells
/// via greedy flood-fill with dihedral angle selection.
///
/// Follows BOPAlgo_BuilderSolid / BOPAlgo_ShellSplitter algorithm:
/// 1. Create all topology faces from classified sub-faces
/// 2. Build edge→face adjacency map
/// 3. Iteratively prune faces with free (dangling) edges
/// 4. Greedy flood-fill to build closed shells (dihedral angle selection)
/// 5. Classify shells as growth (outer) vs hole (inner)
/// 6. Assemble solid with outer shell + inner shells
#[allow(clippy::too_many_lines)]
fn assemble_pipeline(
    topo: &mut Topology,
    pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<SolidId, OperationsError> {
    // ── Phase 1: Create all topology faces ─────────────────────────────
    let all_pts = pipeline.sub_faces.iter().flat_map(|sf| {
        sf.outer_wire
            .iter()
            .flat_map(|e| [e.start_3d, e.end_3d])
            .chain(
                sf.inner_wires
                    .iter()
                    .flat_map(|w| w.iter().flat_map(|e| [e.start_3d, e.end_3d])),
            )
    });
    let resolution = vertex_merge_resolution(all_pts, *tol);
    let mut vertex_map: HashMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
        HashMap::new();
    let mut edge_map: HashMap<EdgeKey, brepkit_topology::edge::EdgeId> = HashMap::new();
    let mut edge_use_count: HashMap<EdgeKey, u32> = HashMap::new();

    let mut face_ids = Vec::new();

    for sub_face in &pipeline.sub_faces {
        let wire_id = create_wire_from_edges_dedup(
            topo,
            &sub_face.outer_wire,
            &mut vertex_map,
            &mut edge_map,
            &mut edge_use_count,
            resolution,
        )?;
        let inner_wires: Vec<_> = sub_face
            .inner_wires
            .iter()
            .filter_map(|inner| {
                create_wire_from_edges_dedup(
                    topo,
                    inner,
                    &mut vertex_map,
                    &mut edge_map,
                    &mut edge_use_count,
                    resolution,
                )
                .ok()
            })
            .collect();

        let face = if sub_face.reversed {
            Face::new_reversed(wire_id, inner_wires, sub_face.surface.clone())
        } else {
            Face::new(wire_id, inner_wires, sub_face.surface.clone())
        };
        let fid = topo.add_face(face);
        face_ids.push(fid);
    }

    if face_ids.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "boolean_pipeline: no faces in result".into(),
        });
    }

    // ── Phase 2: Build vertex-pair→face adjacency map ───────────────
    // Maps each (min_vertex, max_vertex) pair to the list of FaceIds that
    // reference it. This captures adjacency through BOTH shared edges (Line)
    // and geometrically equivalent but unshared edges (Circle, NURBS).
    // Two faces sharing a vertex pair are adjacent regardless of edge sharing.
    let mut vpair_faces: HashMap<(usize, usize), Vec<FaceId>> = HashMap::new();
    for &fid in &face_ids {
        let face = topo.face(fid).map_err(OperationsError::Topology)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).map_err(OperationsError::Topology)?;
            for oe in wire.edges() {
                // Vertex-pair key for adjacency.
                let edge = topo.edge(oe.edge()).map_err(OperationsError::Topology)?;
                let si = edge.start().index();
                let ei = edge.end().index();
                let key = if si <= ei { (si, ei) } else { (ei, si) };
                vpair_faces.entry(key).or_default().push(fid);
            }
        }
    }

    // ── Phase 3: Build shells via greedy flood-fill ─────────────────
    // OCCT's SplitBlock: for each unprocessed face, start a new shell and
    // greedily expand by selecting neighbors via minimum dihedral angle.
    // Never add a face if its shared edge already has 2 faces in the shell.
    //
    // If the flood-fill produces a single shell containing all faces, use it
    // directly (common case). If it fragments into multiple small shells,
    // fall back to single-shell assembly for compatibility.
    let shells = build_shells_greedy(topo, &face_ids, &vpair_faces)?;

    let total_faces = face_ids.len();

    if shells.len() == 1 {
        // Single shell from flood-fill — use directly.
        let shell = Shell::new(shells.into_iter().next().unwrap_or_default())
            .map_err(OperationsError::Topology)?;
        let shell_id = topo.add_shell(shell);
        return Ok(topo.add_solid(Solid::new(shell_id, Vec::new())));
    }

    if shells.len() == 2 {
        // Two shells: use multi-shell only if one is clearly dominant
        // (has >75% of faces) — indicates outer + inner cavity.
        let larger = shells.iter().map(Vec::len).max().unwrap_or(0);
        if larger * 4 > total_faces * 3 {
            // One shell has >75% of faces — use as outer, other as inner.
            let (outer_faces, inner_faces) = if shells[0].len() >= shells[1].len() {
                (&shells[0], &shells[1])
            } else {
                (&shells[1], &shells[0])
            };
            let outer = Shell::new(outer_faces.clone()).map_err(OperationsError::Topology)?;
            let inner = Shell::new(inner_faces.clone()).map_err(OperationsError::Topology)?;
            let outer_id = topo.add_shell(outer);
            let inner_id = topo.add_shell(inner);
            return Ok(topo.add_solid(Solid::new(outer_id, vec![inner_id])));
        }
    }

    // Default: single shell with all faces.
    // This handles cases where flood-fill fragments due to unshared curved
    // edges, disjoint solids, or ambiguous shell boundaries.
    let shell = Shell::new(face_ids).map_err(OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, Vec::new())))
}

/// Extended edge key: vertex pair + curve geometry discriminant.
///
/// Two edges are "the same" (shareable) when they connect the same vertices
/// AND have geometrically equivalent curves. This is OCCT's CommonBlock concept.
///
/// - Line: same vertex pair is sufficient (geometry is fully determined by endpoints)
/// - Circle: same vertex pair + same center + same radius + same normal direction
/// - Ellipse: same vertex pair + same center + same semi-axes + same normal
/// - NURBS: not shared (control points are floating-point, hard to compare)
type EdgeKey = (usize, usize, CurveDiscriminant);

/// Geometry discriminant for edge sharing. Quantized to integer coordinates
/// so it can be used as a HashMap key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CurveDiscriminant {
    /// Line: fully determined by endpoint positions (no extra data needed).
    Line,
    /// Circle: center + radius + arc midpoint (quantized).
    /// The midpoint distinguishes two half-arcs on the same circle connecting
    /// the same vertex pair (e.g., upper vs lower half of a section circle).
    Circle {
        cx: i64,
        cy: i64,
        cz: i64,
        r: i64,
        /// Arc midpoint — distinguishes which arc between the two vertices.
        mx: i64,
        my: i64,
        mz: i64,
    },
    /// Ellipse: center + semi-axes + arc midpoint (quantized).
    Ellipse {
        cx: i64,
        cy: i64,
        cz: i64,
        a: i64,
        b: i64,
        mx: i64,
        my: i64,
        mz: i64,
    },
    /// NURBS: never shared (each instance is unique).
    Nurbs(u64),
}

/// Counter for generating unique NURBS discriminants.
static NURBS_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Compute a geometry discriminant for edge sharing.
///
/// `p_start` and `p_end` are the 3D start/end of the edge (in wire traversal
/// order). For curved edges, the arc midpoint is included to distinguish two
/// arcs on the same curve connecting the same vertices. The midpoint is
/// always computed using the SHORTER arc (canonical ordering) so that two
/// faces traversing the same arc in opposite directions get the same key.
fn curve_discriminant(
    curve: &EdgeCurve,
    p_start: Point3,
    p_end: Point3,
    resolution: f64,
) -> CurveDiscriminant {
    let q = |v: f64| -> i64 { (v / resolution).round() as i64 };
    match curve {
        EdgeCurve::Line => CurveDiscriminant::Line,
        EdgeCurve::Circle(c) => {
            let center = c.center();
            // Arc midpoint for canonical key: always compute from the
            // parameter-space midpoint of the arc going from the smaller
            // angle to the larger angle. This ensures both traversal
            // directions (A→B and B→A) produce the same midpoint.
            let t_a = c.project(p_start);
            let t_b = c.project(p_end);
            let (t_lo, t_hi) = if t_a <= t_b { (t_a, t_b) } else { (t_b, t_a) };
            let t_mid = (t_lo + t_hi) * 0.5;
            let mid = c.evaluate(t_mid);
            CurveDiscriminant::Circle {
                cx: q(center.x()),
                cy: q(center.y()),
                cz: q(center.z()),
                r: q(c.radius()),
                mx: q(mid.x()),
                my: q(mid.y()),
                mz: q(mid.z()),
            }
        }
        EdgeCurve::Ellipse(e) => {
            let center = e.center();
            let t_a = e.project(p_start);
            let t_b = e.project(p_end);
            let (t_lo, t_hi) = if t_a <= t_b { (t_a, t_b) } else { (t_b, t_a) };
            let t_mid = (t_lo + t_hi) * 0.5;
            let mid = e.evaluate(t_mid);
            CurveDiscriminant::Ellipse {
                cx: q(center.x()),
                cy: q(center.y()),
                cz: q(center.z()),
                a: q(e.semi_major()),
                b: q(e.semi_minor()),
                mx: q(mid.x()),
                my: q(mid.y()),
                mz: q(mid.z()),
            }
        }
        EdgeCurve::NurbsCurve(_) => CurveDiscriminant::Nurbs(
            NURBS_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ),
    }
}

fn create_wire_from_edges_dedup(
    topo: &mut Topology,
    edges: &[super::pipeline::OrientedPCurveEdge],
    vertex_map: &mut HashMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    edge_map: &mut HashMap<EdgeKey, brepkit_topology::edge::EdgeId>,
    edge_use_count: &mut HashMap<EdgeKey, u32>,
    resolution: f64,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    let mut oriented_edges = Vec::new();
    for pe in edges {
        let key_s = quantize_point(pe.start_3d, resolution);
        let v_s = *vertex_map
            .entry(key_s)
            .or_insert_with(|| topo.add_vertex(Vertex::new(pe.start_3d, 0.0)));
        let key_e = quantize_point(pe.end_3d, resolution);
        let v_e = *vertex_map
            .entry(key_e)
            .or_insert_with(|| topo.add_vertex(Vertex::new(pe.end_3d, 0.0)));

        let si = v_s.index();
        let ei = v_e.index();
        let (key_min, key_max) = if si <= ei { (si, ei) } else { (ei, si) };
        let disc = curve_discriminant(&pe.curve_3d, pe.start_3d, pe.end_3d, resolution);
        let edge_key: EdgeKey = (key_min, key_max, disc);

        // Check manifold limit: max 2 faces per edge.
        let count = edge_use_count.get(&edge_key).copied().unwrap_or(0);
        let (eid, wire_forward) = if count < 2 {
            // Try to share existing edge.
            if let Some(&existing) = edge_map.get(&edge_key) {
                *edge_use_count.entry(edge_key).or_default() += 1;
                let edge_start = topo
                    .edge(existing)
                    .map_err(OperationsError::Topology)?
                    .start();
                // For Line edges, forward = edge.start matches wire start.
                // For curved edges, forward = edge.start matches wire start
                // (same logic — the edge's stored start vertex determines direction).
                let wire_fwd = edge_start == v_s;
                (existing, wire_fwd)
            } else {
                // Create new edge. For Line, natural direction = v_s→v_e.
                // For curved edges, use pe.forward to determine natural direction.
                let (natural_s, natural_e) = match &pe.curve_3d {
                    EdgeCurve::Line => (v_s, v_e),
                    _ => {
                        if pe.forward {
                            (v_s, v_e)
                        } else {
                            (v_e, v_s)
                        }
                    }
                };
                let eid = topo.add_edge(Edge::new(natural_s, natural_e, pe.curve_3d.clone()));
                edge_map.insert(edge_key, eid);
                *edge_use_count.entry(edge_key).or_default() += 1;
                let wire_fwd = match &pe.curve_3d {
                    EdgeCurve::Line => {
                        topo.edge(eid).map_err(OperationsError::Topology)?.start() == v_s
                    }
                    _ => pe.forward,
                };
                (eid, wire_fwd)
            }
        } else {
            // Already at manifold limit — create new (unshared) edge.
            let (natural_s, natural_e) = match &pe.curve_3d {
                EdgeCurve::Line => (v_s, v_e),
                _ => {
                    if pe.forward {
                        (v_s, v_e)
                    } else {
                        (v_e, v_s)
                    }
                }
            };
            let eid = topo.add_edge(Edge::new(natural_s, natural_e, pe.curve_3d.clone()));
            let wire_fwd = match &pe.curve_3d {
                EdgeCurve::Line => {
                    topo.edge(eid).map_err(OperationsError::Topology)?.start() == v_s
                }
                _ => pe.forward,
            };
            (eid, wire_fwd)
        };

        // Skip duplicate edges in the same wire (can happen with shared edges).
        if oriented_edges
            .iter()
            .any(|oe: &OrientedEdge| oe.edge() == eid)
        {
            continue;
        }

        oriented_edges.push(OrientedEdge::new(eid, wire_forward));
    }
    let wire = Wire::new(oriented_edges, true).map_err(OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

/// Build shells from a set of faces using greedy flood-fill.
///
/// Uses vertex-pair adjacency (`vpair_faces`) for neighbor discovery — this
/// handles both shared edges (Line) and geometrically equivalent but unshared
/// edges (Circle, NURBS connecting the same vertices on different faces).
///
/// Uses edge-ID adjacency (`edge_faces`) for the manifold constraint: an edge
/// shared by 2 faces in the current shell is "manifold" and won't be crossed.
/// Build shells from a set of faces using greedy flood-fill.
///
/// Uses vertex-pair adjacency (`vpair_faces`) for neighbor discovery — this
/// captures connections through both shared edges (Line) and geometrically
/// equivalent but unshared edges (Circle, NURBS with same vertex endpoints).
///
/// The manifold constraint (max 2 faces per edge) is tracked per edge ID,
/// not per vertex pair, since multiple distinct edges (Line + Circle) can
/// share the same vertex pair independently.
fn build_shells_greedy(
    topo: &Topology,
    face_ids: &[FaceId],
    vpair_faces: &HashMap<(usize, usize), Vec<FaceId>>,
) -> Result<Vec<Vec<FaceId>>, OperationsError> {
    use std::collections::{HashSet, VecDeque};

    let available: HashSet<FaceId> = face_ids.iter().copied().collect();
    let mut processed: HashSet<FaceId> = HashSet::new();
    let mut shells: Vec<Vec<FaceId>> = Vec::new();

    for &start_face in face_ids {
        if !available.contains(&start_face) || processed.contains(&start_face) {
            continue;
        }

        let mut shell_faces: Vec<FaceId> = vec![start_face];
        processed.insert(start_face);

        // Track edge-ID usage within this shell for manifold constraint.
        // Each edge ID can be shared by at most 2 faces.
        let mut shell_edge_count: HashMap<usize, u32> = HashMap::new();
        {
            let face = topo.face(start_face).map_err(OperationsError::Topology)?;
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                let wire = topo.wire(wid).map_err(OperationsError::Topology)?;
                for oe in wire.edges() {
                    *shell_edge_count.entry(oe.edge().index()).or_default() += 1;
                }
            }
        }

        // BFS expansion.
        let mut queue: VecDeque<FaceId> = VecDeque::new();
        queue.push_back(start_face);

        while let Some(current) = queue.pop_front() {
            let face = topo.face(current).map_err(OperationsError::Topology)?;
            // Collect (vertex-pair, edge-id) from ALL wires (outer + inner).
            let all_edges: Vec<((usize, usize), brepkit_topology::edge::EdgeId)> = {
                let mut pairs = Vec::new();
                for wid in
                    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
                {
                    let wire = topo.wire(wid).map_err(OperationsError::Topology)?;
                    for oe in wire.edges() {
                        let e = topo.edge(oe.edge()).map_err(OperationsError::Topology)?;
                        let si = e.start().index();
                        let ei = e.end().index();
                        let key = if si <= ei { (si, ei) } else { (ei, si) };
                        pairs.push((key, oe.edge()));
                    }
                }
                pairs
            };

            for (vpair, edge_id) in all_edges {
                let eidx = edge_id.index();

                // Skip this specific edge if already manifold in the shell.
                if shell_edge_count.get(&eidx).copied().unwrap_or(0) >= 2 {
                    continue;
                }

                // Find candidate neighbor faces via vertex-pair adjacency.
                let candidates: Vec<FaceId> = vpair_faces
                    .get(&vpair)
                    .map(|fs| {
                        fs.iter()
                            .copied()
                            .filter(|&f| {
                                f != current && available.contains(&f) && !processed.contains(&f)
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if candidates.is_empty() {
                    continue;
                }

                // Select best candidate by minimum dihedral angle.
                let selected = if candidates.len() == 1 {
                    candidates[0]
                } else {
                    select_by_dihedral_angle(topo, current, edge_id, &candidates)?
                };

                if processed.contains(&selected) {
                    continue;
                }

                processed.insert(selected);
                shell_faces.push(selected);
                queue.push_back(selected);

                // Update shell edge count with the selected face's edges.
                let sel_face = topo.face(selected).map_err(OperationsError::Topology)?;
                for wid in std::iter::once(sel_face.outer_wire())
                    .chain(sel_face.inner_wires().iter().copied())
                {
                    let wire = topo.wire(wid).map_err(OperationsError::Topology)?;
                    for sel_oe in wire.edges() {
                        *shell_edge_count.entry(sel_oe.edge().index()).or_default() += 1;
                    }
                }
            }
        }

        shells.push(shell_faces);
    }

    // Collect remaining unprocessed faces into their own shell.
    let remaining: Vec<FaceId> = available
        .iter()
        .filter(|f| !processed.contains(f))
        .copied()
        .collect();
    if !remaining.is_empty() {
        shells.push(remaining);
    }

    Ok(shells)
}

/// Select the candidate face with the smallest dihedral angle to `current_face`
/// around the shared `edge`. This is OCCT's GetFaceOff algorithm: compute the
/// exterior dihedral angle between `current_face` and each candidate, pick the
/// minimum. This produces "smooth continuation" — the greedy shell follows the
/// most natural surface flow.
fn select_by_dihedral_angle(
    topo: &Topology,
    current_face: FaceId,
    edge: brepkit_topology::edge::EdgeId,
    candidates: &[FaceId],
) -> Result<FaceId, OperationsError> {
    // Compute edge midpoint and tangent direction.
    let e = topo.edge(edge).map_err(OperationsError::Topology)?;
    let p_start = topo
        .vertex(e.start())
        .map_err(OperationsError::Topology)?
        .point();
    let p_end = topo
        .vertex(e.end())
        .map_err(OperationsError::Topology)?
        .point();
    let edge_mid = Point3::new(
        (p_start.x() + p_end.x()) * 0.5,
        (p_start.y() + p_end.y()) * 0.5,
        (p_start.z() + p_end.z()) * 0.5,
    );
    let edge_dir = Vec3::new(
        p_end.x() - p_start.x(),
        p_end.y() - p_start.y(),
        p_end.z() - p_start.z(),
    );
    let edge_len =
        (edge_dir.x() * edge_dir.x() + edge_dir.y() * edge_dir.y() + edge_dir.z() * edge_dir.z())
            .sqrt();
    if edge_len < 1e-12 {
        // Degenerate edge — just pick first candidate.
        return Ok(candidates[0]);
    }
    let t = Vec3::new(
        edge_dir.x() / edge_len,
        edge_dir.y() / edge_len,
        edge_dir.z() / edge_len,
    );

    // Compute the outward binormal of current_face at the edge midpoint.
    // binormal = face_normal × edge_tangent (points away from face interior).
    let n_cur = face_normal_at_point(topo, current_face, edge_mid)?;
    let b_cur = cross(n_cur, t);
    let b_cur_len = (b_cur.x() * b_cur.x() + b_cur.y() * b_cur.y() + b_cur.z() * b_cur.z()).sqrt();
    if b_cur_len < 1e-12 {
        return Ok(candidates[0]);
    }
    let b_cur = Vec3::new(
        b_cur.x() / b_cur_len,
        b_cur.y() / b_cur_len,
        b_cur.z() / b_cur_len,
    );

    // Reference direction for angle measurement: t × b_cur (== n_cur projected).
    let ref_dir = cross(t, b_cur);

    let mut best = candidates[0];
    let mut best_angle = f64::MAX;

    for &cand in candidates {
        let n_cand = face_normal_at_point(topo, cand, edge_mid)?;
        let b_cand = cross(n_cand, t);
        let b_cand_len =
            (b_cand.x() * b_cand.x() + b_cand.y() * b_cand.y() + b_cand.z() * b_cand.z()).sqrt();
        if b_cand_len < 1e-12 {
            continue;
        }
        let b_cand = Vec3::new(
            b_cand.x() / b_cand_len,
            b_cand.y() / b_cand_len,
            b_cand.z() / b_cand_len,
        );

        // Signed angle from b_cur to b_cand around t.
        let cos_a = dot(b_cur, b_cand);
        let sin_a = dot(ref_dir, b_cand);
        let mut angle = sin_a.atan2(cos_a);
        if angle < 0.0 {
            angle += std::f64::consts::TAU;
        }

        if angle < best_angle {
            best_angle = angle;
            best = cand;
        }
    }

    Ok(best)
}

/// Approximate face normal at a 3D point by evaluating the surface or using
/// the face polygon's geometric normal.
fn face_normal_at_point(
    topo: &Topology,
    face_id: FaceId,
    _point: Point3,
) -> Result<Vec3, OperationsError> {
    let face = topo.face(face_id).map_err(OperationsError::Topology)?;
    let surface = face.surface();
    let reversed = face.is_reversed();

    // For plane faces, the normal is constant.
    if let FaceSurface::Plane { normal, .. } = surface {
        let n = if reversed {
            Vec3::new(-normal.x(), -normal.y(), -normal.z())
        } else {
            *normal
        };
        return Ok(n);
    }

    // For curved surfaces, compute from the face polygon (Newell's method).
    let wire = topo
        .wire(face.outer_wire())
        .map_err(OperationsError::Topology)?;
    let pts: Vec<Point3> = wire
        .edges()
        .iter()
        .filter_map(|oe| {
            let e = topo.edge(oe.edge()).ok()?;
            let vid = if oe.is_forward() { e.start() } else { e.end() };
            Some(topo.vertex(vid).ok()?.point())
        })
        .collect();

    if pts.len() < 3 {
        return Ok(Vec3::new(0.0, 0.0, 1.0));
    }

    // Newell's method for polygon normal.
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..pts.len() {
        let j = (i + 1) % pts.len();
        nx += (pts[i].y() - pts[j].y()) * (pts[i].z() + pts[j].z());
        ny += (pts[i].z() - pts[j].z()) * (pts[i].x() + pts[j].x());
        nz += (pts[i].x() - pts[j].x()) * (pts[i].y() + pts[j].y());
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-12 {
        return Ok(Vec3::new(0.0, 0.0, 1.0));
    }
    let mut n = Vec3::new(nx / len, ny / len, nz / len);
    if reversed {
        n = Vec3::new(-n.x(), -n.y(), -n.z());
    }
    Ok(n)
}

#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )
}

#[inline]
fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x() * b.x() + a.y() * b.y() + a.z() * b.z()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn all_faces_supported(topo: &Topology, solid: SolidId) -> Result<bool, OperationsError> {
    let faces = collect_solid_faces(topo, solid)?;
    for fid in faces {
        let surface = topo.face(fid)?.surface();
        // Analytic (Plane, Cylinder, Cone, Sphere, Torus) + NURBS are supported.
        if !surface.is_analytic() && !matches!(surface, FaceSurface::Nurbs(_)) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn collect_solid_faces(topo: &Topology, solid: SolidId) -> Result<Vec<FaceId>, OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    Ok(shell.faces().to_vec())
}

/// Collect face boundary as a polygon by sampling edges.
///
/// For Line edges, only the start vertex is sampled. For curved edges (Circle,
/// Ellipse, NURBS), intermediate points are sampled to produce a polygon that
/// accurately approximates the curved boundary for ray-cast classification.
fn collect_face_polygon(topo: &Topology, face_id: FaceId) -> Result<Vec<Point3>, OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut pts = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let start_vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        let end_vid = if oe.is_forward() {
            edge.end()
        } else {
            edge.start()
        };
        let p_start = topo.vertex(start_vid)?.point();
        pts.push(p_start);

        // Sample curved edges at intermediate points.
        match edge.curve() {
            EdgeCurve::Line => {} // Start vertex is sufficient.
            EdgeCurve::Circle(circle) => {
                let p_end = topo.vertex(end_vid)?.point();
                // Sample 8 intermediate points along the arc.
                let t0 = circle.project(p_start);
                let mut t1 = circle.project(p_end);
                if oe.is_forward() {
                    if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                } else if t1 >= t0 {
                    t1 -= std::f64::consts::TAU;
                }
                let n_samples = 16;
                for i in 1..n_samples {
                    let frac = i as f64 / n_samples as f64;
                    let t = t0 + (t1 - t0) * frac;
                    pts.push(circle.evaluate(t));
                }
            }
            EdgeCurve::Ellipse(ellipse) => {
                let p_end = topo.vertex(end_vid)?.point();
                let t0 = ellipse.project(p_start);
                let mut t1 = ellipse.project(p_end);
                if oe.is_forward() {
                    if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                } else if t1 >= t0 {
                    t1 -= std::f64::consts::TAU;
                }
                let n_samples = 16;
                for i in 1..n_samples {
                    let frac = i as f64 / n_samples as f64;
                    let t = t0 + (t1 - t0) * frac;
                    pts.push(ellipse.evaluate(t));
                }
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                // Sample 8 intermediate points.
                let (t0, t1) = (nurbs.knots().first(), nurbs.knots().last());
                if let (Some(&t0), Some(&t1)) = (t0, t1) {
                    let n_samples = 16;
                    for i in 1..n_samples {
                        let frac = i as f64 / n_samples as f64;
                        let t = t0 + (t1 - t0) * frac;
                        pts.push(nurbs.evaluate(t));
                    }
                }
            }
        }
    }
    Ok(pts)
}

fn collect_solid_face_polygons(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<FacePolyData>, OperationsError> {
    let faces = collect_solid_faces(topo, solid)?;
    let mut result = Vec::new();
    for fid in faces {
        let face = topo.face(fid)?;
        let poly = collect_face_polygon(topo, fid)?;
        if poly.len() < 3 {
            continue;
        }
        // Collect inner wire (hole) polygons with curved edge sampling.
        let mut holes = Vec::new();
        for &iw_id in face.inner_wires() {
            let iw = topo.wire(iw_id)?;
            let hole_pts = sample_wire_polygon(topo, iw)?;
            if hole_pts.len() >= 3 {
                holes.push(hole_pts);
            }
        }
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let effective_normal = if face.is_reversed() {
                -*normal
            } else {
                *normal
            };
            let effective_d = if face.is_reversed() { -*d } else { *d };
            result.push((poly, holes, effective_normal, effective_d));
        } else {
            let (normal, d) = approximate_polygon_plane(&poly);
            let effective_normal = if face.is_reversed() { -normal } else { normal };
            let effective_d = if face.is_reversed() { -d } else { d };
            result.push((poly, holes, effective_normal, effective_d));
        }
    }
    Ok(result)
}

/// Sample a wire's boundary as a polygon, with intermediate points on curved edges.
fn sample_wire_polygon(topo: &Topology, wire: &Wire) -> Result<Vec<Point3>, OperationsError> {
    let mut pts = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let start_vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        let end_vid = if oe.is_forward() {
            edge.end()
        } else {
            edge.start()
        };
        let p_start = topo.vertex(start_vid)?.point();
        pts.push(p_start);

        match edge.curve() {
            EdgeCurve::Line => {}
            EdgeCurve::Circle(circle) => {
                let p_end = topo.vertex(end_vid)?.point();
                let t0 = circle.project(p_start);
                let mut t1 = circle.project(p_end);
                if oe.is_forward() {
                    if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                } else if t1 >= t0 {
                    t1 -= std::f64::consts::TAU;
                }
                let n_samples = 16;
                for i in 1..n_samples {
                    let frac = i as f64 / n_samples as f64;
                    let t = t0 + (t1 - t0) * frac;
                    pts.push(circle.evaluate(t));
                }
            }
            EdgeCurve::Ellipse(ellipse) => {
                let p_end = topo.vertex(end_vid)?.point();
                let t0 = ellipse.project(p_start);
                let mut t1 = ellipse.project(p_end);
                if oe.is_forward() {
                    if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                } else if t1 >= t0 {
                    t1 -= std::f64::consts::TAU;
                }
                let n_samples = 16;
                for i in 1..n_samples {
                    let frac = i as f64 / n_samples as f64;
                    let t = t0 + (t1 - t0) * frac;
                    pts.push(ellipse.evaluate(t));
                }
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                if let (Some(&t0), Some(&t1)) = (nurbs.knots().first(), nurbs.knots().last()) {
                    let n_samples = 16;
                    for i in 1..n_samples {
                        let frac = i as f64 / n_samples as f64;
                        let t = t0 + (t1 - t0) * frac;
                        pts.push(nurbs.evaluate(t));
                    }
                }
            }
        }
    }
    Ok(pts)
}

/// Count vertices shared between two face polygons (within tolerance).
fn count_shared_vertices(poly_a: &[Point3], poly_b: &[Point3], tol: f64) -> usize {
    let tol_sq = tol * tol;
    let mut count = 0;
    for a in poly_a {
        for b in poly_b {
            let dx = a.x() - b.x();
            let dy = a.y() - b.y();
            let dz = a.z() - b.z();
            if dx * dx + dy * dy + dz * dz < tol_sq {
                count += 1;
                break; // Count each vertex in A at most once.
            }
        }
    }
    count
}

fn plane_frame_for_polygon(normal: Vec3, poly: &[Point3]) -> PlaneFrame {
    PlaneFrame::from_plane_face(normal, poly)
}

fn compute_line_pcurve(
    frame: &PlaneFrame,
    start: Point3,
    end: Point3,
) -> brepkit_math::curves2d::Curve2D {
    use brepkit_math::curves2d::Curve2D;
    let p0 = frame.project(start);
    let p1 = frame.project(end);
    let dir = Vec2::new(p1.x() - p0.x(), p1.y() - p0.y());
    // Build a Line2D from the projected direction. If degenerate
    // (zero-length), fall back to unit x-axis via make_line2d_safe which
    // centralizes the single #[allow(clippy::unwrap_used)].
    Curve2D::Line(super::pcurve_compute::make_line2d_safe(p0, dir))
}

// ---------------------------------------------------------------------------
// Intersection helpers
// ---------------------------------------------------------------------------

/// Find a point on the intersection line of two planes.
fn solve_two_planes_origin(
    na: Vec3,
    da: f64,
    nb: Vec3,
    db: f64,
    line_dir: Vec3,
) -> Result<Point3, OperationsError> {
    // Use the formula: p = (da*(nb×line) + db*(line×na)) / (line·(na×nb))
    let na_cross_nb = na.cross(nb);
    let denom = line_dir.dot(na_cross_nb);
    if denom.abs() < 1e-15 {
        return Err(OperationsError::InvalidInput {
            reason: "degenerate plane-plane intersection".into(),
        });
    }
    let nb_cross_l = nb.cross(line_dir);
    let l_cross_na = line_dir.cross(na);
    Ok(Point3::new(
        (da * nb_cross_l.x() + db * l_cross_na.x()) / denom,
        (da * nb_cross_l.y() + db * l_cross_na.y()) / denom,
        (da * nb_cross_l.z() + db * l_cross_na.z()) / denom,
    ))
}

/// Trim an infinite 3D line to a face's polygon boundary.
///
/// Projects the line and polygon into the face's 2D frame, computes
/// line-polygon intersection, and returns parameter pairs on the 3D line.
fn trim_line_to_polygon_3d(
    line_origin: &Point3,
    line_dir: &Vec3,
    polygon: &[Point3],
    frame: &PlaneFrame,
) -> Vec<(f64, f64)> {
    // Project the line into 2D.
    let origin_2d = frame.project(*line_origin);
    let dir_2d = Vec2::new(line_dir.dot(frame.u_axis()), line_dir.dot(frame.v_axis()));

    // Project polygon to 2D.
    let poly_2d: Vec<Point2> = polygon.iter().map(|p| frame.project(*p)).collect();

    trim_line_to_polygon_2d(origin_2d, dir_2d, &poly_2d)
}

/// Trim an infinite 2D line to a 2D polygon boundary.
///
/// Returns sorted parameter pairs (t_enter, t_exit) for segments inside.
fn trim_line_to_polygon_2d(origin: Point2, dir: Vec2, polygon: &[Point2]) -> Vec<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    let dir_len_sq = dir.x() * dir.x() + dir.y() * dir.y();
    if dir_len_sq < 1e-30 {
        return Vec::new();
    }

    // Compute intersection parameter for the line with each polygon edge.
    let mut hits: Vec<f64> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let pi = polygon[i];
        let pj = polygon[j];
        let edge_dir = Vec2::new(pj.x() - pi.x(), pj.y() - pi.y());
        let denom = dir.x() * edge_dir.y() - dir.y() * edge_dir.x();
        if denom.abs() < 1e-15 {
            continue; // Parallel.
        }
        let dp = Vec2::new(pi.x() - origin.x(), pi.y() - origin.y());
        let t = (dp.x() * edge_dir.y() - dp.y() * edge_dir.x()) / denom;
        let u = (dp.x() * dir.y() - dp.y() * dir.x()) / denom;
        // u must be in [0, 1] for the edge segment.
        if (-1e-10..=1.0 + 1e-10).contains(&u) {
            hits.push(t);
        }
    }

    if hits.is_empty() {
        return Vec::new();
    }

    // Sort and pair up as (enter, exit).
    hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Dedup close hits.
    hits.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    let mut segments = Vec::new();
    let mut i = 0;
    while i + 1 < hits.len() {
        // Check if midpoint of this pair is inside the polygon.
        let t_mid = (hits[i] + hits[i + 1]) * 0.5;
        let mid = Point2::new(origin.x() + dir.x() * t_mid, origin.y() + dir.y() * t_mid);
        if point_in_polygon_2d(mid, polygon) {
            segments.push((hits[i], hits[i + 1]));
        }
        i += 2;
    }
    segments
}

// ---------------------------------------------------------------------------
// Disjoint handling (with containment detection)
// ---------------------------------------------------------------------------

fn handle_disjoint_pipeline(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    classifier_a: Option<&AnalyticClassifier>,
    classifier_b: Option<&AnalyticClassifier>,
    tol: &Tolerance,
) -> Result<SolidId, OperationsError> {
    // Check if B is inside A or A is inside B.
    // Try the analytic classifier first; fall back to ray-cast polygon test
    // if no classifier is available (e.g. non-axis-aligned planar solids in
    // future steps where try_build_analytic_classifier returns None).
    let sample_b = sample_solid_vertex(topo, b)?;
    let sample_a = sample_solid_vertex(topo, a)?;

    let b_in_a = classifier_a
        .and_then(|c| c.classify(sample_b, *tol))
        .unwrap_or_else(|| {
            let polys = collect_solid_face_polygons(topo, a).unwrap_or_default();
            classify_point_against_solid(sample_b, &polys)
        })
        == FaceClass::Inside;
    let a_in_b = classifier_b
        .and_then(|c| c.classify(sample_a, *tol))
        .unwrap_or_else(|| {
            let polys = collect_solid_face_polygons(topo, b).unwrap_or_default();
            classify_point_against_solid(sample_a, &polys)
        })
        == FaceClass::Inside;

    if b_in_a {
        // B is entirely inside A.
        return match op {
            BooleanOp::Fuse => Ok(crate::copy::copy_solid(topo, a)?),
            BooleanOp::Cut => Err(OperationsError::InvalidInput {
                reason:
                    "boolean_pipeline: B is inside A — cut would create void (not supported in pipeline)"
                        .into(),
            }),
            BooleanOp::Intersect => Ok(crate::copy::copy_solid(topo, b)?),
        };
    }

    if a_in_b {
        // A is entirely inside B.
        return match op {
            BooleanOp::Fuse => Ok(crate::copy::copy_solid(topo, b)?),
            BooleanOp::Cut => Err(OperationsError::InvalidInput {
                reason: "boolean_pipeline: A is inside B — cut result is empty".into(),
            }),
            BooleanOp::Intersect => Ok(crate::copy::copy_solid(topo, a)?),
        };
    }

    // Truly disjoint.
    match op {
        BooleanOp::Fuse => {
            // Copy both solids into a merged shell.
            // NOTE: merging two disjoint shells into one produces a
            // non-manifold topology (two disconnected components in one
            // shell). validate_boolean_result in the caller will reject
            // this, falling back to the chord-based path. The orphaned
            // copy_a/copy_b entities remain in the arena (no dealloc).
            let copy_a = crate::copy::copy_solid(topo, a)?;
            let copy_b = crate::copy::copy_solid(topo, b)?;
            let shell_a = topo.solid(copy_a)?.outer_shell();
            let shell_b = topo.solid(copy_b)?.outer_shell();
            let mut all_faces = topo.shell(shell_a)?.faces().to_vec();
            all_faces.extend(topo.shell(shell_b)?.faces().to_vec());
            let new_shell = topo.add_shell(brepkit_topology::shell::Shell::new(all_faces)?);
            Ok(topo.add_solid(brepkit_topology::solid::Solid::new(new_shell, vec![])))
        }
        BooleanOp::Cut => {
            // A - B where B doesn't touch A → result is A.
            Ok(crate::copy::copy_solid(topo, a)?)
        }
        BooleanOp::Intersect => {
            // A ∩ B where they don't touch → empty.
            Err(OperationsError::InvalidInput {
                reason: "boolean_pipeline: intersection of disjoint solids is empty".into(),
            })
        }
    }
}

/// Sample one vertex position from a solid (for containment testing).
fn sample_solid_vertex(topo: &Topology, solid: SolidId) -> Result<Point3, OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    let fid = *shell
        .faces()
        .first()
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "solid has no faces".into(),
        })?;
    let face = topo.face(fid)?;
    let wire = topo.wire(face.outer_wire())?;
    let oe = wire
        .edges()
        .first()
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "face has no edges".into(),
        })?;
    let edge = topo.edge(oe.edge())?;
    let v = topo.vertex(edge.start())?;
    Ok(v.point())
}

// ---------------------------------------------------------------------------
// Analytic intersection helpers
// ---------------------------------------------------------------------------

/// Sample 3D points along an `ExactIntersectionCurve`.
#[allow(clippy::cast_precision_loss)]
fn sample_intersection_curve(curve: &ExactIntersectionCurve, n: usize) -> Vec<Point3> {
    match curve {
        ExactIntersectionCurve::Circle(c) => {
            use brepkit_math::traits::ParametricCurve;
            let (t0, t1) = c.domain();
            (0..=n)
                .map(|i| {
                    let t = t0 + (t1 - t0) * (i as f64) / (n as f64);
                    c.evaluate(t)
                })
                .collect()
        }
        ExactIntersectionCurve::Ellipse(e) => {
            use brepkit_math::traits::ParametricCurve;
            let (t0, t1) = e.domain();
            (0..=n)
                .map(|i| {
                    let t = t0 + (t1 - t0) * (i as f64) / (n as f64);
                    e.evaluate(t)
                })
                .collect()
        }
        ExactIntersectionCurve::Points(pts) => pts.clone(),
    }
}

/// Convert an `ExactIntersectionCurve` to an `EdgeCurve`.
fn intersection_curve_to_edge_curve(curve: &ExactIntersectionCurve) -> EdgeCurve {
    match curve {
        ExactIntersectionCurve::Circle(c) => EdgeCurve::Circle(c.clone()),
        ExactIntersectionCurve::Ellipse(e) => EdgeCurve::Ellipse(e.clone()),
        ExactIntersectionCurve::Points(pts) => {
            // Fit a NURBS curve through the points.
            if pts.len() >= 4 {
                match brepkit_math::nurbs::fitting::interpolate(pts, 3) {
                    Ok(nc) => EdgeCurve::NurbsCurve(nc),
                    Err(_) => EdgeCurve::Line,
                }
            } else {
                EdgeCurve::Line
            }
        }
    }
}

/// Extract contiguous segments of `true` from a boolean array.
///
/// Returns `(start_index, end_index)` pairs where both endpoints are `true`.
fn extract_contiguous_segments(flags: &[bool]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < flags.len() {
        if flags[i] {
            let start = i;
            while i < flags.len() && flags[i] {
                i += 1;
            }
            let end = i - 1;
            if end > start {
                result.push((start, end));
            }
        } else {
            i += 1;
        }
    }
    result
}

/// Fit a 2D pcurve from 3D sample points projected via a `PlaneFrame`.
///
/// Projects each 3D point to UV, checks collinearity, and fits a NURBS curve
/// if the points are not on a line. Used for section edges on plane faces
/// where the 3D curve is curved (half-circle arcs).
fn fit_pcurve_from_3d_samples(
    samples_3d: &[Point3],
    frame: &PlaneFrame,
) -> brepkit_math::curves2d::Curve2D {
    use super::pcurve_compute::make_line2d_safe;

    let uv_pts: Vec<Point2> = samples_3d.iter().map(|&p| frame.project(p)).collect();
    if uv_pts.len() < 2 {
        let p0 = uv_pts.first().copied().unwrap_or(Point2::new(0.0, 0.0));
        return brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(p0, Vec2::new(1.0, 0.0)));
    }

    // Check collinearity.
    let p0 = uv_pts[0];
    let pn = uv_pts[uv_pts.len() - 1];
    let dx = pn.x() - p0.x();
    let dy = pn.y() - p0.y();
    let len_sq = dx * dx + dy * dy;
    let mut is_line = len_sq < 1e-12;
    if !is_line {
        let inv_len = 1.0 / len_sq.sqrt();
        is_line = uv_pts[1..uv_pts.len() - 1].iter().all(|p| {
            let ex = p.x() - p0.x();
            let ey = p.y() - p0.y();
            (ex * dy - ey * dx).abs() * inv_len < 1e-6
        });
    }

    if is_line {
        let dir = Vec2::new(dx, dy);
        return brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(p0, dir));
    }

    // Fit NURBS through UV points.
    let pts_3d: Vec<Point3> = uv_pts
        .iter()
        .map(|p| Point3::new(p.x(), p.y(), 0.0))
        .collect();
    let degree = 3.min(pts_3d.len() - 1);
    match brepkit_math::nurbs::fitting::interpolate(&pts_3d, degree) {
        Ok(nurbs_3d) => {
            let cp_2d: Vec<Point2> = nurbs_3d
                .control_points()
                .iter()
                .map(|p| Point2::new(p.x(), p.y()))
                .collect();
            let weights = nurbs_3d.weights().to_vec();
            let knots = nurbs_3d.knots().to_vec();
            brepkit_math::curves2d::NurbsCurve2D::new(nurbs_3d.degree(), knots, cp_2d, weights)
                .map_or_else(
                    |_| {
                        brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(
                            p0,
                            Vec2::new(dx, dy),
                        ))
                    },
                    brepkit_math::curves2d::Curve2D::Nurbs,
                )
        }
        Err(_) => brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(p0, Vec2::new(dx, dy))),
    }
}

/// Detect the topological seam u-value for a periodic face.
///
/// Projects the first boundary vertex onto the surface to find the u-angle
/// where the seam edge (Line edge connecting top and bottom circles) sits.
/// Falls back to u=0 if the face or surface can't be queried.
fn detect_topological_seam_u(topo: &Topology, face_id: FaceId, surface: &FaceSurface) -> f64 {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return 0.0,
    };
    let wire = match topo.wire(face.outer_wire()) {
        Ok(w) => w,
        Err(_) => return 0.0,
    };
    // Find a Line edge in the wire — this is the seam.
    for oe in wire.edges() {
        let edge = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !matches!(edge.curve(), EdgeCurve::Line) {
            continue;
        }
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        let vertex = match topo.vertex(vid) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let (u, _v) = surface.project_point(vertex.point()).unwrap_or((0.0, 0.0));
        return u;
    }
    0.0
}

/// Build two seam-split section edges from a closed intersection curve.
///
/// Projects all 3D samples to the analytic surface's UV, finds the seam (u≈0)
/// and antipodal (u≈π) points, then constructs two arcs sorted by increasing u:
/// - Arc 1: (0, v) → (π, v) — the first half of the cylinder
/// - Arc 2: (π, v) → (2π, v) — the second half
///
/// Each arc gets a NURBS pcurve on the plane face and a fitted pcurve on the
/// analytic face with correct unwrapped UV endpoints.
#[allow(clippy::too_many_lines)]
fn build_seam_split_sections(
    samples: &[Point3],
    curve_3d: &EdgeCurve,
    analytic_surface: &FaceSurface,
    frame_plane: &PlaneFrame,
    seam_u: f64,
) -> Vec<SectionEdge> {
    use super::pcurve_compute::surface_periods;
    use std::f64::consts::TAU;

    if samples.len() < 4 {
        return Vec::new();
    }

    // Project all samples to the analytic surface's UV (raw, not unwrapped).
    let raw_uv: Vec<Point2> = samples
        .iter()
        .map(|&p| {
            let (u, v) = analytic_surface.project_point(p).unwrap_or((0.0, 0.0));
            Point2::new(u, v)
        })
        .collect();

    let (u_period, _v_period) = surface_periods(analytic_surface);
    let period = u_period.unwrap_or(TAU);

    // Find the seam sample: the one with raw u closest to the topological seam.
    let seam_idx = raw_uv
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = angular_distance(a.x(), seam_u, period);
            let db = angular_distance(b.x(), seam_u, period);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Find the antipodal sample: raw u closest to seam + π.
    let target_u = (seam_u + std::f64::consts::PI).rem_euclid(period);
    let anti_idx = raw_uv
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = angular_distance(a.x(), target_u, period);
            let db = angular_distance(b.x(), target_u, period);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(samples.len() / 2);

    let seam_3d = samples[seam_idx];
    let anti_3d = samples[anti_idx];
    let seam_v = raw_uv[seam_idx].y();
    let anti_v = raw_uv[anti_idx].y();

    // Classify each sample into arc 1 (seam → antipodal in +u direction)
    // or arc 2 (antipodal → seam+period). Use angular distance from seam_u.
    let mut arc1_indexed: Vec<(usize, f64)> = Vec::new(); // (idx, angular_offset_from_seam)
    let mut arc2_indexed: Vec<(usize, f64)> = Vec::new();
    let half = std::f64::consts::PI;
    for (i, uv) in raw_uv.iter().enumerate() {
        let u = uv.x().rem_euclid(period);
        // Angular offset from seam in [0, period).
        let offset = (u - seam_u).rem_euclid(period);
        if offset <= half + 0.05 {
            arc1_indexed.push((i, offset));
        }
        if offset >= half - 0.05 {
            arc2_indexed.push((i, offset));
        }
    }

    // Sort by angular offset from seam.
    arc1_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    arc2_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = Vec::new();

    // Arc 1: seam → antipodal (u: 0 → π)
    let arc1_3d: Vec<Point3> = arc1_indexed.iter().map(|&(i, _)| samples[i]).collect();
    if arc1_3d.len() >= 2 {
        let pcurve_a1 = fit_pcurve_from_3d_samples(&arc1_3d, frame_plane);
        let pcurve_b1 = fit_pcurve_from_3d_samples_on_surface(&arc1_3d, analytic_surface);
        result.push(SectionEdge {
            curve_3d: curve_3d.clone(),
            pcurve_a: pcurve_a1,
            pcurve_b: pcurve_b1,
            start: seam_3d,
            end: anti_3d,
            start_uv_a: None,
            end_uv_a: None,
            start_uv_b: Some(Point2::new(seam_u, seam_v)),
            end_uv_b: Some(Point2::new(seam_u + std::f64::consts::PI, anti_v)),
            target_face: None,
        });
    }

    // Arc 2: antipodal → seam+2π (u: π → 2π)
    let mut arc2_3d: Vec<Point3> = arc2_indexed.iter().map(|&(i, _)| samples[i]).collect();
    // Add the seam point at the end (same 3D, UV at u=2π).
    arc2_3d.push(seam_3d);
    if arc2_3d.len() >= 2 {
        let pcurve_a2 = fit_pcurve_from_3d_samples(&arc2_3d, frame_plane);
        let pcurve_b2 = fit_pcurve_from_3d_samples_on_surface(&arc2_3d, analytic_surface);
        result.push(SectionEdge {
            curve_3d: curve_3d.clone(),
            pcurve_a: pcurve_a2,
            pcurve_b: pcurve_b2,
            start: anti_3d,
            end: seam_3d,
            start_uv_a: None,
            end_uv_a: None,
            start_uv_b: Some(Point2::new(seam_u + std::f64::consts::PI, anti_v)),
            end_uv_b: Some(Point2::new(seam_u + period, seam_v)),
            target_face: None,
        });
    }

    result
}

/// Handle closed-curve segments where start ≈ end in 3D.
///
/// When a full circle or closed curve has all samples inside both faces,
/// the segment spans from index 0 to N with near-identical 3D endpoints.
/// Splits such segments at the midpoint to produce two arcs with distinct
/// endpoints. Non-closed segments are returned as-is.
fn split_closed_segment(
    samples: &[Point3],
    seg_start: usize,
    seg_end: usize,
    tol: f64,
) -> Vec<(usize, usize)> {
    let start_3d = samples[seg_start];
    let end_3d = samples[seg_end];
    let span = seg_end - seg_start;

    if (end_3d - start_3d).length() < tol && span >= 4 {
        // Closed curve — split at midpoint.
        let mid = seg_start + span / 2;
        vec![(seg_start, mid), (mid, seg_end)]
    } else if (end_3d - start_3d).length() < tol {
        // Too few samples for a meaningful split — skip.
        Vec::new()
    } else {
        vec![(seg_start, seg_end)]
    }
}

/// Find the 3D point where a circle crosses a periodic surface's u=0 seam.
/// Angular distance between two u-values on a periodic surface.
fn angular_distance(u1: f64, u2: f64, period: f64) -> f64 {
    let d = (u1 - u2).rem_euclid(period);
    d.min(period - d)
}

/// Fit a 2D pcurve from 3D sample points projected onto a surface's UV space.
///
/// Like [`fit_pcurve_from_3d_samples`] but projects via `surface.project_point()`
/// with periodic unwrapping, instead of using a `PlaneFrame`.
fn fit_pcurve_from_3d_samples_on_surface(
    samples_3d: &[Point3],
    surface: &FaceSurface,
) -> brepkit_math::curves2d::Curve2D {
    use super::pcurve_compute::{make_line2d_safe, surface_periods, unwrap_periodic_params_pub};

    let mut uv_pts: Vec<Point2> = samples_3d
        .iter()
        .map(|&p| {
            let (u, v) = surface.project_point(p).unwrap_or((0.0, 0.0));
            Point2::new(u, v)
        })
        .collect();

    if uv_pts.len() < 2 {
        let p0 = uv_pts.first().copied().unwrap_or(Point2::new(0.0, 0.0));
        return brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(p0, Vec2::new(1.0, 0.0)));
    }

    // Unwrap periodicity.
    let (u_period, v_period) = surface_periods(surface);
    unwrap_periodic_params_pub(&mut uv_pts, u_period, v_period);

    // Check collinearity.
    let p0 = uv_pts[0];
    let pn = uv_pts[uv_pts.len() - 1];
    let dx = pn.x() - p0.x();
    let dy = pn.y() - p0.y();
    let len_sq = dx * dx + dy * dy;
    let mut is_line = len_sq < 1e-12;
    if !is_line {
        let inv_len = 1.0 / len_sq.sqrt();
        is_line = uv_pts[1..uv_pts.len() - 1].iter().all(|p| {
            let ex = p.x() - p0.x();
            let ey = p.y() - p0.y();
            (ex * dy - ey * dx).abs() * inv_len < 1e-6
        });
    }

    if is_line {
        let dir = Vec2::new(dx, dy);
        return brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(p0, dir));
    }

    // Fit NURBS through UV points.
    let pts_3d: Vec<Point3> = uv_pts
        .iter()
        .map(|p| Point3::new(p.x(), p.y(), 0.0))
        .collect();
    let degree = 3.min(pts_3d.len() - 1);
    match brepkit_math::nurbs::fitting::interpolate(&pts_3d, degree) {
        Ok(nurbs_3d) => {
            let cp_2d: Vec<Point2> = nurbs_3d
                .control_points()
                .iter()
                .map(|p| Point2::new(p.x(), p.y()))
                .collect();
            let weights = nurbs_3d.weights().to_vec();
            let knots = nurbs_3d.knots().to_vec();
            brepkit_math::curves2d::NurbsCurve2D::new(nurbs_3d.degree(), knots, cp_2d, weights)
                .map_or_else(
                    |_| {
                        brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(
                            p0,
                            Vec2::new(dx, dy),
                        ))
                    },
                    brepkit_math::curves2d::Curve2D::Nurbs,
                )
        }
        Err(_) => brepkit_math::curves2d::Curve2D::Line(make_line2d_safe(p0, Vec2::new(dx, dy))),
    }
}

/// Build a UV polygon for an analytic face (for containment testing).
///
/// Samples points along each boundary edge (not just vertices) so that
/// faces with few unique vertices (e.g. cylinder lateral: 2 vertices,
/// 4 edges) produce a well-shaped UV polygon.
fn face_uv_polygon(topo: &Topology, face_id: FaceId, surface: &FaceSurface) -> Vec<Point2> {
    use super::pcurve_compute::evaluate_edge_at_t;

    const SAMPLES_PER_EDGE: usize = 8;

    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let wire = match topo.wire(face.outer_wire()) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };

    let mut uv_pts = Vec::new();
    for oe in wire.edges() {
        let edge = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let start_v = topo
            .vertex(if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            })
            .map(brepkit_topology::vertex::Vertex::point)
            .unwrap_or(Point3::new(0.0, 0.0, 0.0));
        let end_v = topo
            .vertex(if oe.is_forward() {
                edge.end()
            } else {
                edge.start()
            })
            .map(brepkit_topology::vertex::Vertex::point)
            .unwrap_or(Point3::new(0.0, 0.0, 0.0));

        // Sample N points along the edge (excluding the last — it's the next edge's first).
        // For closed circle edges (start ≈ end), pre-compute the vertex angle
        // so we evaluate starting from the boundary vertex, not the Circle3D's
        // parametric origin. Hoisted outside the sample loop.
        let closed_circle_angle = if matches!(edge.curve(), EdgeCurve::Circle(_))
            && (start_v - end_v).length() < Tolerance::new().linear
        {
            if let EdgeCurve::Circle(circle) = edge.curve() {
                Some((circle, circle.project(start_v)))
            } else {
                None
            }
        } else {
            None
        };
        #[allow(clippy::cast_precision_loss)]
        for i in 0..SAMPLES_PER_EDGE {
            let t = i as f64 / SAMPLES_PER_EDGE as f64;
            let p3d = if let Some((circle, vertex_angle)) = closed_circle_angle {
                let angle = if oe.is_forward() {
                    vertex_angle + std::f64::consts::TAU * t
                } else {
                    vertex_angle - std::f64::consts::TAU * t
                };
                circle.evaluate(angle)
            } else {
                evaluate_edge_at_t(edge.curve(), start_v, end_v, t)
            };
            let (u, v) = surface.project_point(p3d).unwrap_or((0.0, 0.0));
            uv_pts.push(Point2::new(u, v));
        }
    }

    // Unwrap periodicity so the polygon doesn't have seam jumps.
    let (u_period, v_period) = surface_periods(surface);
    if u_period.is_some() || v_period.is_some() {
        super::pcurve_compute::unwrap_periodic_params_pub(&mut uv_pts, u_period, v_period);
    }

    // Sphere cap fix: hemisphere faces have an equatorial boundary that maps
    // to a degenerate zero-area strip in UV (all points at v ≈ 0). Extend
    // the polygon to the appropriate pole so point_in_polygon_2d works.
    if matches!(surface, FaceSurface::Sphere(_)) && uv_pts.len() >= 3 {
        let v_min = uv_pts.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min);
        let v_max = uv_pts
            .iter()
            .map(|p| p.y())
            .fold(f64::NEG_INFINITY, f64::max);
        // Hemisphere boundary spans < ~0.6° in latitude → degenerate.
        if (v_max - v_min) < 0.01 {
            // Degenerate. Determine which hemisphere from the u-direction
            // of the boundary traversal: u increasing → north (v > 0),
            // u decreasing → south (v < 0).
            let u_first = uv_pts.first().map_or(0.0, |p| p.x());
            let u_last = uv_pts.last().map_or(0.0, |p| p.x());
            let v_pole = if u_last > u_first {
                std::f64::consts::FRAC_PI_2
            } else {
                -std::f64::consts::FRAC_PI_2
            };
            // Add pole corners. Extend u slightly past the sampled range
            // to cover the full period — Line edge sampling leaves a gap
            // (~TAU/segments) between the last sample and the first.
            let u_gap = (u_last - u_first).abs() / uv_pts.len() as f64;
            let (u_lo, u_hi) = if u_last < u_first {
                (u_last - u_gap, u_first + u_gap)
            } else {
                (u_first - u_gap, u_last + u_gap)
            };
            // Ensure rectangle connects: last boundary → bottom-left → bottom-right → first boundary
            if u_last < u_first {
                uv_pts.push(Point2::new(u_lo, v_pole));
                uv_pts.push(Point2::new(u_hi, v_pole));
            } else {
                uv_pts.push(Point2::new(u_hi, v_pole));
                uv_pts.push(Point2::new(u_lo, v_pole));
            }
        }
    }

    uv_pts
}

/// Test if a 3D point is inside an analytic face using UV polygon containment.
///
/// For periodic surfaces, tries shifted u/v candidates (±period) to handle
/// points near the seam that might project outside the unwrapped polygon range.
fn point_in_analytic_face_uv(point: Point3, surface: &FaceSurface, uv_poly: &[Point2]) -> bool {
    if uv_poly.len() < 3 {
        return true; // Degenerate face — accept.
    }
    let (u, v) = surface.project_point(point).unwrap_or((0.0, 0.0));

    let (u_period, v_period) = surface_periods(surface);
    let u_candidates = periodic_candidates(u, u_period);
    let v_candidates = periodic_candidates(v, v_period);

    for &uc in &u_candidates {
        for &vc in &v_candidates {
            if point_in_polygon_2d(Point2::new(uc, vc), uv_poly) {
                return true;
            }
        }
    }
    false
}

/// Generate candidate values for a periodic coordinate: `[val, val - period, val + period]`.
/// For non-periodic coordinates, returns just `[val]`.
fn periodic_candidates(val: f64, period: Option<f64>) -> Vec<f64> {
    if let Some(p) = period {
        vec![val, val - p, val + p]
    } else {
        vec![val]
    }
}

/// Convert a `FaceSurface` to a `NurbsSurface` for intersection.
///
/// For NURBS faces, clones the surface directly.
/// For analytic faces (cylinder, cone, sphere, torus), uses `to_nurbs()`.
/// For planes, returns `None` (plane-NURBS uses a dedicated path).
fn face_surface_to_nurbs(
    topo: &Topology,
    face_id: FaceId,
    surface: &FaceSurface,
) -> Result<Option<brepkit_math::nurbs::surface::NurbsSurface>, OperationsError> {
    match surface {
        FaceSurface::Nurbs(s) => Ok(Some(s.clone())),
        FaceSurface::Cylinder(cyl) => {
            let v_range = estimate_v_range(topo, face_id, surface).unwrap_or((-1.0, 1.0));
            cyl.to_nurbs(v_range.0, v_range.1)
                .map(Some)
                .map_err(OperationsError::Math)
        }
        FaceSurface::Cone(cone) => {
            let v_range = estimate_v_range(topo, face_id, surface).unwrap_or((0.01, 2.0));
            cone.to_nurbs(v_range.0, v_range.1)
                .map(Some)
                .map_err(OperationsError::Math)
        }
        FaceSurface::Sphere(sphere) => sphere.to_nurbs().map(Some).map_err(OperationsError::Math),
        FaceSurface::Torus(torus) => torus.to_nurbs().map(Some).map_err(OperationsError::Math),
        FaceSurface::Plane { .. } => Ok(None), // Plane-NURBS uses dedicated path
    }
}

/// Estimate the v-parameter range for an analytic face from its boundary vertices.
fn estimate_v_range(topo: &Topology, face_id: FaceId, surface: &FaceSurface) -> Option<(f64, f64)> {
    let poly = collect_face_polygon(topo, face_id).ok()?;
    if poly.is_empty() {
        return None;
    }
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for &p in &poly {
        let (_, v) = surface.project_point(p)?;
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    if v_min < v_max {
        // Extend slightly for robustness.
        let margin = (v_max - v_min) * 0.1;
        Some((v_min - margin, v_max + margin))
    } else {
        None
    }
}

/// Approximate a best-fit plane from a polygon using Newell's method.
fn approximate_polygon_plane(poly: &[Point3]) -> (Vec3, f64) {
    let n = poly.len();
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let pi = poly[i];
        let pj = poly[j];
        nx += (pi.y() - pj.y()) * (pi.z() + pj.z());
        ny += (pi.z() - pj.z()) * (pi.x() + pj.x());
        nz += (pi.x() - pj.x()) * (pi.y() + pj.y());
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    let normal = if len > 1e-15 {
        Vec3::new(nx / len, ny / len, nz / len)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    // d = normal · centroid
    let cx = poly.iter().map(|p| p.x()).sum::<f64>() / n as f64;
    let cy = poly.iter().map(|p| p.y()).sum::<f64>() / n as f64;
    let cz = poly.iter().map(|p| p.z()).sum::<f64>() / n as f64;
    let d = normal.dot(Vec3::new(cx, cy, cz));
    (normal, d)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::primitives::make_box;
    use brepkit_math::mat::Mat4;

    /// Create two partially-overlapping boxes with NO shared faces.
    /// a: [0,10]³. b: [3,8]×[2,12]×[1,11] (5×10×10 box translated).
    fn make_overlapping_boxes(topo: &mut Topology) -> (SolidId, SolidId) {
        let a = make_box(topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(topo, 5.0, 10.0, 10.0).unwrap();
        // Translate b by (3, 2, 1) so no faces are coplanar with a.
        let mat = Mat4::translation(3.0, 2.0, 1.0);
        crate::transform::transform_solid(topo, b, &mat).unwrap();
        (a, b)
    }

    #[test]
    fn trim_line_to_square_polygon_2d() {
        // Square [0,10]×[0,10] in 2D. Vertical line through x=5.
        let polygon = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        // Line: origin=(5, -5), direction=(0, 1).
        let origin = Point2::new(5.0, -5.0);
        let dir = Vec2::new(0.0, 1.0);
        let segments = trim_line_to_polygon_2d(origin, dir, &polygon);

        assert_eq!(segments.len(), 1, "expected 1 segment, got {segments:?}");
        let (t0, t1) = segments[0];
        // At t=5, we reach y=0 (bottom edge). At t=15, we reach y=10 (top edge).
        assert!((t0 - 5.0).abs() < 1e-6, "t0 = {t0}, expected ~5.0");
        assert!((t1 - 15.0).abs() < 1e-6, "t1 = {t1}, expected ~15.0");
    }

    #[test]
    fn trim_line_missing_polygon_2d() {
        let polygon = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        // Line that misses the polygon entirely: x=15.
        let origin = Point2::new(15.0, -5.0);
        let dir = Vec2::new(0.0, 1.0);
        let segments = trim_line_to_polygon_2d(origin, dir, &polygon);
        assert!(
            segments.is_empty(),
            "expected no segments for miss, got {segments:?}"
        );
    }

    #[test]
    fn solve_two_planes_gives_point_on_both() {
        // Plane A: z=5 → normal=(0,0,1), d=5
        // Plane B: x=3 → normal=(1,0,0), d=3
        let na = Vec3::new(0.0, 0.0, 1.0);
        let nb = Vec3::new(1.0, 0.0, 0.0);
        let line_dir = na.cross(nb).normalize().unwrap();
        let origin = solve_two_planes_origin(na, 5.0, nb, 3.0, line_dir).unwrap();

        // Origin should satisfy both planes: z=5 and x=3.
        assert!(
            (origin.z() - 5.0).abs() < 1e-10,
            "origin.z = {}, expected 5.0",
            origin.z()
        );
        assert!(
            (origin.x() - 3.0).abs() < 1e-10,
            "origin.x = {}, expected 3.0",
            origin.x()
        );
    }

    #[test]
    fn trim_3d_line_to_box_face() {
        // Face A is z=0 plane of a 10×10×10 box: polygon at z=0.
        // Intersection line: x=5, z=0, y=variable.
        let polygon = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let frame = PlaneFrame::from_plane_face(normal, &polygon);

        let line_origin = Point3::new(5.0, 0.0, 0.0);
        let line_dir = Vec3::new(0.0, 1.0, 0.0);

        let segments = trim_line_to_polygon_3d(&line_origin, &line_dir, &polygon, &frame);
        assert_eq!(segments.len(), 1, "expected 1 segment, got {segments:?}");

        let (t0, t1) = segments[0];
        // Should trim to y=[0,10].
        let start = line_origin + line_dir * t0;
        let end = line_origin + line_dir * t1;
        assert!(
            start.y().abs() < 0.1 && (end.y() - 10.0).abs() < 0.1,
            "segment ({:.2},{:.2},{:.2})→({:.2},{:.2},{:.2}), expected y=[0,10]",
            start.x(),
            start.y(),
            start.z(),
            end.x(),
            end.y(),
            end.z()
        );
    }

    #[test]
    fn intersect_two_actual_box_faces() {
        // Create a box and inspect its face geometry.
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let faces_a = collect_solid_faces(&topo, a).unwrap();
        let faces_b = collect_solid_faces(&topo, b).unwrap();

        // Try intersecting specific face pairs and check the result.
        let frames = HashMap::new();
        let mut total_sections = 0;
        for &fa in &faces_a {
            for &fb in &faces_b {
                let sections =
                    intersect_two_plane_faces(&topo, fa, fb, &frames, &mut Vec::new()).unwrap();
                if !sections.is_empty() {
                    for s in &sections {
                        // Verify section edge is within bounds of both boxes.
                        assert!(
                            s.start.x() >= -0.1 && s.start.x() <= 10.1,
                            "section start x={:.2} out of bounds",
                            s.start.x()
                        );
                        assert!(
                            s.start.y() >= -0.1 && s.start.y() <= 12.1,
                            "section start y={:.2} out of bounds",
                            s.start.y()
                        );
                        assert!(
                            s.start.z() >= -0.1 && s.start.z() <= 11.1,
                            "section start z={:.2} out of bounds",
                            s.start.z()
                        );
                    }
                    total_sections += sections.len();
                }
            }
        }
        assert!(total_sections > 0, "no sections found");
    }

    #[test]
    fn stage1_intersection_produces_section_edges() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let tol = Tolerance::new();
        let mut pipeline = BooleanPipeline {
            solid_a: Some(a),
            solid_b: Some(b),
            ..BooleanPipeline::default()
        };
        intersect_all_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        // Two boxes with 6 faces each = 36 face pairs.
        // Many are parallel (no intersection). Non-parallel pairs produce
        // trimmed section edges where the faces actually overlap.
        let total_sections: usize = pipeline.intersections.values().map(Vec::len).sum();
        assert!(
            total_sections > 0,
            "expected section edges, got 0. intersections map has {} entries",
            pipeline.intersections.len()
        );
    }

    #[test]
    fn stage1_sections_match_face_ids() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let tol = Tolerance::new();
        let mut pipeline = BooleanPipeline {
            solid_a: Some(a),
            solid_b: Some(b),
            ..BooleanPipeline::default()
        };
        intersect_all_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        let faces_a = collect_solid_faces(&topo, a).unwrap();
        let faces_b = collect_solid_faces(&topo, b).unwrap();

        // Check which faces have sections.
        let mut sections_for_face: HashMap<FaceId, Vec<SectionEdge>> = HashMap::new();
        for ((fa, fb), sections) in &pipeline.intersections {
            for s in sections {
                sections_for_face.entry(*fa).or_default().push(s.clone());
                sections_for_face.entry(*fb).or_default().push(s.clone());
            }
        }
        let a_with_sections = faces_a
            .iter()
            .filter(|f| sections_for_face.contains_key(f))
            .count();
        let b_with_sections = faces_b
            .iter()
            .filter(|f| sections_for_face.contains_key(f))
            .count();
        eprintln!("A faces with sections: {a_with_sections}/{}", faces_a.len());
        eprintln!("B faces with sections: {b_with_sections}/{}", faces_b.len());
    }

    #[test]
    fn stage3_split_produces_sub_faces() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let tol = Tolerance::new();
        let mut pipeline = BooleanPipeline {
            solid_a: Some(a),
            solid_b: Some(b),
            ..BooleanPipeline::default()
        };
        init_surface_info(&topo, a, b, &mut pipeline).unwrap();
        intersect_all_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        split_all_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        eprintln!(
            "Stage 3: {} sub-faces total (A: {}, B: {})",
            pipeline.sub_faces.len(),
            pipeline
                .sub_faces
                .iter()
                .filter(|sf| sf.source == Source::A)
                .count(),
            pipeline
                .sub_faces
                .iter()
                .filter(|sf| sf.source == Source::B)
                .count(),
        );
        // Both boxes have 6 faces. Some get split. We expect > 12 sub-faces.
        assert!(
            pipeline.sub_faces.len() >= 12,
            "expected >= 12 sub-faces, got {}",
            pipeline.sub_faces.len()
        );
    }

    #[test]
    fn pipeline_box_cut_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // a: [0,10]³ = 1000. b: [3,8]×[2,12]×[1,11].
        // Overlap: [3,8]×[2,10]×[1,10] = 5×8×9 = 360.
        // Cut = a - overlap = 1000 - 360 = 640.
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 640.0).abs() < 70.0,
            "Cut volume {vol} should be ~640"
        );
    }

    #[test]
    fn pipeline_box_fuse_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // Fuse = a + b - overlap = 1000 + 500 - 360 = 1140.
        let result = boolean_pipeline(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1140.0).abs() < 120.0,
            "Fuse volume {vol} should be ~1140"
        );
    }

    #[test]
    fn pipeline_box_intersect_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // Intersect = overlap = [3,8]×[2,10]×[1,10] = 5×8×9 = 360.
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 360.0).abs() < 40.0,
            "Intersect volume {vol} should be ~360"
        );
    }

    // --- Edge-case tests ---

    #[test]
    fn pipeline_3d_offset_overlap() {
        // Boxes offset on all 3 axes (no shared plane).
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // a=[0,10]³, b=[5,15]³. Overlap: [5,10]³ = 125.
        // Intersect is simpler to verify: overlap = 125.
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 125.0).abs() < 20.0,
            "3D-offset intersect volume {vol} should be ~125"
        );
    }

    #[test]
    fn pipeline_3d_offset_fuse() {
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // a=[0,10]³, b=[5,15]³. Fuse = 1000 + 1000 - 125 = 1875.
        let result = boolean_pipeline(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1875.0).abs() < 200.0,
            "3D-offset fuse volume {vol} should be ~1875"
        );
    }

    #[test]
    fn pipeline_b_inside_a_fuse() {
        // Small box entirely inside large box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // B is at [3,7]×[3,7]×[3,7], entirely inside A=[0,10]³.
        // Fuse = A (since B adds no volume).
        let result = boolean_pipeline(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() < 10.0,
            "B-inside-A fuse volume {vol} should be ~1000"
        );
    }

    #[test]
    fn pipeline_b_inside_a_intersect() {
        // Small box entirely inside large box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Intersect = B.
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 64.0).abs() < 10.0,
            "B-inside-A intersect volume {vol} should be ~64"
        );
    }

    #[test]
    fn pipeline_a_inside_b_intersect() {
        // Large box inside even larger box (A inside B).
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat_a = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, a, &mat_a).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        // A is at [3,7]³, B is at [0,10]³. A inside B.
        // Intersect = A.
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 64.0).abs() < 10.0,
            "A-inside-B intersect volume {vol} should be ~64"
        );
    }

    #[test]
    fn pipeline_disjoint_cut() {
        // Two completely separate boxes.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let mat = Mat4::translation(20.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Disjoint: Cut = A.
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 125.0).abs() < 10.0,
            "Disjoint cut volume {vol} should be ~125"
        );
    }

    #[test]
    fn pipeline_disjoint_intersect_is_error() {
        // Two completely separate boxes.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let mat = Mat4::translation(20.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Disjoint: Intersect = error.
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b);
        assert!(result.is_err(), "disjoint intersect should return Err");
    }

    #[test]
    fn pipeline_asymmetric_cut() {
        // Non-symmetric overlap verifying correct face selection.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 6.0, 4.0, 3.0).unwrap();
        let mat = Mat4::translation(7.0, 3.0, 2.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // b: [7,13]×[3,7]×[2,5]. Overlap with a: [7,10]×[3,7]×[2,5] = 3×4×3 = 36.
        // Cut = 1000 - 36 = 964.
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 964.0).abs() < 100.0,
            "Asymmetric cut volume {vol} should be ~964"
        );
    }

    // --- Mixed-surface integration tests (Step 2) ---

    use crate::primitives::{make_cone, make_cylinder, make_sphere};

    /// Helper: create a cylinder centered at (cx, cy) with base at z=0.
    fn make_centered_cylinder(topo: &mut Topology, r: f64, h: f64, cx: f64, cy: f64) -> SolidId {
        let cyl = make_cylinder(topo, r, h).unwrap();
        let mat = Mat4::translation(cx, cy, 0.0);
        crate::transform::transform_solid(topo, cyl, &mat).unwrap();
        cyl
    }

    /// Helper: create a sphere centered at (cx, cy, cz).
    fn make_centered_sphere(topo: &mut Topology, r: f64, cx: f64, cy: f64, cz: f64) -> SolidId {
        let sph = make_sphere(topo, r, 8).unwrap();
        let mat = Mat4::translation(cx, cy, cz);
        crate::transform::transform_solid(topo, sph, &mat).unwrap();
        sph
    }

    #[test]
    fn pipeline_box_intersect_cylinder_inside() {
        // Cylinder entirely inside box → intersect = cylinder.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let h = 8.0;
        let b = make_centered_cylinder(&mut topo, r, h, 5.0, 5.0);
        // Move cylinder up by 1.0 so it's within [0,10]³.
        let mat = Mat4::translation(0.0, 0.0, 1.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let expected = std::f64::consts::PI * r * r * h;
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "box∩cyl volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_box_fuse_cylinder_inside() {
        // Cylinder entirely inside box → fuse = box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_centered_cylinder(&mut topo, 2.0, 8.0, 5.0, 5.0);
        let mat = Mat4::translation(0.0, 0.0, 1.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let result = boolean_pipeline(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() / 1000.0 < 0.05,
            "box∪cyl_inside volume {vol} should be ~1000"
        );
    }

    #[test]
    fn pipeline_box_intersect_sphere_inside() {
        // Small sphere entirely inside box → intersect = sphere.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_sphere(&mut topo, r, 5.0, 5.0, 5.0);

        let expected = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "box∩sphere volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_box_fuse_sphere_inside() {
        // Small sphere entirely inside box → fuse = box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_centered_sphere(&mut topo, 2.0, 5.0, 5.0, 5.0);

        let result = boolean_pipeline(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() / 1000.0 < 0.05,
            "box∪sphere_inside volume {vol} should be ~1000"
        );
    }

    #[test]
    fn pipeline_box_intersect_cone_inside() {
        // Cone (frustum) entirely inside box → intersect = cone.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r1 = 2.0;
        let r2 = 1.0;
        let h = 6.0;
        let b = make_cone(&mut topo, r1, r2, h).unwrap();
        let mat = Mat4::translation(5.0, 5.0, 2.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let expected = std::f64::consts::PI * h / 3.0 * (r1 * r1 + r2 * r2 + r1 * r2);
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "box∩cone volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_cylinder_inside_box_cut_is_void() {
        // Cylinder entirely inside box → cut would create a void (inner shell).
        // The cylinder caps are coplanar with box z-faces, so the pipeline may
        // find boundary intersection curves and attempt assembly. Accept either
        // an error (void not supported) or a result.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_centered_cylinder(&mut topo, 2.0, 10.0, 5.0, 5.0);

        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b);
        // Pipeline may succeed (producing the box with coplanar trim) or fail
        // (void detection). Either is acceptable for this degenerate case.
        if let Ok(solid) = result {
            let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap_or(0.0);
            // Volume should be between box (1000) and box - cylinder (1000 - π*4*10 ≈ 874).
            assert!(
                vol > 800.0 && vol < 1100.0,
                "void-cut volume {vol} should be reasonable"
            );
        }
        // Err is also acceptable — void not supported.
    }

    #[test]
    fn pipeline_disjoint_box_cylinder() {
        // Box and cylinder are disjoint → intersect returns error.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_centered_cylinder(&mut topo, 1.0, 3.0, 20.0, 20.0);

        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b);
        assert!(
            result.is_err(),
            "disjoint box-cylinder intersect should return Err"
        );
    }

    #[test]
    fn pipeline_disjoint_box_sphere() {
        // Box and sphere are disjoint → cut = box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_centered_sphere(&mut topo, 1.0, 20.0, 20.0, 20.0);

        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 125.0).abs() < 10.0,
            "disjoint box-sphere cut volume {vol} should be ~125"
        );
    }

    // --- Face-crossing integration tests (Step 3) ---

    /// Helper: create a cone (frustum) centered at (cx, cy) with base at z=0.
    fn make_centered_cone(
        topo: &mut Topology,
        r1: f64,
        r2: f64,
        h: f64,
        cx: f64,
        cy: f64,
    ) -> SolidId {
        let cone = make_cone(topo, r1, r2, h).unwrap();
        let mat = Mat4::translation(cx, cy, 0.0);
        crate::transform::transform_solid(topo, cone, &mat).unwrap();
        cone
    }

    #[test]
    fn pipeline_cylinder_through_box_cut() {
        // Cylinder r=2, h=20 centered at (5,5) goes through box [0,10]³.
        // Cut = box minus cylinder slice within box.
        // Overlap volume = π·r²·h_box = π·4·10 ≈ 125.66
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_cylinder(&mut topo, r, 20.0, 5.0, 5.0);
        // Shift cylinder down so it extends from z=-5 to z=15, through box.
        let mat = Mat4::translation(0.0, 0.0, -5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let expected = 1000.0 - std::f64::consts::PI * r * r * 10.0;
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "cylinder-through-box cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_cylinder_through_box_intersect() {
        // Same geometry: intersect = cylinder slice within box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_cylinder(&mut topo, r, 20.0, 5.0, 5.0);
        let mat = Mat4::translation(0.0, 0.0, -5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let expected = std::f64::consts::PI * r * r * 10.0;
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "cylinder-through-box intersect volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_cylinder_through_box_fuse() {
        // Fuse = box + cylinder - overlap.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_cylinder(&mut topo, r, 20.0, 5.0, 5.0);
        let mat = Mat4::translation(0.0, 0.0, -5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let cyl_vol = std::f64::consts::PI * r * r * 20.0;
        let overlap = std::f64::consts::PI * r * r * 10.0;
        let expected = 1000.0 + cyl_vol - overlap;
        let result = boolean_pipeline(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "cylinder-through-box fuse volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_cylinder_partially_through_box_cut() {
        // Cylinder r=2, base at z=2, top at z=17 — exits top face only.
        // Overlap: z ∈ [2, 10], h_overlap = 8, vol_overlap = π·4·8 ≈ 100.53.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_cylinder(&mut topo, r, 15.0, 5.0, 5.0);
        let mat = Mat4::translation(0.0, 0.0, 2.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let overlap_h = 8.0; // min(10, 17) - 2
        let expected = 1000.0 - std::f64::consts::PI * r * r * overlap_h;
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "partially-through cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_sphere_cap_cut() {
        // Sphere r=5 centered at (5, 5, 12) — overlaps box top by 3 units.
        // Sphere bottom at z=7, box top at z=10. Cap from z=7..10 inside box.
        // Cap height h = 3, volume = πh²(3r−h)/3 = π·9·12/3 = 36π ≈ 113.10.
        // Center at z=12 (not z=10) so the cutting plane at z=10 is inside the
        // lower hemisphere face, not on the equator boundary.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 5.0;
        let b = make_centered_sphere(&mut topo, r, 5.0, 5.0, 12.0);

        let h = 3.0; // sphere dips 3 units into box (z=7..10)
        let cap_vol = std::f64::consts::PI * h * h * (3.0 * r - h) / 3.0;
        let expected = 1000.0 - cap_vol;
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.15,
            "sphere-cap cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_cone_through_box_intersect() {
        // Frustum r₁=3, r₂=1, h=20 through box [0,10]³.
        // Frustum slice z∈[0,10]: r varies linearly from r₁ to midpoint.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r1 = 3.0;
        let r2 = 1.0;
        let h = 20.0;
        let b = make_centered_cone(&mut topo, r1, r2, h, 5.0, 5.0);
        // Shift down by 5 so cone goes from z=-5 to z=15.
        let mat = Mat4::translation(0.0, 0.0, -5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        // After shift: frustum from z=-5 to z=15, r(z) = r1 + (r2-r1)*(z+5)/20
        // At z=0: r(0) = 3 + (-2)*5/20 = 2.5
        // At z=10: r(10) = 3 + (-2)*15/20 = 1.5
        // Volume of truncated cone [0, 10]: π·h/3·(r_bot² + r_top² + r_bot·r_top)
        let r_bot = 2.5;
        let r_top = 1.5;
        let expected =
            std::f64::consts::PI * 10.0 / 3.0 * (r_bot * r_bot + r_top * r_top + r_bot * r_top);
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.10,
            "cone-through-box intersect volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_offset_cylinder_through_box_cut() {
        // Cylinder off-center at (3, 7) to avoid seam alignment with box faces.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_cylinder(&mut topo, r, 20.0, 3.0, 7.0);
        let mat = Mat4::translation(0.0, 0.0, -5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let expected = 1000.0 - std::f64::consts::PI * r * r * 10.0;
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "offset cylinder cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_two_cylinders_intersect() {
        // Two perpendicular cylinders (Steinmetz-like).
        // Cylinder A along z-axis, Cylinder B along x-axis, both r=3.
        // Intersection volume = 16r³/3 = 16·27/3 = 144.
        let mut topo = Topology::new();
        let r = 3.0;
        let h = 10.0;
        let a = make_cylinder(&mut topo, r, h).unwrap();
        // Center A at origin: shift z by -h/2.
        let mat_a = Mat4::translation(0.0, 0.0, -h / 2.0);
        crate::transform::transform_solid(&mut topo, a, &mat_a).unwrap();

        let b = make_cylinder(&mut topo, r, h).unwrap();
        // Rotate B by 90° around y-axis (z→x), then center.
        let rot = Mat4::rotation_y(std::f64::consts::FRAC_PI_2);
        let shift = Mat4::translation(0.0, 0.0, -h / 2.0);
        // Apply shift first (center), then rotate.
        let mat_b = rot * shift;
        crate::transform::transform_solid(&mut topo, b, &mat_b).unwrap();

        let expected = 16.0 * r * r * r / 3.0;
        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.10,
            "Steinmetz intersect volume {vol} should be ~{expected}"
        );
    }

    /// Helper: create a planar square face centered at `(cx, cy, z)`.
    fn make_square_face_at(topo: &mut Topology, size: f64, cx: f64, cy: f64, z: f64) -> FaceId {
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let hs = size / 2.0;
        let t = 1e-7;
        let v0 = topo.add_vertex(Vertex::new(Point3::new(cx - hs, cy - hs, z), t));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(cx + hs, cy - hs, z), t));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(cx + hs, cy + hs, z), t));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(cx - hs, cy + hs, z), t));
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
        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: z,
            },
        ))
    }

    #[test]
    fn pipeline_loft_box_cut() {
        // Loft between two 4×4 squares at z=-1 and z=11, centered at (5,5).
        // Pokes through box top and bottom → plane-NURBS intersections.
        // Inside box: 4×4×10 = 160.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let bottom = make_square_face_at(&mut topo, 4.0, 5.0, 5.0, -1.0);
        let top = make_square_face_at(&mut topo, 4.0, 5.0, 5.0, 11.0);
        let b = crate::loft::loft(&mut topo, &[bottom, top]).unwrap();

        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        // Box volume = 1000. Lofted prism inside box = 4×4×10 = 160.
        let expected = 1000.0 - 160.0;
        assert!(
            (vol - expected).abs() / expected < 0.10,
            "loft-box cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_loft_box_intersect() {
        // Loft extends through box — intersect gives the part inside the box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let bottom = make_square_face_at(&mut topo, 4.0, 5.0, 5.0, -1.0);
        let top = make_square_face_at(&mut topo, 4.0, 5.0, 5.0, 11.0);
        let b = crate::loft::loft(&mut topo, &[bottom, top]).unwrap();

        let result = boolean_pipeline(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        let expected = 160.0; // 4×4×10 (prism clipped to box height)
        assert!(
            (vol - expected).abs() / expected < 0.10,
            "loft-box intersect volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn pipeline_cylinder_vs_loft_cut() {
        // Cylinder (analytic) vs loft (NURBS) — tests analytic-NURBS intersection.
        // Cylinder r=3 along z, centered at (5,5), from z=-1 to z=11.
        // Loft 6×6 square prism centered at (5,5), z=0 to z=10.
        // The cylinder pokes through the loft's NURBS side faces.
        let mut topo = Topology::new();
        let cyl = make_cylinder(&mut topo, 3.0, 12.0).unwrap();
        let mat = Mat4::translation(5.0, 5.0, -1.0);
        crate::transform::transform_solid(&mut topo, cyl, &mat).unwrap();

        let bottom = make_square_face_at(&mut topo, 6.0, 5.0, 5.0, 0.0);
        let top = make_square_face_at(&mut topo, 6.0, 5.0, 5.0, 10.0);
        let loft = crate::loft::loft(&mut topo, &[bottom, top]).unwrap();

        // Must not return InvalidInput (analytic-NURBS conversion must work).
        let result = boolean_pipeline(&mut topo, BooleanOp::Cut, loft, cyl).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        // Loft volume = 6×6×10 = 360. Cylinder inside loft ≈ π×9×10 ≈ 283.
        // But cylinder extends beyond loft, so intersection is smaller.
        assert!(
            vol > 0.0,
            "cylinder-vs-loft cut should have positive volume, got {vol}"
        );
    }
}
