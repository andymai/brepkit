//! Boolean v2: OCCT-style parameter-space pipeline.
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform a boolean operation using the v2 parameter-space pipeline.
///
/// Supports solids composed entirely of analytic faces: `Plane`, `Cylinder`,
/// `Cone`, `Sphere`, `Torus`. Returns `Err` for solids containing NURBS faces.
pub fn boolean_v2(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, OperationsError> {
    let tol = Tolerance::new();

    // Guard: all faces must be analytic (no NURBS).
    if !all_faces_analytic(topo, a)? || !all_faces_analytic(topo, b)? {
        return Err(OperationsError::InvalidInput {
            reason: "boolean_v2: only analytic faces (plane/cylinder/cone/sphere/torus) supported"
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
        return handle_disjoint_v2(
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

    // Stage 5: Assemble result.
    let result = assemble_v2(topo, &pipeline, &tol)?;

    // Post-processing: healing.
    crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    crate::heal::unify_faces(topo, result)?;

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
    pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;
    let surf_a = face_a.surface();
    let surf_b = face_b.surface();

    match (surf_a, surf_b) {
        (FaceSurface::Plane { .. }, FaceSurface::Plane { .. }) => {
            // Plane-plane: existing intersection.
            intersect_two_plane_faces(topo, fa, fb, &pipeline.plane_frames)
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
        _ => {
            // NURBS or unsupported pair — skip.
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
            } else if (end_3d - start_3d).length() >= tol.linear {
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
                });
            }
        }
    }

    Ok(result)
}

/// Compute the intersection of two plane faces.
///
/// Returns trimmed section edges (finite line segments where the faces overlap).
fn intersect_two_plane_faces(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    plane_frames: &HashMap<FaceId, PlaneFrame>,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;

    // Guarded by `all_faces_plane()` at the entry point of `boolean_v2()`.
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

    // Plane-plane intersection: line direction = na × nb.
    let line_dir = na.cross(nb);
    let line_len = line_dir.length();
    if line_len < 1e-10 {
        // Parallel or coincident planes — no intersection curve.
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
            if t1 - t0 < 1e-10 {
                continue; // No overlap or degenerate.
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
            });
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Stage 3: Split faces
// ---------------------------------------------------------------------------

fn split_all_faces(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    pipeline: &mut BooleanPipeline,
    tol: &Tolerance,
) -> Result<(), OperationsError> {
    let faces_a = collect_solid_faces(topo, a)?;
    let faces_b = collect_solid_faces(topo, b)?;

    // Collect section edges per face.
    let mut sections_for_face: HashMap<FaceId, Vec<SectionEdge>> = HashMap::new();
    for ((fa, fb), sections) in &pipeline.intersections {
        for s in sections {
            sections_for_face.entry(*fa).or_default().push(s.clone());
            sections_for_face.entry(*fb).or_default().push(s.clone());
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

    // Classify each sub-face and mark for selection.
    let mut selected = Vec::new();
    for sub_face in &pipeline.sub_faces {
        // Get test point inside this sub-face.
        let frame = pipeline
            .surface_info
            .get(&sub_face.parent)
            .and_then(SurfaceInfo::as_plane_frame);
        let test_pt = interior_point_3d(sub_face, frame);

        // Classify against opposing solid: try analytic first, fall back to ray-cast.
        // If the interior point is on the boundary (classifier returns None),
        // try wire vertices for a definitive answer before falling back to
        // the ray-cast (which can also be degenerate at corners/edges).
        let class = match sub_face.source {
            Source::A => {
                classify_with_fallback(test_pt, &sub_face.outer_wire, classifier_b, &polys_b, *tol)
            }
            Source::B => {
                classify_with_fallback(test_pt, &sub_face.outer_wire, classifier_a, &polys_a, *tol)
            }
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

/// Classify a sub-face against the opposing solid with robust boundary handling.
///
/// 1. Try the analytic classifier on the interior point.
/// 2. If on-boundary (None), try each wire vertex until one gives a definitive answer.
/// 3. Fall back to ray-casting if no analytic result.
fn classify_with_fallback(
    test_pt: Point3,
    wire: &[super::pipeline::OrientedPCurveEdge],
    classifier: Option<&AnalyticClassifier>,
    face_polys: &[(Vec<Point3>, Vec3, f64)],
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
    // Fall back to ray-casting.
    classify_point_against_solid(test_pt, face_polys)
}

/// Classify a 3D point as inside/outside/on a solid using face polygon ray-casting.
fn classify_point_against_solid(
    point: Point3,
    face_polys: &[(Vec<Point3>, Vec3, f64)],
) -> FaceClass {
    // Cast a ray along +Z and count crossings using even-odd parity.
    // Odd crossing count = inside, even = outside.
    let ray_dir = Vec3::new(0.0, 0.0, 1.0);
    let mut crossings = 0u32;

    for (verts, normal, d) in face_polys {
        // Check if the ray from `point` in `ray_dir` crosses this face polygon.
        let denom = normal.dot(ray_dir);
        if denom.abs() < 1e-15 {
            continue; // Ray parallel to face.
        }
        let t = (*d - normal.dot(Vec3::new(point.x(), point.y(), point.z()))) / denom;
        if t < 1e-10 {
            continue; // Intersection behind ray origin.
        }
        let hit = point + ray_dir * t;
        // Check if hit point is inside the face polygon using 2D projection.
        // Project onto the face's dominant plane.
        if point_in_face_polygon_3d(hit, verts, normal) {
            crossings += 1;
        }
    }

    if crossings != 0 && crossings % 2 != 0 {
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

fn assemble_v2(
    topo: &mut Topology,
    pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<SolidId, OperationsError> {
    // Collect all 3D points for vertex merge resolution.
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

    let mut face_ids = Vec::new();

    for sub_face in &pipeline.sub_faces {
        // Create vertices + edges + wire for the outer boundary.
        let wire_id =
            create_wire_from_edges_dedup(topo, &sub_face.outer_wire, &mut vertex_map, resolution)?;
        let inner_wires: Vec<_> = sub_face
            .inner_wires
            .iter()
            .filter_map(|inner| {
                create_wire_from_edges_dedup(topo, inner, &mut vertex_map, resolution).ok()
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
            reason: "boolean_v2: no faces in result".into(),
        });
    }

    let shell = Shell::new(face_ids).map_err(OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = Solid::new(shell_id, Vec::new());
    Ok(topo.add_solid(solid))
}

fn create_wire_from_edges_dedup(
    topo: &mut Topology,
    edges: &[super::pipeline::OrientedPCurveEdge],
    vertex_map: &mut HashMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    resolution: f64,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    let mut oriented_edges = Vec::new();
    for pe in edges {
        // When pe.forward=false, the wire traverses the edge backward
        // (edge.end → edge.start). To make the backward traversal go
        // pe.start_3d → pe.end_3d, swap the vertex roles so
        // edge.end = pe.start_3d (traversal start) and
        // edge.start = pe.end_3d (traversal end).
        let (natural_start, natural_end) = if pe.forward {
            (pe.start_3d, pe.end_3d)
        } else {
            (pe.end_3d, pe.start_3d)
        };
        let key_start = quantize_point(natural_start, resolution);
        let v_start = *vertex_map
            .entry(key_start)
            .or_insert_with(|| topo.add_vertex(Vertex::new(natural_start, 0.0)));
        let key_end = quantize_point(natural_end, resolution);
        let v_end = *vertex_map
            .entry(key_end)
            .or_insert_with(|| topo.add_vertex(Vertex::new(natural_end, 0.0)));
        let edge = Edge::new(v_start, v_end, pe.curve_3d.clone());
        let eid = topo.add_edge(edge);
        oriented_edges.push(OrientedEdge::new(eid, pe.forward));
    }
    let wire = Wire::new(oriented_edges, true).map_err(OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn all_faces_analytic(topo: &Topology, solid: SolidId) -> Result<bool, OperationsError> {
    let faces = collect_solid_faces(topo, solid)?;
    for fid in faces {
        // is_analytic() returns true for Plane, Cylinder, Cone, Sphere, Torus.
        // Only Nurbs is rejected.
        if !topo.face(fid)?.surface().is_analytic() {
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

fn collect_face_polygon(topo: &Topology, face_id: FaceId) -> Result<Vec<Point3>, OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut pts = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        pts.push(topo.vertex(vid)?.point());
    }
    Ok(pts)
}

fn collect_solid_face_polygons(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<(Vec<Point3>, Vec3, f64)>, OperationsError> {
    let faces = collect_solid_faces(topo, solid)?;
    let mut result = Vec::new();
    for fid in faces {
        let face = topo.face(fid)?;
        let poly = collect_face_polygon(topo, fid)?;
        if poly.len() < 3 {
            continue;
        }
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let effective_normal = if face.is_reversed() {
                -*normal
            } else {
                *normal
            };
            let effective_d = if face.is_reversed() { -*d } else { *d };
            result.push((poly, effective_normal, effective_d));
        } else {
            // Approximate normal/d from polygon for ray-cast fallback.
            let (normal, d) = approximate_polygon_plane(&poly);
            let effective_normal = if face.is_reversed() { -normal } else { normal };
            let effective_d = if face.is_reversed() { -d } else { d };
            result.push((poly, effective_normal, effective_d));
        }
    }
    Ok(result)
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

fn handle_disjoint_v2(
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
                    "boolean_v2: B is inside A — cut would create void (not supported in step 1)"
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
                reason: "boolean_v2: A is inside B — cut result is empty".into(),
            }),
            BooleanOp::Intersect => Ok(crate::copy::copy_solid(topo, a)?),
        };
    }

    // Truly disjoint.
    match op {
        BooleanOp::Fuse => {
            // For disjoint fuse, return a copy of A.
            // TODO: Step 1 limitation — this discards solid B. A full
            // implementation should copy both solids into a compound or
            // merged shell so the result contains all geometry.
            Ok(crate::copy::copy_solid(topo, a)?)
        }
        BooleanOp::Cut => {
            // A - B where B doesn't touch A → result is A.
            Ok(crate::copy::copy_solid(topo, a)?)
        }
        BooleanOp::Intersect => {
            // A ∩ B where they don't touch → empty.
            Err(OperationsError::InvalidInput {
                reason: "boolean_v2: intersection of disjoint solids is empty".into(),
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
        // For closed circle edges (start ≈ end), evaluate starting from the
        // vertex angle instead of the Circle3D's parametric origin. This keeps
        // the UV samples aligned with seam edge endpoints.
        let is_closed_circle = matches!(edge.curve(), EdgeCurve::Circle(_))
            && (start_v - end_v).length() < Tolerance::new().linear;
        #[allow(clippy::cast_precision_loss)]
        for i in 0..SAMPLES_PER_EDGE {
            let t = i as f64 / SAMPLES_PER_EDGE as f64;
            let p3d = if is_closed_circle {
                if let EdgeCurve::Circle(circle) = edge.curve() {
                    // Find the angle of the vertex on the circle.
                    let vertex_angle = circle.project(start_v);
                    let angle = if oe.is_forward() {
                        vertex_angle + std::f64::consts::TAU * t
                    } else {
                        vertex_angle - std::f64::consts::TAU * t
                    };
                    circle.evaluate(angle)
                } else {
                    evaluate_edge_at_t(edge.curve(), start_v, end_v, t)
                }
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
        if (v_max - v_min) < 0.1 {
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
                let sections = intersect_two_plane_faces(&topo, fa, fb, &frames).unwrap();
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
    fn boolean_v2_box_cut_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // a: [0,10]³ = 1000. b: [3,8]×[2,12]×[1,11].
        // Overlap: [3,8]×[2,10]×[1,10] = 5×8×9 = 360.
        // Cut = a - overlap = 1000 - 360 = 640.
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 640.0).abs() < 70.0,
            "Cut volume {vol} should be ~640"
        );
    }

    #[test]
    fn boolean_v2_box_fuse_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // Fuse = a + b - overlap = 1000 + 500 - 360 = 1140.
        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1140.0).abs() < 120.0,
            "Fuse volume {vol} should be ~1140"
        );
    }

    #[test]
    fn boolean_v2_box_intersect_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // Intersect = overlap = [3,8]×[2,10]×[1,10] = 5×8×9 = 360.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 360.0).abs() < 40.0,
            "Intersect volume {vol} should be ~360"
        );
    }

    // --- Edge-case tests ---

    #[test]
    fn boolean_v2_3d_offset_overlap() {
        // Boxes offset on all 3 axes (no shared plane).
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // a=[0,10]³, b=[5,15]³. Overlap: [5,10]³ = 125.
        // Intersect is simpler to verify: overlap = 125.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 125.0).abs() < 20.0,
            "3D-offset intersect volume {vol} should be ~125"
        );
    }

    #[test]
    fn boolean_v2_3d_offset_fuse() {
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // a=[0,10]³, b=[5,15]³. Fuse = 1000 + 1000 - 125 = 1875.
        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1875.0).abs() < 200.0,
            "3D-offset fuse volume {vol} should be ~1875"
        );
    }

    #[test]
    fn boolean_v2_b_inside_a_fuse() {
        // Small box entirely inside large box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // B is at [3,7]×[3,7]×[3,7], entirely inside A=[0,10]³.
        // Fuse = A (since B adds no volume).
        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() < 10.0,
            "B-inside-A fuse volume {vol} should be ~1000"
        );
    }

    #[test]
    fn boolean_v2_b_inside_a_intersect() {
        // Small box entirely inside large box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Intersect = B.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 64.0).abs() < 10.0,
            "B-inside-A intersect volume {vol} should be ~64"
        );
    }

    #[test]
    fn boolean_v2_a_inside_b_intersect() {
        // Large box inside even larger box (A inside B).
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat_a = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, a, &mat_a).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        // A is at [3,7]³, B is at [0,10]³. A inside B.
        // Intersect = A.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 64.0).abs() < 10.0,
            "A-inside-B intersect volume {vol} should be ~64"
        );
    }

    #[test]
    fn boolean_v2_disjoint_cut() {
        // Two completely separate boxes.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let mat = Mat4::translation(20.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Disjoint: Cut = A.
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 125.0).abs() < 10.0,
            "Disjoint cut volume {vol} should be ~125"
        );
    }

    #[test]
    fn boolean_v2_disjoint_intersect_is_error() {
        // Two completely separate boxes.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let mat = Mat4::translation(20.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Disjoint: Intersect = error.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b);
        assert!(result.is_err(), "disjoint intersect should return Err");
    }

    #[test]
    fn boolean_v2_asymmetric_cut() {
        // Non-symmetric overlap verifying correct face selection.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 6.0, 4.0, 3.0).unwrap();
        let mat = Mat4::translation(7.0, 3.0, 2.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // b: [7,13]×[3,7]×[2,5]. Overlap with a: [7,10]×[3,7]×[2,5] = 3×4×3 = 36.
        // Cut = 1000 - 36 = 964.
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
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
    fn boolean_v2_box_intersect_cylinder_inside() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "box∩cyl volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_box_fuse_cylinder_inside() {
        // Cylinder entirely inside box → fuse = box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_centered_cylinder(&mut topo, 2.0, 8.0, 5.0, 5.0);
        let mat = Mat4::translation(0.0, 0.0, 1.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() / 1000.0 < 0.05,
            "box∪cyl_inside volume {vol} should be ~1000"
        );
    }

    #[test]
    fn boolean_v2_box_intersect_sphere_inside() {
        // Small sphere entirely inside box → intersect = sphere.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_sphere(&mut topo, r, 5.0, 5.0, 5.0);

        let expected = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "box∩sphere volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_box_fuse_sphere_inside() {
        // Small sphere entirely inside box → fuse = box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_centered_sphere(&mut topo, 2.0, 5.0, 5.0, 5.0);

        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() / 1000.0 < 0.05,
            "box∪sphere_inside volume {vol} should be ~1000"
        );
    }

    #[test]
    fn boolean_v2_box_intersect_cone_inside() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "box∩cone volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_cylinder_inside_box_cut_is_void() {
        // Cylinder entirely inside box → cut would create a void (inner shell).
        // The cylinder caps are coplanar with box z-faces, so the pipeline may
        // find boundary intersection curves and attempt assembly. Accept either
        // an error (void not supported) or a result.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_centered_cylinder(&mut topo, 2.0, 10.0, 5.0, 5.0);

        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b);
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
    fn boolean_v2_disjoint_box_cylinder() {
        // Box and cylinder are disjoint → intersect returns error.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_centered_cylinder(&mut topo, 1.0, 3.0, 20.0, 20.0);

        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b);
        assert!(
            result.is_err(),
            "disjoint box-cylinder intersect should return Err"
        );
    }

    #[test]
    fn boolean_v2_disjoint_box_sphere() {
        // Box and sphere are disjoint → cut = box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_centered_sphere(&mut topo, 1.0, 20.0, 20.0, 20.0);

        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
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
    fn boolean_v2_cylinder_through_box_cut() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "cylinder-through-box cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_cylinder_through_box_intersect() {
        // Same geometry: intersect = cylinder slice within box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_cylinder(&mut topo, r, 20.0, 5.0, 5.0);
        let mat = Mat4::translation(0.0, 0.0, -5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let expected = std::f64::consts::PI * r * r * 10.0;
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "cylinder-through-box intersect volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_cylinder_through_box_fuse() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "cylinder-through-box fuse volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_cylinder_partially_through_box_cut() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "partially-through cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_sphere_cap_cut() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.10,
            "sphere-cap cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_cone_through_box_intersect() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.10,
            "cone-through-box intersect volume {vol} should be ~{expected}"
        );
    }

    #[test]
    fn boolean_v2_offset_cylinder_through_box_cut() {
        // Cylinder off-center at (3, 7) to avoid seam alignment with box faces.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let r = 2.0;
        let b = make_centered_cylinder(&mut topo, r, 20.0, 3.0, 7.0);
        let mat = Mat4::translation(0.0, 0.0, -5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let expected = 1000.0 - std::f64::consts::PI * r * r * 10.0;
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "offset cylinder cut volume {vol} should be ~{expected}"
        );
    }

    #[test]
    #[ignore = "Steinmetz (cylinder-cylinder) needs analytic-analytic pipeline improvements"]
    fn boolean_v2_two_cylinders_intersect() {
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
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        assert!(
            (vol - expected).abs() / expected < 0.10,
            "Steinmetz intersect volume {vol} should be ~{expected}"
        );
    }
}
