//! Phase FF: Face-face intersection detection.
//!
//! For each (face_a, face_b) pair across solids, computes intersection
//! curves. Results are stored as `IntersectionCurveDS` entries in the
//! GFA arena, with FF interferences referencing them by index.
//!
//! Each raw curve also gets a pave block spanning its full parameter
//! range, with topology vertices and an edge created at the endpoints.

use brepkit_math::aabb::Aabb3;
use brepkit_math::analytic_intersection;
use brepkit_math::nurbs::intersection as nurbs_isect;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use crate::ds::{GfaArena, Interference, IntersectionCurveDS, Pave, PaveBlock};
use crate::error::AlgoError;

/// Default number of samples for NURBS intersection.
const NURBS_SAMPLES: usize = 32;

/// Default march step for NURBS-NURBS intersection.
const NURBS_MARCH_STEP: f64 = 0.01;

/// Detect face-face intersections between the two solids.
///
/// For each face pair (one from each solid), computes intersection
/// curves using surface-type-specific algorithms. Raw intersection
/// curves are stored in the arena without trimming to face boundaries
/// (boundary trimming is a later phase).
///
/// Creates topology vertices and edges for each intersection curve
/// endpoint, and a pave block spanning the full parameter range.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup or intersection computation fails.
#[allow(clippy::too_many_lines)]
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let faces_a = brepkit_topology::explorer::solid_faces(topo, solid_a)?;
    let faces_b = brepkit_topology::explorer::solid_faces(topo, solid_b)?;

    // Pre-compute face AABBs for rejection
    let bboxes_a = compute_face_bboxes(topo, &faces_a)?;
    let bboxes_b = compute_face_bboxes(topo, &faces_b)?;

    // Collect all surface data upfront so we don't borrow topo immutably
    // while mutating it later.
    let surfs_a: Vec<FaceSurface> = faces_a
        .iter()
        .map(|&fa| topo.face(fa).map(|f| f.surface().clone()))
        .collect::<Result<_, _>>()?;
    let surfs_b: Vec<FaceSurface> = faces_b
        .iter()
        .map(|&fb| topo.face(fb).map(|f| f.surface().clone()))
        .collect::<Result<_, _>>()?;

    // Pre-compute v-parameter ranges for analytic surfaces (used by AA intersection)
    let v_ranges_a: Vec<Option<(f64, f64)>> = faces_a
        .iter()
        .zip(surfs_a.iter())
        .map(|(&fid, surf)| face_v_range(topo, fid, surf))
        .collect();
    let v_ranges_b: Vec<Option<(f64, f64)>> = faces_b
        .iter()
        .zip(surfs_b.iter())
        .map(|(&fid, surf)| face_v_range(topo, fid, surf))
        .collect();

    for (idx_a, &fa) in faces_a.iter().enumerate() {
        let bbox_a = &bboxes_a[idx_a];
        let surf_a = &surfs_a[idx_a];

        for (idx_b, &fb) in faces_b.iter().enumerate() {
            let bbox_b = &bboxes_b[idx_b];

            // AABB rejection
            if !bbox_a
                .expanded(tol.linear)
                .intersects(bbox_b.expanded(tol.linear))
            {
                continue;
            }

            let surf_b = &surfs_b[idx_b];

            // Compute raw intersection curves
            let v_range_a = v_ranges_a[idx_a];
            let v_range_b = v_ranges_b[idx_b];
            let raw_curves =
                compute_raw_curves(surf_a, surf_b, bbox_a, bbox_b, v_range_a, v_range_b)?;

            // For plane-plane Line curves with all-straight-edge faces, filter
            // curves whose clipped ranges on each face have a LARGE gap (no
            // overlap). A small gap or touching boundary is kept (the face
            // splitter handles final trimming). Only curves with a significant
            // separation (>10% of curve length) between the two ranges are
            // rejected as truly spurious.
            let raw_curves: Vec<RawCurve> = if matches!(surf_a, FaceSurface::Plane { .. })
                && matches!(surf_b, FaceSurface::Plane { .. })
            {
                raw_curves
                    .into_iter()
                    .filter(|raw| {
                        if !matches!(raw.curve, EdgeCurve::Line) {
                            return true;
                        }
                        let range_a = clip_line_to_face(topo, fa, raw);
                        let range_b = clip_line_to_face(topo, fb, raw);
                        match (range_a, range_b) {
                            (None, None) => false,                     // Outside both faces
                            (Some(_), None) | (None, Some(_)) => true, // Inside one face
                            (Some(a), Some(b)) => {
                                // Check gap between ranges
                                let t_min = a.0.max(b.0);
                                let t_max = a.1.min(b.1);
                                let gap = t_min - t_max;
                                // Keep if gap < 10% of curve length (allows touching)
                                gap < 0.1
                            }
                        }
                    })
                    .collect()
            } else {
                raw_curves
            };

            for raw in raw_curves {
                // Closed Circle3D sections — produced by plane-sphere
                // intersections — get split at face-boundary crossings so
                // downstream face splitters see open arcs they can match
                // up. Without this, the closed curve is dropped by
                // `split_noseam_face_direct` and the spherical sub-face is
                // lost entirely.
                //
                // Note: the split here is structurally correct (the FF
                // arcs now have proper endpoints on the analytic face's
                // boundary), but `split_noseam_face_direct` still can't
                // form a single closed cap loop when 2+ arcs cross on a
                // sphere hemisphere (e.g. the great circles from x=0 and
                // y=0 planes meeting at the pole). The hemisphere falls
                // back to "unsplit" → the spherical sub-face is missing
                // from the GFA output → the boolean pipeline retries with
                // mesh boolean. Closing that gap requires generalising
                // the face splitter for multi-arc sphere hemispheres —
                // see `crates/algo/src/builder/face_splitter/mod.rs:138`
                // dispatch and `special_cases.rs::split_noseam_face_direct`.
                if let EdgeCurve::Circle(circle) = &raw.curve {
                    let is_closed = (raw.p_start - raw.p_end).length() < tol.linear;
                    if is_closed {
                        let crossings = closed_circle_boundary_crossings(topo, fa, fb, circle, tol);
                        if crossings.len() >= 2 {
                            emit_split_circle_arcs(
                                topo, arena, fa, fb, &raw, circle, &crossings, tol,
                            );
                            continue;
                        }
                    }
                }

                // Create topology vertices at the curve endpoints.
                // For closed curves (Circle/Ellipse), start and end are the same
                // 3D point — reuse one vertex for correct seam topology.
                //
                // Snap to existing vertices (from input face boundaries or
                // earlier intersection curves) when within tolerance. This is
                // the PutPavesOnCurve equivalent: it ensures intersection curve
                // endpoints share vertices with face boundaries, so the face
                // splitter produces sub-faces with consistent vertex identity.
                let is_closed = (raw.p_start - raw.p_end).length() < tol.linear;
                let start_vid =
                    super::helpers::find_nearby_pave_vertex(topo, arena, raw.p_start, tol)
                        .or_else(|| find_nearby_face_vertex(topo, fa, raw.p_start, tol))
                        .or_else(|| find_nearby_face_vertex(topo, fb, raw.p_start, tol))
                        .unwrap_or_else(|| topo.add_vertex(Vertex::new(raw.p_start, tol.linear)));
                let end_vid = if is_closed {
                    start_vid
                } else {
                    super::helpers::find_nearby_pave_vertex(topo, arena, raw.p_end, tol)
                        .or_else(|| find_nearby_face_vertex(topo, fa, raw.p_end, tol))
                        .or_else(|| find_nearby_face_vertex(topo, fb, raw.p_end, tol))
                        .unwrap_or_else(|| topo.add_vertex(Vertex::new(raw.p_end, tol.linear)))
                };

                // Create a topology edge for this intersection curve.
                let edge = Edge::new(start_vid, end_vid, raw.curve.clone());
                let edge_id = topo.add_edge(edge);

                // Create a pave block spanning the full parameter range.
                let start_pave = Pave::new(start_vid, raw.t_range.0);
                let end_pave = Pave::new(end_vid, raw.t_range.1);
                let pb = PaveBlock::new(edge_id, start_pave, end_pave);
                let pb_id = arena.pave_blocks.alloc(pb);

                let curve_index = arena.curves.len();
                arena.curves.push(IntersectionCurveDS {
                    curve: raw.curve,
                    face_a: fa,
                    face_b: fb,
                    bbox: raw.bbox,
                    pave_blocks: vec![pb_id],
                    t_range: raw.t_range,
                });

                arena.interference.ff.push(Interference::FF {
                    f1: fa,
                    f2: fb,
                    curve_index,
                });

                log::debug!(
                    "FF: faces {fa:?} and {fb:?} intersect (curve_index={curve_index}, \
                     edge={edge_id:?}, pb={pb_id:?})",
                );
            }
        }
    }

    Ok(())
}

/// Compute AABB for a face by sampling its boundary edges.
fn compute_face_bbox(topo: &Topology, face_id: FaceId) -> Result<Aabb3, AlgoError> {
    let edges = brepkit_topology::explorer::face_edges(topo, face_id)?;
    let mut points = Vec::new();

    for eid in edges {
        let edge = topo.edge(eid)?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);

        // Sample edge at several points
        let n: usize = 8;
        for i in 0..=n {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
            points.push(pt);
        }
    }

    if points.is_empty() {
        // Degenerate face with no edges -- use a zero-volume box at origin
        Ok(Aabb3 {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(0.0, 0.0, 0.0),
        })
    } else {
        Ok(Aabb3::from_points(points))
    }
}

/// Compute AABBs for a list of faces.
fn compute_face_bboxes(topo: &Topology, faces: &[FaceId]) -> Result<Vec<Aabb3>, AlgoError> {
    let mut bboxes = Vec::with_capacity(faces.len());
    for &fid in faces {
        bboxes.push(compute_face_bbox(topo, fid)?);
    }
    Ok(bboxes)
}

/// Compute the v-parameter range of a face by projecting boundary vertices.
/// Returns `None` for planes (which have no UV parameterization) or if projection fails.
fn face_v_range(topo: &Topology, face_id: FaceId, surface: &FaceSurface) -> Option<(f64, f64)> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let sp = topo.vertex(edge.start()).ok()?.point();
        let ep = topo.vertex(edge.end()).ok()?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        // Sample 5 points to capture v-extremes on curved/closed edges
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let t = t0 + (t1 - t0) * frac;
            let pt = edge.curve().evaluate_with_endpoints(t, sp, ep);
            if let Some((_, v)) = surface.project_point(pt) {
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
    }
    if v_min < v_max {
        Some((v_min, v_max))
    } else {
        None
    }
}

/// Intermediate intersection result before face IDs are assigned.
struct RawCurve {
    /// The 3D curve geometry.
    curve: EdgeCurve,
    /// Bounding box of the curve.
    bbox: Aabb3,
    /// Parameter range on the curve.
    t_range: (f64, f64),
    /// 3D position at the start of the parameter range.
    p_start: Point3,
    /// 3D position at the end of the parameter range.
    p_end: Point3,
}

/// Compute raw intersection curves between two surfaces.
///
/// Dispatches by surface type pair. Raw curves are returned without
/// trimming to face boundaries.
#[allow(clippy::too_many_lines)]
fn compute_raw_curves(
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
    bbox_a: &Aabb3,
    bbox_b: &Aabb3,
    v_range_a: Option<(f64, f64)>,
    v_range_b: Option<(f64, f64)>,
) -> Result<Vec<RawCurve>, AlgoError> {
    match (surf_a, surf_b) {
        // Plane-Plane
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            plane_plane_intersection(*na, *da, *nb, *db, bbox_a, bbox_b)
        }

        // Plane-Analytic (plane is A)
        (FaceSurface::Plane { normal, d }, other) if other.as_analytic().is_some() => {
            if let Some(analytic) = other.as_analytic() {
                plane_analytic_intersection(*normal, *d, &analytic)
            } else {
                Ok(Vec::new())
            }
        }

        // Analytic-Plane (plane is B, swap)
        (other, FaceSurface::Plane { normal, d }) if other.as_analytic().is_some() => {
            if let Some(analytic) = other.as_analytic() {
                plane_analytic_intersection(*normal, *d, &analytic)
            } else {
                Ok(Vec::new())
            }
        }

        // Analytic-Analytic
        (a, b) if a.as_analytic().is_some() && b.as_analytic().is_some() => {
            if let (Some(aa), Some(ab)) = (a.as_analytic(), b.as_analytic()) {
                analytic_analytic_intersection(&aa, &ab, v_range_a, v_range_b)
            } else {
                Ok(Vec::new())
            }
        }

        // Plane-NURBS
        (FaceSurface::Plane { normal, d }, FaceSurface::Nurbs(nurbs))
        | (FaceSurface::Nurbs(nurbs), FaceSurface::Plane { normal, d }) => {
            plane_nurbs_intersection(*normal, *d, nurbs)
        }

        // Analytic-NURBS or NURBS-Analytic
        (analytic_surf, FaceSurface::Nurbs(nurbs)) if analytic_surf.as_analytic().is_some() => {
            // Deferred to later phases -- analytic-NURBS is complex
            let _ = nurbs;
            Ok(Vec::new())
        }
        (FaceSurface::Nurbs(nurbs), analytic_surf) if analytic_surf.as_analytic().is_some() => {
            let _ = nurbs;
            Ok(Vec::new())
        }

        // NURBS-NURBS
        (FaceSurface::Nurbs(na), FaceSurface::Nurbs(nb)) => nurbs_nurbs_intersection(na, nb),

        // Fallback: unsupported pair
        _ => Ok(Vec::new()),
    }
}

/// Plane-plane intersection: direction = cross product of normals.
#[allow(clippy::unnecessary_wraps)]
fn plane_plane_intersection(
    na: Vec3,
    da: f64,
    nb: Vec3,
    db: f64,
    bbox_a: &Aabb3,
    bbox_b: &Aabb3,
) -> Result<Vec<RawCurve>, AlgoError> {
    let dir = na.cross(nb);
    let dir_len = dir.length();

    if dir_len < 1e-12 {
        // Planes are parallel or coplanar -- no line intersection.
        // Coplanar case is handled separately by the builder.
        return Ok(Vec::new());
    }

    let dir = dir * (1.0 / dir_len);

    // Find a point on the intersection line.
    let point = find_plane_plane_point(na, da, nb, db, dir);

    // Trim parameter range to the combined face AABBs.
    let t_range = trim_t_range_to_aabb(point, dir, bbox_a, bbox_b);
    let p0 = point + dir * t_range.0;
    let p1 = point + dir * t_range.1;

    let bbox = Aabb3 {
        min: Point3::new(p0.x().min(p1.x()), p0.y().min(p1.y()), p0.z().min(p1.z())),
        max: Point3::new(p0.x().max(p1.x()), p0.y().max(p1.y()), p0.z().max(p1.z())),
    };

    Ok(vec![RawCurve {
        curve: EdgeCurve::Line,
        bbox,
        t_range,
        p_start: p0,
        p_end: p1,
    }])
}

/// Trim a line's parameter range to the combined extent of two AABBs.
///
/// Projects the eight corners of the union of `bbox_a` and `bbox_b` onto
/// the line `origin + t * dir` and returns the (min, max) parameter range.
/// `dir` must be unit-length.
fn trim_t_range_to_aabb(origin: Point3, dir: Vec3, bbox_a: &Aabb3, bbox_b: &Aabb3) -> (f64, f64) {
    let cmin = Point3::new(
        bbox_a.min.x().min(bbox_b.min.x()),
        bbox_a.min.y().min(bbox_b.min.y()),
        bbox_a.min.z().min(bbox_b.min.z()),
    );
    let cmax = Point3::new(
        bbox_a.max.x().max(bbox_b.max.x()),
        bbox_a.max.y().max(bbox_b.max.y()),
        bbox_a.max.z().max(bbox_b.max.z()),
    );

    let mut t_min = f64::MAX;
    let mut t_max = f64::MIN;
    for &x in &[cmin.x(), cmax.x()] {
        for &y in &[cmin.y(), cmax.y()] {
            for &z in &[cmin.z(), cmax.z()] {
                let corner = Point3::new(x, y, z);
                let t = (corner - origin).dot(dir);
                t_min = t_min.min(t);
                t_max = t_max.max(t);
            }
        }
    }

    (t_min, t_max)
}

/// Find a point on the plane-plane intersection line.
///
/// The point lies in the plane spanned by the two normals and satisfies
/// both plane equations.
fn find_plane_plane_point(na: Vec3, da: f64, nb: Vec3, db: f64, dir: Vec3) -> Point3 {
    // P = (da * (nb x dir) + db * (dir x na)) / dot(dir, na x nb)
    let na_cross_nb = na.cross(nb);
    let denom = dir.dot(na_cross_nb);

    if denom.abs() < 1e-15 {
        // Degenerate -- return origin as fallback
        return Point3::new(0.0, 0.0, 0.0);
    }

    let nb_cross_dir = nb.cross(dir);
    let dir_cross_na = dir.cross(na);

    Point3::new(
        (da * nb_cross_dir.x() + db * dir_cross_na.x()) / denom,
        (da * nb_cross_dir.y() + db * dir_cross_na.y()) / denom,
        (da * nb_cross_dir.z() + db * dir_cross_na.z()) / denom,
    )
}

/// Plane-analytic surface intersection using exact curves.
fn plane_analytic_intersection(
    normal: Vec3,
    d: f64,
    analytic: &analytic_intersection::AnalyticSurface<'_>,
) -> Result<Vec<RawCurve>, AlgoError> {
    let exact_curves = analytic_intersection::exact_plane_analytic(*analytic, normal, d)?;

    let mut results = Vec::new();
    for exact in exact_curves {
        match exact {
            analytic_intersection::ExactIntersectionCurve::Circle(circle) => {
                let bbox = circle_bbox(&circle);
                let domain = (0.0, std::f64::consts::TAU);
                let p_start = ParametricCurve::evaluate(&circle, domain.0);
                let p_end = ParametricCurve::evaluate(&circle, domain.1);
                results.push(RawCurve {
                    curve: EdgeCurve::Circle(circle),
                    bbox,
                    t_range: domain,
                    p_start,
                    p_end,
                });
            }
            analytic_intersection::ExactIntersectionCurve::Ellipse(ellipse) => {
                let bbox = ellipse_bbox(&ellipse);
                let domain = (0.0, std::f64::consts::TAU);
                let p_start = ParametricCurve::evaluate(&ellipse, domain.0);
                let p_end = ParametricCurve::evaluate(&ellipse, domain.1);
                results.push(RawCurve {
                    curve: EdgeCurve::Ellipse(ellipse),
                    bbox,
                    t_range: domain,
                    p_start,
                    p_end,
                });
            }
            analytic_intersection::ExactIntersectionCurve::Points(pts) => {
                if pts.len() < 2 {
                    continue;
                }
                // Fit a degree-3 NURBS curve through the sampled points
                let nurbs = brepkit_math::nurbs::fitting::interpolate(&pts, 3)
                    .map_err(|e| AlgoError::IntersectionFailed(format!("NURBS fit failed: {e}")))?;
                let t_range = nurbs.domain();
                let bbox = Aabb3::try_from_points(pts.iter().copied()).ok_or_else(|| {
                    AlgoError::IntersectionFailed("empty points for NURBS fit".into())
                })?;
                let end_pt = pts[pts.len() - 1];
                results.push(RawCurve {
                    curve: EdgeCurve::NurbsCurve(nurbs),
                    bbox,
                    t_range,
                    p_start: pts[0],
                    p_end: end_pt,
                });
            }
        }
    }

    Ok(results)
}

/// Analytic-analytic surface intersection using marching.
fn analytic_analytic_intersection(
    a: &analytic_intersection::AnalyticSurface<'_>,
    b: &analytic_intersection::AnalyticSurface<'_>,
    v_range_a: Option<(f64, f64)>,
    v_range_b: Option<(f64, f64)>,
) -> Result<Vec<RawCurve>, AlgoError> {
    let isect_curves = analytic_intersection::intersect_analytic_analytic_bounded(
        *a, *b, 32, v_range_a, v_range_b,
    )?;

    let mut results = Vec::new();
    for ic in isect_curves {
        let domain = ic.curve.domain();
        let bbox = nurbs_curve_bbox(&ic.curve);
        let p_start = ParametricCurve::evaluate(&ic.curve, domain.0);
        let p_end = ParametricCurve::evaluate(&ic.curve, domain.1);
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
            p_start,
            p_end,
        });
    }

    Ok(results)
}

/// Plane-NURBS intersection.
fn plane_nurbs_intersection(
    normal: Vec3,
    d: f64,
    nurbs: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<Vec<RawCurve>, AlgoError> {
    let isect_curves = nurbs_isect::intersect_plane_nurbs(nurbs, normal, d, NURBS_SAMPLES)?;

    let mut results = Vec::new();
    for ic in isect_curves {
        let domain = ic.curve.domain();
        let bbox = nurbs_curve_bbox(&ic.curve);
        let p_start = ParametricCurve::evaluate(&ic.curve, domain.0);
        let p_end = ParametricCurve::evaluate(&ic.curve, domain.1);
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
            p_start,
            p_end,
        });
    }

    Ok(results)
}

/// NURBS-NURBS intersection.
fn nurbs_nurbs_intersection(
    na: &brepkit_math::nurbs::surface::NurbsSurface,
    nb: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<Vec<RawCurve>, AlgoError> {
    let isect_curves = nurbs_isect::intersect_nurbs_nurbs(na, nb, NURBS_SAMPLES, NURBS_MARCH_STEP)?;

    let mut results = Vec::new();
    for ic in isect_curves {
        let domain = ic.curve.domain();
        let bbox = nurbs_curve_bbox(&ic.curve);
        let p_start = ParametricCurve::evaluate(&ic.curve, domain.0);
        let p_end = ParametricCurve::evaluate(&ic.curve, domain.1);
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
            p_start,
            p_end,
        });
    }

    Ok(results)
}

/// Compute AABB for a circle.
fn circle_bbox(circle: &brepkit_math::curves::Circle3D) -> Aabb3 {
    let n = 16;
    let points: Vec<Point3> = (0..=n)
        .map(|i| {
            let t = std::f64::consts::TAU * (i as f64 / n as f64);
            ParametricCurve::evaluate(circle, t)
        })
        .collect();
    Aabb3::from_points(points)
}

/// Compute AABB for an ellipse.
fn ellipse_bbox(ellipse: &brepkit_math::curves::Ellipse3D) -> Aabb3 {
    let n = 16;
    let points: Vec<Point3> = (0..=n)
        .map(|i| {
            let t = std::f64::consts::TAU * (i as f64 / n as f64);
            ParametricCurve::evaluate(ellipse, t)
        })
        .collect();
    Aabb3::from_points(points)
}

/// Find an existing vertex on a face's boundary within tolerance of a point.
///
/// Iterates the face's outer wire vertices and returns the first one
/// within `tol.linear` of `point`. This implements the "PutPavesOnCurve"
/// vertex snapping: intersection curve endpoints at face boundaries reuse
/// the face's existing boundary vertices instead of creating duplicates.
fn find_nearby_face_vertex(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
    tol: Tolerance,
) -> Option<brepkit_topology::vertex::VertexId> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        for &vid in &[edge.start(), edge.end()] {
            let vpt = topo.vertex(vid).ok()?.point();
            if (vpt - point).length() < tol.linear {
                return Some(vid);
            }
        }
    }
    None
}

/// Find where a closed `Circle3D` section crosses the outer-wire
/// boundary of the analytic-surface face in the pair (the non-plane
/// face, which holds the boundary the curve actually exits at).
///
/// Returns crossings as `(t_on_circle, point_3d)` sorted by `t`, with
/// duplicates removed. Only `Line`-curve boundary edges are considered.
///
/// Rationale: for plane-sphere intersections, the closed curve lies on
/// both the plane face and the sphere hemisphere, but the *meaningful*
/// boundary crossings are on the sphere hemisphere's equator polygon
/// — those are the points where the section enters/leaves the
/// hemisphere region. Using those crossings as split points gives
/// half-arcs that cleanly bound a spherical cap sub-face. The plane
/// face's boundary crossings can introduce points interior to the
/// sphere hemisphere, producing arcs that don't form closed loops on
/// the hemisphere.
fn closed_circle_boundary_crossings(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    circle: &brepkit_math::curves::Circle3D,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    let mut hits: Vec<(f64, Point3)> = Vec::new();

    // Pick the non-plane face for boundary crossings. If both are planes
    // or both are non-plane, fall back to using both (existing behavior).
    let analytic_face = |fid: FaceId| -> Option<FaceId> {
        let face = topo.face(fid).ok()?;
        if matches!(face.surface(), FaceSurface::Plane { .. }) {
            None
        } else {
            Some(fid)
        }
    };
    let analytic_a = analytic_face(face_a);
    let analytic_b = analytic_face(face_b);
    let faces_to_check: Vec<FaceId> = match (analytic_a, analytic_b) {
        (None, Some(b)) => vec![b],
        (Some(a), None) => vec![a],
        _ => vec![face_a, face_b],
    };

    for &fid in &faces_to_check {
        let Ok(face) = topo.face(fid) else {
            continue;
        };
        let Ok(wire) = topo.wire(face.outer_wire()) else {
            continue;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            // Only line boundary edges are supported for now (covers all
            // current sphere-hemisphere + box-face boundaries).
            if !matches!(edge.curve(), EdgeCurve::Line) {
                continue;
            }
            let Ok(sv) = topo.vertex(edge.start()) else {
                continue;
            };
            let Ok(ev) = topo.vertex(edge.end()) else {
                continue;
            };
            for (p, t) in circle.intersect_segment(sv.point(), ev.point(), tol.linear) {
                // Dedup against existing hits by 3D distance.
                let dup = hits
                    .iter()
                    .any(|(_, q)| (*q - p).length() < tol.linear * 10.0);
                if !dup {
                    hits.push((t, p));
                }
            }
        }
    }

    hits.sort_by(|a, b| a.0.total_cmp(&b.0));

    log::trace!(
        "closed_circle_boundary_crossings: face_a={face_a:?} face_b={face_b:?} hits={}",
        hits.len()
    );

    // If we found more than 4 crossings, the circle is almost certainly
    // coincident with one of the face boundaries (e.g. the equator polygon
    // approximating a great circle on a sphere). In that case the curve
    // is the boundary itself, not an interior section — fall back to the
    // existing closed-curve handling instead of splitting.
    if hits.len() > 4 {
        log::debug!(
            "closed_circle_boundary_crossings: {} hits — assuming coincident with boundary, \
             skipping split",
            hits.len()
        );
        return Vec::new();
    }

    hits
}

/// Emit N arc edges + `IntersectionCurveDS` entries for a closed-circle
/// section split at `crossings` (≥ 2 entries, sorted by circle parameter
/// `t`).
///
/// Each arc becomes its own `IntersectionCurveDS` so that the downstream
/// `build_section_edges` path sees N separate (open) section sources
/// instead of one closed curve. `build_section_edges` reconstructs each
/// section's start/end from the curve's `t_range`, so the per-arc t-range
/// here is what carries the split through.
///
/// Arcs whose midpoints fall outside the AABB of either face are dropped
/// — those portions of the original closed curve are not geometrically
/// "on" the face pair, so emitting them as sections would confuse the
/// face splitter.
#[allow(clippy::too_many_arguments)]
fn emit_split_circle_arcs(
    topo: &mut Topology,
    arena: &mut GfaArena,
    face_a: FaceId,
    face_b: FaceId,
    raw: &RawCurve,
    circle: &brepkit_math::curves::Circle3D,
    crossings: &[(f64, Point3)],
    tol: Tolerance,
) {
    // Compute AABBs for each face. For analytic faces the outer wire
    // alone doesn't enclose the face region (e.g. a sphere hemisphere has
    // its outer wire on the equator plane while the surface extends to
    // the pole) — so we union the wire AABB with the surface's AABB to
    // get a usable bounding region.
    let face_aabb = |fid: FaceId| -> Option<Aabb3> {
        let face = topo.face(fid).ok()?;
        let wire = topo.wire(face.outer_wire()).ok()?;
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut any = false;
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                for vid in [edge.start(), edge.end()] {
                    if let Ok(v) = topo.vertex(vid) {
                        let p = v.point();
                        min =
                            Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
                        max =
                            Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
                        any = true;
                    }
                }
            }
        }
        if !any {
            return None;
        }
        // For analytic surfaces with finite extent, union the wire AABB
        // with the surface AABB. This expands a "degenerate-in-Z hemisphere
        // wire" (z=0 only) up to the actual hemisphere extent.
        if let FaceSurface::Sphere(sphere) = face.surface() {
            let r = sphere.radius();
            let c = sphere.center();
            min = Point3::new(
                min.x().min(c.x() - r),
                min.y().min(c.y() - r),
                min.z().min(c.z() - r),
            );
            max = Point3::new(
                max.x().max(c.x() + r),
                max.y().max(c.y() + r),
                max.z().max(c.z() + r),
            );
        }
        Some(Aabb3 { min, max }.expanded(tol.linear * 10.0))
    };
    let bbox_a = face_aabb(face_a);
    let bbox_b = face_aabb(face_b);
    let in_both = |p: Point3| -> bool {
        let a_ok = bbox_a.as_ref().is_none_or(|b| b.contains_point(p));
        let b_ok = bbox_b.as_ref().is_none_or(|b| b.contains_point(p));
        a_ok && b_ok
    };

    // Pass 1: determine which arcs survive the AABB filter, *before*
    // allocating any topology vertices. Otherwise dropping an arc could
    // leave its crossing vertices orphaned in the topology (no edges
    // reference them) — visible to any downstream pass that iterates all
    // vertices. We also split each surviving arc into sub-arcs of span
    // ≤ π so the resulting `EdgeCurve::Circle` is unambiguous: several
    // downstream consumers (tessellation's `shorter_arc_range` in
    // `tessellate/mod.rs`, wire sampling in `topology/builder.rs`)
    // interpret an open circle edge as the *shorter* arc between its
    // endpoints, so a span > π would be flipped to the complementary
    // arc and break face splitting/classification.
    //
    // Each survivor records the inserted crossing points along with
    // their (t, 3D) values; pass 2 then materialises just those
    // vertices (in order) and builds the edges.
    //
    // Each point is `(t, 3D point, original-crossing-index-or-None)`.
    // Endpoints carry `Some(crossing_index)` so we can dedup with
    // adjacent arcs sharing the same crossing vertex; midpoints
    // (inserted to keep span ≤ π) carry `None` and always get a fresh
    // vertex allocation.
    let n = crossings.len();
    let mut survivors: Vec<Vec<(f64, Point3, Option<usize>)>> = Vec::new();
    for i in 0..n {
        let (t0, _p0) = crossings[i];
        // Wrap to next crossing; for the last arc that means going back to
        // the first crossing through the seam (t0 → t1 + 2π).
        let next_i = (i + 1) % n;
        let (mut t1, _p1) = crossings[next_i];
        if t1 <= t0 {
            t1 += std::f64::consts::TAU;
        }

        let t_mid = (t0 + t1) * 0.5;
        let mid_3d = circle.evaluate(t_mid);
        if !in_both(mid_3d) {
            log::debug!(
                "FF: drop closed-circle arc {i}/{n} (midpoint {mid_3d:?} outside face pair)"
            );
            continue;
        }

        // Split spans > π into 2+ sub-arcs by inserting midpoint vertices
        // at evenly spaced t-values. Each sub-arc then has span ≤ π so
        // downstream "shorter arc" interpretation matches the intended arc.
        let arc_span = t1 - t0;
        let n_sub = (arc_span / std::f64::consts::PI).ceil().max(1.0) as usize;
        let step = arc_span / n_sub as f64;
        let mut points: Vec<(f64, Point3, Option<usize>)> = Vec::with_capacity(n_sub + 1);
        points.push((t0, crossings[i].1, Some(i)));
        for k in 1..n_sub {
            let tk = t0 + step * k as f64;
            points.push((tk, circle.evaluate(tk), None));
        }
        points.push((t1, crossings[next_i].1, Some(next_i)));

        survivors.push(points);
    }

    // Pass 2: allocate vertices only for crossings/midpoints used by
    // surviving arcs, then materialise edges + pave blocks.
    let mut crossing_vids: Vec<Option<brepkit_topology::vertex::VertexId>> = vec![None; n];
    let resolve_crossing =
        |topo: &mut Topology, arena: &GfaArena, p: Point3| -> brepkit_topology::vertex::VertexId {
            super::helpers::find_nearby_pave_vertex(topo, arena, p, tol)
                .or_else(|| find_nearby_face_vertex(topo, face_a, p, tol))
                .or_else(|| find_nearby_face_vertex(topo, face_b, p, tol))
                .unwrap_or_else(|| topo.add_vertex(Vertex::new(p, tol.linear)))
        };

    let mut emitted = 0_usize;
    let num_survivors = survivors.len();
    for (a_idx, arc_points) in survivors.into_iter().enumerate() {
        // Walk consecutive (t, point) pairs to emit one edge per sub-arc.
        for w in arc_points.windows(2) {
            let (t_s, p_s, idx_s) = w[0];
            let (t_e, p_e, idx_e) = w[1];

            let start_vid = match idx_s {
                Some(ci) => {
                    *crossing_vids[ci].get_or_insert_with(|| resolve_crossing(topo, arena, p_s))
                }
                None => topo.add_vertex(Vertex::new(p_s, tol.linear)),
            };
            let end_vid = match idx_e {
                Some(ci) => {
                    *crossing_vids[ci].get_or_insert_with(|| resolve_crossing(topo, arena, p_e))
                }
                None => topo.add_vertex(Vertex::new(p_e, tol.linear)),
            };

            let edge = Edge::new(start_vid, end_vid, EdgeCurve::Circle(circle.clone()));
            let edge_id = topo.add_edge(edge);

            let start_pave = Pave::new(start_vid, t_s);
            let end_pave = Pave::new(end_vid, t_e);
            let pb = PaveBlock::new(edge_id, start_pave, end_pave);
            let pb_id = arena.pave_blocks.alloc(pb);

            let curve_index = arena.curves.len();
            arena.curves.push(IntersectionCurveDS {
                curve: EdgeCurve::Circle(circle.clone()),
                face_a,
                face_b,
                // The full-circle bbox is a safe over-approximation for any arc.
                bbox: raw.bbox,
                pave_blocks: vec![pb_id],
                t_range: (t_s, t_e),
            });

            arena.interference.ff.push(Interference::FF {
                f1: face_a,
                f2: face_b,
                curve_index,
            });
            emitted += 1;

            log::debug!(
                "FF: split closed circle arc {a_idx}/{num_survivors}: \
                 faces {face_a:?}/{face_b:?} t=[{t_s:.4},{t_e:.4}] \
                 edge={edge_id:?} pb={pb_id:?}"
            );
        }
    }

    log::debug!("FF: emitted {emitted}/{n} arcs after AABB filter for {face_a:?}/{face_b:?}");
}

/// Compute AABB for a NURBS curve.
fn nurbs_curve_bbox(curve: &brepkit_math::nurbs::curve::NurbsCurve) -> Aabb3 {
    let (t0, t1) = curve.domain();
    let n: usize = 32;
    let points: Vec<Point3> = (0..=n)
        .map(|i| {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            ParametricCurve::evaluate(curve, t)
        })
        .collect();
    Aabb3::from_points(points)
}

// ── FF curve boundary filtering ──────────────────────────────────────

/// Clip a Line curve to a planar face's boundary polygon.
fn clip_line_to_face(topo: &Topology, face_id: FaceId, raw: &RawCurve) -> Option<(f64, f64)> {
    let face = topo.face(face_id).ok()?;
    let FaceSurface::Plane { normal, .. } = face.surface() else {
        return Some((0.0, 1.0));
    };
    let wire = topo.wire(face.outer_wire()).ok()?;
    if !wire.edges().iter().all(|oe| {
        topo.edge(oe.edge())
            .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line))
    }) {
        return Some((0.0, 1.0));
    }

    let mut verts = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        verts.push(topo.vertex(vid).ok()?.point());
    }
    if verts.len() < 3 {
        return Some((0.0, 1.0));
    }

    let frame =
        super::super::builder::plane_frame::PlaneFrame::from_normal_and_point(*normal, verts[0]);
    let poly: Vec<(f64, f64)> = verts
        .iter()
        .map(|v| {
            let uv = frame.project(*v);
            (uv.x(), uv.y())
        })
        .collect();
    let s = frame.project(raw.p_start);
    let e = frame.project(raw.p_end);
    clip_line_to_polygon((s.x(), s.y()), (e.x(), e.y()), &poly)
}

/// Cyrus-Beck line-polygon clipping. Handles CCW and CW winding.
fn clip_line_to_polygon(
    start: (f64, f64),
    end: (f64, f64),
    polygon: &[(f64, f64)],
) -> Option<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return None;
    }
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let area2: f64 = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            polygon[i].0 * polygon[j].1 - polygon[j].0 * polygon[i].1
        })
        .sum();
    let sign = if area2 >= 0.0 { 1.0 } else { -1.0 };
    let mut t_min = 0.0_f64;
    let mut t_max = 1.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let ex = polygon[j].0 - polygon[i].0;
        let ey = polygon[j].1 - polygon[i].1;
        let nx = -ey * sign;
        let ny = ex * sign;
        let denom = nx * dx + ny * dy;
        let num = nx * (start.0 - polygon[i].0) + ny * (start.1 - polygon[i].1);
        if denom.abs() < 1e-15 {
            if num < -1e-10 {
                return None;
            }
            continue;
        }
        let t = -num / denom;
        if denom > 0.0 {
            t_min = t_min.max(t);
        } else {
            t_max = t_max.min(t);
        }
        if t_min > t_max + 1e-6 {
            return None;
        }
    }
    if t_max - t_min < 1e-6 {
        return None;
    }
    Some((t_min.max(0.0), t_max.min(1.0)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn clip_inside_square() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let r = clip_line_to_polygon((0.2, 0.5), (0.8, 0.5), &poly).unwrap();
        assert!((r.0).abs() < 1e-6 && (r.1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn clip_crossing() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let r = clip_line_to_polygon((-1.0, 0.5), (2.0, 0.5), &poly).unwrap();
        assert!((r.0 - 1.0 / 3.0).abs() < 1e-6);
        assert!((r.1 - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn clip_outside() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(clip_line_to_polygon((2.0, 0.5), (3.0, 0.5), &poly).is_none());
    }

    #[test]
    fn clip_cw_polygon() {
        let poly = vec![(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)];
        let r = clip_line_to_polygon((-1.0, 0.5), (2.0, 0.5), &poly).unwrap();
        assert!((r.0 - 1.0 / 3.0).abs() < 1e-6);
        assert!((r.1 - 2.0 / 3.0).abs() < 1e-6);
    }
}
