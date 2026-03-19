//! Phase FF: Face-face intersection detection.
//!
//! For each (face_a, face_b) pair across solids, computes intersection
//! curves. Results are stored as `IntersectionCurveDS` entries in the
//! GFA arena, with FF interferences referencing them by index.

use brepkit_math::aabb::Aabb3;
use brepkit_math::analytic_intersection;
use brepkit_math::nurbs::intersection as nurbs_isect;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::ds::{GfaArena, Interference, IntersectionCurveDS};
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
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup or intersection computation fails.
#[allow(clippy::too_many_lines)]
pub fn perform(
    topo: &Topology,
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

    for (idx_a, &fa) in faces_a.iter().enumerate() {
        let bbox_a = &bboxes_a[idx_a];
        let face_a = topo.face(fa)?;
        let surf_a = face_a.surface();

        for (idx_b, &fb) in faces_b.iter().enumerate() {
            let bbox_b = &bboxes_b[idx_b];

            // AABB rejection
            if !bbox_a
                .expanded(tol.linear)
                .intersects(bbox_b.expanded(tol.linear))
            {
                continue;
            }

            let face_b = topo.face(fb)?;
            let surf_b = face_b.surface();

            // Compute raw intersection curves
            let raw_curves = compute_raw_curves(surf_a, surf_b)?;

            for raw in raw_curves {
                let curve_index = arena.curves.len();
                arena.curves.push(IntersectionCurveDS {
                    curve: raw.curve,
                    face_a: fa,
                    face_b: fb,
                    bbox: raw.bbox,
                    pave_blocks: Vec::new(),
                    t_range: raw.t_range,
                });

                arena.interference.ff.push(Interference::FF {
                    f1: fa,
                    f2: fb,
                    curve_index,
                });

                log::debug!("FF: faces {fa:?} and {fb:?} intersect (curve_index={curve_index})",);
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
        // Degenerate face with no edges — use a zero-volume box at origin
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

/// Intermediate intersection result before face IDs are assigned.
struct RawCurve {
    /// The 3D curve geometry.
    curve: EdgeCurve,
    /// Bounding box of the curve.
    bbox: Aabb3,
    /// Parameter range on the curve.
    t_range: (f64, f64),
}

/// Compute raw intersection curves between two surfaces.
///
/// Dispatches by surface type pair. Raw curves are returned without
/// trimming to face boundaries.
#[allow(clippy::too_many_lines)]
fn compute_raw_curves(
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
) -> Result<Vec<RawCurve>, AlgoError> {
    match (surf_a, surf_b) {
        // Plane-Plane
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            plane_plane_intersection(*na, *da, *nb, *db)
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
                analytic_analytic_intersection(&aa, &ab)
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
            // Deferred to later phases — analytic-NURBS is complex
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
) -> Result<Vec<RawCurve>, AlgoError> {
    let dir = na.cross(nb);
    let dir_len = dir.length();

    if dir_len < 1e-12 {
        // Planes are parallel or coplanar — no line intersection.
        // Coplanar case is handled separately by the builder.
        return Ok(Vec::new());
    }

    let dir = dir * (1.0 / dir_len);

    // Find a point on the intersection line.
    let point = find_plane_plane_point(na, da, nb, db, dir);

    // Represent as a Line with a bounded parameter range.
    // TODO: trim t_range to actual face boundaries using face AABBs
    // passed in from the caller, rather than this fixed range.
    let t_range = (-100.0, 100.0);
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
    }])
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
        // Degenerate — return origin as fallback
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
                results.push(RawCurve {
                    curve: EdgeCurve::Circle(circle),
                    bbox,
                    t_range: domain,
                });
            }
            analytic_intersection::ExactIntersectionCurve::Ellipse(ellipse) => {
                let bbox = ellipse_bbox(&ellipse);
                let domain = (0.0, std::f64::consts::TAU);
                results.push(RawCurve {
                    curve: EdgeCurve::Ellipse(ellipse),
                    bbox,
                    t_range: domain,
                });
            }
            analytic_intersection::ExactIntersectionCurve::Points(_) => {
                // Points can't be represented as edge curves — skip
            }
        }
    }

    Ok(results)
}

/// Analytic-analytic surface intersection using marching.
fn analytic_analytic_intersection(
    a: &analytic_intersection::AnalyticSurface<'_>,
    b: &analytic_intersection::AnalyticSurface<'_>,
) -> Result<Vec<RawCurve>, AlgoError> {
    let isect_curves =
        analytic_intersection::intersect_analytic_analytic_bounded(*a, *b, 32, None, None)?;

    let mut results = Vec::new();
    for ic in isect_curves {
        let domain = ic.curve.domain();
        let bbox = nurbs_curve_bbox(&ic.curve);
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
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
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
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
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
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
            brepkit_math::traits::ParametricCurve::evaluate(circle, t)
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
            brepkit_math::traits::ParametricCurve::evaluate(ellipse, t)
        })
        .collect();
    Aabb3::from_points(points)
}

/// Compute AABB for a NURBS curve.
fn nurbs_curve_bbox(curve: &brepkit_math::nurbs::curve::NurbsCurve) -> Aabb3 {
    let (t0, t1) = curve.domain();
    let n: usize = 32;
    let points: Vec<Point3> = (0..=n)
        .map(|i| {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            brepkit_math::traits::ParametricCurve::evaluate(curve, t)
        })
        .collect();
    Aabb3::from_points(points)
}
