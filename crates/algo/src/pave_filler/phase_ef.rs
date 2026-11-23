//! Phase EF: Edge-face intersection detection.
//!
//! For each (edge, face) pair across solids, finds points where the
//! edge crosses or touches the face surface. Records EF interferences
//! and adds extra paves to the edge for later splitting.

use brepkit_math::madd::Madd;
use std::collections::HashSet;

use crate::builder::classify_2d::{distance_to_polygon_boundary, point_in_polygon_2d};
use crate::builder::plane_frame::PlaneFrame;
use crate::ds::{GfaArena, Interference, Pave};
use crate::error::AlgoError;
use brepkit_math::aabb::Aabb3;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::planar::PlanarRegion;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use super::helpers::{add_pave_to_edge, find_nearby_pave_vertex as find_nearby_vertex};

/// Number of samples along each edge for sign-change detection.
const N_SAMPLES: usize = 64;

/// Number of samples per boundary edge for face containment polygons.
const N_BOUNDARY_SAMPLES: usize = 32;

/// Detect edge-face intersections between the two solids.
///
/// Checks edges of A against faces of B, and edges of B against
/// faces of A. When an edge crosses a face surface (within tolerance),
/// an EF interference is recorded and an extra pave is added to the
/// edge's pave block.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let bbox_a = crate::classifier::compute_solid_bbox(topo, solid_a)?;
    let bbox_b = crate::classifier::compute_solid_bbox(topo, solid_b)?;
    if !bbox_a
        .expanded(tol.linear)
        .intersects(bbox_b.expanded(tol.linear))
    {
        log::debug!("EF: solids are disjoint, skipping");
        return Ok(());
    }

    let edges_a = brepkit_topology::explorer::solid_edges(topo, solid_a)?;
    let edges_b = brepkit_topology::explorer::solid_edges(topo, solid_b)?;
    let faces_a = brepkit_topology::explorer::solid_faces(topo, solid_a)?;
    let faces_b = brepkit_topology::explorer::solid_faces(topo, solid_b)?;

    // Collect face boundary edge sets to skip edges that are already
    // on the face boundary.
    let face_boundary_edges_b = collect_face_boundary_edges(topo, &faces_b)?;
    let face_boundary_edges_a = collect_face_boundary_edges(topo, &faces_a)?;

    check_edge_face_pairs(
        topo,
        &edges_a,
        (&faces_a, &face_boundary_edges_a),
        &faces_b,
        &face_boundary_edges_b,
        tol,
        arena,
    )?;
    check_edge_face_pairs(
        topo,
        &edges_b,
        (&faces_b, &face_boundary_edges_b),
        &faces_a,
        &face_boundary_edges_a,
        tol,
        arena,
    )?;

    Ok(())
}

/// Collect the set of boundary edge IDs for each face.
fn collect_face_boundary_edges(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Vec<HashSet<EdgeId>>, AlgoError> {
    let mut result = Vec::with_capacity(faces.len());
    for &fid in faces {
        let edges = brepkit_topology::explorer::face_edges(topo, fid)?;
        result.push(edges.into_iter().collect());
    }
    Ok(result)
}

/// Spatial containment test for a face, built from sampled boundary edges.
///
/// Surface crossings are found against infinite surfaces; this rejects
/// crossing points that lie outside the trimmed face region.
struct FaceContainment {
    bbox: Aabb3,
    planar: Option<PlanarContainment>,
}

struct PlanarContainment {
    frame: PlaneFrame,
    polygon: Vec<Point2>,
    /// The face's inner wires, sampled as the outer one is.
    holes: Vec<Vec<Point2>>,
    margin: f64,
    /// The outer wire read on its own lines, arcs and NURBS pieces, in
    /// `frame`. The polygon cuts inside each arc by up to a chord's
    /// sagitta, and the margin that covers that reaches up to twice the
    /// sagitta past the arc: 2.4 on a whole circle of radius 249.25.
    outer: Option<PlanarRegion>,
    /// A point this near the outer wire lies on it.
    near: f64,
}

impl PlanarContainment {
    /// Whether `p`, in the face's plane, lies in the face, and whether it
    /// lies within `near` of the face's boundary, which the sampled loops
    /// stand off by up to the margin.
    fn reads(&self, p: Point3, near: f64) -> (bool, bool) {
        let q = self.frame.project(p);
        let inside = point_in_polygon_2d(q, &self.polygon)
            && !self.holes.iter().any(|hole| point_in_polygon_2d(q, hole));
        let close = std::iter::once(&self.polygon)
            .chain(&self.holes)
            .any(|ring| distance_to_polygon_boundary(q, ring) <= near + self.margin);
        (inside, close)
    }
}

impl FaceContainment {
    fn accepts(&self, pt: Point3) -> bool {
        if !self.bbox.contains_point(pt) {
            return false;
        }
        let Some(planar) = &self.planar else {
            return true;
        };
        let p2 = planar.frame.project(pt);
        if let Some(inside) = planar
            .outer
            .as_ref()
            .and_then(|outer| outer.side(p2, planar.near))
        {
            return inside;
        }
        point_in_polygon_2d(p2, &planar.polygon)
            || distance_to_polygon_boundary(p2, &planar.polygon) <= planar.margin
    }
}

/// Sample a face's boundary into an AABB plus, for planar faces, an
/// in-plane outer-wire polygon for exact containment testing.
fn build_face_containment(
    topo: &Topology,
    fid: FaceId,
    tol: Tolerance,
) -> Result<FaceContainment, AlgoError> {
    let face = topo.face(fid)?;
    let surface = face.surface().clone();
    let outer_wire_id = face.outer_wire();

    let mut all_points = Vec::new();
    let mut outer_points = Vec::new();
    let mut max_sag = 0.0_f64;

    let outer_wire = topo.wire(outer_wire_id)?;
    let oriented: Vec<_> = outer_wire.edges().to_vec();
    let mut prev: Option<Point3> = None;
    for oe in &oriented {
        let edge = topo.edge(oe.edge())?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
        // Only curved edges contribute to the sagitta margin: a straight Line
        // edge's sampled chords coincide with the edge exactly (zero sagitta).
        // The margin is each chord's own sagitta, measured at its
        // mid-parameter: half a chord (the bound for samples spanning half a
        // turn) let a quarter circle of radius 3.75 admit crossings 0.18
        // outside its face, and a lattice strut face's 0.84-long corner arc
        // admitted a wall line's crossing 0.0015 past the strut's edge.
        let is_curved = !matches!(edge.curve(), EdgeCurve::Line);
        let n = N_BOUNDARY_SAMPLES;
        // Sample inclusive of the edge's end vertex (0..=n) so the closing
        // segment of a closed wire reaches the true endpoint; consecutive
        // edges share a vertex, so dedup against the previous point.
        let mut prev_t: Option<f64> = None;
        for i in 0..=n {
            let frac = i as f64 / n as f64;
            let frac = if oe.is_forward() { frac } else { 1.0 - frac };
            let t = t0 + (t1 - t0) * frac;
            let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
            if let Some(p) = prev {
                if (pt - p).length() <= tol.linear {
                    // The shared vertex's sample still starts this edge's
                    // first chord.
                    prev_t = Some(t);
                    continue;
                }
                if is_curved && let Some(tp) = prev_t {
                    let mid =
                        edge.curve()
                            .evaluate_with_endpoints(0.5 * (t + tp), start_pos, end_pos);
                    max_sag = max_sag.max(point_to_segment(mid, p, pt));
                }
            }
            prev = Some(pt);
            prev_t = Some(t);
            outer_points.push(pt);
        }
    }
    // The last edge's end vertex coincides with the first edge's start
    // vertex on a closed wire; drop the duplicate so the closing polygon
    // segment isn't degenerate.
    if outer_points.len() >= 2
        && let (Some(&first), Some(&last)) = (outer_points.first(), outer_points.last())
        && (last - first).length() <= tol.linear
    {
        outer_points.pop();
    }
    all_points.extend_from_slice(&outer_points);

    let mut hole_points: Vec<Vec<Point3>> = Vec::new();
    for &inner_wid in face.inner_wires() {
        let inner_wire = topo.wire(inner_wid)?;
        let inner_edges: Vec<_> = inner_wire.edges().to_vec();
        let mut ring = Vec::new();
        for oe in &inner_edges {
            let edge = topo.edge(oe.edge())?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
            let is_curved = !matches!(edge.curve(), EdgeCurve::Line);
            let n = N_BOUNDARY_SAMPLES;
            let at = |i: usize| {
                let frac = i as f64 / n as f64;
                let frac = if oe.is_forward() { frac } else { 1.0 - frac };
                t0 + (t1 - t0) * frac
            };
            // Up to the edge's end, which the next edge's first sample is.
            for i in 0..n {
                let (t, next) = (at(i), at(i + 1));
                let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
                if is_curved {
                    let end = edge
                        .curve()
                        .evaluate_with_endpoints(next, start_pos, end_pos);
                    let mid =
                        edge.curve()
                            .evaluate_with_endpoints(0.5 * (t + next), start_pos, end_pos);
                    max_sag = max_sag.max(point_to_segment(mid, pt, end));
                }
                ring.push(pt);
            }
        }
        all_points.extend_from_slice(&ring);
        hole_points.push(ring);
    }

    let Some(bbox) = Aabb3::try_from_points(all_points) else {
        return Ok(FaceContainment {
            bbox: Aabb3 {
                min: Point3::new(0.0, 0.0, 0.0),
                max: Point3::new(0.0, 0.0, 0.0),
            },
            planar: None,
        });
    };
    let diag = (bbox.max - bbox.min).length();

    if let FaceSurface::Plane { normal, .. } = &surface {
        if outer_points.len() >= 3 {
            // Sampled chords undercut curved boundary arcs by their sagitta;
            // twice the largest keeps true near-boundary crossings accepted.
            let margin = (2.0 * max_sag).max(tol.linear * 10.0);
            let frame = PlaneFrame::from_normal_and_point(*normal, outer_points[0]);
            let outer = brepkit_topology::planar::wire_region_2d(
                topo,
                outer_wire_id,
                frame.origin(),
                frame.u_axis(),
                frame.v_axis(),
                tol.linear,
            )?;
            let polygon: Vec<Point2> = outer_points.iter().map(|&p| frame.project(p)).collect();
            let holes = hole_points
                .iter()
                .map(|ring| ring.iter().map(|&p| frame.project(p)).collect())
                .collect();
            return Ok(FaceContainment {
                bbox: bbox.expanded(margin),
                planar: Some(PlanarContainment {
                    frame,
                    polygon,
                    holes,
                    margin,
                    outer,
                    near: 10.0 * tol.linear,
                }),
            });
        }
        return Ok(FaceContainment {
            bbox: bbox.expanded((diag * 0.5).max(tol.linear * 10.0)),
            planar: None,
        });
    }

    // Curved faces can bulge past their boundary AABB (e.g. a hemisphere
    // bounded by its equator), so expand generously by half the diagonal. A
    // NURBS face can rise further, which its control points bound.
    let bbox = bbox.expanded((diag * 0.5).max(tol.linear * 10.0));
    Ok(FaceContainment {
        bbox: match &surface {
            FaceSurface::Nurbs(n) => super::helpers::nurbs_face_box(topo, fid, n, bbox, tol)?,
            _ => bbox,
        },
        planar: None,
    })
}

/// Distance from `p` to the segment `a`-`b`.
fn point_to_segment(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq <= 0.0 {
        return (p - a).length();
    }
    let f = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    (p - (a + ab * f)).length()
}

/// A face's own trim, to read where a point of its surface lies on it.
#[derive(Clone, Copy)]
enum OnFace<'a> {
    Lateral(&'a crate::classifier::LateralTrim),
    /// A plane face bounded by lines and arcs, read on them exactly.
    Plane(&'a super::fill_face_info::PlaneRegion),
    /// A plane face with a NURBS edge in its boundary, read on sampled loops.
    Sampled(&'a PlanarContainment),
}

impl OnFace<'_> {
    /// Whether `p` lies on the face, within `near` of its boundary counting.
    fn holds(self, p: Point3, near: f64) -> bool {
        match self {
            Self::Lateral(trim) => trim.holds(p, near),
            Self::Plane(region) => region.holds(p, near),
            Self::Sampled(region) => {
                let (on, close) = region.reads(p, near);
                on || close
            }
        }
    }

    /// Whether `p` lies inside the face farther than `near` from its boundary.
    fn holds_clear(self, p: Point3, near: f64) -> bool {
        match self {
            Self::Lateral(trim) => trim.holds_clear(p, near),
            Self::Plane(region) => region.holds_clear(p, near),
            Self::Sampled(region) => {
                let (on, close) = region.reads(p, near);
                on && !close
            }
        }
    }
}

/// Whether `eid`, lying in the plane `normal · p = d` of the other solid's
/// face, is what splits that face: an edge of a NURBS face none of whose own
/// faces lies in the plane. The face-face phase drops a plane's section
/// along a NURBS face's own edge as being that edge, and where one of the
/// edge's faces shares the plane the coplanar pair splits the face instead.
fn stands_in_for_a_section(
    topo: &Topology,
    eid: EdgeId,
    (own_faces, own_boundaries): (&[FaceId], &[HashSet<EdgeId>]),
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> bool {
    let mut on_nurbs = false;
    for (&fid, boundary) in own_faces.iter().zip(own_boundaries) {
        if !boundary.contains(&eid) {
            continue;
        }
        match topo.face(fid).map(brepkit_topology::face::Face::surface) {
            Ok(FaceSurface::Nurbs(_)) => on_nurbs = true,
            Ok(FaceSurface::Plane { normal: n, d: own }) => {
                let along = n.dot(normal);
                if along.abs() > 1.0 - tol.angular.max(1e-9)
                    && (own * along.signum() - d).abs() < tol.linear
                {
                    return false;
                }
            }
            _ => {}
        }
    }
    on_nurbs
}

/// Check each edge against each face.
#[allow(clippy::too_many_lines)]
fn check_edge_face_pairs(
    topo: &mut Topology,
    edges: &[EdgeId],
    own: (&[FaceId], &[HashSet<EdgeId>]),
    faces: &[FaceId],
    face_boundary_edges: &[HashSet<EdgeId>],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    // Surface crossings are found against INFINITE surfaces; without a bounds
    // check an edge "crosses" a face far outside its trimmed region, creating
    // spurious paves that propagate bogus edge splits. The containment test
    // bounds-checks every crossing against the face's sampled boundary (bbox
    // for all faces, in-plane outer + inner-wire polygon for planar faces).
    let mut containments = Vec::with_capacity(faces.len());
    for &fid in faces {
        containments.push(build_face_containment(topo, fid, tol)?);
    }

    // Pre-expand each face's containment AABB by the linear tolerance so the
    // broad-phase reject below is conservative (never skips a real crossing,
    // which by definition lies inside the face's boundary region).
    let face_aabbs: Vec<Aabb3> = containments
        .iter()
        .map(|c| c.bbox.expanded(tol.linear))
        .collect();
    // A NURBS surface lies in its control points' box (the convex hull
    // property), so a point off that box is at least as far off the surface:
    // a far tighter bound than a curved face's grown boundary box, and one
    // that spares the surface projection wherever it already exceeds the
    // largest distance the crossing search compares against.
    let hulls: Vec<Option<Aabb3>> = faces
        .iter()
        .map(|&fid| {
            topo.face(fid).map(|f| match f.surface() {
                FaceSurface::Nurbs(n) if n.weights().iter().flatten().all(|&w| w > 0.0) => {
                    Some(n.aabb())
                }
                _ => None,
            })
        })
        .collect::<Result<_, _>>()?;

    let mut trims: Vec<Option<Option<crate::classifier::LateralTrim>>> =
        (0..faces.len()).map(|_| None).collect();
    let mut sphere_trims: Vec<Option<Option<crate::classifier::SphereFaceTrim>>> =
        (0..faces.len()).map(|_| None).collect();
    let mut regions: Vec<Option<Option<super::fill_face_info::PlaneRegion>>> =
        (0..faces.len()).map(|_| None).collect();
    let mut surveys = survey_pairs(
        topo,
        edges,
        faces,
        face_boundary_edges,
        &face_aabbs,
        &hulls,
        tol,
    );
    for (edge_idx, &eid) in edges.iter().enumerate() {
        // Snapshot edge data to avoid holding immutable borrow across add_vertex
        let (curve, start_pos, end_pos, t0, t1) = {
            let edge = topo.edge(eid)?;
            let sp = topo.vertex(edge.start())?.point();
            let ep = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            (edge.curve().clone(), sp, ep, t0, t1)
        };

        // Broad-phase AABB for the edge, reused across all faces. The vast
        // majority of (edge, face) pairs are spatially disjoint; testing each
        // edge sample against every face surface (an iterative projection for
        // curved faces) is the dominant cost in booleans on solids with many
        // curved faces. Sampling the edge into an AABB once and rejecting
        // disjoint faces collapses that quadratic to the pairs that actually
        // overlap. Sampled densely enough that the inter-sample sagitta is
        // negligible for the analytic edge curves used here.
        let edge_aabb = {
            let mut pts = Vec::with_capacity(N_SAMPLES + 1);
            for i in 0..=N_SAMPLES {
                let t = t0 + (t1 - t0) * (i as f64 / N_SAMPLES as f64);
                pts.push(curve.evaluate_with_endpoints(t, start_pos, end_pos));
            }
            Aabb3::try_from_points(pts).map(|a| a.expanded(tol.linear))
        };

        for (face_idx, &fid) in faces.iter().enumerate() {
            if face_boundary_edges[face_idx].contains(&eid) {
                continue;
            }

            // Broad-phase: skip faces whose region cannot reach this edge.
            if let Some(ea) = &edge_aabb
                && !ea.intersects(face_aabbs[face_idx])
            {
                continue;
            }
            // A line's sampled box is its exact box, so it may also be held
            // to the surface's own box; a curve's samples can undercut it.
            if let (Some(ea), Some(hull), EdgeCurve::Line) = (&edge_aabb, hulls[face_idx], &curve)
                && !ea.intersects(hull.expanded(tol.linear))
            {
                continue;
            }
            let hull = hulls[face_idx];

            let face = topo.face(fid)?;
            let surface = face.surface();

            // An edge lying entirely ON the face's surface is a coincidence
            // handled by the FF/same-domain machinery, not a set of
            // crossings; sampling it here would emit dozens of fake paves
            // (e.g. a cap circle lying in the opposing cap's plane).
            let n_chk = 16;
            let survey = surveys[edge_idx].remove(&face_idx).unwrap_or_else(|| {
                survey(&curve, start_pos, end_pos, t0, t1, surface, hull, None, tol)
            });
            if survey.is_none() {
                // Through the inside of a cylinder or cone face, the edge
                // bounds a region the two solids share there (a bin wall's
                // top arc on the coaxial air wall of a fillet that reaches
                // past it), and the face is split along it.
                let near = 10.0 * tol.linear;
                let lateral = matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
                    && matches!(curve, EdgeCurve::Line | EdgeCurve::Circle(_));
                // In a plane, the edge of a NURBS face is the section the
                // plane's face is split along, where it runs through the face.
                let planar = match surface {
                    FaceSurface::Plane { normal, d } => {
                        stands_in_for_a_section(topo, eid, own, *normal, *d, tol)
                    }
                    _ => false,
                };
                if lateral || planar {
                    let trim = if lateral {
                        trims[face_idx]
                            .get_or_insert_with(|| {
                                crate::classifier::LateralTrim::new(topo, fid)
                                    .ok()
                                    .flatten()
                            })
                            .as_ref()
                            .map(OnFace::Lateral)
                    } else {
                        regions[face_idx]
                            .get_or_insert_with(|| {
                                super::fill_face_info::PlaneRegion::of(topo, fid)
                            })
                            .as_ref()
                            .map(OnFace::Plane)
                            .or_else(|| containments[face_idx].planar.as_ref().map(OnFace::Sampled))
                    };
                    let at = |k: u32| {
                        let t = (t1 - t0).madd(f64::from(k) / f64::from(n_chk), t0);
                        curve.evaluate_with_endpoints(t, start_pos, end_pos)
                    };
                    // Every sample on the face: a circle can run through a
                    // cut-out between points that lie on it.
                    if trim.is_some_and(|trim| {
                        trim.holds_clear(at(n_chk / 2), near)
                            && trim.holds(start_pos, near)
                            && trim.holds(end_pos, near)
                            && (1..n_chk).all(|k| trim.holds(at(k), near))
                    }) {
                        log::debug!("EF: edge {eid:?} lies inside face {fid:?}");
                        if planar {
                            arena.plane_section_edges.insert(eid);
                        }
                        arena.interference.ef.push(Interference::EF {
                            edge: eid,
                            face: fid,
                            new_vertex: None,
                            parameter: None,
                            run: None,
                        });
                    } else if let Some(trim) = trim {
                        // An edge running over several faces of one surface
                        // (a corner arc across a wall split in two) lies in
                        // each along a run its paves already bound.
                        let paves: Vec<f64> = arena
                            .edge_pave_blocks
                            .get(&eid)
                            .into_iter()
                            .flatten()
                            .filter_map(|&pb| arena.pave_blocks.get(pb))
                            .flat_map(|pb| {
                                [pb.start.parameter, pb.end.parameter]
                                    .into_iter()
                                    .chain(pb.extra_paves.iter().map(|p| p.parameter))
                            })
                            .collect();
                        let t_at = |k: u32| (t1 - t0).madd(f64::from(k) / f64::from(n_chk), t0);
                        let eps = (t1 - t0).abs() * 1e-9;
                        let paved = |a: u32, b: u32| {
                            let (lo, hi) = (t_at(a).min(t_at(b)), t_at(a).max(t_at(b)));
                            paves.iter().any(|&p| (lo - eps..=hi + eps).contains(&p))
                        };
                        let clear: Vec<bool> =
                            (0..=n_chk).map(|k| trim.holds_clear(at(k), near)).collect();
                        let mut k = 0;
                        while k <= n_chk {
                            if !clear[k as usize] {
                                k += 1;
                                continue;
                            }
                            let first = k;
                            while k < n_chk && clear[k as usize + 1] {
                                k += 1;
                            }
                            let last = k;
                            let opens = first == 0 || paved(first - 1, first);
                            let closes = last == n_chk || paved(last, last + 1);
                            if opens && closes && (first, last) != (0, n_chk) {
                                let t_mid = 0.5 * (t_at(first) + t_at(last));
                                // The run reaches out to the paves bounding
                                // it, beyond its first and last samples.
                                let pave_near = |a: u32, b: u32, toward: f64| {
                                    let (lo, hi) = (t_at(a).min(t_at(b)), t_at(a).max(t_at(b)));
                                    paves
                                        .iter()
                                        .copied()
                                        .filter(|&p| (lo - eps..=hi + eps).contains(&p))
                                        .min_by(|p, q| {
                                            (p - toward).abs().total_cmp(&(q - toward).abs())
                                        })
                                        .unwrap_or(toward)
                                };
                                let from = if first == 0 {
                                    t_at(0)
                                } else {
                                    pave_near(first - 1, first, t_at(first))
                                };
                                let to = if last == n_chk {
                                    t_at(n_chk)
                                } else {
                                    pave_near(last, last + 1, t_at(last))
                                };
                                log::debug!(
                                    "EF: edge {eid:?} lies inside face {fid:?} around t={t_mid:.6}"
                                );
                                if planar {
                                    arena.plane_section_edges.insert(eid);
                                }
                                arena.interference.ef.push(Interference::EF {
                                    edge: eid,
                                    face: fid,
                                    new_vertex: None,
                                    parameter: Some(t_mid),
                                    run: Some((from.min(to), from.max(to))),
                                });
                            }
                            k += 1;
                        }
                    }
                }
                continue;
            }

            let mut crossings = survey.unwrap_or_default();
            // A curved edge crossing a cylinder or cone between two samples
            // (an envelope's corner curve through a pocket's floor fillet, a
            // half-disc's rim just poking through a frustum's wall) shows only
            // as the side of the surface flipping. Its unbounded surface
            // meets far more of the edge than the face does, and this face's
            // containment is a box, so a flip counts only where the face's
            // own wires hold it. A line's crossings are the roots of a
            // quadratic, exact wherever they fall between two samples.
            if matches!(
                curve,
                EdgeCurve::NurbsCurve(_)
                    | EdgeCurve::Line
                    | EdgeCurve::Circle(_)
                    | EdgeCurve::Ellipse(_)
            ) && matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
                && let Some(trim) = trims[face_idx]
                    .get_or_insert_with(|| {
                        crate::classifier::LateralTrim::new(topo, fid)
                            .ok()
                            .flatten()
                    })
                    .as_ref()
            {
                let found = if matches!(curve, EdgeCurve::Line) {
                    // A chord standing in for a rim crosses the surface off
                    // the faces it bounds.
                    let on_faces = topo.edge(eid).is_ok_and(|e| {
                        e.tolerance()
                            .is_none_or(|t| t <= crate::ds::shape_store::MAX_WIDEN)
                    });
                    if on_faces {
                        line_crossings(start_pos, end_pos, t0, t1, surface)
                    } else {
                        Vec::new()
                    }
                } else {
                    find_side_flips(&curve, start_pos, end_pos, t0, t1, surface, tol)
                };
                // The sample scan may already hold this root; two roots a
                // sample step apart are still two crossings.
                // A rim meeting the face only on one of the face's own wires
                // (a knuckle neck's rim in the plane of a lid knuckle's end)
                // is the coplanar-rim configuration the splitters handle from
                // the faces' boundaries; such a flip adds no crossing.
                let rim = matches!(curve, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_));
                for (t, p) in found {
                    let held = if rim {
                        trim.holds_clear(p, 10.0 * tol.linear)
                    } else {
                        trim.holds(p, 10.0 * tol.linear)
                    };
                    if held
                        && !crossings
                            .iter()
                            .any(|&(_, cp)| (p - cp).length() < tol.linear * 100.0)
                    {
                        crossings.push((t, p));
                    }
                }
                crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            // A curved edge crossing a ball between two samples (a ring's rim
            // through a ball across it) shows only as the side of the sphere
            // flipping, and counts where the face's own wires hold it.
            if matches!(
                curve,
                EdgeCurve::NurbsCurve(_) | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)
            ) && matches!(surface, FaceSurface::Sphere(_))
                && let Some(trim) = sphere_trims[face_idx]
                    .get_or_insert_with(|| {
                        crate::classifier::SphereFaceTrim::new(topo, fid)
                            .ok()
                            .flatten()
                    })
                    .as_ref()
            {
                let rim = !matches!(curve, EdgeCurve::NurbsCurve(_));
                for (t, p) in find_side_flips(&curve, start_pos, end_pos, t0, t1, surface, tol) {
                    let held = if rim {
                        trim.holds_clear(p, 10.0 * tol.linear)
                    } else {
                        trim.holds(p, 10.0 * tol.linear)
                    };
                    if held
                        && !crossings
                            .iter()
                            .any(|&(_, cp)| (p - cp).length() < tol.linear * 100.0)
                    {
                        crossings.push((t, p));
                    }
                }
                crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
            }

            // A line or conic that runs within the tolerance of a quadric
            // over several samples only touches it, at the one contact the
            // scan keeps. Exact geometry on an exact surface, it lies in the
            // face only where it stays within the tolerance: its pieces
            // reaching past that band (a ball's equator 3e-7 inside a rod's
            // wall) touch the face and do not split it.
            let bands: Vec<Option<(f64, f64)>> = crossings
                .iter()
                .map(|&(t, _)| {
                    if matches!(
                        curve,
                        EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)
                    ) && matches!(
                        surface,
                        FaceSurface::Cylinder(_)
                            | FaceSurface::Cone(_)
                            | FaceSurface::Sphere(_)
                            | FaceSurface::Torus(_)
                    ) {
                        tangent_band(&curve, start_pos, end_pos, (t0, t1), t, surface, tol)
                    } else {
                        None
                    }
                })
                .collect();

            // Endpoint-drop windows, one per crossing and per endpoint,
            // computed while the face's surface borrow is still live (the
            // loop below mutates `topo`). A TANGENTIAL (grazing) contact's
            // position along the curve is only accurate to sqrt-of-residual —
            // an arc grazing a coplanar wall at its endpoint solves to a
            // point microns along the arc from the true endpoint despite a
            // ~1e-12 residual. A fixed `tol.linear` window misses that point
            // and mints a near-duplicate vertex next to the edge's own
            // endpoint (the gridfinity lip-corner non-manifold STL family).
            //
            // The widened window (tol / |tangent·normal|, capped at 1e-3)
            // applies ONLY toward an endpoint that itself lies ON the
            // surface: then the contact IS that endpoint's vertex-face
            // incidence and the solver merely mislocated it. An endpoint off
            // the surface keeps the tight `tol.linear` window — a shallow
            // crossing near (but not at) an off-surface endpoint is genuine
            // topology and must keep its pave (dropping those regressed the
            // honeycomb wall-cut raw residual).
            let on_surface = |p: Point3| distance_to_surface(p, surface) <= tol.linear;
            let start_on_surface = on_surface(start_pos);
            let end_on_surface = on_surface(end_pos);
            let endpoint_windows: Vec<(f64, f64, f64)> = crossings
                .iter()
                .map(|&(t, pt)| {
                    let tangent = curve.tangent_with_endpoints(t, start_pos, end_pos);
                    let normal = match surface {
                        FaceSurface::Plane { normal, .. } => Some(*normal),
                        _ => surface.project_point(pt).map(|(u, v)| surface.normal(u, v)),
                    };
                    let sin_angle = match (tangent.normalize(), normal) {
                        (Ok(tangent_unit), Some(n)) => tangent_unit.dot(n).abs(),
                        _ => 1.0,
                    };
                    let widened = (tol.linear / sin_angle.max(1e-9)).min(1e-3);
                    if !start_on_surface && !end_on_surface {
                        return (tol.linear, tol.linear, widened);
                    }
                    // A fitted curve reaches its own end vertex only to its
                    // fit error, and a root solved on it lands that far off
                    // the surface again: a scoop rim fitted 1.4e-6 off its
                    // vertex crossed the plane through that vertex 7e-6 away.
                    let at_end = if matches!(curve, EdgeCurve::NurbsCurve(_)) {
                        widened.max(1e-5)
                    } else {
                        widened
                    };
                    (
                        if start_on_surface { at_end } else { tol.linear },
                        if end_on_surface { at_end } else { tol.linear },
                        widened,
                    )
                })
                .collect();

            // Mid-edge tangential junction snap. A grazing contact solved to
            // within the tolerance WELL (distance to the surface grows only
            // quadratically away from a tangency, so a whole ~sqrt(2r*tol)
            // band of the edge sits "on" the surface) lands microns from the
            // true junction, minting a near-duplicate vertex next to an
            // exact one that already exists in an operand (a socket outline
            // arc ending where the outline's straight run continues along
            // the bin wall). Snap the crossing to an existing pave vertex
            // within the angle-scaled window when that vertex genuinely lies
            // on BOTH the crossed surface and this edge's curve — the
            // incidence checks are what make the widened radius safe. The
            // pave parameter is recomputed for Line edges (exact foot);
            // other curve types keep the tight path.
            let snaps: Vec<Option<(brepkit_topology::vertex::VertexId, f64)>> = crossings
                .iter()
                .zip(&endpoint_windows)
                .map(|(&(t, pt), &(_, _, snap_window))| {
                    if snap_window <= tol.linear
                        || find_nearby_vertex(topo, arena, pt, tol).is_some()
                    {
                        return None;
                    }
                    let _ = t;
                    super::helpers::find_nearby_pave_vertex_widened(
                        topo,
                        arena,
                        pt,
                        snap_window,
                        tol.linear,
                        // The candidate itself must lie on the crossed surface
                        // AND inside the face's boundary region — the solved
                        // point passed containment, but the vertex sits up to
                        // the window away from it.
                        |p| {
                            distance_to_surface(p, surface) <= tol.linear
                                && containments[face_idx].accepts(p)
                        },
                    )
                    .and_then(|vid| {
                        let vp = topo.vertex(vid).ok()?.point();
                        let brepkit_topology::edge::EdgeCurve::Line = &curve else {
                            return None;
                        };
                        let d = end_pos - start_pos;
                        let len_sq = d.length_squared();
                        if len_sq < tol.linear * tol.linear {
                            return None;
                        }
                        let s = ((vp - start_pos).dot(d) / len_sq).clamp(0.0, 1.0);
                        let t_new = (t1 - t0).madd(s, t0);
                        let on_curve = (curve.evaluate_with_endpoints(t_new, start_pos, end_pos)
                            - vp)
                            .length()
                            <= tol.linear;
                        on_curve.then_some((vid, t_new))
                    })
                })
                .collect();

            for ((((t, pt), (start_window, end_window, _)), snap), band) in crossings
                .into_iter()
                .zip(endpoint_windows)
                .zip(snaps)
                .zip(bands)
            {
                if !containments[face_idx].accepts(pt) {
                    log::debug!(
                        "EF: dropping crossing of edge {eid:?} at t={t:.6} ({:.4},{:.4},{:.4}) — outside face {fid:?} boundary",
                        pt.x(),
                        pt.y(),
                        pt.z()
                    );
                    continue;
                }

                // A contact at the edge's own endpoint is a vertex-face
                // incidence (VF territory), not an edge crossing. Recording
                // it as EF marks the adjacent pave block as lying inside the
                // face even though the edge merely touches the face there
                // (e.g. a cap-rim arc tangent to a coplanar wall corner).
                if (pt - start_pos).length() <= start_window
                    || (pt - end_pos).length() <= end_window
                {
                    log::debug!(
                        "EF: dropping endpoint contact of edge {eid:?} at t={t:.6} on face {fid:?} (windows {start_window:.2e}/{end_window:.2e})",
                    );
                    continue;
                }

                let existing = find_nearby_vertex(topo, arena, pt, tol);
                let (vertex_id, t) = if let Some(vid) = existing {
                    (vid, t)
                } else if let Some((vid, t_new)) = snap {
                    log::debug!(
                        "EF: snapping tangential crossing of edge {eid:?} at t={t:.6} to \
                         existing vertex {vid:?} (t={t_new:.6})",
                    );
                    (vid, t_new)
                } else {
                    // The crossing is no closer to the faces than the edge
                    // and its own ends are.
                    let edge_tol = topo.edge(eid).ok().map_or(tol.linear, |e| {
                        [e.start(), e.end()]
                            .into_iter()
                            .filter_map(|v| topo.vertex(v).ok())
                            .map(Vertex::tolerance)
                            .fold(
                                e.tolerance()
                                    .filter(|&t| t <= crate::ds::shape_store::MAX_WIDEN)
                                    .unwrap_or(tol.linear),
                                f64::max,
                            )
                    });
                    let found = super::helpers::pave_vertex_within_tolerance(
                        topo, arena, pt, edge_tol, tol,
                    );
                    match found {
                        Some(vid) => {
                            // The pave sits at the shared vertex's foot, so
                            // the edge's pieces end where that vertex is.
                            let foot =
                                match (topo.edge(eid).map(|e| e.curve().clone()), topo.vertex(vid))
                                {
                                    (Ok(EdgeCurve::Line), Ok(v)) => {
                                        let dir = end_pos - start_pos;
                                        let len_sq = dir.dot(dir);
                                        if len_sq > 0.0 {
                                            let along = (v.point() - start_pos).dot(dir) / len_sq;
                                            (t1 - t0).madd(along, t0)
                                        } else {
                                            t
                                        }
                                    }
                                    _ => t,
                                };
                            (vid, foot)
                        }
                        None => (topo.add_vertex(Vertex::new(pt, edge_tol)), t),
                    }
                };

                let pave = Pave::new(vertex_id, t);
                add_pave_to_edge(arena, topo, eid, pave);

                arena.interference.ef.push(Interference::EF {
                    edge: eid,
                    face: fid,
                    new_vertex: Some(vertex_id),
                    parameter: Some(t),
                    run: band,
                });

                arena.face_info_mut(fid).vertices_in.insert(vertex_id);

                log::debug!("EF: edge {eid:?} crosses face {fid:?} at t={t:.6}");
            }
        }
    }

    Ok(())
}

/// Find edge-plane crossings using algebraic ray-plane intersection.
#[allow(clippy::too_many_arguments)]
pub(super) fn find_edge_plane_crossings(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    if matches!(curve, EdgeCurve::Line) {
        let dir = end_pos - start_pos;
        let denom = dir.dot(normal);

        // 1e-15 checks for mathematical degeneracy (line parallel to
        // plane), not geometric tolerance.
        if denom.abs() < 1e-15 {
            // Line parallel to plane — no single crossing
            return Vec::new();
        }

        let origin_dot =
            start_pos.x() * normal.x() + start_pos.y() * normal.y() + start_pos.z() * normal.z();
        let s = (d - origin_dot) / denom;

        // s is in [0, 1] parameterization of start..end
        if !(-1e-7..=1.0 + 1e-7).contains(&s) {
            return Vec::new();
        }

        let s_clamped = s.clamp(0.0, 1.0);
        let pt = start_pos + dir * s_clamped;
        let t = s_clamped.madd(t1 - t0, t0);
        vec![(t, pt)]
    } else if matches!(curve, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
        conic_plane_crossings(curve, start_pos, end_pos, t0, t1, normal, d, tol)
    } else {
        find_crossings_by_sampling(
            curve,
            start_pos,
            end_pos,
            t0,
            t1,
            &|pt: Point3| pt.x() * normal.x() + pt.y() * normal.y() + pt.z() * normal.z() - d,
            tol.linear,
        )
    }
}

/// A circle or ellipse edge's crossings of the plane `normal·p = d`.
///
/// `normal·C(t) - d` is a sinusoid in the conic's angle, so three evaluations
/// fix it and its roots are closed-form, exact where a sampled search lands a
/// grazing sample's neighbourhood a few 1e-8 off the root. A contact within
/// the tolerance of tangency counts once.
#[allow(clippy::too_many_arguments)]
fn conic_plane_crossings(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let at = |t: f64| curve.evaluate_with_endpoints(t, start_pos, end_pos);
    let f = |t: f64| {
        let p = at(t);
        normal
            .x()
            .madd(p.x(), normal.y().madd(p.y(), normal.z() * p.z()))
            - d
    };
    let (f0, fq, fp) = (f(0.0), f(FRAC_PI_2), f(PI));
    let c = 0.5 * (f0 + fp);
    let (a, b) = (0.5 * (f0 - fp), fq - c);
    let amp = a.hypot(b);
    if amp <= 1e-12 {
        return Vec::new();
    }
    let ratio = -c / amp;
    if ratio.abs() > 1.0 + tol.linear / amp {
        return Vec::new();
    }
    let (phase, delta) = (b.atan2(a), ratio.clamp(-1.0, 1.0).acos());
    // Near tangency `acos` parts the double root by its rounding, so roots
    // whose points lie within the tolerance band are the one contact, at the
    // sinusoid's extreme on the plane's side.
    let extreme = if ratio < 0.0 { phase + PI } else { phase };
    let (pair, tangent) = ([phase - delta, phase + delta], [extreme]);
    let roots: &[f64] = if (at(pair[0]) - at(pair[1])).length() <= 10.0 * tol.linear {
        &tangent
    } else {
        &pair
    };
    // Each of two roots moves no more than halfway to the other.
    let reach = if roots.len() == 2 {
        0.5 * (at(pair[0]) - at(pair[1])).length()
    } else {
        0.0
    };
    let mut out: Vec<(f64, Point3)> = Vec::new();
    for &root in roots {
        let t = t0 + (root - t0).rem_euclid(TAU);
        if t > t1 + 1e-12 {
            continue;
        }
        out.push(match conic_axes(curve) {
            Some(axes) => settle_conic_crossing(at(t), t, axes, normal, d, reach),
            None => (t, at(t)),
        });
    }
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    out
}

/// The semi-axis vectors `a u` and `b v` of a circle or an ellipse
/// `centre + a u cos t + b v sin t`.
pub(super) fn conic_axes(curve: &EdgeCurve) -> Option<(Vec3, Vec3)> {
    match curve {
        EdgeCurve::Circle(c) => Some((c.u_axis() * c.radius(), c.v_axis() * c.radius())),
        EdgeCurve::Ellipse(e) => Some((e.u_axis() * e.semi_major(), e.v_axis() * e.semi_minor())),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
    }
}

/// A conic's crossing of the plane `normal·p = d` from `point`, its point at
/// `t`, a root of the closed-form solve, and the crossing's parameter.
///
/// The conic's point carries the rounding of its centre's coordinates and
/// the root that of the solve: a flat leaning 5e-8 off a bore's axis meets
/// the bore along an ellipse whose centre lies 1e8 away, where the two put
/// a crossing 1.2e-7 off the plane. The crossing is found again in a frame
/// at `point`: the conic's change from `t` over a step `δ`, `a u (cos(t+δ)
/// - cos t) + b v (sin(t+δ) - sin t)`, is written as products with
/// `2 sin(δ/2)`, free of the centre and of the rounding of `t + δ`, and
/// Newton steps on `δ` bring the point onto the plane. `point` is kept when
/// the crossing would move farther than `reach` or lie no nearer the plane.
pub(super) fn settle_conic_crossing(
    point: Point3,
    t: f64,
    (au, bv): (Vec3, Vec3),
    normal: Vec3,
    d: f64,
    reach: f64,
) -> (f64, Point3) {
    let off = |q: Point3| {
        normal
            .x()
            .madd(q.x(), normal.y().madd(q.y(), normal.z() * q.z()))
            - d
    };
    let change = |delta: f64| {
        let chord = 2.0 * (0.5 * delta).sin();
        let mid = 0.5f64.madd(delta, t);
        au * (-chord * mid.sin()) + bv * (chord * mid.cos())
    };
    let start = off(point);
    let mut delta = 0.0;
    for _ in 0..4 {
        let s = t + delta;
        let slope = normal.dot(bv * s.cos() - au * s.sin());
        let step = (start + normal.dot(change(delta))) / slope;
        if !step.is_finite() {
            return (t, point);
        }
        delta -= step;
    }
    let moved = change(delta);
    let settled = point + moved;
    if moved.length() <= reach && off(settled).abs() <= start.abs() {
        (t + delta, settled)
    } else {
        (t, point)
    }
}

/// What an edge's pass over a face finds before anything is added: `None`
/// when the edge lies on the face's surface (16 samples within the
/// tolerance), else where it crosses the surface.
type Survey = Option<Vec<(f64, Point3)>>;

/// [`Survey`] of one edge and one face's surface.
#[allow(clippy::too_many_arguments)]
fn survey(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    surface: &FaceSurface,
    hull: Option<Aabb3>,
    pieces: Option<&[Aabb3]>,
    tol: Tolerance,
) -> Survey {
    // An edge every point of which lies farther from a B-spline surface
    // than the scan below reads as near it crosses it nowhere: the scan's
    // distances are those of points of the surface, or bounds below them,
    // and it keeps only crossings within the tolerance of the surface.
    if let (FaceSurface::Nurbs(nurbs), Some(pieces)) = (surface, pieces)
        && let Some(index) = nurbs.span_index()
        && pieces
            .iter()
            .all(|b| index.box_clears(nurbs, b, 2.0 * NEAR_LIMIT * tol.linear))
    {
        return Some(Vec::new());
    }
    let n_chk = 16;
    let edge_on_surface = (0..=n_chk).all(|i| {
        let t = t0 + (t1 - t0) * (f64::from(i) / f64::from(n_chk));
        let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        distance_to_surface_near(pt, surface, hull, tol) < tol.linear
    });
    if edge_on_surface {
        return None;
    }
    Some(match surface {
        FaceSurface::Plane { normal, d } => {
            find_edge_plane_crossings(curve, start_pos, end_pos, t0, t1, *normal, *d, tol)
        }
        _ => find_edge_surface_crossings(curve, start_pos, end_pos, t0, t1, surface, hull, tol),
    })
}

/// The [`Survey`] of every edge and face pair that [`check_edge_face_pairs`]
/// examines, by edge index and then face index. A survey reads only the
/// edge's curve and the face's surface, which adding vertices leaves as they
/// are, so the pairs are surveyed in parallel up front; the pass then takes
/// them in its own order.
#[allow(clippy::too_many_arguments)]
fn survey_pairs(
    topo: &Topology,
    edges: &[EdgeId],
    faces: &[FaceId],
    face_boundary_edges: &[HashSet<EdgeId>],
    face_aabbs: &[Aabb3],
    hulls: &[Option<Aabb3>],
    tol: Tolerance,
) -> Vec<std::collections::HashMap<usize, Survey>> {
    /// An edge and a face to survey, with the edge's curve and ends.
    struct Pair<'a> {
        edge_idx: usize,
        face_idx: usize,
        curve: &'a EdgeCurve,
        start_pos: Point3,
        end_pos: Point3,
        t0: f64,
        t1: f64,
        surface: &'a FaceSurface,
    }
    let surfaces: Vec<Option<&FaceSurface>> = faces
        .iter()
        .map(|&fid| {
            topo.face(fid)
                .ok()
                .map(brepkit_topology::face::Face::surface)
        })
        .collect();
    let mut pieces: Vec<Option<Vec<Aabb3>>> = (0..edges.len()).map(|_| None).collect();
    let mut pairs = Vec::new();
    for (edge_idx, &eid) in edges.iter().enumerate() {
        let Some((curve, start_pos, end_pos, t0, t1)) = topo.edge(eid).ok().and_then(|edge| {
            let sp = topo.vertex(edge.start()).ok()?.point();
            let ep = topo.vertex(edge.end()).ok()?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            Some((edge.curve(), sp, ep, t0, t1))
        }) else {
            continue;
        };
        // The same box and the same rejections as the pass.
        let edge_aabb = {
            let mut pts = Vec::with_capacity(N_SAMPLES + 1);
            for i in 0..=N_SAMPLES {
                let t = t0 + (t1 - t0) * (i as f64 / N_SAMPLES as f64);
                pts.push(curve.evaluate_with_endpoints(t, start_pos, end_pos));
            }
            Aabb3::try_from_points(pts).map(|a| a.expanded(tol.linear))
        };
        // An edge's piece boxes serve only its pairs with B-spline faces,
        // so they are made for the first such pair that is kept.
        let mut boxed = false;
        for (face_idx, surface) in surfaces.iter().enumerate() {
            let Some(surface) = surface else {
                continue;
            };
            if face_boundary_edges[face_idx].contains(&eid) {
                continue;
            }
            if let Some(ea) = &edge_aabb
                && !ea.intersects(face_aabbs[face_idx])
            {
                continue;
            }
            if let (Some(ea), Some(hull), EdgeCurve::Line) = (&edge_aabb, hulls[face_idx], curve)
                && !ea.intersects(hull.expanded(tol.linear))
            {
                continue;
            }
            if !boxed && matches!(surface, FaceSurface::Nurbs(_)) {
                pieces[edge_idx] = curve_piece_boxes(curve, start_pos, end_pos, t0, t1);
                boxed = true;
            }
            pairs.push(Pair {
                edge_idx,
                face_idx,
                curve,
                start_pos,
                end_pos,
                t0,
                t1,
                surface,
            });
        }
    }
    let run = |pair: &Pair<'_>| {
        survey(
            pair.curve,
            pair.start_pos,
            pair.end_pos,
            pair.t0,
            pair.t1,
            pair.surface,
            hulls[pair.face_idx],
            pieces[pair.edge_idx].as_deref(),
            tol,
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let found: Vec<Survey> = {
        use rayon::prelude::*;
        pairs.par_iter().map(run).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let found: Vec<Survey> = pairs.iter().map(run).collect();
    let mut surveys: Vec<std::collections::HashMap<usize, Survey>> = (0..edges.len())
        .map(|_| std::collections::HashMap::new())
        .collect();
    for (pair, found) in pairs.iter().zip(found) {
        surveys[pair.edge_idx].insert(pair.face_idx, found);
    }
    surveys
}

/// Boxes that together hold every point an edge's curve takes over its
/// parameters `t0` to `t1`: a line's in eight pieces; a circle's or an
/// ellipse's in sixteen arcs, each the box of its ends grown by the arc's sag
/// bound (the most its second derivative reaches, times its angle squared
/// over eight); a
/// B-spline's the boxes of the nets of the Bezier pieces the range runs
/// over, each piece within the hull of its net. `None` where a weight is not
/// positive, the pieces cannot be read, or a box is not finite.
pub fn curve_piece_boxes(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
) -> Option<Vec<Aabb3>> {
    let finite = |p: Point3| p.x().is_finite() && p.y().is_finite() && p.z().is_finite();
    piece_boxes(curve, start_pos, end_pos, t0, t1)
        .filter(|boxes| boxes.iter().all(|b| finite(b.min) && finite(b.max)))
}

/// [`curve_piece_boxes`], whose boxes may not be finite.
fn piece_boxes(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
) -> Option<Vec<Aabb3>> {
    let (lo, hi) = (t0.min(t1), t0.max(t1));
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let grown = |b: Aabb3, by: f64| {
        let size = b.min.x().abs().max(b.min.y().abs()).max(b.min.z().abs())
            + b.max.x().abs().max(b.max.y().abs()).max(b.max.z().abs());
        b.expanded(by + 1e-9 * size.max(1.0))
    };
    let arcs = |at: &dyn Fn(f64) -> Point3, radius: f64| -> Vec<Aabb3> {
        const ARCS: usize = 16;
        let step = (hi - lo) / ARCS as f64;
        (0..ARCS)
            .map(|k| {
                let a = (k as f64).madd(step, lo);
                let b = if k + 1 == ARCS { hi } else { a + step };
                let w = b - a;
                grown(Aabb3::from_points([at(a), at(b)]), radius * w * w / 8.0)
            })
            .collect()
    };
    match curve {
        EdgeCurve::Line => {
            const PIECES: usize = 8;
            let at = |t: f64| start_pos + (end_pos - start_pos) * t;
            let step = (hi - lo) / PIECES as f64;
            Some(
                (0..PIECES)
                    .map(|k| {
                        let a = (k as f64).madd(step, lo);
                        let b = if k + 1 == PIECES { hi } else { a + step };
                        grown(Aabb3::from_points([at(a), at(b)]), 0.0)
                    })
                    .collect(),
            )
        }
        EdgeCurve::Circle(c) => Some(arcs(
            &|t| c.evaluate(t),
            conic_reach(c.radius(), c.radius(), c.u_axis(), c.v_axis()),
        )),
        EdgeCurve::Ellipse(e) => Some(arcs(
            &|t| e.evaluate(t),
            conic_reach(e.semi_major(), e.semi_minor(), e.u_axis(), e.v_axis()),
        )),
        EdgeCurve::NurbsCurve(n) => {
            // A weight below the least normal float can carry the curve out
            // of its control box, as one that is not positive can.
            let unbounded = |w: &f64| *w <= 0.0 || !w.is_normal();
            if n.weights().iter().any(unbounded) {
                return None;
            }
            let segments = brepkit_math::nurbs::decompose::curve_to_bezier_segments(n).ok()?;
            let boxes: Vec<Aabb3> = segments
                .iter()
                .filter(|seg| {
                    let k = seg.knots();
                    k.first().is_some_and(|&a| a <= hi) && k.last().is_some_and(|&b| b >= lo)
                })
                .map(|seg| {
                    if seg.weights().iter().any(unbounded) {
                        return None;
                    }
                    Aabb3::try_from_points(seg.control_points().iter().copied())
                        .map(|b| grown(b, 0.0))
                })
                .collect::<Option<_>>()?;
            (!boxes.is_empty()).then_some(boxes)
        }
    }
}

/// The most `|a cos t u + b sin t v|` reaches over `t`, the size of the
/// second derivative of a conic on axes `u` and `v` that need not be unit or
/// square: its square is no more than the larger of `(a |u|)²` and
/// `(b |v|)²` plus `|a b u . v|`.
fn conic_reach(a: f64, b: f64, u: Vec3, v: Vec3) -> f64 {
    // The axes are scaled before they are multiplied: a large radius on
    // short axes is an ordinary conic, but the product of the two lengths
    // can overflow on its own.
    let (u, v) = (u * a, v * b);
    let most = u.length().max(v.length());
    (most * most + u.dot(v).abs()).sqrt()
}

/// Find edge-surface crossings by sampling the distance to the surface and refining.
#[allow(clippy::too_many_arguments)]
fn find_edge_surface_crossings(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    surface: &FaceSurface,
    hull: Option<Aabb3>,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    let n = N_SAMPLES;
    let mut crossings = Vec::new();
    let mut prev_dist = f64::MAX;
    let mut prev_t = t0;

    let samples: Vec<(f64, Point3)> = (0..=n)
        .map(|i| {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            (t, curve.evaluate_with_endpoints(t, start_pos, end_pos))
        })
        .collect();
    // On a NURBS face each sample's one projection gives its distance and,
    // within a sample step of the patch's hull (a crossing lies within a
    // step of the samples either side of it), the side of the surface it
    // lies on, for the sign-change scan below.
    let nurbs = matches!(surface, FaceSurface::Nurbs(_));
    let step = samples
        .windows(2)
        .map(|w| (w[1].1 - w[0].1).length())
        .fold(0.0_f64, f64::max);
    let hull_gap = |p: Point3| {
        hull.map_or(0.0, |h| {
            let gap = |x: f64, lo: f64, hi: f64| (lo - x).max(x - hi).max(0.0);
            gap(p.x(), h.min.x(), h.max.x())
                .hypot(gap(p.y(), h.min.y(), h.max.y()))
                .hypot(gap(p.z(), h.min.z(), h.max.z()))
        })
    };
    let probed: Vec<(f64, Option<f64>)> = samples
        .iter()
        .map(|&(_, p)| {
            let g = hull_gap(p);
            if !nurbs || g > step + tol.linear {
                return (distance_to_surface_near(p, surface, hull, tol), None);
            }
            let foot = surface
                .project_point(p)
                .and_then(|(u, v)| surface.evaluate(u, v).map(|q| (q, surface.normal(u, v))));
            match foot {
                Some((q, normal)) => {
                    let dist = if g > NEAR_LIMIT * tol.linear {
                        g
                    } else {
                        (p - q).length()
                    };
                    (dist, Some((p - q).dot(normal)))
                }
                None => (distance_to_surface_near(p, surface, hull, tol), None),
            }
        })
        .collect();

    // A run of samples within the tolerance of the surface holds a contact
    // where the edge comes nearest to it, not one at each sample. An edge
    // nearly tangent to the surface stays within the tolerance over a long
    // stretch (a ball's equator 3e-7 inside a rod's wall for a third of a
    // turn), and a crossing at each of its samples would cut it into pieces
    // that all lie on the surface. Where the edge passes to the other side
    // inside the run, each such crossing is kept: a line 6e-8 inside a ball
    // at its middle crosses it twice, 2385 tolerances apart. Each touch the
    // run also holds, a least of the distance apart from the crossings, is
    // kept as well.
    let mut run: Option<(usize, usize)> = None;
    let near_a_crossing = |t: f64, crossings: &[(f64, Point3)]| {
        crossings
            .iter()
            .any(|&(ct, _)| (t - ct).abs() < ((t1 - t0) / (n as f64) * 2.0).abs())
    };
    let close_run = |first: usize, last: usize, best: usize, crossings: &mut Vec<_>| {
        if first == last {
            let t = samples[first].0;
            if first > 0 && !near_a_crossing(t, crossings) {
                crossings.push(refine_crossing(
                    curve,
                    start_pos,
                    end_pos,
                    samples[first - 1].0,
                    t,
                    surface,
                    tol,
                ));
            }
        } else {
            let around: Vec<f64> = (first.saturating_sub(1)..=(last + 1).min(n))
                .map(|k| samples[k].0)
                .collect();
            let flips = run_contacts(curve, start_pos, end_pos, &around, surface, tol);
            if !flips.is_empty() {
                let before = crossings.len();
                for (t, p) in flips {
                    if !near_a_crossing(t, &crossings[..before]) {
                        crossings.push((t, p));
                    }
                }
                return;
            }
            let found = tangent_contact(curve, start_pos, end_pos, &around, surface, tol)
                .unwrap_or_else(|| {
                    refine_crossing(
                        curve,
                        start_pos,
                        end_pos,
                        samples[best.saturating_sub(1)].0,
                        samples[(best + 1).min(n)].0,
                        surface,
                        tol,
                    )
                });
            if !near_a_crossing(found.0, crossings) {
                crossings.push(found);
            }
        }
    };
    for (i, (&(t, _), &(dist, _))) in samples.iter().zip(&probed).enumerate() {
        if dist < tol.linear {
            run = Some(match run {
                Some((first, best)) if probed[best].0 <= dist => (first, best),
                Some((first, _)) => (first, i),
                None => (i, i),
            });
        } else if let Some((first, best)) = run.take() {
            close_run(first, i - 1, best, &mut crossings);
        } else if i > 0 && prev_dist > tol.linear && dist > tol.linear {
            let mid_t = f64::midpoint(prev_t, t);
            let mid_pt = curve.evaluate_with_endpoints(mid_t, start_pos, end_pos);
            let mid_dist = distance_to_surface_near(mid_pt, surface, hull, tol);
            if mid_dist < prev_dist.min(dist) && mid_dist < tol.linear * 2.0 {
                let refined = refine_crossing(curve, start_pos, end_pos, prev_t, t, surface, tol);
                if distance_to_surface(refined.1, surface) < tol.linear {
                    crossings.push(refined);
                }
            }

            // Tangent contact: near-surface sample triggers golden section minimum search
            if prev_dist < NEAR_LIMIT * tol.linear || dist < NEAR_LIMIT * tol.linear {
                let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
                let mut lo = prev_t;
                let mut hi = t;
                for _ in 0..30 {
                    let m1 = hi - phi * (hi - lo);
                    let m2 = lo + phi * (hi - lo);
                    let d1 = distance_to_surface(
                        curve.evaluate_with_endpoints(m1, start_pos, end_pos),
                        surface,
                    );
                    let d2 = distance_to_surface(
                        curve.evaluate_with_endpoints(m2, start_pos, end_pos),
                        surface,
                    );
                    if d1 < d2 {
                        hi = m2;
                    } else {
                        lo = m1;
                    }
                }
                let t_min = f64::midpoint(lo, hi);
                let pt_min = curve.evaluate_with_endpoints(t_min, start_pos, end_pos);
                if distance_to_surface(pt_min, surface) < tol.linear {
                    let is_dup = crossings.iter().any(|&(ct, _): &(f64, Point3)| {
                        (t_min - ct).abs() < ((t1 - t0) / (n as f64) * 2.0).abs()
                    });
                    if !is_dup {
                        let refined =
                            refine_crossing(curve, start_pos, end_pos, lo, hi, surface, tol);
                        crossings.push(refined);
                    }
                }
            }
        }

        prev_dist = dist;
        prev_t = t;
    }
    if let Some((first, best)) = run {
        close_run(first, n, best, &mut crossings);
    }

    // The samples find a crossing only where one lands within a few
    // tolerances of the surface; a transversal crossing between two of them
    // (a fillet's meridian arc through an envelope's NURBS corner) shows only
    // as the side of the surface flipping. On a NURBS face that flip is
    // bisected to the surface, and kept only where it lands on it: a flip
    // from the projection jumping to another part of the patch does not.
    if let FaceSurface::Nurbs(_) = surface {
        let side = |t: f64| -> Option<f64> {
            let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
            let (u, v) = surface.project_point(pt)?;
            let on = surface.evaluate(u, v)?;
            Some((pt - on).dot(surface.normal(u, v)))
        };
        let mut prev: Option<(f64, f64)> = None;
        for (&(t, _), &(_, side_at)) in samples.iter().zip(&probed) {
            let Some(s) = side_at else {
                prev = None;
                continue;
            };
            if let Some((t_prev, s_prev)) = prev
                && (s_prev > 0.0) != (s > 0.0)
                && s_prev != 0.0
                && s != 0.0
            {
                let (mut lo, mut hi, mut s_lo) = (t_prev, t, s_prev);
                for _ in 0..60 {
                    let mid = f64::midpoint(lo, hi);
                    let Some(s_mid) = side(mid) else { break };
                    if (s_mid > 0.0) == (s_lo > 0.0) {
                        lo = mid;
                        s_lo = s_mid;
                    } else {
                        hi = mid;
                    }
                }
                let tc = f64::midpoint(lo, hi);
                let pc = curve.evaluate_with_endpoints(tc, start_pos, end_pos);
                // The sample scan may already hold this root; two roots a
                // sample step apart are still two crossings.
                if distance_to_surface(pc, surface) < tol.linear
                    && !crossings
                        .iter()
                        .any(|&(_, cp): &(f64, Point3)| (pc - cp).length() < tol.linear * 100.0)
                {
                    crossings.push((tc, pc));
                }
            }
            prev = Some((t, s));
        }
        crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    crossings
}

/// A line segment's crossings of a cylinder or cone, at their parameters.
fn line_crossings(
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    surface: &FaceSurface,
) -> Vec<(f64, Point3)> {
    let d = end_pos - start_pos;
    let len_sq = d.dot(d);
    if len_sq <= 0.0 {
        return Vec::new();
    }
    super::phase_ff::line_segment_surface_crossings(start_pos, end_pos, surface)
        .into_iter()
        .map(|p| ((t1 - t0).madd((p - start_pos).dot(d) / len_sq, t0), p))
        .collect()
}

/// Where a curve nearly tangent to a surface comes nearest to it between
/// the first and last of the parameters `ts`: a root of its tangent's part
/// along the surface's normal that lies within the tolerance of the
/// surface, the nearest such root where there are several. The distance is
/// flat to second order there, so a search on it stops anywhere within the
/// square root of its rounding error (3e-5 along a ball's equator 3e-7
/// inside a rod's wall), but the tangent's normal part changes sign at a
/// rate that pins the point.
fn tangent_contact(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    ts: &[f64],
    surface: &FaceSurface,
    tol: Tolerance,
) -> Option<(f64, Point3)> {
    let across = |t: f64| -> Option<f64> {
        let p = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        let (u, v) = surface.project_point(p)?;
        Some(
            curve
                .tangent_with_endpoints(t, start_pos, end_pos)
                .dot(surface.normal(u, v)),
        )
    };
    let mut best: Option<(f64, f64, Point3)> = None;
    let mut prev: Option<(f64, f64)> = None;
    for &t in ts {
        let Some(g) = across(t) else {
            prev = None;
            continue;
        };
        if let Some((t_prev, g_prev)) = prev
            && (g_prev > 0.0) != (g > 0.0)
        {
            let (mut lo, mut hi, mut g_lo) = (t_prev, t, g_prev);
            for _ in 0..60 {
                let mid = f64::midpoint(lo, hi);
                let Some(g_mid) = across(mid) else { break };
                if (g_mid > 0.0) == (g_lo > 0.0) {
                    lo = mid;
                    g_lo = g_mid;
                } else {
                    hi = mid;
                }
            }
            let tc = f64::midpoint(lo, hi);
            let pc = curve.evaluate_with_endpoints(tc, start_pos, end_pos);
            let d = distance_to_surface(pc, surface);
            if d < tol.linear && best.is_none_or(|(bd, _, _)| d < bd) {
                best = Some((d, tc, pc));
            }
        }
        prev = Some((t, g));
    }
    best.map(|(_, t, p)| (t, p))
}

/// The span of the samples that the crossing scan reads, lying within the
/// tolerance of `surface` next to each other around a contact at `t`, where
/// at least two do and the curve stays on one side of the surface across
/// them: the stretch along which the curve grazes the surface. A curve
/// that passes through the surface in that stretch crosses it there.
fn tangent_band(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    (t0, t1): (f64, f64),
    t: f64,
    surface: &FaceSurface,
    tol: Tolerance,
) -> Option<(f64, f64)> {
    let n = N_SAMPLES;
    let at = |i: usize| t0 + (t1 - t0) * (i as f64 / n as f64);
    let within = |i: usize| {
        distance_to_surface(
            curve.evaluate_with_endpoints(at(i), start_pos, end_pos),
            surface,
        ) < tol.linear
    };
    let along = ((t - t0) / (t1 - t0) * n as f64).clamp(0.0, n as f64);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (below, above) = (along.floor() as usize, along.ceil() as usize);
    let start = [below, above].into_iter().find(|&i| within(i))?;
    let (mut lo, mut hi) = (start, start);
    while lo > 0 && within(lo - 1) {
        lo -= 1;
    }
    while hi < n && within(hi + 1) {
        hi += 1;
    }
    if hi == lo {
        return None;
    }
    let around = (lo.saturating_sub(1)..=(hi + 1).min(n)).map(at);
    run_crossings(curve, start_pos, end_pos, around, surface, tol)
        .is_empty()
        .then(|| (at(lo).min(at(hi)), at(lo).max(at(hi))))
}

/// Where a curve passes from one side of a surface to the other between
/// the first and last of the parameters `ts`, each bisected onto the
/// surface and kept where it lands within the tolerance of it. A parameter
/// whose point lies within a thousandth of the tolerance of the surface
/// takes no side: where the curve touches the surface at a sample, as a
/// ball's equator touches a rod's wall, rounding alone sets that sample's
/// sign.
fn run_crossings(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    ts: impl IntoIterator<Item = f64>,
    surface: &FaceSurface,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    let floor = 1e-3 * tol.linear;
    let side = |t: f64| -> Option<f64> {
        let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        let (u, v) = surface.project_point(pt)?;
        let on = surface.evaluate(u, v)?;
        Some((pt - on).dot(surface.normal(u, v)))
    };
    let mut out: Vec<(f64, Point3)> = Vec::new();
    let mut prev: Option<(f64, f64)> = None;
    for t in ts {
        let Some(s) = side(t) else {
            prev = None;
            continue;
        };
        if s.abs() <= floor {
            continue;
        }
        if let Some((t_prev, s_prev)) = prev
            && (s_prev > 0.0) != (s > 0.0)
        {
            let (mut lo, mut hi, mut s_lo) = (t_prev, t, s_prev);
            for _ in 0..60 {
                let mid = f64::midpoint(lo, hi);
                let Some(s_mid) = side(mid) else { break };
                if (s_mid > 0.0) == (s_lo > 0.0) {
                    lo = mid;
                    s_lo = s_mid;
                } else {
                    hi = mid;
                }
            }
            let tc = f64::midpoint(lo, hi);
            let pc = curve.evaluate_with_endpoints(tc, start_pos, end_pos);
            // Each change of side lies between its own two samples, so two
            // roots meet only where both land on one point.
            if distance_to_surface(pc, surface) < tol.linear
                && !out.iter().any(|&(_, q)| (pc - q).length() <= tol.linear)
            {
                out.push((tc, pc));
            }
        }
        prev = Some((t, s));
    }
    out
}

/// Where a curve meets a surface between the first and last of the
/// parameters `ts`: each change of side, as [`run_crossings`] finds them,
/// and each touch, where the distance comes down to a least within the
/// tolerance and rises again on the same side. A touch is a root of the
/// distance's rate, the curve's derivative along the surface's normal. A
/// touch and the contact beside it are one contact where the distance
/// between them rises less than a thousandth of the tolerance above the
/// higher of the two: a crossing is kept over a touch, and of two touches
/// the nearer to the surface. Two crossings stay apart as [`run_crossings`]
/// keeps them.
fn run_contacts(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    ts: &[f64],
    surface: &FaceSurface,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    run_contacts_counted(curve, start_pos, end_pos, ts, surface, tol).0
}

/// [`run_contacts`], and how many intervals its search read.
///
/// Two roots of the rate can lie between two samples where it has one
/// sign, a touch and a most of the distance. Each interval is halved until
/// a bound on the rate's second derivative shows that it holds no root, or
/// one where the rate changes sign, or until it is shorter than the
/// tolerance or the distance can change across it by less than the merge
/// allows. A spline is cut at its knots first, where that derivative can
/// jump, and each knot span takes its own bound. Without a bound (a
/// B-spline surface) each interval is halved four times. Past
/// [`RUN_INTERVALS`] intervals, or [`MOST_HALVINGS`] halvings, an interval
/// is read once at its ends, as the samples alone are: a touch can then
/// hide in it, but no curve holds the search longer.
#[allow(clippy::too_many_lines)]
fn run_contacts_counted(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    ts: &[f64],
    surface: &FaceSurface,
    tol: Tolerance,
) -> (Vec<(f64, Point3)>, usize) {
    /// A contact's parameter, point and distance from the surface, and
    /// whether the curve passes through the surface there.
    struct Contact {
        t: f64,
        p: Point3,
        depth: f64,
        crossing: bool,
    }
    /// A parameter, the signed distance there and its rate.
    type Reading = (f64, f64, f64);
    /// Halvings without a bound on the rate.
    const UNBOUNDED_HALVINGS: u32 = 4;
    let floor = 1e-3 * tol.linear;
    // The point, the signed distance and its rate, on `on`.
    let foot_on = |on: &EdgeCurve, t: f64| -> Option<(Point3, f64, f64)> {
        let pt = on.evaluate_with_endpoints(t, start_pos, end_pos);
        let (u, v) = surface.project_point(pt)?;
        let normal = surface.normal(u, v);
        let side = (pt - surface.evaluate(u, v)?).dot(normal);
        let rate = curve_velocity(on, start_pos, end_pos, t).dot(normal);
        Some((pt, side, rate))
    };
    let foot = |t: f64| foot_on(curve, t);
    // The bounds on the curve's first three derivatives over `[a, b]`,
    // which lies within one knot span of a spline.
    let speeds = |a: f64, b: f64| -> Option<[f64; 3]> {
        match curve {
            EdgeCurve::NurbsCurve(n) => {
                let count = n.control_points().len();
                let span = brepkit_math::nurbs::basis::find_span(
                    count,
                    n.degree(),
                    f64::midpoint(a, b),
                    n.knots(),
                );
                spline_span_derivative_bounds(n, span)
            }
            _ => curve_derivative_bounds(curve, start_pos, end_pos),
        }
    };
    // The bound on the curve's first derivative and on the rate's second
    // derivative over `[a, b]`.
    let bend = |a: f64, b: f64| -> Option<(f64, f64)> {
        let [c1, c2, c3] = speeds(a, b)?;
        let mid = curve.evaluate_with_endpoints(f64::midpoint(a, b), start_pos, end_pos);
        let (k2, k3) = distance_derivative_bounds(surface, mid, 0.5 * c1 * (b - a).abs())?;
        Some((c1, (k3 * c1).madd(c1 * c1, (3.0 * k2 * c1).madd(c2, c3))))
    };

    let mut contacts: Vec<Contact> =
        run_crossings(curve, start_pos, end_pos, ts.iter().copied(), surface, tol)
            .into_iter()
            .map(|(t, p)| Contact {
                t,
                p,
                depth: 0.0,
                crossing: true,
            })
            .collect();
    let mut cuts: Vec<f64> = ts.to_vec();
    // A spline is read in pieces split at its knots, each piece up to the
    // parameter it is keyed by: a piece read at its end gives the rate on
    // its own side of a knot where the derivative jumps, which the whole
    // spline reads from the span after it.
    let mut pieces: Vec<(f64, EdgeCurve)> = Vec::new();
    if let (EdgeCurve::NurbsCurve(nurbs), [first, .., last]) = (curve, ts) {
        let (lo, hi) = (first.min(*last), first.max(*last));
        let mut knots: Vec<f64> = nurbs
            .knots()
            .iter()
            .copied()
            .filter(|&k| lo < k && k < hi)
            .collect();
        knots.dedup();
        let mut rest = nurbs.clone();
        for &k in &knots {
            if let Ok((head, tail)) = brepkit_math::nurbs::knot_ops::curve_split(&rest, k) {
                pieces.push((k, EdgeCurve::NurbsCurve(head)));
                rest = tail;
            }
        }
        pieces.push((f64::INFINITY, EdgeCurve::NurbsCurve(rest)));
        cuts.extend(knots);
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    // The piece that holds `[a, b]`.
    let piece = |a: f64, b: f64| -> &EdgeCurve {
        let mid = f64::midpoint(a, b);
        pieces
            .get(pieces.partition_point(|&(end, _)| end <= mid))
            .map_or(curve, |(_, on)| on)
    };
    // The mosts of the distance the search passes, for the rises below.
    let mut mosts: Vec<(f64, f64)> = Vec::new();
    // Where the rate jumps across a knot from one sign to the other, the
    // distance has a least or a most at the knot itself.
    for pair in pieces.windows(2) {
        let k = pair[0].0;
        let (Some((p, s, g_in)), Some((_, _, g_out))) =
            (foot_on(&pair[0].1, k), foot_on(&pair[1].1, k))
        else {
            continue;
        };
        if (g_in > 0.0) == (g_out > 0.0) {
            continue;
        }
        let sign = if s < 0.0 { -1.0 } else { 1.0 };
        if s.abs() <= floor || (sign * g_in <= 0.0 && sign * g_out >= 0.0) {
            if distance_to_surface(p, surface) < tol.linear {
                contacts.push(Contact {
                    t: k,
                    p,
                    depth: s.abs(),
                    crossing: false,
                });
            }
        } else {
            mosts.push((k, s.abs()));
        }
    }
    let mut stack: Vec<(Reading, Reading, u32)> = Vec::new();
    for pair in cuts.windows(2) {
        let on = piece(pair[0], pair[1]);
        if let (Some((_, sa, ga)), Some((_, sb, gb))) = (foot_on(on, pair[0]), foot_on(on, pair[1]))
        {
            stack.push(((pair[0], sa, ga), (pair[1], sb, gb), 0));
        }
    }
    let mut read = 0_usize;
    while let Some(((a, s_a, g_a), (b, s_b, g_b), halvings)) = stack.pop() {
        read += 1;
        let width = b - a;
        let changes = (g_a > 0.0) != (g_b > 0.0);
        let spent = read > RUN_INTERVALS || halvings >= MOST_HALVINGS;
        let bound = if spent { None } else { bend(a, b) };
        let settled = spent
            || match bound {
                // One root where the rate's slope keeps its sign, none where
                // the rate cannot come down to zero from its values at the
                // ends.
                Some((_, m)) if changes => (g_b - g_a).abs() > m * width * width,
                Some((_, m)) => g_a.abs().min(g_b.abs()) > 0.125 * m * width * width,
                None => halvings >= UNBOUNDED_HALVINGS,
            };
        let short = bound.is_some_and(|(c1, m)| {
            c1 * width <= tol.linear
                || width * (0.125 * m).madd(width * width, g_a.abs().max(g_b.abs())) < floor
        });
        let on = piece(a, b);
        if !(settled || short) {
            let mid = f64::midpoint(a, b);
            if let Some((_, s_m, g_m)) = foot_on(on, mid) {
                stack.push(((a, s_a, g_a), (mid, s_m, g_m), halvings + 1));
                stack.push(((mid, s_m, g_m), (b, s_b, g_b), halvings + 1));
                continue;
            }
        }
        if !changes {
            continue;
        }
        let (mut lo, mut hi, mut g_at_lo) = (a, b, g_a);
        for _ in 0..60 {
            let mid = f64::midpoint(lo, hi);
            let Some((_, _, g_mid)) = foot_on(on, mid) else {
                break;
            };
            if (g_mid > 0.0) == (g_at_lo > 0.0) {
                lo = mid;
                g_at_lo = g_mid;
            } else {
                hi = mid;
            }
        }
        let tm = f64::midpoint(lo, hi);
        let Some((pm, s_m, _)) = foot_on(on, tm) else {
            continue;
        };
        // A most of the distance between two of its leasts is no contact.
        if s_m.abs() > s_a.abs().min(s_b.abs()) {
            mosts.push((tm, s_m.abs()));
            continue;
        }
        if distance_to_surface(pm, surface) < tol.linear {
            contacts.push(Contact {
                t: tm,
                p: pm,
                depth: s_m.abs(),
                crossing: false,
            });
        }
    }
    contacts.sort_by(|a, b| a.t.total_cmp(&b.t));
    mosts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut samples: Vec<f64> = ts.to_vec();
    samples.sort_by(f64::total_cmp);
    // The most of the mosts' distances over each run of `2^j` of them from
    // each index, so any range of them reads in two lookups.
    let mut most_over: Vec<Vec<f64>> = vec![mosts.iter().map(|&(_, s)| s).collect()];
    while let Some(last) = most_over.last()
        && 2 * (1 << (most_over.len() - 1)) <= mosts.len()
    {
        let half = 1 << (most_over.len() - 1);
        let next = (0..=mosts.len() - 2 * half)
            .map(|i| last[i].max(last[i + half]))
            .collect();
        most_over.push(next);
    }
    // The most the distance reaches strictly between two parameters, read
    // at the samples there, at the mosts found and midway.
    let rise = |a: f64, b: f64| {
        let within = |xs: &[f64]| xs.partition_point(|&x| x <= a)..xs.partition_point(|&x| x < b);
        let read = samples[within(&samples)]
            .iter()
            .copied()
            .chain(std::iter::once(f64::midpoint(a, b)))
            .filter_map(|t| foot(t).map(|(_, s, _)| s.abs()))
            .fold(0.0_f64, f64::max);
        let (lo, hi) = (
            mosts.partition_point(|&(t, _)| t <= a),
            mosts.partition_point(|&(t, _)| t < b),
        );
        if hi <= lo {
            return read;
        }
        let level = (usize::BITS - 1 - (hi - lo).leading_zeros()) as usize;
        let span = 1 << level;
        read.max(most_over[level][lo])
            .max(most_over[level][hi - span])
    };
    let mut kept: Vec<Contact> = Vec::new();
    for c in contacts {
        if let Some(last) = kept.last_mut()
            && !(last.crossing && c.crossing)
            && ((last.p - c.p).length() <= tol.linear
                || rise(last.t, c.t) < last.depth.max(c.depth) + floor)
        {
            if !last.crossing && (c.crossing || c.depth < last.depth) {
                *last = c;
            }
            continue;
        }
        kept.push(c);
    }
    (kept.into_iter().map(|c| (c.t, c.p)).collect(), read)
}

/// The most intervals one [`run_contacts`] search reads before it reads the
/// rest once at their ends: four times the most a run of the tests reads,
/// 7905, along a ball's equator 3e-7 inside a rod's wall.
const RUN_INTERVALS: usize = 1 << 15;

/// The most halvings of one sample interval in a [`run_contacts`] search:
/// the runs of the tests take at most 7.
const MOST_HALVINGS: u32 = 20;

/// Bounds on the lengths of the first three derivatives of a spline within
/// its knot span `span`, read from the control points of the derivatives
/// of its homogeneous form, `w (C - o)` and `w` about the span's first
/// control point `o`: `|C - o|` is at most the farthest of the span's
/// control points, `w` at least its least weight, and `C' w = (w (C - o))'
/// - (C - o) w'`, and so on for `C''` and `C'''`. Order `k` of the
/// derivative's control points `i` is `(p - k + 1)` times the difference of
/// order `k - 1`'s at `i + 1` and `i` over `U[i + p + 1] - U[i + k]`, and the
/// span holds those from `span - p` to `span - k`.
fn spline_span_derivative_bounds(
    n: &brepkit_math::nurbs::curve::NurbsCurve,
    span: usize,
) -> Option<[f64; 3]> {
    let (p, knots) = (n.degree(), n.knots());
    let first = span.checked_sub(p)?;
    let points = n.control_points().get(first..=span)?;
    let weights = n.weights().get(first..=span)?;
    let origin = *points.first()?;
    let least = weights.iter().copied().fold(f64::INFINITY, f64::min);
    if !(least > 0.0 && least.is_finite()) {
        return None;
    }
    let far = points
        .iter()
        .map(|&q| (q - origin).length())
        .fold(0.0_f64, f64::max);
    let mut level: Vec<[f64; 4]> = points
        .iter()
        .zip(weights)
        .map(|(&q, &w)| {
            let d = (q - origin) * w;
            [d.x(), d.y(), d.z(), w]
        })
        .collect();
    // The most of each homogeneous derivative's point part and weight, at
    // orders 1, 2 and 3.
    let mut most = [(0.0_f64, 0.0_f64); 3];
    for (k, slot) in (1..=3).zip(&mut most) {
        if k > p {
            break;
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = (p - k + 1) as f64;
        level = (0..level.len() - 1)
            .map(|j| {
                let i = first + j;
                let gap = knots.get(i + p + 1).copied().unwrap_or(0.0)
                    - knots.get(i + k).copied().unwrap_or(0.0);
                if gap <= 0.0 {
                    return [0.0; 4];
                }
                let f = scale / gap;
                let (q0, q1) = (level[j], level[j + 1]);
                [
                    f * (q1[0] - q0[0]),
                    f * (q1[1] - q0[1]),
                    f * (q1[2] - q0[2]),
                    f * (q1[3] - q0[3]),
                ]
            })
            .collect();
        *slot = level.iter().fold((0.0_f64, 0.0_f64), |(a, w), q| {
            (
                a.max(Vec3::new(q[0], q[1], q[2]).length()),
                w.max(q[3].abs()),
            )
        });
    }
    let [(a1, w1), (a2, w2), (a3, w3)] = most;
    let c1 = far.madd(w1, a1) / least;
    let c2 = far.madd(w2, (2.0 * c1).madd(w1, a2)) / least;
    let c3 = far.madd(w3, (3.0 * c1).madd(w2, (3.0 * c2).madd(w1, a3))) / least;
    [c1, c2, c3]
        .iter()
        .all(|c| c.is_finite())
        .then_some([c1, c2, c3])
}

/// The derivative of `curve` at `t`, at the speed its parameter runs: a
/// line's from start to end over `[0, 1]`, a circle's or an ellipse's in
/// its angle.
fn curve_velocity(curve: &EdgeCurve, start_pos: Point3, end_pos: Point3, t: f64) -> Vec3 {
    let zero = Vec3::new(0.0, 0.0, 0.0);
    match curve {
        EdgeCurve::Line => end_pos - start_pos,
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
            conic_axes(curve).map_or(zero, |(au, bv)| bv * t.cos() - au * t.sin())
        }
        EdgeCurve::NurbsCurve(n) => n.derivatives(t, 1).get(1).copied().unwrap_or(zero),
    }
}

/// Bounds on the lengths of the first three derivatives of a line, a
/// circle or an ellipse; a spline's are read per knot span by
/// [`spline_span_derivative_bounds`].
fn curve_derivative_bounds(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
) -> Option<[f64; 3]> {
    match curve {
        EdgeCurve::Line => Some([(end_pos - start_pos).length(), 0.0, 0.0]),
        EdgeCurve::Circle(c) => Some([c.radius(); 3]),
        EdgeCurve::Ellipse(e) => Some([e.semi_major().max(e.semi_minor()); 3]),
        EdgeCurve::NurbsCurve(_) => None,
    }
}

/// Bounds on the second and third derivatives of `surface`'s signed
/// distance over the ball of radius `reach` about `p`. `None` where the
/// ball reaches the axis, centre or core circle the distance bends about,
/// and for a B-spline surface. The distance is `|x| - r` for `x` the point
/// off a cylinder's axis or a sphere's centre; a cone's is linear in the
/// distance off its axis and along it; a torus's is `|w| - r` for `w` the
/// point's offset from the core circle in its plane through the axis. At
/// distance `d`, `|x|` has a second derivative of `1/d` and a third under
/// `3/d²`; the torus's plane turns about the axis as well.
fn distance_derivative_bounds(surface: &FaceSurface, p: Point3, reach: f64) -> Option<(f64, f64)> {
    let off_axis = |origin: Point3, axis: Vec3| {
        let w = p - origin;
        (w - axis * w.dot(axis)).length()
    };
    let bend = |d: f64| (d > 0.0).then(|| (1.0 / d, 3.0 / (d * d)));
    match surface {
        FaceSurface::Plane { .. } => Some((0.0, 0.0)),
        FaceSurface::Cylinder(c) => bend(off_axis(c.origin(), c.axis()) - reach),
        FaceSurface::Cone(c) => bend(off_axis(c.apex(), c.axis()) - reach),
        FaceSurface::Sphere(s) => bend((p - s.center()).length() - reach),
        FaceSurface::Torus(t) => {
            let rho = off_axis(t.center(), t.z_axis());
            let height = (p - t.center()).dot(t.z_axis());
            let core = (rho - t.major_radius()).hypot(height);
            let ((k2_core, k3_core), (k2_axis, k3_axis)) =
                (bend(core - reach)?, bend(rho - reach)?);
            Some((
                k2_core + k2_axis,
                (3.0 * k2_core).madd(k2_axis, k3_core + k3_axis),
            ))
        }
        FaceSurface::Nurbs(_) => None,
    }
}

/// Points where a curve passes from one side of a surface to the other,
/// bisected onto the surface. A flip where the projection jumps to another
/// part of the surface lands off it and is dropped.
fn find_side_flips(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    surface: &FaceSurface,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    let side = |t: f64| -> Option<f64> {
        let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        let (u, v) = surface.project_point(pt)?;
        let on = surface.evaluate(u, v)?;
        Some((pt - on).dot(surface.normal(u, v)))
    };
    let mut out = Vec::new();
    let mut prev: Option<(f64, f64)> = None;
    for i in 0..=N_SAMPLES {
        let t = (t1 - t0).madd(i as f64 / N_SAMPLES as f64, t0);
        let Some(s) = side(t) else {
            prev = None;
            continue;
        };
        if let Some((t_prev, s_prev)) = prev
            && (s_prev > 0.0) != (s > 0.0)
            && s_prev != 0.0
            && s != 0.0
        {
            let (mut lo, mut hi, mut s_lo) = (t_prev, t, s_prev);
            for _ in 0..60 {
                let mid = f64::midpoint(lo, hi);
                let Some(s_mid) = side(mid) else { break };
                if (s_mid > 0.0) == (s_lo > 0.0) {
                    lo = mid;
                    s_lo = s_mid;
                } else {
                    hi = mid;
                }
            }
            let tc = f64::midpoint(lo, hi);
            let pc = curve.evaluate_with_endpoints(tc, start_pos, end_pos);
            if distance_to_surface(pc, surface) < tol.linear {
                out.push((tc, pc));
            }
        }
        prev = Some((t, s));
    }
    out
}

/// Find crossings by sampling a signed distance function and detecting sign changes.
fn find_crossings_by_sampling(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    signed_dist: &dyn Fn(Point3) -> f64,
    tol_linear: f64,
) -> Vec<(f64, Point3)> {
    let n = N_SAMPLES;
    let mut crossings = Vec::new();

    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = t0 + (t1 - t0) * (i as f64 / n as f64);
        let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        let sd = signed_dist(pt);
        samples.push((t, sd));
    }

    for i in 0..n {
        let (t_a, sd_a) = samples[i];
        let (t_b, sd_b) = samples[i + 1];

        if sd_a * sd_b < 0.0 {
            let mut lo = t_a;
            let mut hi = t_b;
            let mut sd_lo = sd_a;

            for _ in 0..30 {
                let mid = f64::midpoint(lo, hi);
                let pt_mid = curve.evaluate_with_endpoints(mid, start_pos, end_pos);
                let sd_mid = signed_dist(pt_mid);

                if sd_mid * sd_lo < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    sd_lo = sd_mid;
                }
            }

            let t = f64::midpoint(lo, hi);
            let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
            crossings.push((t, pt));
        }
        // Tangent contact: minimum approaches zero without sign change
        else if sd_a.abs() < 4.0 * tol_linear || sd_b.abs() < 4.0 * tol_linear {
            let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
            let mut lo = t_a;
            let mut hi = t_b;
            for _ in 0..30 {
                let m1 = hi - phi * (hi - lo);
                let m2 = lo + phi * (hi - lo);
                let d1 = signed_dist(curve.evaluate_with_endpoints(m1, start_pos, end_pos)).abs();
                let d2 = signed_dist(curve.evaluate_with_endpoints(m2, start_pos, end_pos)).abs();
                if d1 < d2 {
                    hi = m2;
                } else {
                    lo = m1;
                }
            }
            let t_min = f64::midpoint(lo, hi);
            let pt_min = curve.evaluate_with_endpoints(t_min, start_pos, end_pos);
            let d_min = signed_dist(pt_min).abs();
            if d_min < tol_linear {
                let is_dup = crossings.iter().any(|&(ct, _): &(f64, Point3)| {
                    (t_min - ct).abs() < ((t1 - t0) / (n as f64) * 2.0).abs()
                });
                if !is_dup {
                    crossings.push((t_min, pt_min));
                }
            }
        }
    }

    crossings
}

/// Every threshold the crossing search compares a distance against (1, 2
/// and 4 tolerances) is at most this many tolerances, and its one relative
/// comparison (a midpoint nearer than its neighbours) counts only with the
/// midpoint under 2 tolerances. So past it a lower bound on the distance
/// reads every comparison as the distance itself does.
const NEAR_LIMIT: f64 = 4.0;
const _: () = assert!(2.0 <= NEAR_LIMIT);

/// [`distance_to_surface`], or a lower bound on it once that exceeds
/// `NEAR_LIMIT` tolerances: the distance to the surface's `hull`, a box
/// holding the whole surface.
fn distance_to_surface_near(
    pt: Point3,
    surface: &FaceSurface,
    hull: Option<Aabb3>,
    tol: Tolerance,
) -> f64 {
    if let Some(h) = hull {
        let gap = |x: f64, lo: f64, hi: f64| (lo - x).max(x - hi).max(0.0);
        let off = gap(pt.x(), h.min.x(), h.max.x())
            .hypot(gap(pt.y(), h.min.y(), h.max.y()))
            .hypot(gap(pt.z(), h.min.z(), h.max.z()));
        if off > NEAR_LIMIT * tol.linear {
            return off;
        }
    }
    distance_to_surface(pt, surface)
}

/// Compute distance from point to surface.
fn distance_to_surface(pt: Point3, surface: &FaceSurface) -> f64 {
    // A cylinder's and a sphere's distances have closed forms: the point's
    // distance from the axis or the centre less the radius. They equal the
    // projection's foot distance without its trigonometry, and the scans
    // here ask for them at every sample of every edge near the face.
    match surface {
        FaceSurface::Plane { normal, d } => {
            return (pt.x() * normal.x() + pt.y() * normal.y() + pt.z() * normal.z() - d).abs();
        }
        FaceSurface::Cylinder(c) => {
            let w = pt - c.origin();
            let radial = w - c.axis() * w.dot(c.axis());
            return (radial.length() - c.radius()).abs();
        }
        FaceSurface::Sphere(sp) => {
            return ((pt - sp.center()).length() - sp.radius()).abs();
        }
        FaceSurface::Cone(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => {}
    }
    if let Some((u, v)) = surface.project_point(pt) {
        if let Some(surf_pt) = surface.evaluate(u, v) {
            (pt - surf_pt).length()
        } else {
            f64::MAX
        }
    } else {
        f64::MAX
    }
}

/// Refine a crossing between two parameter values using ternary search.
fn refine_crossing(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t_lo: f64,
    t_hi: f64,
    surface: &FaceSurface,
    _tol: Tolerance,
) -> (f64, Point3) {
    let mut lo = t_lo;
    let mut hi = t_hi;
    let (mut lo_moved, mut hi_moved) = (false, false);

    for _ in 0..30 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let d1 = distance_to_surface(
            curve.evaluate_with_endpoints(m1, start_pos, end_pos),
            surface,
        );
        let d2 = distance_to_surface(
            curve.evaluate_with_endpoints(m2, start_pos, end_pos),
            surface,
        );
        if d1 < d2 {
            hi = m2;
            hi_moved = true;
        } else {
            lo = m1;
            lo_moved = true;
        }
    }

    // A search that never left one bracket end closed on that end without
    // reaching it: a crossing at the edge's own vertex would stop a few 1e-7
    // short of it and outlive the endpoint-contact window as a sliver pave.
    // A search that left both ends found its point inside, where a nearer
    // end is another contact (a rib touching a rim a millimetre from the
    // rim edge's vertex).
    let at = |t: f64| {
        let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        (distance_to_surface(pt, surface), t, pt)
    };
    let mid = at(f64::midpoint(lo, hi));
    let pinned = match (lo_moved, hi_moved) {
        (false, true) => Some(at(t_lo)),
        (true, false) => Some(at(t_hi)),
        _ => None,
    };
    let best = pinned.filter(|end| end.0 <= mid.0).unwrap_or(mid);
    (best.1, best.2)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::EdgeCurve;

    /// The closed-form distances to a cylinder and a sphere equal the
    /// projection's foot distance they replace, for surfaces placed off the
    /// axes and points on them and moved off them inward and outward.
    #[test]
    fn closed_form_distances_match_the_projection() {
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
        use brepkit_math::vec::Vec3;
        let foot = |p: Point3, s: &FaceSurface| {
            let (u, v) = s.project_point(p).unwrap();
            (p - s.evaluate(u, v).unwrap()).length()
        };
        let surfaces = [
            FaceSurface::Cylinder(
                CylindricalSurface::new(Point3::new(1.0, -2.0, 3.0), Vec3::new(1.0, 2.0, 2.0), 1.5)
                    .unwrap(),
            ),
            FaceSurface::Sphere(SphericalSurface::new(Point3::new(-4.0, 0.5, 2.0), 2.25).unwrap()),
        ];
        for s in &surfaces {
            // Points on each surface, then moved off it along its normal by
            // amounts inside it, near it and out past it.
            for k in 0..30_u32 {
                let t = f64::from(k);
                let (u, v) = (0.7 * t, 0.4 * t - 4.0);
                let on = s.evaluate(u, v.clamp(-1.4, 1.4)).unwrap();
                let n = s.normal(u, v.clamp(-1.4, 1.4));
                let offset = [0.0, 1e-9, -1e-6, -0.8, 0.5, 6.0][usize::try_from(k % 6).unwrap()];
                let p = on + n * offset;
                let (closed, projected) = (distance_to_surface(p, s), foot(p, s));
                assert!(
                    (closed - projected).abs() < 1e-9,
                    "{}: {closed} vs {projected}",
                    s.type_tag()
                );
            }
        }
    }

    /// A cylindrical band `z` in `[0, 6]` around the `z` axis at radius 2,
    /// seamed at angle 0, with a window between 60 and 120 degrees and `z` 2
    /// to 4 when `window` is set; and the full circle at `z` = 3 whose vertex
    /// sits on the seam. Whether that circle is recorded as lying in the band.
    fn circle_recorded_in_band(window: bool) -> bool {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let mut topo = Topology::new();
        let z_axis = Vec3::new(0.0, 0.0, 1.0);
        let at = |deg: f64, z: f64| {
            let a = deg.to_radians();
            Point3::new(2.0 * a.cos(), 2.0 * a.sin(), z)
        };
        let circle = |z: f64| {
            EdgeCurve::Circle(Circle3D::new(Point3::new(0.0, 0.0, z), z_axis, 2.0).unwrap())
        };
        let (v_bot, v_top) = (
            topo.add_vertex(Vertex::new(at(0.0, 0.0), 1e-7)),
            topo.add_vertex(Vertex::new(at(0.0, 6.0), 1e-7)),
        );
        let e_bot = topo.add_edge(Edge::new(v_bot, v_bot, circle(0.0)));
        let e_top = topo.add_edge(Edge::new(v_top, v_top, circle(6.0)));
        let e_seam = topo.add_edge(Edge::new(v_bot, v_top, EdgeCurve::Line));
        let outer = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_bot, true),
                    OrientedEdge::new(e_seam, true),
                    OrientedEdge::new(e_top, false),
                    OrientedEdge::new(e_seam, false),
                ],
                true,
            )
            .unwrap(),
        );
        let mut holes = Vec::new();
        if window {
            let corner = [at(60.0, 2.0), at(120.0, 2.0), at(120.0, 4.0), at(60.0, 4.0)];
            let v = corner.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let low = topo.add_edge(Edge::new(v[0], v[1], circle(2.0)));
            let right = topo.add_edge(Edge::new(v[1], v[2], EdgeCurve::Line));
            let high = topo.add_edge(Edge::new(v[3], v[2], circle(4.0)));
            let left = topo.add_edge(Edge::new(v[0], v[3], EdgeCurve::Line));
            holes.push(
                topo.add_wire(
                    Wire::new(
                        vec![
                            OrientedEdge::new(left, true),
                            OrientedEdge::new(high, true),
                            OrientedEdge::new(right, false),
                            OrientedEdge::new(low, false),
                        ],
                        true,
                    )
                    .unwrap(),
                ),
            );
        }
        let surface = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z_axis, 2.0).unwrap();
        let band = topo.add_face(Face::new(outer, holes, FaceSurface::Cylinder(surface)));
        let v_ring = topo.add_vertex(Vertex::new(at(0.0, 3.0), 1e-7));
        let ring = topo.add_edge(Edge::new(v_ring, v_ring, circle(3.0)));
        let mut arena = GfaArena::new();
        check_edge_face_pairs(
            &mut topo,
            &[ring],
            (&[], &[]),
            &[band],
            &[HashSet::new()],
            Tolerance::new(),
            &mut arena,
        )
        .unwrap();
        arena.interference.ef.iter().any(|interference| {
            matches!(
                interference,
                Interference::EF { edge, face, parameter: None, .. } if *edge == ring && *face == band
            )
        })
    }

    /// A rim circle through a plane holding its centre crosses it twice,
    /// exactly, though a sample of the circle lands on the plane.
    #[test]
    fn a_rim_crosses_a_plane_through_its_centre_twice_exactly() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::Vec3;
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let start = circle.evaluate(0.0);
        let crossings = find_edge_plane_crossings(
            &EdgeCurve::Circle(circle),
            start,
            start,
            0.0,
            std::f64::consts::TAU,
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
            Tolerance::new(),
        );
        assert_eq!(crossings.len(), 2, "{crossings:?}");
        for (_, p) in &crossings {
            assert!(
                p.x().abs() < 1e-12 && (p.y().abs() - 3.0).abs() < 1e-12,
                "{p:?}"
            );
        }
    }

    /// A rim circle touching a plane crosses it once, at the touch point.
    #[test]
    fn a_rim_touching_a_plane_crosses_it_once() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::Vec3;
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let start = circle.evaluate(0.0);
        let crossings = find_edge_plane_crossings(
            &EdgeCurve::Circle(circle),
            start,
            start,
            0.0,
            std::f64::consts::TAU,
            Vec3::new(0.6, 0.8, 0.0),
            3.0 * (1.0 - 2e-16),
            Tolerance::new(),
        );
        assert_eq!(crossings.len(), 1, "{crossings:?}");
        let p = crossings[0].1;
        assert!((p - Point3::new(1.8, 2.4, 0.0)).length() < 1e-7, "{p:?}");
    }

    /// A rim touching a plane on the far side of its angle's phase crosses it
    /// at that touch, not at the opposite extreme.
    #[test]
    fn a_rim_touching_a_plane_behind_it_crosses_it_there() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::Vec3;
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let start = circle.evaluate(0.0);
        let crossings = find_edge_plane_crossings(
            &EdgeCurve::Circle(circle),
            start,
            start,
            0.0,
            std::f64::consts::TAU,
            Vec3::new(0.6, 0.8, 0.0),
            -3.0 * (1.0 - 2e-16),
            Tolerance::new(),
        );
        assert_eq!(crossings.len(), 1, "{crossings:?}");
        let p = crossings[0].1;
        assert!((p - Point3::new(-1.8, -2.4, 0.0)).length() < 1e-7, "{p:?}");
    }

    /// A flat leaning 5e-8 off a bore's axis meets the bore along an ellipse
    /// whose centre lies 1e8 along the axis. Its crossings of a plane square
    /// to the axis lie on that plane, and on the bore and the flat.
    #[test]
    fn a_long_ellipse_crosses_a_plane_on_the_plane() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::vec::Vec3;
        let tilt = 5e-8;
        let flat = Vec3::new(0.0, 1.0, tilt).normalize().unwrap();
        let along = Vec3::new(0.0, -tilt, 1.0).normalize().unwrap();
        let ellipse = Ellipse3D::new_with_ref(
            Point3::new(0.0, 0.0, 5.0 / tilt),
            flat,
            10.0 / flat.z(),
            10.0,
            along,
        )
        .unwrap();
        let start = ellipse.evaluate(0.0);
        for (normal, d) in [
            (Vec3::new(0.0, 0.0, 1.0), 0.0),
            (Vec3::new(0.0, 0.0, 1.0), 60.0),
            (Vec3::new(0.0, 0.0, -1.0), -100.0),
        ] {
            let crossings = find_edge_plane_crossings(
                &EdgeCurve::Ellipse(ellipse.clone()),
                start,
                start,
                0.0,
                std::f64::consts::TAU,
                normal,
                d,
                Tolerance::new(),
            );
            assert_eq!(crossings.len(), 2, "{crossings:?}");
            for (_, p) in crossings {
                let off_plane = normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d;
                let off_bore = p.x().hypot(p.y()) - 10.0;
                let off_flat = flat.dot(Vec3::new(p.x(), p.y(), p.z())) - flat.y() * 5.0;
                assert!(
                    off_plane.abs() < 1e-12 && off_bore.abs() < 1e-7 && off_flat.abs() < 1e-7,
                    "{p:?}: off the plane {off_plane:.2e}, the bore {off_bore:.2e}, the flat {off_flat:.2e}"
                );
            }
        }
    }

    /// A straight edge through a cylinder band crosses it where no sample of
    /// the edge lands on the surface; both crossings take vertices on it.
    #[test]
    fn a_line_through_a_band_crosses_it_between_samples() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let mut topo = Topology::new();
        let z_axis = Vec3::new(0.0, 0.0, 1.0);
        let circle = |z: f64| {
            EdgeCurve::Circle(Circle3D::new(Point3::new(0.0, 0.0, z), z_axis, 2.0).unwrap())
        };
        let v_bot = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 0.0), 1e-7));
        let v_top = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 6.0), 1e-7));
        let e_bot = topo.add_edge(Edge::new(v_bot, v_bot, circle(0.0)));
        let e_top = topo.add_edge(Edge::new(v_top, v_top, circle(6.0)));
        let e_seam = topo.add_edge(Edge::new(v_bot, v_top, EdgeCurve::Line));
        let outer = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_bot, true),
                    OrientedEdge::new(e_seam, true),
                    OrientedEdge::new(e_top, false),
                    OrientedEdge::new(e_seam, false),
                ],
                true,
            )
            .unwrap(),
        );
        let surface = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z_axis, 2.0).unwrap();
        let band = topo.add_face(Face::new(outer, vec![], FaceSurface::Cylinder(surface)));
        let a = topo.add_vertex(Vertex::new(Point3::new(-3.0, 0.37, 1.1), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(3.0, 0.41, 4.9), 1e-7));
        let line = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let mut arena = GfaArena::new();
        check_edge_face_pairs(
            &mut topo,
            &[line],
            (&[], &[]),
            &[band],
            &[HashSet::new()],
            Tolerance::new(),
            &mut arena,
        )
        .unwrap();
        let radii: Vec<f64> = arena
            .interference
            .ef
            .iter()
            .filter_map(|i| match i {
                Interference::EF {
                    new_vertex: Some(v),
                    ..
                } => topo
                    .vertex(*v)
                    .ok()
                    .map(|v| v.point().x().hypot(v.point().y())),
                _ => None,
            })
            .collect();
        assert_eq!(radii.len(), 2, "{radii:?}");
        assert!(radii.iter().all(|r| (r - 2.0).abs() < 1e-9), "{radii:?}");
    }

    /// A circle on a cylinder face splits the face along it, unless it runs
    /// through a window in the face between points that lie on the face.
    #[test]
    fn a_circle_through_a_window_in_a_band_is_not_in_the_band() {
        assert!(circle_recorded_in_band(false));
        assert!(!circle_recorded_in_band(true));
    }

    /// The hull bound spares projections only where the distance is past
    /// every threshold the search compares, so it finds exactly the crossings
    /// the plain search does, and a line whose box misses the grown hull has
    /// none. Seeded patches (bilinear and rational, up to cubic) against
    /// lines through them, lines skimming them at up to 8 tolerances, and
    /// lines running just outside a face of their hull.
    #[test]
    fn hull_bound_keeps_every_nurbs_crossing() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_math::vec::Vec3;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            let f = (state >> 11) as f64 / (1_u64 << 53) as f64;
            f
        };
        let tol = Tolerance::new();
        let (mut cases, mut crossing_cases, mut near_cases) = (0, 0, 0);
        for patch in 0..24_u32 {
            let degree = 1 + (patch as usize % 3);
            let n = degree + 2;
            let mut knots = vec![0.0; degree + 1];
            knots.push(0.5);
            knots.extend(vec![1.0; degree + 1]);
            let rational = patch % 2 == 1;
            let (mut points, mut weights) = (Vec::new(), Vec::new());
            for i in 0..n {
                let (mut row, mut wrow) = (Vec::new(), Vec::new());
                for j in 0..n {
                    #[allow(clippy::cast_precision_loss)]
                    row.push(Point3::new(
                        i as f64 + 0.6 * next() - 0.3,
                        j as f64 + 0.6 * next() - 0.3,
                        2.0 * next() - 1.0,
                    ));
                    wrow.push(if rational { 0.2 + 4.8 * next() } else { 1.0 });
                }
                points.push(row);
                weights.push(wrow);
            }
            let patch =
                NurbsSurface::new(degree, degree, knots.clone(), knots, points, weights).unwrap();
            let hull = patch.aabb();
            let surface = FaceSurface::Nurbs(patch.clone());
            for kind in 0..9_u32 {
                let (u, v) = (next(), next());
                let at = patch.evaluate(u, v);
                let normal = patch.normal(u, v).unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                let dir = Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                let gap = 8.0 * next() * tol.linear;
                let (a, b) = match kind % 3 {
                    0 => (at - dir * 2.0, at + dir * 3.0),
                    1 => {
                        let along = (dir - normal * dir.dot(normal)).normalize().unwrap_or(dir);
                        let q = at + normal * gap;
                        (q - along * 0.5, q + along * 0.5)
                    }
                    _ => {
                        let q = Point3::new(at.x(), at.y(), hull.max.z() + gap);
                        let flat = Vec3::new(dir.x(), dir.y(), 0.0);
                        (q - flat * 2.0, q + flat * 2.0)
                    }
                };
                let plain = find_edge_surface_crossings(
                    &EdgeCurve::Line,
                    a,
                    b,
                    0.0,
                    1.0,
                    &surface,
                    None,
                    tol,
                );
                let bounded = find_edge_surface_crossings(
                    &EdgeCurve::Line,
                    a,
                    b,
                    0.0,
                    1.0,
                    &surface,
                    Some(hull),
                    tol,
                );
                assert_eq!(plain, bounded, "line {a:?} -> {b:?}");
                let edge_box = Aabb3::try_from_points([a, b]).unwrap();
                if !edge_box.intersects(hull.expanded(tol.linear)) {
                    assert!(plain.is_empty(), "a line clear of the hull crossed it");
                }
                for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let p = a + (b - a) * t;
                    let exact = distance_to_surface(p, &surface);
                    let near = distance_to_surface_near(p, &surface, Some(hull), tol);
                    assert_eq!(exact < tol.linear, near < tol.linear);
                    if exact < NEAR_LIMIT * tol.linear {
                        near_cases += 1;
                    }
                }
                cases += 1;
                if !plain.is_empty() {
                    crossing_cases += 1;
                }
            }
        }
        assert_eq!(cases, 216);
        assert!(crossing_cases >= 20, "only {crossing_cases} lines crossed");
        assert!(near_cases >= 20, "only {near_cases} samples came near");

        // A vertical line down through the saddle `z = u + v - 2uv` near the
        // corner (1, 0, 1), where the patch meets its hull's top: its sample
        // above the crossing lies 3.5 tolerances over the hull but some 6e-4
        // off the patch. Read by the hull alone it would trigger the tangent
        // search the plain distance does not.
        let saddle = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 1.0)],
                vec![Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();
        let hull = saddle.aabb();
        let surface = FaceSurface::Nurbs(saddle);
        let (y, step) = (0.0005, 0.01);
        let x = (0.999 - y) / 2.0f64.madd(-y, 1.0);
        let z_top = 3.5f64.madd(tol.linear, 1.0) + step;
        let (a, b) = (
            Point3::new(x, y, 64.0f64.madd(-step, z_top)),
            Point3::new(x, y, z_top),
        );
        let plain =
            find_edge_surface_crossings(&EdgeCurve::Line, a, b, 0.0, 1.0, &surface, None, tol);
        let bounded = find_edge_surface_crossings(
            &EdgeCurve::Line,
            a,
            b,
            0.0,
            1.0,
            &surface,
            Some(hull),
            tol,
        );
        assert_eq!(plain, bounded);
    }

    /// An envelope's leaning corner edge reaching a corner cylinder at its
    /// own end vertex crosses it exactly there, so the contact drops as the
    /// vertex's; a search that stops short of the end left a sliver pave.
    #[test]
    fn a_crossing_at_an_edge_end_lands_on_the_end() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        let surface = FaceSurface::Cylinder(
            CylindricalSurface::new(
                Point3::new(-41.0, -20.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                3.03,
            )
            .unwrap(),
        );
        let (start, end) = (
            Point3::new(-38.0, -20.03, 0.0),
            Point3::new(-41.0, -23.03, 6.0),
        );
        let crossings = find_edge_surface_crossings(
            &EdgeCurve::Line,
            start,
            end,
            0.0,
            1.0,
            &surface,
            None,
            Tolerance::new(),
        );
        let last = crossings.last().expect("the end crossing");
        assert!((last.1 - end).length() < 1e-12, "{crossings:?}");
    }

    /// Two roots in adjacent sample intervals, a fraction of a sample step
    /// apart, are two crossings.
    #[test]
    fn two_nearby_crossings_of_a_nurbs_patch_both_land() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::nurbs::surface::NurbsSurface;
        // z = 100 (t - 0.404)(t - 0.421) along x = t.
        let parabola = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 17.0084),
                Point3::new(0.5, 0.0, -24.2416),
                Point3::new(1.0, 0.0, 34.5084),
            ],
            vec![1.0; 3],
        )
        .unwrap();
        let patch = FaceSurface::Nurbs(
            NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                vec![
                    vec![Point3::new(-1.0, -1.0, 0.0), Point3::new(-1.0, 1.0, 0.0)],
                    vec![Point3::new(2.0, -1.0, 0.0), Point3::new(2.0, 1.0, 0.0)],
                ],
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            )
            .unwrap(),
        );
        let (start, end) = (parabola.evaluate(0.0), parabola.evaluate(1.0));
        let crossings = find_edge_surface_crossings(
            &EdgeCurve::NurbsCurve(parabola),
            start,
            end,
            0.0,
            1.0,
            &patch,
            None,
            Tolerance::new(),
        );
        assert_eq!(crossings.len(), 2, "{crossings:?}");
        assert!((crossings[0].1.x() - 0.404).abs() < 1e-6, "{crossings:?}");
        assert!((crossings[1].1.x() - 0.421).abs() < 1e-6, "{crossings:?}");
    }

    /// Two crossings in one run of samples near a ball, 90 tolerances apart:
    /// a line at `y = sqrt(0.001² - (4.5e-6)²)` crosses a ball of radius
    /// 0.001 at `x = ±4.5e-6`.
    #[test]
    fn two_crossings_of_a_small_ball_90_tolerances_apart_are_both_kept() {
        use brepkit_math::surfaces::SphericalSurface;
        let (radius, x) = (1e-3_f64, 4.5e-6_f64);
        let y = radius.madd(radius, -x * x).sqrt();
        let surface =
            FaceSurface::Sphere(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), radius).unwrap());
        let (a, b) = (Point3::new(-5e-5, y, 0.0), Point3::new(5e-5, y, 0.0));
        let crossings = find_edge_surface_crossings(
            &EdgeCurve::Line,
            a,
            b,
            0.0,
            1.0,
            &surface,
            None,
            Tolerance::new(),
        );
        let xs: Vec<f64> = crossings.iter().map(|(_, p)| p.x()).collect();
        assert_eq!(xs.len(), 2, "crossings at {xs:?}");
        for (got, want) in xs.iter().zip([-x, x]) {
            assert!(
                (got - want).abs() < 1e-10,
                "crossing at {got}, expected {want}"
            );
        }
    }

    /// A line within the tolerance of a ball over a run of samples that
    /// passes through the ball's surface twice in that run: radius
    /// `2002001 s` and the line at `y = 2002000 s`, with `s = 2^-24`, cross
    /// at `x = ±2001 s`, about 2385 tolerances apart.
    #[test]
    fn a_shallow_line_through_a_ball_keeps_both_crossings() {
        use brepkit_math::surfaces::SphericalSurface;
        let s = 2.0_f64.powi(-24);
        let surface = FaceSurface::Sphere(
            SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2_002_001.0 * s).unwrap(),
        );
        let (a, b) = (
            Point3::new(-16.0 * 2001.0 * s, 2_002_000.0 * s, 0.0),
            Point3::new(16.0 * 2001.0 * s, 2_002_000.0 * s, 0.0),
        );
        let tol = Tolerance::new();
        let crossings =
            find_edge_surface_crossings(&EdgeCurve::Line, a, b, 0.0, 1.0, &surface, None, tol);
        let xs: Vec<f64> = crossings.iter().map(|(_, p)| p.x()).collect();
        assert_eq!(xs.len(), 2, "crossings at {xs:?}");
        for (x, want) in xs.iter().zip([-2001.0 * s, 2001.0 * s]) {
            assert!((x - want).abs() < 1e-9, "crossing at {x}, expected {want}");
        }
    }

    /// Where the edge `(x, 0, k (x - r0)(x - r1)...)`, for `x` from -1 to 1
    /// and its height a polynomial with the roots `roots`, meets the wall
    /// of a rod of radius 1 along `x`, from `x = -2` to `x = 2`, whose top
    /// lies at `z = 0`: the `x` of the vertices phase EF makes on the edge.
    /// Along `y = 0` the edge's height is its signed distance from the wall.
    fn contacts_with_a_wall(k: f64, roots: &[f64]) -> Vec<f64> {
        use brepkit_math::nurbs::curve::NurbsCurve;
        // The height in powers of `u`, with `x = 2u - 1`, then as a Bézier
        // curve's control heights.
        let mut power = vec![k];
        for &r in roots {
            let mut next = vec![0.0; power.len() + 1];
            for (i, &c) in power.iter().enumerate() {
                next[i] += c * (-1.0 - r);
                next[i + 1] += c * 2.0;
            }
            power = next;
        }
        let n = roots.len();
        let choose = |a: usize, b: usize| -> f64 {
            (0..b).fold(1.0, |acc, i| acc * (a - i) as f64 / (i + 1) as f64)
        };
        let points: Vec<Point3> = (0..=n)
            .map(|j| {
                let z = (0..=j)
                    .map(|i| choose(j, i) / choose(n, i) * power[i])
                    .sum();
                Point3::new(2.0f64.madd(j as f64 / n as f64, -1.0), 0.0, z)
            })
            .collect();
        let knots = [vec![0.0; n + 1], vec![1.0; n + 1]].concat();
        contacts_of_a_spline_with_a_wall(
            &NurbsCurve::new(n, knots, points, vec![1.0; n + 1]).unwrap(),
            1.0,
        )
    }

    /// The `x` of the vertices phase EF makes on an edge along `curve`,
    /// against the wall of a rod like that of [`contacts_with_a_wall`], of
    /// radius `radius`.
    fn contacts_of_a_spline_with_a_wall(
        curve: &brepkit_math::nurbs::curve::NurbsCurve,
        radius: f64,
    ) -> Vec<f64> {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let (d0, d1) = curve.domain();
        let (first, last) = (curve.evaluate(d0), curve.evaluate(d1));
        let curve = curve.clone();

        let mut topo = Topology::new();
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let wall = CylindricalSurface::with_ref_dir(
            Point3::new(-2.0, 0.0, -radius),
            x_axis,
            radius,
            Vec3::new(0.0, 0.0, -1.0),
        )
        .unwrap();
        let rim = |topo: &mut Topology, x: f64| {
            let v = topo.add_vertex(Vertex::new(Point3::new(x, 0.0, -2.0 * radius), 1e-7));
            let circle = Circle3D::new(Point3::new(x, 0.0, -radius), x_axis, radius).unwrap();
            (v, topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle))))
        };
        let ((v_lo, e_lo), (v_hi, e_hi)) = (rim(&mut topo, -2.0), rim(&mut topo, 2.0));
        let seam = topo.add_edge(Edge::new(v_lo, v_hi, EdgeCurve::Line));
        let outer = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_lo, true),
                    OrientedEdge::new(seam, true),
                    OrientedEdge::new(e_hi, false),
                    OrientedEdge::new(seam, false),
                ],
                true,
            )
            .unwrap(),
        );
        let face = topo.add_face(Face::new(outer, vec![], FaceSurface::Cylinder(wall)));
        let (a, b) = (
            topo.add_vertex(Vertex::new(first, 1e-7)),
            topo.add_vertex(Vertex::new(last, 1e-7)),
        );
        let edge = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve)));
        let mut arena = GfaArena::new();
        check_edge_face_pairs(
            &mut topo,
            &[edge],
            (&[], &[]),
            &[face],
            &[HashSet::new()],
            Tolerance::new(),
            &mut arena,
        )
        .unwrap();
        let mut xs: Vec<f64> = arena
            .interference
            .ef
            .iter()
            .filter_map(|i| match i {
                Interference::EF {
                    new_vertex: Some(v),
                    ..
                } => topo.vertex(*v).ok().map(|v| v.point().x()),
                _ => None,
            })
            .collect();
        xs.sort_by(f64::total_cmp);
        xs
    }

    /// An edge whose samples lie within the tolerance of a wall from
    /// `x = -0.62` to `x = 0.48`, touching it at `x = -0.3` and crossing it
    /// at `x = 0.3`: each takes a vertex.
    #[test]
    fn a_run_that_touches_and_crosses_a_wall_keeps_both() {
        let xs = contacts_with_a_wall(1e-6, &[-0.3, -0.3, 0.3]);
        assert_eq!(xs.len(), 2, "contacts at {xs:?}");
        for (x, want) in xs.iter().zip([-0.3, 0.3]) {
            assert!((x - want).abs() < 1e-9, "contact at {x}, expected {want}");
        }
    }

    /// An edge whose samples lie within the tolerance of a wall from
    /// `x = -0.65` to `x = 0.65`, touching it at `x = -0.3` and `x = 0.3`
    /// and 8.1e-9 off it between: each touch takes a vertex.
    #[test]
    fn a_run_that_touches_a_wall_twice_keeps_both_touches() {
        let xs = contacts_with_a_wall(1e-6, &[-0.3, -0.3, 0.3, 0.3]);
        assert_eq!(xs.len(), 2, "contacts at {xs:?}");
        for (x, want) in xs.iter().zip([-0.3, 0.3]) {
            assert!((x - want).abs() < 1e-9, "contact at {x}, expected {want}");
        }
    }

    /// An edge that touches a wall at `x = -0.3` and crosses it at
    /// `x = -0.28`, its distance 2.37e-9 off the wall between them at
    /// `x = -0.2867`: the touch and the most of the distance both lie
    /// between the samples at `x = -0.3125` and `x = -0.28125`, where the
    /// distance's rate has one sign. Each contact takes a vertex.
    #[test]
    fn a_touch_and_a_most_between_two_samples_keep_the_touch() {
        let xs = contacts_with_a_wall(2e-3, &[-0.3, -0.3, -0.28]);
        assert_eq!(xs.len(), 2, "contacts at {xs:?}");
        for (x, want) in xs.iter().zip([-0.3, -0.28]) {
            assert!((x - want).abs() < 1e-9, "contact at {x}, expected {want}");
        }
    }

    /// A quadratic spline with a double knot at `x = 0`, where it turns
    /// from one parabola to another: it touches the wall of a rod of radius
    /// 1000 at `x = -0.01` and at `x = 0.5`, and is 1e-8 off it at the knot
    /// between, where its rate falls from up to down. Read on the whole
    /// spline, the rate at the knot comes from the span after it, and the
    /// bound then shows no root in the interval before, which holds the
    /// first touch. Each touch takes a vertex.
    #[test]
    fn a_spline_touching_a_wall_before_and_after_a_kink_keeps_both_touches() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let at = |x: f64, z: f64| Point3::new(x, 0.0, z);
        // `1e-4 (x + 0.01)^2` on `[-1, 0]` and `4e-8 (x - 0.5)^2` on `[0, 1]`.
        let spline = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            vec![
                at(-1.0, 9.801e-5),
                at(-0.5, -9.9e-7),
                at(0.0, 1e-8),
                at(0.5, -1e-8),
                at(1.0, 1e-8),
            ],
            vec![1.0; 5],
        )
        .unwrap();
        let xs = contacts_of_a_spline_with_a_wall(&spline, 1000.0);
        assert_eq!(xs.len(), 2, "contacts at {xs:?}");
        for (x, want) in xs.iter().zip([-0.01, 0.5]) {
            assert!((x - want).abs() < 1e-6, "contact at {x}, expected {want}");
        }
    }

    /// A line 1 long along the top of a rod of radius 0.005 lies on the
    /// rod's wall: its distance's rate is zero along it, and the bound on
    /// the rate's second derivative, `3 c1³ / r²`, is 1.2e5. The search
    /// would read 131008 intervals before the distance could change across
    /// each by less than the merge allows; it stops reading past its
    /// budget.
    #[test]
    fn a_line_along_a_thin_rod_stops_at_the_budget() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        let rod = FaceSurface::Cylinder(
            CylindricalSurface::new(
                Point3::new(0.0, 0.0, -0.005),
                Vec3::new(1.0, 0.0, 0.0),
                0.005,
            )
            .unwrap(),
        );
        let (a, b) = (Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0));
        #[allow(clippy::cast_precision_loss)]
        let ts: Vec<f64> = (0..=N_SAMPLES)
            .map(|i| i as f64 / N_SAMPLES as f64)
            .collect();
        let (_, read) = run_contacts_counted(&EdgeCurve::Line, a, b, &ts, &rod, Tolerance::new());
        assert!(read > RUN_INTERVALS, "read {read}: the bound settled");
        assert!(
            read <= RUN_INTERVALS + 2 * (ts.len() + MOST_HALVINGS as usize),
            "read {read}"
        );
    }

    /// A ball's equator turned 1.57 about `x`, against a rod of the ball's
    /// radius along `y`: the circle runs up to 3.2e-7 inside the rod's wall
    /// and touches it at `(1, 0, 0)` only, whether that point falls between
    /// two samples or on one.
    #[test]
    fn a_circle_nearly_tangent_inside_a_wall_touches_it_once() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        let spin = 1.57_f64;
        let circle = Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -spin.sin(), spin.cos()),
            1.0,
        )
        .unwrap();
        let surface = FaceSurface::Cylinder(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0)
                .unwrap(),
        );
        let touch = circle.project(Point3::new(1.0, 0.0, 0.0));
        let curve = EdgeCurve::Circle(circle.clone());
        for (t0, t1) in [(touch - 1.0, touch + 1.5), (touch - 1.5, touch + 1.5)] {
            let crossings = find_edge_surface_crossings(
                &curve,
                circle.evaluate(t0),
                circle.evaluate(t1),
                t0,
                t1,
                &surface,
                None,
                Tolerance::new(),
            );
            assert_eq!(crossings.len(), 1, "crossings {crossings:?}");
            let p = crossings[0].1;
            assert!(
                (p - Point3::new(1.0, 0.0, 0.0)).length() < 1e-9,
                "touch at {p:?}"
            );
        }
    }

    #[test]
    fn sampling_detects_tangent_touch() {
        // Signed distance: parabola touching zero at t=0.5 (exact tangent)
        let curve = EdgeCurve::Line;
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(1.0, 0.0, 0.0);
        let signed_dist = |pt: Point3| -> f64 {
            let t = pt.x();
            (t - 0.5) * (t - 0.5) // minimum = 0 at t=0.5
        };

        let crossings =
            find_crossings_by_sampling(&curve, start, end, 0.0, 1.0, &signed_dist, 1e-7);

        assert!(
            !crossings.is_empty(),
            "tangent touch (minimum=0) should be detected"
        );
        let (t, _) = crossings[0];
        assert!(
            (t - 0.5).abs() < 0.02,
            "tangent point should be near t=0.5, got {t}"
        );
    }

    /// Every point a curve takes over an edge's parameters lies in one of
    /// its piece boxes: a line, a circle's arc, an ellipse's arc past a half
    /// turn, and a rational B-spline's sub-span, sampled densely.
    #[test]
    fn curve_piece_boxes_hold_the_curve() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        use brepkit_math::nurbs::curve::NurbsCurve;
        let n = Vec3::new(0.3, -0.4, 0.866);
        let nurbs = NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 3.0, -1.0),
                Point3::new(4.0, -2.0, 2.0),
                Point3::new(5.0, 1.0, 0.5),
                Point3::new(7.0, 0.0, 3.0),
            ],
            vec![1.0, 0.5, 2.0, 0.8, 1.0],
        )
        .unwrap();
        let (a, b) = (Point3::new(1.0, 2.0, 3.0), Point3::new(-4.0, 0.5, 7.0));
        let cases = [
            (EdgeCurve::Line, a, b, 0.0, 1.0),
            (
                EdgeCurve::Circle(Circle3D::new(a, n, 2.5).unwrap()),
                a,
                a,
                0.4,
                2.9,
            ),
            (
                EdgeCurve::Ellipse(Ellipse3D::new(a, n, 6.0, 1.5).unwrap()),
                a,
                a,
                5.0,
                -0.3,
            ),
            (EdgeCurve::NurbsCurve(nurbs), a, b, 0.25, 0.9),
            // Axes that are neither unit nor square, as a copy through a
            // scaling or shearing transform keeps them.
            (
                EdgeCurve::Circle(
                    Circle3D::with_axes(
                        a,
                        n,
                        2.5,
                        Vec3::new(3.0, 0.0, 0.0),
                        Vec3::new(1.5, 0.5, 0.0),
                    )
                    .unwrap(),
                ),
                a,
                a,
                0.4,
                2.9,
            ),
            (
                EdgeCurve::Ellipse(
                    Ellipse3D::with_axes(
                        a,
                        n,
                        6.0,
                        1.5,
                        Vec3::new(0.0, 4.0, 0.0),
                        Vec3::new(2.0, -3.0, 1.0),
                    )
                    .unwrap(),
                ),
                a,
                a,
                5.0,
                -0.3,
            ),
            // A unit circle, its radius and axes each far from unit: the
            // product of the radius by itself overflows.
            (
                EdgeCurve::Circle(
                    Circle3D::with_axes(
                        a,
                        Vec3::new(0.0, 1.0, 0.0),
                        1e160,
                        Vec3::new(1e-160, 0.0, 0.0),
                        Vec3::new(0.0, 0.0, 1e-160),
                    )
                    .unwrap(),
                ),
                a,
                a,
                0.4,
                2.9,
            ),
        ];
        // Weights below the least normal float give no boxes.
        let tiny = f64::from_bits(1);
        let line = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0)],
            vec![tiny, tiny],
        )
        .unwrap();
        let (s, e) = (Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(curve_piece_boxes(&EdgeCurve::NurbsCurve(line), s, e, 0.0, 1.0).is_none());
        for (curve, start, end, t0, t1) in &cases {
            let boxes = curve_piece_boxes(curve, *start, *end, *t0, *t1).unwrap();
            for k in 0..=2000 {
                let t = t0 + (t1 - t0) * f64::from(k) / 2000.0;
                let p = curve.evaluate_with_endpoints(t, *start, *end);
                assert!(
                    boxes.iter().any(|bx| bx.contains_point(p)),
                    "{curve:?} at {t}: {p:?} in no box"
                );
            }
        }
    }

    /// The radii in `z = 60` at which lines through the ring `245 <= r <=
    /// 249.25` there cross it, each line through the plane at one of
    /// `radii` and at the angle `deg`, its outer circle cut at `cuts`.
    fn ring_crossings(radii: &[f64], deg: f64, cuts: &[f64]) -> Vec<f64> {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::Vec3;
        use brepkit_topology::edge::Edge;
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let mut topo = Topology::new();
        let z_axis = Vec3::new(0.0, 0.0, 1.0);
        let center = Point3::new(0.0, 0.0, 60.0);
        let at = |r: f64, a: f64| Point3::new(r * a.cos(), r * a.sin(), 60.0);
        let mut outer_edges = Vec::new();
        let outer_circle = Circle3D::new(center, z_axis, 249.25).unwrap();
        let ends: Vec<_> = cuts
            .iter()
            .map(|&a| topo.add_vertex(Vertex::new(at(249.25, a), 1e-7)))
            .collect();
        for (k, &v) in ends.iter().enumerate() {
            let next = ends[(k + 1) % ends.len()];
            let arc = Edge::new(v, next, EdgeCurve::Circle(outer_circle.clone()));
            outer_edges.push(OrientedEdge::new(topo.add_edge(arc), true));
        }
        let outer = topo.add_wire(Wire::new(outer_edges, true).unwrap());
        let inner_circle = Circle3D::new(center, z_axis, 245.0).unwrap();
        let v_in = topo.add_vertex(Vertex::new(at(245.0, 0.0), 1e-7));
        let rim = topo.add_edge(Edge::new(v_in, v_in, EdgeCurve::Circle(inner_circle)));
        let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, false)], true).unwrap());
        let ring = topo.add_face(Face::new(
            outer,
            vec![hole],
            FaceSurface::Plane {
                normal: z_axis,
                d: 60.0,
            },
        ));
        let a = deg.to_radians();
        let lines: Vec<_> = radii
            .iter()
            .map(|&r| {
                let p = at(r, a);
                let lean = Vec3::new(0.3, -0.2, 5.0);
                let s = topo.add_vertex(Vertex::new(p - lean, 1e-7));
                let e = topo.add_vertex(Vertex::new(p + lean, 1e-7));
                topo.add_edge(Edge::new(s, e, EdgeCurve::Line))
            })
            .collect();
        let mut arena = GfaArena::new();
        check_edge_face_pairs(
            &mut topo,
            &lines,
            (&[], &[]),
            &[ring],
            &[HashSet::new()],
            Tolerance::new(),
            &mut arena,
        )
        .unwrap();
        arena
            .interference
            .ef
            .iter()
            .filter_map(|i| match i {
                Interference::EF {
                    new_vertex: Some(v),
                    ..
                } => topo
                    .vertex(*v)
                    .ok()
                    .map(|v| v.point().x().hypot(v.point().y())),
                _ => None,
            })
            .collect()
    }

    /// A line through the plane of a ring of radius 249.25 crosses the ring
    /// only where the ring's own circle holds the crossing: not 0.066 or
    /// 0.098 past it, whether the circle is one edge or three, at a sample
    /// of the circle or between two, and up to it from inside.
    #[test]
    fn a_ring_holds_crossings_only_inside_its_circle() {
        for cuts in [vec![0.0], vec![0.0, 2.0, 4.1]] {
            for deg in [0.0, 5.625 / 2.0, 47.3, 200.0] {
                let past = ring_crossings(&[249.316, 249.348, 249.25 + 1e-4], deg, &cuts);
                assert!(past.is_empty(), "cuts {cuts:?}, {deg} deg: {past:?}");
                let held = ring_crossings(&[249.25 - 1e-4, 249.2, 247.0], deg, &cuts);
                assert_eq!(held.len(), 3, "cuts {cuts:?}, {deg} deg: {held:?}");
            }
        }
        let rim = ring_crossings(&[249.25], 47.3, &[0.0]);
        assert_eq!(rim.len(), 1, "{rim:?}");
    }
}
