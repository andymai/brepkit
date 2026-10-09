//! Phase EF: Edge-face intersection detection.
//!
//! For each (edge, face) pair across solids, finds points where the
//! edge crosses or touches the face surface. Records EF interferences
//! and adds extra paves to the edge for later splitting.

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

    check_edge_face_pairs(topo, &edges_a, &faces_b, &face_boundary_edges_b, tol, arena)?;
    check_edge_face_pairs(topo, &edges_b, &faces_a, &face_boundary_edges_a, tol, arena)?;

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
    margin: f64,
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

    for &inner_wid in face.inner_wires() {
        let inner_wire = topo.wire(inner_wid)?;
        let inner_edges: Vec<_> = inner_wire.edges().to_vec();
        for oe in &inner_edges {
            let edge = topo.edge(oe.edge())?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
            let n = N_BOUNDARY_SAMPLES;
            for i in 0..=n {
                let t = t0 + (t1 - t0) * (i as f64 / n as f64);
                all_points.push(edge.curve().evaluate_with_endpoints(t, start_pos, end_pos));
            }
        }
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
            let polygon: Vec<Point2> = outer_points.iter().map(|&p| frame.project(p)).collect();
            return Ok(FaceContainment {
                bbox: bbox.expanded(margin),
                planar: Some(PlanarContainment {
                    frame,
                    polygon,
                    margin,
                }),
            });
        }
        return Ok(FaceContainment {
            bbox: bbox.expanded((diag * 0.5).max(tol.linear * 10.0)),
            planar: None,
        });
    }

    // Curved faces can bulge past their boundary AABB (e.g. a hemisphere
    // bounded by its equator), so expand generously by half the diagonal.
    Ok(FaceContainment {
        bbox: bbox.expanded((diag * 0.5).max(tol.linear * 10.0)),
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

/// Check each edge against each face.
#[allow(clippy::too_many_lines)]
fn check_edge_face_pairs(
    topo: &mut Topology,
    edges: &[EdgeId],
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
    for &eid in edges {
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
            let edge_on_surface = (0..=n_chk).all(|i| {
                let t = t0 + (t1 - t0) * (f64::from(i) / f64::from(n_chk));
                let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
                distance_to_surface_near(pt, surface, hull, tol) < tol.linear
            });
            if edge_on_surface {
                // Through the inside of a cylinder or cone face, the edge
                // bounds a region the two solids share there (a bin wall's
                // top arc on the coaxial air wall of a fillet that reaches
                // past it), and the face is split along it.
                let near = 10.0 * tol.linear;
                let lateral = matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
                    && matches!(curve, EdgeCurve::Line | EdgeCurve::Circle(_));
                if lateral {
                    let trim = trims[face_idx]
                        .get_or_insert_with(|| {
                            crate::classifier::LateralTrim::new(topo, fid)
                                .ok()
                                .flatten()
                        })
                        .as_ref();
                    let at = |k: u32| {
                        let t = (t1 - t0).mul_add(f64::from(k) / f64::from(n_chk), t0);
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
                        let t_at = |k: u32| (t1 - t0).mul_add(f64::from(k) / f64::from(n_chk), t0);
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

            let mut crossings = match surface {
                FaceSurface::Plane { normal, d } => {
                    find_edge_plane_crossings(&curve, start_pos, end_pos, t0, t1, *normal, *d, tol)
                }
                _ => find_edge_surface_crossings(
                    &curve, start_pos, end_pos, t0, t1, surface, hull, tol,
                ),
            };
            // A NURBS edge crossing a cylinder or cone between two samples
            // (an envelope's corner curve through a pocket's floor fillet)
            // shows only as the side of the surface flipping. Its unbounded
            // surface meets far more of the edge than the face does, and this
            // face's containment is a box, so a flip counts only where the
            // face's own wires hold it. A line's crossings are the roots of
            // a quadratic, exact wherever they fall between two samples.
            if matches!(curve, EdgeCurve::NurbsCurve(_) | EdgeCurve::Line)
                && matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
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
                for (t, p) in found {
                    if trim.holds(p, 10.0 * tol.linear)
                        && !crossings
                            .iter()
                            .any(|&(_, cp)| (p - cp).length() < tol.linear * 100.0)
                    {
                        crossings.push((t, p));
                    }
                }
                crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
            }

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
                        let t_new = (t1 - t0).mul_add(s, t0);
                        let on_curve = (curve.evaluate_with_endpoints(t_new, start_pos, end_pos)
                            - vp)
                            .length()
                            <= tol.linear;
                        on_curve.then_some((vid, t_new))
                    })
                })
                .collect();

            for (((t, pt), (start_window, end_window, _)), snap) in
                crossings.into_iter().zip(endpoint_windows).zip(snaps)
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
                                            (t1 - t0).mul_add(along, t0)
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
                add_pave_to_edge(arena, eid, pave);

                arena.interference.ef.push(Interference::EF {
                    edge: eid,
                    face: fid,
                    new_vertex: Some(vertex_id),
                    parameter: Some(t),
                    run: None,
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
        let t = s_clamped.mul_add(t1 - t0, t0);
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
            .mul_add(p.x(), normal.y().mul_add(p.y(), normal.z() * p.z()))
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
    let mut out: Vec<(f64, Point3)> = Vec::new();
    for &root in roots {
        let t = t0 + (root - t0).rem_euclid(TAU);
        if t > t1 + 1e-12 {
            continue;
        }
        out.push((t, at(t)));
    }
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    out
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

    for (i, (&(t, _), &(dist, _))) in samples.iter().zip(&probed).enumerate() {
        if i > 0 && dist < tol.linear {
            let is_dup = crossings.iter().any(|&(ct, _): &(f64, Point3)| {
                (t - ct).abs() < ((t1 - t0) / (n as f64) * 2.0).abs()
            });
            if !is_dup {
                let refined = refine_crossing(curve, start_pos, end_pos, prev_t, t, surface, tol);
                crossings.push(refined);
            }
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
        .map(|p| ((t1 - t0).mul_add((p - start_pos).dot(d) / len_sq, t0), p))
        .collect()
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
        let t = (t1 - t0).mul_add(i as f64 / N_SAMPLES as f64, t0);
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
    if let FaceSurface::Plane { normal, d } = surface {
        (pt.x() * normal.x() + pt.y() * normal.y() + pt.z() * normal.z() - d).abs()
    } else if let Some((u, v)) = surface.project_point(pt) {
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
        let x = (0.999 - y) / 2.0f64.mul_add(-y, 1.0);
        let z_top = 3.5f64.mul_add(tol.linear, 1.0) + step;
        let (a, b) = (
            Point3::new(x, y, 64.0f64.mul_add(-step, z_top)),
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
}
