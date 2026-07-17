//! Special topology handlers for face splitting edge cases.

use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::super::plane_frame::PlaneFrame;
use super::super::split_types::{OrientedPCurveEdge, SectionEdge, SplitSubFace};
use super::conversion::uv_endpoints_from_pcurve;
use super::edge_splitting::split_boundary_edges_at_3d_points;
use crate::ds::Rank;

/// Split a face with no seam edges directly into cap + remainder sub-faces.
///
/// Faces whose boundary consists entirely of Line edges (no seam edges)
/// can't be split by the wire builder (it needs vertical seam connections).
/// This function bypasses the wire builder and constructs sub-faces
/// geometrically from the section edges:
///
/// - **Cap**: bounded by the section arcs chained into one closed loop.
/// - **Remainder**: when the cap loop stays interior, the original boundary
///   with the reversed cap as a hole; when a cap arc runs along the face
///   boundary (e.g. the in-region equator arc of a faceted sphere whose
///   boundary polygon is inscribed in the same circle), the covered boundary
///   edges are replaced by the arc, and the remainder is chained from the
///   uncovered boundary edges plus the reversed non-coincident arcs.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn split_noseam_face_direct(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    wire_pts: &[Point3],
    tol: f64,
) -> Vec<SplitSubFace> {
    let close_tol = tol * 100.0;

    // Helper: return the face unsplit (used in fallback paths).
    let unsplit = || {
        vec![SplitSubFace {
            surface: surface.clone(),
            outer_wire: boundary_edges.to_vec(),
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        }]
    };

    // Collect open section arcs on this face, plus any closed-circle sections
    // (interior cap circles, e.g. a latitude cut that stays inside one
    // hemisphere) which become holes in the assembled region.
    let mut open_sections: Vec<OrientedPCurveEdge> = Vec::new();
    let mut closed_sections: Vec<OrientedPCurveEdge> = Vec::new();
    for section in sections {
        let pcurve_on_this_face = match rank {
            Rank::A => &section.pcurve_a,
            Rank::B => &section.pcurve_b,
        };

        let precomputed_uv = match rank {
            Rank::A => section.start_uv_a.zip(section.end_uv_a),
            Rank::B => section.start_uv_b.zip(section.end_uv_b),
        };
        let (start_uv, end_uv) = precomputed_uv.unwrap_or_else(|| {
            uv_endpoints_from_pcurve(
                pcurve_on_this_face,
                section.start,
                section.end,
                surface,
                wire_pts,
            )
        });

        let edge = OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };

        // Full-circle section edges (start approx end in 3D) are interior caps;
        // open arcs were produced by the FF boundary-crossing split.
        if (section.start - section.end).length() < tol {
            closed_sections.push(edge);
        } else {
            open_sections.push(edge);
        }
    }

    if open_sections.is_empty() {
        return unsplit();
    }

    // Chain the arcs into a single closed loop, greedily matching endpoints
    // (with reversal). When the open arcs are disjoint (each crossing the
    // boundary at distinct points, sharing no endpoints — e.g. a sphere
    // hemisphere cut by several box faces along great-circle arcs), they
    // cannot chain alone; assemble the regions by interleaving the boundary
    // sub-segments between the arcs (a UV-space planar arrangement).
    let Some(cap_edges) = chain_closed_loop(open_sections.clone(), close_tol) else {
        return split_noseam_by_arrangement(
            surface,
            boundary_edges,
            &open_sections,
            &closed_sections,
            rank,
            reversed,
            face_id,
            tol,
        );
    };

    // Boundary edges covered by a cap arc: both segment endpoints lie on
    // the arc's circle within its angular span. Those edges are replaced
    // by the (exact) arc in the cap; the remainder must not reuse them.
    let covered: Vec<bool> = boundary_edges
        .iter()
        .map(|be| cap_edges.iter().any(|arc| arc_covers_segment(arc, be, tol)))
        .collect();
    let coincident: Vec<bool> = cap_edges
        .iter()
        .map(|arc| {
            boundary_edges
                .iter()
                .zip(&covered)
                .any(|(be, &cov)| cov && arc_covers_segment(arc, be, tol))
        })
        .collect();

    let reverse_of = |e: &OrientedPCurveEdge| OrientedPCurveEdge {
        curve_3d: e.curve_3d.clone(),
        pcurve: e.pcurve.clone(),
        start_uv: e.end_uv,
        end_uv: e.start_uv,
        start_3d: e.end_3d,
        end_3d: e.start_3d,
        forward: !e.forward,
        source_edge_idx: e.source_edge_idx,
        pave_block_id: e.pave_block_id,
    };

    if covered.iter().any(|&c| c) {
        // Cap loop runs along part of the boundary: remainder = uncovered
        // boundary edges + reversed non-coincident arcs, chained closed.
        let mut pool: Vec<OrientedPCurveEdge> = boundary_edges
            .iter()
            .zip(&covered)
            .filter(|&(_, &cov)| !cov)
            .map(|(be, _)| be.clone())
            .collect();
        for (arc, &coin) in cap_edges.iter().zip(&coincident) {
            if !coin {
                pool.push(reverse_of(arc));
            }
        }
        let Some(remainder) = chain_closed_loop(pool, close_tol) else {
            return unsplit();
        };

        let cap_interior = sphere_loop_interior(surface, &cap_edges);
        let remainder_interior = sphere_loop_interior(surface, &remainder);
        return vec![
            SplitSubFace {
                surface: surface.clone(),
                outer_wire: cap_edges,
                inner_wires: Vec::new(),
                reversed,
                parent: face_id,
                rank,
                precomputed_interior: cap_interior,
            },
            SplitSubFace {
                surface: surface.clone(),
                outer_wire: remainder,
                inner_wires: Vec::new(),
                reversed,
                parent: face_id,
                rank,
                precomputed_interior: remainder_interior,
            },
        ];
    }

    // Cap loop is interior to the boundary: cap + band-with-hole.
    let hole_edges: Vec<OrientedPCurveEdge> = cap_edges.iter().map(reverse_of).collect();
    vec![
        SplitSubFace {
            surface: surface.clone(),
            outer_wire: cap_edges,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        },
        SplitSubFace {
            surface: surface.clone(),
            outer_wire: boundary_edges.to_vec(),
            inner_wires: vec![hole_edges],
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        },
    ]
}

/// Split a sphere face whose disjoint open arcs cannot chain alone into a
/// region sub-face, by interleaving the seam (boundary) sub-segments between
/// the arcs.
///
/// Each open arc crosses the seam boundary at its two endpoints. The seam is a
/// polygon inscribed in the seam circle, so the arcs land OFF its chords (by
/// the polygon sagitta) and the chords cannot be split at them. Instead
/// reconstruct the seam as its exact circle, split that at the crossings, and
/// trace the arrangement of (seam arcs + open arcs) in UV. The wanted region —
/// the slice of this hemisphere inside the cutting solid — is an annular collar
/// (it wraps fully around longitude) bounded by a scalloped "bottom chain"
/// (seam arcs alternating with great-circle arcs) with any interior latitude
/// cap as an inner hole.
#[allow(clippy::too_many_arguments)]
fn split_noseam_by_arrangement(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    open_sections: &[OrientedPCurveEdge],
    closed_sections: &[OrientedPCurveEdge],
    rank: crate::ds::Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Vec<SplitSubFace> {
    let unsplit = || {
        vec![SplitSubFace {
            surface: surface.clone(),
            outer_wire: boundary_edges.to_vec(),
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        }]
    };

    // Need at least two arcs to interleave; one arc is handled by the cap path.
    if open_sections.len() < 2 {
        return unsplit();
    }

    // Reconstruct the seam as its exact circle and split it at the crossings,
    // so the seam arcs share endpoints EXACTLY with the open arcs.
    let Some(seam_arcs) = build_seam_arcs(surface, boundary_edges, open_sections, tol) else {
        return unsplit();
    };

    // Half-edge soup: every seam arc and every open arc in both orientations,
    // so the angular traversal can bound a region from either side.
    let both = |e: &OrientedPCurveEdge| -> [OrientedPCurveEdge; 2] {
        let rev = OrientedPCurveEdge {
            curve_3d: e.curve_3d.clone(),
            pcurve: e.pcurve.clone(),
            start_uv: e.end_uv,
            end_uv: e.start_uv,
            start_3d: e.end_3d,
            end_3d: e.start_3d,
            forward: !e.forward,
            source_edge_idx: e.source_edge_idx,
            pave_block_id: e.pave_block_id,
        };
        [e.clone(), rev]
    };
    let mut soup: Vec<OrientedPCurveEdge> = Vec::new();
    for e in &seam_arcs {
        soup.extend(both(e));
    }
    for a in open_sections {
        soup.extend(both(a));
    }

    // Trace the arrangement faces in UV. The shared seam plane (all crossings
    // at one latitude) makes the generic wire builder's endpoint-tangent sort
    // ambiguous, so use a dedicated tracer that reads each half-edge's
    // direction from its pcurve.
    let loops = trace_region_loops(&soup, tol * 10.0);

    let hole_loops: Vec<Vec<OrientedPCurveEdge>> =
        closed_sections.iter().map(|c| vec![c.clone()]).collect();

    // Net longitude wound by a loop (≈ ±2π for a chain that encircles the
    // sphere once, ~0 for a lune or a back-and-forth chain). Robust where signed
    // UV area is degenerate (all arrangement vertices sit on the seam, v=0).
    let net_u = |l: &[OrientedPCurveEdge]| -> f64 {
        use std::f64::consts::{PI, TAU};
        let poly = loop_polyline(l);
        if poly.len() < 2 {
            return 0.0;
        }
        // Sum of shortest signed u-steps (each wrapped into (-pi, pi]) — a
        // discretization-independent winding measure.
        (0..poly.len())
            .map(|i| {
                let d = poly[(i + 1) % poly.len()].x() - poly[i].x();
                d - TAU * ((d + PI) / TAU).floor()
            })
            .sum()
    };
    let loop_is_sliver = |l: &[OrientedPCurveEdge]| -> bool {
        for i in 0..l.len() {
            for j in (i + 1)..l.len() {
                if (l[i].start_3d - l[j].end_3d).length() < tol * 100.0
                    && (l[i].end_3d - l[j].start_3d).length() < tol * 100.0
                {
                    return true;
                }
            }
        }
        false
    };

    // The collar's outer wire is the unique non-sliver loop encircling the
    // sphere once in longitude. Orient it to oppose the parent boundary's
    // winding (so the collar, not the discarded lunes, is its interior).
    let parent_net_u = net_u(boundary_edges);
    let mut best: Option<usize> = None;
    for (i, l) in loops.iter().enumerate() {
        if l.len() < 3 || loop_is_sliver(l) {
            continue;
        }
        if net_u(l).abs() < std::f64::consts::PI {
            continue;
        }
        if best.is_none_or(|b| l.len() > loops[b].len()) {
            best = Some(i);
        }
    }

    let Some(region_idx) = best else {
        return unsplit();
    };
    let mut region = loops[region_idx].clone();
    if net_u(&region) * parent_net_u > 0.0 {
        region = reverse_loop(&region);
    }

    // 3D interior sample for classification (a point on the collar surface).
    let interior_3d = patch_interior_point(surface, &hole_loops, open_sections);

    // Each latitude cap on this hemisphere is an inner hole of the collar.
    let region_holes: Vec<Vec<OrientedPCurveEdge>> =
        hole_loops.iter().map(|hl| reverse_loop(hl)).collect();

    vec![SplitSubFace {
        surface: surface.clone(),
        outer_wire: region,
        inner_wires: region_holes,
        reversed,
        parent: face_id,
        rank,
        precomputed_interior: Some(interior_3d),
    }]
}

/// Reconstruct a sphere face's seam (boundary) as its exact circle and split it
/// at the open arcs' crossing points into seam-arc edges.
///
/// The seam circle is the intersection of the boundary polygon's plane with the
/// sphere. Splitting the *circle* (rather than the inscribed chords) means the
/// seam arcs share their endpoints exactly with the open arcs, which is what
/// makes the assembled region watertight.
fn build_seam_arcs(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    open_sections: &[OrientedPCurveEdge],
    tol: f64,
) -> Option<Vec<OrientedPCurveEdge>> {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::Vec3;

    let FaceSurface::Sphere(sphere) = surface else {
        return None;
    };

    // Seam-plane normal + a point on it, from the boundary polygon (Newell).
    let verts: Vec<Point3> = boundary_edges.iter().map(|e| e.start_3d).collect();
    if verts.len() < 3 {
        return None;
    }
    let mut nrm = Vec3::new(0.0, 0.0, 0.0);
    let mut cen = Vec3::new(0.0, 0.0, 0.0);
    let n = verts.len();
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        nrm += Vec3::new(
            (a.y() - b.y()) * (a.z() + b.z()),
            (a.z() - b.z()) * (a.x() + b.x()),
            (a.x() - b.x()) * (a.y() + b.y()),
        );
        cen += Vec3::new(a.x(), a.y(), a.z());
    }
    let plane_n = nrm.normalize().ok()?;
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / n as f64;
    let plane_pt = Point3::new(cen.x() * inv_n, cen.y() * inv_n, cen.z() * inv_n);

    // Seam circle on the sphere: centre offset from the sphere centre along the
    // plane normal by the plane's signed distance; radius from Pythagoras.
    let h = (plane_pt - sphere.center()).dot(plane_n);
    let rr = sphere.radius() * sphere.radius() - h * h;
    if rr <= tol * tol {
        return None;
    }
    let seam_radius = rr.sqrt();
    let seam_center = sphere.center() + plane_n * h;
    let seam_circle = Circle3D::new(seam_center, plane_n, seam_radius).ok()?;

    // Crossing points = the open arcs' endpoints (they lie on the seam circle).
    let mut unique: Vec<Point3> = Vec::new();
    for p in open_sections.iter().flat_map(|a| [a.start_3d, a.end_3d]) {
        if !unique.iter().any(|q| (*q - p).length() < tol * 100.0) {
            unique.push(p);
        }
    }
    if unique.len() < 2 {
        return None;
    }

    // Sort crossings by angle on the seam circle, then build the arc between
    // each consecutive pair (closing the loop).
    let mut by_angle: Vec<(f64, Point3)> = unique
        .into_iter()
        .map(|p| (seam_circle.project(p), p))
        .collect();
    by_angle.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut arcs: Vec<OrientedPCurveEdge> = Vec::new();
    let m = by_angle.len();
    for i in 0..m {
        let (_, start_3d) = by_angle[i];
        let (_, end_3d) = by_angle[(i + 1) % m];
        if (start_3d - end_3d).length() < tol * 100.0 {
            continue;
        }
        let curve = EdgeCurve::Circle(seam_circle.clone());
        let pcurve = super::super::pcurve_compute::compute_pcurve_on_surface(
            &curve,
            start_3d,
            end_3d,
            surface,
            &[],
            None,
        );
        let start_uv =
            super::super::pcurve_compute::project_point_on_surface(start_3d, surface, &[], None);
        let end_uv =
            super::super::pcurve_compute::project_point_on_surface(end_3d, surface, &[], None);
        arcs.push(OrientedPCurveEdge {
            curve_3d: curve,
            pcurve,
            start_uv,
            end_uv,
            start_3d,
            end_3d,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        });
    }
    if arcs.len() < 2 {
        return None;
    }
    Some(arcs)
}

/// Trace the faces of a planar half-edge arrangement in UV.
///
/// `soup` holds every undirected edge in both orientations. Each directed
/// half-edge belongs to exactly one face loop; the next half-edge in a face is
/// the first one counter-clockwise from the incoming edge's reverse at the
/// shared vertex (the standard DCEL face walk). Directions are read from each
/// pcurve so curved arcs that share a vertex with straight seam segments at the
/// same latitude are distinguished. Returns every traced loop (callers drop the
/// outer face / lunes by winding).
fn trace_region_loops(soup: &[OrientedPCurveEdge], tol: f64) -> Vec<Vec<OrientedPCurveEdge>> {
    use std::collections::HashMap;
    use std::f64::consts::TAU;

    let q = |v: f64| -> i64 {
        #[allow(clippy::cast_possible_truncation)]
        let r = (v / tol).round() as i64;
        r
    };
    // u wraps; quantize u modulo 2π so seam-opposite endpoints share a key.
    let key = |p: brepkit_math::vec::Point2| -> (i64, i64) { (q(p.x().rem_euclid(TAU)), q(p.y())) };

    // Direction of a half-edge at one endpoint, from the pcurve's analytic
    // tangent (it already encodes the correct arc and bulge — avoiding the
    // shorter-arc ambiguity of half-circle 3D arcs and the straight-vs-curved
    // confusion at a shared latitude). A reversed half-edge reuses the forward
    // pcurve, so its logical start is the pcurve's domain end. `from_start=true`
    // returns the OUTGOING direction at the logical start; `false` the INCOMING
    // direction at the logical end (pointing back toward the start).
    let edge_dir = |e: &OrientedPCurveEdge, from_start: bool| -> f64 {
        use brepkit_math::curves2d::Curve2D;
        let (mut dx, dy) = if let Curve2D::Nurbs(nurbs) = &e.pcurve {
            let (t0, t1) = nurbs.domain();
            let (t_at, sign) = match (from_start, e.forward) {
                (true, true) => (t0, 1.0),
                (true, false) => (t1, -1.0),
                (false, true) => (t1, -1.0),
                (false, false) => (t0, 1.0),
            };
            let tan = nurbs.tangent(t_at);
            (tan.x() * sign, tan.y() * sign)
        } else if from_start {
            (e.end_uv.x() - e.start_uv.x(), e.end_uv.y() - e.start_uv.y())
        } else {
            (e.start_uv.x() - e.end_uv.x(), e.start_uv.y() - e.end_uv.y())
        };
        if dx.abs() > TAU * 0.5 {
            dx -= dx.signum() * TAU;
        }
        dy.atan2(dx).rem_euclid(TAU)
    };
    let out_dir = |e: &OrientedPCurveEdge| -> f64 { edge_dir(e, true) };
    let in_dir = |e: &OrientedPCurveEdge| -> f64 { edge_dir(e, false) };

    let mut outgoing: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, e) in soup.iter().enumerate() {
        outgoing.entry(key(e.start_uv)).or_default().push(i);
    }

    let mut used = vec![false; soup.len()];
    let mut loops: Vec<Vec<OrientedPCurveEdge>> = Vec::new();

    for start in 0..soup.len() {
        if used[start] {
            continue;
        }
        let mut loop_edges: Vec<OrientedPCurveEdge> = Vec::new();
        let mut cur = start;
        let mut closed = false;
        for _ in 0..=soup.len() {
            used[cur] = true;
            loop_edges.push(soup[cur].clone());
            let end_key = key(soup[cur].end_uv);
            if end_key == key(soup[start].start_uv) && loop_edges.len() >= 2 {
                closed = true;
                break;
            }
            let arrive_back = in_dir(&soup[cur]);
            let Some(cands) = outgoing.get(&end_key) else {
                break;
            };
            let mut best: Option<usize> = None;
            let mut best_score = f64::MAX;
            let mut fallback: Option<usize> = None;
            let mut fb_score = f64::MAX;
            for &c in cands {
                if used[c] {
                    continue;
                }
                let cd = out_dir(&soup[c]);
                // First edge counter-clockwise from the reverse of the arriving
                // edge (CCW-interior face walk).
                let ccw = (cd - arrive_back).rem_euclid(TAU);
                if ccw < fb_score {
                    fb_score = ccw;
                    fallback = Some(c);
                }
                if ccw < 1e-4 || (TAU - ccw) < 1e-4 {
                    continue; // near-exact reverse (U-turn)
                }
                if ccw < best_score {
                    best_score = ccw;
                    best = Some(c);
                }
            }
            let Some(next) = best.or(fallback) else {
                break;
            };
            cur = next;
        }
        if closed && loop_edges.len() >= 2 {
            loops.push(loop_edges);
        }
    }
    loops
}

/// UV point on an oriented half-edge at fraction `f` in [0,1] of its logical
/// start→end. A reversed half-edge reuses the forward pcurve, so its start is
/// the pcurve's domain end; curved (NURBS) pcurves are sampled over their own
/// domain (which is not generally [0,1]).
fn sample_half_edge_uv(e: &OrientedPCurveEdge, f: f64) -> brepkit_math::vec::Point2 {
    use brepkit_math::curves2d::Curve2D;
    use brepkit_math::vec::Point2;
    match &e.pcurve {
        Curve2D::Nurbs(nurbs) => {
            let (t0, t1) = nurbs.domain();
            let p = if e.forward {
                t0 + (t1 - t0) * f
            } else {
                t1 - (t1 - t0) * f
            };
            nurbs.evaluate(p)
        }
        _ => Point2::new(
            e.start_uv.x() + (e.end_uv.x() - e.start_uv.x()) * f,
            e.start_uv.y() + (e.end_uv.y() - e.start_uv.y()) * f,
        ),
    }
}

/// Polyline (UV) approximation of a loop, sampling curved edges.
fn loop_polyline(loop_edges: &[OrientedPCurveEdge]) -> Vec<brepkit_math::vec::Point2> {
    let mut poly = Vec::new();
    for e in loop_edges {
        let n = if matches!(e.curve_3d, EdgeCurve::Line) {
            1
        } else {
            16
        };
        for k in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let f = k as f64 / f64::from(n);
            poly.push(sample_half_edge_uv(e, f));
        }
    }
    poly
}

/// Reverse a loop's orientation (a hole is traversed opposite to the containing
/// region's outer wire).
fn reverse_loop(loop_edges: &[OrientedPCurveEdge]) -> Vec<OrientedPCurveEdge> {
    loop_edges
        .iter()
        .rev()
        .map(|e| OrientedPCurveEdge {
            curve_3d: e.curve_3d.clone(),
            pcurve: e.pcurve.clone(),
            start_uv: e.end_uv,
            end_uv: e.start_uv,
            start_3d: e.end_3d,
            end_3d: e.start_3d,
            forward: !e.forward,
            source_edge_idx: e.source_edge_idx,
            pave_block_id: e.pave_block_id,
        })
        .collect()
}

/// A 3D interior sample on the in-solid collar patch, for classification.
///
/// When the face has a latitude cap, the sample is a point on the cap's
/// latitude nudged toward the equator so it lands on the collar surface (not in
/// the removed cap). Otherwise the patch reaches the pole, so use a near-pole
/// point on the hemisphere the open arcs bulge toward.
fn patch_interior_point(
    surface: &FaceSurface,
    hole_loops: &[Vec<OrientedPCurveEdge>],
    open_sections: &[OrientedPCurveEdge],
) -> Point3 {
    use brepkit_math::vec::Vec3;
    let FaceSurface::Sphere(sphere) = surface else {
        return Point3::new(0.0, 0.0, 0.0);
    };

    if let Some(cap) = hole_loops.first().and_then(|h| h.first()) {
        let (u_cap, v_cap) = sphere.project_point(cap.start_3d);
        let v_sample = v_cap - v_cap.signum() * (v_cap.abs() * 0.25 + 0.05);
        return sphere.evaluate(u_cap, v_sample);
    }

    // No cap: aim toward the pole the open arcs bulge to.
    let mut dir = Vec3::new(0.0, 0.0, 0.0);
    for e in open_sections {
        let mid = super::super::pcurve_compute::evaluate_edge_at_t(
            &e.curve_3d,
            e.start_3d,
            e.end_3d,
            0.5,
        );
        if let Ok(d) = (mid - sphere.center()).normalize() {
            dir += d;
        }
    }
    match dir.normalize() {
        Ok(d) => sphere.center() + d * sphere.radius(),
        Err(_) => sphere.center() + Vec3::new(0.0, 0.0, sphere.radius()),
    }
}

/// Greedily chain edges into one closed loop by matching 3D endpoints,
/// reversing edges as needed. Returns `None` when the edges are
/// disconnected, leave leftovers, or do not close.
fn chain_closed_loop(
    mut pool: Vec<OrientedPCurveEdge>,
    close_tol: f64,
) -> Option<Vec<OrientedPCurveEdge>> {
    if pool.is_empty() {
        return None;
    }
    let mut chain = vec![pool.remove(0)];
    while !pool.is_empty() {
        let cur_end = chain.last()?.end_3d;
        if let Some(pos) = pool
            .iter()
            .position(|e| (e.start_3d - cur_end).length() < close_tol)
        {
            chain.push(pool.remove(pos));
        } else if let Some(pos) = pool
            .iter()
            .position(|e| (e.end_3d - cur_end).length() < close_tol)
        {
            let mut e = pool.remove(pos);
            std::mem::swap(&mut e.start_uv, &mut e.end_uv);
            std::mem::swap(&mut e.start_3d, &mut e.end_3d);
            e.forward = !e.forward;
            chain.push(e);
        } else {
            return None;
        }
    }
    let gap = (chain.last()?.end_3d - chain.first()?.start_3d).length();
    if gap > close_tol { None } else { Some(chain) }
}

/// Whether a straight boundary segment lies along (is a chord of) the
/// arc: both endpoints on the arc's circle, within the arc's angular span.
fn arc_covers_segment(arc: &OrientedPCurveEdge, segment: &OrientedPCurveEdge, tol: f64) -> bool {
    use brepkit_topology::edge::EdgeCurve;
    let EdgeCurve::Circle(c) = &arc.curve_3d else {
        return false;
    };
    if !matches!(segment.curve_3d, EdgeCurve::Line) {
        return false;
    }
    let on_circle = |p: Point3| -> bool {
        let r = p - c.center();
        let axial = c.normal().dot(r);
        let radial = (r - c.normal() * axial).length();
        axial.abs() < tol * 100.0 && (radial - c.radius()).abs() < tol * 100.0
    };
    if !on_circle(segment.start_3d) || !on_circle(segment.end_3d) {
        return false;
    }
    // The arc's traversal direction comes from `forward` relative to the
    // circle's native (CCW) parameterization — not from the endpoint angles
    // alone, which are direction-ambiguous. Measuring membership as the
    // forward-progress delta from the start handles spans across the full
    // (0, 2pi) range, including arcs longer than pi (a 270° arc no longer
    // collapses to its complementary 90° short arc).
    let tau = std::f64::consts::TAU;
    let t0 = c.project(arc.start_3d);
    // Forward progress (always in [0, 2pi)) of `p` from the arc start in the
    // arc's traversal direction.
    let progress = |p: Point3| -> f64 {
        let delta = c.project(p) - t0;
        if arc.forward {
            delta.rem_euclid(tau)
        } else {
            (-delta).rem_euclid(tau)
        }
    };
    let span = progress(arc.end_3d);
    let eps = 1e-6;
    let within = |p: Point3| -> bool {
        let d = progress(p);
        // A point at the arc start wraps to ~2pi under rem_euclid; treat it
        // as 0 so segment endpoints coincident with the start still match.
        let d = if d > tau - eps { 0.0 } else { d };
        d <= span + eps
    };
    within(segment.start_3d) && within(segment.end_3d)
}

/// Interior point for a loop on a sphere face: the spherical centroid of
/// the loop edges' midpoints, projected back onto the sphere. `None` for
/// non-sphere surfaces (callers fall back to UV-based interior sampling).
fn sphere_loop_interior(surface: &FaceSurface, edges: &[OrientedPCurveEdge]) -> Option<Point3> {
    use brepkit_math::vec::Vec3;
    let FaceSurface::Sphere(s) = surface else {
        return None;
    };
    let center = s.center();
    let mut dir = Vec3::new(0.0, 0.0, 0.0);
    for e in edges {
        let mid = super::super::pcurve_compute::evaluate_edge_at_t(
            &e.curve_3d,
            e.start_3d,
            e.end_3d,
            0.5,
        );
        if let Ok(d) = (mid - center).normalize() {
            dir += d;
        }
    }
    let d = dir.normalize().ok()?;
    Some(center + d * s.radius())
}

// ---------------------------------------------------------------------------
// Periodic faces take `split_periodic_face_into_bands` below; the disc-loop
// interpretation in `split_face_with_internal_loops` applies to plane faces
// (and to periodic faces only as a fallback when the band preconditions
// don't hold). The band path depends on the seam-anchor pre-pass in
// `fill_images_faces`, which re-parameterizes closed section circles to
// start at the face's seam — without it, the band's seam segments would
// connect arbitrary section angles and cut through the surface interior.
// ---------------------------------------------------------------------------

/// Split a u-periodic face (cylinder/cone lateral) into stacked bands at
/// its closed section circles.
///
/// A closed circle on a u-periodic surface does not bound a disc — it
/// separates the surface into bands. For N internal circles sorted by v,
/// emits N+1 band sub-faces, each bounded by:
/// lower circle + seam segment up + upper circle reversed + seam segment
/// down. The end bands reuse the face's original boundary circle edges.
///
/// Preconditions (returns `None` so the caller can fall back otherwise):
/// - surface is a cylinder or cone
/// - boundary is exactly 2 closed circle edges plus seam Line edges, all
///   seam endpoints at the same u
/// - every section is a full closed circle whose start point sits on the
///   seam (guaranteed by the seam-anchor pre-pass) at a v strictly between
///   the boundary circles, with no two circles at the same v
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::items_after_statements
)]
pub(super) fn split_periodic_face_into_bands(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2};
    use std::f64::consts::{PI, TAU};

    if !matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) {
        return None;
    }
    let close_tol = tol * 100.0;

    // Partition boundary into closed circle edges and seam Line edges.
    let mut boundary_circles: Vec<&OrientedPCurveEdge> = Vec::new();
    let mut seam_edges: Vec<&OrientedPCurveEdge> = Vec::new();
    for e in boundary_edges {
        let is_closed = (e.start_3d - e.end_3d).length() < close_tol;
        match (&e.curve_3d, is_closed) {
            (EdgeCurve::Circle(_), true) => boundary_circles.push(e),
            (EdgeCurve::Line, false) => seam_edges.push(e),
            _ => return None,
        }
    }
    if boundary_circles.len() != 2 || seam_edges.is_empty() {
        return None;
    }

    // Seam u — shared by every seam edge endpoint (mod 2π).
    let (seam_u, _) = surface.project_point(seam_edges[0].start_3d)?;
    for e in &seam_edges {
        for p in [e.start_3d, e.end_3d] {
            let (u, _) = surface.project_point(p)?;
            let du = (u - seam_u + PI).rem_euclid(TAU) - PI;
            if du.abs() > 1e-6 {
                return None;
            }
        }
    }

    // Every circle must start on the seam; collect (v, lower_fwd, edge).
    let circle_v = |e: &OrientedPCurveEdge| -> Option<f64> {
        let (_, v) = surface.project_point(e.start_3d)?;
        let on_seam = surface.evaluate(seam_u, v)?;
        ((on_seam - e.start_3d).length() < close_tol).then_some(v)
    };

    let v0 = circle_v(boundary_circles[0])?;
    let v1 = circle_v(boundary_circles[1])?;
    let (v_bot, bot_edge, v_top, top_edge) = if v0 < v1 {
        (v0, boundary_circles[0], v1, boundary_circles[1])
    } else {
        (v1, boundary_circles[1], v0, boundary_circles[0])
    };
    if v_top - v_bot < close_tol {
        return None;
    }

    // Reference traversal tangent: how the original bottom circle is
    // traversed at the seam. Section circles in the lower role must
    // traverse the same way; in the upper role, the opposite way.
    let traversal_tangent = |e: &OrientedPCurveEdge| -> Option<brepkit_math::vec::Vec3> {
        let EdgeCurve::Circle(c) = &e.curve_3d else {
            return None;
        };
        let t = c.tangent(c.project(e.start_3d));
        Some(if e.forward { t } else { -t })
    };
    let ref_tan = traversal_tangent(bot_edge)?;

    // Collect section circles with their v and natural-direction alignment.
    struct BandCircle {
        v: f64,
        lower: OrientedPCurveEdge,
        upper: OrientedPCurveEdge,
    }
    let mut mids: Vec<BandCircle> = Vec::with_capacity(sections.len());
    for s in sections {
        if (s.start - s.end).length() > close_tol {
            return None;
        }
        let EdgeCurve::Circle(c) = &s.curve_3d else {
            return None;
        };
        let (_, v) = surface.project_point(s.start)?;
        let on_seam = surface.evaluate(seam_u, v)?;
        if (on_seam - s.start).length() > close_tol {
            return None;
        }
        // A section circle at a boundary circle's v duplicates the existing
        // boundary edge (flush cap configuration) — no split there.
        if (v - v_bot).abs() < close_tol || (v_top - v).abs() < close_tol {
            continue;
        }
        if v < v_bot || v > v_top {
            return None;
        }
        let natural_tan = c.tangent(c.project(s.start));
        let lower_fwd = natural_tan.dot(ref_tan) > 0.0;
        let pcurve = match rank {
            Rank::A => &s.pcurve_a,
            Rank::B => &s.pcurve_b,
        };
        let mk = |forward: bool| OrientedPCurveEdge {
            curve_3d: s.curve_3d.clone(),
            pcurve: pcurve.clone(),
            start_uv: Point2::new(seam_u, v),
            end_uv: Point2::new(seam_u, v),
            start_3d: s.start,
            end_3d: s.start,
            forward,
            source_edge_idx: None,
            pave_block_id: s.pave_block_id,
        };
        mids.push(BandCircle {
            v,
            lower: mk(lower_fwd),
            upper: mk(!lower_fwd),
        });
    }
    if mids.is_empty() {
        // Every section coincided with a boundary circle (fully flush
        // tool): no band split applies — let the generic paths handle the
        // coplanar-cap interaction.
        return None;
    }
    mids.sort_by(|a, b| a.v.partial_cmp(&b.v).unwrap_or(std::cmp::Ordering::Equal));
    if mids.windows(2).any(|w| w[1].v - w[0].v < close_tol) {
        return None;
    }

    let mk_seam = |va: f64, vb: f64| -> Option<OrientedPCurveEdge> {
        let pa = surface.evaluate(seam_u, va)?;
        let pb = surface.evaluate(seam_u, vb)?;
        let dir = Vec2::new(0.0, if vb > va { 1.0 } else { -1.0 });
        let pcurve = Curve2D::Line(Line2D::new(Point2::new(seam_u, va), dir).ok()?);
        Some(OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: Point2::new(seam_u, va),
            end_uv: Point2::new(seam_u, vb),
            start_3d: pa,
            end_3d: pb,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        })
    };

    // Assemble bands bottom-to-top. Levels: bot boundary, sections, top
    // boundary. Each band: lower circle, seam up, upper circle, seam down.
    let mut levels: Vec<(f64, OrientedPCurveEdge, OrientedPCurveEdge)> = Vec::new();
    levels.push((v_bot, bot_edge.clone(), bot_edge.clone()));
    for m in mids {
        levels.push((m.v, m.lower, m.upper));
    }
    levels.push((v_top, top_edge.clone(), top_edge.clone()));

    let mut bands = Vec::with_capacity(levels.len() - 1);
    for w in levels.windows(2) {
        let (va, lower, _) = &w[0];
        let (vb, _, upper) = &w[1];
        let (va, vb) = (*va, *vb);
        let wire = vec![
            lower.clone(),
            mk_seam(va, vb)?,
            upper.clone(),
            mk_seam(vb, va)?,
        ];
        let interior = surface.evaluate((seam_u + PI).rem_euclid(TAU), f64::midpoint(va, vb))?;
        bands.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(interior),
        });
    }
    Some(bands)
}

/// Build an `OrientedPCurveEdge` (forward) from a section on this (torus) face.
fn torus_section_to_edge(
    section: &SectionEdge,
    surface: &FaceSurface,
    rank: Rank,
) -> OrientedPCurveEdge {
    let pcurve = match rank {
        Rank::A => &section.pcurve_a,
        Rank::B => &section.pcurve_b,
    };
    let (start_uv, end_uv) = match rank {
        Rank::A => section.start_uv_a.zip(section.end_uv_a),
        Rank::B => section.start_uv_b.zip(section.end_uv_b),
    }
    .unwrap_or_else(|| uv_endpoints_from_pcurve(pcurve, section.start, section.end, surface, &[]));
    OrientedPCurveEdge {
        curve_3d: section.curve_3d.clone(),
        pcurve: pcurve.clone(),
        start_uv,
        end_uv,
        start_3d: section.start,
        end_3d: section.end,
        forward: true,
        source_edge_idx: None,
        pave_block_id: section.pave_block_id,
    }
}

/// Contained tracer for the `torus − box`-style cut: a box notch removes a
/// connected sector of the ring whose surface boundary, in the torus `(θ, φ)`
/// parameter space, is TWO closed loops each WRAPPING the tube angle `φ` fully
/// (one at each θ-band where the box walls cut the tube partially; the box
/// swallows the whole tube cross-section in between). The kept toroidal surface
/// is the annular `u`-band between those two loops — emitted as ONE band face
/// (outer wire = one φ-loop, the other as an inner wire, like a cylinder band
/// between two rims).
///
/// Returns `None` (defer to the generic path) unless the in-box arcs stitch into
/// exactly two φ-wrapping closed loops. Relies on the FF exact-crossing trim
/// (`trim_torus_oval_to_box_face`) having given the arcs faithful endpoints that
/// share the box-edge crossing vertices.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_torus_band_by_arrangement(
    surface: &FaceSurface,
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    use std::f64::consts::{PI, TAU};
    let FaceSurface::Torus(torus) = surface else {
        return None;
    };
    // Open arcs only (closed sections would be the lobe-hole case, not a band).
    let open: Vec<OrientedPCurveEdge> = sections
        .iter()
        .filter(|s| (s.start - s.end).length() > tol)
        .map(|s| torus_section_to_edge(s, surface, rank))
        .collect();
    if open.len() < 2 {
        return None;
    }

    // Stitch arcs into closed loops by chaining shared 3D endpoints. The FF
    // exact-crossing trim snaps adjacent arcs to the SAME box-edge crossing
    // (and the arc-split shares the exact midpoint), so a tight tolerance
    // suffices — a loose one could stitch unrelated endpoints into false loops.
    let join_tol = (tol * 100.0).max(tol);
    let mut used = vec![false; open.len()];
    let mut loops: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
    for s0 in 0..open.len() {
        if used[s0] {
            continue;
        }
        used[s0] = true;
        let mut chain = vec![open[s0].clone()];
        loop {
            let tail = chain.last().map_or(chain[0].start_3d, |e| e.end_3d);
            if (tail - chain[0].start_3d).length() < join_tol && chain.len() >= 2 {
                break; // closed
            }
            let nxt = open.iter().enumerate().find_map(|(i, e)| {
                if used[i] {
                    None
                } else if (e.start_3d - tail).length() < join_tol {
                    Some((i, false))
                } else if (e.end_3d - tail).length() < join_tol {
                    Some((i, true))
                } else {
                    None
                }
            });
            match nxt {
                Some((i, rev)) => {
                    used[i] = true;
                    let mut e = open[i].clone();
                    if rev {
                        std::mem::swap(&mut e.start_uv, &mut e.end_uv);
                        std::mem::swap(&mut e.start_3d, &mut e.end_3d);
                        e.forward = !e.forward;
                    }
                    chain.push(e);
                }
                None => break,
            }
        }
        let closed = chain.len() >= 2
            && chain
                .last()
                .is_some_and(|e| (e.end_3d - chain[0].start_3d).length() < join_tol);
        if closed {
            loops.push(chain);
        }
    }

    // The kept band needs exactly two φ-wrapping boundary loops.
    if loops.len() != 2 {
        return None;
    }
    // Each loop must wrap φ fully: net φ-traversal ≈ ±2π.
    let net_phi = |l: &[OrientedPCurveEdge]| -> f64 {
        let mut phis: Vec<f64> = Vec::new();
        for e in l {
            let (_, v) = torus.project_point(e.start_3d);
            phis.push(v);
            let (_, vm) = torus.project_point(
                e.curve_3d
                    .evaluate_with_endpoints(0.5, e.start_3d, e.end_3d),
            );
            phis.push(vm);
        }
        let mut acc = 0.0;
        for i in 0..phis.len() {
            let d = phis[(i + 1) % phis.len()] - phis[i];
            acc += d - TAU * ((d + PI) / TAU).floor();
        }
        acc
    };
    if loops.iter().any(|l| net_phi(l).abs() < PI) {
        return None;
    }

    // Snap each loop's internal junctions to the midpoint of the two meeting
    // endpoints so the loop is internally watertight at the box-edge vertices.
    let snap_loop = |l: &mut Vec<OrientedPCurveEdge>| {
        let n = l.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let mid = l[i].end_3d + (l[j].start_3d - l[i].end_3d) * 0.5;
            l[i].end_3d = mid;
            l[j].start_3d = mid;
        }
    };
    for l in &mut loops {
        snap_loop(l);
    }

    let outer = loops[0].clone();
    // Interior sample on the KEPT band: the band spans the ring angle u the LONG
    // way between the two boundary loops, which sit at roughly constant u (the
    // box-wall cuts). Take each loop's mean u (wrap-safe via summed unit vectors)
    // and the MIDPOINT of the long arc between them — NOT a hardcoded u = π,
    // which is only correct when the notch is on the +x side (a mirrored or
    // rotated cut puts the kept band's centre elsewhere). The band wraps the tube
    // v fully at this interior u, so v = 0 is on it.
    let loop_mean_u = |l: &[OrientedPCurveEdge]| -> f64 {
        let (mut sx, mut sy) = (0.0, 0.0);
        for e in l {
            let (u, _) = torus.project_point(e.start_3d);
            sx += u.cos();
            sy += u.sin();
        }
        sy.atan2(sx).rem_euclid(TAU)
    };
    let u0 = loop_mean_u(&loops[0]);
    let u1 = loop_mean_u(&loops[1]);
    // Long-arc midpoint between u0 and u1 (the short gap is the removed notch).
    let fwd = (u1 - u0).rem_euclid(TAU);
    let u_mid = if fwd >= PI {
        (u0 + fwd / 2.0).rem_euclid(TAU) // a->b the long way is increasing
    } else {
        (u0 - (TAU - fwd) / 2.0).rem_euclid(TAU) // long way is decreasing
    };
    let interior = torus.evaluate(u_mid, 0.0);
    let inner_rev = reverse_loop(&loops[1]);

    Some(vec![SplitSubFace {
        surface: surface.clone(),
        outer_wire: outer,
        inner_wires: vec![inner_rev],
        reversed,
        parent: face_id,
        rank,
        precomputed_interior: Some(interior),
    }])
}

/// Split a face when ALL section edges are interior (don't touch the boundary).
///
/// Groups section edges into closed loops by chaining shared 3D endpoints.
/// Each closed loop produces:
/// - An "inside" sub-face with the loop as outer wire
/// - A reversed copy added as an inner wire (hole) of the "outside" sub-face
///
/// The "outside" sub-face has the original boundary as outer wire with all
/// loops as holes.
///
/// The disc-loop interpretation is only correct for plane faces (and
/// sphere caps). Cylinder/cone laterals are routed to
/// [`split_periodic_face_into_bands`] first and only reach this fallback
/// when the band preconditions don't hold.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_face_with_internal_loops(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    original_inner_wires: &[Vec<OrientedPCurveEdge>],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    _wire_pts: &[Point3],
) -> Vec<SplitSubFace> {
    let tol_3d = brepkit_math::tolerance::Tolerance::new().linear;

    // Convert each section edge to an OrientedPCurveEdge, preserving the
    // original EdgeCurve (NURBS, Circle, etc.) without polyline approximation.
    let mut forward_edges: Vec<OrientedPCurveEdge> = Vec::new();

    for section in sections {
        let pcurve_on_face = match rank {
            Rank::A => &section.pcurve_a,
            Rank::B => &section.pcurve_b,
        };

        let (start_uv, end_uv) = match rank {
            Rank::A => section.start_uv_a.zip(section.end_uv_a).unwrap_or_else(|| {
                uv_endpoints_from_pcurve(pcurve_on_face, section.start, section.end, surface, &[])
            }),
            Rank::B => section.start_uv_b.zip(section.end_uv_b).unwrap_or_else(|| {
                uv_endpoints_from_pcurve(pcurve_on_face, section.start, section.end, surface, &[])
            }),
        };

        forward_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
            source_edge_idx: None,
            // Preserve the section's pave_block_id so cross-face edge
            // sharing (box face inner wire ↔ cylinder face outer wire)
            // works through `resolve_edge_vertices`'s PaveBlock path.
            pave_block_id: section.pave_block_id,
        });
    }

    // Group edges into closed loops by chaining: edge.end_3d approx next.start_3d.
    let mut used = vec![false; forward_edges.len()];
    let mut loops: Vec<Vec<OrientedPCurveEdge>> = Vec::new();

    for start_idx in 0..forward_edges.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let mut chain = vec![forward_edges[start_idx].clone()];
        let loop_start_3d = chain[0].start_3d;

        loop {
            let last_end = chain.last().map_or(loop_start_3d, |e| e.end_3d);

            // Check if the loop is closed (includes single-edge circles
            // where start ~= end).
            if (last_end - loop_start_3d).length() < tol_3d * 100.0 {
                break;
            }

            // Find the next unused edge connecting to last_end. Section
            // edges arrive with arbitrary orientation (each face-pair
            // emits its own direction), so reversed matches are accepted
            // and flipped into chain order.
            let next = forward_edges.iter().enumerate().find_map(|(i, e)| {
                if used[i] {
                    None
                } else if (e.start_3d - last_end).length() < tol_3d * 100.0 {
                    Some((i, false))
                } else if (e.end_3d - last_end).length() < tol_3d * 100.0 {
                    Some((i, true))
                } else {
                    None
                }
            });

            if let Some((idx, rev)) = next {
                used[idx] = true;
                let mut e = forward_edges[idx].clone();
                if rev {
                    std::mem::swap(&mut e.start_uv, &mut e.end_uv);
                    std::mem::swap(&mut e.start_3d, &mut e.end_3d);
                    e.forward = !e.forward;
                }
                chain.push(e);
            } else {
                break; // Can't continue -- open chain.
            }
        }

        // Accept only closed chains (single-edge circles or multi-edge
        // closed loops). Reject open chains from orphaned arcs.
        let chain_end = chain.last().map_or(loop_start_3d, |e| e.end_3d);
        if !chain.is_empty() && (chain_end - loop_start_3d).length() < tol_3d * 100.0 {
            loops.push(chain);
        }
    }

    log::debug!(
        "split_face_with_internal_loops: face {face_id:?} {} sections -> {} loops (sizes {:?})",
        sections.len(),
        loops.len(),
        loops.iter().map(Vec::len).collect::<Vec<_>>()
    );

    // Deepened-opening union: a later cut's internal loop can OVERLAP an
    // existing inner wire (the cutter re-opens an area an earlier cut already
    // opened, offset slightly — the snapClip slot stack's 0.01 ledge margins).
    // Emitting loop and hole as independent wires double-covers the overlap
    // band: both rims stay as unpaired edges and the collinear band pieces
    // trace twice. Merge each such pair instead: the frame keeps ONE union
    // outline, and the removable disc is bounded by the loop pieces outside
    // the hole plus the hole pieces inside the loop.
    let mut consumed_holes: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut union_hole_by_loop: Vec<Option<Vec<OrientedPCurveEdge>>> = vec![None; loops.len()];
    if let FaceSurface::Plane { normal, .. } = surface {
        // Stored UVs on hole wires can be fitted in a foreign frame; build ONE
        // local frame and project every 3D endpoint through it for all 2D
        // tests (the pcurve-convention lesson).
        let frame = PlaneFrame::from_plane_face(*normal, _wire_pts);
        for (li, loop_edges) in loops.iter_mut().enumerate() {
            let mut hit: Option<usize> = None;
            for (hi, hole) in original_inner_wires.iter().enumerate() {
                if consumed_holes.contains(&hi) {
                    continue;
                }
                if internal_loop_hole_interact(loop_edges, hole, &frame) {
                    if hit.is_some() {
                        hit = None; // >1 overlapping hole: bail to current behavior
                        break;
                    }
                    hit = Some(hi);
                }
            }
            if let Some(hi) = hit
                && let Some((disc, union_hole)) = union_internal_loop_with_hole(
                    loop_edges,
                    &original_inner_wires[hi],
                    &frame,
                    tol_3d,
                )
            {
                *loop_edges = disc;
                union_hole_by_loop[li] = Some(union_hole);
                consumed_holes.insert(hi);
            }
        }
    }

    let mut result = Vec::new();

    // For each closed loop: create an "inside" sub-face.
    // The loop winding determines which region of the face is enclosed.
    // We want the SMALLER region (the Steinmetz lobe), so check signed area
    // in UV and reverse if the loop encloses the larger region.
    let mut all_holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
    for (li, loop_edges) in loops.iter_mut().enumerate() {
        // Compute signed area in UV. For single-edge closed curves
        // (circles), sample points along the pcurve since start_uv ~= end_uv
        // gives zero area with just the endpoints.
        let signed_area = if loop_edges.len() == 1 {
            // For single-edge closed curves (circles), sample UV points
            // along the 3D curve and project to UV. The pcurve evaluation
            // gives proper UV coordinates for the full circle.
            let edge = &loop_edges[0];
            let n = 32;
            let mut area = 0.0;
            for k in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let t_cur = k as f64 / n as f64;
                #[allow(clippy::cast_precision_loss)]
                let t_next = (k + 1) as f64 / n as f64;
                let uv0 = edge.pcurve.evaluate(t_cur);
                let uv1 = edge.pcurve.evaluate(t_next);
                area += (uv1.x() - uv0.x()) * (uv1.y() + uv0.y());
            }
            area
        } else {
            let mut area = 0.0;
            for edge in loop_edges.iter() {
                area +=
                    (edge.end_uv.x() - edge.start_uv.x()) * (edge.end_uv.y() + edge.start_uv.y());
            }
            area
        };
        // If signed area is positive (CW in standard UV), the loop encloses
        // the "right" region. If negative (CCW), it encloses the complement.
        // Heuristic: use signed_area sign directly -- negative means CCW in
        // UV which corresponds to the exterior. Reverse to get interior.
        if signed_area < 0.0 {
            // CCW -> enclosing exterior. Reverse to CW -> interior.
            loop_edges.reverse();
            for edge in loop_edges.iter_mut() {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        }

        // Compute the interior point for the disc sub-face.
        // For closed section curves (circles) that form internal loops,
        // the interior point on the plane can land ON the opposing solid's
        // coplanar boundary face, causing ambiguous ray-cast classification.
        // Offset the point slightly along the face normal to break the tie.
        let disc_interior = {
            // Sample 3D points along every loop edge to find the loop's
            // centroid (the circle center for single-circle loops, the
            // polygon centroid for multi-Line footprint loops).
            let n_samples = 16;
            let mut sum = brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0);
            let mut count = 0_usize;
            for edge in loop_edges.iter() {
                let (t0, t1) = edge
                    .curve_3d
                    .domain_with_endpoints(edge.start_3d, edge.end_3d);
                for k in 0..n_samples {
                    #[allow(clippy::cast_precision_loss)]
                    let t = t0 + (t1 - t0) * (k as f64 / n_samples as f64);
                    let pt = edge
                        .curve_3d
                        .evaluate_with_endpoints(t, edge.start_3d, edge.end_3d);
                    sum += brepkit_math::vec::Vec3::new(pt.x(), pt.y(), pt.z());
                    count += 1;
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let centroid = Point3::new(
                sum.x() / count as f64,
                sum.y() / count as f64,
                sum.z() / count as f64,
            );
            // Offset along the face normal by a small amount to ensure
            // the point is clearly inside the opposing solid (not on the
            // coplanar boundary). Use the surface normal direction.
            let normal_offset = match &surface {
                FaceSurface::Plane { normal, .. } => {
                    let n = if reversed { -*normal } else { *normal };
                    // Offset INTO the solid (opposite to the face normal).
                    brepkit_math::vec::Vec3::new(-n.x(), -n.y(), -n.z()) * 1e-6
                }
                _ => brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0),
            };
            Point3::new(
                centroid.x() + normal_offset.x(),
                centroid.y() + normal_offset.y(),
                centroid.z() + normal_offset.z(),
            )
        };

        // The loop as outer wire of the inside sub-face.
        result.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: loop_edges.clone(),
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(disc_interior),
        });

        // Build the outside sub-face's hole: the merged union outline when
        // this loop consumed an overlapping pre-existing hole, otherwise the
        // reversed loop.
        let hole: Vec<OrientedPCurveEdge> = if let Some(u) = union_hole_by_loop[li].take() {
            // Normalize to hole winding (CCW in UV — opposite of the CW
            // interior-enclosing convention applied to the loop above).
            let area: f64 = u
                .iter()
                .map(|e| (e.end_uv.x() - e.start_uv.x()) * (e.end_uv.y() + e.start_uv.y()))
                .sum();
            let mut u = u;
            if area > 0.0 {
                u.reverse();
                for edge in &mut u {
                    std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                    std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                    edge.forward = !edge.forward;
                }
            }
            u
        } else {
            loop_edges
                .iter()
                .rev()
                .map(|e| OrientedPCurveEdge {
                    curve_3d: e.curve_3d.clone(),
                    pcurve: e.pcurve.clone(),
                    start_uv: e.end_uv,
                    end_uv: e.start_uv,
                    start_3d: e.end_3d,
                    end_3d: e.start_3d,
                    forward: !e.forward,
                    source_edge_idx: None,
                    pave_block_id: None,
                })
                .collect()
        };
        // Verify hole is closed.
        if let (Some(first), Some(last)) = (hole.first(), hole.last())
            && (last.end_3d - first.start_3d).length() < tol_3d * 100.0
        {
            all_holes.push(hole);
        }
    }

    // For all-Line hole loops, compute the frame interior point in 3D:
    // midway between the longest outer boundary edge's midpoint and its
    // projection onto the hole polyline. The UV-based hole avoidance in
    // `interior_point_3d` can miss multi-Line holes (sampled hole polygon
    // in a mismatched frame), which classifies the frame as inside the
    // opposing solid and silently drops the whole cap.
    let all_line_holes = !all_holes.is_empty()
        && all_holes
            .iter()
            .flatten()
            .all(|e| matches!(e.curve_3d, EdgeCurve::Line));
    let frame_interior = if all_line_holes {
        boundary_edges
            .iter()
            .map(|e| {
                let mid = e.start_3d + (e.end_3d - e.start_3d) * 0.5;
                ((e.end_3d - e.start_3d).length(), mid)
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, outer_mid)| {
                let mut nearest: Option<(f64, Point3)> = None;
                for e in all_holes.iter().flatten() {
                    let dir = e.end_3d - e.start_3d;
                    let len2 = dir.dot(dir);
                    let t = if len2 > 1e-18 {
                        ((outer_mid - e.start_3d).dot(dir) / len2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let foot = e.start_3d + dir * t;
                    let d = (foot - outer_mid).length();
                    if nearest.is_none_or(|(dn, _)| d < dn) {
                        nearest = Some((d, foot));
                    }
                }
                match nearest {
                    Some((_, hp)) => outer_mid + (hp - outer_mid) * 0.5,
                    None => outer_mid,
                }
            })
    } else {
        None
    };

    // The "outside" sub-face: original boundary with all loops as holes.
    // Pre-existing holes (from earlier boolean operations) stay with the
    // outside sub-face — dropping them would leave the faces ringing those
    // holes with free edges. Holes consumed by a deepened-opening union are
    // already represented by their union outline.
    all_holes.extend(
        original_inner_wires
            .iter()
            .enumerate()
            .filter(|(hi, _)| !consumed_holes.contains(hi))
            .map(|(_, h)| h.clone()),
    );
    let remainder = SplitSubFace {
        surface: surface.clone(),
        outer_wire: boundary_edges.to_vec(),
        inner_wires: all_holes,
        reversed,
        parent: face_id,
        rank,
        precomputed_interior: frame_interior,
    };
    // A curved analytic lateral remainder (cylinder/cone) keeps its hole loops
    // as CURVED edges (e.g. the closed ellipses where two perpendicular
    // cylinders cross), so the all-Line `frame_interior` heuristic above leaves
    // `precomputed_interior` unset. Without it the classifier falls back to
    // sampling the WHOLE parent face — which lands at the axial midpoint inside
    // a lens hole and misclassifies the remainder (a Fuse then drops the entire
    // wall). The generic UV hole-avoidance can't help either: each lens loop is
    // a single closed edge with a degenerate start/end UV, so it forms no usable
    // hole polygon. Sample the wall on a (u,v) grid and pick the point whose
    // nearest 3D distance to every hole loop is greatest — guaranteed on the
    // kept wall, away from every lens.
    let mut remainder = remainder;
    if remainder.precomputed_interior.is_none()
        && matches!(
            remainder.surface,
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
        )
        && !remainder.inner_wires.is_empty()
    {
        remainder.precomputed_interior = cylinder_cone_remainder_interior(&remainder);
    }
    result.push(remainder);

    result
}

/// True when an internal section loop and a pre-existing inner wire (both
/// all-Line, in UV) overlap: a proper edge crossing, a vertex of one strictly
/// inside the other, or a collinear edge-overlap span. Disjoint pairs return
/// false and keep the independent-wires behavior.
fn internal_loop_hole_interact(
    loop_edges: &[OrientedPCurveEdge],
    hole: &[OrientedPCurveEdge],
    frame: &PlaneFrame,
) -> bool {
    let all_line =
        |es: &[OrientedPCurveEdge]| es.iter().all(|e| matches!(e.curve_3d, EdgeCurve::Line));
    if !all_line(loop_edges) || !all_line(hole) {
        return false;
    }
    let segs = |es: &[OrientedPCurveEdge]| -> Vec<(Point2, Point2)> {
        es.iter()
            .map(|e| (frame.project(e.start_3d), frame.project(e.end_3d)))
            .collect()
    };
    let (ls, hs) = (segs(loop_edges), segs(hole));
    let lp: Vec<Point2> = ls.iter().map(|s| s.0).collect();
    let hp: Vec<Point2> = hs.iter().map(|s| s.0).collect();
    let eps = 1e-5;
    for &(a0, a1) in &ls {
        for &(b0, b1) in &hs {
            let (rx, ry) = (a1.x() - a0.x(), a1.y() - a0.y());
            let (sx, sy) = (b1.x() - b0.x(), b1.y() - b0.y());
            let denom = rx.mul_add(sy, -(ry * sx));
            let scale = (rx.hypot(ry) * sx.hypot(sy)).max(f64::MIN_POSITIVE);
            if denom.abs() > 1e-9 * scale {
                let (qx, qy) = (b0.x() - a0.x(), b0.y() - a0.y());
                let t = qx.mul_add(sy, -(qy * sx)) / denom;
                let u = qx.mul_add(ry, -(qy * rx)) / denom;
                if t > 1e-6 && t < 1.0 - 1e-6 && u > 1e-6 && u < 1.0 - 1e-6 {
                    return true;
                }
            } else {
                // Parallel: collinear overlap counts as interaction.
                let la = rx.hypot(ry).max(f64::MIN_POSITIVE);
                let d0 = ((b0.x() - a0.x()) * ry - (b0.y() - a0.y()) * rx).abs() / la;
                if d0 < eps {
                    let proj =
                        |p: Point2| ((p.x() - a0.x()) * rx + (p.y() - a0.y()) * ry) / (la * la);
                    let (u0, u1) = (proj(b0), proj(b1));
                    let (lo, hi) = (u0.min(u1).max(0.0), u0.max(u1).min(1.0));
                    if hi - lo > eps / la {
                        return true;
                    }
                }
            }
        }
    }
    let strictly_inside = |p: Point2, poly: &[Point2], other: &[(Point2, Point2)]| -> bool {
        let on = other.iter().any(|&(s0, s1)| {
            let d = point_seg_dist_2d(p, s0, s1);
            d < eps
        });
        !on && super::super::classify_2d::point_in_polygon_2d(p, poly)
    };
    lp.iter().any(|&p| strictly_inside(p, &hp, &hs))
        || hp.iter().any(|&p| strictly_inside(p, &lp, &ls))
}

fn point_seg_dist_2d(p: Point2, a: Point2, b: Point2) -> f64 {
    let (dx, dy) = (b.x() - a.x(), b.y() - a.y());
    let len2 = dx.mul_add(dx, dy * dy);
    if len2 < 1e-30 {
        return (p.x() - a.x()).hypot(p.y() - a.y());
    }
    let t = ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let (fx, fy) = (a.x() + dx * t, a.y() + dy * t);
    (p.x() - fx).hypot(p.y() - fy)
}

/// Merge an internal section loop with the pre-existing inner wire it
/// overlaps (both all-Line). Returns `(disc_outer, union_hole)`:
///
/// - `disc_outer` bounds the removable region `loop \ hole` — the loop pieces
///   outside the hole plus the hole pieces inside the loop;
/// - `union_hole` is the merged opening outline `loop ∪ hole` — the pieces of
///   both outside each other, with collinear overlap spans contributed once
///   (the hole's copy, so the pieces pair with the pave-split neighbor faces).
///
/// Any inconsistency (chain fails to close, either chain empty, leftover
/// pieces) returns None and the caller keeps the current independent-wires
/// emission.
#[allow(clippy::too_many_lines)]
fn union_internal_loop_with_hole(
    loop_edges: &[OrientedPCurveEdge],
    hole: &[OrientedPCurveEdge],
    frame: &PlaneFrame,
    tol: f64,
) -> Option<(Vec<OrientedPCurveEdge>, Vec<OrientedPCurveEdge>)> {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::Vec2;
    #[derive(PartialEq, Clone, Copy)]
    enum Verdict {
        In,
        On,
        Out,
    }
    let eps = 1e-5;
    let seg_of = |e: &OrientedPCurveEdge| (frame.project(e.start_3d), frame.project(e.end_3d));
    let lsegs: Vec<(Point2, Point2)> = loop_edges.iter().map(seg_of).collect();
    let hsegs: Vec<(Point2, Point2)> = hole.iter().map(seg_of).collect();
    let lp: Vec<Point2> = lsegs.iter().map(|s| s.0).collect();
    let hp: Vec<Point2> = hsegs.iter().map(|s| s.0).collect();

    let mk_piece = |e: &OrientedPCurveEdge, t0: f64, t1: f64| -> Option<OrientedPCurveEdge> {
        let lerp3 = |t: f64| e.start_3d + (e.end_3d - e.start_3d) * t;
        let (s3, e3) = (lerp3(t0), lerp3(t1));
        let (s_uv, e_uv) = (frame.project(s3), frame.project(e3));
        if (e3 - s3).length() < tol {
            return None;
        }
        let d = Vec2::new(e_uv.x() - s_uv.x(), e_uv.y() - s_uv.y());
        let len = d.x().hypot(d.y());
        let dir = if len > 1e-12 {
            Vec2::new(d.x() / len, d.y() / len)
        } else {
            Vec2::new(1.0, 0.0)
        };
        let pcurve = Curve2D::Line(
            Line2D::new(s_uv, dir)
                .or_else(|_| Line2D::new(s_uv, Vec2::new(1.0, 0.0)))
                .ok()?,
        );
        Some(OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: s_uv,
            end_uv: e_uv,
            start_3d: s3,
            end_3d: e3,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        })
    };

    // Split parameters on `edge` from proper crossings with the other set's
    // segments plus the other outline's vertices lying on the edge interior
    // (T-junctions and collinear overlap ends).
    let split_ts =
        |a0: Point2, a1: Point2, other: &[(Point2, Point2)], verts: &[Point2]| -> Vec<f64> {
            let (rx, ry) = (a1.x() - a0.x(), a1.y() - a0.y());
            let la = rx.hypot(ry).max(f64::MIN_POSITIVE);
            let mut ts = vec![0.0, 1.0];
            for &(b0, b1) in other {
                let (sx, sy) = (b1.x() - b0.x(), b1.y() - b0.y());
                let denom = rx.mul_add(sy, -(ry * sx));
                let scale = (la * sx.hypot(sy)).max(f64::MIN_POSITIVE);
                if denom.abs() <= 1e-9 * scale {
                    continue;
                }
                let (qx, qy) = (b0.x() - a0.x(), b0.y() - a0.y());
                let t = qx.mul_add(sy, -(qy * sx)) / denom;
                let u = qx.mul_add(ry, -(qy * rx)) / denom;
                if t > 1e-9 && t < 1.0 - 1e-9 && u > -1e-6 && u < 1.0 + 1e-6 {
                    ts.push(t);
                }
            }
            for &v in verts {
                if point_seg_dist_2d(v, a0, a1) < eps {
                    let t = ((v.x() - a0.x()) * rx + (v.y() - a0.y()) * ry) / (la * la);
                    if t > 1e-9 && t < 1.0 - 1e-9 {
                        ts.push(t);
                    }
                }
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ts.dedup_by(|a, b| (*a - *b).abs() * la < tol * 10.0);
            ts
        };

    let classify = |mid: Point2, poly: &[Point2], segs: &[(Point2, Point2)]| -> Verdict {
        if segs
            .iter()
            .any(|&(s0, s1)| point_seg_dist_2d(mid, s0, s1) < eps)
        {
            Verdict::On
        } else if super::super::classify_2d::point_in_polygon_2d(mid, poly) {
            Verdict::In
        } else {
            Verdict::Out
        }
    };

    let mut union_pieces: Vec<OrientedPCurveEdge> = Vec::new();
    let mut disc_pieces: Vec<OrientedPCurveEdge> = Vec::new();
    for (e, &(s0, s1)) in loop_edges.iter().zip(&lsegs) {
        let ts = split_ts(s0, s1, &hsegs, &hp);
        for w in ts.windows(2) {
            let Some(piece) = mk_piece(e, w[0], w[1]) else {
                continue;
            };
            let mid = Point2::new(
                (piece.start_uv.x() + piece.end_uv.x()) * 0.5,
                (piece.start_uv.y() + piece.end_uv.y()) * 0.5,
            );
            if classify(mid, &hp, &hsegs) == Verdict::Out {
                union_pieces.push(piece.clone());
                disc_pieces.push(piece);
            }
            // In: air inside the old opening. On: the hole's copy is kept.
        }
    }
    for (e, &(s0, s1)) in hole.iter().zip(&hsegs) {
        let ts = split_ts(s0, s1, &lsegs, &lp);
        for w in ts.windows(2) {
            let Some(piece) = mk_piece(e, w[0], w[1]) else {
                continue;
            };
            let mid = Point2::new(
                (piece.start_uv.x() + piece.end_uv.x()) * 0.5,
                (piece.start_uv.y() + piece.end_uv.y()) * 0.5,
            );
            match classify(mid, &lp, &lsegs) {
                Verdict::In => disc_pieces.push(piece),
                Verdict::On | Verdict::Out => union_pieces.push(piece),
            }
        }
    }

    let disc = chain_single_closed_loop(disc_pieces, tol)?;
    let union_hole = chain_single_closed_loop(union_pieces, tol)?;
    Some((disc, union_hole))
}

/// Chain pieces into exactly one closed loop consuming every piece; None on
/// any leftover, open chain, or fewer than 3 pieces.
fn chain_single_closed_loop(
    mut pieces: Vec<OrientedPCurveEdge>,
    tol: f64,
) -> Option<Vec<OrientedPCurveEdge>> {
    let close = tol * 100.0;
    if pieces.len() < 3 {
        return None;
    }
    let mut chain = vec![pieces.swap_remove(0)];
    let start = chain[0].start_3d;
    while !pieces.is_empty() {
        let last_end = chain.last()?.end_3d;
        let next = pieces
            .iter()
            .position(|e| (e.start_3d - last_end).length() < close);
        let next_rev = if next.is_none() {
            pieces
                .iter()
                .position(|e| (e.end_3d - last_end).length() < close)
        } else {
            None
        };
        if let Some(i) = next {
            chain.push(pieces.swap_remove(i));
        } else if let Some(i) = next_rev {
            let mut e = pieces.swap_remove(i);
            std::mem::swap(&mut e.start_uv, &mut e.end_uv);
            std::mem::swap(&mut e.start_3d, &mut e.end_3d);
            e.forward = !e.forward;
            chain.push(e);
        } else {
            return None;
        }
    }
    ((chain.last()?.end_3d - start).length() < close).then_some(chain)
}

/// Interior point on a cylinder/cone lateral remainder face that carries
/// CURVED hole loops (the lens loops where another quadric crosses the wall).
///
/// The generic UV hole-avoidance fails for these — each lens is a single closed
/// edge whose start/end UV coincide, so it yields no point-in-polygon hole. This
/// instead samples the wall on a `(u, v)` grid (`u` over the full revolution,
/// `v` over the boundary's axial span) and returns the 3D point whose minimum
/// distance to every hole loop is largest, which is guaranteed to lie on the
/// kept wall well clear of every lens. Returns `None` if the surface evaluator
/// or boundary v-range is unusable, so the caller keeps the unset interior.
pub fn cylinder_cone_remainder_interior(remainder: &SplitSubFace) -> Option<Point3> {
    use std::f64::consts::{PI, TAU};
    // Densely sample every hole loop into 3D points AND project each to a (u, v)
    // polyline so candidate interior points can be tested for face containment
    // (outside every hole). Endpoint UVs are unusable here — a lens hole is one
    // closed edge whose start/end UV coincide — so sample the curve.
    let mut hole_pts: Vec<Point3> = Vec::new();
    // Inner-loop (u, v) segments (split at the seam), for the even-odd
    // vertical-ray hole-containment test below.
    let mut hole_segs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let n = 48;
    for hole in &remainder.inner_wires {
        let mut prev_uv: Option<(f64, f64)> = None;
        for edge in hole {
            let (t0, t1) = edge
                .curve_3d
                .domain_with_endpoints(edge.start_3d, edge.end_3d);
            for k in 0..=n {
                #[allow(clippy::cast_precision_loss)]
                let t = t0 + (t1 - t0) * (k as f64 / f64::from(n));
                let p = edge
                    .curve_3d
                    .evaluate_with_endpoints(t, edge.start_3d, edge.end_3d);
                hole_pts.push(p);
                if let Some((u, v)) = remainder.surface.project_point(p) {
                    if let Some((pu, pv)) = prev_uv {
                        // Unwrap `u` relative to the previous sample so a
                        // seam-crossing step is a single continuous segment in an
                        // extended u-frame (`pu → pu + wrapped_delta`, |Δ| ≤ π)
                        // instead of being dropped. `point_in_hole_loops_uv` tests
                        // every 2π translate, so a segment living past the seam is
                        // still matched for a query on the other side — the lens
                        // boundary stays closed.
                        let wrapped_delta = ((u - pu + PI).rem_euclid(TAU)) - PI;
                        let u_unwrapped = pu + wrapped_delta;
                        let (a_u, a_v, b_u, b_v) = if pu <= u_unwrapped {
                            (pu, pv, u_unwrapped, v)
                        } else {
                            (u_unwrapped, v, pu, pv)
                        };
                        hole_segs.push((a_u, a_v, b_u, b_v));
                    }
                    prev_uv = Some((u, v));
                }
            }
        }
    }
    if hole_pts.is_empty() {
        return None;
    }

    // Angular (u) and axial (v) extent the boundary wire actually covers —
    // densely sampled so a partial-arc face (e.g. a rounded-rect corner
    // quarter-cylinder) is searched only over its OWN span, not the full
    // revolution (a 0..TAU grid could pick an interior point off the sub-face,
    // mis-classifying it). A full-revolution wall covers the whole period and
    // its boundary samples span it.
    let mut u_samples: Vec<f64> = Vec::new();
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for e in &remainder.outer_wire {
        let (t0, t1) = e.curve_3d.domain_with_endpoints(e.start_3d, e.end_3d);
        let n = 16;
        for k in 0..=n {
            #[allow(clippy::cast_precision_loss)]
            let t = t0 + (t1 - t0) * (k as f64 / f64::from(n));
            let p = e.curve_3d.evaluate_with_endpoints(t, e.start_3d, e.end_3d);
            if let Some((u, v)) = remainder.surface.project_point(p) {
                u_samples.push(u);
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
    }
    if !v_min.is_finite() || !v_max.is_finite() || (v_max - v_min) <= 0.0 || u_samples.is_empty() {
        return None;
    }

    // The covered u-interval `[u_lo, u_lo + u_span]`: the complement of the
    // largest gap between sorted u-samples. `u_span ≈ TAU` is a full revolution
    // (search the whole period); a smaller span restricts the grid to the arc
    // the face occupies.
    let (u_lo, u_span) = covered_u_interval(&u_samples);

    // Grid-search (u, v) for the wall point maximising the minimum 3D distance
    // to the hole loops, over the face's actual extent only, accepting ONLY
    // candidates that are face-CONTAINED (inside the covered u/v extent — by
    // construction — AND outside every lens hole). A point outside the sub-face
    // fed to classification would drop valid curved wall faces, so an
    // uncontained best is rejected.
    let search = |n_u: u32, n_v: u32| -> Option<Point3> {
        let mut best: Option<(f64, Point3)> = None;
        for iu in 0..n_u {
            #[allow(clippy::cast_precision_loss)]
            let u = u_lo + u_span * f64::from(iu) / f64::from(n_u);
            for iv in 1..n_v {
                #[allow(clippy::cast_precision_loss)]
                let v = v_min + (v_max - v_min) * f64::from(iv) / f64::from(n_v);
                if point_in_hole_loops_uv(&hole_segs, u, v) {
                    continue; // Inside a lens hole — not the kept region.
                }
                let Some(p) = remainder.surface.evaluate(u, v) else {
                    continue;
                };
                let min_d = hole_pts
                    .iter()
                    .map(|h| (*h - p).length())
                    .fold(f64::INFINITY, f64::min);
                if best.is_none_or(|(bd, _)| min_d > bd) {
                    best = Some((min_d, p));
                }
            }
        }
        best.map(|(_, p)| p)
    };

    // First pass at the standard density (the census finds a contained interior
    // here). If a thin kept strip or a close pair of lens loops yields nothing
    // contained at this resolution, take ONE refined denser pass before giving
    // up — a present-but-narrow remainder strip then resolves. Returning `None`
    // (no contained point even dense) signals the caller to abort the analytic
    // split rather than fall back to an uncontained generic interior point.
    search(48, 5).or_else(|| search(256, 17))
}

/// Whether `(u, v)` lies inside the region bounded by the combined inner-loop
/// `(u, v)` segments — an even-odd vertical-ray test. Each segment is tested in
/// every 2π `u`-translate so a seam-wrapping lens loop is matched. A wall
/// interior point must be OUTSIDE every hole (this returns `false`) to lie in
/// the kept remainder region.
fn point_in_hole_loops_uv(hole_segs: &[(f64, f64, f64, f64)], u: f64, v: f64) -> bool {
    use std::f64::consts::TAU;
    let seg_u_min = hole_segs.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
    let seg_u_max = hole_segs
        .iter()
        .map(|s| s.2)
        .fold(f64::NEG_INFINITY, f64::max);
    if !seg_u_min.is_finite() {
        return false;
    }
    #[allow(clippy::cast_possible_truncation)]
    let k0 = ((seg_u_min - u) / TAU).floor() as i64;
    #[allow(clippy::cast_possible_truncation)]
    let k1 = ((seg_u_max - u) / TAU).ceil() as i64;
    let mut crossings = 0u32;
    for k in k0..=k1 {
        #[allow(clippy::cast_precision_loss)]
        let uu = u + (k as f64) * TAU;
        for &(a_u, a_v, b_u, b_v) in hole_segs {
            if uu >= a_u && uu < b_u && (b_u - a_u) > 1e-15 {
                let t = (uu - a_u) / (b_u - a_u);
                if a_v + t * (b_v - a_v) > v {
                    crossings += 1;
                }
            }
        }
    }
    crossings % 2 == 1
}

/// Given angular samples (radians, any 2π window), return `(u_lo, u_span)` —
/// the contiguous interval the samples cover, computed as the complement of the
/// largest angular gap between consecutive sorted samples. A full revolution
/// returns `u_span ≈ TAU`; a partial arc returns its true (shorter) span.
fn covered_u_interval(u_samples: &[f64]) -> (f64, f64) {
    use std::f64::consts::TAU;
    let mut us: Vec<f64> = u_samples.iter().map(|u| u.rem_euclid(TAU)).collect();
    us.sort_by(f64::total_cmp);
    us.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    if us.len() < 2 {
        return (us.first().copied().unwrap_or(0.0), TAU);
    }
    // Largest gap between consecutive samples (including the wrap gap).
    let mut max_gap = us[0] + TAU - us[us.len() - 1]; // wrap-around gap
    let mut gap_after = us[us.len() - 1]; // the sample BEFORE the wrap gap
    for w in us.windows(2) {
        let gap = w[1] - w[0];
        if gap > max_gap {
            max_gap = gap;
            gap_after = w[0];
        }
    }
    // The covered interval starts just after the largest gap and spans the rest.
    let u_lo = (gap_after + max_gap).rem_euclid(TAU);
    (u_lo, TAU - max_gap)
}

/// Reorder and reverse boundary edges to form a closed chain.
#[allow(clippy::expect_used)]
pub(super) fn chain_boundary_edges(
    edges: Vec<OrientedPCurveEdge>,
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    if edges.len() < 2 {
        return edges;
    }
    let mut remaining: Vec<Option<OrientedPCurveEdge>> = edges.into_iter().map(Some).collect();
    let mut chain = Vec::with_capacity(remaining.len());
    chain.push(remaining[0].take().expect("first edge"));
    for _ in 0..remaining.len() {
        let tail = chain.last().expect("non-empty").end_3d;
        let mut best_idx = None;
        let mut best_reversed = false;
        let mut best_dist = f64::MAX;
        for (i, opt) in remaining.iter().enumerate() {
            let Some(e) = opt else { continue };
            let d_fwd = (tail - e.start_3d).length();
            if d_fwd < best_dist {
                best_dist = d_fwd;
                best_idx = Some(i);
                best_reversed = false;
            }
            let d_rev = (tail - e.end_3d).length();
            if d_rev < best_dist {
                best_dist = d_rev;
                best_idx = Some(i);
                best_reversed = true;
            }
        }
        if best_dist > tol * 100.0 {
            break;
        }
        if let Some(idx) = best_idx {
            let mut e = remaining[idx].take().expect("edge");
            if best_reversed {
                std::mem::swap(&mut e.start_uv, &mut e.end_uv);
                std::mem::swap(&mut e.start_3d, &mut e.end_3d);
                e.forward = !e.forward;
            }
            chain.push(e);
        }
    }
    for e in remaining.into_iter().flatten() {
        chain.push(e);
    }
    chain
}

/// Split a plane face with crossing section edges into 4 quadrant sub-faces.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(super) fn try_split_crossing_plane_face(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    frame: &PlaneFrame,
    tol: &brepkit_math::tolerance::Tolerance,
) -> Option<Vec<SplitSubFace>> {
    let cross_3d;
    let section_endpoints: Vec<Point3>;

    if sections.len() == 2 {
        let (s0, s1) = (&sections[0], &sections[1]);
        let d0 = s0.end - s0.start;
        let d1 = s1.end - s1.start;
        if d0.length() < tol.linear || d1.length() < tol.linear {
            return None;
        }
        let normal = d0.cross(d1);
        let ptol = d0.length() * d1.length() * tol.linear;
        if normal.x().abs() < ptol && normal.y().abs() < ptol && normal.z().abs() < ptol {
            return None;
        }
        let d = s1.start - s0.start;
        let ax = normal.x().abs();
        let ay = normal.y().abs();
        let az = normal.z().abs();
        #[allow(clippy::similar_names)]
        let t0 = if az >= ax && az >= ay {
            let det = d0.x().mul_add(d1.y(), -(d0.y() * d1.x()));
            if det.abs() < ptol {
                return None;
            }
            d.x().mul_add(d1.y(), -(d.y() * d1.x())) / det
        } else if ay >= ax {
            let det = d0.x().mul_add(d1.z(), -(d0.z() * d1.x()));
            if det.abs() < ptol {
                return None;
            }
            d.x().mul_add(d1.z(), -(d.z() * d1.x())) / det
        } else {
            let det = d0.y().mul_add(d1.z(), -(d0.z() * d1.y()));
            if det.abs() < ptol {
                return None;
            }
            d.y().mul_add(d1.z(), -(d.z() * d1.y())) / det
        };
        cross_3d = s0.start + d0 * t0;
        let t1 = (cross_3d - s1.start).dot(d1) / d1.dot(d1);
        // The infinite lines must meet within both segments (endpoints allowed).
        if !(-0.01..=1.01).contains(&t0) || !(-0.01..=1.01).contains(&t1) {
            return None;
        }
        let mid = |t: f64| (0.01..=0.99).contains(&t);
        section_endpoints = if mid(t0) && mid(t1) {
            // X-crossing: both sections split mid-way → 4 regions.
            vec![s0.start, s0.end, s1.start, s1.end]
        } else if mid(t1) {
            // T-junction: s0's endpoint lands mid-way on s1 → 3 regions.
            // `cross_3d` is that endpoint; keep s0's far end + s1's two ends.
            let s0_far = if t0 < 0.5 { s0.end } else { s0.start };
            vec![s0_far, s1.start, s1.end]
        } else if mid(t0) {
            let s1_far = if t1 < 0.5 { s1.end } else { s1.start };
            vec![s1_far, s0.start, s0.end]
        } else {
            // L-junction (shared endpoint) or no interior crossing — no split.
            return None;
        };
    } else if sections.len() == 4 {
        let all_pts: Vec<Point3> = sections.iter().flat_map(|s| [s.start, s.end]).collect();
        let mut common = None;
        for &pt in &all_pts {
            let count = all_pts
                .iter()
                .filter(|&&o| (o - pt).length() < tol.linear * 10.0)
                .count();
            if count >= 4 {
                common = Some(pt);
                break;
            }
        }
        let cp = common?;
        cross_3d = cp;
        section_endpoints = all_pts
            .into_iter()
            .filter(|&pt| (pt - cp).length() > tol.linear * 10.0)
            .collect();
        if section_endpoints.len() != 4 {
            return None;
        }
        let dirs: Vec<_> = sections
            .iter()
            .map(|s| {
                let other = if (s.start - cp).length() < tol.linear * 10.0 {
                    s.end
                } else {
                    s.start
                };
                let d = other - cp;
                let l = d.length();
                if l > 1e-12 { d * (1.0 / l) } else { d }
            })
            .collect();
        let mut matched = [false; 4];
        let mut groups = 0u32;
        for i in 0..4 {
            if matched[i] {
                continue;
            }
            for j in (i + 1)..4 {
                if !matched[j] && dirs[i].dot(dirs[j]) < -0.9 {
                    matched[i] = true;
                    matched[j] = true;
                    groups += 1;
                    break;
                }
            }
        }
        if groups != 2 {
            return None;
        }
    } else {
        return None;
    }

    // Verify the crossing point is in the face INTERIOR (not on a boundary edge).
    // For fuse, sections meet at a boundary vertex — splitting would be wrong.
    let on_boundary = boundary_edges.iter().any(|e| {
        let to_pt = cross_3d - e.start_3d;
        let edge_dir = e.end_3d - e.start_3d;
        let edge_len = edge_dir.length();
        if edge_len < tol.linear {
            return (cross_3d - e.start_3d).length() < tol.linear;
        }
        let t = to_pt.dot(edge_dir) / (edge_len * edge_len);
        if !(-0.01..=1.01).contains(&t) {
            return false;
        }
        let closest = e.start_3d + edge_dir * t.clamp(0.0, 1.0);
        (cross_3d - closest).length() < tol.linear * 10.0
    });
    if on_boundary {
        return None;
    }

    let split_boundary = split_boundary_edges_at_3d_points(
        boundary_edges.to_vec(),
        &section_endpoints,
        Some(frame),
        surface,
        tol.linear,
    );
    let split_boundary = chain_boundary_edges(split_boundary, tol.linear);
    let find_idx = |pt: Point3| -> Option<usize> {
        split_boundary
            .iter()
            .position(|e| (e.start_3d - pt).length() < tol.linear * 100.0)
    };
    let mut section_indices = Vec::with_capacity(4);
    for &pt in &section_endpoints {
        section_indices.push(find_idx(pt)?);
    }
    section_indices.sort_unstable();
    section_indices.dedup();
    if section_indices.len() != section_endpoints.len() || section_indices.len() < 3 {
        return None;
    }

    let n = split_boundary.len();
    let make_edge = |start: Point3, end: Point3| -> OrientedPCurveEdge {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::vec::Vec2;
        let su = frame.project(start);
        let eu = frame.project(end);
        let dir = eu - su;
        let len = dir.length();
        let direction = if len > 1e-12 {
            Vec2::new(dir.x() / len, dir.y() / len)
        } else {
            Vec2::new(1.0, 0.0)
        };
        #[allow(clippy::expect_used)]
        let pcurve = Curve2D::Line(
            Line2D::new(su, direction)
                .or_else(|_| Line2D::new(su, Vec2::new(1.0, 0.0)))
                .expect("unit direction"),
        );
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: su,
            end_uv: eu,
            start_3d: start,
            end_3d: end,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        }
    };

    let n_regions = section_indices.len();
    let mut result = Vec::new();
    for qi in 0..n_regions {
        let arc_start = section_indices[qi];
        let arc_end = section_indices[(qi + 1) % n_regions];
        let mut wire = Vec::new();
        let mut idx = arc_start;
        loop {
            wire.push(split_boundary[idx].clone());
            idx = (idx + 1) % n;
            if idx == arc_end || wire.len() > n {
                break;
            }
        }
        wire.push(make_edge(split_boundary[arc_end].start_3d, cross_3d));
        wire.push(make_edge(cross_3d, split_boundary[arc_start].start_3d));
        let wn = wire.len() as f64;
        let sum = wire.iter().fold(Point3::new(0.0, 0.0, 0.0), |acc, e| {
            acc + (e.start_3d - Point3::new(0.0, 0.0, 0.0))
        });
        result.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: Some(Point3::new(sum.x() / wn, sum.y() / wn, sum.z() / wn)),
        });
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::{OrientedPCurveEdge, arc_covers_segment, point_in_hole_loops_uv};
    use brepkit_math::curves::Circle3D;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;
    use std::f64::consts::{PI, TAU};

    fn dummy_pcurve() -> Curve2D {
        Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap())
    }

    fn arc_edge(
        circle: &Circle3D,
        start_angle: f64,
        end_angle: f64,
        forward: bool,
    ) -> OrientedPCurveEdge {
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(circle.clone()),
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(0.0, 0.0),
            start_3d: circle.evaluate(start_angle),
            end_3d: circle.evaluate(end_angle),
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    fn line_chord(start: Point3, end: Point3) -> OrientedPCurveEdge {
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: dummy_pcurve(),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(0.0, 0.0),
            start_3d: start,
            end_3d: end,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    #[test]
    fn arc_covers_chord_within_270_degree_span() {
        // A 270° arc (0 → 3π/2, CCW). A chord whose endpoints lie within
        // that span is covered; a chord in the complementary 90° gap is not.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let arc = arc_edge(&circle, 0.0, 1.5 * PI, true);
        let tol = 1e-7;

        // Chord between angles 0.5π and π — inside the 270° sweep.
        let inside = line_chord(circle.evaluate(0.5 * PI), circle.evaluate(PI));
        assert!(arc_covers_segment(&arc, &inside, tol));

        // Chord between angles 1.6π and 1.9π — inside the complementary
        // 90° gap, which the arc does NOT cover. The old half-turn wrap
        // mistook this complementary short arc for the real one.
        let outside = line_chord(circle.evaluate(1.6 * PI), circle.evaluate(1.9 * PI));
        assert!(!arc_covers_segment(&arc, &outside, tol));
    }

    #[test]
    fn arc_covers_chord_on_reversed_arc() {
        // Same geometry traversed CW (forward = false): the swept region is
        // the complementary 90° arc, so the membership flips.
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let arc = arc_edge(&circle, 0.0, 1.5 * PI, false);
        let tol = 1e-7;

        let in_gap = line_chord(circle.evaluate(1.6 * PI), circle.evaluate(1.9 * PI));
        assert!(arc_covers_segment(&arc, &in_gap, tol));

        let in_long = line_chord(circle.evaluate(0.5 * PI), circle.evaluate(PI));
        assert!(!arc_covers_segment(&arc, &in_long, tol));
    }

    #[test]
    fn band_interior_antipode_wraps_into_domain() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);

        // Seam in (π, 2π): the unwrapped antipode seam_u + π exceeds 2π.
        for &seam_u in &[1.1 * PI, 1.5 * PI, 1.9 * PI] {
            let wrapped = (seam_u + PI).rem_euclid(TAU);
            assert!((0.0..TAU).contains(&wrapped));
            // The wrap is behavior-preserving on a periodic surface: the
            // in-domain parameter evaluates to the same 3D interior point.
            let a = surface.evaluate(seam_u + PI, 3.0).unwrap();
            let b = surface.evaluate(wrapped, 3.0).unwrap();
            assert!((a - b).length() < 1e-9);
        }
    }

    #[test]
    fn hole_containment_even_odd_inside_vs_outside() {
        // A square hole loop in (u, v) around (u=1.0, v=10.0), as the segment
        // list `cylinder_cone_remainder_interior` builds. The remainder
        // interior-point search rejects any candidate INSIDE this hole.
        let (uc, vc, h) = (1.0_f64, 10.0_f64, 0.5_f64);
        let corners = [
            (uc - h, vc - h),
            (uc + h, vc - h),
            (uc + h, vc + h),
            (uc - h, vc + h),
        ];
        let mut segs: Vec<(f64, f64, f64, f64)> = Vec::new();
        for i in 0..4 {
            let (mut a_u, mut a_v) = corners[i];
            let (mut b_u, mut b_v) = corners[(i + 1) % 4];
            if a_u > b_u {
                std::mem::swap(&mut a_u, &mut b_u);
                std::mem::swap(&mut a_v, &mut b_v);
            }
            segs.push((a_u, a_v, b_u, b_v));
        }
        // Centre of the hole → inside.
        assert!(
            point_in_hole_loops_uv(&segs, uc, vc),
            "hole centre is inside"
        );
        // Well outside the hole (different u and v) → outside.
        assert!(
            !point_in_hole_loops_uv(&segs, uc + 2.0, vc),
            "a point clear of the hole in u is outside"
        );
        assert!(
            !point_in_hole_loops_uv(&segs, uc, vc + 2.0),
            "a point clear of the hole in v is outside"
        );
        // No segments → never inside.
        assert!(!point_in_hole_loops_uv(&[], uc, vc));
    }

    #[test]
    fn hole_containment_handles_seam_crossing_loop() {
        // A square hole loop straddling the u-seam: its corners sit at u≈TAU−h
        // and u≈+h (i.e. the loop wraps across u=0). The projection unwraps each
        // step (|Δu| ≤ π) so the loop is a CLOSED boundary in an extended u-frame;
        // a point inside it (at the seam, u=0) must read inside, and a point on
        // the far side (u=π) must read outside. Before the seam-crossing fix the
        // wrap segment was dropped, leaving an open boundary that mis-classified.
        use std::f64::consts::{PI, TAU};
        let vc = 10.0_f64;
        let h = 0.5_f64;
        // Corners walked in order around the seam: (TAU−h, lo) → (h, lo) →
        // (h, hi) → (TAU−h, hi). Build segments by unwrapping like the
        // production projection does.
        let corners = [
            (TAU - h, vc - h),
            (h, vc - h),
            (h, vc + h),
            (TAU - h, vc + h),
        ];
        let mut segs: Vec<(f64, f64, f64, f64)> = Vec::new();
        let mut prev = corners[corners.len() - 1];
        for &(u, v) in &corners {
            let (pu, pv) = prev;
            let wrapped_delta = ((u - pu + PI).rem_euclid(TAU)) - PI;
            let u_un = pu + wrapped_delta;
            let seg = if pu <= u_un {
                (pu, pv, u_un, v)
            } else {
                (u_un, v, pu, pv)
            };
            segs.push(seg);
            prev = (u, v);
        }
        // At the seam (u=0, inside the wrapped loop) → inside.
        assert!(
            point_in_hole_loops_uv(&segs, 0.0, vc),
            "seam-crossing hole contains the seam point"
        );
        // Just inside on the +u side and the wrap side.
        assert!(point_in_hole_loops_uv(&segs, 0.25, vc));
        assert!(point_in_hole_loops_uv(&segs, TAU - 0.25, vc));
        // Far side of the cylinder (u=π) → outside.
        assert!(
            !point_in_hole_loops_uv(&segs, PI, vc),
            "the opposite side of the wall is outside the seam-crossing hole"
        );
        // Outside in v at the seam → outside.
        assert!(!point_in_hole_loops_uv(&segs, 0.0, vc + 2.0));
    }

    #[test]
    fn remainder_interior_point_is_outside_the_lens_hole() {
        // A full cylinder wall (z-axis, r=3, v∈[0,20]) with one closed-circle
        // hole on the wall near (u=0, v=10): the chosen interior point must be
        // OUTSIDE the hole (Finding B — an uncontained point would drop the
        // wall face at classification).
        use super::super::super::split_types::SplitSubFace;
        use crate::ds::Rank;
        use brepkit_topology::topology::Topology;

        // A real FaceId for the `parent` field (never read by the function under
        // test, but the struct requires a valid handle).
        let mut dummy_topo = Topology::new();
        let parent = brepkit_topology::test_utils::make_unit_square_face(&mut dummy_topo);

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);

        // Outer wire: bottom rim (v=0), seam up (u=0), top rim (v=20), seam down.
        let bot = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let top =
            Circle3D::new(Point3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let outer = vec![
            arc_edge(&bot, 0.0, TAU, true),
            line_chord(Point3::new(3.0, 0.0, 0.0), Point3::new(3.0, 0.0, 20.0)),
            arc_edge(&top, 0.0, TAU, true),
            line_chord(Point3::new(3.0, 0.0, 20.0), Point3::new(3.0, 0.0, 0.0)),
        ];

        // Hole: a small circle ON the cylinder wall near (u≈0, v=10). Centre on
        // the surface at (3,0,10); the hole loop is a circle of radius 1 in the
        // tangent plane (u,z) — small enough to stay on the wall band.
        let hole_center = Point3::new(3.0, 0.0, 10.0);
        let hole_normal = Vec3::new(1.0, 0.0, 0.0); // wall outward normal at u=0
        let hole = Circle3D::new(hole_center, hole_normal, 1.0).unwrap();
        let inner = vec![vec![arc_edge(&hole, 0.0, TAU, true)]];

        let sub = SplitSubFace {
            surface,
            outer_wire: outer,
            inner_wires: inner,
            reversed: false,
            parent,
            rank: Rank::A,
            precomputed_interior: None,
        };

        let p = super::cylinder_cone_remainder_interior(&sub).unwrap();
        // The point is ON the cylinder (radius 3 from the axis).
        let radial = (p.x() * p.x() + p.y() * p.y()).sqrt();
        assert!((radial - 3.0).abs() < 1e-6, "interior point on the wall");
        // The point is CLEAR of the hole (well outside its 1-unit circle).
        assert!(
            (p - hole_center).length() > 1.5,
            "interior point must be clear of the hole, got dist {}",
            (p - hole_center).length()
        );
    }

    #[test]
    fn remainder_interior_found_for_thin_kept_strip() {
        // A narrow wall band v∈[8,12] mostly filled by a large lens hole centred
        // at v=10, leaving only thin kept strips near the rims. The two-pass grid
        // (coarse then dense) must still return an ON-WALL point CLEAR of the
        // hole — never `None`, which would abort the analytic split. Exercises
        // robustness of the contained-interior search on a thin remainder.
        use super::super::super::split_types::SplitSubFace;
        use crate::ds::Rank;
        use brepkit_topology::topology::Topology;

        let mut dummy_topo = Topology::new();
        let parent = brepkit_topology::test_utils::make_unit_square_face(&mut dummy_topo);

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);

        // Narrow band v∈[8,12].
        let bot = Circle3D::new(Point3::new(0.0, 0.0, 8.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let top =
            Circle3D::new(Point3::new(0.0, 0.0, 12.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let outer = vec![
            arc_edge(&bot, 0.0, TAU, true),
            line_chord(Point3::new(3.0, 0.0, 8.0), Point3::new(3.0, 0.0, 12.0)),
            arc_edge(&top, 0.0, TAU, true),
            line_chord(Point3::new(3.0, 0.0, 12.0), Point3::new(3.0, 0.0, 8.0)),
        ];

        // Big hole centred at (3,0,10) nearly filling the 4-unit band.
        let hole_center = Point3::new(3.0, 0.0, 10.0);
        let hole = Circle3D::new(hole_center, Vec3::new(1.0, 0.0, 0.0), 1.9).unwrap();
        let inner = vec![vec![arc_edge(&hole, 0.0, TAU, true)]];

        let sub = SplitSubFace {
            surface,
            outer_wire: outer,
            inner_wires: inner,
            reversed: false,
            parent,
            rank: Rank::A,
            precomputed_interior: None,
        };

        // The two-pass search must find a contained point in the thin kept strip.
        let p = super::cylinder_cone_remainder_interior(&sub).unwrap();
        let radial = (p.x() * p.x() + p.y() * p.y()).sqrt();
        assert!((radial - 3.0).abs() < 1e-6, "interior point on the wall");
        assert!(
            (p - hole_center).length() > 1.9,
            "interior point must be outside the hole, got dist {}",
            (p - hole_center).length()
        );
    }
}
