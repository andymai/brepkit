//! Face splitting via 2D wire construction.
//!
//! For each face, collects boundary edges and section edges, converts
//! them to [`OrientedPCurveEdge`]s in the face's parameter space, calls
//! the wire builder, and produces [`SplitSubFace`]s.

mod containment;
mod conversion;
mod edge_splitting;
mod sampling;
mod special_cases;

pub use conversion::collect_wire_points;

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::classify_2d::{sample_interior_point, signed_area_2d};
use super::pcurve_compute::evaluate_edge_at_t;
use super::plane_frame::PlaneFrame;
use super::split_types::{OrientedPCurveEdge, SectionEdge, SplitSubFace, SurfaceInfo};
use super::wire_builder::{build_wire_loops, build_wire_loops_with_winding};
use crate::ds::Rank;

use containment::{find_point_outside_holes, is_inside_any_hole};
use conversion::{
    boundary_edges_to_pcurve, extract_plane_normal, is_point_on_boundary_uv,
    uv_endpoints_from_pcurve,
};
use edge_splitting::split_boundary_edges_at_3d_points;
use sampling::{sample_wire_loop_uv, sample_wire_loop_uv_periodic};
use special_cases::{
    split_face_with_internal_loops, split_noseam_face_direct, split_periodic_face_into_bands,
    try_split_crossing_plane_face,
};

/// Number of probe points (plus one for the closing sample) walked along a
/// section edge when testing whether it lies entirely inside an existing hole.
const HOLE_PROBE_SAMPLES: usize = 8;

/// Split a face by its section edges, producing sub-faces.
///
/// If there are no section edges, returns a single sub-face covering
/// the entire face (pass-through).
///
/// # Arguments
/// - `topo` -- the topology arena (immutable read)
/// - `face_id` -- the face to split
/// - `sections` -- intersection curves that cut this face (already trimmed)
/// - `rank` -- which solid this face belongs to (A or B)
/// - `tol` -- tolerance (`.linear` for 3D matching, UV tol derived internally)
/// - `frame` -- cached `PlaneFrame` for this face (avoids origin mismatch)
/// - `info` -- cached `SurfaceInfo` for periodicity flags
#[allow(clippy::too_many_lines)]
pub fn split_face_2d(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    rank: Rank,
    tol: &brepkit_math::tolerance::Tolerance,
    frame: Option<&PlaneFrame>,
    info: Option<&SurfaceInfo>,
) -> Vec<SplitSubFace> {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let is_plane = matches!(surface, FaceSurface::Plane { .. });

    // Use provided frame or build one from wire points (plane faces only).
    let wire_pts = collect_wire_points(topo, face.outer_wire());
    let owned_frame;
    let frame = if let Some(f) = frame {
        f
    } else if is_plane {
        let normal = extract_plane_normal(&surface);
        owned_frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        &owned_frame
    } else {
        // For non-plane faces, PlaneFrame is not used -- set a dummy.
        // All UV projection goes through surface.project_point().
        owned_frame = PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &[]);
        &owned_frame
    };

    // Extract periodicity from SurfaceInfo.
    // Periodic quantization is needed for boundary wire connectivity (circle
    // end at u=2pi connects to seam start at u=0). Keep it enabled.
    let (u_periodic, v_periodic) = info.map_or((false, false), SurfaceInfo::periodicity);

    // Convert boundary edges to OrientedPCurveEdge.
    let mut boundary_edges = if is_plane {
        boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, Some(frame))
    } else {
        boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, None)
    };

    // Convert original inner wires (holes) to OrientedPCurveEdge.
    let original_inner_wires: Vec<Vec<OrientedPCurveEdge>> = face
        .inner_wires()
        .iter()
        .filter_map(|&iw_id| {
            let iw_pts = collect_wire_points(topo, iw_id);
            let edges = if is_plane {
                boundary_edges_to_pcurve(topo, iw_id, &surface, &iw_pts, Some(frame))
            } else {
                boundary_edges_to_pcurve(topo, iw_id, &surface, &iw_pts, None)
            };
            // A hole bounded by closed curved edges (e.g. a single full
            // circle) has fewer than 3 distinct wire points but is a valid
            // inner wire; only polyline-style wires need 3+ points.
            let has_closed_curve = edges.iter().any(|e| {
                !matches!(e.curve_3d, EdgeCurve::Line)
                    && (e.start_3d - e.end_3d).length() < tol.linear * 100.0
            });
            if edges.is_empty() || (iw_pts.len() < 3 && !has_closed_curve) {
                None
            } else {
                Some(edges)
            }
        })
        .collect();

    // A section edge lying entirely inside an existing hole runs through
    // air, not face material (a tool passing through a cavity opening still
    // intersects the face's surface plane inside the hole). Keeping it would
    // stamp a spurious nested loop onto the face, leaving free edges.
    let filtered_sections: Vec<SectionEdge>;
    let sections = if original_inner_wires.is_empty() {
        sections
    } else {
        let to_uv = |p: Point3| -> Option<Point2> {
            if is_plane {
                Some(frame.project(p))
            } else {
                surface.project_point(p).map(|(u, v)| Point2::new(u, v))
            }
        };
        // Sample along the actual curve, not the start/mid/end chord: a
        // strongly curved section edge can bow outside the hole while its
        // endpoints and chord midpoint all sit inside it. Walking the curve
        // via `evaluate_edge_at_t` also covers closed-circle sections
        // (start == end), where chord sampling collapses to a single point.
        filtered_sections = sections
            .iter()
            .filter(|s| {
                let all_in_hole = (0..=HOLE_PROBE_SAMPLES).all(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = i as f64 / HOLE_PROBE_SAMPLES as f64;
                    let p = evaluate_edge_at_t(&s.curve_3d, s.start, s.end, t);
                    to_uv(p).is_some_and(|uv| is_inside_any_hole(&uv, &original_inner_wires))
                });
                !all_in_hole
            })
            .cloned()
            .collect();
        &filtered_sections
    };

    // If no section edges, the face is unsplit -- return as-is with original holes.
    if sections.is_empty() {
        return vec![SplitSubFace {
            surface,
            outer_wire: boundary_edges,
            inner_wires: original_inner_wires,
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        }];
    }

    // No-seam face shortcut: faces whose boundary is entirely Line edges
    // (no seam edges) can't be split by the wire builder (it needs vertical
    // seam connections to form rectangular bands). Construct cap + band
    // sub-faces directly instead. Applies to sphere hemispheres and any
    // other face topology without seam edges.
    let all_boundary_line = boundary_edges.iter().all(|e| {
        matches!(e.curve_3d, EdgeCurve::Line)
            // Exclude degenerate seam edges (start approx end) -- those are periodic
            // seam connections (e.g., torus), not true line boundaries.
            && (e.start_3d - e.end_3d).length() > tol.linear
    });
    if all_boundary_line && !is_plane {
        return split_noseam_face_direct(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            &wire_pts,
            tol.linear,
        );
    }

    // Band shortcut: closed section circles on a u-periodic face split it
    // into stacked bands, not discs. Requires seam-anchored circles (see
    // the seam-anchor pre-pass in fill_images_faces); falls through to the
    // generic paths when preconditions don't hold.
    if u_periodic && !is_plane && original_inner_wires.is_empty() {
        if let Some(bands) = split_periodic_face_into_bands(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            tol.linear,
        ) {
            return bands;
        }
    }

    // Internal section edge shortcut: when section edges form closed loops
    // entirely within the face (not connecting to boundary edges), the wire
    // builder struggles with periodic UV and 4-way junctions. Instead, group
    // the section edges into closed loops and construct sub-faces directly.
    //
    // Detection: check if ALL section endpoints are far from the face
    // boundary in UV space. Project each section endpoint to UV and test
    // if it lies on any boundary edge's UV segment (within tolerance).
    // This is surface-type agnostic and handles curved boundary edges.
    let mut deduped_line_loops: Option<Vec<SectionEdge>> = None;
    let all_sections_internal = if sections.is_empty() {
        false
    } else if is_plane {
        // Plane faces: exactly 1 closed section curve, or all-Line
        // sections forming closed loops strictly inside the boundary
        // (nested coplanar footprints). Multiple circles on the same
        // plane face still need the wire builder for loop formation.
        let single_closed = sections.len() == 1
            && sections.iter().all(|s| {
                (s.start - s.end).length() < tol.linear // closed curve
            });
        if single_closed {
            true
        } else {
            deduped_line_loops =
                plane_internal_line_loops(sections, frame, &boundary_edges, tol.linear);
            deduped_line_loops.is_some()
        }
    } else {
        // Non-plane faces: check if all section endpoints are off the
        // boundary in UV space.
        let uv_tol = 0.01; // ~0.6 deg in angular coordinates
        sections.iter().all(|s| {
            let start_on_boundary =
                is_point_on_boundary_uv(s.start, &surface, &boundary_edges, uv_tol);
            let end_on_boundary = is_point_on_boundary_uv(s.end, &surface, &boundary_edges, uv_tol);
            !start_on_boundary && !end_on_boundary
        })
    };

    if all_sections_internal {
        let secs = deduped_line_loops.as_deref().unwrap_or(sections);
        log::debug!(
            "split_face_2d: face {face_id:?} routed to internal-loops path ({} sections)",
            secs.len()
        );
        return split_face_with_internal_loops(
            &surface,
            &boundary_edges,
            &original_inner_wires,
            secs,
            rank,
            reversed,
            face_id,
            &wire_pts,
        );
    }

    // Stage 2: Split boundary edges at section edge endpoints (3D matching).
    let mut split_pts_3d: Vec<Point3> = sections.iter().flat_map(|s| [s.start, s.end]).collect();

    // For periodic faces, align closed boundary edge UV with seam edge UV.
    // The same 3D vertex projects to u=0 (from circle unwrapping) and u=seam
    // (from Line edge projection). Shift the circle UV so it starts at seam_u.
    if u_periodic {
        let seam_u_opt = boundary_edges.iter().find_map(|e| {
            if matches!(e.curve_3d, EdgeCurve::Line) {
                surface.project_point(e.start_3d).map(|(u, _)| u)
            } else {
                None
            }
        });
        if let Some(seam_u) = seam_u_opt {
            for edge in &mut boundary_edges {
                if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                    // Closed edge: shift UV so start_uv.x() == seam_u.
                    let shift = seam_u - edge.start_uv.x();
                    if shift.abs() > 0.01 {
                        edge.start_uv = Point2::new(edge.start_uv.x() + shift, edge.start_uv.y());
                        edge.end_uv = Point2::new(edge.end_uv.x() + shift, edge.end_uv.y());
                    }
                }
            }
        }
    }

    // For periodic faces with section edges, split closed boundary edges
    // (full circles) at the point diametrically opposite the seam vertex
    // in the surface's UV parameterization (u = seam_u + pi).
    //
    // The seam vertex (where the boundary circle starts/ends) is shared
    // with the seam Line edge. Splitting the circle at the UV-antipodal
    // point creates half-arcs whose endpoints match the seam edge vertices,
    // enabling the wire builder to form proper rectangular bands.
    if u_periodic && !sections.is_empty() {
        // Find the seam Line edge's vertex UV to determine seam_u.
        let seam_u = {
            let mut su = 0.0_f64;
            for edge in &boundary_edges {
                if matches!(edge.curve_3d, EdgeCurve::Line) {
                    if let Some((u, _)) = surface.project_point(edge.start_3d) {
                        su = u;
                        break;
                    }
                }
            }
            su
        };
        let anti_u = (seam_u + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU);

        for edge in &boundary_edges {
            if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                // Closed edge: find the 3D point at u = seam_u + pi on the surface.
                // Project the boundary vertex to get v, then evaluate surface at (anti_u, v).
                if let Some((_, v)) = surface.project_point(edge.start_3d) {
                    if let Some(anti_pt) = surface.evaluate(anti_u, v) {
                        split_pts_3d.push(anti_pt);
                    }
                }
            }
        }
    }

    let boundary_edges = split_boundary_edges_at_3d_points(
        boundary_edges,
        &split_pts_3d,
        if is_plane { Some(frame) } else { None },
        &surface,
        tol.linear,
    );

    // Reorder boundary edges: Line (seam) edges first, then curved (circle)
    // edges. This ensures the wire builder starts loops from seam edges,
    // forming rectangular bands before circle arcs can self-close.
    let boundary_edges = if u_periodic && !sections.is_empty() {
        let (mut lines, curves): (Vec<_>, Vec<_>) = boundary_edges
            .into_iter()
            .partition(|e| matches!(e.curve_3d, EdgeCurve::Line));
        lines.extend(curves);
        lines
    } else {
        boundary_edges
    };

    let boundary_edges_backup = if is_plane && sections.len() >= 2 {
        Some(boundary_edges.clone())
    } else {
        None
    };

    // Convert section edges to OrientedPCurveEdge (both orientations).
    let mut all_edges = boundary_edges;
    let n_boundary_edges = all_edges.len();
    for section in sections {
        // Skip full-circle section edges on plane faces -- they have
        // start approx end in 3D and would produce degenerate UV edges.
        // The half-arc section edges handle the plane face correctly.
        let is_closed_edge = (section.start - section.end).length() < 1e-10;
        if is_closed_edge && is_plane {
            continue;
        }

        // Curved sections on plane faces must live in the same PlaneFrame
        // as the boundary edges. The pcurve from build_section_edges was
        // fitted in a frame anchored at the original (pre-split) wire, so
        // its UV space — and its NURBS parameter domain — need not match
        // `frame`; using it would disconnect the section from the boundary
        // in UV. Refit it in this face's frame.
        let owned_pcurve;
        let pcurve_on_this_face = if is_plane && !matches!(section.curve_3d, EdgeCurve::Line) {
            owned_pcurve = super::pcurve_compute::compute_pcurve_on_surface(
                &section.curve_3d,
                section.start,
                section.end,
                &surface,
                &wire_pts,
                Some(frame),
            );
            &owned_pcurve
        } else {
            match rank {
                Rank::A => &section.pcurve_a,
                Rank::B => &section.pcurve_b,
            }
        };

        // Project section endpoints to UV.
        // Use pre-computed UV endpoints when available (e.g. seam-split half-arcs
        // where the unwrapped UV was computed from the arc samples). Otherwise,
        // for non-plane faces, use the pcurve's endpoint evaluations instead
        // of independent surface projection -- this ensures UV endpoints are
        // consistent with the pcurve's unwrapped parameterization (e.g. arc
        // ending at u=2pi rather than u=0 after periodic unwrapping).
        let (start_uv, end_uv) = if is_plane {
            // Plane faces: project in the boundary's frame. Precomputed UVs
            // (when present) come from build_section_edges' own frame and
            // would not connect to the boundary edges in UV.
            (frame.project(section.start), frame.project(section.end))
        } else {
            match rank {
                Rank::A => {
                    if let (Some(su), Some(eu)) = (section.start_uv_a, section.end_uv_a) {
                        (su, eu)
                    } else {
                        uv_endpoints_from_pcurve(
                            pcurve_on_this_face,
                            section.start,
                            section.end,
                            &surface,
                            &wire_pts,
                        )
                    }
                }
                Rank::B => {
                    if let (Some(su), Some(eu)) = (section.start_uv_b, section.end_uv_b) {
                        (su, eu)
                    } else {
                        uv_endpoints_from_pcurve(
                            pcurve_on_this_face,
                            section.start,
                            section.end,
                            &surface,
                            &wire_pts,
                        )
                    }
                }
            }
        };

        // Forward direction. Both forward and reverse share the same
        // source_edge_idx so build_topology_face creates one shared edge.
        let section_idx = all_edges.len();
        let pb_id = section.pave_block_id;
        all_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
            source_edge_idx: Some(section_idx),
            pave_block_id: pb_id,
        });
        // Reverse direction (for the adjacent sub-face).
        all_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv: end_uv,
            end_uv: start_uv,
            start_3d: section.end,
            end_3d: section.start,
            forward: false,
            source_edge_idx: Some(section_idx),
            pave_block_id: pb_id,
        });
    }

    // Partial-band u unwrap: a face whose u-window touches the period seam
    // (e.g. a rounded-rect corner cylinder spanning [3pi/2, 2pi]) carries
    // mixed u anchors — surface projection returns u in [0, 2pi), so
    // endpoints exactly on the seam come back as 0 while their neighbours
    // sit near 2pi. Partial bands are treated as non-periodic (see
    // build_surface_info), so quantized junction keys would never match.
    // Remap every endpoint u into the continuous window that starts after
    // the largest angular gap.
    if !u_periodic && !is_plane {
        if let (Some(u_period), _) = super::pcurve_compute::surface_periods(&surface) {
            let mut us: Vec<f64> = all_edges
                .iter()
                .flat_map(|e| [e.start_uv.x(), e.end_uv.x()])
                .map(|u| u.rem_euclid(u_period))
                .collect();
            us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if us.len() >= 2 {
                let mut gap_start = us[us.len() - 1];
                let mut max_gap = u_period - (us[us.len() - 1] - us[0]);
                for w in us.windows(2) {
                    if w[1] - w[0] > max_gap {
                        max_gap = w[1] - w[0];
                        gap_start = w[0];
                    }
                }
                if max_gap > 0.05 {
                    let lo = gap_start + max_gap;
                    for e in &mut all_edges {
                        let remap = |uv: Point2| -> Point2 {
                            let mut d = (uv.x() - lo).rem_euclid(u_period);
                            if d > u_period - 1e-6 {
                                d = 0.0;
                            }
                            Point2::new(lo + d, uv.y())
                        };
                        e.start_uv = remap(e.start_uv);
                        e.end_uv = remap(e.end_uv);
                    }
                }
            }
        }
    }

    // Build wire loops via angular-sorting traversal.
    let mut loops = build_wire_loops(&all_edges, tol.linear, u_periodic, v_periodic);

    // Clockwise-boundary retry: the min-clockwise turn rule merges
    // everything into a single loop when the boundary winds clockwise in
    // this UV frame (the frame derives from the raw surface normal, not the
    // effective face orientation). If the default traversal failed to split
    // despite having sections, and the boundary is CW, retry with the
    // mirrored turn rule. Loop areas from the retry are sign-flipped for
    // the outer/hole classification below.
    let mut cw_loops = false;
    if loops.len() <= 1 && all_edges.len() > n_boundary_edges && !u_periodic && !v_periodic {
        let boundary_pts = sample_wire_loop_uv(&all_edges[..n_boundary_edges]);
        if signed_area_2d(&boundary_pts) < 0.0 {
            let retry =
                build_wire_loops_with_winding(&all_edges, tol.linear, u_periodic, v_periodic, true);
            if retry.len() > loops.len() {
                loops = retry;
                cw_loops = true;
            }
        }
    }

    // Fallback: wire builder produced only 1 loop despite having 2+ section
    // edges that cross in the face interior. Use direct geometric quadrant
    // construction. The wire builder struggles with 4-way junctions when
    // boundary edges have inconsistent winding.
    if loops.len() <= 1 && sections.len() >= 2 && is_plane {
        if let Some(ref boundary) = boundary_edges_backup {
            if let Some(result) = try_split_crossing_plane_face(
                &surface, boundary, sections, rank, reversed, face_id, frame, tol,
            ) {
                return result;
            }
        }
    }

    // Classify each loop as outer (positive area) or hole (negative).
    // For loops with curved edges, sample intermediate UV points to get
    // an accurate area -- using only start_uv gives degenerate polygons
    // for 2-edge circles.
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, f64)> = Vec::new();
    let mut holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();

    let u_per_opt = if u_periodic {
        Some(std::f64::consts::TAU)
    } else {
        None
    };
    let v_per_opt = if v_periodic {
        Some(std::f64::consts::TAU)
    } else {
        None
    };

    // For periodic faces with section edges, use structural classification
    // instead of signed area. Band loops (containing seam + section edges)
    // are outer wires. Circle-only self-loops are holes. Signed area on
    // periodic surfaces is unreliable because UV wraps around the period.
    let use_structural_classification = u_periodic && !sections.is_empty();

    for wire_loop in loops {
        if use_structural_classification {
            // Structural: a loop containing both Line edges (seam) and
            // non-Line edges (section arcs / circles) is a band = outer.
            let has_line = wire_loop
                .iter()
                .any(|e| matches!(e.curve_3d, EdgeCurve::Line));
            let has_nonline = wire_loop
                .iter()
                .any(|e| !matches!(e.curve_3d, EdgeCurve::Line));
            if has_line && has_nonline {
                outers.push((wire_loop, 1.0)); // area placeholder
            } else {
                holes.push(wire_loop);
            }
        } else {
            let pts = sample_wire_loop_uv_periodic(&wire_loop, u_per_opt, v_per_opt);
            let area = if cw_loops {
                -signed_area_2d(&pts)
            } else {
                signed_area_2d(&pts)
            };
            // Sliver guard: a loop enclosing less area than a tol-wide band
            // along its own perimeter is degenerate — e.g. an arc traversed
            // forward then backward when a coplanar partner's boundary
            // coincides with the face's own corner arc. Classifying it as
            // outer creates a zero-area face; as hole, a spurious inner wire.
            let perimeter: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
            if area.abs() <= perimeter * tol.linear {
                continue;
            }
            if area > 0.0 {
                outers.push((wire_loop, area));
            } else {
                holes.push(wire_loop);
            }
        }
    }

    // If all loops are CW (negative area), the winding is reversed.
    if !use_structural_classification && outers.is_empty() && !holes.is_empty() {
        for hole in &mut holes {
            hole.reverse();
            for edge in hole.iter_mut() {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        }
        let pts: Vec<Point2> = holes[0].iter().map(|e| e.start_uv).collect();
        let area = signed_area_2d(&pts);
        outers.push((holes.remove(0), area));
    }

    // Match holes to containing outer wires.
    let mut sub_faces = Vec::new();
    for (outer_wire, _area) in outers {
        sub_faces.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        });
    }

    // Simple hole matching: each hole goes to the outer that contains its
    // first vertex (via 2D point-in-polygon). Uses sampled UV points for
    // accurate containment with curved outer wires.
    for hole in holes {
        if let Some(first_pt) = hole.first().map(|e| e.start_uv) {
            let mut assigned = false;
            for sf in &mut sub_faces {
                let outer_pts = sample_wire_loop_uv(&sf.outer_wire);
                if super::classify_2d::point_in_polygon_2d(first_pt, &outer_pts) {
                    sf.inner_wires.push(hole.clone());
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                if let Some(sf) = sub_faces.first_mut() {
                    sf.inner_wires.push(hole);
                }
            }
        }
    }

    // Distribute original inner wires (holes from the source face) to sub-faces.
    // Each hole is assigned to the sub-face whose outer wire contains its
    // interior sample point (a point inside the hole's enclosed area, not
    // its first vertex — that vertex often sits exactly on a sub-face
    // boundary when the split passes through it, and `point_in_polygon_2d`'s
    // strict ray-cast returns false for every sub-face). If no sub-face
    // claims the hole — degenerate UV sample, hole straddling multiple
    // sub-faces, etc. — fall back to the largest-area sub-face. A warning
    // fires for the fallback so the case stays visible; what we never do is
    // silently drop the hole as the earlier code did.
    if !original_inner_wires.is_empty() {
        let largest_sub_face_idx = |sub_faces: &[SplitSubFace]| -> Option<usize> {
            sub_faces
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let area_a =
                        super::classify_2d::signed_area_2d(&sample_wire_loop_uv(&a.outer_wire))
                            .abs();
                    let area_b =
                        super::classify_2d::signed_area_2d(&sample_wire_loop_uv(&b.outer_wire))
                            .abs();
                    area_a
                        .partial_cmp(&area_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        };

        for hole in &original_inner_wires {
            let hole_pts = sample_wire_loop_uv(hole);
            let assigned = if hole_pts.len() >= 3 {
                let probe = super::classify_2d::sample_interior_point(&hole_pts);
                sub_faces.iter_mut().find_map(|sf| {
                    let outer_pts = sample_wire_loop_uv(&sf.outer_wire);
                    super::classify_2d::point_in_polygon_2d(probe, &outer_pts).then(|| {
                        sf.inner_wires.push(hole.clone());
                    })
                })
            } else {
                None
            };
            if assigned.is_some() {
                continue;
            }
            // Fallback path: degenerate sample OR no sub-face contained the
            // probe point. Attach to the largest sub-face so the geometry is
            // preserved.
            let reason = if hole_pts.len() < 3 {
                "produced a degenerate UV sample (<3 pts)"
            } else {
                "did not contain-test in any sub-face"
            };
            log::warn!(
                "face_splitter: hole with {} edges {reason}; attaching to largest sub-face \
                 as fallback",
                hole.len()
            );
            if let Some(idx) = largest_sub_face_idx(&sub_faces) {
                sub_faces[idx].inner_wires.push(hole.clone());
            }
        }
    }

    sub_faces
}

/// Get a point guaranteed inside a sub-face's outer wire (in UV space),
/// not inside any inner wire (hole), then evaluate it to 3D via the surface.
#[allow(clippy::too_many_lines)]
pub fn interior_point_3d(sub_face: &SplitSubFace, frame: Option<&PlaneFrame>) -> Point3 {
    let pts_2d = sample_wire_loop_uv(&sub_face.outer_wire);
    let mut interior_uv = sample_interior_point(&pts_2d);

    // Sphere cap fix: sphere sub-faces with degenerate UV boundaries (thin
    // strip at constant v) need the interior UV offset toward the pole.
    // The outer wire of a sphere cap maps to a horizontal line in UV,
    // producing a near-zero-area polygon whose centroid lies on the boundary.
    if let FaceSurface::Sphere(_) = &sub_face.surface {
        if !pts_2d.is_empty() {
            let v_min = pts_2d.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min);
            let v_max = pts_2d
                .iter()
                .map(|p| p.y())
                .fold(f64::NEG_INFINITY, f64::max);
            if (v_max - v_min) < 0.1 {
                let v_boundary = (v_min + v_max) * 0.5;
                let v_pole = if v_boundary >= 0.0 {
                    std::f64::consts::FRAC_PI_2
                } else {
                    -std::f64::consts::FRAC_PI_2
                };
                let u_center = pts_2d.iter().map(|p| p.x()).sum::<f64>() / pts_2d.len() as f64;
                interior_uv = Point2::new(u_center, (v_boundary + v_pole) * 0.5);
            }
        }
    }

    // If the point falls inside a hole, find a point between the outer wire
    // and the nearest hole boundary.
    if is_inside_any_hole(&interior_uv, &sub_face.inner_wires) {
        interior_uv = find_point_outside_holes(&pts_2d, &sub_face.inner_wires);
    }

    // Secondary hole check: sample_wire_loop_uv for curved hole wires may
    // produce an under-sampled polygon that misses containment. Cross-check
    // using the hole's 3D boundary: if the interior 3D point is close to
    // the centroid of any hole, it's likely inside and needs displacement.
    if !sub_face.inner_wires.is_empty() {
        let eval_3d = |uv: Point2| -> Option<Point3> {
            if let Some(p) = sub_face.surface.evaluate(uv.x(), uv.y()) {
                return Some(p);
            }
            if let FaceSurface::Plane { normal, .. } = &sub_face.surface {
                if let Some(f) = frame {
                    return Some(f.evaluate(uv.x(), uv.y()));
                }
                let wire_pts: Vec<Point3> =
                    sub_face.outer_wire.iter().map(|e| e.start_3d).collect();
                let f = PlaneFrame::from_plane_face(*normal, &wire_pts);
                return Some(f.evaluate(uv.x(), uv.y()));
            }
            None
        };

        if let Some(test_3d) = eval_3d(interior_uv) {
            for hole in &sub_face.inner_wires {
                // Compute hole centroid in 3D.
                if hole.is_empty() {
                    continue;
                }
                let hc: Point3 = {
                    let sum = hole.iter().fold(Point3::new(0.0, 0.0, 0.0), |acc, e| {
                        acc + (e.start_3d - Point3::new(0.0, 0.0, 0.0))
                    });
                    let n = hole.len() as f64;
                    Point3::new(sum.x() / n, sum.y() / n, sum.z() / n)
                };
                // Compute hole boundary radius from centroid.
                let max_r = hole
                    .iter()
                    .map(|e| (e.start_3d - hc).length())
                    .fold(0.0_f64, f64::max);

                if (test_3d - hc).length() < max_r * 0.95 {
                    // Interior point is inside the hole in 3D. Try outer wire
                    // vertex that's farthest from the hole centroid.
                    let best = sub_face
                        .outer_wire
                        .iter()
                        .max_by(|a, b| {
                            let da = (a.start_3d - hc).length();
                            let db = (b.start_3d - hc).length();
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|e| e.start_uv);
                    if let Some(uv) = best {
                        // Nudge slightly toward the centroid so the point
                        // is strictly interior, not on the boundary vertex.
                        interior_uv = Point2::new(
                            uv.x() * 0.95 + interior_uv.x() * 0.05,
                            uv.y() * 0.95 + interior_uv.y() * 0.05,
                        );
                    }
                    break;
                }
            }
        }
    }

    // Evaluate back to 3D.
    if let Some(p) = sub_face.surface.evaluate(interior_uv.x(), interior_uv.y()) {
        return p;
    }

    // For plane faces, evaluate via PlaneFrame.
    if let FaceSurface::Plane { normal, .. } = &sub_face.surface {
        if let Some(f) = frame {
            return f.evaluate(interior_uv.x(), interior_uv.y());
        }
        let wire_pts: Vec<Point3> = sub_face.outer_wire.iter().map(|e| e.start_3d).collect();
        let f = PlaneFrame::from_plane_face(*normal, &wire_pts);
        return f.evaluate(interior_uv.x(), interior_uv.y());
    }

    // Last resort: average of 3D endpoints.
    let sum: Point3 = sub_face
        .outer_wire
        .iter()
        .fold(Point3::new(0.0, 0.0, 0.0), |acc, e| {
            acc + (e.start_3d - Point3::new(0.0, 0.0, 0.0))
        });
    let n = sub_face.outer_wire.len() as f64;
    Point3::new(sum.x() / n, sum.y() / n, sum.z() / n)
}

/// Detect all-Line section edges forming closed loops strictly inside a
/// plane face's boundary (nested coplanar footprints), and dedup repeated
/// segments. Both the coplanar-contact pass and adjacent-face plane-plane
/// intersections can emit the same footprint segment, so identical
/// segments (by unordered quantized endpoints) collapse to one.
///
/// Returns the deduped sections when every quantized endpoint has degree
/// exactly 2 (disjoint closed loops) and every endpoint lies strictly
/// interior to the boundary polygon; `None` routes back to the generic
/// wire-builder path.
fn plane_internal_line_loops(
    sections: &[SectionEdge],
    frame: &PlaneFrame,
    boundary_edges: &[OrientedPCurveEdge],
    tol_linear: f64,
) -> Option<Vec<SectionEdge>> {
    use std::collections::{HashMap, HashSet};

    type QPt = (i64, i64, i64);

    // Accept Line and open arc (Circle/Ellipse) sections: a rounded-rect
    // tool footprint stamps a mixed line+arc loop onto a coplanar cap.
    // Closed curves (start == end) are handled by the single-closed path.
    if sections.len() < 3
        || !sections.iter().all(|s| match s.curve_3d {
            EdgeCurve::Line => true,
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => (s.start - s.end).length() > tol_linear,
            EdgeCurve::NurbsCurve(_) => false,
        })
    {
        return None;
    }
    let polygon: Vec<Point2> = boundary_edges.iter().map(|e| e.start_uv).collect();
    if polygon.len() < 3 {
        return None;
    }

    let quant = |p: Point3| -> QPt {
        let s = 1.0 / (tol_linear * 100.0);
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };

    let margin = tol_linear * 100.0;
    let on_plane = |p: Point3| {
        let uv = frame.project(p);
        (frame.evaluate(uv.x(), uv.y()) - p).length() <= margin
    };

    let mut seen: HashSet<(QPt, QPt)> = HashSet::new();
    let mut deduped: Vec<SectionEdge> = Vec::new();
    for s in sections {
        // A section can only bound a sub-face of this plane if it lies on
        // the plane; off-plane segments (e.g. lateral edges grazing the
        // face at one endpoint) are noise for this face.
        if !on_plane(s.start) || !on_plane(s.end) {
            continue;
        }
        let a = quant(s.start);
        let b = quant(s.end);
        if a == b {
            return None;
        }
        let key = if a <= b { (a, b) } else { (b, a) };
        if seen.insert(key) {
            deduped.push(s.clone());
        }
    }
    if deduped.len() < 3 {
        return None;
    }

    // The same footprint side can arrive both whole and as sub-segments
    // split at paves. Drop any segment that another section's endpoint
    // subdivides (collinear, strictly interior) — the sub-segments carry
    // the same geometry. If the sub-segments turn out incomplete, the
    // degree check below rejects and the generic path takes over.
    let endpoints: Vec<Point3> = deduped.iter().flat_map(|s| [s.start, s.end]).collect();
    deduped.retain(|s| {
        if !matches!(s.curve_3d, EdgeCurve::Line) {
            return true;
        }
        let dir = s.end - s.start;
        let len2 = dir.dot(dir);
        if len2 < margin * margin {
            return true;
        }
        !endpoints.iter().any(|&p| {
            if (p - s.start).length() < margin || (p - s.end).length() < margin {
                return false;
            }
            let t = (p - s.start).dot(dir) / len2;
            if !(0.0..=1.0).contains(&t) {
                return false;
            }
            let foot = s.start + dir * t;
            (p - foot).length() < margin
        })
    });

    let mut degree: HashMap<QPt, u32> = HashMap::new();
    for s in &deduped {
        *degree.entry(quant(s.start)).or_insert(0) += 1;
        *degree.entry(quant(s.end)).or_insert(0) += 1;
        for p in [s.start, s.end] {
            let uv = frame.project(p);
            if !super::classify_2d::point_in_polygon_2d(uv, &polygon)
                || super::classify_2d::distance_to_polygon_boundary(uv, &polygon) <= margin
            {
                log::debug!(
                    "plane_internal_line_loops: endpoint {p:?} not strictly interior (dist {})",
                    super::classify_2d::distance_to_polygon_boundary(uv, &polygon)
                );
                return None;
            }
        }
    }
    if degree.values().any(|&d| d != 2) {
        let bad: Vec<_> = degree.iter().filter(|&(_, &d)| d != 2).collect();
        log::debug!(
            "plane_internal_line_loops: {} endpoints with degree != 2 (deduped {} of {}): {bad:?}",
            bad.len(),
            deduped.len(),
            degree.len()
        );
        return None;
    }
    Some(deduped)
}
