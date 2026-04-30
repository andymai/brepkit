//! Special topology handlers for face splitting edge cases.

use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::super::plane_frame::PlaneFrame;
use super::super::split_types::{OrientedPCurveEdge, SectionEdge, SplitSubFace};
use super::conversion::uv_endpoints_from_pcurve;
use super::edge_splitting::split_boundary_edges_at_3d_points;
use crate::ds::Rank;

/// Split a face with no seam edges directly into cap + band sub-faces.
///
/// Faces whose boundary consists entirely of Line edges (no seam edges)
/// can't be split by the wire builder (it needs vertical seam connections).
/// This function bypasses the wire builder and constructs sub-faces
/// geometrically from the section edges:
///
/// - **Cap**: bounded by the section circle (2 half-arcs).
/// - **Band**: bounded by the original boundary, with the section as hole.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_noseam_face_direct(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    wire_pts: &[Point3],
) -> Vec<SplitSubFace> {
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

    // Collect section forward/reverse edges on this face.
    let mut cap_edges = Vec::new();
    let mut hole_edges = Vec::new();

    for section in sections {
        let pcurve_on_this_face = match rank {
            Rank::A => &section.pcurve_a,
            Rank::B => &section.pcurve_b,
        };

        // Skip full-circle section edges (start approx end in 3D) -- only use
        // the half-arcs produced by build_seam_split_sections.
        if (section.start - section.end).length() < brepkit_math::tolerance::Tolerance::new().linear
        {
            continue;
        }

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

        // Forward: for the cap outer wire.
        cap_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        });

        // Reverse: for the band's inner wire (hole).
        hole_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv: end_uv,
            end_uv: start_uv,
            start_3d: section.end,
            end_3d: section.start,
            forward: false,
            source_edge_idx: None,
            pave_block_id: None,
        });
    }

    if cap_edges.is_empty() {
        // No valid section edges -- return the face unsplit.
        return unsplit();
    }

    // Validate: cap edges must form a single closed loop (last end approx first start).
    // If the topology is unexpected (multiple loops, open chain), fall back to unsplit.
    let loop_gap = (cap_edges
        .last()
        .map_or(Point3::new(0.0, 0.0, 0.0), |e| e.end_3d)
        - cap_edges
            .first()
            .map_or(Point3::new(0.0, 0.0, 0.0), |e| e.start_3d))
    .length();
    if loop_gap > brepkit_math::tolerance::Tolerance::new().linear * 100.0 {
        return unsplit();
    }

    // Cap sub-face: outer wire = section forward half-arcs.
    // The half-arcs connect end-to-end, forming a closed loop (the section circle).
    // Band sub-face: outer wire = equatorial boundary, inner wire = section reversed.
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

/// Split a face when ALL section edges are interior (don't touch the boundary).
///
/// Groups section edges into closed loops by chaining shared 3D endpoints.
/// Each closed loop produces:
/// - An "inside" sub-face with the loop as outer wire
/// - A reversed copy added as an inner wire (hole) of the "outside" sub-face
///
/// The "outside" sub-face has the original boundary as outer wire with all
/// loops as holes.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_face_with_internal_loops(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
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
            // Previously dropped to None, which forced position-fallback
            // lookup that created duplicate vertices on the cylinder
            // side of cylinder-cut booleans.
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

        // Follow the chain until we close the loop.
        loop {
            let last_end = chain.last().map_or(loop_start_3d, |e| e.end_3d);

            // Check if the loop is closed (includes single-edge circles
            // where start ~= end).
            if (last_end - loop_start_3d).length() < tol_3d * 100.0 {
                break;
            }

            // Find the next unused edge whose start matches last_end.
            let next = forward_edges
                .iter()
                .enumerate()
                .find(|(i, e)| !used[*i] && (e.start_3d - last_end).length() < tol_3d * 100.0);

            if let Some((idx, _)) = next {
                used[idx] = true;
                chain.push(forward_edges[idx].clone());
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

    // Build sub-faces.
    let mut result = Vec::new();

    // For each closed loop: create an "inside" sub-face.
    // The loop winding determines which region of the face is enclosed.
    // We want the SMALLER region (the Steinmetz lobe), so check signed area
    // in UV and reverse if the loop encloses the larger region.
    let mut all_holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
    for loop_edges in &mut loops {
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
            // Sample 3D points along the circle to find its centroid.
            let edge = &loop_edges[0];
            let (t0, t1) = edge
                .curve_3d
                .domain_with_endpoints(edge.start_3d, edge.end_3d);
            let n_samples = 16;
            let mut sum = brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0);
            for k in 0..n_samples {
                #[allow(clippy::cast_precision_loss)]
                let t = t0 + (t1 - t0) * (k as f64 / n_samples as f64);
                let pt = edge
                    .curve_3d
                    .evaluate_with_endpoints(t, edge.start_3d, edge.end_3d);
                sum += brepkit_math::vec::Vec3::new(pt.x(), pt.y(), pt.z());
            }
            #[allow(clippy::cast_precision_loss)]
            let centroid = Point3::new(
                sum.x() / n_samples as f64,
                sum.y() / n_samples as f64,
                sum.z() / n_samples as f64,
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

        // Build reversed loop for the outside sub-face's hole.
        let hole: Vec<OrientedPCurveEdge> = loop_edges
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
            .collect();
        // Verify hole is closed.
        if let (Some(first), Some(last)) = (hole.first(), hole.last()) {
            if (last.end_3d - first.start_3d).length() < tol_3d * 100.0 {
                all_holes.push(hole);
            }
        }
    }

    // The "outside" sub-face: original boundary with all loops as holes.
    result.push(SplitSubFace {
        surface: surface.clone(),
        outer_wire: boundary_edges.to_vec(),
        inner_wires: all_holes,
        reversed,
        parent: face_id,
        rank,
        precomputed_interior: None,
    });

    result
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
        if !(0.01..=0.99).contains(&t0) {
            return None;
        }
        cross_3d = s0.start + d0 * t0;
        section_endpoints = vec![s0.start, s0.end, s1.start, s1.end];
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
    if section_indices.len() != 4 {
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

    let mut result = Vec::new();
    for qi in 0..4 {
        let arc_start = section_indices[qi];
        let arc_end = section_indices[(qi + 1) % 4];
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
