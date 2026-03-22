//! Split faces using `FaceInfo` data from the PaveFiller.
//!
//! For each face that has section pave blocks, converts them to
//! [`SectionEdge`] entries and calls [`split_face_2d`] to produce
//! geometric sub-faces. Faces without intersection data pass through
//! unchanged.

use std::collections::HashMap;
use std::hash::BuildHasher;

use std::collections::BTreeMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::ds::{GfaArena, PaveBlockId, Rank};

use super::SubFace;
use super::face_class::FaceClass;
use super::face_splitter::split_face_2d;
use super::split_types::{SectionEdge, SurfaceInfo};

/// Build sub-faces for all faces that have intersection data.
///
/// For faces with section edges (from FF intersection), calls the full
/// face splitter to produce geometrically split sub-faces. Faces
/// without intersection data pass through as single sub-faces.
#[allow(clippy::too_many_lines)]
pub fn fill_images_faces<S: BuildHasher, S2: BuildHasher>(
    topo: &mut Topology,
    arena: &GfaArena,
    _edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
    face_ranks: &HashMap<FaceId, Rank, S2>,
    tol: Tolerance,
) -> Vec<SubFace> {
    let mut sub_faces = Vec::new();

    // Shared edge cache: (face_id, source_edge_idx) → EdgeId. Ensures section
    // edges (which appear in both forward and reverse in adjacent loops from
    // the SAME face's split) reference the SAME topology edge entity.
    let mut shared_edge_cache: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> =
        HashMap::new();

    // Pre-compute which faces have section edges from which curves
    let section_map = build_section_map(arena);

    for (&face_id, &rank) in face_ranks {
        let fi = arena.face_info(face_id);
        let has_sections = fi.is_some_and(|fi| !fi.pave_blocks_sc.is_empty());

        log::debug!("fill_images_faces: face {face_id:?} has_sections={has_sections}");

        if !has_sections {
            // No sections: face passes through unchanged
            sub_faces.push(SubFace {
                face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point: None,
            });
            continue;
        }

        // Build SectionEdge entries from pave block data
        let sections = build_section_edges(topo, arena, face_id, &section_map, tol.linear);

        log::debug!(
            "fill_images_faces: face {face_id:?} got {} section edges",
            sections.len()
        );

        if sections.is_empty() {
            sub_faces.push(SubFace {
                face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point: None,
            });
            continue;
        }

        // Build SurfaceInfo for periodicity
        let info = build_surface_info(topo, face_id);

        // Call the face splitter
        let split_results = split_face_2d(
            topo,
            face_id,
            &sections,
            rank,
            &tol,
            None, // PlaneFrame built internally by face_splitter
            info.as_ref(),
        );

        log::debug!(
            "fill_images_faces: face {face_id:?} split into {} sub-faces",
            split_results.len()
        );

        if split_results.is_empty() {
            log::warn!("fill_images_faces: split_face_2d returned empty for face {face_id:?}");
            sub_faces.push(SubFace {
                face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point: None,
            });
            continue;
        }

        // Each SplitSubFace represents a geometric sub-region.
        // Build real topology entities (Vertex → Edge → Wire → Face) for each,
        // and compute a distinct interior point for classification.
        for split in &split_results {
            let new_face_id =
                build_topology_face(topo, split, tol, face_id, &mut shared_edge_cache);
            let pt = split
                .precomputed_interior
                .unwrap_or_else(|| super::face_splitter::interior_point_3d(split, None));

            sub_faces.push(SubFace {
                face_id: new_face_id.unwrap_or(face_id),
                classification: FaceClass::Unknown,
                rank,
                interior_point: Some(pt),
            });
        }
    }

    sub_faces
}

/// Map from face ID to section pave block IDs (from FF intersection curves).
fn build_section_map(arena: &GfaArena) -> HashMap<FaceId, Vec<PaveBlockId>> {
    let mut map: HashMap<FaceId, Vec<PaveBlockId>> = HashMap::new();
    for curve in &arena.curves {
        for &pb_id in &curve.pave_blocks {
            map.entry(curve.face_a).or_default().push(pb_id);
            map.entry(curve.face_b).or_default().push(pb_id);
        }
    }
    map
}

/// Convert pave block section data to `SectionEdge` entries.
#[allow(clippy::too_many_lines)]
fn build_section_edges(
    topo: &Topology,
    arena: &GfaArena,
    face_id: FaceId,
    section_map: &HashMap<FaceId, Vec<PaveBlockId>>,
    tol: f64,
) -> Vec<SectionEdge> {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2};

    let pb_ids = match section_map.get(&face_id) {
        Some(ids) => ids,
        None => return Vec::new(),
    };

    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let mut sections = Vec::new();

    for &pb_id in pb_ids {
        let pb = match arena.pave_blocks.get(pb_id) {
            Some(pb) => pb,
            None => continue,
        };

        let edge_id = match pb.split_edge {
            Some(eid) => eid,
            None => continue,
        };

        let edge = match topo.edge(edge_id) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let raw_start = match topo.vertex(edge.start()) {
            Ok(v) => v.point(),
            Err(_) => continue,
        };
        let raw_end = match topo.vertex(edge.end()) {
            Ok(v) => v.point(),
            Err(_) => continue,
        };

        // Clip straight section edges to the face boundary polygon.
        // Non-line curves (Circle, Ellipse, NURBS) pass through unclipped —
        // their endpoints are already bounded by the curve geometry.
        let (start, end) = if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
            match clip_line_to_face_boundary(topo, face_id, raw_start, raw_end, tol) {
                Some(pair) => pair,
                None => continue,
            }
        } else {
            (raw_start, raw_end)
        };

        // Project start/end to UV on this face
        let start_uv = face.surface().project_point(start);
        let end_uv = face.surface().project_point(end);

        // Build a simple Line2D pcurve from UV endpoints
        let make_pcurve = |s: Option<(f64, f64)>, e: Option<(f64, f64)>| -> Curve2D {
            let s2 = s.map_or(Point2::new(0.0, 0.0), |(u, v)| Point2::new(u, v));
            let e2 = e.map_or(Point2::new(1.0, 0.0), |(u, v)| Point2::new(u, v));
            let dir = e2 - s2;
            let len = dir.length();
            let direction = if len > 1e-12 {
                Vec2::new(dir.x() / len, dir.y() / len)
            } else {
                Vec2::new(1.0, 0.0)
            };
            // Try the actual direction; fall back to unit X if degenerate.
            // Line2D::new can only fail if direction length < 1e-15,
            // which can't happen for Vec2::new(1.0, 0.0).
            #[allow(clippy::expect_used)]
            let line = Line2D::new(s2, direction)
                .or_else(|_| Line2D::new(s2, Vec2::new(1.0, 0.0)))
                .expect("unit direction (1,0) is always valid");
            Curve2D::Line(line)
        };

        let pcurve = make_pcurve(start_uv, end_uv);

        sections.push(SectionEdge {
            curve_3d: edge.curve().clone(),
            pcurve_a: pcurve.clone(),
            pcurve_b: pcurve,
            start,
            end,
            start_uv_a: start_uv.map(|(u, v)| Point2::new(u, v)),
            end_uv_a: end_uv.map(|(u, v)| Point2::new(u, v)),
            start_uv_b: start_uv.map(|(u, v)| Point2::new(u, v)),
            end_uv_b: end_uv.map(|(u, v)| Point2::new(u, v)),
            target_face: None,
            pave_block_id: Some(pb_id.index()),
        });
    }

    sections
}

/// Clip a 3D line segment to a face's boundary polygon.
///
/// Collects the outer wire vertices as line segments, then finds where
/// the section line enters and exits the polygon. Returns the trimmed
/// `(start, end)` or `None` if the line doesn't cross the face.
#[allow(clippy::too_many_lines)]
fn clip_line_to_face_boundary(
    topo: &Topology,
    face_id: FaceId,
    line_start: Point3,
    line_end: Point3,
    tol: f64,
) -> Option<(Point3, Point3)> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;

    // Collect boundary edges as line segments (vertex positions in traversal order)
    let edges = wire.edges();
    let mut boundary_segments: Vec<(Point3, Point3)> = Vec::with_capacity(edges.len());
    for oe in edges {
        let edge = topo.edge(oe.edge()).ok()?;
        let sp = topo.vertex(oe.oriented_start(edge)).ok()?.point();
        let ep = topo.vertex(oe.oriented_end(edge)).ok()?.point();
        boundary_segments.push((sp, ep));
    }

    let line_dir = line_end - line_start;
    let line_len = line_dir.length();
    if line_len < tol {
        return None;
    }

    // Find all intersection parameters (t) of the section line with boundary segments.
    // The section line is: P(t) = line_start + t * line_dir, t in [0, 1].
    let mut crossings: Vec<f64> = Vec::new();

    for (seg_start, seg_end) in &boundary_segments {
        let seg_dir = *seg_end - *seg_start;
        let seg_len = seg_dir.length();

        // Scaled tolerance for parallel/determinant checks — proportional to
        // both vector magnitudes, consistent with the project tolerance framework.
        let parallel_tol = line_len * seg_len * tol;

        // For two coplanar 3D line segments, project to the dominant 2D plane.
        let normal = line_dir.cross(seg_dir);
        let ax = normal.x().abs();
        let ay = normal.y().abs();
        let az = normal.z().abs();

        // If lines are parallel (cross product near zero), skip
        if ax < parallel_tol && ay < parallel_tol && az < parallel_tol {
            continue;
        }

        let d = *seg_start - line_start;

        let (t, s) = if az >= ax && az >= ay {
            let det = line_dir.x() * seg_dir.y() - line_dir.y() * seg_dir.x();
            if det.abs() < parallel_tol {
                continue;
            }
            let t = (d.x() * seg_dir.y() - d.y() * seg_dir.x()) / det;
            let s = (d.x() * line_dir.y() - d.y() * line_dir.x()) / det;
            (t, s)
        } else if ay >= ax {
            let det = line_dir.x() * seg_dir.z() - line_dir.z() * seg_dir.x();
            if det.abs() < parallel_tol {
                continue;
            }
            let t = (d.x() * seg_dir.z() - d.z() * seg_dir.x()) / det;
            let s = (d.x() * line_dir.z() - d.z() * line_dir.x()) / det;
            (t, s)
        } else {
            let det = line_dir.y() * seg_dir.z() - line_dir.z() * seg_dir.y();
            if det.abs() < parallel_tol {
                continue;
            }
            let t = (d.y() * seg_dir.z() - d.z() * seg_dir.y()) / det;
            let s = (d.y() * line_dir.z() - d.z() * line_dir.y()) / det;
            (t, s)
        };

        // Boundary segment parameter must be within [0, 1] (with tolerance)
        let s_tol = tol / seg_dir.length().max(tol);
        if s >= -s_tol && s <= 1.0 + s_tol {
            crossings.push(t);
        }
    }

    if crossings.len() < 2 {
        return None;
    }

    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Take the outermost pair of crossings as entry/exit
    let t0 = crossings[0].clamp(0.0, 1.0);
    let t1 = crossings[crossings.len() - 1].clamp(0.0, 1.0);

    let t_tol = tol / line_len;
    if (t1 - t0).abs() < t_tol {
        return None;
    }

    let clipped_start = line_start + line_dir * t0;
    let clipped_end = line_start + line_dir * t1;

    Some((clipped_start, clipped_end))
}

/// Build `SurfaceInfo` for a face (periodicity flags).
fn build_surface_info(topo: &Topology, face_id: FaceId) -> Option<SurfaceInfo> {
    let face = topo.face(face_id).ok()?;
    match face.surface() {
        FaceSurface::Plane { .. } => None,
        FaceSurface::Cylinder(_) => Some(SurfaceInfo::Parametric {
            u_periodic: true,
            v_periodic: false,
        }),
        FaceSurface::Cone(_) => Some(SurfaceInfo::Parametric {
            u_periodic: true,
            v_periodic: false,
        }),
        FaceSurface::Sphere(_) => Some(SurfaceInfo::Parametric {
            u_periodic: true,
            v_periodic: false,
        }),
        FaceSurface::Torus(_) => Some(SurfaceInfo::Parametric {
            u_periodic: true,
            v_periodic: true,
        }),
        FaceSurface::Nurbs(_) => Some(SurfaceInfo::Parametric {
            u_periodic: false,
            v_periodic: false,
        }),
    }
}

/// Build a topology `Face` from a `SplitSubFace`.
///
/// Creates vertices at each 3D endpoint (deduplicating by position),
/// edges between consecutive vertices, a wire from the edges, and
/// a face with the split's surface.
#[allow(clippy::too_many_lines)]
fn build_topology_face(
    topo: &mut Topology,
    split: &super::split_types::SplitSubFace,
    tol: Tolerance,
    parent_face_id: FaceId,
    shared_edge_cache: &mut HashMap<(usize, usize), brepkit_topology::edge::EdgeId>,
) -> Option<FaceId> {
    if split.outer_wire.is_empty() {
        return None;
    }

    // Step 1: Create/find vertices for each unique 3D endpoint.
    // Use a BTreeMap keyed by quantized position to deduplicate.
    let mut vertex_cache: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
        BTreeMap::new();

    let quantize = |p: Point3| -> (i64, i64, i64) {
        let scale = 1.0 / tol.linear;
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    let get_or_create_vertex =
        |topo: &mut Topology,
         cache: &mut BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
         pt: Point3| {
            let key = quantize(pt);
            *cache
                .entry(key)
                .or_insert_with(|| topo.add_vertex(Vertex::new(pt, tol.linear)))
        };

    // Step 2: Create edges and oriented edges for the outer wire.
    let mut oriented_edges = Vec::with_capacity(split.outer_wire.len());

    for pcurve_edge in &split.outer_wire {
        let start_vid = get_or_create_vertex(topo, &mut vertex_cache, pcurve_edge.start_3d);
        let end_vid = get_or_create_vertex(topo, &mut vertex_cache, pcurve_edge.end_3d);

        // Edge sharing: prefer pave_block_id (cross-face, from FF intersection),
        // fall back to source_edge_idx (within-face, from forward+reverse loops).
        let cache_key = if let Some(pb_id) = pcurve_edge.pave_block_id {
            // Cross-face key: all faces sharing this FF section curve
            // get the same edge entity. Use a distinct namespace (usize::MAX)
            // to avoid collision with within-face keys.
            Some((usize::MAX, pb_id))
        } else {
            pcurve_edge
                .source_edge_idx
                .map(|idx| (parent_face_id.index(), idx))
        };
        let edge_id = if let Some(key) = cache_key {
            *shared_edge_cache.entry(key).or_insert_with(|| {
                topo.add_edge(Edge::new(start_vid, end_vid, pcurve_edge.curve_3d.clone()))
            })
        } else {
            topo.add_edge(Edge::new(start_vid, end_vid, pcurve_edge.curve_3d.clone()))
        };
        let forward = pcurve_edge.forward;
        oriented_edges.push(OrientedEdge::new(edge_id, forward));
    }

    if oriented_edges.is_empty() {
        return None;
    }

    // Step 3: Build wire.
    let wire = Wire::new(oriented_edges, true).ok()?;
    let wire_id = topo.add_wire(wire);

    // Step 4: Build inner wires (holes).
    let mut inner_wire_ids = Vec::new();
    for inner in &split.inner_wires {
        let mut inner_oriented = Vec::with_capacity(inner.len());
        for pcurve_edge in inner {
            let start_vid = get_or_create_vertex(topo, &mut vertex_cache, pcurve_edge.start_3d);
            let end_vid = get_or_create_vertex(topo, &mut vertex_cache, pcurve_edge.end_3d);
            let edge = Edge::new(start_vid, end_vid, pcurve_edge.curve_3d.clone());
            let edge_id = topo.add_edge(edge);
            inner_oriented.push(OrientedEdge::new(edge_id, pcurve_edge.forward));
        }
        if let Ok(inner_wire) = Wire::new(inner_oriented, true) {
            inner_wire_ids.push(topo.add_wire(inner_wire));
        }
    }

    // Step 5: Build face.
    let mut face = Face::new(wire_id, inner_wire_ids, split.surface.clone());
    if split.reversed {
        face.set_reversed(true);
    }
    let face_id = topo.add_face(face);

    Some(face_id)
}
