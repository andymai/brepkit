//! Split faces using `FaceInfo` data from the PaveFiller.
//!
//! For each face that has section pave blocks, converts them to
//! [`SectionEdge`] entries and calls [`split_face_2d`] to produce
//! geometric sub-faces. Faces without intersection data pass through
//! unchanged.

use std::collections::HashMap;
use std::hash::BuildHasher;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};

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
    topo: &Topology,
    arena: &GfaArena,
    _edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
    face_ranks: &HashMap<FaceId, Rank, S2>,
    tol: Tolerance,
) -> Vec<SubFace> {
    let mut sub_faces = Vec::new();

    // Pre-compute which faces have section edges from which curves
    let section_map = build_section_map(arena);

    for (&face_id, &rank) in face_ranks {
        let fi = arena.face_info(face_id);
        let has_sections = fi.is_some_and(|fi| !fi.pave_blocks_sc.is_empty());

        if !has_sections {
            // No sections: face passes through unchanged
            sub_faces.push(SubFace {
                parent_face: face_id,
                face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point: None,
            });
            continue;
        }

        // Build SectionEdge entries from pave block data
        let sections = build_section_edges(topo, arena, face_id, &section_map);

        if sections.is_empty() {
            sub_faces.push(SubFace {
                parent_face: face_id,
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

        if split_results.is_empty() {
            log::warn!("fill_images_faces: split_face_2d returned empty for face {face_id:?}");
            sub_faces.push(SubFace {
                parent_face: face_id,
                face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point: None,
            });
            continue;
        }

        // Each SplitSubFace represents a geometric sub-region of the face.
        // Compute a distinct interior point for each so the classifier
        // can determine inside/outside independently per sub-region.
        for split in &split_results {
            let pt = super::face_splitter::interior_point_3d(split, None);
            sub_faces.push(SubFace {
                parent_face: face_id,
                face_id, // same topology face — classification uses interior_point
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

        let start = match topo.vertex(edge.start()) {
            Ok(v) => v.point(),
            Err(_) => continue,
        };
        let end = match topo.vertex(edge.end()) {
            Ok(v) => v.point(),
            Err(_) => continue,
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
        });
    }

    sections
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
