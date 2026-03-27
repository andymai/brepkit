//! Split faces using `FaceInfo` data from the PaveFiller.
//!
//! For each face that has section pave blocks, converts them to
//! [`SectionEdge`] entries and calls [`split_face_2d`] to produce
//! geometric sub-faces. Faces without intersection data pass through
//! unchanged.

use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasher;

/// Quantized 3D position pair for CommonBlock edge matching.
type CbEdgeKey = ((i64, i64, i64), (i64, i64, i64));

/// Scale for vertex deduplication in the face splitter.
///
/// Uses 1e10 to match vertices from the same computation path that
/// may differ by floating-point noise (~1e-14). This is coarser than
/// bit-identical (1e12) but much tighter than modeling tolerance (1e7).
/// Vertices from the same plane-plane intersection that land on
/// different face splits will share VertexIds, reducing the Euler
/// vertex count. Geometrically distinct vertices (>1e-10 apart)
/// remain separate.
const VERTEX_DEDUP_SCALE: f64 = 1e10;

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
#[allow(clippy::too_many_lines, clippy::type_complexity)]
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

    // CommonBlock position-pair → shared EdgeId. When building sub-face edges,
    // if the edge endpoints match a CB's split_edge endpoints (by quantized
    // position), reuse the CB's edge entity. This ensures faces from different
    // input solids share the same EdgeId at their common boundaries.
    let cb_qpair_edges: HashMap<CbEdgeKey, brepkit_topology::edge::EdgeId> = {
        let scale = VERTEX_DEDUP_SCALE;
        let qpt = |p: brepkit_math::vec::Point3| -> (i64, i64, i64) {
            (
                (p.x() * scale).round() as i64,
                (p.y() * scale).round() as i64,
                (p.z() * scale).round() as i64,
            )
        };
        let mut map = HashMap::new();
        for (_, cb) in arena.common_blocks.iter() {
            if let Some(edge_id) = cb.split_edge {
                if let Ok(edge) = topo.edge(edge_id) {
                    if let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) {
                        let qs = qpt(sv.point());
                        let qe = qpt(ev.point());
                        let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
                        map.insert(key, edge_id);
                    }
                }
            }
        }
        map
    };

    // Build vertex seed from VV-phase merged vertices.
    let vv_vertex_seed: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> = {
        let scale = VERTEX_DEDUP_SCALE;
        let mut seed = BTreeMap::new();
        let canonical_vids: std::collections::HashSet<brepkit_topology::vertex::VertexId> =
            arena.same_domain_vertices.values().copied().collect();
        for &vid in &canonical_vids {
            if let Ok(v) = topo.vertex(vid) {
                let pt = v.point();
                let key = (
                    (pt.x() * scale).round() as i64,
                    (pt.y() * scale).round() as i64,
                    (pt.z() * scale).round() as i64,
                );
                seed.entry(key).or_insert(vid);
            }
        }
        seed
    };

    // Shared quantization helper for vertex position dedup.
    let qpos = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * VERTEX_DEDUP_SCALE).round() as i64,
            (p.y() * VERTEX_DEDUP_SCALE).round() as i64,
            (p.z() * VERTEX_DEDUP_SCALE).round() as i64,
        )
    };

    // PB vertex registry: cross-face pool of FRESH vertices at CB positions.
    let mut pb_vertex_registry: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
        BTreeMap::new();

    // ── CommonBlock vertex pre-pass ─────────────────────────────────
    // Create FRESH vertices at CommonBlock split edge positions.
    {
        let cb_positions: Vec<Point3> = arena
            .common_blocks
            .iter()
            .filter_map(|(_, cb)| {
                let eid = cb.split_edge?;
                let e = topo.edge(eid).ok()?;
                let mut pts = Vec::new();
                if let Ok(v) = topo.vertex(e.start()) {
                    pts.push(v.point());
                }
                if let Ok(v) = topo.vertex(e.end()) {
                    pts.push(v.point());
                }
                Some(pts)
            })
            .flatten()
            .collect();
        for pt in cb_positions {
            pb_vertex_registry
                .entry(qpos(pt))
                .or_insert_with(|| topo.add_vertex(Vertex::new(pt, tol.linear)));
        }
    }

    // ── Cross-rank fresh-vertex pool ──────────────────────────────
    // Create FRESH vertices at positions of face vertices shared by
    // 2+ unique faces (any rank). This covers box corners (3 faces)
    // and shared edge endpoints (2 faces). Using a single cross-rank
    // pool ensures that faces from different solids sharing a vertex
    // position get the SAME fresh VertexId, eliminating the Euler
    // vertex excess from per-rank duplication. Fresh vertices don't
    // connect to input solid topology (no contamination).
    let shared_vertex_pool: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> = {
        // Count UNIQUE faces per resolved vertex position (any rank).
        let mut vid_faces: HashMap<usize, (Point3, std::collections::HashSet<usize>)> =
            HashMap::new();
        for (&face_id, &_rank) in face_ranks {
            if let Ok(face) = topo.face(face_id) {
                if let Ok(wire) = topo.wire(face.outer_wire()) {
                    for oe in wire.edges() {
                        if let Ok(edge) = topo.edge(oe.edge()) {
                            for &vid in &[edge.start(), edge.end()] {
                                let rv = arena.resolve_vertex(vid);
                                if let Ok(v) = topo.vertex(rv) {
                                    let entry = vid_faces.entry(rv.index()).or_insert_with(|| {
                                        (v.point(), std::collections::HashSet::new())
                                    });
                                    entry.1.insert(face_id.index());
                                }
                            }
                        }
                    }
                }
            }
        }
        // Create fresh vertices for positions with 2+ unique faces.
        // Reuse CB pre-pass vertex if available at this position.
        let mut pool = BTreeMap::new();
        for (pt, faces) in vid_faces.values() {
            if faces.len() >= 2 {
                let key = qpos(*pt);
                pool.entry(key).or_insert_with(|| {
                    pb_vertex_registry
                        .get(&key)
                        .copied()
                        .unwrap_or_else(|| topo.add_vertex(Vertex::new(*pt, tol.linear)))
                });
            }
        }
        pool
    };

    // (pb_vertex_registry and CB pre-pass moved above rank pool)

    // Pre-compute which faces have section edges from which curves
    let section_map = build_section_map(arena);

    // No boundary edge cache — each face creates its own edges with its own
    // vertices. Cross-face edge sharing is handled by merge_duplicate_edges
    // in builder_solid. Sharing edges across parent faces via a position-pair
    // cache caused VertexId mismatches at wire junctions (different parent
    // faces have different vertex caches producing different IDs at the same
    // position).

    // Sort faces by ID index for deterministic processing order.
    // HashMap iteration order varies between compilations (different hash
    // seeds), which causes non-deterministic edge sharing in the
    // shared_edge_cache — an edge created by the first face processed
    // gets shared with later faces. Sorting ensures consistent results.
    let mut sorted_faces: Vec<(FaceId, Rank)> =
        face_ranks.iter().map(|(&fid, &r)| (fid, r)).collect();
    sorted_faces.sort_by_key(|(fid, _)| fid.index());

    for (face_id, rank) in sorted_faces {
        let fi = arena.face_info(face_id);
        let has_sections = fi.is_some_and(|fi| !fi.pave_blocks_sc.is_empty());

        log::debug!("fill_images_faces: face {face_id:?} has_sections={has_sections}");

        if !has_sections {
            // TODO: Use fresh-vertex face to achieve V=16.
            // Currently disabled — the fresh faces have correct topology
            // but incorrect geometry (volume 2/3 of expected, one face
            // normal flipped). Root cause under investigation.
            let _fresh = rebuild_face_with_fresh_vertices(
                topo,
                face_id,
                Some(&shared_vertex_pool),
                &mut pb_vertex_registry,
                &qpos,
                tol,
            );
            let rebuilt =
                rebuild_face_with_cb_edges(topo, face_id, &cb_qpair_edges, &vv_vertex_seed, tol);
            sub_faces.push(SubFace {
                face_id: rebuilt.unwrap_or(face_id),
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

        // Build the parent face's PlaneFrame for consistent UV→3D conversion.
        // interior_point_3d needs the SAME frame that the face splitter used
        // for UV projection; creating a new frame from sub-face wire points
        // uses a different origin and produces wrong 3D coordinates.
        let parent_frame = {
            let face = topo.face(face_id).ok();
            let is_plane = face
                .as_ref()
                .is_some_and(|f| matches!(f.surface(), FaceSurface::Plane { .. }));
            if is_plane {
                let normal = face
                    .as_ref()
                    .and_then(|f| match f.surface() {
                        FaceSurface::Plane { normal, .. } => Some(*normal),
                        _ => None,
                    })
                    .unwrap_or(brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0));
                let wire_pts: Vec<_> = face
                    .as_ref()
                    .and_then(|f| topo.wire(f.outer_wire()).ok())
                    .map(|w| {
                        w.edges()
                            .iter()
                            .filter_map(|oe| {
                                topo.edge(oe.edge())
                                    .ok()
                                    .and_then(|e| topo.vertex(e.start()).ok())
                                    .map(brepkit_topology::vertex::Vertex::point)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Some(super::plane_frame::PlaneFrame::from_plane_face(
                    normal, &wire_pts,
                ))
            } else {
                None
            }
        };

        // Each SplitSubFace represents a geometric sub-region.
        // Build real topology entities (Vertex → Edge → Wire → Face) for each,
        // and compute a distinct interior point for classification.
        for split in &split_results {
            let rank_pool = Some(&shared_vertex_pool);
            let new_face_id = build_topology_face(
                topo,
                split,
                tol,
                face_id,
                &mut shared_edge_cache,
                &cb_qpair_edges,
                &vv_vertex_seed,
                rank_pool,
                &mut pb_vertex_registry,
                arena,
            );
            let pt = split.precomputed_interior.unwrap_or_else(|| {
                super::face_splitter::interior_point_3d(split, parent_frame.as_ref())
            });

            sub_faces.push(SubFace {
                face_id: new_face_id.unwrap_or(face_id),
                classification: FaceClass::Unknown,
                rank,
                interior_point: Some(pt),
            });
        }
    }

    // ── Post-processing: merge duplicate vertices via wire rebuild ──
    // The per-face vertex cache creates separate vertices at the same
    // position. Instead of mutating shared edges in-place (which creates
    // crossed polygons), rebuild each face's wire with NEW edges using
    // canonical vertices. Each face gets its own edges — no sharing.
    let all_planar = sub_faces.iter().all(|sf| {
        topo.face(sf.face_id)
            .is_ok_and(|f| matches!(f.surface(), FaceSurface::Plane { .. }))
    });
    let ranks: std::collections::HashSet<_> = sub_faces.iter().map(|sf| sf.rank).collect();
    let no_inner_wires = sub_faces.iter().all(|sf| {
        topo.face(sf.face_id)
            .is_ok_and(|f| f.inner_wires().is_empty())
    });
    if all_planar && ranks.len() == 2 && no_inner_wires {
        let q12 = |p: Point3| -> (i64, i64, i64) {
            (
                (p.x() * 1e12).round() as i64,
                (p.y() * 1e12).round() as i64,
                (p.z() * 1e12).round() as i64,
            )
        };

        // Build per-rank merge maps.
        let mut rank_merge_maps: HashMap<Rank, HashMap<usize, brepkit_topology::vertex::VertexId>> =
            HashMap::new();
        {
            let mut rank_edges: HashMap<Rank, Vec<EdgeId>> = HashMap::new();
            for sf in &sub_faces {
                let edges = rank_edges.entry(sf.rank).or_default();
                if let Ok(face) = topo.face(sf.face_id) {
                    let mut seen = std::collections::HashSet::new();
                    if let Ok(wire) = topo.wire(face.outer_wire()) {
                        for oe in wire.edges() {
                            if seen.insert(oe.edge().index()) {
                                edges.push(oe.edge());
                            }
                        }
                    }
                }
            }
            for (&rank, edges) in &rank_edges {
                let mut canonical: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
                    BTreeMap::new();
                let mut merge_map: HashMap<usize, brepkit_topology::vertex::VertexId> =
                    HashMap::new();
                for &eid in edges {
                    if let Ok(edge) = topo.edge(eid) {
                        for &vid in &[edge.start(), edge.end()] {
                            if let Ok(v) = topo.vertex(vid) {
                                let key = q12(v.point());
                                let canon = *canonical.entry(key).or_insert(vid);
                                if canon != vid {
                                    merge_map.insert(vid.index(), canon);
                                }
                            }
                        }
                    }
                }
                if !merge_map.is_empty() {
                    rank_merge_maps.insert(rank, merge_map);
                }
            }
        }

        // Rebuild each SubFace's wire with NEW edges using merged vertices.
        for sf in &sub_faces {
            let merge_map = match rank_merge_maps.get(&sf.rank) {
                Some(m) => m,
                None => continue,
            };
            let (outer_oes, surface, is_reversed) = {
                let Ok(face) = topo.face(sf.face_id) else {
                    continue;
                };
                let Ok(wire) = topo.wire(face.outer_wire()) else {
                    continue;
                };
                (
                    wire.edges().to_vec(),
                    face.surface().clone(),
                    face.is_reversed(),
                )
            };
            let mut any_changed = false;
            let mut new_oes = Vec::with_capacity(outer_oes.len());
            for oe in &outer_oes {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    new_oes.push(*oe);
                    continue;
                };
                // Get the TRAVERSAL-ORDER vertices (what the wire sees).
                let (trav_start, trav_end) = if oe.is_forward() {
                    (edge.start(), edge.end())
                } else {
                    (edge.end(), edge.start())
                };
                let ns = merge_map
                    .get(&trav_start.index())
                    .copied()
                    .unwrap_or(trav_start);
                let ne = merge_map
                    .get(&trav_end.index())
                    .copied()
                    .unwrap_or(trav_end);
                if ns == ne {
                    // Degenerate after merge — skip
                    continue;
                }
                if ns != trav_start || ne != trav_end {
                    // Create NEW edge in traversal order (start→end = forward).
                    let new_eid = topo.add_edge(Edge::new(ns, ne, edge.curve().clone()));
                    new_oes.push(OrientedEdge::new(new_eid, true));
                    any_changed = true;
                } else {
                    new_oes.push(*oe);
                }
            }
            if any_changed && new_oes.len() >= 3 {
                if let Ok(new_wire) = Wire::new(new_oes, true) {
                    let wid = topo.add_wire(new_wire);
                    let new_face = if is_reversed {
                        Face::new_reversed(wid, vec![], surface)
                    } else {
                        Face::new(wid, vec![], surface)
                    };
                    if let Ok(face) = topo.face_mut(sf.face_id) {
                        *face = new_face;
                    }
                }
            }
        }
    }

    sub_faces
}

/// Create a NEW face from an unsplit face using fresh pool vertices.
#[allow(dead_code)]
fn rebuild_face_with_fresh_vertices(
    topo: &mut Topology,
    face_id: FaceId,
    rank_pool: Option<&BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>>,
    pb_registry: &mut BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    qpos: &dyn Fn(Point3) -> (i64, i64, i64),
    tol: Tolerance,
) -> Option<FaceId> {
    let face = topo.face(face_id).ok()?;
    let surface = face.surface().clone();
    let is_reversed = face.is_reversed();
    let outer_wid = face.outer_wire();
    let inner_wids: Vec<_> = face.inner_wires().to_vec();

    let wire = topo.wire(outer_wid).ok()?;
    let orig_edges: Vec<_> = wire
        .edges()
        .iter()
        .map(|oe| {
            let edge = topo.edge(oe.edge()).ok()?;
            let sv = topo.vertex(edge.start()).ok()?;
            let ev = topo.vertex(edge.end()).ok()?;
            Some((
                oe.is_forward(),
                edge.curve().clone(),
                sv.point(),
                ev.point(),
            ))
        })
        .collect::<Option<Vec<_>>>()?;

    let mut new_edges: Vec<(bool, brepkit_topology::edge::EdgeId)> = Vec::new();
    for (is_fwd, curve, sp, ep) in &orig_edges {
        let start_vid = {
            let key = qpos(*sp);
            rank_pool
                .and_then(|p| p.get(&key).copied())
                .unwrap_or_else(|| {
                    *pb_registry
                        .entry(key)
                        .or_insert_with(|| topo.add_vertex(Vertex::new(*sp, tol.linear)))
                })
        };
        let end_vid = {
            let key = qpos(*ep);
            rank_pool
                .and_then(|p| p.get(&key).copied())
                .unwrap_or_else(|| {
                    *pb_registry
                        .entry(key)
                        .or_insert_with(|| topo.add_vertex(Vertex::new(*ep, tol.linear)))
                })
        };
        let eid = topo.add_edge(Edge::new(start_vid, end_vid, curve.clone()));
        new_edges.push((*is_fwd, eid));
    }

    let oes: Vec<_> = new_edges
        .iter()
        .map(|(is_fwd, eid)| OrientedEdge::new(*eid, *is_fwd))
        .collect();
    let new_wire = topo.add_wire(Wire::new(oes, true).ok()?);

    let new_face = if is_reversed {
        Face::new_reversed(new_wire, inner_wids, surface)
    } else {
        Face::new(new_wire, inner_wids, surface)
    };
    Some(topo.add_face(new_face))
}

/// Map from face ID to section pave block IDs (from FF intersection curves).
/// Rebuild a face expanding boundary edges that have been split into
/// multiple children. Only expands edges with 2+ split images; single-edge
/// replacements (1:1 CB mappings) are left for `merge_duplicate_edges`.
#[allow(clippy::too_many_lines, dead_code)]
fn rebuild_face_with_edge_images<S: BuildHasher>(
    topo: &mut Topology,
    face_id: FaceId,
    edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
) -> Option<FaceId> {
    let (surface, is_reversed, outer_edges, inner_edges_list) = {
        let face = topo.face(face_id).ok()?;
        let surface = face.surface().clone();
        let is_reversed = face.is_reversed();
        let outer_wire = topo.wire(face.outer_wire()).ok()?;
        let outer_edges: Vec<(EdgeId, bool)> = outer_wire
            .edges()
            .iter()
            .map(|oe| (oe.edge(), oe.is_forward()))
            .collect();
        let inner_wids = face.inner_wires().to_vec();
        let mut inner_edges_list = Vec::new();
        for &iw in &inner_wids {
            if let Ok(w) = topo.wire(iw) {
                inner_edges_list.push(
                    w.edges()
                        .iter()
                        .map(|oe| (oe.edge(), oe.is_forward()))
                        .collect::<Vec<_>>(),
                );
            }
        }
        (surface, is_reversed, outer_edges, inner_edges_list)
    };

    // Only expand LINE edges with multi-split images. Curved edges
    // (Circle, Ellipse, NURBS) need special angular-range handling
    // that this simple expand_edge doesn't support.
    let has_multi_split = outer_edges
        .iter()
        .chain(inner_edges_list.iter().flatten())
        .any(|(eid, _)| {
            edge_images.get(eid).is_some_and(|imgs| imgs.len() > 1)
                && topo
                    .edge(*eid)
                    .is_ok_and(|e| matches!(e.curve(), brepkit_topology::edge::EdgeCurve::Line))
        });

    if !has_multi_split {
        return None;
    }

    let new_outer_oes: Vec<OrientedEdge> = outer_edges
        .iter()
        .flat_map(|&(eid, fwd)| expand_edge(topo, eid, fwd, edge_images))
        .collect();
    let new_outer = Wire::new(new_outer_oes, true).ok()?;
    let new_outer_id = topo.add_wire(new_outer);

    let mut new_inner_ids = Vec::new();
    for inner_edges in &inner_edges_list {
        let oes: Vec<OrientedEdge> = inner_edges
            .iter()
            .flat_map(|&(eid, fwd)| expand_edge(topo, eid, fwd, edge_images))
            .collect();
        if let Ok(w) = Wire::new(oes, true) {
            new_inner_ids.push(topo.add_wire(w));
        } else {
            // Inner wire reconstruction failed — fall back to the
            // original face to avoid silently dropping holes.
            log::warn!(
                "rebuild_face_with_edge_images: inner wire failed for \
                 face {face_id:?}, keeping original"
            );
            return None;
        }
    }

    let mut new_face = Face::new(new_outer_id, new_inner_ids, surface);
    if is_reversed {
        new_face.set_reversed(true);
    }
    let new_fid = topo.add_face(new_face);
    log::debug!("rebuild_face_with_edge_images: face {face_id:?} → {new_fid:?}");
    Some(new_fid)
}

/// Expand a single edge into its multi-split image edges.
/// Only expands Line edges with 2+ children; keeps everything else as-is.
#[allow(dead_code)]
fn expand_edge<S: BuildHasher>(
    topo: &Topology,
    eid: EdgeId,
    fwd: bool,
    edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
) -> Vec<OrientedEdge> {
    let imgs = match edge_images.get(&eid) {
        Some(imgs) if imgs.len() > 1 => imgs,
        _ => return vec![OrientedEdge::new(eid, fwd)],
    };
    // Only expand Line edges
    if !topo
        .edge(eid)
        .is_ok_and(|e| matches!(e.curve(), brepkit_topology::edge::EdgeCurve::Line))
    {
        return vec![OrientedEdge::new(eid, fwd)];
    }
    if fwd {
        imgs.iter()
            .map(|&img| OrientedEdge::new(img, true))
            .collect()
    } else {
        imgs.iter()
            .rev()
            .map(|&img| OrientedEdge::new(img, false))
            .collect()
    }
}

/// Rebuild an unsplit face replacing boundary edges with CommonBlock shared edges.
///
/// For each boundary edge of the face, checks if its PaveBlock belongs to a
/// CommonBlock. If so, replaces the edge with the CB's `split_edge`. This
/// ensures that unsplit faces from different solids share edge entities at
/// their common boundaries.
///
/// Returns `Some(new_face_id)` if any edges were replaced, `None` if unchanged.
/// Falls back to `None` (keeping the original face) if any wire rebuild fails.
#[allow(clippy::too_many_lines)]
fn rebuild_face_with_cb_edges(
    topo: &mut Topology,
    face_id: FaceId,
    cb_qpair_edges: &HashMap<CbEdgeKey, brepkit_topology::edge::EdgeId>,
    vv_vertex_seed: &std::collections::BTreeMap<
        (i64, i64, i64),
        brepkit_topology::vertex::VertexId,
    >,
    _tol: Tolerance,
) -> Option<FaceId> {
    if cb_qpair_edges.is_empty() && vv_vertex_seed.is_empty() {
        return None;
    }

    let face = topo.face(face_id).ok()?;
    let surface = face.surface().clone();
    let is_reversed = face.is_reversed();
    let outer_wid = face.outer_wire();
    let inner_wids: Vec<_> = face.inner_wires().to_vec();

    // Use VERTEX_DEDUP_SCALE consistently for all position lookups —
    // both VV vertex seed and CB edge matching.
    let scale = VERTEX_DEDUP_SCALE;
    let qpt = |p: brepkit_math::vec::Point3| -> (i64, i64, i64) {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    // Check if any edge needs replacement (CB edge or vertex canonicalization).
    // Uses a block scope so the immutable borrow of `topo` is released before
    // the mutable `remap_wire` closure below.
    let any_replaced = {
        let check_wire = |wid: brepkit_topology::wire::WireId| -> bool {
            let Ok(wire) = topo.wire(wid) else {
                return false;
            };
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                let Ok(sv) = topo.vertex(edge.start()) else {
                    continue;
                };
                let Ok(ev) = topo.vertex(edge.end()) else {
                    continue;
                };
                let qs = qpt(sv.point());
                let qe = qpt(ev.point());
                // Check CB edge replacement
                let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
                if let Some(&cb_edge) = cb_qpair_edges.get(&key) {
                    if cb_edge != oe.edge() {
                        return true;
                    }
                }
                // Check VV vertex canonicalization
                if vv_vertex_seed
                    .get(&qs)
                    .is_some_and(|&vid| vid != edge.start())
                {
                    return true;
                }
                if vv_vertex_seed
                    .get(&qe)
                    .is_some_and(|&vid| vid != edge.end())
                {
                    return true;
                }
            }
            false
        };
        let mut found = check_wire(outer_wid);
        if !found {
            for &iw in &inner_wids {
                if check_wire(iw) {
                    found = true;
                    break;
                }
            }
        }
        found
    };

    if !any_replaced {
        return None;
    }

    // Rebuild wires with CB edge replacements + vertex canonicalization.
    // For each edge: (1) if it matches a CB, use the CB's shared edge.
    // (2) Otherwise, if its start or end vertex has a canonical VV vertex
    //     at the same position, create a new edge with the canonical vertex.
    // This ensures ALL boundary edges share canonical vertices, not just
    // CB-matched edges.
    let remap_wire = |topo: &mut Topology,
                      wid: brepkit_topology::wire::WireId|
     -> Option<brepkit_topology::wire::WireId> {
        // Snapshot wire data (snapshot-then-allocate pattern)
        let wire = topo.wire(wid).ok()?;
        let snap: Vec<_> = wire
            .edges()
            .iter()
            .map(|oe| {
                let edge = topo.edge(oe.edge()).ok();
                let (start_vid, end_vid, start_q, end_q, curve) = if let Some(e) = edge {
                    let sv = topo
                        .vertex(e.start())
                        .ok()
                        .map(brepkit_topology::vertex::Vertex::point);
                    let ev = topo
                        .vertex(e.end())
                        .ok()
                        .map(brepkit_topology::vertex::Vertex::point);
                    let qs = sv.map(&qpt);
                    let qe = ev.map(&qpt);
                    (
                        Some(e.start()),
                        Some(e.end()),
                        qs,
                        qe,
                        Some(e.curve().clone()),
                    )
                } else {
                    (None, None, None, None, None)
                };
                (
                    oe.edge(),
                    oe.is_forward(),
                    start_vid,
                    end_vid,
                    start_q,
                    end_q,
                    curve,
                )
            })
            .collect();

        // Pre-lookup CB edge start positions (needed for orientation)
        let cb_start_qs: HashMap<brepkit_topology::edge::EdgeId, (i64, i64, i64)> = {
            let mut m = HashMap::new();
            for &eid in cb_qpair_edges.values() {
                if let Ok(e) = topo.edge(eid) {
                    if let Ok(v) = topo.vertex(e.start()) {
                        m.insert(eid, qpt(v.point()));
                    }
                }
            }
            m
        };

        // Allocate new edges where needed
        let mut oes = Vec::with_capacity(snap.len());
        for (eid, fwd, start_vid, end_vid, start_q, end_q, curve) in snap {
            let (Some(sv), Some(ev), Some(qs), Some(qe)) = (start_vid, end_vid, start_q, end_q)
            else {
                oes.push(OrientedEdge::new(eid, fwd));
                continue;
            };

            // (1) CB edge replacement
            let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
            if let Some(&cb_edge) = cb_qpair_edges.get(&key) {
                if cb_edge != eid {
                    let oriented_start_q = if fwd { qs } else { qe };
                    // If we can't look up the CB edge's start position,
                    // preserve the original orientation rather than
                    // guessing `false`.
                    let new_fwd = cb_start_qs
                        .get(&cb_edge)
                        .map_or(fwd, |&cs| cs == oriented_start_q);
                    oes.push(OrientedEdge::new(cb_edge, new_fwd));
                    continue;
                }
            }

            // (2) Vertex canonicalization via VV seed
            let canon_start = vv_vertex_seed.get(&qs).copied().filter(|&vid| vid != sv);
            let canon_end = vv_vertex_seed.get(&qe).copied().filter(|&vid| vid != ev);
            if let (true, Some(curve)) = (canon_start.is_some() || canon_end.is_some(), curve) {
                let new_s = canon_start.unwrap_or(sv);
                let new_e = canon_end.unwrap_or(ev);
                let new_edge = Edge::new(new_s, new_e, curve);
                let new_eid = topo.add_edge(new_edge);
                oes.push(OrientedEdge::new(new_eid, fwd));
                continue;
            }

            oes.push(OrientedEdge::new(eid, fwd));
        }
        let new_wire = Wire::new(oes, true).ok()?;
        Some(topo.add_wire(new_wire))
    };

    let new_outer = remap_wire(topo, outer_wid)?;
    let mut new_inner_ids = Vec::new();
    for &iw in &inner_wids {
        // If remapping fails, keep the original inner wire rather than
        // silently dropping it (which would remove a hole from the face).
        new_inner_ids.push(remap_wire(topo, iw).unwrap_or(iw));
    }

    let mut new_face = Face::new(new_outer, new_inner_ids, surface);
    if is_reversed {
        new_face.set_reversed(true);
    }
    let new_fid = topo.add_face(new_face);
    log::debug!(
        "rebuild_face_with_cb_edges: face {face_id:?} → {new_fid:?} (replaced CB boundary edges)"
    );
    Some(new_fid)
}

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

    // Deduplicate: remove section edges that are subsets of longer
    // collinear edges. This happens when both the FF phase and the
    // coplanar phase create section edges on the same line — the FF
    // edge spans the full face, the coplanar edge spans the inner
    // region only. Keeping both creates degenerate face splits.
    dedup_collinear_sections(&mut sections, tol);

    sections
}

/// Remove section edges that are subsets of longer collinear edges.
fn dedup_collinear_sections(sections: &mut Vec<SectionEdge>, tol: f64) {
    if sections.len() < 2 {
        return;
    }

    let n = sections.len();
    let mut to_remove = vec![false; n];

    for i in 0..n {
        if to_remove[i] {
            continue;
        }
        for j in (i + 1)..n {
            if to_remove[j] {
                continue;
            }

            let si = &sections[i];
            let sj = &sections[j];

            // Check collinearity: direction vectors must be parallel
            let di = sj.end - sj.start;
            let dj = si.end - si.start;
            let cross = di.cross(dj);
            if cross.length() > tol * 10.0 {
                continue;
            }

            // Check if on the same line: distance from si.start to line(sj)
            let to_sj = si.start - sj.start;
            let dj_len = dj.length();
            if dj_len < tol {
                continue;
            }
            let dj_unit = dj * (1.0 / dj_len);
            let perp = to_sj - dj_unit * to_sj.dot(dj_unit);
            if perp.length() > tol * 10.0 {
                continue;
            }

            // Collinear and on the same line. Remove the shorter one.
            let len_i = (si.end - si.start).length();
            let len_j = (sj.end - sj.start).length();
            if len_i < len_j - tol {
                to_remove[i] = true;
            } else if len_j < len_i - tol {
                to_remove[j] = true;
            }
            // If equal length, keep both (they might be distinct edges)
        }
    }

    let removed = to_remove.iter().filter(|&&r| r).count();
    if removed > 0 {
        let mut idx = 0;
        sections.retain(|_| {
            let keep = !to_remove[idx];
            idx += 1;
            keep
        });
        log::debug!("dedup_collinear_sections: removed {removed} subset edges");
    }
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

    // Discard section edges that lie entirely ON a single face boundary edge.
    // This catches the case where the FF intersection of an adjacent coplanar
    // face produces a section line that coincides with one boundary edge.
    // Only discard if BOTH endpoints lie on the SAME boundary segment.
    for (seg_start, seg_end) in &boundary_segments {
        let start_dist = point_to_segment_dist_3d(clipped_start, *seg_start, *seg_end);
        let end_dist = point_to_segment_dist_3d(clipped_end, *seg_start, *seg_end);
        if start_dist < tol && end_dist < tol {
            return None;
        }
    }

    Some((clipped_start, clipped_end))
}

/// Distance from a 3D point to a line segment.
fn point_to_segment_dist_3d(pt: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let len_sq = ab.dot(ab);
    if len_sq < 1e-30 {
        return (pt - a).length();
    }
    let t = ((pt - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (pt - proj).length()
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

/// Compute quantized position pair for CommonBlock edge lookup.
///
/// When the edge has a `pave_block_id`, uses the PaveBlock's resolved vertex
/// positions (authoritative, from PaveFiller). Otherwise falls back to the
/// edge's `start_3d`/`end_3d` (UV→3D converted, may have floating-point noise).
///
/// Returns `None` if the PaveBlock or vertex lookup fails.
#[allow(dead_code)] // Used by rebuild_face_with_cb_edges; disabled for split sub-faces
fn cb_quantize_pair(
    topo: &Topology,
    arena: &crate::ds::GfaArena,
    edge: &super::split_types::OrientedPCurveEdge,
    scale: f64,
) -> Option<CbEdgeKey> {
    let qpt = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    // Prefer PaveBlock vertex positions when available.
    // pave_block_id is the raw arena index of the PaveBlock.
    let (sp, ep) = if let Some(pb_idx) = edge.pave_block_id {
        let pb_id = arena.pave_blocks.id_from_index(pb_idx);
        let pb = pb_id.and_then(|id| arena.pave_blocks.get(id));
        if let Some(pb) = pb {
            let sv = arena.resolve_vertex(pb.start.vertex);
            let ev = arena.resolve_vertex(pb.end.vertex);
            let sp = topo.vertex(sv).ok()?.point();
            let ep = topo.vertex(ev).ok()?.point();
            (sp, ep)
        } else {
            (edge.start_3d, edge.end_3d)
        }
    } else {
        (edge.start_3d, edge.end_3d)
    };

    let qs = qpt(sp);
    let qe = qpt(ep);
    Some(if qs <= qe { (qs, qe) } else { (qe, qs) })
}

/// Build a topology `Face` from a `SplitSubFace`.
///
/// Creates vertices at each 3D endpoint (deduplicating by position),
/// edges between consecutive vertices, a wire from the edges, and
/// a face with the split's surface.
/// Resolve vertices for a wire edge, using PaveBlock identity when available.
///
/// For section edges (with `pave_block_id`): looks up the PaveBlock's
/// start/end vertices from the arena. These are the authoritative vertices
/// created by the PaveFiller, ensuring consistent vertex identity across faces.
///
/// For boundary edges (without `pave_block_id`): falls back to position-based
/// cache lookup, creating new vertices only when none exists at the position.
fn resolve_edge_vertices(
    topo: &mut Topology,
    cache: &mut BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    pb_registry: &mut BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    edge: &super::split_types::OrientedPCurveEdge,
    arena: &crate::ds::GfaArena,
    quantize: &dyn Fn(Point3) -> (i64, i64, i64),
    tol: Tolerance,
) -> (
    brepkit_topology::vertex::VertexId,
    brepkit_topology::vertex::VertexId,
) {
    // Try PaveBlock-based vertex lookup for SHARED section edges only.
    // Only use split-edge vertices when the PB belongs to a CommonBlock
    // (shared across input solids). Non-CB section edges are local to
    // one solid and don't need vertex identity sharing.
    if let Some(pb_idx) = edge.pave_block_id {
        let pb_id = arena.pave_blocks.id_from_index(pb_idx);
        let is_cb = pb_id.is_some_and(|id| arena.pb_to_cb.contains_key(&id));
        let pb = pb_id.and_then(|id| arena.pave_blocks.get(id));
        if let Some(pb) = pb {
            if let (true, Some(split_edge)) = (is_cb, pb.split_edge) {
                // Use the split edge's actual vertices — these are the topology
                // entities created by MakeSplitEdges and shared via CommonBlocks.
                if let Ok(se) = topo.edge(split_edge) {
                    let se_start = se.start();
                    let se_end = se.end();

                    // Verify position match (section edges can be forward or reversed)
                    let start_pos = topo
                        .vertex(se_start)
                        .ok()
                        .map(brepkit_topology::vertex::Vertex::point);
                    let end_pos = topo
                        .vertex(se_end)
                        .ok()
                        .map(brepkit_topology::vertex::Vertex::point);

                    if let (Some(sp), Some(ep)) = (start_pos, end_pos) {
                        let fwd_match = (sp - edge.start_3d).length() < tol.linear
                            && (ep - edge.end_3d).length() < tol.linear;
                        let rev_match = (sp - edge.end_3d).length() < tol.linear
                            && (ep - edge.start_3d).length() < tol.linear;

                        if fwd_match {
                            let qs = quantize(edge.start_3d);
                            let qe = quantize(edge.end_3d);
                            // Use fresh vertex from cache/registry if available
                            // (from rank pool or CB pre-pass). Fall back to
                            // the split_edge's actual vertex only if no fresh
                            // vertex exists. This prevents topology connections
                            // between the GFA result and the PaveFiller's
                            // intermediate split edges.
                            let vs = *cache
                                .entry(qs)
                                .or_insert_with(|| *pb_registry.entry(qs).or_insert(se_start));
                            let ve = *cache
                                .entry(qe)
                                .or_insert_with(|| *pb_registry.entry(qe).or_insert(se_end));
                            return (vs, ve);
                        }
                        if rev_match {
                            let qs = quantize(edge.start_3d);
                            let qe = quantize(edge.end_3d);
                            let vs = *cache
                                .entry(qs)
                                .or_insert_with(|| *pb_registry.entry(qs).or_insert(se_end));
                            let ve = *cache
                                .entry(qe)
                                .or_insert_with(|| *pb_registry.entry(qe).or_insert(se_start));
                            return (vs, ve);
                        }
                    }
                }
            }
        }
    }

    // Fallback: position-based cache lookup.
    // Consult the PB registry first — if another face's PaveBlock
    // vertex was registered at this position, reuse it to ensure
    // cross-face vertex sharing.
    let start_vid = {
        let key = quantize(edge.start_3d);
        *cache.entry(key).or_insert_with(|| {
            pb_registry
                .get(&key)
                .copied()
                .unwrap_or_else(|| topo.add_vertex(Vertex::new(edge.start_3d, tol.linear)))
        })
    };
    let end_vid = {
        let key = quantize(edge.end_3d);
        *cache.entry(key).or_insert_with(|| {
            pb_registry
                .get(&key)
                .copied()
                .unwrap_or_else(|| topo.add_vertex(Vertex::new(edge.end_3d, tol.linear)))
        })
    };
    (start_vid, end_vid)
}

#[allow(
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::too_many_arguments
)]
fn build_topology_face(
    topo: &mut Topology,
    split: &super::split_types::SplitSubFace,
    tol: Tolerance,
    parent_face_id: FaceId,
    shared_edge_cache: &mut HashMap<(usize, usize), brepkit_topology::edge::EdgeId>,
    _cb_qpair_edges: &HashMap<CbEdgeKey, brepkit_topology::edge::EdgeId>,
    vv_vertex_seed: &BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    rank_pool: Option<&BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>>,
    pb_vertex_registry: &mut BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    arena: &crate::ds::GfaArena,
) -> Option<FaceId> {
    if split.outer_wire.is_empty() {
        return None;
    }

    // Step 1: Create/find vertices for each unique 3D endpoint.
    // Seed from VV-merged vertices, then from this rank's fresh-vertex
    // pool. The rank pool provides per-solid shared fresh vertices at
    // PaveBlock endpoint positions, avoiding cross-solid contamination.
    let mut vertex_cache: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
        vv_vertex_seed.clone();
    if let Some(pool) = rank_pool {
        for (&key, &vid) in pool {
            vertex_cache.entry(key).or_insert(vid);
        }
    }

    let quantize = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * VERTEX_DEDUP_SCALE).round() as i64,
            (p.y() * VERTEX_DEDUP_SCALE).round() as i64,
            (p.z() * VERTEX_DEDUP_SCALE).round() as i64,
        )
    };

    // Step 2: Create edges and oriented edges for the outer wire.
    let mut oriented_edges = Vec::with_capacity(split.outer_wire.len());

    for pcurve_edge in &split.outer_wire {
        // Vertex resolution priority:
        // 1. PaveBlock vertex identity (section edges from FF intersection)
        // 2. Position-based cache (boundary edges, degenerate edges)
        let (start_vid, end_vid) = resolve_edge_vertices(
            topo,
            &mut vertex_cache,
            pb_vertex_registry,
            pcurve_edge,
            arena,
            &quantize,
            tol,
        );

        // Edge sharing priority:
        // 0. CommonBlock position match — ONLY for edges with pave_block_id
        //    (section edges from FF intersection). Boundary edges must NOT
        //    use CB lookup because the global cb_qpair_edges map can match
        //    CB edges from unrelated face pairs at the same position
        //    (e.g., edge (1,0,0)→(1,0,1) exists on y=0, y=1, z=0, z=1 planes).
        // 1. pave_block_id cache (cross-face, from FF intersection)
        // 2. source_edge_idx cache (within-face, from forward+reverse loops)
        // 3. New edge (no sharing)
        // Edge sharing for split sub-faces uses pave_block_id cache (cross-face
        // sharing from FF intersection) and source_edge_idx cache (within-face
        // sharing from forward+reverse loops). The global cb_qpair_edges map is
        // NOT used here because it can match CB edges from unrelated face pairs
        // at the same position (e.g., edge at (1,0,0)→(1,0,1) exists on y=0,
        // z=0, and x=1 planes). cb_qpair_edges is only used by
        // rebuild_face_with_cb_edges for unsplit faces.
        let edge_id = if let Some(cb_edge) = None::<brepkit_topology::edge::EdgeId> {
            cb_edge
        } else if let Some(pb_id) = pcurve_edge.pave_block_id {
            let key = (usize::MAX, pb_id);
            *shared_edge_cache.entry(key).or_insert_with(|| {
                topo.add_edge(Edge::new(start_vid, end_vid, pcurve_edge.curve_3d.clone()))
            })
        } else if let Some(idx) = pcurve_edge.source_edge_idx {
            let key = (parent_face_id.index(), idx);
            *shared_edge_cache.entry(key).or_insert_with(|| {
                topo.add_edge(Edge::new(start_vid, end_vid, pcurve_edge.curve_3d.clone()))
            })
        } else {
            // Each face creates its own boundary edges with its own vertices.
            // Cross-face sharing is handled by merge_duplicate_edges in
            // builder_solid. This avoids VertexId mismatches that occur when
            // edges are shared across parent faces with different vertex caches.
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
            let (start_vid, end_vid) = resolve_edge_vertices(
                topo,
                &mut vertex_cache,
                pb_vertex_registry,
                pcurve_edge,
                arena,
                &quantize,
                tol,
            );
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
