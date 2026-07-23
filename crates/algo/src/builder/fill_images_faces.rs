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
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
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
    edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
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
            if let Some(edge_id) = cb.split_edge
                && let Ok(edge) = topo.edge(edge_id)
                && let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
            {
                let qs = qpt(sv.point());
                let qe = qpt(ev.point());
                let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
                map.insert(key, edge_id);
            }
        }
        map
    };

    // Build vertex seed from VV-phase merged vertices.
    //
    // `same_domain_vertices` is a BTreeMap (deterministic iteration), but
    // the previous implementation collected its values into a HashSet for
    // dedup before iterating. HashSet iteration uses a randomized hasher,
    // so the order of `seed.entry(key).or_insert(vid)` calls varied per
    // run — when two distinct VertexIds quantized to the same key, the
    // "winning" vid was nondeterministic. That nondeterminism propagated
    // into face ordering in the result solid and ultimately drove
    // 100-500× variance in `bench_boolean_64_holes`.
    //
    // Fix: dedup the BTreeMap values via a sorted Vec. The canonical
    // policy is now explicit — lowest VertexId wins per quantized key,
    // independent of how `same_domain_vertices` was populated. Different
    // from the BTreeMap's natural value order (which is the order keys
    // were inserted), but that order wasn't load-bearing here, and an
    // explicit "lowest id wins" policy is easier to reason about than
    // "whatever the BTreeMap's value iteration happened to be."
    let vv_vertex_seed: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> = {
        let scale = VERTEX_DEDUP_SCALE;
        let mut seed = BTreeMap::new();
        let mut canonical_vids: Vec<brepkit_topology::vertex::VertexId> =
            arena.same_domain_vertices.values().copied().collect();
        canonical_vids.sort();
        canonical_vids.dedup();
        for vid in canonical_vids {
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
            if let Ok(face) = topo.face(face_id)
                && let Ok(wire) = topo.wire(face.outer_wire())
            {
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
        // Create fresh vertices for positions with 2+ unique faces.
        // Reuse CB pre-pass vertex if available at this position.
        //
        // Sort by the resolved VertexId index so that `topo.add_vertex` is
        // called in a deterministic order. Iterating `vid_faces` directly
        // picks up HashMap iteration order, which uses a random per-process
        // seed — different orderings assign different fresh VertexIds to
        // the same quantized positions, producing nondeterministic topology
        // downstream and intermittent compound-boolean failures.
        let mut pool = BTreeMap::new();
        let mut sorted_entries: Vec<(usize, &(Point3, std::collections::HashSet<usize>))> =
            vid_faces.iter().map(|(&k, v)| (k, v)).collect();
        sorted_entries.sort_by_key(|(k, _)| *k);
        for (_, (pt, faces)) in sorted_entries {
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

    let section_map = build_section_map(topo, arena);

    // ── Periodic seam anchor pre-pass ───────────────────────────────
    // Closed circle intersection curves on a u-periodic face (cylinder /
    // cone lateral) are re-anchored so the circle's start/end vertex sits
    // on the face's seam. The band splitter connects consecutive circles
    // with seam segments; without re-anchoring, those segments would join
    // arbitrary section angles and cut through the surface interior.
    // Pre-registering the anchor vertices makes the periodic face's band
    // wires and the opposing plane faces' hole wires resolve to the same
    // VertexId, so merge_duplicate_edges can share the circle edge.
    let seam_anchors = compute_seam_anchors(topo, arena);
    for &anchor in seam_anchors.values() {
        pb_vertex_registry
            .entry(qpos(anchor))
            .or_insert_with(|| topo.add_vertex(Vertex::new(anchor, tol.linear)));
    }

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
    // Plane faces first: their arrangements register section split points
    // (keyed by pave block) that curved faces sharing the same section curves
    // consume, so both sides of a shared curve split at identical 3D points.
    sorted_faces.sort_by_key(|(fid, _)| {
        let curved = topo
            .face(*fid)
            .map(|f| {
                !matches!(
                    f.surface(),
                    brepkit_topology::face::FaceSurface::Plane { .. }
                )
            })
            .unwrap_or(true);
        (curved, fid.index())
    });
    let mut section_split_registry: std::collections::HashMap<
        usize,
        Vec<brepkit_math::vec::Point3>,
    > = std::collections::HashMap::new();

    for (face_id, rank) in sorted_faces {
        let fi = arena.face_info(face_id);
        let has_sections =
            fi.is_some_and(|fi| !fi.pave_blocks_sc.is_empty() || !fi.pave_blocks_in.is_empty());

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
            let expanded =
                rebuild_face_with_edge_images(topo, face_id, edge_images).unwrap_or(face_id);
            let rebuilt =
                rebuild_face_with_cb_edges(topo, expanded, &cb_qpair_edges, &vv_vertex_seed, tol);
            sub_faces.push(SubFace {
                face_id: rebuilt.unwrap_or(expanded),
                source_face: face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point: None,
            });
            continue;
        }

        let sections = build_section_edges(
            topo,
            arena,
            face_id,
            &section_map,
            &seam_anchors,
            tol.linear,
        );

        log::debug!(
            "fill_images_faces: face {:?} has_sections={} sections={}",
            face_id,
            has_sections,
            sections.len()
        );
        if sections.is_empty() {
            let expanded =
                rebuild_face_with_edge_images(topo, face_id, edge_images).unwrap_or(face_id);
            sub_faces.push(SubFace {
                face_id: expanded,
                source_face: face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point: None,
            });
            continue;
        }

        // Build SurfaceInfo for periodicity
        let info = build_surface_info(topo, face_id);

        // Curved faces: pre-split sections at the registry's points for the
        // same pave block (the plane-side arrangement split the shared curve
        // there; the endpoint-T machinery cascades the knock-on splits).
        let is_plane_face = topo
            .face(face_id)
            .map(|f| {
                matches!(
                    f.surface(),
                    brepkit_topology::face::FaceSurface::Plane { .. }
                )
            })
            .unwrap_or(false);
        let sections = if !is_plane_face && !section_split_registry.is_empty() {
            presplit_sections_at_registry(&sections, &section_split_registry, tol.linear)
        } else {
            sections
        };

        let split_results = split_face_2d(
            topo,
            face_id,
            &sections,
            rank,
            &tol,
            None, // PlaneFrame built internally by face_splitter
            info.as_ref(),
            edge_images,
            Some(&mut section_split_registry),
        );

        log::debug!(
            "fill_images_faces: face {face_id:?} split into {} sub-faces",
            split_results.len()
        );

        if split_results.is_empty() {
            log::warn!("fill_images_faces: split_face_2d returned empty for face {face_id:?}");
            let expanded =
                rebuild_face_with_edge_images(topo, face_id, edge_images).unwrap_or(face_id);
            sub_faces.push(SubFace {
                face_id: expanded,
                source_face: face_id,
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
            let resolved_face_id = new_face_id.unwrap_or(face_id);
            // For a curved-lens-hole wall (cylinder/cone with closed
            // Circle/Ellipse/NURBS holes) the generic `interior_point_3d` is
            // unsafe — it can sample inside the removed lens. The dedicated
            // search ran in the splitter; if it found a contained point it is in
            // `precomputed_interior`. When it is unset for such a face, leave the
            // interior unset so the classifier aborts the analytic split (→ mesh
            // fallback) instead of classifying the wall from inside the lens.
            let interior_point = match split.precomputed_interior {
                Some(pt) => Some(pt),
                None if super::face_splitter::face_has_curved_lens_holes(
                    topo,
                    resolved_face_id,
                ) =>
                {
                    // Not every split path that can leave a closed curved hole
                    // on a cylinder/cone wall runs the dedicated remainder
                    // search (e.g. a wall keeping a pre-existing lens hole
                    // through the generic split). Run it here as a last resort;
                    // only when even the dense grid finds nothing does the
                    // classifier abort to mesh.
                    super::face_splitter::cylinder_cone_remainder_interior(split)
                }
                None => Some(super::face_splitter::interior_point_3d(
                    split,
                    parent_frame.as_ref(),
                )),
            };

            sub_faces.push(SubFace {
                face_id: resolved_face_id,
                source_face: face_id,
                classification: FaceClass::Unknown,
                rank,
                interior_point,
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
            if any_changed
                && new_oes.len() >= 3
                && let Ok(new_wire) = Wire::new(new_oes, true)
            {
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
#[allow(clippy::too_many_lines)]
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
            non_degenerate_image_count(topo, *eid, edge_images) > 1
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

/// True when an image edge collapses to a point (shared start/end vertex).
///
/// Phase EF intersects each edge against the INFINITE plane of every face,
/// so a split point landing on an edge's own endpoint yields a zero-length
/// stub image. Propagating that stub into a rebuilt wire corrupts the face
/// and produces free edges, so such images must be discarded.
fn is_degenerate_image(topo: &Topology, img: EdgeId) -> bool {
    topo.edge(img).is_ok_and(|e| e.start() == e.end())
}

/// Count of an edge's split images that are not zero-length stubs.
fn non_degenerate_image_count<S: BuildHasher>(
    topo: &Topology,
    eid: EdgeId,
    edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
) -> usize {
    edge_images.get(&eid).map_or(0, |imgs| {
        imgs.iter()
            .filter(|&&img| !is_degenerate_image(topo, img))
            .count()
    })
}

/// Expand a single edge into its multi-split image edges.
/// Only expands Line edges with 2+ non-degenerate children; keeps
/// everything else (including edges whose only extra images are
/// zero-length stubs) as-is.
fn expand_edge<S: BuildHasher>(
    topo: &Topology,
    eid: EdgeId,
    fwd: bool,
    edge_images: &HashMap<EdgeId, Vec<EdgeId>, S>,
) -> Vec<OrientedEdge> {
    // Only expand Line edges
    if !topo
        .edge(eid)
        .is_ok_and(|e| matches!(e.curve(), brepkit_topology::edge::EdgeCurve::Line))
    {
        return vec![OrientedEdge::new(eid, fwd)];
    }
    let real_imgs: Vec<EdgeId> = match edge_images.get(&eid) {
        Some(imgs) => imgs
            .iter()
            .copied()
            .filter(|&img| !is_degenerate_image(topo, img))
            .collect(),
        None => return vec![OrientedEdge::new(eid, fwd)],
    };
    if real_imgs.len() < 2 {
        return vec![OrientedEdge::new(eid, fwd)];
    }
    if fwd {
        real_imgs
            .iter()
            .map(|&img| OrientedEdge::new(img, true))
            .collect()
    } else {
        real_imgs
            .iter()
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
                let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
                if let Some(&cb_edge) = cb_qpair_edges.get(&key)
                    && cb_edge != oe.edge()
                {
                    return true;
                }
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
                if let Ok(e) = topo.edge(eid)
                    && let Ok(v) = topo.vertex(e.start())
                {
                    m.insert(eid, qpt(v.point()));
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
            if let Some(&cb_edge) = cb_qpair_edges.get(&key)
                && cb_edge != eid
            {
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

/// Source of a section edge — either a complete intersection curve or
/// an individual PaveBlock (for IN edges that don't belong to a curve).
enum SectionSource {
    /// Complete intersection curve — use the curve's geometry, not individual PBs.
    Curve(usize),
    /// Individual PaveBlock. The optional face is the OPPOSING FF face that the
    /// same intersection line bounds; a Line section is clipped to it as well as
    /// to this face, so a wide face (e.g. a box cap) does not extend the section
    /// past where the narrow opposing face (e.g. a notch tool's front) actually
    /// is. `None` for IN edges from EF, which have no single opposing face.
    PaveBlock(PaveBlockId, Option<FaceId>),
}

/// Tolerance on `|t_range| - 2π` for treating a circle edge as a full circle.
/// Angular (radian) comparison on a parameter span.
const FULL_CIRCLE_T_TOL: f64 = 1e-9;

/// Minimum 3D length for a seam Line edge to count as non-degenerate.
const SEAM_DEGENERATE_TOL: f64 = 1e-10;

/// Radial/planar tolerance for accepting the seam point as lying on the
/// circle. Looser than the linear default (1e-7) because the anchor is built
/// from two `project_point` + `evaluate` round-trips whose float error
/// accumulates; tightening it would spuriously reject valid anchors.
const SEAM_ON_CIRCLE_TOL: f64 = 1e-6;

/// Compute seam-anchored start points for closed circle intersection curves.
///
/// For each full-circle FF curve whose face pair includes a u-periodic
/// surface (cylinder/cone) with a seam Line edge, returns the point on the
/// circle at the seam's u parameter. Keyed by the curve's arena index.
fn compute_seam_anchors(topo: &Topology, arena: &GfaArena) -> BTreeMap<usize, Point3> {
    use std::f64::consts::TAU;

    let mut anchors = BTreeMap::new();
    for (idx, curve_ds) in arena.curves.iter().enumerate() {
        let EdgeCurve::Circle(circle) = &curve_ds.curve else {
            continue;
        };
        let (t0, t1) = curve_ds.t_range;
        if ((t1 - t0).abs() - TAU).abs() > FULL_CIRCLE_T_TOL {
            continue;
        }
        for fid in [curve_ds.face_a, curve_ds.face_b] {
            let Ok(face) = topo.face(fid) else { continue };
            let surface = face.surface();
            if !matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) {
                continue;
            }
            let Some(anchor) = seam_anchor_on_circle(topo, face, circle) else {
                continue;
            };
            anchors.insert(idx, anchor);
            break;
        }
    }
    anchors
}

/// Find the point on `circle` at the seam u of `face`'s periodic surface.
///
/// Returns `None` if the face has no non-degenerate seam Line edge, the
/// projections fail, or the seam point at the circle's v does not actually
/// lie on the circle (e.g. the circle is not a constant-v iso-curve).
fn seam_anchor_on_circle(
    topo: &Topology,
    face: &Face,
    circle: &brepkit_math::curves::Circle3D,
) -> Option<Point3> {
    let surface = face.surface();
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut seam_pt = None;
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        if matches!(edge.curve(), EdgeCurve::Line) {
            let sp = topo.vertex(edge.start()).ok()?.point();
            let ep = topo.vertex(edge.end()).ok()?.point();
            if (sp - ep).length() > SEAM_DEGENERATE_TOL {
                seam_pt = Some(sp);
                break;
            }
        }
    }
    let (seam_u, _) = surface.project_point(seam_pt?)?;
    let (_, v_circle) = surface.project_point(circle.evaluate(0.0))?;
    let anchor = surface.evaluate(seam_u, v_circle)?;
    let radial = anchor - circle.center();
    let on_circle = (radial.length() - circle.radius()).abs() < SEAM_ON_CIRCLE_TOL
        && radial.dot(circle.normal()).abs() < SEAM_ON_CIRCLE_TOL;
    on_circle.then_some(anchor)
}

fn build_section_map(topo: &Topology, arena: &GfaArena) -> HashMap<FaceId, Vec<SectionSource>> {
    let mut map: HashMap<FaceId, Vec<SectionSource>> = HashMap::new();
    // Section edges from FF intersection curves.
    // For non-Line curves (Circle, Ellipse, NURBS): use one Curve entry per curve
    // so the face splitter gets the complete geometric loop instead of fragments.
    // For Line curves: use individual PaveBlocks (the old approach) since
    // line segments work correctly with the face splitter.
    for (idx, curve) in arena.curves.iter().enumerate() {
        if curve.pave_blocks.is_empty() {
            continue;
        }
        let is_line = matches!(&curve.curve, brepkit_topology::edge::EdgeCurve::Line);
        if is_line {
            // Feed individual PaveBlocks — works correctly for Lines.
            for &pb_id in &curve.pave_blocks {
                map.entry(curve.face_a)
                    .or_default()
                    .push(SectionSource::PaveBlock(pb_id, Some(curve.face_b)));
                map.entry(curve.face_b)
                    .or_default()
                    .push(SectionSource::PaveBlock(pb_id, Some(curve.face_a)));
            }
        } else {
            // Feed the complete curve — critical for closed curves (circles).
            map.entry(curve.face_a)
                .or_default()
                .push(SectionSource::Curve(idx));
            map.entry(curve.face_b)
                .or_default()
                .push(SectionSource::Curve(idx));
        }
    }
    // IN edges from EF interferences — individual PaveBlocks.
    // For faces that already have a curved (non-Line) FF curve, skip ALL
    // IN PBs — the complete FF curve already captures the same geometry.
    // This prevents 28 arc fragments from being added alongside the single
    // complete circle intersection curve.
    let faces_with_curved_ff: std::collections::HashSet<usize> = map
        .iter()
        .filter_map(|(fid, sources)| {
            sources
                .iter()
                .any(|s| matches!(s, SectionSource::Curve(_)))
                .then_some(fid.index())
        })
        .collect();
    for (&face_id, fi) in &arena.face_info {
        if faces_with_curved_ff.contains(&face_id.index()) {
            continue; // Face has curved FF curve — IN PBs are redundant.
        }
        // A plane cap disc (outer boundary is a single closed circle, e.g.
        // the cutting tool's own cap lying flush on a wall) needs no interior
        // splitting: its boundary already trims it. IN edges projected onto
        // such a cap from the wall's rectangle and other holes are spurious
        // arcs that fragment the clean disc into a many-sided polygon, which
        // then fails edge-set same-domain pairing with the wall's matching
        // cap disc and survives the cut as a stray face. Keep only IN edges
        // strictly inside the disc; drop those on or outside its boundary.
        let cap_disc = cap_disc_circle(topo, face_id);
        for &pb_id in &fi.pave_blocks_in {
            if let Some(circle) = &cap_disc
                && !pb_strictly_inside_circle(topo, arena, pb_id, circle)
            {
                continue;
            }
            map.entry(face_id)
                .or_default()
                .push(SectionSource::PaveBlock(pb_id, None));
        }
    }
    map
}

/// If `face_id` is a plane whose outer wire is a single closed circular edge
/// (a cap disc), return that circle; otherwise `None`.
fn cap_disc_circle(topo: &Topology, face_id: FaceId) -> Option<brepkit_math::curves::Circle3D> {
    let face = topo.face(face_id).ok()?;
    if !matches!(face.surface(), FaceSurface::Plane { .. }) {
        return None;
    }
    let wire = topo.wire(face.outer_wire()).ok()?;
    if wire.edges().len() != 1 {
        return None;
    }
    let edge = topo.edge(wire.edges()[0].edge()).ok()?;
    match edge.curve() {
        EdgeCurve::Circle(c) => Some(c.clone()),
        _ => None,
    }
}

/// Whether the pave block's edge midpoint lies strictly inside (not on) the
/// disc bounded by `circle` — its in-plane distance from the centre is below
/// the radius by more than tolerance.
fn pb_strictly_inside_circle(
    topo: &Topology,
    arena: &GfaArena,
    pb_id: crate::ds::PaveBlockId,
    circle: &brepkit_math::curves::Circle3D,
) -> bool {
    let Some(pb) = arena.pave_blocks.get(pb_id) else {
        return false;
    };
    let Ok(edge) = topo.edge(pb.original_edge) else {
        return false;
    };
    let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
        return false;
    };
    let (sp, ep) = (sv.point(), ev.point());
    let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
    let mid = edge
        .curve()
        .evaluate_with_endpoints(0.5 * (t0 + t1), sp, ep);
    let radial = mid - circle.center();
    let in_plane = radial - circle.normal() * radial.dot(circle.normal());
    in_plane.length() < circle.radius() - SEAM_ON_CIRCLE_TOL
}

/// Parametric span below which a chord-collapsed arc section is treated as
/// degenerate: ~2 orders above the observed remnant span (~5e-8) and ~6 below a
/// quarter-arc (π/2), so it rejects a coincident-fuse arc remnant while
/// preserving a genuine full circle (span ~2π).
const DEGENERATE_ARC_SPAN: f64 = 1e-6;

/// Convert section sources to `SectionEdge` entries.
///
/// For intersection curves, uses the complete curve geometry (not individual
/// PaveBlock fragments). For IN edges, uses the individual PaveBlock edge.
#[allow(clippy::too_many_lines)]
fn build_section_edges(
    topo: &Topology,
    arena: &GfaArena,
    face_id: FaceId,
    section_map: &HashMap<FaceId, Vec<SectionSource>>,
    seam_anchors: &BTreeMap<usize, Point3>,
    tol: f64,
) -> Vec<SectionEdge> {
    use brepkit_math::vec::Point3;

    let sources = match section_map.get(&face_id) {
        Some(s) => s,
        None => return Vec::new(),
    };

    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    // Pre-compute wire points for PCurve (needed for plane frame).
    let wire_pts: Vec<Point3> = topo
        .wire(face.outer_wire())
        .ok()
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

    let mut sections = Vec::new();

    for source in sources {
        match source {
            SectionSource::Curve(curve_idx) => {
                use brepkit_math::vec::Point2;
                // Use the COMPLETE intersection curve, not individual PBs.
                // The face splitter needs whole curves to form proper loops.
                let curve_ds = match arena.curves.get(*curve_idx) {
                    Some(c) => c,
                    None => continue,
                };

                // A closed section curve that coincides with one of this
                // face's own closed boundary edges does not split the face —
                // it lies entirely on the boundary. Feeding it to the face
                // splitter would corrupt the wire (the boundary circle gets
                // re-split against its own geometry).
                //
                // This only holds when the circle is genuinely redundant: it
                // bounds a region that lies fully on the partner FF face (a
                // flush coplanar cap sitting inside a wall). When the circle
                // is a real section larger than the partner — e.g. a cylinder
                // rim where another, narrower cylinder's cap plane slices the
                // lateral — keeping the lateral whole drops the split the
                // cut/fuse needs and yields invalid topology. Require the
                // partner face to host the full circle within its extent
                // before treating the coincidence as a pure self-boundary.
                let partner = if curve_ds.face_a == face_id {
                    curve_ds.face_b
                } else {
                    curve_ds.face_a
                };
                if closed_curve_coincides_with_boundary(topo, face_id, &curve_ds.curve, tol)
                    && circle_inside_face(topo, partner, &curve_ds.curve, tol)
                {
                    continue;
                }

                // Find start/end 3D points by evaluating the curve at its
                // parametric endpoints. For closed curves (circles), start ≈ end.
                let (start, end) = curve_endpoints(topo, arena, curve_ds);
                let (start, end) = match (start, end) {
                    (Some(s), Some(e)) => (s, e),
                    _ => continue,
                };

                // An OPEN section that re-traces PART of one of this face's
                // EXISTING boundary edges (outer or inner wire) adds no interior
                // split — that partition is already in the arrangement.
                // Threading it makes the planar arrangement weave a degenerate
                // region: a zero-area annulus over a hole ring (the 2×1/1×2
                // stacking-lip fuse failure, where the lip's bottom annulus sits
                // flush on the body's top and the FF intersection re-traces the
                // annulus's own opening ring), or a multiply-traversed snake
                // wire over the outer corner arcs (the tilted-divider lip fuse,
                // where the body's corner cylinders meet the lip's bottom plane
                // exactly at its flush outer rim). Sample the section by its own
                // parametric range (precise arc direction) and drop it when
                // every interior sample lies on a boundary edge — EXCEPT when it
                // duplicates a single existing edge exactly (see the exemption
                // in `section_on_existing_boundary`).
                if section_on_existing_boundary(
                    topo,
                    face_id,
                    &sample_curve_interior(curve_ds),
                    Some((start, end)),
                    tol,
                ) {
                    // Salvage the parts of a straight section that extend past
                    // the collinear boundary edge it rides — see the PaveBlock
                    // arm below for the full rationale (label-tab corner
                    // crescents).
                    if matches!(curve_ds.curve, EdgeCurve::Line) {
                        for (es, ee) in
                            line_section_boundary_extensions(topo, face_id, start, end, tol)
                        {
                            push_plain_line_section(&mut sections, face, es, ee);
                        }
                    }
                    continue;
                }

                // Skip a degenerate curved intersection curve: a non-Line curve
                // whose 3D chord has collapsed to a point AND whose parametric
                // span is near-zero (the `!Line` filter matches the PaveBlock
                // sibling below so a degenerate NurbsCurve section is covered too;
                // the chord test is the universal degeneracy signal and a genuine
                // section always has a non-tiny chord). The FF intersection of two coaxial corner
                // cylinders at a shared rim can emit such a remnant on one of two
                // mismatched-segmentation corner patches (gridfinity 3×3
                // stacking-lip fuse: one body eighth received a clean π/4 split
                // arc, its twin received that arc PLUS this zero-span remnant at
                // the lip-corner vertex). Threading it wove an out-and-back spur
                // and a degenerate self-loop into that face's wire, so the patch
                // never split into its lower/upper bands and the shell stayed
                // open. This mirrors the PaveBlock-branch guard below; the span
                // test (`curve_ds.t_range`) preserves a genuine full circle
                // (~2π span) and a genuine partial arc (~π/4 span).
                if !matches!(curve_ds.curve, EdgeCurve::Line)
                    && (end - start).length() < tol * 100.0
                {
                    let (ct0, ct1) = curve_ds.t_range;
                    if (ct1 - ct0).abs() < DEGENERATE_ARC_SPAN {
                        continue;
                    }
                }

                // Seam-anchored closed circles: re-parameterize so the
                // circle starts at the periodic face's seam point. Both
                // faces of the pair receive the same anchored geometry.
                let (curve_3d, start, end) = match seam_anchors.get(curve_idx) {
                    Some(&anchor) => {
                        let reanchored = if let EdgeCurve::Circle(c) = &curve_ds.curve {
                            brepkit_math::curves::Circle3D::new_with_ref(
                                c.center(),
                                c.normal(),
                                c.radius(),
                                anchor - c.center(),
                            )
                            .ok()
                            .map(EdgeCurve::Circle)
                        } else {
                            None
                        };
                        match reanchored {
                            Some(c) => (c, anchor, anchor),
                            None => (curve_ds.curve.clone(), start, end),
                        }
                    }
                    None => (curve_ds.curve.clone(), start, end),
                };

                let pcurve = super::pcurve_compute::compute_pcurve_on_surface(
                    &curve_3d,
                    start,
                    end,
                    face.surface(),
                    &wire_pts,
                    None,
                );

                // For closed curves on periodic surfaces (e.g. circle on cylinder),
                // the pcurve wraps around and evaluate(0) ≈ evaluate(1). We need
                // UV endpoints that span the full period so the face splitter
                // sees the section edge as a full-width cut.
                let is_closed = (start - end).length() < tol * 100.0;
                let (u_per, _v_per) = super::pcurve_compute::surface_periods(face.surface());
                let (start_uv_pt, end_uv_pt) = if is_closed && u_per.is_some() {
                    let period = u_per.unwrap_or(std::f64::consts::TAU);
                    // Project the start 3D point to UV.
                    let start_uv = face.surface().project_point(start);
                    if let Some((su, sv)) = start_uv {
                        // Sample the curve at t=0.25 to determine winding direction.
                        let (t0, t1) = curve_3d.domain_with_endpoints(start, end);
                        let mid_3d =
                            curve_3d.evaluate_with_endpoints(t0 + (t1 - t0) * 0.25, start, end);
                        let mid_uv = face.surface().project_point(mid_3d);
                        if let Some((mu, _mv)) = mid_uv {
                            // Determine winding: does the curve go in +u or -u
                            // direction from start?
                            let du = mu - su;
                            // Normalize du to [-period/2, period/2].
                            let du_norm = du - (du / period).round() * period;
                            let end_u = if du_norm < 0.0 {
                                su - period
                            } else {
                                su + period
                            };
                            (Some(Point2::new(su, sv)), Some(Point2::new(end_u, sv)))
                        } else {
                            let s = pcurve.evaluate(0.0);
                            (Some(s), Some(Point2::new(s.x() - period, s.y())))
                        }
                    } else {
                        // Plane surface — project via pcurve.
                        (Some(pcurve.evaluate(0.0)), Some(pcurve.evaluate(1.0)))
                    }
                } else if matches!(pcurve, brepkit_math::curves2d::Curve2D::Line(_)) {
                    // Line2D pcurves use arc-length parameterization, so
                    // `evaluate(1.0)` is one UV unit along the line, not the
                    // endpoint (e.g. a horizontal circle on a cylinder maps
                    // to a Line2D spanning the angular extent). Leave the
                    // endpoints unset; downstream consumers fall back to
                    // `uv_endpoints_from_pcurve`, which measures the true
                    // 2D length.
                    (None, None)
                } else {
                    (Some(pcurve.evaluate(0.0)), Some(pcurve.evaluate(1.0)))
                };

                sections.push(SectionEdge {
                    curve_3d,
                    pcurve_a: pcurve.clone(),
                    pcurve_b: pcurve,
                    start,
                    end,
                    start_uv_a: start_uv_pt,
                    end_uv_a: end_uv_pt,
                    start_uv_b: start_uv_pt,
                    end_uv_b: end_uv_pt,
                    target_face: None,
                    pave_block_id: None,
                });
            }
            SectionSource::PaveBlock(pb_id, opposing_face) => {
                // Individual PaveBlock edge — use the old Line2D pcurve approach.
                // This preserves the existing behavior for Line section edges
                // that the face splitter already handles correctly.
                use brepkit_math::curves2d::{Curve2D, Line2D};
                use brepkit_math::vec::{Point2, Vec2};

                let pb = match arena.pave_blocks.get(*pb_id) {
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

                let intervals: Vec<(Point3, Point3)> =
                    if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                        let Some(clipped_list) =
                            clip_line_to_face_boundary(topo, face_id, raw_start, raw_end, tol)
                        else {
                            continue;
                        };
                        // Also clip each piece to the OPPOSING FF face so a wide
                        // face does not extend the section past the narrower
                        // face's boundary. Adopt an opposing-clip endpoint ONLY
                        // where it genuinely tightens (moves by > tol): a
                        // flush/coincident opposing face re-derives the SAME
                        // endpoint with sub-tolerance float noise, and snapping
                        // to that noisy value shifts the section off the shared
                        // corner vertex this face's clip landed on exactly (the
                        // stacking-lip fuse regression). Interior split points
                        // from a multi-window opposing clip are genuine and kept
                        // as-is.
                        let mut finals = Vec::new();
                        for (cs, ce) in clipped_list {
                            match opposing_face
                                .and_then(|of| clip_line_to_face_boundary(topo, of, cs, ce, tol))
                            {
                                Some(subs) => {
                                    let n = subs.len();
                                    for (i, (ss, ee)) in subs.into_iter().enumerate() {
                                        let s = if i == 0 && (ss - cs).length() <= tol {
                                            cs
                                        } else {
                                            ss
                                        };
                                        let e = if i + 1 == n && (ee - ce).length() <= tol {
                                            ce
                                        } else {
                                            ee
                                        };
                                        finals.push((s, e));
                                    }
                                }
                                None => finals.push((cs, ce)),
                            }
                        }
                        finals
                    } else {
                        vec![(raw_start, raw_end)]
                    };
                let mut work: Vec<(Point3, Point3)> = intervals;
                let mut work_i = 0;
                while work_i < work.len() {
                    let (start, end) = work[work_i];
                    work_i += 1;
                    // Skip a degenerate curved section: a Circle/Ellipse PaveBlock
                    // fragment whose arc has collapsed to a point (3D chord below
                    // tolerance AND a near-zero parametric span). A coincident-edge
                    // fuse can split a rounded corner's arc so that one fragment
                    // carries the whole quarter-arc and its twin is a ~1e-7-long
                    // remnant; the remnant has no boundary to contribute, but
                    // threading it into the face wire makes the builder weave an
                    // out-and-back spur (the over-shared depth-wall edge of the
                    // 2×1/1×2 stacking-lip fuse). The Line path already rejects
                    // zero-length sections inside `clip_line_to_face_boundary`; this
                    // guards the curved path, which uses the raw endpoints unclipped.
                    // The span test preserves a genuine full circle (coincident
                    // endpoints but a ~2π span); the chord test uses the
                    // weld-scale band (100·tol) because the remnant's endpoints are
                    // the same near-coincident vertices the assembler later welds.
                    if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line)
                        && (end - start).length() < tol * 100.0
                    {
                        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
                        if (t1 - t0).abs() < DEGENERATE_ARC_SPAN {
                            continue;
                        }
                    }

                    // Same boundary-re-trace rejection as the Curve arm above: a
                    // PaveBlock section whose interior lies entirely on one of this
                    // face's existing boundary edges contributes no interior split.
                    // Keeping such sections made the arrangement of the lip bottom
                    // ring weave a degenerate region wherever the body's inner-wall
                    // planes intersect the lip plane exactly along its flush inner
                    // rim, and the whole fuse fell back to the mesh boolean (the
                    // top-row-merged compartment bin).
                    let interior_samples: Vec<Point3> = {
                        const SAMPLES: usize = 9;
                        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
                        (1..SAMPLES)
                            .map(|k| {
                                #[allow(clippy::cast_precision_loss)]
                                let frac = k as f64 / SAMPLES as f64;
                                edge.curve().evaluate_with_endpoints(
                                    (t1 - t0).mul_add(frac, t0),
                                    start,
                                    end,
                                )
                            })
                            .collect()
                    };
                    if section_on_existing_boundary(
                        topo,
                        face_id,
                        &interior_samples,
                        Some((start, end)),
                        tol,
                    ) {
                        // A straight section can ride a collinear boundary edge
                        // for most of its span yet extend past it at one or both
                        // ends (the label-tab corner: the tab's back-plane chord
                        // rides the cavity's back line but juts 2.55mm into each
                        // rounded-corner crescent). The interior samples all land
                        // on the covered middle, so the whole section reads as a
                        // re-trace and the crescent is never split off. Salvage
                        // the uncovered extensions — they carry the genuine
                        // interior split. Exact interval math, not sampling: the
                        // extension fraction can be arbitrarily small.
                        if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                            for ext in
                                line_section_boundary_extensions(topo, face_id, start, end, tol)
                            {
                                work.push(ext);
                            }
                        }
                        continue;
                    }

                    // Project start/end to UV using surface projection (original approach).
                    let start_uv = face.surface().project_point(start);
                    let end_uv = face.surface().project_point(end);
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
            }
        }
    }

    // Deduplicate: remove section edges that are subsets of longer
    // collinear edges. This happens when both the FF phase and the
    // coplanar phase create section edges on the same line — the FF
    // edge spans the full face, the coplanar edge spans the inner
    // region only. Keeping both creates degenerate face splits.
    dedup_collinear_sections(&mut sections, tol);

    sections
}

/// Whether the closed `curve` (a circle/ellipse) is contained within
/// `face`'s 3D extent (its outer-wire bounding box, expanded by tolerance).
///
/// Distinguishes a flush coplanar cap whose rim sits inside a wall (the
/// circle fits the wall → redundant, skip-safe) from a rim that extends
/// beyond a smaller partner face (e.g. a cylinder rim slicing a narrower
/// cylinder's cap → a real section that must split the partner, never skip).
fn circle_inside_face(
    topo: &Topology,
    face_id: FaceId,
    curve: &brepkit_topology::edge::EdgeCurve,
    tol: f64,
) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return false;
    };
    let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    let mut have = false;
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            return false;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            return false;
        };
        let (sp, ep) = (sv.point(), ev.point());
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        for k in 0..=8 {
            #[allow(clippy::cast_precision_loss)]
            let frac = k as f64 / 8.0;
            let p = edge
                .curve()
                .evaluate_with_endpoints((t1 - t0).mul_add(frac, t0), sp, ep);
            min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
            max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
            have = true;
        }
    }
    if !have {
        return false;
    }

    // Closed Circle/Ellipse: evaluate_with_endpoints ignores the reference
    // points and dispatches to the parametric domain directly.
    let origin = Point3::new(0.0, 0.0, 0.0);
    let (t0, t1) = curve.domain_with_endpoints(origin, origin);
    for k in 0..16 {
        #[allow(clippy::cast_precision_loss)]
        let frac = k as f64 / 16.0;
        let p = curve.evaluate_with_endpoints((t1 - t0).mul_add(frac, t0), origin, origin);
        if p.x() < min.x() - tol
            || p.x() > max.x() + tol
            || p.y() < min.y() - tol
            || p.y() > max.y() + tol
            || p.z() < min.z() - tol
            || p.z() > max.z() + tol
        {
            return false;
        }
    }
    true
}

/// Check whether a closed section curve (Circle/Ellipse) coincides with
/// one of the face's own closed boundary edges.
fn closed_curve_coincides_with_boundary(
    topo: &Topology,
    face_id: FaceId,
    curve: &brepkit_topology::edge::EdgeCurve,
    tol: f64,
) -> bool {
    use brepkit_topology::edge::EdgeCurve;

    let circles_match = |a: &brepkit_math::curves::Circle3D, b: &brepkit_math::curves::Circle3D| {
        (a.center() - b.center()).length() < tol
            && (a.radius() - b.radius()).abs() < tol
            && a.normal().dot(b.normal()).abs() > 1.0 - 1e-9
    };
    let ellipses_match = |a: &brepkit_math::curves::Ellipse3D,
                          b: &brepkit_math::curves::Ellipse3D| {
        let geometry_matches = (a.center() - b.center()).length() < tol
            && (a.semi_major() - b.semi_major()).abs() < tol
            && (a.semi_minor() - b.semi_minor()).abs() < tol
            && a.normal().dot(b.normal()).abs() > 1.0 - 1e-9;
        if !geometry_matches {
            return false;
        }
        // When the semi-axes are equal the ellipse is a circle and the
        // major-axis direction is undefined, so only its length matters.
        if (a.semi_major() - a.semi_minor()).abs() < tol {
            return true;
        }
        // The major axis and its negation describe the same ellipse, so
        // compare directions up to sign.
        a.u_axis().dot(b.u_axis()).abs() > 1.0 - 1e-9
    };

    if !matches!(curve, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
        return false;
    }

    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let wires: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    for wid in wires {
        let Ok(wire) = topo.wire(wid) else {
            continue;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            if edge.start() != edge.end() {
                continue;
            }
            let coincides = match (curve, edge.curve()) {
                (EdgeCurve::Circle(a), EdgeCurve::Circle(b)) => circles_match(a, b),
                (EdgeCurve::Ellipse(a), EdgeCurve::Ellipse(b)) => ellipses_match(a, b),
                _ => false,
            };
            if coincides {
                return true;
            }
        }
    }
    false
}

/// Whether a section sampled as `section_pts` (3D points along its interior)
/// lies entirely on one of `face`'s own boundary edges — i.e. it is a redundant
/// self-hole intersection that does not partition the face interior.
///
/// A section that rides one of the face's existing INNER-wire (hole) edges is
/// redundant: the hole boundary is already present, so re-tracing it adds no new
/// region. Threading it makes the planar arrangement weave a zero-area annulus
/// (a sub-face whose outer wire equals its inner wire), which over-shares every
/// ring edge and inverts the assembled shell — the 2×1/1×2 stacking-lip fuse
/// failure, where the lip's bottom annulus sits flush on the body's top and the
/// FF intersection re-traces the annulus's own opening ring.
///
/// Both wire kinds are tested. An inner-wire (hole) re-trace weaves a zero-area
/// annulus (the case above). An OUTER-wire re-trace is just as destructive: the
/// body's corner cylinders meet the lip's bottom plane exactly at its flush
/// outer rim, so the FF intersection arcs re-trace the lip ring's own outer
/// corner arcs; threading them makes the wire builder weave a snake wire that
/// multiply-traverses the rim, which saturates the shell flood-fill's edge
/// counts and orphans every face across the junction (the tilted-divider lip
/// fuse collapsed to the 14-face body exterior). Sections that only PARTLY ride
/// a boundary edge keep their pave — the all-interior-samples rule keeps any
/// section that enters face material, so a genuinely needed seam section (one
/// that crosses INTO the face) is never dropped.
fn section_on_existing_boundary(
    topo: &Topology,
    face_id: FaceId,
    section_pts: &[Point3],
    sec_endpoints: Option<(Point3, Point3)>,
    tol: f64,
) -> bool {
    if section_pts.is_empty() {
        return false;
    }
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    'sample: for p in section_pts {
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let Ok(wire) = topo.wire(wid) else { continue };
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                    continue;
                };
                let (bs, be) = (sv.point(), ev.point());
                if point_on_edge(edge.curve(), bs, be, *p, tol) {
                    continue 'sample;
                }
            }
        }
        // This sample is off every boundary edge → the section is not a pure
        // boundary re-trace → keep it (it may split the interior or seam a
        // neighbour).
        return false;
    }

    // Exemption: an OPEN section that duplicates a single existing OUTER-wire
    // edge EXACTLY (endpoint pair matches within the weld band) is safe to
    // thread — the wire builder welds identical duplicates — and load-bearing:
    // threading it is what routes the face through the split/rebuild path,
    // aligning its boundary partition with the other solid's coincident faces
    // (the shelled cup's rim/lip-base corner arcs; dropping them left the
    // lip-bottom ring unsplit and its inner rim unpaired against the cup
    // walls). Only SUB-SPAN re-traces (a 45°-split half of a whole corner-arc
    // edge, a straight run split at a divider crossing) corrupt the
    // arrangement, because the coincident pieces partition the same curve
    // differently.
    //
    // The exemption deliberately excludes INNER (hole) wires and
    // closed/degenerate sections (start ≈ end): a hole ring is a complete
    // closed boundary already, so re-threading ANY of its edges — whole or
    // partial — recreates the zero-area annulus this guard exists for (the
    // 2×1/1×2 stacking-lip fuse).
    if let Some((ss, se)) = sec_endpoints {
        let weld = tol * 100.0;
        // Endpoint matching uses a wider band than the closed-section test: a
        // grazing/tangential intersection can mislocate a section endpoint by
        // microns ALONG the boundary curve (positional error ~ sqrt of the
        // solver residual — measured 3.1 µm on the lip-corner arc) while the
        // section still rides the edge within `tol`. Genuine sub-span split
        // points (a 45° arc midpoint, a divider crossing) sit millimetres from
        // the edge endpoints, so 10 µm keeps a 100× separation margin.
        let endpoint_band = (tol * 1e5).max(weld);
        if (ss - se).length() >= weld
            && let Ok(wire) = topo.wire(face.outer_wire())
        {
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                    continue;
                };
                let (bs, be) = (sv.point(), ev.point());
                let fwd = (ss - bs).length() < endpoint_band && (se - be).length() < endpoint_band;
                let rev = (ss - be).length() < endpoint_band && (se - bs).length() < endpoint_band;
                if fwd || rev {
                    return false;
                }
            }
        }
    }

    true
}

/// Whether `p` lies on the edge's true geometry within `tol` (perpendicular
/// distance for a Line; in-plane + radial distance with an arc-span containment
/// test for a Circle/Ellipse). NURBS edges are not boundary candidates here.
fn point_on_edge(
    curve: &brepkit_topology::edge::EdgeCurve,
    start: Point3,
    end: Point3,
    p: Point3,
    tol: f64,
) -> bool {
    use brepkit_topology::edge::EdgeCurve;
    match curve {
        EdgeCurve::Line => point_to_segment_dist_3d(p, start, end) < tol,
        EdgeCurve::Circle(c) => {
            // True 3D distance from p to the circle is the hypotenuse of the
            // off-plane and radial errors; checking the two independently
            // (each < tol) would admit a point up to √2·tol off the curve and
            // could drop a non-redundant section.
            let radial = p - c.center();
            let off_plane = radial.dot(c.normal());
            let in_plane = radial - c.normal() * off_plane;
            let radial_err = in_plane.length() - c.radius();
            if off_plane.hypot(radial_err) > tol {
                return false;
            }
            // p is on the full circle; confirm its angle lies within the arc
            // span [t0, t1] derived from the edge's endpoints. `c.project`
            // returns the angle in the circle's frame; compare via the same
            // CCW-delta convention `domain_with_endpoints` uses.
            arc_param_contains(c.project(p), c.project(start), c.project(end), start, end)
        }
        EdgeCurve::Ellipse(e) => {
            let radial = p - e.center();
            let off_plane = radial.dot(e.normal());
            if off_plane.abs() > tol {
                return false;
            }
            // On-ellipse test: the evaluated point at p's projected angle must
            // match p (the ellipse radius varies with angle, so a fixed-radius
            // test does not apply).
            let ang = e.project(p);
            if (e.evaluate(ang) - p).length() > tol {
                return false;
            }
            arc_param_contains(ang, e.project(start), e.project(end), start, end)
        }
        EdgeCurve::NurbsCurve(_) => false,
    }
}

/// Whether angle `a` lies within the CCW arc span from `a0` to `a1`, matching
/// the convention `EdgeCurve::domain_with_endpoints` uses (a closed full circle
/// when the endpoints coincide). Angles are in radians in the curve's frame.
fn arc_param_contains(a: f64, a0: f64, a1: f64, start: Point3, end: Point3) -> bool {
    use std::f64::consts::TAU;
    if (start - end).length() < 1e-9 {
        return true; // full closed circle/ellipse
    }
    let span = (a1 - a0).rem_euclid(TAU);
    let span = if span < 1e-12 { TAU } else { span };
    let rel = (a - a0).rem_euclid(TAU);
    // `rel` is the CCW offset from the start, so the start (rel == 0) is
    // inherently inclusive; the `+ 1e-6` keeps the end inclusive against
    // round-off (a point just past the start wraps to rel ≈ TAU and stays out).
    rel <= span + 1e-6
}

/// Sample interior 3D points along an intersection curve using its OWN
/// parametric range (`curve_ds.t_range`). The endpoints are excluded — they
/// always coincide with the face boundary where the section meets it; only the
/// interior reveals whether the section enters the face material or rides a
/// boundary edge. Used by [`section_on_existing_boundary`].
fn sample_curve_interior(curve_ds: &crate::ds::IntersectionCurveDS) -> Vec<Point3> {
    use brepkit_math::vec::Point3;
    const SAMPLES: usize = 9;
    let (t0, t1) = curve_ds.t_range;
    let dummy = Point3::new(0.0, 0.0, 0.0);
    let mut pts = Vec::with_capacity(SAMPLES - 1);
    for k in 1..SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let frac = k as f64 / SAMPLES as f64;
        pts.push(
            curve_ds
                .curve
                .evaluate_with_endpoints((t1 - t0).mul_add(frac, t0), dummy, dummy),
        );
    }
    pts
}

/// Find the overall 3D start/end points of an intersection curve
/// by evaluating at the curve's parametric endpoints.
///
/// This function is only called for non-Line curves (Circle, Ellipse, NURBS).
/// For these types, `evaluate_with_endpoints` ignores the start/end reference
/// points entirely — they dispatch to `ParametricCurve::evaluate(t)`.
fn curve_endpoints(
    topo: &Topology,
    arena: &GfaArena,
    curve_ds: &crate::ds::IntersectionCurveDS,
) -> (Option<Point3>, Option<Point3>) {
    use brepkit_math::vec::Point3;

    let (t0, t1) = curve_ds.t_range;
    // Non-Line curves evaluate directly at their parametric endpoints.
    // The dummy reference points are unused by Circle/Ellipse/NURBS evaluation.
    let dummy = Point3::new(0.0, 0.0, 0.0);
    let start_3d = curve_ds.curve.evaluate_with_endpoints(t0, dummy, dummy);
    let end_3d = curve_ds.curve.evaluate_with_endpoints(t1, dummy, dummy);
    // A marched/fitted curve evaluates ~1e-6 off the exact junction its mint
    // snapped the pave vertices to (boundary-foot anchors); downstream boundary
    // splitters gate on the exact 1e-7 tolerance, so hand back the VERTEX
    // positions when they are the same endpoints (within the weld band) —
    // never a repositioning, only the exact-snap variant of the same point.
    if curve_ds.pave_blocks.len() == 1
        && !matches!(curve_ds.curve, brepkit_topology::edge::EdgeCurve::Line)
        && let Some(pb) = arena.pave_blocks.get(curve_ds.pave_blocks[0])
        && let (Ok(sv), Ok(ev)) = (topo.vertex(pb.start.vertex), topo.vertex(pb.end.vertex))
    {
        let weld = 1e-5;
        let (svp, evp) = (sv.point(), ev.point());
        if (svp - start_3d).length() <= weld && (evp - end_3d).length() <= weld {
            return (Some(svp), Some(evp));
        }
    }
    (Some(start_3d), Some(end_3d))
}

/// Sub-intervals of the straight section `start..end` NOT covered by any
/// collinear boundary Line edge of `face_id` (outer or inner wires).
///
/// A section flagged as a boundary re-trace can still extend past the edge it
/// rides at one or both ends; those extensions are genuine interior splitters
/// (the label-tab back-plane chord riding the cavity back line but jutting
/// into each rounded-corner crescent). Coverage is computed with exact
/// interval arithmetic so an arbitrarily small extension survives. Extensions
/// shorter than the weld band (100·tol) are noise and skipped.
fn line_section_boundary_extensions(
    topo: &Topology,
    face_id: FaceId,
    start: Point3,
    end: Point3,
    tol: f64,
) -> Vec<(Point3, Point3)> {
    let weld = tol * 100.0;
    let dir = end - start;
    let len = dir.length();
    if len <= weld {
        return Vec::new();
    }
    let u = dir * (1.0 / len);
    let Ok(face) = topo.face(face_id) else {
        return Vec::new();
    };
    let mut cover: Vec<(f64, f64)> = Vec::new();
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let Ok(wire) = topo.wire(wid) else { continue };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                continue;
            }
            let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                continue;
            };
            let (bs, be) = (sv.point(), ev.point());
            let off0 = ((bs - start) - u * (bs - start).dot(u)).length();
            let off1 = ((be - start) - u * (be - start).dot(u)).length();
            if off0 > weld || off1 > weld {
                continue;
            }
            let t0 = (bs - start).dot(u);
            let t1 = (be - start).dot(u);
            let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(len));
            if hi - lo > tol {
                cover.push((lo, hi));
            }
        }
    }
    if cover.is_empty() {
        return Vec::new();
    }
    cover.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut exts = Vec::new();
    let mut cur = 0.0_f64;
    for &(lo, hi) in &cover {
        if lo > cur + weld {
            exts.push((cur, lo));
        }
        cur = cur.max(hi);
    }
    if len > cur + weld {
        exts.push((cur, len));
    }
    exts.into_iter()
        .map(|(a, b)| (start + u * a, start + u * b))
        .collect()
}

/// Push a plain straight section with a Line2D pcurve (the PaveBlock-arm
/// construction) — used for salvaged boundary-extension segments.
fn push_plain_line_section(
    sections: &mut Vec<SectionEdge>,
    face: &brepkit_topology::face::Face,
    start: Point3,
    end: Point3,
) {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2};

    let start_uv = face.surface().project_point(start);
    let end_uv = face.surface().project_point(end);
    let s2 = start_uv.map_or(Point2::new(0.0, 0.0), |(u, v)| Point2::new(u, v));
    let e2 = end_uv.map_or(Point2::new(1.0, 0.0), |(u, v)| Point2::new(u, v));
    let d = e2 - s2;
    let dl = d.length();
    let direction = if dl > 1e-12 {
        Vec2::new(d.x() / dl, d.y() / dl)
    } else {
        Vec2::new(1.0, 0.0)
    };
    let Ok(line) = Line2D::new(s2, direction).or_else(|_| Line2D::new(s2, Vec2::new(1.0, 0.0)))
    else {
        return;
    };
    let pcurve = Curve2D::Line(line);
    sections.push(SectionEdge {
        curve_3d: brepkit_topology::edge::EdgeCurve::Line,
        pcurve_a: pcurve.clone(),
        pcurve_b: pcurve,
        start,
        end,
        start_uv_a: start_uv.map(|(u, v)| Point2::new(u, v)),
        end_uv_a: end_uv.map(|(u, v)| Point2::new(u, v)),
        start_uv_b: start_uv.map(|(u, v)| Point2::new(u, v)),
        end_uv_b: end_uv.map(|(u, v)| Point2::new(u, v)),
        target_face: None,
        pave_block_id: None,
    });
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

            // Collinear and on the same infinite line, but possibly DISJOINT or
            // merely ADJACENT:
            //  - DISJOINT: two notches on opposite walls cut the same rim along
            //    the same x = ±cut_hw line, yet their cut segments sit at
            //    opposite ends of the face and must both survive.
            //  - ADJACENT: two PaveBlock fragments of one wall meeting end-to-end
            //    at a shared junction vertex (e.g. a wall split into two faces by
            //    a wall cutout). The junction carries sub-tolerance float noise,
            //    so the two intervals overlap by a tiny sliver (~1e-3, well above
            //    the 1e-7 linear tol) — but neither is a subset of the other and
            //    dropping the shorter one deletes a genuine boundary piece (the
            //    honeycomb-cut cap corner cell then never closes → open shell).
            // Only treat the shorter section as a redundant subset when it is
            // (nearly) FULLY CONTAINED in the longer one (a coplanar inner-region
            // edge nested in the full-face FF edge): the overlap must cover
            // essentially the whole shorter segment, not just a junction sliver.
            // Project both segments onto the shared line and compare intervals.
            let proj = |p: Point3, origin: Point3| (p - origin).dot(dj_unit);
            let (ia0, ia1) = (proj(si.start, sj.start), proj(si.end, sj.start));
            let (ib0, ib1) = (proj(sj.start, sj.start), proj(sj.end, sj.start));
            let (ia_lo, ia_hi) = (ia0.min(ia1), ia0.max(ia1));
            let (ib_lo, ib_hi) = (ib0.min(ib1), ib0.max(ib1));
            let overlap = ia_hi.min(ib_hi) - ia_lo.max(ib_lo);
            if overlap <= tol {
                continue;
            }

            let len_i = (si.end - si.start).length();
            let len_j = (sj.end - sj.start).length();
            let shorter_len = len_i.min(len_j);
            // Containment guard: the shorter segment is a redundant subset only
            // when the overlap spans (almost) its entire length. An adjacent pair
            // sharing a noisy junction overlaps by far less than the shorter
            // length, so it falls through and both survive.
            if overlap < shorter_len - tol * 10.0 {
                continue;
            }

            // Overlapping collinear sections, shorter fully contained — remove it.
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

/// Intersect a section line with a curved boundary edge's TRUE geometry,
/// keeping only crossings that fall on the actual arc span (between the edge's
/// start/end vertices, on the side through its midpoint) — not the full circle.
///
/// Returns the 3D crossing points (with the edge's angle, unused by callers).
fn arc_segment_crossings(
    curve: &EdgeCurve,
    edge_start: Point3,
    edge_end: Point3,
    closed: bool,
    line_start: Point3,
    line_end: Point3,
    tol: f64,
) -> Vec<(Point3, f64)> {
    let circle = match curve {
        EdgeCurve::Circle(c) => c,
        // Only circular arcs are handled here. Ellipse arcs are not produced on
        // the corner-straddle path, and lines/NURBS sections have no true-arc
        // geometry — all fall back to the chord (handled by the line-line
        // crossing in the caller).
        EdgeCurve::Ellipse(_) | EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return Vec::new(),
    };
    let hits = circle.intersect_segment(line_start, line_end, tol);
    if hits.is_empty() {
        return hits;
    }
    // Angular interval of the arc edge: from start angle to end angle on the
    // side that passes through the edge's geometric midpoint.
    let a_start = circle.project(edge_start);
    let a_end = circle.project(edge_end);
    let mid = super::pcurve_compute::evaluate_edge_at_t(curve, edge_start, edge_end, 0.5);
    let a_mid = circle.project(mid);
    // Normalize so the test is "is `a` between a_start and a_end the short/long
    // way that contains a_mid". Use unsigned angular distances on the circle.
    let ang_dist = |x: f64, y: f64| -> f64 {
        let d = (x - y).abs() % std::f64::consts::TAU;
        d.min(std::f64::consts::TAU - d)
    };
    let span = ang_dist(a_start, a_end);
    // A CLOSED rim edge (vertex identity, per the caller) covers the whole
    // circle: every hit is on the arc. Without this, the major-arc
    // complement filter below excludes a hit that lands exactly at the seam
    // angle (dsa + dae = 0), losing one of a cap disc's two chord crossings
    // and with it the cap's half-disc split.
    if closed {
        return hits;
    }
    // `a` is on the arc iff dist(start,a)+dist(a,end) ≈ the arc span that
    // contains the midpoint. Validate the midpoint satisfies this first so a
    // degenerate (near-full) circle doesn't admit everything.
    let on_arc = |a: f64| -> bool {
        let dsa = ang_dist(a_start, a);
        let dae = ang_dist(a, a_end);
        (dsa + dae - span).abs() < 1e-6 || (dsa + dae) <= span + 1e-6
    };
    if !on_arc(a_mid) {
        // Edge is the COMPLEMENT (major) arc; flip the test.
        let major = |a: f64| -> bool {
            let dsa = ang_dist(a_start, a);
            let dae = ang_dist(a, a_end);
            (dsa + dae) > span + 1e-9
        };
        return hits.into_iter().filter(|(_, t)| major(*t)).collect();
    }
    hits.into_iter().filter(|(_, t)| on_arc(*t)).collect()
}

/// Clip a 3D line segment to a face's boundary polygon.
///
/// Collects the outer wire vertices as line segments, then finds where
/// the section line enters and exits the polygon. Returns the trimmed
/// Every in-face `(start, end)` interval of the line (a section can cross a
/// face in multiple material windows), or `None` when nothing lies inside.
#[allow(clippy::too_many_lines)]
fn clip_line_to_face_boundary(
    topo: &Topology,
    face_id: FaceId,
    line_start: Point3,
    line_end: Point3,
    tol: f64,
) -> Option<Vec<(Point3, Point3)>> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;

    // Collect boundary edges as line segments (vertex positions in traversal
    // order); for curved edges keep the geometry so the section line can be
    // clipped to the TRUE arc rather than its chord.
    let edges = wire.edges();
    let mut boundary_segments: Vec<(Point3, Point3)> = Vec::with_capacity(edges.len());
    // (curve, oriented start/end points, closed-by-vertex-identity)
    let mut boundary_arcs: Vec<Option<(EdgeCurve, Point3, Point3, bool)>> =
        Vec::with_capacity(edges.len());
    for oe in edges {
        let edge = topo.edge(oe.edge()).ok()?;
        let sp = topo.vertex(oe.oriented_start(edge)).ok()?.point();
        let ep = topo.vertex(oe.oriented_end(edge)).ok()?.point();
        boundary_segments.push((sp, ep));
        match edge.curve() {
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                let closed = oe.oriented_start(edge) == oe.oriented_end(edge);
                boundary_arcs.push(Some((edge.curve().clone(), sp, ep, closed)));
            }
            // A straight edge already equals its chord, and a NURBS boundary edge
            // has no analytic arc to clip against, so neither contributes a
            // beyond-the-chord crossing; the chord segment in
            // `boundary_segments` covers them.
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => boundary_arcs.push(None),
        }
    }

    let line_dir = line_end - line_start;
    let line_len = line_dir.length();
    if line_len < tol {
        return None;
    }

    // Find all intersection parameters (t) of the section line with boundary segments.
    // The section line is: P(t) = line_start + t * line_dir, t in [0, 1].
    let mut crossings: Vec<f64> = Vec::new();
    let mut crossings_ext: Vec<f64> = Vec::new();

    for (seg_idx, (seg_start, seg_end)) in boundary_segments.iter().enumerate() {
        // For a curved boundary edge, also record the crossing with the TRUE
        // arc geometry. A convex rounded corner bulges OUTWARD past its chord,
        // so the arc crossing extends the section to where it actually exits
        // the face (e.g. a notch corner that straddles a wall's top edge clips
        // to x=±13.236, not the chord's x=±12). The crossings are merged below;
        // the outermost pair is taken, so adding the arc crossing never drops a
        // chord crossing the existing cases rely on — it only reaches farther
        // out when the arc genuinely does. Arcs the line misses contribute
        // nothing (the lip-cut sections that graze a corner chord keep working).
        if let Some((curve, asp, aep, closed)) = &boundary_arcs[seg_idx] {
            // Intersect the arc with the EXTENDED line, not just the segment: a
            // section whose FF-clipped endpoint lies in the sliver between a
            // convex arc and its chord (the slot walls crossing a socket-edge
            // half-disc) has its true arc crossing slightly OUTSIDE the
            // segment; rejecting it leaves only the chord crossings, clipping
            // the section to the chord triangle 0.65 short of the corner. The
            // outermost-pair clamp to [0, 1] below bounds the result to the
            // original segment, so extension can never grow the section.
            let ext_start = line_start - line_dir;
            let ext_end = line_end + line_dir;
            let arc_hits =
                arc_segment_crossings(curve, *asp, *aep, *closed, ext_start, ext_end, tol);
            for (p, _) in arc_hits {
                let t = (p - line_start).dot(line_dir) / (line_len * line_len);
                crossings_ext.push(t);
                // The historical outermost path only ever saw crossings ON the
                // segment; keep its input unchanged.
                if (-1e-9..=1.0 + 1e-9).contains(&t) {
                    crossings.push(t);
                }
            }
        }
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

    // A single boundary crossing is legitimate when a section ENDPOINT lies
    // INSIDE the face: the material window runs from that crossing to the
    // interior endpoint, not between two crossings. A box corner biting a
    // cylinder cap disc is the canonical case — each of the two chords runs
    // from the interior corner vertex out to the rim, crossing the boundary
    // circle exactly once. The midpoint-classification path below admits the
    // section endpoints (t=0/t=1) as interval borders and keeps only the
    // truly-inside window, so one crossing suffices there. The non-plane /
    // holed fallback still selects an outermost crossing PAIR, so it keeps
    // requiring two.
    let single_crossing_ok = crossings.len() == 1
        && face.inner_wires().is_empty()
        && matches!(face.surface(), FaceSurface::Plane { .. });
    if crossings.len() < 2 && !single_crossing_ok {
        return None;
    }

    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Select the in-face interval by MIDPOINT CLASSIFICATION rather than the
    // outermost crossing pair. Outermost-pair is only right when every arc
    // bulges OUTWARD (a convex rounded corner extending the face past its
    // chord). A boundary arc that bulges INWARD — a socket-bite circle carving
    // the face, as on the plate-edge junction web between two socket corners —
    // makes the chord crossings OVERSHOOT the true material: the delivered
    // section then extends into the bite (air on this face), the splitter
    // weaves a wrong region, and the cut falls to a mesh fallback (the
    // snap-slot hole cuts). Classify each candidate sub-interval's midpoint
    // against the arc-true boundary polygon and keep EVERY interval that is
    // actually inside the face (a section can cross the face in multiple
    // material windows).
    let plane_frame = match face.surface() {
        FaceSurface::Plane { normal, .. } => {
            let pts: Vec<Point3> = boundary_segments.iter().map(|&(s, _)| s).collect();
            Some(crate::builder::plane_frame::PlaneFrame::from_plane_face(
                *normal, &pts,
            ))
        }
        _ => None,
    };
    let poly: Option<Vec<brepkit_math::vec::Point2>> = plane_frame.as_ref().map(|frame| {
        let mut poly = Vec::new();
        for (seg_idx, (sp, ep)) in boundary_segments.iter().enumerate() {
            poly.push(frame.project(*sp));
            if let Some((curve, asp, aep, closed)) = &boundary_arcs[seg_idx] {
                // Dense sampling: at 12 samples an r=4 quarter-arc's chord
                // sagitta is ~0.14 — larger than the ~0.1 mm groove-mouth
                // slivers this polygon must classify. 96 samples keep the
                // polygon error well under the features decided by it.
                //
                // A CLOSED rim edge (a disc rim: the same seam vertex at both
                // ends) has a degenerate endpoint span — `evaluate_edge_at_t`
                // would collapse every sample to the seam and the polygon to
                // a point, so everything but the seam neighbourhood
                // classified outside (the lite pad top-disc's wall chords).
                // Sample its full period from the seam angle instead.
                // Closedness is vertex IDENTITY, so a genuine open arc with
                // near-coincident endpoints keeps the span sampling.
                if *closed && let EdgeCurve::Circle(c) = curve {
                    let a_seam = c.project(*asp);
                    for k in 1..96 {
                        let a = std::f64::consts::TAU.mul_add(f64::from(k) / 96.0, a_seam);
                        poly.push(frame.project(c.evaluate(a)));
                    }
                } else if *closed && let EdgeCurve::Ellipse(el) = curve {
                    let a_seam = el.project(*asp);
                    for k in 1..96 {
                        let a = std::f64::consts::TAU.mul_add(f64::from(k) / 96.0, a_seam);
                        poly.push(frame.project(el.evaluate(a)));
                    }
                } else {
                    for k in 1..96 {
                        let f = f64::from(k) / 96.0;
                        let p3 = super::pcurve_compute::evaluate_edge_at_t(curve, *asp, *aep, f);
                        poly.push(frame.project(p3));
                    }
                }
            }
            let _ = ep;
        }
        poly
    });

    // Midpoint classification only steers hole-free faces: a holed face's
    // sections are the hole weave's input, and that machinery (promotion,
    // coincidence splitting, re-trace discriminants) is calibrated on WHOLE
    // sections — pre-splitting them here breaks its bookkeeping (the groove
    // chain regressed). Hole-free faces have no weave; their concave bites
    // live on the OUTER wire where the outermost-pair heuristic overshoots.
    let t_intervals: Vec<(f64, f64)> = if face.inner_wires().is_empty()
        && let (Some(frame), Some(poly)) = (plane_frame.as_ref(), poly.as_ref())
    {
        // Every in-face sub-interval, not just one: a section can cross the
        // face in MULTIPLE material windows (the slot's z=-1.2 closer runs
        // across the socket bite, leaving material on both sides), and every
        // window is a real section piece the splitter chain needs.
        let mut borders: Vec<f64> = crossings
            .iter()
            .chain(crossings_ext.iter())
            .map(|t| t.clamp(0.0, 1.0))
            .chain([0.0, 1.0])
            .collect();
        borders.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        borders.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        borders
            .windows(2)
            .filter_map(|w| {
                let mid_uv = frame.project(line_start + line_dir * f64::midpoint(w[0], w[1]));
                // Boundary-coincident midpoints count as in-face: a section
                // that re-traces a boundary edge (the load-bearing
                // whole-edge duplicate of the shelled-cup lip fuses) has
                // every midpoint ON the polygon, where the even-odd test
                // is unreliable.
                let inside = crate::builder::classify_2d::point_in_polygon_2d(mid_uv, poly)
                    || crate::builder::classify_2d::distance_to_polygon_boundary(mid_uv, poly)
                        <= crate::builder::classify_2d::boundary_eps(poly);
                inside.then_some((w[0], w[1]))
            })
            .collect()
    } else {
        // Non-plane fallback: the historical outermost pair.
        vec![(
            crossings[0].clamp(0.0, 1.0),
            crossings[crossings.len() - 1].clamp(0.0, 1.0),
        )]
    };

    let t_tol = tol / line_len;
    let mut out: Vec<(Point3, Point3)> = Vec::new();
    'interval: for (t0, t1) in t_intervals {
        if (t1 - t0).abs() < t_tol {
            continue;
        }
        let clipped_start = line_start + line_dir * t0;
        let clipped_end = line_start + line_dir * t1;
        // Discard pieces that lie entirely ON a single face boundary edge
        // (an adjacent coplanar face's FF section coinciding with a boundary
        // edge contributes no split).
        for (seg_start, seg_end) in &boundary_segments {
            let start_dist = point_to_segment_dist_3d(clipped_start, *seg_start, *seg_end);
            let end_dist = point_to_segment_dist_3d(clipped_end, *seg_start, *seg_end);
            if start_dist < tol && end_dist < tol {
                continue 'interval;
            }
        }
        out.push((clipped_start, clipped_end));
    }
    if out.is_empty() { None } else { Some(out) }
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

/// Angular u-extent of a face's outer wire on a u-periodic surface,
/// measured as the period minus the largest angular gap between boundary
/// samples (robust against the 2pi wrap).
fn face_u_span(topo: &Topology, face: &brepkit_topology::face::Face) -> Option<f64> {
    const TAU: f64 = std::f64::consts::TAU;
    let surface = face.surface();
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut us: Vec<f64> = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let sp = topo.vertex(edge.start()).ok()?.point();
        let ep = topo.vertex(edge.end()).ok()?.point();
        for i in 0..=8 {
            #[allow(clippy::cast_precision_loss)]
            let t = f64::from(i) / 8.0;
            let p = super::pcurve_compute::evaluate_edge_at_t(edge.curve(), sp, ep, t);
            if let Some((u, _)) = surface.project_point(p) {
                us.push(u.rem_euclid(TAU));
            }
        }
    }
    if us.len() < 2 {
        return None;
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut max_gap = TAU - (us[us.len() - 1] - us[0]);
    for w in us.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }
    Some(TAU - max_gap)
}

/// Build `SurfaceInfo` for a face (periodicity flags).
fn build_surface_info(topo: &Topology, face_id: FaceId) -> Option<SurfaceInfo> {
    let face = topo.face(face_id).ok()?;
    // A partial angular band (e.g. a quarter-cylinder corner of an extruded
    // rounded rectangle) must not be treated as u-periodic: the periodic
    // wire-builder normalizes u into [0, 2pi) and rejects "seam-crossing"
    // loop closures, both of which corrupt loops on faces that merely touch
    // u = 2pi without wrapping.
    // Only quarter-or-less bands are demoted: half-cylinder faces (common
    // after a box cut) still rely on the periodic seam machinery for their
    // own splits, and the wrap-corner unwrap in `split_face_2d` handles
    // sub-half bands without it.
    let u_wraps = || face_u_span(topo, face).is_none_or(|span| span > std::f64::consts::PI + 0.05);
    match face.surface() {
        FaceSurface::Plane { .. } => None,
        FaceSurface::Cylinder(_) => Some(SurfaceInfo::Parametric {
            u_periodic: u_wraps(),
            v_periodic: false,
        }),
        FaceSurface::Cone(_) => Some(SurfaceInfo::Parametric {
            u_periodic: u_wraps(),
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
/// Resolve a quantized vertex key against the layered vertex pools without
/// copying the shared pools into a per-sub-face map.
///
/// Lookup order reproduces the former seeded-cache semantics exactly: a vertex
/// already created during THIS sub-face (`local`) wins, then the VV-merged
/// `seed`, then the rank `pool`; only a genuine miss runs `fallback` (which
/// creates or registry-resolves the vertex) and records it in `local`. Cloning
/// `seed` and copying `pool` into a fresh map for every sub-face was O(pool)
/// per call — quadratic on faces that end up with many inner wires (holes).
fn layered_vertex(
    local: &mut BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    seed: &BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    pool: Option<&BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>>,
    key: (i64, i64, i64),
    fallback: impl FnOnce() -> brepkit_topology::vertex::VertexId,
) -> brepkit_topology::vertex::VertexId {
    if let Some(&v) = local.get(&key) {
        return v;
    }
    if let Some(&v) = seed.get(&key) {
        return v;
    }
    if let Some(&v) = pool.and_then(|p| p.get(&key)) {
        return v;
    }
    let v = fallback();
    crate::perf::bump_local_vertex_insert();
    local.insert(key, v);
    v
}

#[allow(clippy::too_many_arguments)]
fn resolve_edge_vertices(
    topo: &mut Topology,
    local: &mut BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    seed: &BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    pool: Option<&BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>>,
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
        if let Some(pb) = pb
            && let (true, Some(split_edge)) = (is_cb, pb.split_edge)
        {
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
                        let vs = layered_vertex(local, seed, pool, qs, || {
                            *pb_registry.entry(qs).or_insert(se_start)
                        });
                        let ve = layered_vertex(local, seed, pool, qe, || {
                            *pb_registry.entry(qe).or_insert(se_end)
                        });
                        return (vs, ve);
                    }
                    if rev_match {
                        let qs = quantize(edge.start_3d);
                        let qe = quantize(edge.end_3d);
                        let vs = layered_vertex(local, seed, pool, qs, || {
                            *pb_registry.entry(qs).or_insert(se_end)
                        });
                        let ve = layered_vertex(local, seed, pool, qe, || {
                            *pb_registry.entry(qe).or_insert(se_start)
                        });
                        return (vs, ve);
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
        layered_vertex(local, seed, pool, key, || {
            pb_registry
                .get(&key)
                .copied()
                .unwrap_or_else(|| topo.add_vertex(Vertex::new(edge.start_3d, tol.linear)))
        })
    };
    let end_vid = {
        let key = quantize(edge.end_3d);
        layered_vertex(local, seed, pool, key, || {
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
    _parent_face_id: FaceId,
    _shared_edge_cache: &mut HashMap<(usize, usize), brepkit_topology::edge::EdgeId>,
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
    // Existing vertices are resolved by reference from the VV-merged seed and
    // this rank's fresh-vertex pool (see `layered_vertex`); only vertices
    // created during THIS sub-face land in `local_vertices`. Seeding a fresh
    // per-sub-face map from those shared pools was O(pool) × O(sub-faces) —
    // quadratic on a face that ends up with many inner wires (holes).
    let mut local_vertices: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
        BTreeMap::new();

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
            &mut local_vertices,
            vv_vertex_seed,
            rank_pool,
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
        // Each sub-face creates its OWN edges with its own per-call vertices.
        // No edge sharing between sub-faces — different sub-faces have
        // different per-call vertex caches, so shared edges would have wrong
        // VertexId connections at wire junctions.
        // merge_duplicate_edges in BuilderSolid handles cross-face sharing.
        let (edge_id, forward) = instantiate_wire_edge(topo, start_vid, end_vid, pcurve_edge);
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
                &mut local_vertices,
                vv_vertex_seed,
                rank_pool,
                pb_vertex_registry,
                pcurve_edge,
                arena,
                &quantize,
                tol,
            );
            let (edge_id, forward) = instantiate_wire_edge(topo, start_vid, end_vid, pcurve_edge);
            inner_oriented.push(OrientedEdge::new(edge_id, forward));
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

/// Create a topology edge for a wire's pcurve edge, returning the edge id
/// and the oriented-edge forward flag for the wire traversal.
///
/// Open Line edges encode the traversal in their vertex order and are
/// always forward. Open Circle/Ellipse edges must keep their vertex order
/// aligned with the curve parameterization — an open arc edge implicitly
/// spans the CCW range from its start vertex to its end vertex, so storing
/// traversal order for a reverse-traversed arc would flip the geometry to
/// the complementary arc. Closed curved edges (start == end) cannot encode
/// direction via vertices and keep the pcurve flag for winding.
fn instantiate_wire_edge(
    topo: &mut Topology,
    start_vid: brepkit_topology::vertex::VertexId,
    end_vid: brepkit_topology::vertex::VertexId,
    pcurve_edge: &super::split_types::OrientedPCurveEdge,
) -> (brepkit_topology::edge::EdgeId, bool) {
    let is_arc = matches!(
        pcurve_edge.curve_3d,
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)
    );
    if is_arc && start_vid != end_vid && !pcurve_edge.forward {
        let edge_id = topo.add_edge(Edge::new(end_vid, start_vid, pcurve_edge.curve_3d.clone()));
        (edge_id, false)
    } else {
        let edge_id = topo.add_edge(Edge::new(start_vid, end_vid, pcurve_edge.curve_3d.clone()));
        (edge_id, start_vid != end_vid || pcurve_edge.forward)
    }
}

/// Pre-split sections at registered split points that lie on their curves.
///
/// A plane face's arrangement can split a section at a true interior crossing;
/// a curved face sharing the same FF curve must split at the identical 3D
/// point or the two faces' partitions desynchronize and the shared pieces
/// pair with nothing. Curved faces' marched sections carry no pave block id,
/// so registered points are matched GEOMETRICALLY: a point cuts a section
/// when it lies on the section's curve within the weld band and strictly
/// inside its span. Pieces drop the parent's pave block id (vertex resolution
/// would snap both halves to the un-split PaveBlock endpoints) and clear UV
/// hints so consumers re-derive them.
fn presplit_sections_at_registry(
    sections: &[crate::builder::split_types::SectionEdge],
    registry: &std::collections::HashMap<usize, Vec<brepkit_math::vec::Point3>>,
    tol: f64,
) -> Vec<crate::builder::split_types::SectionEdge> {
    // Sections on curved faces arrive without pave-block ids, so match
    // registered points GEOMETRICALLY: a point splits a section when it lies
    // on the section's curve within the weld band (the two faces' copies of
    // one FF curve agree to fit error). Dense sampling pre-locates; a local
    // ternary refine certifies the distance.
    let weld = tol * 100.0;
    let on_curve =
        |s: &crate::builder::split_types::SectionEdge, p: brepkit_math::vec::Point3| -> bool {
            const N: usize = 64;
            let (d0, d1) = s.curve_3d.domain_with_endpoints(s.start, s.end);
            let mut best_k = 0;
            let mut best = f64::MAX;
            for k in 0..=N {
                #[allow(clippy::cast_precision_loss)]
                let t = d0 + (d1 - d0) * (k as f64 / N as f64);
                let d = (s.curve_3d.evaluate_with_endpoints(t, s.start, s.end) - p).length();
                if d < best {
                    best = d;
                    best_k = k;
                }
            }
            if best <= weld {
                return true;
            }
            #[allow(clippy::cast_precision_loss)]
            let mut lo = d0 + (d1 - d0) * ((best_k.saturating_sub(1)) as f64 / N as f64);
            #[allow(clippy::cast_precision_loss)]
            let mut hi = d0 + (d1 - d0) * (((best_k + 1).min(N)) as f64 / N as f64);
            for _ in 0..40 {
                let m1 = lo + (hi - lo) / 3.0;
                let m2 = hi - (hi - lo) / 3.0;
                let f1 = (s.curve_3d.evaluate_with_endpoints(m1, s.start, s.end) - p).length();
                let f2 = (s.curve_3d.evaluate_with_endpoints(m2, s.start, s.end) - p).length();
                if f1 < f2 {
                    hi = m2;
                } else {
                    lo = m1;
                }
            }
            let tm = 0.5 * (lo + hi);
            (s.curve_3d.evaluate_with_endpoints(tm, s.start, s.end) - p).length() <= weld
        };
    let all_points: Vec<brepkit_math::vec::Point3> = registry.values().flatten().copied().collect();
    let mut out = Vec::with_capacity(sections.len());
    for s in sections {
        let chord = s.end - s.start;
        let cl2 = chord.dot(chord);
        if cl2 <= tol * tol {
            out.push(s.clone());
            continue;
        }
        let guard = tol * 10.0;
        let mut cuts: Vec<(f64, brepkit_math::vec::Point3)> = all_points
            .iter()
            .filter(|p| ((**p) - s.start).length() > guard && ((**p) - s.end).length() > guard)
            .filter(|p| on_curve(s, **p))
            .map(|p| (((*p) - s.start).dot(chord) / cl2, *p))
            .filter(|(t, _)| *t > 0.0 && *t < 1.0)
            .collect();
        if cuts.is_empty() {
            out.push(s.clone());
            continue;
        }
        cuts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        cuts.dedup_by(|a, b| (a.1 - b.1).length() < guard);
        let mut prev = s.start;
        for (_, p) in &cuts {
            let mut piece = s.clone();
            piece.start = prev;
            piece.end = *p;
            piece.start_uv_a = None;
            piece.end_uv_a = None;
            piece.start_uv_b = None;
            piece.end_uv_b = None;
            piece.pave_block_id = None;
            out.push(piece);
            prev = *p;
        }
        let mut last = s.clone();
        last.start = prev;
        last.start_uv_a = None;
        last.end_uv_a = None;
        last.start_uv_b = None;
        last.end_uv_b = None;
        last.pave_block_id = None;
        out.push(last);
    }
    out
}

#[cfg(test)]
mod clip_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::clip_line_to_face_boundary;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::topology::Topology;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn disc_face(topo: &mut Topology, r: f64) -> FaceId {
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        let seam = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        let e = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle)));
        let wire = Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    #[test]
    fn through_chord_far_from_seam_clips_to_rim_crossings() {
        // A chord crossing the disc well away from its seam vertex (the seam
        // sits at (+r, 0)). The rim is ONE closed circle edge; sampling it
        // through the endpoint span collapses every polygon vertex to the
        // seam, so the in-face interval's midpoint classified outside and the
        // whole chord was dropped (the lite pad top-disc's north wall chord —
        // its east twin only survived by passing within the boundary band of
        // the degenerate seam-point polygon).
        let mut topo = Topology::new();
        let face = disc_face(&mut topo, 4.45);
        let out = clip_line_to_face_boundary(
            &topo,
            face,
            Point3::new(-20.0, 4.4, 0.0),
            Point3::new(20.0, 4.4, 0.0),
            1e-7,
        );
        let segs = out.expect("a through-chord far from the seam must be kept");
        assert_eq!(segs.len(), 1);
        let (a, b) = segs[0];
        let half = (4.45_f64 * 4.45 - 4.4 * 4.4).sqrt();
        assert!((a.x().abs() - half).abs() < 1e-6, "start {a:?}");
        assert!((b.x().abs() - half).abs() < 1e-6, "end {b:?}");
        assert!((a.y() - 4.4).abs() < 1e-6 && (b.y() - 4.4).abs() < 1e-6);
    }

    #[test]
    fn single_crossing_interior_endpoint_chord_is_kept() {
        // A chord from an interior point out past the rim crosses the disc's
        // boundary circle exactly once. It must clip to [interior, rim], not be
        // dropped — the box-corner cap-disc bite, where each chord runs from the
        // interior corner vertex to the rim. Before the fix the `crossings < 2`
        // bail discarded it and the whole cap disc fell through unsplit.
        let mut topo = Topology::new();
        let face = disc_face(&mut topo, 8.0);
        let out = clip_line_to_face_boundary(
            &topo,
            face,
            Point3::new(5.0, 5.0, 0.0),
            Point3::new(12.0, 5.0, 0.0),
            1e-7,
        );
        let segs = out.expect("single-crossing interior→rim chord must be kept");
        assert_eq!(segs.len(), 1);
        let (s, e) = segs[0];
        let rim = 39.0_f64.sqrt();
        let eq = |p: Point3, q: Point3| (p - q).length() < 1e-6;
        assert!(
            (eq(s, Point3::new(5.0, 5.0, 0.0)) && eq(e, Point3::new(rim, 5.0, 0.0)))
                || (eq(e, Point3::new(5.0, 5.0, 0.0)) && eq(s, Point3::new(rim, 5.0, 0.0))),
            "clipped to interior→rim, got {s:?}..{e:?}"
        );
    }

    #[test]
    fn chord_entirely_outside_disc_is_dropped() {
        // The relaxed bail must still reject a segment with no material inside:
        // the midpoint classification returns no in-face window.
        let mut topo = Topology::new();
        let face = disc_face(&mut topo, 8.0);
        let out = clip_line_to_face_boundary(
            &topo,
            face,
            Point3::new(20.0, 20.0, 0.0),
            Point3::new(20.0, -20.0, 0.0),
            1e-7,
        );
        assert!(out.is_none());
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::Vec3;

    #[test]
    fn closed_rim_chord_crossing_keeps_seam_angle_hit() {
        // A diameter chord of a closed cap rim crosses it twice; one of the
        // crossings lands exactly at the rim's seam angle. The closed-rim
        // path must keep BOTH (the major-arc complement filter used to drop
        // the seam-angle hit, costing the cap its half-disc split).
        let circle =
            Circle3D::new(Point3::new(10.0, 10.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let seam = brepkit_math::traits::ParametricCurve::evaluate(&circle, 0.0);
        let hits = arc_segment_crossings(
            &EdgeCurve::Circle(circle),
            seam,
            seam,
            true,
            Point3::new(20.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
            1e-7,
        );
        assert_eq!(hits.len(), 2, "both chord crossings must survive");
        let xs: Vec<f64> = hits.iter().map(|(p, _)| p.x()).collect();
        assert!(
            xs.iter().any(|&x| (x - 13.0).abs() < 1e-6)
                && xs.iter().any(|&x| (x - 7.0).abs() < 1e-6),
            "expected crossings at x=7 and x=13, got {xs:?}"
        );
    }
}
