//! Same-domain face detection via edge-set hashing.
//!
//! When two faces from opposing solids share the same underlying surface
//! AND identical boundary edge sets (same vertex pairs), they are "same-domain"
//! faces. This module detects SD groups using edge-set hashing and union-find,
//! returning `SameDomainPair` records for downstream use.
//!
//! The SD pair list is used by [`crate::bop::select_faces`] to apply
//! operation-specific deduplication (fuse keeps one representative,
//! cut keeps B reversed, etc.) without encoding operation semantics
//! into the classification pipeline.
//!
//! **Note:** Representative replacement (substituting all group members'
//! images with a single representative face) is not yet implemented.
//! Currently only pairwise SD records are emitted.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use super::SubFace;
use crate::ds::{GfaArena, Rank};
use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

/// A detected same-domain face pair.
#[derive(Debug, Clone)]
pub struct SameDomainPair {
    /// Sub-face index from solid A.
    pub idx_a: usize,
    /// Sub-face index from solid B.
    pub idx_b: usize,
    /// `true` if the effective oriented normals (surface normal combined
    /// with face reversal) point the same direction, `false` if opposite.
    pub same_orientation: bool,
    /// `true` if B's face is fully contained within A's boundary.
    /// For edge-set matched faces, both faces have identical boundaries,
    /// so this is always `false` (touching, not contained).
    pub b_contained_in_a: bool,
    /// Sub-face index to keep when a Fuse/Intersect retains one representative
    /// of this pair. For coextensive (edge-set) pairs this is `idx_a` — either
    /// face spans the same domain, so keeping A matches historical behaviour.
    /// For geometric-containment pairs the two faces have **different extent**
    /// (one is contained inside the other), so the representative is the
    /// larger (containing) face — chosen by geometry, not by which operand is
    /// A vs B. Keeping the smaller face would leave the containing face's
    /// surplus area uncovered (free edges), and which face is A flips with
    /// operand order, so an A-only rule makes the result order-dependent.
    pub representative: usize,
}

/// A within-rank duplicate sub-face: same edge set, same surface, same input
/// solid as another face. Issue #696: sequential boolean operations
/// (`booleanPipeline` in the consumer) accumulate stale coincident faces in
/// the input solid; when the next boolean splits its inputs into sub-faces,
/// these duplicates produce 3+-face junctions in the output topology that
/// tessellate as branching mesh edges. The `representative` is the lowest-
/// indexed sub-face in the group; `duplicate` should be excluded from the
/// boolean result.
#[derive(Debug, Clone, Copy)]
pub struct WithinRankDuplicate {
    /// Sub-face index that stays in the result.
    pub representative: usize,
    /// Sub-face index that should be dropped.
    pub duplicate: usize,
}

/// Output of [`detect_same_domain`].
#[derive(Debug, Default, Clone)]
pub struct SameDomainResult {
    /// Cross-rank pairs (one face from A, one from B).
    pub pairs: Vec<SameDomainPair>,
    /// Within-rank duplicates (multiple faces from the same input solid
    /// occupying the same domain — boolean residue that needs removing
    /// before classification).
    pub within_rank_dups: Vec<WithinRankDuplicate>,
}

/// Quantized 3D grid position — collision-free vertex identity.
type QVert = (i64, i64, i64);

/// Canonical representation of a face's edge set for SD detection.
///
/// Each edge is stored as a sorted quantized vertex pair `(min, max)`.
/// The set of pairs is sorted for deterministic comparison.
type EdgeSet = Vec<(QVert, QVert)>;

/// Detect same-domain face pairs using edge-set hashing.
///
/// Algorithm:
/// 1. For each sub-face, compute its canonical edge set (sorted vertex pairs)
/// 2. Hash the edge set and group faces with identical sets
/// 3. Within each group, verify surface equivalence across opposing solids
/// 4. Build SD pairs via union-find for transitive closure
///
/// Returns a list of SD pairs WITHOUT modifying sub-face classifications.
/// The BOP selector uses these pairs for operation-specific handling.
#[allow(clippy::too_many_lines)]
pub fn detect_same_domain<S: BuildHasher>(
    topo: &Topology,
    arena: &GfaArena,
    sub_faces: &[SubFace],
    _face_ranks: &HashMap<FaceId, Rank, S>,
    tol: Tolerance,
) -> SameDomainResult {
    let n = sub_faces.len();
    if n < 2 {
        return SameDomainResult::default();
    }

    // Use quantized vertex positions (not VertexId) so that VV-merged
    // vertices from different solids that share the same position produce
    // matching edge sets.
    let scale = 1.0 / tol.linear;

    let edge_sets: Vec<Option<EdgeSet>> = sub_faces
        .iter()
        .map(|sf| compute_edge_set_quantized(topo, arena, sf.face_id, scale))
        .collect();

    // Key = edge set, Value = list of sub-face indices with that set.
    let mut groups: HashMap<EdgeSet, Vec<usize>> = HashMap::new();
    for (idx, edge_set) in edge_sets.iter().enumerate() {
        if let Some(es) = edge_set
            && !es.is_empty()
        {
            groups.entry(es.clone()).or_default().push(idx);
        }
    }

    let surfaces: Vec<Option<&FaceSurface>> = sub_faces
        .iter()
        .map(|sf| {
            topo.face(sf.face_id)
                .ok()
                .map(brepkit_topology::face::Face::surface)
        })
        .collect();

    // Surface normals alone don't define orientation: faces kept through a
    // Cut carry their original surface with a reversal flag, so the
    // effective normal is the surface normal flipped when reversed.
    let reversed: Vec<bool> = sub_faces
        .iter()
        .map(|sf| {
            topo.face(sf.face_id)
                .is_ok_and(brepkit_topology::face::Face::is_reversed)
        })
        .collect();

    let mut uf = UnionFind::new(n);
    let mut pair_data: HashMap<(usize, usize), bool> = HashMap::new(); // (min,max) → same_orientation
    // Tracks pairs unioned by the geometric containment pass (Step 3b).
    // Cross-rank groups containing such pairs are "overlapping" same-domain
    // faces, not "touching" — `b_contained_in_a` must be true for them so
    // `apply_sd_selection` cancels both faces under Cut instead of keeping A.
    let mut geometric_overlap_groups: HashSet<usize> = HashSet::new();

    for members in groups.values() {
        if members.len() < 2 {
            continue;
        }

        // Check all pairs within this edge-set group. Pairs can be cross-rank
        // (the classic SD case — same domain across two input solids) or
        // within-rank (issue #696 — boolean residue accumulated in one input
        // across sequential operations). Both unify into the same group; the
        // representative-emission step below splits them by rank composition.
        for (mi, &i) in members.iter().enumerate() {
            let Some(surf_i) = surfaces[i] else {
                continue;
            };

            for &j in &members[mi + 1..] {
                let Some(surf_j) = surfaces[j] else {
                    continue;
                };

                if let Some(same_dir) = surfaces_same_domain(surf_i, surf_j, tol) {
                    uf.union(i, j);
                    let key = (i.min(j), i.max(j));
                    pair_data.insert(key, same_dir ^ (reversed[i] != reversed[j]));
                }
            }
        }
    }

    // Step 3b (issue #696): geometric containment pass for planar faces.
    // Edge-set hashing alone misses the common boolean-residue pattern where
    // one face is fully contained inside another with a different boundary
    // (e.g., a stale nub-bottom face filling the hole in a slab-top face).
    // For planar faces with the same surface, test whether one's
    // pre-computed interior point lies inside the other's wire — if so, the
    // contained face is a duplicate. Limited to planar faces because the
    // analytic surfaces (cylinder/sphere/etc) produce well-defined trimmed
    // patches that rarely accumulate residue, and a 2D containment test on
    // their parametric domains needs surface-specific handling.
    {
        let mut planar_indices: Vec<usize> = Vec::new();
        for (idx, surf) in surfaces.iter().enumerate() {
            if matches!(surf, Some(FaceSurface::Plane { .. })) {
                planar_indices.push(idx);
            }
        }
        for (mi, &i) in planar_indices.iter().enumerate() {
            for &j in &planar_indices[mi + 1..] {
                // Cheap surface-match guard first.
                let same_dir = match (surfaces[i], surfaces[j]) {
                    (Some(si), Some(sj)) => surfaces_same_domain(si, sj, tol),
                    _ => None,
                };
                let Some(same_dir) = same_dir else { continue };
                if uf.find(i) == uf.find(j) {
                    continue; // already grouped
                }
                if planar_faces_overlap(topo, sub_faces, i, j, tol) {
                    uf.union(i, j);
                    let key = (i.min(j), i.max(j));
                    pair_data.insert(key, same_dir ^ (reversed[i] != reversed[j]));
                    // Mark the post-union root so the emission code knows
                    // this group came from geometric containment, not from
                    // boundary-identical edge sets.
                    geometric_overlap_groups.insert(uf.find(i));
                }
            }
        }
    }

    // Collect all roots that participate in pairs (O(m) not O(n*m)).
    let mut active_roots: HashSet<usize> = HashSet::new();
    for &(a, b) in pair_data.keys() {
        active_roots.insert(uf.find(a));
        active_roots.insert(uf.find(b));
    }

    // Each group picks A's face with smallest index as representative.
    let mut sd_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for idx in 0..n {
        let root = uf.find(idx);
        if active_roots.contains(&root) {
            sd_groups.entry(root).or_default().push(idx);
        }
    }

    let mut pairs = Vec::new();
    let mut within_rank_dups = Vec::new();

    for (root, members) in &sd_groups {
        if members.len() < 2 {
            continue;
        }

        let repr_a = members
            .iter()
            .filter(|&&idx| sub_faces[idx].rank == Rank::A)
            .min()
            .copied();
        let repr_b = members
            .iter()
            .filter(|&&idx| sub_faces[idx].rank == Rank::B)
            .min()
            .copied();

        // True if any pair in this group was unioned by the geometric
        // containment pass. Cross-rank groups flagged here have actual
        // interior overlap (one face fully contained in another), not just
        // a shared boundary — `apply_sd_selection` needs `b_contained_in_a`
        // to be true so Cut cancels both faces instead of keeping A.
        let geometric_overlap = geometric_overlap_groups.contains(root);

        match (repr_a, repr_b) {
            // Cross-rank: classic SD pair — emit for operation-specific selection.
            (Some(idx_a), Some(idx_b)) => {
                let key = (idx_a.min(idx_b), idx_a.max(idx_b));
                let same_orientation = pair_data.get(&key).copied().unwrap_or(true);

                // Pick the Fuse/Intersect representative by geometry, not rank.
                // Coextensive (edge-set) pairs share the same domain, so A is a
                // fine representative (and matches historical behaviour). A
                // geometric-containment pair has two faces of different extent;
                // the larger (containing) face must be the representative so its
                // surplus area stays covered. Which face is A flips with operand
                // order, so deferring to area keeps the result order-independent.
                let representative = if geometric_overlap {
                    let area_a = planar_face_area(topo, sub_faces[idx_a].face_id);
                    let area_b = planar_face_area(topo, sub_faces[idx_b].face_id);
                    match (area_a, area_b) {
                        (Some(aa), Some(ab)) if ab > aa => idx_b,
                        _ => idx_a,
                    }
                } else {
                    idx_a
                };

                pairs.push(SameDomainPair {
                    idx_a,
                    idx_b,
                    same_orientation,
                    b_contained_in_a: geometric_overlap,
                    representative,
                });

                // The group may also contain additional same-rank members
                // (rare — a 3+ member group spanning both ranks). Treat those
                // as within-rank duplicates against the matching-rank repr.
                for &idx in members {
                    if idx == idx_a || idx == idx_b {
                        continue;
                    }
                    let rep = if sub_faces[idx].rank == Rank::A {
                        idx_a
                    } else {
                        idx_b
                    };
                    within_rank_dups.push(WithinRankDuplicate {
                        representative: rep,
                        duplicate: idx,
                    });
                }
            }
            // Within-rank only (A-only or B-only): cumulative boolean residue.
            // Keep the lowest-indexed face as representative; mark the rest
            // as duplicates so the BOP selector can drop them before
            // classification (issue #696).
            (Some(rep), None) | (None, Some(rep)) => {
                for &idx in members {
                    if idx != rep {
                        within_rank_dups.push(WithinRankDuplicate {
                            representative: rep,
                            duplicate: idx,
                        });
                    }
                }
            }
            (None, None) => {}
        }
    }

    // Sort outputs deterministically — `sd_groups.values()` iterates a
    // HashMap, so without sorting the pair order varies per run and
    // propagates into face ordering in the result shell (drove 100–500×
    // perf variance in `bench_boolean_64_holes`).
    pairs.sort_unstable_by_key(|p| (p.idx_a, p.idx_b));
    within_rank_dups.sort_unstable_by_key(|d| (d.representative, d.duplicate));

    log::debug!(
        "detect_same_domain: {} cross-rank pairs, {} within-rank duplicates (edge-set hash)",
        pairs.len(),
        within_rank_dups.len()
    );

    SameDomainResult {
        pairs,
        within_rank_dups,
    }
}

/// Compute the canonical edge set for a face using quantized vertex positions.
///
/// Each edge in the outer wire is represented as a sorted pair of quantized
/// 3D positions. The pairs are sorted for deterministic comparison.
/// Using quantized positions instead of `VertexId` ensures that vertices
/// from different solids that share the same position (merged by VV phase)
/// produce matching edge sets.
///
/// Only the outer wire is considered. Inner wires (holes) are intentionally
/// excluded: SD faces in boolean operations share the same outer boundary
/// but may differ in holes (which are handled by the BOP selector).
fn compute_edge_set_quantized(
    topo: &Topology,
    arena: &GfaArena,
    face_id: FaceId,
    scale: f64,
) -> Option<EdgeSet> {
    use brepkit_topology::vertex::VertexId;

    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;

    let mut pairs: Vec<(QVert, QVert)> = Vec::with_capacity(wire.edges().len());

    // Cache resolved vertex positions to avoid redundant resolve_vertex() calls
    // when the same vertex appears in multiple edges.
    let mut vertex_cache: HashMap<VertexId, QVert> = HashMap::new();
    let mut resolve_and_quantize = |vid: VertexId| -> Option<QVert> {
        if let Some(&cached) = vertex_cache.get(&vid) {
            return Some(cached);
        }
        let resolved = arena.resolve_vertex(vid);
        let pos = topo.vertex(resolved).ok()?.point();
        let q = quantize_point(pos, scale);
        vertex_cache.insert(vid, q);
        Some(q)
    };

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;

        let qs = resolve_and_quantize(edge.start())?;
        let qe = resolve_and_quantize(edge.end())?;

        // Canonical ordering: smaller first
        let pair = if qs <= qe { (qs, qe) } else { (qe, qs) };
        pairs.push(pair);
    }

    pairs.sort_unstable();
    Some(pairs)
}

/// Test whether two planar sub-faces are geometrically coincident or one
/// is fully contained inside the other.
///
/// Returns `true` only when ALL outer-wire vertices of one face lie inside
/// or on the boundary of the other face's outer polygon (and the interior
/// sample point confirms it). A weaker "interior-only" containment test was
/// tried and rejected: adjacent coplanar faces with concave geometry could
/// have an interior point that happens to land inside a neighbor's polygon
/// without the faces actually overlapping. Requiring whole-wire containment
/// is the conservative criterion that catches boolean residue (issue #696)
/// — typically a small "filling" face inside a larger face's outer
/// boundary — without firing on legitimate adjacent face pairs.
fn planar_faces_overlap(
    topo: &Topology,
    sub_faces: &[SubFace],
    i: usize,
    j: usize,
    tol: Tolerance,
) -> bool {
    let Ok(face_i) = topo.face(sub_faces[i].face_id) else {
        return false;
    };
    let Ok(face_j) = topo.face(sub_faces[j].face_id) else {
        return false;
    };
    let FaceSurface::Plane {
        normal: normal_i, ..
    } = *face_i.surface()
    else {
        return false;
    };

    // Sample each edge into several points along its curve, not just the
    // start vertex. A closed wire built from a single circular edge (a
    // circular hole left by an earlier cut) has one start vertex, so a
    // vertex-only polygon collapses to a single point and the hole
    // containment test silently treats the hole as absent — letting a
    // coincident coplanar face be wrongly cancelled through the hole.
    let wire_points = |wire_id: brepkit_topology::wire::WireId| -> Vec<brepkit_math::vec::Point3> {
        let samples_per_edge: usize = 8;
        let mut pts = Vec::new();
        let Ok(wire) = topo.wire(wire_id) else {
            return pts;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                continue;
            };
            let (sp, ep) = (sv.point(), ev.point());
            // Sample via the shorter-arc evaluator: split faces can store
            // arc edges whose vertex order opposes the circle's CCW
            // parameterization, and domain-based sampling would then trace
            // the complementary (long-way) arc, corrupting the polygon used
            // for the containment tests below.
            for k in 0..samples_per_edge {
                #[allow(clippy::cast_precision_loss)]
                let frac = k as f64 / samples_per_edge as f64;
                let frac = if oe.is_forward() { frac } else { 1.0 - frac };
                pts.push(super::pcurve_compute::evaluate_edge_at_t(
                    edge.curve(),
                    sp,
                    ep,
                    frac,
                ));
            }
        }
        pts
    };

    let pts_i = wire_points(face_i.outer_wire());
    let pts_j = wire_points(face_j.outer_wire());
    if pts_i.len() < 3 || pts_j.len() < 3 {
        return false;
    }
    let frame = super::plane_frame::PlaneFrame::from_plane_face(normal_i, &pts_i);
    let poly_i: Vec<_> = pts_i.iter().map(|&p| frame.project(p)).collect();
    let poly_j: Vec<_> = pts_j.iter().map(|&p| frame.project(p)).collect();

    // Passthrough faces arrive without a pre-computed interior point;
    // derive one from the projected outer polygon so coincident-outline
    // pairs (split disc vs. unsplit opposing cap) are still testable.
    let p_i_2d = sub_faces[i].interior_point.map_or_else(
        || super::classify_2d::sample_interior_point(&poly_i),
        |p| frame.project(p),
    );
    let p_j_2d = sub_faces[j].interior_point.map_or_else(
        || super::classify_2d::sample_interior_point(&poly_j),
        |p| frame.project(p),
    );

    // Strict containment: every vertex of `verts` lies inside `poly` by the
    // ray-cast test, no boundary tolerance.
    let all_inside_strict =
        |verts: &[brepkit_math::vec::Point2], poly: &[brepkit_math::vec::Point2]| -> bool {
            verts
                .iter()
                .all(|&v| super::classify_2d::point_in_polygon_2d(v, poly))
        };

    // Boundary-tolerant containment: a coincident-outline pair (e.g. a
    // section-loop disc vs. the opposing solid's cap with differently split
    // boundary edges) has every vertex exactly ON the container's polygon,
    // where the strict ray-cast is unpredictable.
    let all_inside_tol =
        |verts: &[brepkit_math::vec::Point2], poly: &[brepkit_math::vec::Point2]| -> bool {
            let boundary_eps = super::classify_2d::boundary_eps(poly);
            verts.iter().all(|&v| {
                super::classify_2d::point_in_polygon_2d(v, poly)
                    || super::classify_2d::distance_to_polygon_boundary(v, poly) <= boundary_eps
            })
        };

    // Two coplanar faces that tile disjoint side-by-side regions share a
    // boundary segment, so every vertex of one lands ON the other's polygon
    // and `all_inside_tol` reports a false containment in a single direction.
    // A genuine coincident-outline pair (the case boundary tolerance exists
    // for) instead has BOTH faces' interior points mutually inside, because
    // the outlines coincide. Require that mutual containment before trusting
    // a boundary-tolerant match; strict containment needs no such guard.
    let ip_i_in_j = super::classify_2d::point_in_polygon_2d(p_i_2d, &poly_j);
    let ip_j_in_i = super::classify_2d::point_in_polygon_2d(p_j_2d, &poly_i);
    let outlines_coincide = ip_i_in_j && ip_j_in_i;
    let all_inside =
        |verts: &[brepkit_math::vec::Point2], poly: &[brepkit_math::vec::Point2]| -> bool {
            all_inside_strict(verts, poly) || (outlines_coincide && all_inside_tol(verts, poly))
        };

    // A point landing inside one of the container's inner wires sits in a
    // hole, not on the face — e.g. a frame face whose hole exactly hosts
    // the candidate. Containment through a hole is not overlap.
    let in_hole = |p: brepkit_math::vec::Point2, face: &brepkit_topology::face::Face| -> bool {
        face.inner_wires().iter().any(|&wid| {
            let pts = wire_points(wid);
            if pts.len() < 3 {
                return false;
            }
            let poly: Vec<_> = pts.iter().map(|&q| frame.project(q)).collect();
            super::classify_2d::point_in_polygon_2d(p, &poly)
        })
    };

    // A single interior sample can miss the hole for a non-convex candidate
    // straddling a hole boundary: the sample may land on solid material while
    // the candidate's footprint actually sits entirely over the container's
    // holes. As an additional (not replacement) suppressor, also reject when
    // EVERY sampled point of the candidate that lies inside the container's
    // outer boundary falls inside one of the container's holes. This keeps
    // the common case (interior sample alone) identical and only fires extra
    // for footprints fully over holes.
    let footprint_in_holes = |sample: brepkit_math::vec::Point2,
                              verts: &[brepkit_math::vec::Point2],
                              outer: &[brepkit_math::vec::Point2],
                              face: &brepkit_topology::face::Face|
     -> bool {
        if face.inner_wires().is_empty() {
            return false;
        }
        std::iter::once(sample)
            .chain(verts.iter().copied())
            .filter(|&p| super::classify_2d::point_in_polygon_2d(p, outer))
            .all(|p| in_hole(p, face))
    };

    // i fully contained in j: every vertex of i (plus its interior sample)
    // is inside j's polygon.
    if ip_i_in_j
        && all_inside(&poly_i, &poly_j)
        && !in_hole(p_i_2d, face_j)
        && !footprint_in_holes(p_i_2d, &poly_i, &poly_j, face_j)
    {
        return true;
    }
    // j fully contained in i.
    if ip_j_in_i
        && all_inside(&poly_j, &poly_i)
        && !in_hole(p_j_2d, face_i)
        && !footprint_in_holes(p_j_2d, &poly_j, &poly_i, face_i)
    {
        return true;
    }

    // Partial overlap. Two coplanar faces can share a genuine 2D area without
    // either being fully contained in the other — e.g. a faceted scoop ramp's
    // staircase-shaped wall sub-face lying against a rectangular ramp side
    // facet. Full-containment misses these; the result is a coincident face
    // pair that survives the boolean and goes non-manifold.
    //
    // Detect it by the intersection AREA of the projected polygons. A positive
    // intersection area means real overlap; faces that merely tile side-by-side
    // (sharing only a boundary segment) have zero intersection area, so this
    // does not reintroduce the side-by-side false positive the containment
    // guards above defend against. Require the overlap to cover a meaningful
    // fraction of the smaller face so a sliver of numerical overlap along a
    // shared edge does not pair disjoint faces.
    if face_j.inner_wires().is_empty() && face_i.inner_wires().is_empty() {
        let inter = brepkit_math::polygon_boolean::polygon_boolean(
            &poly_i,
            &poly_j,
            brepkit_math::polygon_boolean::BooleanOp::Intersection,
            tol.linear,
        );
        let overlap = inter.area().abs();
        let area_i = super::classify_2d::signed_area_2d(&poly_i).abs();
        let area_j = super::classify_2d::signed_area_2d(&poly_j).abs();
        let smaller = area_i.min(area_j);
        // `smaller` and `overlap` are areas, so the degenerate-face guard
        // compares against the squared linear tolerance (area), not `linear`.
        if smaller > tol.linear_sq() && overlap > smaller * 0.5 {
            return true;
        }
    }
    false
}

/// Projected outer-wire area of a planar sub-face, in its own plane.
///
/// Returns `None` for non-planar faces or faces whose outer wire samples to
/// fewer than three points. Used to pick the representative of a
/// geometric-containment SD group: the largest-area face contains the others,
/// so keeping it covers the whole shared domain regardless of operand order.
fn planar_face_area(topo: &Topology, face_id: FaceId) -> Option<f64> {
    let face = topo.face(face_id).ok()?;
    let FaceSurface::Plane { normal, .. } = *face.surface() else {
        return None;
    };
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut pts: Vec<brepkit_math::vec::Point3> = Vec::with_capacity(wire.edges().len() * 8);
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let sv = topo.vertex(edge.start()).ok()?;
        let ev = topo.vertex(edge.end()).ok()?;
        let (sp, ep) = (sv.point(), ev.point());
        // Sample each edge so arc boundaries contribute their true swept area,
        // mirroring `planar_faces_overlap`'s shorter-arc sampling.
        for k in 0..8usize {
            #[allow(clippy::cast_precision_loss)]
            let frac = k as f64 / 8.0;
            let frac = if oe.is_forward() { frac } else { 1.0 - frac };
            pts.push(super::pcurve_compute::evaluate_edge_at_t(
                edge.curve(),
                sp,
                ep,
                frac,
            ));
        }
    }
    if pts.len() < 3 {
        return None;
    }
    let frame = super::plane_frame::PlaneFrame::from_plane_face(normal, &pts);
    let poly: Vec<_> = pts.iter().map(|&p| frame.project(p)).collect();
    Some(super::classify_2d::signed_area_2d(&poly).abs())
}

/// Quantize a 3D point to integer grid coordinates.
///
/// Returns the collision-free `(i64, i64, i64)` triple directly.
fn quantize_point(p: brepkit_math::vec::Point3, scale: f64) -> QVert {
    (
        (p.x() * scale).round() as i64,
        (p.y() * scale).round() as i64,
        (p.z() * scale).round() as i64,
    )
}

/// Simple union-find (disjoint set) with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

/// Check if two surfaces represent the same geometric domain.
///
/// Returns `Some(true)` for same-direction normals (CoplanarSame),
/// `Some(false)` for opposite normals (CoplanarOpposite), or
/// `None` if not the same domain.
///
/// Visible to `crate::diagnostic` (the boolean preflight API). The
/// `redundant_pub_crate` allow is required because the enclosing
/// `builder` module is private — clippy folds `pub(crate)` to `pub`
/// in that scope, but we keep `pub(crate)` to make the intent
/// explicit in the source.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn surfaces_same_domain(
    a: &FaceSurface,
    b: &FaceSurface,
    tol: Tolerance,
) -> Option<bool> {
    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            let dot = na.dot(*nb);
            if dot > 1.0 - tol.angular {
                // Same direction — check distance
                if (da - db).abs() < tol.linear {
                    return Some(true);
                }
            } else if dot < -1.0 + tol.angular {
                // Opposite direction — check distance
                if (da + db).abs() < tol.linear {
                    return Some(false);
                }
            }
            None
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            // Same cylinder: same origin, same axis, same radius
            if (ca.radius() - cb.radius()).abs() > tol.linear {
                return None;
            }
            let axis_dot = ca.axis().dot(cb.axis());
            if axis_dot.abs() < 1.0 - tol.angular {
                return None;
            }
            // Check if origins lie on the same axis line
            let diff = cb.origin() - ca.origin();
            let along_axis = diff.dot(ca.axis());
            let perp_dist = (diff - ca.axis() * along_axis).length();
            if perp_dist > tol.linear {
                return None;
            }
            Some(axis_dot > 0.0)
        }
        (FaceSurface::Sphere(sa), FaceSurface::Sphere(sb)) => {
            if (sa.radius() - sb.radius()).abs() > tol.linear {
                return None;
            }
            let dist = (sa.center() - sb.center()).length();
            if dist > tol.linear {
                return None;
            }
            Some(true)
        }
        (FaceSurface::Cone(ca), FaceSurface::Cone(cb)) => {
            if (ca.half_angle() - cb.half_angle()).abs() > tol.angular {
                return None;
            }
            let axis_dot = ca.axis().dot(cb.axis());
            if axis_dot.abs() < 1.0 - tol.angular {
                return None;
            }
            let dist = (ca.apex() - cb.apex()).length();
            if dist > tol.linear {
                return None;
            }
            Some(axis_dot > 0.0)
        }
        (FaceSurface::Torus(ta), FaceSurface::Torus(tb)) => {
            if (ta.major_radius() - tb.major_radius()).abs() > tol.linear {
                return None;
            }
            if (ta.minor_radius() - tb.minor_radius()).abs() > tol.linear {
                return None;
            }
            let axis_dot = ta.z_axis().dot(tb.z_axis());
            if axis_dot.abs() < 1.0 - tol.angular {
                return None;
            }
            let dist = (ta.center() - tb.center()).length();
            if dist > tol.linear {
                return None;
            }
            Some(axis_dot > 0.0)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
