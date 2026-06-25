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
    /// `true` if this is a geometric-overlap pair — one face is contained in
    /// or partially over the other, so the two faces differ in extent. For
    /// edge-set matched (coextensive) faces both boundaries coincide, so this
    /// is `false`. Detection signal observed by tests; the BOP selector orders
    /// the pair via [`Self::representative`] (the larger face) rather than
    /// reading this flag directly.
    #[allow(dead_code)]
    pub geometric_overlap: bool,
    /// Sub-face index of the **larger** face of this pair by projected outer-
    /// wire area (see [`repr_face_area`] — planar faces in their plane,
    /// cylinder/cone faces in `(arc-length, axial)` space), used to keep
    /// representative selection order-independent.
    ///
    /// For coextensive (edge-set) pairs both faces span the same domain, so
    /// area ties and this is `idx_a` — matching historical behaviour. For a
    /// geometric-overlap pair (`geometric_overlap == true`) the two faces have
    /// **different extent**, so the larger is chosen by area rather than by
    /// which operand is A; `idx_a` flips with operand order, so an A-only rule
    /// would make the result order-dependent.
    ///
    /// "Larger" — not "containing": a geometric-overlap pair may be strict
    /// containment OR partial overlap ([`planar_faces_overlap`] and
    /// [`analytic_faces_overlap`] both accept either), so neither face
    /// necessarily contains the other. The consumer
    /// ([`crate::bop::select_faces`]) keeps this face for Fuse (it covers the
    /// most boundary) and the *other* (smaller) face for Intersect (whose
    /// footprint is bounded by both solids).
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

/// Number of points sampled along each outer-wire edge when building the
/// projected polygon for the coplanar containment / overlap / area tests.
/// Defined once so [`planar_faces_overlap`] and [`planar_face_area`] keep the
/// same density — an arc boundary must sample to the same polygon in both, or
/// the area-based representative pick could disagree with the overlap test.
const SD_EDGE_SAMPLES: usize = 8;

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
    // faces (one face contained in / partially over the other) rather than
    // exactly coextensive. `geometric_overlap` records this so the BOP selector
    // can pick the larger face for Fuse and the smaller for Intersect; the two
    // faces always differ in extent for these pairs.
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
                    // Two curved faces of the same underlying surface can share an
                    // outer-wire edge set yet cover DIFFERENT regions — e.g. the
                    // two hemisphere bands of a bored sphere share the equator
                    // polygon but lie on opposite halves. A genuine same-domain
                    // duplicate is coincident (same region → same interior
                    // sample); distinct glued patches have far-apart interiors.
                    // Skip the union when their interior samples disagree.
                    if !planar(surf_i) && distinct_curved_regions(sub_faces, i, j, tol) {
                        continue;
                    }
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
        let mut planar_aabbs: Vec<(usize, brepkit_math::aabb::Aabb3)> = Vec::new();
        for (idx, surf) in surfaces.iter().enumerate() {
            if matches!(surf, Some(FaceSurface::Plane { .. }))
                && let Some(bb) = face_outer_aabb(topo, sub_faces[idx].face_id)
            {
                planar_aabbs.push((idx, bb));
            }
        }
        // Broad-phase: only test pairs whose AABBs overlap. Two planar faces
        // that don't share 3D space (expanded by tol for boundary-coincident
        // outlines) cannot pass `planar_faces_overlap`, so pruning them is
        // result-preserving while collapsing the former O(n²) scan to
        // O(near). Candidate pairs come back in ascending (i, j) order — the
        // same order the nested loop visited — so the union-find sequence (and
        // hence representative selection) is unchanged.
        for (i, j) in overlap_candidate_pairs(&planar_aabbs, tol.linear) {
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

    // Step 3c: geometric-overlap pass for coaxial cylinder/cone faces.
    // Edge-set hashing pairs faces only when their boundaries coincide
    // exactly. Two operands can carry the SAME coincident curved wall with
    // MISMATCHED segmentation — e.g. a body whose rounded corner arrives split
    // into two angular eighth-cylinders against a lip whose corner is one
    // quarter-cylinder (gridfinity 3×3 stacking-lip fuse). The eighths and the
    // quarter share an identical infinite cylinder over an overlapping band but
    // no edge, so Step 1 misses them and the redundant interior pieces survive,
    // leaving the shell open. Test overlap in the surface's (arc-length, axial)
    // parameter space; the BOP selector then keeps the larger patch for Fuse /
    // the smaller for Intersect, exactly as for the planar geometric-overlap
    // pairs above.
    {
        let mut analytic_aabbs: Vec<(usize, brepkit_math::aabb::Aabb3)> = Vec::new();
        for (idx, surf) in surfaces.iter().enumerate() {
            if matches!(surf, Some(FaceSurface::Cylinder(_) | FaceSurface::Cone(_)))
                && let Some(bb) = face_outer_aabb(topo, sub_faces[idx].face_id)
            {
                analytic_aabbs.push((idx, bb));
            }
        }
        // Broad-phase: as with the planar pass, only test AABB-overlapping
        // candidate pairs. Two coaxial patches whose 3D AABBs are disjoint
        // (expanded by tol) cannot share a band, so pruning them preserves the
        // result. Candidate order is ascending (i, j), matching the former
        // nested loop, so the union-find sequence is unchanged.
        for (i, j) in overlap_candidate_pairs(&analytic_aabbs, tol.linear) {
            let same_dir = match (surfaces[i], surfaces[j]) {
                (Some(si), Some(sj)) => surfaces_same_domain(si, sj, tol),
                _ => None,
            };
            let Some(same_dir) = same_dir else { continue };
            if uf.find(i) == uf.find(j) {
                continue; // already grouped (e.g. identical edge sets)
            }
            if analytic_faces_overlap(topo, sub_faces, i, j, tol) {
                uf.union(i, j);
                let key = (i.min(j), i.max(j));
                pair_data.insert(key, same_dir ^ (reversed[i] != reversed[j]));
                geometric_overlap_groups.insert(uf.find(i));
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
        // interior overlap (one face contained in / partially over another),
        // not just a shared boundary, so the two faces differ in extent. The
        // BOP selector uses this to keep the larger face for Fuse and the
        // smaller for Intersect (see `representative` below).
        let geometric_overlap = geometric_overlap_groups.contains(root);

        match (repr_a, repr_b) {
            // Cross-rank: classic SD pair — emit for operation-specific selection.
            (Some(idx_a), Some(idx_b)) => {
                let key = (idx_a.min(idx_b), idx_a.max(idx_b));
                let same_orientation = pair_data.get(&key).copied().unwrap_or(true);

                // Record the LARGER face (by projected area) as the
                // representative, so the choice is geometry-based not rank-based.
                // Coextensive (edge-set) pairs share the same domain (area ties),
                // so A is a fine representative and matches historical behaviour.
                // A geometric-overlap pair has two faces of different extent;
                // tagging the larger lets the BOP selector keep it for Fuse and
                // the smaller for Intersect. Which face is A flips with operand
                // order, so deferring to area keeps the result order-independent.
                let representative = if geometric_overlap {
                    let area_a = repr_face_area(topo, sub_faces[idx_a].face_id);
                    let area_b = repr_face_area(topo, sub_faces[idx_b].face_id);
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
                    geometric_overlap,
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
        let samples_per_edge: usize = SD_EDGE_SAMPLES;
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
        let area_i = super::classify_2d::signed_area_2d(&poly_i).abs();
        let area_j = super::classify_2d::signed_area_2d(&poly_j).abs();
        let smaller = area_i.min(area_j);
        // The polygon intersection is contained in the overlap of the two 2D
        // bounding boxes, so `area(poly∩poly) ≤ area(bbox∩bbox)`. The exact
        // (and costly) polygon clip can only clear the 50%-of-smaller threshold
        // below when the box overlap already does — so gate the clip on the
        // cheap box test. This skips the clip for the common touching /
        // side-by-side coplanar pairs (e.g. stacked wall-piece bands) without
        // changing the result. `smaller`/`overlap` are areas, so the degenerate
        // guard compares against the squared linear tolerance.
        if smaller > tol.linear_sq() && bbox2d_overlap_area(&poly_i, &poly_j) > smaller * 0.5 {
            crate::perf::bump_sd_poly_clip();
            let inter = brepkit_math::polygon_boolean::polygon_boolean(
                &poly_i,
                &poly_j,
                brepkit_math::polygon_boolean::BooleanOp::Intersection,
                tol.linear,
            );
            if inter.area().abs() > smaller * 0.5 {
                return true;
            }
        }
    }
    false
}

/// Area of the overlap of two 2D point sets' axis-aligned bounding boxes.
///
/// A conservative (over-)estimate of the polygons' intersection area: the
/// intersection lies inside both boxes, so its area never exceeds this. Used to
/// skip the exact polygon clip when no meaningful overlap is possible.
fn bbox2d_overlap_area(a: &[brepkit_math::vec::Point2], b: &[brepkit_math::vec::Point2]) -> f64 {
    let bounds = |poly: &[brepkit_math::vec::Point2]| {
        let (mut lo_x, mut lo_y) = (f64::MAX, f64::MAX);
        let (mut hi_x, mut hi_y) = (f64::MIN, f64::MIN);
        for p in poly {
            lo_x = lo_x.min(p.x());
            lo_y = lo_y.min(p.y());
            hi_x = hi_x.max(p.x());
            hi_y = hi_y.max(p.y());
        }
        (lo_x, lo_y, hi_x, hi_y)
    };
    let (alx, aly, ahx, ahy) = bounds(a);
    let (blx, bly, bhx, bhy) = bounds(b);
    let ox = (ahx.min(bhx) - alx.max(blx)).max(0.0);
    let oy = (ahy.min(bhy) - aly.max(bly)).max(0.0);
    ox * oy
}

/// Approximate projected outer-wire area of a planar sub-face, in its own
/// plane.
///
/// Returns `None` for non-planar faces or faces whose outer wire samples to
/// fewer than three points. The area is an approximation: each edge is sampled
/// at [`SD_EDGE_SAMPLES`] points (so arc boundaries contribute their swept
/// area, matching [`planar_faces_overlap`]), then the projected polygon's
/// signed area is taken — a finer arc is under-counted by the chord polygon.
///
/// Used only to order the two faces of a geometric-overlap SD pair (see
/// [`SameDomainPair::representative`]). The two faces always share a plane, so
/// the same under-counting applies to both and their relative order is stable;
/// the absolute area is never compared against a tolerance.
fn planar_face_area(topo: &Topology, face_id: FaceId) -> Option<f64> {
    let face = topo.face(face_id).ok()?;
    let FaceSurface::Plane { normal, .. } = *face.surface() else {
        return None;
    };
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut pts: Vec<brepkit_math::vec::Point3> =
        Vec::with_capacity(wire.edges().len() * SD_EDGE_SAMPLES);
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let sv = topo.vertex(edge.start()).ok()?;
        let ev = topo.vertex(edge.end()).ok()?;
        let (sp, ep) = (sv.point(), ev.point());
        // Sample each edge so arc boundaries contribute their true swept area,
        // mirroring `planar_faces_overlap`'s shorter-arc sampling.
        for k in 0..SD_EDGE_SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let frac = k as f64 / SD_EDGE_SAMPLES as f64;
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

/// Sample a cylinder/cone sub-face's outer wire into 3D points, [`SD_EDGE_SAMPLES`]
/// per edge.
///
/// Returns `None` for non-(cylinder/cone) faces or wires that sample to fewer
/// than three points. Mirrors [`planar_faces_overlap`]'s shorter-arc edge
/// sampling so arc boundaries contribute their true swept extent. The raw 3D
/// points (not parameters) are returned so [`analytic_faces_overlap`] can
/// project BOTH faces through a single shared reference surface — projecting
/// each face through its own surface would reference the axial coordinate to a
/// different origin and falsely align disjoint z-bands.
fn wire_points_3d(topo: &Topology, face_id: FaceId) -> Option<Vec<brepkit_math::vec::Point3>> {
    let face = topo.face(face_id).ok()?;
    if !matches!(
        face.surface(),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
    ) {
        return None;
    }
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut pts = Vec::with_capacity(wire.edges().len() * SD_EDGE_SAMPLES);
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let sv = topo.vertex(edge.start()).ok()?;
        let ev = topo.vertex(edge.end()).ok()?;
        let (sp, ep) = (sv.point(), ev.point());
        for k in 0..SD_EDGE_SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let frac = k as f64 / SD_EDGE_SAMPLES as f64;
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
    Some(pts)
}

/// Project 3D points into the `(θ, axial)` parameter space of `surface`
/// (cylinder or cone), returning the `(θ, axial)` samples and the arc-length
/// scale radius.
///
/// `θ` is the raw angular parameter (radians, not yet seam-unwrapped); `axial`
/// is `v` along the surface's own axis from its origin/apex. Because both faces
/// of a candidate pair are projected through the SAME `surface`, the axial
/// reference is shared and the parameter spaces are directly comparable. The
/// returned `radius` scales `θ` into arc length so the 2D tests operate in mm;
/// for a cone it is the radius at the samples' mid-axial coordinate.
fn project_points_through_surface(
    surface: &FaceSurface,
    pts: &[brepkit_math::vec::Point3],
) -> Option<(Vec<(f64, f64)>, f64)> {
    let samples: Vec<(f64, f64)> = match surface {
        FaceSurface::Cylinder(c) => pts.iter().map(|&p| c.project_point(p)).collect(),
        FaceSurface::Cone(c) => pts.iter().map(|&p| c.project_point(p)).collect(),
        _ => return None,
    };
    if samples.len() < 3 {
        return None;
    }
    let radius = match surface {
        FaceSurface::Cylinder(c) => c.radius(),
        FaceSurface::Cone(c) => {
            let v_min = samples
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::INFINITY, f64::min);
            let v_max = samples
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::NEG_INFINITY, f64::max);
            // `radius_at` is signed: it returns a negative value when the
            // cone's axis points apex→base and the patch sits on the negative
            // side. Only the magnitude scales θ into arc length (the
            // tessellation path takes `.abs()` for the same reason).
            c.radius_at(0.5 * (v_min + v_max)).abs()
        }
        _ => return None,
    };
    // A degenerate (apex-touching) cone band has ~zero radius; the arc-length
    // scaling would collapse θ and make the 2D test meaningless.
    if radius <= 0.0 {
        return None;
    }
    Some((samples, radius))
}

/// Unwrap a sequence of raw angular samples (each in `[0, 2π)`) into a
/// continuous run by adding the multiple of `2π` to each successive sample that
/// minimizes the step from its predecessor.
///
/// A trimmed cylinder/cone patch spans less than a full turn, so its boundary
/// θ values form a continuous arc once seam-wrapping is removed.
fn unwrap_angles(samples: &[(f64, f64)]) -> Vec<(f64, f64)> {
    use std::f64::consts::TAU;
    debug_assert!(
        !samples.is_empty(),
        "unwrap_angles requires at least one sample (callers guard len >= 3)"
    );
    let mut out = Vec::with_capacity(samples.len());
    let mut prev = samples[0].0;
    out.push(samples[0]);
    for &(u, axial) in &samples[1..] {
        // Add the integer multiple of 2π that brings `u` closest to `prev`
        // (i.e. the step into [-π, π]).
        let uu = u - ((u - prev) / TAU).round() * TAU;
        out.push((uu, axial));
        prev = uu;
    }
    out
}

/// Test whether two cylinder/cone sub-faces on the **same** coaxial surface
/// have overlapping trimmed patches in `(arc-length, axial)` parameter space.
///
/// The caller must have already confirmed the two faces share an infinite
/// surface (via [`surfaces_same_domain`]) — that guarantees a `(θ, axial)`
/// pair maps to the *same* 3D point on both, so a genuine parameter-space
/// overlap is a genuine 3D overlap, with one exception this function guards:
/// the angular seam. Each face's boundary is unwrapped into a continuous θ-arc,
/// then face `j`'s arc is shifted by the multiple of `2π` that maximizes its
/// 1D overlap with face `i`'s arc. Because `P(θ, ·) = P(θ + 2π, ·)` on the
/// surface, that shift is an identity in 3D; it only selects which periodic
/// representative to compare, so two patches on *opposite* sides (no genuine
/// overlap) yield no positive overlap under any shift and are not paired.
///
/// θ is scaled by the surface radius (the patch's mid-axial radius for a cone)
/// so both axes are in mm; the 2D containment / overlap-area tests then mirror
/// [`planar_faces_overlap`] exactly, including its area-fraction guard against
/// pairing faces that merely share a boundary segment.
fn analytic_faces_overlap(
    topo: &Topology,
    sub_faces: &[SubFace],
    i: usize,
    j: usize,
    tol: Tolerance,
) -> bool {
    use std::f64::consts::TAU;

    // Project BOTH faces through face i's surface so the axial coordinate and
    // angular origin share one reference frame. The two faces are coaxial with
    // equal radius (the caller's `surfaces_same_domain` guard), so each face's
    // 3D wire points lie on face i's surface too; projecting them through it is
    // exact. Projecting each face through its OWN surface would reference the
    // axial v to a different origin (e.g. a body cylinder at z=0 vs a lip
    // cylinder at z=13.3) and falsely overlap disjoint z-bands.
    let Ok(ref_surface) = topo.face(sub_faces[i].face_id).map(|f| f.surface().clone()) else {
        return false;
    };
    let Some(pts_i) = wire_points_3d(topo, sub_faces[i].face_id) else {
        return false;
    };
    let Some(pts_j) = wire_points_3d(topo, sub_faces[j].face_id) else {
        return false;
    };
    let Some((samples_i, radius_i)) = project_points_through_surface(&ref_surface, &pts_i) else {
        return false;
    };
    let Some((samples_j, _radius_j)) = project_points_through_surface(&ref_surface, &pts_j) else {
        return false;
    };

    let unwrapped_i = unwrap_angles(&samples_i);
    let unwrapped_j = unwrap_angles(&samples_j);

    let theta_span = |pts: &[(f64, f64)]| -> (f64, f64) {
        let lo = pts.iter().map(|&(u, _)| u).fold(f64::INFINITY, f64::min);
        let hi = pts
            .iter()
            .map(|&(u, _)| u)
            .fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    };
    let (i_lo, i_hi) = theta_span(&unwrapped_i);
    let (j_lo, j_hi) = theta_span(&unwrapped_j);

    // A patch spanning (near) a full turn is a closed/seam surface, not the
    // partial corner patches this pass targets; comparing it parametrically is
    // ambiguous, so bail and let edge-set matching handle it.
    if (i_hi - i_lo) >= TAU - tol.angular || (j_hi - j_lo) >= TAU - tol.angular {
        return false;
    }

    // Shift j's θ-branch by the k·2π that maximizes 1D interval overlap with i.
    // Real trimmed patches span < 2π, so the physically-correct alignment is the
    // one with the largest overlap; when no genuine overlap exists every shift
    // gives a non-positive overlap and the 2D test below sees disjoint polygons.
    let mut best_shift = 0.0_f64;
    let mut best_overlap = f64::NEG_INFINITY;
    for k in -1..=1 {
        let shift = f64::from(k) * TAU;
        let lo = (j_lo + shift).max(i_lo);
        let hi = (j_hi + shift).min(i_hi);
        let overlap = hi - lo;
        if overlap > best_overlap {
            best_overlap = overlap;
            best_shift = shift;
        }
    }

    // Scale θ to arc length (mm) using the (shared) reference radius so the two
    // polygons are metrically consistent.
    let scale = radius_i;
    let to_2d = |&(u, axial): &(f64, f64), shift: f64| {
        brepkit_math::vec::Point2::new((u + shift) * scale, axial)
    };
    let poly_i: Vec<_> = unwrapped_i.iter().map(|s| to_2d(s, 0.0)).collect();
    let poly_j: Vec<_> = unwrapped_j.iter().map(|s| to_2d(s, best_shift)).collect();
    if poly_i.len() < 3 || poly_j.len() < 3 {
        return false;
    }

    let p_i = super::classify_2d::sample_interior_point(&poly_i);
    let p_j = super::classify_2d::sample_interior_point(&poly_j);

    let all_inside_strict =
        |verts: &[brepkit_math::vec::Point2], poly: &[brepkit_math::vec::Point2]| -> bool {
            verts
                .iter()
                .all(|&v| super::classify_2d::point_in_polygon_2d(v, poly))
        };
    let all_inside_tol =
        |verts: &[brepkit_math::vec::Point2], poly: &[brepkit_math::vec::Point2]| -> bool {
            let boundary_eps = super::classify_2d::boundary_eps(poly);
            verts.iter().all(|&v| {
                super::classify_2d::point_in_polygon_2d(v, poly)
                    || super::classify_2d::distance_to_polygon_boundary(v, poly) <= boundary_eps
            })
        };

    let ip_i_in_j = super::classify_2d::point_in_polygon_2d(p_i, &poly_j);
    let ip_j_in_i = super::classify_2d::point_in_polygon_2d(p_j, &poly_i);
    let outlines_coincide = ip_i_in_j && ip_j_in_i;
    let all_inside =
        |verts: &[brepkit_math::vec::Point2], poly: &[brepkit_math::vec::Point2]| -> bool {
            all_inside_strict(verts, poly) || (outlines_coincide && all_inside_tol(verts, poly))
        };

    // i contained in j, or j contained in i — the eighth-in-quarter case.
    if ip_i_in_j && all_inside(&poly_i, &poly_j) {
        return true;
    }
    if ip_j_in_i && all_inside(&poly_j, &poly_i) {
        return true;
    }

    // Partial overlap by intersection area (mirrors the planar path). Faces
    // that merely tile side-by-side share only a boundary segment (zero
    // intersection area) and are not paired; a genuine shared band covers a
    // meaningful fraction of the smaller patch.
    let inter = brepkit_math::polygon_boolean::polygon_boolean(
        &poly_i,
        &poly_j,
        brepkit_math::polygon_boolean::BooleanOp::Intersection,
        tol.linear,
    );
    let overlap_area = inter.area().abs();
    let area_i = super::classify_2d::signed_area_2d(&poly_i).abs();
    let area_j = super::classify_2d::signed_area_2d(&poly_j).abs();
    let smaller = area_i.min(area_j);
    smaller > tol.linear_sq() && overlap_area > smaller * 0.5
}

/// Approximate `(arc-length, axial)` parameter-space area of a cylinder/cone
/// sub-face's outer wire.
///
/// Returns `None` for non-(cylinder/cone) faces or wires that sample to fewer
/// than three points. Used only to order the two faces of a coaxial
/// geometric-overlap SD pair (see [`SameDomainPair::representative`]); both
/// faces share the surface so the same arc-length scaling applies to each and
/// their relative order is stable. The absolute area is never compared against
/// a tolerance.
fn analytic_face_param_area(topo: &Topology, face_id: FaceId) -> Option<f64> {
    let face = topo.face(face_id).ok()?;
    let surface = face.surface().clone();
    let pts = wire_points_3d(topo, face_id)?;
    let (samples, radius) = project_points_through_surface(&surface, &pts)?;
    let unwrapped = unwrap_angles(&samples);
    let poly: Vec<_> = unwrapped
        .iter()
        .map(|&(u, axial)| brepkit_math::vec::Point2::new(u * radius, axial))
        .collect();
    if poly.len() < 3 {
        return None;
    }
    Some(super::classify_2d::signed_area_2d(&poly).abs())
}

/// Outer-wire area used to pick the larger face of a geometric-overlap SD pair,
/// dispatched by surface type: planar area in the face plane, or
/// (arc-length, axial) parameter-space area for cylinder/cone faces.
///
/// Areas are only ever compared between the two faces of one pair, which share
/// a surface, so the (possibly different) projection per surface type is never
/// compared across surfaces.
fn repr_face_area(topo: &Topology, face_id: FaceId) -> Option<f64> {
    let face = topo.face(face_id).ok()?;
    match face.surface() {
        FaceSurface::Plane { .. } => planar_face_area(topo, face_id),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => analytic_face_param_area(topo, face_id),
        _ => None,
    }
}

/// Outer-wire AABB of a sub-face, sampled at [`SD_EDGE_SAMPLES`] points per
/// edge to match the polygons the overlap tests build. Returns `None` when the
/// wire has no usable points.
///
/// Used purely as a broad-phase reject for the geometric-overlap passes: both
/// [`planar_faces_overlap`] and [`analytic_faces_overlap`] can only return
/// `true` when the two faces share real 3D area, which requires their AABBs
/// (expanded by tolerance for boundary-coincident cases) to intersect.
fn face_outer_aabb(topo: &Topology, face_id: FaceId) -> Option<brepkit_math::aabb::Aabb3> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut pts: Vec<brepkit_math::vec::Point3> = Vec::new();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            continue;
        };
        let (sp, ep) = (sv.point(), ev.point());
        // Sample `0..SD_EDGE_SAMPLES` (not `..=`) to match the
        // `planar_faces_overlap` / `planar_face_area` polygons exactly: the next
        // edge's `frac=0` already covers each shared vertex, so this drops the
        // redundant per-vertex duplicate without changing the AABB.
        for k in 0..SD_EDGE_SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let frac = k as f64 / SD_EDGE_SAMPLES as f64;
            let frac = if oe.is_forward() { frac } else { 1.0 - frac };
            pts.push(super::pcurve_compute::evaluate_edge_at_t(
                edge.curve(),
                sp,
                ep,
                frac,
            ));
        }
    }
    brepkit_math::aabb::Aabb3::try_from_points(pts)
}

/// Generate the spatially-overlapping candidate pairs among `indices` using a
/// uniform grid over the faces' (tolerance-expanded) AABBs.
///
/// Each face is inserted into every grid cell its expanded AABB touches; any
/// two faces that ever land in the same cell become a candidate pair (emitted
/// once, with `i < j` in original-index order, deduplicated). Faces whose AABBs
/// never share a cell cannot overlap, so they are never tested — turning the
/// former all-pairs O(n²) scan into O(n + candidate pairs). The pair set is a
/// superset of the truly-overlapping pairs; the caller still runs the exact
/// `*_faces_overlap` test on each.
fn overlap_candidate_pairs(
    aabbs: &[(usize, brepkit_math::aabb::Aabb3)],
    margin: f64,
) -> Vec<(usize, usize)> {
    if aabbs.len() < 2 {
        return Vec::new();
    }
    // Cell size: the cube root of the per-face AABB volume budget across the
    // populated region, never below the average face extent, so a face spans a
    // bounded number of cells.
    let mut union = aabbs[0].1;
    let mut ext_sum = 0.0_f64;
    for &(_, bb) in aabbs {
        union = union.union(bb);
        let e = bb.max - bb.min;
        ext_sum += e.x().abs().max(e.y().abs()).max(e.z().abs());
    }
    #[allow(clippy::cast_precision_loss)]
    let avg_ext = ext_sum / aabbs.len() as f64;
    let span = {
        let e = union.max - union.min;
        e.x().abs().max(e.y().abs()).max(e.z().abs())
    };
    #[allow(clippy::cast_precision_loss)]
    let n = aabbs.len() as f64;
    let cell = avg_ext
        .max(span / n.cbrt().max(1.0))
        .max(margin)
        .max(f64::MIN_POSITIVE);
    let inv = 1.0 / cell;
    let cell_of = |c: f64| (c * inv).floor() as i64;

    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut pairs: HashSet<(usize, usize)> = HashSet::new();
    // A face whose expanded AABB spans more cells than this would cost O(cells)
    // to grid (a broad wall patch among many tiny facets); above it, AABB-test
    // the face against all faces directly instead — mirrors PointGrid's guard.
    let cell_budget = i64::try_from(aabbs.len())
        .unwrap_or(i64::MAX)
        .saturating_mul(4)
        .max(64);
    for &(idx, bb) in aabbs {
        let e = bb.expanded(margin);
        let (lo, hi) = (e.min, e.max);
        let (cx0, cx1) = (cell_of(lo.x()), cell_of(hi.x()));
        let (cy0, cy1) = (cell_of(lo.y()), cell_of(hi.y()));
        let (cz0, cz1) = (cell_of(lo.z()), cell_of(hi.z()));
        let cells = cx1
            .saturating_sub(cx0)
            .saturating_add(1)
            .saturating_mul(cy1.saturating_sub(cy0).saturating_add(1))
            .saturating_mul(cz1.saturating_sub(cz0).saturating_add(1));
        if cells > cell_budget {
            // Candidate set stays a superset of true overlaps (the narrow phase
            // filters and pairs are sorted before union-find), so the same-domain
            // result is unchanged — only the cost path differs.
            for &(jdx, jbb) in aabbs {
                if jdx != idx && e.intersects(jbb.expanded(margin)) {
                    let (a, b) = if jdx < idx { (jdx, idx) } else { (idx, jdx) };
                    pairs.insert((a, b));
                }
            }
            continue;
        }
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                for cz in cz0..=cz1 {
                    let bucket = buckets.entry((cx, cy, cz)).or_default();
                    for &other in bucket.iter() {
                        let (a, b) = if other < idx {
                            (other, idx)
                        } else {
                            (idx, other)
                        };
                        if a != b {
                            pairs.insert((a, b));
                        }
                    }
                    bucket.push(idx);
                }
            }
        }
    }
    let mut out: Vec<(usize, usize)> = pairs.into_iter().collect();
    // Deterministic order so union-find sequencing (and thus representative
    // selection) is reproducible regardless of HashSet iteration order.
    out.sort_unstable();
    out
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

/// Whether a surface is planar (the planar SD passes have their own geometric
/// containment tests, so the curved-region guard only applies to non-planes).
fn planar(surf: &FaceSurface) -> bool {
    matches!(surf, FaceSurface::Plane { .. })
}

/// Whether two curved sub-faces of the same underlying surface, paired by a
/// shared outer-wire edge set, actually cover DIFFERENT regions of that
/// surface (so they are glued neighbours, not coincident duplicates).
///
/// The discriminator is their precomputed interior sample: a genuine
/// same-domain duplicate is coincident (identical region → coincident
/// interior), whereas the two hemisphere bands of a bored sphere share the
/// equator boundary yet have interiors on opposite halves. Returns `false`
/// (defer to the edge-set union) when either interior is unavailable, keeping
/// the conservative pre-existing behaviour.
fn distinct_curved_regions(sub_faces: &[SubFace], i: usize, j: usize, tol: Tolerance) -> bool {
    match (sub_faces[i].interior_point, sub_faces[j].interior_point) {
        (Some(pi), Some(pj)) => (pi - pj).length() > tol.linear * 100.0,
        _ => false,
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
