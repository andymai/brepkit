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
    /// `true` if normals point the same direction, `false` if opposite.
    pub same_orientation: bool,
    /// `true` if B's face is fully contained within A's boundary.
    /// For edge-set matched faces, both faces have identical boundaries,
    /// so this is always `false` (touching, not contained).
    pub b_contained_in_a: bool,
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

    // Step 1: Compute canonical edge sets for each sub-face.
    // Use quantized vertex positions (not VertexId) so that VV-merged
    // vertices from different solids that share the same position produce
    // matching edge sets.
    let scale = 1.0 / tol.linear;

    let edge_sets: Vec<Option<EdgeSet>> = sub_faces
        .iter()
        .map(|sf| compute_edge_set_quantized(topo, arena, sf.face_id, scale))
        .collect();

    // Step 2: Group sub-faces by edge-set hash.
    // Key = edge set, Value = list of sub-face indices with that set.
    let mut groups: HashMap<EdgeSet, Vec<usize>> = HashMap::new();
    for (idx, edge_set) in edge_sets.iter().enumerate() {
        if let Some(es) = edge_set {
            if !es.is_empty() {
                groups.entry(es.clone()).or_default().push(idx);
            }
        }
    }

    // Step 3: For each group with 2+ faces from opposing solids,
    // verify surface equivalence and build SD pairs.
    let surfaces: Vec<Option<&FaceSurface>> = sub_faces
        .iter()
        .map(|sf| {
            topo.face(sf.face_id)
                .ok()
                .map(brepkit_topology::face::Face::surface)
        })
        .collect();

    let mut uf = UnionFind::new(n);
    let mut pair_data: HashMap<(usize, usize), bool> = HashMap::new(); // (min,max) → same_orientation

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

                // Verify surface equivalence.
                if let Some(same_dir) = surfaces_same_domain(surf_i, surf_j, tol) {
                    uf.union(i, j);
                    let key = (i.min(j), i.max(j));
                    pair_data.insert(key, same_dir);
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
            if sub_faces[i].interior_point.is_none() {
                continue;
            }
            for &j in &planar_indices[mi + 1..] {
                if sub_faces[j].interior_point.is_none() {
                    continue;
                }
                // Cheap surface-match guard first.
                let same_dir = match (surfaces[i], surfaces[j]) {
                    (Some(si), Some(sj)) => surfaces_same_domain(si, sj, tol),
                    _ => None,
                };
                let Some(same_dir) = same_dir else { continue };
                if uf.find(i) == uf.find(j) {
                    continue; // already grouped
                }
                if planar_faces_overlap(topo, sub_faces, i, j) {
                    uf.union(i, j);
                    let key = (i.min(j), i.max(j));
                    pair_data.insert(key, same_dir);
                }
            }
        }
    }

    // Step 4: Build SD pairs from union-find groups.
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

    for members in sd_groups.values() {
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

        match (repr_a, repr_b) {
            // Cross-rank: classic SD pair — emit for operation-specific selection.
            (Some(idx_a), Some(idx_b)) => {
                let key = (idx_a.min(idx_b), idx_a.max(idx_b));
                let same_orientation = pair_data.get(&key).copied().unwrap_or(true);

                pairs.push(SameDomainPair {
                    idx_a,
                    idx_b,
                    same_orientation,
                    b_contained_in_a: false,
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
fn planar_faces_overlap(topo: &Topology, sub_faces: &[SubFace], i: usize, j: usize) -> bool {
    let Some(p_i) = sub_faces[i].interior_point else {
        return false;
    };
    let Some(p_j) = sub_faces[j].interior_point else {
        return false;
    };
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

    let wire_points = |face: &brepkit_topology::face::Face| -> Vec<brepkit_math::vec::Point3> {
        let mut pts = Vec::new();
        let Ok(wire) = topo.wire(face.outer_wire()) else {
            return pts;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            if let Ok(v) = topo.vertex(vid) {
                pts.push(v.point());
            }
        }
        pts
    };

    let pts_i = wire_points(face_i);
    let pts_j = wire_points(face_j);
    if pts_i.len() < 3 || pts_j.len() < 3 {
        return false;
    }
    let frame = super::plane_frame::PlaneFrame::from_plane_face(normal_i, &pts_i);
    let poly_i: Vec<_> = pts_i.iter().map(|&p| frame.project(p)).collect();
    let poly_j: Vec<_> = pts_j.iter().map(|&p| frame.project(p)).collect();

    // Containment test using a small inward tolerance on the boundary —
    // point_in_polygon_2d's strict ray-cast treats on-boundary points
    // inconsistently, so we also accept "close to boundary" as inside.
    let p_i_2d = frame.project(p_i);
    let p_j_2d = frame.project(p_j);

    let all_inside =
        |verts: &[brepkit_math::vec::Point2], poly: &[brepkit_math::vec::Point2]| -> bool {
            verts
                .iter()
                .all(|&v| super::classify_2d::point_in_polygon_2d(v, poly))
        };

    // i fully contained in j: every vertex of i (plus its interior sample)
    // is inside j's polygon.
    if super::classify_2d::point_in_polygon_2d(p_i_2d, &poly_j) && all_inside(&poly_i, &poly_j) {
        return true;
    }
    // j fully contained in i.
    if super::classify_2d::point_in_polygon_2d(p_j_2d, &poly_i) && all_inside(&poly_j, &poly_i) {
        return true;
    }
    false
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
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};

    #[test]
    fn planes_same_domain_same_direction() {
        let tol = Tolerance::new();
        let a = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 5.0,
        };
        let b = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 5.0,
        };
        assert_eq!(surfaces_same_domain(&a, &b, tol), Some(true));
    }

    #[test]
    fn planes_same_domain_opposite_direction() {
        let tol = Tolerance::new();
        let a = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 5.0,
        };
        let b = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -5.0,
        };
        assert_eq!(surfaces_same_domain(&a, &b, tol), Some(false));
    }

    #[test]
    fn planes_different_distance_not_same_domain() {
        let tol = Tolerance::new();
        let a = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 5.0,
        };
        let b = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 10.0,
        };
        assert_eq!(surfaces_same_domain(&a, &b, tol), None);
    }

    #[test]
    fn mixed_surface_types_not_same_domain() {
        let tol = Tolerance::new();
        let a = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let b = FaceSurface::Sphere(
            brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0)
                .expect("valid sphere"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), None);
    }

    #[test]
    fn cones_same_domain_same_direction() {
        let tol = Tolerance::new();
        let a = FaceSurface::Cone(
            brepkit_math::surfaces::ConicalSurface::with_ref_dir(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                std::f64::consts::FRAC_PI_6,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .expect("valid cone"),
        );
        let b = FaceSurface::Cone(
            brepkit_math::surfaces::ConicalSurface::with_ref_dir(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                std::f64::consts::FRAC_PI_6,
                Vec3::new(0.0, 1.0, 0.0),
            )
            .expect("valid cone"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), Some(true));
    }

    #[test]
    fn cones_different_half_angle_not_same_domain() {
        let tol = Tolerance::new();
        let a = FaceSurface::Cone(
            brepkit_math::surfaces::ConicalSurface::with_ref_dir(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                std::f64::consts::FRAC_PI_6,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .expect("valid cone"),
        );
        let b = FaceSurface::Cone(
            brepkit_math::surfaces::ConicalSurface::with_ref_dir(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                std::f64::consts::FRAC_PI_4,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .expect("valid cone"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), None);
    }

    #[test]
    fn torus_same_domain_same_direction_ignores_ref_dir() {
        let tol = Tolerance::new();
        let a = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        // Same surface, but constructed with a different ref direction —
        // x_axis/y_axis differ but z_axis matches, so this is the same surface.
        let b = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis_and_ref_dir(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 0.0),
            )
            .expect("valid torus"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), Some(true));
    }

    #[test]
    fn torus_same_domain_opposite_direction() {
        let tol = Tolerance::new();
        let a = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(1.0, 2.0, 3.0),
                5.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        let b = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(1.0, 2.0, 3.0),
                5.0,
                1.0,
                Vec3::new(0.0, 0.0, -1.0),
            )
            .expect("valid torus"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), Some(false));
    }

    #[test]
    fn torus_different_major_radius_not_same_domain() {
        let tol = Tolerance::new();
        let a = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        let b = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                4.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), None);
    }

    #[test]
    fn torus_different_minor_radius_not_same_domain() {
        let tol = Tolerance::new();
        let a = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        let b = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                0.5,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), None);
    }

    #[test]
    fn torus_different_center_not_same_domain() {
        let tol = Tolerance::new();
        let a = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        let b = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(1.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), None);
    }

    #[test]
    fn torus_skew_axes_not_same_domain() {
        let tol = Tolerance::new();
        let a = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(0.0, 0.0, 1.0),
            )
            .expect("valid torus"),
        );
        let b = FaceSurface::Torus(
            brepkit_math::surfaces::ToroidalSurface::with_axis(
                Point3::new(0.0, 0.0, 0.0),
                3.0,
                1.0,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .expect("valid torus"),
        );
        assert_eq!(surfaces_same_domain(&a, &b, tol), None);
    }

    #[test]
    fn quantize_point_deterministic() {
        let scale = 1.0 / 1e-7; // default tolerance
        let p = Point3::new(1.0, 2.0, 3.0);
        let q1 = quantize_point(p, scale);
        let q2 = quantize_point(p, scale);
        assert_eq!(q1, q2, "quantization should be deterministic");
    }

    #[test]
    fn quantize_nearby_points_collapse() {
        let tol = Tolerance::new();
        let scale = 1.0 / tol.linear;
        let p1 = Point3::new(1.0, 2.0, 3.0);
        let p2 = Point3::new(1.0 + tol.linear * 0.4, 2.0, 3.0);
        let q1 = quantize_point(p1, scale);
        let q2 = quantize_point(p2, scale);
        assert_eq!(q1, q2, "nearby points should collapse to same grid cell");
    }

    #[test]
    fn quantize_distant_points_differ() {
        let tol = Tolerance::new();
        let scale = 1.0 / tol.linear;
        let p1 = Point3::new(1.0, 2.0, 3.0);
        let p2 = Point3::new(1.0 + tol.linear * 2.0, 2.0, 3.0);
        let q1 = quantize_point(p1, scale);
        let q2 = quantize_point(p2, scale);
        assert_ne!(
            q1, q2,
            "points separated by 2x tolerance should be in different cells"
        );
    }

    #[test]
    fn union_find_basic_groups() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));
        // Transitive closure
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(3));
    }
}
