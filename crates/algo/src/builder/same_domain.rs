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
) -> Vec<SameDomainPair> {
    let n = sub_faces.len();
    if n < 2 {
        return Vec::new();
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

        // Check all pairs within this edge-set group
        for (mi, &i) in members.iter().enumerate() {
            let rank_i = sub_faces[i].rank;
            let Some(surf_i) = surfaces[i] else {
                continue;
            };

            for &j in &members[mi + 1..] {
                // Only pair faces from opposing solids
                if sub_faces[j].rank == rank_i {
                    continue;
                }
                let Some(surf_j) = surfaces[j] else {
                    continue;
                };

                // Verify surface equivalence
                if let Some(same_dir) = surfaces_same_domain(surf_i, surf_j, tol) {
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

    for members in sd_groups.values() {
        if members.len() < 2 {
            continue;
        }

        // Find the best representative from Rank::A (smallest index)
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

        // Build a pair for each A-B combination in the group.
        // For simple cases (1 from A, 1 from B), this produces exactly 1 pair.
        if let (Some(idx_a), Some(idx_b)) = (repr_a, repr_b) {
            let key = (idx_a.min(idx_b), idx_a.max(idx_b));
            let same_orientation = pair_data.get(&key).copied().unwrap_or(true);

            pairs.push(SameDomainPair {
                idx_a,
                idx_b,
                same_orientation,
                // Edge-set matched faces have identical boundaries.
                // b_contained_in_a=false → touching (default for same-boundary faces).
                b_contained_in_a: false,
            });
        }
    }

    log::debug!(
        "detect_same_domain: {} same-domain pairs found (edge-set hash)",
        pairs.len()
    );
    pairs
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
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;

    let mut pairs: Vec<(QVert, QVert)> = Vec::with_capacity(wire.edges().len());

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;

        // Resolve vertices through VV-phase same-domain mapping
        let sv = arena.resolve_vertex(edge.start());
        let ev = arena.resolve_vertex(edge.end());

        let sp = topo.vertex(sv).ok()?.point();
        let ep = topo.vertex(ev).ok()?.point();

        // Quantize to tolerance grid
        let qs = quantize_point(sp, scale);
        let qe = quantize_point(ep, scale);

        // Canonical ordering: smaller first
        let pair = if qs <= qe { (qs, qe) } else { (qe, qs) };
        pairs.push(pair);
    }

    pairs.sort_unstable();
    Some(pairs)
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
fn surfaces_same_domain(a: &FaceSurface, b: &FaceSurface, tol: Tolerance) -> Option<bool> {
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
        _ => None,
    }
}
