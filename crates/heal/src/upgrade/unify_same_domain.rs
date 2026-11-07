//! Unify same-domain faces — merge adjacent faces sharing the same
//! underlying surface.
//!
//! This is the most impactful healing operation in production.  After
//! boolean operations, a box may have 72 faces instead of 6 because
//! intersection curves split each original face.  `unify_same_domain`
//! detects adjacent faces on the same plane/cylinder/cone/sphere/torus
//! and merges them back, dramatically reducing face count.

use std::collections::{HashMap, HashSet};

use brepkit_math::frame::Frame3;
use brepkit_math::region2d::{Boundary2, point_in_region};
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::vec::{Point2, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::HealError;
use crate::analysis::surface::surfaces_equivalent;
use crate::status::Status;

/// Options for the unify-same-domain operation.
#[derive(Debug, Clone)]
pub struct UnifyOptions {
    /// Merge co-surface adjacent faces.
    pub unify_faces: bool,
    /// Merge collinear adjacent edges after face merge.
    pub unify_edges: bool,
    /// Linear tolerance for "same position" checks.
    pub linear_tolerance: f64,
    /// Angular tolerance for "same direction" checks.
    pub angular_tolerance: f64,
}

impl Default for UnifyOptions {
    fn default() -> Self {
        Self {
            unify_faces: true,
            unify_edges: true,
            linear_tolerance: 1e-7,
            angular_tolerance: 1e-12,
        }
    }
}

/// Result of the unify operation.
#[derive(Debug, Clone)]
pub struct UnifyResult {
    /// Number of faces that were merged away.
    pub faces_merged: usize,
    /// Number of edges that were merged.
    pub edges_merged: usize,
    /// Status flags.
    pub status: Status,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

/// Merge adjacent faces that share the same underlying surface.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail during merging.
#[allow(clippy::too_many_lines)]
pub fn unify_same_domain(
    topo: &mut Topology,
    solid_id: SolidId,
    options: &UnifyOptions,
) -> Result<(SolidId, UnifyResult), HealError> {
    if !options.unify_faces {
        return Ok((
            solid_id,
            UnifyResult {
                faces_merged: 0,
                edges_merged: 0,
                status: Status::OK,
            },
        ));
    }

    let solid_data = topo.solid(solid_id)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();
    let n_faces = face_ids.len();

    if n_faces < 2 {
        return Ok((
            solid_id,
            UnifyResult {
                faces_merged: 0,
                edges_merged: 0,
                status: Status::OK,
            },
        ));
    }

    let mut edge_faces: HashMap<usize, Vec<usize>> = HashMap::new();
    for (fi, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_faces.entry(oe.edge().index()).or_default().push(fi);
        }
        for &iw_id in face.inner_wires() {
            let iw = topo.wire(iw_id)?;
            for oe in iw.edges() {
                edge_faces.entry(oe.edge().index()).or_default().push(fi);
            }
        }
    }

    let face_surfaces: Vec<_> = face_ids
        .iter()
        .map(|&fid| topo.face(fid).map(|f| f.surface().clone()))
        .collect::<Result<Vec<_>, _>>()?;

    let mut uf = UnionFind::new(n_faces);
    let tol = brepkit_math::tolerance::Tolerance {
        linear: options.linear_tolerance,
        angular: options.angular_tolerance,
        ..brepkit_math::tolerance::Tolerance::new()
    };

    for faces in edge_faces.values() {
        if faces.len() == 2 {
            let fi = faces[0];
            let fj = faces[1];
            if fi != fj && surfaces_equivalent(&face_surfaces[fi], &face_surfaces[fj], &tol) {
                uf.union(fi, fj);
            }
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n_faces {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    // `into_values()` yields groups in seed-dependent HashMap order. The merge
    // order cascades through face creation/removal and vertex welding, so an
    // unstable order makes sequential booleans non-deterministic across
    // processes. Each group's first element is its minimum face index (members
    // are pushed in 0..n order), giving a stable key to sort on.
    let mut merge_groups: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
    merge_groups.sort_unstable_by_key(|g| g[0]);

    if merge_groups.is_empty() {
        return Ok((
            solid_id,
            UnifyResult {
                faces_merged: 0,
                edges_merged: 0,
                status: Status::OK,
            },
        ));
    }

    let mut faces_to_remove: HashSet<FaceId> = HashSet::new();
    let mut new_faces_to_add: Vec<FaceId> = Vec::new();
    let mut total_merged = 0;

    for group in &merge_groups {
        let group_face_ids: Vec<FaceId> = group.iter().map(|&i| face_ids[i]).collect();

        let has_holes = group_face_ids
            .iter()
            .any(|&fid| topo.face(fid).is_ok_and(|f| !f.inner_wires().is_empty()));

        let surface = face_surfaces[group[0]].clone();

        let merged = if has_holes || surface.is_planar() {
            // The reassembly path keeps holes attached and correctly orders the
            // surviving loops; it is the only path that can merge groups with
            // inner wires. Periodic surfaces with holes are deferred (a hole
            // touching a seam cannot be classified in UV yet), except a sphere's
            // two halves meeting on their whole outer wires, which have no seam
            // left to touch.
            if has_holes && !surface.is_planar() {
                if let Some(whole) = merge_whole_sphere(topo, &group_face_ids, &surface)? {
                    for &fid in &group_face_ids {
                        faces_to_remove.insert(fid);
                    }
                    total_merged += group_face_ids.len() - 1;
                    new_faces_to_add.push(whole);
                    continue;
                }
                log::warn!(
                    "unify_same_domain: skipping non-planar group with {} faces (holes on periodic surface)",
                    group.len()
                );
                continue;
            }
            merge_group_with_holes(topo, &group_face_ids, &surface, options)?
        } else {
            merge_group_simple(topo, &group_face_ids, &surface)?
        };

        let Some(new_face_ids) = merged else {
            continue;
        };

        for &fid in &group_face_ids {
            faces_to_remove.insert(fid);
        }
        total_merged += group_face_ids.len() - new_face_ids.len();
        new_faces_to_add.extend(new_face_ids);
    }

    // Merging is an OPTIMISATION — it must never leave the shell worse connected
    // than it found it. Each phase is applied, measured, and rolled back on its
    // own so a bad phase does not discard a good one.
    //
    // Both phases can orphan edges: a group whose reassembly loses a boundary
    // edge, and `merge_collinear_edges`, which rewrites only the NEW faces'
    // wires — a neighbour still referencing the pre-merge edges is left
    // unpaired. That is invisible on analytic solids (few collinear runs) and
    // catastrophic on mesh-fallback ones (thousands of coplanar facets: 1630
    // edge merges orphaned 4574 edges on a drilled 15648-face solid).
    let mut bad_after_faces = 0;
    if total_merged > 0 {
        let shell = topo.shell(shell_id)?;
        let current_faces: Vec<FaceId> = shell.faces().to_vec();
        let bad_before = unpaired_edge_count(topo, &current_faces)?;

        let mut final_faces: Vec<FaceId> = current_faces
            .iter()
            .copied()
            .filter(|f| !faces_to_remove.contains(f))
            .collect();
        final_faces.extend(&new_faces_to_add);

        // Removed faces are only dropped from the shell's list, never deleted
        // from the arena, so restoring that list is a complete revert.
        bad_after_faces = unpaired_edge_count(topo, &final_faces)?;
        if bad_after_faces > bad_before {
            log::warn!(
                "unify_same_domain: skipping merge of {total_merged} faces — unpaired edges would rise {bad_before} -> {bad_after_faces}"
            );
            return Ok((
                solid_id,
                UnifyResult {
                    faces_merged: 0,
                    edges_merged: 0,
                    status: Status::OK,
                },
            ));
        }
        let new_shell = Shell::new(final_faces)?;
        let shell_mut = topo.shell_mut(shell_id)?;
        *shell_mut = new_shell;
    }

    let mut total_edges_merged = 0;
    if options.unify_edges && total_merged > 0 {
        // `merge_collinear_edges` rewrites each wire in place under the same
        // WireId, so snapshotting the wires is enough to undo just this phase
        // and keep the face merges above.
        let mut wire_snapshots: Vec<(WireId, Wire)> = Vec::with_capacity(new_faces_to_add.len());
        for &fid in &new_faces_to_add {
            let wid = topo.face(fid)?.outer_wire();
            wire_snapshots.push((wid, topo.wire(wid)?.clone()));
        }

        for &(wid, _) in &wire_snapshots {
            total_edges_merged += merge_collinear_edges(topo, wid, options)?;
        }

        if total_edges_merged > 0 {
            let faces_now: Vec<FaceId> = topo.shell(shell_id)?.faces().to_vec();
            let bad_after_edges = unpaired_edge_count(topo, &faces_now)?;
            if bad_after_edges > bad_after_faces {
                log::warn!(
                    "unify_same_domain: reverting {total_edges_merged} collinear-edge merges — unpaired edges would rise {bad_after_faces} -> {bad_after_edges} (face merges kept)"
                );
                for (wid, original) in wire_snapshots {
                    let wire_mut = topo.wire_mut(wid)?;
                    *wire_mut = original;
                }
                total_edges_merged = 0;
            }
        }
    }

    let status = if total_merged > 0 || total_edges_merged > 0 {
        Status::DONE1
    } else {
        Status::OK
    };

    Ok((
        solid_id,
        UnifyResult {
            faces_merged: total_merged,
            edges_merged: total_edges_merged,
            status,
        },
    ))
}

/// Count edges of `faces` that are NOT shared by exactly two of them.
///
/// A closed manifold shell pairs every edge exactly twice; free (once) and
/// over-shared (3+) edges are both defects. Used to prove a merge did not make
/// the shell worse before committing it.
fn unpaired_edge_count(topo: &Topology, faces: &[FaceId]) -> Result<usize, HealError> {
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    Ok(uses.values().filter(|&&c| c != 2).count())
}

/// Two faces of a sphere that share edges of their outer wires, as a primitive's
/// hemispheres share their equator, as one face. Their other edges close into
/// loops, each with the face on its left as an outer wire has it, and these and
/// their holes are all the boundary the sphere keeps, so the first serves as the
/// outer wire and the rest as holes. The surface's axis turns to pass through
/// that loop, which then winds the axis with neither pole on it, the layout the
/// sphere mesher reads a band or a large cap in. `None` when the faces share no
/// edge, or their other edges do not close into separate loops.
fn merge_whole_sphere(
    topo: &mut Topology,
    group: &[FaceId],
    surface: &FaceSurface,
) -> Result<Option<FaceId>, HealError> {
    /// Samples per edge when reading where a loop's cap lies.
    const SAMPLES: u32 = 8;
    let FaceSurface::Sphere(sphere) = surface else {
        return Ok(None);
    };
    let &[first, second] = group else {
        return Ok(None);
    };
    let (a, b) = (topo.face(first)?, topo.face(second)?);
    let reversed = a.is_reversed();
    if reversed != b.is_reversed() {
        return Ok(None);
    }
    let mut holes: Vec<WireId> = a.inner_wires().to_vec();
    holes.extend_from_slice(b.inner_wires());
    let (outer_a, outer_b) = (
        topo.wire(a.outer_wire())?.edges(),
        topo.wire(b.outer_wire())?.edges(),
    );
    // Each face runs a shared edge once and the other way, and the edge goes.
    // An edge a wire runs twice is a seam, which this merge does not take.
    let senses = |edges: &[OrientedEdge]| -> Option<HashMap<EdgeId, bool>> {
        let mut seen = HashMap::new();
        edges
            .iter()
            .all(|oe| seen.insert(oe.edge(), oe.is_forward()).is_none())
            .then_some(seen)
    };
    let (Some(senses_a), Some(senses_b)) = (senses(outer_a), senses(outer_b)) else {
        return Ok(None);
    };
    let shared: HashSet<EdgeId> = senses_a
        .keys()
        .copied()
        .filter(|edge| senses_b.contains_key(edge))
        .collect();
    if shared.is_empty() || shared.iter().any(|edge| senses_a[edge] == senses_b[edge]) {
        return Ok(None);
    }
    let kept: Vec<OrientedEdge> = outer_a
        .iter()
        .chain(outer_b)
        .copied()
        .filter(|oe| !shared.contains(&oe.edge()))
        .collect();

    // Chain them, a vertex two loops pass through being no separate loop.
    let mut leaving: HashMap<VertexId, usize> = HashMap::new();
    for (i, oe) in kept.iter().enumerate() {
        let start = oe.oriented_start(topo.edge(oe.edge())?);
        if leaving.insert(start, i).is_some() {
            return Ok(None);
        }
    }
    let mut walked = vec![false; kept.len()];
    let mut loops: Vec<Vec<OrientedEdge>> = Vec::new();
    for from in 0..kept.len() {
        if walked[from] {
            continue;
        }
        let mut chain = Vec::new();
        let mut at = from;
        while !walked[at] {
            walked[at] = true;
            chain.push(kept[at]);
            let end = kept[at].oriented_end(topo.edge(kept[at].edge())?);
            let Some(&next) = leaving.get(&end) else {
                return Ok(None);
            };
            at = next;
        }
        if at != from {
            return Ok(None);
        }
        loops.push(chain);
    }
    for chain in loops {
        holes.insert(0, topo.add_wire(Wire::new(chain, true)?));
    }
    let Some((&outer_wire, rest)) = holes.split_first() else {
        return Ok(None);
    };
    let rest = rest.to_vec();

    // The loop's cap lies along the mean of its samples from the centre, in the
    // order the wire runs them.
    let center = sphere.center();
    let mut samples: Vec<Vec3> = Vec::new();
    for oe in topo.wire(outer_wire)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (p, q) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(p, q);
        for k in 0..SAMPLES {
            let fraction = f64::from(k) / f64::from(SAMPLES);
            let along = if oe.is_forward() {
                fraction
            } else {
                1.0 - fraction
            };
            let t = (t1 - t0).mul_add(along, t0);
            samples.push(edge.curve().evaluate_with_endpoints(t, p, q) - center);
        }
    }
    let toward = samples
        .iter()
        .fold(Vec3::new(0.0, 0.0, 0.0), |sum, &s| sum + s);
    let Ok(axis) = toward.normalize() else {
        return Ok(None);
    };
    // The mesher wants the loop to wind this axis once. The mean of a concave
    // or unevenly split loop can lie outside its cap, and then it does not.
    let turned_by = |a: Vec3, b: Vec3| {
        a.cross(b)
            .dot(axis)
            .atan2(a.dot(b) - a.dot(axis) * b.dot(axis))
    };
    let winding: f64 = (0..samples.len())
        .map(|i| turned_by(samples[i], samples[(i + 1) % samples.len()]))
        .sum();
    if (winding.abs() - std::f64::consts::TAU).abs() > 1e-6 {
        return Ok(None);
    }
    let turned = FaceSurface::Sphere(SphericalSurface::with_axis(center, sphere.radius(), axis)?);
    let face = if reversed {
        Face::new_reversed(outer_wire, rest, turned)
    } else {
        Face::new(outer_wire, rest, turned)
    };
    Ok(Some(topo.add_face(face)))
}

/// Merge a hole-free group on a curved (non-planar) surface.
///
/// Surviving edges (referenced exactly once across the group) are emitted as a
/// single unordered closed wire — adequate for analytic surfaces whose merged
/// boundary is one loop. Returns `None` if the merge cannot be formed.
fn merge_group_simple(
    topo: &mut Topology,
    group_face_ids: &[FaceId],
    surface: &brepkit_topology::face::FaceSurface,
) -> Result<Option<Vec<FaceId>>, HealError> {
    let mut edge_count: HashMap<usize, usize> = HashMap::new();
    let mut all_oes: Vec<OrientedEdge> = Vec::new();

    for &fid in group_face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            *edge_count.entry(oe.edge().index()).or_insert(0) += 1;
            all_oes.push(*oe);
        }
    }

    let boundary_edges: Vec<OrientedEdge> = all_oes
        .into_iter()
        .filter(|oe| edge_count.get(&oe.edge().index()).copied().unwrap_or(0) == 1)
        .collect();

    if boundary_edges.is_empty() {
        return Ok(None);
    }
    // The faces' surviving edges come face by face; a wire must run them in
    // traversal order.
    let Some(boundary_edges) = chain_into_loop(topo, &boundary_edges)? else {
        log::debug!(
            "unify_same_domain: boundary of {} faces is not one loop",
            group_face_ids.len()
        );
        return Ok(None);
    };

    let Ok(merged_wire) = Wire::new(boundary_edges, true) else {
        log::warn!(
            "unify_same_domain: failed to build merged wire for {} faces",
            group_face_ids.len()
        );
        return Ok(None);
    };

    let new_wire_id = topo.add_wire(merged_wire);
    let reversed = topo.face(group_face_ids[0])?.is_reversed();
    let new_face = if reversed {
        brepkit_topology::face::Face::new_reversed(new_wire_id, Vec::new(), surface.clone())
    } else {
        brepkit_topology::face::Face::new(new_wire_id, Vec::new(), surface.clone())
    };
    Ok(Some(vec![topo.add_face(new_face)]))
}

/// `edges` reordered into one closed loop, each starting where the previous
/// ends, or `None` when they do not chain into exactly one loop.
fn chain_into_loop(
    topo: &Topology,
    edges: &[OrientedEdge],
) -> Result<Option<Vec<OrientedEdge>>, HealError> {
    let mut rest = Vec::with_capacity(edges.len());
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let (s, t) = if oe.is_forward() {
            (edge.start(), edge.end())
        } else {
            (edge.end(), edge.start())
        };
        rest.push((*oe, s, t));
    }
    let Some((first, start, mut at)) = rest.first().copied() else {
        return Ok(None);
    };
    rest.remove(0);
    let mut chained = vec![first];
    while !rest.is_empty() {
        let Some(i) = rest.iter().position(|&(_, s, _)| s == at) else {
            return Ok(None);
        };
        let (oe, _, t) = rest.remove(i);
        chained.push(oe);
        at = t;
    }
    Ok((at == start).then_some(chained))
}

/// A surviving edge with its traversal endpoints and its 2D directions of
/// travel as it leaves its start and as it arrives at its end, which differ
/// on a curve.
struct SurvEdge {
    oe: OrientedEdge,
    start: brepkit_topology::vertex::VertexId,
    end: brepkit_topology::vertex::VertexId,
    tangent_uv: (f64, f64),
    arrival_uv: (f64, f64),
}

/// An edge's outline in UV as it is traversed, its end left out: its start
/// alone for a line, its start and interior samples for a curve.
fn edge_uv_run(
    topo: &Topology,
    oe: &OrientedEdge,
    to_uv: &impl Fn(brepkit_math::vec::Point3) -> (f64, f64),
) -> Result<Vec<(f64, f64)>, HealError> {
    const SAMPLES: usize = 16;
    let edge = topo.edge(oe.edge())?;
    // The curve runs from the edge's own start; the traversal may not.
    let (es, ee) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    if matches!(edge.curve(), EdgeCurve::Line) {
        return Ok(vec![to_uv(if oe.is_forward() { es } else { ee })]);
    }
    let (t0, t1) = edge.curve().domain_with_endpoints(es, ee);
    Ok((0..SAMPLES)
        .map(|k| {
            #[allow(clippy::cast_precision_loss)]
            let f = k as f64 / SAMPLES as f64;
            let t = if oe.is_forward() {
                t0 + f * (t1 - t0)
            } else {
                t1 - f * (t1 - t0)
            };
            to_uv(edge.curve().evaluate_with_endpoints(t, es, ee))
        })
        .collect())
}

/// A reassembled closed loop with its signed area in UV.
struct LoopInfo {
    edge_indices: Vec<usize>,
    signed_area: f64,
}

/// Merge a planar group, preserving holes.
///
/// Implements the XOR / loop-reassembly / hole-routing algorithm: shared seams
/// (referenced exactly twice) become interior and vanish; every other edge
/// (including all hole edges, referenced once) survives. Surviving edges are
/// rewalked into closed loops, classified as outer/hole by UV containment, and
/// holes are routed to the outer region that contains them.
///
/// Returns `None` (group left unmerged) when the surviving edge set is
/// non-manifold, fails to close into loops, or contains a hole that classifies
/// into no outer region.
#[allow(clippy::too_many_lines)]
fn merge_group_with_holes(
    topo: &mut Topology,
    group_face_ids: &[FaceId],
    surface: &brepkit_topology::face::FaceSurface,
    options: &UnifyOptions,
) -> Result<Option<Vec<FaceId>>, HealError> {
    use brepkit_topology::vertex::VertexId;

    let brepkit_topology::face::FaceSurface::Plane { normal, d } = surface else {
        return Ok(None);
    };
    let origin = brepkit_math::vec::Point3::new(normal.x() * *d, normal.y() * *d, normal.z() * *d);
    let Ok(frame) = brepkit_math::frame::Frame3::from_normal(origin, *normal) else {
        return Ok(None);
    };

    let lin = options.linear_tolerance;

    // XOR edge selection across outer + inner wires of every member face,
    // preserving each surviving edge's orientation as it appeared.
    let mut edge_count: HashMap<usize, usize> = HashMap::new();
    let mut all_oes: Vec<OrientedEdge> = Vec::new();
    for &fid in group_face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            *edge_count.entry(oe.edge().index()).or_insert(0) += 1;
            all_oes.push(*oe);
        }
        for &iw_id in face.inner_wires() {
            let iw = topo.wire(iw_id)?;
            for oe in iw.edges() {
                *edge_count.entry(oe.edge().index()).or_insert(0) += 1;
                all_oes.push(*oe);
            }
        }
    }

    if edge_count.values().any(|&c| c > 2) {
        log::warn!(
            "unify_same_domain: skipping group with {} faces (non-manifold edge)",
            group_face_ids.len()
        );
        return Ok(None);
    }

    let surviving: Vec<OrientedEdge> = all_oes
        .into_iter()
        .filter(|oe| edge_count.get(&oe.edge().index()).copied().unwrap_or(0) == 1)
        .collect();

    if surviving.is_empty() {
        return Ok(None);
    }

    let to_uv = |p: brepkit_math::vec::Point3| -> (f64, f64) {
        let rel = p - origin;
        (rel.dot(frame.x), rel.dot(frame.y))
    };

    let mut edges: Vec<SurvEdge> = Vec::with_capacity(surviving.len());
    for oe in &surviving {
        let edge = topo.edge(oe.edge())?;
        let start = oe.oriented_start(edge);
        let end = oe.oriented_end(edge);
        let sp = topo.vertex(start)?.point();
        let ep = topo.vertex(end)?.point();
        // A full circle is stored as a CLOSED edge: start == end with a zero
        // chord, which a chord test cannot tell apart from a point; its
        // samples can. Such a loop is a real boundary, typically a bore
        // opening: it is kept as a loop of its own, outlined by its samples.
        let closed = start == end
            && edge_uv_run(topo, oe, &to_uv)?.iter().any(|&q| {
                let p = to_uv(sp);
                (q.0 - p.0).hypot(q.1 - p.1) > lin
            });
        if (ep - sp).length() < lin && start == end && !closed {
            continue;
        }
        let (es, ee) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (tangent_uv, arrival_uv) =
            travel_directions(edge.curve(), es, ee, oe.is_forward(), frame.x, frame.y);
        edges.push(SurvEdge {
            oe: *oe,
            start,
            end,
            tangent_uv,
            arrival_uv,
        });
    }

    if edges.is_empty() {
        return Ok(None);
    }

    // Walk surviving edges into closed loops. At a pinch vertex (multiple
    // unused candidates), pick the one turning most tightly to the left.
    let mut outgoing: HashMap<VertexId, Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        outgoing.entry(e.start).or_default().push(i);
    }

    let mut used = vec![false; edges.len()];
    let mut loops: Vec<Vec<usize>> = Vec::new();
    // A closed edge is a loop of its own, kept out of the walk.
    for (i, e) in edges.iter().enumerate() {
        if e.start == e.end {
            used[i] = true;
            loops.push(vec![i]);
        }
    }

    for seed in 0..edges.len() {
        if used[seed] {
            continue;
        }
        let mut loop_edges: Vec<usize> = Vec::new();
        let start_vertex = edges[seed].start;
        let mut current = seed;

        loop {
            used[current] = true;
            loop_edges.push(current);
            let next_vertex = edges[current].end;
            if next_vertex == start_vertex {
                break;
            }

            // Incoming direction reversed so the turn angle is measured between
            // the edges meeting at `next_vertex`.
            let in_dir = (-edges[current].arrival_uv.0, -edges[current].arrival_uv.1);
            let candidates = outgoing.get(&next_vertex);
            let mut best: Option<usize> = None;
            let mut best_angle = f64::NEG_INFINITY;
            if let Some(cands) = candidates {
                for &ci in cands {
                    if used[ci] {
                        continue;
                    }
                    let out = edges[ci].tangent_uv;
                    let angle = signed_turn(in_dir, out);
                    if angle > best_angle {
                        best_angle = angle;
                        best = Some(ci);
                    }
                }
            }

            let Some(next) = best else {
                // Dead end — surviving edges did not close into a loop.
                log::warn!(
                    "unify_same_domain: surviving edges failed to close for {} faces",
                    group_face_ids.len()
                );
                return Ok(None);
            };
            current = next;
        }

        loops.push(loop_edges);
    }

    // Build per-loop UV polygons and signed areas for classification.
    // Each loop's exact boundary (segments and conic arcs) and points on it
    // (the middle of each of its edges), for the containment tests below: a
    // sampled outline misreads a point between an arc and its chords, a
    // small hole close inside a curved outline among them.
    let mut pieces: Vec<Vec<Boundary2>> = Vec::with_capacity(loops.len());
    let mut probes: Vec<Vec<(f64, f64)>> = Vec::with_capacity(loops.len());
    for loop_edges in &loops {
        let mut loop_pieces = Vec::new();
        let mut loop_probes = Vec::new();
        for &i in loop_edges {
            let (p, q) = edge_pieces(topo, &edges[i].oe, &to_uv, &frame)?;
            loop_pieces.extend(p);
            loop_probes.push(q);
        }
        pieces.push(loop_pieces);
        probes.push(loop_probes);
    }
    // Whether loop `i` lies inside loop `j`, read at a point on `i` (loops
    // do not cross, so one of `i`'s points answers for all of it); `None`
    // when every probe lies on `j`'s boundary.
    let inside = |i: usize, j: usize| -> Option<bool> {
        probes[i]
            .iter()
            .find_map(|&(u, v)| point_in_region(&pieces[j], Point2::new(u, v), lin))
    };

    // Each loop's winding from its exact area: a sampled outline can wind a
    // thin curved loop (a sector a ten-thousandth wide) the wrong way round.
    let mut infos: Vec<LoopInfo> = Vec::with_capacity(loops.len());
    for loop_edges in loops {
        let oes: Vec<OrientedEdge> = loop_edges.iter().map(|&i| edges[i].oe).collect();
        let Some(signed_area) =
            brepkit_topology::planar::loop_area_2d(topo, &oes, frame.x, frame.y)?
        else {
            return Ok(None);
        };
        infos.push(LoopInfo {
            edge_indices: loop_edges,
            signed_area,
        });
    }

    // A loop is an outer region if no other loop contains it. Containment is by
    // point-in-polygon of a representative interior sample.
    let n_loops = infos.len();
    let mut is_outer = vec![true; n_loops];
    for i in 0..n_loops {
        for j in 0..n_loops {
            if i == j {
                continue;
            }
            match inside(i, j) {
                Some(true) => {
                    is_outer[i] = false;
                    break;
                }
                Some(false) => {}
                None => return Ok(None),
            }
        }
    }

    let outer_indices: Vec<usize> = (0..n_loops).filter(|&i| is_outer[i]).collect();
    let hole_indices: Vec<usize> = (0..n_loops).filter(|&i| !is_outer[i]).collect();

    if outer_indices.is_empty() {
        return Ok(None);
    }

    // Route each hole to the outer region whose polygon contains its sample.
    let mut region_holes: HashMap<usize, Vec<usize>> = HashMap::new();
    for &h in &hole_indices {
        let mut owner: Option<usize> = None;
        for &o in &outer_indices {
            match inside(h, o) {
                Some(true) => {
                    owner = Some(o);
                    break;
                }
                Some(false) => {}
                None => return Ok(None),
            }
        }
        if let Some(o) = owner {
            region_holes.entry(o).or_default().push(h);
        } else {
            log::warn!("unify_same_domain: hole could not be routed to an outer region");
            return Ok(None);
        }
    }

    let reversed = topo.face(group_face_ids[0])?.is_reversed();

    // The merged outer loops wind as the members' outer wires do, and holes the
    // other way; a reversed face may store its wire either way round its
    // surface's normal, so read the sense off the members rather than the flag.
    // Reverse a loop's edge run when its current sign disagrees.
    let mut widest = 0.0_f64;
    for &fid in group_face_ids {
        let wire = topo.face(fid)?.outer_wire();
        let Some(area) = brepkit_topology::planar::wire_area_2d(topo, wire, frame.x, frame.y)?
        else {
            return Ok(None);
        };
        if area.abs() > widest.abs() {
            widest = area;
        }
    }
    let outer_positive = if widest.abs() > lin * lin {
        widest > 0.0
    } else {
        !reversed
    };
    let build_wire =
        |topo: &mut Topology, info: &LoopInfo, want_positive: bool| -> Result<WireId, HealError> {
            let mut oes: Vec<OrientedEdge> =
                info.edge_indices.iter().map(|&i| edges[i].oe).collect();
            let has_positive = info.signed_area > 0.0;
            if has_positive != want_positive {
                oes.reverse();
                for oe in &mut oes {
                    *oe = OrientedEdge::new(oe.edge(), !oe.is_forward());
                }
            }
            let wire = Wire::new(oes, true)?;
            Ok(topo.add_wire(wire))
        };

    let mut new_face_ids: Vec<FaceId> = Vec::with_capacity(outer_indices.len());
    let mut infos_by_index: HashMap<usize, &LoopInfo> = HashMap::new();
    for (idx, info) in infos.iter().enumerate() {
        infos_by_index.insert(idx, info);
    }

    for &o in &outer_indices {
        let outer_wire = build_wire(topo, infos_by_index[&o], outer_positive)?;
        let mut inner_wires: Vec<WireId> = Vec::new();
        if let Some(holes) = region_holes.get(&o) {
            for &h in holes {
                inner_wires.push(build_wire(topo, infos_by_index[&h], !outer_positive)?);
            }
        }
        let new_face = if reversed {
            brepkit_topology::face::Face::new_reversed(outer_wire, inner_wires, surface.clone())
        } else {
            brepkit_topology::face::Face::new(outer_wire, inner_wires, surface.clone())
        };
        new_face_ids.push(topo.add_face(new_face));
    }

    Ok(Some(new_face_ids))
}

/// Signed left turn from incoming direction `a` to outgoing direction `b`,
/// returned as an angle in `(-π, π]`. Larger means a tighter left turn.
/// Running straight on lies on the cut at `±π`, where a rounding of the
/// cross product would rank it last: it is read as `π`.
fn signed_turn(a: (f64, f64), b: (f64, f64)) -> f64 {
    let cross = a.0 * b.1 - a.1 * b.0;
    let dot = a.0 * b.0 + a.1 * b.1;
    let angle = cross.atan2(dot);
    if angle < 1e-6 - std::f64::consts::PI {
        std::f64::consts::PI
    } else {
        angle
    }
}

/// An edge's 2D directions of travel as it leaves its start and arrives at
/// its end, traversed forward or not, in the plane of `x` and `y`: the
/// curve's own tangents at its ends, a NURBS curve's taken just inside them,
/// so an edge trimmed at a corner of its curve (a knot it is only C0 across)
/// reads the side it runs on.
fn travel_directions(
    curve: &EdgeCurve,
    es: brepkit_math::vec::Point3,
    ee: brepkit_math::vec::Point3,
    forward: bool,
    x: brepkit_math::vec::Vec3,
    y: brepkit_math::vec::Vec3,
) -> ((f64, f64), (f64, f64)) {
    let (t0, t1) = curve.domain_with_endpoints(es, ee);
    let sense = if t1 < t0 { -1.0 } else { 1.0 };
    let inward = if matches!(curve, EdgeCurve::NurbsCurve(_)) {
        1e-9 * (t1 - t0)
    } else {
        0.0
    };
    let along = |t: f64, flip: f64| {
        let d = curve.tangent_with_endpoints(t, es, ee) * (sense * flip);
        let (u, v) = (d.dot(x), d.dot(y));
        let len = u.hypot(v);
        if len < f64::EPSILON {
            (1.0, 0.0)
        } else {
            (u / len, v / len)
        }
    };
    let (leave, arrive) = (t0 + inward, t1 - inward);
    if forward {
        (along(leave, 1.0), along(arrive, 1.0))
    } else {
        (along(arrive, -1.0), along(leave, -1.0))
    }
}

/// Shoelace signed area of a closed UV polygon.
#[cfg(test)]
fn polygon_signed_area(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..n {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % n];
        acc += x0 * y1 - x1 * y0;
    }
    acc * 0.5
}

/// An edge's exact boundary in the plane's UV frame, with a point in its
/// middle: a segment for a line, an arc for a circle or an ellipse, and the
/// chords of its samples for a fitted curve.
fn edge_pieces(
    topo: &Topology,
    oe: &OrientedEdge,
    to_uv: &impl Fn(brepkit_math::vec::Point3) -> (f64, f64),
    frame: &Frame3,
) -> Result<(Vec<Boundary2>, (f64, f64)), HealError> {
    let edge = topo.edge(oe.edge())?;
    let (es, ee) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let pt = |p: brepkit_math::vec::Point3| {
        let (u, v) = to_uv(p);
        Point2::new(u, v)
    };
    let dir = |d: brepkit_math::vec::Vec3| Vec2::new(d.dot(frame.x), d.dot(frame.y));
    let (t0, t1) = edge.curve().domain_with_endpoints(es, ee);
    let middle = to_uv(
        edge.curve()
            .evaluate_with_endpoints(0.5 * (t0 + t1), es, ee),
    );
    let arc = |center, u, v, a, b| {
        let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        Boundary2::Arc {
            center: pt(center),
            u: dir(u),
            v: dir(v),
            a,
            b,
            t0: lo,
            t1: hi,
        }
    };
    let pieces = match edge.curve() {
        EdgeCurve::Line => vec![Boundary2::Segment(pt(es), pt(ee))],
        EdgeCurve::Circle(c) => vec![arc(
            c.center(),
            c.u_axis(),
            c.v_axis(),
            c.radius(),
            c.radius(),
        )],
        EdgeCurve::Ellipse(e) => vec![arc(
            e.center(),
            e.u_axis(),
            e.v_axis(),
            e.semi_major(),
            e.semi_minor(),
        )],
        EdgeCurve::NurbsCurve(_) => {
            const SAMPLES: usize = 32;
            let at = |k: usize| {
                #[allow(clippy::cast_precision_loss)]
                let t = t0 + (t1 - t0) * (k as f64 / SAMPLES as f64);
                pt(edge.curve().evaluate_with_endpoints(t, es, ee))
            };
            (0..SAMPLES)
                .map(|k| Boundary2::Segment(at(k), at(k + 1)))
                .collect()
        }
    };
    Ok((pieces, middle))
}

/// Merge runs of adjacent edges that share a common geometric host.
///
/// Scans consecutive edge pairs and merges them when they:
///  - share a topology vertex,
///  - have the same orientation (both forward in the wire), and
///  - share the same underlying analytic curve.
///
/// Supported curve kinds:
///  - `Line` — collinear Line+Line, merged into a single Line edge.
///  - `Circle` — co-circular arcs (same center, normal, radius), merged into
///    a single Circle arc.
///  - `Ellipse` — co-elliptical arcs (same center, axes, semi-radii).
///
/// `NurbsCurve` edges are not merged — they have no closed-form equivalence
/// test cheap enough to run inline, and merging them requires knot-vector
/// stitching that belongs in a separate upgrade pass.
///
/// Mixed-orientation runs (e.g. one forward + one reversed) are left alone:
/// reversing an arc on a Circle3D requires flipping the surface normal, which
/// is a substantive change rather than a textual merge.
///
/// Returns the number of edge merges performed.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
fn merge_collinear_edges(
    topo: &mut Topology,
    wire_id: WireId,
    options: &UnifyOptions,
) -> Result<usize, HealError> {
    use brepkit_math::curves::{Circle3D, Ellipse3D};

    /// Curve-kind snapshot used to decide whether two consecutive edges live
    /// on the same geometric host. `Other` blocks merging (NurbsCurve, etc.).
    #[derive(Clone)]
    enum EdgeKind {
        Line,
        Circle(Circle3D),
        Ellipse(Ellipse3D),
        Other,
    }

    struct EdgeData {
        oe: OrientedEdge,
        start_vid: brepkit_topology::vertex::VertexId,
        end_vid: brepkit_topology::vertex::VertexId,
        start_pos: brepkit_math::vec::Point3,
        end_pos: brepkit_math::vec::Point3,
        kind: EdgeKind,
    }

    let lin = options.linear_tolerance;
    let ang = options.angular_tolerance;

    let circles_match = |a: &Circle3D, b: &Circle3D| -> bool {
        (a.center() - b.center()).length() < lin
            && (a.radius() - b.radius()).abs() < lin
            && a.normal().dot(b.normal()) > 1.0 - ang
    };
    let ellipses_match = |a: &Ellipse3D, b: &Ellipse3D| -> bool {
        (a.center() - b.center()).length() < lin
            && (a.semi_major() - b.semi_major()).abs() < lin
            && (a.semi_minor() - b.semi_minor()).abs() < lin
            && a.normal().dot(b.normal()) > 1.0 - ang
            // The angular reference axis must agree, otherwise arc params don't match.
            && a.u_axis().dot(b.u_axis()) > 1.0 - ang
    };

    let kinds_match = |a: &EdgeKind, b: &EdgeKind| -> bool {
        match (a, b) {
            (EdgeKind::Line, EdgeKind::Line) => true,
            (EdgeKind::Circle(c1), EdgeKind::Circle(c2)) => circles_match(c1, c2),
            (EdgeKind::Ellipse(e1), EdgeKind::Ellipse(e2)) => ellipses_match(e1, e2),
            _ => false,
        }
    };

    let wire = topo.wire(wire_id)?;
    let edges_list: Vec<OrientedEdge> = wire.edges().to_vec();
    let is_closed = wire.is_closed();
    let n = edges_list.len();

    if n < 2 {
        return Ok(0);
    }

    let mut data = Vec::with_capacity(n);
    for oe in &edges_list {
        let edge = topo.edge(oe.edge())?;
        let kind = match edge.curve() {
            EdgeCurve::Line => EdgeKind::Line,
            EdgeCurve::Circle(c) => EdgeKind::Circle(c.clone()),
            EdgeCurve::Ellipse(e) => EdgeKind::Ellipse(e.clone()),
            EdgeCurve::NurbsCurve(_) => EdgeKind::Other,
        };
        let start_vid = oe.oriented_start(edge);
        let end_vid = oe.oriented_end(edge);
        let start_pos = topo.vertex(start_vid)?.point();
        let end_pos = topo.vertex(end_vid)?.point();
        data.push(EdgeData {
            oe: *oe,
            start_vid,
            end_vid,
            start_pos,
            end_pos,
            kind,
        });
    }

    let mut new_edges: Vec<OrientedEdge> = Vec::with_capacity(n);
    let mut merged_count = 0usize;
    let mut i = 0;

    while i < n {
        // Capture the curve template the merged edge would inherit if the run
        // extends. `None` ⇒ this edge cannot start a mergeable run; we never
        // re-derive the curve at merge time, so the merge path has no
        // unreachable arm.
        let head = &data[i];
        let merge_template: Option<EdgeCurve> = if head.oe.is_forward() {
            match &head.kind {
                EdgeKind::Line => Some(EdgeCurve::Line),
                EdgeKind::Circle(c) => Some(EdgeCurve::Circle(c.clone())),
                EdgeKind::Ellipse(e) => Some(EdgeCurve::Ellipse(e.clone())),
                EdgeKind::Other => None,
            }
        } else {
            None
        };

        let Some(merged_curve) = merge_template else {
            new_edges.push(head.oe);
            i += 1;
            continue;
        };

        // For lines we additionally require subsequent segments to be parallel
        // — colocated start/end positions on the same line — so capture the
        // direction up front.
        let head_line_dir = if matches!(head.kind, EdgeKind::Line) {
            if let Ok(d) = (head.end_pos - head.start_pos).normalize() {
                Some(d)
            } else {
                new_edges.push(head.oe);
                i += 1;
                continue;
            }
        } else {
            None
        };

        let mut run_end_vid = head.end_vid;
        let mut j = i + 1;
        while j < n {
            let next = &data[j];

            if !next.oe.is_forward() {
                break;
            }
            if !kinds_match(&head.kind, &next.kind) {
                break;
            }
            if next.start_vid != run_end_vid {
                break;
            }

            // Line-specific parallelism check: same direction along the run.
            if let Some(dir0) = head_line_dir {
                let span = next.end_pos - next.start_pos;
                if span.length() < lin {
                    break;
                }
                let Ok(dir1) = span.normalize() else {
                    break;
                };
                if dir0.dot(dir1) < 1.0 - ang {
                    break;
                }
            }

            run_end_vid = next.end_vid;
            j += 1;
        }

        let run_length = j - i;
        if run_length == 1 {
            new_edges.push(head.oe);
            i += 1;
            continue;
        }

        let new_edge = Edge::new(head.start_vid, run_end_vid, merged_curve);
        let new_edge_id = topo.add_edge(new_edge);
        new_edges.push(OrientedEdge::new(new_edge_id, true));
        merged_count += run_length - 1;
        i = j;
    }

    if merged_count == 0 {
        return Ok(0);
    }

    let new_wire = Wire::new(new_edges, is_closed)?;
    let wire_mut = topo.wire_mut(wire_id)?;
    *wire_mut = new_wire;

    Ok(merged_count)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod hole_merge_tests {
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::{Vertex, VertexId};
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    fn add_vertex(topo: &mut Topology, x: f64, y: f64) -> VertexId {
        topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7))
    }

    fn line(topo: &mut Topology, a: VertexId, b: VertexId) -> EdgeId {
        topo.add_edge(Edge::new(a, b, EdgeCurve::Line))
    }

    fn loop_of(topo: &mut Topology, edges: &[(EdgeId, bool)]) -> WireId {
        let oes: Vec<_> = edges
            .iter()
            .map(|&(e, f)| OrientedEdge::new(e, f))
            .collect();
        topo.add_wire(Wire::new(oes, true).unwrap())
    }

    fn plane_z0() -> FaceSurface {
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        }
    }

    #[test]
    fn unify_two_coplanar_faces_sharing_a_hole_merges_outer_and_preserves_inner_wire() {
        // Left face spans x∈[0,2], right face spans x∈[2,4], both y∈[0,2],
        // sharing the middle edge x=2. A unit-square hole sits inside the
        // left face: x∈[0.5,1.5], y∈[0.5,1.5].
        let mut topo = Topology::default();

        let v00 = add_vertex(&mut topo, 0.0, 0.0);
        let v20 = add_vertex(&mut topo, 2.0, 0.0);
        let v40 = add_vertex(&mut topo, 4.0, 0.0);
        let v42 = add_vertex(&mut topo, 4.0, 2.0);
        let v22 = add_vertex(&mut topo, 2.0, 2.0);
        let v02 = add_vertex(&mut topo, 0.0, 2.0);

        // Shared middle edge x=2 (forward for left, reversed for right).
        let e_mid = line(&mut topo, v20, v22);

        // Left outer wire CCW: 0,0 → 2,0 → 2,2 → 0,2 → 0,0
        let l_b = line(&mut topo, v00, v20);
        let l_t = line(&mut topo, v22, v02);
        let l_l = line(&mut topo, v02, v00);
        let left_outer = loop_of(
            &mut topo,
            &[(l_b, true), (e_mid, true), (l_t, true), (l_l, true)],
        );

        // Hole square (CW so material lies outside it).
        let h00 = add_vertex(&mut topo, 0.5, 0.5);
        let h10 = add_vertex(&mut topo, 1.5, 0.5);
        let h11 = add_vertex(&mut topo, 1.5, 1.5);
        let h01 = add_vertex(&mut topo, 0.5, 1.5);
        let hb = line(&mut topo, h00, h01);
        let hl = line(&mut topo, h01, h11);
        let ht = line(&mut topo, h11, h10);
        let hr = line(&mut topo, h10, h00);
        let hole_wire = loop_of(&mut topo, &[(hb, true), (hl, true), (ht, true), (hr, true)]);

        let left_face = topo.add_face(Face::new(left_outer, vec![hole_wire], plane_z0()));

        // Right outer wire CCW: 2,0 → 4,0 → 4,2 → 2,2 → 2,0
        let r_b = line(&mut topo, v20, v40);
        let r_r = line(&mut topo, v40, v42);
        let r_t = line(&mut topo, v42, v22);
        let right_outer = loop_of(
            &mut topo,
            &[(r_b, true), (r_r, true), (r_t, true), (e_mid, false)],
        );
        let right_face = topo.add_face(Face::new(right_outer, vec![], plane_z0()));

        let shell_id = topo.add_shell(Shell::new(vec![left_face, right_face]).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let opts = UnifyOptions {
            unify_edges: false,
            ..UnifyOptions::default()
        };
        let (_, result) = unify_same_domain(&mut topo, solid_id, &opts).unwrap();

        assert_eq!(result.faces_merged, 1, "two faces collapse to one");

        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 1, "exactly one face remains");

        let merged_fid = shell.faces()[0];
        let merged = topo.face(merged_fid).unwrap();
        assert_eq!(merged.inner_wires().len(), 1, "hole preserved");

        // The shared middle edge must not appear on any wire of the merged face.
        let mut all_edges: Vec<EdgeId> = Vec::new();
        let outer = topo.wire(merged.outer_wire()).unwrap();
        all_edges.extend(outer.edges().iter().map(super::OrientedEdge::edge));
        for &iw in merged.inner_wires() {
            let w = topo.wire(iw).unwrap();
            all_edges.extend(w.edges().iter().map(super::OrientedEdge::edge));
        }
        assert!(!all_edges.contains(&e_mid), "shared seam edge XOR-removed");

        // Hole corners preserved within tolerance.
        let inner = topo.wire(merged.inner_wires()[0]).unwrap();
        let mut hole_pts: Vec<Point3> = Vec::new();
        for oe in inner.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            hole_pts.push(topo.vertex(oe.oriented_start(edge)).unwrap().point());
        }
        let expected = [
            Point3::new(0.5, 0.5, 0.0),
            Point3::new(1.5, 0.5, 0.0),
            Point3::new(1.5, 1.5, 0.0),
            Point3::new(0.5, 1.5, 0.0),
        ];
        for e in &expected {
            assert!(
                hole_pts.iter().any(|p| (*p - *e).length() < 1e-7),
                "hole corner {e:?} preserved"
            );
        }

        // Area check: outer area minus hole area equals union of member areas
        // minus hole.
        let outer_uv: Vec<(f64, f64)> = {
            let outer_w = topo.wire(merged.outer_wire()).unwrap();
            outer_w
                .edges()
                .iter()
                .map(|oe| {
                    let edge = topo.edge(oe.edge()).unwrap();
                    let p = topo.vertex(oe.oriented_start(edge)).unwrap().point();
                    (p.x(), p.y())
                })
                .collect()
        };
        let hole_uv: Vec<(f64, f64)> = hole_pts.iter().map(|p| (p.x(), p.y())).collect();
        let outer_area = super::polygon_signed_area(&outer_uv).abs();
        let hole_area = super::polygon_signed_area(&hole_uv).abs();
        // left (4) + right (4) - hole (1) = 7
        assert!(
            ((outer_area - hole_area) - 7.0).abs() < 1e-7,
            "net area {} should be 7",
            outer_area - hole_area
        );
    }

    #[test]
    fn unify_two_split_regions_each_preserve_their_hole() {
        // Two disjoint co-planar regions, each split into two members and each
        // carrying its own hole. They form two same-domain groups; each must
        // merge to a single face keeping its own inner wire.
        let mut topo = Topology::default();

        let make_region = |topo: &mut Topology, x0: f64| -> (FaceId, FaceId, EdgeId) {
            let y0 = 0.0;
            let y1 = 2.0;
            let xm = x0 + 2.0;
            let x1 = x0 + 4.0;
            let a = add_vertex(topo, x0, y0);
            let b = add_vertex(topo, xm, y0);
            let c = add_vertex(topo, x1, y0);
            let d = add_vertex(topo, x1, y1);
            let e = add_vertex(topo, xm, y1);
            let f = add_vertex(topo, x0, y1);
            let mid = line(topo, b, e);
            let lb = line(topo, a, b);
            let lt = line(topo, e, f);
            let ll = line(topo, f, a);
            let left_outer = loop_of(topo, &[(lb, true), (mid, true), (lt, true), (ll, true)]);
            // Hole in the left half.
            let h0 = add_vertex(topo, x0 + 0.5, 0.5);
            let h1 = add_vertex(topo, x0 + 1.5, 0.5);
            let h2 = add_vertex(topo, x0 + 1.5, 1.5);
            let h3 = add_vertex(topo, x0 + 0.5, 1.5);
            let hb = line(topo, h0, h3);
            let hl = line(topo, h3, h2);
            let ht = line(topo, h2, h1);
            let hr = line(topo, h1, h0);
            let hole = loop_of(topo, &[(hb, true), (hl, true), (ht, true), (hr, true)]);
            let lf = topo.add_face(Face::new(left_outer, vec![hole], plane_z0()));
            let rb = line(topo, b, c);
            let rr = line(topo, c, d);
            let rt = line(topo, d, e);
            let right_outer = loop_of(topo, &[(rb, true), (rr, true), (rt, true), (mid, false)]);
            let rf = topo.add_face(Face::new(right_outer, vec![], plane_z0()));
            (lf, rf, mid)
        };

        let (a_l, a_r, _) = make_region(&mut topo, 0.0);
        let (b_l, b_r, _) = make_region(&mut topo, 6.0);

        let shell_id = topo.add_shell(Shell::new(vec![a_l, a_r, b_l, b_r]).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let opts = UnifyOptions {
            unify_edges: false,
            ..UnifyOptions::default()
        };
        let (_, result) = unify_same_domain(&mut topo, solid_id, &opts).unwrap();

        // Each region merges its two members → 2 merges total.
        assert_eq!(result.faces_merged, 2);

        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 2, "two merged regions remain");
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            assert_eq!(f.inner_wires().len(), 1, "each region keeps its hole");
        }
    }

    #[test]
    fn unify_keeps_a_round_hole_where_the_merged_outline_centres() {
        // A 4 x 2 plate holed by a full circle at (3, 1), r = 0.5, and a
        // 1 x 2 tab beside it, sharing the plate's edge x = 4. The merged
        // outline's vertices average to (3, 1), the hole's centre.
        let mut topo = Topology::default();
        let v00 = add_vertex(&mut topo, 0.0, 0.0);
        let v40 = add_vertex(&mut topo, 4.0, 0.0);
        let v50 = add_vertex(&mut topo, 5.0, 0.0);
        let v52 = add_vertex(&mut topo, 5.0, 2.0);
        let v42 = add_vertex(&mut topo, 4.0, 2.0);
        let v02 = add_vertex(&mut topo, 0.0, 2.0);
        let e_mid = line(&mut topo, v40, v42);
        let p_b = line(&mut topo, v00, v40);
        let p_t = line(&mut topo, v42, v02);
        let p_l = line(&mut topo, v02, v00);
        let plate_outer = loop_of(
            &mut topo,
            &[(p_b, true), (e_mid, true), (p_t, true), (p_l, true)],
        );
        let circle = brepkit_math::curves::Circle3D::new(
            Point3::new(3.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.5,
        )
        .unwrap();
        let hv = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        let rim = topo.add_edge(Edge::new(hv, hv, EdgeCurve::Circle(circle)));
        let hole = loop_of(&mut topo, &[(rim, false)]);
        let plate = topo.add_face(Face::new(plate_outer, vec![hole], plane_z0()));
        let t_b = line(&mut topo, v40, v50);
        let t_r = line(&mut topo, v50, v52);
        let t_t = line(&mut topo, v52, v42);
        let tab_outer = loop_of(
            &mut topo,
            &[(t_b, true), (t_r, true), (t_t, true), (e_mid, false)],
        );
        let tab = topo.add_face(Face::new(tab_outer, vec![], plane_z0()));
        let shell_id = topo.add_shell(Shell::new(vec![plate, tab]).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let opts = UnifyOptions {
            unify_edges: false,
            ..UnifyOptions::default()
        };
        let (_, result) = unify_same_domain(&mut topo, solid_id, &opts).unwrap();
        assert_eq!(
            result.faces_merged, 1,
            "the plate and the tab become one face"
        );
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 1);
        let merged = topo.face(shell.faces()[0]).unwrap();
        assert_eq!(merged.inner_wires().len(), 1, "the round hole stays a hole");
        let inner = topo.wire(merged.inner_wires()[0]).unwrap();
        assert_eq!(inner.edges().len(), 1);
        assert_eq!(inner.edges()[0].edge(), rim);
        let outer = topo.wire(merged.outer_wire()).unwrap();
        assert_eq!(
            outer.edges().len(),
            6,
            "the outline is plate and tab, the seam gone"
        );
    }

    #[test]
    fn unify_keeps_a_small_hole_close_inside_a_curved_outline() {
        // A unit disc in three 120-degree sectors, one holed by a circle of
        // radius 0.0002 at polar (0.999, pi/48): inside the disc, but outside
        // a chord polygon of its rim, so a sampled outline reads the hole as
        // lying outside the merged face.
        let mut topo = Topology::default();
        let unit = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        let centre = add_vertex(&mut topo, 0.0, 0.0);
        let rim: Vec<VertexId> = (0..3)
            .map(|k| {
                let a = std::f64::consts::TAU * f64::from(k) / 3.0;
                add_vertex(&mut topo, a.cos(), a.sin())
            })
            .collect();
        let spokes: Vec<EdgeId> = rim.iter().map(|&r| line(&mut topo, centre, r)).collect();
        let arcs: Vec<EdgeId> = (0..3)
            .map(|k| {
                topo.add_edge(Edge::new(
                    rim[k],
                    rim[(k + 1) % 3],
                    EdgeCurve::Circle(unit.clone()),
                ))
            })
            .collect();
        let at = std::f64::consts::PI / 48.0;
        let small = brepkit_math::curves::Circle3D::new(
            Point3::new(0.999 * at.cos(), 0.999 * at.sin(), 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0002,
        )
        .unwrap();
        let hv = topo.add_vertex(Vertex::new(small.evaluate(0.0), 1e-7));
        let hole_rim = topo.add_edge(Edge::new(hv, hv, EdgeCurve::Circle(small)));
        let faces: Vec<FaceId> = (0..3)
            .map(|k| {
                let outer = loop_of(
                    &mut topo,
                    &[
                        (spokes[k], true),
                        (arcs[k], true),
                        (spokes[(k + 1) % 3], false),
                    ],
                );
                let holes = if k == 0 {
                    vec![loop_of(&mut topo, &[(hole_rim, false)])]
                } else {
                    vec![]
                };
                topo.add_face(Face::new(outer, holes, plane_z0()))
            })
            .collect();
        let shell_id = topo.add_shell(Shell::new(faces).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let opts = UnifyOptions {
            unify_edges: false,
            ..UnifyOptions::default()
        };
        unify_same_domain(&mut topo, solid_id, &opts).unwrap();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 1, "the three sectors become one face");
        let merged = topo.face(shell.faces()[0]).unwrap();
        assert_eq!(merged.inner_wires().len(), 1, "the small hole stays a hole");
        let inner = topo.wire(merged.inner_wires()[0]).unwrap();
        assert_eq!(inner.edges()[0].edge(), hole_rim);
    }

    /// A plate in two faces sharing x = 2, the left one holed by a circle
    /// about (-1, 0) split at (-2, 0) and (0, 0), by the triangle on (0, 0),
    /// `p` and `q` (clockwise), and by a whole circle about (-2, 2): the
    /// first two holes touch at (0, 0); all of it turned `turn` about the
    /// origin, every loop wound the other way round when `flipped`. The
    /// sizes of the merged face's holes.
    fn merged_hole_sizes(p: (f64, f64), q: (f64, f64), turn: f64, flipped: bool) -> Vec<usize> {
        let mut topo = Topology::default();
        // Every loop wound the other way round when `flipped`.
        let loop_of = |topo: &mut Topology, edges: &[(EdgeId, bool)]| {
            if flipped {
                let back: Vec<(EdgeId, bool)> = edges.iter().rev().map(|&(e, f)| (e, !f)).collect();
                loop_of(topo, &back)
            } else {
                loop_of(topo, edges)
            }
        };
        let (sin, cos) = turn.sin_cos();
        let at = |x: f64, y: f64| (cos * x - sin * y, sin * x + cos * y);
        let add_vertex = |topo: &mut Topology, x: f64, y: f64| {
            let (x, y) = at(x, y);
            add_vertex(topo, x, y)
        };
        let corners = |topo: &mut Topology, x0: f64, x1: f64| {
            [(x0, -3.0), (x1, -3.0), (x1, 3.0), (x0, 3.0)].map(|(x, y)| add_vertex(topo, x, y))
        };
        let a = corners(&mut topo, -3.0, 2.0);
        let b0 = add_vertex(&mut topo, 4.0, -3.0);
        let b1 = add_vertex(&mut topo, 4.0, 3.0);
        let seam = line(&mut topo, a[1], a[2]);
        let a_edges = [
            line(&mut topo, a[0], a[1]),
            line(&mut topo, a[2], a[3]),
            line(&mut topo, a[3], a[0]),
        ];
        let left_outer = loop_of(
            &mut topo,
            &[
                (a_edges[0], true),
                (seam, true),
                (a_edges[1], true),
                (a_edges[2], true),
            ],
        );
        let (b_bottom, b_right, b_top) = (
            line(&mut topo, a[1], b0),
            line(&mut topo, b0, b1),
            line(&mut topo, b1, a[2]),
        );
        let right_outer = loop_of(
            &mut topo,
            &[
                (b_bottom, true),
                (b_right, true),
                (b_top, true),
                (seam, false),
            ],
        );

        let up = Vec3::new(0.0, 0.0, 1.0);
        let centre = |x: f64, y: f64| {
            let (x, y) = at(x, y);
            Point3::new(x, y, 0.0)
        };
        let circle = brepkit_math::curves::Circle3D::new(centre(-1.0, 0.0), up, 1.0).unwrap();
        let (touch, west) = (
            add_vertex(&mut topo, 0.0, 0.0),
            add_vertex(&mut topo, -2.0, 0.0),
        );
        let upper = topo.add_edge(Edge::new(touch, west, EdgeCurve::Circle(circle.clone())));
        let lower = topo.add_edge(Edge::new(west, touch, EdgeCurve::Circle(circle)));
        // Walked from (-2, 0), so the choice at (0, 0) comes mid-walk.
        let round = loop_of(&mut topo, &[(upper, false), (lower, false)]);
        let (pv, qv) = (
            add_vertex(&mut topo, p.0, p.1),
            add_vertex(&mut topo, q.0, q.1),
        );
        let sides = [
            line(&mut topo, touch, pv),
            line(&mut topo, pv, qv),
            line(&mut topo, qv, touch),
        ];
        let tri = loop_of(
            &mut topo,
            &[(sides[0], true), (sides[1], true), (sides[2], true)],
        );
        let whole = brepkit_math::curves::Circle3D::new(centre(-2.0, 2.0), up, 0.3).unwrap();
        let wv = topo.add_vertex(Vertex::new(whole.evaluate(0.0), 1e-7));
        let rim = topo.add_edge(Edge::new(wv, wv, EdgeCurve::Circle(whole)));
        let bore = loop_of(&mut topo, &[(rim, false)]);

        let left = topo.add_face(Face::new(left_outer, vec![round, tri, bore], plane_z0()));
        let right = topo.add_face(Face::new(right_outer, vec![], plane_z0()));
        let shell_id = topo.add_shell(Shell::new(vec![left, right]).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let opts = UnifyOptions {
            unify_edges: false,
            ..UnifyOptions::default()
        };
        unify_same_domain(&mut topo, solid_id, &opts).unwrap();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 1, "the two faces become one");
        let merged = topo.face(shell.faces()[0]).unwrap();
        let mut sizes: Vec<usize> = merged
            .inner_wires()
            .iter()
            .map(|&w| topo.wire(w).unwrap().edges().len())
            .collect();
        sizes.sort_unstable();
        sizes
    }

    #[test]
    fn unify_keeps_two_holes_touching_at_a_vertex_apart() {
        // Arriving at (0, 0) along its upper half, the circle's walk must
        // turn onto its lower half, not onto the triangle, so each hole stays
        // a loop of its own: a triangle opening wide from (0, 0), and one
        // leaving it within a few degrees of the circle's tangent there,
        // each turned about the origin so straight on falls either side of
        // the turn's cut.
        let at = |deg: f64| {
            let a = deg.to_radians();
            (a.cos(), a.sin())
        };
        for (p, q) in [((1.0, 1.0), (1.0, -1.0)), (at(-85.0), at(-89.0))] {
            for turn in [0.0, 0.3, 1.1, 2.0, 3.7, 5.5] {
                for flipped in [false, true] {
                    assert_eq!(
                        merged_hole_sizes(p, q, turn, flipped),
                        [1, 2, 3],
                        "the circles and the triangle on {p:?}, {q:?}, turned {turn}, \
                         wound {}, stay three holes",
                        if flipped { "back" } else { "forward" }
                    );
                }
            }
        }
    }

    #[test]
    fn running_straight_on_ranks_as_the_widest_turn() {
        for tilt in [1e-18, -1e-18, 0.0] {
            let turn = super::signed_turn((0.0, 1.0), (tilt, -1.0));
            assert!((turn - std::f64::consts::PI).abs() < 1e-12, "{turn}");
        }
    }

    #[test]
    fn an_edge_ending_at_a_corner_of_its_curve_arrives_along_its_own_side() {
        // A polyline from (0, 0) up to (1, 1) and down to (2, 0), C0 at the
        // corner; the edge is its first leg.
        let curve = EdgeCurve::NurbsCurve(
            brepkit_math::nurbs::curve::NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 2.0, 2.0],
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(2.0, 0.0, 0.0),
                ],
                vec![1.0; 3],
            )
            .unwrap(),
        );
        let (a, b) = (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0));
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let near = |d: (f64, f64), want: (f64, f64)| (d.0 - want.0).hypot(d.1 - want.1) < 1e-6;
        let (leave, arrive) = super::travel_directions(&curve, a, b, true, x, y);
        assert!(
            near(leave, (s, s)) && near(arrive, (s, s)),
            "{leave:?} {arrive:?}"
        );
        let (leave, arrive) = super::travel_directions(&curve, a, b, false, x, y);
        assert!(
            near(leave, (-s, -s)) && near(arrive, (-s, -s)),
            "{leave:?} {arrive:?}"
        );
    }

    #[test]
    fn unify_winds_a_thin_curved_band_by_its_exact_area() {
        // Two 60-degree sectors of the band between radii 1 and 0.9999, each
        // bounded by one outer arc and two inner arcs, the first holed by a
        // circle of radius 0.00002 midway across the band. Sampled per edge,
        // a member's outline winds the wrong way round, so the merged hole
        // was turned to wind with the outline.
        let mut topo = Topology::default();
        let circle = |r: f64| {
            brepkit_math::curves::Circle3D::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                r,
            )
            .unwrap()
        };
        let (outer_r, inner_r) = (1.0, 0.9999);
        let at = |r: f64, deg: f64| {
            let a = deg.to_radians();
            (r * a.cos(), r * a.sin())
        };
        let vertex = |topo: &mut Topology, (x, y): (f64, f64)| add_vertex(topo, x, y);
        let outer: Vec<VertexId> = [0.0, 60.0, 120.0]
            .iter()
            .map(|&d| vertex(&mut topo, at(outer_r, d)))
            .collect();
        let inner: Vec<VertexId> = [0.0, 30.0, 60.0, 90.0, 120.0]
            .iter()
            .map(|&d| vertex(&mut topo, at(inner_r, d)))
            .collect();
        let arc = |topo: &mut Topology, a: VertexId, b: VertexId, r: f64| {
            topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle(r))))
        };
        let outer_arcs: Vec<EdgeId> = (0..2)
            .map(|k| arc(&mut topo, outer[k], outer[k + 1], outer_r))
            .collect();
        let inner_arcs: Vec<EdgeId> = (0..4)
            .map(|k| arc(&mut topo, inner[k], inner[k + 1], inner_r))
            .collect();
        let radii: Vec<EdgeId> = (0..3)
            .map(|k| line(&mut topo, inner[2 * k], outer[k]))
            .collect();
        let mid = 0.5 * (outer_r + inner_r);
        let small = brepkit_math::curves::Circle3D::new(
            Point3::new(
                mid * 30_f64.to_radians().cos(),
                mid * 30_f64.to_radians().sin(),
                0.0,
            ),
            Vec3::new(0.0, 0.0, 1.0),
            0.00002,
        )
        .unwrap();
        let hv = topo.add_vertex(Vertex::new(small.evaluate(0.0), 1e-7));
        let hole_rim = topo.add_edge(Edge::new(hv, hv, EdgeCurve::Circle(small)));
        let faces: Vec<FaceId> = (0..2)
            .map(|k| {
                let outline = loop_of(
                    &mut topo,
                    &[
                        (radii[k], true),
                        (outer_arcs[k], true),
                        (radii[k + 1], false),
                        (inner_arcs[2 * k + 1], false),
                        (inner_arcs[2 * k], false),
                    ],
                );
                let holes = if k == 0 {
                    vec![loop_of(&mut topo, &[(hole_rim, false)])]
                } else {
                    vec![]
                };
                topo.add_face(Face::new(outline, holes, plane_z0()))
            })
            .collect();
        let shell_id = topo.add_shell(Shell::new(faces).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let opts = UnifyOptions {
            unify_edges: false,
            ..UnifyOptions::default()
        };
        unify_same_domain(&mut topo, solid_id, &opts).unwrap();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 1, "the two sectors become one face");
        let merged = topo.face(shell.faces()[0]).unwrap();
        assert_eq!(merged.inner_wires().len(), 1, "the small hole stays a hole");
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let area = |w| {
            brepkit_topology::planar::wire_area_2d(&topo, w, x, y)
                .unwrap()
                .unwrap()
        };
        let (outline, hole) = (area(merged.outer_wire()), area(merged.inner_wires()[0]));
        assert!(outline > 0.0, "the band winds as its members do: {outline}");
        assert!(hole < 0.0, "the hole winds against it: {hole}");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod merge_tests {
    use brepkit_math::curves::{Circle3D, Ellipse3D};
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::{Vertex, VertexId};
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    fn z_axis() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }
    fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    fn add_vertex(topo: &mut Topology, p: Point3) -> VertexId {
        topo.add_vertex(Vertex::new(p, 1e-7))
    }

    /// Build a wire from a sequence of edges, all forward-oriented.
    fn build_wire_from_edges(
        topo: &mut Topology,
        edge_ids: &[brepkit_topology::edge::EdgeId],
        is_closed: bool,
    ) -> WireId {
        let oes: Vec<_> = edge_ids
            .iter()
            .map(|&e| OrientedEdge::new(e, true))
            .collect();
        topo.add_wire(Wire::new(oes, is_closed).unwrap())
    }

    /// Two quarter-cylinder faces stacked along the axis share the arc
    /// between them, which sits mid-wire in each; the merged face's wire runs
    /// its surviving edges end to end.
    #[test]
    fn stacked_quarter_cylinders_merge_into_a_chained_wire() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::face::{Face, FaceSurface};

        let mut topo = Topology::default();
        let cyl = CylindricalSurface::new(origin(), z_axis(), 1.0).unwrap();
        let at = |z: f64| Circle3D::new(Point3::new(0.0, 0.0, z), z_axis(), 1.0).unwrap();
        let quarter = std::f64::consts::FRAC_PI_2;
        let [a0, b0, a1, b1, a2, b2] = [0.0, 1.0, 2.0]
            .map(|z| {
                [at(z).evaluate(0.0), at(z).evaluate(quarter)].map(|p| add_vertex(&mut topo, p))
            })
            .concat()
            .try_into()
            .unwrap();
        let arc =
            |topo: &mut Topology, a, b, z| topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(at(z))));
        let (bottom, middle, top) = (
            arc(&mut topo, a0, b0, 0.0),
            arc(&mut topo, a1, b1, 1.0),
            arc(&mut topo, a2, b2, 2.0),
        );
        let [ra, rb, sa, sb] = [(a0, a1), (b0, b1), (a1, a2), (b1, b2)]
            .map(|(p, q)| topo.add_edge(Edge::new(p, q, EdgeCurve::Line)));
        let face = |topo: &mut Topology, edges: [(brepkit_topology::edge::EdgeId, bool); 4]| {
            let oes = edges.map(|(e, fwd)| OrientedEdge::new(e, fwd)).to_vec();
            let wire = topo.add_wire(Wire::new(oes, true).unwrap());
            topo.add_face(Face::new(
                wire,
                Vec::new(),
                FaceSurface::Cylinder(cyl.clone()),
            ))
        };
        let low = face(
            &mut topo,
            [(rb, true), (middle, false), (ra, false), (bottom, true)],
        );
        let high = face(
            &mut topo,
            [(middle, true), (sb, true), (top, false), (sa, false)],
        );

        let merged =
            merge_group_simple(&mut topo, &[low, high], &FaceSurface::Cylinder(cyl.clone()))
                .unwrap()
                .unwrap();
        let wire = topo
            .wire(topo.face(merged[0]).unwrap().outer_wire())
            .unwrap();
        let ends: Vec<_> = wire
            .edges()
            .iter()
            .map(|oe| {
                let e = topo.edge(oe.edge()).unwrap();
                if oe.is_forward() {
                    (e.start(), e.end())
                } else {
                    (e.end(), e.start())
                }
            })
            .collect();
        assert_eq!(ends.len(), 6);
        for (i, &(_, end)) in ends.iter().enumerate() {
            assert_eq!(
                end,
                ends[(i + 1) % ends.len()].0,
                "edge {i} does not meet the next"
            );
        }
    }

    #[test]
    fn three_circle_arcs_merge_into_one_full_circle() {
        // Three quarter-arcs at 0°→120°→240°→360°.
        let mut topo = Topology::default();
        let circle = Circle3D::new(origin(), z_axis(), 1.0).unwrap();
        let p_at = |theta: f64| circle.evaluate(theta);

        let v0 = add_vertex(&mut topo, p_at(0.0));
        let v1 = add_vertex(&mut topo, p_at(2.0 * std::f64::consts::PI / 3.0));
        let v2 = add_vertex(&mut topo, p_at(4.0 * std::f64::consts::PI / 3.0));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Circle(circle.clone())));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Circle(circle.clone())));
        let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Circle(circle.clone())));
        let wire = build_wire_from_edges(&mut topo, &[e0, e1, e2], true);

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        assert_eq!(merged, 2, "three arcs → one merged edge means 2 merges");

        let new_wire = topo.wire(wire).unwrap();
        assert_eq!(new_wire.edges().len(), 1);

        let merged_eid = new_wire.edges()[0].edge();
        let merged_edge = topo.edge(merged_eid).unwrap();
        assert!(matches!(merged_edge.curve(), EdgeCurve::Circle(_)));
        // Closed-loop arc should reduce to a single closed circle edge.
        assert_eq!(merged_edge.start(), merged_edge.end());
    }

    #[test]
    fn arcs_on_different_circles_do_not_merge() {
        let mut topo = Topology::default();
        let circle_a = Circle3D::new(origin(), z_axis(), 1.0).unwrap();
        let circle_b = Circle3D::new(Point3::new(1.0, 0.0, 0.0), z_axis(), 1.0).unwrap();

        let v0 = add_vertex(&mut topo, circle_a.evaluate(0.0));
        let v1 = add_vertex(&mut topo, circle_a.evaluate(std::f64::consts::PI));
        let v2 = add_vertex(&mut topo, circle_b.evaluate(std::f64::consts::PI));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Circle(circle_a)));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Circle(circle_b)));
        let wire = build_wire_from_edges(&mut topo, &[e0, e1], false);

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        assert_eq!(merged, 0);
    }

    #[test]
    fn antiparallel_normal_circles_do_not_merge() {
        // Same center & radius but opposite normal — these are different
        // parameterizations and merging would silently flip orientation.
        let mut topo = Topology::default();
        let c_up = Circle3D::new(origin(), z_axis(), 1.0).unwrap();
        let c_down = Circle3D::new(origin(), Vec3::new(0.0, 0.0, -1.0), 1.0).unwrap();

        let v0 = add_vertex(&mut topo, c_up.evaluate(0.0));
        let v1 = add_vertex(&mut topo, c_up.evaluate(std::f64::consts::PI));
        let v2 = add_vertex(&mut topo, c_up.evaluate(std::f64::consts::PI * 1.5));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Circle(c_up)));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Circle(c_down)));
        let wire = build_wire_from_edges(&mut topo, &[e0, e1], false);

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        assert_eq!(merged, 0);
    }

    #[test]
    fn reversed_orientation_blocks_arc_merge() {
        // Two arcs that share a circle and a vertex, but the second is
        // traversed in reverse. Merging would silently change topology
        // direction, so we leave them alone.
        let mut topo = Topology::default();
        let circle = Circle3D::new(origin(), z_axis(), 1.0).unwrap();

        let v0 = add_vertex(&mut topo, circle.evaluate(0.0));
        let v1 = add_vertex(&mut topo, circle.evaluate(std::f64::consts::PI));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Circle(circle.clone())));
        let e1 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Circle(circle)));

        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(e0, true), OrientedEdge::new(e1, false)],
                false,
            )
            .unwrap(),
        );

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        assert_eq!(merged, 0);
    }

    #[test]
    fn ellipse_arcs_on_same_ellipse_merge() {
        let mut topo = Topology::default();
        let ellipse = Ellipse3D::new(origin(), z_axis(), 4.0, 2.0).unwrap();
        let p = |t: f64| ellipse.evaluate(t);

        let v0 = add_vertex(&mut topo, p(0.0));
        let v1 = add_vertex(&mut topo, p(std::f64::consts::PI));
        let v2 = add_vertex(&mut topo, p(2.0 * std::f64::consts::PI - 1e-9));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Ellipse(ellipse.clone())));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Ellipse(ellipse.clone())));
        let wire = build_wire_from_edges(&mut topo, &[e0, e1], false);

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        assert_eq!(merged, 1);
        let new_wire = topo.wire(wire).unwrap();
        assert_eq!(new_wire.edges().len(), 1);
        assert!(matches!(
            topo.edge(new_wire.edges()[0].edge()).unwrap().curve(),
            EdgeCurve::Ellipse(_)
        ));
    }

    #[test]
    fn nurbs_curve_edges_never_merge() {
        // NurbsCurve edges are intentionally skipped — they are not handled
        // by this pass and should be left untouched even when they share a
        // vertex.
        use brepkit_math::nurbs::curve::NurbsCurve;
        let mut topo = Topology::default();
        let nurbs = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![1.0, 1.0],
        )
        .unwrap();
        let v0 = add_vertex(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let v1 = add_vertex(&mut topo, Point3::new(1.0, 0.0, 0.0));
        let v2 = add_vertex(&mut topo, Point3::new(2.0, 0.0, 0.0));
        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::NurbsCurve(nurbs.clone())));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::NurbsCurve(nurbs)));
        let wire = build_wire_from_edges(&mut topo, &[e0, e1], false);

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        assert_eq!(merged, 0);
    }

    #[test]
    fn mixed_line_and_circle_runs_handled_independently() {
        // Three collinear lines along +x ending at (3, 0, 0), then two arcs
        // on a circle centered at (4, 0, 0) starting at (3, 0, 0):
        //   L→L→L | A→A
        // Each run merges within itself; nothing crosses the kind boundary.
        let mut topo = Topology::default();
        let circle = Circle3D::new(Point3::new(4.0, 0.0, 0.0), z_axis(), 1.0).unwrap();

        let v0 = add_vertex(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let v1 = add_vertex(&mut topo, Point3::new(1.0, 0.0, 0.0));
        let v2 = add_vertex(&mut topo, Point3::new(2.0, 0.0, 0.0));
        // Frame3::from_normal gives u_axis=(0,1,0), v_axis=(-1,0,0) for
        // normal=+z, so evaluate(π/2) lands at (3, 0, 0) — the natural
        // continuation of the +x line run.
        let v3 = add_vertex(&mut topo, circle.evaluate(std::f64::consts::FRAC_PI_2));
        let v4 = add_vertex(&mut topo, circle.evaluate(std::f64::consts::PI));
        let v5 = add_vertex(&mut topo, circle.evaluate(1.49 * std::f64::consts::PI));

        let l0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let l1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let l2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let a0 = topo.add_edge(Edge::new(v3, v4, EdgeCurve::Circle(circle.clone())));
        let a1 = topo.add_edge(Edge::new(v4, v5, EdgeCurve::Circle(circle)));

        let wire = build_wire_from_edges(&mut topo, &[l0, l1, l2, a0, a1], false);

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        // 3 lines → 1 line (2 merges) + 2 arcs → 1 arc (1 merge) = 3 merges.
        assert_eq!(merged, 3);
        let new_wire = topo.wire(wire).unwrap();
        assert_eq!(new_wire.edges().len(), 2);
        let kind0 = topo.edge(new_wire.edges()[0].edge()).unwrap().curve();
        let kind1 = topo.edge(new_wire.edges()[1].edge()).unwrap().curve();
        assert!(matches!(kind0, EdgeCurve::Line));
        assert!(matches!(kind1, EdgeCurve::Circle(_)));
    }

    #[test]
    fn line_only_run_still_merges_after_refactor() {
        // Regression — the existing Line+Line behavior must survive the
        // generalization.
        let mut topo = Topology::default();
        let v0 = add_vertex(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let v1 = add_vertex(&mut topo, Point3::new(1.0, 0.0, 0.0));
        let v2 = add_vertex(&mut topo, Point3::new(2.0, 0.0, 0.0));
        let v3 = add_vertex(&mut topo, Point3::new(3.0, 0.0, 0.0));
        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let wire = build_wire_from_edges(&mut topo, &[e0, e1, e2], false);

        let merged = merge_collinear_edges(&mut topo, wire, &UnifyOptions::default()).unwrap();
        assert_eq!(merged, 2);
        assert_eq!(topo.wire(wire).unwrap().edges().len(), 1);
    }
}
