//! Unify same-domain faces — merge adjacent faces sharing the same
//! underlying surface.
//!
//! This is the most impactful healing operation in production.  After
//! boolean operations, a box may have 72 faces instead of 6 because
//! intersection curves split each original face.  `unify_same_domain`
//! detects adjacent faces on the same plane/cylinder/cone/sphere/torus
//! and merges them back, dramatically reducing face count.

use std::collections::{HashMap, HashSet};

use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
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

// ── Union-Find ──────────────────────────────────────────────────────

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

    // 1. Build edge → face adjacency.
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

    // 2. Snapshot face surfaces for comparison.
    let face_surfaces: Vec<_> = face_ids
        .iter()
        .map(|&fid| topo.face(fid).map(|f| f.surface().clone()))
        .collect::<Result<Vec<_>, _>>()?;

    // 3. Union-find: group adjacent faces with equivalent surfaces.
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

    // 4. Group faces by root.
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

    // 5. Merge each group.
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
            // touching a seam cannot be classified in UV yet).
            if has_holes && !surface.is_planar() {
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

    // 6. Rebuild shell.
    if total_merged > 0 {
        let shell = topo.shell(shell_id)?;
        let current_faces: Vec<FaceId> = shell.faces().to_vec();

        let mut final_faces: Vec<FaceId> = current_faces
            .into_iter()
            .filter(|f| !faces_to_remove.contains(f))
            .collect();
        final_faces.extend(&new_faces_to_add);

        let new_shell = Shell::new(final_faces)?;
        let shell_mut = topo.shell_mut(shell_id)?;
        *shell_mut = new_shell;
    }

    // 7. Merge collinear adjacent edges in the newly created wires.
    let mut total_edges_merged = 0;
    if options.unify_edges && total_merged > 0 {
        for &fid in &new_faces_to_add {
            let outer_wire = topo.face(fid)?.outer_wire();
            let merged = merge_collinear_edges(topo, outer_wire, options)?;
            total_edges_merged += merged;
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

/// A surviving edge with its traversal endpoints and 2D outgoing tangent.
struct SurvEdge {
    oe: OrientedEdge,
    start: brepkit_topology::vertex::VertexId,
    end: brepkit_topology::vertex::VertexId,
    start_uv: (f64, f64),
    tangent_uv: (f64, f64),
}

/// A reassembled closed loop with its UV polygon and signed area.
struct LoopInfo {
    edge_indices: Vec<usize>,
    uv_points: Vec<(f64, f64)>,
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
        if (ep - sp).length() < lin && start == end {
            continue;
        }
        let start_uv = to_uv(sp);
        let end_uv = to_uv(ep);
        let dx = end_uv.0 - start_uv.0;
        let dy = end_uv.1 - start_uv.1;
        let len = dx.hypot(dy);
        let tangent_uv = if len < f64::EPSILON {
            (1.0, 0.0)
        } else {
            (dx / len, dy / len)
        };
        edges.push(SurvEdge {
            oe: *oe,
            start,
            end,
            start_uv,
            tangent_uv,
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
            let in_dir = (-edges[current].tangent_uv.0, -edges[current].tangent_uv.1);
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
    let mut infos: Vec<LoopInfo> = Vec::with_capacity(loops.len());
    for loop_edges in loops {
        let uv_points: Vec<(f64, f64)> = loop_edges.iter().map(|&i| edges[i].start_uv).collect();
        let signed_area = polygon_signed_area(&uv_points);
        infos.push(LoopInfo {
            edge_indices: loop_edges,
            uv_points,
            signed_area,
        });
    }

    // A loop is an outer region if no other loop contains it. Containment is by
    // point-in-polygon of a representative interior sample.
    let n_loops = infos.len();
    let mut is_outer = vec![true; n_loops];
    for i in 0..n_loops {
        let Some(sample) = loop_interior_point(&infos[i].uv_points) else {
            return Ok(None);
        };
        for j in 0..n_loops {
            if i == j {
                continue;
            }
            if point_in_polygon(sample, &infos[j].uv_points) {
                is_outer[i] = false;
                break;
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
        let Some(sample) = loop_interior_point(&infos[h].uv_points) else {
            return Ok(None);
        };
        let mut owner: Option<usize> = None;
        for &o in &outer_indices {
            if point_in_polygon(sample, &infos[o].uv_points) {
                owner = Some(o);
                break;
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

    // An outer loop bounds material on its left → positive signed UV area for a
    // non-reversed face (flipped when the face inherits a reversed orientation).
    // Holes carry the opposite sign. Reverse a loop's edge run when its current
    // sign disagrees.
    let outer_positive = !reversed;
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
fn signed_turn(a: (f64, f64), b: (f64, f64)) -> f64 {
    let cross = a.0 * b.1 - a.1 * b.0;
    let dot = a.0 * b.0 + a.1 * b.1;
    cross.atan2(dot)
}

/// Shoelace signed area of a closed UV polygon.
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

/// A representative interior point of a UV polygon (centroid; falls back to the
/// midpoint of the first edge for non-convex shapes by nudging inward).
fn loop_interior_point(pts: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pts.len() < 3 {
        return None;
    }
    // Midpoint of the first edge nudged toward the polygon centroid keeps the
    // sample strictly inside even for non-convex loops.
    let cx: f64 = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
    let cy: f64 = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
    let mid = ((pts[0].0 + pts[1].0) * 0.5, (pts[0].1 + pts[1].1) * 0.5);
    let sample = (mid.0 * 0.001 + cx * 0.999, mid.1 * 0.001 + cy * 0.999);
    Some(sample)
}

/// Even-odd point-in-polygon test in UV.
fn point_in_polygon(p: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let (px, py) = p;
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        let intersects = ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
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
