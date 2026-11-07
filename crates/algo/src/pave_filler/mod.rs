//! PaveFiller — intersection engine that builds pave blocks.
//!
//! Runs phases in two stages:
//!
//! **Stage 1 — Intersection** (reads `&Topology`, writes `&mut GfaArena`):
//! VV, VE, EE, VF, EF, FF.
//!
//! **Stage 2 — Resolution** (writes `&mut Topology` from arena data):
//! `MakeBlocks`, `MakeSplitEdges`, `MakePCurves`, `FillFaceInfo`.

pub mod fill_face_info;
pub mod force_interf_ee;
mod helpers;
pub mod link_existing;
pub mod make_blocks;
pub mod make_pcurves;
pub mod make_split_edges;
pub mod phase_ee;
pub mod phase_ef;
pub mod phase_ff;
pub mod phase_ff_coplanar;
pub mod phase_ve;
pub mod phase_vf;
pub mod phase_vv;

#[cfg(test)]
mod tests;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::ds::GfaArena;
use crate::error::AlgoError;

/// PaveFiller intersects all shape pairs between two solids,
/// building pave blocks and populating the GFA arena.
pub struct PaveFiller<'a> {
    /// The topology containing both solids.
    topo: &'a mut Topology,
    /// Solid A (first boolean argument).
    solid_a: SolidId,
    /// Solid B (second boolean argument).
    solid_b: SolidId,
    /// Tolerance for geometric comparisons.
    tol: Tolerance,
}

impl<'a> PaveFiller<'a> {
    /// Creates a new `PaveFiller` for two solids.
    #[allow(dead_code)]
    pub fn new(topo: &'a mut Topology, solid_a: SolidId, solid_b: SolidId) -> Self {
        Self {
            topo,
            solid_a,
            solid_b,
            tol: Tolerance::default(),
        }
    }

    /// Creates a `PaveFiller` with custom tolerance.
    pub fn with_tolerance(
        topo: &'a mut Topology,
        solid_a: SolidId,
        solid_b: SolidId,
        tol: Tolerance,
    ) -> Self {
        Self {
            topo,
            solid_a,
            solid_b,
            tol,
        }
    }

    /// Run intersection phases (VV through FF), populating the GFA arena.
    ///
    /// Creates new vertices in `Topology` when intersection points have
    /// no nearby existing vertex (EE, EF, FF phases).
    /// Call [`run_pave_filler`] instead to run both stages.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup or intersection fails.
    pub fn perform(&mut self, arena: &mut GfaArena) -> Result<(), AlgoError> {
        use crate::perf::gfa_time;
        self.init_pave_blocks(arena)?;

        gfa_time!(
            "phase_vv",
            phase_vv::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)
        )?;
        // VV is the only phase that registers same-domain vertices, and
        // `edge_pave_blocks` is fixed at init — so the pave-vertex coincidence
        // index is stable for the remaining phases. Build it once here instead
        // of linear-scanning every pave block per intersection endpoint.
        arena.build_pave_vertex_index(self.topo, self.tol.linear);
        gfa_time!(
            "phase_ve",
            phase_ve::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)
        )?;
        gfa_time!(
            "phase_ee",
            phase_ee::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)
        )?;
        gfa_time!(
            "phase_vf",
            phase_vf::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)
        )?;
        gfa_time!(
            "phase_ef",
            phase_ef::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)
        )?;
        gfa_time!(
            "phase_ff",
            phase_ff::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)
        )?;

        // Coplanar face splitting: parallel planes are skipped by Phase FF.
        gfa_time!(
            "phase_ff_coplanar",
            phase_ff_coplanar::perform(self.topo, self.solid_a, self.solid_b, self.tol, arena)
        )?;

        pave_co_endpoint_curves(self.topo, &[self.solid_a, self.solid_b], self.tol, arena)?;
        split_co_endpoint_sections(self.topo, self.tol, arena)?;

        Ok(())
    }

    /// Initialize pave blocks for all edges of both solids.
    fn init_pave_blocks(&self, arena: &mut GfaArena) -> Result<(), AlgoError> {
        for &solid in &[self.solid_a, self.solid_b] {
            let edges = brepkit_topology::explorer::solid_edges(self.topo, solid)?;
            for edge_id in edges {
                // Skip if already initialized (shared edges between solids)
                if arena.edge_pave_blocks.contains_key(&edge_id) {
                    continue;
                }
                let edge = self.topo.edge(edge_id)?;
                let start_pos = self.topo.vertex(edge.start())?.point();
                let end_pos = self.topo.vertex(edge.end())?.point();
                let (t0, t1) = edge_pave_span(edge.curve(), start_pos, end_pos);
                arena.init_edge_pave_block(edge_id, edge.start(), t0, edge.end(), t1);
            }
        }
        Ok(())
    }
}

/// Pave at its middle every curved edge that shares both endpoints with
/// another edge along a different path (a chord and its arc, two arcs
/// closing a circle) and that nothing else splits. The two edges may come
/// from one solid or from both (two half rings meeting along their flat
/// sides each bring half of one tube circle), so endpoints are compared
/// after the vertex merge. The assembler keys duplicate edges on their
/// endpoint pair and would weld the two into one, dropping the region
/// between them; the face splitter keeps every edge paved here in pieces
/// ([`GfaArena::co_endpoint_middles`]).
fn pave_co_endpoint_curves(
    topo: &mut Topology,
    solids: &[SolidId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use std::collections::BTreeMap;

    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::vertex::{Vertex, VertexId};

    use crate::ds::Pave;

    let mut by_ends: BTreeMap<(usize, usize), Vec<EdgeId>> = BTreeMap::new();
    for &solid in solids {
        for edge_id in brepkit_topology::explorer::solid_edges(topo, solid)? {
            let edge = topo.edge(edge_id)?;
            let (s, e) = (
                arena.resolve_vertex(edge.start()).index(),
                arena.resolve_vertex(edge.end()).index(),
            );
            let group = by_ends.entry((s.min(e), s.max(e))).or_default();
            if s != e && !group.contains(&edge_id) {
                group.push(edge_id);
            }
        }
    }
    for group in by_ends.values().filter(|g| g.len() > 1) {
        let mut middles = Vec::with_capacity(group.len());
        for &edge_id in group {
            let edge = topo.edge(edge_id)?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            let tm = 0.5 * (t0 + t1);
            let curved = !matches!(edge.curve(), EdgeCurve::Line);
            middles.push((
                edge_id,
                curved,
                tm,
                edge.curve().evaluate_with_endpoints(tm, a, b),
            ));
        }
        // Coincident copies (an edge of each operand along one path, where
        // two stacked prisms meet) are one path: the group needs pieces only
        // when it holds two. Copies whose middles agree within the vertex
        // tolerance share one middle vertex so they split alike; a copy
        // farther off gets a vertex on its own curve.
        let mut paths: Vec<(Point3, Vec<usize>)> = Vec::new();
        for (i, m) in middles.iter().enumerate() {
            match paths
                .iter_mut()
                .find(|(mid, _)| (*mid - m.3).length() <= tol.linear * 100.0)
            {
                Some((_, copies)) => copies.push(i),
                None => paths.push((m.3, vec![i])),
            }
        }
        if paths.len() < 2 {
            continue;
        }
        for (_, copies) in &paths {
            let mut placed: Vec<(VertexId, Point3)> = Vec::new();
            for &i in copies {
                let (edge_id, curved, tm, mid) = middles[i];
                let Some(&[pb_id]) = arena.edge_pave_blocks.get(&edge_id).map(Vec::as_slice) else {
                    continue;
                };
                let unsplit = arena
                    .pave_blocks
                    .get(pb_id)
                    .is_some_and(|pb| pb.extra_paves.is_empty());
                if !(curved && unsplit) {
                    continue;
                }
                let shared = placed
                    .iter()
                    .find(|(_, at)| (*at - mid).length() <= tol.linear)
                    .map(|&(vertex, _)| vertex);
                let vertex = if let Some(vertex) = shared {
                    vertex
                } else {
                    let vertex = topo.add_vertex(Vertex::new(mid, tol.linear));
                    arena.co_endpoint_middles.insert(vertex);
                    placed.push((vertex, mid));
                    vertex
                };
                arena.add_extra_pave(topo, pb_id, Pave::new(vertex, tm));
            }
        }
    }
    Ok(())
}

/// Divide at its middle every section arc other than a circle's that shares
/// both ends with another section along a different path (a plane across a half
/// ellipse prism meets its wall in the arc and its flat side in the chord). The
/// assembler keys duplicate edges on their endpoint pair and would weld the two,
/// leaving the wall bounded by the chord. A circle's arc has that split later,
/// in the face splitter, which reads its middle exactly.
#[allow(clippy::too_many_lines)]
fn split_co_endpoint_sections(
    topo: &mut Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use std::collections::HashMap;

    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;

    use crate::ds::{Interference, Pave, PaveBlock};

    struct Section {
        index: usize,
        curved: bool,
        at: f64,
        start: Point3,
        end: Point3,
        path: Vec<Point3>,
    }

    /// Ends match by distance within the band the assembler later welds them at,
    /// whatever the boolean's own tolerance: a coarser reading here would leave
    /// the pair for the weld to join.
    const WELD: f64 = 1e-5;
    /// Chords a section's path is read through.
    const PATH_SAMPLES: u32 = 32;

    let point_to_chord = |p: Point3, a: Point3, b: Point3| {
        let ab = b - a;
        let along = (p - a).dot(ab) / ab.dot(ab).max(f64::MIN_POSITIVE);
        (p - (a + ab * along.clamp(0.0, 1.0))).length()
    };
    let mut sections: Vec<Section> = Vec::new();
    for (index, curve) in arena.curves.iter().enumerate() {
        let &[pb_id] = curve.pave_blocks.as_slice() else {
            continue;
        };
        let Some(pb) = arena.pave_blocks.get(pb_id) else {
            continue;
        };
        let edge = topo.edge(pb.original_edge)?;
        let start = topo.vertex(arena.resolve_vertex(pb.start.vertex))?.point();
        let end = topo.vertex(arena.resolve_vertex(pb.end.vertex))?.point();
        if (start - end).length() <= WELD {
            continue;
        }
        let at = 0.5 * (pb.start.parameter + pb.end.parameter);
        // A line section's parameter is a distance along its raw line, not the
        // fraction its edge reads, so a path is sampled in fractions of its span.
        let is_line = matches!(edge.curve(), EdgeCurve::Line);
        let (t0, t1) = (pb.start.parameter, pb.end.parameter);
        let path: Vec<Point3> = (0..=PATH_SAMPLES)
            .map(|k| {
                let f = f64::from(k) / f64::from(PATH_SAMPLES);
                if is_line {
                    start + (end - start) * f
                } else {
                    edge.curve()
                        .evaluate_with_endpoints((t1 - t0).mul_add(f, t0), start, end)
                }
            })
            .collect();
        sections.push(Section {
            index,
            curved: matches!(
                edge.curve(),
                EdgeCurve::NurbsCurve(_) | EdgeCurve::Ellipse(_)
            ),
            at,
            start,
            end,
            path,
        });
    }
    // Candidates by their chord's middle: sections sharing ends share it to
    // within the band, and a neighbouring cell catches it across a cell's edge.
    let cell_of = |p: Point3| {
        #[allow(clippy::cast_possible_truncation)]
        (
            (p.x() / (2.0 * WELD)).floor() as i64,
            (p.y() / (2.0 * WELD)).floor() as i64,
            (p.z() / (2.0 * WELD)).floor() as i64,
        )
    };
    let middle_of = |s: &Section| s.start + (s.end - s.start) * 0.5;
    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, section) in sections.iter().enumerate() {
        cells
            .entry(cell_of(middle_of(section)))
            .or_default()
            .push(i);
    }
    let same_ends = |p: &Section, q: &Section| {
        let close = |a: Point3, b: Point3| (a - b).length() <= WELD;
        (close(p.start, q.start) && close(p.end, q.end))
            || (close(p.start, q.end) && close(p.end, q.start))
    };
    // Off the other's path anywhere: read as paths, since two copies of one path
    // need not run at the same speed, nor their samples land alike. A sample can
    // lie off the other's chords by their sag, bounded here by the sag of chords
    // twice as long.
    let leaves = |p: &Section, q: &Section| {
        let sag = q
            .path
            .windows(3)
            .step_by(2)
            .map(|w| point_to_chord(w[1], w[0], w[2]))
            .fold(0.0, f64::max);
        p.path.iter().any(|&sample| {
            q.path
                .windows(2)
                .map(|w| point_to_chord(sample, w[0], w[1]))
                .fold(f64::INFINITY, f64::min)
                > WELD.max(sag)
        })
    };
    let mut to_split: Vec<(usize, f64, Point3)> = Vec::new();
    for (i, section) in sections.iter().enumerate() {
        if !section.curved {
            continue;
        }
        let (cx, cy, cz) = cell_of(middle_of(section));
        let shares_its_ends = (-1..=1).any(|dx| {
            (-1..=1).any(|dy| {
                (-1..=1).any(|dz| {
                    cells
                        .get(&(cx + dx, cy + dy, cz + dz))
                        .is_some_and(|candidates| {
                            candidates.iter().any(|&j| {
                                j != i
                                    && same_ends(section, &sections[j])
                                    && leaves(section, &sections[j])
                            })
                        })
                })
            })
        });
        if shares_its_ends {
            let middle = section.path[section.path.len() / 2];
            to_split.push((section.index, section.at, middle));
        }
    }
    for (index, at, at_point) in to_split {
        let original = arena.curves[index].clone();
        let Some(&pb_id) = original.pave_blocks.first() else {
            continue;
        };
        let Some(pb) = arena.pave_blocks.get(pb_id) else {
            continue;
        };
        let (start, end) = (pb.start, pb.end);
        let middle = topo.add_vertex(Vertex::new(at_point, tol.linear));
        let mid_pave = Pave::new(middle, at);
        let first_edge = topo.add_edge(Edge::new(start.vertex, middle, original.curve.clone()));
        let second_edge = topo.add_edge(Edge::new(middle, end.vertex, original.curve.clone()));
        let first_pb = arena
            .pave_blocks
            .alloc(PaveBlock::new(first_edge, start, mid_pave));
        let second_pb = arena
            .pave_blocks
            .alloc(PaveBlock::new(second_edge, mid_pave, end));

        let second_index = arena.curves.len();
        let mut second = original.clone();
        second.pave_blocks = vec![second_pb];
        second.t_range = (at, original.t_range.1);
        arena.curves.push(second);
        arena.interference.ff.push(Interference::FF {
            f1: original.face_a,
            f2: original.face_b,
            curve_index: second_index,
        });
        if let Some(skipped) = arena.curve_skip_faces.get(&index).cloned() {
            arena.curve_skip_faces.insert(second_index, skipped);
        }
        let first = &mut arena.curves[index];
        first.pave_blocks = vec![first_pb];
        first.t_range = (original.t_range.0, at);
    }
    Ok(())
}

/// Run the complete PaveFiller pipeline (both stages).
///
/// **Stage 1 — Intersection** (reads `&Topology`):
/// Runs VV, VE, EE, VF, EF, FF phases to discover all interferences
/// and populate pave blocks with extra paves.
///
/// **Stage 2 — Resolution** (writes `&mut Topology`):
/// - `MakeBlocks` — splits pave blocks at extra paves
/// - `MakeSplitEdges` — creates new topology edges for leaf pave blocks
/// - `MakePCurves` — builds 2D curves on faces (stub)
/// - `FillFaceInfo` — classifies pave blocks as On/In/Sc per face
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup or intersection fails.
pub fn run_pave_filler(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use crate::perf::gfa_time;
    // Stage 1: Intersection (may create new vertices for EE/EF/FF crossings)
    {
        let mut filler = PaveFiller::with_tolerance(topo, solid_a, solid_b, tol);
        filler.perform(arena)?;
    }

    // Stage 2: Resolution (mutable Topology)
    gfa_time!("make_blocks", make_blocks::perform(topo, arena))?;
    gfa_time!(
        "force_interf_ee",
        force_interf_ee::perform(topo, tol, arena)
    )?;
    gfa_time!("link_existing", link_existing::perform(topo, tol, arena))?;
    gfa_time!("make_split_edges", make_split_edges::perform(topo, arena))?;
    gfa_time!("make_pcurves", make_pcurves::perform(topo, arena))?;
    gfa_time!("fill_face_info", fill_face_info::perform(topo, arena))?;

    Ok(())
}

/// Run the PaveFiller pipeline over **N** source solids for an N-way fuse.
///
/// The Stage-1 intersection phases are inherently pairwise (each section is
/// between the faces of two solids) and, crucially, deposit only geometric
/// split data into the shared `arena` — they carry no `Rank`. So the two-solid
/// phase code is reused verbatim: run each phase for every spatially-interacting
/// source pair into ONE arena, and the paves/sections accumulate correctly. A
/// bbox broad-phase skips non-interacting pairs, keeping the stage O(n·k) for
/// the sparse interaction graphs a fused lattice produces rather than O(n²).
///
/// Phase order preserves the two-solid pipeline's invariant that the
/// pave-vertex coincidence index is built ONCE, after all VV coincidences are
/// registered and before the phases that query it: every pair's VV runs first,
/// then the index is built, then the remaining phases run per pair. Stage-2
/// resolution (which is already solid-agnostic — it reads the accumulated arena)
/// runs once.
///
/// For `sources.len() == 2` this is behaviourally identical to
/// [`run_pave_filler`]. Cut/Intersect are unaffected; this path is fuse-only.
///
/// # Errors
///
/// Returns [`AlgoError`] if `sources` is empty or any stage fails.
pub fn run_pave_filler_n(
    topo: &mut Topology,
    sources: &[SolidId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    if sources.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "N-way pave filler needs at least one source solid".into(),
        ));
    }

    // Stage 1: Intersection over interacting pairs, accumulating into one arena.
    init_pave_blocks_n(topo, sources, arena)?;
    let pairs = interacting_pairs(topo, sources, tol);

    for &(i, j) in &pairs {
        phase_vv::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    // VV is the only phase that registers same-domain vertices and the edge
    // pave blocks are fixed at init, so the coincidence index is stable for the
    // remaining phases — build it once, after every pair's VV (mirrors the
    // two-solid `PaveFiller::perform`).
    arena.build_pave_vertex_index(topo, tol.linear);
    for &(i, j) in &pairs {
        phase_ve::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ee::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_vf::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ef::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ff::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    for &(i, j) in &pairs {
        phase_ff_coplanar::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    pave_co_endpoint_curves(topo, sources, tol, arena)?;
    split_co_endpoint_sections(topo, tol, arena)?;

    // Stage 2: Resolution (solid-agnostic — reads the accumulated arena).
    make_blocks::perform(topo, arena)?;
    force_interf_ee::perform(topo, tol, arena)?;
    link_existing::perform(topo, tol, arena)?;
    make_split_edges::perform(topo, arena)?;
    make_pcurves::perform(topo, arena)?;
    fill_face_info::perform(topo, arena)?;

    Ok(())
}

/// Initialize a pave block for every edge across all `sources`, de-duplicating
/// edges shared between solids (coincident walls). Mirrors
/// [`PaveFiller::init_pave_blocks`] but over N solids.
fn init_pave_blocks_n(
    topo: &Topology,
    sources: &[SolidId],
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    for &solid in sources {
        for edge_id in brepkit_topology::explorer::solid_edges(topo, solid)? {
            if arena.edge_pave_blocks.contains_key(&edge_id) {
                continue;
            }
            let edge = topo.edge(edge_id)?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge_pave_span(edge.curve(), start_pos, end_pos);
            arena.init_edge_pave_block(edge_id, edge.start(), t0, edge.end(), t1);
        }
    }
    Ok(())
}

/// An edge's pave span, start vertex first. A NURBS edge on its whole curve
/// reports the curve's own domain in either orientation, so one stored
/// against its curve starts at the domain's far end and its span descends.
fn edge_pave_span(
    curve: &brepkit_topology::edge::EdgeCurve,
    start: brepkit_math::vec::Point3,
    end: brepkit_math::vec::Point3,
) -> (f64, f64) {
    // The weld band within which an edge's vertices sit on its curve's ends:
    // `EdgeCurve::domain_with_endpoints` (topology's edge.rs) accepts on-curve
    // endpoints within the same `WELD_EPS`, and reports the whole domain for
    // ends within its tighter `END_EPS`, which this band contains.
    const ON_END: f64 = 1e-5;
    let (t0, t1) = curve.domain_with_endpoints(start, end);
    if matches!(curve, brepkit_topology::edge::EdgeCurve::NurbsCurve(_))
        && (start - end).length() > 1e-9
        && (curve.evaluate_with_endpoints(t1, start, end) - start).length() < ON_END
        && (curve.evaluate_with_endpoints(t0, start, end) - end).length() < ON_END
    {
        return (t1, t0);
    }
    (t0, t1)
}

/// Axis-aligned bounding box of a solid, curved edges included: a cylinder's
/// only vertices sit on its seam.
fn solid_aabb(topo: &Topology, solid: SolidId) -> Option<brepkit_math::aabb::Aabb3> {
    crate::classifier::compute_solid_bbox(topo, solid).ok()
}

/// Source-index pairs `(i, j)` with `i < j` whose bounding boxes overlap (each
/// expanded by tolerance so coincident-wall pairs are not missed). Two solids
/// with disjoint boxes cannot intersect, so pruning them is result-preserving.
/// A solid without a computable box is treated as interacting with all others
/// (conservative — never drops a real interaction).
fn interacting_pairs(topo: &Topology, sources: &[SolidId], tol: Tolerance) -> Vec<(usize, usize)> {
    let boxes: Vec<Option<brepkit_math::aabb::Aabb3>> =
        sources.iter().map(|&s| solid_aabb(topo, s)).collect();
    let mut pairs = Vec::new();
    for i in 0..sources.len() {
        for j in (i + 1)..sources.len() {
            let interact = match (boxes[i], boxes[j]) {
                (Some(bi), Some(bj)) => bi.expanded(tol.linear).intersects(bj.expanded(tol.linear)),
                _ => true,
            };
            if interact {
                pairs.push((i, j));
            }
        }
    }
    pairs
}
