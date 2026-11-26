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
mod nurbs_trim;
pub mod phase_ee;
pub mod phase_ef;
pub mod phase_ff;
pub mod phase_ff_coplanar;
pub mod phase_ve;
pub mod phase_vf;
pub mod phase_vv;
mod tool_edges;

#[cfg(test)]
mod tests;

use brepkit_math::madd::Madd;
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

        split_sections_at_boundary_crossings(self.topo, self.tol, arena)?;
        pave_co_endpoint_curves(self.topo, &[self.solid_a, self.solid_b], self.tol, arena)?;
        pave_chords_across_conics(self.topo, self.tol, arena)?;
        split_co_endpoint_sections(self.topo, self.tol, arena)?;
        split_sections_at_rim_vertices(self.topo, self.tol, arena)?;

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
        // when it holds two. An edge is a copy of a path's first edge only
        // where the two are proved one curve within 100 tolerances along
        // their whole length; a circle's and an ellipse's arcs through the
        // same ends and middle are two paths. Copies whose middles agree
        // within the vertex tolerance share one middle vertex so they split
        // alike; a copy farther off gets a vertex on its own curve.
        let one_path = |x: EdgeId, y: EdgeId| -> bool {
            let read = |e: EdgeId| -> Option<(&EdgeCurve, Point3, Point3)> {
                let edge = topo.edge(e).ok()?;
                Some((
                    edge.curve(),
                    topo.vertex(edge.start()).ok()?.point(),
                    topo.vertex(edge.end()).ok()?.point(),
                ))
            };
            match (read(x), read(y)) {
                (Some(a), Some(b)) => {
                    crate::builder::face_splitter::curves_one_curve(a, b, tol.linear * 100.0)
                }
                _ => false,
            }
        };
        let mut paths: Vec<(Point3, Vec<usize>)> = Vec::new();
        for (i, m) in middles.iter().enumerate() {
            match paths
                .iter_mut()
                .find(|(_, copies)| one_path(middles[copies[0]].0, m.0))
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
                // A pave at one of the block's own ends (a crossing with
                // another edge there) splits nothing: the copy is whole, and
                // left so it would not split as its twin does.
                let unsplit = arena.pave_blocks.get(pb_id).is_some_and(|pb| {
                    let ends = [
                        arena.resolve_vertex(pb.start.vertex),
                        arena.resolve_vertex(pb.end.vertex),
                    ];
                    pb.extra_paves
                        .iter()
                        .all(|pave| ends.contains(&arena.resolve_vertex(pave.vertex)))
                });
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

/// Pave at its middle each straight edge's piece between two of its crossings
/// with one circle or ellipse edge, where that conic's piece between the same
/// two points is whole (a slab's bottom edge across the round hole in the
/// plate it stands on). The piece is a chord of the conic: it runs through the
/// conic's inside, so it is never the arc, but the assembler keys duplicate
/// edges on their endpoint pair and would weld the two into one edge on four
/// faces. Only crossings inside the line count: a straight edge that ends on
/// the conic can stand for its arc. The face splitter keeps the line in
/// pieces ([`GfaArena::co_endpoint_middles`]).
fn pave_chords_across_conics(
    topo: &mut Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use std::collections::BTreeMap;

    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::vertex::{Vertex, VertexId};

    use crate::ds::{Interference, Pave};

    /// Points closer than the band the assembler welds at are one point, and
    /// a chord within it of its arc is that arc.
    const WELD: f64 = 1e-5;

    let mut crossings: BTreeMap<(EdgeId, EdgeId), Vec<VertexId>> = BTreeMap::new();
    for interference in &arena.interference.ee {
        let Interference::EE {
            e1,
            e2,
            new_vertex: Some(vertex),
            ..
        } = interference
        else {
            continue;
        };
        let conic = |c: &EdgeCurve| matches!(c, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_));
        let key = match (topo.edge(*e1)?.curve(), topo.edge(*e2)?.curve()) {
            (EdgeCurve::Line, c) if conic(c) => (*e1, *e2),
            (c, EdgeCurve::Line) if conic(c) => (*e2, *e1),
            _ => continue,
        };
        crossings.entry(key).or_default().push(*vertex);
    }

    for ((line_id, conic_id), vertices) in crossings {
        if vertices.len() < 2 {
            continue;
        }
        let Some(&[pb_id]) = arena.edge_pave_blocks.get(&line_id).map(Vec::as_slice) else {
            continue;
        };
        let Some(pb) = arena.pave_blocks.get(pb_id) else {
            continue;
        };
        let line = topo.edge(line_id)?;
        let start = topo.vertex(line.start())?.point();
        let seg = topo.vertex(line.end())?.point() - start;
        let len = seg.length();
        if len <= WELD {
            continue;
        }
        let along = |p: Point3| (p - start).dot(seg) / (len * len);
        let (t0, t1) = (pb.start.parameter, pb.end.parameter);
        let mut breaks: Vec<f64> = vec![0.0, 1.0];
        for pave in &pb.extra_paves {
            breaks.push(along(
                topo.vertex(arena.resolve_vertex(pave.vertex))?.point(),
            ));
        }
        let mut points: Vec<(f64, Point3)> = Vec::with_capacity(vertices.len());
        for vertex in vertices {
            let p = topo.vertex(arena.resolve_vertex(vertex))?.point();
            points.push((along(p), p));
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        points.dedup_by(|b, a| (b.1 - a.1).length() <= WELD);
        let band = WELD / len;

        let mut middles: Vec<(f64, Point3)> = Vec::new();
        for w in points.windows(2) {
            let ((fa, pa), (fb, pq)) = (w[0], w[1]);
            if fa <= band || fb >= 1.0 - band {
                continue;
            }
            if breaks.iter().any(|&f| f > fa + band && f < fb - band) {
                continue;
            }
            let Some(arc_middle) = whole_conic_arc_middle(topo, arena, conic_id, pa, pq, WELD)?
            else {
                continue;
            };
            let f = 0.5 * (fa + fb);
            let middle = start + seg * f;
            if (arc_middle - middle).length() > WELD {
                middles.push(((t1 - t0).madd(f, t0), middle));
            }
        }
        for (t, at) in middles {
            let vertex = topo.add_vertex(Vertex::new(at, tol.linear));
            arena.co_endpoint_middles.insert(vertex);
            arena.add_extra_pave(topo, pb_id, Pave::new(vertex, t));
        }
    }
    Ok(())
}

/// The middle of an arc of a circle or ellipse edge running from `a` to `b`
/// with no pave or end of the edge strictly inside it, if there is one.
fn whole_conic_arc_middle(
    topo: &Topology,
    arena: &GfaArena,
    conic_id: brepkit_topology::edge::EdgeId,
    a: brepkit_math::vec::Point3,
    b: brepkit_math::vec::Point3,
    weld: f64,
) -> Result<Option<brepkit_math::vec::Point3>, AlgoError> {
    use std::f64::consts::TAU;

    use brepkit_topology::edge::EdgeCurve;

    let edge = topo.edge(conic_id)?;
    let (cs, ce) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let curve = edge.curve();
    if !matches!(curve, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
        return Ok(None);
    }
    let project = |p| match curve {
        EdgeCurve::Circle(c) => Some(c.project(p)),
        EdgeCurve::Ellipse(e) => Some(e.project(p)),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
    };
    let evaluate = |t| match curve {
        EdgeCurve::Circle(c) => c.evaluate(t),
        EdgeCurve::Ellipse(e) => e.evaluate(t),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => cs,
    };
    let (t0, t1) = curve.domain_with_endpoints(cs, ce);
    let span = t1 - t0;
    let closed = span >= TAU - 1e-9;
    let rel = |p| project(p).map_or(0.0, |a| (a - t0).rem_euclid(TAU));
    let (ra, rb) = (rel(a), rel(b));
    let (lo, hi) = (ra.min(rb), ra.max(rb));

    let mut breaks: Vec<brepkit_math::vec::Point3> = vec![cs, ce];
    for &pb_id in arena.edge_pave_blocks.get(&conic_id).into_iter().flatten() {
        if let Some(pb) = arena.pave_blocks.get(pb_id) {
            for pave in &pb.extra_paves {
                breaks.push(topo.vertex(arena.resolve_vertex(pave.vertex))?.point());
            }
        }
    }
    let inside: Vec<f64> = breaks
        .into_iter()
        .filter(|&p| (p - a).length() > weld && (p - b).length() > weld)
        .map(rel)
        .collect();
    if hi <= span && !inside.iter().any(|&r| r > lo && r < hi) {
        return Ok(Some(evaluate(t0 + 0.5 * (lo + hi))));
    }
    if closed && !inside.iter().any(|&r| r > hi || r < lo) {
        return Ok(Some(evaluate(t0 + 0.5 * (hi + lo + TAU))));
    }
    Ok(None)
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
                        .evaluate_with_endpoints((t1 - t0).madd(f, t0), start, end)
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

/// Split the sections at the points where the faces of three solids meet.
///
/// Two sections on one face, from partner faces of two other solids, cross
/// where the three faces meet, and the third pair's section runs through the
/// same point. A pairwise boolean meets that point where an edge of its
/// first result crosses the third solid's face; a single pass over three or
/// more solids has no such edge. Each section is cut there at one vertex the
/// three share, so the faces split along whole pieces that meet end to end.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
#[allow(clippy::too_many_lines)]
fn split_sections_at_triple_points(
    topo: &mut Topology,
    sources: &[SolidId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use std::collections::{BTreeMap, HashMap};

    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::FaceId;
    use brepkit_topology::vertex::{Vertex, VertexId};

    use crate::ds::{Interference, Pave, PaveBlock};

    /// The band two copies of one point are welded at downstream; a crossing
    /// this near a section's end is that end.
    const WELD: f64 = 1e-5;

    /// How a section's edge parameter maps to its paves'.
    enum Reading {
        /// The fraction of the span.
        Fraction,
        /// An angle, in any turn.
        Angle,
        /// The curve's own parameter.
        Own,
    }
    struct Section {
        index: usize,
        edge: brepkit_topology::edge::EdgeId,
        start: Point3,
        end: Point3,
        params: (f64, f64),
        reading: Reading,
        /// A box proven to hold the section, where one can be read.
        bbox: Option<brepkit_math::aabb::Aabb3>,
        /// A whole loop, from its one vertex round to it.
        closed: bool,
        /// The vertices at its start and its end.
        ends: [VertexId; 2],
        curve: EdgeCurve,
    }
    /// A section a crossing lies on, where the crossing cuts it, and its
    /// vertex there where the crossing is at one of its ends.
    type Side = (usize, Option<f64>, Option<VertexId>);
    /// The vertex for the crossings at one place, the point it stands for,
    /// and the sections those crossings lie on.
    struct Place {
        at: Point3,
        vertex: Option<VertexId>,
        sections: Vec<usize>,
    }

    if sources.len() < 3 {
        return Ok(());
    }
    let mut source_of: HashMap<FaceId, usize> = HashMap::new();
    for (i, &solid) in sources.iter().enumerate() {
        for face in brepkit_topology::explorer::solid_faces(topo, solid)? {
            source_of.insert(face, i);
        }
    }

    let mut sections: Vec<Section> = Vec::new();
    let mut on_face: BTreeMap<FaceId, Vec<usize>> = BTreeMap::new();
    for (index, curve) in arena.curves.iter().enumerate() {
        let &[pb_id] = curve.pave_blocks.as_slice() else {
            continue;
        };
        let Some(pb) = arena.pave_blocks.get(pb_id) else {
            continue;
        };
        let edge = topo.edge(pb.original_edge)?;
        let start = topo.vertex(edge.start())?.point();
        let end = topo.vertex(edge.end())?.point();
        // Ends that meet bound a whole loop (a circle round a wall) when the
        // curve turns between them, and a section of no length otherwise.
        let closed = (edge.start() == edge.end() || (start - end).length() < 1e-9)
            && !matches!(edge.curve(), EdgeCurve::Line)
            && (pb.end.parameter - pb.start.parameter).abs() > 1e-9;
        if (start - end).length() <= WELD && !closed {
            continue;
        }
        let reading = match edge.curve() {
            EdgeCurve::Line => Reading::Fraction,
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => Reading::Angle,
            EdgeCurve::NurbsCurve(_) => Reading::Own,
        };
        let k = sections.len();
        sections.push(Section {
            index,
            edge: pb.original_edge,
            start,
            end,
            params: (pb.start.parameter, pb.end.parameter),
            reading,
            bbox: phase_ee::proven_span_box(edge.curve(), start, end).map(|b| b.expanded(WELD)),
            closed,
            ends: [pb.start.vertex, pb.end.vertex],
            curve: edge.curve().clone(),
        });
        for face in [curve.face_a, curve.face_b] {
            on_face.entry(face).or_default().push(k);
        }
    }

    // The section's pave parameter of a crossing read along its edge: a line
    // edge reads the fraction of its span, a circle or an ellipse its angle
    // in any turn.
    let pave_parameter = |s: &Section, t: f64| -> Option<f64> {
        let (p0, p1) = s.params;
        let (lo, hi) = (p0.min(p1), p0.max(p1));
        let at = match s.reading {
            Reading::Fraction => t.madd(p1 - p0, p0),
            Reading::Angle => lo + (t - lo).rem_euclid(std::f64::consts::TAU),
            Reading::Own => t,
        };
        (lo < at && at < hi).then_some(at)
    };
    // How far `q` lies from a section's curve.
    let off_curve = |s: &Section, q: Point3| -> f64 {
        match &s.curve {
            EdgeCurve::Line => {
                let d = s.end - s.start;
                let w = q - s.start;
                (w - d * (w.dot(d) / d.dot(d).max(f64::MIN_POSITIVE))).length()
            }
            EdgeCurve::Circle(c) => (c.evaluate(c.project(q)) - q).length(),
            EdgeCurve::Ellipse(e) => (e.evaluate(e.project(q)) - q).length(),
            EdgeCurve::NurbsCurve(n) => {
                brepkit_math::nurbs::projection::project_point_to_curve(n, q, 1e-12)
                    .map_or(f64::INFINITY, |r| r.distance)
            }
        }
    };
    let on_all =
        |q: Point3, ks: &[usize]| ks.iter().all(|&k| off_curve(&sections[k], q) <= tol.linear);

    // Each crossing with, for both sections, where it cuts the section (a
    // crossing inside its span) and the section's vertex there (a crossing
    // within the tolerance of one of its ends).
    let mut hits: Vec<(Point3, [Side; 2])> = Vec::new();
    for (&face, here) in &on_face {
        let Some(&own) = source_of.get(&face) else {
            continue;
        };
        let partner = |k: usize| {
            let curve = &arena.curves[sections[k].index];
            let other = if curve.face_a == face {
                curve.face_b
            } else {
                curve.face_a
            };
            source_of.get(&other).copied()
        };
        for (n, &i) in here.iter().enumerate() {
            for &j in &here[n + 1..] {
                let (Some(si), Some(sj)) = (partner(i), partner(j)) else {
                    continue;
                };
                if si == sj || si == own || sj == own {
                    continue;
                }
                let (a, b) = (&sections[i], &sections[j]);
                if let (Some(box_a), Some(box_b)) = (a.bbox, b.bbox)
                    && !box_a.intersects(box_b)
                {
                    continue;
                }
                for (ta, tb, p) in phase_ee::edge_crossings(topo, a.edge, b.edge, tol)? {
                    let side = |k: usize, s: &Section, t: f64| -> Side {
                        let end = if (p - s.start).length() <= tol.linear {
                            Some(s.ends[0])
                        } else if (p - s.end).length() <= tol.linear {
                            Some(s.ends[1])
                        } else {
                            None
                        };
                        let cut = if end.is_none() {
                            pave_parameter(s, t)
                        } else {
                            None
                        };
                        (k, cut, end)
                    };
                    let sides = [side(i, a, ta), side(j, b, tb)];
                    if sides.iter().any(|&(_, cut, _)| cut.is_some()) {
                        hits.push((p, sides));
                    }
                }
            }
        }
    }

    // One vertex for the crossings at one place. A crossing joins a place
    // whose point lies near it and within the tolerance of both its curves
    // (two curves can cross more than once), and a
    // section's own end vertex stands for the place only where it lies
    // within the tolerance of every curve there; else the place keeps a new
    // vertex at the refined crossing. A crossing within the tolerance of a
    // section's end never cuts that section, so no piece is shorter than the
    // tolerance.
    let mut places: Vec<Place> = Vec::new();
    let mut place_of: Vec<usize> = Vec::with_capacity(hits.len());
    for &(p, sides) in &hits {
        let ks = [sides[0].0, sides[1].0];
        let place = if let Some(k) = places
            .iter()
            .position(|place| (place.at - p).length() <= WELD && on_all(place.at, &ks))
        {
            places[k].sections.extend(ks);
            k
        } else {
            places.push(Place {
                at: p,
                vertex: None,
                sections: ks.to_vec(),
            });
            places.len() - 1
        };
        if places[place].vertex.is_none() {
            for end in sides.iter().filter_map(|&(_, _, end)| end) {
                let q = topo.vertex(end)?.point();
                if on_all(q, &places[place].sections) {
                    places[place].vertex = Some(end);
                    places[place].at = q;
                    break;
                }
            }
        }
        place_of.push(place);
    }
    let vertices: Vec<VertexId> = places
        .iter()
        .map(|place| {
            place
                .vertex
                .unwrap_or_else(|| topo.add_vertex(Vertex::new(place.at, tol.linear)))
        })
        .collect();
    let mut cuts: BTreeMap<usize, Vec<(f64, VertexId)>> = BTreeMap::new();
    for ((_, sides), &place) in hits.iter().zip(&place_of) {
        let vertex = vertices[place];
        for &(k, cut, _) in sides {
            let Some(at) = cut else {
                continue;
            };
            let list = cuts.entry(k).or_default();
            if !list.iter().any(|&(_, v)| v == vertex) {
                list.push((at, vertex));
            }
        }
    }

    for (k, mut list) in cuts {
        let index = sections[k].index;
        let original = arena.curves[index].clone();
        let Some(&pb_id) = original.pave_blocks.first() else {
            continue;
        };
        let Some(pb) = arena.pave_blocks.get(pb_id) else {
            continue;
        };
        let (start, end) = (pb.start, pb.end);
        let ascending = start.parameter <= end.parameter;
        list.sort_by(|x, y| {
            if ascending {
                x.0.total_cmp(&y.0)
            } else {
                y.0.total_cmp(&x.0)
            }
        });
        let mut cut_at = vec![start];
        cut_at.extend(list.into_iter().map(|(at, v)| Pave::new(v, at)));
        cut_at.push(end);
        // A loop's pieces each run less than half its turn: an open arc of a
        // circle or an ellipse is read the shorter way round between its
        // ends, and two pieces of a loop cut once would share both ends.
        let turn = (end.parameter - start.parameter).abs();
        let mut paves = vec![start];
        for w in cut_at.windows(2) {
            let span = w[1].parameter - w[0].parameter;
            if sections[k].closed {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let parts = (span.abs() / (0.499 * turn)).ceil().max(1.0) as u32;
                for m in 1..parts {
                    let t = span.madd(f64::from(m) / f64::from(parts), w[0].parameter);
                    let p = original.curve.evaluate_with_endpoints(
                        t,
                        sections[k].start,
                        sections[k].end,
                    );
                    paves.push(Pave::new(topo.add_vertex(Vertex::new(p, tol.linear)), t));
                }
            }
            paves.push(w[1]);
        }
        for (n, w) in paves.windows(2).enumerate() {
            let edge = topo.add_edge(Edge::new(w[0].vertex, w[1].vertex, original.curve.clone()));
            let piece = arena.pave_blocks.alloc(PaveBlock::new(edge, w[0], w[1]));
            let t_range = (w[0].parameter, w[1].parameter);
            if n == 0 {
                let first = &mut arena.curves[index];
                first.pave_blocks = vec![piece];
                first.t_range = t_range;
                continue;
            }
            let next = arena.curves.len();
            let mut curve = original.clone();
            curve.pave_blocks = vec![piece];
            curve.t_range = t_range;
            arena.curves.push(curve);
            arena.interference.ff.push(Interference::FF {
                f1: original.face_a,
                f2: original.face_b,
                curve_index: next,
            });
            if let Some(skipped) = arena.curve_skip_faces.get(&index).cloned() {
                arena.curve_skip_faces.insert(next, skipped);
            }
            if let Some(extra) = arena.curve_extra_faces.get(&index).cloned() {
                arena.curve_extra_faces.insert(next, extra);
            }
        }
    }
    Ok(())
}

/// Split each open circle section at the vertices of the boundary arcs of
/// its two faces that run along its circle, inside its span. A section along
/// a chain of rim arcs (a cone's rim on a coaxial wall, broken where two
/// faces of the part's back meet it) then runs arc for arc with the chain,
/// and each piece links to the boundary edge it lies on instead of standing
/// beside the chain as a second copy of it.
fn split_sections_at_rim_vertices(
    topo: &mut Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use std::f64::consts::TAU;

    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::FaceId;
    use brepkit_topology::vertex::VertexId;

    use crate::ds::{Interference, Pave, PaveBlock};

    let near = tol.linear * 10.0;
    for index in 0..arena.curves.len() {
        let original = arena.curves[index].clone();
        let EdgeCurve::Circle(circle) = &original.curve else {
            continue;
        };
        let &[pb_id] = original.pave_blocks.as_slice() else {
            continue;
        };
        let Some(pb) = arena.pave_blocks.get(pb_id) else {
            continue;
        };
        let (start, end) = (pb.start, pb.end);
        let (sv, ev) = (
            arena.resolve_vertex(start.vertex),
            arena.resolve_vertex(end.vertex),
        );
        let (t0, t1) = (start.parameter, end.parameter);
        let whole = sv == ev && (t1 - t0 - TAU).abs() <= 1e-9;
        if (sv == ev && !whole) || t1 - t0 <= 0.0 || t1 - t0 > TAU + 1e-9 {
            continue;
        }
        let along = |p: brepkit_math::vec::Point3| {
            let t = circle.project(p);
            (t - t0).rem_euclid(TAU) + t0
        };
        let ang = near / circle.radius();
        // Each candidate carries the rank MakeBlocks gives it on its rim: a
        // block's own ends stay, and of waiting paves that coincide it keeps
        // the first along the block. Coincident candidates here take the
        // same one, by the same test (`make_blocks::same_pave`), so that each
        // cut is a vertex of the rim's pieces.
        let mut candidates: Vec<(VertexId, usize)> = Vec::new();
        for face_id in [original.face_a, original.face_b] {
            let face = topo.face(face_id)?;
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wid)?.edges() {
                    let edge = topo.edge(oe.edge())?;
                    let EdgeCurve::Circle(rim) = edge.curve() else {
                        continue;
                    };
                    if (rim.radius() - circle.radius()).abs() > near
                        || (rim.center() - circle.center()).length() > near
                        || rim.normal().cross(circle.normal()).length() > 1e-9
                    {
                        continue;
                    }
                    candidates.extend([(edge.start(), 0), (edge.end(), 0)]);
                    // The rim's blocks split only in MakeBlocks, after this
                    // pass: a crossing found by EE or EF still waits among a
                    // block's extra paves.
                    if let Some(blocks) = arena.edge_pave_blocks.get(&oe.edge()) {
                        for leaf in arena.collect_leaf_pave_blocks(blocks) {
                            let Some(piece) = arena.pave_blocks.get(leaf) else {
                                continue;
                            };
                            candidates.extend(
                                make_blocks::piece_vertices(topo, arena, piece)
                                    .into_iter()
                                    .enumerate()
                                    .map(|(k, vertex)| (vertex, usize::from(k >= 2))),
                            );
                        }
                    }
                }
            }
        }
        let linear = tol.linear;
        let same = |topo: &Topology, a: VertexId, b: VertexId| {
            a == b
                || matches!(
                    (topo.vertex(a), topo.vertex(b)),
                    (Ok(p), Ok(q)) if (p.point() - q.point()).length() <= linear
                )
        };
        let mut cuts: Vec<(f64, VertexId, usize)> = Vec::new();
        for (vertex, rank) in candidates {
            let vid = arena.resolve_vertex(vertex);
            if same(topo, vid, sv) || same(topo, vid, ev) {
                continue;
            }
            let p = topo.vertex(vid)?.point();
            if (circle.evaluate(circle.project(p)) - p).length() > near {
                continue;
            }
            let t = along(p);
            if t <= t0 + ang || t >= t1 - ang {
                continue;
            }
            match cuts.iter_mut().find(|c| same(topo, c.1, vid)) {
                Some(kept) => {
                    if (rank, vid.index()) < (kept.2, kept.1.index()) {
                        *kept = (t, vid, rank);
                    }
                }
                None => cuts.push((t, vid, rank)),
            }
        }
        let mut cuts: Vec<(f64, VertexId)> = cuts.into_iter().map(|(t, v, _)| (t, v)).collect();
        if cuts.is_empty() {
            continue;
        }
        cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut paves = vec![start];
        paves.extend(cuts.iter().map(|&(t, vid)| Pave::new(vid, t)));
        paves.push(end);
        // A whole circle splits only into arcs under half a turn, at least
        // three of them: an open arc's pcurve runs the short way round, and
        // two arcs between the same two vertices would weld into one edge.
        // An arc a hair under half a turn (cut at two vertices opposite each
        // other) reads either way round too.
        if whole
            && (paves.len() < 4
                || paves
                    .windows(2)
                    .any(|w| w[1].parameter - w[0].parameter >= std::f64::consts::PI - 1e-9))
        {
            continue;
        }
        // A piece from one boundary vertex of a face to another that runs
        // off that face's boundary gets a vertex at its middle. The boundary
        // joins the two ends along another path, and a reader that takes the
        // piece by its chord or by its ends sees that path: the hole weave
        // leaves out an arc whose chord lies along a hole's straight side,
        // and the assembler welds two edges between the same two vertices.
        let mut rims: Vec<(FaceId, Vec<VertexId>)> = Vec::new();
        for face_id in [original.face_a, original.face_b] {
            let face = topo.face(face_id)?;
            let mut vertices = Vec::new();
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wid)?.edges() {
                    let edge = topo.edge(oe.edge())?;
                    vertices.extend([edge.start(), edge.end()]);
                    if let Some(blocks) = arena.edge_pave_blocks.get(&oe.edge()) {
                        for leaf in arena.collect_leaf_pave_blocks(blocks) {
                            if let Some(piece) = arena.pave_blocks.get(leaf) {
                                vertices.extend([piece.start.vertex, piece.end.vertex]);
                                vertices.extend(piece.extra_paves.iter().map(|pave| pave.vertex));
                            }
                        }
                    }
                }
            }
            let mut resolved: Vec<VertexId> = vertices
                .into_iter()
                .map(|v| arena.resolve_vertex(v))
                .collect();
            resolved.sort_unstable_by_key(|v| v.index());
            resolved.dedup();
            rims.push((face_id, resolved));
        }
        let mut with_middles: Vec<Pave> = Vec::with_capacity(2 * paves.len());
        for w in paves.windows(2) {
            with_middles.push(w[0]);
            let (a, b) = (
                arena.resolve_vertex(w[0].vertex),
                arena.resolve_vertex(w[1].vertex),
            );
            let tm = 0.5 * (w[0].parameter + w[1].parameter);
            let middle = circle.evaluate(tm);
            let on_rim = |vertices: &[VertexId], v: VertexId| {
                vertices
                    .binary_search_by_key(&v.index(), |r| r.index())
                    .is_ok()
            };
            let across = rims.iter().any(|(face_id, vertices)| {
                on_rim(vertices, a)
                    && on_rim(vertices, b)
                    && !phase_ff::point_on_face_edges(topo, *face_id, middle, near)
            });
            if across {
                let vid =
                    topo.add_vertex(brepkit_topology::vertex::Vertex::new(middle, tol.linear));
                with_middles.push(Pave::new(vid, tm));
            }
        }
        if let Some(&last) = paves.last() {
            with_middles.push(last);
        }
        let paves = with_middles;
        for (k, w) in paves.windows(2).enumerate() {
            let piece_edge =
                topo.add_edge(Edge::new(w[0].vertex, w[1].vertex, original.curve.clone()));
            let piece_pb = arena
                .pave_blocks
                .alloc(PaveBlock::new(piece_edge, w[0], w[1]));
            if k == 0 {
                let first = &mut arena.curves[index];
                first.pave_blocks = vec![piece_pb];
                first.t_range = (w[0].parameter, w[1].parameter);
                continue;
            }
            let piece_index = arena.curves.len();
            let mut piece = original.clone();
            piece.pave_blocks = vec![piece_pb];
            piece.t_range = (w[0].parameter, w[1].parameter);
            arena.curves.push(piece);
            arena.interference.ff.push(Interference::FF {
                f1: original.face_a,
                f2: original.face_b,
                curve_index: piece_index,
            });
            if let Some(skipped) = arena.curve_skip_faces.get(&index).cloned() {
                arena.curve_skip_faces.insert(piece_index, skipped);
            }
            if let Some(extra) = arena.curve_extra_faces.get(&index).cloned() {
                arena.curve_extra_faces.insert(piece_index, extra);
            }
        }
    }
    Ok(())
}

/// Cut each line, circle or ellipse section between two curved faces at the
/// vertices where a boundary edge of one of the two faces crosses the other
/// face (the EF crossings of the pair), inside the section's span. The
/// section leaves the first face there, or meets the sections of a third
/// face there, and the boundary edge is split at the same vertex. The window
/// such a section is trimmed to cannot see every boundary: a whole cylinder
/// band notched by an earlier boolean keeps its `v` window, and the section
/// runs on past the notch's edge without a vertex where it crosses it. A
/// section on a plane face stays whole: the face's outline clips it later,
/// and a piece wholly off the face slips past that clip.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
fn split_sections_at_boundary_crossings(
    topo: &mut Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use std::collections::HashMap;

    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{FaceId, FaceSurface};
    use brepkit_topology::vertex::VertexId;

    use crate::ds::{Interference, Pave, PaveBlock};

    /// A crossing this near a section's end is that end.
    const WELD: f64 = 1e-5;

    let pair = |a: FaceId, b: FaceId| (a.index().min(b.index()), a.index().max(b.index()));
    let linear = tol.linear;
    let same = |topo: &Topology, a: VertexId, b: VertexId| {
        a == b
            || matches!(
                (topo.vertex(a), topo.vertex(b)),
                (Ok(p), Ok(q)) if (p.point() - q.point()).length() <= linear
            )
    };
    // Each crossing stands for the vertex its edge keeps there once
    // MakeBlocks splits it: of the paves that coincide on a block it keeps
    // one, and the section must end at that same vertex. The rank is the
    // one that vertex has on its edge: a block's own end, or a pave.
    let mut crossings: HashMap<(usize, usize), Vec<(VertexId, usize)>> = HashMap::new();
    for interf in &arena.interference.ef {
        let Interference::EF {
            edge,
            face,
            new_vertex: Some(vertex),
            ..
        } = interf
        else {
            continue;
        };
        let vertex = arena.resolve_vertex(*vertex);
        let kept = arena
            .edge_pave_blocks
            .get(edge)
            .map(|blocks| arena.collect_leaf_pave_blocks(blocks))
            .unwrap_or_default()
            .into_iter()
            .filter_map(|leaf| arena.pave_blocks.get(leaf))
            .flat_map(|piece| {
                make_blocks::piece_vertices(topo, arena, piece)
                    .into_iter()
                    .enumerate()
                    .map(|(k, v)| (arena.resolve_vertex(v), usize::from(k >= 2)))
            })
            .find(|&(v, _)| same(topo, v, vertex))
            .unwrap_or((vertex, 1));
        for owner in topo.faces_of_edge(*edge) {
            if owner != *face {
                let list = crossings.entry(pair(owner, *face)).or_default();
                if !list.contains(&kept) {
                    list.push(kept);
                }
            }
        }
    }
    if crossings.is_empty() {
        return Ok(());
    }

    for index in 0..arena.curves.len() {
        let original = arena.curves[index].clone();
        let Some(list) = crossings.get(&pair(original.face_a, original.face_b)) else {
            continue;
        };
        let &[pb_id] = original.pave_blocks.as_slice() else {
            continue;
        };
        let Some(pb) = arena.pave_blocks.get(pb_id) else {
            continue;
        };
        if !pb.extra_paves.is_empty() {
            continue;
        }
        let planar = |face: FaceId| {
            topo.face(face)
                .map(|f| matches!(f.surface(), FaceSurface::Plane { .. }))
        };
        if planar(original.face_a)? || planar(original.face_b)? {
            continue;
        }
        let (start, end) = (pb.start, pb.end);
        let (p0, p1) = (start.parameter, end.parameter);
        let (lo, hi) = (p0.min(p1), p0.max(p1));
        let edge = topo.edge(pb.original_edge)?;
        let a = topo.vertex(edge.start())?.point();
        let b = topo.vertex(edge.end())?.point();
        if (a - b).length() <= WELD {
            continue;
        }
        // Each crossing on the curve, as a pave parameter inside the span: a
        // line's paves read its own parameter, from `p0` at its start to `p1`
        // at its end, and a conic's its angle.
        // Crossings at one place cut the section once, at the vertex the
        // earliest rank and then the lowest index keeps, as the rim pass
        // does: no piece is shorter than the linear tolerance.
        let mut cuts: Vec<(f64, VertexId, usize)> = Vec::new();
        for &(vertex, rank) in list {
            let p = topo.vertex(vertex)?.point();
            if (p - a).length() <= WELD || (p - b).length() <= WELD {
                continue;
            }
            let (at, foot) = match &original.curve {
                EdgeCurve::Line => {
                    let d = b - a;
                    let f = (p - a).dot(d) / d.dot(d);
                    (f.madd(p1 - p0, p0), a + d * f)
                }
                EdgeCurve::Circle(c) => {
                    let t = c.project(p);
                    (
                        lo + (t - lo).rem_euclid(std::f64::consts::TAU),
                        c.evaluate(t),
                    )
                }
                EdgeCurve::Ellipse(e) => {
                    let t = e.project(p);
                    (
                        lo + (t - lo).rem_euclid(std::f64::consts::TAU),
                        e.evaluate(t),
                    )
                }
                EdgeCurve::NurbsCurve(_) => continue,
            };
            if (foot - p).length() > tol.linear || !(lo < at && at < hi) {
                continue;
            }
            match cuts.iter_mut().find(|c| same(topo, c.1, vertex)) {
                Some(kept) => {
                    if (rank, vertex.index()) < (kept.2, kept.1.index()) {
                        *kept = (at, vertex, rank);
                    }
                }
                None => cuts.push((at, vertex, rank)),
            }
        }
        if cuts.is_empty() {
            continue;
        }
        let ascending = p0 <= p1;
        cuts.sort_by(|x, y| {
            if ascending {
                x.0.total_cmp(&y.0)
            } else {
                y.0.total_cmp(&x.0)
            }
        });
        let mut paves = vec![start];
        paves.extend(cuts.into_iter().map(|(at, v, _)| Pave::new(v, at)));
        paves.push(end);
        for (n, w) in paves.windows(2).enumerate() {
            let piece_edge =
                topo.add_edge(Edge::new(w[0].vertex, w[1].vertex, original.curve.clone()));
            let piece = arena
                .pave_blocks
                .alloc(PaveBlock::new(piece_edge, w[0], w[1]));
            let t_range = (w[0].parameter, w[1].parameter);
            if n == 0 {
                let first = &mut arena.curves[index];
                first.pave_blocks = vec![piece];
                first.t_range = t_range;
                continue;
            }
            let next = arena.curves.len();
            let mut curve = original.clone();
            curve.pave_blocks = vec![piece];
            curve.t_range = t_range;
            arena.curves.push(curve);
            arena.interference.ff.push(Interference::FF {
                f1: original.face_a,
                f2: original.face_b,
                curve_index: next,
            });
            if let Some(skipped) = arena.curve_skip_faces.get(&index).cloned() {
                arena.curve_skip_faces.insert(next, skipped);
            }
            if let Some(extra) = arena.curve_extra_faces.get(&index).cloned() {
                arena.curve_extra_faces.insert(next, extra);
            }
        }
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
    run_pave_filler_over(topo, sources, &[], tol, arena, true)
}

/// [`run_pave_filler_n`] with loose edges among the tools, for imprinting them
/// on `sources[0]`: their vertices and edges meet that solid's and each
/// other's, and each block lying in a face of the solid becomes its section.
///
/// # Errors
///
/// Returns [`AlgoError`] if `sources` is empty or any stage fails.
pub fn run_pave_filler_with_edges(
    topo: &mut Topology,
    sources: &[SolidId],
    tool_edges: &[brepkit_topology::edge::EdgeId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    run_pave_filler_over(topo, sources, tool_edges, tol, arena, false)
}

/// The pave filler over `sources` and loose `tool_edges`. With `fuse`, the
/// sections are also cut where the faces of three sources meet
/// ([`split_sections_at_triple_points`]). An imprint or a split by faces
/// takes each tool face as a source of its own and splits the tools'
/// crossing sections in its own builders, so it skips that cut.
fn run_pave_filler_over(
    topo: &mut Topology,
    sources: &[SolidId],
    tool_edges: &[brepkit_topology::edge::EdgeId],
    tol: Tolerance,
    arena: &mut GfaArena,
    fuse: bool,
) -> Result<(), AlgoError> {
    let Some(&body) = sources.first() else {
        return Err(AlgoError::AssemblyFailed(
            "N-way pave filler needs at least one source solid".into(),
        ));
    };

    // Stage 1: Intersection over interacting pairs, accumulating into one arena.
    init_pave_blocks_n(topo, sources, arena)?;
    let mut tool_vertices = Vec::with_capacity(2 * tool_edges.len());
    for &edge_id in tool_edges {
        init_edge_pave_block(topo, edge_id, arena)?;
        let edge = topo.edge(edge_id)?;
        tool_vertices.extend([edge.start(), edge.end()]);
    }
    let (body_vertices, body_edges) = if tool_edges.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        (
            brepkit_topology::explorer::solid_vertices(topo, body)?,
            brepkit_topology::explorer::solid_edges(topo, body)?,
        )
    };
    let pairs = interacting_pairs(topo, sources, tol);

    for &(i, j) in &pairs {
        phase_vv::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    phase_vv::merge_coincident(topo, &tool_vertices, &body_vertices, tol, arena)?;
    phase_vv::merge_coincident(topo, &tool_vertices, &tool_vertices, tol, arena)?;
    // VV is the only phase that registers same-domain vertices and the edge
    // pave blocks are fixed at init, so the coincidence index is stable for the
    // remaining phases — build it once, after every pair's VV (mirrors the
    // two-solid `PaveFiller::perform`).
    arena.build_pave_vertex_index(topo, tol.linear);
    for &(i, j) in &pairs {
        phase_ve::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    phase_ve::check_vertex_edge_pairs(topo, &tool_vertices, &body_edges, tol, arena)?;
    phase_ve::check_vertex_edge_pairs(topo, &body_vertices, tool_edges, tol, arena)?;
    phase_ve::check_vertex_edge_pairs(topo, &tool_vertices, tool_edges, tol, arena)?;
    for &(i, j) in &pairs {
        phase_ee::perform(topo, sources[i], sources[j], tol, arena)?;
    }
    phase_ee::check_edge_edge_pairs(topo, tool_edges, &body_edges, tol, arena)?;
    for (k, edge) in tool_edges.iter().enumerate() {
        phase_ee::check_edge_edge_pairs(
            topo,
            std::slice::from_ref(edge),
            &tool_edges[k + 1..],
            tol,
            arena,
        )?;
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
    if fuse {
        split_sections_at_triple_points(topo, sources, tol, arena)?;
        split_sections_at_boundary_crossings(topo, tol, arena)?;
    }
    pave_co_endpoint_curves(topo, sources, tol, arena)?;
    pave_chords_across_conics(topo, tol, arena)?;
    split_co_endpoint_sections(topo, tol, arena)?;
    split_sections_at_rim_vertices(topo, tol, arena)?;

    // Stage 2: Resolution (solid-agnostic — reads the accumulated arena).
    make_blocks::perform(topo, arena)?;
    tool_edges::perform(topo, body, tool_edges, tol, arena)?;
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
            init_edge_pave_block(topo, edge_id, arena)?;
        }
    }
    Ok(())
}

/// An edge's first pave block, unless the edge already has one.
fn init_edge_pave_block(
    topo: &Topology,
    edge_id: brepkit_topology::edge::EdgeId,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    if arena.edge_pave_blocks.contains_key(&edge_id) {
        return Ok(());
    }
    let edge = topo.edge(edge_id)?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();
    let (t0, t1) = edge_pave_span(edge.curve(), start_pos, end_pos);
    arena.init_edge_pave_block(edge_id, edge.start(), t0, edge.end(), t1);
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
