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

    let merge_groups: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();

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

        // Check if any face in the group has inner wires (holes).
        let has_holes = group_face_ids
            .iter()
            .any(|&fid| topo.face(fid).is_ok_and(|f| !f.inner_wires().is_empty()));
        if has_holes {
            log::warn!(
                "unify_same_domain: skipping group with {} faces (contains holes)",
                group.len()
            );
            continue;
        }

        // Count edge usage within this group.
        let mut edge_count: HashMap<usize, usize> = HashMap::new();
        let mut all_oes: Vec<OrientedEdge> = Vec::new();

        for &fid in &group_face_ids {
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

        // Boundary edges: appear exactly once in the group.
        let boundary_edges: Vec<OrientedEdge> = all_oes
            .into_iter()
            .filter(|oe| edge_count.get(&oe.edge().index()).copied().unwrap_or(0) == 1)
            .collect();

        if boundary_edges.is_empty() {
            continue;
        }

        // Build merged wire from boundary edges.
        let Ok(merged_wire) = Wire::new(boundary_edges, true) else {
            log::warn!(
                "unify_same_domain: failed to build merged wire for {} faces",
                group.len()
            );
            continue;
        };

        let new_wire_id = topo.add_wire(merged_wire);
        let surface = face_surfaces[group[0]].clone();
        let new_face = brepkit_topology::face::Face::new(new_wire_id, Vec::new(), surface);
        let new_face_id = topo.add_face(new_face);

        for &fid in &group_face_ids {
            faces_to_remove.insert(fid);
        }
        new_faces_to_add.push(new_face_id);
        total_merged += group.len() - 1;
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
