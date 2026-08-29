//! G1 continuity chain expansion for fillet edge propagation.
//!
//! Given a set of seed edges, iteratively expands along manifold edges
//! that share the same face pair and are tangent-continuous at the
//! shared vertex.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// Sample the tangent of an edge curve at normalized parameter `t` in `[0, 1]`.
fn sample_edge_tangent(curve: &EdgeCurve, p_start: Point3, p_end: Point3, t: f64) -> Vec3 {
    match curve {
        EdgeCurve::Line => p_end - p_start,
        EdgeCurve::Circle(circle) => {
            let ts = circle.project(p_start);
            let mut te = circle.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ParametricCurve::tangent(circle, ts + (te - ts) * t)
        }
        EdgeCurve::Ellipse(ellipse) => {
            let ts = ellipse.project(p_start);
            let mut te = ellipse.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ParametricCurve::tangent(ellipse, ts + (te - ts) * t)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let (u0, u1) = nurbs.domain();
            let u = u0 + (u1 - u0) * t;
            let d = nurbs.derivatives(u, 1);
            d[1]
        }
    }
}

/// Expand a seed edge set by G1 (tangent-continuity) chain propagation.
pub fn expand_g1_chain(
    topo: &Topology,
    solid: SolidId,
    seed_edges: &[EdgeId],
    tol: Tolerance,
) -> Result<Vec<EdgeId>, crate::BlendError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();
    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    let mut vertex_to_edges: HashMap<usize, Vec<EdgeId>> = HashMap::new();
    let mut edge_ids: HashMap<usize, EdgeId> = HashMap::new();
    for &fid in &shell_face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                edge_to_faces.entry(eid.index()).or_default().push(fid);
                edge_ids.insert(eid.index(), eid);
                let edge = topo.edge(eid)?;
                vertex_to_edges
                    .entry(edge.start().index())
                    .or_default()
                    .push(eid);
                vertex_to_edges
                    .entry(edge.end().index())
                    .or_default()
                    .push(eid);
            }
        }
    }
    for edges in vertex_to_edges.values_mut() {
        edges.sort_unstable_by_key(|edge| edge.index());
        edges.dedup_by_key(|edge| edge.index());
    }
    let mut expanded: HashSet<usize> = seed_edges.iter().map(|edge| edge.index()).collect();
    let mut queue = seed_edges.to_vec();
    while let Some(current) = queue.pop() {
        let Some(cf) = edge_to_faces.get(&current.index()) else {
            continue;
        };
        if cf.len() != 2 {
            continue;
        }
        let (first, second) = (cf[0].index(), cf[1].index());
        let (cf1, cf2) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        let cur_edge = topo.edge(current)?;
        let cur_start = topo.vertex(cur_edge.start())?.point();
        let cur_end = topo.vertex(cur_edge.end())?.point();
        for &shared_vid in &[cur_edge.start(), cur_edge.end()] {
            let t_raw = if shared_vid == cur_edge.start() {
                sample_edge_tangent(cur_edge.curve(), cur_start, cur_end, 0.0)
            } else {
                -sample_edge_tangent(cur_edge.curve(), cur_start, cur_end, 1.0)
            };
            let len = t_raw.length();
            if len < tol.linear {
                continue;
            }
            let t_cur = t_raw * (1.0 / len);
            let Some(neighbors) = vertex_to_edges.get(&shared_vid.index()) else {
                continue;
            };
            for &nb in neighbors {
                if expanded.contains(&nb.index()) {
                    continue;
                }
                let Some(nf) = edge_to_faces.get(&nb.index()) else {
                    continue;
                };
                if nf.len() != 2 {
                    continue;
                }
                let (first, second) = (nf[0].index(), nf[1].index());
                let pair = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                if (cf1, cf2) != pair {
                    continue;
                }
                let nb_edge = topo.edge(nb)?;
                let nb_start = topo.vertex(nb_edge.start())?.point();
                let nb_end = topo.vertex(nb_edge.end())?.point();
                let raw = if shared_vid == nb_edge.start() {
                    sample_edge_tangent(nb_edge.curve(), nb_start, nb_end, 0.0)
                } else {
                    -sample_edge_tangent(nb_edge.curve(), nb_start, nb_end, 1.0)
                };
                let len = raw.length();
                if len < tol.linear {
                    continue;
                }
                if t_cur.dot(raw * (1.0 / len)) < -0.985 {
                    expanded.insert(nb.index());
                    queue.push(nb);
                }
            }
        }
    }
    let mut result: Vec<EdgeId> = expanded
        .iter()
        .filter_map(|idx| edge_ids.get(idx).copied())
        .collect();
    result.sort_unstable_by_key(|edge| edge.index());
    Ok(result)
}

/// Group selected edges into deterministic ordered G1 contours.
///
/// Only requested edges participate; unlike [`expand_g1_chain`], this never
/// adds an unselected edge.  Edges are connected when they share a vertex,
/// have the same manifold face pair, and have anti-parallel away tangents.
pub fn group_g1_contours(
    topo: &Topology,
    solid: SolidId,
    selected_edges: &[EdgeId],
    tol: Tolerance,
) -> Result<Vec<Vec<EdgeId>>, crate::BlendError> {
    let shell = topo.shell(topo.solid(solid)?.outer_shell())?;
    let mut edge_to_faces = HashMap::<usize, Vec<FaceId>>::new();
    let mut vertex_to_edges = HashMap::<usize, Vec<EdgeId>>::new();
    let mut edge_ids = HashMap::<usize, EdgeId>::new();
    for &face_id in shell.faces() {
        let face = topo.face(face_id)?;
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            for oriented in topo.wire(wire_id)?.edges() {
                let edge_id = oriented.edge();
                edge_ids.insert(edge_id.index(), edge_id);
                edge_to_faces
                    .entry(edge_id.index())
                    .or_default()
                    .push(face_id);
                let edge = topo.edge(edge_id)?;
                vertex_to_edges
                    .entry(edge.start().index())
                    .or_default()
                    .push(edge_id);
                vertex_to_edges
                    .entry(edge.end().index())
                    .or_default()
                    .push(edge_id);
            }
        }
    }
    for faces in edge_to_faces.values_mut() {
        faces.sort_unstable_by_key(|face| face.index());
        faces.dedup_by_key(|face| face.index());
    }
    for edges in vertex_to_edges.values_mut() {
        edges.sort_unstable_by_key(|edge| edge.index());
        edges.dedup_by_key(|edge| edge.index());
    }
    let mut selected = selected_edges.to_vec();
    selected.sort_unstable_by_key(|edge| edge.index());
    selected.dedup_by_key(|edge| edge.index());
    let selected_set: HashSet<usize> = selected.iter().map(|edge| edge.index()).collect();
    let face_pair = |edge: EdgeId| -> Option<(usize, usize)> {
        let faces = edge_to_faces.get(&edge.index())?;
        if faces.len() != 2 {
            return None;
        }
        let (first, second) = (faces[0].index(), faces[1].index());
        Some(if first < second {
            (first, second)
        } else {
            (second, first)
        })
    };
    let away_tangent = |edge_id: EdgeId, vertex: brepkit_topology::vertex::VertexId| {
        let edge = topo.edge(edge_id).ok()?;
        let start = topo.vertex(edge.start()).ok()?.point();
        let end = topo.vertex(edge.end()).ok()?.point();
        let raw = if vertex == edge.start() {
            sample_edge_tangent(edge.curve(), start, end, 0.0)
        } else {
            -sample_edge_tangent(edge.curve(), start, end, 1.0)
        };
        let length = raw.length();
        (length >= tol.linear).then(|| raw * (1.0 / length))
    };
    let mut graph = HashMap::<usize, Vec<usize>>::new();
    for &edge_id in &selected {
        let Some(pair) = face_pair(edge_id) else {
            graph.entry(edge_id.index()).or_default();
            continue;
        };
        let edge = topo.edge(edge_id)?;
        for vertex in [edge.start(), edge.end()] {
            let Some(candidates) = vertex_to_edges.get(&vertex.index()) else {
                continue;
            };
            let Some(current_tangent) = away_tangent(edge_id, vertex) else {
                continue;
            };
            for &candidate in candidates {
                if candidate == edge_id
                    || !selected_set.contains(&candidate.index())
                    || face_pair(candidate) != Some(pair)
                {
                    continue;
                }
                let Some(candidate_tangent) = away_tangent(candidate, vertex) else {
                    continue;
                };
                if current_tangent.dot(candidate_tangent) < -0.985 {
                    graph
                        .entry(edge_id.index())
                        .or_default()
                        .push(candidate.index());
                    graph
                        .entry(candidate.index())
                        .or_default()
                        .push(edge_id.index());
                }
            }
        }
    }
    for neighbors in graph.values_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut contours = Vec::new();
    let mut visited = HashSet::<usize>::new();
    for &root in &selected {
        if !visited.insert(root.index()) {
            continue;
        }
        let mut component = vec![root.index()];
        let mut stack = vec![root.index()];
        while let Some(edge_index) = stack.pop() {
            for &neighbor in graph.get(&edge_index).into_iter().flatten() {
                if visited.insert(neighbor) {
                    stack.push(neighbor);
                    component.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        let simple = component
            .iter()
            .all(|edge| graph.get(edge).map_or(0, Vec::len) <= 2);
        let endpoints: Vec<_> = component
            .iter()
            .copied()
            .filter(|edge| graph.get(edge).map_or(0, Vec::len) != 2)
            .collect();
        let start = endpoints.first().copied().unwrap_or(component[0]);
        let ordered = if simple {
            let mut result = vec![start];
            let mut previous = None;
            let mut current = start;
            while result.len() < component.len() {
                let next = graph
                    .get(&current)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|candidate| Some(*candidate) != previous && *candidate != start)
                    .min();
                let Some(next) = next else { break };
                result.push(next);
                previous = Some(current);
                current = next;
            }
            if result.len() == component.len() {
                result
            } else {
                component.clone()
            }
        } else {
            component.clone()
        };
        contours.push(
            ordered
                .into_iter()
                .filter_map(|index| edge_ids.get(&index).copied())
                .collect::<Vec<_>>(),
        );
    }
    contours.sort_by_key(|contour: &Vec<EdgeId>| contour[0].index());
    Ok(contours)
}
