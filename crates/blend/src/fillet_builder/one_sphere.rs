//! A round whose rim passes from one face of a sphere to another.
//!
//! A ball is two faces, which meet at its equator. A rim that crosses the
//! equator (a plane cut tilted to it) runs on one face, then on the other.
//! The round's contact on the sphere crosses the equator too, but not where
//! the rim does: its section at the rim's vertex on the equator ends past
//! the equator, on the face beyond. A band built for each face then has its
//! contact run on to the other face, and the trim of each face along it
//! leaves the two faces overlapping. So the two faces are joined first,
//! along the edges between them, and the round meets the one face.

use std::collections::{HashMap, HashSet};

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::fillet_plan::faces_tangent_along_edge;

/// Joins, in place, each two faces of `solid`'s outer shell that lie on one
/// sphere where a run of the `picked` edges passes from one to the other: at
/// a vertex two picked edges share, each lies between a third face and one
/// of the two, and an edge between the two that is not picked ends there.
/// The caller restores the solid's shell after the round.
///
/// The picked edges are read as the plan reads them: an edge between two
/// faces tangent along it has no corner to round, and the plan drops it, so
/// it neither joins a run nor counts as picked. An edge the solid does not
/// hold is left to the build, which fails it alone. Returns the edges the
/// joins took out of the shell, each one between two faces tangent along it.
pub(super) fn join_sphere_faces_the_rim_crosses(
    topo: &mut Topology,
    solid: SolidId,
    picked: &[EdgeId],
) -> Result<HashSet<EdgeId>, BlendError> {
    let mut pairs: Vec<(FaceId, FaceId)> = Vec::new();
    let mut gone = HashSet::new();
    {
        let adjacency = topo.build_adjacency(solid)?;
        let mut effective: HashSet<EdgeId> = HashSet::new();
        let mut at_vertex: HashMap<VertexId, Vec<EdgeId>> = HashMap::new();
        for &edge_id in picked {
            let Ok(edge) = topo.edge(edge_id) else {
                continue;
            };
            if let &[f0, f1] = adjacency.faces_for_edge(edge_id)
                && faces_tangent_along_edge(topo, edge_id, f0, f1)
            {
                continue;
            }
            if !effective.insert(edge_id) || edge.start() == edge.end() {
                continue;
            }
            at_vertex.entry(edge.start()).or_default().push(edge_id);
            at_vertex.entry(edge.end()).or_default().push(edge_id);
        }
        let picked = effective;
        for (&vertex, edges) in &at_vertex {
            let [e1, e2] = edges[..] else {
                continue;
            };
            let (&[a0, a1], &[b0, b1]) =
                (adjacency.faces_for_edge(e1), adjacency.faces_for_edge(e2))
            else {
                continue;
            };
            // The face the two edges share, and each one's other face.
            let (a, b) = if a0 == b0 {
                (a1, b1)
            } else if a0 == b1 {
                (a1, b0)
            } else if a1 == b0 {
                (a0, b1)
            } else if a1 == b1 {
                (a0, b0)
            } else {
                continue;
            };
            let (a, b) = if a.index() <= b.index() {
                (a, b)
            } else {
                (b, a)
            };
            if a == b || pairs.contains(&(a, b)) {
                continue;
            }
            let (fa, fb) = (topo.face(a)?, topo.face(b)?);
            if fa.is_reversed() != fb.is_reversed() || !one_sphere(fa.surface(), fb.surface()) {
                continue;
            }
            let between = shared_edges(topo, a, b)?;
            let mut meets_here = false;
            for &edge_id in &between {
                let edge = topo.edge(edge_id)?;
                meets_here |=
                    !picked.contains(&edge_id) && (edge.start() == vertex || edge.end() == vertex);
            }
            if meets_here {
                pairs.push((a, b));
            }
        }
    }
    pairs.sort_unstable_by_key(|&(a, b)| (a.index(), b.index()));
    let mut joined: HashMap<FaceId, FaceId> = HashMap::new();
    let find = |joined: &HashMap<FaceId, FaceId>, mut face: FaceId| {
        while let Some(&next) = joined.get(&face) {
            face = next;
        }
        face
    };
    for (a, b) in pairs {
        let (a, b) = (find(&joined, a), find(&joined, b));
        if a == b {
            continue;
        }
        if let Some((face, between)) = join(topo, solid, a, b)? {
            joined.insert(a, face);
            joined.insert(b, face);
            gone.extend(between);
        }
    }
    Ok(gone)
}

/// Whether two faces' surfaces are one sphere read alike: each takes the
/// same point at each parameter pair, so either face's edges lie on the
/// other's surface where they lie on their own.
fn one_sphere(a: &FaceSurface, b: &FaceSurface) -> bool {
    let (FaceSurface::Sphere(a), FaceSurface::Sphere(b)) = (a, b) else {
        return false;
    };
    let tol = 1e-9 * (1.0 + a.radius());
    (0..6).all(|i| {
        (-2..=2).all(|j| {
            let (u, v) = (1.1 * f64::from(i), 0.6 * f64::from(j));
            (a.evaluate(u, v) - b.evaluate(u, v)).length() <= tol
        })
    })
}

/// The edges both faces' outer wires run.
fn shared_edges(topo: &Topology, a: FaceId, b: FaceId) -> Result<Vec<EdgeId>, BlendError> {
    let of = |face: FaceId| -> Result<HashSet<EdgeId>, BlendError> {
        let wire = topo.wire(topo.face(face)?.outer_wire())?;
        Ok(wire.edges().iter().map(OrientedEdge::edge).collect())
    };
    let (ea, eb) = (of(a)?, of(b)?);
    let mut shared: Vec<EdgeId> = ea.intersection(&eb).copied().collect();
    shared.sort_unstable_by_key(|e| e.index());
    Ok(shared)
}

/// The face `a` and `b` make together, put in the solid's outer shell in
/// their place. Both must be faces of that shell, each outer wire must run
/// the edges the two share in one stretch, and the rest of `a`'s wire and
/// the rest of `b`'s must close one loop. Holes of either stay holes. The
/// new face takes the pcurves its edges had on `a` or `b`, which read alike
/// on it. Returns the face and the edges the two shared, which go, or `None`,
/// and changes nothing, for any other pair.
fn join(
    topo: &mut Topology,
    solid: SolidId,
    a: FaceId,
    b: FaceId,
) -> Result<Option<(FaceId, HashSet<EdgeId>)>, BlendError> {
    let shell_id = topo.solid(solid)?.outer_shell();
    let outer = topo.shell(shell_id)?.faces();
    if !outer.contains(&a) || !outer.contains(&b) {
        return Ok(None);
    }
    let shared: HashSet<EdgeId> = shared_edges(topo, a, b)?.into_iter().collect();
    if shared.is_empty() {
        return Ok(None);
    }
    // The wire's edges from the one after its shared stretch round to the
    // one before it, or `None` when the shared edges are not one stretch.
    let rest = |face: FaceId| -> Result<Option<Vec<OrientedEdge>>, BlendError> {
        let edges = topo.wire(topo.face(face)?.outer_wire())?.edges().to_vec();
        let n = edges.len();
        let on = |i: usize| shared.contains(&edges[i % n].edge());
        let starts: Vec<usize> = (0..n).filter(|&i| !on(i) && on(i + n - 1)).collect();
        let [start] = starts[..] else {
            return Ok(None);
        };
        let run: Vec<OrientedEdge> = (start..start + n)
            .map(|i| edges[i % n])
            .take_while(|oe| !shared.contains(&oe.edge()))
            .collect();
        Ok((run.len() + shared.len() == n).then_some(run))
    };
    let (Some(run_a), Some(run_b)) = (rest(a)?, rest(b)?) else {
        return Ok(None);
    };
    let ends = |run: &[OrientedEdge]| -> Result<(VertexId, VertexId), BlendError> {
        let (first, last) = (run[0], run[run.len() - 1]);
        Ok((
            first.oriented_start(topo.edge(first.edge())?),
            last.oriented_end(topo.edge(last.edge())?),
        ))
    };
    let ((a_from, a_to), (b_from, b_to)) = (ends(&run_a)?, ends(&run_b)?);
    if a_to != b_from || b_to != a_from {
        return Ok(None);
    }
    let (fa, fb) = (topo.face(a)?.clone(), topo.face(b)?.clone());
    // Each edge the new face keeps, and the face it came from.
    let mut kept: Vec<(EdgeId, FaceId)> = Vec::new();
    for (run, from) in [(&run_a, a), (&run_b, b)] {
        kept.extend(run.iter().map(|oe| (oe.edge(), from)));
    }
    for (holes, from) in [(fa.inner_wires(), a), (fb.inner_wires(), b)] {
        for &hole in holes {
            kept.extend(topo.wire(hole)?.edges().iter().map(|oe| (oe.edge(), from)));
        }
    }
    let mut edges = run_a;
    edges.extend(run_b);
    let wire = topo.add_wire(Wire::new(edges, true)?);
    let holes: Vec<_> = fa
        .inner_wires()
        .iter()
        .chain(fb.inner_wires())
        .copied()
        .collect();
    let face = if fa.is_reversed() {
        Face::new_reversed(wire, holes, fa.surface().clone())
    } else {
        Face::new(wire, holes, fa.surface().clone())
    };
    let face = topo.add_face(face);
    let mut faces: Vec<FaceId> = Vec::new();
    for &f in topo.shell(shell_id)?.faces() {
        if f == a {
            faces.push(face);
        } else if f != b {
            faces.push(f);
        }
    }
    *topo.shell_mut(shell_id)? = Shell::new(faces)?;
    for (edge, from) in kept {
        if let Some(pcurve) = topo.pcurves().get(edge, from).cloned() {
            topo.pcurves_mut().set(edge, face, pcurve);
        }
    }
    Ok(Some((face, shared)))
}
