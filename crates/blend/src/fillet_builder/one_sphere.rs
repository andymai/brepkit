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
//!
//! Two faces of one cylinder, cone or torus (two halves of a rod's wall)
//! meet the same way where a tilted rim crosses from one to the other. They
//! share two edges, one at each crossing, so they are joined along the
//! edges at one crossing, and the other edges between them stay as the
//! seam of the one face, which a round crosses as it crosses any seam. A
//! rim square to the edge it crosses (a level rim across a ruling) has the
//! round's contact cross that edge where the rim does, and the two faces
//! stay apart.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::fillet_plan::faces_tangent_along_edge;

/// For two faces on one surface: whether it is a sphere, and the edges
/// their outer wires share.
type PairData = (bool, Vec<EdgeId>);

/// Joins, in place, each two faces of `solid`'s outer shell that lie on one
/// sphere, cylinder, cone or torus where a run of the `picked` edges passes
/// from one to the other: at a vertex two picked edges share, each lies
/// between a third face and one of the two, and an edge between the two that
/// is not picked ends there (see [`join`] for the edges they are joined
/// along). Off a sphere, the run must cross that edge at an angle other than
/// square. The caller restores the solid's shell after the round.
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
    let mut pairs: Vec<(FaceId, FaceId, EdgeId)> = Vec::new();
    let mut gone = HashSet::new();
    {
        let adjacency = topo.build_adjacency(solid)?;
        let mut effective: HashSet<EdgeId> = HashSet::new();
        let mut at_vertex: HashMap<VertexId, Vec<EdgeId>> = HashMap::new();
        let mut pair_data: HashMap<(FaceId, FaceId), Option<PairData>> = HashMap::new();
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
            if a == b {
                continue;
            }
            // Each pair is read once, however many times the rim crosses
            // between its faces.
            let found = match pair_data.entry((a, b)) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let (fa, fb) = (topo.face(a)?, topo.face(b)?);
                    let data = if fa.is_reversed() == fb.is_reversed() && one_surface(topo, a, b)? {
                        let sphere = matches!(fa.surface(), FaceSurface::Sphere(_));
                        Some((sphere, shared_edges(topo, a, b)?))
                    } else {
                        None
                    };
                    entry.insert(data)
                }
            };
            let Some((sphere, between)) = found else {
                continue;
            };
            let sphere = *sphere;
            let mut meets_here = None;
            for &edge_id in between.iter() {
                let edge = topo.edge(edge_id)?;
                if !picked.contains(&edge_id) && (edge.start() == vertex || edge.end() == vertex) {
                    meets_here = Some(edge_id);
                }
            }
            let Some(edge) = meets_here else {
                continue;
            };
            // Off a sphere, a rim square to the edge it crosses (a level
            // rim across a ruling) has its round's contact cross that edge
            // where the rim does: each face takes its own part of the round.
            let at_an_angle = sphere
                || match (
                    tangent_at(topo, e1, vertex)?,
                    tangent_at(topo, edge, vertex)?,
                ) {
                    (Some(along_rim), Some(along_edge)) => along_rim.dot(along_edge).abs() > 1e-9,
                    _ => false,
                };
            if at_an_angle {
                pairs.push((a, b, edge));
            }
        }
    }
    pairs.sort_unstable_by_key(|&(a, b, edge)| (a.index(), b.index(), edge.index()));
    pairs.dedup_by_key(|&mut (a, b, _)| (a, b));
    let mut joined: HashMap<FaceId, FaceId> = HashMap::new();
    let find = |joined: &HashMap<FaceId, FaceId>, mut face: FaceId| {
        while let Some(&next) = joined.get(&face) {
            face = next;
        }
        face
    };
    for (a, b, at) in pairs {
        let (a, b) = (find(&joined, a), find(&joined, b));
        if a == b {
            continue;
        }
        if let Some((face, between)) = join(topo, solid, a, b, at)? {
            joined.insert(a, face);
            joined.insert(b, face);
            gone.extend(between);
        }
    }
    Ok(gone)
}

/// The unit tangent of `edge` at its end `vertex`, along its curve.
fn tangent_at(topo: &Topology, edge: EdgeId, vertex: VertexId) -> Result<Option<Vec3>, BlendError> {
    let data = topo.edge(edge)?;
    let (start, end) = (
        topo.vertex(data.start())?.point(),
        topo.vertex(data.end())?.point(),
    );
    let (t0, t1) = data.curve().domain_with_endpoints(start, end);
    let t = if data.start() == vertex { t0 } else { t1 };
    Ok(data
        .curve()
        .tangent_with_endpoints(t, start, end)
        .normalize()
        .ok())
}

/// Joins, in place, each face of `solid`'s outer shell on a sphere that is a
/// band between a closed run of the `picked` edges and one other loop, with
/// the one face beyond that loop on the same sphere. A ball cut below its
/// equator by a tilted plane keeps such a band of its lower face. The round
/// of the rim can run its contact past the equator on one side and short of
/// it on the other, where no trim of either face alone holds it. On the
/// joined face, the rim is the outer wire and the other loops of the face
/// beyond are holes. The picked edges are read as
/// [`join_sphere_faces_the_rim_crosses`] reads them. The caller restores the
/// solid's shell after the round. Returns the edges the joins took out of
/// the shell, those of each band's other loop.
pub(super) fn join_sphere_bands_at_picked_rims(
    topo: &mut Topology,
    solid: SolidId,
    picked: &[EdgeId],
) -> Result<HashSet<EdgeId>, BlendError> {
    let mut joins: Vec<(FaceId, usize, FaceId)> = Vec::new();
    {
        let adjacency = topo.build_adjacency(solid)?;
        let mut effective: HashSet<EdgeId> = HashSet::new();
        for &edge_id in picked {
            if let &[f0, f1] = adjacency.faces_for_edge(edge_id)
                && faces_tangent_along_edge(topo, edge_id, f0, f1)
            {
                continue;
            }
            effective.insert(edge_id);
        }
        let shell = topo.solid(solid)?.outer_shell();
        for &band in topo.shell(shell)?.faces() {
            let face = topo.face(band)?;
            if !matches!(face.surface(), FaceSurface::Sphere(_)) {
                continue;
            }
            let loops = loops_of(topo, band)?;
            let [l0, l1] = &loops[..] else {
                continue;
            };
            let all_picked = |l: &[OrientedEdge]| l.iter().all(|oe| effective.contains(&oe.edge()));
            let none_picked =
                |l: &[OrientedEdge]| !l.iter().any(|oe| effective.contains(&oe.edge()));
            let (rim, far) = if all_picked(l0) && none_picked(l1) {
                (0, l1)
            } else if all_picked(l1) && none_picked(l0) {
                (1, l0)
            } else {
                continue;
            };
            let mut beyond = None;
            for oe in far {
                let next = match *adjacency.faces_for_edge(oe.edge()) {
                    [a, b] if a == band && b != band => Some(b),
                    [a, b] if b == band && a != band => Some(a),
                    _ => None,
                };
                if next.is_none() || (beyond.is_some() && beyond != next) {
                    beyond = None;
                    break;
                }
                beyond = next;
            }
            let Some(beyond) = beyond else {
                continue;
            };
            let other = topo.face(beyond)?;
            if other.is_reversed() != face.is_reversed() || !one_sphere(topo, band, beyond)? {
                continue;
            }
            let far_edges: HashSet<EdgeId> = far.iter().map(OrientedEdge::edge).collect();
            let matching = loops_of(topo, beyond)?
                .iter()
                .filter(|l| {
                    l.len() == far_edges.len() && l.iter().all(|oe| far_edges.contains(&oe.edge()))
                })
                .count();
            if matching == 1
                && !joins
                    .iter()
                    .any(|&(b, _, o)| [b, o].contains(&band) || [b, o].contains(&beyond))
            {
                joins.push((band, rim, beyond));
            }
        }
    }
    let mut gone = HashSet::new();
    for (band, rim, beyond) in joins {
        gone.extend(join_band(topo, solid, band, rim, beyond)?);
    }
    Ok(gone)
}

/// The edges of each loop of `face`, its outer wire first.
fn loops_of(topo: &Topology, face: FaceId) -> Result<Vec<Vec<OrientedEdge>>, BlendError> {
    let data = topo.face(face)?;
    std::iter::once(data.outer_wire())
        .chain(data.inner_wires().iter().copied())
        .map(|wire| Ok(topo.wire(wire)?.edges().to_vec()))
        .collect()
}

/// The face `band` and `beyond` make together, put in the solid's outer
/// shell in their place: `band`'s loop `rim` is its outer wire, and the
/// loops of `beyond` other than the one it shares with `band` are its holes.
/// The new face takes the pcurves its edges had on either face. Returns the
/// edges the two shared, which go.
fn join_band(
    topo: &mut Topology,
    solid: SolidId,
    band: FaceId,
    rim: usize,
    beyond: FaceId,
) -> Result<HashSet<EdgeId>, BlendError> {
    let band_loops = loops_of(topo, band)?;
    let shared: HashSet<EdgeId> = band_loops[1 - rim].iter().map(OrientedEdge::edge).collect();
    let (fb, fo) = (topo.face(band)?.clone(), topo.face(beyond)?.clone());
    let mut kept: Vec<(EdgeId, FaceId)> =
        band_loops[rim].iter().map(|oe| (oe.edge(), band)).collect();
    let mut holes = Vec::new();
    for wire in std::iter::once(fo.outer_wire()).chain(fo.inner_wires().iter().copied()) {
        let edges = topo.wire(wire)?.edges();
        if edges.iter().all(|oe| shared.contains(&oe.edge())) {
            continue;
        }
        kept.extend(edges.iter().map(|oe| (oe.edge(), beyond)));
        holes.push(wire);
    }
    let outer = topo.add_wire(Wire::new(band_loops[rim].clone(), true)?);
    let face = if fb.is_reversed() {
        Face::new_reversed(outer, holes, fb.surface().clone())
    } else {
        Face::new(outer, holes, fb.surface().clone())
    };
    let face = topo.add_face(face);
    let shell_id = topo.solid(solid)?.outer_shell();
    let mut faces: Vec<FaceId> = Vec::new();
    for &f in topo.shell(shell_id)?.faces() {
        if f == band {
            faces.push(face);
        } else if f != beyond {
            faces.push(f);
        }
    }
    *topo.shell_mut(shell_id)? = Shell::new(faces)?;
    for (edge, from) in kept {
        if let Some(pcurve) = topo.pcurves().get(edge, from).cloned() {
            topo.pcurves_mut().set(edge, face, pcurve);
        }
    }
    Ok(shared)
}

/// Whether faces `a` and `b` lie on one sphere read alike (see
/// [`read_alike`]): each surface takes the same point at each parameter
/// pair, so either face's edges and pcurves lie on the other's surface where
/// they lie on their own.
fn one_sphere(topo: &Topology, a: FaceId, b: FaceId) -> Result<bool, BlendError> {
    let (sa, sb) = (topo.face(a)?.surface(), topo.face(b)?.surface());
    if !matches!((sa, sb), (FaceSurface::Sphere(_), FaceSurface::Sphere(_))) {
        return Ok(false);
    }
    same_surface(topo, a, b)
}

/// Whether faces `a` and `b` lie on one sphere, cylinder, cone or torus
/// read alike (see [`one_sphere`]).
fn one_surface(topo: &Topology, a: FaceId, b: FaceId) -> Result<bool, BlendError> {
    let (sa, sb) = (topo.face(a)?.surface(), topo.face(b)?.surface());
    if !matches!(
        (sa, sb),
        (FaceSurface::Sphere(_), FaceSurface::Sphere(_))
            | (FaceSurface::Cylinder(_), FaceSurface::Cylinder(_))
            | (FaceSurface::Cone(_), FaceSurface::Cone(_))
            | (FaceSurface::Torus(_), FaceSurface::Torus(_))
    ) {
        return Ok(false);
    }
    same_surface(topo, a, b)
}

/// [`read_alike`] for the surfaces of `a` and `b`, over the reach of the two
/// faces from `a`'s surface's centre, origin or apex.
fn same_surface(topo: &Topology, a: FaceId, b: FaceId) -> Result<bool, BlendError> {
    let (sa, sb) = (topo.face(a)?.surface(), topo.face(b)?.surface());
    let from = match sa {
        FaceSurface::Sphere(s) => s.center(),
        FaceSurface::Cylinder(s) => s.origin(),
        FaceSurface::Cone(s) => s.apex(),
        FaceSurface::Torus(s) => s.center(),
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => return Ok(false),
    };
    let reach = reach_from(topo, &[a, b], from)?;
    Ok(read_alike(sa, sb, reach))
}

/// A bound on how far any point of the wires of `faces` lies from `from`:
/// a line by its ends, a circle or an ellipse by its centre's distance plus
/// its larger radius, and a NURBS curve by its farthest control point, as a
/// curve with positive weights lies in their hull. A NURBS curve with a
/// weight that is not positive has no such bound, and the reach is infinite.
fn reach_from(topo: &Topology, faces: &[FaceId], from: Point3) -> Result<f64, BlendError> {
    let mut reach = 0.0_f64;
    for &face in faces {
        let data = topo.face(face)?;
        for wire in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
            for oe in topo.wire(wire)?.edges() {
                let edge = topo.edge(oe.edge())?;
                let bound = match edge.curve() {
                    EdgeCurve::Line => {
                        let (p, q) = (
                            topo.vertex(edge.start())?.point(),
                            topo.vertex(edge.end())?.point(),
                        );
                        (p - from).length().max((q - from).length())
                    }
                    EdgeCurve::Circle(circle) => {
                        (circle.center() - from).length() + circle.radius()
                    }
                    EdgeCurve::Ellipse(ellipse) => {
                        (ellipse.center() - from).length()
                            + ellipse.semi_major().max(ellipse.semi_minor())
                    }
                    EdgeCurve::NurbsCurve(curve) => {
                        if curve.weights().iter().any(|&w| w <= 0.0) {
                            f64::INFINITY
                        } else {
                            curve
                                .control_points()
                                .iter()
                                .map(|&p| (p - from).length())
                                .fold(0.0, f64::max)
                        }
                    }
                };
                reach = reach.max(bound);
            }
        }
    }
    Ok(reach)
}

/// Whether two surfaces are one sphere, cylinder, cone or torus in one
/// frame: the same centre, origin or apex, the same axes, and the same radii
/// or half-angle. Lengths agree within a part in 1e9 of the surface's own
/// size (its radius, a torus's major radius, 1 for a cone), and never by more
/// than the linear tolerance, so where the surface lies does not change the
/// answer. An axis or a half-angle that turns moves the surface by the angle
/// times the distance from its centre, so the angles, times `reach` (how far
/// the faces lie from the centre), agree within the same length. Then each
/// takes the same point at each parameter pair over the faces.
fn read_alike(a: &FaceSurface, b: &FaceSurface, reach: f64) -> bool {
    const PART: f64 = 1e-9;
    let linear = Tolerance::new().linear;
    let tol = |size: f64| (PART * size).min(linear);
    let near = |p: Point3, q: Point3, size: f64| (p - q).length() <= tol(size);
    let along = |u: Vec3, v: Vec3, size: f64| (u - v).length() * reach <= tol(size);
    let equal = |x: f64, y: f64, size: f64| (x - y).abs() <= tol(size);
    match (a, b) {
        (FaceSurface::Sphere(a), FaceSurface::Sphere(b)) => {
            let size = a.radius();
            near(a.center(), b.center(), size)
                && equal(a.radius(), b.radius(), size)
                && along(a.x_axis(), b.x_axis(), size)
                && along(a.y_axis(), b.y_axis(), size)
                && along(a.z_axis(), b.z_axis(), size)
        }
        (FaceSurface::Cylinder(a), FaceSurface::Cylinder(b)) => {
            let size = a.radius();
            near(a.origin(), b.origin(), size)
                && equal(a.radius(), b.radius(), size)
                && along(a.axis(), b.axis(), size)
                && along(a.x_axis(), b.x_axis(), size)
                && along(a.y_axis(), b.y_axis(), size)
        }
        (FaceSurface::Cone(a), FaceSurface::Cone(b)) => {
            near(a.apex(), b.apex(), 1.0)
                && (a.half_angle() - b.half_angle()).abs() * reach <= tol(1.0)
                && along(a.axis(), b.axis(), 1.0)
                && along(a.x_axis(), b.x_axis(), 1.0)
                && along(a.y_axis(), b.y_axis(), 1.0)
        }
        (FaceSurface::Torus(a), FaceSurface::Torus(b)) => {
            let size = a.major_radius();
            near(a.center(), b.center(), size)
                && equal(a.major_radius(), b.major_radius(), size)
                && equal(a.minor_radius(), b.minor_radius(), size)
                && along(a.x_axis(), b.x_axis(), size)
                && along(a.y_axis(), b.y_axis(), size)
                && along(a.z_axis(), b.z_axis(), size)
        }
        _ => false,
    }
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
/// their place. Both must be faces of that shell, and the rest of `a`'s
/// outer wire and the rest of `b`'s past the edges they are joined along
/// must close one loop. On a sphere they are joined along every edge they
/// share, which each outer wire must run in one stretch. On a cylinder, cone
/// or torus they are joined along the stretch of shared edges in `a`'s outer
/// wire that holds `at`, and each other edge they share is run twice by the
/// new face's wire, as its seam. Holes of either stay holes. The new face
/// takes the pcurves its edges other than such a seam had on `a` or `b`,
/// which read alike on it. Returns the face and the edges it was joined
/// along, which go, or `None`, and changes nothing, for any other pair.
fn join(
    topo: &mut Topology,
    solid: SolidId,
    a: FaceId,
    b: FaceId,
    at: EdgeId,
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
    let along: HashSet<EdgeId> = if matches!(topo.face(a)?.surface(), FaceSurface::Sphere(_)) {
        shared.clone()
    } else {
        let edges = topo.wire(topo.face(a)?.outer_wire())?.edges().to_vec();
        let n = edges.len();
        let on = |i: usize| shared.contains(&edges[i % n].edge());
        let Some(i) = edges.iter().position(|oe| oe.edge() == at) else {
            return Ok(None);
        };
        let mut stretch = HashSet::from([at]);
        let mut k = 1;
        while k < n && on(i + k) {
            stretch.insert(edges[(i + k) % n].edge());
            k += 1;
        }
        let mut k = 1;
        while k < n && on(i + n - k) {
            stretch.insert(edges[(i + n - k) % n].edge());
            k += 1;
        }
        stretch
    };
    // The wire's edges from the one after the stretch it is joined along
    // round to the one before it, or `None` when those edges are not one
    // stretch.
    let rest = |face: FaceId| -> Result<Option<Vec<OrientedEdge>>, BlendError> {
        let edges = topo.wire(topo.face(face)?.outer_wire())?.edges().to_vec();
        let n = edges.len();
        let on = |i: usize| along.contains(&edges[i % n].edge());
        let starts: Vec<usize> = (0..n).filter(|&i| !on(i) && on(i + n - 1)).collect();
        let [start] = starts[..] else {
            return Ok(None);
        };
        let run: Vec<OrientedEdge> = (start..start + n)
            .map(|i| edges[i % n])
            .take_while(|oe| !along.contains(&oe.edge()))
            .collect();
        Ok((run.len() + along.len() == n).then_some(run))
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
        if shared.contains(&edge) {
            continue;
        }
        if let Some(pcurve) = topo.pcurves().get(edge, from).cloned() {
            topo.pcurves_mut().set(edge, face, pcurve);
        }
    }
    Ok(Some((face, along)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::surfaces::{
        ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
    };
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::{one_surface, reach_from, read_alike};

    const PLACES: [f64; 2] = [0.0, 1e6];

    fn ball(x: f64) -> FaceSurface {
        FaceSurface::Sphere(SphericalSurface::new(Point3::new(x, 0.0, 0.0), 1.0).unwrap())
    }

    /// Two unit balls 1e-4 apart are two spheres wherever they lie, and the
    /// two faces of one ball are one sphere wherever it lies.
    #[test]
    fn balls_apart_stay_apart_and_one_ball_joins_at_any_place() {
        for x in PLACES {
            assert!(!read_alike(&ball(x), &ball(x + 1e-4), 2.0), "apart at {x}");
            assert!(read_alike(&ball(x), &ball(x), 2.0), "one ball at {x}");
        }
    }

    /// The same for two rods, cones and rings 1e-4 apart, and for one of
    /// each.
    #[test]
    fn rods_cones_and_rings_apart_stay_apart_at_any_place() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let rod = |x: f64| {
            FaceSurface::Cylinder(
                CylindricalSurface::new(Point3::new(x, 0.0, 0.0), up, 1.0).unwrap(),
            )
        };
        let cone = |x: f64| {
            FaceSurface::Cone(ConicalSurface::new(Point3::new(x, 0.0, 0.0), up, 0.4).unwrap())
        };
        let ring = |x: f64| {
            FaceSurface::Torus(ToroidalSurface::new(Point3::new(x, 0.0, 0.0), 2.0, 0.5).unwrap())
        };
        for x in PLACES {
            for make in [&rod as &dyn Fn(f64) -> FaceSurface, &cone, &ring] {
                assert!(!read_alike(&make(x), &make(x + 1e-4), 3.0), "apart at {x}");
                assert!(read_alike(&make(x), &make(x), 3.0), "one at {x}");
            }
        }
    }

    /// A face of the unit cylinder about `axis` through the origin, `length`
    /// long, bounded by lines between two of its rulings.
    fn wall(topo: &mut Topology, axis: Vec3, length: f64) -> FaceId {
        let surface = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), axis, 1.0).unwrap();
        let corners = [
            (1.0, 0.0, 0.0),
            (1.0, 0.0, length),
            (0.0, 1.0, length),
            (0.0, 1.0, 0.0),
        ]
        .map(|(x, y, z)| topo.add_vertex(Vertex::new(Point3::new(x, y, z), 1e-7)));
        let edges: Vec<OrientedEdge> = (0..4)
            .map(|k| {
                let edge = Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line);
                OrientedEdge::new(topo.add_edge(edge), true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        topo.add_face(Face::new(wire, vec![], FaceSurface::Cylinder(surface)))
    }

    /// Two faces of cylinders whose axes differ by 5e-10 radians lie apart by
    /// 5e-4 at a million from the origin: they are two surfaces there. Faces
    /// of them a unit long lie within 1e-9 of each other: one surface.
    #[test]
    fn nearly_coaxial_long_walls_stay_apart() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let leaning = Vec3::new(5e-10, 0.0, 1.0).normalize().unwrap();
        for (length, one) in [(1e6, false), (1.0, true)] {
            let mut topo = Topology::new();
            let a = wall(&mut topo, up, length);
            let b = wall(&mut topo, leaning, length);
            assert_eq!(one_surface(&topo, a, b).unwrap(), one, "{length} long");
            assert!(one_surface(&topo, a, a).unwrap(), "{length} long, one face");
        }
    }

    /// A face bounded by a line and a spline that runs from `(1, 0, 0)` up
    /// to `(0, 0, 40)` and down to `(-1, 0, 0)`, its corner a third of the way
    /// along, between points read at even steps. The reach from the origin is
    /// at least 40.
    #[test]
    fn the_reach_bounds_a_spline_between_its_samples() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let mut topo = Topology::new();
        let (a, b) = (Point3::new(1.0, 0.0, 0.0), Point3::new(-1.0, 0.0, 0.0));
        let spline = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.3, 1.0, 1.0],
            vec![a, Point3::new(0.0, 0.0, 40.0), b],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        let (va, vb) = (
            topo.add_vertex(Vertex::new(a, 1e-7)),
            topo.add_vertex(Vertex::new(b, 1e-7)),
        );
        let up = topo.add_edge(Edge::new(va, vb, EdgeCurve::NurbsCurve(spline)));
        let back = topo.add_edge(Edge::new(vb, va, EdgeCurve::Line));
        let wire = Wire::new(
            vec![OrientedEdge::new(up, true), OrientedEdge::new(back, true)],
            true,
        )
        .unwrap();
        let wire = topo.add_wire(wire);
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: 0.0,
        };
        let face = topo.add_face(Face::new(wire, vec![], plane));
        let reach = reach_from(&topo, &[face], Point3::new(0.0, 0.0, 0.0)).unwrap();
        assert!(reach >= 40.0, "reach {reach}");
    }
}
