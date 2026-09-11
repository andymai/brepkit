//! Reconstruct and jointly rebuild an isolated, accepted cylindrical fillet.
//!
//! The ordinary polygon trimmer cannot trim a cylindrical face touched only
//! at a vertex: passing that face through leaves a missing transition. For a
//! convex quarter-cylinder between perpendicular planar supports, bounded by
//! two perpendicular planar caps, the sharp topology is uniquely recoverable.
//! This module recognizes that geometry (not operation history), extends only
//! collinear straight edges, and rebuilds the inherited and requested blends
//! together. It never sews unrelated boundaries or changes tolerances/radii.
//!
//! Other cylinders, unequal radii, inner shells, and non-isolated strips do
//! not enter this route. Both the source and recovered topology stay immutable
//! during reconstruction; no partial reconstruction is a successful result.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::OperationsError;

struct Strip {
    face: FaceId,
    axial: [EdgeId; 2],
    ends: [([VertexId; 2], Point3); 2],
}

/// Return `None` unless exactly one isolated equal-radius strip adjoins the
/// single requested sharp edge. Allocation starts only after recognition.
pub(super) fn try_adjoining_strip(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<Option<SolidId>, OperationsError> {
    let [target] = edges else { return Ok(None) };
    if !radius.is_finite() || !topo.solid(solid)?.inner_shells().is_empty() {
        return Ok(None);
    }
    let adjacency = topo.build_adjacency(solid)?;
    if !adjacency.is_manifold() || adjacency.faces_for_edge(*target).len() != 2 {
        return Ok(None);
    }
    let faces = topo
        .shell(topo.solid(solid)?.outer_shell())?
        .faces()
        .to_vec();
    let mut strips = Vec::new();
    for &face in &faces {
        if let Some(strip) = recognize(topo, &adjacency, face, *target, radius)? {
            strips.push(strip);
        }
    }
    let [strip] = strips.as_slice() else {
        return Ok(None);
    };
    if !can_extend(topo, solid, strip)? {
        return Ok(None);
    }
    let (recovered, inherited, requested) = reconstruct(topo, solid, strip, *target)?;
    if !crate::validate::validate_solid(topo, recovered)?.is_valid() {
        return Err(OperationsError::NonManifoldResult);
    }
    // Rebuild the inherited strip with a G1-constrained rational setback
    // corner. A positional Coons patch closes the shell but leaves creases.
    let corner = SetbackCorner::new(topo, recovered, inherited, requested, radius)?;
    let result = super::rolling_ball::build(
        topo,
        recovered,
        &[inherited, requested],
        radius,
        Some(&corner),
    )?;
    let validation = crate::validate::validate_solid(topo, result)?;
    if !validation.is_valid() {
        let adjacency = topo.build_adjacency(result)?;
        for &id in adjacency.boundary_edges() {
            let edge = topo.edge(id)?;
            log::debug!(
                "N339 free edge {id:?}: {:?} -> {:?}, curve={:?}, faces={:?}",
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
                edge.curve(),
                adjacency.faces_for_edge(id)
            );
        }
        return Err(OperationsError::Blend(
            brepkit_blend::BlendError::PlanningFailure {
                reason: format!(
                    "adjoining fillet output failed validation: {:?}",
                    validation.issues
                ),
            },
        ));
    }
    let volume = crate::measure::solid_volume(topo, result, 0.01)?;
    if !volume.is_finite() || volume <= 0.0 {
        return Err(OperationsError::NonManifoldResult);
    }
    Ok(Some(result))
}

/// G1 rational transition for the recognized convex right-angle/equal-radius case.
pub(super) struct SetbackCorner {
    pub vertex: usize,
    origin: Point3,
    first: brepkit_math::vec::Vec3,
    second: brepkit_math::vec::Vec3,
    inward: brepkit_math::vec::Vec3,
    radius: f64,
}

impl SetbackCorner {
    fn new(
        topo: &Topology,
        solid: SolidId,
        first: EdgeId,
        second: EdgeId,
        radius: f64,
    ) -> Result<Self, OperationsError> {
        let fail = || OperationsError::InvalidInput {
            reason:
                "unsupported adjoining fillet corner: expected convex perpendicular planar supports"
                    .into(),
        };
        let a = topo.edge(first)?;
        let b = topo.edge(second)?;
        let shared: Vec<_> = [a.start(), a.end()]
            .into_iter()
            .filter(|&v| v == b.start() || v == b.end())
            .collect();
        let [vertex] = shared.as_slice() else {
            return Err(fail());
        };
        let origin = topo.vertex(*vertex)?.point();
        let adjacency = topo.build_adjacency(solid)?;
        let common: Vec<_> = adjacency
            .faces_for_edge(first)
            .iter()
            .filter(|id| adjacency.faces_for_edge(second).contains(id))
            .copied()
            .collect();
        let [common] = common.as_slice() else {
            return Err(fail());
        };
        let normal = topo
            .face(*common)?
            .effective_plane_normal()
            .ok_or_else(fail)?;
        // The transition stays inside the three rectangular support faces.
        // Curved boundaries, notches and holes need a different corner solver.
        for &face_id in adjacency
            .faces_for_edge(first)
            .iter()
            .chain(adjacency.faces_for_edge(second))
        {
            let face = topo.face(face_id)?;
            let wire = topo.wire(face.outer_wire())?;
            if !face.inner_wires().is_empty()
                || wire.edges().len() != 4
                || wire.edges().iter().any(|oe| {
                    !topo
                        .edge(oe.edge())
                        .is_ok_and(|edge| matches!(edge.curve(), EdgeCurve::Line))
                })
            {
                return Err(fail());
            }
            let mut sides = Vec::new();
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                sides.push(
                    (topo.vertex(edge.end())?.point() - topo.vertex(edge.start())?.point())
                        .normalize()?,
                );
            }
            for i in 0..4 {
                if sides[i].dot(sides[(i + 1) % 4]).abs() > Tolerance::new().angular {
                    return Err(fail());
                }
            }
        }
        let mut directions = Vec::new();
        for (eid, edge) in [(first, a), (second, b)] {
            let far = if edge.start() == *vertex {
                edge.end()
            } else {
                edge.start()
            };
            let direction = (topo.vertex(far)?.point() - origin).normalize()?;
            if (topo.vertex(far)?.point() - origin).length() <= 2.0 * radius {
                return Err(OperationsError::InvalidInput { reason: "adjoining fillet setback requires both sharp edges longer than twice the radius".into() });
            }
            let other = other_face(&adjacency, eid, *common).ok_or_else(fail)?;
            let side_normal = topo
                .face(other)?
                .effective_plane_normal()
                .ok_or_else(fail)?;
            if side_normal.dot(normal).abs() > Tolerance::new().angular
                || direction.dot(normal).abs() > Tolerance::new().angular
            {
                return Err(fail());
            }
            directions.push((direction, side_normal));
        }
        if directions[0].0.dot(directions[1].0).abs() > Tolerance::new().angular
            || directions[0].0.dot(directions[1].1) > -1.0 + Tolerance::new().angular
            || directions[1].0.dot(directions[0].1) > -1.0 + Tolerance::new().angular
        {
            return Err(fail());
        }
        let mut retained = Vec::new();
        for eid in solid_edges(topo, solid)? {
            let edge = topo.edge(eid)?;
            if eid != first && eid != second && (edge.start() == *vertex || edge.end() == *vertex) {
                retained.push(eid);
            }
        }
        let [retained] = retained.as_slice() else {
            return Err(fail());
        };
        let edge = topo.edge(*retained)?;
        let far = if edge.start() == *vertex {
            edge.end()
        } else {
            edge.start()
        };
        let span = topo.vertex(far)?.point() - origin;
        if !matches!(edge.curve(), EdgeCurve::Line)
            || span.cross(normal).length() > Tolerance::new().linear
            || span.dot(-normal) <= 2.0 * radius
        {
            return Err(OperationsError::InvalidInput { reason: "adjoining fillet setback requires more than twice the radius of clearance along the retained sharp edge".into() });
        }
        Ok(Self {
            vertex: vertex.index(),
            origin,
            first: directions[0].0,
            second: directions[1].0,
            inward: -normal,
            radius,
        })
    }

    fn point(&self, local: Point3) -> Point3 {
        self.origin
            + (self.first * local.x() + self.inward * local.y() + self.second * local.z())
                * self.radius
    }

    pub(super) fn preserved(&self) -> Point3 {
        self.point(Point3::new(0.0, 2.0, 0.0))
    }

    pub(super) fn face_spec(&self) -> Result<crate::boolean::FaceSpec, OperationsError> {
        let local = super::setback_patch::surface()?;
        let points = local
            .control_points()
            .iter()
            .map(|row| row.iter().map(|&p| self.point(p)).collect())
            .collect();
        let surface = brepkit_math::nurbs::surface::NurbsSurface::new(
            5,
            5,
            local.knots_u().to_vec(),
            local.knots_v().to_vec(),
            points,
            local.weights().to_vec(),
        )?;
        let mut vertices = [
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ]
        .map(|p| self.point(p))
        .to_vec();
        // The local (first, inward, second) frame may be left-handed.
        let reversed = self.first.cross(self.inward).dot(self.second) < 0.0;
        if reversed {
            vertices.reverse();
        }
        Ok(crate::boolean::FaceSpec::Surface {
            vertices,
            surface: FaceSurface::Nurbs(surface),
            reversed,
            inner_wires: vec![],
        })
    }

    /// Replace the two planar-support chords by the exact cubic boundaries
    /// of the corner patch. Each shared edge is replaced once, for both faces.
    pub(super) fn install_contact_curves(
        &self,
        topo: &mut Topology,
        solid: SolidId,
    ) -> Result<(), OperationsError> {
        for control in super::setback_patch::planar_controls() {
            let points = control.map(|p| self.point(p));
            let [a, _, _, b] = points;
            let mut matches = Vec::new();
            for id in solid_edges(topo, solid)? {
                let edge = topo.edge(id)?;
                let p = topo.vertex(edge.start())?.point();
                let q = topo.vertex(edge.end())?.point();
                let tol = Tolerance::new().linear;
                if ((p - a).length() < tol && (q - b).length() < tol)
                    || ((p - b).length() < tol && (q - a).length() < tol)
                {
                    matches.push((id, (p - a).length() < tol));
                }
            }
            let [(id, forward)] = matches.as_slice() else {
                return Err(OperationsError::NonManifoldResult);
            };
            let mut controls = points.to_vec();
            if !forward {
                controls.reverse();
            }
            let knots = [0.0; 4].into_iter().chain([1.0; 4]).collect();
            let curve =
                brepkit_math::nurbs::curve::NurbsCurve::new(3, knots, controls, vec![1.0; 4])?;
            topo.edge_mut(*id)?.set_curve(EdgeCurve::NurbsCurve(curve));
        }
        Ok(())
    }
}

/// Recognize geometry before allocating anything. In particular, a circle's
/// endpoints alone do not prove the selected arc is the quarter-circle rather
/// than its three-quarter complement.
#[allow(clippy::too_many_lines)]
fn recognize(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    face_id: FaceId,
    target: EdgeId,
    radius: f64,
) -> Result<Option<Strip>, OperationsError> {
    let tol = Tolerance::new();
    let face = topo.face(face_id)?;
    let FaceSurface::Cylinder(cylinder) = face.surface() else {
        return Ok(None);
    };
    if face.is_reversed()
        || !face.inner_wires().is_empty()
        || (cylinder.radius() - radius).abs() > tol.linear
    {
        return Ok(None);
    }
    let wire = topo.wire(face.outer_wire())?;
    if wire.edges().len() != 4 || wire.edges().iter().any(|oe| oe.edge() == target) {
        return Ok(None);
    }
    let selected = topo.edge(target)?;
    if !matches!(selected.curve(), EdgeCurve::Line) {
        return Ok(None);
    }
    let mut axial = Vec::new();
    let mut arcs = Vec::new();
    for oe in wire.edges() {
        match topo.edge(oe.edge())?.curve() {
            EdgeCurve::Line => axial.push(oe.edge()),
            EdgeCurve::Circle(_) => arcs.push(oe.edge()),
            _ => return Ok(None),
        }
    }
    let ([a0, a1], [c0, c1]) = (axial.as_slice(), arcs.as_slice()) else {
        return Ok(None);
    };
    let mut supports = Vec::new();
    for &edge_id in &[*a0, *a1] {
        let Some(other) = other_face(adjacency, edge_id, face_id) else {
            return Ok(None);
        };
        let support = topo.face(other)?;
        let Some(normal) = support.effective_plane_normal() else {
            return Ok(None);
        };
        if !support.inner_wires().is_empty() || normal.dot(cylinder.axis()).abs() > tol.angular {
            return Ok(None);
        }
        let edge = topo.edge(edge_id)?;
        let p = topo.vertex(edge.start())?.point();
        let q = topo.vertex(edge.end())?.point();
        if (q - p).cross(cylinder.axis()).length() > tol.linear || (q - p).length() <= 2.0 * radius
        {
            return Ok(None);
        }
        for point in [p, q] {
            let center = cylinder.origin()
                + cylinder.axis() * (point - cylinder.origin()).dot(cylinder.axis());
            if (point - center - normal * radius).length() > tol.linear {
                return Ok(None);
            }
        }
        supports.push((other, normal));
    }
    if supports[0].0 == supports[1].0 || supports[0].1.dot(supports[1].1).abs() > tol.angular {
        return Ok(None);
    }
    let mut ends = Vec::new();
    let mut caps = Vec::new();
    let mut touches = 0;
    for &arc_id in &[*c0, *c1] {
        let Some(cap_id) = other_face(adjacency, arc_id, face_id) else {
            return Ok(None);
        };
        let cap = topo.face(cap_id)?;
        let Some(normal) = cap.effective_plane_normal() else {
            return Ok(None);
        };
        if !cap.inner_wires().is_empty()
            || normal.cross(cylinder.axis()).length() > tol.angular
            || supports.iter().any(|&(id, _)| id == cap_id)
        {
            return Ok(None);
        }
        let edge = topo.edge(arc_id)?;
        let EdgeCurve::Circle(circle) = edge.curve() else {
            return Ok(None);
        };
        let (va, vb) = (edge.start(), edge.end());
        let (pa, pb) = (topo.vertex(va)?.point(), topo.vertex(vb)?.point());
        let center =
            cylinder.origin() + cylinder.axis() * (pa - cylinder.origin()).dot(cylinder.axis());
        let (t0, t1) = edge.curve().domain_with_endpoints(pa, pb);
        if (circle.center() - center).length() > tol.linear
            || (circle.radius() - radius).abs() > tol.linear
            || circle.normal().cross(cylinder.axis()).length() > tol.angular
            || ((pb - center).length() - radius).abs() > tol.linear
            || (pa - pb).dot(cylinder.axis()).abs() > tol.linear
            || (t1 - t0 - std::f64::consts::FRAC_PI_2).abs() > tol.angular
        {
            return Ok(None);
        }
        // Each circular end must connect one vertex of each axial contact.
        for &axial_id in &[*a0, *a1] {
            let line = topo.edge(axial_id)?;
            if [va, vb]
                .iter()
                .filter(|&&v| v == line.start() || v == line.end())
                .count()
                != 1
            {
                return Ok(None);
            }
        }
        let sharp = pa + (pb - center);
        for &(id, _) in &supports {
            let FaceSurface::Plane { normal, d } = topo.face(id)?.surface() else {
                return Ok(None);
            };
            if (crate::dot_normal_point(*normal, sharp) - d).abs() > tol.linear {
                return Ok(None);
            }
        }
        if [va, vb]
            .iter()
            .any(|&v| v == selected.start() || v == selected.end())
        {
            let target_faces = adjacency.faces_for_edge(target);
            if !target_faces.contains(&cap_id)
                || !supports.iter().any(|&(id, _)| target_faces.contains(&id))
            {
                return Ok(None);
            }
            touches += 1;
        }
        ends.push(([va, vb], sharp));
        caps.push(cap_id);
    }
    if caps[0] == caps[1] || touches != 1 {
        return Ok(None);
    }
    Ok(Some(Strip {
        face: face_id,
        axial: [*a0, *a1],
        ends: [ends[0], ends[1]],
    }))
}

fn other_face(adjacency: &AdjacencyIndex, edge: EdgeId, face: FaceId) -> Option<FaceId> {
    match adjacency.faces_for_edge(edge) {
        [a, b] if *a == face && *b != face => Some(*b),
        [a, b] if *b == face && *a != face => Some(*a),
        _ => None,
    }
}

fn mapped_point(strip: &Strip, vertex: VertexId) -> Option<Point3> {
    strip
        .ends
        .iter()
        .find_map(|(vertices, point)| vertices.contains(&vertex).then_some(*point))
}

/// All moved edges outside the strip must merely extend their original line.
/// A curved or additional face at any affected vertex is not an isolated strip.
fn can_extend(topo: &Topology, solid: SolidId, strip: &Strip) -> Result<bool, OperationsError> {
    let tolerance = Tolerance::new().linear;
    for edge_id in solid_edges(topo, solid)? {
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            mapped_point(strip, edge.start()),
            mapped_point(strip, edge.end()),
        );
        let ((Some(sharp), None) | (None, Some(sharp))) = (a, b) else {
            continue;
        };
        if !matches!(edge.curve(), EdgeCurve::Line) {
            return Ok(false);
        }
        let p = topo.vertex(edge.start())?.point();
        let q = topo.vertex(edge.end())?.point();
        if (q - p).length() <= tolerance
            || (sharp - p).cross(q - p).length() > tolerance * (q - p).length()
        {
            return Ok(false);
        }
        // Extending, not collapsing/reversing a neighbor through its far end.
        let (contact, far) = if a.is_some() { (p, q) } else { (q, p) };
        if (sharp - contact).dot(far - contact) >= 0.0 {
            return Ok(false);
        }
    }
    for &fid in topo.shell(topo.solid(solid)?.outer_shell())?.faces() {
        if fid == strip.face {
            continue;
        }
        let face = topo.face(fid)?;
        let mut touched = false;
        for &wid in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                if mapped_point(strip, edge.start()).is_some()
                    || mapped_point(strip, edge.end()).is_some()
                {
                    touched = true;
                    if wid != face.outer_wire() {
                        return Ok(false);
                    }
                }
            }
        }
        if touched && !face.surface().is_planar() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn reconstruct(
    topo: &mut Topology,
    solid: SolidId,
    strip: &Strip,
    target: EdgeId,
) -> Result<(SolidId, EdgeId, EdgeId), OperationsError> {
    let mut vertices = HashMap::new();
    let mut sharp = Vec::new();
    for (old, point) in strip.ends {
        let new = topo.add_vertex(Vertex::new(point, Tolerance::new().linear));
        sharp.push(new);
        for vertex in old {
            vertices.insert(vertex, new);
        }
    }
    let inherited = topo.add_edge(Edge::new(sharp[0], sharp[1], EdgeCurve::Line));
    let mut replacements = HashMap::new();
    for edge_id in solid_edges(topo, solid)? {
        let edge = topo.edge(edge_id)?;
        let start = vertices
            .get(&edge.start())
            .copied()
            .unwrap_or_else(|| edge.start());
        let end = vertices
            .get(&edge.end())
            .copied()
            .unwrap_or_else(|| edge.end());
        let replacement = if start == end {
            None
        } else if strip.axial.contains(&edge_id) {
            Some((inherited, start == sharp[0]))
        } else if start != edge.start() || end != edge.end() {
            Some((
                topo.add_edge(Edge::new(start, end, edge.curve().clone())),
                true,
            ))
        } else {
            Some((edge_id, true))
        };
        replacements.insert(edge_id, replacement);
    }
    let requested = replacements
        .get(&target)
        .and_then(|value| *value)
        .ok_or(OperationsError::NonManifoldResult)?
        .0;
    let mut faces = Vec::new();
    let original_faces = topo
        .shell(topo.solid(solid)?.outer_shell())?
        .faces()
        .to_vec();
    for face_id in original_faces {
        if face_id == strip.face {
            continue;
        }
        let mut face = topo.face(face_id)?.clone();
        let wire = topo.wire(face.outer_wire())?;
        let mut changed = false;
        let mut edges = Vec::new();
        for oe in wire.edges() {
            match replacements[&oe.edge()] {
                Some((edge, same_direction)) => {
                    changed |= edge != oe.edge();
                    edges.push(OrientedEdge::new(edge, oe.is_forward() == same_direction));
                }
                None => changed = true,
            }
        }
        if changed {
            let wire = topo.add_wire(Wire::new(edges, true)?);
            face.set_outer_wire(wire);
            faces.push(topo.add_face(face));
        } else {
            faces.push(face_id);
        }
    }
    let shell = topo.add_shell(Shell::new(faces)?);
    Ok((
        topo.add_solid(Solid::new(shell, vec![])),
        inherited,
        requested,
    ))
}
