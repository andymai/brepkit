//! A round along a straight convex edge between two planes that ends at a
//! concave corner of the solid.
//!
//! At such an end the edge runs off the far side of a third plane, and the
//! material goes on past that plane under the round. The round stops there
//! against a flat tab in the third plane, between its arc and the corner, so
//! the plane's face grows where a convex end's would be notched. The removed
//! material is the corner's section swept along the edge, so a cut takes it off
//! exactly: the tool stops on the third plane, and runs past any end that does
//! not stop there.

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::OperationsError;
use crate::boolean::{BooleanOp, boolean};

/// How near a cap's normal must lie to the edge to count as square to it.
const SQUARE: f64 = 1e-9;
/// How far past a strip's rim a boundary must reach to count as inside it, so
/// the faces beside the strip and a contact line landing on a rim both pass.
const STRIP_MARGIN: f64 = 1e-7;
/// Chords a curved boundary is read through against a strip.
const STRIP_SPANS: usize = 64;

/// How an edge's end meets the cap past it.
#[derive(Clone, Copy, PartialEq)]
enum End {
    /// The edge runs off the far side of the cap's plane, so the round stops on it.
    Concave,
    /// The cap ends the edge, and the round runs out through it.
    Convex,
}

/// One edge's cut: its section, and how far along the edge it runs.
struct Cut {
    start: Point3,
    along: Vec3,
    length: f64,
    first: Vec3,
    second: Vec3,
    center_offset: Vec3,
}

/// The two faces on an edge and the caps at its ends, each end's other faces.
fn faces_at(
    topo: &Topology,
    faces: &[FaceId],
    vertex: VertexId,
) -> Result<Vec<FaceId>, OperationsError> {
    let mut around = Vec::new();
    for &face_id in faces {
        let face = topo.face(face_id)?;
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        'wires: for wire in wires {
            for oe in topo.wire(wire)?.edges() {
                let edge = topo.edge(oe.edge())?;
                if edge.start() == vertex || edge.end() == vertex {
                    around.push(face_id);
                    break 'wires;
                }
            }
        }
    }
    Ok(around)
}

/// A planar face's outward normal.
fn outward(topo: &Topology, face: FaceId) -> Result<Option<Vec3>, OperationsError> {
    Ok(topo.face(face)?.effective_plane_normal())
}

/// Reads an edge as a straight convex edge between two planes, with a square cap
/// at each end. `None` for anything else.
fn plan(
    topo: &Topology,
    faces: &[FaceId],
    edge_id: EdgeId,
    radius: f64,
) -> Result<Option<(Cut, [End; 2])>, OperationsError> {
    let edge = topo.edge(edge_id)?;
    if !matches!(edge.curve(), EdgeCurve::Line) {
        return Ok(None);
    }
    let (from, to) = (edge.start(), edge.end());
    let (start, end) = (topo.vertex(from)?.point(), topo.vertex(to)?.point());
    let length = (end - start).length();
    let Ok(along) = (end - start).normalize() else {
        return Ok(None);
    };
    let mut sides = Vec::new();
    for &face_id in faces {
        let face = topo.face(face_id)?;
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for wire in wires {
            if topo
                .wire(wire)?
                .edges()
                .iter()
                .any(|oe| oe.edge() == edge_id)
            {
                sides.push(face_id);
            }
        }
    }
    let [side1, side2] = sides[..] else {
        return Ok(None);
    };
    let (Some(first), Some(second)) = (outward(topo, side1)?, outward(topo, side2)?) else {
        return Ok(None);
    };
    let cosine = first.dot(second);
    if 1.0 + cosine < 1e-6 || 1.0 - cosine < 1e-6 {
        return Ok(None);
    }

    let end_of = |vertex: VertexId, into: Vec3| -> Result<Option<End>, OperationsError> {
        let around = faces_at(topo, faces, vertex)?;
        let caps: Vec<FaceId> = around
            .into_iter()
            .filter(|&face| face != side1 && face != side2)
            .collect();
        let [cap] = caps[..] else {
            return Ok(None);
        };
        let Some(normal) = outward(topo, cap)? else {
            return Ok(None);
        };
        let toward = normal.dot(into);
        if 1.0 - toward.abs() > SQUARE {
            return Ok(None);
        }
        Ok(Some(if toward > 0.0 {
            End::Concave
        } else {
            End::Convex
        }))
    };
    let (Some(at_start), Some(at_end)) = (end_of(from, along)?, end_of(to, -along)?) else {
        return Ok(None);
    };
    if at_start != End::Concave && at_end != End::Concave {
        return Ok(None);
    }

    // The ball's centre stands `radius` inside both planes.
    let center_offset = (first + second) * (-radius / (1.0 + cosine));
    for (face, normal) in [(side1, first), (side2, second)] {
        let reach = center_offset + normal * radius;
        if !strip_fits(topo, face, edge_id, start, along, length, reach)? {
            return Ok(None);
        }
    }
    Ok(Some((
        Cut {
            start,
            along,
            length,
            first,
            second,
            center_offset,
        },
        [at_start, at_end],
    )))
}

/// The tool: the corner's section at `cut.start`, swept along the edge.
fn tool(topo: &mut Topology, cut: &Cut, radius: f64) -> Result<SolidId, OperationsError> {
    let center = cut.start + cut.center_offset;
    let (a, b) = (center + cut.first * radius, center + cut.second * radius);
    let point = |topo: &mut Topology, p: Point3| topo.add_vertex(Vertex::new(p, 1e-7));
    let (va, vc, vb) = (point(topo, a), point(topo, cut.start), point(topo, b));
    let normal = (b - center).cross(a - center).normalize()?;
    let circle = Circle3D::new(center, normal, radius)?;
    let edges = [
        (va, vc, EdgeCurve::Line),
        (vc, vb, EdgeCurve::Line),
        (vb, va, EdgeCurve::Circle(circle)),
    ]
    .map(|(s, e, curve)| OrientedEdge::new(topo.add_edge(Edge::new(s, e, curve)), true));
    let wire = topo.add_wire(Wire::new(edges.to_vec(), true)?);
    let d = crate::dot_normal_point(-cut.along, cut.start);
    let face = topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: -cut.along,
            d,
        },
    ));
    crate::extrude::extrude(topo, face, cut.along, cut.length)
}

/// Whether the strip a round takes off `face` holds none of the face's other
/// boundary: the rectangle from the edge at `origin`, `length` along and `reach`
/// across (whose length is its width). A boundary inside it means the round's
/// contact line leaves the face, which the volume it removes barely shows.
fn strip_fits(
    topo: &Topology,
    face: FaceId,
    edge_id: EdgeId,
    origin: Point3,
    along: Vec3,
    length: f64,
    reach: Vec3,
) -> Result<bool, OperationsError> {
    let width = reach.length();
    let across = reach.normalize()?;
    let local = |p: Point3| ((p - origin).dot(along), (p - origin).dot(across));
    let enters = |p: Point3, q: Point3| {
        let (from, to) = (local(p), local(q));
        segment_enters_box(
            from,
            to,
            (STRIP_MARGIN, length - STRIP_MARGIN),
            (STRIP_MARGIN, width - STRIP_MARGIN),
        )
    };
    let face = topo.face(face)?;
    let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for wire in wires {
        for oe in topo.wire(wire)?.edges() {
            if oe.edge() == edge_id {
                continue;
            }
            let edge = topo.edge(oe.edge())?;
            let (p, q) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            if matches!(edge.curve(), EdgeCurve::Line) {
                if enters(p, q) {
                    return Ok(false);
                }
                continue;
            }
            let (t0, t1) = edge.curve().domain_with_endpoints(p, q);
            let at = |i: usize| {
                let t = t0 + (t1 - t0) * (i as f64) / (STRIP_SPANS as f64);
                edge.curve().evaluate_with_endpoints(t, p, q)
            };
            if (0..STRIP_SPANS).any(|i| enters(at(i), at(i + 1))) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Whether the segment between two points reaches the box, by clipping its
/// parameter range against each slab in turn.
fn segment_enters_box(
    from: (f64, f64),
    to: (f64, f64),
    first: (f64, f64),
    second: (f64, f64),
) -> bool {
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for (start, end, (min, max)) in [(from.0, to.0, first), (from.1, to.1, second)] {
        let delta = end - start;
        if delta.abs() < f64::EPSILON {
            if start < min || start > max {
                return false;
            }
            continue;
        }
        let (a, b) = ((min - start) / delta, (max - start) / delta);
        low = low.max(a.min(b));
        high = high.min(a.max(b));
    }
    low <= high
}

/// The section's area: the kite between the corner and the ball's centre, less
/// the ball's sector within it.
fn section_area(cut: &Cut, radius: f64) -> f64 {
    let turn = cut.first.dot(cut.second).clamp(-1.0, 1.0).acos();
    radius * radius * ((turn / 2.0).tan() - turn / 2.0)
}

/// Rounds straight convex edges that end at a concave corner, by a cut.
///
/// Each edge lies between two planes, with a square cap at each end, at least
/// one of them across a concave corner, and no two edges touch. `None` where the
/// selection is anything else, where a contact line would leave its face, or
/// where the cut takes off anything but the swept section (a part beyond a convex
/// end, say).
///
/// # Errors
///
/// Returns an error if a topology lookup or a boolean fails.
pub fn fillet_concave_ends(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<Option<SolidId>, OperationsError> {
    let faces = solid_faces(topo, solid)?;
    let known = solid_edges(topo, solid)?;
    let mut touched: Vec<VertexId> = Vec::new();
    let mut cuts = Vec::with_capacity(edges.len());
    for &edge_id in edges {
        if !known.contains(&edge_id) {
            return Ok(None);
        }
        let edge = topo.edge(edge_id)?;
        let ends = [edge.start(), edge.end()];
        if ends.iter().any(|v| touched.contains(v)) {
            return Ok(None);
        }
        touched.extend(ends);
        let Some(planned) = plan(topo, &faces, edge_id, radius)? else {
            return Ok(None);
        };
        cuts.push(planned);
    }
    if cuts.is_empty() {
        return Ok(None);
    }

    let before = crate::measure::solid_volume(topo, solid, 0.001)?;
    let mut removed = 0.0;
    let mut current = solid;
    for (mut cut, [at_start, at_end]) in cuts {
        // The swept section the fillet takes off spans the edge alone.
        removed += section_area(&cut, radius) * cut.length;
        // Run past a convex end, so its cap takes the section as a notch.
        let past = radius + 0.1 * cut.length;
        if at_start == End::Convex {
            cut.start = cut.start - cut.along * past;
            cut.length += past;
        }
        if at_end == End::Convex {
            cut.length += past;
        }
        let tool_solid = tool(topo, &cut, radius)?;
        let Ok(result) = boolean(topo, BooleanOp::Cut, current, tool_solid) else {
            return Ok(None);
        };
        current = result;
    }
    let after = crate::measure::solid_volume(topo, current, 0.001)?;
    // Rounding in the two whole-part readings sets a floor far below any cut.
    let allowed = 1e-6 * removed + 1e-14 * before.abs();
    if ((before - after) - removed).abs() > allowed {
        return Ok(None);
    }
    Ok(Some(current))
}
