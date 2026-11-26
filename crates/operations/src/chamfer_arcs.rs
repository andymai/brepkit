//! Chamfers of a cylinder's rim in arcs, cut as the bevel's ring.
//!
//! A rim between a cylinder and a plane square to its axis, in arcs, some of
//! them convex (the cap on top of the wall's material) and others concave
//! (the wall meeting a face that overhangs it), cannot be walked as one
//! chamfer: the chain turns from convex to concave where a wall rises from
//! the cap. Each convex arc is chamfered exactly by cutting away the ring
//! its bevel sweeps round the axis, kept to the prism of the cap it runs
//! along, so the cut ends where the cap does, on the walls rising from it. A
//! concave arc whose face is narrower than the chamfer all along it, so that
//! no chamfer of that distance fits on it, is left as it is, as another
//! kernel leaves it; so is the wall's seam, which is no edge between faces.

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

/// How far parallel two directions must be, as `1 - |a . b|`.
const PARALLEL_TOL: f64 = 1e-9;

/// One convex arc's cut: the cap it runs along, its circle, and the bevel's
/// triangle in a half-plane through the axis.
struct Ring {
    cap: FaceId,
    origin: Point3,
    axis: Vec3,
    radius: f64,
    /// Out of the material across the cap.
    up: Vec3,
    triangle: [Point3; 3],
    /// How deep the bevel runs down the wall.
    depth: f64,
}

/// A surface's own normal at `point` (not turned by the face's orientation):
/// a plane's, or a cylinder's, away from its axis.
fn surface_normal(surface: &FaceSurface, point: Point3) -> Option<Vec3> {
    match surface {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        FaceSurface::Cylinder(c) => {
            let off = point - c.origin();
            (off - c.axis() * off.dot(c.axis())).normalize().ok()
        }
        _ => None,
    }
}

/// The ways `face`'s wires run along `edge` from its start to its end, as
/// `true` for forward.
fn ways_along(
    topo: &Topology,
    face_id: FaceId,
    edge: EdgeId,
) -> Result<Vec<bool>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let mut ways = Vec::new();
    for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire)?.edges() {
            if oe.edge() == edge {
                ways.push(oe.is_forward());
            }
        }
    }
    Ok(ways)
}

/// The direction into `face` square to the edge at `point`, where the edge
/// runs along `tangent`: the face lies to the left of its wires seen from
/// the side its surface's normal points to.
fn into_face(
    topo: &Topology,
    face_id: FaceId,
    edge: EdgeId,
    point: Point3,
    tangent: Vec3,
) -> Result<Option<Vec3>, crate::OperationsError> {
    let ways = ways_along(topo, face_id, edge)?;
    let [forward] = ways[..] else {
        return Ok(None);
    };
    let Some(normal) = surface_normal(topo.face(face_id)?.surface(), point) else {
        return Ok(None);
    };
    let way = if forward { tangent } else { -tangent };
    Ok(normal.cross(way).normalize().ok())
}

/// The face's outward normal at `point`.
fn outward(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
) -> Result<Option<Vec3>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    Ok(surface_normal(face.surface(), point).map(|n| if face.is_reversed() { -n } else { n }))
}

/// Points along every edge of `face`'s wires.
fn boundary_samples(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let mut points = Vec::new();
    for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            for k in 0..=16 {
                let t = t0 + (t1 - t0) * f64::from(k) / 16.0;
                points.push(edge.curve().evaluate_with_endpoints(t, a, b));
            }
        }
    }
    Ok(points)
}

/// What a picked edge is to this chamfer.
enum Pick {
    /// A convex arc, cut as its ring.
    Cut(Ring),
    /// A concave arc too wide for its face, or the wall's seam: left.
    Left,
}

/// Whether the edge from `vertex` along the cap, other than the arcs, runs
/// along a plane wall rising from the cap (its points all on or above the
/// cap's plane, its normal square to the axis), which ends the arc's cut.
fn ends_on_a_rising_wall(
    topo: &Topology,
    faces: &[FaceId],
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    cap: FaceId,
    arc: EdgeId,
    vertex: brepkit_topology::vertex::VertexId,
    up: Vec3,
) -> Result<bool, crate::OperationsError> {
    let level = topo.vertex(vertex)?.point();
    let tol = 1e-7;
    let cap_face = topo.face(cap)?;
    for wire in std::iter::once(cap_face.outer_wire()).chain(cap_face.inner_wires().iter().copied())
    {
        for oe in topo.wire(wire)?.edges() {
            let next = oe.edge();
            if next == arc {
                continue;
            }
            let edge = topo.edge(next)?;
            if edge.start() != vertex && edge.end() != vertex {
                continue;
            }
            if matches!(edge.curve(), EdgeCurve::Circle(_)) {
                // Another arc of the rim carries the cut on.
                continue;
            }
            if !matches!(edge.curve(), EdgeCurve::Line) {
                return Ok(false);
            }
            for &other in adjacency.faces_for_edge(next) {
                if other == cap || !faces.contains(&other) {
                    continue;
                }
                let face = topo.face(other)?;
                let FaceSurface::Plane { normal, .. } = face.surface() else {
                    return Ok(false);
                };
                if normal.dot(up).abs() > PARALLEL_TOL {
                    return Ok(false);
                }
                let rises = boundary_samples(topo, other)?
                    .into_iter()
                    .all(|p| (p - level).dot(up) >= -tol);
                if !rises {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// What `edge` is to this chamfer, `d_cap` into the cap and `d_wall` down
/// the wall, or `None` when it is neither a rim arc between a cylinder and a
/// plane square to its axis nor the cylinder's seam.
#[allow(clippy::too_many_lines)]
fn pick(
    topo: &Topology,
    faces: &[FaceId],
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    edge: EdgeId,
    reference: Option<FaceId>,
    d1: f64,
    d2: f64,
) -> Result<Option<Pick>, crate::OperationsError> {
    let data = topo.edge(edge)?;
    let owners = adjacency.faces_for_edge(edge);
    // The seam of a cylinder: one face on both sides, along a ruling.
    let one_face = match owners {
        [f] => Some(*f),
        [f, g] if f == g => Some(*f),
        _ => None,
    };
    if let Some(face) = one_face {
        let surface = topo.face(face)?.surface();
        let (a, b) = (
            topo.vertex(data.start())?.point(),
            topo.vertex(data.end())?.point(),
        );
        if let (FaceSurface::Cylinder(c), EdgeCurve::Line) = (surface, data.curve())
            && let Ok(along) = (b - a).normalize()
            && 1.0 - along.dot(c.axis()).abs() <= PARALLEL_TOL
        {
            return Ok(Some(Pick::Left));
        }
        return Ok(None);
    }
    let EdgeCurve::Circle(circle) = data.curve() else {
        return Ok(None);
    };
    let &[f1, f2] = owners else {
        return Ok(None);
    };
    let (wall, cap) = match (topo.face(f1)?.surface(), topo.face(f2)?.surface()) {
        (FaceSurface::Cylinder(_), FaceSurface::Plane { .. }) => (f1, f2),
        (FaceSurface::Plane { .. }, FaceSurface::Cylinder(_)) => (f2, f1),
        _ => return Ok(None),
    };
    let FaceSurface::Cylinder(cylinder) = topo.face(wall)?.surface().clone() else {
        return Ok(None);
    };
    let FaceSurface::Plane {
        normal: cap_normal, ..
    } = topo.face(cap)?.surface().clone()
    else {
        return Ok(None);
    };
    let axis = cylinder.axis();
    let tol = 1e-7 * cylinder.radius().max(1.0);
    let off = circle.center() - cylinder.origin();
    if 1.0 - circle.normal().dot(axis).abs() > PARALLEL_TOL
        || 1.0 - cap_normal.dot(axis).abs() > PARALLEL_TOL
        || (off - axis * off.dot(axis)).length() > tol
        || (circle.radius() - cylinder.radius()).abs() > tol
    {
        return Ok(None);
    }
    // `d1` into the reference face, the cap when none is given.
    let (d_cap, d_wall) = if reference == Some(wall) {
        (d2, d1)
    } else {
        (d1, d2)
    };
    let (a, b) = (
        topo.vertex(data.start())?.point(),
        topo.vertex(data.end())?.point(),
    );
    let (t0, t1) = data.curve().domain_with_endpoints(a, b);
    let mid_t = 0.5 * (t0 + t1);
    let point = data.curve().evaluate_with_endpoints(mid_t, a, b);
    let Ok(tangent) = data.curve().tangent_with_endpoints(mid_t, a, b).normalize() else {
        return Ok(None);
    };
    let (Some(w_cap), Some(w_wall)) = (
        into_face(topo, cap, edge, point, tangent)?,
        into_face(topo, wall, edge, point, tangent)?,
    ) else {
        return Ok(None);
    };
    let (Some(n_cap), Some(n_wall)) = (outward(topo, cap, point)?, outward(topo, wall, point)?)
    else {
        return Ok(None);
    };
    let convex = w_cap.dot(n_wall) < -1e-9 && w_wall.dot(n_cap) < -1e-9;
    let concave = w_cap.dot(n_wall) > 1e-9 && w_wall.dot(n_cap) > 1e-9;
    let radial = {
        let off = point - cylinder.origin();
        (off - axis * off.dot(axis)).normalize()?
    };
    // Which way the cap runs off the rim, out from the axis or in.
    let side = w_cap.dot(radial).signum();
    if concave {
        // Left only when no chamfer of this distance fits on the cap: every
        // point of it nearer the rim than the chamfer's contact.
        let widest = boundary_samples(topo, cap)?
            .into_iter()
            .map(|p| {
                let off = p - cylinder.origin();
                side * ((off - axis * off.dot(axis)).length() - cylinder.radius())
            })
            .fold(f64::NEG_INFINITY, f64::max);
        return Ok((widest < d_cap - tol).then_some(Pick::Left));
    }
    if !convex {
        return Ok(None);
    }
    for vertex in [data.start(), data.end()] {
        if data.start() == data.end() {
            break;
        }
        if !ends_on_a_rising_wall(topo, faces, adjacency, cap, edge, vertex, n_cap)? {
            return Ok(None);
        }
    }
    // The bevel's line run on past both contacts, out over the corner, so
    // it crosses the cap and the wall square enough to cut cleanly; the
    // cap's prism keeps the cut to the cap's side of both.
    let on_cap = point + w_cap * d_cap;
    let on_wall = point + w_wall * d_wall;
    let (past_cap, past_wall) = (
        on_cap + (on_cap - on_wall) * 0.5,
        on_wall + (on_wall - on_cap) * 0.5,
    );
    let corner = point + n_cap * (0.5 * d_wall) + n_wall * (0.5 * d_cap);
    Ok(Some(Pick::Cut(Ring {
        cap,
        origin: cylinder.origin(),
        axis,
        radius: cylinder.radius(),
        up: n_cap,
        triangle: [corner, past_cap, past_wall],
        depth: d_wall,
    })))
}

/// Chamfers `edges` when they are arcs of cylinders' rims on planes square
/// to their axes.
///
/// Each edge is chamfered `d1` into its reference face (or into the cap with
/// `None`) and `d2` into the other: each convex arc's bevel ring, kept to its
/// cap's prism, is cut away; a concave arc on a face narrower than the
/// chamfer, and a cylinder's seam, are left. `None` for any other pick, when
/// no arc is convex, when a convex arc ends other than on another arc or a
/// plane wall rising from its cap, or when a cut fails, leaves an invalid
/// solid, or takes off other than the rings' own volume.
///
/// # Errors
///
/// Returns an error on a failed topology lookup.
pub fn chamfer_rim_arcs(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[(EdgeId, Option<FaceId>)],
    d1: f64,
    d2: f64,
) -> Result<Option<SolidId>, crate::OperationsError> {
    if edges.is_empty() || d1 <= 0.0 || d2 <= 0.0 {
        return Ok(None);
    }
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    let adjacency = brepkit_topology::adjacency::AdjacencyIndex::build_from_faces(topo, &faces)?;
    // One ring for each cap: its arcs on one circle, at one distance.
    let mut rings: HashMap<FaceId, Ring> = HashMap::new();
    for &(edge, reference) in edges {
        match pick(topo, &faces, &adjacency, edge, reference, d1, d2)? {
            None => return Ok(None),
            Some(Pick::Left) => {}
            Some(Pick::Cut(ring)) => {
                if let Some(other) = rings.get(&ring.cap) {
                    let off = ring.origin - other.origin;
                    let same = 1.0 - ring.axis.dot(other.axis).abs() <= PARALLEL_TOL
                        && (off - other.axis * off.dot(other.axis)).length() <= 1e-7
                        && (ring.radius - other.radius).abs() <= 1e-7
                        && (ring.depth - other.depth).abs() <= 1e-9;
                    if !same {
                        return Ok(None);
                    }
                } else {
                    rings.insert(ring.cap, ring);
                }
            }
        }
    }
    if rings.is_empty() {
        return Ok(None);
    }
    let opts = crate::boolean::BooleanOptions {
        mesh_fallback: false,
        ..crate::boolean::BooleanOptions::default()
    };
    let before = crate::measure::solid_volume(topo, solid, 0.01)?;
    let mut removed = 0.0;
    let mut result = solid;
    let mut caps: Vec<FaceId> = rings.keys().copied().collect();
    caps.sort_by_key(|f| f.index());
    for cap in caps {
        let ring = &rings[&cap];
        let margin = 1e-3 * ring.depth;
        let base = brepkit_topology::builder::make_planar_face(topo, &ring.triangle, 1e-7)?;
        let band = crate::revolve::revolve(
            topo,
            base,
            ring.origin,
            ring.axis,
            2.0 * std::f64::consts::PI,
        )?;
        // The cap's prism, from the cap to below the bevel. Its sides stop
        // at the cap: lifted past it, they would lie on the walls rising
        // from the cap and leave slivers of them.
        let top = crate::copy::copy_face(topo, ring.cap)?;
        let prism = crate::extrude::extrude(topo, top, -ring.up, ring.depth + margin)?;
        let cutter = match crate::boolean::boolean_with_options(
            topo,
            crate::boolean::BooleanOp::Intersect,
            band,
            prism,
            opts,
        ) {
            Ok(cutter) => cutter,
            Err(e) => {
                log::debug!("chamfer of rim arcs: the ring's cut to its cap failed: {e}");
                return Ok(None);
            }
        };
        removed += crate::measure::solid_volume(topo, cutter, 0.01)?;
        match crate::boolean::boolean_with_options(
            topo,
            crate::boolean::BooleanOp::Cut,
            result,
            cutter,
            opts,
        ) {
            Ok(cut) => result = cut,
            Err(e) => {
                log::debug!("chamfer of rim arcs: a cut failed: {e}");
                return Ok(None);
            }
        }
    }
    if !crate::validate::validate_solid(topo, result)?.is_valid() {
        log::debug!("chamfer of rim arcs: the result is not valid");
        return Ok(None);
    }
    let after = crate::measure::solid_volume(topo, result, 0.01)?;
    // The cutters reach a little past the corner, into the air.
    let tol = 1e-3 * removed + 1e-7 * before;
    if ((before - after) - removed).abs() > tol {
        log::debug!(
            "chamfer of rim arcs: took off {}, the rings hold {removed}",
            before - after
        );
        return Ok(None);
    }
    Ok(Some(result))
}
