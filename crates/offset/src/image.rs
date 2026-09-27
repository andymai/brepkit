//! Exact offsets by topological image.
//!
//! While no face collapses and no edge turns round, offsetting a solid
//! bounded by analytic faces keeps its topology: each face lies on its
//! surface's offset, each vertex where the offsets of its faces meet, and
//! each line or circle edge on the same kind of curve through the moved
//! vertices. This builds that image directly and declines (`None`) wherever
//! the premise fails, so the caller can take another route.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

use brepkit_math::curves::Circle3D;
use brepkit_math::frame::Frame3;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::error::OffsetError;
use crate::offset::offset_surface;

/// The offset of every face of `solid` by `distance`, built as the image of
/// the solid's own topology, or `None` when that image is not the offset. A
/// solid with cavities is declined: a cavity's image can pass its outer
/// wall's without either turning round.
///
/// # Errors
///
/// Returns [`OffsetError`] if a topology lookup fails.
pub fn offset_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    tol: f64,
) -> Result<Option<SolidId>, OffsetError> {
    let solid_data = topo.solid(solid)?;
    if !solid_data.inner_shells().is_empty() {
        log::debug!("offset image: the solid has cavities");
        return Ok(None);
    }
    let faces = topo.shell(solid_data.outer_shell())?.faces().to_vec();
    let keep: BTreeSet<usize> = faces.iter().map(|f| f.index()).collect();
    let Some(image) = build_image(topo, &faces, &|_| distance, &keep, false, tol)? else {
        return Ok(None);
    };
    if !image
        .faces
        .values()
        .all(|&f| loops_stay_apart(topo, f, tol))
    {
        log::debug!("offset image: a face's loops meet");
        return Ok(None);
    }
    let mapped = faces.iter().map(|f| image.faces[&f.index()]).collect();
    let shell = topo.add_shell(Shell::new(mapped)?);
    Ok(Some(topo.add_solid(Solid::new(shell, vec![]))))
}

/// `solid` hollowed to a wall of thickness `|distance|`, inside its faces
/// when `distance` is negative and outside them when positive, with each of
/// `open` removed and a rim closing the wall where it was; built as two
/// images of the solid's topology, or `None` when they are not the offset.
///
/// The open faces must be planar, without holes or seams, and touch no
/// other open face; a solid with cavities is declined.
///
/// # Errors
///
/// Returns [`OffsetError`] if a topology lookup fails.
pub fn thick_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
    open: &[FaceId],
    tol: f64,
) -> Result<Option<SolidId>, OffsetError> {
    let solid_data = topo.solid(solid)?;
    if !solid_data.inner_shells().is_empty() {
        return Ok(None);
    }
    let faces = topo.shell(solid_data.outer_shell())?.faces().to_vec();
    let open_set: BTreeSet<usize> = open.iter().map(|f| f.index()).collect();
    if open_set
        .iter()
        .any(|i| !faces.iter().any(|f| f.index() == *i))
    {
        return Ok(None);
    }
    let mut open_vertices: BTreeSet<usize> = BTreeSet::new();
    for &f in open {
        let face = topo.face(f)?;
        if !face.inner_wires().is_empty() || !matches!(face.surface(), FaceSurface::Plane { .. }) {
            return Ok(None);
        }
        let wire = topo.wire(face.outer_wire())?;
        let mut seen = BTreeSet::new();
        let mut vertices = BTreeSet::new();
        for oe in wire.edges() {
            if !seen.insert(oe.edge().index()) {
                return Ok(None);
            }
            let edge = topo.edge(oe.edge())?;
            vertices.insert(edge.start().index());
            vertices.insert(edge.end().index());
        }
        if !vertices.is_disjoint(&open_vertices) {
            return Ok(None);
        }
        open_vertices.extend(vertices);
    }

    let closed: BTreeSet<usize> = faces
        .iter()
        .map(|f| f.index())
        .filter(|i| !open_set.contains(i))
        .collect();
    let (outside, inside) = if distance < 0.0 {
        (0.0, distance)
    } else {
        (distance, 0.0)
    };
    let open_ref = &open_set;
    let at = |d: f64| {
        move |f: FaceId| {
            if open_ref.contains(&f.index()) {
                0.0
            } else {
                d
            }
        }
    };
    let Some(outer) = build_image(topo, &faces, &at(outside), &closed, false, tol)? else {
        return Ok(None);
    };
    let Some(inner) = build_image(topo, &faces, &at(inside), &closed, true, tol)? else {
        return Ok(None);
    };

    let mut shell_faces: Vec<FaceId> = Vec::new();
    for image in [&outer, &inner] {
        for &f in image.faces.values() {
            if !loops_stay_apart(topo, f, tol) {
                log::debug!("offset image: a face's loops meet");
                return Ok(None);
            }
            shell_faces.push(f);
        }
    }
    for &f in open {
        let face = topo.face(f)?;
        let (reversed, surface) = (face.is_reversed(), face.surface().clone());
        let wire = topo.wire(face.outer_wire())?.edges().to_vec();
        let rim_outer: Vec<OrientedEdge> = wire
            .iter()
            .map(|oe| OrientedEdge::new(outer.edges[&oe.edge().index()], oe.is_forward()))
            .collect();
        let rim_inner: Vec<OrientedEdge> = wire
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(inner.edges[&oe.edge().index()], !oe.is_forward()))
            .collect();
        let outer_wire = topo.add_wire(Wire::new(rim_outer, true)?);
        let inner_wire = topo.add_wire(Wire::new(rim_inner, true)?);
        let rim = if reversed {
            Face::new_reversed(outer_wire, vec![inner_wire], surface)
        } else {
            Face::new(outer_wire, vec![inner_wire], surface)
        };
        let rim = topo.add_face(rim);
        if !loops_stay_apart(topo, rim, tol) {
            log::debug!("offset image: a rim's loops meet");
            return Ok(None);
        }
        shell_faces.push(rim);
    }

    if open.is_empty() {
        let outer_shell = topo.add_shell(Shell::new(outer.faces.values().copied().collect())?);
        let cavity = topo.add_shell(Shell::new(inner.faces.values().copied().collect())?);
        return Ok(Some(topo.add_solid(Solid::new(outer_shell, vec![cavity]))));
    }
    let shell = topo.add_shell(Shell::new(shell_faces)?);
    Ok(Some(topo.add_solid(Solid::new(shell, vec![]))))
}

/// The new entities of an image, keyed by the originals' arena indices.
struct Image {
    edges: BTreeMap<usize, EdgeId>,
    faces: BTreeMap<usize, FaceId>,
}

/// A face read before any entity is added.
struct FaceData {
    id: FaceId,
    reversed: bool,
    surface: FaceSurface,
    offset: FaceSurface,
    wires: Vec<Vec<OrientedEdge>>,
}

/// The image of `faces` with each face moved `distance(face)` along its
/// outward normal: every vertex and edge of `faces`, and a face for each of
/// `keep` (facing the other way when `flip`, as a cavity's wall does).
/// `None` when a face is not analytic, a vertex sits on a cone's apex or
/// finds no point on all its faces' offsets, or an edge is not a line or a
/// circle, turns round, collapses, or leaves its faces' offsets.
#[allow(clippy::too_many_lines)]
fn build_image(
    topo: &mut Topology,
    faces: &[FaceId],
    distance: &dyn Fn(FaceId) -> f64,
    keep: &BTreeSet<usize>,
    flip: bool,
    tol: f64,
) -> Result<Option<Image>, OffsetError> {
    let mut data = Vec::with_capacity(faces.len());
    for &id in faces {
        let face = topo.face(id)?;
        let surface = face.surface().clone();
        if matches!(surface, FaceSurface::Nurbs(_)) {
            log::debug!("offset image: face {} is NURBS", id.index());
            return Ok(None);
        }
        let d = distance(id);
        let outward = if face.is_reversed() { -d } else { d };
        let Ok(offset) = offset_surface(id, &surface, outward) else {
            log::debug!("offset image: face {} collapses", id.index());
            return Ok(None);
        };
        let wires = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .map(|w| topo.wire(w).map(|w| w.edges().to_vec()))
            .collect::<Result<_, _>>()?;
        data.push(FaceData {
            id,
            reversed: face.is_reversed(),
            surface,
            offset,
            wires,
        });
    }

    let mut edge_faces: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut vertex_faces: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut edges: BTreeMap<usize, (VertexId, VertexId, EdgeCurve)> = BTreeMap::new();
    let mut vertex_ids: BTreeMap<usize, VertexId> = BTreeMap::new();
    for (k, fd) in data.iter().enumerate() {
        for oe in fd.wires.iter().flatten() {
            let edge = topo.edge(oe.edge())?;
            let (s, e) = (edge.start(), edge.end());
            edge_faces.entry(oe.edge().index()).or_default().insert(k);
            for v in [s, e] {
                vertex_faces.entry(v.index()).or_default().insert(k);
                vertex_ids.insert(v.index(), v);
            }
            edges
                .entry(oe.edge().index())
                .or_insert_with(|| (s, e, edge.curve().clone()));
        }
    }

    let mut moved: BTreeMap<usize, Point3> = BTreeMap::new();
    for (vi, ks) in &vertex_faces {
        let p = topo.vertex(vertex_ids[vi])?.point();
        let on_apex = ks.iter().any(
            |&k| matches!(&data[k].surface, FaceSurface::Cone(c) if (p - c.apex()).length() <= tol),
        );
        if on_apex {
            log::debug!("offset image: vertex {vi} is a cone's apex");
            return Ok(None);
        }
        let surfaces: Vec<&FaceSurface> = ks.iter().map(|&k| &data[k].offset).collect();
        let Some(q) = place(p, &surfaces) else {
            log::debug!("offset image: vertex {vi}'s {} faces do not meet", ks.len());
            return Ok(None);
        };
        moved.insert(*vi, q);
    }

    let mut curves: BTreeMap<usize, EdgeCurve> = BTreeMap::new();
    for (ei, (s, e, curve)) in &edges {
        let (p0, p1) = (topo.vertex(*s)?.point(), topo.vertex(*e)?.point());
        let (q0, q1) = (moved[&s.index()], moved[&e.index()]);
        let image = match curve {
            EdgeCurve::Line if s == e => {
                curves.insert(*ei, EdgeCurve::Line);
                continue;
            }
            EdgeCurve::Line => {
                let (was, now) = (p1 - p0, q1 - q0);
                if now.length() <= tol || was.dot(now) <= 0.0 {
                    log::debug!("offset image: line {ei} turns round or collapses");
                    return Ok(None);
                }
                EdgeCurve::Line
            }
            EdgeCurve::Circle(c) => {
                let Some(c) = image_circle(c, p0, q0, q1, tol) else {
                    log::debug!("offset image: circle {ei} leaves its axis or collapses");
                    return Ok(None);
                };
                EdgeCurve::Circle(c)
            }
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                log::debug!("offset image: edge {ei} is not a line or a circle");
                return Ok(None);
            }
        };
        // Where the edge lies on its faces, its image does too, and an arc's
        // middle stays on its side of the axis (not the complement).
        let (t0, t1) = curve.domain_with_endpoints(p0, p1);
        let x = curve.evaluate_with_endpoints(0.5 * (t0 + t1), p0, p1);
        let ks = &edge_faces[ei];
        let on_faces = ks.iter().all(|&k| {
            residual(&data[k].surface, x).is_some_and(|(r, _)| r.abs() <= 1e-9 * scale_of(x))
        });
        if on_faces {
            let (s0, s1) = image.domain_with_endpoints(q0, q1);
            let z = image.evaluate_with_endpoints(0.5 * (s0 + s1), q0, q1);
            let stays = ks.iter().all(|&k| {
                residual(&data[k].offset, z).is_some_and(|(r, _)| r.abs() <= 1e-8 * scale_of(z))
            });
            let same_side = match (curve, &image) {
                (EdgeCurve::Circle(was), EdgeCurve::Circle(now)) => {
                    (z - now.center()).dot(x - was.center()) > 0.0
                }
                _ => true,
            };
            if !stays || !same_side {
                log::debug!("offset image: edge {ei}'s image leaves its faces' offsets");
                return Ok(None);
            }
        }
        curves.insert(*ei, image);
    }

    let new_vertices: BTreeMap<usize, VertexId> = moved
        .iter()
        .map(|(&vi, &q)| (vi, topo.add_vertex(Vertex::new(q, tol))))
        .collect();
    let mut new_edges: BTreeMap<usize, EdgeId> = BTreeMap::new();
    for (ei, curve) in curves {
        let (s, e, _) = &edges[&ei];
        let edge = Edge::new(new_vertices[&s.index()], new_vertices[&e.index()], curve);
        new_edges.insert(ei, topo.add_edge(edge));
    }
    let mut new_faces: BTreeMap<usize, FaceId> = BTreeMap::new();
    for fd in data.iter().filter(|fd| keep.contains(&fd.id.index())) {
        let mut wire_ids: Vec<WireId> = Vec::with_capacity(fd.wires.len());
        for wire in &fd.wires {
            let mapped = wire
                .iter()
                .map(|oe| OrientedEdge::new(new_edges[&oe.edge().index()], oe.is_forward()))
                .collect();
            wire_ids.push(topo.add_wire(Wire::new(mapped, true)?));
        }
        let outer = wire_ids.remove(0);
        let face = if fd.reversed == flip {
            Face::new(outer, wire_ids, fd.offset.clone())
        } else {
            Face::new_reversed(outer, wire_ids, fd.offset.clone())
        };
        new_faces.insert(fd.id.index(), topo.add_face(face));
    }
    Ok(Some(Image {
        edges: new_edges,
        faces: new_faces,
    }))
}

/// A point's size, for tolerances relative to its coordinates.
fn scale_of(p: Point3) -> f64 {
    1.0 + (p - Point3::new(0.0, 0.0, 0.0)).length()
}

/// A point's signed distance from a surface along the surface's normal at
/// its foot, and that normal.
fn residual(surface: &FaceSurface, p: Point3) -> Option<(f64, Vec3)> {
    if let FaceSurface::Plane { normal, d } = surface {
        return Some((normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d, *normal));
    }
    let (u, v) = surface.project_point(p)?;
    let foot = surface.evaluate(u, v)?;
    let n = surface.normal(u, v);
    Some(((p - foot).dot(n), n))
}

/// The point nearest `start` on every one of `surfaces`, by Gauss-Newton
/// steps of least length. `None` if the surfaces do not meet there.
fn place(start: Point3, surfaces: &[&FaceSurface]) -> Option<Point3> {
    let scale = scale_of(start);
    let mut p = start;
    for _ in 0..64 {
        let rows: Vec<(f64, Vec3)> = surfaces
            .iter()
            .map(|s| residual(s, p))
            .collect::<Option<_>>()?;
        if rows.iter().all(|(r, _)| r.abs() <= 1e-14 * scale) {
            return Some(p);
        }
        // Tangent faces share a normal: step along an independent subset.
        let mut basis: Vec<Vec3> = Vec::with_capacity(3);
        let mut kept: Vec<(f64, Vec3)> = Vec::with_capacity(3);
        for &(r, n) in &rows {
            let w = basis.iter().fold(n, |w, q| w - *q * w.dot(*q));
            let len = w.length();
            if len > 1e-6 {
                basis.push(w * (1.0 / len));
                kept.push((r, n));
            }
        }
        p = p + least_step(&kept)?;
    }
    let settled = surfaces
        .iter()
        .all(|s| residual(s, p).is_some_and(|(r, _)| r.abs() <= 1e-9 * scale));
    settled.then_some(p)
}

/// The shortest step `s` with `n_i · s = -r_i` for independent rows.
fn least_step(rows: &[(f64, Vec3)]) -> Option<Vec3> {
    let k = rows.len();
    let mut a = [[0.0_f64; 4]; 3];
    for i in 0..k {
        for j in 0..k {
            a[i][j] = rows[i].1.dot(rows[j].1);
        }
        a[i][3] = rows[i].0;
    }
    // The Gram matrix of independent rows is positive definite.
    for col in 0..k {
        let pivot = a[col][col];
        if pivot.abs() < 1e-14 {
            return None;
        }
        for row in 0..k {
            if row != col {
                let f = a[row][col] / pivot;
                for c in col..4 {
                    a[row][c] -= f * a[col][c];
                }
            }
        }
    }
    Some(
        rows.iter()
            .enumerate()
            .fold(Vec3::new(0.0, 0.0, 0.0), |s, (i, &(_, n))| {
                s - n * (a[i][3] / a[i][i])
            }),
    )
}

/// The circle of `c`'s axis through the moved start `q0`, or `None` if the
/// start crossed the axis or the moved end `q1` is off it.
fn image_circle(c: &Circle3D, p0: Point3, q0: Point3, q1: Point3, tol: f64) -> Option<Circle3D> {
    let n = c.normal();
    let center = c.center() + n * (q0 - c.center()).dot(n);
    let radius = (q0 - center).length();
    if radius <= tol || (q0 - center).dot(p0 - c.center()) <= 0.0 {
        return None;
    }
    let to_end = q1 - center;
    let slack = 1e-8 * scale_of(q1);
    if to_end.dot(n).abs() > slack || (to_end.length() - radius).abs() > slack {
        return None;
    }
    Circle3D::with_axes(center, n, radius, c.u_axis(), c.v_axis()).ok()
}

/// Whether a planar face's loops keep clear of one another and of
/// themselves, each hole inside its outer loop and outside the other holes.
/// A curved face is taken as it is when it has no holes and declined when it
/// has. Walls passing through each other far apart cross on the planar faces
/// between them, as a prism's neck does on its caps.
fn loops_stay_apart(topo: &Topology, face_id: FaceId, tol: f64) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let FaceSurface::Plane { normal, .. } = face.surface() else {
        return face.inner_wires().is_empty();
    };
    let Ok(frame) = Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), *normal) else {
        return false;
    };
    let flat = |p: Point3| {
        let v = p - Point3::new(0.0, 0.0, 0.0);
        (v.dot(frame.x), v.dot(frame.y))
    };
    let mut loops: Vec<Vec<Segment>> = Vec::new();
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let Some(points) = wire_points(topo, wid) else {
            return false;
        };
        let n = points.len();
        loops.push(
            (0..n)
                .map(|i| Segment {
                    a: flat(points[i].0),
                    b: flat(points[(i + 1) % n].0),
                    sag: points[i].1,
                })
                .collect(),
        );
    }
    let near =
        |s: &Segment, t: &Segment| segment_distance(s.a, s.b, t.a, t.b) <= tol + s.sag + t.sag;
    for (i, lp) in loops.iter().enumerate() {
        let n = lp.len();
        for a in 0..n {
            for b in (a + 2)..n {
                if !(a == 0 && b == n - 1) && near(&lp[a], &lp[b]) {
                    return false;
                }
            }
        }
        for other in &loops[i + 1..] {
            if lp.iter().any(|s| other.iter().any(|t| near(s, t))) {
                return false;
            }
        }
    }
    let polygon = |lp: &[Segment]| lp.iter().map(|s| s.a).collect::<Vec<_>>();
    let outer = polygon(&loops[0]);
    let holes: Vec<Vec<(f64, f64)>> = loops[1..].iter().map(|lp| polygon(lp)).collect();
    holes.iter().enumerate().all(|(i, hole)| {
        inside(hole[0], &outer)
            && !inside(outer[0], hole)
            && holes
                .iter()
                .enumerate()
                .all(|(j, other)| j == i || !inside(hole[0], other))
    })
}

/// A chord of a flattened loop, and how far the curve it stands for may bow
/// away from it.
struct Segment {
    a: (f64, f64),
    b: (f64, f64),
    sag: f64,
}

/// A wire's points in traversal order, each with the sagitta of the chord
/// to the next: each edge's start, and 128 steps a turn along a circle.
fn wire_points(topo: &Topology, wire: WireId) -> Option<Vec<(Point3, f64)>> {
    let mut points = Vec::new();
    for oe in topo.wire(wire).ok()?.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let (a, b) = (
            topo.vertex(edge.start()).ok()?.point(),
            topo.vertex(edge.end()).ok()?.point(),
        );
        let curve = edge.curve();
        let (t0, t1) = curve.domain_with_endpoints(a, b);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (steps, sag) = match curve {
            EdgeCurve::Circle(c) => {
                let steps = ((t1 - t0).abs() / TAU * 128.0).ceil().max(1.0) as usize;
                #[allow(clippy::cast_precision_loss)]
                let half = (t1 - t0).abs() / (2.0 * steps as f64);
                (steps, c.radius() * (1.0 - half.cos()))
            }
            _ => (1, 0.0),
        };
        for k in 0..steps {
            #[allow(clippy::cast_precision_loss)]
            let f = k as f64 / steps as f64;
            let f = if oe.is_forward() { f } else { 1.0 - f };
            points.push((
                curve.evaluate_with_endpoints((t1 - t0).mul_add(f, t0), a, b),
                sag,
            ));
        }
    }
    Some(points)
}

/// The distance between two 2D segments.
fn segment_distance(p: (f64, f64), q: (f64, f64), r: (f64, f64), s: (f64, f64)) -> f64 {
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0).mul_add(b.1 - o.1, -((a.1 - o.1) * (b.0 - o.0)))
    };
    let (d1, d2) = (cross(p, q, r), cross(p, q, s));
    let (d3, d4) = (cross(r, s, p), cross(r, s, q));
    if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
        return 0.0;
    }
    let to_segment = |x: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len2 = dx.mul_add(dx, dy * dy);
        let t = if len2 > 0.0 {
            ((x.0 - a.0).mul_add(dx, (x.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (x.0 - t.mul_add(dx, a.0)).hypot(x.1 - t.mul_add(dy, a.1))
    };
    to_segment(p, r, s)
        .min(to_segment(q, r, s))
        .min(to_segment(r, p, q))
        .min(to_segment(s, p, q))
}

/// Whether a point is inside a closed polyline, by crossings of a ray.
fn inside(x: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let mut odd = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a.1 > x.1) != (b.1 > x.1) {
            let at = (b.0 - a.0).mul_add((x.1 - a.1) / (b.1 - a.1), a.0);
            if x.0 < at {
                odd = !odd;
            }
        }
    }
    odd
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
    use brepkit_operations::transform::transform_solid;

    /// Two 4 x 4 blocks joined by a neck 1 wide, 2 tall.
    fn necked_prism(topo: &mut Topology) -> SolidId {
        let corners = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.5),
            (6.0, 1.5),
            (6.0, 0.0),
            (10.0, 0.0),
            (10.0, 4.0),
            (6.0, 4.0),
            (6.0, 2.5),
            (4.0, 2.5),
            (4.0, 4.0),
            (0.0, 4.0),
        ];
        let ids: Vec<VertexId> = corners
            .iter()
            .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
            .collect();
        let n = ids.len();
        let edges = (0..n)
            .map(|i| {
                let e = topo.add_edge(Edge::new(ids[i], ids[(i + 1) % n], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap()
    }

    /// Offset in by 0.75 the neck's walls pass each other, though no edge
    /// turns round: the image is declined. By 0.2 it is taken.
    #[test]
    fn a_neck_offset_past_its_width_is_declined() {
        let mut topo = Topology::new();
        let solid = necked_prism(&mut topo);
        assert!(
            offset_solid(&mut topo, solid, -0.75, 1e-7)
                .unwrap()
                .is_none()
        );
        assert!(
            offset_solid(&mut topo, solid, -0.2, 1e-7)
                .unwrap()
                .is_some()
        );
    }

    /// A 10 x 10 x 4 plate bored by a radius-2 hole turned a 64th of a turn
    /// about its axis, offset in by 1.505: the hole's image reaches past the
    /// plate's sides by 0.01, less than a 32-chord polygon of it bows, and is
    /// declined. By 1.49 it stays 0.02 clear and is taken.
    #[test]
    fn a_hole_reaching_a_side_by_less_than_a_chord_is_declined() {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 10.0, 10.0, 4.0).unwrap();
        let rod = make_cylinder(&mut topo, 2.0, 10.0).unwrap();
        let place =
            Mat4::translation(5.0, 5.0, -3.0) * Mat4::rotation_z(std::f64::consts::PI / 32.0);
        transform_solid(&mut topo, rod, &place).unwrap();
        let bored = boolean(&mut topo, BooleanOp::Cut, plate, rod).unwrap();
        assert!(
            offset_solid(&mut topo, bored, -1.505, 1e-7)
                .unwrap()
                .is_none()
        );
        assert!(
            offset_solid(&mut topo, bored, -1.49, 1e-7)
                .unwrap()
                .is_some()
        );
    }

    /// A ball hollowed to a wall 1 thick offset in by 0.6 would put its
    /// cavity's image outside its outer wall's: a solid with a cavity is
    /// declined.
    #[test]
    fn a_solid_with_a_cavity_is_declined() {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, 5.0, 32).unwrap();
        let hollow = thick_solid(&mut topo, ball, -1.0, &[], 1e-7)
            .unwrap()
            .unwrap();
        assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
        assert!(
            offset_solid(&mut topo, hollow, -0.6, 1e-7)
                .unwrap()
                .is_none()
        );
    }
}
