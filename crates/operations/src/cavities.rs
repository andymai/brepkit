//! A solid's cavities kept through an operation that rebuilds only its
//! outer shell.

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Points whose box holds every point of a NURBS curve or surface: its
/// control points while every weight is positive, else an unbounded box.
fn hull_or_everything<'a>(
    control_points: impl Iterator<Item = &'a Point3>,
    mut weights: impl Iterator<Item = f64>,
) -> Vec<Point3> {
    if weights.all(|w| w > 0.0) {
        control_points.copied().collect()
    } else {
        let d = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        vec![
            Point3::new(0.0, 0.0, 0.0) - d,
            Point3::new(0.0, 0.0, 0.0) + d,
        ]
    }
}

/// A box holding a face: its measured box, widened to a NURBS face's
/// control points (the measured box samples it at a few points only), to a
/// sphere's or torus's whole surface (it takes that only for a whole one,
/// and a trimmed band can bulge past its boundary), and to everything where
/// a boundary NURBS edge has a weight at or below zero (such an edge can
/// run outside its control points).
fn face_bounds(topo: &Topology, face: FaceId) -> Result<Aabb3, OperationsError> {
    let measured = crate::measure::face_set_bounding_box(topo, &[face])?;
    let mut points = vec![measured.min, measured.max];
    let data = topo.face(face)?;
    match data.surface() {
        FaceSurface::Nurbs(n) => points.extend(hull_or_everything(
            n.control_points().iter().flatten(),
            n.weights().iter().flatten().copied(),
        )),
        FaceSurface::Sphere(sphere) => {
            let whole = sphere.aabb();
            points.extend([whole.min, whole.max]);
        }
        FaceSurface::Torus(torus) => {
            let whole = torus.aabb();
            points.extend([whole.min, whole.max]);
        }
        FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {}
    }
    for wire in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wire)?.edges() {
            if let EdgeCurve::NurbsCurve(n) = topo.edge(oe.edge())?.curve() {
                points.extend(hull_or_everything(
                    n.control_points().iter(),
                    n.weights().iter().copied(),
                ));
            }
        }
    }
    Ok(Aabb3::from_points(points))
}

/// A box holding every face of `shell`.
fn shell_bounds(topo: &Topology, shell: ShellId) -> Result<Aabb3, OperationsError> {
    let mut bounds: Option<Aabb3> = None;
    for &face in topo.shell(shell)?.faces() {
        let b = face_bounds(topo, face)?;
        bounds = Some(bounds.map_or(b, |a| a.union(b)));
    }
    bounds.ok_or(OperationsError::Topology(
        brepkit_topology::TopologyError::Empty {
            entity: "cavity shell faces",
        },
    ))
}

/// Whether `a` and `b` are one surface, within a little of each other's
/// defining values. A surface reads as new where in doubt.
#[allow(clippy::many_single_char_names)]
fn same_surface(a: &FaceSurface, b: &FaceSurface) -> bool {
    const EPS: f64 = 1e-9;
    // A face passed through keeps its surface as it was, so points match
    // exactly; a margin growing with distance from the origin would let a
    // different surface far away through.
    let near = |p: Point3, q: Point3| (p - q).length() <= EPS;
    let close = |x: f64, y: f64| (x - y).abs() <= EPS * (1.0 + x.abs());
    let parallel = |u: Vec3, v: Vec3| u.cross(v).length() <= EPS;
    // A point on an axis through `origin` along `axis`.
    let on_axis = |origin: Point3, axis: Vec3, p: Point3| {
        let w = p - origin;
        (w - axis * axis.dot(w)).length() <= EPS * (1.0 + w.length())
    };
    match (a, b) {
        (FaceSurface::Plane { normal: n, d }, FaceSurface::Plane { normal: m, d: e }) => {
            let dot = n.dot(*m);
            (close(dot, 1.0) && close(*d, *e)) || (close(dot, -1.0) && close(*d, -*e))
        }
        (FaceSurface::Cylinder(x), FaceSurface::Cylinder(y)) => {
            close(x.radius(), y.radius())
                && parallel(x.axis(), y.axis())
                && on_axis(x.origin(), x.axis(), y.origin())
        }
        (FaceSurface::Cone(x), FaceSurface::Cone(y)) => {
            near(x.apex(), y.apex())
                && x.axis().dot(y.axis()) > 1.0 - EPS
                && close(x.half_angle(), y.half_angle())
        }
        (FaceSurface::Sphere(x), FaceSurface::Sphere(y)) => {
            near(x.center(), y.center()) && close(x.radius(), y.radius())
        }
        (FaceSurface::Torus(x), FaceSurface::Torus(y)) => {
            near(x.center(), y.center())
                && parallel(x.z_axis(), y.z_axis())
                && close(x.major_radius(), y.major_radius())
                && close(x.minor_radius(), y.minor_radius())
        }
        (FaceSurface::Nurbs(x), FaceSurface::Nurbs(y)) => {
            x.degree_u() == y.degree_u()
                && x.degree_v() == y.degree_v()
                && x.knots_u() == y.knots_u()
                && x.knots_v() == y.knots_v()
                && x.weights() == y.weights()
                && x.control_points().len() == y.control_points().len()
                && x.control_points()
                    .iter()
                    .flatten()
                    .zip(y.control_points().iter().flatten())
                    .all(|(p, q)| near(*p, *q))
        }
        _ => false,
    }
}

/// Whether a planar face is convex, with straight edges and no holes.
fn convex_without_holes(
    topo: &Topology,
    face: FaceId,
    normal: Vec3,
) -> Result<bool, OperationsError> {
    let data = topo.face(face)?;
    if !data.inner_wires().is_empty() {
        return Ok(false);
    }
    let mut corners = Vec::new();
    for oe in topo.wire(data.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        if !matches!(edge.curve(), EdgeCurve::Line) {
            return Ok(false);
        }
        corners.push(topo.vertex(oe.oriented_start(edge))?.point());
    }
    let n = corners.len();
    if n < 3 {
        return Ok(false);
    }
    let mut left: Option<bool> = None;
    for i in 0..n {
        let (a, b, c) = (corners[i], corners[(i + 1) % n], corners[(i + 2) % n]);
        let turn = (b - a).cross(c - b).dot(normal);
        if turn.abs() <= 1e-12 * (b - a).length() * (c - b).length() {
            continue;
        }
        match left {
            None => left = Some(turn > 0.0),
            Some(side) if side != (turn > 0.0) => return Ok(false),
            Some(_) => {}
        }
    }
    Ok(true)
}

/// Whether a face of the result lies on one of `source_faces`: on its
/// surface and, for a plane, a convex face without holes holding every
/// vertex on the plane and in or on its region (a face the operation made
/// can share the plane of a face far from it), for another surface within
/// its box.
fn on_source_face(
    topo: &Topology,
    face: FaceId,
    source_faces: &[FaceId],
) -> Result<bool, OperationsError> {
    let surface = topo.face(face)?.surface().clone();
    let vertices = brepkit_topology::explorer::face_vertices(topo, face)?
        .into_iter()
        .map(|v| topo.vertex(v).map(brepkit_topology::vertex::Vertex::point))
        .collect::<Result<Vec<_>, _>>()?;
    let Some(&origin) = vertices.first() else {
        return Ok(false);
    };
    for &source_face in source_faces {
        let source_surface = topo.face(source_face)?.surface();
        if !same_surface(source_surface, &surface) {
            continue;
        }
        let within = if let FaceSurface::Plane { normal, d } = source_surface {
            // The surfaces match within a margin relative to the plane's
            // offset, which far from the origin lets a slightly tilted cut
            // through: every vertex must lie on the plane itself.
            let length = normal.length();
            if vertices
                .iter()
                .any(|p| ((normal.dot(*p - Point3::new(0.0, 0.0, 0.0)) - d) / length).abs() > 1e-7)
            {
                continue;
            }
            // Inside a convex face with straight edges and no holes, a face
            // whose vertices all lie in it lies in it whole; inside any
            // other it could span a hole or notch, and reads as made.
            if !convex_without_holes(topo, source_face, *normal)? {
                continue;
            }
            let frame = brepkit_math::frame::Frame3::from_normal(origin, *normal)?;
            let Some(pieces) = brepkit_topology::planar::face_boundary_2d(
                topo,
                source_face,
                origin,
                frame.x,
                frame.y,
            )?
            else {
                continue;
            };
            vertices.iter().all(|&p| {
                let d = p - origin;
                let q = brepkit_math::vec::Point2::new(d.dot(frame.x), d.dot(frame.y));
                brepkit_math::region2d::point_in_region(&pieces, q, 1e-7) != Some(false)
            })
        } else {
            let bounds = face_bounds(topo, source_face)?.expanded(1e-7);
            vertices.iter().all(|&p| bounds.contains_point(p))
        };
        if within {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Registers on `copy`'s faces the pcurves of `original`'s, which
/// `copy_shells` leaves behind: a curved cavity face keeps its UV trims.
/// The copy lists faces, wires and edges in the original's order.
fn copy_pcurves(
    topo: &mut Topology,
    original: ShellId,
    copy: ShellId,
) -> Result<(), OperationsError> {
    let faces: Vec<(FaceId, FaceId)> = topo
        .shell(original)?
        .faces()
        .iter()
        .copied()
        .zip(topo.shell(copy)?.faces().iter().copied())
        .collect();
    for (old_face, new_face) in faces {
        let wires = |f: FaceId| -> Result<Vec<brepkit_topology::wire::WireId>, OperationsError> {
            let face = topo.face(f)?;
            Ok(std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .collect())
        };
        let mut edges = std::collections::HashMap::new();
        for (old_wire, new_wire) in wires(old_face)?.into_iter().zip(wires(new_face)?) {
            for (a, b) in topo
                .wire(old_wire)?
                .edges()
                .iter()
                .zip(topo.wire(new_wire)?.edges())
            {
                edges.insert(a.edge(), b.edge());
            }
        }
        let pcurves: Vec<_> = topo
            .pcurves()
            .pcurves_for_face(old_face)
            .into_iter()
            .map(|(edge, pcurve)| (edge, pcurve.clone()))
            .collect();
        for (edge, pcurve) in pcurves {
            if let Some(&new_edge) = edges.get(&edge) {
                topo.pcurves_mut().set(new_edge, new_face, pcurve);
            }
        }
    }
    Ok(())
}

/// `result`, rebuilt from `source`'s outer shell alone, given copies of
/// `source`'s cavities.
///
/// The material taken off is bounded by parts of `source`'s outer faces,
/// which a cavity, lying strictly inside, never reaches, and by the faces
/// the operation made, those of `result` on no face of `source`. A
/// cavity meets that material only by crossing one of those faces or by
/// lying wholly in it, so one whose box meets such a face's box, or whose
/// vertex is not inside `result`, refuses the operation, named `what` in
/// the error.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if a cavity could meet the
/// material taken off, or a topology error if a lookup fails.
pub fn keep_cavities(
    topo: &mut Topology,
    source: SolidId,
    result: SolidId,
    what: &str,
) -> Result<SolidId, OperationsError> {
    let cavities = topo.solid(source)?.inner_shells().to_vec();
    if cavities.is_empty() {
        return Ok(result);
    }
    let source_faces = topo
        .shell(topo.solid(source)?.outer_shell())?
        .faces()
        .to_vec();
    let mut made = Vec::new();
    for &face in topo.shell(topo.solid(result)?.outer_shell())?.faces() {
        if !on_source_face(topo, face, &source_faces)? {
            made.push(face);
        }
    }
    clear_of_cavities(topo, &cavities, result, &made, what)?;
    // The result gets its own copy: a shell shared with the source would
    // move with it under an in-place transform of either.
    let copied = crate::copy::copy_shells(topo, &cavities)?;
    for (&original, &copy) in cavities.iter().zip(&copied) {
        copy_pcurves(topo, original, copy)?;
    }
    let solid = topo.solid_mut(result)?;
    for cavity in copied {
        solid.add_inner_shell(cavity);
    }
    Ok(result)
}

/// A walking blend's `result` as a solid of its own, holding copies of
/// `source`'s cavities.
///
/// The walking engine rebuilds only the outer shell: the faces it leaves
/// as they were come back as clones whose wires run on `source`'s own
/// edges, and the cavity shells as they are. The result's outer shell and
/// the cavities are copied, with their pcurves, so moving the result in
/// place leaves `source` where it was. The faces the blend made are those
/// whose edges are no face of `source`'s and that lie on no face of it, and
/// a cavity one of them could reach refuses the blend, named `what` in the
/// error, as in [`keep_cavities`].
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if a cavity could meet the
/// material taken off, or a topology error if a lookup fails.
pub fn detach_walked(
    topo: &mut Topology,
    source: SolidId,
    result: SolidId,
    what: &str,
) -> Result<SolidId, OperationsError> {
    let edges_of = |topo: &Topology, face: FaceId| -> Result<Vec<usize>, OperationsError> {
        let data = topo.face(face)?;
        let mut edges = Vec::new();
        for wire in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
            edges.extend(topo.wire(wire)?.edges().iter().map(|oe| oe.edge().index()));
        }
        edges.sort_unstable();
        Ok(edges)
    };
    let outer = topo.solid(result)?.outer_shell();
    let cavities = topo.solid(source)?.inner_shells().to_vec();
    if !cavities.is_empty() {
        let source_faces = topo
            .shell(topo.solid(source)?.outer_shell())?
            .faces()
            .to_vec();
        let mut kept = std::collections::HashSet::new();
        for &face in &source_faces {
            kept.insert(edges_of(topo, face)?);
        }
        let mut made = Vec::new();
        for &face in topo.shell(outer)?.faces() {
            if !kept.contains(&edges_of(topo, face)?) && !on_source_face(topo, face, &source_faces)?
            {
                made.push(face);
            }
        }
        // The cavities are read against the outer shell alone: the walked
        // result still holds them, so their vertices lie on its boundary.
        let bare = topo.add_solid(brepkit_topology::solid::Solid::new(outer, Vec::new()));
        clear_of_cavities(topo, &cavities, bare, &made, what)?;
    }

    let shells: Vec<ShellId> = std::iter::once(outer)
        .chain(cavities.iter().copied())
        .collect();
    let copied = crate::copy::copy_shells(topo, &shells)?;
    for (&original, &copy) in shells.iter().zip(&copied) {
        copy_pcurves(topo, original, copy)?;
    }
    let (&own_outer, own_cavities) =
        copied
            .split_first()
            .ok_or_else(|| OperationsError::InvalidInput {
                reason: format!("the {what} left no shell"),
            })?;
    Ok(topo.add_solid(brepkit_topology::solid::Solid::new(
        own_outer,
        own_cavities.to_vec(),
    )))
}

/// Refuses a blend that made `made` on `result`, which carries its input's
/// cavities, if one of the cavities holding none of `made` could meet the
/// material the blend took off.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if a cavity could, and
/// propagates topology and classification errors.
pub fn clear_of_own_cavities(
    topo: &mut Topology,
    result: SolidId,
    made: &[FaceId],
    what: &str,
) -> Result<(), OperationsError> {
    let data = topo.solid(result)?;
    let outer = data.outer_shell();
    let mut untouched = Vec::new();
    for &cavity in data.inner_shells() {
        if !topo.shell(cavity)?.faces().iter().any(|f| made.contains(f)) {
            untouched.push(cavity);
        }
    }
    if untouched.is_empty() {
        return Ok(());
    }
    // The cavities are read against the outer shell alone: the result holds
    // them, so their vertices lie on its boundary.
    let bare = topo.add_solid(brepkit_topology::solid::Solid::new(outer, Vec::new()));
    clear_of_cavities(topo, &untouched, bare, made, what)
}

/// Refuses an operation that made `made` on `result` if a cavity could
/// meet the material it took off: a cavity whose box meets a made face's
/// box, or whose vertex is not inside `result`.
fn clear_of_cavities(
    topo: &Topology,
    cavities: &[ShellId],
    result: SolidId,
    made: &[FaceId],
    what: &str,
) -> Result<(), OperationsError> {
    let mut made_bounds = Vec::with_capacity(made.len());
    for &face in made {
        made_bounds.push(face_bounds(topo, face)?);
    }
    let made = made_bounds;
    let refused = || OperationsError::InvalidInput {
        reason: format!("the {what} would cut into a cavity of the solid"),
    };
    for &cavity in cavities {
        let bounds = shell_bounds(topo, cavity)?;
        if made.iter().any(|m| m.intersects(bounds)) {
            return Err(refused());
        }
        let face = topo.shell(cavity)?.faces().first().copied();
        let vertex = match face {
            Some(f) => brepkit_topology::explorer::face_vertices(topo, f)?
                .first()
                .copied(),
            None => None,
        };
        if let Some(v) = vertex {
            let p = topo.vertex(v)?.point();
            let inside = crate::classify::classify_point(topo, result, p, 0.0, 1e-7)?;
            if inside != crate::classify::PointClassification::Inside {
                return Err(refused());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::nurbs::NurbsSurface;
    use brepkit_topology::edge::Edge;
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// A unit square patch, flat but for one control point raised 10 whose
    /// basis lives on (0.05, 0.2) in u and v: between every interior sample
    /// the measured box takes, and clear of the boundary.
    #[test]
    fn a_nurbs_bump_between_samples_is_inside_the_bounds() {
        let knots = vec![0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 1.0, 1.0, 1.0];
        let n = knots.len() - 3;
        #[allow(clippy::cast_precision_loss)]
        let at = |i: usize| i as f64 / (n - 1) as f64;
        let points: Vec<Vec<Point3>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| Point3::new(at(i), at(j), if (i, j) == (3, 3) { 10.0 } else { 0.0 }))
                    .collect()
            })
            .collect();
        let surface =
            NurbsSurface::new(2, 2, knots.clone(), knots, points, vec![vec![1.0; n]; n]).unwrap();

        let mut topo = Topology::new();
        let corners: Vec<_> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
            .collect();
        let sides: Vec<_> = (0..4)
            .map(|k| {
                let e = topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(sides, true).unwrap());
        let face = topo.add_face(Face::new(wire, Vec::new(), FaceSurface::Nurbs(surface)));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());

        let measured = crate::measure::face_set_bounding_box(&topo, &[face]).unwrap();
        assert!(measured.max.z() < 1e-9, "{measured:?}");
        let bounds = shell_bounds(&topo, shell).unwrap();
        assert!(bounds.max.z() >= 10.0, "{bounds:?}");
    }

    /// A band of a torus (major 10, minor 3) a quarter turn round its ring
    /// and over the outer half of its tube: its boundary arcs stay within
    /// x <= 10, but the band reaches x = 13 at its middle.
    #[test]
    fn a_trimmed_torus_band_bulging_past_its_boundary_is_inside_the_bounds() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ToroidalSurface;
        use std::f64::consts::FRAC_PI_4;

        let mut topo = Topology::new();
        let corner = |u: f64, z: f64| Point3::new(10.0 * u.cos(), 10.0 * u.sin(), z);
        let v: Vec<_> = [
            (-FRAC_PI_4, -3.0),
            (FRAC_PI_4, -3.0),
            (FRAC_PI_4, 3.0),
            (-FRAC_PI_4, 3.0),
        ]
        .iter()
        .map(|&(u, z)| topo.add_vertex(Vertex::new(corner(u, z), 1e-7)))
        .collect();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let ring = |z: f64| Circle3D::new(Point3::new(0.0, 0.0, z), up, 10.0).unwrap();
        let meridian = |u: f64| {
            Circle3D::new(
                Point3::new(10.0 * u.cos(), 10.0 * u.sin(), 0.0),
                Vec3::new(-u.sin(), u.cos(), 0.0),
                3.0,
            )
            .unwrap()
        };
        let curves = [
            EdgeCurve::Circle(ring(-3.0)),
            EdgeCurve::Circle(meridian(FRAC_PI_4)),
            EdgeCurve::Circle(ring(3.0)),
            EdgeCurve::Circle(meridian(-FRAC_PI_4)),
        ];
        let sides: Vec<_> = curves
            .into_iter()
            .enumerate()
            .map(|(k, curve)| {
                let e = topo.add_edge(Edge::new(v[k], v[(k + 1) % 4], curve));
                OrientedEdge::new(e, true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(sides, true).unwrap());
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 10.0, 3.0).unwrap();
        let face = topo.add_face(Face::new(wire, Vec::new(), FaceSurface::Torus(torus)));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());

        let measured = crate::measure::face_set_bounding_box(&topo, &[face]).unwrap();
        assert!(measured.max.x() < 13.0 - 1e-6, "{measured:?}");
        let bounds = shell_bounds(&topo, shell).unwrap();
        assert!(bounds.max.x() >= 13.0, "{bounds:?}");
    }

    /// Keeping a hollow block's cavity copies its six faces and nothing
    /// more.
    #[test]
    fn keeping_a_cavity_copies_only_it() {
        use brepkit_math::mat::Mat4;
        use brepkit_topology::solid::Solid;

        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let cube = crate::primitives::make_box(&mut topo, 3.0, 3.0, 3.0).unwrap();
        crate::transform::transform_solid(&mut topo, cube, &Mat4::translation(3.5, 3.5, 3.5))
            .unwrap();
        let hollow =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, block, cube)
                .unwrap();
        let outer = topo.solid(hollow).unwrap().outer_shell();
        let result = topo.add_solid(Solid::new(outer, Vec::new()));
        let faces = topo.num_faces();
        keep_cavities(&mut topo, hollow, result, "test").unwrap();
        assert_eq!(topo.num_faces() - faces, 6);
        assert_eq!(topo.solid(result).unwrap().inner_shells().len(), 1);
    }

    /// A boundary NURBS edge with a weight below zero can run outside its
    /// control points, so a face it bounds leaves the cavity unbounded.
    #[test]
    fn a_nonpositive_weight_boundary_leaves_the_bounds_open() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(-1.0, 0.0, 0.0), 1e-7));
        let arc = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
            ],
            vec![1.0, -0.5, 1.0],
        )
        .unwrap();
        let curved = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(arc)));
        let straight = topo.add_edge(Edge::new(b, a, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(curved, true),
                    OrientedEdge::new(straight, true),
                ],
                true,
            )
            .unwrap(),
        );
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let face = topo.add_face(Face::new(wire, Vec::new(), plane));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        let bounds = shell_bounds(&topo, shell).unwrap();
        assert!(bounds.max.x().is_infinite(), "{bounds:?}");
    }

    /// A result that leaves the cavity out wholly, its new face (the top at
    /// z = 3) well clear of it: the cavity's vertex outside the result
    /// refuses it.
    #[test]
    fn a_cavity_wholly_in_the_material_taken_off_is_refused() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let cube = crate::primitives::make_box(&mut topo, 3.0, 3.0, 3.0).unwrap();
        crate::transform::transform_solid(&mut topo, cube, &Mat4::translation(3.5, 3.5, 3.5))
            .unwrap();
        let hollow =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, block, cube)
                .unwrap();
        let slab = crate::primitives::make_box(&mut topo, 10.0, 10.0, 3.0).unwrap();
        let refused = keep_cavities(&mut topo, hollow, slab, "test").unwrap_err();
        assert!(refused.to_string().contains("cavity"), "{refused}");
    }

    /// A cut a hair off a plane far from the origin matches the plane's
    /// surface within its relative margin; its vertices 1e-5 below it make
    /// it a face of its own.
    #[test]
    fn a_slightly_tilted_cut_far_from_the_origin_is_no_source_face() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        crate::transform::transform_solid(&mut topo, block, &Mat4::translation(0.0, 0.0, 99_990.0))
            .unwrap();
        let top_face = topo
            .shell(topo.solid(block).unwrap().outer_shell())
            .unwrap()
            .faces()
            .iter()
            .copied()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(),
                    FaceSurface::Plane { normal, .. } if normal.z() > 0.5)
            })
            .unwrap();

        let top = 100_000.0;
        let corners = [
            Point3::new(0.0, 0.0, top - 1e-5),
            Point3::new(10.0, 0.0, top - 1e-5),
            Point3::new(10.0, 1.0, top),
            Point3::new(0.0, 1.0, top),
        ];
        let wire = brepkit_topology::builder::make_polygon_wire(&mut topo, &corners, 1e-7).unwrap();
        let normal = (corners[1] - corners[0])
            .cross(corners[3] - corners[0])
            .normalize()
            .unwrap();
        let d = normal.dot(corners[0] - Point3::new(0.0, 0.0, 0.0));
        let bevel = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane { normal, d },
        ));

        assert!(same_surface(
            topo.face(top_face).unwrap().surface(),
            topo.face(bevel).unwrap().surface()
        ));
        assert!(!on_source_face(&topo, bevel, &[top_face]).unwrap());
    }

    /// Over a source square with a hole, a face can cross the hole with
    /// every vertex on the face and none of the hole's inside it: a face
    /// with holes proves nothing, and the rectangle reads as made. Over a
    /// plain square it is the source's.
    #[test]
    fn a_face_over_a_holed_source_face_is_no_source_face() {
        let mut topo = Topology::new();
        let rectangle = |topo: &mut Topology, (x0, y0): (f64, f64), (x1, y1): (f64, f64)| {
            let corners: Vec<Point3> = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                .iter()
                .map(|&(x, y)| Point3::new(x, y, 0.0))
                .collect();
            brepkit_topology::builder::make_polygon_wire(topo, &corners, 1e-7).unwrap()
        };
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let outer = rectangle(&mut topo, (-5.0, -5.0), (5.0, 5.0));
        let hole = rectangle(&mut topo, (-1.0, -3.0), (1.0, 3.0));
        let holed = topo.add_face(Face::new(outer, vec![hole], plane.clone()));
        let plain_outer = rectangle(&mut topo, (-5.0, -5.0), (5.0, 5.0));
        let plain = topo.add_face(Face::new(plain_outer, Vec::new(), plane.clone()));
        let strip = rectangle(&mut topo, (-4.0, -1.0), (4.0, 1.0));
        let crossing = topo.add_face(Face::new(strip, Vec::new(), plane));
        assert!(!on_source_face(&topo, crossing, &[holed]).unwrap());
        assert!(on_source_face(&topo, crossing, &[plain]).unwrap());
    }
}
