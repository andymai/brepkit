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
    let origin = Point3::new(0.0, 0.0, 0.0);
    let near = |p: Point3, q: Point3| (p - q).length() <= EPS * (1.0 + (p - origin).length());
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

/// `result`, rebuilt from `source`'s outer shell alone, given copies of
/// `source`'s cavities.
///
/// The material taken off is bounded by parts of `source`'s outer faces,
/// which a cavity, lying strictly inside, never reaches, and by the faces
/// the operation made, those of `result` on no surface of `source`. A
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
    let surfaces: Vec<FaceSurface> = topo
        .shell(topo.solid(source)?.outer_shell())?
        .faces()
        .iter()
        .map(|&f| topo.face(f).map(|face| face.surface().clone()))
        .collect::<Result<_, _>>()?;
    let mut made = Vec::new();
    for &face in topo.shell(topo.solid(result)?.outer_shell())?.faces() {
        let surface = topo.face(face)?.surface();
        if !surfaces.iter().any(|s| same_surface(s, surface)) {
            made.push(face_bounds(topo, face)?);
        }
    }
    let refused = || OperationsError::InvalidInput {
        reason: format!("the {what} would cut into a cavity of the solid"),
    };
    for &cavity in &cavities {
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
    // The result gets its own copy: a shell shared with the source would
    // move with it under an in-place transform of either.
    let copied = crate::copy::copy_shells(topo, &cavities)?;
    let solid = topo.solid_mut(result)?;
    for cavity in copied {
        solid.add_inner_shell(cavity);
    }
    Ok(result)
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
}
