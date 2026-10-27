//! A solid's cavities kept through an operation that rebuilds only its
//! outer shell.

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Two corners of the box of the whole conic `center + a cos t u + b sin t
/// v`, which reaches `hypot(a u_i, b v_i)` either side along each axis.
fn conic_box(center: Point3, (a, u): (f64, Vec3), (b, v): (f64, Vec3)) -> Vec<Point3> {
    let d = Vec3::new(
        (a * u.x()).hypot(b * v.x()),
        (a * u.y()).hypot(b * v.y()),
        (a * u.z()).hypot(b * v.z()),
    );
    vec![center - d, center + d]
}

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

/// A box holding everything within `reach` of `edge`: the edge's own box (a
/// line's ends, the box of the whole circle or ellipse it runs along, a
/// NURBS curve's control points) grown by `reach`.
pub fn edge_reach(topo: &Topology, edge: EdgeId, reach: f64) -> Result<Aabb3, OperationsError> {
    let data = topo.edge(edge)?;
    let points = match data.curve() {
        EdgeCurve::Line => vec![
            topo.vertex(data.start())?.point(),
            topo.vertex(data.end())?.point(),
        ],
        EdgeCurve::Circle(c) => conic_box(
            c.center(),
            (c.radius(), c.u_axis()),
            (c.radius(), c.v_axis()),
        ),
        EdgeCurve::Ellipse(e) => conic_box(
            e.center(),
            (e.semi_major(), e.u_axis()),
            (e.semi_minor(), e.v_axis()),
        ),
        EdgeCurve::NurbsCurve(n) => {
            hull_or_everything(n.control_points().iter(), n.weights().iter().copied())
        }
    };
    Ok(Aabb3::from_points(points).expanded(reach))
}

/// A box holding every face of `shell`: the measured box of its faces,
/// widened to a NURBS face's control points (the measured box samples it at
/// a few points only) and to a sphere's or torus's whole surface (it takes
/// that only for a whole one, and a trimmed band can bulge past its
/// boundary).
fn shell_bounds(topo: &Topology, shell: ShellId) -> Result<Aabb3, OperationsError> {
    let faces = topo.shell(shell)?.faces().to_vec();
    let measured = crate::measure::face_set_bounding_box(topo, &faces)?;
    let mut points = vec![measured.min, measured.max];
    for &face in &faces {
        match topo.face(face)?.surface() {
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
    }
    Ok(Aabb3::from_points(points))
}

/// `result`, rebuilt from `source`'s outer shell alone, given copies of
/// `source`'s cavities. A cavity whose box meets one of `removed` (boxes
/// holding the material the operation took off) could be cut into, and
/// refuses the operation, named `what` in the error.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if a cavity comes within
/// `removed`, or a topology error if a lookup fails.
pub fn keep_cavities(
    topo: &mut Topology,
    source: SolidId,
    result: SolidId,
    removed: &[Aabb3],
    what: &str,
) -> Result<SolidId, OperationsError> {
    let cavities = topo.solid(source)?.inner_shells().to_vec();
    if cavities.is_empty() {
        return Ok(result);
    }
    for &cavity in &cavities {
        let bounds = shell_bounds(topo, cavity)?;
        if removed.iter().any(|r| r.intersects(bounds)) {
            return Err(OperationsError::InvalidInput {
                reason: format!("the {what} would cut into a cavity of the solid"),
            });
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
        keep_cavities(&mut topo, hollow, result, &[], "test").unwrap();
        assert_eq!(topo.num_faces() - faces, 6);
        assert_eq!(topo.solid(result).unwrap().inner_shells().len(), 1);
    }
}
