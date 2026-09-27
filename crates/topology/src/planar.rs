//! A planar face's boundary flattened into its plane.

use brepkit_math::region2d::Boundary2;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};

use crate::Topology;
use crate::TopologyError;
use crate::edge::EdgeCurve;
use crate::face::FaceId;

/// A planar face's boundary flattened into its plane, holes included.
///
/// The plane runs through `origin`, spanned by the orthonormal `x` and `y`.
/// A line becomes a segment, a circle or ellipse arc the arc it is. `None`
/// when an edge is a NURBS curve.
///
/// # Errors
///
/// Returns [`TopologyError`] if an entity lookup fails.
pub fn face_boundary_2d(
    topo: &Topology,
    face: FaceId,
    origin: Point3,
    x: Vec3,
    y: Vec3,
) -> Result<Option<Vec<Boundary2>>, TopologyError> {
    let face = topo.face(face)?;
    let flat = |p: Point3| {
        let d = p - origin;
        Point2::new(d.dot(x), d.dot(y))
    };
    let along = |v: Vec3| Vec2::new(v.dot(x), v.dot(y));
    let mut pieces = Vec::new();
    for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            pieces.push(match edge.curve() {
                EdgeCurve::Line => Boundary2::Segment(flat(a), flat(b)),
                EdgeCurve::Circle(c) => Boundary2::Arc {
                    center: flat(c.center()),
                    u: along(c.u_axis()),
                    v: along(c.v_axis()),
                    a: c.radius(),
                    b: c.radius(),
                    t0,
                    t1,
                },
                EdgeCurve::Ellipse(e) => Boundary2::Arc {
                    center: flat(e.center()),
                    u: along(e.u_axis()),
                    v: along(e.v_axis()),
                    a: e.semi_major(),
                    b: e.semi_minor(),
                    t0,
                    t1,
                },
                EdgeCurve::NurbsCurve(_) => return Ok(None),
            });
        }
    }
    Ok(Some(pieces))
}
