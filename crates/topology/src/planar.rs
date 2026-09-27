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
    let wires: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .map(|w| topo.wire(w))
        .collect::<Result<_, _>>()?;
    // A NURBS edge declines before any curve is projected: finding its
    // span costs point projections the caller's fallback repeats.
    for wire in &wires {
        for oe in wire.edges() {
            if matches!(topo.edge(oe.edge())?.curve(), EdgeCurve::NurbsCurve(_)) {
                return Ok(None);
            }
        }
    }
    let mut pieces = Vec::new();
    for wire in &wires {
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let curve = edge.curve();
            pieces.push(match curve {
                EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => Boundary2::Segment(flat(a), flat(b)),
                EdgeCurve::Circle(c) => {
                    let (t0, t1) = curve.domain_with_endpoints(a, b);
                    Boundary2::Arc {
                        center: flat(c.center()),
                        u: along(c.u_axis()),
                        v: along(c.v_axis()),
                        a: c.radius(),
                        b: c.radius(),
                        t0,
                        t1,
                    }
                }
                EdgeCurve::Ellipse(e) => {
                    let (t0, t1) = curve.domain_with_endpoints(a, b);
                    Boundary2::Arc {
                        center: flat(e.center()),
                        u: along(e.u_axis()),
                        v: along(e.v_axis()),
                        a: e.semi_major(),
                        b: e.semi_minor(),
                        t0,
                        t1,
                    }
                }
            });
        }
    }
    Ok(Some(pieces))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::region2d::point_in_region;

    use super::*;
    use crate::edge::Edge;
    use crate::face::{Face, FaceSurface};
    use crate::vertex::Vertex;
    use crate::wire::{OrientedEdge, Wire};

    /// The upper half of the unit disc in `z = 0`, its diameter a line, or a
    /// straight NURBS when `nurbs`.
    fn half_disc(topo: &mut Topology, nurbs: bool) -> FaceId {
        let (a, b) = (Point3::new(1.0, 0.0, 0.0), Point3::new(-1.0, 0.0, 0.0));
        let (va, vb) = (
            topo.add_vertex(Vertex::new(a, 1e-7)),
            topo.add_vertex(Vertex::new(b, 1e-7)),
        );
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let arc = topo.add_edge(Edge::new(va, vb, EdgeCurve::Circle(circle)));
        let curve = if nurbs {
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![b, a], vec![1.0, 1.0]).unwrap(),
            )
        } else {
            EdgeCurve::Line
        };
        let diameter = topo.add_edge(Edge::new(vb, va, curve));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(arc, true),
                OrientedEdge::new(diameter, true),
            ],
            true,
        )
        .unwrap();
        let wire = topo.add_wire(wire);
        topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    /// A half disc flattens to its arc and its diameter: a point just inside
    /// the arc reads inside, one past the diameter outside.
    #[test]
    fn a_half_disc_flattens_to_its_arc_and_diameter() {
        let mut topo = Topology::new();
        let face = half_disc(&mut topo, false);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let pieces = face_boundary_2d(&topo, face, Point3::new(0.0, 0.0, 0.0), x, y)
            .unwrap()
            .unwrap();
        assert_eq!(pieces.len(), 2);
        let inside = |px: f64, py: f64| point_in_region(&pieces, Point2::new(px, py), 1e-9);
        assert_eq!(inside(0.0, 0.999), Some(true));
        assert_eq!(inside(0.7, 0.7), Some(true));
        assert_eq!(inside(0.0, -0.01), Some(false));
    }

    /// A NURBS edge declines the flattening.
    #[test]
    fn a_nurbs_edge_declines() {
        let mut topo = Topology::new();
        let face = half_disc(&mut topo, true);
        let (x, y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(
            face_boundary_2d(&topo, face, Point3::new(0.0, 0.0, 0.0), x, y)
                .unwrap()
                .is_none()
        );
    }
}
