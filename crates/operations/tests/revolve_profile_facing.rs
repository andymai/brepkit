//! A circle revolved part of a turn about an axis it clears makes a trimmed
//! torus band with two disc caps, valid whichever way the profile faces.

#![allow(clippy::unwrap_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::measure::{solid_surface_area, solid_volume};
use brepkit_operations::revolve::revolve;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

#[test]
fn a_circle_revolved_part_way_is_valid_whichever_way_it_faces() {
    let (big, small) = (5.0, 1.5);
    for facing in [1.0, -1.0] {
        for (turn, sweep) in [(FRAC_PI_2, "a quarter"), (PI, "half"), (4.0, "4 radians")] {
            let label = format!("{sweep} turn, profile facing {facing} y");
            let mut topo = Topology::default();
            let normal = Vec3::new(0.0, facing, 0.0);
            let circle = Circle3D::new(Point3::new(big, 0.0, 0.0), normal, small).unwrap();
            let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
            let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
            let plane = FaceSurface::Plane { normal, d: 0.0 };
            let profile = topo.add_face(Face::new(wire, vec![], plane));
            let axis = Vec3::new(0.0, 0.0, 1.0);
            let solid =
                revolve(&mut topo, profile, Point3::new(0.0, 0.0, 0.0), axis, turn).unwrap();

            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let volume = solid_volume(&topo, solid, 0.001).unwrap();
            let want = PI * small * small * big * turn;
            assert!(
                (volume - want).abs() < 1e-9 * want,
                "{label}: volume {volume} against {want}"
            );
            let area = solid_surface_area(&topo, solid, 0.001).unwrap();
            let want = 2.0 * PI * small * big * turn + 2.0 * PI * small * small;
            assert!(
                (area - want).abs() < 1e-9 * want,
                "{label}: area {area} against {want}"
            );
        }
    }
}
