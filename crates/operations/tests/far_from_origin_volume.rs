//! A solid's volume does not depend on where it sits. Divergence sums about
//! the world origin must not multiply far-off coordinates together (a
//! triangle's `a · (b × c)`, a polygon's shoelace), or a solid a million units
//! out cancels its own volume away.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const FAR: [f64; 3] = [1.0e6, 2.0e6, -3.0e6];

fn far() -> Mat4 {
    Mat4::translation(FAR[0], FAR[1], FAR[2])
}

/// A slanted triangular prism: every face planar, none axis-aligned.
fn prism(topo: &mut Topology) -> SolidId {
    let corners = [(0.0, 0.0), (4.0, 1.0), (1.0, 3.0)];
    let vs: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let edges = (0..3)
        .map(|k| {
            let e = topo.add_edge(Edge::new(vs[k], vs[(k + 1) % 3], EdgeCurve::Line));
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
    let solid =
        brepkit_operations::extrude::extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    transform_solid(topo, solid, &Mat4::rotation_x(0.4)).unwrap();
    solid
}

#[test]
fn planar_prism_far_away_keeps_its_volume() {
    // Triangle area 5.5 times height 2.
    let truth = 11.0;
    let mut topo = Topology::new();
    let solid = prism(&mut topo);
    transform_solid(&mut topo, solid, &far()).unwrap();
    let exact = solid_volume(&topo, solid, 0.001).unwrap();
    assert!((exact - truth).abs() < 1e-8 * truth, "solid_volume {exact}");
    let mesh = oriented_solid_volume(&topo, solid, 0.001).unwrap();
    assert!((mesh - truth).abs() < 1e-8 * truth, "mesh volume {mesh}");
}

/// A window through a tube wall: the holed wall's flux, the exact planar
/// caps and the meshed faces all sum about the world origin.
#[test]
fn windowed_tube_far_away_keeps_its_volume() {
    let (r, h) = (1.5, 4.0);
    let strip = |y0: f64, y1: f64| {
        let f = |y: f64| y * (r * r - y * y).sqrt() + r * r * (y / r).asin();
        f(y1) - f(y0)
    };
    let truth = std::f64::consts::PI * r * r * h - 0.5 * strip(0.3, 0.8);
    let mut topo = Topology::new();
    let tube = make_cylinder(&mut topo, r, h).unwrap();
    let cutter = make_box(&mut topo, 10.0, 0.5, 0.5).unwrap();
    transform_solid(&mut topo, cutter, &Mat4::translation(-5.0, 0.3, 1.0)).unwrap();
    let solid = boolean(&mut topo, BooleanOp::Cut, tube, cutter).unwrap();
    transform_solid(&mut topo, solid, &far()).unwrap();
    let exact = solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (exact - truth).abs() < 1e-7 * truth,
        "solid_volume {exact}, truth {truth}"
    );
    let mesh = oriented_solid_volume(&topo, solid, 0.0005).unwrap();
    assert!(
        (mesh - truth).abs() < 1e-3 * truth,
        "mesh volume {mesh}, truth {truth}"
    );
}

/// A napkin ring: every face goes through the bored-quadric integrator.
#[test]
fn bored_sphere_far_away_keeps_its_volume() {
    let mut topo = Topology::new();
    let sphere = make_sphere(&mut topo, 6.0, 24).unwrap();
    let bore = make_cylinder(&mut topo, 3.0, 20.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
    let solid = boolean(&mut topo, BooleanOp::Cut, sphere, bore).unwrap();
    let near = solid_volume(&topo, solid, 0.001).unwrap();
    transform_solid(&mut topo, solid, &far()).unwrap();
    let away = solid_volume(&topo, solid, 0.001).unwrap();
    assert!((away - near).abs() < 1e-6 * near, "near {near}, far {away}");
}
