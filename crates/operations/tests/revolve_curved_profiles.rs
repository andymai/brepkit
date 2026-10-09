//! Profiles with curved edges revolved about an axis: each band is the
//! surface the edge's own curve sweeps, so the solid holds its volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{PI, TAU};

use brepkit_geometry::convert::ellipse_to_nurbs;
use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// Valid, watertight, of the given volume, few faces, every NURBS face on
/// the swept surface (`gap` measures a point's distance off it), and the
/// points classified as expected.
fn assert_solid(
    topo: &Topology,
    solid: SolidId,
    volume: f64,
    gap: impl Fn(Point3) -> f64,
    points: &[(Point3, bool)],
    label: &str,
) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{label}"
    );
    let faces = solid_faces(topo, solid).unwrap();
    assert!(faces.len() <= 40, "{label}: {} faces", faces.len());
    for &f in &faces {
        let FaceSurface::Nurbs(surface) = topo.face(f).unwrap().surface() else {
            continue;
        };
        let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
        for i in 0..=20 {
            for j in 0..=20 {
                let p = surface.evaluate(
                    (u1 - u0).mul_add(f64::from(i) / 20.0, u0),
                    (v1 - v0).mul_add(f64::from(j) / 20.0, v0),
                );
                assert!(gap(p).abs() < 1e-9, "{label}: {p:?} off the surface");
            }
        }
    }
    let measured = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (measured - volume).abs() < 1e-4 * volume,
        "{label}: {measured} against {volume}"
    );
    // The signed volume of the oriented mesh: a face turned inward would
    // subtract its share.
    let oriented = oriented_solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (oriented - volume).abs() < 1e-3 * volume,
        "{label}: oriented {oriented} against {volume}"
    );
    for &(point, inside) in points {
        let expected = if inside {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        let got = classify_point(topo, solid, point, 0.01, 1e-7).unwrap();
        assert_eq!(got, expected, "{label}: {point:?}");
    }
}

/// The face bounded by one closed curve edge.
fn closed_face(topo: &mut Topology, curve: EdgeCurve, start: Point3) -> FaceId {
    let vertex = topo.add_vertex(Vertex::new(start, 1e-7));
    let edge = topo.add_edge(Edge::new(vertex, vertex, curve));
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
    make_face_from_wire(topo, wire).unwrap()
}

/// A half-ellipse from pole to pole (semi-axes 2 out from the axis, 1 along
/// it), closed along the axis, as an ellipse arc and as its NURBS: an oblate
/// spheroid of volume 4/3 pi a^2 b.
#[test]
fn a_half_ellipse_revolves_to_a_spheroid() {
    let ellipse = Ellipse3D::new_with_ref(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        2.0,
        1.0,
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    let (south, north) = (Point3::new(0.0, 0.0, -1.0), Point3::new(0.0, 0.0, 1.0));
    let (t_south, t_north) = (ellipse.project(south), ellipse.project(north));
    let t_north = if t_north > t_south {
        t_north
    } else {
        t_north + TAU
    };
    // The NURBS also as an edge stored north to south, its curve running
    // from the edge's end, used reversed.
    for (as_nurbs, stored_backward) in [(false, false), (true, false), (true, true)] {
        let mut topo = Topology::new();
        let vs = topo.add_vertex(Vertex::new(south, 1e-7));
        let vn = topo.add_vertex(Vertex::new(north, 1e-7));
        let curve = if as_nurbs {
            EdgeCurve::NurbsCurve(ellipse_to_nurbs(&ellipse, t_south, t_north).unwrap())
        } else {
            EdgeCurve::Ellipse(ellipse.clone())
        };
        let arc = if stored_backward {
            OrientedEdge::new(topo.add_edge(Edge::new(vn, vs, curve)), false)
        } else {
            OrientedEdge::new(topo.add_edge(Edge::new(vs, vn, curve)), true)
        };
        let axis = topo.add_edge(Edge::new(vn, vs, EdgeCurve::Line));
        let wire =
            topo.add_wire(Wire::new(vec![arc, OrientedEdge::new(axis, true)], true).unwrap());
        let face = make_face_from_wire(&mut topo, wire).unwrap();
        let solid = revolve(&mut topo, face, Point3::new(0.0, 0.0, 0.0), Z, TAU).unwrap();
        assert_solid(
            &topo,
            solid,
            4.0 / 3.0 * PI * 4.0,
            |p| (p.x().hypot(p.y()) / 2.0).hypot(p.z()) - 1.0,
            &[],
            &format!("spheroid, NURBS {as_nurbs}, stored backward {stored_backward}"),
        );
    }
}

/// An ellipse (semi-axes 1 out from the axis, 0.5 along it) centred 3 from
/// the axis sweeps a ring of volume 2 pi R * pi a b.
#[test]
fn an_ellipse_revolves_to_an_elliptic_ring() {
    let mut topo = Topology::new();
    let ellipse = Ellipse3D::new_with_ref(
        Point3::new(3.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        1.0,
        0.5,
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    let start = ellipse.evaluate(0.0);
    let face = closed_face(&mut topo, EdgeCurve::Ellipse(ellipse), start);
    let solid = revolve(&mut topo, face, Point3::new(0.0, 0.0, 0.0), Z, TAU).unwrap();
    assert_solid(
        &topo,
        solid,
        2.0 * PI * 3.0 * PI * 0.5,
        |p| (p.x().hypot(p.y()) - 3.0).hypot(p.z() / 0.5) - 1.0,
        &[],
        "ring",
    );
    // Its bands are rational surfaces of revolution: points round the ring
    // read inside its tube and outside it at every turn.
    for k in 0..24 {
        let a = TAU * f64::from(k) / 24.0;
        for (r, z, inside) in [
            (3.0, 0.3, true),
            (3.9, 0.0, true),
            (3.0, 0.6, false),
            (4.2, 0.0, false),
        ] {
            let p = Point3::new(r * a.cos(), r * a.sin(), z);
            let want = if inside {
                PointClassification::Inside
            } else {
                PointClassification::Outside
            };
            let got = classify_point(&topo, solid, p, 0.01, 1e-7).unwrap();
            assert_eq!(got, want, "ring at {p:?}");
        }
    }
}

/// A 3 x 4 rectangle 2 to 5 from the axis with a hole of radius 1 at 3.5
/// sweeps a ring with a toroidal tunnel.
#[test]
fn a_profile_with_a_circular_hole_revolves_to_a_tunnelled_ring() {
    let mut topo = Topology::new();
    let outer = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(2.0, 0.0, -2.0),
            Point3::new(5.0, 0.0, -2.0),
            Point3::new(5.0, 0.0, 2.0),
            Point3::new(2.0, 0.0, 2.0),
        ],
        1e-7,
    )
    .unwrap();
    let circle = Circle3D::new(Point3::new(3.5, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0).unwrap();
    let vertex = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
    let edge = topo.add_edge(Edge::new(vertex, vertex, EdgeCurve::Circle(circle)));
    let inner = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
    let face = topo.add_face(Face::new(
        outer,
        vec![inner],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, -1.0, 0.0),
            d: 0.0,
        },
    ));
    let solid = revolve(&mut topo, face, Point3::new(0.0, 0.0, 0.0), Z, TAU).unwrap();
    assert_solid(
        &topo,
        solid,
        2.0 * PI * 3.5 * (12.0 - PI),
        |p| (p.x().hypot(p.y()) - 3.5).hypot(p.z()) - 1.0,
        &[
            (Point3::new(0.0, 3.5, 0.0), false),
            (Point3::new(0.0, 3.5, 1.5), true),
            (Point3::new(0.0, 4.8, 0.0), true),
        ],
        "tunnelled ring",
    );
}
