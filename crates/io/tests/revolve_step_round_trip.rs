//! Revolved solids hold their volume, in brepkit and written to STEP and read
//! back. A revolve checked by Pappus' theorem: a half ellipse's
//! spheroid measured 1.2e-6 short (its NURBS bands read on their meshes), and
//! a slot's end bitten by a circle about a point a hair off the tilted axis
//! swept that bite as a NURBS band (no sphere in its STEP) and measured
//! 5.5e-6 over.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use brepkit_geometry::convert::ellipse_to_nurbs;
use brepkit_io::step::{read_step, write_step};
use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::revolve::revolve;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_face_from_wire;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The solid measures `exact`, and so does its STEP read back.
fn assert_volume_round_trips(topo: &Topology, solid: SolidId, exact: f64, tol: f64, label: &str) {
    let measured = solid_volume(topo, solid, 0.05).unwrap();
    assert!(
        (measured - exact).abs() < tol * exact,
        "{label}: {measured} against {exact}"
    );
    let step = write_step(topo, &[solid]).unwrap();
    let mut back = Topology::new();
    let ids = read_step(&step, &mut back).unwrap();
    assert_eq!(ids.len(), 1, "{label}");
    let read = solid_volume(&back, ids[0], 0.05).unwrap();
    assert!(
        (read - exact).abs() < tol * exact,
        "{label}: read back {read} against {exact}"
    );
}

fn face_with(topo: &mut Topology, edges: Vec<OrientedEdge>) -> FaceId {
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    make_face_from_wire(topo, wire).unwrap()
}

/// A half ellipse 30 out from the axis and 40 along it, closed by the axis,
/// as an ellipse arc and as its rational NURBS, a full turn and a quarter.
#[test]
fn a_revolved_half_ellipse_holds_its_volume() {
    let ellipse = Ellipse3D::new_with_ref(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        40.0,
        30.0,
        Vec3::new(0.0, 0.0, 1.0),
    )
    .unwrap();
    let (south, north) = (Point3::new(0.0, 0.0, -40.0), Point3::new(0.0, 0.0, 40.0));
    let t_north = ellipse.project(north);
    let t_south = ellipse.project(south);
    let t_south = if t_south > t_north {
        t_south
    } else {
        t_south + TAU
    };
    for (as_nurbs, angle) in [(false, TAU), (true, TAU), (true, FRAC_PI_2)] {
        let mut topo = Topology::new();
        let vs = topo.add_vertex(Vertex::new(south, 1e-7));
        let vn = topo.add_vertex(Vertex::new(north, 1e-7));
        let curve = if as_nurbs {
            EdgeCurve::NurbsCurve(ellipse_to_nurbs(&ellipse, t_north, t_south).unwrap())
        } else {
            EdgeCurve::Ellipse(ellipse.clone())
        };
        let arc = OrientedEdge::new(topo.add_edge(Edge::new(vn, vs, curve)), true);
        let axis = OrientedEdge::new(topo.add_edge(Edge::new(vs, vn, EdgeCurve::Line)), true);
        let face = face_with(&mut topo, vec![arc, axis]);
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            angle,
        )
        .unwrap();
        let exact = 4.0 / 3.0 * PI * 30.0 * 30.0 * 40.0 * angle / TAU;
        assert_volume_round_trips(
            &topo,
            solid,
            exact,
            1e-9,
            &format!("half ellipse, NURBS {as_nurbs}, angle {angle}"),
        );
    }
}

/// A half slot along a tilted axis (two quarter discs of radius 18.5 and the
/// rectangle between), its far end bitten by a half disc of radius 16.5
/// whose centre lies 9.6e-7 off the axis. The bite sweeps a sphere, which
/// the STEP carries.
#[test]
fn a_revolved_slot_with_a_bitten_end_sweeps_spheres() {
    let o = Point3::new(26.0, 0.0, 488.0);
    let d = Vec3::new(0.911_895_205_361_810_8, 0.0, -0.410_423_116_354_501_9)
        .normalize()
        .unwrap();
    let n = Vec3::new(0.0, -1.0, 0.0);
    let side = d.cross(n).normalize().unwrap();
    let (big, small, len) = (18.516_656_037_207_447, 16.5, 23.538_099_582_662_824);
    let at = |along: f64, out: f64| o + d * along + side * out;
    let bite_centre = at(len, 0.0) + side * 9.6e-7;
    let circle = |c: Point3, r: f64, from: Point3, normal: Vec3| {
        EdgeCurve::Circle(Circle3D::new_with_ref(c, normal, r, from - c).unwrap())
    };
    let r_quarter = 4.0 * big / (3.0 * PI);
    let r_half = 4.0 * small / (3.0 * PI);
    let moment = PI * big * big / 2.0 * r_quarter + big * len * big / 2.0
        - PI * small * small / 2.0 * r_half;
    for angle in [TAU, FRAC_PI_2] {
        let mut topo = Topology::new();
        let pts = [
            at(-big, 0.0),
            at(0.0, big),
            at(len, big),
            at(len + big, 0.0),
            at(len + small, 0.0),
            at(len - small, 0.0),
        ];
        let vs: Vec<_> = pts
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        let mut edge = |a: usize, b: usize, curve: EdgeCurve| {
            OrientedEdge::new(topo.add_edge(Edge::new(vs[a], vs[b], curve)), true)
        };
        let edges = vec![
            edge(0, 1, circle(at(0.0, 0.0), big, pts[0], n)),
            edge(1, 2, EdgeCurve::Line),
            edge(2, 3, circle(at(len, 0.0), big, pts[2], n)),
            edge(3, 4, EdgeCurve::Line),
            edge(4, 5, circle(bite_centre, small, pts[4], -n)),
            edge(5, 0, EdgeCurve::Line),
        ];
        let face = face_with(&mut topo, edges);
        let solid = revolve(&mut topo, face, o, d, angle).unwrap();
        let label = format!("bitten slot, angle {angle}");
        // (the bite's centre off the axis moves the volume by about 1e-9)
        assert_volume_round_trips(&topo, solid, angle * moment, 1e-8, &label);
        let surfaces: Vec<FaceSurface> = solid_faces(&topo, solid)
            .unwrap()
            .into_iter()
            .map(|f| topo.face(f).unwrap().surface().clone())
            .collect();
        assert!(
            !surfaces.iter().any(|s| matches!(s, FaceSurface::Nurbs(_))),
            "{label}: a NURBS band"
        );
        let mut radii: Vec<f64> = surfaces
            .iter()
            .filter_map(|s| match s {
                FaceSurface::Sphere(s) => Some(s.radius()),
                _ => None,
            })
            .collect();
        radii.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert!(
            radii.iter().any(|r| (r - small).abs() < 1e-9),
            "{label}: no sphere of the bite in {radii:?}"
        );
        let step = write_step(&topo, &[solid]).unwrap();
        assert!(
            step.contains("SPHERICAL_SURFACE") && step.contains("1.650000000000000E1"),
            "{label}: the STEP has no sphere of the bite"
        );
        assert!(!step.contains("B_SPLINE_SURFACE"), "{label}");
    }
}
