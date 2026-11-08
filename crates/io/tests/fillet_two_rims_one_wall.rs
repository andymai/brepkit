//! A puck (radius 25, height 20) with a pocket sunk 5 into its top, the
//! pocket's outline one closed B-spline edge, both rims rounded at once.
//!
//! The top cap's hole has a B-spline outline, so that rim's round takes the
//! normal trim path, while the bottom rim's disc cap took the closed-rim
//! rebuild, which rebuilds the wall with its own contact circle. The normal
//! trim then started from the wall as it came: it dropped the rebuild's
//! contact circle and kept the bottom rim, and the incidence audit refused
//! the fillet. A rim whose wall a normal-path stripe also trims now takes the
//! normal path too, which trims both contacts on the one wall.
//!
//! Read back through STEP, as a file gives the part: a file starts each rim
//! circle at its seam vertex.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_geometry::convert::ellipse_to_nurbs;
use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::curves::Ellipse3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const RADIUS: f64 = 25.0;
const HEIGHT: f64 = 20.0;

/// The pocketed puck, written to STEP and read back.
fn pocketed_puck(topo: &mut Topology) -> SolidId {
    let mut built = Topology::new();
    let puck = make_cylinder(&mut built, RADIUS, HEIGHT).unwrap();
    let outline = Ellipse3D::new(
        Point3::new(0.0, 0.0, 15.0),
        Vec3::new(0.0, 0.0, 1.0),
        8.0,
        5.0,
    )
    .unwrap();
    // One control point pushed out, so the outline is no conic.
    let ellipse = ellipse_to_nurbs(&outline, 0.0, 2.0 * PI).unwrap();
    let mut poles = ellipse.control_points().to_vec();
    poles[2] = poles[2] + Vec3::new(1.5, 1.5, 0.0);
    let spline = NurbsCurve::new(
        ellipse.degree(),
        ellipse.knots().to_vec(),
        poles,
        ellipse.weights().to_vec(),
    )
    .unwrap();
    let start = spline.evaluate(spline.domain().0);
    let v = built.add_vertex(Vertex::new(start, 1e-7));
    let edge = built.add_edge(Edge::new(v, v, EdgeCurve::NurbsCurve(spline)));
    let wire = built.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
    let face = built.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 15.0,
        },
    ));
    let tool = extrude(&mut built, face, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();
    let pocketed = boolean(&mut built, BooleanOp::Cut, puck, tool).unwrap();
    let step = write_step(&built, &[pocketed]).unwrap();
    read_step(&step, topo).unwrap()[0]
}

/// The puck's rim circle at height `z`.
fn rim_at(topo: &Topology, solid: SolidId, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c) if (c.radius() - RADIUS).abs() < 1e-6)
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-6
        })
        .expect("a rim circle at that height")
}

/// Whether the top cap (the plane holding the top rim) has a hole with a
/// B-spline edge, which is what sends its round down the normal path.
fn top_cap_hole_is_a_spline(topo: &Topology, solid: SolidId, top: EdgeId) -> bool {
    let spline = |w| {
        topo.wire(w).unwrap().edges().iter().any(|oe| {
            matches!(
                topo.edge(oe.edge()).unwrap().curve(),
                EdgeCurve::NurbsCurve(_)
            )
        })
    };
    solid_faces(topo, solid).unwrap().into_iter().any(|f| {
        let face = topo.face(f).unwrap();
        face.surface().is_planar()
            && topo
                .wire(face.outer_wire())
                .unwrap()
                .edges()
                .iter()
                .any(|oe| oe.edge() == top)
            && face.inner_wires().iter().any(|&w| spline(w))
    })
}

/// The ring a round of radius `r` removes from a convex rim of radius `rim`:
/// the square of side `r` at the rim less a quarter disc, swept round by its
/// centroid, `r (10 - 3 pi) / (3 (4 - pi))` in from the rim.
fn rounded_off(rim: f64, r: f64) -> f64 {
    let centroid = r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI));
    2.0 * PI * (rim - centroid) * (1.0 - PI / 4.0) * r * r
}

#[test]
fn a_pocketed_pucks_two_rims_round_off_together() {
    for r in [3.0, 7.5] {
        let mut topo = Topology::new();
        let puck = pocketed_puck(&mut topo);
        let top = rim_at(&topo, puck, HEIGHT);
        let bottom = rim_at(&topo, puck, 0.0);
        assert!(
            top_cap_hole_is_a_spline(&topo, puck, top),
            "the top cap's hole has a B-spline edge"
        );
        let before = solid_volume(&topo, puck, 0.001).unwrap();
        let result = fillet_v2(&mut topo, puck, &[top, bottom], r).unwrap();
        assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
        let changed = solid_volume(&topo, result.solid, 0.001).unwrap() - before;
        let truth = -2.0 * rounded_off(RADIUS, r);
        assert!(
            (changed - truth).abs() < 1e-6 * before,
            "r = {r}: changed by {changed}, truth {truth}"
        );
    }
}
