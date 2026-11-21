//! A plate 100 thick whose outline is one closed B-spline, its wall drafted
//! (the seam line leans 23 across the plate's thickness), its top rim
//! rounded at r = 100. Near one tip of the outline the ball cannot roll on:
//! the wall's contact runs away along the wall, 6, then 30, then 68 times as
//! fast as the rim, and one step of the walk then lands on another
//! solution, 520 along the wall. The band built through those sections
//! stood up to 36 off the wall, so the result was not the solid it claimed
//! to be, and meshing it ran for minutes. The walk is refused there, so the
//! fillet fails (as it does in a reference kernel), and does so at once.
//! Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{edge_to_face_map, solid_edges};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/drafted_spline_plate.step");
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

/// The plate's closed B-spline rim on its top face, `y = 1100`.
fn top_rim(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
                && edge.start() == edge.end()
                && (topo.vertex(edge.start()).unwrap().point().y() - 1100.0).abs() < 1e-6
        })
        .expect("the plate's top rim")
}

/// How far the farthest of 40 samples along any edge of `solid` lies off a
/// face that uses it.
fn worst_edge_off_its_faces(topo: &Topology, solid: SolidId) -> f64 {
    let faces_of = edge_to_face_map(topo, solid).unwrap();
    let mut worst: f64 = 0.0;
    for edge_id in solid_edges(topo, solid).unwrap() {
        let edge = topo.edge(edge_id).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        for face_id in &faces_of[&edge_id.index()] {
            let surface = topo.face(*face_id).unwrap().surface();
            for k in 1..40 {
                let p = edge.curve().evaluate_with_endpoints(
                    (t1 - t0).mul_add(f64::from(k) / 40.0, t0),
                    a,
                    b,
                );
                let off = if let FaceSurface::Plane { normal, d } = surface {
                    (normal.dot(p - brepkit_math::vec::Point3::new(0.0, 0.0, 0.0)) - d).abs()
                } else {
                    let (u, v) = surface.project_point(p).unwrap();
                    (surface.evaluate(u, v).unwrap() - p).length()
                };
                worst = worst.max(off);
            }
        }
    }
    worst
}

#[test]
fn a_ball_wedged_at_a_drafted_walls_tip_is_refused() {
    let mut topo = Topology::new();
    let plate = load(&mut topo);
    let rim = top_rim(&topo, plate);
    // Refused outright, or the round left out; never a band off its wall.
    if let Ok(result) = fillet_v2(&mut topo, plate, &[rim], 100.0)
        && result.failed.is_empty()
    {
        let off = worst_edge_off_its_faces(&topo, result.solid);
        assert!(
            off < 1e-3,
            "the rounded plate has an edge {off} off a face that uses it"
        );
    }
}
