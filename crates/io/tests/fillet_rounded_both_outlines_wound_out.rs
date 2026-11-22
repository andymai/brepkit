//! A leg plate rounded round both its outlines, top and bottom (r = 7 on a
//! 40 mm plate whose outline runs round a 11.4 mm end arc and B-spline
//! sides): every face of the input is rebuilt, so none is left to read the
//! solid's boundary-walk convention from, and the rebuilt faces' own
//! readings decided it. A dozen bands read backwards against seven trimmed
//! input faces read right, the vote went backwards, and every face was
//! turned to match: each loop ran clockwise about a normal facing out,
//! which a file reader takes as a hole on a closed surface (the end arc's
//! spindle torus bands read back as the whole torus less the face, the
//! part 1.8% light). The convention is the one the readings are taken in,
//! a loop counterclockwise about its face's effective normal.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};

/// Twice the signed area of a plane face's outer loop about its effective
/// normal, from points along its edges in the order the loop runs them.
fn plane_winding(topo: &Topology, face: &Face) -> f64 {
    let FaceSurface::Plane { normal, .. } = *face.surface() else {
        unreachable!()
    };
    let normal = if face.is_reversed() { -normal } else { normal };
    let mut points = Vec::new();
    for oe in topo.wire(face.outer_wire()).unwrap().edges() {
        let e = topo.edge(oe.edge()).unwrap();
        let (a, b) = (
            topo.vertex(e.start()).unwrap().point(),
            topo.vertex(e.end()).unwrap().point(),
        );
        let (t0, t1) = e.curve().domain_with_endpoints(a, b);
        for k in 0..16 {
            let f = f64::from(k) / 16.0;
            let f = if oe.is_forward() { f } else { 1.0 - f };
            points.push(e.curve().evaluate_with_endpoints(t0 + (t1 - t0) * f, a, b));
        }
    }
    let o = points[0];
    (1..points.len())
        .map(|k| {
            (points[k] - o)
                .cross(points[(k + 1) % points.len()] - o)
                .dot(normal)
        })
        .sum()
}

#[test]
fn both_outlines_rounded_leave_every_loop_wound_about_its_normal() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/leg_rounded_both_outlines.step");
    let mut topo = Topology::new();
    let solid = read_step(&std::fs::read_to_string(path).unwrap(), &mut topo).unwrap()[0];
    let edges: Vec<_> = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let z = |v| topo.vertex(v).unwrap().point().z();
            let (a, b) = (z(edge.start()), z(edge.end()));
            (a - b).abs() < 1e-6 && (a.abs() - 20.0).abs() < 1e-6
        })
        .collect();
    assert_eq!(edges.len(), 10);
    let result = fillet_v2(&mut topo, solid, &edges, 7.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    // Another kernel's measure of its result of the same round.
    let volume = solid_volume(&topo, result.solid, 1e-3).unwrap();
    assert!(
        (volume - 486_549.72).abs() < 1e-4 * 486_549.72,
        "volume {volume}"
    );
    let mut caps = 0;
    for f in solid_faces(&topo, result.solid).unwrap() {
        let face = topo.face(f).unwrap();
        if matches!(face.surface(), FaceSurface::Plane { .. }) {
            caps += 1;
            let winding = plane_winding(&topo, face);
            assert!(
                winding > 0.0,
                "cap {f:?} runs clockwise about its normal ({winding})"
            );
        }
    }
    assert_eq!(caps, 2);
}
