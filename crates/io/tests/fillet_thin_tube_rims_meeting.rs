//! A tube whose walls are B-spline surfaces 2 apart, both rims of its top
//! rounded at r = 1: a full round. The two walked contacts on the top lie
//! on each other all round, and the top is left a ring with no area, which
//! goes, the inner rim's round taking the outer's contact.
//!
//! Here the two contacts start at one point as two vertices (the walked
//! inner contact closes 5e-5 from where the outer one starts). The ring was
//! dropped only where the loops shared their start vertex or the inner one
//! started inside the outer's curve, so it stayed, a strip of no area
//! whose mesh closed the tube's top: the solid's own volume read 65% high
//! and its mesh was not closed. Loops starting within their match of each
//! other now start together, the inner round's seam re-pointed to the
//! outer contact's vertex.
//!
//! Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{TriangleMesh, is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/thin_tube_rims_meeting.step");
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

/// The edge of `solid` passing nearest `p`.
fn edge_through(topo: &Topology, solid: SolidId, p: Point3) -> EdgeId {
    let off = |e: EdgeId| {
        let edge = topo.edge(e).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        (0..=20_000)
            .map(|k| {
                let t = (t1 - t0).mul_add(f64::from(k) / 20_000.0, t0);
                (edge.curve().evaluate_with_endpoints(t, a, b) - p).length()
            })
            .fold(f64::MAX, f64::min)
    };
    let edges = solid_edges(topo, solid).unwrap();
    let best = edges
        .iter()
        .copied()
        .min_by(|&a, &b| off(a).total_cmp(&off(b)))
        .unwrap();
    assert!(off(best) < 0.05, "no edge through {p:?}: {}", off(best));
    best
}

fn mesh_volume(mesh: &TriangleMesh) -> f64 {
    let origin = mesh.positions[0];
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let p = |i: u32| mesh.positions[i as usize] - origin;
            p(t[0]).dot(p(t[1]).cross(p(t[2]))) / 6.0
        })
        .sum()
}

#[test]
fn rims_meeting_from_two_start_vertices_leave_no_ring() {
    let mut topo = Topology::new();
    let solid = load(&mut topo);
    let before = solid_volume(&topo, solid, 0.05).unwrap();
    let outer = edge_through(
        &topo,
        solid,
        Point3::new(
            -44.236_611_778_667_69,
            11.577_980_938_862_195,
            -28.614_907_363_513_3,
        ),
    );
    let inner = edge_through(
        &topo,
        solid,
        Point3::new(
            -46.328_959_136_463,
            11.577_980_938_862_195,
            -28.614_907_363_513_3,
        ),
    );
    assert_ne!(outer, inner);
    let result = fillet_v2(&mut topo, solid, &[outer, inner], 1.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    // The two walls, the bottom, and the two rounds: the top is gone.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 5);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    // Closed as a coarse mesh gives it. (Finer meshes leave one triangle
    // loose by the inner round's seam, which now ends at the outer
    // contact's vertex, 3e-5 across the round from its own rim.)
    for deflection in [0.5, 0.2] {
        let mesh = tessellate_solid(&topo, result.solid, deflection).unwrap();
        assert!(is_watertight(&mesh), "mesh open at {deflection}");
    }
    // With the ring kept, its mesh closed the tube's top and the solid's
    // volume read 65% high.
    let after = solid_volume(&topo, result.solid, 0.05).unwrap();
    let meshed = mesh_volume(&tessellate_solid(&topo, result.solid, 0.02).unwrap());
    assert!(
        (after - meshed).abs() <= 1e-3 * meshed,
        "the solid's volume {after} is not its mesh's {meshed}"
    );
    // A full round of a 2 wide rim takes off a little of the tube.
    assert!(
        after < before && after > 0.99 * before,
        "{before} -> {after}"
    );
}
