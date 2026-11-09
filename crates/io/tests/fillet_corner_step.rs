//! A set-back corner's patches and links through a STEP file and back.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::{CornerKind, FilletCorner, fillet_v2_with_corners};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};

#[test]
fn a_set_back_corner_comes_back_from_step() {
    let mut topo = Topology::new();
    let body = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    let at = Point3::new(0.0, 0.0, 0.0);
    let corner = solid_vertices(&topo, body)
        .unwrap()
        .into_iter()
        .find(|&v| (topo.vertex(v).unwrap().point() - at).length() < 1e-6)
        .unwrap();
    let edges: Vec<_> = solid_edges(&topo, body)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            edge.start() == corner || edge.end() == corner
        })
        .collect();
    let corners = [FilletCorner {
        vertex: corner,
        setbacks: edges.iter().copied().zip([3.0, 4.0, 5.0]).collect(),
        kind: CornerKind::G2,
    }];
    let solid = fillet_v2_with_corners(&mut topo, body, &edges, 2.0, &corners).unwrap();
    let before = solid_volume(&topo, solid, 0.01).unwrap();

    let text = write_step(&topo, &[solid]).unwrap();
    let mut read = Topology::new();
    let solids = read_step(&text, &mut read).unwrap();
    assert_eq!(solids.len(), 1);
    let back = solids[0];
    assert_eq!(solid_faces(&read, back).unwrap().len(), 11);
    let report = validate_solid(&read, back).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&read, back, 0.02).unwrap();
    assert!(is_watertight(&mesh), "the mesh leaks");
    let after = solid_volume(&read, back, 0.01).unwrap();
    assert!((after - before).abs() < 1e-4, "{before} became {after}");
}
