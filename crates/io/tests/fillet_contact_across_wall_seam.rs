//! A plate 240 thick whose wall is one closed B-spline surface, the two
//! B-spline pieces of its bottom rim along that wall rounded at r = 50. One
//! round's contact on the wall runs across the wall's seam line.
//!
//! The trim kept that contact as one edge, from one side of the seam to the
//! other, which no run of the wall's outline between the seam's copies
//! could take, so the wall's trim failed and the fillet with it. The
//! contact is now cut where it crosses the seam: each piece lies on one
//! side of it, and the band takes both pieces as its side.
//!
//! Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology) -> SolidId {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/spline_plate_two_rounds.step");
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

#[test]
fn a_contact_across_the_wall_s_seam_is_cut_there() {
    let mut topo = Topology::new();
    let plate = load(&mut topo);
    let picks = [
        Point3::new(-415.159_424_679_209_7, 351.839_643_089_86, -120.0),
        Point3::new(240.998_040_990_944_65, 69.993_421_862_153_55, -120.0),
    ]
    .map(|p| edge_through(&topo, plate, p));
    let result = fillet_v2(&mut topo, plate, &picks, 50.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    // The part's six faces and the two bands.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 8);
    let mesh = tessellate_solid(&topo, result.solid, 0.5).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // As a reference kernel measures it.
    let volume = solid_volume(&topo, result.solid, 0.1).unwrap();
    let expected = 93_285_967.06;
    assert!(
        (volume - expected).abs() < 1e-5 * expected,
        "volume {volume}, expected {expected}"
    );
}
