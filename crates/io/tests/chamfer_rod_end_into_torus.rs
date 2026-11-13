//! A rod's end chamfered, the rod's wall running on at its other end into a
//! torus along two arcs (not one closed circle). The rim chamfer took only
//! a wall between two whole circles, so the chamfer fell to the flat-bevel
//! rebuild, which refuses curved edges. The far arcs lie at one height on
//! the wall, which the chamfer leaves alone: it now takes them, and the
//! result is valid, watertight, and takes off the revolved triangle,
//! pi d^2 (R - d/3). Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;
use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/rod_end_into_torus.step");
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

/// The rod's end rim: the closed circle at `y = -55`, and its radius.
fn end_rim(topo: &Topology, solid: SolidId) -> (EdgeId, f64) {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find_map(|e| {
            let edge = topo.edge(e).unwrap();
            match edge.curve() {
                EdgeCurve::Circle(c)
                    if edge.start() == edge.end() && (c.center().y() + 55.0).abs() < 1e-6 =>
                {
                    Some((e, c.radius()))
                }
                _ => None,
            }
        })
        .expect("the rod's end rim")
}

#[test]
fn a_rod_end_running_into_a_torus_takes_its_chamfer() {
    for d in [3.0, 9.0, 12.5] {
        let mut topo = Topology::new();
        let rod = load(&mut topo);
        let before = solid_volume(&topo, rod, 0.01).unwrap();
        let (rim, radius) = end_rim(&topo, rod);
        let result = chamfer(&mut topo, rod, &[rim], d).unwrap();
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "d {d}: {:?}", report.issues);
        // The end disc, the cone, the wall, the torus and its far disc.
        assert_eq!(solid_faces(&topo, result).unwrap().len(), 5);
        let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
        assert!(is_watertight(&mesh), "d {d}: mesh not watertight");
        let truth = (PI * d * d).mul_add(-(radius - d / 3.0), before);
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "d {d}: volume {volume}, truth {truth}"
        );
    }
}
