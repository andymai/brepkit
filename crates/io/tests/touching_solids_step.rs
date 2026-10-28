//! A fuse of two solids touching at a point keeps both pieces joined there,
//! and survives a trip through STEP with its volume and validity.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

#[test]
fn touching_solids_round_trip_through_step() {
    for (label, want) in [("boxes", 2.0), ("balls", 8.0 * PI / 3.0)] {
        let mut topo = Topology::default();
        let (a, b, at) = if label == "boxes" {
            let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            (a, b, [1.0, 1.0, 1.0])
        } else {
            let a = make_sphere(&mut topo, 1.0, 16).unwrap();
            let b = make_sphere(&mut topo, 1.0, 16).unwrap();
            (a, b, [2.0, 0.0, 0.0])
        };
        transform_solid(&mut topo, b, &Mat4::translation(at[0], at[1], at[2])).unwrap();
        let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let faces = solid_faces(&topo, fused).unwrap().len();

        let text = write_step(&topo, &[fused]).unwrap();
        let mut back = Topology::default();
        let solids = read_step(&text, &mut back).unwrap();
        assert_eq!(solids.len(), 1, "{label}: solids read back");
        let solid = solids[0];
        assert_eq!(
            solid_faces(&back, solid).unwrap().len(),
            faces,
            "{label}: faces"
        );
        let report = validate_solid(&back, solid).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        let got = solid_volume(&back, solid, 0.001).unwrap();
        assert!(
            (got - want).abs() < 1e-9 * want,
            "{label}: volume {got} against {want}"
        );
    }
}
