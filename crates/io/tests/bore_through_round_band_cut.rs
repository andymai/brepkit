//! A plate rounded on both rims, its end round about a bore's axis, and the
//! bore cut back through the rounds: the cut a fillet over a bore's mouth
//! makes once the round is built on the plate with the bore filled.
//!
//! Where the two rounds cross at the plate's end, a small B-spline band
//! meets the top round's band along a crease, a quarter of a circle, which
//! the small band walks against the circle's sense. The face splitter read
//! such an arc on a B-spline face from its walk's start to its end, which
//! there is the other three quarters, so the bore's section, ending on the
//! crease, never split it. The small band kept its edges inside the bore,
//! the section was left with one face, and the cut fell back to a mesh. The
//! operands are a case's, written by brepkit's STEP writer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn a_bore_cut_back_through_crossing_rounds_is_exact() {
    let poses = [
        Mat4::identity(),
        Mat4::translation(3.0, -7.0, 2.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-1.1),
    ];
    for pose in poses {
        let mut parts = Vec::new();
        for op in [BooleanOp::Cut, BooleanOp::Intersect] {
            let mut topo = Topology::new();
            let plate = load(&mut topo, "bore_through_round_band_plate.step");
            let bore = load(&mut topo, "bore_through_round_band_bore.step");
            transform_solid(&mut topo, plate, &pose).unwrap();
            transform_solid(&mut topo, bore, &pose).unwrap();
            let whole = solid_volume(&topo, plate, 1e-3).unwrap();
            let opts = BooleanOptions {
                mesh_fallback: false,
                ..BooleanOptions::default()
            };
            let piece = boolean_with_options(&mut topo, op, plate, bore, opts).unwrap();
            assert!(
                validate_solid(&topo, piece).unwrap().is_valid(),
                "{op:?} invalid"
            );
            let faces = solid_faces(&topo, piece).unwrap();
            if op == BooleanOp::Cut {
                let count = |pick: fn(&FaceSurface) -> bool| {
                    faces
                        .iter()
                        .filter(|&&f| pick(topo.face(f).unwrap().surface()))
                        .count()
                };
                // The plate's two round ends and the bore's wall, the five
                // bands and the two flats.
                assert_eq!(count(|s| matches!(s, FaceSurface::Cylinder(_))), 3);
                assert_eq!(count(|s| matches!(s, FaceSurface::Nurbs(_))), 5);
                assert_eq!(count(|s| matches!(s, FaceSurface::Plane { .. })), 2);
            }
            parts.push((solid_volume(&topo, piece, 1e-3).unwrap(), whole));
        }
        // What the bore takes off and what is left make up the plate.
        let ((cut, whole), (inside, _)) = (parts[0], parts[1]);
        assert!(
            (cut + inside - whole).abs() <= 1e-6 * whole,
            "cut {cut} + inside {inside} != {whole}"
        );
    }
}
