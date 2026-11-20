//! A plate whose round hole's wall is two faces, a 324 and a 36 degree arc,
//! and a sector-shaped pin standing in the hole against the wall, fused.
//!
//! The ray classifier reads a partial arc of a cylinder or cone by the angle
//! its boundary leaves uncovered: the largest gap between its points' `u`.
//! It read the points of the boundary polygon, three inside each open arc,
//! which along the 324 degree arc stand 81 degrees apart: wider than its real
//! gap, so a stretch of the wall read as open and another as doubled. A
//! point in the hole then read inside the plate, the pin's wall facing the
//! hole was dropped, and the fuse fell back to a mesh. The operands are a
//! case's, written by brepkit's STEP writer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
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
fn a_pin_against_a_split_hole_wall_fuses_exactly() {
    let poses = [
        Mat4::identity(),
        Mat4::translation(3.0, -7.0, 2.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-1.1),
    ];
    for pose in poses {
        let mut topo = Topology::new();
        let plate = load(&mut topo, "plate_hole_split_wall_plate.step");
        let pin = load(&mut topo, "plate_hole_split_wall_tool.step");
        transform_solid(&mut topo, plate, &pose).unwrap();
        transform_solid(&mut topo, pin, &pose).unwrap();
        // The pin fills part of the hole and does not overlap the plate.
        let want =
            solid_volume(&topo, plate, 1e-3).unwrap() + solid_volume(&topo, pin, 1e-3).unwrap();
        let fused = boolean(&mut topo, BooleanOp::Fuse, plate, pin).unwrap();
        assert!(validate_solid(&topo, fused).unwrap().is_valid());
        let faces = solid_faces(&topo, fused).unwrap();
        let count = |pick: fn(&FaceSurface) -> bool| {
            faces
                .iter()
                .filter(|&&f| pick(topo.face(f).unwrap().surface()))
                .count()
        };
        // The plate's two spline sides, the hole's wall, the pin's walls and
        // flats, the plate's caps and the pin's beside them (the same-domain
        // merge joins those later): not the mesh fallback's facets.
        assert_eq!(count(|s| matches!(s, FaceSurface::Nurbs(_))), 2, "splines");
        assert_eq!(count(|s| matches!(s, FaceSurface::Cylinder(_))), 4, "walls");
        assert_eq!(
            count(|s| matches!(s, FaceSurface::Plane { .. })),
            6,
            "planes"
        );
        let volume = solid_volume(&topo, fused, 1e-3).unwrap();
        assert!(
            (volume - want).abs() <= 1e-6 * want,
            "volume {volume}, want {want}"
        );
    }
}
