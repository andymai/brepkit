//! A lens (the common part of two balls) fused with a collar that only
//! touches it: the collar lies inside a third ball and outside both of the
//! lens's, so the two meet along the lens's rim alone, the circle where its
//! spheres cross, and the collar's inner faces are bands of the lens's own
//! spheres. Both are turned off the axes, as a file gives them.
//!
//! The file's spheres were read in a frame of their own, their rim circles
//! off its latitudes, so the lens's caps took sections their spheres'
//! circles with the third sphere make beside them, and were cut along
//! circles they do not hold: the fuse came out a whole ball and more, ten
//! times too big, and valid. Read in the file's frame, the circles are
//! latitudes, and the four faces on the rim pair off into the lens's wedge
//! and the collar's, each a closed piece.
//!
//! The operands are read from STEP, written by brepkit's own writer.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/{name}.step"));
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn a_lens_fused_with_a_collar_on_its_rim_keeps_both() {
    let mut topo = Topology::new();
    let lens = load(&mut topo, "sphere_lens");
    let collar = load(&mut topo, "sphere_lens_collar");
    let (a, b) = (
        solid_volume(&topo, lens, 1e-3).unwrap(),
        solid_volume(&topo, collar, 1e-3).unwrap(),
    );
    // The lens's two caps, each a radius of its sphere times the cap's
    // height round: r1 = 125 and r2 = 175, their centres 242.3 apart.
    let caps = |r: f64, h: f64| std::f64::consts::PI * h * h * (3.0 * r - h) / 3.0;
    let d = 168.751_503_822_24_f64.hypot(173.850_331_508_36);
    let along = (d * d + 125.0 * 125.0 - 175.0 * 175.0) / (2.0 * d);
    let closed_form = caps(125.0, 125.0 - along) + caps(175.0, 175.0 - (d - along));
    assert!(
        (a - closed_form).abs() <= 1e-6 * closed_form,
        "lens {a}, closed form {closed_form}"
    );
    let fused = boolean(&mut topo, BooleanOp::Fuse, lens, collar).unwrap();
    let faces = solid_faces(&topo, fused).unwrap();
    assert!(
        faces
            .iter()
            .all(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_))),
        "the fuse fell back to a mesh"
    );
    // The lens's two caps and the collar's three bands.
    assert_eq!(faces.len(), 5);
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    // They only touch, so the fuse holds exactly both.
    let v = solid_volume(&topo, fused, 1e-3).unwrap();
    assert!(
        (v - (a + b)).abs() <= 1e-6 * (a + b),
        "fuse {v}, operands {a} + {b}"
    );
}
