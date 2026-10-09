//! Point classification at a rounded pocket's corners, where the pocket's
//! floor rim (r 2.45) turns a corner of r 2.55 through a spindle torus
//! (major radius 0.1). The material is the gridfinity tool's interior fillet
//! material for "a scoop beside tapered side walls keeps the plain fillet"
//! (`binGenerator.export.interiorFilletScoops.test.ts`): a rounded box (half
//! sizes 44.03 by 20.03, corners r 3.03 about (+-41, +-17), z 1.25 to 22.55)
//! less its pocket (floor at z 2.25), with a scoop along the -y wall.
//!
//! Data: `scoop_beside_taper_material.bin`.

#![allow(clippy::unwrap_used)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_topology::Topology;

/// A ray through a corner torus also meets the tube swung across its axis,
/// which projects onto the face's own tube and once counted as a crossing.
#[test]
fn pocket_air_in_a_spindle_torus_corner_is_outside() {
    let mut topo = Topology::new();
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/scoop_beside_taper_material.bin");
    let material = deserialize_solid(&std::fs::read(path).unwrap(), &mut topo).unwrap();
    let at = |x: f64, y: f64, z: f64| {
        classify_point(&topo, material, Point3::new(x, y, z), 0.01, 1e-7).unwrap()
    };
    // Inside the corner tubes: the pocket's air.
    for (x, y, z) in [
        (42.36, 18.31, 3.4),
        (41.97, 17.91, 3.0),
        (-42.36, 18.31, 3.4),
        (-41.97, 17.91, 3.0),
    ] {
        assert_eq!(at(x, y, z), PointClassification::Outside, "({x}, {y}, {z})");
    }
    // Between a corner tube and the outer corner, and in the floor slab.
    for (x, y, z) in [(43.8, 18.0, 3.0), (-43.8, 18.0, 3.0), (42.0, 17.5, 1.8)] {
        assert_eq!(at(x, y, z), PointClassification::Inside, "({x}, {y}, {z})");
    }
}
