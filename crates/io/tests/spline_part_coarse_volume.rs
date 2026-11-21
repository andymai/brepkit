//! An imported part with B-spline faces measured on a coarse mesh. Some of
//! its faces' boundary points stand a hair off their surfaces where the
//! proof of a nearest point gives up; the mesher lost those points, and the
//! faces' meshes round them, and the part read 0.7% light. Such a point
//! takes the nearest point found, unproven, and the coarse volume holds to
//! the fine one. The part is a case's, written by brepkit's STEP writer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;

#[test]
fn a_spline_part_measures_alike_coarse_and_fine() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/spline_part_coarse_volume.step");
    let mut topo = Topology::new();
    let solid = read_step(&std::fs::read_to_string(path).unwrap(), &mut topo).unwrap()[0];
    let fine = solid_volume(&topo, solid, 1e-3).unwrap();
    let coarse = solid_volume(&topo, solid, 0.05).unwrap();
    assert!((fine - 4606.03).abs() < 0.1, "fine {fine}");
    assert!(
        (coarse - fine).abs() <= 1e-3 * fine,
        "coarse {coarse}, fine {fine}"
    );
}
