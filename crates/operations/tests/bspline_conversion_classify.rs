//! A solid converted to B-splines still holds its own interior: points on its
//! axis classify inside it, and a cut by it removes it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::heal::convert_to_bspline;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// A rod of radius 0.6 along x through `(0, y, 3)`. Laid on its side, its
/// seam edge points down while its cylinder's angles start a quarter turn
/// away, so a patch converted from the cylinder as it stands starts off the
/// seam.
fn rod(topo: &mut Topology, y: f64) -> SolidId {
    let rod = make_cylinder(topo, 0.6, 10.0).unwrap();
    let place = Mat4::translation(-5.0, y, 3.0) * Mat4::rotation_y(std::f64::consts::FRAC_PI_2);
    transform_solid(topo, rod, &place).unwrap();
    rod
}

const INSIDE: [(f64, f64); 4] = [(-4.0, 0.0), (0.0, 0.0), (0.0, 0.5), (3.0, -0.5)];
const OUTSIDE: [(f64, f64); 2] = [(0.0, 0.7), (-3.0, -0.7)];

/// Both point classifiers read a converted rod's axis inside it and points
/// past its wall outside.
#[test]
fn a_converted_rod_holds_its_own_axis() {
    let mut topo = Topology::new();
    let rod = rod(&mut topo, 0.3);
    convert_to_bspline(&mut topo, rod).unwrap();
    for (inside, points) in [(true, &INSIDE[..]), (false, &OUTSIDE[..])] {
        for &(x, dy) in points {
            let p = Point3::new(x, 0.3 + dy, 3.0);
            let class = classify_point(&topo, rod, p, 0.01, 1e-7).unwrap();
            let want = if inside {
                PointClassification::Inside
            } else {
                PointClassification::Outside
            };
            assert_eq!(class, want, "classify_point at {p:?}");
            let engine = brepkit_algo::classifier::classify_point(&topo, rod, p).unwrap();
            assert_eq!(
                engine == brepkit_algo::FaceClass::Inside,
                inside,
                "the boolean engine's classifier at {p:?}"
            );
        }
    }
}

/// A cylinder cut by a converted rod through its axis loses the rod: the
/// cut once read the rod as empty and returned the whole cylinder, 5.6%
/// over.
#[test]
fn a_cylinder_cut_by_a_converted_rod_loses_the_rod() {
    let mut topo = Topology::new();
    let exact = {
        let cylinder = make_cylinder(&mut topo, 2.25, 6.0).unwrap();
        let rod = rod(&mut topo, 0.0);
        let cut = boolean(&mut topo, BooleanOp::Cut, cylinder, rod).unwrap();
        solid_volume(&topo, cut, 0.001).unwrap()
    };
    let cylinder = make_cylinder(&mut topo, 2.25, 6.0).unwrap();
    let rod = rod(&mut topo, 0.0);
    convert_to_bspline(&mut topo, rod).unwrap();
    let cut = boolean(&mut topo, BooleanOp::Cut, cylinder, rod).unwrap();
    let volume = solid_volume(&topo, cut, 0.001).unwrap();
    assert!(
        (volume - exact).abs() < 0.02 * exact,
        "cut volume {volume}, exact {exact}"
    );
}
