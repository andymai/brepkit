//! Point classification in a solid with closed cavities: a point inside a
//! cavity is outside the solid, as is a point beyond it, and material
//! between a cavity and the outer skin is inside.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_check::classify::{ClassifyOptions, PointClassification};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// A 10-unit block less a cylinder (r = 1.5, z 3..7) centred at (3, 5) and
/// a 3-unit cube centred at (7, 5, 5), both enclosed.
fn block_with_two_cavities(topo: &mut Topology) -> SolidId {
    let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    let rod = make_cylinder(topo, 1.5, 4.0).unwrap();
    transform_solid(topo, rod, &Mat4::translation(3.0, 5.0, 3.0)).unwrap();
    let cube = make_box(topo, 3.0, 3.0, 3.0).unwrap();
    transform_solid(topo, cube, &Mat4::translation(5.5, 3.5, 3.5)).unwrap();
    let once = boolean(topo, BooleanOp::Cut, block, rod).unwrap();
    let twice = boolean(topo, BooleanOp::Cut, once, cube).unwrap();
    assert_eq!(topo.solid(twice).unwrap().inner_shells().len(), 2);
    twice
}

type CheckClassifier = fn(
    &Topology,
    SolidId,
    Point3,
    &ClassifyOptions,
) -> Result<PointClassification, brepkit_check::CheckError>;
type OperationsClassifier = fn(
    &Topology,
    SolidId,
    Point3,
    f64,
    f64,
) -> Result<
    brepkit_operations::classify::PointClassification,
    brepkit_operations::OperationsError,
>;

const INSIDE: [(f64, f64, f64); 4] = [
    (1.0, 1.0, 1.0),
    (5.0, 5.0, 5.0),
    (3.0, 5.0, 8.0),
    (7.0, 5.0, 7.5),
];
const OUTSIDE: [(f64, f64, f64); 5] = [
    (3.0, 5.0, 5.0),
    (3.8, 5.6, 3.5),
    (7.0, 5.0, 5.0),
    (7.5, 4.5, 6.3),
    (11.0, 5.0, 5.0),
];

/// The winding variants integrate fan-triangulated wire polygons, which do
/// not cover a curved face, so only the ray classifier is held to this.
#[test]
fn check_classifier_sees_cavities() {
    let mut topo = Topology::new();
    let solid = block_with_two_cavities(&mut topo);
    let options = ClassifyOptions::default();
    let classifiers: [(&str, CheckClassifier); 1] =
        [("ray", brepkit_check::classify::classify_point)];
    for (name, classify) in classifiers {
        for (points, expected) in [
            (&INSIDE[..], PointClassification::Inside),
            (&OUTSIDE[..], PointClassification::Outside),
        ] {
            for &(x, y, z) in points {
                let p = Point3::new(x, y, z);
                assert_eq!(
                    classify(&topo, solid, p, &options).unwrap(),
                    expected,
                    "{name} at {p:?}"
                );
            }
        }
    }
}

#[test]
fn operations_classifiers_see_cavities() {
    use brepkit_operations::classify::{
        PointClassification as Class, classify_point, classify_point_robust, classify_point_winding,
    };
    let mut topo = Topology::new();
    let solid = block_with_two_cavities(&mut topo);
    let classifiers: [(&str, OperationsClassifier); 3] = [
        ("ray", classify_point),
        ("winding", classify_point_winding),
        ("robust", classify_point_robust),
    ];
    for (name, classify) in classifiers {
        for (points, expected) in [(&INSIDE[..], Class::Inside), (&OUTSIDE[..], Class::Outside)] {
            for &(x, y, z) in points {
                let p = Point3::new(x, y, z);
                assert_eq!(
                    classify(&topo, solid, p, 0.01, 1e-6).unwrap(),
                    expected,
                    "{name} at {p:?}"
                );
            }
        }
    }
}

/// With planar faces only, the check crate's winding number is exact, so its
/// winding and robust classifiers are held to cavities too: a block less an
/// enclosed cube.
#[test]
fn check_winding_classifiers_see_a_planar_cavity() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let cube = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    transform_solid(&mut topo, cube, &Mat4::translation(3.0, 3.0, 3.0)).unwrap();
    let hollow = boolean(&mut topo, BooleanOp::Cut, block, cube).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let options = ClassifyOptions::default();
    let classifiers: [(&str, CheckClassifier); 2] = [
        ("winding", brepkit_check::classify::classify_point_winding),
        ("robust", brepkit_check::classify::classify_point_robust),
    ];
    for (name, classify) in classifiers {
        for (x, y, z, expected) in [
            (1.0, 1.0, 1.0, PointClassification::Inside),
            (5.0, 5.0, 8.0, PointClassification::Inside),
            (5.0, 5.0, 5.0, PointClassification::Outside),
            (4.0, 6.5, 3.5, PointClassification::Outside),
            (11.0, 5.0, 5.0, PointClassification::Outside),
        ] {
            let p = Point3::new(x, y, z);
            assert_eq!(
                classify(&topo, hollow, p, &options).unwrap(),
                expected,
                "{name} at {p:?}"
            );
        }
    }
}
