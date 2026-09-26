//! A Cut that leaves a piece ringing the tool keeps its exact result: the
//! box `|x|, |y| < 3`, `-1 < z < 11` less `make_cone(5, 2, 10)`, whose
//! frustum fills the box's section below `z = 2.52` and leaves a slab under
//! it and, above, the box's corners joined into a ring around its top.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use brepkit_algo::FaceClass;
use brepkit_algo::classifier::classify_ray_cast;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cone};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// The box's volume, 432, less the frustum's part within it: the section
/// is a disc while `r = 5 - 0.3 z` is at most 3, the disc less four
/// segments up to `3 sqrt(2)`, then the whole square, integrated in `z`
/// (Simpson over each stretch, 20000 panels).
const TRUTH: f64 = 135.395_725_959_044_06;

#[test]
fn a_box_less_a_frustum_through_it_keeps_both_pieces() {
    let poses = [
        ("upright", Mat4::identity()),
        ("turned", Mat4::rotation_z(1.0) * Mat4::rotation_y(0.3)),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ];
    for (name, pose) in poses {
        let mut topo = Topology::new();
        let frustum = make_cone(&mut topo, 5.0, 2.0, 10.0).unwrap();
        let block = make_box(&mut topo, 6.0, 6.0, 12.0).unwrap();
        transform_solid(
            &mut topo,
            block,
            &(pose * Mat4::translation(-3.0, -3.0, -1.0)),
        )
        .unwrap();
        transform_solid(&mut topo, frustum, &pose).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, block, frustum).unwrap();

        let mut census = BTreeMap::new();
        for face in solid_faces(&topo, result).unwrap() {
            *census
                .entry(topo.face(face).unwrap().surface().type_tag())
                .or_insert(0) += 1;
        }
        assert_eq!(
            census,
            BTreeMap::from([("cone", 1), ("plane", 12)]),
            "{name}: faces"
        );
        assert!(
            validate_solid(&topo, result).unwrap().is_valid(),
            "{name}: invalid"
        );
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{name}: mesh open");
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - TRUTH).abs() < 1e-9 * TRUTH,
            "{name}: volume {volume}, truth {TRUTH}"
        );

        for (at, inside) in [
            ((0.0, 0.0, -0.5), true),
            ((2.9, 2.9, 5.0), true),
            ((0.0, 0.0, 10.5), true),
            ((0.0, 0.0, 5.0), false),
            ((2.9, 0.0, 1.0), false),
            ((2.9, 2.9, 1.0), false),
        ] {
            let p = pose.mul_point(Point3::new(at.0, at.1, at.2));
            let got = classify_ray_cast(&topo, result, p).unwrap() == FaceClass::Inside;
            assert_eq!(got, inside, "{name}: {at:?} read inside {got}");
        }
    }
}
