//! A closed rim rounded at exactly its wall's height: a pin as long as the
//! round's radius, its end rounded over, or a blind hole as deep, its mouth
//! rounded. The round's contact on the wall lands on the wall's other rim,
//! so no wall is left: the round runs from the cap to that rim.
//!
//! The radius is the height, or differs from it by a file's digits either
//! way. Each case runs upright, turned and shifted off the axes, and
//! mirrored. The pin keeps three faces and the holed plate eight; each
//! result is valid and watertight, and its volume agrees to 1e-4 with the
//! volume the round takes off.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

/// The wall's radius.
const RC: f64 = 3.0;
/// The wall's height.
const H: f64 = 2.0;

/// How far the centroid of a round's section (the corner's square less its
/// quarter disc) lies from the corner, for each unit of radius.
fn centroid_inset() -> f64 {
    (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI)
}

/// The volume a round of radius `r` takes off a rim of radius `RC`, its
/// section's centroid `RC + side * inset` from the axis (`side` -1 on a pin,
/// +1 round a hole).
fn removed(r: f64, side: f64) -> f64 {
    let area = (1.0 - PI / 4.0) * r * r;
    2.0 * PI * area * (side * centroid_inset()).mul_add(r, RC)
}

/// The radii each case rounds by: the wall's height, and a file's digits
/// short of it and past it.
const RADII: [f64; 3] = [H, H * (1.0 - 1e-8), H * (1.0 + 1e-8)];

fn poses() -> [(&'static str, Mat4); 3] {
    [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
        ),
        (
            "mirrored",
            Mat4::translation(1.0, 0.5, -2.0) * Mat4::scale(-1.0, 1.0, 1.0),
        ),
    ]
}

/// The part's circle edges at height `z`.
fn rims_at(topo: &Topology, part: SolidId, z: f64) -> Vec<EdgeId> {
    solid_edges(topo, part)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c) if (c.radius() - RC).abs() < 1e-9)
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
        })
        .collect()
}

/// Rounds `rim` by `r` and checks the result has `faces` faces and the
/// volume `truth`.
fn round(
    topo: &mut Topology,
    part: SolidId,
    rim: &[EdgeId],
    (r, name): (f64, &str),
    faces: usize,
    truth: f64,
) {
    let case = format!("{name}, r = {r}");
    assert_eq!(rim.len(), 1, "{case}");
    let result = fillet_v2(topo, part, rim, r).unwrap();
    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    assert_eq!(
        solid_faces(topo, result.solid).unwrap().len(),
        faces,
        "{case}"
    );
    for deflection in [0.1, 0.01] {
        let mesh = tessellate_solid(topo, result.solid, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "{case}: mesh not watertight at {deflection}"
        );
    }
    let volume = solid_volume(topo, result.solid, 0.001).unwrap();
    assert!(
        (volume - truth).abs() < 1e-4 * truth,
        "{case}: volume {volume}, truth {truth}"
    );
}

#[test]
fn a_pin_as_long_as_the_round_is_wide_is_rounded_to_its_far_rim() {
    for (name, pose) in poses() {
        for r in RADII {
            let mut topo = Topology::new();
            let pin = make_cylinder(&mut topo, RC, H).unwrap();
            let rim = rims_at(&topo, pin, H);
            transform_solid(&mut topo, pin, &pose).unwrap();
            let truth = PI * RC * RC * H - removed(r, -1.0);
            // The end cap, now the round's flat top, the round and the base.
            round(&mut topo, pin, &rim, (r, name), 3, truth);
        }
    }
}

#[test]
fn a_blind_hole_as_deep_as_the_round_is_wide_is_rounded_to_its_floor() {
    let (side, thick) = (20.0, 10.0);
    for (name, pose) in poses() {
        for r in RADII {
            let mut topo = Topology::new();
            let plate = make_box(&mut topo, side, side, thick).unwrap();
            let drill = make_cylinder(&mut topo, RC, H + 1.0).unwrap();
            transform_solid(
                &mut topo,
                drill,
                &Mat4::translation(side / 2.0, side / 2.0, thick - H),
            )
            .unwrap();
            let part = boolean(&mut topo, BooleanOp::Cut, plate, drill).unwrap();
            let rim = rims_at(&topo, part, thick);
            transform_solid(&mut topo, part, &pose).unwrap();
            let truth = side * side * thick - PI * RC * RC * H - removed(r, 1.0);
            // The plate's six faces, the hole's floor and the round.
            round(&mut topo, part, &rim, (r, name), 8, truth);
        }
    }
}
