//! A rod whose end sits at the centre of a ball of its own radius (a wire
//! guard's strut and the ball on its end): the end cap's rim lies on the ball
//! and the rod's wall touches the ball along it.
//!
//! The ball's faces meet the cap and the wall along that one rim, which both
//! sections must break where the rim's vertex is, whichever way the rod is
//! turned about its own axis.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_cylinder, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 0.9;
const LENGTH: f64 = 25.0;

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        *m.entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    m
}

fn edges_not_used_twice(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wire).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

#[test]
fn a_rod_ending_at_a_balls_centre_fuses_onto_it() {
    let expected = PI * RADIUS * RADIUS * LENGTH + 2.0 / 3.0 * PI * RADIUS.powi(3);
    let at = Point3::new(28.638, 28.638, 2.65);
    for roll in [0.0, 0.3, 1.2, 2.5] {
        for yaw in [0.0, FRAC_PI_2 / 2.0, 2.0] {
            let name = format!("roll {roll}, yaw {yaw}");
            // The rod runs along -x from the ball's centre once placed.
            let place = Mat4::translation(at.x(), at.y(), at.z())
                * Mat4::rotation_z(yaw)
                * Mat4::rotation_y(-FRAC_PI_2)
                * Mat4::rotation_z(roll);
            let mut topo = Topology::new();
            let rod = make_cylinder(&mut topo, RADIUS, LENGTH).unwrap();
            transform_solid(&mut topo, rod, &place).unwrap();
            let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
            transform_solid(&mut topo, ball, &Mat4::translation(at.x(), at.y(), at.z())).unwrap();
            let before = boolean::mesh_fallback_count();
            let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, rod, ball).unwrap();
            assert_eq!(
                boolean::mesh_fallback_count(),
                before,
                "mesh fallback, {name}"
            );
            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            assert_eq!(
                census(&topo, fused),
                BTreeMap::from([("cylinder", 1), ("plane", 1), ("sphere", 2)]),
                "{name}"
            );
            assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
            let volume = solid_volume(&topo, fused, 0.0005).unwrap();
            assert!(
                (volume - expected).abs() < 1e-4 * expected,
                "{name}: volume {volume} should be {expected}"
            );
            // Along the rod, out past its end into the ball, and beside it.
            let local = |x: f64, y: f64, z: f64| {
                (Mat4::translation(at.x(), at.y(), at.z()) * Mat4::rotation_z(yaw))
                    .mul_point(Point3::new(x, y, z))
            };
            let class = |p: Point3| classify_point(&topo, fused, p, 0.01, 1e-7).unwrap();
            assert_eq!(
                class(local(-5.0, 0.0, 0.0)),
                PointClassification::Inside,
                "{name}"
            );
            assert_eq!(
                class(local(0.6, 0.0, 0.3)),
                PointClassification::Inside,
                "{name}"
            );
            assert_eq!(
                class(local(0.8, 0.0, 0.5)),
                PointClassification::Outside,
                "{name}"
            );
            assert_eq!(
                class(local(-5.0, 1.0, 0.0)),
                PointClassification::Outside,
                "{name}"
            );
        }
    }
}
