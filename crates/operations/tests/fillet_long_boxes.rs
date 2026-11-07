//! Fillets of a box's edges, whatever its proportions. Set back to the rolling
//! ball, a corner where three rounds meet is a patch of that ball, and the box
//! is the Minkowski sum of a smaller box and the ball. A box whose longest side
//! was over twice its shortest used to skip the set-back, and came back closed
//! and valid but thousands of cubic millimetres short, with an open mesh.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

/// A `w x d x h` box with every edge rounded at `r`: the box shrunk by `r` all
/// round, swept by a ball of radius `r`.
fn rounded_box(w: f64, d: f64, h: f64, r: f64) -> f64 {
    let (a, b, c) = (w - 2.0 * r, d - 2.0 * r, h - 2.0 * r);
    a * b * c
        + 2.0 * r * (a * b + b * c + a * c)
        + PI * r * r * (a + b + c)
        + 4.0 / 3.0 * PI * r * r * r
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut faces = BTreeMap::new();
    for face in solid_faces(topo, solid).unwrap() {
        *faces
            .entry(topo.face(face).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    faces
}

fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, what: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let exact = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (exact - volume).abs() < 1e-3,
        "{what}: volume {exact}, expected {volume}"
    );
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
    let meshed = oriented_solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (meshed - volume).abs() < 1e-3 * volume,
        "{what}: meshed volume {meshed}, expected {volume}"
    );
}

/// Every edge at once, proportions either side of two to one, in the plain pose
/// and turned, each corner a patch of its ball.
#[test]
fn every_edge_rounds_at_any_proportions() {
    let turn = Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3);
    for (w, d, h) in [
        (40.0, 30.0, 20.0),
        (40.0, 30.0, 19.0),
        (40.0, 30.0, 10.0),
        (40.0, 30.0, 5.0),
        (30.0, 20.0, 10.0),
        (100.0, 10.0, 10.0),
        (10.0, 100.0, 6.0),
        (4.0, 50.0, 80.0),
    ] {
        for turned in [false, true] {
            let mut topo = Topology::new();
            let solid = make_box(&mut topo, w, d, h).unwrap();
            let edges = solid_edges(&topo, solid).unwrap();
            if turned {
                transform_solid(&mut topo, solid, &turn).unwrap();
            }

            let result = fillet_v2(&mut topo, solid, &edges, 1.0).unwrap();

            let what = format!("{w} x {d} x {h} turned {turned}");
            assert!(result.failed.is_empty(), "{what}");
            assert_sound(&topo, result.solid, rounded_box(w, d, h, 1.0), &what);
            let faces = census(&topo, result.solid);
            assert_eq!(faces.get("sphere"), Some(&8), "{what}: {faces:?}");
            assert_eq!(faces.get("cylinder"), Some(&12), "{what}: {faces:?}");
        }
    }
}

/// A side too short for the balls at both its ends leaves no stripe between
/// them, which is refused rather than built past the box.
#[test]
fn a_side_shorter_than_its_two_balls_is_refused() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 40.0, 30.0, 1.5).unwrap();
    let edges = solid_edges(&topo, solid).unwrap();

    assert!(fillet_v2(&mut topo, solid, &edges, 1.0).is_err());
}
