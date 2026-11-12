//! A box's four top edges chamfered: each bevel lands on the box's sides at
//! its distance below the top, so a distance past the box's height has no
//! side to land on. Such a chamfer is refused; it used to come back as a
//! closed, valid solid of the wrong volume. Distances that fit build the
//! hip roof's closed-form volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const LENGTH: f64 = 100.0;
const WIDTH: f64 = 60.0;
const HEIGHT: f64 = 20.0;

fn top_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()]
                .iter()
                .all(|&v| (topo.vertex(v).unwrap().point().z() - HEIGHT).abs() < 1e-9)
        })
        .collect()
}

fn chamfer_top(d: f64) -> Result<(Topology, SolidId), brepkit_operations::OperationsError> {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, LENGTH, WIDTH, HEIGHT).unwrap();
    let edges = top_edges(&topo, solid);
    assert_eq!(edges.len(), 4);
    let result = chamfer(&mut topo, solid, &edges, d)?;
    Ok((topo, result))
}

#[test]
fn chamfers_that_fit_build_the_hip_roof() {
    for d in [5.0, 12.5, 19.999] {
        let (topo, solid) = chamfer_top(d).unwrap();
        let report = validate_solid(&topo, solid).unwrap();
        assert!(report.is_valid(), "d {d}: {:?}", report.issues);
        // The top slab's section at depth t below the top is
        // (L - 2t)(W - 2t) for t < d.
        let truth = (4.0_f64 / 3.0).mul_add(
            d * d * d,
            (LENGTH + WIDTH).mul_add(-d * d, LENGTH * WIDTH * HEIGHT),
        );
        let volume = solid_volume(&topo, solid, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-9 * truth,
            "d {d}: volume {volume}, truth {truth}"
        );
    }
}

#[test]
fn chamfers_past_the_sides_are_refused() {
    for d in [20.0, 25.0, 29.999, 30.0, 35.0] {
        assert!(chamfer_top(d).is_err(), "d {d} was built");
    }
}
