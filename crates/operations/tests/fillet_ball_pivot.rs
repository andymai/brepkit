//! A round whose radius is that of a round it meets.
//!
//! Rounding the rim where an earlier round of radius `r` meets a face, again
//! at `r`, puts the ball's centre on the earlier round's axis: the ball
//! touches that round all along a circle and the face at one point, and
//! pivots there while it rolls. The band is a patch of the ball, and its
//! contact on the face is a point. The contact was built as an edge from that
//! point back to itself, which the band and the face both used in one sense,
//! so the faces' orientations could not agree.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const A: f64 = 20.0;
const H: f64 = 10.0;
const R: f64 = 2.0;

/// The edges of `solid` whose ends both satisfy `pick`.
fn edges_where(
    topo: &Topology,
    solid: SolidId,
    pick: impl Fn(f64, f64, f64) -> bool,
) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                pick(p.x(), p.y(), p.z())
            })
        })
        .collect()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

/// The box's vertical edge at `x = y = A`.
fn corner_edge(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    edges_where(topo, solid, |x, y, _| near(x, A) && near(y, A))
}

/// The edges round the box's top: its four sides, and the arc where the
/// rounded corner meets the top once it is rounded.
fn top_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    edges_where(topo, solid, |_, _, z| near(z, H))
}

#[test]
fn rounding_a_round_s_rim_at_its_own_radius_pivots_the_ball() {
    // The corner rounded first, then the top's rim with its arc.
    let mut topo = Topology::new();
    let block = make_box(&mut topo, A, A, H).unwrap();
    let corner = corner_edge(&topo, block);
    assert_eq!(corner.len(), 1);
    let rounded = fillet_v2(&mut topo, block, &corner, R).unwrap();
    assert!(rounded.failed.is_empty(), "{:?}", rounded.failed);
    let rim = top_edges(&topo, rounded.solid);
    assert!(
        rim.iter()
            .any(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_))),
        "the rounded corner's top arc"
    );
    let twice = fillet_v2(&mut topo, rounded.solid, &rim, R).unwrap();
    assert!(twice.failed.is_empty(), "{:?}", twice.failed);
    let report = validate_solid(&topo, twice.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    // The same rounds at once: the corner and the top's four sides.
    let mut once_topo = Topology::new();
    let block = make_box(&mut once_topo, A, A, H).unwrap();
    let mut edges = corner_edge(&once_topo, block);
    edges.extend(top_edges(&once_topo, block));
    let once = fillet_v2(&mut once_topo, block, &edges, R).unwrap();
    assert!(once.failed.is_empty(), "{:?}", once.failed);

    let v_twice = solid_volume(&topo, twice.solid, 1e-3).unwrap();
    let v_once = solid_volume(&once_topo, once.solid, 1e-3).unwrap();
    assert!(
        (v_twice - v_once).abs() < 1e-4 * v_once,
        "rounded twice {v_twice}, at once {v_once}"
    );
}
