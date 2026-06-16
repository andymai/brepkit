//! Regression: a Cut between two coaxial truncated cones that share their top
//! cap must keep the inner cone wall as the cavity boundary.
//!
//! The inner cone (r9->8) lies strictly inside the outer cone (r10->8) except
//! at the shared top rim (r8 @ z=10). For the Cut, the inner wall must be kept
//! and reversed as the cavity boundary. The classifier samples an interior
//! point of each sub-face; before the `interior_point_3d` axial-midpoint fix,
//! the inner wall's sample landed on a bounding circle (the shared top rim, a
//! boundary of the opposing solid), so it classified `Outside` and the Cut
//! dropped it — collapsing the result to a single open face.

#![allow(clippy::unwrap_used)]

use brepkit_algo::bop::BooleanOp;
use brepkit_algo::gfa;
use brepkit_operations::primitives::make_cone;
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::explorer::solid_faces;

#[test]
fn coaxial_cone_cut_keeps_inner_cavity_wall() {
    let mut topo = Topology::new();
    let outer = make_cone(&mut topo, 10.0, 8.0, 10.0).unwrap();
    let inner = make_cone(&mut topo, 9.0, 8.0, 10.0).unwrap();

    let result = gfa::boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();

    // The cut is a conical washer: outer cone wall + reversed inner cavity
    // wall + bottom annulus = three analytic faces, all curved/planar (never a
    // mesh fallback). The shared top rim (where the cone-cone intersection
    // circle coincides with both cones' top boundary) must be a single linked
    // edge, so the shell is watertight (every edge shared by exactly 2 faces).
    let faces = solid_faces(&topo, result).unwrap();
    let cones = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "cone")
        .count();
    let planes = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "plane")
        .count();
    assert_eq!(cones, 2, "expected both cone walls, got {cones}");
    assert_eq!(planes, 1, "expected the bottom annulus, got {planes}");

    let adj = AdjacencyIndex::build_from_faces(&topo, &faces).unwrap();
    let free = adj
        .edge_faces_iter()
        .filter(|(_, fs)| fs.len() != 2)
        .count();
    assert_eq!(
        free, 0,
        "cut shell must be watertight, got {free} free edges"
    );
}
