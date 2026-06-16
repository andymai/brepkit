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
use brepkit_topology::explorer::solid_faces;

#[test]
fn coaxial_cone_cut_keeps_inner_cavity_wall() {
    let mut topo = Topology::new();
    let outer = make_cone(&mut topo, 10.0, 8.0, 10.0).unwrap();
    let inner = make_cone(&mut topo, 9.0, 8.0, 10.0).unwrap();

    let result = gfa::boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();

    let cones = solid_faces(&topo, result)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "cone")
        .count();

    assert!(
        cones >= 2,
        "Cut must keep both cone walls (outer + reversed inner cavity wall); got {cones}"
    );
}
