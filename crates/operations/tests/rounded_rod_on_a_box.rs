//! A rod whose bottom rim is rounded, standing on a box. The fillet's torus
//! rests on the box's top along the rod's bottom cap rim, a plane tangent to
//! the torus's tube, so the union is the two solids side by side: its volume
//! is their sum. A small fillet gives a ring torus, one past half the rod's
//! radius a spindle torus.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::solid_edges;

#[test]
fn a_rod_rounded_at_its_foot_fuses_onto_a_box_exactly() {
    for r in [1.0, 3.0] {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 5.0, 6.0).unwrap();
        let foot = solid_edges(&topo, rod)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), EdgeCurve::Circle(_))
                    && topo.vertex(edge.start()).unwrap().point().z().abs() < 1e-9
            })
            .unwrap();
        let rounded = fillet_v2(&mut topo, rod, &[foot], r).unwrap();
        assert!(rounded.failed.is_empty(), "r = {r}");
        let block = make_box(&mut topo, 20.0, 20.0, 4.0).unwrap();
        transform_solid(&mut topo, block, &Mat4::translation(-10.0, -10.0, -4.0)).unwrap();
        let parts = solid_volume(&topo, rounded.solid, 0.001).unwrap()
            + solid_volume(&topo, block, 0.001).unwrap();

        let before = mesh_fallback_count();
        let fused = boolean(&mut topo, BooleanOp::Fuse, rounded.solid, block).unwrap();
        assert_eq!(mesh_fallback_count(), before, "r = {r}");
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r = {r}");
        let volume = solid_volume(&topo, fused, 0.001).unwrap();
        assert!(
            (volume - parts).abs() < 1e-6 * parts,
            "r = {r}: {volume} against {parts}"
        );
    }
}
