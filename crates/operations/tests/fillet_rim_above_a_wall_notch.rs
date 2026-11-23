//! A rod's top rim rounded above a steep notch cut low in its wall. The
//! notch's sides are planes nearly along the axis, so they cut the wall in
//! short arcs of ellipses whose whole curves reach far up the wall, past
//! where the round's contact lands. Only the arcs bound the wall, and they
//! lie well below the contact: the round is built, and takes off what it
//! takes off the plain rod.
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
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

/// The rod's radius.
const RC: f64 = 10.0;
/// The rod's height.
const H: f64 = 30.0;

/// The volume a round of radius `r` takes off a convex rim of radius `RC`:
/// the corner's square less its quarter disc, swept round at its centroid.
fn removed(r: f64) -> f64 {
    let inset = (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
    2.0 * PI * (1.0 - PI / 4.0) * r * r * (-inset).mul_add(r, RC)
}

/// The rod's circle edges of radius `RC` at height `z`.
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

/// A rod with a slot cut into its side by a slab leaning `lean` radians
/// off the axis, centred `depth` below the top.
fn notched_rod(topo: &mut Topology, lean: f64, depth: f64) -> SolidId {
    let rod = make_cylinder(topo, RC, H).unwrap();
    let (w, t, l) = (40.0, 3.0, 20.0);
    let slab = make_box(topo, w, t, l).unwrap();
    // The slab's middle at the origin, leant about x, then pushed out to the
    // wall at y = RC (half its depth into the rod) and down to its height.
    let place = Mat4::translation(0.0, RC, H - depth)
        * Mat4::rotation_x(lean)
        * Mat4::translation(-w / 2.0, -t / 2.0, -l / 2.0);
    transform_solid(topo, slab, &place).unwrap();
    boolean(topo, BooleanOp::Cut, rod, slab).unwrap()
}

#[test]
fn a_rim_above_a_steep_notch_is_rounded() {
    for lean in [0.15, 0.3] {
        for r in [1.0, 2.5] {
            let case = format!("lean {lean}, r = {r}");
            let mut topo = Topology::new();
            let part = notched_rod(&mut topo, lean, 18.0);
            let ellipses = solid_edges(&topo, part)
                .unwrap()
                .into_iter()
                .filter(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Ellipse(_)))
                .count();
            assert!(ellipses > 0, "{case}: the notch cut no ellipse arcs");
            let before = solid_volume(&topo, part, 0.001).unwrap();
            let rim = rims_at(&topo, part, H);
            assert_eq!(rim.len(), 1, "{case}");

            let result = fillet_v2(&mut topo, part, &rim, r).unwrap();
            assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
            assert!(is_watertight(&mesh), "{case}: mesh not watertight");
            let after = solid_volume(&topo, result.solid, 0.001).unwrap();
            let want = removed(r);
            assert!(
                ((before - after) - want).abs() < 1e-4 * want.max(1.0),
                "{case}: took off {}, want {want}",
                before - after
            );
        }
    }
}
