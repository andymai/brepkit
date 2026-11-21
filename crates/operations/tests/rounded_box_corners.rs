//! A box with all twelve edges filleted at one radius is the box shrunk by
//! `r` on every side, swept by a ball of radius `r`: its volume is
//! `A'B'C' + 2r(A'B' + B'C' + C'A') + pi r^2 (A' + B' + C') + 4/3 pi r^3`
//! for the shrunk sides `A' = a - 2r` and so on. Each corner is an eighth
//! of that ball, which a NURBS patch only approximated (0.15% of the
//! volume short on a 10 x 10 x 5 box with r = 1).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;

#[test]
fn a_box_rounded_on_every_edge_has_ball_corners() {
    let (a, b, c, r) = (10.0, 10.0, 5.0, 1.0);
    let mut topo = Topology::new();
    let block = make_box(&mut topo, a, b, c).unwrap();
    let edges = solid_edges(&topo, block).unwrap();
    assert_eq!(edges.len(), 12);
    let result = fillet_v2(&mut topo, block, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);

    let faces = solid_faces(&topo, result.solid).unwrap();
    let balls = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .count();
    assert_eq!((faces.len(), balls), (26, 8));
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");

    let (sa, sb, sc) = (a - 2.0 * r, b - 2.0 * r, c - 2.0 * r);
    let truth = 2.0f64.mul_add(r * sc.mul_add(sa, sa.mul_add(sb, sb * sc)), sa * sb * sc)
        + PI * r * r * (sa + sb + sc)
        + 4.0 / 3.0 * PI * r * r * r;
    let vol = solid_volume(&topo, result.solid, 0.001).unwrap();
    assert!(
        (vol - truth).abs() < 1e-5 * truth,
        "volume {vol}, truth {truth}"
    );
}
