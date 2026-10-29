//! Filleting every edge of a box at one radius rounds each corner with a
//! patch of the ball that touches the corner's three faces: an eighth of a
//! sphere. The result is the inner box grown by the ball (Steiner), so its
//! volume and area have closed forms.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::{face_area, solid_surface_area, solid_volume};
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;

#[test]
fn a_boxs_corners_round_into_eighths_of_the_ball() {
    let (x, y, z, r) = (40.0, 30.0, 20.0, 3.0);
    let mut topo = Topology::new();
    let block = make_box(&mut topo, x, y, z).unwrap();
    let edges = solid_edges(&topo, block).unwrap();
    let result = fillet_v2(&mut topo, block, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");

    let (a, b, c) = (x - 2.0 * r, y - 2.0 * r, z - 2.0 * r);
    let volume = a * b * c
        + 2.0 * r * (a * b + b * c + c * a)
        + PI * r * r * (a + b + c)
        + 4.0 / 3.0 * PI * r.powi(3);
    let area = 2.0 * (a * b + b * c + c * a) + 2.0 * PI * r * (a + b + c) + 4.0 * PI * r * r;
    let measured = solid_volume(&topo, result.solid, 0.001).unwrap();
    assert!(
        (measured - volume).abs() < 1e-6 * volume,
        "{measured} vs {volume}"
    );
    let measured = solid_surface_area(&topo, result.solid, 0.001).unwrap();
    assert!(
        (measured - area).abs() < 1e-6 * area,
        "{measured} vs {area}"
    );

    let corners: Vec<f64> = solid_faces(&topo, result.solid)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .map(|f| face_area(&topo, f, 0.001).unwrap())
        .collect();
    assert_eq!(corners.len(), 8);
    for corner in corners {
        assert!(
            (corner - PI * r * r / 2.0).abs() < 1e-9,
            "corner area {corner}"
        );
    }
}
