//! A concave edge whose one face is a wide wall notched at the edge: a
//! pocket cut into a block's corner under a groove that runs the block's
//! whole width, so the pocket's back wall and the groove's are one face.
//! Beside the edge that face runs away from the pocket's side wall, but its
//! far end reaches past that wall's plane on the other side. The plane/plane
//! round read the edge's convexity from each face's farthest vertex, took
//! the edge for convex, and built its round on the material side, a mirror
//! of the right one through the edge.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::heal::unify_faces;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// A box `size` with its low corner at `at`.
fn block(topo: &mut Topology, at: [f64; 3], size: [f64; 3]) -> SolidId {
    let solid = make_box(topo, size[0], size[1], size[2]).unwrap();
    transform_solid(topo, solid, &Mat4::translation(at[0], at[1], at[2])).unwrap();
    solid
}

/// The pocket's edge at `x = 20, y = 10`, rounded by `r`, the whole part
/// turned about `z` by `turn`. The part's bottom is tilted about `x` by
/// `tilt`.
fn round_the_pocket_corner(r: f64, turn: f64, tilt: f64) {
    let mut topo = Topology::new();
    let bottom = if tilt == 0.0 { 1.0 } else { -5.0 };
    let part = block(&mut topo, [-30.0, 0.0, bottom], [70.0, 20.0, 10.0 - bottom]);
    let pocket = block(
        &mut topo,
        [20.0, 10.0, bottom - 1.0],
        [25.0, 15.0, 9.0 - bottom],
    );
    let part = boolean(&mut topo, BooleanOp::Cut, part, pocket).unwrap();
    let groove = block(&mut topo, [-31.0, 10.0, 8.0], [76.0, 15.0, 1.0]);
    let part = boolean(&mut topo, BooleanOp::Cut, part, groove).unwrap();
    // Below the plane through (0, 0, 1) tilted about x.
    let part = if tilt == 0.0 {
        part
    } else {
        let below = block(&mut topo, [-100.0, -100.0, -50.0], [200.0, 200.0, 50.0]);
        transform_solid(&mut topo, below, &Mat4::rotation_x(tilt)).unwrap();
        transform_solid(&mut topo, below, &Mat4::translation(0.0, 0.0, 1.0)).unwrap();
        boolean(&mut topo, BooleanOp::Cut, part, below).unwrap()
    };
    unify_faces(&mut topo, part).unwrap();
    // The back wall is one face, notched at the edge.
    let walls = solid_faces(&topo, part)
        .unwrap()
        .into_iter()
        .filter(|&f| {
            let face = topo.face(f).unwrap();
            matches!(face.surface(), FaceSurface::Plane { normal, d }
                if normal.y().abs() > 0.999 && (d.abs() - 10.0).abs() < 1e-9)
        })
        .count();
    assert_eq!(walls, 1, "the back wall is one face");
    let edge: Vec<_> = solid_edges(&topo, part)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                (p.x() - 20.0).abs() < 1e-9 && (p.y() - 10.0).abs() < 1e-9
            })
        })
        .collect();
    assert_eq!(edge.len(), 1);
    transform_solid(&mut topo, part, &Mat4::rotation_z(turn)).unwrap();
    let before = solid_volume(&topo, part, 0.01).unwrap();
    let result = fillet_v2(&mut topo, part, &edge, r).unwrap();
    let case = format!("r = {r}, turned {turn}, tilted {tilt}");
    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{case}: mesh not watertight");
    // The round's axis stands in the pocket's corner, a radius out from
    // both walls, not mirrored into the material.
    let axis = Mat4::rotation_z(turn).mul_point(Point3::new(20.0 + r, 10.0 + r, 0.0));
    let rounds: Vec<_> = solid_faces(&topo, result.solid)
        .unwrap()
        .into_iter()
        .filter_map(|f| match topo.face(f).unwrap().surface() {
            FaceSurface::Cylinder(c) if (c.radius() - r).abs() < 1e-9 => Some(c.origin()),
            _ => None,
        })
        .collect();
    assert_eq!(rounds.len(), 1, "{case}");
    let off = rounds[0] - axis;
    assert!(
        off.x().hypot(off.y()) < 1e-9,
        "{case}: the round's axis is at {:?}, not {axis:?}",
        rounds[0]
    );
    // It fills the corner: (1 - pi/4) r^2 along the edge, from the bottom
    // up to the groove's floor.
    let added = solid_volume(&topo, result.solid, 0.01).unwrap() - before;
    let along = 8.0 - (1.0 + 10.0 * tilt.tan());
    let expected = (1.0 - PI / 4.0) * r * r * along;
    assert!(
        (added - expected).abs() < 0.05 * expected,
        "{case}: added {added}, about {expected}"
    );
}

#[test]
fn a_concave_edge_of_a_notched_wall_is_filled() {
    for r in [1.0, 2.0] {
        round_the_pocket_corner(r, 0.0, 0.0);
        round_the_pocket_corner(r, 0.7, 0.0);
    }
}
