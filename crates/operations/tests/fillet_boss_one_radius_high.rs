//! A boss exactly a radius high, its top outline rounded: each round's
//! contact on the boss's wall lands on the wall's bottom edge, so no wall
//! is left. Each round meets the plate along that edge: the result is
//! valid, has fifteen faces and no wall among them, its mesh is watertight,
//! and its volume agrees with the volume the rounds take off.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

const R: f64 = 3.0;
const PLATE: [f64; 3] = [40.0, 30.0, 10.0];

/// The plate with `boss` fused on its top.
fn with_boss(topo: &mut Topology, boss: SolidId) -> SolidId {
    let plate = make_box(topo, PLATE[0], PLATE[1], PLATE[2]).unwrap();
    boolean(topo, BooleanOp::Fuse, plate, boss).unwrap()
}

/// The part's edges with both ends at height `z`, inside the plate's
/// outline by more than `margin`.
fn top_edges(topo: &Topology, part: SolidId, z: f64, margin: f64) -> Vec<EdgeId> {
    solid_edges(topo, part)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                (p.z() - z).abs() < 1e-9
                    && p.x() > margin
                    && p.x() < PLATE[0] - margin
                    && p.y() > margin
                    && p.y() < PLATE[1] - margin
            })
        })
        .collect()
}

/// Rounds `edges` by `R` and checks the result has `faces` faces and the
/// volume `truth`.
fn round(topo: &mut Topology, part: SolidId, edges: &[EdgeId], faces: usize, truth: f64) {
    let result = fillet_v2(topo, part, edges, R).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(solid_faces(topo, result.solid).unwrap().len(), faces);
    for deflection in [0.5, 0.05] {
        let mesh = tessellate_solid(topo, result.solid, deflection).unwrap();
        assert!(is_watertight(&mesh), "mesh not watertight at {deflection}");
    }
    // The corner rounds are walked bands, measured on their mesh: 1e-5
    // light at this deflection, closing on the truth as it shrinks.
    let vol = solid_volume(topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - truth).abs() < 3e-5 * truth,
        "volume {vol}, truth {truth}"
    );
}

#[test]
fn a_rounded_boss_one_radius_high_rounded_round_its_top() {
    let mut topo = Topology::new();
    let (dx, dy, rho) = (20.0, 14.0, 4.5);
    let boss = make_box(&mut topo, dx, dy, R).unwrap();
    let verticals: Vec<_> = solid_edges(&topo, boss)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let p = topo.vertex(edge.start()).unwrap().point();
            let q = topo.vertex(edge.end()).unwrap().point();
            (p.z() - q.z()).abs() > 1.0
        })
        .collect();
    assert_eq!(verticals.len(), 4);
    let boss = fillet_v2(&mut topo, boss, &verticals, rho).unwrap().solid;
    transform_solid(&mut topo, boss, &Mat4::translation(10.0, 8.0, PLATE[2])).unwrap();
    let part = with_boss(&mut topo, boss);
    let edges = top_edges(&topo, part, PLATE[2] + R, 1.0);
    assert_eq!(edges.len(), 8);
    // Along the straight sides the rounds take (1 - pi/4) r^2; round each
    // corner, that corner square less a quarter disc turned a quarter turn
    // about the corner's axis.
    let straight = 2.0 * (dx + dy - 4.0 * rho);
    let moment = (R * R).mul_add(
        rho - R / 2.0,
        -(PI * R * R / 4.0) * (rho - R + 4.0 * R / (3.0 * PI)),
    );
    let removed = ((1.0 - PI / 4.0) * R * R).mul_add(straight, 2.0 * PI * moment);
    let foot = (4.0 - PI).mul_add(-rho * rho, dx * dy);
    let truth = PLATE[0] * PLATE[1] * PLATE[2] + foot * R - removed;
    // The plate's six faces (its top holed by the boss's foot), the boss's
    // top, four straight rounds and four corner rounds: no walls.
    round(&mut topo, part, &edges, 6 + 1 + 8, truth);
}
