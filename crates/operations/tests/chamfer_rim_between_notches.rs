//! A pocket's rim edge chamfered between two notches cut off the rim, each
//! notch exactly as deep as the chamfer is wide and narrower than it: the
//! bevel ends on the notches' walls, its end triangles reaching past them.
//! The plate is drilled too, so the body has a curved edge. The chamfer
//! takes off its prism, half the distances' product along the edge.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::{chamfer, chamfer_two_on};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

/// The chamfer's width, and the notches' depth.
const D: f64 = 5.0;

fn cut_box(topo: &mut Topology, part: SolidId, at: [f64; 3], size: [f64; 3]) -> SolidId {
    let tool = make_box(topo, size[0], size[1], size[2]).unwrap();
    transform_solid(topo, tool, &Mat4::translation(at[0], at[1], at[2])).unwrap();
    boolean(topo, BooleanOp::Cut, part, tool).unwrap()
}

/// A 100 x 60 x 20 plate with a pocket 10 deep (x 10..90, y 10..40), two
/// notches `D` deep and 4 wide off its rim at y = 40 (x 10..20 and 60..70),
/// and a bore. Returns the plate and the rim edge between the notches.
fn notched_pocket(topo: &mut Topology) -> (SolidId, EdgeId) {
    let plate = make_box(topo, 100.0, 60.0, 20.0).unwrap();
    let plate = cut_box(topo, plate, [10.0, 10.0, 10.0], [80.0, 30.0, 11.0]);
    let plate = cut_box(topo, plate, [10.0, 39.0, 20.0 - D], [10.0, 5.0, D + 1.0]);
    let plate = cut_box(topo, plate, [60.0, 39.0, 20.0 - D], [10.0, 5.0, D + 1.0]);
    let drill = make_cylinder(topo, 3.0, 22.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(90.0, 50.0, -1.0)).unwrap();
    let plate = boolean(topo, BooleanOp::Cut, plate, drill).unwrap();
    let edge = solid_edges(topo, plate)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            let on = |p: brepkit_math::vec::Point3| {
                (p.y() - 40.0).abs() < 1e-9 && (p.z() - 20.0).abs() < 1e-9
            };
            on(a) && on(b) && ((a.x() - 20.0).abs() < 1e-9 || (b.x() - 20.0).abs() < 1e-9)
        })
        .expect("the rim edge between the notches");
    (plate, edge)
}

fn check(topo: &Topology, before: f64, result: SolidId, want: f64, label: &str) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{label}: mesh not watertight");
    let removed = before - solid_volume(topo, result, 0.01).unwrap();
    assert!(
        (removed - want).abs() < 1e-6 * want,
        "{label}: took off {removed}, want {want}"
    );
}

#[test]
fn a_rim_edge_between_notches_as_deep_as_the_chamfer_is_chamfered() {
    let mut topo = Topology::new();
    let (plate, edge) = notched_pocket(&mut topo);
    let before = solid_volume(&topo, plate, 0.01).unwrap();
    let result = chamfer(&mut topo, plate, &[edge], D).unwrap();
    check(&topo, before, result, 0.5 * D * D * 40.0, "equal");
}

#[test]
fn a_rim_edge_between_notches_takes_two_distances() {
    let mut topo = Topology::new();
    let (plate, edge) = notched_pocket(&mut topo);
    let before = solid_volume(&topo, plate, 0.01).unwrap();
    // 5 down the pocket's wall, 3 across the top.
    let faces = solid_faces(&topo, plate).unwrap();
    let adjacency = AdjacencyIndex::build_from_faces(&topo, &faces).unwrap();
    let wall: FaceId = adjacency
        .faces_for_edge(edge)
        .iter()
        .copied()
        .find(|&f| {
            matches!(topo.face(f).unwrap().surface(),
                FaceSurface::Plane { normal, .. } if normal.z().abs() < 0.5)
        })
        .unwrap();
    let result = chamfer_two_on(&mut topo, plate, &[(edge, wall)], D, 3.0).unwrap();
    check(&topo, before, result, 0.5 * D * 3.0 * 40.0, "two distances");
}
