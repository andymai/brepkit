//! Two knuckles of one radius meeting end to end, each bored along its axis
//! by a keyhole (a rod with a triangular tail), the second's keyhole wider and
//! turned about the axis: a hinge's bin and lid knuckles with the lid swung
//! open. The shared end plane carries each knuckle's ring around its own
//! keyhole, so the two end faces split into pieces the same-domain pass
//! cannot pair, and the solids only touch there.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_face_from_wire;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const KNUCKLE_R: f64 = 1.8;
const KNUCKLE_LEN: f64 = 4.0;
const BIN_KEY: (f64, f64) = (0.925, 1.308);
const LID_KEY: (f64, f64) = (1.0, 1.414);

/// A rod of radius `r` along `x` from `x0` for `len`.
fn rod_x(topo: &mut Topology, r: f64, x0: f64, len: f64) -> SolidId {
    let rod = make_cylinder(topo, r, len).unwrap();
    let place = Mat4::translation(x0, 0.0, 0.0) * Mat4::rotation_y(FRAC_PI_2);
    transform_solid(topo, rod, &place).unwrap();
    rod
}

/// A keyhole of rod radius `r` along `x` from `x0` for `len`: the rod with a
/// tail whose base is the rod's horizontal diameter and whose tip is `tip`
/// above the axis, turned `turn` about the axis.
fn keyhole(topo: &mut Topology, (r, tip): (f64, f64), x0: f64, len: f64, turn: f64) -> SolidId {
    let rod = rod_x(topo, r, x0, len);
    let corners = [(-r, 0.0), (r, 0.0), (0.0, tip)];
    let vids: Vec<_> = corners
        .iter()
        .map(|&(y, z)| topo.add_vertex(Vertex::new(Point3::new(x0, y, z), 1e-7)))
        .collect();
    let edges: Vec<_> = (0..3)
        .map(|i| {
            let e = topo.add_edge(Edge::new(vids[i], vids[(i + 1) % 3], EdgeCurve::Line));
            OrientedEdge::new(e, true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = make_face_from_wire(topo, wire).unwrap();
    let tail = extrude(topo, face, Vec3::new(1.0, 0.0, 0.0), len).unwrap();
    let key = boolean(topo, BooleanOp::Fuse, rod, tail).unwrap();
    transform_solid(topo, key, &Mat4::rotation_x(turn)).unwrap();
    key
}

/// A knuckle along `x` from `x0`, bored by `key` turned `turn`; the keyhole
/// overshoots both ends so the cut leaves no sliver.
fn knuckle(topo: &mut Topology, key: (f64, f64), x0: f64, turn: f64) -> SolidId {
    let body = rod_x(topo, KNUCKLE_R, x0, KNUCKLE_LEN);
    let bore = keyhole(topo, key, x0 - 1.0, KNUCKLE_LEN + 2.0, turn);
    boolean(topo, BooleanOp::Cut, body, bore).unwrap()
}

/// The keyhole's area: the rod's disc and the part of the tail beyond it, a
/// triangle between the tip and the points where the tail's sides leave the
/// circle, less the circle's segment under that triangle.
fn keyhole_area((r, h): (f64, f64)) -> f64 {
    let s = 2.0 * r * r / (r * r + h * h);
    let tip = r * h * (1.0 - s).powi(2);
    let alpha = (1.0 - s).asin();
    let segment = r * r * (alpha - alpha.sin() * alpha.cos());
    PI * r * r + tip - segment
}

fn knuckle_volume(key: (f64, f64)) -> f64 {
    (PI * KNUCKLE_R * KNUCKLE_R - keyhole_area(key)) * KNUCKLE_LEN
}

fn poses() -> [Mat4; 2] {
    [
        Mat4::identity(),
        Mat4::rotation_z(0.7) * Mat4::rotation_y(0.4),
    ]
}

/// Runs `op` on the bin knuckle (`x` from 0) and the lid knuckle (`x` up to
/// 0, its keyhole turned `turn`) in every pose, and returns each result's
/// volume, after checking it is exact and valid.
fn volumes(op: BooleanOp, turn: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for (k, pose) in poses().iter().enumerate() {
        let mut topo = Topology::new();
        let bin = knuckle(&mut topo, BIN_KEY, 0.0, 0.0);
        let lid = knuckle(&mut topo, LID_KEY, -KNUCKLE_LEN, turn);
        transform_solid(&mut topo, bin, pose).unwrap();
        transform_solid(&mut topo, lid, pose).unwrap();
        let before = mesh_fallback_count();
        let result = boolean(&mut topo, op, bin, lid).unwrap();
        assert_eq!(
            mesh_fallback_count(),
            before,
            "{op:?} turn {turn} pose {k}: mesh fallback"
        );
        if topo.is_empty_solid(result) {
            out.push(0.0);
            continue;
        }
        let report = validate_solid(&topo, result).unwrap();
        assert!(
            report.is_valid(),
            "{op:?} turn {turn} pose {k}: {:?}",
            report.issues
        );
        out.push(solid_volume(&topo, result, 0.001).unwrap());
    }
    out
}

const TURNS: [f64; 3] = [0.35, 0.7, 1.05];

#[test]
fn keyholed_knuckles_end_to_end_share_nothing() {
    for turn in TURNS {
        for vol in volumes(BooleanOp::Intersect, turn) {
            assert!(vol.abs() < 1e-9, "turn {turn}: common volume {vol}");
        }
    }
}

#[test]
fn keyholed_knuckles_end_to_end_fuse_to_both() {
    let truth = knuckle_volume(BIN_KEY) + knuckle_volume(LID_KEY);
    for turn in TURNS {
        for vol in volumes(BooleanOp::Fuse, turn) {
            assert!(
                (vol - truth).abs() < 1e-6 * truth,
                "turn {turn}: volume {vol}, expected {truth}"
            );
        }
    }
}

#[test]
fn keyholed_knuckle_cut_by_its_neighbour_is_unchanged() {
    let truth = knuckle_volume(BIN_KEY);
    for turn in TURNS {
        for vol in volumes(BooleanOp::Cut, turn) {
            assert!(
                (vol - truth).abs() < 1e-6 * truth,
                "turn {turn}: volume {vol}, expected {truth}"
            );
        }
    }
}
