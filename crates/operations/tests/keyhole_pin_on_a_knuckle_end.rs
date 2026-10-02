//! A keyhole pin (a rod fused with a triangular tail) ending flat on a
//! knuckle's end disc, coaxial with it, as a hinge lid's short pin ends on its
//! first knuckle. The pin's cap meets the disc in a cluster of sections that
//! never reaches the disc's rim: the tail's two sides each run between the
//! ends of a rod-cap arc, and the cluster's outline is the hole of the ring
//! left around it. Every result is exact and valid, upright, turned and
//! mirrored, with the pin turned about its axis to eighths of a turn.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

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
const PIN_R: f64 = 0.925;
const TAIL_TIP: f64 = 1.308;
const PIN_LEN: f64 = 5.0;

/// A rod of radius `r` along `x` from `x0` for `len`.
fn rod_x(topo: &mut Topology, r: f64, x0: f64, len: f64) -> SolidId {
    let rod = make_cylinder(topo, r, len).unwrap();
    let place = Mat4::translation(x0, 0.0, 0.0) * Mat4::rotation_y(FRAC_PI_2);
    transform_solid(topo, rod, &place).unwrap();
    rod
}

/// The pin along `x` from `-PIN_LEN` to `0`: a rod with a tail whose base is
/// the rod's horizontal diameter and whose tip is at `z = TAIL_TIP`.
fn keyhole_pin(topo: &mut Topology) -> SolidId {
    let rod = rod_x(topo, PIN_R, -PIN_LEN, PIN_LEN);
    let corners = [(-PIN_R, 0.0), (PIN_R, 0.0), (0.0, TAIL_TIP)];
    let vids: Vec<_> = corners
        .iter()
        .map(|&(y, z)| topo.add_vertex(Vertex::new(Point3::new(-PIN_LEN, y, z), 1e-7)))
        .collect();
    let edges: Vec<_> = (0..3)
        .map(|i| {
            let e = topo.add_edge(Edge::new(vids[i], vids[(i + 1) % 3], EdgeCurve::Line));
            OrientedEdge::new(e, true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = make_face_from_wire(topo, wire).unwrap();
    let tail = extrude(topo, face, Vec3::new(1.0, 0.0, 0.0), PIN_LEN).unwrap();
    boolean(topo, BooleanOp::Fuse, rod, tail).unwrap()
}

/// The keyhole's area: the rod's disc and the part of the tail beyond it,
/// a triangle between the tip and the points where the tail's sides leave
/// the circle, less the circle's segment under that triangle.
fn keyhole_area() -> f64 {
    let (r, h) = (PIN_R, TAIL_TIP);
    let s = 2.0 * r * r / (r * r + h * h);
    let tip = r * h * (1.0 - s).powi(2);
    let alpha = (1.0 - s).asin();
    let segment = r * r * (alpha - alpha.sin() * alpha.cos());
    PI * r * r + tip - segment
}

fn poses() -> [Mat4; 3] {
    [
        Mat4::identity(),
        Mat4::rotation_z(0.7) * Mat4::rotation_y(0.4),
        Mat4::scale(1.0, -1.0, 1.0),
    ]
}

/// Runs `op` on the knuckle and the pin in every pose, the pin turned about
/// the shared axis, requiring an exact valid result of volume `expected`.
fn op_is_exact(op: BooleanOp, expected: f64) {
    for pose in poses() {
        for k in 0..8 {
            let mut topo = Topology::new();
            let knuckle = rod_x(&mut topo, KNUCKLE_R, 0.0, KNUCKLE_LEN);
            let pin = keyhole_pin(&mut topo);
            let spin = Mat4::rotation_x(f64::from(k) * FRAC_PI_4);
            transform_solid(&mut topo, pin, &(pose * spin)).unwrap();
            transform_solid(&mut topo, knuckle, &pose).unwrap();
            let before = mesh_fallback_count();
            let result = boolean(&mut topo, op, knuckle, pin).unwrap();
            assert_eq!(mesh_fallback_count(), before, "mesh fallback, spin {k}");
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "spin {k}: {:?}", report.issues);
            let vol = solid_volume(&topo, result, 0.001).unwrap();
            assert!(
                (vol - expected).abs() < 1e-6 * expected,
                "spin {k}: volume {vol}, expected {expected}"
            );
        }
    }
}

fn knuckle_volume() -> f64 {
    PI * KNUCKLE_R * KNUCKLE_R * KNUCKLE_LEN
}

/// The pin only touches the knuckle, so the cut leaves it whole. The disc's
/// greedy walk took the tail's side where it should have closed a rod-cap
/// segment on its chord, the arrangement rebuilt from the rim and sections
/// kept only the cluster's own pieces and dropped the ring, and the face
/// trace measured the disc's one-edge rim as enclosing nothing.
#[test]
fn a_keyhole_pin_ending_on_a_knuckle_is_cut_exactly() {
    op_is_exact(BooleanOp::Cut, knuckle_volume());
}

#[test]
fn a_keyhole_pin_ending_on_a_knuckle_fuses_exactly() {
    op_is_exact(BooleanOp::Fuse, knuckle_volume() + keyhole_area() * PIN_LEN);
}
