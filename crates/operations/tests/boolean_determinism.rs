//! Booleans give the same result on every run: the same faces in the same
//! order, bounded by the same edges at the same bit-exact points.
//!
//! `std` hash maps seed every instance differently, so repeating an
//! operation in one process exercises the orders a later process would see.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const RUNS: usize = 6;

fn moved(topo: &mut Topology, s: SolidId, m: &Mat4) -> SolidId {
    transform_solid(topo, s, m).unwrap();
    s
}

fn fingerprint(topo: &Topology, solid: SolidId) -> Vec<String> {
    let mut out = Vec::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        out.push(format!(
            "face {} rev={}",
            face.surface().type_tag(),
            face.is_reversed()
        ));
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for wid in wires {
            out.push("  wire".to_owned());
            for oe in topo.wire(wid).unwrap().edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                let p = |v| {
                    let q = topo.vertex(v).unwrap().point();
                    format!(
                        "({:016x},{:016x},{:016x})",
                        q.x().to_bits(),
                        q.y().to_bits(),
                        q.z().to_bits()
                    )
                };
                out.push(format!(
                    "    {} fwd={} {} -> {}",
                    edge.curve().type_tag(),
                    oe.is_forward(),
                    p(edge.start()),
                    p(edge.end())
                ));
            }
        }
    }
    out
}

fn assert_repeatable(name: &str, build: impl Fn(&mut Topology) -> SolidId) {
    let mut first: Option<Vec<String>> = None;
    for run in 0..RUNS {
        let mut topo = Topology::default();
        let solid = build(&mut topo);
        let fp = fingerprint(&topo, solid);
        match &first {
            None => first = Some(fp),
            Some(want) => {
                let at = want
                    .iter()
                    .zip(&fp)
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| want.len().min(fp.len()));
                assert!(
                    *want == fp,
                    "{name}: run {run} differs from run 0 at line {at} \
                     ({} vs {} lines)\n  run 0: {:?}\n  run {run}: {:?}",
                    want.len(),
                    fp.len(),
                    want.get(at),
                    fp.get(at)
                );
            }
        }
    }
}

#[test]
fn box_fuse_offset_box() {
    assert_repeatable("box fuse box", |t| {
        let a = make_box(t, 2.0, 2.0, 2.0).unwrap();
        let b = make_box(t, 2.0, 2.0, 2.0).unwrap();
        let b = moved(t, b, &Mat4::translation(1.0, 0.7, 0.4));
        boolean(t, BooleanOp::Fuse, a, b).unwrap()
    });
}

#[test]
fn plate_cut_cylinders() {
    assert_repeatable("plate cut cylinders", |t| {
        let mut plate = make_box(t, 10.0, 6.0, 1.0).unwrap();
        for (x, y) in [(2.0, 2.0), (5.0, 3.0), (8.0, 4.0)] {
            let c = make_cylinder(t, 0.8, 3.0).unwrap();
            let c = moved(t, c, &Mat4::translation(x, y, -1.0));
            plate = boolean(t, BooleanOp::Cut, plate, c).unwrap();
        }
        plate
    });
}

#[test]
fn box_intersect_sphere() {
    assert_repeatable("box intersect sphere", |t| {
        let a = make_box(t, 2.0, 2.0, 2.0).unwrap();
        let s = make_sphere(t, 1.3, 32).unwrap();
        let s = moved(t, s, &Mat4::translation(1.0, 1.0, 1.0));
        boolean(t, BooleanOp::Intersect, a, s).unwrap()
    });
}

#[test]
fn crossed_cylinders_fuse() {
    assert_repeatable("crossed cylinders fuse", |t| {
        let a = make_cylinder(t, 1.0, 6.0).unwrap();
        let a = moved(t, a, &Mat4::translation(0.0, 0.0, -3.0));
        let b = make_cylinder(t, 0.7, 6.0).unwrap();
        let b = moved(
            t,
            b,
            &(Mat4::rotation_x(std::f64::consts::FRAC_PI_2) * Mat4::translation(0.0, 0.0, -3.0)),
        );
        boolean(t, BooleanOp::Fuse, a, b).unwrap()
    });
}

#[test]
fn box_cut_cone() {
    assert_repeatable("box cut cone", |t| {
        let a = make_box(t, 4.0, 4.0, 2.0).unwrap();
        let c = make_cone(t, 1.5, 0.5, 4.0).unwrap();
        let c = moved(t, c, &Mat4::translation(2.0, 2.0, -1.0));
        boolean(t, BooleanOp::Cut, a, c).unwrap()
    });
}

#[test]
fn torus_fuse_box() {
    assert_repeatable("torus fuse box", |t| {
        let a = make_torus(t, 3.0, 1.0, 32).unwrap();
        let b = make_box(t, 2.0, 2.0, 4.0).unwrap();
        let b = moved(t, b, &Mat4::translation(2.0, -1.0, -2.0));
        boolean(t, BooleanOp::Fuse, a, b).unwrap()
    });
}

#[test]
fn sphere_cut_sphere() {
    assert_repeatable("sphere cut sphere", |t| {
        let a = make_sphere(t, 2.0, 32).unwrap();
        let b = make_sphere(t, 1.5, 32).unwrap();
        let b = moved(t, b, &Mat4::translation(1.6, 0.3, 0.2));
        boolean(t, BooleanOp::Cut, a, b).unwrap()
    });
}

#[test]
fn coplanar_boxes_fuse() {
    assert_repeatable("coplanar boxes fuse", |t| {
        let mut acc = make_box(t, 2.0, 2.0, 1.0).unwrap();
        for (x, y) in [(2.0, 0.0), (1.0, 1.5), (3.0, 1.0), (0.5, -1.5)] {
            let b = make_box(t, 2.0, 2.0, 1.0).unwrap();
            let b = moved(t, b, &Mat4::translation(x, y, 0.0));
            acc = boolean(t, BooleanOp::Fuse, acc, b).unwrap();
        }
        acc
    });
}

#[test]
fn coaxial_cylinders_fuse_then_cut() {
    assert_repeatable("coaxial cylinders", |t| {
        let a = make_cylinder(t, 2.0, 3.0).unwrap();
        let b = make_cylinder(t, 2.0, 3.0).unwrap();
        let b = moved(t, b, &Mat4::translation(0.0, 0.0, 2.0));
        let ab = boolean(t, BooleanOp::Fuse, a, b).unwrap();
        let bore = make_cylinder(t, 1.0, 8.0).unwrap();
        let bore = moved(t, bore, &Mat4::translation(0.0, 0.0, -1.0));
        boolean(t, BooleanOp::Cut, ab, bore).unwrap()
    });
}
