//! A fan frame's strut under the window of its lead slot, built as the
//! brepjs playground example `axial-fan` builds it: a duct wall of radius 38
//! with a slot cut through it from z = 3 to 4.5, and a strut whose top at
//! z = 3 is the slot's floor, running into the wall under the slot. The
//! strut's footprint on the bore meets the window along the window's
//! bottom edge.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_4;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
use brepkit_topology::solid::SolidId;

const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

fn turn() -> Mat4 {
    Mat4::translation(5.0, -4.0, 3.0) * Mat4::rotation_x(0.43) * Mat4::rotation_z(0.71)
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(-10.0, 3.0, 1.0), Vec3::new(0.2, 1.0, 0.4))
}

fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            transform_solid(topo, solid, &turn()).unwrap();
            solid
        }
        "mirrored" => {
            let (at, normal) = mirror_plane();
            mirror(topo, solid, at, normal).unwrap()
        }
        _ => solid,
    }
}

fn placed(p: Point3, pose: &str) -> Point3 {
    match pose {
        "turned" => turn().mul_point(p),
        "mirrored" => {
            let (at, normal) = mirror_plane();
            let unit = normal.normalize().unwrap();
            p - unit * (2.0 * (p - at).dot(unit))
        }
        _ => p,
    }
}

fn exact() -> BooleanOptions {
    BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    }
}

/// A box of `size` centred on `at`, turned -45 degrees about z.
fn turned_box(topo: &mut Topology, size: [f64; 3], at: [f64; 3]) -> SolidId {
    let b = make_box(topo, size[0], size[1], size[2]).unwrap();
    let place = Mat4::rotation_z(-FRAC_PI_4)
        * Mat4::translation(
            at[0] - size[0] / 2.0,
            at[1] - size[1] / 2.0,
            at[2] - size[2] / 2.0,
        );
    transform_solid(topo, b, &place).unwrap();
    b
}

/// The duct wall from radius 38 to 42 and z = 1 to 23, less the lead
/// slot: 3.2 wide, from z = 3 to 4.5, through the wall.
fn wall(topo: &mut Topology) -> SolidId {
    let outer = make_cylinder(topo, 42.0, 22.0).unwrap();
    transform_solid(topo, outer, &Mat4::translation(0.0, 0.0, 1.0)).unwrap();
    let bore = make_cylinder(topo, 38.0, 24.0).unwrap();
    let tube = boolean_with_options(topo, BooleanOp::Cut, outer, bore, exact()).unwrap();
    let slot = turned_box(topo, [30.0, 3.2, 1.5], [51.0, -6.0, 3.75]);
    boolean_with_options(topo, BooleanOp::Cut, tube, slot, exact()).unwrap()
}

/// [`wall`] with the slot's sides leaning over by 1 across its height, so
/// that they cut the bore in ellipses and the window's sides are curved in
/// the bore's parameters.
fn wall_with_leaning_slot(topo: &mut Topology) -> SolidId {
    let outer = make_cylinder(topo, 42.0, 22.0).unwrap();
    transform_solid(topo, outer, &Mat4::translation(0.0, 0.0, 1.0)).unwrap();
    let bore = make_cylinder(topo, 38.0, 24.0).unwrap();
    let tube = boolean_with_options(topo, BooleanOp::Cut, outer, bore, exact()).unwrap();
    let section: Vec<Point3> = [(-7.6, 3.0), (-4.4, 3.0), (-3.4, 4.5), (-6.6, 4.5)]
        .iter()
        .map(|&(y, z)| Point3::new(36.0, y, z))
        .collect();
    let wire = make_polygon_wire(topo, &section, 1e-7).unwrap();
    let face = make_planar_face_from_wire(topo, wire).unwrap();
    let slot = extrude(topo, face, Vec3::new(1.0, 0.0, 0.0), 30.0).unwrap();
    transform_solid(topo, slot, &Mat4::rotation_z(-FRAC_PI_4)).unwrap();
    boolean_with_options(topo, BooleanOp::Cut, tube, slot, exact()).unwrap()
}

fn holds(topo: &Topology, solid: SolidId, p: Point3) -> Option<bool> {
    match classify_point(topo, solid, p, 0.01, 1e-7).unwrap() {
        PointClassification::Inside => Some(true),
        PointClassification::Outside => Some(false),
        PointClassification::OnBoundary => None,
    }
}

fn strut_fuses_exactly(build: fn(&mut Topology) -> SolidId) {
    for pose in POSES {
        let mut topo = Topology::new();
        let body = build(&mut topo);
        let strut = turned_box(&mut topo, [28.9, 4.4, 3.0], [24.45, -6.0, 1.5]);
        let (a, b) = (posed(&mut topo, body, pose), posed(&mut topo, strut, pose));
        let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, exact())
            .map_err(|e| format!("{pose}: {e}"))
            .unwrap();
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{pose}: invalid: {report:?}");
        let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{pose}: open or non-manifold mesh");
        let common = boolean_with_options(&mut topo, BooleanOp::Intersect, a, b, exact())
            .map_err(|e| format!("{pose}: {e}"))
            .unwrap();
        let volume = |s| solid_volume(&topo, s, 0.01).unwrap();
        let (va, vb, vc, vf) = (volume(a), volume(b), volume(common), volume(fused));
        assert!(
            (vf - (va + vb - vc)).abs() < 1e-6 * vf,
            "{pose}: fused {vf}, the wall {va} and the strut {vb} less {vc}"
        );
        // Round where the strut meets the bore under the window.
        let centre = Mat4::rotation_z(-FRAC_PI_4).mul_point(Point3::new(38.0, -6.0, 3.0));
        let n = 16;
        let mut read = 0;
        #[allow(clippy::cast_precision_loss)]
        let at = |s: usize| (s as f64 + 0.5) / n as f64 - 0.5;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let p = placed(
                        Point3::new(
                            centre.x() + 6.0 * at(i),
                            centre.y() + 6.0 * at(j),
                            centre.z() + 5.0 * at(k),
                        ),
                        pose,
                    );
                    let (Some(f), Some(ia), Some(ib)) = (
                        holds(&topo, fused, p),
                        holds(&topo, a, p),
                        holds(&topo, b, p),
                    ) else {
                        continue;
                    };
                    assert_eq!(f, ia || ib, "{pose}: at {p:?}");
                    read += 1;
                }
            }
        }
        assert!(read > n * n * n / 2, "{pose}: only {read} points read");
    }
}

/// The wide strut, 4.4 wide from z = 0 to 3, fused into the wall under the
/// slot, is exact, valid and watertight in every pose. Its volume is the
/// two operands' less their common part, and over a grid round the slot's
/// mouth it holds a point where either operand does.
#[test]
fn a_strut_fuses_into_a_wall_under_its_slot() {
    strut_fuses_exactly(wall);
}

/// [`a_strut_fuses_into_a_wall_under_its_slot`] under a slot whose sides
/// lean: the window's sides are curved in the bore's parameters, and the
/// window is woven into the bore a period away from where its own loop
/// lies.
#[test]
fn a_strut_fuses_into_a_wall_under_a_leaning_slot() {
    strut_fuses_exactly(wall_with_leaning_slot);
}
