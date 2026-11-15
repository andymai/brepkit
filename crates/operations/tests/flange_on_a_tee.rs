//! A pipe tee, a run with a branch standing on it, with a flange fused onto
//! the branch's far end. The run's wall holds the saddle where the branch
//! leaves it, one closed curve, and the flange meets no part of that wall.
//! The wall is then classified whole, from a point clear of the saddle (it
//! had no such point, and the fuse fell back to a mesh).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

const RUN_R: f64 = 18.0;
const RUN_LEN: f64 = 130.0;
const BRANCH_R: f64 = 15.0;
const BRANCH_H: f64 = 60.0;
const FLANGE_R: f64 = 26.0;
const FLANGE_T: f64 = 8.0;

fn exact() -> BooleanOptions {
    BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    }
}

/// The tee with its flange, the flange fused last or the branch fused last.
fn check(pose: &Mat4, flange_last: bool) {
    let label = format!("flange last {flange_last}");
    let mut topo = Topology::new();
    let run = make_cylinder(&mut topo, RUN_R, RUN_LEN).unwrap();
    let along_x = Mat4::translation(-RUN_LEN / 2.0, 0.0, 0.0) * Mat4::rotation_y(FRAC_PI_2);
    transform_solid(&mut topo, run, &along_x).unwrap();
    let branch = make_cylinder(&mut topo, BRANCH_R, BRANCH_H).unwrap();
    let flange = make_cylinder(&mut topo, FLANGE_R, FLANGE_T).unwrap();
    transform_solid(
        &mut topo,
        flange,
        &Mat4::translation(0.0, 0.0, BRANCH_H - FLANGE_T),
    )
    .unwrap();
    for solid in [run, branch, flange] {
        transform_solid(&mut topo, solid, pose).unwrap();
    }

    let fuse = |topo: &mut Topology, a, b| {
        boolean_with_options(topo, BooleanOp::Fuse, a, b, exact())
            .unwrap_or_else(|e| panic!("{label}: {e}"))
    };
    let tee = fuse(&mut topo, run, branch);
    let tee_volume = solid_volume(&topo, tee, 0.001).unwrap();
    let whole = if flange_last {
        fuse(&mut topo, tee, flange)
    } else {
        let run_and_flange = fuse(&mut topo, run, flange);
        fuse(&mut topo, run_and_flange, branch)
    };

    let report = validate_solid(&topo, whole).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(&topo, whole, 0.05).unwrap()),
        "{label}: mesh not watertight"
    );
    let faces = solid_faces(&topo, whole).unwrap();
    let count = |want: fn(&FaceSurface) -> bool| {
        faces
            .iter()
            .filter(|&&f| want(topo.face(f).unwrap().surface()))
            .count()
    };
    // The run, the branch and the flange's rim; the run's two ends, the
    // flange's ring below and its top with the branch's.
    assert_eq!(
        count(|s| matches!(s, FaceSurface::Cylinder(_))),
        3,
        "{label}: walls"
    );
    assert_eq!(
        count(|s| matches!(s, FaceSurface::Plane { .. })),
        faces.len() - 3,
        "{label}: every other face is flat"
    );
    assert!(faces.len() <= 8, "{label}: {} faces", faces.len());

    // The flange adds its ring round the branch.
    let want = tee_volume + PI * (FLANGE_R * FLANGE_R - BRANCH_R * BRANCH_R) * FLANGE_T;
    let volume = solid_volume(&topo, whole, 0.001).unwrap();
    assert!(
        (volume - want).abs() <= 1e-7 * want,
        "{label}: volume {volume}, want {want}"
    );

    let at = |x: f64, y: f64, z: f64| pose.mul_point(Point3::new(x, y, z));
    let tol = brepkit_math::tolerance::Tolerance::new();
    for (p, inside) in [
        (at(60.0, 0.0, 0.0), true),
        (at(0.0, 0.0, 40.0), true),
        (at(20.0, 0.0, 56.0), true),
        (at(20.0, 0.0, 45.0), false),
        (at(40.0, 0.0, 25.0), false),
        (at(0.0, 30.0, 56.0), false),
    ] {
        let class = classify_point(&topo, whole, p, 0.01, tol.linear).unwrap();
        let want = if inside {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        assert_eq!(class, want, "{label}: point {p:?}");
    }
}

#[test]
fn a_flange_fuses_onto_a_tees_branch() {
    check(&Mat4::identity(), true);
}

#[test]
fn the_branch_fused_last_gives_the_same_solid() {
    check(&Mat4::identity(), false);
}

#[test]
fn a_turned_tee_takes_its_flange() {
    let pose =
        Mat4::translation(12.5, -40.0, 7.25) * Mat4::rotation_z(0.6) * Mat4::rotation_x(-1.1);
    check(&pose, true);
    let mirrored = Mat4::scale(1.0, -1.0, 1.0) * Mat4::rotation_y(0.4);
    check(&mirrored, true);
}
