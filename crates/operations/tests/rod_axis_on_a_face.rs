//! A rod whose axis lies on the other solid's boundary: a second rod's wall,
//! or a box side. The sections halve the rod's wall along rulings that end
//! on a closed rim, and each half had to be traced in the same turn of `u`
//! as the rims it shares with the caps. Every result is exact, valid and
//! measures its closed form, upright, turned and mirrored, whichever way the
//! rod's seam points.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_4, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// A rod of radius `r` and height `h` along `z`, its base centred at
/// `(x, y, z)`, turned `spin` about its axis.
fn rod(topo: &mut Topology, r: f64, h: f64, (x, y, z): (f64, f64, f64), spin: f64) -> SolidId {
    let rod = make_cylinder(topo, r, h).unwrap();
    let place = Mat4::translation(x, y, z) * Mat4::rotation_z(spin);
    transform_solid(topo, rod, &place).unwrap();
    rod
}

/// Upright, turned about two axes, and mirrored through `x = 0`, each with
/// the rod turned about its own axis to eighths of a turn.
fn poses_and_spins() -> impl Iterator<Item = (Mat4, f64)> {
    [
        Mat4::identity(),
        Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4),
        Mat4::scale(-1.0, 1.0, 1.0),
    ]
    .into_iter()
    .flat_map(|pose| (0..8).map(move |k| (pose, f64::from(k) * FRAC_PI_4)))
}

/// `op` on `a` and `b` in `pose`, requiring an exact valid result of plane
/// and cylinder faces holding `expected`.
fn is_exact(
    topo: &mut Topology,
    op: BooleanOp,
    (a, b): (SolidId, SolidId),
    pose: &Mat4,
    expected: f64,
    label: &str,
) {
    transform_solid(topo, a, pose).unwrap();
    transform_solid(topo, b, pose).unwrap();
    let before = mesh_fallback_count();
    let result = boolean(topo, op, a, b).unwrap();
    assert_eq!(mesh_fallback_count(), before, "{label}: mesh fallback");
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    for f in solid_faces(topo, result).unwrap() {
        let surface = topo.face(f).unwrap().surface();
        assert!(
            matches!(
                surface,
                FaceSurface::Plane { .. } | FaceSurface::Cylinder(_)
            ),
            "{label}: a {} face",
            surface.type_tag()
        );
    }
    let vol = solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (vol - expected).abs() < 1e-6 * expected,
        "{label}: volume {vol}, expected {expected}"
    );
}

/// Two rods of radius 5 and height 10 on one base, their axes 5 apart, so
/// each axis lies on the other's wall and the caps are coplanar.
#[test]
fn rods_through_each_others_axes_fuse_exactly() {
    let (r, d, h) = (5.0_f64, 5.0_f64, 10.0);
    let lens = 2.0 * r * r * (d / (2.0 * r)).acos() - d / 2.0 * (4.0 * r * r - d * d).sqrt();
    let expected = h * (2.0 * PI * r * r - lens);
    for (pose, spin) in poses_and_spins() {
        let label = format!("spin {spin}");
        let mut topo = Topology::new();
        let a = rod(&mut topo, r, h, (d, 0.0, 0.0), spin);
        let b = rod(&mut topo, r, h, (0.0, 0.0, 0.0), spin);
        is_exact(&mut topo, BooleanOp::Fuse, (a, b), &pose, expected, &label);
    }
}

/// A rod of radius 12 standing on a box's base plane with its axis in the
/// box's side `x = 40`, twice the box's height.
#[test]
fn rod_on_a_box_side_fuses_exactly() {
    let (r, h, box_h) = (12.0_f64, 40.0, 20.0);
    let expected = (40.0_f64 * 40.0).mul_add(box_h, PI * r * r * (h - box_h / 2.0));
    for (pose, spin) in poses_and_spins() {
        let label = format!("spin {spin}");
        let mut topo = Topology::new();
        let base = make_box(&mut topo, 40.0, 40.0, box_h).unwrap();
        let tool = rod(&mut topo, r, h, (40.0, 20.0, 0.0), spin);
        is_exact(
            &mut topo,
            BooleanOp::Fuse,
            (base, tool),
            &pose,
            expected,
            &label,
        );
    }
}

/// A rod of radius 2.5 with its axis in the side `x = 40` of a 40 x 30 x 20
/// box, from mid height up past the top, cut from the box.
#[test]
fn rod_on_a_box_side_through_its_top_cuts_exactly() {
    let (r, h) = (2.5_f64, 12.5);
    let expected = (PI * r * r / 2.0).mul_add(-10.0, 40.0 * 30.0 * 20.0);
    for (pose, spin) in poses_and_spins() {
        let label = format!("spin {spin}");
        let mut topo = Topology::new();
        let base = make_box(&mut topo, 40.0, 30.0, 20.0).unwrap();
        let tool = rod(&mut topo, r, h, (40.0, 15.0, 10.0), spin);
        is_exact(
            &mut topo,
            BooleanOp::Cut,
            (base, tool),
            &pose,
            expected,
            &label,
        );
    }
}
