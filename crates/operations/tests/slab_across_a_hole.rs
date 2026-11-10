//! A plate with a round through hole, fused with a slab or a box whose flat
//! face runs across the hole. The slab standing on the plate has its bottom
//! edges cut by the hole's rim into chords; the slab sunk into the plate, and
//! a box whose top crosses a hole through the plate's side, are split along
//! sections that run across the hole. Each chord shares both ends with the
//! rim's arc over the hole when the hole's seam lies on the far side, and stays
//! an edge of its own: every fuse is exact, valid, watertight and at its
//! closed-form volume, with the open part of the hole outside it, in every
//! pose and wherever the hole's seam lies.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const HOLE_R: f64 = 5.0;
const SLAB_W: f64 = 3.0;
const SLAB_D: f64 = 30.0;
const SLAB_H: f64 = 20.0;
const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

fn centred_box(topo: &mut Topology, size: (f64, f64, f64), c: (f64, f64, f64)) -> SolidId {
    let b = make_box(topo, size.0, size.1, size.2).unwrap();
    let to = Mat4::translation(c.0 - size.0 / 2.0, c.1 - size.1 / 2.0, c.2 - size.2 / 2.0);
    transform_solid(topo, b, &to).unwrap();
    b
}

fn posed(topo: &mut Topology, s: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            transform_solid(topo, s, &turn()).unwrap();
            s
        }
        "mirrored" => mirror(
            topo,
            s,
            Point3::new(0.3, 0.0, 0.0),
            Vec3::new(1.0, 0.3, 0.0),
        )
        .unwrap(),
        _ => s,
    }
}

fn turn() -> Mat4 {
    Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6)
}

/// Where `p` lands in `pose`.
fn place(pose: &str, p: Point3) -> Point3 {
    match pose {
        "turned" => turn().mul_point(p),
        "mirrored" => {
            let n = Vec3::new(1.0, 0.3, 0.0).normalize().unwrap();
            let off = (p - Point3::new(0.3, 0.0, 0.0)).dot(n);
            p - n * (2.0 * off)
        }
        _ => p,
    }
}

/// Exact (no mesh fallback since `before`, no NURBS face, the hole's wall a
/// cylinder), valid, watertight, holding `want`, with each of `outside` out
/// of it and each of `inside` in it (upright coordinates).
#[allow(clippy::too_many_arguments)]
fn check(
    topo: &Topology,
    fused: SolidId,
    before: u64,
    want: f64,
    pose: &str,
    outside: &[Point3],
    inside: &[Point3],
    case: &str,
) {
    assert_eq!(mesh_fallback_count(), before, "{case}: mesh fallback");
    let report = validate_solid(topo, fused).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let surfaces: Vec<FaceSurface> = solid_faces(topo, fused)
        .unwrap()
        .iter()
        .map(|&f| topo.face(f).unwrap().surface().clone())
        .collect();
    assert!(
        !surfaces.iter().any(|s| matches!(s, FaceSurface::Nurbs(_))),
        "{case}: a NURBS face"
    );
    assert!(
        surfaces
            .iter()
            .any(|s| matches!(s, FaceSurface::Cylinder(_))),
        "{case}: the hole lost its wall"
    );
    // Finer than the shallowest arc's 0.01 sag over its chord: at that
    // deflection the hole's wall above the box meshes onto the chord.
    let mesh = tessellate_solid(topo, fused, 0.002).unwrap();
    assert!(is_watertight(&mesh), "{case}: the mesh leaks");
    let volume = solid_volume(topo, fused, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "{case}: volume {volume}, want {want}"
    );
    for (points, class) in [
        (outside, PointClassification::Outside),
        (inside, PointClassification::Inside),
    ] {
        for &p in points {
            assert_eq!(
                classify_point(topo, fused, place(pose, p), 0.0, 1e-7).unwrap(),
                class,
                "{case}: {p:?}"
            );
        }
    }
}

/// The plate x -30..30, y -20..20, z 0..5, holed about (20, 0) with its seam
/// turned `spin` degrees, fused with the slab x 18.5..21.5, y -15..15 from
/// `sink` below the plate's top to 20 above it.
fn fuse_slab(pose: &str, spin: f64, sink: f64) {
    let case = format!("{pose}, sunk {sink}, hole seam turned {spin}");
    let mut topo = Topology::new();
    let plate = centred_box(&mut topo, (60.0, 40.0, 5.0), (0.0, 0.0, 2.5));
    let tool = make_cylinder(&mut topo, HOLE_R, 7.0).unwrap();
    let m = Mat4::translation(20.0, 0.0, -1.0) * Mat4::rotation_z(spin.to_radians());
    transform_solid(&mut topo, tool, &m).unwrap();
    let slab = centred_box(
        &mut topo,
        (SLAB_W, SLAB_D, SLAB_H + sink),
        (20.0, 0.0, 5.0 + (SLAB_H - sink) / 2.0),
    );
    let plate = posed(&mut topo, plate, pose);
    let tool = posed(&mut topo, tool, pose);
    let slab = posed(&mut topo, slab, pose);

    let holed = boolean(&mut topo, BooleanOp::Cut, plate, tool).unwrap();
    let before = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, holed, slab).unwrap();

    // The slab's footprint over the hole: the disc between x = +-1.5 of its
    // centre.
    let w = SLAB_W / 2.0;
    let over_hole =
        2.0 * (w * (HOLE_R * HOLE_R - w * w).sqrt() + HOLE_R * HOLE_R * (w / HOLE_R).asin());
    let want = 60.0 * 40.0 * 5.0 - PI * HOLE_R * HOLE_R * 5.0
        + SLAB_W * SLAB_D * SLAB_H
        + sink * over_hole;
    let below = 5.0 - sink;
    check(
        &topo,
        fused,
        before,
        want,
        pose,
        &[
            Point3::new(16.0, 0.0, 2.5),
            Point3::new(20.0, 0.0, below - 0.25),
            Point3::new(16.0, 0.0, 5.5),
        ],
        &[
            Point3::new(20.0, 0.0, 5.5),
            Point3::new(20.0, 0.0, below + 0.25),
            Point3::new(10.0, 0.0, 2.5),
        ],
        &case,
    );
}

#[test]
fn a_slab_across_a_through_hole_fuses_exactly() {
    for pose in POSES {
        for spin in [0.0_f64, 45.0, 90.0, 180.0, 270.0] {
            fuse_slab(pose, spin, 0.0);
        }
    }
}

#[test]
fn a_slab_sunk_into_a_plate_across_a_through_hole_fuses_exactly() {
    for pose in POSES {
        for spin in [0.0_f64, 45.0, 90.0, 180.0, 270.0] {
            fuse_slab(pose, spin, 0.5);
        }
    }
}

/// The plate x -10..10, y 0..3, z 0..20, holed along y about z = 10 with
/// radius 1.7, the hole's seam turned `spin` degrees from its bottom, fused
/// with the box x -4..4, y -6..2, z -5..`top`: the box's top crosses the
/// hole, and its strip inside the hole and the hole's wall above it stay.
fn fuse_box_across(pose: &str, spin: f64, top: f64) {
    let case = format!("{pose}, top {top}, hole seam turned {spin}");
    let r = 1.7_f64;
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 20.0, 3.0, 20.0).unwrap();
    transform_solid(&mut topo, plate, &Mat4::translation(-10.0, 0.0, 0.0)).unwrap();
    let pin = make_cylinder(&mut topo, r, 5.0).unwrap();
    for m in [
        Mat4::rotation_z(FRAC_PI_2 + spin.to_radians()),
        Mat4::rotation_x(-FRAC_PI_2),
        Mat4::translation(0.0, -1.0, 10.0),
    ] {
        transform_solid(&mut topo, pin, &m).unwrap();
    }
    let block = make_box(&mut topo, 8.0, 8.0, top + 5.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-4.0, -6.0, -5.0)).unwrap();
    let plate = posed(&mut topo, plate, pose);
    let pin = posed(&mut topo, pin, pose);
    let block = posed(&mut topo, block, pose);

    let holed = boolean(&mut topo, BooleanOp::Cut, plate, pin).unwrap();
    let before = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, holed, block).unwrap();

    // The hole's part below the box's top, a chord `d` off its axis, 2 deep.
    let d = top - 10.0;
    let below = r * r * (-d / r).acos() + d * (r * r - d * d).sqrt();
    let want = 20.0 * 3.0 * 20.0 - PI * r * r * 3.0 + 8.0 * 8.0 * (top + 5.0)
        - (8.0 * 2.0 * top - 2.0 * below);
    check(
        &topo,
        fused,
        before,
        want,
        pose,
        &[
            Point3::new(0.0, 1.0, 0.5 * (top + 10.0 + r)),
            Point3::new(0.0, 2.5, 10.0),
        ],
        &[
            Point3::new(0.3, 1.0, top - 0.1),
            Point3::new(3.0, 1.0, top + 0.1),
        ],
        &case,
    );
}

#[test]
fn a_box_whose_top_crosses_a_hole_fuses_exactly() {
    for pose in POSES {
        // At 11.69 the top runs 0.01 below the rim, its run's middle between
        // two of a 16-gon's corners with the seam turned 11.25 degrees.
        for spin in [0.0_f64, 11.25, 90.0, 180.0, 270.0] {
            for top in [9.0, 11.5, 11.69] {
                fuse_box_across(pose, spin, top);
            }
        }
    }
}
