//! A ball cut by a plane that stays clear of its equator, level or tilted,
//! keeping either side. With `d` the centre's distance to the plane along the
//! kept piece's outward normal there, the piece holds
//! `pi (2 r^3 + 3 r^2 d - d^3) / 3`, its sphere faces `4 pi r^2 - 2 pi r (r - d)`
//! and its disc `pi (r^2 - d^2)`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

#[test]
fn ball_cut_by_a_plane_clear_of_its_equator() {
    let r = 3.0_f64;
    // The plane z = z0 + slope (x cos + y sin); each section circle stays in
    // one hemisphere of `make_sphere`'s chordal equator.
    for (z0, slope) in [
        (0.5, 0.0),
        (-1.0, 0.0),
        (2.0, 0.0),
        (-2.5, 0.0),
        (1.0, 0.2),
        (-1.5, 0.3),
        (0.5, 0.1),
        (-2.0, 0.4),
    ] {
        for turn_deg in [0.0_f64, 90.0, 200.0] {
            for keep_below in [true, false] {
                let label = format!(
                    "z0 {z0}, slope {slope}, turned {turn_deg} degrees, keeping {}",
                    if keep_below { "below" } else { "above" }
                );
                let turn = turn_deg.to_radians();
                let mut topo = Topology::new();
                let ball = make_sphere(&mut topo, r, 32).unwrap();
                let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
                let place = Mat4::rotation_z(turn)
                    * Mat4::translation(0.0, 0.0, z0)
                    * Mat4::rotation_y(-f64::atan(slope))
                    * Mat4::translation(-10.0, -10.0, 0.0);
                transform_solid(&mut topo, lid, &place).unwrap();
                let op = if keep_below {
                    BooleanOp::Cut
                } else {
                    BooleanOp::Intersect
                };
                let piece = boolean(&mut topo, op, ball, lid).unwrap();

                let report = validate_solid(&topo, piece).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let faces = solid_faces(&topo, piece).unwrap();
                let (mut discs, mut disc_area, mut sphere_area) = (0, 0.0, 0.0);
                for &f in &faces {
                    let area = face_area(&topo, f, 0.01).unwrap();
                    match topo.face(f).unwrap().surface() {
                        FaceSurface::Plane { .. } => {
                            discs += 1;
                            disc_area += area;
                        }
                        FaceSurface::Sphere(_) => sphere_area += area,
                        other => panic!("{label}: a {} face", other.type_tag()),
                    }
                }
                assert_eq!(discs, 1, "{label}: plane faces");

                let above = z0 / slope.hypot(1.0);
                let d = if keep_below { above } else { -above };
                let truth = PI * (2.0 * r.powi(3) + 3.0 * r * r * d - d.powi(3)) / 3.0;
                let volume = solid_volume(&topo, piece, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
                let disc_truth = PI * (r * r - d * d);
                assert!(
                    (disc_area - disc_truth).abs() < 1e-9 * disc_truth,
                    "{label}: disc area {disc_area}, truth {disc_truth}"
                );
                let sphere_truth = 4.0 * PI * r * r - 2.0 * PI * r * (r - d);
                assert!(
                    (sphere_area - sphere_truth).abs() < 1e-9 * sphere_truth,
                    "{label}: sphere area {sphere_area}, truth {sphere_truth}"
                );

                let (s, c) = turn.sin_cos();
                let plane_z = |x: f64, y: f64| z0 + slope * (x * c + y * s);
                let (below, over) = if keep_below {
                    (PointClassification::Inside, PointClassification::Outside)
                } else {
                    (PointClassification::Outside, PointClassification::Inside)
                };
                let classify = |x: f64, y: f64, z: f64| {
                    classify_point(
                        &topo,
                        piece,
                        Point3::new(x, y, z),
                        &ClassifyOptions::default(),
                    )
                    .unwrap()
                };
                for (x, y) in [(0.0, 0.0), (0.5 * c, 0.5 * s), (-0.5 * s, 0.5 * c)] {
                    let z = plane_z(x, y);
                    assert_eq!(classify(x, y, z - 0.05), below, "{label}: below ({x}, {y})");
                    assert_eq!(classify(x, y, z + 0.05), over, "{label}: above ({x}, {y})");
                }

                let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let meshed = oriented_solid_volume(&topo, piece, 0.01).unwrap();
                assert!(
                    meshed <= truth && truth - meshed < 1e-2 * truth,
                    "{label}: mesh volume {meshed}, truth {truth}"
                );
            }
        }
    }
}

/// Two caps of one ball fused into one solid: every disc is still a full
/// circle on the sphere, but the caps each removes overlap.
#[test]
fn two_caps_of_one_ball_keep_both_volumes() {
    let r = 3.0_f64;
    let cap = |topo: &mut Topology, z0: f64, keep_below: bool| {
        let ball = make_sphere(topo, r, 32).unwrap();
        let lid = make_box(topo, 20.0, 20.0, 20.0).unwrap();
        transform_solid(topo, lid, &Mat4::translation(-10.0, -10.0, z0)).unwrap();
        let op = if keep_below {
            BooleanOp::Cut
        } else {
            BooleanOp::Intersect
        };
        boolean(topo, op, ball, lid).unwrap()
    };
    let mut topo = Topology::new();
    let top = cap(&mut topo, 2.0, false);
    let bottom = cap(&mut topo, -2.5, true);
    let both = boolean(&mut topo, BooleanOp::Fuse, top, bottom).unwrap();

    // Caps 1 and 0.5 high.
    let truth = PI * (1.0 * (3.0 * r - 1.0) + 0.25 * (3.0 * r - 0.5)) / 3.0;
    let volume = solid_volume(&topo, both, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-2 * truth,
        "volume {volume}, truth {truth}"
    );
}

/// A shell turned inside out still bounds the same piece of the ball.
#[test]
fn inverted_ball_piece_keeps_its_volume() {
    let r = 3.0_f64;
    for keep_below in [true, false] {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, r, 32).unwrap();
        let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        transform_solid(&mut topo, lid, &Mat4::translation(-10.0, -10.0, 1.0)).unwrap();
        let op = if keep_below {
            BooleanOp::Cut
        } else {
            BooleanOp::Intersect
        };
        let piece = boolean(&mut topo, op, ball, lid).unwrap();
        let before = solid_volume(&topo, piece, 0.01).unwrap();
        for f in solid_faces(&topo, piece).unwrap() {
            let face = topo.face_mut(f).unwrap();
            let flipped = !face.is_reversed();
            face.set_reversed(flipped);
        }
        let after = solid_volume(&topo, piece, 0.01).unwrap();
        assert!(
            (after - before).abs() < 1e-9 * before,
            "keeping {}: volume {after} after inverting, {before} before",
            if keep_below { "below" } else { "above" }
        );
    }
}
