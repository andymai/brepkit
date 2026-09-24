//! A torus cut by a plane across its axis or through it, keeping either
//! side. Across the axis, with the plane `c` above the centre and
//! `a = asin(c / r)` its angle up the tube, Pappus gives the part below as
//! `2 pi R` times the tube disc's area below the chord at `c`, its torus face
//! covers `2 pi r R (pi + 2a)` and the plane face is the annulus
//! `4 pi R sqrt(r^2 - c^2)`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

#[test]
fn torus_cut_across_its_axis() {
    let (big, small) = (4.0_f64, 1.5_f64);
    // The whole scene is also tipped over, so the torus's frame is off the axes.
    let tips = [
        Mat4::identity(),
        Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3),
    ];
    for c in [0.0_f64, 0.5, -1.0, 1.2] {
        for (k, tip) in tips.iter().enumerate() {
            for keep_below in [true, false] {
                let label = format!(
                    "height {c}, tip {k}, keeping {}",
                    if keep_below { "below" } else { "above" }
                );
                let mut topo = Topology::new();
                let torus = make_torus(&mut topo, big, small, 32).unwrap();
                let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
                transform_solid(&mut topo, lid, &Mat4::translation(-10.0, -10.0, c)).unwrap();
                transform_solid(&mut topo, torus, tip).unwrap();
                transform_solid(&mut topo, lid, tip).unwrap();
                let op = if keep_below {
                    BooleanOp::Cut
                } else {
                    BooleanOp::Intersect
                };
                let piece = boolean(&mut topo, op, torus, lid).unwrap();

                let report = validate_solid(&topo, piece).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let faces = solid_faces(&topo, piece).unwrap();
                assert_eq!(faces.len(), 2, "{label}: faces");

                let chord = small.mul_add(small, -(c * c)).sqrt();
                let above = small * small * (c / small).acos() - c * chord;
                let slice = if keep_below {
                    PI * small * small - above
                } else {
                    above
                };
                let truth = 2.0 * PI * big * slice;
                let volume = solid_volume(&topo, piece, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
                let up = (c / small).asin();
                let band = if keep_below {
                    PI + 2.0 * up
                } else {
                    PI - 2.0 * up
                };
                for &f in &faces {
                    let area = face_area(&topo, f, 0.01).unwrap();
                    let truth = match topo.face(f).unwrap().surface() {
                        FaceSurface::Torus(_) => 2.0 * PI * small * big * band,
                        FaceSurface::Plane { .. } => 4.0 * PI * big * chord,
                        other => panic!("{label}: a {} face", other.type_tag()),
                    };
                    assert!(
                        (area - truth).abs() < 1e-9 * truth,
                        "{label}: {} area {area}, truth {truth}",
                        topo.face(f).unwrap().surface().type_tag()
                    );
                }

                let at = |x: f64, y: f64, z: f64| {
                    classify_point(
                        &topo,
                        piece,
                        tip.mul_point(Point3::new(x, y, z)),
                        &ClassifyOptions::default(),
                    )
                    .unwrap()
                };
                let (below, over) = if keep_below {
                    (PointClassification::Inside, PointClassification::Outside)
                } else {
                    (PointClassification::Outside, PointClassification::Inside)
                };
                for (x, y) in [(big, 0.0), (0.0, -big), (-big * 0.6, big * 0.8)] {
                    assert_eq!(at(x, y, c - 0.1), below, "{label}: below at ({x}, {y})");
                    assert_eq!(at(x, y, c + 0.1), over, "{label}: above at ({x}, {y})");
                }
                assert_eq!(
                    at(0.0, 0.0, c),
                    PointClassification::Outside,
                    "{label}: the hole"
                );

                let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let meshed = oriented_solid_volume(&topo, piece, 0.01).unwrap();
                assert!(
                    (meshed - truth).abs() < 1e-2 * truth,
                    "{label}: mesh volume {meshed}, truth {truth}"
                );
            }
        }
    }
}

/// A torus halved by a plane through its axis, turned about the axis: each
/// half holds `pi^2 R r^2`, its torus face `2 pi^2 R r`, and two tube
/// cross-sections `pi r^2` each.
#[test]
fn torus_halved_through_its_axis() {
    let (big, small) = (4.0_f64, 1.5_f64);
    for turn_deg in [0.0_f64, 60.0, 200.0] {
        for keep_behind in [true, false] {
            let label = format!(
                "turned {turn_deg} degrees, keeping {}",
                if keep_behind { "behind" } else { "ahead" }
            );
            let turn = Mat4::rotation_z(turn_deg.to_radians());
            let mut topo = Topology::new();
            let torus = make_torus(&mut topo, big, small, 32).unwrap();
            // The box fills x > 0 before the turn.
            let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
            let place = turn
                * Mat4::rotation_y(std::f64::consts::FRAC_PI_2)
                * Mat4::translation(-10.0, -10.0, 0.0);
            transform_solid(&mut topo, lid, &place).unwrap();
            let op = if keep_behind {
                BooleanOp::Cut
            } else {
                BooleanOp::Intersect
            };
            let piece = boolean(&mut topo, op, torus, lid).unwrap();

            let report = validate_solid(&topo, piece).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let faces = solid_faces(&topo, piece).unwrap();
            assert_eq!(faces.len(), 3, "{label}: faces");

            let truth = PI * PI * big * small * small;
            let volume = solid_volume(&topo, piece, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-9 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            for &f in &faces {
                let area = face_area(&topo, f, 0.01).unwrap();
                let truth = match topo.face(f).unwrap().surface() {
                    FaceSurface::Torus(_) => 2.0 * PI * PI * big * small,
                    FaceSurface::Plane { .. } => PI * small * small,
                    other => panic!("{label}: a {} face", other.type_tag()),
                };
                assert!(
                    (area - truth).abs() < 1e-9 * truth,
                    "{label}: {} area {area}, truth {truth}",
                    topo.face(f).unwrap().surface().type_tag()
                );
            }

            let at = |x: f64, y: f64, z: f64| {
                classify_point(
                    &topo,
                    piece,
                    turn.mul_point(Point3::new(x, y, z)),
                    &ClassifyOptions::default(),
                )
                .unwrap()
            };
            let (behind, ahead) = if keep_behind {
                (PointClassification::Inside, PointClassification::Outside)
            } else {
                (PointClassification::Outside, PointClassification::Inside)
            };
            for (y, z) in [(big, 0.0), (-big, 0.5), (big + 1.0, -0.3)] {
                assert_eq!(at(-0.1, y, z), behind, "{label}: behind at ({y}, {z})");
                assert_eq!(at(0.1, y, z), ahead, "{label}: ahead at ({y}, {z})");
            }
            assert_eq!(at(-big, 0.0, 1.0), behind, "{label}: far side");

            let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let meshed = oriented_solid_volume(&topo, piece, 0.01).unwrap();
            assert!(
                (meshed - truth).abs() < 1e-2 * truth,
                "{label}: mesh volume {meshed}, truth {truth}"
            );
        }
    }
}
