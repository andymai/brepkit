//! A rod cut by an oblique plane that crosses its whole wall: the wall is
//! then bounded by its bottom rim and an ellipse, and the solid holds
//! pi r^2 times the plane's height at the axis, whatever the tilt.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

#[test]
fn rod_cut_by_an_oblique_plane() {
    let (radius, height_at_axis) = (3.0_f64, 3.0_f64);
    for slope in [0.1_f64, 0.3, 0.5, 0.8] {
        for turn_deg in [0.0_f64, 60.0, 90.0, 200.0] {
            let label = format!("slope {slope}, turned {turn_deg} degrees");
            let turn = turn_deg.to_radians();
            let mut topo = Topology::new();
            let rod = make_cylinder(&mut topo, radius, 6.0).unwrap();
            // Everything above the plane z = 3 + slope * (x cos + y sin).
            let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
            let place = Mat4::rotation_z(turn)
                * Mat4::translation(0.0, 0.0, height_at_axis)
                * Mat4::rotation_y(-slope.atan())
                * Mat4::translation(-10.0, -10.0, 0.0);
            transform_solid(&mut topo, lid, &place).unwrap();
            let cut = boolean(&mut topo, BooleanOp::Cut, rod, lid).unwrap();

            let report = validate_solid(&topo, cut).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let faces = solid_faces(&topo, cut).unwrap();
            assert_eq!(faces.len(), 3, "{label}: faces");

            let truth = PI * radius * radius * height_at_axis;
            let volume = solid_volume(&topo, cut, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-9 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            let wall = faces
                .iter()
                .copied()
                .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
                .expect("a cylinder wall");
            let wall_truth = 2.0 * PI * radius * height_at_axis;
            let area = face_area(&topo, wall, 0.01).unwrap();
            assert!(
                (area - wall_truth).abs() < 1e-9 * wall_truth,
                "{label}: wall area {area}, truth {wall_truth}"
            );

            // The base disc and the ellipse, which leans off it by the tilt.
            let caps_truth = PI * radius * radius * (1.0 + slope.hypot(1.0));
            let caps: f64 = faces
                .iter()
                .filter(|&&f| topo.face(f).unwrap().surface().is_planar())
                .map(|&f| face_area(&topo, f, 0.01).unwrap())
                .sum();
            assert!(
                (caps - caps_truth).abs() < 1e-9 * caps_truth,
                "{label}: cap area {caps}, truth {caps_truth}"
            );

            let (s, c) = turn.sin_cos();
            let plane_z = |x: f64, y: f64| height_at_axis + slope * (x * c + y * s);
            let classify = |x: f64, y: f64, z: f64| {
                classify_point(
                    &topo,
                    cut,
                    Point3::new(x, y, z),
                    &ClassifyOptions::default(),
                )
                .unwrap()
            };
            for (x, y) in [
                (0.0, 0.0),
                (2.5 * c, 2.5 * s),
                (-2.5 * c, -2.5 * s),
                (-2.0 * s, 2.0 * c),
            ] {
                let z = plane_z(x, y);
                assert_eq!(
                    classify(x, y, z - 0.05),
                    PointClassification::Inside,
                    "{label}: below at ({x}, {y})"
                );
                assert_eq!(
                    classify(x, y, z + 0.05),
                    PointClassification::Outside,
                    "{label}: above at ({x}, {y})"
                );
            }

            let mesh = tessellate_solid(&topo, cut, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let meshed: f64 = mesh
                .indices
                .chunks_exact(3)
                .map(|t| {
                    let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                    (a - Point3::new(0.0, 0.0, 0.0)).dot((b - a).cross(c - a)) / 6.0
                })
                .sum();
            // Inscribed: short of the solid by the rims' chords only.
            assert!(
                meshed <= truth && truth - meshed < 5e-3 * truth,
                "{label}: mesh volume {meshed}, truth {truth}"
            );
        }
    }
}
