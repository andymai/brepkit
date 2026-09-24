//! A rod halved by a plane through its axis, turned in 15 degree steps so
//! the plane passes through the rod's seam (at 90 and 270 degrees here) as
//! well as clear of it. Each cut keeps an analytic, valid half of the rod.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

#[test]
fn rod_halved_through_its_axis_at_every_angle() {
    for step in 0..24 {
        let angle = f64::from(step) * 15.0_f64.to_radians();
        let label = format!("{} degrees", step * 15);
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
        // The block removes the half on the far side of the plane through
        // the axis whose normal points along `angle`.
        let block = make_box(&mut topo, 5.0, 10.0, 6.0).unwrap();
        let place = Mat4::rotation_z(angle) * Mat4::translation(-5.0, -5.0, -1.0);
        transform_solid(&mut topo, block, &place).unwrap();
        let half = boolean(&mut topo, BooleanOp::Cut, rod, block).unwrap();

        let report = validate_solid(&topo, half).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        let faces = solid_faces(&topo, half).unwrap();
        assert!(faces.len() <= 6, "{label}: {} faces", faces.len());
        for &f in &faces {
            assert!(
                matches!(
                    topo.face(f).unwrap().surface(),
                    FaceSurface::Plane { .. } | FaceSurface::Cylinder(_)
                ),
                "{label}: the cut stays analytic"
            );
        }

        let (s, c) = angle.sin_cos();
        let classify = |r: f64, z: f64| {
            classify_point(
                &topo,
                half,
                Point3::new(r * c, r * s, z),
                &ClassifyOptions::default(),
            )
            .unwrap()
        };
        assert_eq!(
            classify(1.5, 2.0),
            PointClassification::Inside,
            "{label}: kept"
        );
        assert_eq!(
            classify(2.9, 0.1),
            PointClassification::Inside,
            "{label}: kept"
        );
        assert_eq!(
            classify(-1.5, 2.0),
            PointClassification::Outside,
            "{label}: cut"
        );
        assert_eq!(
            classify(-2.9, 3.9),
            PointClassification::Outside,
            "{label}: cut"
        );

        let volume = solid_volume(&topo, half, 0.001).unwrap();
        let truth = 18.0 * PI;
        assert!(
            (volume - truth).abs() < 2e-3 * truth,
            "{label}: volume {volume}, truth {truth}"
        );
        let mesh = tessellate_solid(&topo, half, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    }
}
