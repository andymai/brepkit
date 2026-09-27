//! Recognizing a NURBS face as an analytic surface keeps the side the face
//! faces: a patch parameterized against the new surface's own normal is
//! turned over with it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::heal::{convert_to_bspline, convert_to_elementary};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::{SolidId, Topology};

type Make = fn(&mut Topology) -> SolidId;

/// A ball, a cylinder, a frustum, a torus, a plate bored through and a ball bored
/// through (faces with holes) converted to NURBS and recognized back,
/// upright, turned and mirrored (the mirror flips each patch's flag and keeps
/// its wire): every face comes back analytic, each plane stores its outward
/// normal unflagged, and each solid is valid, meshes watertight, measures
/// its volume within `1e-6`, and faces outward (its mesh's signed volume
/// within `1e-2` of the truth).
#[test]
fn recognized_faces_keep_their_side() {
    let frustum = PI * 10.0 / 3.0 * (25.0 + 10.0 + 4.0);
    // A radius-3 ball less a radius-0.5 bore through its middle: the bore's
    // barrel and the two caps it takes off.
    let (a, r) = (0.5_f64, 3.0_f64);
    let half = r.mul_add(r, -(a * a)).sqrt();
    let cap = (r - half).powi(2) * PI * (3.0 * r - (r - half)) / 3.0;
    let bored_ball = 4.0 / 3.0 * PI * r.powi(3) - PI * a * a * 2.0 * half - 2.0 * cap;
    let cases: [(&str, Make, f64); 6] = [
        ("ball", |t| make_sphere(t, 3.0, 32).unwrap(), 36.0 * PI),
        (
            "cylinder",
            |t| make_cylinder(t, 2.0, 5.0).unwrap(),
            20.0 * PI,
        ),
        (
            "frustum",
            |t| make_cone(t, 5.0, 2.0, 10.0).unwrap(),
            frustum,
        ),
        (
            "torus",
            |t| make_torus(t, 6.0, 2.0, 32).unwrap(),
            48.0 * PI * PI,
        ),
        (
            "bored plate",
            |t| {
                let plate = make_box(t, 10.0, 10.0, 4.0).unwrap();
                let rod = make_cylinder(t, 2.0, 10.0).unwrap();
                transform_solid(t, rod, &Mat4::translation(5.0, 5.0, -3.0)).unwrap();
                boolean(t, BooleanOp::Cut, plate, rod).unwrap()
            },
            400.0 - 16.0 * PI,
        ),
        (
            "bored ball",
            |t| {
                let ball = make_sphere(t, 3.0, 32).unwrap();
                let rod = make_cylinder(t, 0.5, 10.0).unwrap();
                transform_solid(t, rod, &Mat4::translation(0.0, 0.0, -5.0)).unwrap();
                boolean(t, BooleanOp::Cut, ball, rod).unwrap()
            },
            bored_ball,
        ),
    ];
    let poses = [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ];
    for (name, make, truth) in cases {
        for (pose, place) in &poses {
            let label = format!("{name} {pose}");
            let mut topo = Topology::new();
            let solid = make(&mut topo);
            convert_to_bspline(&mut topo, solid).unwrap();
            transform_solid(&mut topo, solid, place).unwrap();
            convert_to_elementary(&mut topo, solid, 1e-6).unwrap();
            for f in solid_faces(&topo, solid).unwrap() {
                let face = topo.face(f).unwrap();
                assert!(
                    !matches!(face.surface(), FaceSurface::Nurbs(_)),
                    "{label}: a face stayed NURBS"
                );
                if matches!(face.surface(), FaceSurface::Plane { .. }) {
                    assert!(!face.is_reversed(), "{label}: a plane came back flagged");
                }
            }
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-6 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            let signed = oriented_solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (signed - truth).abs() < 1e-2 * truth,
                "{label}: signed volume {signed}, truth {truth}"
            );
        }
    }
}
