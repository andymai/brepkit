//! Both point classifiers read a ball by its sphere faces' boundary planes:
//! every point of a grid through and around the ball classifies as its
//! closed form says, with the ball upright, turned about an oblique axis or
//! mirrored through a slanted plane, and so does the ball less a box corner,
//! an off-axis rod or a coaxial bore. A sphere face bounded by a loop in one
//! plane is the sphere's part on that plane's side (a polygon through the
//! loop cuts the chords' sagitta off it), and a tilted face is no graph over
//! the nearest axis plane, so a test projected onto one misreads its side.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 3.0;

/// Grid points through and around the ball, clear of the sphere and of
/// `near` by 0.04.
fn grid(near: &dyn Fn(Point3) -> f64) -> Vec<Point3> {
    let mut pts = Vec::new();
    for i in -4..=4 {
        for j in -4..=4 {
            for k in -4..=4 {
                let p = Point3::new(
                    f64::from(i) * 0.83,
                    f64::from(j) * 0.87,
                    f64::from(k) * 0.89,
                );
                let r = (p.x() * p.x() + p.y() * p.y() + p.z() * p.z()).sqrt();
                if (r - RADIUS).abs() > 0.04 && near(p) > 0.04 {
                    pts.push(p);
                }
            }
        }
    }
    pts
}

/// The grid points each classifier reads against `inside`, placed like the
/// solid: `(point, check says inside, operations says inside)`.
fn misreads(
    topo: &Topology,
    solid: SolidId,
    place: &dyn Fn(Point3) -> Point3,
    near: &dyn Fn(Point3) -> f64,
    inside: &dyn Fn(Point3) -> bool,
) -> (Vec<Point3>, Vec<Point3>) {
    let (mut check, mut ops) = (Vec::new(), Vec::new());
    for p in grid(near) {
        let q = place(p);
        let by_check = classify_point(topo, solid, q, &ClassifyOptions::default()).unwrap()
            == PointClassification::Inside;
        let by_ops = brepkit_operations::classify::classify_point(topo, solid, q, 0.01, 1e-7)
            .unwrap()
            == brepkit_operations::classify::PointClassification::Inside;
        if by_check != inside(p) {
            check.push(p);
        }
        if by_ops != inside(p) {
            ops.push(p);
        }
    }
    (check, ops)
}

fn in_ball(p: Point3) -> bool {
    p.x() * p.x() + p.y() * p.y() + p.z() * p.z() < RADIUS * RADIUS
}

#[test]
fn a_ball_classifies_in_every_pose() {
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
    let at = Point3::new(0.3, 0.0, 0.0);
    let normal = Vec3::new(1.0, 0.2, 0.1);
    let unit = normal.normalize().unwrap();
    for pose in ["upright", "turned", "mirrored"] {
        let mut topo = Topology::new();
        let mut ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
        match pose {
            "turned" => transform_solid(&mut topo, ball, &turn).unwrap(),
            "mirrored" => ball = mirror(&mut topo, ball, at, normal).unwrap(),
            _ => {}
        }
        let place = |p: Point3| match pose {
            "turned" => turn.mul_point(p),
            "mirrored" => p - unit * (2.0 * (p - at).dot(unit)),
            _ => p,
        };
        let (check, ops) = misreads(&topo, ball, &place, &|_| 1.0, &in_ball);
        assert!(check.is_empty(), "{pose}: check misreads {check:?}");
        assert!(ops.is_empty(), "{pose}: operations misreads {ops:?}");
    }
}

#[test]
fn a_ball_less_a_tool_classifies() {
    for tool in ["corner", "rod", "bore"] {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let block = match tool {
            "corner" => {
                let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                transform_solid(&mut topo, b, &Mat4::translation(1.0, 1.2, 0.8)).unwrap();
                b
            }
            "rod" => {
                let c = make_cylinder(&mut topo, 0.6, 20.0).unwrap();
                let place = Mat4::translation(0.5, 10.0, 1.0)
                    * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
                transform_solid(&mut topo, c, &place).unwrap();
                c
            }
            _ => {
                let c = make_cylinder(&mut topo, 1.0, 20.0).unwrap();
                transform_solid(&mut topo, c, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
                c
            }
        };
        let result = boolean(&mut topo, BooleanOp::Cut, ball, block).unwrap();
        let spheres = solid_faces(&topo, result)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
            .count();
        assert!(spheres > 0, "{tool}: the result keeps no sphere face");
        let near = |p: Point3| match tool {
            "corner" => (p.x() - 1.0)
                .abs()
                .min((p.y() - 1.2).abs())
                .min((p.z() - 0.8).abs()),
            "rod" => ((p.x() - 0.5).hypot(p.z() - 1.0) - 0.6).abs(),
            _ => (p.x().hypot(p.y()) - 1.0).abs(),
        };
        let in_tool = |p: Point3| match tool {
            "corner" => p.x() > 1.0 && p.y() > 1.2 && p.z() > 0.8,
            "rod" => (p.x() - 0.5).hypot(p.z() - 1.0) < 0.6,
            _ => p.x().hypot(p.y()) < 1.0,
        };
        let (check, _) = misreads(&topo, result, &|p| p, &near, &|p| in_ball(p) && !in_tool(p));
        assert!(check.is_empty(), "{tool}: check misreads {check:?}");
    }
}
