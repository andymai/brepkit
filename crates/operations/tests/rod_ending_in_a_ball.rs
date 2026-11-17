//! A rod whose end sits at the centre of a ball of its own radius (a wire
//! guard's strut and the ball on its end): the end cap's rim lies on the ball
//! and the rod's wall touches the ball along it.
//!
//! The ball's faces meet the cap and the wall along that one rim, which both
//! sections must break where the rim's vertex is, whichever way the rod is
//! turned about its own axis.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_cylinder, make_sphere};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 0.9;
const LENGTH: f64 = 25.0;

/// A fuse that errors where it would fall back to a mesh.
fn exact_fuse(topo: &mut Topology, a: SolidId, b: SolidId, name: &str) -> SolidId {
    let opts = BooleanOptions {
        mesh_fallback: false,
        unify_faces: false,
        ..BooleanOptions::default()
    };
    boolean::boolean_with_options(topo, BooleanOp::Fuse, a, b, opts)
        .map_err(|e| format!("{name}: {e}"))
        .unwrap()
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        *m.entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    m
}

fn edges_not_used_twice(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wire).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

#[test]
fn a_rod_ending_at_a_balls_centre_fuses_onto_it() {
    let expected = PI * RADIUS * RADIUS * LENGTH + 2.0 / 3.0 * PI * RADIUS.powi(3);
    let at = Point3::new(28.638, 28.638, 2.65);
    for roll in [0.0, 0.3, 1.2, 2.5] {
        for yaw in [0.0, FRAC_PI_2 / 2.0, 2.0] {
            let name = format!("roll {roll}, yaw {yaw}");
            // The rod runs along -x from the ball's centre once placed.
            let place = Mat4::translation(at.x(), at.y(), at.z())
                * Mat4::rotation_z(yaw)
                * Mat4::rotation_y(-FRAC_PI_2)
                * Mat4::rotation_z(roll);
            let mut topo = Topology::new();
            let rod = make_cylinder(&mut topo, RADIUS, LENGTH).unwrap();
            transform_solid(&mut topo, rod, &place).unwrap();
            let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
            transform_solid(&mut topo, ball, &Mat4::translation(at.x(), at.y(), at.z())).unwrap();
            let fused = exact_fuse(&mut topo, rod, ball, &name);
            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            assert_eq!(
                census(&topo, fused),
                BTreeMap::from([("cylinder", 1), ("plane", 1), ("sphere", 2)]),
                "{name}"
            );
            assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
            let volume = solid_volume(&topo, fused, 0.0005).unwrap();
            assert!(
                (volume - expected).abs() < 1e-4 * expected,
                "{name}: volume {volume} should be {expected}"
            );
            // Along the rod, out past its end into the ball, and beside it.
            let local = |x: f64, y: f64, z: f64| {
                (Mat4::translation(at.x(), at.y(), at.z()) * Mat4::rotation_z(yaw))
                    .mul_point(Point3::new(x, y, z))
            };
            let class = |p: Point3| classify_point(&topo, fused, p, 0.01, 1e-7).unwrap();
            assert_eq!(
                class(local(-5.0, 0.0, 0.0)),
                PointClassification::Inside,
                "{name}"
            );
            assert_eq!(
                class(local(0.6, 0.0, 0.3)),
                PointClassification::Inside,
                "{name}"
            );
            assert_eq!(
                class(local(0.8, 0.0, 0.5)),
                PointClassification::Outside,
                "{name}"
            );
            assert_eq!(
                class(local(-5.0, 1.0, 0.0)),
                PointClassification::Outside,
                "{name}"
            );
        }
    }
}

/// The volume two rods of radius `RADIUS` and length `length`, both from the
/// origin along `a` and `b`, share: each line along `a` through the first
/// rod's cross-section meets the second rod in one stretch, read exactly,
/// summed over the cross-section in polar steps.
fn rods_overlap(a: Vec3, b: Vec3, length: f64) -> f64 {
    let e1 = if a.x().abs() < 0.9 {
        a.cross(Vec3::new(1.0, 0.0, 0.0))
    } else {
        a.cross(Vec3::new(0.0, 1.0, 0.0))
    };
    let e1 = e1 * (1.0 / e1.length());
    let e2 = a.cross(e1);
    let c = a.dot(b);
    let n = 2000;
    let (dr, dphi) = (RADIUS / f64::from(n), 2.0 * PI / f64::from(n));
    let mut sum = 0.0;
    for i in 0..n {
        let rho = (f64::from(i) + 0.5) * dr;
        for j in 0..n {
            let phi = (f64::from(j) + 0.5) * dphi;
            let q = e1 * (rho * phi.cos()) + e2 * (rho * phi.sin());
            let qb = q.dot(b);
            // Within the second rod's radius: s^2 (1 - c^2) - 2 s qb c
            // + |q|^2 - qb^2 - R^2 <= 0.
            let (qa2, qb2) = (1.0 - c * c, -2.0 * qb * c);
            let q0 = q.dot(q) - qb * qb - RADIUS * RADIUS;
            let disc = qb2 * qb2 - 4.0 * qa2 * q0;
            if disc <= 0.0 {
                continue;
            }
            let root = disc.sqrt();
            let (mut lo, mut hi) = ((-qb2 - root) / (2.0 * qa2), (-qb2 + root) / (2.0 * qa2));
            // Between the second rod's ends: 0 <= qb + s c <= length.
            let (s0, s1) = ((0.0 - qb) / c, (length - qb) / c);
            lo = lo.max(s0.min(s1)).max(0.0);
            hi = hi.min(s0.max(s1)).min(length);
            if hi > lo {
                sum += (hi - lo) * rho;
            }
        }
    }
    sum * dr * dphi
}

#[test]
fn two_rods_ending_at_one_balls_centre_fuse_onto_it() {
    // A wire guard's two struts meeting at one ball: each rod's end cap
    // lies in a great circle of the ball.
    let length = 10.0;
    for (roll, yaw, pitch) in [
        (0.3, 2.2, 0.4),
        (0.0, 2.2, 0.4),
        (1.1, 1.3, -0.7),
        // A wire guard's rib stepping down: the caps meet on the equator.
        (0.3, PI, 0.41),
    ] {
        for at in [Point3::new(0.0, 0.0, 0.0), Point3::new(28.638, -8.1, 2.65)] {
            let name = format!("roll {roll}, yaw {yaw}, pitch {pitch}, at {at:?}");
            let lay = |yaw: f64, pitch: f64| {
                Mat4::translation(at.x(), at.y(), at.z())
                    * Mat4::rotation_z(yaw)
                    * Mat4::rotation_y(pitch)
                    * Mat4::rotation_y(-FRAC_PI_2)
                    * Mat4::rotation_z(roll)
            };
            let along = |yaw: f64, pitch: f64| {
                let p = (Mat4::rotation_z(yaw) * Mat4::rotation_y(pitch))
                    .mul_point(Point3::new(-1.0, 0.0, 0.0));
                Vec3::new(p.x(), p.y(), p.z())
            };
            let (a, b) = (along(0.0, 0.0), along(yaw, pitch));
            let mut topo = Topology::new();
            let rod = make_cylinder(&mut topo, RADIUS, length).unwrap();
            transform_solid(&mut topo, rod, &lay(0.0, 0.0)).unwrap();
            let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
            transform_solid(&mut topo, ball, &Mat4::translation(at.x(), at.y(), at.z())).unwrap();
            let other = make_cylinder(&mut topo, RADIUS, length).unwrap();
            transform_solid(&mut topo, other, &lay(yaw, pitch)).unwrap();
            let first = exact_fuse(&mut topo, rod, ball, &name);
            let fused = exact_fuse(&mut topo, first, other, &name);
            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let kinds = census(&topo, fused);
            assert_eq!(kinds.get("cylinder"), Some(&2), "{name}: {kinds:?}");
            assert_eq!(kinds.get("plane"), Some(&2), "{name}: {kinds:?}");
            assert!(kinds.contains_key("sphere"), "{name}: {kinds:?}");
            assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
            // The rods, less their overlap, and the wedge of the ball
            // outside both: its two planes meet at the angle between the
            // rods' axes.
            let wedge = 2.0 / 3.0 * (PI - a.dot(b).acos()) * RADIUS.powi(3);
            let expected = 2.0 * PI * RADIUS * RADIUS * length - rods_overlap(a, b, length) + wedge;
            let volume = solid_volume(&topo, fused, 0.0005).unwrap();
            assert!(
                (volume - expected).abs() < 1e-4 * expected,
                "{name}: volume {volume} should be {expected}"
            );
            let class = |v: Vec3| classify_point(&topo, fused, at + v, 0.01, 1e-7).unwrap();
            let away = -(a + b) * (1.0 / (a + b).length());
            let side = a.cross(b) * (1.0 / a.cross(b).length());
            for (v, want) in [
                (a * 5.0, PointClassification::Inside),
                (b * 5.0, PointClassification::Inside),
                (away * 0.6, PointClassification::Inside),
                (away * 0.6 + side * 0.4, PointClassification::Inside),
                (away * 0.95, PointClassification::Outside),
                (a * 5.0 + side * 1.0, PointClassification::Outside),
                (b * 5.0 - side * 1.0, PointClassification::Outside),
            ] {
                assert_eq!(class(v), want, "{name} at {v:?}");
            }
        }
    }
}
