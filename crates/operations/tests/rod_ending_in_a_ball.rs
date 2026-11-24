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
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
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

/// How far the edges of the solid's faces stray from the surfaces of the
/// faces they bound, read at points along each edge.
fn edges_off_their_faces(topo: &Topology, solid: SolidId) -> f64 {
    let mut worst = 0.0_f64;
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        let surface = face.surface();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wire).unwrap().edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                let (p0, p1) = (
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                );
                let (t0, t1) = edge.curve().domain_with_endpoints(p0, p1);
                for k in 1..8 {
                    let t = t0 + (t1 - t0) * f64::from(k) / 8.0;
                    let p = edge.curve().evaluate_with_endpoints(t, p0, p1);
                    let off = match surface {
                        FaceSurface::Plane { normal, d } => {
                            (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs()
                        }
                        s => {
                            let (u, v) = s.project_point(p).unwrap();
                            (s.evaluate(u, v).unwrap() - p).length()
                        }
                    };
                    worst = worst.max(off);
                }
            }
        }
    }
    worst
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

/// The volume two rods of radius `radius`, both from the origin along unit
/// `a` for `la` and unit `b` for `lb`, share: each line along `a` through the
/// first rod's cross-section meets the second rod in one stretch, read
/// exactly, summed over the cross-section in polar steps.
fn rods_overlap(radius: f64, (a, la): (Vec3, f64), (b, lb): (Vec3, f64)) -> f64 {
    let e1 = if a.x().abs() < 0.9 {
        a.cross(Vec3::new(1.0, 0.0, 0.0))
    } else {
        a.cross(Vec3::new(0.0, 1.0, 0.0))
    };
    let e1 = e1 * (1.0 / e1.length());
    let e2 = a.cross(e1);
    let c = a.dot(b);
    let n = 2000;
    let (dr, dphi) = (radius / f64::from(n), 2.0 * PI / f64::from(n));
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
            let q0 = q.dot(q) - qb * qb - radius * radius;
            let disc = qb2 * qb2 - 4.0 * qa2 * q0;
            if disc <= 0.0 {
                continue;
            }
            let root = disc.sqrt();
            let (mut lo, mut hi) = ((-qb2 - root) / (2.0 * qa2), (-qb2 + root) / (2.0 * qa2));
            // Between the second rod's ends: 0 <= qb + s c <= lb.
            let (s0, s1) = ((0.0 - qb) / c, (lb - qb) / c);
            lo = lo.max(s0.min(s1)).max(0.0);
            hi = hi.min(s0.max(s1)).min(la);
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
    // The first rod lies along -x and the second at (yaw, pitch) from it,
    // each built from the ball's centre, or from its far end where its flag
    // is set (its end cap at the ball). With `swap` the second rod meets the
    // ball first.
    // `mirror_too` adds the scene mirrored in `y`.
    for (roll, yaw, pitch, swap, far_ends, mirror_too) in [
        (0.3, 2.2, 0.4, false, (false, false), true),
        (0.0, 2.2, 0.4, false, (false, false), true),
        (1.1, 1.3, -0.7, false, (false, false), true),
        // A wire guard's rib stepping down: the caps meet on the equator.
        (0.3, PI, 0.41, false, (false, false), true),
        (0.3, PI, -0.41, false, (true, false), false),
        (0.3, PI, 0.41, true, (false, true), true),
        // Both axes in the equator's plane: the caps meet at the poles.
        (0.3, 2.2, 0.0, false, (false, false), true),
        // The second rod's cap runs through both poles.
        (0.0, 2.2, 0.4, true, (false, false), true),
        // The second rod's cap crosses the ball's upper quarter along a
        // short arc of its rim, by the first rod's cap.
        (0.0, 2.9, 1.0, false, (false, false), true),
        (0.3, 2.9, 1.0, true, (true, true), true),
        (0.0, 2.9, 1.0, true, (false, true), true),
    ] {
        for (at, mirrored) in poses().into_iter().filter(|&(_, m)| mirror_too || !m) {
            two_rods_at_one_ball(roll, yaw, pitch, swap, far_ends, at, mirrored);
        }
    }
}

#[test]
fn two_rods_whose_caps_meet_at_a_balls_poles_fuse_onto_it() {
    // Each section of the ball ends at a pole of its first chart, or at the
    // rim of the cap fused first: the ball's faces are charted clear of the
    // poles, and the pieces either side of a section run across that chart's
    // seam.
    for (roll, yaw, pitch, swap, far_ends) in [
        (0.0, 2.2, 0.0, true, (true, false)),
        (0.3, 2.2, 0.0, true, (true, true)),
        (0.0, 2.9, 1.0, true, (true, true)),
        (0.3, 2.9, 1.0, true, (true, false)),
        (0.0, PI, -0.41, false, (false, true)),
        (0.3, PI, -0.41, false, (true, true)),
        (0.0, PI, 1.0, true, (true, false)),
        (0.0, 2.9, 0.4, true, (true, false)),
        // A thin lune of the ball, between two caps 0.24 radians apart.
        (0.3, 2.9, 0.0, false, (false, true)),
        // The second rod's cap rides the arcs that cut it from its rim.
        (0.3, 2.9, 0.4, false, (false, true)),
        (0.3, 2.9, 0.4, false, (true, true)),
        (0.0, 1.0, 0.0, false, (false, false)),
        (0.0, 2.2, 1.0, true, (true, false)),
    ] {
        for (at, mirrored) in poses() {
            two_rods_at_one_ball(roll, yaw, pitch, swap, far_ends, at, mirrored);
        }
        two_rods_at_one_ball(
            roll,
            yaw,
            pitch,
            swap,
            far_ends,
            Point3::new(0.0, 0.0, 0.0),
            true,
        );
    }
}

#[test]
fn two_nearly_opposite_rods_with_their_caps_at_a_ball_fuse_exactly() {
    // The rods lie a few degrees from opposite, so their walls touch at the
    // ball's poles, and each rod's cap rim and seam stay within a thousandth
    // of the other rod's wall along a long arc there. Neither lies on that
    // wall, and the first rod's cap meets it at the poles only.
    for roll in [0.3, 1.0] {
        for (at, mirrored) in poses()
            .into_iter()
            .chain([(Point3::new(0.0, 0.0, 0.0), true)])
        {
            two_rods_at_one_ball(roll, 3.05, 0.0, true, (true, true), at, mirrored);
        }
    }
}

#[test]
fn two_nearly_opposite_rods_at_a_ball_fuse_exactly_or_refuse() {
    // Rods a few degrees from opposite, in poses where a rim or a seam runs
    // within a thousandth of the other rod's wall: each fuse returns the
    // exact solid or an error, never a solid with an edge off a face it
    // bounds.
    for (roll, yaw, swap, far_ends) in [
        (0.3, 3.0, true, (true, true)),
        (0.3, 3.05, true, (true, true)),
        (1.0, 3.1, true, (true, false)),
        (2.0, 3.1, true, (false, false)),
        (3.0, 3.1, true, (false, false)),
        (4.0, 3.1, true, (true, true)),
        (5.0, 3.1, true, (true, true)),
        (5.0, 3.1, false, (false, false)),
    ] {
        for (at, mirrored) in poses()
            .into_iter()
            .chain([(Point3::new(0.0, 0.0, 0.0), true)])
        {
            two_rods_at_one_ball_or_refused(
                roll,
                yaw,
                0.0,
                swap,
                far_ends,
                at,
                mirrored,
                Join::Pairwise,
            );
        }
    }
}

#[test]
fn two_rods_and_a_ball_fuse_at_once() {
    // `fuse_n` cuts each of the ball's hemispheres, bounded by its equator,
    // with both caps' great circles, which cross on one of them. Every axis
    // has a pole on such a face, and the circles' crossing can lie at the
    // hemisphere's own pole.
    for (roll, yaw, pitch, swap, far_ends) in [
        (0.0, 1.0, 0.0, false, (false, false)),
        (0.0, 1.0, 0.4, false, (false, false)),
        (0.0, 1.0, -0.41, true, (true, true)),
        (0.0, 2.2, 0.4, false, (false, false)),
        (0.0, 2.2, 1.0, false, (true, false)),
        (0.0, 2.9, 0.4, false, (true, true)),
        (0.3, 1.0, 0.4, true, (false, false)),
        (0.3, 2.2, 1.0, true, (false, false)),
        (0.3, 2.9, 0.4, false, (false, false)),
    ] {
        for (at, mirrored) in poses() {
            two_rods_joined_at_one_ball(
                roll,
                yaw,
                pitch,
                swap,
                far_ends,
                at,
                mirrored,
                Join::AtOnce,
            );
        }
    }
}

#[test]
fn two_rods_fuse_onto_a_ball_with_its_faces_unified() {
    // With its faces joined, the first fuse leaves the ball one face bounded
    // by the first cap's great circle. Every axis has a pole on that face,
    // and the second cap's great circle crosses it.
    for (roll, yaw, pitch, swap, far_ends) in [
        (0.0, 1.0, 0.0, true, (true, false)),
        (0.0, 1.0, 1.0, true, (false, true)),
        (0.0, 2.9, 0.0, false, (false, true)),
        (0.0, PI, 1.0, false, (false, false)),
        (0.3, 2.2, 0.4, false, (false, true)),
        (0.3, 2.9, -0.41, false, (false, false)),
        // The second cap's sections end a whole turn of the chart from the
        // outline's copies of their ends.
        (0.0, 2.9, -0.41, false, (false, false)),
        (0.3, 2.9, -0.41, false, (false, true)),
    ] {
        for (at, mirrored) in poses() {
            two_rods_joined_at_one_ball(
                roll,
                yaw,
                pitch,
                swap,
                far_ends,
                at,
                mirrored,
                Join::PairwiseUnified,
            );
        }
    }
}

/// The ball's centre, and whether the scene is mirrored in `y` about the
/// origin once placed.
fn poses() -> [(Point3, bool); 3] {
    [
        (Point3::new(0.0, 0.0, 0.0), false),
        (Point3::new(28.638, -8.1, 2.65), false),
        (Point3::new(28.638, -8.1, 2.65), true),
    ]
}

/// How the rods and the ball are fused.
#[derive(Clone, Copy, Debug)]
enum Join {
    /// The first rod and the ball, then the second rod, faces kept apart.
    Pairwise,
    /// As `Pairwise`, with the faces of one surface joined after each fuse
    /// (the default, and brepjs's `simplify`).
    PairwiseUnified,
    /// All three at once, with `fuse_n`.
    AtOnce,
}

/// Two rods of length 10 from the centre of a ball of their radius at `at`,
/// the first along -x and the second at (`yaw`, `pitch`) from it, each
/// turned `roll` about its own axis and built from its far end where its
/// flag is set; fused exactly, the second rod first with `swap`.
fn two_rods_at_one_ball(
    roll: f64,
    yaw: f64,
    pitch: f64,
    swap: bool,
    far_ends: (bool, bool),
    at: Point3,
    mirrored: bool,
) {
    two_rods_joined_at_one_ball(
        roll,
        yaw,
        pitch,
        swap,
        far_ends,
        at,
        mirrored,
        Join::Pairwise,
    );
}

/// [`two_rods_at_one_ball`], fused as `join` says.
#[allow(clippy::too_many_arguments)]
fn two_rods_joined_at_one_ball(
    roll: f64,
    yaw: f64,
    pitch: f64,
    swap: bool,
    far_ends: (bool, bool),
    at: Point3,
    mirrored: bool,
    join: Join,
) {
    assert!(
        two_rods_at_one_ball_or_refused(roll, yaw, pitch, swap, far_ends, at, mirrored, join),
        "roll {roll}, yaw {yaw}, pitch {pitch}, swap {swap}, far ends {far_ends:?}, at {at:?}, mirrored {mirrored}, {join:?}: refused"
    );
}

/// As [`two_rods_joined_at_one_ball`], but a fuse that refuses passes:
/// `false` where the last fuse returns an error. A fuse that returns a
/// solid must return the exact one.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn two_rods_at_one_ball_or_refused(
    roll: f64,
    yaw: f64,
    pitch: f64,
    swap: bool,
    (first_far, second_far): (bool, bool),
    at: Point3,
    mirrored: bool,
    join: Join,
) -> bool {
    let length = 10.0;
    let name = format!(
        "roll {roll}, yaw {yaw}, pitch {pitch}, swap {swap}, far ends {first_far} {second_far}, at {at:?}, mirrored {mirrored}, {join:?}"
    );
    let pose = if mirrored {
        Mat4::scale(1.0, -1.0, 1.0)
    } else {
        Mat4::identity()
    };
    let lay = |yaw: f64, pitch: f64, far_end: bool| {
        let base = Mat4::translation(at.x(), at.y(), at.z())
            * Mat4::rotation_z(yaw)
            * Mat4::rotation_y(pitch)
            * Mat4::rotation_y(-FRAC_PI_2);
        if far_end {
            base * Mat4::translation(0.0, 0.0, length)
                * Mat4::rotation_x(PI)
                * Mat4::rotation_z(roll)
        } else {
            base * Mat4::rotation_z(roll)
        }
    };
    let along = |yaw: f64, pitch: f64| {
        let p = (Mat4::rotation_z(yaw) * Mat4::rotation_y(pitch))
            .mul_point(Point3::new(-1.0, 0.0, 0.0));
        Vec3::new(p.x(), p.y(), p.z())
    };
    let (a, b) = (along(0.0, 0.0), along(yaw, pitch));
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, RADIUS, length).unwrap();
    transform_solid(&mut topo, rod, &lay(0.0, 0.0, first_far)).unwrap();
    let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
    transform_solid(&mut topo, ball, &Mat4::translation(at.x(), at.y(), at.z())).unwrap();
    let other = make_cylinder(&mut topo, RADIUS, length).unwrap();
    transform_solid(&mut topo, other, &lay(yaw, pitch, second_far)).unwrap();
    if mirrored {
        for solid in [rod, ball, other] {
            transform_solid(&mut topo, solid, &pose).unwrap();
        }
    }
    let (one, two) = if swap { (other, rod) } else { (rod, other) };
    let opts = BooleanOptions {
        mesh_fallback: false,
        unify_faces: matches!(join, Join::PairwiseUnified),
        ..BooleanOptions::default()
    };
    let fused = match join {
        Join::AtOnce => brepkit_algo::gfa::fuse_n(&mut topo, &[one, ball, two]).ok(),
        Join::Pairwise | Join::PairwiseUnified => {
            let first = boolean::boolean_with_options(&mut topo, BooleanOp::Fuse, one, ball, opts)
                .map_err(|e| format!("{name}: {e}"))
                .unwrap();
            boolean::boolean_with_options(&mut topo, BooleanOp::Fuse, first, two, opts).ok()
        }
    };
    let Some(fused) = fused else {
        return false;
    };
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{name}: {:?}", report.issues);
    let kinds = census(&topo, fused);
    assert_eq!(kinds.get("cylinder"), Some(&2), "{name}: {kinds:?}");
    assert_eq!(kinds.get("plane"), Some(&2), "{name}: {kinds:?}");
    assert!(kinds.contains_key("sphere"), "{name}: {kinds:?}");
    assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
    let off = edges_off_their_faces(&topo, fused);
    assert!(
        off < 1e-9,
        "{name}: an edge stands {off:e} off a face it bounds"
    );
    let mesh = tessellate_solid(&topo, fused, 0.005).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
    // The rods, less their overlap, and the wedge of the ball outside both:
    // its two planes meet at the angle between the rods' axes.
    let wedge = 2.0 / 3.0 * (PI - a.dot(b).acos()) * RADIUS.powi(3);
    let expected = 2.0 * PI * RADIUS * RADIUS * length
        - rods_overlap(RADIUS, (a, length), (b, length))
        + wedge;
    let volume = solid_volume(&topo, fused, 0.0005).unwrap();
    assert!(
        (volume - expected).abs() < 1e-6 * expected,
        "{name}: volume {volume} should be {expected}"
    );
    let class = |v: Vec3| classify_point(&topo, fused, pose.mul_point(at + v), 0.01, 1e-7).unwrap();
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
    true
}

/// A turn by `angle` about `axis` through the origin.
fn turn_about(axis: Vec3, angle: f64) -> Mat4 {
    let u = axis * (1.0 / axis.length());
    let (ux, uy, uz) = (u.x(), u.y(), u.z());
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    Mat4([
        [
            t * ux * ux + c,
            t * ux * uy - s * uz,
            t * ux * uz + s * uy,
            0.0,
        ],
        [
            t * uy * ux + s * uz,
            t * uy * uy + c,
            t * uy * uz - s * ux,
            0.0,
        ],
        [
            t * uz * ux - s * uy,
            t * uz * uy + s * ux,
            t * uz * uz + c,
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

#[test]
fn cable_cores_bent_at_balls_fuse_exactly() {
    // The cable cores of a junction box: each a strand of three rods of
    // radius 1 with a ball at each bend, its middle rod a few degrees off
    // level. Each rod stands up from its start and turns about the
    // horizontal axis across its direction.
    let r = 1.0;
    for (n, k) in [(0, 0), (1, 2), (2, 2), (2, 0), (1, 1)] {
        let name = format!("gland {n} core {k}");
        let from = Point3::new(
            31.0,
            f64::from(n - 1) * 36.0 + f64::from(k - 1) * 2.1,
            if k == 1 { 26.4 } else { 22.8 },
        );
        let to = Point3::new(10.0, f64::from(3 * n + k - 4) * 6.2, 20.5);
        let pts = [
            from,
            Point3::new(from.x() - 2.0, from.y(), from.z()),
            Point3::new(to.x() + 3.0, to.y(), to.z()),
            Point3::new(to.x() - 7.0, to.y(), to.z()),
        ];
        let mut topo = Topology::new();
        let mut parts = Vec::new();
        for i in 1..pts.len() {
            let d = pts[i] - pts[i - 1];
            let rod = make_cylinder(&mut topo, r, d.length()).unwrap();
            let n = d * (1.0 / d.length());
            let turn = turn_about(Vec3::new(-n.y(), n.x(), 0.0), n.z().clamp(-1.0, 1.0).acos());
            let at = pts[i - 1];
            transform_solid(
                &mut topo,
                rod,
                &(Mat4::translation(at.x(), at.y(), at.z()) * turn),
            )
            .unwrap();
            parts.push(rod);
            if i < pts.len() - 1 {
                let ball = make_sphere(&mut topo, r, 32).unwrap();
                let at = pts[i];
                transform_solid(&mut topo, ball, &Mat4::translation(at.x(), at.y(), at.z()))
                    .unwrap();
                parts.push(ball);
            }
        }
        let fused = parts[1..]
            .iter()
            .fold(parts[0], |acc, &p| exact_fuse(&mut topo, acc, p, &name));
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        assert_eq!(edges_not_used_twice(&topo, fused), 0, "{name}");
        let kinds = census(&topo, fused);
        assert_eq!(kinds.get("cylinder"), Some(&3), "{name}: {kinds:?}");
        assert_eq!(kinds.get("plane"), Some(&2), "{name}: {kinds:?}");
        let mesh = tessellate_solid(&topo, fused, 0.005).unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
        // The rods, less each two that meet at a bend share there, and the
        // wedge of each ball outside the two rods that meet in it.
        let mut expected = 0.0;
        for i in 1..pts.len() {
            expected += PI * r * r * (pts[i] - pts[i - 1]).length();
        }
        for j in 1..pts.len() - 1 {
            let (a, b) = (pts[j - 1] - pts[j], pts[j + 1] - pts[j]);
            let (la, lb) = (a.length(), b.length());
            let (a, b) = (a * (1.0 / la), b * (1.0 / lb));
            expected +=
                2.0 / 3.0 * (PI - a.dot(b).acos()) * r.powi(3) - rods_overlap(r, (a, la), (b, lb));
        }
        let volume = solid_volume(&topo, fused, 0.0005).unwrap();
        assert!(
            (volume - expected).abs() < 1e-5 * expected,
            "{name}: volume {volume} should be {expected}"
        );
    }
}
