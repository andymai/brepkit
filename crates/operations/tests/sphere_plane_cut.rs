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

/// A ball cut by a plane across its equator (and not through its poles),
/// keeping either side: on each hemisphere the section is one chain of arcs
/// from the seam to the seam, and the hemisphere splits into the collar
/// holding its pole and the lune past the chain. With `d` as above, each
/// piece is valid, has one disc, matches the closed forms, classifies points
/// just either side of the plane, and meshes closed.
#[test]
fn ball_cut_by_a_plane_across_its_equator() {
    let r = 3.0_f64;
    // The half-space x' > x0 of a frame tilted about y, then turned about z.
    for (x0, tilt) in [
        (0.5, 0.0),
        (-1.2, 0.0),
        (2.5, 0.0),
        (0.0, 0.3),
        (0.5, 0.3),
        (-1.2, 0.6),
        (1.2, 0.2),
        (-0.3, 1.2),
    ] {
        for turn in [0.0_f64, 0.3, 1.0] {
            for keep_past in [true, false] {
                let label = format!(
                    "x0 {x0}, tilt {tilt}, turned {turn}, keeping {}",
                    if keep_past { "past" } else { "short of" }
                );
                let frame = Mat4::rotation_z(turn) * Mat4::rotation_y(tilt);
                let mut topo = Topology::new();
                let ball = make_sphere(&mut topo, r, 32).unwrap();
                let slab = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
                let place = frame * Mat4::translation(x0, -10.0, -10.0);
                transform_solid(&mut topo, slab, &place).unwrap();
                let op = if keep_past {
                    BooleanOp::Intersect
                } else {
                    BooleanOp::Cut
                };
                let piece = boolean(&mut topo, op, ball, slab).unwrap();

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

                let d = if keep_past { -x0 } else { x0 };
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

                let (past, short) = if keep_past {
                    (PointClassification::Inside, PointClassification::Outside)
                } else {
                    (PointClassification::Outside, PointClassification::Inside)
                };
                for (y, z) in [(0.0, 0.0), (0.4, 0.7), (-0.6, -0.3)] {
                    for (dx, want) in [(0.05, past), (-0.05, short)] {
                        let p = frame.mul_point(Point3::new(x0 + dx, y, z));
                        let got =
                            classify_point(&topo, piece, p, &ClassifyOptions::default()).unwrap();
                        assert_eq!(got, want, "{label}: {p:?}");
                    }
                }

                let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            }
        }
    }
}

/// Whether `piece` is exact (sphere and plane faces, at most `max` of
/// them), valid, meshes watertight, holds `truth` to within rounding, and
/// reads each of `probes` as held or not.
fn assert_exact_piece(
    topo: &Topology,
    piece: brepkit_topology::solid::SolidId,
    max: usize,
    truth: f64,
    probes: &[(Point3, bool)],
    label: &str,
) {
    let faces = solid_faces(topo, piece).unwrap();
    assert!(faces.len() <= max, "{label}: {} faces", faces.len());
    assert!(
        faces.iter().all(|&f| matches!(
            topo.face(f).unwrap().surface(),
            FaceSurface::Sphere(_) | FaceSurface::Plane { .. }
        )),
        "{label}: a face neither sphere nor plane"
    );
    let report = validate_solid(topo, piece).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, piece, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    let volume = solid_volume(topo, piece, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
    for &(at, held) in probes {
        let got = classify_point(topo, piece, at, &ClassifyOptions::default()).unwrap();
        let want = if held {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        assert_eq!(got, want, "{label}: {at:?} reads {got:?}");
    }
}

/// A ball cut by a plane through its axis, keeping either half: the section
/// runs from the equator through a pole and back on each hemisphere, which
/// then splits into the two lunes either side of it. Each half is exact,
/// valid and watertight, and holds half the ball.
#[test]
fn a_plane_through_the_axis_keeps_half_the_ball() {
    let r = 3.0_f64;
    let half = 2.0 * PI * r.powi(3) / 3.0;
    for turn in [0.0_f64, 0.3, 1.0, 2.5] {
        let (s, c) = turn.sin_cos();
        // The slab holds x' > 0 in its turned frame.
        let side = |d: f64| Point3::new(c * d, s * d, 0.4);
        for op in [BooleanOp::Intersect, BooleanOp::Cut] {
            let label = format!("turned {turn}, {op:?}");
            let mut topo = Topology::new();
            let ball = make_sphere(&mut topo, r, 32).unwrap();
            let slab = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
            let place = Mat4::rotation_z(turn) * Mat4::translation(0.0, -10.0, -10.0);
            transform_solid(&mut topo, slab, &place).unwrap();
            let piece = boolean(&mut topo, op, ball, slab).unwrap();
            let kept = matches!(op, BooleanOp::Intersect);
            let probes = [(side(1.5), kept), (side(-1.5), !kept)];
            assert_exact_piece(&topo, piece, 3, half, &probes, &label);
        }
    }
}

/// A ball against a quarter wedge through its axis (the box over `x' > 0`,
/// `y' > 0` in a frame turned about z): its two walls meet on the axis, so
/// each hemisphere's section turns a right angle at the pole and splits it
/// into lunes of a quarter and three quarters. Each piece is exact.
#[test]
fn a_wedge_through_the_axis_keeps_its_quarter() {
    let ball = 36.0 * PI;
    for turn in [0.0_f64, 0.3, 1.0, 2.0] {
        let (s, c) = turn.sin_cos();
        let at = |x: f64, y: f64| Point3::new(c * x - s * y, s * x + c * y, 0.4);
        for (op, truth, inside) in [
            (BooleanOp::Intersect, ball / 4.0, true),
            (BooleanOp::Cut, 0.75 * ball, false),
        ] {
            let label = format!("turned {turn}, {op:?}");
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, 3.0, 32).unwrap();
            let wedge = make_box(&mut topo, 10.0, 10.0, 20.0).unwrap();
            let place = Mat4::rotation_z(turn) * Mat4::translation(0.0, 0.0, -10.0);
            transform_solid(&mut topo, wedge, &place).unwrap();
            let piece = boolean(&mut topo, op, sphere, wedge).unwrap();
            let probes = [(at(1.0, 1.0), inside), (at(-1.0, 1.0), !inside)];
            assert_exact_piece(&topo, piece, 4, truth, &probes, &label);
        }
    }
}

/// The ball within and less the box over `x > x0`, `y > -1` with the wall
/// `x = x0` on the axis or 1e-3 off it: on each hemisphere that wall's arc
/// passes the pole on its way to the wall `y = -1`, and the chain of the two
/// splits the hemisphere into lunes. Each piece is exact.
#[test]
fn a_wall_holding_the_axis_keeps_its_piece() {
    let ball = 36.0 * PI;
    for x0 in [0.0_f64, 1e-3] {
        // The ball's slice past the wall `y = -1`, then less the slab
        // `0 < x < x0` of it, whose section is that slice's disc.
        let cap = PI * 2.0_f64.powi(2) * (3.0 * 3.0 - 2.0) / 3.0;
        let slice = PI * 9.0 - 9.0 * (1.0_f64 / 3.0).acos() + 8.0_f64.sqrt();
        let within = 0.5 * (ball - cap) - x0 * slice;
        for (op, truth) in [
            (BooleanOp::Intersect, within),
            (BooleanOp::Cut, ball - within),
        ] {
            let label = format!("x0 {x0}, {op:?}");
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, 3.0, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 11.0, 20.0).unwrap();
            transform_solid(&mut topo, block, &Mat4::translation(x0, -1.0, -10.0)).unwrap();
            let piece = boolean(&mut topo, op, sphere, block).unwrap();
            let held = matches!(op, BooleanOp::Intersect);
            let probes = [
                (Point3::new(1.0, 0.5, 0.3), held),
                (Point3::new(-1.0, 0.5, 0.3), !held),
            ];
            assert_exact_piece(&topo, piece, 4, truth, &probes, &label);
        }
    }
}

/// The ball against a box whose top face is the plane of its chordal
/// equator (built on `z = 0`, or tipped there by a quarter turn about y,
/// which leaves it 1e-16 off), turned about z: the ball within it is the
/// lower hemisphere, the ball less it the upper one, the box less the ball
/// keeps the lower one's dent and the two fused keep the upper one's dome,
/// each exact, valid, watertight, within `1e-9` of `18 pi` or the box less
/// or with it, and with points either side of the
/// equator on the right side. The plane's section circle is the hemispheres'
/// boundary in `(u, v)`, so it sections only the plane, and an unsplit
/// hemisphere's sample sits on its own side of the equator.
#[test]
fn a_box_on_the_equator_plane_keeps_a_hemisphere() {
    let half = 18.0 * PI;
    for (build, tipped) in [("on z = 0", false), ("tipped", true)] {
        for turn in [0.0_f64, 0.3, 1.0] {
            let place = if tipped {
                Mat4::rotation_z(turn)
                    * Mat4::rotation_y(std::f64::consts::FRAC_PI_2)
                    * Mat4::translation(0.0, -5.0, -5.0)
            } else {
                Mat4::rotation_z(turn) * Mat4::translation(-5.0, -5.0, -10.0)
            };
            for (name, truth, low, high) in [
                (
                    "ball within box",
                    half,
                    PointClassification::Inside,
                    PointClassification::Outside,
                ),
                (
                    "ball less box",
                    half,
                    PointClassification::Outside,
                    PointClassification::Inside,
                ),
                (
                    "box less ball",
                    1000.0 - half,
                    PointClassification::Outside,
                    PointClassification::Outside,
                ),
                (
                    "ball and box fused",
                    1000.0 + half,
                    PointClassification::Inside,
                    PointClassification::Inside,
                ),
            ] {
                let label = format!("{name}, {build}, turned {turn}");
                let mut topo = Topology::new();
                let ball = make_sphere(&mut topo, 3.0, 32).unwrap();
                let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                transform_solid(&mut topo, block, &place).unwrap();
                let piece = match name {
                    "ball within box" => boolean(&mut topo, BooleanOp::Intersect, ball, block),
                    "ball less box" => boolean(&mut topo, BooleanOp::Cut, ball, block),
                    "ball and box fused" => boolean(&mut topo, BooleanOp::Fuse, ball, block),
                    _ => boolean(&mut topo, BooleanOp::Cut, block, ball),
                }
                .unwrap();
                let faces = solid_faces(&topo, piece).unwrap();
                assert!(
                    faces.len() <= 8
                        && faces.iter().any(|&f| matches!(
                            topo.face(f).unwrap().surface(),
                            FaceSurface::Sphere(_)
                        )),
                    "{label}: fell back to a mesh ({} faces)",
                    faces.len()
                );
                let report = validate_solid(&topo, piece).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let volume = solid_volume(&topo, piece, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
                let at = |z: f64| {
                    classify_point(
                        &topo,
                        piece,
                        Point3::new(0.3, 0.2, z),
                        &ClassifyOptions::default(),
                    )
                    .unwrap()
                };
                assert_eq!(at(-1.0), low, "{label}: below the equator");
                assert_eq!(at(1.0), high, "{label}: above the equator");
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

/// `make_sphere(3, 32)` less a slab, clear of the equator (`1 < z < 2`, and
/// `0.2 < z < 0.7`) or across it (`-0.5 < z < 0.5`), upright, turned and
/// mirrored: the two pieces share one shell and the solid is exact (a few
/// sphere and plane faces), valid, meshes watertight, holds material on
/// both sides of the slab and none in it, and measures the ball less the
/// slab's disc stack within `1e-6`.
#[test]
fn a_ball_less_a_slab_keeps_both_pieces() {
    let r = 3.0_f64;
    let poses = [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ];
    for (z0, z1) in [(1.0_f64, 2.0_f64), (0.2, 0.7), (-0.5, 0.5)] {
        for (pose, place) in &poses {
            let label = format!("slab {z0} < z < {z1}, {pose}");
            let mut topo = Topology::new();
            let ball = make_sphere(&mut topo, r, 32).unwrap();
            let slab = make_box(&mut topo, 10.0, 10.0, z1 - z0).unwrap();
            transform_solid(&mut topo, slab, &Mat4::translation(-5.0, -5.0, z0)).unwrap();
            transform_solid(&mut topo, ball, place).unwrap();
            transform_solid(&mut topo, slab, place).unwrap();
            let cut = boolean(&mut topo, BooleanOp::Cut, ball, slab).unwrap();
            let report = validate_solid(&topo, cut).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, cut, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let faces = solid_faces(&topo, cut).unwrap();
            let spheres = faces
                .iter()
                .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
                .count();
            let exact = faces.iter().all(|&f| {
                matches!(
                    topo.face(f).unwrap().surface(),
                    FaceSurface::Sphere(_) | FaceSurface::Plane { .. }
                )
            });
            assert!(
                exact && spheres >= 2 && faces.len() <= 8,
                "{label}: {} faces, {spheres} spheres: not exact",
                faces.len()
            );
            let removed = PI * (r * r * (z1 - z0) - (z1.powi(3) - z0.powi(3)) / 3.0);
            let truth = 4.0 / 3.0 * PI * r.powi(3) - removed;
            let volume = solid_volume(&topo, cut, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-6 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            for (z, want) in [
                (0.5 * (z0 + z1), PointClassification::Outside),
                (z0 - 0.1, PointClassification::Inside),
                (z1 + 0.1, PointClassification::Inside),
            ] {
                let p = place.mul_point(Point3::new(0.3, 0.2, z));
                let got = classify_point(&topo, cut, p, &ClassifyOptions::default()).unwrap();
                assert_eq!(got, want, "{label}: z {z}");
            }
        }
    }
}
