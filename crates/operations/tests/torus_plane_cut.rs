//! A torus cut by a plane across its axis or through it, keeping either
//! side. Across the axis, with the plane `c` above the centre and
//! `a = asin(c / r)` its angle up the tube, Pappus gives the part below as
//! `2 pi R` times the tube disc's area below the chord at `c`, its torus face
//! covers `2 pi r R (pi + 2a)` and the plane face is the annulus
//! `4 pi R sqrt(r^2 - c^2)`. A plane parallel to the axis closer to it than
//! `R - r` meets the tube in two loops winding around it.
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
use brepkit_topology::solid::SolidId;

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

/// A box over one side of a plane through the torus's axis, fused with the
/// torus: the box, plus the half of the ring outside it.
#[test]
fn box_fused_over_half_a_torus() {
    let (big, small) = (4.0_f64, 1.5_f64);
    let mut topo = Topology::new();
    let torus = make_torus(&mut topo, big, small, 32).unwrap();
    let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    let place =
        Mat4::rotation_y(std::f64::consts::FRAC_PI_2) * Mat4::translation(-10.0, -10.0, 0.0);
    transform_solid(&mut topo, lid, &place).unwrap();
    let both = boolean(&mut topo, BooleanOp::Fuse, torus, lid).unwrap();

    let report = validate_solid(&topo, both).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let truth = 8000.0 + PI * PI * big * small * small;
    let volume = solid_volume(&topo, both, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
    let mesh = tessellate_solid(&topo, both, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let inside = |x: f64, y: f64, z: f64| {
        classify_point(
            &topo,
            both,
            Point3::new(x, y, z),
            &ClassifyOptions::default(),
        )
        .unwrap()
    };
    assert_eq!(inside(-big, 0.0, 0.0), PointClassification::Inside);
    assert_eq!(inside(-big, 0.0, 1.6), PointClassification::Outside);
    assert_eq!(inside(5.0, 5.0, 5.0), PointClassification::Inside);
}

/// The volume of the `(big, small)` ring beyond the plane `x = h`, for `|h|`
/// under `big - small`: over its height (`z = small sin t`), each annulus
/// keeps the share `acos(h / rho) / pi` of its circle at `rho`.
fn ring_beyond(big: f64, small: f64, h: f64) -> f64 {
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let annulus = |w: f64| {
        simpson(800, big - w, big + w, &|rho: f64| {
            2.0 * rho * (h / rho).acos()
        })
    };
    simpson(800, -PI / 2.0, PI / 2.0, &|t: f64| {
        let w = small * t.cos();
        w * annulus(w)
    })
}

/// Checks a boolean piece: exactly the given faces by surface type, valid
/// when it is one piece, `solid_volume` within 1e-8 of `truth` (the loops
/// around the tube are fitted through exact points, which bounds it), and a
/// closed mesh within 1% of it.
fn check_piece(
    topo: &Topology,
    piece: SolidId,
    census: &[(&str, usize)],
    one_piece: bool,
    truth: f64,
    label: &str,
) {
    let mut found: Vec<(&str, usize)> = Vec::new();
    for face in solid_faces(topo, piece).unwrap() {
        let tag = topo.face(face).unwrap().surface().type_tag();
        match found.iter_mut().find(|(t, _)| *t == tag) {
            Some((_, n)) => *n += 1,
            None => found.push((tag, 1)),
        }
    }
    found.sort_unstable();
    let mut census = census.to_vec();
    census.sort_unstable();
    assert_eq!(found, census, "{label}: faces");
    // Two pieces share one shell, which the Euler check reads as one.
    if one_piece {
        let report = validate_solid(topo, piece).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
    }
    let volume = solid_volume(topo, piece, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-8 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
    let mesh = tessellate_solid(topo, piece, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    let meshed = oriented_solid_volume(topo, piece, 0.005).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-2 * truth,
        "{label}: mesh volume {meshed}, truth {truth}"
    );
}

fn at(topo: &Topology, piece: SolidId, p: Point3) -> PointClassification {
    classify_point(topo, piece, p, &ClassifyOptions::default()).unwrap()
}

/// A 20 x 20 x 10 slab over `x > 1` with the torus, upright and with the
/// whole scene tipped over.
#[test]
fn slab_over_one_side_of_the_ring() {
    use PointClassification::{Inside, Outside};
    let (big, small) = (4.0_f64, 1.5_f64);
    let ring = 2.0 * PI * PI * big * small * small;
    let beyond = ring_beyond(big, small, 1.0);
    let tips = [
        Mat4::identity(),
        Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3),
    ];
    let sector: &[(&str, usize)] = &[("plane", 2), ("torus", 1)];
    let fused: &[(&str, usize)] = &[("plane", 6), ("torus", 1)];
    for (k, tip) in tips.iter().enumerate() {
        for (op, census, truth, kept) in [
            (
                BooleanOp::Intersect,
                sector,
                beyond,
                [Inside, Outside, Outside],
            ),
            (
                BooleanOp::Cut,
                sector,
                ring - beyond,
                [Outside, Inside, Outside],
            ),
            (
                BooleanOp::Fuse,
                fused,
                4000.0 + ring - beyond,
                [Inside, Inside, Inside],
            ),
        ] {
            let label = format!("tip {k}, {op:?}");
            let mut topo = Topology::new();
            let torus = make_torus(&mut topo, big, small, 32).unwrap();
            let slab = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
            transform_solid(&mut topo, slab, &Mat4::translation(1.0, -10.0, -5.0)).unwrap();
            transform_solid(&mut topo, torus, tip).unwrap();
            transform_solid(&mut topo, slab, tip).unwrap();
            let piece = boolean(&mut topo, op, torus, slab).unwrap();
            check_piece(&topo, piece, census, true, truth, &label);
            // The ring past the plane, the ring behind it, the slab alone.
            let probes = [(big, 0.0, 0.0), (-big, 0.0, 0.0), (10.0, 5.0, 4.0)];
            for ((x, y, z), class) in probes.into_iter().zip(kept) {
                let p = tip.mul_point(Point3::new(x, y, z));
                assert_eq!(at(&topo, piece, p), class, "{label}: ({x}, {y}, {z})");
            }
            assert_eq!(
                at(&topo, piece, tip.mul_point(Point3::new(0.0, 0.0, 0.0))),
                Outside
            );
        }
    }
}

/// A 2 x 20 x 4 bar through the ring's hole and across its tube at
/// `|x| < 1`: its cut and its common part are each two pieces.
#[test]
fn bar_through_the_ring() {
    use PointClassification::{Inside, Outside};
    let (big, small) = (4.0_f64, 1.5_f64);
    let ring = 2.0 * PI * PI * big * small * small;
    let within = ring_beyond(big, small, -1.0) - ring_beyond(big, small, 1.0);
    let sectors: &[(&str, usize)] = &[("plane", 4), ("torus", 2)];
    let fused: &[(&str, usize)] = &[("plane", 6), ("torus", 2)];
    for (op, census, one_piece, truth, kept) in [
        (
            BooleanOp::Intersect,
            sectors,
            false,
            within,
            [Outside, Inside, Outside],
        ),
        (
            BooleanOp::Cut,
            sectors,
            false,
            ring - within,
            [Inside, Outside, Outside],
        ),
        (
            BooleanOp::Fuse,
            fused,
            true,
            160.0 + ring - within,
            [Inside, Inside, Inside],
        ),
    ] {
        let label = format!("{op:?}");
        let mut topo = Topology::new();
        let torus = make_torus(&mut topo, big, small, 32).unwrap();
        let bar = make_box(&mut topo, 2.0, 20.0, 4.0).unwrap();
        transform_solid(&mut topo, bar, &Mat4::translation(-1.0, -10.0, -2.0)).unwrap();
        let piece = boolean(&mut topo, op, torus, bar).unwrap();
        check_piece(&topo, piece, census, one_piece, truth, &label);
        // The ring clear of the bar, the ring in it, the bar alone.
        let probes = [(-big, 0.0, 0.5), (0.0, big, 0.0), (0.0, 0.0, 0.0)];
        for ((x, y, z), class) in probes.into_iter().zip(kept) {
            assert_eq!(
                at(&topo, piece, Point3::new(x, y, z)),
                class,
                "{label}: ({x}, {y}, {z})"
            );
        }
        assert_eq!(at(&topo, piece, Point3::new(big, 0.0, 1.6)), Outside);
    }
}

/// A slab tipped 0.15 rad off the axis: its plane still winds around the
/// tube. The cut and the common part make up the ring, and the fuse is the
/// cut plus the slab.
#[test]
fn slab_tilted_off_the_axis() {
    let (big, small) = (4.0_f64, 1.5_f64);
    let ring = 2.0 * PI * PI * big * small * small;
    let mut volumes = Vec::new();
    for op in [BooleanOp::Intersect, BooleanOp::Cut, BooleanOp::Fuse] {
        let mut topo = Topology::new();
        let torus = make_torus(&mut topo, big, small, 32).unwrap();
        let slab = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
        let place = Mat4::translation(1.0, 0.0, 0.0)
            * Mat4::rotation_y(0.15)
            * Mat4::translation(0.0, -10.0, -5.0);
        transform_solid(&mut topo, slab, &place).unwrap();
        let piece = boolean(&mut topo, op, torus, slab).unwrap();
        let census: &[(&str, usize)] = if op == BooleanOp::Fuse {
            &[("plane", 6), ("torus", 1)]
        } else {
            &[("plane", 2), ("torus", 1)]
        };
        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        check_piece(&topo, piece, census, true, volume, &format!("{op:?}"));
        volumes.push(volume);
    }
    let (common, cut, fused) = (volumes[0], volumes[1], volumes[2]);
    assert!(
        (common + cut - ring).abs() < 1e-9 * ring,
        "{common} + {cut}"
    );
    assert!(
        (fused - cut - 4000.0).abs() < 1e-9 * fused,
        "{fused} - {cut}"
    );
}
