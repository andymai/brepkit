//! A rod or a cone frustum standing through a plate's edge, fused with the
//! plate: the plate takes a window out of the tool's wall, and where the
//! window straddles the wall's seam the wall's outer wire carries it as a
//! notch between two rims and the seam's two copies, which is no box in
//! `(u, v)`. Each fuse matches its closed form and classifies points on
//! either side of the tool's wall right, each wall matches its area, and
//! each wall's own mesh stays within the deflection of the surface, whichever
//! way the tool's seam points. The rod cut by the plate, cutting it and within
//! it is exact too, and joins an N-way fuse, with its seam away from the plate.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The area of a disc of radius `r` about `x = cx` where `x > 0`.
fn disc_past_zero(r: f64, cx: f64) -> f64 {
    let d = (-cx / r).clamp(-1.0, 1.0);
    r * r * (d.acos() - d * (1.0 - d * d).sqrt())
}

/// The plate `[0, 10] x [0, 10] x [0, 2]` fused with `tool` placed at
/// `(cx, 5, -4)`, turned `spin` about its axis.
fn fuse(cone: bool, cx: f64, spin: f64) -> (Topology, SolidId) {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
    let tool = if cone {
        make_cone(&mut topo, 1.2, 0.6, 10.0).unwrap()
    } else {
        make_cylinder(&mut topo, 1.0, 10.0).unwrap()
    };
    let place = Mat4::translation(cx, 5.0, -4.0) * Mat4::rotation_z(spin);
    transform_solid(&mut topo, tool, &place).unwrap();
    let fused = boolean(&mut topo, BooleanOp::Fuse, plate, tool).unwrap();
    (topo, fused)
}

/// A wall's own mesh at deflection `DEFLECTION`: its area, and how far the
/// midpoint of any of its edges falls inside the wall, whose radius at
/// height `z` is `radius(z)` about the vertical axis through `(cx, 5)`.
fn wall_mesh(
    topo: &Topology,
    face: brepkit_topology::face::FaceId,
    cx: f64,
    radius: &dyn Fn(f64) -> f64,
) -> (f64, f64) {
    let mesh = tessellate(topo, face, DEFLECTION).unwrap();
    let (mut area, mut sag) = (0.0, 0.0_f64);
    for t in mesh.indices.chunks(3) {
        let corners = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
        area += (corners[1] - corners[0])
            .cross(corners[2] - corners[0])
            .length()
            / 2.0;
        for k in 0..3 {
            let (a, b) = (corners[k], corners[(k + 1) % 3]);
            let (x, y, z) = (
                0.5 * (a.x() + b.x()),
                0.5 * (a.y() + b.y()),
                0.5 * (a.z() + b.z()),
            );
            sag = sag.max(radius(z) - (x - cx).hypot(y - 5.0));
        }
    }
    (area, sag)
}

const DEFLECTION: f64 = 0.002;

/// Points just inside and just outside the tool's wall beside the plate,
/// above it and below it, and in the plate past the tool.
fn assert_classifies(
    topo: &Topology,
    fused: SolidId,
    cx: f64,
    radius: &dyn Fn(f64) -> f64,
    label: &str,
) {
    use PointClassification::{Inside, Outside};
    for (x, z, want) in [
        (cx - 0.9 * radius(1.0), 1.0, Inside),
        (cx - 1.1 * radius(1.0), 1.0, Outside),
        (cx - 0.9 * radius(-2.0), -2.0, Inside),
        (cx + 1.1 * radius(-2.0), -2.0, Outside),
        (cx + 0.9 * radius(4.0), 4.0, Inside),
        (cx + 1.1 * radius(4.0), 4.0, Outside),
        (cx + 1.5, 1.0, Inside),
    ] {
        let p = Point3::new(x, 5.0, z);
        let got = classify_point(topo, fused, p, &ClassifyOptions::default()).unwrap();
        assert_eq!(got, want, "{label}: ({x}, 5, {z})");
    }
}

#[test]
fn a_rod_through_a_plate_edge_fuses_whole() {
    for cx in [0.0, 0.4] {
        let overlap = 2.0 * disc_past_zero(1.0, cx);
        let truth = 200.0 + 10.0 * PI - overlap;
        // The window's angular span, where the rod's wall lies over x > 0.
        let span = 2.0 * (-cx).clamp(-1.0, 1.0).acos();
        let wall = 2.0 * PI * 10.0 - 2.0 * span;
        // At cx = 0 turned 1 radian the fuse falls back to a mesh.
        let spins: &[f64] = if cx == 0.0 {
            &[0.0, PI]
        } else {
            &[0.0, 1.0, PI]
        };
        for &spin in spins {
            let label = format!("cx {cx} spin {spin}");
            let (topo, fused) = fuse(false, cx, spin);
            let faces = solid_faces(&topo, fused).unwrap();
            assert!(faces.len() <= 12, "{label}: fell back to a mesh");
            assert!(
                validate_solid(&topo, fused).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, fused, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-3 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            assert_classifies(&topo, fused, cx, &|_| 1.0, &label);
            let walls: Vec<_> = faces
                .into_iter()
                .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
                .collect();
            assert_eq!(walls.len(), 1, "{label}: walls");
            let area = face_area(&topo, walls[0], 0.01).unwrap();
            assert!(
                (area - wall).abs() < 1e-9 * wall,
                "{label}: wall area {area}, truth {wall}"
            );
            let (meshed, sag) = wall_mesh(&topo, walls[0], cx, &|_| 1.0);
            assert!(
                (meshed - wall).abs() < 2e-3 * wall,
                "{label}: wall mesh area {meshed}, truth {wall}"
            );
            assert!(
                sag <= DEFLECTION * (1.0 + 1e-6),
                "{label}: wall mesh sags {sag}"
            );
        }
    }
}

#[test]
fn a_cone_through_a_plate_edge_fuses_whole() {
    // The cone's radius at height z (the plate spans z in [0, 2]).
    let radius = |z: f64| 0.06f64.mul_add(-(z + 4.0), 1.2);
    let cone = PI * 10.0 / 3.0 * (1.2f64.powi(2) + 1.2 * 0.6 + 0.6f64.powi(2));
    // The wall's area per unit height, per radian, over its radius.
    let slant = 0.06f64.hypot(1.0);
    for cx in [0.0, 0.4] {
        let simpson = |f: &dyn Fn(f64) -> f64| {
            let n = 2000_u32;
            let step = 2.0 / f64::from(n);
            let mut sum = f(0.0) + f(2.0);
            for k in 1..n {
                sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step * f64::from(k));
            }
            sum * step / 3.0
        };
        let truth = 200.0 + cone - simpson(&|z| disc_past_zero(radius(z), cx));
        // The whole side less the window, where the wall lies over x > 0.
        let window = simpson(&|z| {
            let r = radius(z);
            r * 2.0 * (-cx / r).clamp(-1.0, 1.0).acos()
        });
        let wall = slant * (PI * (1.2 + 0.6) * 10.0 - window);
        for spin in [0.0, 1.0, PI] {
            let label = format!("cx {cx} spin {spin}");
            let (topo, fused) = fuse(true, cx, spin);
            let faces = solid_faces(&topo, fused).unwrap();
            assert!(faces.len() <= 12, "{label}: fell back to a mesh");
            assert!(
                validate_solid(&topo, fused).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, fused, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-3 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            assert_classifies(&topo, fused, cx, &radius, &label);
            let walls: Vec<_> = faces
                .into_iter()
                .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
                .collect();
            assert_eq!(walls.len(), 1, "{label}: walls");
            let area = face_area(&topo, walls[0], 0.01).unwrap();
            assert!(
                (area - wall).abs() < 1e-6 * wall,
                "{label}: wall area {area}, truth {wall}"
            );
            let (meshed, sag) = wall_mesh(&topo, walls[0], cx, &radius);
            assert!(
                (meshed - wall).abs() < 2e-3 * wall,
                "{label}: wall mesh area {meshed}, truth {wall}"
            );
            assert!(
                sag <= DEFLECTION * (1.0 + 1e-6),
                "{label}: wall mesh sags {sag}"
            );
        }
    }
}

/// The rod `make_cylinder(1, 10)` at `(cx, 5, -4)`, turned `spin` about its
/// axis, with the plate `[0, 10] x [0, 10] x [0, 2]`.
fn rod_and_plate(cx: f64, spin: f64) -> (Topology, SolidId, SolidId) {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
    let rod = make_cylinder(&mut topo, 1.0, 10.0).unwrap();
    let place = Mat4::translation(cx, 5.0, -4.0) * Mat4::rotation_z(spin);
    transform_solid(&mut topo, rod, &place).unwrap();
    (topo, rod, plate)
}

/// Turned so its seam lies away from the plate, the rod's floor and top
/// sections each run from one crossing of the plate's edge at `x = 0` to the
/// other, sharing both ends with that edge's piece; the Cut, the plate less
/// the rod and the Intersect keep the segments between them. At `x = -0.3`
/// turned `acos(0.3)` the seam runs through the window's corner, and turned
/// 2 its widest rulings are the window's sides. Each is exact, valid,
/// watertight, within `1e-3` of its closed form, and puts a point in the
/// overlap, one in the rod beside the plate and one in the plate right.
#[test]
fn a_rod_cut_at_a_plate_edge_keeps_its_segments() {
    use PointClassification::{Inside, Outside};
    for (cx, spins) in [
        (-0.5, vec![0.0, 1.0, 2.0, 2.5, PI]),
        (-0.3, vec![0.0, 1.0, 0.3f64.acos(), 2.0, 2.5, PI]),
        (0.4, vec![0.0, 1.0, 2.5, PI]),
    ] {
        let overlap = 2.0 * disc_past_zero(1.0, cx);
        let (in_overlap, beside) = (
            Point3::new(0.5 * (cx.max(0.0) + cx + 1.0), 5.0, 1.0),
            Point3::new(0.5 * (cx - 1.0), 5.0, 1.0),
        );
        let in_plate = Point3::new(5.0, 5.0, 1.0);
        for spin in spins {
            for (name, truth, classes) in [
                (
                    "rod less plate",
                    10.0 * PI - overlap,
                    [Outside, Inside, Outside],
                ),
                (
                    "plate less rod",
                    200.0 - overlap,
                    [Outside, Outside, Inside],
                ),
                ("rod within plate", overlap, [Inside, Outside, Outside]),
            ] {
                let label = format!("cx {cx} spin {spin}: {name}");
                let (mut topo, rod, plate) = rod_and_plate(cx, spin);
                let result = match name {
                    "rod less plate" => boolean(&mut topo, BooleanOp::Cut, rod, plate),
                    "plate less rod" => boolean(&mut topo, BooleanOp::Cut, plate, rod),
                    _ => boolean(&mut topo, BooleanOp::Intersect, rod, plate),
                }
                .unwrap();
                assert!(
                    solid_faces(&topo, result).unwrap().len() <= 12,
                    "{label}: fell back to a mesh"
                );
                assert!(
                    validate_solid(&topo, result).unwrap().is_valid(),
                    "{label}: invalid"
                );
                let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let volume = solid_volume(&topo, result, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-3 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
                for (p, want) in [in_overlap, beside, in_plate].into_iter().zip(classes) {
                    let got =
                        classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
                    assert_eq!(got, want, "{label}: {p:?}");
                }
            }
        }
    }
}

/// An N-way fuse of the plate, a tool and a far unit box: valid (the far box
/// a piece of the same shell), watertight, within `1e-3` of `truth`, and
/// holding a point in the tool clear of the plate, one in the plate, one in
/// the far box, and none beside them.
fn assert_n_way_fuse(topo: &Topology, fused: SolidId, truth: f64, in_tool: Point3, label: &str) {
    use PointClassification::{Inside, Outside};
    let report = validate_solid(topo, fused).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, fused, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    let volume = solid_volume(topo, fused, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-3 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
    for (p, want) in [
        (in_tool, Inside),
        (Point3::new(8.0, 8.0, 1.0), Inside),
        (Point3::new(20.5, 20.5, 20.5), Inside),
        (Point3::new(-5.0, 5.0, 1.0), Outside),
        (Point3::new(5.0, 5.0, 4.0), Outside),
    ] {
        let got = classify_point(topo, fused, p, &ClassifyOptions::default()).unwrap();
        assert_eq!(got, want, "{label}: {p:?}");
    }
}

/// The plate, the rod turned so its only vertices lie off the plate, and a
/// box far from both, fused at once: the rod's curved rims reach the plate,
/// so the pair takes part and the fuse matches its closed form.
#[test]
fn a_turned_rod_joins_an_n_way_fuse() {
    for cx in [-0.5, 0.4] {
        let (mut topo, rod, plate) = rod_and_plate(cx, PI);
        let far = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        transform_solid(&mut topo, far, &Mat4::translation(20.0, 20.0, 20.0)).unwrap();
        let fused = brepkit_algo::gfa::fuse_n(&mut topo, &[plate, rod, far]).unwrap();
        let truth = 200.0 + 10.0 * PI - 2.0 * disc_past_zero(1.0, cx) + 1.0;
        let in_rod = Point3::new(cx, 5.0, -2.0);
        assert_n_way_fuse(&topo, fused, truth, in_rod, &format!("cx {cx}"));
    }
}

/// The plate turned about `y`, cut by, cutting and within the rod turned so
/// its seam crosses the plate's window: each of the window's halves either
/// side of the seam is sampled inside itself, one of them wider than half a
/// turn. Each is exact, valid, watertight and within `1e-3` of its volume,
/// the rod's and the plate's less their overlap: over the rod's disc, the
/// length of the rod's vertical line that lies inside the turned slab.
#[test]
fn a_turned_plate_cuts_a_turned_rod() {
    for (tilt, cx, spin) in [(0.2, -0.2, 1.0), (-0.3, 0.3, 1.0), (0.5, 0.3, 0.5 * PI)] {
        let back = Mat4::rotation_y(-tilt);
        // The plate's own coordinates of `(x, y, z)`: affine in `z`.
        let local = |x: f64, y: f64, z: f64| back.mul_point(Point3::new(x, y, z));
        let inside_length = |x: f64, y: f64| {
            let (at0, at1) = (local(x, y, 0.0), local(x, y, 1.0));
            let (mut lo, mut hi) = (-4.0_f64, 6.0_f64);
            for (a, slope, (min, max)) in [
                (at0.x(), at1.x() - at0.x(), (0.0, 10.0)),
                (at0.z(), at1.z() - at0.z(), (0.0, 2.0)),
            ] {
                let (t0, t1) = ((min - a) / slope, (max - a) / slope);
                lo = lo.max(t0.min(t1));
                hi = hi.min(t0.max(t1));
            }
            (hi - lo).max(0.0)
        };
        let (nr, nt) = (400_u32, 800_u32);
        let simpson = |k: u32, n: u32| {
            if k == 0 || k == n {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            }
        };
        let mut overlap = 0.0;
        for i in 0..=nr {
            let r = f64::from(i) / f64::from(nr);
            for j in 0..=nt {
                let th = 2.0 * PI * f64::from(j) / f64::from(nt);
                overlap += simpson(i, nr)
                    * simpson(j, nt)
                    * r
                    * inside_length(r.mul_add(th.cos(), cx), r.mul_add(th.sin(), 5.0));
            }
        }
        overlap *= (1.0 / f64::from(nr)) * (2.0 * PI / f64::from(nt)) / 9.0;
        for (name, truth) in [
            ("rod less plate", 10.0 * PI - overlap),
            ("plate less rod", 200.0 - overlap),
            ("rod within plate", overlap),
        ] {
            let label = format!("tilt {tilt} cx {cx} spin {spin}: {name}");
            let (mut topo, rod, plate) = rod_and_plate(cx, spin);
            transform_solid(&mut topo, plate, &Mat4::rotation_y(tilt)).unwrap();
            let result = match name {
                "rod less plate" => boolean(&mut topo, BooleanOp::Cut, rod, plate),
                "plate less rod" => boolean(&mut topo, BooleanOp::Cut, plate, rod),
                _ => boolean(&mut topo, BooleanOp::Intersect, rod, plate),
            }
            .unwrap();
            assert!(
                solid_faces(&topo, result).unwrap().len() <= 12,
                "{label}: fell back to a mesh"
            );
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-3 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}

/// The plate with a ball dipping below it and a ring lying through its edge,
/// each with a box far off, fused at once: a ball's hemispheres meet on a
/// chordal equator and a ring's seams are points, so their boxes come from
/// their surfaces, and each pair takes part. The ball measures the plate and
/// the ball less its cap in the plate `pi h² (3r - h) / 3`; the ring the
/// plate and the ring less its part past `x = 0`, by Simpson over the radius
/// about its axis.
#[test]
fn a_ball_and_a_ring_join_an_n_way_fuse() {
    let (rr, r0, tcx) = (2.0_f64, 0.5_f64, -2.2_f64);
    let n = 200_000_u32;
    let (lo, hi) = (rr - r0, rr + r0);
    let step = (hi - lo) / f64::from(n);
    let past = |rho: f64| {
        let h2 = r0.mul_add(r0, -(rho - rr).powi(2));
        if h2 <= 0.0 || rho <= -tcx {
            0.0
        } else {
            2.0 * h2.sqrt() * rho * 2.0 * (-tcx / rho).acos()
        }
    };
    let mut sum = past(lo) + past(hi);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * past(step.mul_add(f64::from(k), lo));
    }
    let ring_overlap = sum * step / 3.0;
    let ring = 2.0 * PI * PI * rr * r0 * r0;
    let cap = PI * 0.25 * 0.5_f64.mul_add(-1.0, 3.0) / 3.0;
    for ball in [true, false] {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
        let (tool, truth, in_tool) = if ball {
            let s = make_sphere(&mut topo, 1.0, 32).unwrap();
            transform_solid(&mut topo, s, &Mat4::translation(5.0, 5.0, -0.5)).unwrap();
            (
                s,
                200.0 + 4.0 * PI / 3.0 - cap + 1.0,
                Point3::new(5.0, 5.0, -1.0),
            )
        } else {
            let t = make_torus(&mut topo, rr, r0, 16).unwrap();
            let place = Mat4::translation(tcx, 5.0, 1.0) * Mat4::rotation_z(PI);
            transform_solid(&mut topo, t, &place).unwrap();
            (
                t,
                200.0 + ring - ring_overlap + 1.0,
                Point3::new(tcx - rr, 5.0, 1.0),
            )
        };
        let far = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        transform_solid(&mut topo, far, &Mat4::translation(20.0, 20.0, 20.0)).unwrap();
        let fused = brepkit_algo::gfa::fuse_n(&mut topo, &[plate, tool, far]).unwrap();
        assert_n_way_fuse(&topo, fused, truth, in_tool, &format!("ball {ball}"));
    }
}
