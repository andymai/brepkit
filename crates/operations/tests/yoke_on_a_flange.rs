//! A universal joint's yoke on its flange, built as the brepjs playground
//! example `universal-joint` builds it: two ears and a bridge fused into a
//! fork, trimmed to a turned outside that tapers along a cone to radius 34
//! at x = -62, a disc of radius 52 from x = -72 to -62 under it, and a disc
//! of radius 34 from x = -65 to -59 through the plane of the yoke's back.
//! The yoke's back is three faces in the plane x = -62 (the bridge between
//! the ears and two caps past them), all inside the large disc's face there.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::compound_ops::fuse_all;
use brepkit_operations::copy::copy_solid;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
use brepkit_topology::compound::Compound;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

fn turn() -> Mat4 {
    Mat4::translation(5.0, -4.0, 3.0) * Mat4::rotation_x(0.43) * Mat4::rotation_z(0.71)
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(-10.0, 3.0, 1.0), Vec3::new(0.2, 1.0, 0.4))
}

fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            transform_solid(topo, solid, &turn()).unwrap();
            solid
        }
        "mirrored" => {
            let (at, normal) = mirror_plane();
            mirror(topo, solid, at, normal).unwrap()
        }
        _ => solid,
    }
}

fn placed(p: Point3, pose: &str) -> Point3 {
    match pose {
        "turned" => turn().mul_point(p),
        "mirrored" => {
            let (at, normal) = mirror_plane();
            let unit = normal.normalize().unwrap();
            p - unit * (2.0 * (p - at).dot(unit))
        }
        _ => p,
    }
}

fn exact() -> BooleanOptions {
    BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    }
}

/// The yoke: a fork of two ears round the bore on z and a bridge, trimmed to
/// the turned outside about x.
fn yoke(topo: &mut Topology) -> SolidId {
    let mut outline: Vec<Point3> = (0..25)
        .map(|i| {
            let a = (7.5 * f64::from(i) - 90.0).to_radians();
            Point3::new(21.0 * a.cos(), 21.0 * a.sin(), 28.0)
        })
        .collect();
    outline.extend([
        Point3::new(-62.0, 21.0, 28.0),
        Point3::new(-62.0, -21.0, 28.0),
    ]);
    let wire = make_polygon_wire(topo, &outline, 1e-7).unwrap();
    let face = make_planar_face_from_wire(topo, wire).unwrap();
    let ear = extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 16.5).unwrap();
    let other = copy_solid(topo, ear).unwrap();
    transform_solid(topo, other, &Mat4::rotation_x(PI)).unwrap();
    let bridge = make_box(topo, 24.0, 42.0, 89.0).unwrap();
    transform_solid(topo, bridge, &Mat4::translation(-62.0, -21.0, -44.5)).unwrap();
    let parts = topo.add_compound(Compound::new(vec![ear, other, bridge]));
    let fork = fuse_all(topo, parts).unwrap();
    let profile: Vec<Point3> = [
        (-62.0, 0.0),
        (22.0, 0.0),
        (22.0, 47.0),
        (-46.0, 47.0),
        (-62.0, 34.0),
    ]
    .iter()
    .map(|&(x, r)| Point3::new(x, r, 0.0))
    .collect();
    let wire = make_polygon_wire(topo, &profile, 1e-7).unwrap();
    let face = make_planar_face_from_wire(topo, wire).unwrap();
    let outside = revolve(
        topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        2.0 * PI,
    )
    .unwrap();
    boolean_with_options(topo, BooleanOp::Intersect, fork, outside, exact())
        .map_err(|e| e.to_string())
        .unwrap()
}

/// A disc of `radius` about the x axis from `x0` to `x1`.
fn disc(topo: &mut Topology, radius: f64, (x0, x1): (f64, f64)) -> SolidId {
    let solid = make_cylinder(topo, radius, x1 - x0).unwrap();
    transform_solid(
        topo,
        solid,
        &(Mat4::translation(x0, 0.0, 0.0) * Mat4::rotation_y(PI / 2.0)),
    )
    .unwrap();
    solid
}

fn holds(topo: &Topology, solid: SolidId, p: Point3) -> Option<bool> {
    match classify_point(topo, solid, p, 0.01, 1e-7).unwrap() {
        PointClassification::Inside => Some(true),
        PointClassification::Outside => Some(false),
        PointClassification::OnBoundary => None,
    }
}

/// The yoke fused onto the flange's disc is exact, valid and watertight in
/// every pose. The two only touch, so its volume is theirs together, and
/// over a grid round the back it holds a point where either does.
#[test]
fn a_yoke_fuses_onto_its_flange_disc() {
    for pose in POSES {
        let mut topo = Topology::new();
        let body = yoke(&mut topo);
        let flange = disc(&mut topo, 52.0, (-72.0, -62.0));
        let (a, b) = (posed(&mut topo, body, pose), posed(&mut topo, flange, pose));
        let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, exact())
            .map_err(|e| format!("{pose}: {e}"))
            .unwrap();
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{pose}: invalid: {report:?}");
        let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{pose}: open or non-manifold mesh");
        assert!(
            solid_faces(&topo, fused).unwrap().iter().any(|&f| topo
                .face(f)
                .unwrap()
                .surface()
                .type_tag()
                == "cone"),
            "{pose}: the taper is gone"
        );
        let volume = |s| solid_volume(&topo, s, 0.01).unwrap();
        let (va, vb, vf) = (volume(a), volume(b), volume(fused));
        assert!(
            (vf - (va + vb)).abs() < 1e-6 * vf,
            "{pose}: fused {vf}, the yoke {va} and the disc {vb}"
        );
        let n = 14;
        let mut read = 0;
        #[allow(clippy::cast_precision_loss)]
        let at = |s: usize, lo: f64, hi: f64| lo + (hi - lo) * (s as f64 + 0.5) / n as f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let p = placed(
                        Point3::new(at(i, -66.0, -58.0), at(j, -40.0, 40.0), at(k, -40.0, 40.0)),
                        pose,
                    );
                    let (Some(f), Some(ia), Some(ib)) = (
                        holds(&topo, fused, p),
                        holds(&topo, a, p),
                        holds(&topo, b, p),
                    ) else {
                        continue;
                    };
                    assert_eq!(f, ia || ib, "{pose}: at {p:?}");
                    read += 1;
                }
            }
        }
        assert!(read > n * n * n / 2, "{pose}: only {read} points read");
    }
}

/// The yoke, the flange's disc and the spigot through the plane of the
/// yoke's back fused in one go are exact, valid and watertight in every
/// pose. The spigot's wall meets the plane where the cone's rim runs, and the
/// yoke's sides cut it along four rulings. The volume is the three solids'
/// less their overlaps: the spigot's half inside the disc, and its slab
/// between the yoke's sides on the far half.
#[test]
fn a_yoke_fuses_onto_its_stepped_flange() {
    let r: f64 = 34.0;
    let half = 21.0;
    let slab = 2.0 * (half * (r * r - half * half).sqrt() + r * r * (half / r).asin());
    for pose in POSES {
        let mut topo = Topology::new();
        let body = yoke(&mut topo);
        let flange = disc(&mut topo, 52.0, (-72.0, -62.0));
        let spigot = disc(&mut topo, r, (-65.0, -59.0));
        let (a, b, c) = (
            posed(&mut topo, body, pose),
            posed(&mut topo, flange, pose),
            posed(&mut topo, spigot, pose),
        );
        let volume = |topo: &Topology, s| solid_volume(topo, s, 0.01).unwrap();
        let (va, vb, vc) = (volume(&topo, a), volume(&topo, b), volume(&topo, c));
        let parts = topo.add_compound(Compound::new(vec![a, b, c]));
        let fused = fuse_all(&mut topo, parts)
            .map_err(|e| format!("{pose}: {e}"))
            .unwrap();
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{pose}: invalid: {report:?}");
        let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{pose}: open or non-manifold mesh");
        let faces = solid_faces(&topo, fused).unwrap();
        assert!(
            faces.len() < 120,
            "{pose}: {} faces, a mesh in place of the B-Rep",
            faces.len()
        );
        assert!(
            faces
                .iter()
                .any(|&f| topo.face(f).unwrap().surface().type_tag() == "cone"),
            "{pose}: the taper is gone"
        );
        let vf = volume(&topo, fused);
        let expected = va + vb + vc - PI * r * r * 3.0 - slab * 3.0;
        assert!(
            (vf - expected).abs() < 1e-6 * vf,
            "{pose}: fused {vf}, expected {expected}"
        );
        let n = 14;
        let mut read = 0;
        #[allow(clippy::cast_precision_loss)]
        let at = |s: usize, lo: f64, hi: f64| lo + (hi - lo) * (s as f64 + 0.5) / n as f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let p = placed(
                        Point3::new(at(i, -66.0, -56.0), at(j, -40.0, 40.0), at(k, -40.0, 40.0)),
                        pose,
                    );
                    let (Some(f), Some(ia), Some(ib), Some(ic)) = (
                        holds(&topo, fused, p),
                        holds(&topo, a, p),
                        holds(&topo, b, p),
                        holds(&topo, c, p),
                    ) else {
                        continue;
                    };
                    assert_eq!(f, ia || ib || ic, "{pose}: at {p:?}");
                    read += 1;
                }
            }
        }
        assert!(read > n * n * n / 2, "{pose}: only {read} points read");
    }
}
