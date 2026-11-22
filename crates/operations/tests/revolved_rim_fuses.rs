//! Fuses of revolved parts that meet at a rim, built as two brepjs
//! playground examples build them.
//!
//! - `conical-pour-funnel`: a whole torus lip rolled onto the top rim of a
//!   funnel's body. The tube crosses the two cones of the bowl's wall and
//!   covers the flat ring between them.
//! - `rotary-potentiometer`: a lug, revolved 20 degrees, that stands on the
//!   flat rim of a cup. Its bottom lies in the rim's plane and its inner and
//!   outer faces lie on the cylinders of the cup's bore and wall.
//!
//! Each fuse is exact, valid and watertight in every pose, holds its volume
//! from the closed forms, and holds a point of a grid round the contact
//! exactly where one of its operands holds it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_torus};
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

fn turn() -> Mat4 {
    Mat4::translation(3.0, -2.0, 1.5) * Mat4::rotation_x(0.61) * Mat4::rotation_z(-0.43)
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(2.0, -1.0, 0.5), Vec3::new(0.3, 1.0, 0.2))
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

/// The closed profile `(r, z)` in the half-plane `y = 0`, `x = r`,
/// revolved `angle` about the `z` axis.
fn lathe(topo: &mut Topology, profile: &[(f64, f64)], angle: f64) -> SolidId {
    let points: Vec<Point3> = profile
        .iter()
        .map(|&(r, z)| Point3::new(r, 0.0, z))
        .collect();
    let wire = make_polygon_wire(topo, &points, 1e-7).unwrap();
    let face = make_planar_face_from_wire(topo, wire).unwrap();
    revolve(
        topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        angle,
    )
    .unwrap()
}

/// `2π ∫∫ r dA` over the closed profile, by Green's theorem: the volume of
/// its whole revolution.
fn pappus(profile: &[(f64, f64)]) -> f64 {
    let n = profile.len();
    let twice: f64 = (0..n)
        .map(|i| {
            let ((r1, z1), (r2, z2)) = (profile[i], profile[(i + 1) % n]);
            (z2 - z1) * r2.mul_add(r1 + r2, r1 * r1) / 3.0
        })
        .sum();
    PI * twice.abs()
}

fn class_of(topo: &Topology, solid: SolidId, p: Point3) -> PointClassification {
    classify_point(topo, solid, p, 0.01, 1e-7).unwrap()
}

/// Whether `solid` holds `p`; `None` on its boundary.
fn holds(topo: &Topology, solid: SolidId, p: Point3) -> Option<bool> {
    match class_of(topo, solid, p) {
        PointClassification::Inside => Some(true),
        PointClassification::Outside => Some(false),
        PointClassification::OnBoundary => None,
    }
}

/// The exact fuse of `a` and `b` in `pose`: valid and watertight, with
/// `faces` faces of the surface types `types` (counted alike), the volume
/// `truth`, and over the `n`-step grid from `lo` to `hi` (given upright) a
/// point held exactly where `a` or `b` holds it.
#[allow(clippy::too_many_arguments)]
fn fuse_exactly(
    topo: &mut Topology,
    (a, b): (SolidId, SolidId),
    pose: &str,
    types: &[(&str, usize)],
    truth: f64,
    (lo, hi, n): ([f64; 3], [f64; 3], usize),
) {
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let fused = boolean_with_options(topo, BooleanOp::Fuse, a, b, exact)
        .map_err(|e| format!("{pose}: {e}"))
        .unwrap();
    let report = validate_solid(topo, fused).unwrap();
    assert!(report.is_valid(), "{pose}: invalid: {report:?}");
    let mesh = tessellate_solid(topo, fused, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{pose}: open or non-manifold mesh");
    let mut census: Vec<(&str, usize)> = Vec::new();
    for f in solid_faces(topo, fused).unwrap() {
        let tag = topo.face(f).unwrap().surface().type_tag();
        match census.iter_mut().find(|(t, _)| *t == tag) {
            Some((_, n)) => *n += 1,
            None => census.push((tag, 1)),
        }
    }
    census.sort_unstable();
    let mut want = types.to_vec();
    want.sort_unstable();
    assert_eq!(census, want, "{pose}: faces");
    let volume = solid_volume(topo, fused, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-6 * truth,
        "{pose}: volume {volume}, truth {truth}"
    );
    let mut read = 0;
    #[allow(clippy::cast_precision_loss)]
    let at = |k: usize, i: usize| lo[k] + (hi[k] - lo[k]) * (i as f64 + 0.5) / n as f64;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = placed(Point3::new(at(0, i), at(1, j), at(2, k)), pose);
                let (Some(f), Some(ia), Some(ib)) =
                    (holds(topo, fused, p), holds(topo, a, p), holds(topo, b, p))
                else {
                    continue;
                };
                assert_eq!(f, ia || ib, "{pose}: at {p:?}");
                read += 1;
            }
        }
    }
    assert!(read > n.pow(3) / 2, "{pose}: only {read} points read");
}

/// The funnel's bowl and spout in `(r, z)`: its wall at 45 degrees, 1.6
/// thick, from the spout at `z = 0` up to the rim at `z = 70.5`, where the
/// flat ring between the two cones runs from `r = 80` out to `80 + skin`.
fn funnel_profile() -> Vec<(f64, f64)> {
    let skin = 1.6 / FRAC_1_SQRT_2;
    let (rim, top) = (80.0, 70.5);
    let outside = 9.5 + skin;
    vec![
        (6.0, -60.0),
        (7.6, -60.0),
        (7.6, -33.6),
        (outside, -28.0),
        (outside, 0.0),
        (rim + skin, top),
        (rim, top),
        (9.5, 0.0),
        (9.5, -28.0),
        (6.0, -33.0),
    ]
}

/// `2π ∫∫ r dA` over the part of the lip's tube section (radius 2 about
/// `(major, 70.5)`) that lies in the funnel's wall: below the rim, outside
/// the inner cone `z = r - 9.5` and inside the outer cone `z = r - outside`.
/// Read along rays from the tube's centre, each as far as the nearest of
/// the tube and the three bounds.
fn lip_in_wall(major: f64) -> f64 {
    let skin = 1.6 / FRAC_1_SQRT_2;
    let (top, outside) = (70.5, 9.5 + skin);
    let reach = |angle: f64| {
        let (s, c) = angle.sin_cos();
        let mut most: f64 = 2.0;
        if s - c > 0.0 {
            most = most.min((major - 9.5 - top) / (s - c));
        }
        if c - s > 0.0 {
            most = most.min((top - major + outside) / (c - s));
        }
        most.max(0.0)
    };
    let steps = 20_000;
    let gauss = gauss_legendre_points(4);
    let mut total = 0.0;
    for i in 0..steps {
        #[allow(clippy::cast_precision_loss)]
        let (lo, hi) = (
            PI + PI * i as f64 / steps as f64,
            PI + PI * (i + 1) as f64 / steps as f64,
        );
        for g in gauss {
            let angle = 0.5 * (hi - lo).mul_add(g.x + 1.0, 2.0 * lo);
            let s = reach(angle);
            total += 0.5 * (hi - lo) * g.w * (major * s * s / 2.0 + angle.cos() * s * s * s / 3.0);
        }
    }
    2.0 * PI * total
}

/// The lip's tube covers the flat ring of the rim and crosses both cones of
/// the wall, each along one circle: the fuse keeps the body's cones cut at
/// those circles and the part of the tube outside the wall, and drops the
/// ring.
#[test]
fn a_torus_lip_fuses_onto_a_funnels_rim() {
    let skin = 1.6 / FRAC_1_SQRT_2;
    let major = 80.0 + skin / 2.0 + 0.8;
    let profile = funnel_profile();
    let lip = 2.0 * PI * PI * major * 4.0;
    let truth = pappus(&profile) + lip - lip_in_wall(major);
    for pose in POSES {
        let mut topo = Topology::new();
        let body = lathe(&mut topo, &profile, 2.0 * PI);
        let torus = make_torus(&mut topo, major, 2.0, 32).unwrap();
        transform_solid(&mut topo, torus, &Mat4::translation(0.0, 0.0, 70.5)).unwrap();
        let operands = (posed(&mut topo, body, pose), posed(&mut topo, torus, pose));
        fuse_exactly(
            &mut topo,
            operands,
            pose,
            &[("cone", 4), ("cylinder", 4), ("plane", 1), ("torus", 1)],
            truth,
            ([77.0, -4.0, 67.0], [85.0, 4.0, 73.0], 14),
        );
    }
}

/// `∫ f` over `[lo, hi]` by four-point Gauss rules on `steps` equal pieces.
fn integrate(f: impl Fn(f64) -> f64, (lo, hi): (f64, f64), steps: usize) -> f64 {
    let gauss = gauss_legendre_points(4);
    #[allow(clippy::cast_precision_loss)]
    let step = (hi - lo) / steps as f64;
    let mut total = 0.0;
    for i in 0..steps {
        #[allow(clippy::cast_precision_loss)]
        let a = step.mul_add(i as f64, lo);
        for g in gauss {
            total += 0.5 * step * g.w * f(step.mul_add(0.5 * (g.x + 1.0), a));
        }
    }
    total
}

/// The volume that the hang tab's plate, `x` from -108 to -80, `y` from -11
/// to 11 and `z` from 69.3 to 71.7, shares with the lipped funnel (the body
/// and the lip's tube of radius 2 about the circle of radius `major` at
/// `z = 70.5`). At a distance `ρ` from the axis, the body and the tube hold
/// the same heights all round, so the shared volume is the integral over `ρ`
/// of the heights in the plate held there, times the arc of the circle of
/// radius `ρ` over the plate.
fn tab_in_lipped_funnel(major: f64) -> f64 {
    let skin = 1.6 / FRAC_1_SQRT_2;
    let (top, outside) = (70.5, 9.5 + skin);
    let (low, high) = (69.3, 71.7);
    let held = |rho: f64| {
        let mut spans: Vec<(f64, f64)> = Vec::new();
        let off = rho - major;
        if off.abs() < 2.0 {
            let h = off.mul_add(-off, 4.0).sqrt();
            spans.push((top - h, top + h));
        }
        if rho - outside < top {
            spans.push((rho - outside, top));
        }
        let mut clipped: Vec<(f64, f64)> = spans
            .into_iter()
            .map(|(a, b)| (a.max(low), b.min(high)))
            .filter(|(a, b)| b > a)
            .collect();
        clipped.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut length = 0.0;
        let mut reach = f64::NEG_INFINITY;
        for (a, b) in clipped {
            let a = a.max(reach);
            if b > a {
                length += b - a;
            }
            reach = reach.max(b);
        }
        length
    };
    let arc = |rho: f64| 2.0 * rho * (80.0 / rho).acos().min((11.0 / rho).asin());
    integrate(|rho| held(rho) * arc(rho), (80.0, major + 2.0), 40_000)
}

/// The hang tab's plate runs out over the rim through the lip's tube: it
/// cuts two islands out of the tube's band, one round its outer side and a
/// lens on the inner side under the plate's top, and fuses with the wall
/// below the rim.
#[test]
fn a_hang_tab_fuses_through_a_funnels_lip() {
    let skin = 1.6 / FRAC_1_SQRT_2;
    let major = 80.0 + skin / 2.0 + 0.8;
    let profile = funnel_profile();
    let lip = 2.0 * PI * PI * major * 4.0;
    let lipped = pappus(&profile) + lip - lip_in_wall(major);
    let truth = lipped + 28.0 * 22.0 * 2.4 - tab_in_lipped_funnel(major);
    for pose in POSES {
        let mut topo = Topology::new();
        let body = lathe(&mut topo, &profile, 2.0 * PI);
        let torus = make_torus(&mut topo, major, 2.0, 32).unwrap();
        transform_solid(&mut topo, torus, &Mat4::translation(0.0, 0.0, 70.5)).unwrap();
        let exact = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let funnel = boolean_with_options(&mut topo, BooleanOp::Fuse, body, torus, exact).unwrap();
        let plate = make_box(&mut topo, 28.0, 22.0, 2.4).unwrap();
        transform_solid(&mut topo, plate, &Mat4::translation(-108.0, -11.0, 69.3)).unwrap();
        let operands = (
            posed(&mut topo, funnel, pose),
            posed(&mut topo, plate, pose),
        );
        fuse_exactly(
            &mut topo,
            operands,
            pose,
            &[("cone", 4), ("cylinder", 4), ("plane", 8), ("torus", 1)],
            truth,
            ([-86.0, -12.0, 68.0], [-78.0, 12.0, 73.0], 14),
        );
    }
}

/// The lug's bottom covers a 20 degree piece of the cup's rim, and its two
/// curved faces continue the cup's wall and bore above the rim. The fuse
/// cuts the rim along the lug's two sides, drops the piece under the lug,
/// and holds the cup's volume and the lug's.
#[test]
fn a_lug_standing_on_a_cups_rim_fuses_onto_it() {
    let (bottom, rim) = (-11.8, -3.2);
    let cup_profile = [
        (0.0, bottom),
        (12.0, bottom),
        (12.0, rim),
        (11.5, rim),
        (11.5, bottom + 0.5),
        (0.0, bottom + 0.5),
    ];
    let lug_profile = [
        (11.5, rim),
        (12.0, rim),
        (12.0, 0.4),
        (10.0, 0.4),
        (10.0, 0.0),
        (11.5, 0.0),
    ];
    let sweep = PI / 9.0;
    let truth = pappus(&cup_profile) + pappus(&lug_profile) * sweep / (2.0 * PI);
    for pose in POSES {
        let mut topo = Topology::new();
        let cup = lathe(&mut topo, &cup_profile, 2.0 * PI);
        let lug = lathe(&mut topo, &lug_profile, sweep);
        transform_solid(&mut topo, lug, &Mat4::rotation_z(sweep)).unwrap();
        let operands = (posed(&mut topo, cup, pose), posed(&mut topo, lug, pose));
        fuse_exactly(
            &mut topo,
            operands,
            pose,
            &[("cylinder", 5), ("plane", 7)],
            truth,
            ([7.0, 2.5, -5.0], [12.5, 9.0, 1.0], 14),
        );
    }
}
