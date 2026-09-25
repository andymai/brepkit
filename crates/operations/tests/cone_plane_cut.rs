//! A cone cut by a plane that crosses its whole wall, level or tilted, keeping
//! either side. The plane meets the wall in a circle or an ellipse, and the
//! cone between the apex and the plane is a cone over that section: a third
//! of its area times the apex's distance to the plane. Seen along the axis the
//! section encloses `proj`, the wall between the apex and the plane covers
//! the same region, and the wall's normal makes a fixed angle with the axis.
//!
//! A plane parallel to the axis meets the wall in two rulings when it holds
//! the axis and in a hyperbola otherwise; the part of the cone past it is,
//! at every height, the circular segment past the plane.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cone};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

struct Case {
    label: String,
    top_radius: f64,
    slope: f64,
    turn: f64,
    keep_tip: bool,
    planes: usize,
    volume: f64,
    wall: f64,
    caps: f64,
}

/// Cones of base radius 3 and height 6 under the plane
/// `z = 3 + slope (x cos(turn) + y sin(turn))`.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for slope in [0.0_f64, 0.3, 0.6, 0.9] {
        for turn_deg in [0.0_f64, 60.0, 90.0, 200.0] {
            // Pointed: apex (0, 0, 6), 2 units of rise per unit of radius.
            let proj = 18.0 * PI / (4.0 - slope * slope).powf(1.5);
            let root5 = 5.0_f64.sqrt();
            // Frustum to radius 1.5: apex (0, 0, 12), 4 units of rise.
            let fproj = 324.0 * PI / (16.0 - slope * slope).powf(1.5);
            let root17 = 17.0_f64.sqrt();
            // The section leans off the base plane by the plane's tilt.
            let lean = slope.hypot(1.0);
            for (top_radius, keep_tip, planes, volume, wall, caps) in [
                (
                    0.0,
                    false,
                    2,
                    18.0 * PI - proj,
                    root5 * (9.0 * PI - proj),
                    9.0 * PI + lean * proj,
                ),
                (0.0, true, 1, proj, root5 * proj, lean * proj),
                (
                    1.5,
                    false,
                    2,
                    36.0 * PI - 3.0 * fproj,
                    root17 * (9.0 * PI - fproj),
                    9.0 * PI + lean * fproj,
                ),
                (
                    1.5,
                    true,
                    2,
                    3.0 * fproj - 4.5 * PI,
                    root17 * (fproj - 2.25 * PI),
                    2.25 * PI + lean * fproj,
                ),
            ] {
                let label = format!(
                    "{} {}, slope {slope}, turned {turn_deg} degrees",
                    if top_radius > 0.0 { "frustum" } else { "cone" },
                    if keep_tip { "tip" } else { "base" },
                );
                out.push(Case {
                    label,
                    top_radius,
                    slope,
                    turn: turn_deg.to_radians(),
                    keep_tip,
                    planes,
                    volume,
                    wall,
                    caps,
                });
            }
        }
    }
    out
}

#[test]
fn cone_cut_by_a_plane_across_its_wall() {
    for case in cases() {
        let label = &case.label;
        let mut topo = Topology::new();
        let cone = make_cone(&mut topo, 3.0, case.top_radius, 6.0).unwrap();
        let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        let place = Mat4::rotation_z(case.turn)
            * Mat4::translation(0.0, 0.0, 3.0)
            * Mat4::rotation_y(-case.slope.atan())
            * Mat4::translation(-10.0, -10.0, 0.0);
        transform_solid(&mut topo, lid, &place).unwrap();
        let op = if case.keep_tip {
            BooleanOp::Intersect
        } else {
            BooleanOp::Cut
        };
        let piece = boolean(&mut topo, op, cone, lid).unwrap();

        let report = validate_solid(&topo, piece).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        let faces = solid_faces(&topo, piece).unwrap();
        let planes = faces
            .iter()
            .filter(|&&f| topo.face(f).unwrap().surface().is_planar())
            .count();
        assert_eq!(
            (faces.len(), planes),
            (case.planes + 1, case.planes),
            "{label}: faces"
        );

        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        assert!(
            (volume - case.volume).abs() < 1e-9 * case.volume,
            "{label}: volume {volume}, truth {}",
            case.volume
        );
        let wall = faces
            .iter()
            .copied()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
            .expect("a cone wall");
        let area = face_area(&topo, wall, 0.01).unwrap();
        assert!(
            (area - case.wall).abs() < 1e-9 * case.wall,
            "{label}: wall area {area}, truth {}",
            case.wall
        );

        let caps: f64 = faces
            .iter()
            .filter(|&&f| topo.face(f).unwrap().surface().is_planar())
            .map(|&f| face_area(&topo, f, 0.01).unwrap())
            .sum();
        assert!(
            (caps - case.caps).abs() < 1e-9 * case.caps,
            "{label}: cap area {caps}, truth {}",
            case.caps
        );

        let (s, c) = case.turn.sin_cos();
        let plane_z = |x: f64, y: f64| 3.0 + case.slope * (x * c + y * s);
        let (below, above) = if case.keep_tip {
            (PointClassification::Outside, PointClassification::Inside)
        } else {
            (PointClassification::Inside, PointClassification::Outside)
        };
        for (x, y) in [
            (0.0, 0.0),
            (0.8 * c, 0.8 * s),
            (-0.8 * c, -0.8 * s),
            (-0.8 * s, 0.8 * c),
        ] {
            let z = plane_z(x, y);
            let at = |z: f64| {
                classify_point(
                    &topo,
                    piece,
                    Point3::new(x, y, z),
                    &ClassifyOptions::default(),
                )
                .unwrap()
            };
            assert_eq!(at(z - 0.05), below, "{label}: below at ({x}, {y})");
            assert_eq!(at(z + 0.05), above, "{label}: above at ({x}, {y})");
        }

        let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
        let meshed: f64 = mesh
            .indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                (a - Point3::new(0.0, 0.0, 0.0)).dot((b - a).cross(c - a)) / 6.0
            })
            .sum();
        // Inscribed: short of the solid by the rims' chords only (a level
        // cut's r = 1.5 rim takes 27 chords at this deflection).
        assert!(
            meshed <= case.volume && case.volume - meshed < 1.5e-2 * case.volume,
            "{label}: mesh volume {meshed}, truth {}",
            case.volume
        );
    }
}

/// The cone of base radius 3 and height `h` narrowing to `top`, turned by
/// `turn` about its axis, against a box over `x < off`.
fn cone_and_box(top: f64, h: f64, turn: f64, off: f64, op: BooleanOp) -> (Topology, SolidId) {
    let mut topo = Topology::new();
    let cone = make_cone(&mut topo, 3.0, top, h).unwrap();
    transform_solid(&mut topo, cone, &Mat4::rotation_z(turn.to_radians())).unwrap();
    let block = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    transform_solid(
        &mut topo,
        block,
        &Mat4::translation(off - 20.0, -10.0, -5.0),
    )
    .unwrap();
    let piece = boolean(&mut topo, op, cone, block).unwrap();
    (topo, piece)
}

/// The volume of that cone over `x < off`: the circular segment past the
/// plane at every height, integrated by Simpson's rule split where the
/// radius reaches `|off|`, substituted `z = kink ∓ s²` on each side of it,
/// where the segment grows as the 3/2 power of the distance.
fn cone_below(top: f64, h: f64, off: f64) -> f64 {
    let radius = |z: f64| (top - 3.0).mul_add(z / h, 3.0);
    let segment = |z: f64| {
        let r = radius(z);
        if off >= r {
            PI * r * r
        } else if off <= -r {
            0.0
        } else {
            (r * r).mul_add(PI - (off / r).acos(), off * off.mul_add(-off, r * r).sqrt())
        }
    };
    let simpson = |span: f64, f: &dyn Fn(f64) -> f64| {
        let n = 2000;
        let step = span / f64::from(n);
        let mut sum = f(0.0) + f(span);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step * f64::from(k));
        }
        sum * step / 3.0
    };
    let kink = (3.0 - off.abs()) / (3.0 - top) * h;
    if kink > 0.0 && kink < h {
        simpson(kink.sqrt(), &|s: f64| {
            segment(s.mul_add(-s, kink)) * 2.0 * s
        }) + simpson((h - kink).sqrt(), &|s: f64| {
            segment(s.mul_add(s, kink)) * 2.0 * s
        })
    } else {
        simpson(h, &segment)
    }
}

/// A pointed cone and a frustum halved by a plane holding the axis, turned so
/// the plane's rulings fall on, beside and across the seam.
#[test]
fn cone_halved_through_its_axis() {
    for (top, h) in [(0.0_f64, 6.0_f64), (1.0, 4.0)] {
        let whole = PI * h / 3.0 * top.mul_add(top + 3.0, 9.0);
        for turn in [0.0_f64, 17.0, 90.0, 200.0] {
            for (op, truth) in [
                (BooleanOp::Cut, whole / 2.0),
                (BooleanOp::Intersect, whole / 2.0),
                (BooleanOp::Fuse, 8000.0 + whole / 2.0),
            ] {
                let label = format!("top {top}, turned {turn}, {op:?}");
                let (topo, piece) = cone_and_box(top, h, turn, 0.0, op);
                let report = validate_solid(&topo, piece).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let faces = solid_faces(&topo, piece).unwrap();
                assert!(
                    faces
                        .iter()
                        .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_))),
                    "{label}: no cone face"
                );
                let volume = solid_volume(&topo, piece, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
                let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                // The half past the plane, the half before it (the cone is
                // round, so the turn moves neither).
                let (before, past) = match op {
                    BooleanOp::Cut => (PointClassification::Outside, PointClassification::Inside),
                    BooleanOp::Intersect => {
                        (PointClassification::Inside, PointClassification::Outside)
                    }
                    BooleanOp::Fuse => (PointClassification::Inside, PointClassification::Inside),
                };
                for (x, class) in [(-1.5, before), (1.5, past)] {
                    let at = classify_point(
                        &topo,
                        piece,
                        Point3::new(x, 0.0, 0.5),
                        &ClassifyOptions::default(),
                    )
                    .unwrap();
                    assert_eq!(at, class, "{label}: at x = {x}");
                }
            }
        }
    }
}

/// A pointed cone and a frustum cut by planes parallel to the axis, turned so
/// the seam lies on either side of the cut. The listed cases build exactly;
/// the rest (the piece cut off holding the seam) may fall back to a mesh, but
/// never come out exact and wrong.
#[test]
fn cone_cut_parallel_to_its_axis() {
    use PointClassification::{Inside, Outside};
    let exact = |top: f64, turn: f64, off: f64| -> bool {
        let offsets: &[f64] = match (top > 0.0, turn) {
            (false, 0.0) => &[-2.5, -1.5, -1.0, -0.7, 0.3, 0.5, 1.2, 2.0, 2.4],
            (false, 17.0) => &[-2.5, -1.5, -1.0, -0.7],
            (false, _) => &[0.3, 0.5, 1.2, 2.0, 2.4],
            (true, 0.0 | 17.0) => &[-2.5, -1.5, -1.0, -0.7, 0.3, 0.5],
            (true, _) => &[-0.7, 0.3, 0.5, 1.2, 2.0, 2.4],
        };
        offsets.contains(&off)
    };
    for (top, h) in [(0.0_f64, 6.0_f64), (1.0, 4.0)] {
        let whole = PI * h / 3.0 * top.mul_add(top + 3.0, 9.0);
        for turn in [0.0_f64, 17.0, 200.0] {
            for off in [-2.5_f64, -1.5, -1.0, -0.7, 0.3, 0.5, 1.2, 2.0, 2.4] {
                let below = cone_below(top, h, off);
                let must_build = exact(top, turn, off);
                for (op, truth) in [
                    (BooleanOp::Cut, whole - below),
                    (BooleanOp::Intersect, below),
                    (BooleanOp::Fuse, 8000.0 + whole - below),
                ] {
                    let label = format!("top {top}, turned {turn}, x < {off}, {op:?}");
                    let (topo, piece) = cone_and_box(top, h, turn, off, op);
                    let faces = solid_faces(&topo, piece).unwrap();
                    let built = faces
                        .iter()
                        .any(|&f| !topo.face(f).unwrap().surface().is_planar());
                    assert!(built || !must_build, "{label}: fell back to a mesh");
                    if built {
                        let report = validate_solid(&topo, piece).unwrap();
                        assert!(report.is_valid(), "{label}: {:?}", report.issues);
                        // Halfway between the wall and the plane on either
                        // side, a tenth of the way up, where the cone is round
                        // whatever its turn.
                        let z = 0.1 * h;
                        let r = (top - 3.0).mul_add(0.1, 3.0);
                        let (inside_box, beyond) = match op {
                            BooleanOp::Cut => (Outside, Inside),
                            BooleanOp::Intersect => (Inside, Outside),
                            BooleanOp::Fuse => (Inside, Inside),
                        };
                        for (x, class) in [(0.5 * (off - r), inside_box), (0.5 * (off + r), beyond)]
                        {
                            let at = classify_point(
                                &topo,
                                piece,
                                Point3::new(x, 0.0, z),
                                &ClassifyOptions::default(),
                            )
                            .unwrap();
                            assert_eq!(at, class, "{label}: at x = {x}");
                        }
                    }
                    let volume = solid_volume(&topo, piece, 0.01).unwrap();
                    let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
                    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                    // A built piece's hyperbola-trimmed wall is measured along
                    // its boundary; a fallback's mesh errs on the scale of the
                    // whole cone. A fuse is judged by what the cone adds to
                    // the box.
                    let box_volume = if op == BooleanOp::Fuse { 8000.0 } else { 0.0 };
                    let (added, expected) = (volume - box_volume, truth - box_volume);
                    // A fallback's bound never exceeds half the piece, so a
                    // result that lost the piece altogether still fails.
                    let bound = if built {
                        5e-9 * expected
                    } else {
                        (3e-2 * whole).min(0.5 * expected)
                    };
                    assert!(
                        (added - expected).abs() < bound,
                        "{label}: volume {volume}, truth {truth}"
                    );
                }
            }
        }
    }
}

/// The frustum (radius 3 to 1.5 over height 6) and the box over `x > 0.5`,
/// upright, turned about an oblique axis and mirrored through a slanted
/// plane: each piece's volume, with its wall trimmed by the plane's
/// hyperbola, matches the segment integral in every pose.
#[test]
fn frustum_half_space_in_any_pose() {
    let whole = PI * 6.0 / 3.0 * (9.0 + 4.5 + 2.25);
    let beyond = whole - cone_below(1.5, 6.0, 0.5);
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
    for (op, truth) in [
        (BooleanOp::Intersect, beyond),
        (BooleanOp::Cut, whole - beyond),
    ] {
        for pose in ["upright", "turned", "mirrored"] {
            let mut topo = Topology::new();
            let mut cone = make_cone(&mut topo, 3.0, 1.5, 6.0).unwrap();
            let mut block = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
            transform_solid(&mut topo, block, &Mat4::translation(0.5, -10.0, -5.0)).unwrap();
            if pose == "turned" {
                transform_solid(&mut topo, cone, &turn).unwrap();
                transform_solid(&mut topo, block, &turn).unwrap();
            } else if pose == "mirrored" {
                let (at, normal) = (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
                cone = mirror(&mut topo, cone, at, normal).unwrap();
                block = mirror(&mut topo, block, at, normal).unwrap();
            }
            let before = mesh_fallback_count();
            let piece = boolean(&mut topo, op, cone, block).unwrap();
            assert_eq!(mesh_fallback_count(), before, "{op:?} {pose}: fell back");
            let volume = solid_volume(&topo, piece, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-8 * truth,
                "{op:?} {pose}: volume {volume}, truth {truth}"
            );
            assert!(
                validate_solid(&topo, piece).unwrap().is_valid(),
                "{op:?} {pose}: invalid"
            );
            let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{op:?} {pose}: open mesh");
            // (2, 0, 1) lies in the frustum beyond the plane, (0, 0, 1) short of it.
            let place = |p: Point3| match pose {
                "turned" => turn.mul_point(p),
                "mirrored" => {
                    let (at, n) = (
                        Point3::new(0.3, 0.0, 0.0),
                        Vec3::new(1.0, 0.2, 0.1).normalize().unwrap(),
                    );
                    p - n * (2.0 * (p - at).dot(n))
                }
                _ => p,
            };
            let (kept, removed) = if op == BooleanOp::Intersect {
                (Point3::new(2.0, 0.0, 1.0), Point3::new(0.0, 0.0, 1.0))
            } else {
                (Point3::new(0.0, 0.0, 1.0), Point3::new(2.0, 0.0, 1.0))
            };
            let opts = ClassifyOptions::default();
            assert_eq!(
                classify_point(&topo, piece, place(kept), &opts).unwrap(),
                PointClassification::Inside,
                "{op:?} {pose}: lost its material"
            );
            assert_eq!(
                classify_point(&topo, piece, place(removed), &opts).unwrap(),
                PointClassification::Outside,
                "{op:?} {pose}: kept the removed side"
            );
        }
    }
}
