//! A ring (a rod less a coaxial rod) against a block whose side faces pass
//! through the ring's axis. One side face meets each rim of the ring at the
//! rim's own vertex, and the next side face meets the rim a quarter turn on.
//! The rims come from the cut that made the ring, so their vertex is not at
//! the origin of the circle's parameter. The crossing at that origin was
//! read as the end of the rim's block and dropped: the rim was not split
//! there, the ring's end face split into nothing, and the boolean fell back
//! to a mesh. A sheet-metal bend is made the same way: a tube of the bend
//! radii, intersected with a wedge on the tube's axis.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const POSES: [&str; 5] = ["upright", "turned", "shifted", "mirrored", "scaled"];

fn turn() -> Mat4 {
    Mat4::translation(0.4, -0.3, 0.2) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3)
}

fn shift() -> Mat4 {
    Mat4::translation(-37.2, 5.5, 140.0)
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1))
}

fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            transform_solid(topo, solid, &turn()).unwrap();
            solid
        }
        "shifted" => {
            transform_solid(topo, solid, &shift()).unwrap();
            solid
        }
        "mirrored" => {
            let (at, normal) = mirror_plane();
            mirror(topo, solid, at, normal).unwrap()
        }
        "scaled" => {
            transform_solid(topo, solid, &Mat4::scale(-1.0, 1.0, 1.0)).unwrap();
            solid
        }
        _ => solid,
    }
}

fn placed(p: Point3, pose: &str) -> Point3 {
    match pose {
        "turned" => turn().mul_point(p),
        "shifted" => shift().mul_point(p),
        "mirrored" => {
            let (at, normal) = mirror_plane();
            let unit = normal.normalize().unwrap();
            p - unit * (2.0 * (p - at).dot(unit))
        }
        "scaled" => Point3::new(-p.x(), p.y(), p.z()),
        _ => p,
    }
}

/// A ring of radii `inner` and `outer` on the z axis, from 0 to `height`.
fn ring(topo: &mut Topology, inner: f64, outer: f64, height: f64) -> SolidId {
    let rod = make_cylinder(topo, outer, height).unwrap();
    let bore = make_cylinder(topo, inner, height).unwrap();
    boolean(topo, BooleanOp::Cut, rod, bore).unwrap()
}

/// A box with one corner at `at`.
fn block(topo: &mut Topology, size: [f64; 3], at: [f64; 3]) -> SolidId {
    let b = make_box(topo, size[0], size[1], size[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(at[0], at[1], at[2])).unwrap();
    b
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId, label: &str) -> SolidId {
    let opts = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    boolean_with_options(topo, op, a, b, opts).unwrap_or_else(|e| panic!("{label}: {e}"))
}

/// `(planes, cylinders, other faces)`.
fn census(topo: &Topology, solid: SolidId) -> (usize, usize, usize) {
    let mut n = (0, 0, 0);
    for f in solid_faces(topo, solid).unwrap() {
        match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { .. } => n.0 += 1,
            FaceSurface::Cylinder(_) => n.1 += 1,
            FaceSurface::Nurbs(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => n.2 += 1,
        }
    }
    n
}

struct Want<'a> {
    faces: (usize, usize, usize),
    volume: f64,
    inside: &'a [[f64; 3]],
    outside: &'a [[f64; 3]],
}

fn check(topo: &Topology, solid: SolidId, pose: &str, label: &str, want: &Want) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{label}: mesh not watertight"
    );
    assert_eq!(census(topo, solid), want.faces, "{label}: faces");
    let volume = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (volume - want.volume).abs() <= 1e-6 * want.volume,
        "{label}: volume {volume}, want {}",
        want.volume
    );
    for (points, side) in [
        (want.inside, PointClassification::Inside),
        (want.outside, PointClassification::Outside),
    ] {
        for p in points {
            let at = placed(Point3::new(p[0], p[1], p[2]), pose);
            let got = classify_point(topo, solid, at, 0.01, 1e-7).unwrap();
            assert_eq!(got, side, "{label}: point {p:?}");
        }
    }
}

/// The issue's ring and block: the block's side faces `x = 0` and `y = 0`
/// pass through the axis, and its ends lie in the planes of the ring's ends.
#[test]
fn a_block_with_its_edge_on_the_axis_takes_a_quarter_of_a_ring() {
    let whole = PI * (9.0 - 4.0) * 10.0;
    for pose in POSES {
        for (name, op) in [
            ("intersect", BooleanOp::Intersect),
            ("cut", BooleanOp::Cut),
            ("fuse", BooleanOp::Fuse),
        ] {
            let label = format!("{name}, {pose}");
            let mut topo = Topology::new();
            let r = ring(&mut topo, 2.0, 3.0, 10.0);
            let b = block(&mut topo, [5.0, 5.0, 10.0], [0.0, 0.0, 0.0]);
            let (r, b) = (posed(&mut topo, r, pose), posed(&mut topo, b, pose));
            let result = exact(&mut topo, op, r, b, &label);
            let quarter = [[1.8, 1.8, 5.0], [2.4, 0.3, 9.5], [0.3, 2.9, 0.5]];
            let rest = [[-1.8, 1.8, 5.0], [-2.4, -0.3, 9.5], [0.3, -2.9, 0.5]];
            let corner = [[4.5, 4.5, 5.0], [0.5, 0.5, 5.0], [3.5, 0.5, 9.5]];
            let off = [[0.0, 0.0, 12.0], [-1.0, -1.0, 5.0], [1.8, 1.8, -0.5]];
            let want = match op {
                BooleanOp::Intersect => Want {
                    faces: (4, 2, 0),
                    volume: whole / 4.0,
                    inside: &quarter,
                    outside: &[rest.as_slice(), &corner, &off].concat(),
                },
                BooleanOp::Cut => Want {
                    faces: (4, 2, 0),
                    volume: whole * 0.75,
                    inside: &rest,
                    outside: &[quarter.as_slice(), &corner, &off].concat(),
                },
                BooleanOp::Fuse => Want {
                    // Each end of the ring is one face with an end of the
                    // block, and the ring divides each of the block's two
                    // sides through the axis in two.
                    faces: (8, 2, 0),
                    volume: 250.0 + whole * 0.75,
                    inside: &[quarter.as_slice(), &rest, &corner].concat(),
                    outside: &off,
                },
            };
            check(&topo, result, pose, &label, &want);
        }
    }
}

/// The same block longer than the ring, so no face of one lies in a face of
/// the other, and a block that takes half the ring: its one side face
/// through the axis meets each rim twice, away from the rim's vertex.
#[test]
fn a_block_past_the_rings_ends_takes_a_quarter_and_a_half() {
    let whole = PI * (9.0 - 4.0) * 10.0;
    for pose in POSES {
        for (name, size, at, part) in [
            ("quarter", [6.0, 6.0, 12.0], [0.0, 0.0, -1.0], 0.25),
            ("half", [6.0, 12.0, 12.0], [0.0, -6.0, -1.0], 0.5),
            (
                "half between the ends",
                [6.0, 12.0, 10.0],
                [0.0, -6.0, 0.0],
                0.5,
            ),
        ] {
            let label = format!("{name}, {pose}");
            let mut topo = Topology::new();
            let r = ring(&mut topo, 2.0, 3.0, 10.0);
            let b = block(&mut topo, size, at);
            let (r, b) = (posed(&mut topo, r, pose), posed(&mut topo, b, pose));
            let result = exact(&mut topo, BooleanOp::Intersect, r, b, &label);
            let want = Want {
                faces: (4, 2, 0),
                volume: whole * part,
                inside: &[[1.8, 1.8, 5.0], [2.4, 0.3, 9.5]],
                outside: &[[-1.8, 1.8, 5.0], [0.5, 0.5, 5.0], [1.8, 1.8, -0.5]],
            };
            check(&topo, result, pose, &label, &want);
        }
    }
}

/// A sheet-metal bend: a tube of the bend radii along `y`, intersected with
/// the wedge between a half block on the tube's axis and that half block
/// turned about the axis.
#[test]
fn a_tube_and_a_wedge_on_its_axis_make_a_bend() {
    let (inner, thickness, width) = (2.0, 1.0, 10.0);
    let outer = inner + thickness;
    for pose in POSES {
        for degrees in [90.0_f64, 60.0, 120.0] {
            let label = format!("bend {degrees}, {pose}");
            let mut topo = Topology::new();
            let tube = ring(&mut topo, inner, outer, width);
            // The tube's axis goes from z to y.
            let along_y = Mat4::rotation_x(-PI / 2.0);
            transform_solid(&mut topo, tube, &along_y).unwrap();
            let span = outer.mul_add(2.0, 2.0);
            let half = |topo: &mut Topology| {
                block(topo, [span, width + 2.0, 2.0 * span], [0.0, -1.0, -span])
            };
            let half_a = half(&mut topo);
            let half_b = half(&mut topo);
            transform_solid(
                &mut topo,
                half_b,
                &Mat4::rotation_y((180.0 - degrees).to_radians()),
            )
            .unwrap();
            let wedge = exact(
                &mut topo,
                BooleanOp::Intersect,
                half_a,
                half_b,
                &format!("{label}, wedge"),
            );
            let (tube, wedge) = (posed(&mut topo, tube, pose), posed(&mut topo, wedge, pose));
            // The bend sweeps from straight down (-z) toward +x.
            let at =
                |radius: f64, angle: f64, y: f64| [radius * angle.sin(), y, -radius * angle.cos()];
            let sweep = degrees.to_radians();
            let bend = [at(2.5, sweep * 0.5, 5.0), at(2.9, sweep * 0.1, 0.5)];
            let rest = [at(2.5, -0.2, 5.0), at(2.5, sweep + 0.2, 5.0)];
            let off = [at(1.5, sweep * 0.5, 5.0), at(2.5, sweep * 0.5, -0.5)];
            let sector = sweep / 2.0 * (outer * outer - inner * inner) * width;
            let whole = PI * (outer * outer - inner * inner) * width;
            for (name, op, volume, inside, outside) in [
                ("bend", BooleanOp::Intersect, sector, bend, rest),
                ("rest", BooleanOp::Cut, whole - sector, rest, bend),
            ] {
                let label = format!("{label}, {name}");
                let result = exact(&mut topo, op, tube, wedge, &label);
                let want = Want {
                    faces: (4, 2, 0),
                    volume,
                    inside: &inside,
                    outside: &[outside.as_slice(), &off].concat(),
                };
                check(&topo, result, pose, &label, &want);
            }
        }
    }
}

/// The part of a disc of radius `r` about the origin with `x` and `y` both
/// above `a`.
fn corner_of_disc(r: f64, a: f64) -> f64 {
    let top = (r * r - a * a).sqrt();
    let under = |x: f64| 0.5 * (x * (r * r - x * x).sqrt() + r * r * (x / r).asin());
    under(top) - under(a) - a * (top - a)
}

/// A block whose side faces pass half a unit from the axis: each meets a
/// rim once, away from the rim's vertex and less than half a turn from the
/// other, so the rest of the rim is one piece of more than half a turn.
#[test]
fn a_block_beside_the_axis_takes_a_corner_of_a_ring() {
    let corner = (corner_of_disc(3.0, 0.5) - corner_of_disc(2.0, 0.5)) * 10.0;
    let whole = PI * (9.0 - 4.0) * 10.0;
    for pose in POSES {
        let mut topo = Topology::new();
        let r = ring(&mut topo, 2.0, 3.0, 10.0);
        let b = block(&mut topo, [6.0, 6.0, 12.0], [0.5, 0.5, -1.0]);
        let (r, b) = (posed(&mut topo, r, pose), posed(&mut topo, b, pose));
        let taken = [[1.8, 1.8, 5.0], [2.4, 0.7, 9.5]];
        let rest = [[-1.8, 1.8, 5.0], [2.4, 0.3, 9.5], [0.3, 2.4, 0.5]];
        let off = [[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [1.8, 1.8, -0.5]];
        for (name, op, volume, inside, outside) in [
            (
                "corner",
                BooleanOp::Intersect,
                corner,
                taken.as_slice(),
                rest.as_slice(),
            ),
            ("rest", BooleanOp::Cut, whole - corner, &rest, &taken),
        ] {
            let label = format!("{name}, {pose}");
            let result = exact(&mut topo, op, r, b, &label);
            let want = Want {
                faces: (4, 2, 0),
                volume,
                inside,
                outside: &[outside, &off].concat(),
            };
            check(&topo, result, pose, &label, &want);
        }
    }
}
