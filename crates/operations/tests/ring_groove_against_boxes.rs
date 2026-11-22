//! A rod with a groove, the cut of a torus ring round its axis, against
//! boxes that take parts of the groove: a flat across the groove's seam, the
//! top off down to the middle of the groove, a slot across it, a notch out
//! of its side and a corner whose floor lies in it. Each is cut, intersected and fused, upright, turned and
//! mirrored, without the mesh fallback, and measured against an integral.
//!
//! The groove's torus band is a whole turn about the axis, with a meridian
//! arc for its seam. Traced glued at that seam, a rim piece over half a turn
//! long read as running back the short way, and a flat that took off the seam
//! fell back to a mesh. The plane through the middle of the groove meets the
//! band in a closed circle whose way round was read from the circle's own
//! angle origin: a quarter of its domain on from there was the start's
//! antipode, which reads either way. And a groove over more than half the
//! tube (a ring narrower than the rod) read as a flat polygon in the ray
//! cast, its tube angles centred on the mean of its edges' start vertices,
//! which leans toward one rim. A box wall's loop round the tube was cut at
//! the box's edges alone, so its part over the top of the tube, off the
//! band, stayed as a section of the band.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{PI, TAU};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 6.0;
const HEIGHT: f64 = 16.0;
const TUBE: f64 = 0.32;
const GROOVE_Z: f64 = 2.0;

/// An axis-aligned box by its low corner and its size.
#[derive(Clone, Copy, Debug)]
struct Block {
    lo: [f64; 3],
    size: [f64; 3],
}

impl Block {
    fn hi(&self, k: usize) -> f64 {
        self.lo[k] + self.size[k]
    }

    fn holds_xy(&self, x: f64, y: f64) -> bool {
        x > self.lo[0] && x < self.hi(0) && y > self.lo[1] && y < self.hi(1)
    }
}

/// Whether a point at distance `rho` from the axis and height `z`, inside
/// the block's outline or not, is in the result.
fn in_result(op: BooleanOp, major: f64, block: &Block, rho: f64, z: f64, in_xy: bool) -> bool {
    let in_rod = rho < RADIUS && z > 0.0 && z < HEIGHT;
    let in_ring = (rho - major).hypot(z - GROOVE_Z) < TUBE;
    let in_block = in_xy && z > block.lo[2] && z < block.hi(2);
    let grooved = in_rod && !in_ring;
    match op {
        BooleanOp::Cut => grooved && !in_block,
        BooleanOp::Intersect => grooved && in_block,
        BooleanOp::Fuse => grooved || in_block,
    }
}

/// The turn of the circle of radius `rho` about the axis inside the block's
/// outline.
fn turn_in_block(rho: f64, block: &Block) -> f64 {
    let mut angles = vec![0.0, TAU];
    for x in [block.lo[0], block.hi(0)] {
        if x.abs() < rho {
            let a = (x / rho).acos();
            angles.extend([a, TAU - a]);
        }
    }
    for y in [block.lo[1], block.hi(1)] {
        if y.abs() < rho {
            let a = (y / rho).asin();
            angles.extend([a.rem_euclid(TAU), (PI - a).rem_euclid(TAU)]);
        }
    }
    angles.sort_by(f64::total_cmp);
    angles
        .windows(2)
        .filter(|w| {
            let a = f64::midpoint(w[0], w[1]);
            block.holds_xy(rho * a.cos(), rho * a.sin())
        })
        .map(|w| w[1] - w[0])
        .sum()
}

/// The length of the result's line at distance `rho` from the axis, inside
/// the block's outline or not.
fn height_in_result(op: BooleanOp, major: f64, block: &Block, rho: f64, in_xy: bool) -> f64 {
    let mut zs = vec![-1e3, 1e3, 0.0, HEIGHT, block.lo[2], block.hi(2)];
    let off = rho - major;
    if off.abs() < TUBE {
        let half = (TUBE * TUBE - off * off).sqrt();
        zs.extend([GROOVE_Z - half, GROOVE_Z + half]);
    }
    zs.sort_by(f64::total_cmp);
    zs.windows(2)
        .filter(|w| in_result(op, major, block, rho, f64::midpoint(w[0], w[1]), in_xy))
        .map(|w| w[1] - w[0])
        .sum()
}

/// Tanh-sinh quadrature over `[a, b]`, exact to rounding for a smooth
/// integrand with square-root ends.
fn tanh_sinh(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let h = 1.0 / 64.0;
    let half = 0.5 * (b - a);
    let mut sum = 0.0;
    for k in -240..=240 {
        let t = f64::from(k) * h;
        let s = 0.5 * PI * t.sinh();
        let weight = 0.5 * PI * t.cosh() / (s.cosh() * s.cosh());
        // The distance from the nearer end, without cancellation.
        let gap = half / (s.abs().exp() * s.cosh());
        let x = if s >= 0.0 { b - gap } else { a + gap };
        if x > a && x < b {
            sum += weight * f(x);
        }
    }
    sum * h * half
}

/// The result's volume, integrated over the distance from the axis: at
/// each distance the result is a turn of the circle inside the block's
/// outline times one height, and the rest of the turn times another. The
/// integrand is smooth between the distances where a box edge or corner,
/// the rod or the ring's tube starts or ends.
fn volume_of(op: BooleanOp, major: f64, block: &Block) -> f64 {
    let mut breaks = vec![0.0, RADIUS, major - TUBE, major + TUBE];
    for x in [block.lo[0], block.hi(0)] {
        breaks.push(x.abs());
        for y in [block.lo[1], block.hi(1)] {
            breaks.extend([y.abs(), x.hypot(y)]);
        }
    }
    for z in [block.lo[2], block.hi(2), 0.0, HEIGHT] {
        let dz = z - GROOVE_Z;
        if dz.abs() < TUBE {
            let half = (TUBE * TUBE - dz * dz).sqrt();
            breaks.extend([major - half, major + half]);
        }
    }
    breaks.retain(|r| *r >= 0.0);
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let slice = |rho: f64| {
        let turn = turn_in_block(rho, block);
        rho * (turn * height_in_result(op, major, block, rho, true)
            + (TAU - turn) * height_in_result(op, major, block, rho, false))
    };
    breaks
        .windows(2)
        .map(|w| tanh_sinh(&slice, w[0], w[1]))
        .sum()
}

fn turned() -> Mat4 {
    Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9)
}

fn mirrored() -> Mat4 {
    Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_y(0.4)
}

fn poses() -> [(&'static str, Mat4); 3] {
    [
        ("upright", Mat4::identity()),
        ("turned", turned()),
        ("mirrored", mirrored()),
    ]
}

fn check(
    major: f64,
    block: &Block,
    op: BooleanOp,
    pose: &Mat4,
    label: &str,
) -> (Topology, SolidId) {
    let label = format!("{label}, ring of radius {major}, {op:?}");
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let ring = make_torus(&mut topo, major, TUBE, 32).unwrap();
    transform_solid(&mut topo, ring, &Mat4::translation(0.0, 0.0, GROOVE_Z)).unwrap();
    let [x, y, z] = block.size;
    let tool = make_box(&mut topo, x, y, z).unwrap();
    let [x, y, z] = block.lo;
    transform_solid(&mut topo, tool, &Mat4::translation(x, y, z)).unwrap();
    for solid in [rod, ring, tool] {
        transform_solid(&mut topo, solid, pose).unwrap();
    }
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let grooved = boolean_with_options(&mut topo, BooleanOp::Cut, rod, ring, exact)
        .unwrap_or_else(|e| panic!("{label}: the groove: {e}"));
    let solid = boolean_with_options(&mut topo, op, grooved, tool, exact)
        .unwrap_or_else(|e| panic!("{label}: {e}"));

    let report = validate_solid(&topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(&topo, solid, 0.01).unwrap()),
        "{label}: mesh not watertight"
    );
    for face in solid_faces(&topo, solid).unwrap() {
        let surface = topo.face(face).unwrap().surface();
        assert!(
            matches!(
                surface,
                FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Torus(_)
            ),
            "{label}: a {} face",
            surface.type_tag()
        );
    }

    let volume = solid_volume(&topo, solid, 0.001).unwrap();
    let want = volume_of(op, major, block);
    assert!(
        (volume - want).abs() <= 1e-6 * want,
        "{label}: volume {volume}, want {want}"
    );

    // Points round the groove, at the seam's turn and five others, kept
    // clear of every surface.
    let tol = brepkit_math::tolerance::Tolerance::new();
    for rho in [3.0, major - 0.15, RADIUS - 0.05] {
        for k in 0..6 {
            let a = f64::from(k) * PI / 3.0;
            let (px, py) = (rho * a.cos(), rho * a.sin());
            for pz in [1.0, GROOVE_Z - 0.15, GROOVE_Z + 0.15] {
                let clear = [
                    (rho - RADIUS).abs(),
                    ((rho - major).hypot(pz - GROOVE_Z) - TUBE).abs(),
                    (px - block.lo[0]).abs(),
                    (px - block.hi(0)).abs(),
                    (py - block.lo[1]).abs(),
                    (py - block.hi(1)).abs(),
                    (pz - block.lo[2]).abs(),
                    (pz - block.hi(2)).abs(),
                ]
                .into_iter()
                .fold(f64::INFINITY, f64::min);
                if clear < 0.02 {
                    continue;
                }
                let inside = in_result(op, major, block, rho, pz, block.holds_xy(px, py));
                let p = Point3::new(px, py, pz);
                let class =
                    classify_point(&topo, solid, pose.mul_point(p), 0.01, tol.linear).unwrap();
                let want = if inside {
                    PointClassification::Inside
                } else {
                    PointClassification::Outside
                };
                assert_eq!(class, want, "{label}: point {p:?}");
            }
        }
    }
    (topo, solid)
}

fn check_all(majors: &[f64], block: &Block, label: &str) {
    for &major in majors {
        for op in [BooleanOp::Cut, BooleanOp::Intersect, BooleanOp::Fuse] {
            for (pose_name, pose) in poses() {
                check(major, block, op, &pose, &format!("{label}, {pose_name}"));
            }
        }
    }
}

/// The volume integral against closed forms: the ring inside the rod is a
/// cavity of `2 pi^2 R r^2`, and half the grooved rod is half its volume.
#[test]
fn the_volume_integral_meets_closed_forms() {
    let away = Block {
        lo: [100.0, 100.0, 100.0],
        size: [1.0, 1.0, 1.0],
    };
    let cavity = volume_of(BooleanOp::Cut, 5.5, &away);
    let want = PI * RADIUS * RADIUS * HEIGHT - 2.0 * PI * PI * 5.5 * TUBE * TUBE;
    assert!(
        (cavity - want).abs() <= 1e-10 * want,
        "{cavity} against {want}"
    );

    let inner = Block {
        lo: [-1.0, -1.0, 5.0],
        size: [2.0, 2.0, 2.0],
    };
    let cube = volume_of(BooleanOp::Intersect, RADIUS, &inner);
    assert!((cube - 8.0).abs() <= 1e-10, "{cube}");

    let half = Block {
        lo: [0.0, -10.0, -2.0],
        size: [20.0, 20.0, 20.0],
    };
    let whole = volume_of(BooleanOp::Cut, RADIUS, &away);
    let part = volume_of(BooleanOp::Intersect, RADIUS, &half);
    assert!(
        (2.0 * part - whole).abs() <= 1e-10 * whole,
        "{part} against {whole}"
    );
}

/// The flat past `x = 2` takes off the band's seam at `u = 0`: its sections
/// at `u = +-1.23` leave a rim piece 3.82 long between them.
#[test]
fn a_flat_that_takes_off_the_grooves_seam() {
    let block = Block {
        lo: [2.0, -10.0, -2.0],
        size: [20.0, 20.0, 20.0],
    };
    check_all(&[5.9, 6.0, 6.1, 6.3], &block, "flat past x = 2");
}

/// The plane `z = 2` meets the band in the circle round the inside of its
/// tube, through the seam's middle vertex.
#[test]
fn the_top_off_down_to_the_middle_of_the_groove() {
    let block = Block {
        lo: [-10.0, -10.0, 2.0],
        size: [20.0, 20.0, 20.0],
    };
    check_all(&[5.9, 6.0], &block, "top off");
}

#[test]
fn a_slot_across_the_groove() {
    let block = Block {
        lo: [-1.0, -10.0, 1.0],
        size: [2.0, 20.0, 2.0],
    };
    check_all(&[5.9], &block, "slot");
}

#[test]
fn a_notch_out_of_the_grooves_side() {
    let block = Block {
        lo: [3.0, 3.0, 1.5],
        size: [5.0, 5.0, 1.0],
    };
    check_all(&[5.9], &block, "notch");
}

/// The box's floor lies inside the groove. Each wall meets the whole tube in
/// a loop, of which only the part between the band's rims lies on the band:
/// with a ring of radius 5.9 or 6 the rest runs on over the top of the tube.
/// With a ring of radius 6.1 or 6.3 the band's lower rim runs 0.2 or 0.011
/// below the floor, and the wall's arc between the two lies off the wall.
#[test]
fn a_corner_whose_floor_lies_in_the_groove() {
    let block = Block {
        lo: [2.0, 2.0, 1.9],
        size: [10.0, 10.0, 10.0],
    };
    check_all(&[5.9, 6.0, 6.1, 6.3], &block, "corner");
}

/// A flat through the axis meets the groove's band in two meridian
/// circles. The pieces of the band keep them as exact circular arcs. Each
/// circle's arc round the outside of the tube lies off the band; the box of
/// the band's rims, taken in the world's axes, held it when turned, and the
/// ring of radius 5.9 fell back to a mesh.
#[test]
fn a_flat_through_the_axis_keeps_the_grooves_meridians_circular() {
    let block = Block {
        lo: [0.0, -10.0, -2.0],
        size: [20.0, 20.0, 20.0],
    };
    for major in [5.9, 6.0, 6.1, 6.3] {
        for op in [BooleanOp::Cut, BooleanOp::Intersect, BooleanOp::Fuse] {
            for (pose_name, pose) in poses() {
                let label = format!("flat through the axis, {pose_name}");
                let (topo, solid) = check(major, &block, op, &pose, &label);
                for face in solid_faces(&topo, solid).unwrap() {
                    let face = topo.face(face).unwrap();
                    for wire in
                        std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
                    {
                        for oe in topo.wire(wire).unwrap().edges() {
                            let curve = topo.edge(oe.edge()).unwrap().curve();
                            assert!(
                                matches!(curve, EdgeCurve::Line | EdgeCurve::Circle(_)),
                                "{label}, ring of radius {major}, {op:?}: a {} edge",
                                curve.type_tag()
                            );
                        }
                    }
                }
            }
        }
    }
}
