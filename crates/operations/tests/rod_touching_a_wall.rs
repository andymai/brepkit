//! A rod across a cylinder's wall, its side running along the wall. The wall
//! (radius 2, axis z) and the rod (radius 0.3, along y through `(x0, ., 1)`)
//! meet in two loops around the rod. At `x0 = 1.7` the rod's outermost ruling
//! rests on the wall and the loops touch there, as the two halves of one
//! curve crossing itself; just inside, they pass through a narrow neck.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const OPS: [BooleanOp; 3] = [BooleanOp::Cut, BooleanOp::Intersect, BooleanOp::Fuse];

/// The wall turned `turn` about its axis, and the rod turned `spin` about
/// its own axis before it is laid along y through `(x0, ., 1)`, both moved
/// by `(shift, -2 shift, shift / 2)`.
fn wall_and_rod(
    topo: &mut Topology,
    x0: f64,
    spin: f64,
    turn: f64,
    shift: f64,
) -> (SolidId, SolidId) {
    let moved = Mat4::translation(shift, -2.0 * shift, 0.5 * shift);
    let wall = make_cylinder(topo, 2.0, 6.0).unwrap();
    transform_solid(
        topo,
        wall,
        &(moved * Mat4::translation(0.0, 0.0, -3.0) * Mat4::rotation_z(turn)),
    )
    .unwrap();
    let rod = make_cylinder(topo, 0.3, 20.0).unwrap();
    transform_solid(
        topo,
        rod,
        &(moved
            * Mat4::translation(x0, 10.0, 1.0)
            * Mat4::rotation_x(FRAC_PI_2)
            * Mat4::rotation_z(spin)),
    )
    .unwrap();
    (wall, rod)
}

/// The volume the wall and the rod share: across the rod's section, the
/// wall's chord `2 sqrt(4 - x^2)` along y, integrated with
/// `x = x0 + 0.3 sin(t)` so both square roots stay smooth.
fn shared_volume(x0: f64) -> f64 {
    const STEPS: u32 = 20_000;
    let h = PI / f64::from(STEPS);
    let f = |t: f64| {
        let x = 0.3f64.mul_add(t.sin(), x0);
        4.0 * (4.0 - x * x).max(0.0).sqrt() * 0.09 * t.cos() * t.cos()
    };
    let mut sum = f(-FRAC_PI_2) + f(FRAC_PI_2);
    for k in 1..STEPS {
        let t = h.mul_add(f64::from(k), -FRAC_PI_2);
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(t);
    }
    sum * h / 3.0
}

/// Whether a result is exact rather than a mesh fallback, which is all
/// planes and dozens of them. (The fallback counter is process-wide, and
/// other tests in this binary fall back while one runs.)
fn exact(topo: &Topology, solid: SolidId) -> bool {
    let faces = solid_faces(topo, solid).unwrap();
    faces.len() <= 8
        && faces
            .iter()
            .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
}

/// How far the result's curved edges stray from the wall and the rod,
/// both moved by `shift` as in [`wall_and_rod`].
fn section_gap(topo: &Topology, solid: SolidId, x0: f64, shift: f64) -> f64 {
    let mut gap: f64 = 0.0;
    for e in solid_edges(topo, solid).unwrap() {
        let EdgeCurve::NurbsCurve(curve) = topo.edge(e).unwrap().curve() else {
            continue;
        };
        let (t0, t1) = curve.domain();
        for k in 0..=2000 {
            let p = curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 2000.0);
            let (x, y, z) = (p.x() - shift, p.y() + 2.0 * shift, p.z() - 0.5 * shift);
            let off_wall = (x.hypot(y) - 2.0).abs();
            let off_rod = ((x - x0).hypot(z - 1.0) - 0.3).abs();
            gap = gap.max(off_wall.max(off_rod));
        }
    }
    gap
}

/// Each op is exact, valid and watertight, with the volume the shared part
/// gives, and its curved edges on both surfaces.
fn assert_exact(x0: f64, spin: f64, shift: f64) {
    let shared = shared_volume(x0);
    let (wall_volume, rod_volume) = (24.0 * PI, 0.09 * PI * 20.0);
    for (op, truth) in OPS.into_iter().zip([
        wall_volume - shared,
        shared,
        wall_volume + rod_volume - shared,
    ]) {
        let label = format!("{op:?} at x0 = {x0}, spin {spin}, shift {shift}");
        let mut topo = Topology::new();
        let (wall, rod) = wall_and_rod(&mut topo, x0, spin, 0.0, shift);
        let result = boolean(&mut topo, op, wall, rod).unwrap();
        assert!(exact(&topo, result), "{label}: fell back");
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        assert!(
            is_watertight(&tessellate_solid(&topo, result, 0.01).unwrap()),
            "{label}"
        );
        let volume = solid_volume(&topo, result, 0.001).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "{label}: {volume} against {truth}"
        );
        let gap = section_gap(&topo, result, x0, shift);
        assert!(gap < 1e-6, "{label}: edges {gap} off the surfaces");
    }
}

#[test]
fn a_rod_resting_on_a_wall_meets_it_exactly() {
    assert_exact(1.7, 2.5, 0.0);
}

/// Whether the loops touch is read against their own size, so the same pair
/// far from the origin comes out the same.
#[test]
fn a_rod_resting_on_a_wall_far_from_the_origin_meets_it_exactly() {
    assert_exact(1.7, 2.5, 1.0e6);
}

#[test]
fn a_rod_just_inside_a_wall_meets_it_exactly() {
    for x0 in [1.69, 1.698, 1.6999] {
        assert_exact(x0, 0.0, 0.0);
    }
}

/// With the rod's seam on the ruling that rests on the wall, both loops
/// would meet the seam at the one point they share; the exact path declines
/// and the mesh fallback is still a valid solid with the op's volume. The
/// fallback meshes both operands, so a curved solid loses up to a couple of
/// percent to the chords, and the small common a larger share of its own.
#[test]
fn a_rod_resting_on_a_wall_along_its_seam_stays_valid() {
    let shared = shared_volume(1.7);
    let (wall_volume, rod_volume) = (24.0 * PI, 0.09 * PI * 20.0);
    for turn in [0.0, 0.5, 2.0] {
        for (op, truth, within) in [
            (BooleanOp::Cut, wall_volume - shared, 0.02),
            (BooleanOp::Intersect, shared, 0.1),
            (BooleanOp::Fuse, wall_volume + rod_volume - shared, 0.02),
        ] {
            let label = format!("{op:?}, wall turned {turn}");
            let mut topo = Topology::new();
            let (wall, rod) = wall_and_rod(&mut topo, 1.7, 0.0, turn, 0.0);
            let result = boolean(&mut topo, op, wall, rod).unwrap();
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            assert!(
                is_watertight(&tessellate_solid(&topo, result, 0.01).unwrap()),
                "{label}"
            );
            let volume = solid_volume(&topo, result, 0.001).unwrap();
            assert!(
                (volume - truth).abs() < within * truth,
                "{label}: {volume} against {truth}"
            );
        }
    }
}
