//! Chamfers the walking engine builds on a curved face: the ellipse where a
//! slanted plane cuts a rod, which no closed form chamfers, a closed circular
//! rim, and an arc that ends at two planes through its axis.
//!
//! A chamfer of `c` into the cap and `w` along the wall puts each contact in
//! the section square to the edge, `c` from the edge's point along the cap
//! and a chord `w` from it on the wall. The ellipse lies in its plane, so the
//! solid the chamfer takes off sweeps each section's region along the edge
//! by the tube formula: with `a` the distance in from the edge toward its
//! centre of curvature and `b` the distance off the plane, a point stands
//! for `|G'(θ)| (1 - κ a) dθ da db`. The region lies between the wall's trace
//! in the section and the chord from the wall's contact to the cap's.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{PI, TAU};

use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::chamfer_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The rod: radius 2, 5 high.
const R: f64 = 2.0;
/// The cut plane's turn about `x`, through `(0, 0, 4)`.
const TILT: f64 = 0.3;

/// Upright, turned about one axis and about all three, and mirrored.
fn poses() -> Vec<(&'static str, Mat4)> {
    vec![
        ("upright", Mat4::translation(0.0, 0.0, 0.0)),
        ("turned", Mat4::rotation_z(0.7)),
        (
            "tilted",
            Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

/// `f` integrated from `a` to `b` in `panels` equal panels.
fn integral(a: f64, b: f64, panels: u32, f: impl Fn(f64) -> f64) -> f64 {
    let h = (b - a) / f64::from(panels);
    (0..panels)
        .map(|k| {
            let lo = a + h * f64::from(k);
            gauss_legendre_points(20)
                .iter()
                .map(|g| 0.5 * h * g.w * f(0.5 * h * g.x + lo + 0.5 * h))
                .sum::<f64>()
        })
        .sum()
}

/// The rod cut by the slanted plane, placed by `pose`, and its rim.
fn slanted_rod(topo: &mut Topology, pose: &Mat4) -> (SolidId, EdgeId) {
    let rod = make_cylinder(topo, R, 5.0).unwrap();
    let cutter = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    let place = Mat4::translation(0.0, 0.0, 4.0)
        * Mat4::rotation_x(TILT)
        * Mat4::translation(-5.0, -5.0, 0.0);
    transform_solid(topo, cutter, &place).unwrap();
    let cut = boolean(topo, BooleanOp::Cut, rod, cutter).unwrap();
    transform_solid(topo, cut, pose).unwrap();
    let rim = solid_edges(topo, cut)
        .unwrap()
        .into_iter()
        .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Ellipse(_)))
        .expect("the ellipse");
    (cut, rim)
}

/// The volume a chamfer of `cap` into the slanted cap and `wall` along the
/// wall takes off the rod's ellipse, by the tube formula.
fn ellipse_chamfer(cap: f64, wall: f64) -> f64 {
    let t = TILT.tan();
    let up = Vec3::new(0.0, -TILT.sin(), TILT.cos());
    let section = |th: f64| {
        let g = Point3::new(R * th.cos(), R * th.sin(), 4.0 + R * t * th.sin());
        let d1 = Vec3::new(-R * th.sin(), R * th.cos(), R * t * th.cos());
        let d2 = Vec3::new(-R * th.cos(), -R * th.sin(), -R * t * th.sin());
        let speed = d1.length();
        let tangent = d1 * (1.0 / speed);
        let kappa = d1.cross(d2).length() / speed.powi(3);
        // In from the edge toward its centre of curvature.
        let mut inward = up.cross(tangent);
        if inward.dot(d2 - tangent * d2.dot(tangent)) < 0.0 {
            inward = -inward;
        }
        // Where the wall crosses the section `b` off the plane: the root of
        // `|(g + a N + b B)_xy| = R` nearest the edge.
        let wall_at = |b: f64| {
            let q = g + up * b;
            let a2 = inward.x().powi(2) + inward.y().powi(2);
            let b1 = 2.0 * (q.x() * inward.x() + q.y() * inward.y());
            let c0 = q.x().powi(2) + q.y().powi(2) - R * R;
            let root = (b1 * b1 - 4.0 * a2 * c0).sqrt();
            let (r1, r2) = ((-b1 + root) / (2.0 * a2), (-b1 - root) / (2.0 * a2));
            if r1.abs() < r2.abs() { r1 } else { r2 }
        };
        // The wall's contact: the chord `wall` from the edge, below the plane.
        let (mut lo, mut hi) = (-wall, 0.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if wall_at(mid).powi(2) + mid * mid > wall * wall {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let b2 = 0.5 * (lo + hi);
        let a2 = wall_at(b2);
        let strip = |b: f64| {
            let (from, to) = (wall_at(b), cap + (a2 - cap) * (b / b2));
            let f = |a: f64| a - 0.5 * kappa * a * a;
            f(to) - f(from)
        };
        speed * integral(b2, 0.0, 4, strip)
    };
    integral(0.0, TAU, 16, section)
}

/// How far each face of `edge` the chamfer reaches when `chamfer_v2` sets
/// `d1` on its first face in the solid's face order: into the plane, then
/// along the curved face.
fn on_faces(topo: &Topology, solid: SolidId, edge: EdgeId, (d1, d2): (f64, f64)) -> (f64, f64) {
    let first = AdjacencyIndex::build(topo, solid)
        .unwrap()
        .faces_for_edge(edge)[0];
    match topo.face(first).unwrap().surface() {
        FaceSurface::Plane { .. } => (d1, d2),
        _ => (d2, d1),
    }
}

/// Valid, watertight, and `removed` less than `before`, to `1e-6` of what
/// it took off.
fn assert_removed(topo: &Topology, result: SolidId, before: f64, removed: f64, what: &str) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
    let took = before - solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (took - removed).abs() < 1e-6 * removed,
        "{what}: took off {took}, expected {removed}"
    );
}

/// The ellipse where a plane slanted 0.3 rad cuts a rod: `chamfer` at 0.5
/// and `chamfer_v2` at 0.5 and 0.3 walk a ruled band round it, which cuts
/// the cap and the wall along its contacts.
#[test]
fn the_ellipse_where_a_slanted_plane_cuts_a_rod_takes_a_chamfer() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (rod, rim) = slanted_rod(&mut topo, &pose);
        let before = solid_volume(&topo, rod, 0.001).unwrap();
        let result = chamfer(&mut topo, rod, &[rim], 0.5).unwrap();
        assert_removed(&topo, result, before, ellipse_chamfer(0.5, 0.5), name);
        let bands = solid_faces(&topo, result)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
            .count();
        assert_eq!(bands, 1, "{name}");

        let (cap, wall) = on_faces(&topo, rod, rim, (0.5, 0.3));
        let built = chamfer_v2(&mut topo, rod, &[rim], 0.5, 0.3).unwrap();
        assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
        assert_removed(&topo, built.solid, before, ellipse_chamfer(cap, wall), name);
    }
}

/// The walking chamfer cuts a cylinder's and a frustum's bottom rims
/// exactly, along a cone: the ring their section triangle sweeps round the
/// axis.
#[test]
fn the_walking_chamfer_cuts_a_closed_rim_exactly() {
    let (d1, d2) = (0.4, 0.25);
    // The frustum's wall rises 4 as it closes in by 2.
    let slope = 2.0_f64.atan2(1.0);
    for (frustum, rise) in [(false, 0.5 * PI), (true, slope)] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = if frustum {
                make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap()
            } else {
                make_cylinder(&mut topo, 3.0, 4.0).unwrap()
            };
            let rim = solid_edges(&topo, solid)
                .unwrap()
                .into_iter()
                .find(|&e| {
                    matches!(topo.edge(e).unwrap().curve(),
                        EdgeCurve::Circle(c) if c.center().z().abs() < 1e-9)
                })
                .unwrap();
            let (cap, wall) = on_faces(&topo, solid, rim, (d1, d2));
            transform_solid(&mut topo, solid, &pose).unwrap();
            let before = solid_volume(&topo, solid, 0.001).unwrap();
            let mut builder = ChamferBuilder::new(&mut topo, solid);
            builder.add_edges_asymmetric(&[rim], d1, d2);
            let built = builder.build().unwrap();
            assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
            let area = 0.5 * cap * wall * rise.sin();
            let centroid = 3.0 - (cap + wall * rise.cos()) / 3.0;
            let what = format!("frustum {frustum} {name}");
            assert_removed(&topo, built.solid, before, TAU * centroid * area, &what);
        }
    }
}

/// An arc of a sector of a rod, between two planes through its axis: the
/// chamfer's cone runs from one plane to the other, each of which takes its
/// end section.
#[test]
fn a_sectors_arc_ends_on_its_radial_planes() {
    let (d1, d2) = (0.4, 0.25);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, R, 4.0).unwrap();
        let block = make_box(&mut topo, 3.0, 3.0, 6.0).unwrap();
        let place = Mat4::rotation_z(0.5) * Mat4::translation(0.0, 0.0, -1.0);
        transform_solid(&mut topo, block, &place).unwrap();
        let sector = boolean(&mut topo, BooleanOp::Intersect, rod, block).unwrap();
        let arc = solid_edges(&topo, sector)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), EdgeCurve::Circle(_))
                    && (topo.vertex(edge.start()).unwrap().point().z() - 4.0).abs() < 1e-9
            })
            .unwrap();
        let (cap, wall) = on_faces(&topo, sector, arc, (d1, d2));
        transform_solid(&mut topo, sector, &pose).unwrap();
        let before = solid_volume(&topo, sector, 0.001).unwrap();
        let built = chamfer_v2(&mut topo, sector, &[arc], d1, d2).unwrap();
        assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
        // A quarter turn of the ring.
        let removed = 0.25 * PI * cap * wall * (R - cap / 3.0);
        assert_removed(&topo, built.solid, before, removed, name);
    }
}
