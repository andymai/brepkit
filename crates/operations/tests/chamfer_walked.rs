//! Chamfers the walking engine builds on a curved face: the ellipse where a
//! slanted plane cuts a rod, which no closed form chamfers, a closed circular
//! rim, arcs of a circle and of the ellipse that end at planes square to
//! them, the rim of a ball cut by a plane, and a rim round a torus's tube.
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

use std::collections::HashSet;
use std::f64::consts::{PI, TAU};

use brepkit_blend::chamfer_builder::ChamferBuilder;
use brepkit_check::validate::{CheckId, ValidateOptions};
use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::chamfer_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::builder::make_planar_face_from_wire;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

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

/// The rod cut by the slanted plane.
fn slanted_cut(topo: &mut Topology) -> SolidId {
    let rod = make_cylinder(topo, R, 5.0).unwrap();
    let cutter = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    let place = Mat4::translation(0.0, 0.0, 4.0)
        * Mat4::rotation_x(TILT)
        * Mat4::translation(-5.0, -5.0, 0.0);
    transform_solid(topo, cutter, &place).unwrap();
    boolean(topo, BooleanOp::Cut, rod, cutter).unwrap()
}

/// The rod cut by the slanted plane, placed by `pose`, and its rim.
fn slanted_rod(topo: &mut Topology, pose: &Mat4) -> (SolidId, EdgeId) {
    let cut = slanted_cut(topo);
    transform_solid(topo, cut, pose).unwrap();
    let rim = solid_edges(topo, cut)
        .unwrap()
        .into_iter()
        .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Ellipse(_)))
        .expect("the ellipse");
    (cut, rim)
}

/// The volume a chamfer of `cap` into the slanted cap and `wall` along the
/// wall takes off the rod's ellipse from `from` to `to` round the axis, by
/// the tube formula.
fn ellipse_chamfer(cap: f64, wall: f64, (from, to): (f64, f64)) -> f64 {
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
    integral(from, to, 16, section)
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
    assert_removed_measured(topo, result, before, removed, what, 0.001);
}

/// [`assert_removed`], with the result measured at `deflection`.
fn assert_removed_measured(
    topo: &Topology,
    result: SolidId,
    before: f64,
    removed: f64,
    what: &str,
    deflection: f64,
) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
    let took = before - solid_volume(topo, result, deflection).unwrap();
    assert!(
        (took - removed).abs() < 1e-6 * removed,
        "{what}: took off {took}, expected {removed}"
    );
}

/// The edges of `solid` and the curved face of its `edge`, by its surface
/// written out, which a trim of the face keeps.
fn wall_of(topo: &Topology, solid: SolidId, edge: EdgeId) -> (HashSet<EdgeId>, String) {
    let wall = AdjacencyIndex::build(topo, solid)
        .unwrap()
        .faces_for_edge(edge)
        .iter()
        .map(|&f| topo.face(f).unwrap().surface())
        .find(|surface| !surface.is_planar())
        .expect("a curved face");
    let edges = solid_edges(topo, solid).unwrap().into_iter().collect();
    (edges, format!("{wall:?}"))
}

/// The curves of the edges that the chamfer added to the wall in `result`,
/// each with its pcurve there. Before that, it asserts the same-parameter
/// check on each edge of `result`: the edge's pcurve read at one fraction of
/// its span gives the point of its curve at that fraction of the edge's span.
fn wall_contacts(
    topo: &Topology,
    (old, wall): &(HashSet<EdgeId>, String),
    result: SolidId,
    what: &str,
) -> Vec<&'static str> {
    let report =
        brepkit_check::validate::validate_solid(topo, result, &ValidateOptions::default()).unwrap();
    let off: Vec<_> = report
        .issues
        .iter()
        .filter(|issue| issue.check == CheckId::EdgeSameParameter)
        .collect();
    assert!(off.is_empty(), "{what}: {off:?}");
    let mut curves = Vec::new();
    for face in solid_faces(topo, result).unwrap() {
        let f = topo.face(face).unwrap();
        if format!("{:?}", f.surface()) != *wall {
            continue;
        }
        for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oe in topo.wire(w).unwrap().edges() {
                if !old.contains(&oe.edge()) && topo.pcurves().get(oe.edge(), face).is_some() {
                    curves.push(topo.edge(oe.edge()).unwrap().curve().type_tag());
                }
            }
        }
    }
    curves
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
        let side = wall_of(&topo, rod, rim);
        let result = chamfer(&mut topo, rod, &[rim], 0.5).unwrap();
        assert_removed(
            &topo,
            result,
            before,
            ellipse_chamfer(0.5, 0.5, (0.0, TAU)),
            name,
        );
        let contacts = wall_contacts(&topo, &side, result, name);
        assert_eq!(contacts, ["nurbs_curve"], "{name}");
        let bands = solid_faces(&topo, result)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
            .count();
        assert_eq!(bands, 1, "{name}");

        let (cap, wall) = on_faces(&topo, rod, rim, (0.5, 0.3));
        let built = chamfer_v2(&mut topo, rod, &[rim], 0.5, 0.3).unwrap();
        assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
        assert_removed(
            &topo,
            built.solid,
            before,
            ellipse_chamfer(cap, wall, (0.0, TAU)),
            name,
        );
        let contacts = wall_contacts(&topo, &side, built.solid, name);
        assert_eq!(contacts, ["nurbs_curve"], "{name}");
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
            let side = wall_of(&topo, solid, rim);
            let mut builder = ChamferBuilder::new(&mut topo, solid);
            builder.add_edges_asymmetric(&[rim], d1, d2);
            let built = builder.build().unwrap();
            assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
            let area = 0.5 * cap * wall * rise.sin();
            let centroid = 3.0 - (cap + wall * rise.cos()) / 3.0;
            let what = format!("frustum {frustum} {name}");
            assert_removed(&topo, built.solid, before, TAU * centroid * area, &what);
            let contacts = wall_contacts(&topo, &side, built.solid, &what);
            assert_eq!(contacts, ["circle"], "{what}");
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

/// The sector of a rod from `from` round the axis for `turns` of a turn,
/// placed by `pose`, and the arc of its top rim.
fn rod_sector(topo: &mut Topology, from: f64, turns: f64, pose: &Mat4) -> (SolidId, EdgeId) {
    let at = |th: f64| Point3::new(R * th.cos(), R * th.sin(), 0.0);
    let centre = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
    let first = topo.add_vertex(Vertex::new(at(from), 1e-7));
    let last = topo.add_vertex(Vertex::new(at(from + turns * TAU), 1e-7));
    let rim = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let edges = [
        Edge::new(centre, first, EdgeCurve::Line),
        Edge::new(first, last, EdgeCurve::Circle(rim)),
        Edge::new(last, centre, EdgeCurve::Line),
    ]
    .map(|edge| OrientedEdge::new(topo.add_edge(edge), true));
    let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
    let profile = make_planar_face_from_wire(topo, wire).unwrap();
    let sector = extrude(topo, profile, Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
    let arc = solid_edges(topo, sector)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(_))
                && (topo.vertex(edge.start()).unwrap().point().z() - 4.0).abs() < 1e-9
        })
        .unwrap();
    transform_solid(topo, sector, pose).unwrap();
    (sector, arc)
}

/// The walking chamfer cuts the wall of a rod's sector along a circle that
/// runs a quarter turn or three quarters, each to a plane through the axis,
/// and across the wall's seam: the contact's pcurve on the wall runs its
/// whole arc, the long way round for the three quarters.
#[test]
fn the_walking_chamfer_runs_a_sectors_arc_on_its_wall() {
    let (d1, d2) = (0.4, 0.25);
    for (from, turns) in [(0.5, 0.25), (0.25 * PI, 0.75)] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (sector, arc) = rod_sector(&mut topo, from, turns, &pose);
            let (cap, wall) = on_faces(&topo, sector, arc, (d1, d2));
            let before = solid_volume(&topo, sector, 0.001).unwrap();
            let side = wall_of(&topo, sector, arc);
            let mut builder = ChamferBuilder::new(&mut topo, sector);
            builder.add_edges_asymmetric(&[arc], d1, d2);
            let built = builder.build().unwrap();
            let what = format!("{turns} turn {name}");
            assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
            let removed = turns * TAU * 0.5 * cap * wall * (R - cap / 3.0);
            assert_removed(&topo, built.solid, before, removed, &what);
            let contacts = wall_contacts(&topo, &side, built.solid, &what);
            assert_eq!(contacts, ["circle"], "{what}");
        }
    }
}

/// The slanted rod cut back to the arc of its ellipse from `θ = π/6` to
/// `5π/6` by the two planes square to the ellipse there, placed by `pose`,
/// and that arc.
fn ellipse_arc(topo: &mut Topology, pose: &Mat4) -> (SolidId, EdgeId) {
    let mut solid = slanted_cut(topo);
    let t = TILT.tan();
    // Each block's face `x = 0` lies on the plane square to the ellipse at
    // `th`, and the block reaches away from the arc.
    for (th, away) in [(5.0 * PI / 6.0, 1.0), (PI / 6.0, -1.0)] {
        let at = Point3::new(R * th.cos(), R * th.sin(), 4.0 + R * t * th.sin());
        let ahead = Vec3::new(-th.sin(), th.cos(), t * th.cos()) * away;
        let ahead = ahead * (1.0 / ahead.length());
        let place = Mat4::translation(at.x(), at.y(), at.z())
            * Mat4::rotation_z(ahead.y().atan2(ahead.x()))
            * Mat4::rotation_y(-ahead.z().asin())
            * Mat4::translation(0.0, -5.0, -5.0);
        let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
        transform_solid(topo, block, &place).unwrap();
        solid = boolean(topo, BooleanOp::Cut, solid, block).unwrap();
    }
    let arc = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Ellipse(_))
                && [edge.start(), edge.end()]
                    .into_iter()
                    .all(|v| topo.vertex(v).unwrap().point().z() > 4.0)
        })
        .expect("the arc");
    transform_solid(topo, solid, pose).unwrap();
    (solid, arc)
}

/// An arc of the slanted rod's ellipse that ends on the planes square to
/// it: the walking chamfer's contact on the wall rises and falls as it runs
/// round, and its pcurve follows it.
#[test]
fn the_walking_chamfer_runs_an_ellipse_arc_on_its_wall() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (solid, arc) = ellipse_arc(&mut topo, &pose);
        let (cap, wall) = on_faces(&topo, solid, arc, (0.5, 0.3));
        let before = solid_volume(&topo, solid, 0.001).unwrap();
        let side = wall_of(&topo, solid, arc);
        let mut builder = ChamferBuilder::new(&mut topo, solid);
        builder.add_edges_asymmetric(&[arc], 0.5, 0.3);
        let built = builder.build().unwrap();
        assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
        let removed = ellipse_chamfer(cap, wall, (PI / 6.0, 5.0 * PI / 6.0));
        assert_removed(&topo, built.solid, before, removed, name);
        let contacts = wall_contacts(&topo, &side, built.solid, name);
        assert_eq!(contacts, ["nurbs_curve"], "{name}");
    }
}

/// The plane that cuts the ball: `z = BALL_CUT`.
const BALL_CUT: f64 = 1.0;

/// A ball of radius `R` cut by the plane `z = BALL_CUT`, its part below kept,
/// placed by `pose`, and the circle where the plane cuts it. With `turned`,
/// the ball's faces read the same sphere about poles turned 0.6 rad off `z`.
fn capped_ball(topo: &mut Topology, turned: bool, pose: &Mat4) -> (SolidId, EdgeId) {
    let ball = make_sphere(topo, R, 32).unwrap();
    let lid = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(topo, lid, &Mat4::translation(-5.0, -5.0, BALL_CUT)).unwrap();
    let capped = boolean(topo, BooleanOp::Cut, ball, lid).unwrap();
    if turned {
        let poles = Vec3::new(0.0, -0.6_f64.sin(), 0.6_f64.cos());
        for face in solid_faces(topo, capped).unwrap() {
            let FaceSurface::Sphere(sphere) = topo.face(face).unwrap().surface() else {
                continue;
            };
            let turned = SphericalSurface::with_axis(sphere.center(), R, poles).unwrap();
            topo.face_mut(face)
                .unwrap()
                .set_surface(FaceSurface::Sphere(turned));
        }
    }
    let rim = solid_edges(topo, capped)
        .unwrap()
        .into_iter()
        .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_)))
        .expect("the rim");
    transform_solid(topo, capped, pose).unwrap();
    (capped, rim)
}

/// A chamfer of `cap` into the plane and a chord `wall` down the ball from
/// the rim: the volume it takes off, by Pappus's theorem, and each contact's
/// distance from the axis and height, the plane's first. The region it takes
/// off in a half-plane through the axis lies between the rim's point, the
/// two contacts and the ball's meridian, and `∮ r²/2 dz` round it gives the
/// moment `∫∫ r dA`.
fn ball_chamfer(cap: f64, wall: f64) -> (f64, [(f64, f64); 2]) {
    let edge = (BALL_CUT, (R * R - BALL_CUT * BALL_CUT).sqrt());
    let rim = (edge.1, edge.0);
    let from = edge.0.atan2(edge.1);
    let to = from - 2.0 * (0.5 * wall / R).asin();
    let plane = (rim.0 - cap, rim.1);
    let ball = (R * to.cos(), R * to.sin());
    let line = |(r0, z0): (f64, f64), (r1, z1): (f64, f64)| {
        0.5 * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
    };
    let meridian = |a: f64| 0.5 * R.powi(3) * (a.sin() - a.sin().powi(3) / 3.0);
    let moment = line(rim, plane) + line(plane, ball) + meridian(from) - meridian(to);
    (TAU * moment.abs(), [plane, ball])
}

/// The farthest from the circle round the axis at `(r, z)` that a new edge
/// of `result` with a pcurve on a face of kind `kind` lies, read with `pose`
/// undone, and how many such edges there are.
fn off_circle(
    topo: &Topology,
    old: &HashSet<EdgeId>,
    result: SolidId,
    pose: &Mat4,
    kind: &str,
    (r, z): (f64, f64),
) -> (usize, f64) {
    let undo = pose.inverse().unwrap();
    let (mut count, mut worst) = (0, 0.0_f64);
    for face in solid_faces(topo, result).unwrap() {
        let f = topo.face(face).unwrap();
        if f.surface().type_tag() != kind {
            continue;
        }
        for w in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oe in topo.wire(w).unwrap().edges() {
                let id = oe.edge();
                if old.contains(&id) || topo.pcurves().get(id, face).is_none() {
                    continue;
                }
                count += 1;
                let edge = topo.edge(id).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                for k in 0..=64 {
                    let t = t0 + (t1 - t0) * f64::from(k) / 64.0;
                    let p = undo.mul_point(edge.curve().evaluate_with_endpoints(t, a, b));
                    let off = (p.x().hypot(p.y()) - r).hypot(p.z() - z);
                    worst = worst.max(off);
                }
            }
        }
    }
    (count, worst)
}

/// A chamfer's distance along a ball is a chord, as along every other
/// curved face: the rim of a ball cut by a plane takes one chamfer whether
/// the ball's poles lie along the plane's normal (a cone of the closed form)
/// or turned off it (a walked band).
#[test]
fn a_balls_rim_takes_one_chamfer_whichever_way_its_poles_lie() {
    let (d1, d2) = (0.3, 0.6);
    for turned in [false, true] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (ball, rim) = capped_ball(&mut topo, turned, &pose);
            let (cap, wall) = on_faces(&topo, ball, rim, (d1, d2));
            let (removed, [plane, sphere]) = ball_chamfer(cap, wall);
            let before = solid_volume(&topo, ball, 0.001).unwrap();
            let old: HashSet<EdgeId> = solid_edges(&topo, ball).unwrap().into_iter().collect();
            let mut builder = ChamferBuilder::new(&mut topo, ball);
            builder.add_edges_asymmetric(&[rim], d1, d2);
            let built = builder.build().unwrap();
            let what = format!("turned {turned} {name}");
            assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
            let band = solid_faces(&topo, built.solid)
                .unwrap()
                .into_iter()
                .map(|f| topo.face(f).unwrap().surface().type_tag())
                .find(|&tag| tag == "cone" || tag == "nurbs");
            assert_eq!(band, Some(if turned { "nurbs" } else { "cone" }), "{what}");
            assert_removed(&topo, built.solid, before, removed, &what);
            for (kind, at) in [("plane", plane), ("sphere", sphere)] {
                let (count, off) = off_circle(&topo, &old, built.solid, &pose, kind, at);
                assert_eq!(count, 1, "{what}: contacts on the {kind}");
                assert!(off < 1e-6, "{what}: the {kind}'s contact lies {off} off");
            }
        }
    }
}

/// The ball of [`capped_ball`] with both its faces read about `poles`,
/// placed by `pose`, and its rim.
fn capped_ball_about(topo: &mut Topology, poles: Vec3, pose: &Mat4) -> (SolidId, EdgeId) {
    let (ball, rim) = capped_ball(topo, false, &Mat4::translation(0.0, 0.0, 0.0));
    for face in solid_faces(topo, ball).unwrap() {
        let FaceSurface::Sphere(sphere) = topo.face(face).unwrap().surface() else {
            continue;
        };
        let about = SphericalSurface::with_axis(sphere.center(), R, poles).unwrap();
        topo.face_mut(face)
            .unwrap()
            .set_surface(FaceSurface::Sphere(about));
    }
    transform_solid(topo, ball, pose).unwrap();
    (ball, rim)
}

/// A ball's rim whose contact on the ball runs through a pole of the ball's
/// parameters, the north or the south one, at two places round the contact:
/// the chamfer is the one Pappus's theorem gives, its contacts lie on their
/// circles, and their pcurves follow them.
#[test]
fn a_balls_rim_chamfers_when_its_contact_runs_through_a_pole() {
    let (d1, d2) = (0.3, 0.6);
    for (round, side) in [(0.0_f64, 1.0), (1.0, -1.0)] {
        for (name, pose) in poses() {
            let what = format!("round {round} side {side} {name}");
            let mut topo = Topology::new();
            let (ball, rim) = capped_ball(&mut topo, false, &Mat4::translation(0.0, 0.0, 0.0));
            let (cap, wall) = on_faces(&topo, ball, rim, (d1, d2));
            let (removed, [plane, sphere]) = ball_chamfer(cap, wall);
            let on_contact = Vec3::new(sphere.0 * round.cos(), sphere.0 * round.sin(), sphere.1);
            let mut topo = Topology::new();
            let (ball, rim) = capped_ball_about(&mut topo, on_contact * (side / R), &pose);
            let before = solid_volume(&topo, ball, 0.001).unwrap();
            let old: HashSet<EdgeId> = solid_edges(&topo, ball).unwrap().into_iter().collect();
            let side_of = wall_of(&topo, ball, rim);
            let mut builder = ChamferBuilder::new(&mut topo, ball);
            builder.add_edges_asymmetric(&[rim], d1, d2);
            let built = builder.build().unwrap();
            assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
            assert_removed(&topo, built.solid, before, removed, &what);
            for (kind, at) in [("plane", plane), ("sphere", sphere)] {
                let (count, off) = off_circle(&topo, &old, built.solid, &pose, kind, at);
                assert_eq!(count, 1, "{what}: contacts on the {kind}");
                assert!(off < 1e-6, "{what}: the {kind}'s contact lies {off} off");
            }
            assert_eq!(wall_contacts(&topo, &side_of, built.solid, &what).len(), 1);
            if name != "upright" {
                continue;
            }
            let built = chamfer_v2(&mut topo, ball, &[rim], d1, d2).unwrap();
            assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
            assert_removed(&topo, built.solid, before, removed, &what);
        }
    }
}

/// The torus: radius 4 round its axis, 1.5 round its tube.
const TORUS: (f64, f64) = (4.0, 1.5);

/// Half the torus, on the side `x >= 0`, placed by `pose`, and one of the
/// two circles where the plane `x = 0` cuts its tube.
fn half_torus(topo: &mut Topology, pose: &Mat4) -> (SolidId, EdgeId) {
    let torus = make_torus(topo, TORUS.0, TORUS.1, 32).unwrap();
    let half = make_box(topo, 10.0, 20.0, 10.0).unwrap();
    transform_solid(topo, half, &Mat4::translation(0.0, -10.0, -5.0)).unwrap();
    let kept = boolean(topo, BooleanOp::Intersect, torus, half).unwrap();
    let rim = solid_edges(topo, kept)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            edge.start() == edge.end() && matches!(edge.curve(), EdgeCurve::Circle(_))
        })
        .expect("a rim");
    transform_solid(topo, kept, pose).unwrap();
    (kept, rim)
}

/// The volume a chamfer of `cap` into the plane `x = 0` and a chord `wall`
/// along the torus takes off one of the half torus's rims, by the tube
/// formula. The rim is a circle of radius `r` round the tube's centre, at
/// the angle `v` round it. In the section square to the rim, `a` runs in
/// toward the tube's centre and `b` along `x`, and a point stands for
/// `r (1 - a / r) dv da db`. The region lies between the torus's trace in
/// the section and the chord from the plane's contact to the torus's.
fn torus_rim_chamfer(cap: f64, wall: f64) -> f64 {
    let (big, r) = TORUS;
    let section = |v: f64| {
        // The torus's trace `b` along `x`: its root in `a` nearest the rim.
        let wall_at = |b: f64| {
            let off = |a: f64| {
                let (rho, z) = (big + (r - a) * v.cos(), (r - a) * v.sin());
                (rho.hypot(b) - big).powi(2) + z * z - r * r
            };
            let (mut lo, mut hi) = (-0.5 * r, 0.5 * r);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if off(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        // The torus's contact: the chord `wall` from the rim.
        let (mut lo, mut hi) = (0.0, wall);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if wall_at(mid).powi(2) + mid * mid < wall * wall {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let b_t = 0.5 * (lo + hi);
        let a_t = wall_at(b_t);
        let strip = |b: f64| {
            let (from, to) = (wall_at(b), cap + (a_t - cap) * (b / b_t));
            let f = |a: f64| a - 0.5 * a * a / r;
            f(to) - f(from)
        };
        r * integral(0.0, b_t, 4, strip)
    };
    integral(0.0, TAU, 16, section)
}

/// A rim of the half torus winds round the tube, so the torus face's seam
/// there runs along `v`, not along `u`: the walking chamfer starts its
/// contacts on that seam and cuts the rim exactly.
#[test]
fn a_rim_round_a_torus_tube_takes_a_chamfer() {
    let (d1, d2) = (0.3, 0.25);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (half, rim) = half_torus(&mut topo, &pose);
        let (cap, wall) = on_faces(&topo, half, rim, (d1, d2));
        let before = solid_volume(&topo, half, 0.001).unwrap();
        let side = wall_of(&topo, half, rim);
        let mut builder = ChamferBuilder::new(&mut topo, half);
        builder.add_edges_asymmetric(&[rim], d1, d2);
        let built = builder.build().unwrap();
        assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
        assert_removed(
            &topo,
            built.solid,
            before,
            torus_rim_chamfer(cap, wall),
            name,
        );
        let contacts = wall_contacts(&topo, &side, built.solid, name);
        assert_eq!(contacts, ["nurbs_curve"], "{name}");
    }
}

/// That a chamfer refused `edge` and left `solid` as it was: the result is
/// the input, the edge is in `failed` and nothing in `succeeded`, and the
/// input is still valid and watertight, with its volume `before`. The text
/// of the edge's failure.
fn refused(
    topo: &Topology,
    solid: SolidId,
    built: &brepkit_blend::BlendResult,
    edge: EdgeId,
    before: f64,
    what: &str,
) -> String {
    assert_eq!(built.solid, solid, "{what}: a new solid");
    assert!(built.succeeded.is_empty(), "{what}: {:?}", built.succeeded);
    let failure = built
        .failed
        .iter()
        .find(|(e, _)| *e == edge)
        .unwrap_or_else(|| panic!("{what}: the edge is not in failed"));
    assert!(validate_solid(topo, solid).unwrap().is_valid(), "{what}");
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
    let now = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (now - before).abs() <= 1e-12 * before,
        "{what}: {now} against {before}"
    );
    format!("{:?}", failure.1)
}

/// A cylinder (3, 4) or a frustum (3 to 1, 4 high) whose wall is put in its
/// B-spline form, which closes round on itself, placed by `pose`, and its
/// bottom or top rim.
fn b_spline_walled(
    topo: &mut Topology,
    frustum: bool,
    top: bool,
    pose: &Mat4,
) -> (SolidId, EdgeId) {
    let solid = if frustum {
        make_cone(topo, 3.0, 1.0, 4.0).unwrap()
    } else {
        make_cylinder(topo, 3.0, 4.0).unwrap()
    };
    let height = if top { 4.0 } else { 0.0 };
    let rim = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            matches!(topo.edge(e).unwrap().curve(),
                EdgeCurve::Circle(c) if (c.center().z() - height).abs() < 1e-9)
        })
        .unwrap();
    for face in solid_faces(topo, solid).unwrap() {
        if topo.face(face).unwrap().surface().is_planar() {
            continue;
        }
        let wall = brepkit_heal::custom::convert_to_bspline::face_bspline_surface(topo, face)
            .unwrap()
            .unwrap();
        topo.face_mut(face)
            .unwrap()
            .set_surface(FaceSurface::Nurbs(wall));
    }
    transform_solid(topo, solid, pose).unwrap();
    (solid, rim)
}

/// A cylinder's rims and a frustum's bottom rim on their walls put in
/// B-spline form, which closes round on itself: the walking chamfer cuts
/// each exactly, as on the analytic walls, and the pcurve of each contact
/// on the wall follows its edge within the wall's knot domain.
#[test]
fn a_rim_on_a_closed_b_spline_wall_chamfers_exactly() {
    let (d1, d2) = (0.4, 0.25);
    // The frustum's wall rises 4 as it closes in by 2.
    let slope = 2.0_f64.atan2(1.0);
    for (frustum, top) in [(false, false), (false, true), (true, false)] {
        for (name, pose) in poses() {
            let what = format!("frustum {frustum} top {top} {name}");
            let mut topo = Topology::new();
            let (solid, rim) = b_spline_walled(&mut topo, frustum, top, &pose);
            let (cap, wall) = on_faces(&topo, solid, rim, (d1, d2));
            let rise = if frustum { slope } else { 0.5 * PI };
            let area = 0.5 * cap * wall * rise.sin();
            let removed = TAU * (3.0 - (cap + wall * rise.cos()) / 3.0) * area;
            // The measure of a B-spline cone trimmed along a circle is
            // 2.2e-5 high at 0.001 and comes down to 2.3e-8 at 1e-6.
            let deflection = if frustum { 1e-5 } else { 0.001 };
            let before = solid_volume(&topo, solid, deflection).unwrap();
            let side = wall_of(&topo, solid, rim);
            let mut builder = ChamferBuilder::new(&mut topo, solid);
            builder.add_edges_asymmetric(&[rim], d1, d2);
            let built = builder.build().unwrap();
            assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
            assert_removed_measured(&topo, built.solid, before, removed, &what, deflection);
            assert_eq!(wall_contacts(&topo, &side, built.solid, &what).len(), 1);
            if name != "upright" {
                continue;
            }
            let built = chamfer_v2(&mut topo, solid, &[rim], d1, d2).unwrap();
            assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
            assert_removed_measured(&topo, built.solid, before, removed, &what, deflection);
            assert_eq!(wall_contacts(&topo, &side, built.solid, &what).len(), 1);
        }
    }
}

/// The common part of two cylinders of radius 2 and height 4 whose axes lie
/// 3 apart, and its two ridges, one stored up and one down.
fn lens(topo: &mut Topology) -> (SolidId, Vec<EdgeId>) {
    let a = make_cylinder(topo, 2.0, 4.0).unwrap();
    let b = make_cylinder(topo, 2.0, 4.0).unwrap();
    transform_solid(topo, b, &Mat4::translation(3.0, 0.0, 0.0)).unwrap();
    let lens = boolean(topo, BooleanOp::Intersect, a, b).unwrap();
    let ridges: Vec<EdgeId> = solid_edges(topo, lens)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()]
                .into_iter()
                .all(|v| (topo.vertex(v).unwrap().point().x() - 1.5).abs() < 1e-6)
        })
        .collect();
    (lens, ridges)
}

/// A lens's ridge, where the builder cannot close the chamfer's ends: each
/// ridge is refused on its own, and the solid comes back as it was, from
/// the builder and from `chamfer_v2`.
#[test]
fn a_ridge_whose_chamfer_ends_do_not_close_is_refused_on_its_own() {
    let mut topo = Topology::new();
    let (solid, ridges) = lens(&mut topo);
    assert_eq!(ridges.len(), 2);
    let before = solid_volume(&topo, solid, 0.001).unwrap();
    for &ridge in &ridges {
        let what = format!("{ridge:?}");
        let mut builder = ChamferBuilder::new(&mut topo, solid);
        builder.add_edges_symmetric(&[ridge], 0.3);
        let built = builder.build().unwrap();
        refused(&topo, solid, &built, ridge, before, &what);
        let built = chamfer_v2(&mut topo, solid, &[ridge], 0.3, 0.3).unwrap();
        refused(&topo, solid, &built, ridge, before, &what);
    }
    let mut builder = ChamferBuilder::new(&mut topo, solid);
    builder.add_edges_symmetric(&ridges, 0.3);
    let built = builder.build().unwrap();
    for &ridge in &ridges {
        refused(&topo, solid, &built, ridge, before, "both ridges");
    }
}
