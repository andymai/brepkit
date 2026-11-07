//! Blend surfaces between edge chains, with each continuity claim measured
//! along the seam in several poses.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::panic
)]

mod surface_fixtures;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::blend_surface::{BlendContinuity, BlendSurfaceOptions, blend_surface};
use brepkit_operations::measure::face_area;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::tessellate;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{face_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::wire::{OrientedEdge, Wire};
use surface_fixtures::{
    Builder, KINDS, Pose, Sheet, Worst, across_pose, assert_wire_sound, chain_exit, gap_to_chain,
    leaves_the_face, measure_seam, measure_seam_at, net_sheet, sheet, surface_of,
};

const SAMPLES: usize = 400;
const CHAIN_SAMPLES: usize = 100;
type MakePose = fn() -> Pose;
const POSES: [(&str, MakePose); 3] = [
    ("upright", Pose::upright),
    ("turned", Pose::turned),
    ("mirrored", Pose::mirrored),
];

fn nurbs_of(topo: &Topology, face: FaceId) -> NurbsSurface {
    match topo.face(face).unwrap().surface() {
        FaceSurface::Nurbs(surface) => surface.clone(),
        other => panic!(
            "a blend surface should be a spline, got {}",
            other.type_tag()
        ),
    }
}

fn surface_of_face(topo: &Topology, face: FaceId) -> FaceSurface {
    topo.face(face).unwrap().surface().clone()
}

/// Both seams of a blend between `a` and `b`, with the chain gaps counted in
/// the position figure.
fn both_seams(topo: &Topology, result: &NurbsSurface, a: &Sheet, b: &Sheet) -> (Worst, Worst) {
    let mut at_a = measure_seam(result, 0.0, &surface_of_face(topo, a.face), SAMPLES);
    let mut at_b = measure_seam(result, 1.0, &surface_of_face(topo, b.face), SAMPLES);
    at_a.g0 = at_a
        .g0
        .max(gap_to_chain(topo, result, 0.0, &a.chain, CHAIN_SAMPLES));
    at_b.g0 = at_b
        .g0
        .max(gap_to_chain(topo, result, 1.0, &b.chain, CHAIN_SAMPLES));
    (at_a, at_b)
}

fn assert_seams(context: &str, continuity: BlendContinuity, seams: &[Worst]) {
    for seam in seams {
        assert!(seam.g0 < 1e-8, "{context}: {seam:?}");
        assert!(seam.g1 < 1e-7, "{context}: {seam:?}");
        if continuity == BlendContinuity::G2 {
            assert!(seam.g2 < 1e-6, "{context}: {seam:?}");
        }
    }
}

#[derive(Default)]
struct Table {
    rows: Vec<(String, Worst, Worst)>,
}

impl Table {
    fn add(&mut self, surface: &str, continuity: BlendContinuity, worst: Worst) {
        let at = if let Some(at) = self.rows.iter().position(|row| row.0 == surface) {
            at
        } else {
            self.rows
                .push((surface.to_string(), Worst::default(), Worst::default()));
            self.rows.len() - 1
        };
        match continuity {
            BlendContinuity::G1 => self.rows[at].1.join(worst),
            BlendContinuity::G2 => self.rows[at].2.join(worst),
        }
    }

    fn print(&self, title: &str) {
        println!("{title}");
        println!(
            "{:<10} {:>10} {:>10} | {:>10} {:>10} {:>10} {:>9}",
            "surface", "G1:gap", "G1:angle", "G2:gap", "G2:angle", "G2:dk", "|k|max"
        );
        for (name, g1, g2) in &self.rows {
            println!(
                "{name:<10} {:>10.2e} {:>10.2e} | {:>10.2e} {:>10.2e} {:>10.2e} {:>9.3}",
                g1.g0, g1.g1, g2.g0, g2.g1, g2.g2, g2.curvature
            );
        }
    }
}

/// Every fixture kind against a copy of itself, in every pose, at both
/// continuities.
#[test]
fn blends_keep_their_continuity_on_every_surface_type_in_every_pose() {
    let mut table = Table::default();
    for kind in KINDS {
        for (pose_name, pose) in POSES {
            for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
                let mut topo = Topology::new();
                let a = sheet(&mut topo, kind, pose());
                let b = sheet(&mut topo, kind, across_pose(kind, pose()));
                let face = blend_surface(
                    &mut topo,
                    &a.chain,
                    a.face,
                    &b.chain,
                    b.face,
                    &BlendSurfaceOptions::new(continuity),
                )
                .unwrap_or_else(|e| panic!("{kind} {pose_name} {continuity:?}: {e}"));
                let result = nurbs_of(&topo, face);
                if pose_name == "turned" && continuity == BlendContinuity::G2 {
                    println!(
                        "{kind}: {} by {} control points",
                        result.control_points().len(),
                        result.control_points()[0].len()
                    );
                }
                assert_wire_sound(&topo, face);
                let (at_a, at_b) = both_seams(&topo, &result, &a, &b);
                let context = format!("{kind} {pose_name} {continuity:?}");
                assert!(leaves_the_face(&result, 0.0, a.inside), "{context}");
                assert!(leaves_the_face(&result, 1.0, b.inside), "{context}");
                assert_seams(&context, continuity, &[at_a, at_b]);
                let mut joined = at_a;
                joined.join(at_b);
                table.add(surface_of(kind), continuity, joined);
            }
        }
    }
    table.print("blend_surface, like against like: worst over 400 samples a seam, three poses");
}

/// Each fixture kind faces every other, so a seam's figures cannot rest on the
/// two sides being alike.
#[test]
fn blends_keep_their_continuity_between_different_surface_types() {
    let mut table = Table::default();
    for kind_a in KINDS {
        for kind_b in KINDS {
            let pose = Pose::turned();
            let (at, exit) = chain_exit(kind_a);
            let (at_b, exit_b) = chain_exit(kind_b);
            let meet = Pose::turn(
                exit_b
                    .cross(-exit)
                    .normalize()
                    .unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
                exit_b.dot(-exit).clamp(-1.0, 1.0).acos(),
                [0.0, 0.0, 0.0],
            );
            let moved = meet.point(at_b);
            let place = Pose {
                rows: meet.rows,
                shift: [
                    at.x() + exit.x() * 8.0 - moved.x(),
                    at.y() + exit.y() * 8.0 - moved.y(),
                    at.z() + exit.z() * 8.0 - moved.z(),
                ],
            };
            let mut topo = Topology::new();
            let a = sheet(&mut topo, kind_a, pose);
            let b = sheet(&mut topo, kind_b, place.then(pose));
            let face = blend_surface(
                &mut topo,
                &a.chain,
                a.face,
                &b.chain,
                b.face,
                &BlendSurfaceOptions::new(BlendContinuity::G2),
            )
            .unwrap_or_else(|e| panic!("{kind_a} to {kind_b}: {e}"));
            let result = nurbs_of(&topo, face);
            let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
            assert_seams(
                &format!("{kind_a} to {kind_b}"),
                BlendContinuity::G2,
                &[seam_a, seam_b],
            );
            table.add(surface_of(kind_a), BlendContinuity::G2, seam_a);
            table.add(surface_of(kind_b), BlendContinuity::G2, seam_b);
        }
    }
    table.print("blend_surface, every pair of types, turned pose");
}

/// A chain of two sides of each fixture: a corner on every surface type.
#[test]
fn a_chain_may_turn_a_corner_on_every_surface_type() {
    let mut table = Table::default();
    for kind in KINDS {
        for (pose_name, pose) in POSES {
            for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
                let mut topo = Topology::new();
                let mut a = sheet(&mut topo, kind, pose());
                let mut b = sheet(&mut topo, kind, across_pose(kind, pose()));
                let at = a.boundary.iter().position(|e| *e == a.chain[0]).unwrap();
                let next = (at + 1) % a.boundary.len();
                a.chain.push(a.boundary[next]);
                b.chain.push(b.boundary[next]);
                let face = blend_surface(
                    &mut topo,
                    &a.chain,
                    a.face,
                    &b.chain,
                    b.face,
                    &BlendSurfaceOptions::new(continuity),
                )
                .unwrap_or_else(|e| panic!("{kind} {pose_name} {continuity:?}: {e}"));
                let result = nurbs_of(&topo, face);
                let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
                assert_seams(
                    &format!("{kind} corner {pose_name} {continuity:?}"),
                    continuity,
                    &[seam_a, seam_b],
                );
                let mut joined = seam_a;
                joined.join(seam_b);
                table.add(surface_of(kind), continuity, joined);
            }
        }
    }
    table.print("blend_surface, chains turning a corner: worst over 400 samples a seam");
}

/// A line, a tangent arc and a spline on a plane, each stored either way; the
/// spline starts at a kink.
fn mixed_plane_sheet(topo: &mut Topology, pose: Pose, flips: [bool; 3], z: f64) -> Sheet {
    let mut b = Builder { topo, pose };
    let p = |x: f64, y: f64| Point3::new(x, y, z);
    let corners = [
        p(0.0, 0.0),
        p(0.0, 4.0),
        p(-2.0, 6.0),
        p(-4.0, 10.0),
        p(-9.0, 10.0),
        p(-9.0, 0.0),
    ];
    let v: Vec<_> = corners.iter().map(|&c| b.vertex(c)).collect();
    let up = Vec3::new(0.0, 0.0, 1.0);
    let line = if flips[0] {
        b.line(v[1], v[0])
    } else {
        b.line(v[0], v[1])
    };
    let arc = if flips[1] {
        b.arc(p(-2.0, 4.0), -up, 2.0, v[2], v[1])
    } else {
        b.arc(p(-2.0, 4.0), up, 2.0, v[1], v[2])
    };
    let through = [p(-2.0, 6.0), p(-3.5, 7.0), p(-3.0, 8.5), p(-4.0, 10.0)];
    let spline = if flips[2] {
        let back: Vec<Point3> = through.iter().rev().copied().collect();
        b.spline(&back, v[3], v[2])
    } else {
        b.spline(&through, v[2], v[3])
    };
    let rest: Vec<EdgeId> = (3..6).map(|i| b.line(v[i], v[(i + 1) % 6])).collect();
    let mut walk = vec![
        OrientedEdge::new(line, !flips[0]),
        OrientedEdge::new(arc, !flips[1]),
        OrientedEdge::new(spline, !flips[2]),
    ];
    walk.extend(rest.iter().map(|&e| OrientedEdge::new(e, true)));
    let wire = b.topo.add_wire(Wire::new(walk, true).unwrap());
    let normal = pose.vector(up);
    let at = pose.point(p(0.0, 0.0));
    let d = normal.dot(at - Point3::new(0.0, 0.0, 0.0));
    let face = b.topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane { normal, d },
    ));
    Sheet {
        face,
        chain: vec![line, arc, spline],
        inside: pose.point(p(-6.0, 5.0)),
        boundary: Vec::new(),
    }
}

/// A chain's own measure and the same surface however its edges are stored
/// and whichever way the list runs.
#[test]
fn a_mixed_chain_blends_the_same_however_it_is_stored() {
    let across = Pose::reflection(Point3::new(4.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0))
        .then(Pose::turn(Vec3::new(0.0, 0.0, 1.0), 0.1, [0.0, 0.5, 1.0]));
    for (pose_name, pose) in POSES {
        for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
            let mut reference: Option<NurbsSurface> = None;
            for flips in [
                [false, false, false],
                [true, true, true],
                [true, false, true],
            ] {
                for reverse_list in [false, true] {
                    let mut topo = Topology::new();
                    let mut a = mixed_plane_sheet(&mut topo, pose(), flips, 0.0);
                    let b = mixed_plane_sheet(&mut topo, across.then(pose()), [false; 3], 0.0);
                    if reverse_list {
                        a.chain.reverse();
                    }
                    let face = blend_surface(
                        &mut topo,
                        &a.chain,
                        a.face,
                        &b.chain,
                        b.face,
                        &BlendSurfaceOptions::new(continuity),
                    )
                    .unwrap_or_else(|e| panic!("{pose_name} {flips:?}: {e}"));
                    let result = nurbs_of(&topo, face);
                    let seam_a = measure_seam(&result, 0.0, &surface_of_face(&topo, a.face), 120);
                    let seam_b = measure_seam(&result, 1.0, &surface_of_face(&topo, b.face), 120);
                    assert_seams(
                        &format!("mixed {pose_name} {flips:?} {reverse_list}"),
                        continuity,
                        &[seam_a, seam_b],
                    );
                    assert!(gap_to_chain(&topo, &result, 0.0, &a.chain, 60) < 1e-8);
                    if let Some(first) = &reference {
                        for i in 0..=40 {
                            for j in 0..=10 {
                                let (u, v) = (f64::from(i) / 40.0, f64::from(j) / 10.0);
                                let gap = (first.evaluate(u, v) - result.evaluate(u, v)).length();
                                assert!(gap < 1e-8, "{flips:?} {reverse_list} at {u} {v}: {gap}");
                            }
                        }
                    } else {
                        reference = Some(result);
                    }
                }
            }
        }
    }
}

/// Two chains of different edge counts pair by arc length.
#[test]
fn chains_of_different_edge_counts_pair_by_arc_length() {
    let mut topo = Topology::new();
    let a = mixed_plane_sheet(&mut topo, Pose::turned(), [false, true, false], 0.0);
    let far =
        Pose::reflection(Point3::new(3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).then(Pose::turned());
    let b = sheet(&mut topo, "plane", far);
    let face = blend_surface(
        &mut topo,
        &a.chain,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    let result = nurbs_of(&topo, face);
    let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
    assert_seams("mixed to line", BlendContinuity::G2, &[seam_a, seam_b]);
}

/// A larger weight sends the surface further along the tangent plane, without
/// costing any continuity.
#[test]
fn tangent_weights_scale_the_leaving_speed_only() {
    let measure = |weight_a: f64, weight_b: f64| {
        let mut topo = Topology::new();
        let a = sheet(&mut topo, "cylinder across", Pose::turned());
        let b = sheet(
            &mut topo,
            "cylinder across",
            across_pose("cylinder across", Pose::turned()),
        );
        let face = blend_surface(
            &mut topo,
            &a.chain,
            a.face,
            &b.chain,
            b.face,
            &BlendSurfaceOptions {
                continuity: BlendContinuity::G2,
                weight_a,
                weight_b,
            },
        )
        .unwrap();
        let result = nurbs_of(&topo, face);
        let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
        assert_seams("weighted", BlendContinuity::G2, &[seam_a, seam_b]);
        let leaving = |v: f64| result.derivatives(0.5, v, 1)[0][1].length();
        (leaving(0.0), leaving(1.0))
    };
    let (a1, b1) = measure(1.0, 1.0);
    let (a2, b2) = measure(2.5, 0.4);
    assert!((a2 / a1 - 2.5).abs() < 1e-7, "{a1} {a2}");
    assert!((b2 / b1 - 0.4).abs() < 1e-7, "{b1} {b2}");
}

fn circle_edges(topo: &Topology, face: FaceId) -> Vec<EdgeId> {
    face_edges(topo, face)
        .unwrap()
        .into_iter()
        .filter(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_)))
        .collect()
}

fn lateral_face(topo: &Topology, solid: brepkit_topology::solid::SolidId) -> FaceId {
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .unwrap()
}

/// Two closed rims on two solids, the second turned so that its seam is
/// elsewhere: the surface closes on itself and does not spiral.
#[test]
fn closed_chains_bridge_two_solids() {
    for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
        let mut topo = Topology::new();
        let lower = make_cylinder(&mut topo, 3.0, 5.0).unwrap();
        let upper = make_cylinder(&mut topo, 2.0, 5.0).unwrap();
        let place = Mat4::translation(0.0, 0.0, 9.0) * Mat4::rotation_z(1.1);
        transform_solid(&mut topo, upper, &place).unwrap();
        let (lower_side, upper_side) = (lateral_face(&topo, lower), lateral_face(&topo, upper));
        let height = |topo: &Topology, e: EdgeId| {
            topo.vertex(topo.edge(e).unwrap().start())
                .unwrap()
                .point()
                .z()
        };
        let rim_of = |topo: &Topology, face: FaceId, z: f64| {
            circle_edges(topo, face)
                .into_iter()
                .find(|&e| (height(topo, e) - z).abs() < 1e-9)
                .unwrap()
        };
        let top_rim = rim_of(&topo, lower_side, 5.0);
        let bottom_rim = rim_of(&topo, upper_side, 9.0);
        let face = blend_surface(
            &mut topo,
            &[top_rim],
            lower_side,
            &[bottom_rim],
            upper_side,
            &BlendSurfaceOptions::new(continuity),
        )
        .unwrap();
        let result = nurbs_of(&topo, face);
        for i in 0..=20 {
            let v = f64::from(i) / 20.0;
            let gap = (result.evaluate(0.0, v) - result.evaluate(1.0, v)).length();
            assert!(gap < 1e-12, "seam open by {gap} at {v}");
        }
        let at_lower = measure_seam(&result, 0.0, &surface_of_face(&topo, lower_side), SAMPLES);
        let at_upper = measure_seam(&result, 1.0, &surface_of_face(&topo, upper_side), SAMPLES);
        assert_seams("closed rims", continuity, &[at_lower, at_upper]);
        println!("closed rims {continuity:?}: {at_lower:?} {at_upper:?}");
        for i in 0..=8 {
            let u = f64::from(i) / 8.0;
            let rail = result.evaluate(u, 1.0) - result.evaluate(u, 0.0);
            let radial = rail - Vec3::new(0.0, 0.0, rail.z());
            assert!(radial.length() < 1.2, "the bridge twists: {rail:?} at {u}");
        }
        assert_wire_sound(&topo, face);
        let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
        assert_eq!(wire.edges().len(), 4, "two rims and a seam used twice");
        assert_eq!(wire.edges()[1].edge(), wire.edges()[3].edge());
    }
}

/// Four lines round a box's top against a circle: closed, with corners on one
/// side only, on a plane on one side and a plane on the other.
#[test]
fn a_closed_polygon_bridges_to_a_circle() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 6.0, 6.0, 2.0).unwrap();
    let post = make_cylinder(&mut topo, 6.0, 3.0).unwrap();
    transform_solid(&mut topo, post, &Mat4::translation(3.0, 3.0, 8.0)).unwrap();
    let top = solid_faces(&topo, block)
        .unwrap()
        .into_iter()
        .find(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { normal, d } => normal.z() > 0.5 && (*d - 2.0).abs() < 1e-9,
            _ => false,
        })
        .unwrap();
    let wire = topo.wire(topo.face(top).unwrap().outer_wire()).unwrap();
    let square: Vec<EdgeId> = wire.edges().iter().map(OrientedEdge::edge).collect();
    let cap = solid_faces(&topo, post)
        .unwrap()
        .into_iter()
        .find(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { normal, .. } => normal.z() < -0.5,
            _ => false,
        })
        .unwrap();
    let rim = circle_edges(&topo, cap);
    assert_eq!(rim.len(), 1);
    let face = blend_surface(
        &mut topo,
        &square,
        top,
        &rim,
        cap,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    let result = nurbs_of(&topo, face);
    let at_top = measure_seam(&result, 0.0, &surface_of_face(&topo, top), SAMPLES);
    let at_cap = measure_seam(&result, 1.0, &surface_of_face(&topo, cap), SAMPLES);
    assert_seams("square to circle", BlendContinuity::G2, &[at_top, at_cap]);
    println!("square to circle: {at_top:?} {at_cap:?}");
    assert!(gap_to_chain(&topo, &result, 0.0, &square, SAMPLES) < 1e-8);
    assert!(gap_to_chain(&topo, &result, 1.0, &rim, SAMPLES) < 1e-8);
}

/// An edge of a hole leaves into the hole: the face lies outside the loop.
#[test]
fn a_hole_chain_leaves_the_face_into_the_hole() {
    let mut topo = Topology::new();
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
    let corners = [p(-5.0, -5.0), p(5.0, -5.0), p(5.0, 5.0), p(-5.0, 5.0)];
    let v: Vec<_> = corners.iter().map(|&c| b.vertex(c)).collect();
    let outer: Vec<_> = (0..4).map(|i| b.line(v[i], v[(i + 1) % 4])).collect();
    let center = b.vertex(p(2.0, 0.0));
    let hole = b.arc(p(0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), 2.0, center, center);
    let outer_wire = b.topo.add_wire(
        Wire::new(
            outer.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let hole_wire = b
        .topo
        .add_wire(Wire::new(vec![OrientedEdge::new(hole, true)], true).unwrap());
    let plate = b.topo.add_face(Face::new(
        outer_wire,
        vec![hole_wire],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let lid = make_cylinder(&mut topo, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, lid, &Mat4::translation(0.0, 0.0, 3.0)).unwrap();
    let side = lateral_face(&topo, lid);
    let rim = circle_edges(&topo, side)
        .into_iter()
        .find(|&e| {
            let z = topo
                .vertex(topo.edge(e).unwrap().start())
                .unwrap()
                .point()
                .z();
            (z - 3.0).abs() < 1e-9
        })
        .unwrap();
    let face = blend_surface(
        &mut topo,
        &[hole],
        plate,
        &[rim],
        side,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    let result = nurbs_of(&topo, face);
    let at = |u: f64| {
        let leaving = result.derivatives(u, 0.0, 1)[0][1];
        let spot = result.evaluate(u, 0.0);
        let toward_center = Vec3::new(-spot.x(), -spot.y(), 0.0);
        leaving.dot(toward_center) > 0.0
    };
    assert!(at(0.1) && at(0.5) && at(0.9));
    let seam = measure_seam(&result, 0.0, &surface_of_face(&topo, plate), SAMPLES);
    assert_seams("hole rim", BlendContinuity::G2, &[seam]);
}

/// The same face wound the other way gives the same surface: it is the area
/// its wire bounds that says which side the face is on, not how the wire was
/// built.
#[test]
fn a_face_wound_either_way_is_left_the_same_way() {
    for kind in ["plane", "cylinder across", "spline"] {
        let run = |wind_back: bool| {
            let mut topo = Topology::new();
            let a = sheet(&mut topo, kind, Pose::turned());
            let b = sheet(&mut topo, kind, across_pose(kind, Pose::turned()));
            if wind_back {
                let outer = topo.face(a.face).unwrap().outer_wire();
                let reversed: Vec<OrientedEdge> = topo
                    .wire(outer)
                    .unwrap()
                    .edges()
                    .iter()
                    .rev()
                    .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                    .collect();
                let wire = topo.add_wire(Wire::new(reversed, true).unwrap());
                topo.face_mut(a.face).unwrap().set_outer_wire(wire);
            }
            let face = blend_surface(
                &mut topo,
                &a.chain,
                a.face,
                &b.chain,
                b.face,
                &BlendSurfaceOptions::new(BlendContinuity::G2),
            )
            .unwrap();
            let result = nurbs_of(&topo, face);
            assert!(
                leaves_the_face(&result, 0.0, a.inside),
                "{kind} {wind_back}"
            );
            result
        };
        let (kept, wound) = (run(false), run(true));
        for i in 0..=20 {
            let (u, v) = (f64::from(i) / 20.0, 0.3);
            let gap = (kept.evaluate(u, v) - wound.evaluate(u, v)).length();
            assert!(gap < 1e-8, "{kind}: {gap}");
        }
    }
}

/// The blend's normal agrees with face A's, reversed flag counted.
#[test]
fn the_blend_faces_the_way_face_a_does() {
    for reversed in [false, true] {
        let mut topo = Topology::new();
        let a = sheet(&mut topo, "cylinder across", Pose::turned());
        let b = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
        topo.face_mut(a.face).unwrap().set_reversed(reversed);
        let face = blend_surface(
            &mut topo,
            &a.chain,
            a.face,
            &b.chain,
            b.face,
            &BlendSurfaceOptions::new(BlendContinuity::G1),
        )
        .unwrap();
        let result = nurbs_of(&topo, face);
        assert!(!topo.face(face).unwrap().is_reversed());
        let p = result.evaluate(0.5, 0.0);
        let neighbour = surface_of_face(&topo, a.face);
        let (u, v) = surface_fixtures::foot_on(&neighbour, p);
        let facing = neighbour.normal(u, v);
        let facing = if reversed { -facing } else { facing };
        let own = result.normal(0.5, 0.0).unwrap();
        assert!(
            own.dot(facing) > 0.99,
            "{reversed}: {own:?} against {facing:?}"
        );
    }
}

/// Chains meeting at one end: the surface closes to a point there, three
/// sided, and holds its continuity everywhere it has a normal.
#[test]
fn chains_meeting_at_one_end_close_to_a_point() {
    let mut topo = Topology::new();
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::turned(),
    };
    let p = |x: f64, y: f64, z: f64| Point3::new(x, y, z);
    let apex = b.vertex(p(0.0, 0.0, 0.0));
    let square = |b: &mut Builder<'_>, points: [Point3; 3]| -> (EdgeId, FaceId) {
        let others: Vec<_> = points.iter().map(|&q| b.vertex(q)).collect();
        let e0 = b.line(apex, others[0]);
        let e1 = b.line(others[0], others[1]);
        let e2 = b.line(others[1], others[2]);
        let e3 = b.line(others[2], apex);
        let wire = b.topo.add_wire(
            Wire::new(
                [e0, e1, e2, e3]
                    .iter()
                    .map(|&e| OrientedEdge::new(e, true))
                    .collect(),
                true,
            )
            .unwrap(),
        );
        let (u, w) = (points[0] - p(0.0, 0.0, 0.0), points[2] - p(0.0, 0.0, 0.0));
        let normal = b.pose.vector(u.cross(w).normalize().unwrap());
        let d = normal.dot(b.pose.point(p(0.0, 0.0, 0.0)) - p(0.0, 0.0, 0.0));
        let face = b.topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane { normal, d },
        ));
        (e0, face)
    };
    let (edge_a, face_a) = square(
        &mut b,
        [p(0.0, 10.0, 0.0), p(-5.0, 10.0, 0.0), p(-5.0, 0.0, 0.0)],
    );
    let (edge_b, face_b) = square(
        &mut b,
        [p(6.0, 8.0, 0.0), p(10.0, 5.0, 0.0), p(4.0, -3.0, 0.0)],
    );
    let face = blend_surface(
        &mut topo,
        &[edge_a],
        face_a,
        &[edge_b],
        face_b,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    let result = nurbs_of(&topo, face);
    assert_wire_sound(&topo, face);
    let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
    assert_eq!(wire.edges().len(), 3, "two sides and one rail");
    let seam_a = measure_seam(&result, 0.0, &surface_of_face(&topo, face_a), SAMPLES);
    let seam_b = measure_seam(&result, 1.0, &surface_of_face(&topo, face_b), SAMPLES);
    println!("closing to a point: {seam_a:?} {seam_b:?}");
    assert!(seam_a.g0 < 1e-8 && seam_b.g0 < 1e-8);
    assert!(
        seam_a.g1 < 1e-5 && seam_b.g1 < 1e-5,
        "{seam_a:?} {seam_b:?}"
    );
}

/// The result works as a face: it meshes, and it has an area.
#[test]
fn a_blend_face_meshes_and_measures() {
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "torus ring", Pose::turned());
    let b = sheet(
        &mut topo,
        "torus ring",
        across_pose("torus ring", Pose::turned()),
    );
    let face = blend_surface(
        &mut topo,
        &a.chain,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    let mesh = tessellate(&topo, face, 0.05).unwrap();
    assert!(mesh.indices.len() >= 3);
    let area = face_area(&topo, face, 0.05).unwrap();
    assert!(area > 1.0, "{area}");
}

fn refused(result: Result<FaceId, OperationsError>) -> String {
    match result {
        Err(OperationsError::InvalidInput { reason }) => reason,
        Err(other) => panic!("wanted InvalidInput, got {other}"),
        Ok(_) => panic!("wanted a refusal, got a face"),
    }
}

#[test]
fn non_positive_weights_are_refused() {
    for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        for at_a in [false, true] {
            let mut topo = Topology::new();
            let a = sheet(&mut topo, "plane", Pose::upright());
            let b = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
            let options = BlendSurfaceOptions {
                continuity: BlendContinuity::G1,
                weight_a: if at_a { weight } else { 1.0 },
                weight_b: if at_a { 1.0 } else { weight },
            };
            let reason = refused(blend_surface(
                &mut topo, &a.chain, a.face, &b.chain, b.face, &options,
            ));
            assert!(reason.contains("weight"), "{reason}");
        }
    }
}

#[test]
fn an_edge_not_on_the_face_is_refused() {
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "plane", Pose::upright());
    let b = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
    let reason = refused(blend_surface(
        &mut topo,
        &b.chain,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G1),
    ));
    assert!(reason.contains("not on the face"), "{reason}");
}

#[test]
fn a_chain_with_a_gap_or_repeat_or_nothing_is_refused() {
    let options = BlendSurfaceOptions::new(BlendContinuity::G1);
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "plane", Pose::upright());
    let b = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
    let opposite = vec![a.boundary[0], a.boundary[2]];
    let reason = refused(blend_surface(
        &mut topo, &opposite, a.face, &b.chain, b.face, &options,
    ));
    assert!(reason.contains("does not meet"), "{reason}");
    let repeated = vec![a.boundary[0], a.boundary[0]];
    let reason = refused(blend_surface(
        &mut topo, &repeated, a.face, &b.chain, b.face, &options,
    ));
    assert!(reason.contains("repeats"), "{reason}");
    let reason = refused(blend_surface(
        &mut topo,
        &[],
        a.face,
        &b.chain,
        b.face,
        &options,
    ));
    assert!(reason.contains("at least one"), "{reason}");
}

#[test]
fn a_closed_chain_against_an_open_one_is_refused() {
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "plane", Pose::upright());
    let b = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
    let reason = refused(blend_surface(
        &mut topo,
        &a.boundary,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    ));
    assert!(reason.contains("closed"), "{reason}");
}

#[test]
fn a_seam_is_refused() {
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 2.0, 4.0).unwrap();
    let side = lateral_face(&topo, solid);
    let seam = face_edges(&topo, side)
        .unwrap()
        .into_iter()
        .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Line))
        .unwrap();
    let other = sheet(
        &mut topo,
        "plane",
        Pose::turn(Vec3::new(0.0, 1.0, 0.0), 0.3, [9.0, 0.0, 0.0]),
    );
    let reason = refused(blend_surface(
        &mut topo,
        &[seam],
        side,
        &other.chain,
        other.face,
        &BlendSurfaceOptions::new(BlendContinuity::G1),
    ));
    assert!(reason.contains("seam"), "{reason}");
}

#[test]
fn an_edge_off_its_surface_is_refused() {
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "plane", Pose::upright());
    let b = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
    let outer = topo.face(a.face).unwrap().outer_wire();
    let wire = topo.wire(outer).unwrap().clone();
    let bad_surface = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 3.0,
    };
    drop(wire);
    topo.face_mut(a.face).unwrap().set_surface(bad_surface);
    let reason = refused(blend_surface(
        &mut topo,
        &a.chain,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G1),
    ));
    assert!(reason.contains("leaves"), "{reason}");
}

#[test]
fn chains_touching_midway_are_refused() {
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "plane", Pose::upright());
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
    let v = [
        b.vertex(p(3.0, 0.0)),
        b.vertex(p(0.0, 5.0)),
        b.vertex(p(3.0, 10.0)),
        b.vertex(p(8.0, 5.0)),
    ];
    let e0 = b.line(v[0], v[1]);
    let e1 = b.line(v[1], v[2]);
    let e2 = b.line(v[2], v[3]);
    let e3 = b.line(v[3], v[0]);
    let wire = b.topo.add_wire(
        Wire::new(
            [e0, e1, e2, e3]
                .iter()
                .map(|&e| OrientedEdge::new(e, true))
                .collect(),
            true,
        )
        .unwrap(),
    );
    let face = b.topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let reason = refused(blend_surface(
        &mut topo,
        &a.chain,
        a.face,
        &[e0, e1],
        face,
        &BlendSurfaceOptions::new(BlendContinuity::G1),
    ));
    assert!(reason.contains("touch"), "{reason}");
}

/// A G1 surface shows its curvature gap, and a turned surface shows its angle:
/// the measure is not reading zero for want of looking.
#[test]
fn the_measure_sees_what_a_blend_does_not_promise() {
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "cylinder across", Pose::turned());
    let b = sheet(
        &mut topo,
        "cylinder across",
        across_pose("cylinder across", Pose::turned()),
    );
    let face = blend_surface(
        &mut topo,
        &a.chain,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G1),
    )
    .unwrap();
    let result = nurbs_of(&topo, face);
    let seam = measure_seam(&result, 0.0, &surface_of_face(&topo, a.face), SAMPLES);
    assert!(seam.g1 < 1e-7, "{seam:?}");
    assert!(
        seam.g2 > 1e-2,
        "a cubic does not match a radius of 5: {seam:?}"
    );

    let c = topo.face(a.face).unwrap().surface().clone();
    let FaceSurface::Cylinder(cylinder) = c else {
        panic!("a cylinder fixture");
    };
    let fatter = brepkit_math::surfaces::CylindricalSurface::new(
        cylinder.origin(),
        cylinder.axis(),
        cylinder.radius() * 1.01,
    )
    .unwrap();
    let off_by_a_percent = measure_seam(&result, 0.0, &FaceSurface::Cylinder(fatter), SAMPLES);
    assert!(off_by_a_percent.g0 > 1e-3, "{off_by_a_percent:?}");

    let tilted = Pose::turn(Vec3::new(1.0, 0.0, 0.0), 0.01, [0.0, 0.0, 0.0]);
    let plane = sheet(&mut topo, "plane", Pose::upright());
    let g1_face = {
        let far = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
        blend_surface(
            &mut topo,
            &plane.chain,
            plane.face,
            &far.chain,
            far.face,
            &BlendSurfaceOptions::new(BlendContinuity::G1),
        )
        .unwrap()
    };
    let flat = nurbs_of(&topo, g1_face);
    let normal = tilted.vector(Vec3::new(0.0, 0.0, 1.0));
    let tilted_plane = FaceSurface::Plane { normal, d: 0.0 };
    let turned_away = measure_seam(&flat, 0.0, &tilted_plane, SAMPLES);
    assert!((turned_away.g1 - 0.01).abs() < 1e-3, "{turned_away:?}");
}

/// A spline neighbour's interior knots and a spline chain's own come through
/// without losing a digit: the fit breaks at them.
#[test]
fn spline_chains_and_knots_keep_their_continuity() {
    let mut topo = Topology::new();
    let a = sheet(&mut topo, "spline", Pose::mirrored());
    let b = sheet(&mut topo, "spline", across_pose("spline", Pose::mirrored()));
    let face = blend_surface(
        &mut topo,
        &a.chain,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    let result = nurbs_of(&topo, face);
    let distinct = |knots: &[f64]| {
        let mut unique = knots.to_vec();
        unique.dedup();
        unique.len()
    };
    assert!(distinct(result.knots_u()) > 2);
    let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
    assert_seams("spline", BlendContinuity::G2, &[seam_a, seam_b]);
    let edge = topo.edge(a.chain[0]).unwrap().curve();
    assert!(matches!(edge, EdgeCurve::NurbsCurve(_)));
}

/// A cap of a sphere above a circle: one closed edge, and curvature worth
/// matching. A wire winding round a surface bounds nothing in `(u, v)`, so it
/// goes by the kernel's rule that it keeps its face on its left about the
/// outward normal, and a mirrored one has to be walked the other way to do so.
fn sphere_cap(topo: &mut Topology, pose: Pose, center: Point3, at_z: f64, up: bool) -> Sheet {
    let radius: f64 = 5.0;
    let sign = if up { 1.0 } else { -1.0 };
    let ring = (radius * radius - at_z * at_z).sqrt();
    let level = center + Vec3::new(0.0, 0.0, sign * at_z);
    let mut b = Builder { topo, pose };
    let start = b.vertex(level + Vec3::new(ring, 0.0, 0.0));
    let edge = b.arc(level, Vec3::new(0.0, 0.0, sign), ring, start, start);
    let wire = b
        .topo
        .add_wire(Wire::new(vec![OrientedEdge::new(edge, !pose.flips())], true).unwrap());
    let surface = brepkit_math::surfaces::SphericalSurface::with_axis(
        pose.point(center),
        radius,
        pose.vector(Vec3::new(0.0, 0.0, 1.0)),
    )
    .unwrap();
    let face = b
        .topo
        .add_face(Face::new(wire, Vec::new(), FaceSurface::Sphere(surface)));
    Sheet {
        face,
        chain: vec![edge],
        inside: pose.point(center + Vec3::new(0.0, 0.0, sign * radius)),
        boundary: vec![edge],
    }
}

/// Closed rims with real curvature across them: two caps of spheres face to
/// face.
#[test]
fn closed_chains_keep_curvature_on_a_sphere() {
    for (pose_name, pose) in POSES {
        for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
            let mut topo = Topology::new();
            let a = sphere_cap(&mut topo, pose(), Point3::new(0.0, 0.0, 0.0), 4.0, false);
            let b = sphere_cap(&mut topo, pose(), Point3::new(0.0, 0.0, 12.0), 4.0, true);
            let face = blend_surface(
                &mut topo,
                &a.chain,
                a.face,
                &b.chain,
                b.face,
                &BlendSurfaceOptions::new(continuity),
            )
            .unwrap_or_else(|e| panic!("{pose_name}: {e}"));
            let result = nurbs_of(&topo, face);
            let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
            println!("sphere caps {pose_name} {continuity:?}: {seam_a:?}");
            assert_seams("sphere caps", continuity, &[seam_a, seam_b]);
            assert!(leaves_the_face(&result, 0.0, a.inside), "{pose_name}");
            assert!(leaves_the_face(&result, 1.0, b.inside), "{pose_name}");
            assert!(seam_a.curvature > 0.1, "{seam_a:?}");
            for i in 0..=20 {
                let v = f64::from(i) / 20.0;
                let gap = (result.evaluate(0.0, v) - result.evaluate(1.0, v)).length();
                assert!(gap < 1e-12, "{pose_name}: seam open by {gap}");
            }
        }
    }
}

/// A whole wall of a cylinder as a sheet, in a pose: a rim and a seam, and a
/// bridge from its top rim to the bottom rim of another, seams apart.
#[test]
fn closed_chains_on_whole_walls_hold_in_every_pose() {
    let wall = |topo: &mut Topology, pose: Pose, base: f64, turn: f64| -> Sheet {
        let mut b = Builder { topo, pose };
        let up = Vec3::new(0.0, 0.0, 1.0);
        let at = |z: f64| Point3::new(3.0 * turn.cos(), 3.0 * turn.sin(), z);
        let (low, high) = (b.vertex(at(base)), b.vertex(at(base + 4.0)));
        let bottom = b.arc(Point3::new(0.0, 0.0, base), up, 3.0, low, low);
        let top = b.arc(Point3::new(0.0, 0.0, base + 4.0), up, 3.0, high, high);
        let seam = b.line(low, high);
        let wire = b.topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(bottom, true),
                    OrientedEdge::new(seam, true),
                    OrientedEdge::new(top, false),
                    OrientedEdge::new(seam, false),
                ],
                true,
            )
            .unwrap(),
        );
        let surface = brepkit_math::surfaces::CylindricalSurface::new(
            pose.point(Point3::new(0.0, 0.0, base)),
            pose.vector(up),
            3.0,
        )
        .unwrap();
        let face = b
            .topo
            .add_face(Face::new(wire, Vec::new(), FaceSurface::Cylinder(surface)));
        Sheet {
            face,
            chain: vec![top, bottom],
            inside: pose.point(Point3::new(3.0, 0.0, base + 2.0)),
            boundary: vec![bottom, seam, top],
        }
    };
    for (pose_name, pose) in POSES {
        for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
            let mut topo = Topology::new();
            let mut a = wall(&mut topo, pose(), 0.0, 0.0);
            let mut b = wall(&mut topo, pose(), 7.0, 1.3);
            a.chain.truncate(1);
            b.chain.remove(0);
            let face = blend_surface(
                &mut topo,
                &a.chain,
                a.face,
                &b.chain,
                b.face,
                &BlendSurfaceOptions::new(continuity),
            )
            .unwrap_or_else(|e| panic!("{pose_name}: {e}"));
            assert_wire_sound(&topo, face);
            let result = nurbs_of(&topo, face);
            let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
            assert_seams(pose_name, continuity, &[seam_a, seam_b]);
            assert!(leaves_the_face(&result, 0.0, a.inside), "{pose_name}");
            assert!(leaves_the_face(&result, 1.0, b.inside), "{pose_name}");
            let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
            assert_eq!(wire.edges().len(), 4, "{pose_name}");
        }
    }
}

/// A chain that turns straight back has no direction off its face to agree on
/// at the turn.
#[test]
fn a_chain_doubling_back_is_refused() {
    let mut topo = Topology::new();
    let other = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
    let corners = [
        p(0.0, 0.0),
        p(0.0, 10.0),
        p(-0.1, 0.0),
        p(-5.0, 0.0),
        p(-5.0, 10.0),
    ];
    let v: Vec<_> = corners.iter().map(|&c| b.vertex(c)).collect();
    let out = b.line(v[0], v[1]);
    let back = b.line(v[1], v[2]);
    let e2 = b.line(v[2], v[3]);
    let e3 = b.line(v[3], v[4]);
    let e4 = b.line(v[4], v[0]);
    let wire = b.topo.add_wire(
        Wire::new(
            [out, back, e2, e3, e4]
                .iter()
                .map(|&e| OrientedEdge::new(e, true))
                .collect(),
            true,
        )
        .unwrap(),
    );
    let face = b.topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let reason = refused(blend_surface(
        &mut topo,
        &[out, back],
        face,
        &other.chain,
        other.face,
        &BlendSurfaceOptions::new(BlendContinuity::G1),
    ));
    assert!(reason.contains("doubles back"), "{reason}");
}

/// A sheet and a solid: the top of a box, along one edge, against a spline
/// sheet across from it. The solid's edge bounds two faces; the named one is
/// the neighbour.
#[test]
fn a_sheet_blends_to_a_solid() {
    for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 10.0, 10.0, 4.0).unwrap();
        let top = solid_faces(&topo, block)
            .unwrap()
            .into_iter()
            .find(|&f| match topo.face(f).unwrap().surface() {
                FaceSurface::Plane { normal, d } => normal.z() > 0.5 && (*d - 4.0).abs() < 1e-9,
                _ => false,
            })
            .unwrap();
        let edge = topo
            .wire(topo.face(top).unwrap().outer_wire())
            .unwrap()
            .edges()
            .iter()
            .map(OrientedEdge::edge)
            .find(|&e| {
                let data = topo.edge(e).unwrap();
                let (p, q) = (
                    topo.vertex(data.start()).unwrap().point(),
                    topo.vertex(data.end()).unwrap().point(),
                );
                p.x().abs().max(q.x().abs()) > 9.0 && (p.x() - q.x()).abs() < 1e-9
            })
            .unwrap();
        let place = Pose::turn(
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::PI,
            [24.0, 10.0, 6.0],
        );
        let other = sheet(&mut topo, "spline", place);
        let face = blend_surface(
            &mut topo,
            &[edge],
            top,
            &other.chain,
            other.face,
            &BlendSurfaceOptions::new(continuity),
        )
        .unwrap();
        let result = nurbs_of(&topo, face);
        let at_top = measure_seam(&result, 0.0, &surface_of_face(&topo, top), SAMPLES);
        let at_sheet = measure_seam(&result, 1.0, &surface_of_face(&topo, other.face), SAMPLES);
        assert_seams("sheet to solid", continuity, &[at_top, at_sheet]);
        let leaving = result.derivatives(0.5, 0.0, 1)[0][1];
        assert!(leaving.x() > 0.0, "off the box's top: {leaving:?}");
    }
}

/// Two squares, the second turned a quarter and a bit: same edge counts, so
/// they pair corner to corner, whichever corner is nearest, without a cut.
#[test]
fn closed_chains_of_equal_counts_pair_corner_to_corner() {
    let mut topo = Topology::new();
    let lower = make_box(&mut topo, 6.0, 6.0, 2.0).unwrap();
    let upper = make_box(&mut topo, 6.0, 6.0, 2.0).unwrap();
    let place = Mat4::translation(3.0, 3.0, 8.0)
        * Mat4::rotation_z(std::f64::consts::FRAC_PI_2 + 0.1)
        * Mat4::translation(-3.0, -3.0, 0.0);
    transform_solid(&mut topo, upper, &place).unwrap();
    let plane_face = |topo: &Topology, solid, up: bool, height: f64| {
        solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&f| match topo.face(f).unwrap().surface() {
                FaceSurface::Plane { normal, d } => {
                    (normal.z() > 0.5) == up
                        && normal.z().abs() > 0.5
                        && (d.abs() - height).abs() < 1e-9
                }
                _ => false,
            })
            .unwrap()
    };
    let top = plane_face(&topo, lower, true, 2.0);
    let bottom = plane_face(&topo, upper, false, 8.0);
    let edges = |topo: &Topology, face: FaceId| -> Vec<EdgeId> {
        topo.wire(topo.face(face).unwrap().outer_wire())
            .unwrap()
            .edges()
            .iter()
            .map(OrientedEdge::edge)
            .collect()
    };
    let (chain_a, chain_b) = (edges(&topo, top), edges(&topo, bottom));
    let face = blend_surface(
        &mut topo,
        &chain_a,
        top,
        &chain_b,
        bottom,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    assert_wire_sound(&topo, face);
    let wire_len = topo
        .wire(topo.face(face).unwrap().outer_wire())
        .unwrap()
        .edges()
        .len();
    assert_eq!(
        wire_len,
        4 + 4 + 2,
        "each side keeps its four edges, and one seam twice"
    );
    let result = nurbs_of(&topo, face);
    for i in 0..4 {
        let u = f64::from(i) / 4.0;
        let (p, q) = (result.evaluate(u, 0.0), result.evaluate(u, 1.0));
        let across = q - p;
        let sideways = (across.x() * across.x() + across.y() * across.y()).sqrt();
        assert!(sideways < 1.0, "corners do not pair at {u}: {p:?} to {q:?}");
    }
    let seam = measure_seam(&result, 0.0, &surface_of_face(&topo, top), SAMPLES);
    assert_seams("squares", BlendContinuity::G2, &[seam]);
}

/// A line, a spline along a helix and an arc on one cylinder, each stored
/// either way. A spline can only lie on a cylinder to its fit, and its tangent
/// plane with it, so the figures here are held to that and not to rounding.
#[test]
fn a_mixed_chain_keeps_its_continuity_on_a_curved_face() {
    let cylinder_sheet = |topo: &mut Topology, pose: Pose, flips: [bool; 3]| -> Sheet {
        let mut b = Builder { topo, pose };
        let at = |theta: f64, y: f64| Point3::new(5.0 * theta.cos(), y, 5.0 * theta.sin());
        let helix: Vec<Point3> = (0..=80)
            .map(|k| {
                let t = f64::from(k) / 80.0;
                at(0.4 + 0.5 * t, 4.0 + 2.0 * t)
            })
            .collect();
        let v = [
            b.vertex(at(0.4, 0.0)),
            b.vertex(at(0.4, 4.0)),
            b.vertex(at(0.9, 6.0)),
            b.vertex(at(2.0, 6.0)),
            b.vertex(at(2.0, 0.0)),
        ];
        let line = if flips[0] {
            b.line(v[1], v[0])
        } else {
            b.line(v[0], v[1])
        };
        let spline = if flips[1] {
            let back: Vec<Point3> = helix.iter().rev().copied().collect();
            b.spline(&back, v[2], v[1])
        } else {
            b.spline(&helix, v[1], v[2])
        };
        let up = Vec3::new(0.0, 1.0, 0.0);
        let round = Point3::new(0.0, 6.0, 0.0);
        let arc = if flips[2] {
            b.arc(round, up, 5.0, v[3], v[2])
        } else {
            b.arc(round, -up, 5.0, v[2], v[3])
        };
        let far = b.line(v[3], v[4]);
        let bottom = b.arc(Point3::new(0.0, 0.0, 0.0), up, 5.0, v[4], v[0]);
        let walk = vec![
            OrientedEdge::new(line, !flips[0]),
            OrientedEdge::new(spline, !flips[1]),
            OrientedEdge::new(arc, !flips[2]),
            OrientedEdge::new(far, true),
            OrientedEdge::new(bottom, true),
        ];
        let wire = b.topo.add_wire(Wire::new(walk, true).unwrap());
        let surface = brepkit_math::surfaces::CylindricalSurface::new(
            pose.point(Point3::new(0.0, 0.0, 0.0)),
            pose.vector(up),
            5.0,
        )
        .unwrap();
        let face = b
            .topo
            .add_face(Face::new(wire, Vec::new(), FaceSurface::Cylinder(surface)));
        Sheet {
            face,
            chain: vec![line, spline, arc],
            inside: pose.point(at(1.5, 3.0)),
            boundary: Vec::new(),
        }
    };
    for (pose_name, pose) in POSES {
        for flips in [[false, false, false], [true, true, true]] {
            let mut topo = Topology::new();
            let a = cylinder_sheet(&mut topo, pose(), flips);
            let across = Pose::reflection(Point3::new(9.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.3));
            let b = cylinder_sheet(&mut topo, across.then(pose()), [false; 3]);
            let face = blend_surface(
                &mut topo,
                &a.chain,
                a.face,
                &b.chain,
                b.face,
                &BlendSurfaceOptions::new(BlendContinuity::G2),
            )
            .unwrap_or_else(|e| panic!("{pose_name} {flips:?}: {e}"));
            let result = nurbs_of(&topo, face);
            let seam_a = measure_seam(&result, 0.0, &surface_of_face(&topo, a.face), SAMPLES);
            let seam_b = measure_seam(&result, 1.0, &surface_of_face(&topo, b.face), SAMPLES);
            println!("line, helix, arc on a cylinder {pose_name}: {seam_a:?}");
            for seam in [seam_a, seam_b] {
                assert!(seam.g0 < 1e-6, "{pose_name}: {seam:?}");
                assert!(seam.g1 < 1e-5, "{pose_name}: {seam:?}");
                assert!(seam.g2 < 1e-3, "{pose_name}: {seam:?}");
            }
            assert!(gap_to_chain(&topo, &result, 0.0, &a.chain, 100) < 1e-8);
        }
    }
}

/// A spline neighbour with only curvature-breaking continuity at a knot line
/// the chain crosses: it has no one curvature there, so the blend closes the
/// jump over a narrow band and holds G2 on either side of it, and G1 throughout.
#[test]
fn a_neighbour_with_a_curvature_jump_is_matched_either_side_of_it() {
    let net: Vec<Vec<Point3>> = (0..5)
        .map(|i| {
            let x = -4.0 + f64::from(i);
            (0..6)
                .map(|j| {
                    let y = 2.0 * f64::from(j);
                    let bend = [0.0, 0.8, -0.6, 0.9, -0.4, 0.3][j as usize];
                    Point3::new(x + 0.4 * y, y, bend * (1.0 + 0.1 * x))
                })
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; 6]; 5];
    let knots_u = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0];
    let build = |topo: &mut Topology, pose: Pose| -> Sheet {
        let mut b = Builder { topo, pose };
        let (face, edges) = net_sheet(
            &mut b,
            (3, 2),
            (knots_u.clone(), knots_v.clone()),
            &net,
            &weights,
        );
        Sheet {
            face,
            chain: vec![edges[1]],
            inside: pose.point(net[2][2]),
            boundary: edges.to_vec(),
        }
    };
    let mut topo = Topology::new();
    let a = build(&mut topo, Pose::turned());
    let across =
        Pose::reflection(Point3::new(6.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).then(Pose::turned());
    let b = build(&mut topo, across);
    let face = blend_surface(
        &mut topo,
        &a.chain,
        a.face,
        &b.chain,
        b.face,
        &BlendSurfaceOptions::new(BlendContinuity::G2),
    )
    .unwrap();
    let result = nurbs_of(&topo, face);
    let neighbour = surface_of_face(&topo, a.face);
    let whole = measure_seam(&result, 0.0, &neighbour, 1600);
    assert!(whole.g0 < 1e-8, "{whole:?}");
    assert!(whole.g1 < 1e-7, "{whole:?}");
    assert!(whole.g2 > 1e-3, "the jump should show up: {whole:?}");

    let mut settled = 0;
    let mut in_band = 0;
    for i in 0..=1600 {
        let u = f64::from(i) / 1600.0;
        let seam = measure_seam_at(&result, 0.0, &neighbour, &[u]);
        if seam.g2 < 1e-5 {
            settled += 1;
        } else {
            in_band += 1;
        }
    }
    println!("curvature jump: {whole:?}; {settled} samples hold G2, {in_band} in the bands");
    assert!(in_band * 4 < settled, "{in_band} of {settled}");
}

/// A faceted neighbour has no tangent plane at a crease: no surface could be
/// G1 there, and the blend says so.
#[test]
fn a_crease_in_the_neighbour_is_refused() {
    let peaks = [0.0, 1.0, -0.5, 0.8, 0.0];
    let net: Vec<Vec<Point3>> = (0..3)
        .map(|i| {
            let x = -4.0 + 2.0 * f64::from(i);
            (0..5)
                .map(|j| Point3::new(x, 2.5 * j as f64, peaks[j] * (1.0 + 0.2 * x)))
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; 5]; 3];
    let knots_u = vec![0.0, 0.0, 0.5, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0];
    let build = |topo: &mut Topology, pose: Pose| -> (FaceId, EdgeId) {
        let mut b = Builder { topo, pose };
        let (face, edges) = net_sheet(
            &mut b,
            (1, 1),
            (knots_u.clone(), knots_v.clone()),
            &net,
            &weights,
        );
        (face, edges[1])
    };
    let mut topo = Topology::new();
    let (face_a, edge_a) = build(&mut topo, Pose::upright());
    let other = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
    let reason = refused(blend_surface(
        &mut topo,
        &[edge_a],
        face_a,
        &other.chain,
        other.face,
        &BlendSurfaceOptions::new(BlendContinuity::G1),
    ));
    assert!(reason.contains("crease"), "{reason}");
}

/// An ellipse arc for a chain, stored either way: not arc-length parameterised,
/// so its tangent and its speed both have work to do.
#[test]
fn an_ellipse_arc_is_a_chain_too() {
    let build = |topo: &mut Topology, pose: Pose, against: bool| -> Sheet {
        let mut b = Builder { topo, pose };
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let (bottom, top) = (p(0.0, -5.0), p(0.0, 5.0));
        let v = [
            b.vertex(bottom),
            b.vertex(top),
            b.vertex(p(-6.0, 5.0)),
            b.vertex(p(-6.0, -5.0)),
        ];
        let flip = if pose.flips() { -1.0 } else { 1.0 };
        let normal = pose.vector(Vec3::new(0.0, 0.0, 1.0));
        let v_axis = pose.vector(Vec3::new(0.0, 1.0, 0.0)) * flip;
        let curve = |normal: Vec3, v_axis: Vec3| {
            EdgeCurve::Ellipse(
                brepkit_math::curves::Ellipse3D::with_axes(
                    pose.point(p(0.0, 0.0)),
                    normal,
                    5.0,
                    2.0,
                    pose.vector(Vec3::new(0.0, -1.0, 0.0)),
                    pose.vector(Vec3::new(2.0, 0.0, 0.0)).normalize().unwrap() * flip,
                )
                .unwrap_or_else(|e| panic!("{e}: {v_axis:?}")),
            )
        };
        let along = if against {
            b.topo
                .add_edge(Edge::new(v[1], v[0], curve(normal * flip, v_axis)))
        } else {
            b.topo
                .add_edge(Edge::new(v[0], v[1], curve(normal * flip, v_axis)))
        };
        let rest = [b.line(v[1], v[2]), b.line(v[2], v[3]), b.line(v[3], v[0])];
        let mut walk = vec![OrientedEdge::new(along, !against)];
        walk.extend(rest.iter().map(|&e| OrientedEdge::new(e, true)));
        let wire = b.topo.add_wire(Wire::new(walk, true).unwrap());
        let d = normal.dot(pose.point(p(0.0, 0.0)) - Point3::new(0.0, 0.0, 0.0));
        let face = b.topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane { normal, d },
        ));
        Sheet {
            face,
            chain: vec![along],
            inside: pose.point(p(-3.0, 0.0)),
            boundary: Vec::new(),
        }
    };
    let across = Pose::reflection(Point3::new(5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    for (pose_name, pose) in POSES {
        let mut topo = Topology::new();
        let a = build(&mut topo, pose(), false);
        let b = build(&mut topo, across.then(pose()), false);
        let face = blend_surface(
            &mut topo,
            &a.chain,
            a.face,
            &b.chain,
            b.face,
            &BlendSurfaceOptions::new(BlendContinuity::G2),
        )
        .unwrap_or_else(|e| panic!("{pose_name}: {e}"));
        let result = nurbs_of(&topo, face);
        let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
        assert_seams(pose_name, BlendContinuity::G2, &[seam_a, seam_b]);
    }
}

/// The same fixture at a thousandth and a thousand times the size: the seam's
/// angle stays where it was, and its curvature gap in step with the size.
#[test]
fn a_blend_holds_at_other_sizes() {
    for size in [1e-3, 1e3] {
        let pose = Pose::turned().then(Pose::scaled(size));
        let mut topo = Topology::new();
        let a = sheet(&mut topo, "spline", pose);
        let b = sheet(&mut topo, "spline", across_pose("spline", pose));
        let face = blend_surface(
            &mut topo,
            &a.chain,
            a.face,
            &b.chain,
            b.face,
            &BlendSurfaceOptions::new(BlendContinuity::G2),
        )
        .unwrap_or_else(|e| panic!("at {size}: {e}"));
        let result = nurbs_of(&topo, face);
        let seam = measure_seam(&result, 0.0, &surface_of_face(&topo, a.face), SAMPLES);
        println!("at size {size}: {seam:?}");
        assert!(seam.g0 < 1e-8 * size, "at {size}: {seam:?}");
        assert!(seam.g1 < 1e-7, "at {size}: {seam:?}");
        assert!(seam.g2 < 1e-6 / size, "at {size}: {seam:?}");
        assert!(seam.curvature > 0.05 / size, "at {size}: {seam:?}");
    }
}

/// A rational spline form of a sphere and of a torus for the neighbour: knots
/// where the spline is only C1 in its parameters though smooth in its shape,
/// which a chain crossing them should not notice.
#[test]
fn a_neighbour_smooth_in_shape_across_its_knots_is_no_trouble() {
    for (kind, to_spline) in [
        ("sphere meridian", true),
        ("sphere parallel", true),
        ("torus ring", false),
        ("torus tube", false),
    ] {
        for (pose_name, pose) in POSES {
            let mut topo = Topology::new();
            let a = sheet(&mut topo, kind, pose());
            let b = sheet(&mut topo, kind, across_pose(kind, pose()));
            for side in [a.face, b.face] {
                let surface = surface_of_face(&topo, side);
                let spline = match (&surface, to_spline) {
                    (FaceSurface::Sphere(s), true) => {
                        brepkit_heal::construct::convert_surface::sphere_to_nurbs(s).unwrap()
                    }
                    (FaceSurface::Torus(t), false) => {
                        brepkit_heal::construct::convert_surface::torus_to_nurbs(t).unwrap()
                    }
                    _ => panic!("{kind}: unexpected surface"),
                };
                topo.face_mut(side)
                    .unwrap()
                    .set_surface(FaceSurface::Nurbs(spline));
            }
            let face = blend_surface(
                &mut topo,
                &a.chain,
                a.face,
                &b.chain,
                b.face,
                &BlendSurfaceOptions::new(BlendContinuity::G2),
            )
            .unwrap_or_else(|e| panic!("{kind} {pose_name}: {e}"));
            let result = nurbs_of(&topo, face);
            let (seam_a, seam_b) = both_seams(&topo, &result, &a, &b);
            println!("{kind} as a rational spline, {pose_name}: {seam_a:?}");
            assert_seams(kind, BlendContinuity::G2, &[seam_a, seam_b]);
            assert!(
                leaves_the_face(&result, 0.0, a.inside),
                "{kind} {pose_name}"
            );
            assert!(seam_a.curvature > 0.1, "{seam_a:?}");
        }
    }
}
