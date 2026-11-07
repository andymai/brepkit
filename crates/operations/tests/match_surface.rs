//! Moving a surface's edge on to another's, with each continuity measured
//! along the seam in several poses, and the rest of the surface checked for
//! staying put.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::panic
)]

mod surface_fixtures;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::blend_surface::{MatchContinuity, match_surface, match_surface_on};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::tessellate;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{face_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::wire::{OrientedEdge, Wire};
use surface_fixtures::{
    Builder, KINDS, Pose, Worst, across_pose, assert_wire_sound, chain_exit, foot_on, gap_to_chain,
    measure_seam, net_sheet, sheet, surface_of,
};

const SAMPLES: usize = 400;
type MakePose = fn() -> Pose;
const POSES: [(&str, MakePose); 3] = [
    ("upright", Pose::upright),
    ("turned", Pose::turned),
    ("mirrored", Pose::mirrored),
];
const ALL: [MatchContinuity; 3] = [
    MatchContinuity::G0,
    MatchContinuity::G1,
    MatchContinuity::G2,
];

fn nurbs_of(topo: &Topology, face: FaceId) -> NurbsSurface {
    match topo.face(face).unwrap().surface() {
        FaceSurface::Nurbs(surface) => surface.clone(),
        other => panic!(
            "a matched surface should be a spline, got {}",
            other.type_tag()
        ),
    }
}

fn surface_of_face(topo: &Topology, face: FaceId) -> FaceSurface {
    topo.face(face).unwrap().surface().clone()
}

fn assert_seam(context: &str, continuity: MatchContinuity, seam: &Worst) {
    assert!(seam.g0 < 1e-8, "{context}: {seam:?}");
    if continuity != MatchContinuity::G0 {
        assert!(seam.g1 < 1e-7, "{context}: {seam:?}");
    }
    if continuity == MatchContinuity::G2 {
        assert!(seam.g2 < 1e-6, "{context}: {seam:?}");
    }
}

/// A smooth spline sheet leaving `chain` along `exit`, its near edge `offset`
/// off the chain and its far side `width` along: not quite in place, not quite
/// tangent, and with no curvature to speak of along the way.
fn source_near(
    topo: &mut Topology,
    target: EdgeId,
    exit: Vec3,
    lift: Vec3,
    width: f64,
) -> (FaceId, EdgeId, NurbsSurface) {
    let data = topo.edge(target).unwrap();
    let start = topo.vertex(data.start()).unwrap().point();
    let end = topo.vertex(data.end()).unwrap().point();
    let (t0, t1) = data.curve().domain_with_endpoints(start, end);
    let columns = 9;
    let rows = 9;
    let mut points: Vec<Vec<Point3>> = Vec::with_capacity(columns);
    for i in 0..columns {
        let along = i as f64 / (columns - 1) as f64;
        let at = data
            .curve()
            .evaluate_with_endpoints(t0 + (t1 - t0) * along, start, end);
        let mut row = Vec::with_capacity(rows);
        for j in 0..rows {
            let t = j as f64 / (rows - 1) as f64;
            let bow = (1.0 - t) * (1.0 - t);
            row.push(
                at + exit * (0.3 + width * t)
                    + lift * bow
                    + Vec3::new(0.0, 0.0, 0.1) * (along * 3.0).sin() * bow,
            );
        }
        points.push(row);
    }
    let surface = interpolate_surface(&points, 3, 3).unwrap();
    let net = surface.control_points().to_vec();
    let weights = surface.weights().to_vec();
    let mut b = Builder {
        topo,
        pose: Pose::upright(),
    };
    let (face, edges) = net_sheet(
        &mut b,
        (surface.degree_u(), surface.degree_v()),
        (surface.knots_u().to_vec(), surface.knots_v().to_vec()),
        &net,
        &weights,
    );
    (face, edges[0], surface)
}

#[derive(Default)]
struct Table {
    rows: Vec<(String, [Worst; 3])>,
}

impl Table {
    fn add(&mut self, surface: &str, continuity: MatchContinuity, worst: Worst) {
        let at = if let Some(at) = self.rows.iter().position(|row| row.0 == surface) {
            at
        } else {
            self.rows.push((surface.to_string(), [Worst::default(); 3]));
            self.rows.len() - 1
        };
        let slot = match continuity {
            MatchContinuity::G0 => 0,
            MatchContinuity::G1 => 1,
            MatchContinuity::G2 => 2,
        };
        self.rows[at].1[slot].join(worst);
    }

    fn print(&self, title: &str) {
        println!("{title}");
        println!(
            "{:<10} {:>10} | {:>10} {:>10} | {:>10} {:>10} {:>10} {:>9}",
            "surface", "G0:gap", "G1:gap", "G1:angle", "G2:gap", "G2:angle", "G2:dk", "|k|max"
        );
        for (name, [g0, g1, g2]) in &self.rows {
            println!(
                "{name:<10} {:>10.2e} | {:>10.2e} {:>10.2e} | {:>10.2e} {:>10.2e} {:>10.2e} {:>9.3}",
                g0.g0, g1.g0, g1.g1, g2.g0, g2.g1, g2.g2, g2.curvature
            );
        }
    }
}

/// A spline sheet on to each surface type's edge, in every pose, at each
/// continuity.
#[test]
fn a_spline_matches_every_surface_type_in_every_pose() {
    let mut table = Table::default();
    for kind in KINDS {
        for (pose_name, pose) in POSES {
            for continuity in ALL {
                let mut topo = Topology::new();
                let target = sheet(&mut topo, kind, pose());
                let (_, exit) = chain_exit(kind);
                let exit = pose().vector(exit);
                let lift = pose().vector(Vec3::new(0.05, 0.07, 0.11));
                let (face, edge, _) = source_near(&mut topo, target.chain[0], exit, lift, 6.0);
                let result = match_surface(&mut topo, face, edge, target.chain[0], continuity)
                    .unwrap_or_else(|e| panic!("{kind} {pose_name} {continuity:?}: {e}"));
                let surface = nurbs_of(&topo, result);
                let mut seam =
                    measure_seam(&surface, 0.0, &surface_of_face(&topo, target.face), SAMPLES);
                seam.g0 = seam
                    .g0
                    .max(gap_to_chain(&topo, &surface, 0.0, &target.chain, 200));
                assert_seam(
                    &format!("{kind} {pose_name} {continuity:?}"),
                    continuity,
                    &seam,
                );
                table.add(surface_of(kind), continuity, seam);
            }
        }
    }
    table.print("match_surface, a spline on to each type: worst over 400 samples, three poses");
}

/// Every surface type as the source, on to the matching edge of a copy of
/// itself.
#[test]
fn every_surface_type_can_be_matched() {
    let mut table = Table::default();
    for kind in KINDS {
        for (pose_name, pose) in POSES {
            for continuity in ALL {
                let mut topo = Topology::new();
                let source = sheet(&mut topo, kind, pose());
                let (at, exit) = chain_exit(kind);
                let mirror = Pose::reflection(at + exit * 0.3, exit);
                let target = sheet(&mut topo, kind, mirror.then(pose()));
                let result = match_surface(
                    &mut topo,
                    source.face,
                    source.chain[0],
                    target.chain[0],
                    continuity,
                )
                .unwrap_or_else(|e| panic!("{kind} {pose_name} {continuity:?}: {e}"));
                let surface = nurbs_of(&topo, result);
                assert_wire_sound(&topo, result);
                let mut seam =
                    measure_seam(&surface, 0.0, &surface_of_face(&topo, target.face), SAMPLES);
                seam.g0 = seam
                    .g0
                    .max(gap_to_chain(&topo, &surface, 0.0, &target.chain, 200));
                let context = format!("{kind} {pose_name} {continuity:?}");
                assert_seam(&context, continuity, &seam);
                table.add(surface_of(kind), continuity, seam);

                // Its far side is where it was.
                let original = surface_of_face(&topo, source.face);
                let far = far_side_distance(&surface, &original, 1.0);
                assert!(far < 1e-7, "{context}: the far side moved {far}");
                // The result still faces the way the source did.
                let at_inside = source.inside;
                let (fu, fv) = foot_on(&original, at_inside);
                let facing = original.normal(fu, fv);
                let (mut best, mut own) = (f64::INFINITY, Vec3::new(0.0, 0.0, 0.0));
                for i in 1..24 {
                    for j in 1..24 {
                        let (u, v) = (f64::from(i) / 24.0, f64::from(j) / 24.0);
                        let gap = (surface.evaluate(u, v) - at_inside).length();
                        if gap < best {
                            best = gap;
                            own = surface.normal(u, v).unwrap();
                        }
                    }
                }
                assert!(own.dot(facing) > 0.0, "{context}: faces the other way");
            }
        }
    }
    table.print("match_surface, each type as the source: worst over 400 samples, three poses");
}

/// The farthest the row of `surface` at `v` is from `original`.
fn far_side_distance(surface: &NurbsSurface, original: &FaceSurface, v: f64) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..=120 {
        let p = surface.evaluate(f64::from(i) / 120.0, v);
        let gap = if let FaceSurface::Plane { normal, d } = original {
            ((p - Point3::new(0.0, 0.0, 0.0)).dot(*normal) - d).abs() / normal.length()
        } else {
            let (u, w) = foot_on(original, p);
            (original.evaluate(u, w).unwrap() - p).length()
        };
        worst = worst.max(gap);
    }
    worst
}

/// `source` at parameters read off the matched result: `s` along and `t` away,
/// whichever way round the net was.
fn deviation_by_row(result: &NurbsSurface, source: &NurbsSurface, rows: usize) -> Vec<(f64, f64)> {
    let same_way = (result.evaluate(1.0, 1.0) - source.evaluate(1.0, 1.0)).length()
        <= (result.evaluate(1.0, 1.0) - source.evaluate(0.0, 1.0)).length();
    (0..=rows)
        .map(|j| {
            let t = j as f64 / rows as f64;
            let worst = (0..=60)
                .map(|i| {
                    let s = f64::from(i) / 60.0;
                    let on_source = source.evaluate(if same_way { s } else { 1.0 - s }, t);
                    (result.evaluate(s, t) - on_source).length()
                })
                .fold(0.0, f64::max);
            (t, worst)
        })
        .collect()
}

/// A small move touches a quarter of the surface and leaves the rest alone, to
/// rounding: far from where a whole new surface would be.
#[test]
fn a_small_move_leaves_the_rest_of_the_surface_alone() {
    for continuity in ALL {
        let mut topo = Topology::new();
        let target = sheet(&mut topo, "plane", Pose::turned());
        let mut b = Builder {
            topo: &mut topo,
            pose: Pose::turned(),
        };
        let points: Vec<Vec<Point3>> = (0..=10)
            .map(|i| {
                let y = f64::from(i);
                (0..=10)
                    .map(|j| {
                        let x = f64::from(j);
                        let wave = (0.2 * x).cos() * (0.3 * y).sin() * 0.05 + 0.03;
                        Point3::new(0.04 + x, y, wave)
                    })
                    .collect()
            })
            .collect();
        let surface = interpolate_surface(&points, 3, 3).unwrap();
        let (face, edges) = net_sheet(
            &mut b,
            (3, 3),
            (surface.knots_u().to_vec(), surface.knots_v().to_vec()),
            surface.control_points(),
            surface.weights(),
        );
        let original = match topo.face(face).unwrap().surface() {
            FaceSurface::Nurbs(n) => n.clone(),
            _ => panic!("spline"),
        };
        let result = match_surface(&mut topo, face, edges[0], target.chain[0], continuity).unwrap();
        let moved = nurbs_of(&topo, result);
        let deviations = deviation_by_row(&moved, &original, 20);
        println!("{continuity:?} deviation by row: {deviations:?}");
        for (t, deviation) in deviations {
            if t >= 0.25 {
                assert!(deviation < 1e-9, "at t = {t}: {deviation}");
            }
        }
        // And the source's own net survives where the band does not reach.
        let source_rows = original.control_points()[0].len();
        assert!(moved.control_points()[0].len() >= source_rows);
    }
}

/// With a large move the band widens to the whole width, the far edge still
/// held, and nothing folds.
#[test]
fn a_large_move_stays_in_one_piece() {
    let mut topo = Topology::new();
    let target = sheet(&mut topo, "cylinder across", Pose::turned());
    let (_, exit) = chain_exit("cylinder across");
    let exit = Pose::turned().vector(exit);
    let lift = Pose::turned().vector(Vec3::new(1.0, 2.0, 2.0));
    let (face, edge, _) = source_near(&mut topo, target.chain[0], exit, lift, 6.0);
    let result =
        match_surface(&mut topo, face, edge, target.chain[0], MatchContinuity::G2).unwrap();
    let surface = nurbs_of(&topo, result);
    let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, target.face), SAMPLES);
    assert_seam("large move", MatchContinuity::G2, &seam);
    for i in 1..30 {
        for j in 1..30 {
            let (u, v) = (f64::from(i) / 30.0, f64::from(j) / 30.0);
            let d = surface.derivatives(u, v, 1);
            let area = d[1][0].cross(d[0][1]).length();
            assert!(area > 1e-3, "folds at {u} {v}: {area}");
        }
    }
}

/// A plane's edge on to a cylinder's generator, when the plane meets it edge
/// on: the plane's own tangent has nothing off the face, so the surface has to
/// set out along the off direction and turn.
#[test]
fn a_source_that_runs_back_over_the_face_is_turned_out_of_it() {
    let mut topo = Topology::new();
    let target = sheet(&mut topo, "plane", Pose::turned());
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::turned(),
    };
    let corners = [
        Point3::new(0.2, 0.0, 0.0),
        Point3::new(-5.0, 0.0, 3.0),
        Point3::new(-5.0, 10.0, 3.0),
        Point3::new(0.2, 10.0, 0.0),
    ];
    let v: Vec<_> = corners.iter().map(|&p| b.vertex(p)).collect();
    let edges: Vec<_> = (0..4).map(|i| b.line(v[i], v[(i + 1) % 4])).collect();
    let wire = b.topo.add_wire(
        Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let normal = Pose::turned()
        .vector((corners[1] - corners[0]).cross(corners[3] - corners[0]))
        .normalize()
        .unwrap();
    let at = Pose::turned().point(corners[0]);
    let d = normal.dot(at - Point3::new(0.0, 0.0, 0.0));
    let face = b.topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane { normal, d },
    ));
    let result = match_surface(
        &mut topo,
        face,
        edges[3],
        target.chain[0],
        MatchContinuity::G2,
    )
    .unwrap();
    let surface = nurbs_of(&topo, result);
    let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, target.face), SAMPLES);
    assert_seam("run back", MatchContinuity::G2, &seam);
    assert!(surface_fixtures::leaves_the_face(
        &surface,
        0.0,
        target.inside
    ));
}

/// A disc's rim on to a cylinder's rim gives a dome tangent to the wall.
#[test]
fn a_disc_caps_a_rim() {
    for continuity in [MatchContinuity::G1, MatchContinuity::G2] {
        let mut topo = Topology::new();
        let post = make_cylinder(&mut topo, 3.0, 5.0).unwrap();
        let wall = solid_faces(&topo, post)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .unwrap();
        let rim = face_edges(&topo, wall)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let data = topo.edge(e).unwrap();
                matches!(data.curve(), EdgeCurve::Circle(_))
                    && (topo.vertex(data.start()).unwrap().point().z() - 5.0).abs() < 1e-9
            })
            .unwrap();
        let mut b = Builder {
            topo: &mut topo,
            pose: Pose::upright(),
        };
        let at = b.vertex(Point3::new(2.9, 0.0, 6.0));
        let edge = b.arc(
            Point3::new(0.0, 0.0, 6.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.9,
            at,
            at,
        );
        let wire = b
            .topo
            .add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
        let disc = b.topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 6.0,
            },
        ));
        let result = match_surface_on(&mut topo, disc, edge, rim, wall, continuity).unwrap();
        let surface = nurbs_of(&topo, result);
        let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, wall), SAMPLES);
        assert_seam("disc cap", continuity, &seam);
        println!("disc cap {continuity:?}: {seam:?}");
        for i in 0..=8 {
            let u = f64::from(i) / 8.0;
            let gap = (surface.evaluate(u, 0.5) - surface.evaluate(0.0, 0.5)).length();
            assert!(!(0.1..0.9).contains(&u) || gap > 1e-3);
            let closes = (surface.evaluate(0.0, u) - surface.evaluate(1.0, u)).length();
            assert!(closes < 1e-9, "the cap's seam opens by {closes}");
        }
        let pole = surface.evaluate(0.3, 1.0);
        assert!(
            (pole - Point3::new(0.0, 0.0, 6.0)).length() < 1e-7,
            "{pole:?}"
        );
        assert!(tessellate(&topo, result, 0.2).unwrap().indices.len() >= 3);
    }
}

/// A band between two rims on a cylinder's whole wall: both sides closed, one
/// moved on to a larger circle, the seam still one edge.
#[test]
fn a_whole_wall_matches_a_closed_edge() {
    let mut topo = Topology::new();
    let post = make_cylinder(&mut topo, 3.0, 5.0).unwrap();
    let wall = solid_faces(&topo, post)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .unwrap();
    let rim = face_edges(&topo, wall)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let data = topo.edge(e).unwrap();
            matches!(data.curve(), EdgeCurve::Circle(_))
                && (topo.vertex(data.start()).unwrap().point().z() - 5.0).abs() < 1e-9
        })
        .unwrap();
    let top = make_cylinder(&mut topo, 3.5, 4.0).unwrap();
    transform_solid(
        &mut topo,
        top,
        &(Mat4::translation(0.0, 0.0, 6.0) * Mat4::rotation_z(0.9)),
    )
    .unwrap();
    let upper = solid_faces(&topo, top)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .unwrap();
    let target = face_edges(&topo, upper)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let data = topo.edge(e).unwrap();
            matches!(data.curve(), EdgeCurve::Circle(_))
                && (topo.vertex(data.start()).unwrap().point().z() - 6.0).abs() < 1e-9
        })
        .unwrap();
    let result =
        match_surface_on(&mut topo, wall, rim, target, upper, MatchContinuity::G2).unwrap();
    let surface = nurbs_of(&topo, result);
    let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, upper), SAMPLES);
    assert_seam("whole wall", MatchContinuity::G2, &seam);
    assert_wire_sound(&topo, result);
    let wire = topo.wire(topo.face(result).unwrap().outer_wire()).unwrap();
    assert_eq!(wire.edges().len(), 4);
    assert_eq!(wire.edges()[1].edge(), wire.edges()[3].edge());
    let far = far_side_distance(&surface, &surface_of_face(&topo, wall), 1.0);
    assert!(far < 1e-7, "{far}");
}

/// A hole clear of the band stays, a copy on the new face; one in its way
/// narrows the band, and one too near is refused.
#[test]
fn a_hole_narrows_the_band_or_is_refused() {
    let plate = |topo: &mut Topology, hole_x: f64| -> (FaceId, EdgeId) {
        let mut b = Builder {
            topo,
            pose: Pose::upright(),
        };
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let corners = [p(0.5, 0.0), p(10.0, 0.0), p(10.0, 10.0), p(0.5, 10.0)];
        let v: Vec<_> = corners.iter().map(|&c| b.vertex(c)).collect();
        let outer: Vec<_> = (0..4).map(|i| b.line(v[i], v[(i + 1) % 4])).collect();
        let at = b.vertex(p(hole_x + 1.0, 5.0));
        let hole = b.arc(p(hole_x, 5.0), Vec3::new(0.0, 0.0, -1.0), 1.0, at, at);
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
        let face = b.topo.add_face(Face::new(
            outer_wire,
            vec![hole_wire],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        (face, outer[3])
    };
    let mut topo = Topology::new();
    let target = sheet(&mut topo, "plane", Pose::upright());
    let (face, edge) = plate(&mut topo, 7.0);
    let result =
        match_surface(&mut topo, face, edge, target.chain[0], MatchContinuity::G1).unwrap();
    assert_eq!(topo.face(result).unwrap().inner_wires().len(), 1);
    let surface = nurbs_of(&topo, result);
    let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, target.face), SAMPLES);
    assert_seam("with a hole", MatchContinuity::G1, &seam);

    let mut topo = Topology::new();
    let target = sheet(&mut topo, "plane", Pose::upright());
    let (face, edge) = plate(&mut topo, 1.6);
    let refused = match_surface(&mut topo, face, edge, target.chain[0], MatchContinuity::G1);
    assert!(matches!(refused, Err(OperationsError::InvalidInput { .. })));
}

fn refusal(result: Result<FaceId, OperationsError>) -> String {
    match result {
        Err(OperationsError::InvalidInput { reason }) => reason,
        Err(other) => panic!("wanted InvalidInput, got {other}"),
        Ok(_) => panic!("wanted a refusal, got a face"),
    }
}

#[test]
fn bad_edges_are_refused() {
    let mut topo = Topology::new();
    let target = sheet(&mut topo, "plane", Pose::upright());
    let source = sheet(&mut topo, "plane", across_pose("plane", Pose::upright()));
    let reason = refusal(match_surface(
        &mut topo,
        source.face,
        target.chain[0],
        target.chain[0],
        MatchContinuity::G1,
    ));
    assert!(reason.contains("itself"), "{reason}");
    let reason = refusal(match_surface(
        &mut topo,
        source.face,
        target.chain[0],
        source.chain[0],
        MatchContinuity::G1,
    ));
    assert!(reason.contains("own face"), "{reason}");
    let other = sheet(&mut topo, "plane", Pose::turned());
    let reason = refusal(match_surface(
        &mut topo,
        source.face,
        other.chain[0],
        target.chain[0],
        MatchContinuity::G1,
    ));
    assert!(reason.contains("not on the outer boundary"), "{reason}");
    let reason = refusal(match_surface_on(
        &mut topo,
        source.face,
        source.chain[0],
        target.chain[0],
        other.face,
        MatchContinuity::G1,
    ));
    assert!(reason.contains("not on the face named"), "{reason}");
}

#[test]
fn a_target_that_bounds_no_face_is_only_good_for_position() {
    let mut topo = Topology::new();
    let source = sheet(&mut topo, "plane", Pose::upright());
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let v = [
        b.vertex(Point3::new(0.5, 0.0, 0.2)),
        b.vertex(Point3::new(0.5, 10.0, -0.1)),
    ];
    let loose = b.line(v[0], v[1]);
    let reason = refusal(match_surface(
        &mut topo,
        source.face,
        source.chain[0],
        loose,
        MatchContinuity::G1,
    ));
    assert!(reason.contains("no other face"), "{reason}");
    let moved = match_surface(
        &mut topo,
        source.face,
        source.chain[0],
        loose,
        MatchContinuity::G0,
    )
    .unwrap();
    let surface = nurbs_of(&topo, moved);
    assert!(gap_to_chain(&topo, &surface, 0.0, &[loose], 100) < 1e-8);
}

/// A solid's edge bounds two faces, so the caller says which is meant.
#[test]
fn a_shared_target_edge_has_to_be_named() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let faces = solid_faces(&topo, block).unwrap();
    let top = faces
        .iter()
        .copied()
        .find(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { normal, d } => normal.z() > 0.5 && (*d - 10.0).abs() < 1e-9,
            _ => false,
        })
        .unwrap();
    let edge = topo
        .wire(topo.face(top).unwrap().outer_wire())
        .unwrap()
        .edges()[0]
        .edge();
    let data = topo.edge(edge).unwrap();
    let (start, end) = (
        topo.vertex(data.start()).unwrap().point(),
        topo.vertex(data.end()).unwrap().point(),
    );
    let sideways = (end - start)
        .cross(Vec3::new(0.0, 0.0, 1.0))
        .normalize()
        .unwrap();
    let outward = {
        let mid = start + (end - start) * 0.5;
        let out = mid - Point3::new(5.0, 5.0, 10.0);
        if sideways.dot(out) > 0.0 {
            sideways
        } else {
            -sideways
        }
    };
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let nudge = outward * 0.2 + Vec3::new(0.0, 0.0, 0.1);
    let far = outward * 8.0;
    let corners = [start + nudge, end + nudge, end + far, start + far];
    let v: Vec<_> = corners.iter().map(|&p| b.vertex(p)).collect();
    let edges: Vec<_> = (0..4).map(|i| b.line(v[i], v[(i + 1) % 4])).collect();
    let wire = b.topo.add_wire(
        Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let normal = (corners[1] - corners[0])
        .cross(corners[3] - corners[0])
        .normalize()
        .unwrap();
    let d = normal.dot(corners[0] - Point3::new(0.0, 0.0, 0.0));
    let source = b.topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane { normal, d },
    ));
    let reason = refusal(match_surface(
        &mut topo,
        source,
        edges[0],
        edge,
        MatchContinuity::G1,
    ));
    assert!(reason.contains("several faces"), "{reason}");
    let result =
        match_surface_on(&mut topo, source, edges[0], edge, top, MatchContinuity::G2).unwrap();
    let surface = nurbs_of(&topo, result);
    let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, top), SAMPLES);
    assert_seam("solid's top", MatchContinuity::G2, &seam);
}

/// One closed side and one open is no shape to match.
#[test]
fn a_closed_target_for_an_open_edge_is_refused() {
    let mut topo = Topology::new();
    let source = sheet(&mut topo, "plane", Pose::upright());
    let post = make_cylinder(&mut topo, 2.0, 3.0).unwrap();
    let cap = solid_faces(&topo, post)
        .unwrap()
        .into_iter()
        .find(|&f| topo.face(f).unwrap().surface().is_planar())
        .unwrap();
    let rim = face_edges(&topo, cap).unwrap()[0];
    let reason = refusal(match_surface_on(
        &mut topo,
        source.face,
        source.chain[0],
        rim,
        cap,
        MatchContinuity::G1,
    ));
    assert!(reason.contains("closed"), "{reason}");
}

/// Sides with more than one edge come back as that many, on the new surface.
#[test]
fn a_many_sided_outline_keeps_its_edges() {
    let mut topo = Topology::new();
    let target = sheet(&mut topo, "plane", Pose::upright());
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let p = |x: f64, y: f64| Point3::new(x, y, 0.5);
    let corners = [
        p(0.5, 0.0),
        p(10.0, 0.0),
        p(10.0, 5.0),
        p(10.0, 10.0),
        p(5.0, 10.0),
        p(0.5, 10.0),
    ];
    let v: Vec<_> = corners.iter().map(|&c| b.vertex(c)).collect();
    let edges: Vec<_> = (0..6).map(|i| b.line(v[i], v[(i + 1) % 6])).collect();
    let wire = b.topo.add_wire(
        Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let face = b.topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.5,
        },
    ));
    let result = match_surface(
        &mut topo,
        face,
        edges[5],
        target.chain[0],
        MatchContinuity::G1,
    )
    .unwrap();
    assert_wire_sound(&topo, result);
    let new_wire = topo.wire(topo.face(result).unwrap().outer_wire()).unwrap();
    assert_eq!(new_wire.edges().len(), 6);
    let surface = nurbs_of(&topo, result);
    let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, target.face), SAMPLES);
    assert_seam("six sides", MatchContinuity::G1, &seam);
}

/// An L is too far from four convex sides to map, and says so.
#[test]
fn an_outline_that_folds_is_refused() {
    let mut topo = Topology::new();
    let target = sheet(&mut topo, "plane", Pose::upright());
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let p = |x: f64, y: f64| Point3::new(x, y, 0.5);
    let corners = [
        p(0.5, 0.0),
        p(2.0, 0.0),
        p(2.0, 8.0),
        p(10.0, 8.0),
        p(10.0, 10.0),
        p(0.5, 10.0),
    ];
    let v: Vec<_> = corners.iter().map(|&c| b.vertex(c)).collect();
    let edges: Vec<_> = (0..6).map(|i| b.line(v[i], v[(i + 1) % 6])).collect();
    let wire = b.topo.add_wire(
        Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let face = b.topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.5,
        },
    ));
    let reason = refusal(match_surface(
        &mut topo,
        face,
        edges[5],
        target.chain[0],
        MatchContinuity::G1,
    ));
    assert!(reason.contains("fold"), "{reason}");
}

/// A rational curve as the edge, and its surface refitted, match as well.
#[test]
fn a_target_spline_edge_on_a_spline_surface_keeps_its_figures() {
    let mut topo = Topology::new();
    let target = sheet(&mut topo, "spline", Pose::turned());
    let curve_of_target = |topo: &Topology, edge: EdgeId| -> NurbsCurve {
        match topo.edge(edge).unwrap().curve() {
            EdgeCurve::NurbsCurve(curve) => curve.clone(),
            _ => panic!("a spline edge"),
        }
    };
    let curve = curve_of_target(&topo, target.chain[0]);
    assert!(curve.knots().len() > 8);
    let exit = Pose::turned().vector(Vec3::new(1.0, 0.0, 0.0));
    let lift = Pose::turned().vector(Vec3::new(0.0, 0.1, 0.2));
    let (face, edge, _) = source_near(&mut topo, target.chain[0], exit, lift, 5.0);
    let result =
        match_surface(&mut topo, face, edge, target.chain[0], MatchContinuity::G2).unwrap();
    let surface = nurbs_of(&topo, result);
    let seam = measure_seam(&surface, 0.0, &surface_of_face(&topo, target.face), SAMPLES);
    assert_seam("spline to spline", MatchContinuity::G2, &seam);
    assert!(gap_to_chain(&topo, &surface, 0.0, &target.chain, 200) < 1e-8);
}
