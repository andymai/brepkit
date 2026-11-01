//! Sweeps whose sections are the profile carried along the path by a
//! similarity keep the profile's own curves: a circle scaled along a line
//! sweeps a frustum, a square turned by a guide about a straight spine keeps
//! its area at every height, and a circle carried round a quarter circle
//! sweeps a quarter torus. Each result is valid, meshes closed and holds its
//! closed-form volume, upright and turned off the axes.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::helix::make_helix_curve;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::sweep::{SweepContactMode, SweepOptions, sweep_guided, sweep_with_options};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_circle_edge;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::{solid_faces, solid_vertices};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const TOL: f64 = 1e-7;

fn poses() -> [(&'static str, Mat4); 2] {
    [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(3.0, -2.0, 1.0)
                * Mat4::rotation_z(0.7)
                * Mat4::rotation_x(0.4)
                * Mat4::rotation_y(0.3),
        ),
    ]
}

fn dir(pose: Mat4, v: Vec3) -> Vec3 {
    pose.mul_point(Point3::new(v.x(), v.y(), v.z())) - pose.mul_point(Point3::new(0.0, 0.0, 0.0))
}

fn plane(pose: Mat4, normal: Vec3, on: Point3) -> FaceSurface {
    let normal = dir(pose, normal);
    let on = pose.mul_point(on);
    FaceSurface::Plane {
        normal,
        d: normal.dot(Vec3::new(on.x(), on.y(), on.z())),
    }
}

/// A disc of radius `r` about the origin in the plane normal to `normal`.
fn disc(topo: &mut Topology, pose: Mat4, normal: Vec3, r: f64) -> FaceId {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let edge = make_circle_edge(topo, pose.mul_point(origin), dir(pose, normal), r, TOL).unwrap();
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
    topo.add_face(Face::new(wire, vec![], plane(pose, normal, origin)))
}

/// A square of side `2 * h` about the origin in the plane z = 0, with an
/// optional round hole of radius `r` about `c`.
fn square(topo: &mut Topology, pose: Mat4, h: f64, hole: Option<(Point3, f64)>) -> FaceId {
    let corners = [(-h, -h), (h, -h), (h, h), (-h, h)];
    let verts: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(pose.mul_point(Point3::new(x, y, 0.0)), TOL)))
        .collect();
    let edges: Vec<_> = (0..4)
        .map(|i| {
            let e = topo.add_edge(Edge::new(verts[i], verts[(i + 1) % 4], EdgeCurve::Line));
            OrientedEdge::new(e, true)
        })
        .collect();
    let outer = topo.add_wire(Wire::new(edges, true).unwrap());
    let inner = hole.map(|(c, r)| {
        // Normal -z: the hole runs clockwise about the face's normal.
        let e = make_circle_edge(
            topo,
            pose.mul_point(c),
            dir(pose, Vec3::new(0.0, 0.0, -1.0)),
            r,
            TOL,
        )
        .unwrap();
        topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap())
    });
    let origin = Point3::new(0.0, 0.0, 0.0);
    topo.add_face(Face::new(
        outer,
        inner.into_iter().collect(),
        plane(pose, Vec3::new(0.0, 0.0, 1.0), origin),
    ))
}

/// The line from the origin up the z axis to height `len`.
fn spine(pose: Mat4, len: f64) -> NurbsCurve {
    NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            pose.mul_point(Point3::new(0.0, 0.0, 0.0)),
            pose.mul_point(Point3::new(0.0, 0.0, len)),
        ],
        vec![1.0; 2],
    )
    .unwrap()
}

fn assert_exact(topo: &Topology, solid: SolidId, want: f64, case: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{case}: the mesh is open");
    let volume = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        ((volume - want) / want).abs() < 1e-6,
        "{case}: volume {volume}, want {want}"
    );
    let integrated = brepkit_check::properties::exact_solid_volume(topo, solid)
        .unwrap()
        .expect("an exact volume");
    assert!(
        ((integrated - want) / want).abs() < 1e-6,
        "{case}: integrated volume {integrated}, want {want}"
    );
}

/// A circle of radius 10 scaled to half its size along 40 sweeps a frustum
/// of radii 10 and 5 in four sides and two caps, however many segments it
/// is given.
#[test]
fn a_circle_scaled_along_a_line_sweeps_a_frustum() {
    let want = PI * 40.0 / 3.0 * (100.0 + 50.0 + 25.0);
    for (name, pose) in poses() {
        for segments in [0, 64, 10_000] {
            let case = format!("{name}, {segments} segments");
            let mut topo = Topology::new();
            let profile = disc(&mut topo, pose, Vec3::new(0.0, 0.0, 1.0), 10.0);
            let options = SweepOptions {
                scale_law: Some(Box::new(|t: f64| 0.5f64.mul_add(-t, 1.0))),
                segments,
                ..Default::default()
            };
            let solid = sweep_with_options(&mut topo, profile, &spine(pose, 40.0), &options)
                .map_err(|e| format!("{case}: {e}"))
                .unwrap();
            assert_exact(&topo, solid, want, &case);
            assert_eq!(
                solid_faces(&topo, solid).unwrap().len(),
                6,
                "{case}: four arcs' sides and two caps"
            );
        }
    }
}

/// A square of side 10 with a hole of radius 2, scaled to half its size
/// along 40: every section is the holed square scaled about the spine, the
/// square's sides stay planes and the hole's wall runs on the hole's arcs.
#[test]
fn a_holed_square_scaled_along_a_line_keeps_its_hole() {
    let area = 4.0f64.mul_add(-PI, 100.0);
    let want = area * 40.0 * (1.0 + 0.5 + 0.25) / 3.0;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let profile = square(
            &mut topo,
            pose,
            5.0,
            Some((Point3::new(1.0, 0.5, 0.0), 2.0)),
        );
        let options = SweepOptions {
            scale_law: Some(Box::new(|t: f64| 0.5f64.mul_add(-t, 1.0))),
            ..Default::default()
        };
        let solid = sweep_with_options(&mut topo, profile, &spine(pose, 40.0), &options)
            .map_err(|e| format!("{name}: {e}"))
            .unwrap();
        assert_exact(&topo, solid, want, name);
        let faces = solid_faces(&topo, solid).unwrap();
        let planes = faces
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }))
            .count();
        assert_eq!(faces.len(), 10, "{name}: 4 sides, 4 hole walls, 2 caps");
        assert_eq!(planes, 6, "{name}: the sides and caps are planes");
    }
}

/// A square of side 10 swept up a straight spine of 40 by a guide that turns
/// 45, 90 and 360 degrees about it: each section keeps the square's 100, and
/// the end's corners are the start's turned through the guide's own turn.
#[test]
fn a_square_turned_by_a_guide_keeps_its_area() {
    for (name, pose) in poses() {
        for turn in [45.0_f64, 90.0, 360.0] {
            let case = format!("{name}, {turn} degrees");
            let mut topo = Topology::new();
            let profile = square(&mut topo, pose, 5.0, None);
            let axis = dir(pose, Vec3::new(0.0, 0.0, 1.0));
            let base = pose.mul_point(Point3::new(0.0, 0.0, 0.0));
            let guide =
                make_helix_curve(base, axis, 1.0, 40.0 * 360.0 / turn, turn / 360.0, 8).unwrap();
            let path = spine(pose, 40.0);
            let solid = sweep_guided(&mut topo, profile, &path, guide.clone())
                .map_err(|e| format!("{case}: {e}"))
                .unwrap();
            assert_exact(&topo, solid, 4000.0, &case);
            assert_eq!(solid_faces(&topo, solid).unwrap().len(), 6, "{case}");

            let (g0, g1) = guide.domain();
            let start = guide.evaluate(g0) - path.evaluate(0.0);
            let end = guide.evaluate(g1) - path.evaluate(1.0);
            let angle = start.cross(end).dot(axis).atan2(start.dot(end));
            let top = pose.mul_point(Point3::new(0.0, 0.0, 40.0));
            let ends: Vec<Point3> = solid_vertices(&topo, solid)
                .unwrap()
                .iter()
                .map(|&v| topo.vertex(v).unwrap().point())
                .filter(|p| (*p - top).dot(axis).abs() < 1e-9)
                .collect();
            assert_eq!(ends.len(), 4, "{case}: the end's corners");
            for (x, y) in [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)] {
                let p = dir(pose, Vec3::new(x, y, 0.0));
                let (s, c) = angle.sin_cos();
                let turned = p * c + axis.cross(p) * s + axis * axis.dot(p) * (1.0 - c);
                let want = top + turned;
                assert!(
                    ends.iter().any(|q| (*q - want).length() < 1e-9),
                    "{case}: no end corner at {want:?}, have {ends:?}"
                );
            }
        }
    }
}

/// A circle of radius 1 carried round a quarter circle of radius 10 on
/// rotation-minimizing frames sweeps a quarter torus, its volume the
/// circle's area times the path's length.
#[test]
fn a_circle_round_a_quarter_circle_sweeps_a_quarter_torus() {
    let want = PI * 1.0 * (PI / 2.0 * 10.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let profile = disc(&mut topo, pose, Vec3::new(1.0, 0.0, 0.0), 1.0);
        let path = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                pose.mul_point(Point3::new(0.0, 0.0, 0.0)),
                pose.mul_point(Point3::new(10.0, 0.0, 0.0)),
                pose.mul_point(Point3::new(10.0, 0.0, 10.0)),
            ],
            vec![1.0, FRAC_1_SQRT_2, 1.0],
        )
        .unwrap();
        let solid = sweep_with_options(&mut topo, profile, &path, &SweepOptions::default())
            .map_err(|e| format!("{name}: {e}"))
            .unwrap();
        assert_exact(&topo, solid, want, name);
    }
}

/// A circle of radius 1 normal to x, carried without turning along a
/// quadratic from the origin to (10, 0, 5) whose tangent turns from x to 45
/// degrees off it: each section is the circle moved, and the body holds its
/// area times the path's run along x.
#[test]
fn a_circle_carried_without_turning_holds_its_area_times_its_run() {
    let want = PI * 10.0;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let profile = disc(&mut topo, pose, Vec3::new(1.0, 0.0, 0.0), 1.0);
        let path = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                pose.mul_point(Point3::new(0.0, 0.0, 0.0)),
                pose.mul_point(Point3::new(5.0, 0.0, 0.0)),
                pose.mul_point(Point3::new(10.0, 0.0, 5.0)),
            ],
            vec![1.0; 3],
        )
        .unwrap();
        let options = SweepOptions {
            contact_mode: SweepContactMode::Fixed,
            ..Default::default()
        };
        let solid = sweep_with_options(&mut topo, profile, &path, &options)
            .map_err(|e| format!("{name}: {e}"))
            .unwrap();
        assert_exact(&topo, solid, want, name);
    }
}

/// A square whose fourth side is a degree-1 NURBS through a point raised 1
/// out of the square's plane over the knot span from 0.02 to 0.04, which no
/// sixteenth of the side's parameter falls in: the profile is not planar,
/// so no plane caps it, and every plane face of the sweep holds its edges.
#[test]
fn a_profile_leaving_its_plane_between_samples_takes_no_plane_cap() {
    let mut topo = Topology::new();
    let p = Point3::new;
    let corners = [
        p(-5.0, -5.0, 0.0),
        p(5.0, -5.0, 0.0),
        p(5.0, 5.0, 0.0),
        p(-5.0, 5.0, 0.0),
    ];
    let verts: Vec<_> = corners
        .iter()
        .map(|&c| topo.add_vertex(Vertex::new(c, TOL)))
        .collect();
    let bump = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 0.02, 0.04, 1.0, 1.0],
        vec![corners[3], p(-5.0, 4.6, 1.0), p(-5.0, 4.2, 0.0), corners[0]],
        vec![1.0; 4],
    )
    .unwrap();
    let mut edges = Vec::new();
    for i in 0..3 {
        let e = topo.add_edge(Edge::new(verts[i], verts[i + 1], EdgeCurve::Line));
        edges.push(OrientedEdge::new(e, true));
    }
    let e = topo.add_edge(Edge::new(verts[3], verts[0], EdgeCurve::NurbsCurve(bump)));
    edges.push(OrientedEdge::new(e, true));
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let profile = topo.add_face(Face::new(
        wire,
        vec![],
        plane(Mat4::identity(), Vec3::new(0.0, 0.0, 1.0), p(0.0, 0.0, 0.0)),
    ));
    let options = SweepOptions {
        scale_law: Some(Box::new(|t: f64| 0.5f64.mul_add(-t, 1.0))),
        ..Default::default()
    };
    let Ok(solid) =
        sweep_with_options(&mut topo, profile, &spine(Mat4::identity(), 40.0), &options)
    else {
        return;
    };
    for f in solid_faces(&topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        let FaceSurface::Plane { normal, d } = *face.surface() else {
            continue;
        };
        for oe in topo.wire(face.outer_wire()).unwrap().edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let a = topo.vertex(edge.start()).unwrap().point();
            let b = topo.vertex(edge.end()).unwrap().point();
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            for k in 0..=200 {
                let t = (t1 - t0).mul_add(f64::from(k) / 200.0, t0);
                let q = edge.curve().evaluate_with_endpoints(t, a, b);
                let off = normal.dot(Vec3::new(q.x(), q.y(), q.z())) - d;
                assert!(
                    off.abs() < 1e-6,
                    "an edge of face {f:?} stands {off} off its plane"
                );
            }
        }
    }
}

/// A path whose knots 0.5 and 0.5 + 1e-12 bound a span narrower than the
/// fit can sample on its own: the sweep leaves it to the ring-by-ring
/// sweep, which reads such a span inside, and comes out valid.
#[test]
fn a_path_with_a_knot_span_too_narrow_to_fit_sweeps_ring_by_ring() {
    let mut topo = Topology::new();
    let profile = disc(&mut topo, Mat4::identity(), Vec3::new(0.0, 0.0, 1.0), 1.0);
    let path = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 0.5, 0.5 + 1e-12, 1.0, 1.0, 1.0],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(5.0, 0.0, 15.0),
            Point3::new(10.0, 0.0, 15.0),
            Point3::new(20.0, 0.0, 15.0),
        ],
        vec![1.0; 5],
    )
    .unwrap();
    let options = SweepOptions {
        scale_law: Some(Box::new(|t: f64| 0.5f64.mul_add(-t, 1.0))),
        ..Default::default()
    };
    let solid = sweep_with_options(&mut topo, profile, &path, &options).unwrap();
    assert!(solid_faces(&topo, solid).unwrap().len() > 6, "ring by ring");
    let report = validate_solid(&topo, solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
}

/// A circle of radius 2 swept three turns round a helix of radius 20 and
/// pitch 8, perpendicular to it by `helical_sweep` and as it stands,
/// upright across the helix's start, by `sweep`. On rotation-minimizing frames each
/// section keeps the angle its normal makes with the path, so the coil holds
/// the circle's area times the path's length times that angle's cosine.
#[test]
fn a_circle_swept_round_a_helix_holds_its_area_times_its_length() {
    let mut topo = Topology::new();
    let helix = make_helix_curve(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        20.0,
        8.0,
        3.0,
        8,
    )
    .unwrap();
    let (h0, _) = helix.domain();
    let start = helix.evaluate(h0);
    let tangent = helix.tangent(h0).unwrap();
    let length = helix.arc_length(200_000);
    let area = PI * 4.0;

    let profile = disc(&mut topo, Mat4::identity(), Vec3::new(0.0, 0.0, 1.0), 2.0);
    let coil = brepkit_operations::helix::helical_sweep(
        &mut topo,
        profile,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        20.0,
        8.0,
        3.0,
        8,
    )
    .unwrap();
    assert_exact(&topo, coil, area * length, "perpendicular");
    assert_eq!(solid_faces(&topo, coil).unwrap().len(), 6, "perpendicular");

    // Standing upright across the helix's start, its normal the start
    // tangent's horizontal part.
    let normal = Vec3::new(tangent.x(), tangent.y(), 0.0)
        .normalize()
        .unwrap();
    let e = make_circle_edge(&mut topo, start, normal, 2.0, TOL).unwrap();
    let w = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
    let d = normal.dot(Vec3::new(start.x(), start.y(), start.z()));
    let profile = topo.add_face(Face::new(w, vec![], FaceSurface::Plane { normal, d }));
    let coil = brepkit_operations::sweep::sweep(&mut topo, profile, &helix).unwrap();
    let want = area * length * normal.dot(tangent).abs();
    assert_exact(&topo, coil, want, "as it stands");
}

/// A square of side 2 swept 10 up a straight spine in 4 segments, scaled
/// by 1 + 4e-6 w((t - c) / 0.005) with w a smooth bump of width 2: the
/// scale moves its corners 5.7e-6 out within 0.005 of t = c, past the fit's
/// bound of some 1e-6. The first fit reads that at a midpoint for c =
/// 0.125, and at a sample for c = 0.25; however its samples grow after,
/// the corner's rail passes within 1.5e-6 of the scaled corner there.
#[test]
fn a_narrow_scale_bump_a_fit_has_read_is_kept() {
    for c in [0.125_f64, 0.25] {
        let off = rail_off_a_scale_bump(c);
        assert!(
            off <= 1.5e-6,
            "bump at {c}: the rail passes {off} from the scaled corner"
        );
    }
}

/// How far the corner's rail of the sweep in
/// [`a_narrow_scale_bump_a_fit_has_read_is_kept`] passes from the corner
/// scaled at the bump's middle `c`.
fn rail_off_a_scale_bump(c: f64) -> f64 {
    let bump = |x: f64| {
        if x.abs() < 1.0 {
            (0.5 * PI * x).cos().powi(2)
        } else {
            0.0
        }
    };
    let mut topo = Topology::new();
    let profile = square(&mut topo, Mat4::identity(), 1.0, None);
    let options = SweepOptions {
        scale_law: Some(Box::new(move |t: f64| {
            4e-6f64.mul_add(bump((t - c) / 0.005), 1.0)
        })),
        segments: 4,
        ..Default::default()
    };
    let solid =
        sweep_with_options(&mut topo, profile, &spine(Mat4::identity(), 10.0), &options).unwrap();
    let corner = |p: Point3| (p - Point3::new(1.0, 1.0, p.z())).length() < 1e-3;
    let rail = brepkit_topology::explorer::solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            corner(a) && corner(b) && (a.z() - b.z()).abs() > 9.0
        })
        .expect("the corner's rail");
    let edge = topo.edge(rail).unwrap();
    let scaled = Point3::new(1.0 + 4e-6, 1.0 + 4e-6, 10.0 * c);
    if let EdgeCurve::NurbsCurve(curve) = edge.curve() {
        brepkit_math::nurbs::projection::project_point_to_curve(curve, scaled, 1e-12)
            .unwrap()
            .distance
    } else {
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let ab = b - a;
        let f = ((scaled - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
        (scaled - (a + ab * f)).length()
    }
}

/// A square of side 2 about the origin swept one turn round a helix of
/// radius 20 by `helical_sweep`, which puts the profile's centre on the
/// path: with its top side a line or the same side held as a degree-1
/// NURBS, the centre is the square's, and the two coils match.
#[test]
fn a_profiles_centre_on_a_helix_does_not_hang_on_how_its_edges_are_held() {
    let coil = |nurbs_top: bool| {
        let mut topo = Topology::new();
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let v: Vec<_> = corners
            .iter()
            .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), TOL)))
            .collect();
        let edges = (0..4)
            .map(|k| {
                let (a, b) = (v[k], v[(k + 1) % 4]);
                let curve = if nurbs_top && k == 2 {
                    let (p, q) = (corners[2], corners[3]);
                    EdgeCurve::NurbsCurve(
                        NurbsCurve::new(
                            1,
                            vec![0.0, 0.0, 1.0, 1.0],
                            vec![Point3::new(p.0, p.1, 0.0), Point3::new(q.0, q.1, 0.0)],
                            vec![1.0; 2],
                        )
                        .unwrap(),
                    )
                } else {
                    EdgeCurve::Line
                };
                OrientedEdge::new(topo.add_edge(Edge::new(a, b, curve)), true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let profile = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let solid = brepkit_operations::helix::helical_sweep(
            &mut topo,
            profile,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            20.0,
            8.0,
            1.0,
            8,
        )
        .unwrap();
        let volume = brepkit_check::properties::exact_solid_volume(&topo, solid)
            .unwrap()
            .unwrap();
        let bbox = brepkit_operations::measure::solid_bounding_box(&topo, solid).unwrap();
        (volume, bbox)
    };
    let ((v_line, b_line), (v_nurbs, b_nurbs)) = (coil(false), coil(true));
    assert!(
        ((v_line - v_nurbs) / v_line).abs() < 1e-9,
        "{v_line} and {v_nurbs}"
    );
    let off = (b_line.min - b_nurbs.min)
        .length()
        .max((b_line.max - b_nurbs.max).length());
    assert!(off < 1e-6, "the coils' boxes differ by {off}");
}
