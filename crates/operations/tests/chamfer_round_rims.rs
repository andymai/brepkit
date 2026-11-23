//! Chamfers of closed circular rims, where a flat cap meets a coaxial
//! cylinder or cone wall: a rod's ends, a tube's mouth, a frustum's top and
//! a hole's countersink. Each removes a ring whose cross-section is a
//! triangle, so its volume is the triangle's area times the path of its
//! centroid.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::{chamfer_distance_angle, chamfer_v2, chamfer_v2_on};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::{chamfer, chamfer_two_on};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{face_edges, solid_edges, solid_faces};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

/// The closed circle edges of `solid` at height `z`.
fn rims_at(topo: &Topology, solid: SolidId, z: f64) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(_))
                && edge.start() == edge.end()
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
        })
        .collect()
}

/// The ring a chamfer cuts from a rim of radius `rim`: the triangle between
/// the rim point `(rim, 0)`, the cap point `(rim + cap, 0)` and the wall
/// point `(rim + wall.0, wall.1)`, revolved.
fn ring(rim: f64, cap: f64, wall: (f64, f64)) -> f64 {
    let area = 0.5 * (cap * wall.1).abs();
    let centroid = rim + (cap + wall.0) / 3.0;
    2.0 * PI * centroid * area
}

fn check(
    topo: &Topology,
    result: SolidId,
    expected: f64,
    census: &[(&str, usize)],
    inside: &[Point3],
    outside: &[Point3],
) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mut faces: BTreeMap<&str, usize> = BTreeMap::new();
    for face in solid_faces(topo, result).unwrap() {
        *faces
            .entry(topo.face(face).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    assert_eq!(faces, census.iter().copied().collect(), "face census");

    let classify =
        |p: Point3| classify_point(topo, result, p, &ClassifyOptions::default()).unwrap();
    for &p in inside {
        assert_eq!(classify(p), PointClassification::Inside, "{p:?} is kept");
    }
    for &p in outside {
        assert_eq!(classify(p), PointClassification::Outside, "{p:?} is cut");
    }

    let exact = solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (exact - expected).abs() < 1e-7 * expected,
        "volume {exact}, truth {expected}"
    );
    let mesh = tessellate_solid(topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let meshed = oriented_solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (meshed - expected).abs() < 2e-3 * expected,
        "mesh volume {meshed}, truth {expected}"
    );
}

#[test]
fn rod_end_rim() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let rims = rims_at(&topo, rod, 10.0);
    let result = chamfer(&mut topo, rod, &rims, 1.0).unwrap();
    check(
        &topo,
        result,
        90.0 * PI - ring(3.0, -1.0, (0.0, -1.0)),
        &[("cone", 1), ("cylinder", 1), ("plane", 2)],
        &[Point3::new(0.0, 0.0, 9.9), Point3::new(2.4, 0.0, 9.5)],
        &[Point3::new(2.9, 0.0, 9.9), Point3::new(0.0, -2.8, 9.8)],
    );
}

#[test]
fn rod_both_rims() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let mut rims = rims_at(&topo, rod, 10.0);
    rims.extend(rims_at(&topo, rod, 0.0));
    let result = chamfer(&mut topo, rod, &rims, 1.0).unwrap();
    check(
        &topo,
        result,
        90.0 * PI - 2.0 * ring(3.0, -1.0, (0.0, -1.0)),
        &[("cone", 2), ("cylinder", 1), ("plane", 2)],
        &[Point3::new(2.4, 0.0, 0.5), Point3::new(2.4, 0.0, 9.5)],
        &[Point3::new(2.9, 0.0, 0.1), Point3::new(2.9, 0.0, 9.9)],
    );
}

#[test]
fn tube_mouth_rims() {
    let mut topo = Topology::new();
    let outer = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let inner = make_cylinder(&mut topo, 2.0, 12.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
    let rims = rims_at(&topo, tube, 10.0);
    assert_eq!(rims.len(), 2);
    let d = 1.0 / 3.0;
    let result = chamfer(&mut topo, tube, &rims, d).unwrap();
    check(
        &topo,
        result,
        50.0 * PI - ring(3.0, -d, (0.0, -d)) - ring(2.0, d, (0.0, -d)),
        &[("cone", 2), ("cylinder", 2), ("plane", 2)],
        &[Point3::new(2.5, 0.0, 9.9), Point3::new(0.0, 2.5, 5.0)],
        &[
            Point3::new(2.95, 0.0, 9.95),
            Point3::new(0.0, 2.05, 9.95),
            Point3::new(0.0, 0.0, 9.0),
        ],
    );
}

/// A tube counterbored at its mouth: the cap is the ring between the outer
/// wall and the counterbore, and the counterbore's wall ends at its step.
#[test]
fn counterbored_tube_mouth_rims() {
    let mut topo = Topology::new();
    let outer = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let bore = make_cylinder(&mut topo, 1.5, 12.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, outer, bore).unwrap();
    let counterbore = make_cylinder(&mut topo, 2.0, 4.0).unwrap();
    transform_solid(&mut topo, counterbore, &Mat4::translation(0.0, 0.0, 7.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, tube, counterbore).unwrap();
    let rims = rims_at(&topo, tube, 10.0);
    assert_eq!(rims.len(), 2);
    let d = 1.0 / 3.0;
    let result = chamfer(&mut topo, tube, &rims, d).unwrap();
    check(
        &topo,
        result,
        PI * (6.75 * 10.0 - 1.75 * 3.0) - ring(3.0, -d, (0.0, -d)) - ring(2.0, d, (0.0, -d)),
        &[("cone", 2), ("cylinder", 3), ("plane", 3)],
        &[Point3::new(2.5, 0.0, 9.9), Point3::new(0.0, 1.7, 5.0)],
        &[
            Point3::new(2.95, 0.0, 9.95),
            Point3::new(0.0, 2.05, 9.95),
            Point3::new(0.0, 1.8, 8.0),
        ],
    );
}

/// The wall runs from the top rim (radius 2 at z = 6) down to radius 3 at
/// z = 0, so the chamfer's wall point moves `d` along that slant.
#[test]
fn frustum_top_rim() {
    let mut topo = Topology::new();
    let frustum = make_cone(&mut topo, 3.0, 2.0, 6.0).unwrap();
    let rims = rims_at(&topo, frustum, 6.0);
    let d = 0.5;
    let slant = 37.0_f64.sqrt();
    let result = chamfer(&mut topo, frustum, &rims, d).unwrap();
    check(
        &topo,
        result,
        38.0 * PI - ring(2.0, -d, (d / slant, -6.0 * d / slant)),
        &[("cone", 2), ("plane", 2)],
        &[Point3::new(1.4, 0.0, 5.9), Point3::new(2.3, 0.0, 3.0)],
        &[Point3::new(1.95, 0.0, 5.97)],
    );
}

/// A through hole in a block: the chamfer's cap circle grows into the top
/// face and its wall circle moves down the bore.
#[test]
fn hole_countersink() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let bore = make_cylinder(&mut topo, 2.0, 12.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(5.0, 5.0, -1.0)).unwrap();
    let holed = boolean(&mut topo, BooleanOp::Cut, block, bore).unwrap();
    let rims = rims_at(&topo, holed, 10.0);
    assert_eq!(rims.len(), 1);
    let d = 0.5;
    let result = chamfer(&mut topo, holed, &rims, d).unwrap();
    check(
        &topo,
        result,
        1000.0 - 40.0 * PI - ring(2.0, d, (0.0, -d)),
        &[("cone", 1), ("cylinder", 1), ("plane", 6)],
        &[Point3::new(7.6, 5.0, 9.9), Point3::new(5.0, 7.3, 5.0)],
        &[Point3::new(7.05, 5.0, 9.95), Point3::new(5.0, 5.0, 9.9)],
    );
}

/// The floor rim of a blind hole is concave: its chamfer fills the corner,
/// adding the ring instead of removing it.
#[test]
fn blind_hole_floor_rim() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let bore = make_cylinder(&mut topo, 2.0, 6.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(5.0, 5.0, 5.0)).unwrap();
    let holed = boolean(&mut topo, BooleanOp::Cut, block, bore).unwrap();
    let rims = rims_at(&topo, holed, 5.0);
    assert_eq!(rims.len(), 1);
    let d = 0.5;
    let result = chamfer(&mut topo, holed, &rims, d).unwrap();
    check(
        &topo,
        result,
        1000.0 - 20.0 * PI + ring(2.0, -d, (0.0, d)),
        &[("cone", 1), ("cylinder", 1), ("plane", 7)],
        &[Point3::new(6.9, 5.0, 5.05), Point3::new(5.0, 5.0, 4.9)],
        &[Point3::new(6.6, 5.0, 5.3), Point3::new(5.0, 5.0, 5.1)],
    );
}

/// A closed cylindrical cavity inside a block: its rims bound an inner
/// shell, and the chamfer's band joins that shell, filling the corner.
#[test]
fn enclosed_cavity_rim() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let cavity = make_cylinder(&mut topo, 2.0, 4.0).unwrap();
    transform_solid(&mut topo, cavity, &Mat4::translation(5.0, 5.0, 3.0)).unwrap();
    let hollow = boolean(&mut topo, BooleanOp::Cut, block, cavity).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let rims = rims_at(&topo, hollow, 7.0);
    assert_eq!(rims.len(), 1);
    let d = 0.5;
    let result = chamfer(&mut topo, hollow, &rims, d).unwrap();
    assert_eq!(topo.solid(result).unwrap().inner_shells().len(), 1);
    check(
        &topo,
        result,
        1000.0 - 16.0 * PI + ring(2.0, -d, (0.0, -d)),
        &[("cone", 1), ("cylinder", 1), ("plane", 8)],
        &[Point3::new(6.9, 5.0, 6.95), Point3::new(8.0, 5.0, 5.0)],
        &[Point3::new(6.6, 5.0, 6.6), Point3::new(5.0, 5.0, 5.0)],
    );
}

/// A distance that consumes the cap or the wall is refused, not built.
#[test]
fn oversized_distances_are_refused() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let rims = rims_at(&topo, rod, 10.0);
    assert!(chamfer(&mut topo, rod, &rims, 3.5).is_err());

    let short = make_cylinder(&mut topo, 3.0, 1.5).unwrap();
    let mut rims = rims_at(&topo, short, 1.5);
    rims.extend(rims_at(&topo, short, 0.0));
    assert!(chamfer(&mut topo, short, &rims, 1.0).is_err());

    let outer = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let inner = make_cylinder(&mut topo, 2.0, 12.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
    let rims = rims_at(&topo, tube, 10.0);
    assert!(chamfer(&mut topo, tube, &rims, 0.5).is_err());
}

/// A rod of radius 3, 10 high, less a cavity: a cylinder of radius 1 from
/// z = 3 to 7 about the rod's axis.
fn rod_with_a_cavity(topo: &mut Topology) -> SolidId {
    let rod = make_cylinder(topo, 3.0, 10.0).unwrap();
    let cavity = make_cylinder(topo, 1.0, 4.0).unwrap();
    transform_solid(topo, cavity, &Mat4::translation(0.0, 0.0, 3.0)).unwrap();
    let hollow = boolean(topo, BooleanOp::Cut, rod, cavity).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    hollow
}

/// Registers a line pcurve, its own for each, on every edge of every face of
/// `solid`, standing in for the UV trims a curved face is meshed by.
fn trim_every_face(topo: &mut Topology, solid: SolidId) {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2};
    use brepkit_topology::pcurve::PCurve;

    let mut k = 0.0;
    for face in solid_faces(topo, solid).unwrap() {
        for edge in brepkit_topology::explorer::face_edges(topo, face).unwrap() {
            k += 1.0;
            let line = Line2D::new(Point2::new(k, 0.5 * k), Vec2::new(1.0, k)).unwrap();
            topo.pcurves_mut()
                .set(edge, face, PCurve::new(Curve2D::Line(line), 0.0, 1.0));
        }
    }
}

/// Each face's pcurves, read at the ends and middle of each one, in the
/// order of the face's wires.
fn trims(topo: &Topology, faces: &[brepkit_topology::face::FaceId]) -> Vec<Vec<[f64; 2]>> {
    faces
        .iter()
        .map(|&face| {
            let mut points = Vec::new();
            for edge in brepkit_topology::explorer::face_edges(topo, face).unwrap() {
                if let Some(pcurve) = topo.pcurves().get(edge, face) {
                    let (t0, t1) = (pcurve.t_start(), pcurve.t_end());
                    for t in [t0, 0.5 * (t0 + t1), t1] {
                        let p = pcurve.evaluate(t);
                        points.push([p.x(), p.y()]);
                    }
                }
            }
            points
        })
        .collect()
}

/// The faces of `solid`'s cavity, and its bottom cap.
fn untouched(topo: &Topology, solid: SolidId) -> Vec<brepkit_topology::face::FaceId> {
    let shell = topo.solid(solid).unwrap().inner_shells()[0];
    let mut faces = topo.shell(shell).unwrap().faces().to_vec();
    let bottom = rims_at(topo, solid, 0.0)[0];
    faces.extend(solid_faces(topo, solid).unwrap().into_iter().filter(|&f| {
        matches!(
            topo.face(f).unwrap().surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        ) && brepkit_topology::explorer::face_edges(topo, f)
            .unwrap()
            .contains(&bottom)
    }));
    faces
}

/// A rim chamfer keeps the trims of the faces it does not touch, the
/// cavity's and the bottom cap's, and drops the trims of the edge uses it
/// replaces: every pcurve left on a face lies on an edge the face uses.
/// Through `chamfer`, and through `chamfer_v2` with two distances. The
/// trims are markers that do not follow their edges, so each chamfer is
/// validated as it is made without them.
#[test]
fn a_rim_chamfer_keeps_the_trims_it_does_not_replace() {
    let mut topo = Topology::new();
    let hollow = rod_with_a_cavity(&mut topo);
    trim_every_face(&mut topo, hollow);
    let before = trims(&topo, &untouched(&topo, hollow));
    assert!(before.iter().all(|points| !points.is_empty()), "{before:?}");
    let rim = rims_at(&topo, hollow, 10.0)[0];

    let equal = chamfer(&mut topo, hollow, &[rim], 0.5).unwrap();
    let two = chamfer_v2(&mut topo, hollow, &[rim], 0.5, 0.3).unwrap();
    assert!(two.failed.is_empty(), "{:?}", two.failed);

    for result in [equal, two.solid] {
        assert_eq!(trims(&topo, &untouched(&topo, result)), before);
        for face in solid_faces(&topo, result).unwrap() {
            let used = brepkit_topology::explorer::face_edges(&topo, face).unwrap();
            for (edge, _) in topo.pcurves().pcurves_for_face(face) {
                assert!(used.contains(&edge), "{face:?} keeps a trim of {edge:?}");
            }
        }
    }

    let mut plain = Topology::new();
    let hollow = rod_with_a_cavity(&mut plain);
    let rim = rims_at(&plain, hollow, 10.0)[0];
    let equal = chamfer(&mut plain, hollow, &[rim], 0.5).unwrap();
    let two = chamfer_v2(&mut plain, hollow, &[rim], 0.5, 0.3).unwrap();
    for result in [equal, two.solid] {
        let report = validate_solid(&plain, result).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
    }
}

/// Both rims of a cylinder of radius 3, 1 high, or of a frustum from radius
/// 3 down to 2 over a height of 1 (its wall `sqrt 2` long), each with its
/// cap as the reference face when a call takes one.
fn short_wall(topo: &mut Topology, frustum: bool) -> (SolidId, Vec<(EdgeId, FaceId)>) {
    let solid = if frustum {
        make_cone(topo, 3.0, 2.0, 1.0).unwrap()
    } else {
        make_cylinder(topo, 3.0, 1.0).unwrap()
    };
    let mut rims = rims_at(topo, solid, 0.0);
    rims.extend(rims_at(topo, solid, 1.0));
    let picks = rims
        .into_iter()
        .map(|rim| {
            let cap = solid_faces(topo, solid)
                .unwrap()
                .into_iter()
                .find(|&f| {
                    matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. })
                        && face_edges(topo, f).unwrap().contains(&rim)
                })
                .unwrap();
            (rim, cap)
        })
        .collect();
    (solid, picks)
}

/// Two rims of one short wall whose chamfers take the whole wall, or cross
/// on it, are refused through each entry point. Each chamfer as long as the
/// wall left the other rim's band running to a rim that was gone, and
/// came back as success with an open shell.
#[test]
fn two_rims_whose_chamfers_take_or_cross_their_wall_are_refused() {
    let slant = 2.0_f64.sqrt();
    for (frustum, whole, past) in [(false, 1.0, 0.6), (true, slant, 0.8)] {
        for wall in [whole, past] {
            let what = format!("frustum {frustum}, {wall} along the wall");
            let mut topo = Topology::new();
            let (solid, picks) = short_wall(&mut topo, frustum);
            let rims: Vec<EdgeId> = picks.iter().map(|&(e, _)| e).collect();
            assert!(chamfer(&mut topo, solid, &rims, wall).is_err(), "{what}");
            assert!(
                chamfer_v2_on(&mut topo, solid, &picks, 0.4, wall).is_err(),
                "{what}"
            );
            assert!(
                chamfer_two_on(&mut topo, solid, &picks, 0.4, wall).is_err(),
                "{what}"
            );
            // `chamfer_v2` and `chamfer_distance_angle` set their first
            // distance on each rim's first face, the wall.
            let first = topo.build_adjacency(solid).unwrap().faces_for_edge(rims[0])[0];
            assert!(!matches!(
                topo.face(first).unwrap().surface(),
                FaceSurface::Plane { .. }
            ));
            assert!(
                chamfer_v2(&mut topo, solid, &rims, wall, 0.4).is_err(),
                "{what}"
            );
            assert!(
                chamfer_distance_angle(&mut topo, solid, &rims, wall, (0.4 / wall).atan()).is_err(),
                "{what}"
            );
        }
    }
}

/// Two rims of one short wall whose chamfers meet on it take the wall
/// between them: the two cones share their wall circle. Each takes off its
/// revolved triangle.
#[test]
fn two_rims_whose_chamfers_meet_take_their_wall() {
    let half = 0.5 * 2.0_f64.sqrt();
    // The bottom rim's cap runs in from radius 3, the top's from 2, and the
    // wall runs in and up at 45 degrees on the frustum, straight up on the
    // cylinder.
    let cases = [
        (false, 0.5, 9.0 * PI, (3.0, 3.0), (0.0, 1.0)),
        (true, half, 19.0 * PI / 3.0, (3.0, 2.0), (-half, half)),
    ];
    for (frustum, wall, whole, (bottom, top), (run, rise)) in cases {
        for cap in [wall, 0.4] {
            let what = format!("frustum {frustum}, cap {cap}");
            let expected = whole
                - ring(bottom, -cap, (run * wall, rise * wall))
                - ring(top, -cap, (-run * wall, rise * wall));
            let mut topo = Topology::new();
            let (solid, picks) = short_wall(&mut topo, frustum);
            let result = if (cap - wall).abs() < 1e-12 {
                let rims: Vec<EdgeId> = picks.iter().map(|&(e, _)| e).collect();
                chamfer(&mut topo, solid, &rims, wall).unwrap()
            } else {
                let built = chamfer_v2_on(&mut topo, solid, &picks, cap, wall).unwrap();
                assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
                built.solid
            };
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{what}: {:?}", report.issues);
            let volume = solid_volume(&topo, result, 0.001).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{what}: volume {volume}, expected {expected}"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{what}: its mesh is open");
            assert_eq!(solid_faces(&topo, result).unwrap().len(), 4, "{what}");
        }
    }
}
