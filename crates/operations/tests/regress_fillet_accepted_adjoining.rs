//! N339 / OttoCAD N332: two 1 mm blends, separated by acceptance of the first.
//! The second target must be resolved on the trimmed first result, not reused
//! from the original 40 x 25 x 30 mm extrusion. Closure is an assertion.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

fn source(topo: &mut Topology) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(40.0, 0.0, 0.0),
            Point3::new(40.0, 25.0, 0.0),
            Point3::new(0.0, 25.0, 0.0),
        ],
        Tolerance::new().linear,
    )
    .unwrap();
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 30.0).unwrap()
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let tolerance = Tolerance::new().linear;
    let matches: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let edge = topo.edge(id).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            ((start - a).length() < tolerance && (end - b).length() < tolerance)
                || ((start - b).length() < tolerance && (end - a).length() < tolerance)
        })
        .collect();
    assert_eq!(matches.len(), 1, "unique target {a:?}–{b:?}");
    matches[0]
}

fn check(topo: &Topology, solid: SolidId) {
    let report = validate_solid(topo, solid).unwrap();
    let volume = solid_volume(topo, solid, 0.01).unwrap();
    eprintln!(
        "N339 solid={solid:?} volume={volume} issues={:?}",
        report.issues
    );
    assert!(report.is_valid(), "{:?}", report.issues);
    assert!(volume.is_finite() && volume > 0.0 && volume <= 30_000.0);
}

#[test]
#[ignore = "bounded native-family geometry comparison, not an acceptance test"]
#[allow(deprecated)]
fn compare_native_geometry() {
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::explorer::solid_faces;
    for legacy in [true, false] {
        for grouped in [false, true] {
            let mut topo = Topology::new();
            let source = source(&mut topo);
            let mut edges = vec![edge_at(
                &topo,
                source,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(40.0, 0.0, 0.0),
            )];
            if grouped {
                edges.push(edge_at(
                    &topo,
                    source,
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.0, 0.0, 30.0),
                ));
            }
            let solid = if legacy {
                brepkit_operations::fillet::fillet_rolling_ball(&mut topo, source, &edges, 1.0)
                    .unwrap()
            } else {
                let result = fillet_v2(&mut topo, source, &edges, 1.0).unwrap();
                assert!(result.failed.is_empty(), "{:?}", result.failed);
                result.solid
            };
            eprintln!(
                "legacy={legacy} grouped={grouped} volume={} issues={:?}",
                solid_volume(&topo, solid, 0.001).unwrap(),
                validate_solid(&topo, solid).unwrap().issues
            );
            for fid in solid_faces(&topo, solid).unwrap() {
                let face = topo.face(fid).unwrap();
                eprintln!(
                    "face={fid:?} reverse={} surface={:?}",
                    face.is_reversed(),
                    face.surface()
                );
                for oe in topo.wire(face.outer_wire()).unwrap().edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    let a = topo.vertex(edge.start()).unwrap().point();
                    let b = topo.vertex(edge.end()).unwrap().point();
                    let domain = edge.curve().domain_with_endpoints(a, b);
                    let mid =
                        edge.curve()
                            .evaluate_with_endpoints(domain.0.midpoint(domain.1), a, b);
                    eprintln!(
                        "edge={:?} forward={} a={a:?} b={b:?} mid={mid:?} circle={}",
                        oe.edge(),
                        oe.is_forward(),
                        matches!(edge.curve(), EdgeCurve::Circle(_))
                    );
                }
            }
        }
    }
}

#[test]
#[allow(deprecated)]
fn accepted_first_then_adjoining_rolling_ball() {
    use brepkit_operations::fillet::fillet_rolling_ball;
    let _ = env_logger::try_init();
    let mut topo = Topology::new();
    let source = source(&mut topo);
    let edge = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first = fillet_rolling_ball(&mut topo, source, &[edge], 1.0).unwrap();
    check(&topo, first);
    let edge = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_rolling_ball(&mut topo, first, &[edge], 1.0).unwrap();
    check(&topo, source);
    check(&topo, first);
    check(&topo, second);
    check_corner(&topo, second);
    assert!(solid_volume(&topo, second, 0.01).unwrap() < solid_volume(&topo, first, 0.01).unwrap());
}

#[test]
#[ignore = "known v3.4.0 v2 sequential failure; explicit negative qualification probe, not repaired by N339"]
fn accepted_first_then_adjoining_fillet_v2() {
    let _ = env_logger::try_init();
    let mut topo = Topology::new();
    let source = source(&mut topo);
    check(&topo, source);
    let edge = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first = fillet_v2(&mut topo, source, &[edge], 1.0).unwrap();
    assert_eq!(first.succeeded, [edge]);
    assert!(first.failed.is_empty());
    check(&topo, first.solid);
    let edge = edge_at(
        &topo,
        first.solid,
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_v2(&mut topo, first.solid, &[edge], 1.0).unwrap();
    check(&topo, first.solid);
    assert_eq!(second.succeeded, [edge], "{:?}", second.failed);
    assert!(second.failed.is_empty());
    check(&topo, second.solid);
}

#[test]
fn adjoining_edges_grouped_control() {
    let _ = env_logger::try_init();
    let mut topo = Topology::new();
    let source = source(&mut topo);
    let first = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let second = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 30.0),
    );
    let result = fillet_v2(&mut topo, source, &[first, second], 1.0).unwrap();
    assert_eq!(result.succeeded.len(), 2, "{:?}", result.failed);
    assert!(result.failed.is_empty());
    check(&topo, result.solid);
    check(&topo, source);
}

fn check_corner(topo: &Topology, solid: SolidId) {
    let mut seams = 0;
    let adjacency = topo.build_adjacency(solid).unwrap();
    for edge_id in solid_edges(topo, solid).unwrap() {
        let edge = topo.edge(edge_id).unwrap();
        let p = topo.vertex(edge.start()).unwrap().point();
        let q = topo.vertex(edge.end()).unwrap().point();
        let (t0, t1) = edge.curve().domain_with_endpoints(p, q);
        let mid = edge.curve().evaluate_with_endpoints(t0.midpoint(t1), p, q);
        let faces = adjacency.faces_for_edge(edge_id);
        let left = topo.face(faces[0]).unwrap();
        let right = topo.face(faces[1]).unwrap();
        if matches!(
            left.surface(),
            FaceSurface::Nurbs(_) | FaceSurface::Torus(_)
        ) || matches!(
            right.surface(),
            FaceSurface::Nurbs(_) | FaceSurface::Torus(_)
        ) {
            seams += 1;
            let normals: Vec<_> = [left, right]
                .iter()
                .map(|face| {
                    let (u, v) = face.surface().project_point(mid).unwrap_or((0.0, 0.0));
                    let normal = face.surface().normal(u, v);
                    if face.is_reversed() { -normal } else { normal }
                })
                .collect();
            eprintln!(
                "N339 corner seam {edge_id:?} {p:?}–{q:?} normals dot={}",
                normals[0].dot(normals[1])
            );
            assert!(
                normals[0].dot(normals[1]) > 1.0 - Tolerance::new().angular,
                "corner seam {edge_id:?} must be G1"
            );
        }
    }
    assert_eq!(seams, 4, "both cylinder and both planar corner joins");
}

fn snapshot(topo: &Topology, solid: SolidId) -> Vec<String> {
    let mut data = Vec::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        data.push(format!("{fid:?} {face:?}"));
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            data.push(format!("{wid:?} {wire:?}"));
            for oe in wire.edges() {
                let eid = oe.edge();
                let edge = topo.edge(eid).unwrap();
                data.push(format!(
                    "{eid:?} {edge:?} {:?} {:?}",
                    topo.vertex(edge.start()).unwrap(),
                    topo.vertex(edge.end()).unwrap()
                ));
            }
        }
    }
    data
}

fn geometry_samples(topo: &Topology, solid: SolidId) -> Vec<Point3> {
    let mut points = Vec::new();
    for id in solid_edges(topo, solid).unwrap() {
        let edge = topo.edge(id).unwrap();
        let p = topo.vertex(edge.start()).unwrap().point();
        let q = topo.vertex(edge.end()).unwrap().point();
        let (lo, hi) = edge.curve().domain_with_endpoints(p, q);
        for i in 0..=16 {
            points.push(edge.curve().evaluate_with_endpoints(
                lo + (hi - lo) * f64::from(i) / 16.0,
                p,
                q,
            ));
        }
    }
    for id in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        if let FaceSurface::Nurbs(surface) = topo.face(id).unwrap().surface() {
            for i in 0..=6 {
                for j in 0..=6 {
                    points.push(surface.evaluate(f64::from(i) / 6.0, f64::from(j) / 6.0));
                }
            }
        }
    }
    points
}

#[test]
#[allow(deprecated)]
fn transformed_both_ends_both_supports_preserve_sources() {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::fillet::fillet_rolling_ball;
    use brepkit_operations::transform::transform_solid;
    let transforms = [
        Mat4::identity(),
        Mat4::rotation_z(0.37) * Mat4::rotation_x(-0.21),
        Mat4::translation(150.0, -230.0, 42.0) * Mat4::rotation_y(0.63),
    ];
    let mut reference = std::collections::HashMap::new();
    for (pose, transform) in transforms.iter().enumerate() {
        for (size, radius) in [1.0, 0.5, 2.0].into_iter().enumerate() {
            for (end, x) in [0.0, 40.0].into_iter().enumerate() {
                for vertical in [true, false] {
                    eprintln!("N339 pose={pose} radius={radius} end={x} vertical={vertical}");
                    let mut topo = Topology::new();
                    let source = source(&mut topo);
                    transform_solid(&mut topo, source, transform).unwrap();
                    let source_state = snapshot(&topo, source);
                    let edge = edge_at(
                        &topo,
                        source,
                        transform.mul_point(Point3::new(0.0, 0.0, 0.0)),
                        transform.mul_point(Point3::new(40.0, 0.0, 0.0)),
                    );
                    let first = fillet_rolling_ball(&mut topo, source, &[edge], radius).unwrap();
                    check(&topo, first);
                    let state = snapshot(&topo, first);
                    let (a, b) = if vertical {
                        (Point3::new(x, 0.0, radius), Point3::new(x, 0.0, 30.0))
                    } else {
                        (Point3::new(x, radius, 0.0), Point3::new(x, 25.0, 0.0))
                    };
                    let edge =
                        edge_at(&topo, first, transform.mul_point(a), transform.mul_point(b));
                    let second = fillet_rolling_ball(&mut topo, first, &[edge], radius).unwrap();
                    check(&topo, second);
                    check_corner(&topo, second);
                    let samples = geometry_samples(&topo, second);
                    let key = (size, end, vertical);
                    if pose == 0 {
                        reference.insert(key, samples);
                    } else {
                        let expected = &reference[&key];
                        assert_eq!(samples.len(), expected.len());
                        for &point in expected {
                            let transformed = transform.mul_point(point);
                            assert!(
                                samples
                                    .iter()
                                    .any(|&p| (p - transformed).length() < Tolerance::new().linear),
                                "pose {pose}: boundary/patch geometry did not transform rigidly at {point:?}"
                            );
                        }
                    }
                    assert_eq!(
                        snapshot(&topo, source),
                        source_state,
                        "original source modified"
                    );
                    assert_eq!(snapshot(&topo, first), state, "accepted source modified");
                    assert!(
                        solid_volume(&topo, second, 0.01).unwrap()
                            < solid_volume(&topo, first, 0.01).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
#[allow(deprecated)]
fn insufficient_corner_clearance_refuses_without_modifying_accepted_source() {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::fillet::fillet_rolling_ball;
    use brepkit_operations::transform::transform_solid;
    let mut topo = Topology::new();
    let source = source(&mut topo);
    transform_solid(&mut topo, source, &Mat4::scale(1.0, 1.5 / 25.0, 1.0)).unwrap();
    let first_edge = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first = fillet_rolling_ball(&mut topo, source, &[first_edge], 1.0).unwrap();
    check(&topo, first);
    let state = snapshot(&topo, first);
    let second_edge = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, 30.0),
    );
    let error = fillet_rolling_ball(&mut topo, first, &[second_edge], 1.0).unwrap_err();
    assert!(error.to_string().contains("clearance"), "{error}");
    assert_eq!(snapshot(&topo, first), state);
    let excessive = fillet_rolling_ball(&mut topo, first, &[second_edge], 100.0);
    assert!(excessive.is_err());
    assert_eq!(snapshot(&topo, first), state);
}

#[test]
#[allow(deprecated)]
fn independent_edges_grouped_and_sequential_preserve_sources() {
    use brepkit_operations::fillet::fillet_rolling_ball;
    let mut topo = Topology::new();
    let source = source(&mut topo);
    let state = snapshot(&topo, source);
    let first_edge = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let other_edge = edge_at(
        &topo,
        source,
        Point3::new(0.0, 25.0, 30.0),
        Point3::new(40.0, 25.0, 30.0),
    );
    let grouped = fillet_rolling_ball(&mut topo, source, &[first_edge, other_edge], 1.0).unwrap();
    check(&topo, grouped);
    assert_eq!(snapshot(&topo, source), state);
    let first = fillet_rolling_ball(&mut topo, source, &[first_edge], 1.0).unwrap();
    let first_state = snapshot(&topo, first);
    let other_edge = edge_at(
        &topo,
        first,
        Point3::new(0.0, 25.0, 30.0),
        Point3::new(40.0, 25.0, 30.0),
    );
    let sequential = fillet_rolling_ball(&mut topo, first, &[other_edge], 1.0).unwrap();
    check(&topo, sequential);
    assert_eq!(snapshot(&topo, source), state);
    assert_eq!(snapshot(&topo, first), first_state);
    for solid in [grouped, sequential] {
        let surfaces: Vec<_> = brepkit_topology::explorer::solid_faces(&topo, solid)
            .unwrap()
            .into_iter()
            .map(|id| topo.face(id).unwrap().surface())
            .collect();
        assert_eq!(surfaces.len(), 8);
        assert_eq!(
            surfaces
                .iter()
                .filter(|s| matches!(s, FaceSurface::Cylinder(_)))
                .count(),
            2
        );
        assert_eq!(
            surfaces
                .iter()
                .filter(|s| matches!(s, FaceSurface::Plane { .. }))
                .count(),
            6
        );
    }
    assert!(
        (solid_volume(&topo, grouped, 0.01).unwrap()
            - solid_volume(&topo, sequential, 0.01).unwrap())
        .abs()
            < Tolerance::new().linear * 5900.0
    );
}
