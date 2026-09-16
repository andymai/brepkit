//! N411 (updated N414 for the 2026-09-14 sharp-miter amendment): equal-radius
//! TwoEdge corners must have incident wires and closed supports, no torus,
//! and — per the amendment — no corner face of any kind.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

fn source(topo: &mut Topology, size: f64) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(size, 0.0, 0.0),
            Point3::new(size, size, 0.0),
            Point3::new(0.0, size, 0.0),
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
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), size).unwrap()
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

#[test]
fn n411_two_edge_and_face_loop_have_incident_closed_boundaries() {
    for size in [1.0, 254.0] {
        for fraction in [0.05, 0.1] {
            for count in [2, 4] {
                for reverse in [false, true] {
                    let mut topo = Topology::new();
                    let input = source(&mut topo, size);
                    let corners = [
                        Point3::new(0., 0., size),
                        Point3::new(size, 0., size),
                        Point3::new(size, size, size),
                        Point3::new(0., size, size),
                    ];
                    let mut selected: Vec<_> = (0..count)
                        .map(|i| edge_at(&topo, input, corners[i], corners[(i + 1) % 4]))
                        .collect();
                    if reverse {
                        selected.reverse();
                    }
                    let result = fillet_v2(&mut topo, input, &selected, size * fraction).unwrap();
                    assert_eq!(result.succeeded.len(), count);
                    assert!(result.failed.is_empty());
                    let mut uses = std::collections::HashMap::new();
                    let mut tori = 0;
                    let mut rational_corners = 0;
                    for fid in solid_faces(&topo, result.solid).unwrap() {
                        let face = topo.face(fid).unwrap();
                        for wid in std::iter::once(face.outer_wire())
                            .chain(face.inner_wires().iter().copied())
                        {
                            let wire = topo.wire(wid).unwrap();
                            for oe in wire.edges() {
                                uses.entry(oe.edge())
                                    .or_insert_with(Vec::new)
                                    .push(oe.is_forward() ^ face.is_reversed());
                                if let FaceSurface::Torus(torus) = face.surface() {
                                    let edge = topo.edge(oe.edge()).unwrap();
                                    assert!(matches!(edge.curve(), EdgeCurve::Circle(_)));
                                    let a = topo.vertex(edge.start()).unwrap().point();
                                    let b = topo.vertex(edge.end()).unwrap().point();
                                    let (lo, hi) = edge.curve().domain_with_endpoints(a, b);
                                    for sample in 0..=16 {
                                        let p = edge.curve().evaluate_with_endpoints(
                                            lo + (hi - lo) * f64::from(sample) / 16.,
                                            a,
                                            b,
                                        );
                                        let q = p - torus.center();
                                        let z = q.dot(torus.z_axis());
                                        let rho = (q - torus.z_axis() * z).length();
                                        let residual = ((rho - torus.major_radius()).hypot(z)
                                            - torus.minor_radius())
                                        .abs();
                                        assert!(
                                            residual < 1e-7,
                                            "size={size} count={count} residual={residual}"
                                        );
                                    }
                                }
                            }
                        }
                        if matches!(face.surface(), FaceSurface::Torus(_)) {
                            tori += 1;
                        }
                        if matches!(face.surface(), FaceSurface::Nurbs(_)) {
                            rational_corners += 1;
                        }
                    }
                    assert_eq!(tori, 0, "the superseded horn construction must be absent");
                    // N413/N414 amendment (2026-09-14) supersedes the product
                    // patch this test originally targeted: the sharp mitered
                    // corner has no corner face of any kind (rational NURBS
                    // included) and no torus, only the two stripe cylinders
                    // and their shared crease. Counts match
                    // `docs/N413-twoedge-construction-diagnosis.md`'s
                    // amendment exactly (11/17/8, 12/20/10), not the old
                    // product-patch counts (13/20/9, 20/32/14).
                    assert_eq!(
                        rational_corners, 0,
                        "the amendment has no corner face at all, rational or otherwise"
                    );
                    assert_eq!(
                        (
                            solid_vertices(&topo, result.solid).unwrap().len(),
                            solid_edges(&topo, result.solid).unwrap().len(),
                            solid_faces(&topo, result.solid).unwrap().len(),
                        ),
                        if count == 2 {
                            (11, 17, 8)
                        } else {
                            (12, 20, 10)
                        },
                    );
                    for (edge, senses) in uses {
                        assert_eq!(senses.len(), 2, "edge={edge:?}");
                        assert_ne!(senses[0], senses[1], "edge={edge:?}");
                    }
                    assert!(
                        brepkit_operations::validate::validate_solid(&topo, result.solid)
                            .unwrap()
                            .is_valid()
                    );
                }
            }
        }
    }
}

/// A planar support outer wire must not cross itself in edge interiors.
/// These contact curves are straight degree-one NURBS, so solving their
/// endpoint segments is exact and independent of tessellation.
#[test]
fn n411_planar_support_contact_boundaries_do_not_self_intersect() {
    let mut failures = Vec::new();
    for size in [1.0, 254.0] {
        for count in [2, 4] {
            let mut topo = Topology::new();
            let input = source(&mut topo, size);
            let corners = [
                Point3::new(0., 0., size),
                Point3::new(size, 0., size),
                Point3::new(size, size, size),
                Point3::new(0., size, size),
            ];
            let selected: Vec<_> = (0..count)
                .map(|i| edge_at(&topo, input, corners[i], corners[(i + 1) % 4]))
                .collect();
            let result = fillet_v2(&mut topo, input, &selected, size * 0.1).unwrap();
            for fid in solid_faces(&topo, result.solid).unwrap() {
                let face = topo.face(fid).unwrap();
                let FaceSurface::Plane { normal, .. } = face.surface() else {
                    continue;
                };
                if normal.z().abs() < 0.99 {
                    continue;
                }
                let wire = topo.wire(face.outer_wire()).unwrap();
                let contacts: Vec<_> = wire
                    .edges()
                    .iter()
                    .filter_map(|oe| {
                        let edge = topo.edge(oe.edge()).unwrap();
                        match edge.curve() {
                            EdgeCurve::NurbsCurve(curve)
                                if curve.degree() == 1 && curve.control_points().len() == 2 =>
                            {
                                Some((
                                    oe.edge(),
                                    topo.vertex(edge.start()).unwrap().point(),
                                    topo.vertex(edge.end()).unwrap().point(),
                                ))
                            }
                            _ => None,
                        }
                    })
                    .collect();
                for (i, &(eid, a, b)) in contacts.iter().enumerate() {
                    for &(other, c, d) in &contacts[i + 1..] {
                        let (u, v, w) = (b - a, d - c, c - a);
                        let cross = |a: Vec3, b: Vec3| a.x() * b.y() - a.y() * b.x();
                        let denominator = cross(u, v);
                        if denominator.abs() < 1e-12 {
                            continue;
                        }
                        let t = cross(w, v) / denominator;
                        let s = cross(w, u) / denominator;
                        if t > 1e-7 && t < 1. - 1e-7 && s > 1e-7 && s < 1. - 1e-7 {
                            failures.push(format!("size={size} count={count} face={fid:?} edges={eid:?}/{other:?} crossing={:?}",a+u*t));
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "native support-wire self-intersections: {failures:#?}"
    );
}
